"""Train one modeling-group specialist's candidate and report its CV score.

Reused by the parallel modeling-group agents (gbdt / linear / trees / full). It is
fully dataset-agnostic — everything is driven by schema discovery +
``data/DATA_DESCRIPTION.md`` and the shared, tested engine in ``src/data_agent``.
It writes a *candidate* submission + a candidate JSON and NEVER overwrites the
repo-root ``submission.csv`` (only the supervisor does that, keep-best).

Usage:
    python scripts/run_modeling_agent.py --approach gbdt --run-id 20260608_x
    python scripts/run_modeling_agent.py --approach linear --run-id 20260608_x
    python scripts/run_modeling_agent.py --approach full --run-id 20260608_x
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_APPROACH_FAMILIES = {
    "gbdt": {"gbdt"},
    "linear": {"linear"},
    "trees": {"trees"},
    "full": None,  # the entire pool (same as the deterministic floor)
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", default="full", choices=sorted(_APPROACH_FAMILIES))
    ap.add_argument("--run-id", default="agent")
    ap.add_argument("--repo", default=".")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))

    # Live progress heartbeat for the modeling-watchdog. Env-gated and
    # dataset-agnostic: a no-op unless AWARDB_HEARTBEAT_PATH is set, so default
    # behaviour is unchanged. The orchestrator may pre-set the path; otherwise we
    # default it next to the other run logs so a directly-invoked run still streams.
    role = f"{args.approach}-specialist"
    logs = repo / "outputs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AWARDB_HEARTBEAT_PATH", str(logs / f"{args.run_id}_{role}_progress.jsonl"))
    os.environ.setdefault("AWARDB_ROLE", role)
    os.environ.setdefault("AWARDB_RUN_ID", args.run_id)

    from src.data_agent.heartbeat import emit
    from src.data_agent.schema import discover_schema
    from src.data_agent.features import build_feature_bundle, _read_description
    from src.data_agent.gates import write_verdict
    from src.data_agent.leakage_guard import guard_to_verdict
    from src.data_agent.models import train_and_predict
    from src.data_agent.runner import apply_monotonic_constraints, _build_submission

    families = _APPROACH_FAMILIES[args.approach]
    schema = discover_schema(repo / "data")
    bundle = build_feature_bundle(schema)

    # Code-enforced leakage floor over the static feature set this candidate trains
    # on (deterministic; an LLM auditor verdict overrides via the llm_gate file).
    write_verdict(guard_to_verdict(bundle.leakage_guard), logs, args.run_id)
    if bundle.leakage_guard.get("status") == "fail":
        emit("leakage_guard_fail", summary=bundle.leakage_guard.get("summary"))

    mr = train_and_predict(bundle, block_column=schema.block_column, families=families)

    desc = _read_description(schema.description_path)
    preds, mono = apply_monotonic_constraints(
        mr.predictions, bundle.sample_submission, schema, desc)
    sub = _build_submission(bundle, schema.row_id_column, schema.target_column, preds, mr.output_kind)

    cand_csv = logs / f"{args.run_id}_cand_{args.approach}.csv"
    sub.to_csv(cand_csv, index=False)

    cv_score = next(
        (s.get("score") for s in mr.model_scores
         if s.get("name") == mr.selected_model_name and s.get("status") == "ok"),
        None,
    )
    out = {
        "role": f"{args.approach}-specialist",
        "approach": args.approach,
        "selected_model": mr.selected_model_name,
        "cv_metric": mr.metric_name,
        "cv_score": cv_score,
        "lower_is_better": not mr.greater_is_better,
        "candidate_submission": str(cand_csv),
        "monotonic_applied": bool(mono.get("applied")),
    }
    (logs / f"{args.run_id}_agent_{args.approach}.json").write_text(json.dumps(out, indent=2))
    emit("final_done", best_so_far=cv_score)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
