"""Train one modeling-group specialist's candidate and report its CV score.

Reused by the parallel modeling-group agents (gbdt / linear / trees / full). It is
fully dataset-agnostic — everything is driven by schema discovery +
``data/DATA_DESCRIPTION.md`` and the shared, tested engine in ``src/data_agent``.
It writes a *candidate* submission + a candidate JSON + an OOF file and NEVER
overwrites the repo-root ``submission.csv`` (only the keep-best owner does that).

When ``--cv-folds`` is given it scores on the canonical shared folds so its CV is
directly comparable to the floor + other candidates (keep-best). When
``--feature-spec`` is given it appends the analysis-programmer's authored features
to the floor's frozen features.

Usage:
    python scripts/run_modeling_agent.py --approach gbdt --run-id 20260608_x \
        --cv-folds outputs/logs/20260608_x_cv_folds.json \
        --feature-spec outputs/logs/20260608_x_feature_spec.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_APPROACH_FAMILIES = {
    "gbdt": {"gbdt"},
    "linear": {"linear"},
    "trees": {"trees"},
    "full": None,  # the entire pool (same as the deterministic floor)
}


def _append_authored_features(bundle, feature_spec_path: Path) -> list[str]:
    """Left-join the analysis-programmer's authored feature matrices onto the
    floor's bundle by row order. Returns the names of the appended columns (empty
    on any failure — the specialist still runs on the floor's frozen features)."""
    try:
        spec = json.loads(Path(feature_spec_path).read_text(encoding="utf-8"))
        tr_path = spec.get("features_train")
        pr_path = spec.get("features_pred")
        cols = spec.get("feature_columns") or []
        if not (tr_path and pr_path and cols):
            return []
        rd = (lambda p: pd.read_parquet(p) if str(p).endswith(".parquet") else pd.read_csv(p))
        ftr = rd(tr_path)
        fpr = rd(pr_path)
        if len(ftr) != len(bundle.train_df) or len(fpr) != len(bundle.predict_df):
            return []  # row counts must match for a positional join
        added = []
        for c in cols:
            if c in ftr.columns and c in fpr.columns and c not in bundle.train_df.columns:
                bundle.train_df[c] = ftr[c].to_numpy()
                bundle.predict_df[c] = fpr[c].to_numpy()
                bundle.feature_columns.append(c)
                if pd.api.types.is_numeric_dtype(ftr[c]):
                    bundle.numeric_columns.append(c)
                added.append(c)
        return added
    except Exception as exc:  # noqa: BLE001 — authored features are best-effort
        print(f"[features] authored features not applied: {type(exc).__name__}: {exc}")
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", default="full", choices=sorted(_APPROACH_FAMILIES))
    ap.add_argument("--run-id", default="agent")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--cv-folds", default=None, help="path to {run_id}_cv_folds.json (canonical shared folds)")
    ap.add_argument("--feature-spec", default=None, help="path to {run_id}_feature_spec.json (authored features)")
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
    from src.data_agent.cv import load_canonical_folds, write_oof

    families = _APPROACH_FAMILIES[args.approach]
    schema = discover_schema(repo / "data")
    bundle = build_feature_bundle(schema)

    # Code-enforced leakage floor over the static feature set this candidate trains
    # on (deterministic; an LLM auditor verdict overrides via the llm_gate file).
    write_verdict(guard_to_verdict(bundle.leakage_guard), logs, args.run_id)
    if bundle.leakage_guard.get("status") == "fail":
        emit("leakage_guard_fail", summary=bundle.leakage_guard.get("summary"))

    added_features = []
    if args.feature_spec and Path(args.feature_spec).exists():
        added_features = _append_authored_features(bundle, Path(args.feature_spec))

    # Canonical shared folds (aligned to the valid-target frame train_and_predict uses).
    folds = scored_mask = None
    if args.cv_folds and Path(args.cv_folds).exists():
        try:
            valid_mask = pd.to_numeric(bundle.target, errors="coerce").notna().to_numpy()
            cf = load_canonical_folds(args.cv_folds, valid_mask=valid_mask)
            if cf.folds:
                folds, scored_mask = cf.folds, cf.scored_mask
        except Exception as exc:  # noqa: BLE001
            print(f"[cv] canonical folds not applied: {type(exc).__name__}: {exc}")

    mr = train_and_predict(bundle, block_column=schema.block_column, families=families,
                           folds=folds, scored_mask=scored_mask)

    desc = _read_description(schema.description_path)
    preds, mono = apply_monotonic_constraints(
        mr.predictions, bundle.sample_submission, schema, desc)
    sub = _build_submission(bundle, schema.row_id_column, schema.target_column, preds, mr.output_kind)

    cand_csv = logs / f"{args.run_id}_cand_{args.approach}.csv"
    sub.to_csv(cand_csv, index=False)

    # OOF on the (canonical) folds — train-row aligned, for common-OOF keep-best.
    oof_path = None
    oof_full = (getattr(mr, "holdout_by_model", None) or {}).get(mr.selected_model_name)
    if oof_full is not None:
        oof_path = logs / f"{args.run_id}_oof_{args.approach}.csv"
        write_oof(oof_path, oof_full, scored_mask=scored_mask)

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
        "oof_path": str(oof_path) if oof_path else None,
        "canonical_folds": bool(folds),
        "authored_features_used": added_features,
        "monotonic_applied": bool(mono.get("applied")),
    }
    (logs / f"{args.run_id}_agent_{args.approach}.json").write_text(json.dumps(out, indent=2))
    emit("final_done", best_so_far=cv_score)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
