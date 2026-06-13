"""Data-pattern / feature-influence analysis — the deterministic engine behind the
Step-3c ``data-pattern-analyzer`` subagent.

Runs BEFORE the planner so the plan can be time-series-aware and influence-driven:
it reuses ``src/data_agent/pattern_analysis.py`` (per-feature target correlations,
interactions, distribution shift, sub-target/target-component checks) and adds the
two diagnostics the planner needs but the engine lacked —

  * ``time_series_shape``  — series length (n_periods), rows-per-period, and
    per-(panel-group) series length, so lag/rolling windows are sized to the data;
  * ``target_autocorrelation`` — within-group target ACF at lags 1/3/6/12, so the
    plan picks WHICH lag features carry signal.

It also fixes the engine's opaque-period blind spot: ``period_id`` is a hashed token
that cannot date-parse, so the engine wrongly reported ``has_time=False`` and
``random_holdout``. Here we resolve a token→ordinal ``period_rank`` from the
guardian's ``validation_strategy.json`` (``holdout_parameters.period_order``) and
pass it in, so the chronological structure is recognised.

Strictly advisory & additive: the influence ranking only PRIORITISES the plan — it
never becomes a model feature and never touches OOF scoring (the real aggregates are
still built per-fold by the programmer and guarded by the feature-leakage-reviewer).
On any failure it writes a minimal, valid report so the planner can still proceed.

Usage:
    python scripts/run_pattern_analysis.py --run-id 20260613_x \
        --spec outputs/logs/spec_parse.json \
        --data-profile outputs/logs/data_profile.json \
        --validation-strategy outputs/logs/validation_strategy.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from pathlib import Path


def _load_json(path: str | Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_period_rank(validation_strategy: dict) -> dict[str, int] | None:
    """token→ordinal from the guardian's resolved period order (the authority that
    mapped opaque period ids to dates). ``None`` when unavailable — the engine then
    falls back to date-parsing."""
    order = (((validation_strategy or {}).get("holdout_parameters") or {})
             .get("period_order"))
    if isinstance(order, list) and order:
        return {str(tok): i for i, tok in enumerate(order)}
    return None


def _resolve_group_keys(spec: dict, time_col: str | None) -> list[str]:
    """Panel-series identity = the group/categorical dimensions (excluding the time
    column). Read from spec_parse detected_structure; dataset-agnostic."""
    ds = (spec or {}).get("detected_structure") or {}
    gks: list[str] = []
    for key in ("group_columns", "block_columns"):
        for c in ds.get(key) or []:
            if c and c != time_col and c not in gks:
                gks.append(c)
    return gks


def _recommendations(report: dict) -> dict:
    """Deterministic planner-facing prioritisation derived from the report (the
    subagent may enrich this; having a baseline makes the artifact useful even if the
    agent layer is thin)."""
    fc = report.get("feature_target_correlations") or {}
    pos = fc.get("top_positive") or []
    neg = fc.get("top_negative") or []
    ranked = sorted(
        [{"feature": c["feature"], "abs_corr": c.get("abs_r"),
          "sign": "+" if c.get("pearson_r", 0) >= 0 else "-"}
         for c in (pos + neg) if c.get("feature")],
        key=lambda x: (x["abs_corr"] is not None, x["abs_corr"] or 0), reverse=True)
    high_influence = [r["feature"] for r in ranked[:6]]
    inter = [p.get("features") for p in
             ((report.get("cross_feature_target_patterns") or {}).get("top_interactions") or [])
             if p.get("features")]
    ac = report.get("target_autocorrelation") or {}
    strongest = ac.get("strongest_lags") or []
    shape = report.get("time_series_shape") or {}
    pgs = (shape.get("per_group_series_length") or {})
    min_len = pgs.get("min")
    # mark lags experimental when the shortest per-group series can't support them
    experimental = bool(min_len is not None and strongest and min_len < 3 * max(strongest))
    return {
        "high_influence_direct_features": high_influence,
        "prioritize_target_aggregates_on": high_influence[:4],
        "prioritize_interactions": inter[:3],
        "lag_features": {
            "suggested_lags": strongest,
            "rolling_windows": [3] if strongest else [],
            "experimental": experimental,
        },
        "series_length_periods": shape.get("n_periods"),
        "note": ("Advisory prioritisation only — these correlations never become "
                 "model features; the programmer still builds aggregates per-fold."),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=os.environ.get("AWARDB_RUN_ID", "agent"))
    ap.add_argument("--repo", default=".")
    ap.add_argument("--spec", default="outputs/logs/spec_parse.json")
    ap.add_argument("--data-profile", default="outputs/logs/data_profile.json")
    ap.add_argument("--validation-strategy", default="outputs/logs/validation_strategy.json")
    ap.add_argument("--out", default=None, help="output filename (default {run_id}_feature_influence.json)")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    logs = repo / "outputs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    out_name = args.out or f"{args.run_id}_feature_influence.json"
    out_path = logs / out_name

    def _degraded(reason: str) -> None:
        spec = _load_json(args.spec)
        prof = _load_json(args.data_profile)
        ss = (prof.get("split_structure") or {})
        payload = {
            "run_id": args.run_id, "status": "degraded", "reason": reason,
            "is_timeseries": ss.get("type", "").startswith("chronological")
                              or bool(ss.get("train_period_range")),
            "time_series_shape": {
                "available": bool(ss.get("train_period_range")),
                "n_periods": (ss.get("train_period_range") or {}).get("n_periods"),
            },
            "target_autocorrelation": {"available": False},
            "feature_influence": {"method": "pearson", "ranked": []},
            "recommendations_for_planner": {
                "high_influence_direct_features": [], "prioritize_target_aggregates_on": [],
                "prioritize_interactions": [], "lag_features": {"suggested_lags": [],
                "rolling_windows": [], "experimental": True},
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"run_id": args.run_id, "pattern_analysis": "degraded", "reason": reason}))

    try:
        from src.data_agent.schema import discover_schema
        from src.data_agent.features import build_feature_bundle
        from src.data_agent.pattern_analysis import run_pattern_analysis

        spec = _load_json(args.spec)
        vstrat = _load_json(args.validation_strategy)
        period_rank = _resolve_period_rank(vstrat)

        schema = discover_schema(repo / "data")
        bundle = build_feature_bundle(schema)
        time_col = getattr(schema, "time_column", None) or getattr(schema, "row_id_column", None)
        group_keys = _resolve_group_keys(spec, time_col) or None

        report = run_pattern_analysis(
            schema, bundle, repo_root=repo, run_id=args.run_id,
            period_rank=period_rank, group_keys=group_keys, out_name=out_name)

        # add the ranked feature_influence + planner recommendations, then re-write
        fc = report.get("feature_target_correlations") or {}
        report["feature_influence"] = {
            "method": "pearson",
            "ranked": sorted(
                [{"feature": c["feature"], "abs_corr": c.get("abs_r"),
                  "sign": "+" if c.get("pearson_r", 0) >= 0 else "-"}
                 for c in ((fc.get("top_positive") or []) + (fc.get("top_negative") or []))
                 if c.get("feature")],
                key=lambda x: (x["abs_corr"] or 0), reverse=True),
        }
        report["sub_target_correlations"] = (
            (report.get("target_component_analysis") or {}).get("components") or [])
        report["recommendations_for_planner"] = _recommendations(report)
        report["status"] = "ok"
        out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps({
            "run_id": args.run_id, "pattern_analysis": "ok",
            "is_timeseries": report.get("is_timeseries"),
            "n_periods": (report.get("time_series_shape") or {}).get("n_periods"),
            "strongest_lags": (report.get("target_autocorrelation") or {}).get("strongest_lags"),
            "top_influence": report["feature_influence"]["ranked"][:3],
        }))
    except Exception as exc:  # never block the workflow
        _degraded(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
