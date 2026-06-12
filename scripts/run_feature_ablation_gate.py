"""Feature ablation gate — prove each authored feature GROUP earns its place
*before* the specialist fleet trains.

The analysis-programmer authors a feature matrix + ``{run_id}_feature_spec.json``;
``run_modeling_agent.py`` then joins **every** declared column into the model
unconditionally (declaration == inclusion). That let a regressing feature group (e.g.
period-lagged target aggregates) reach the specialists and only surface *after*
~100 min of training. This gate closes that gap: it scores the canonical-fold OOF
``block_mae`` with and without each logical feature group using ONE fast
HistGradientBoosting candidate, then prunes any group whose removal does not worsen
OOF — turning "declared" into "ablation-proven".

It reuses the *consumed* engine so preprocessing + skew-driven log1p handling are
identical to the floor and specialists:
  - ``build_feature_bundle`` (features.py) + ``_append_authored_features``
    (run_modeling_agent.py — single-sourced join) build the exact bundle the
    specialists train on;
  - ``_build_candidates`` (models.py) yields the same HGB pipeline (incl. the
    ``hgb_log`` target-transform variant when the target is skewed);
  - ``load_canonical_folds`` (cv.py) gives the same shared folds + scored_mask;
  - ``_score_function`` (models.py) resolves the same ``block_mae`` scorer.

Contract & safety:
  - Saves the original spec to ``{run_id}_feature_spec_full.json`` (audit trail) and
    overwrites ``{run_id}_feature_spec.json`` with the pruned set so
    ``_append_authored_features`` consumes the survivors with **zero** code change.
  - Strictly additive: on ANY failure (or AWARDB_SKIP_ABLATION=1) it leaves the spec
    untouched and exits 0 — the declared features still flow through and the floor
    deliverable is never blocked.
  - Round 1 ablates all logical groups; rounds 2-3 ablate only THIS round's NEW
    columns (diffed against the prior ``feature_spec_full.json``), keeping it cheap.

Usage:
    python scripts/run_feature_ablation_gate.py --run-id 20260612_x \
        --cv-folds outputs/logs/20260612_x_cv_folds.json \
        --feature-spec outputs/logs/20260612_x_feature_spec.json --round 1
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

# Prevent BLAS/OpenMP thread-pool deadlocks on macOS fork-based multiprocessing.
# Must precede numpy import — the thread pool is initialised at import time.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

# Logical feature groups carried in the spec, in pruning-priority order. Any
# authored column not in a named group falls into "other_features". Order is fixed
# so group assignment (first-match) is deterministic and the prune decision is
# reproducible run-to-run.
_NAMED_GROUP_KEYS = (
    ("per_fold_aggregates", "per_fold_aggregates"),  # value is list[{name,...}]
    ("text_svd", "text_svd"),
    ("image_features", "image_features"),            # static per-key image-derived columns
    ("datetime_derived", "datetime_derived"),
    ("distribution_shift_interactions", "distribution_shift_interactions"),
)


def _pick_gbdt_factory(bundle, train_df):
    """Return ``(name, factory)`` for the single fast GBDT candidate used to score
    ablations. Prefers the skew-driven ``hgb_log`` variant when the engine built it
    (matches the recommended log1p), else the plain HGB; both are sklearn-only and
    fast. Falls back to the first gbdt-family candidate. Never hardcodes the log
    decision — it follows the engine's own candidate set."""
    from src.data_agent.models import _build_candidates, _candidate_family

    cands = _build_candidates(bundle, train_df, SEED)
    by_name = {n: f for n, f in cands}
    for name in ("hgb_log", "hist_gradient_boosting"):
        if name in by_name:
            return name, by_name[name]
    for n, f in cands:
        if _candidate_family(n) == "gbdt":
            return n, f
    raise RuntimeError("no gbdt candidate available for ablation")


def _pruned_view(base, drop: set[str]):
    """A lightweight bundle copy with ``drop`` removed from the column LISTS only —
    the DataFrames are shared (read-only during training). Because the OOF loop does
    ``X = train_df[feature_columns]`` and ``_build_candidates`` reads the column
    lists, dropping the names suffices to ablate a group without touching data."""
    b = copy.copy(base)  # shallow — shares train_df/predict_df/target
    b.feature_columns = [c for c in base.feature_columns if c not in drop]
    b.numeric_columns = [c for c in base.numeric_columns if c not in drop]
    b.categorical_columns = [c for c in base.categorical_columns if c not in drop]
    b.text_columns = [c for c in (base.text_columns or []) if c not in drop]
    return b


def _oof_block_mae(bundle, schema, folds, scored_mask) -> tuple[float, str, str]:
    """One HGB candidate, fit per canonical fold, scored with the resolved metric
    (``block_mae`` when a block column exists) on the scored_mask rows. log1p is
    handled inside the candidate (LogTargetRegressor) exactly as for the specialists
    — no separate log logic here. Returns ``(score, metric_name, model_name)``."""
    from src.data_agent.models import _sanitize_predictions, _score_function

    y = pd.to_numeric(bundle.target, errors="coerce")
    valid = y.notna()
    train_df = bundle.train_df.loc[valid].reset_index(drop=True)
    y = y.loc[valid].reset_index(drop=True)
    X = train_df[bundle.feature_columns].copy()

    name, factory = _pick_gbdt_factory(bundle, train_df)
    oof = np.full(len(X), np.nan, dtype=float)
    for tr_idx, va_idx in folds:
        est = factory()
        est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = _sanitize_predictions(est.predict(X.iloc[va_idx]), y)

    mask = ~np.isnan(oof)
    if scored_mask is not None:
        sm = np.asarray(scored_mask, dtype=bool)
        if len(sm) == len(mask):
            mask = mask & sm
    idx = np.flatnonzero(mask)
    if len(idx) < 5:
        raise RuntimeError("insufficient OOF coverage for ablation scoring")
    sfn, mname = _score_function(
        getattr(bundle.task, "metric", None), schema.block_column,
        train_df.iloc[idx].reset_index(drop=True))
    return float(sfn(y.iloc[idx].to_numpy(), oof[idx])), mname, name


def _feature_groups(spec: dict, present: set[str]) -> dict[str, list[str]]:
    """Map the spec's logical groups -> their columns present in the bundle.
    First-match assignment over a fixed key order; ungrouped authored columns go to
    'other_features'. Deterministic & order-stable."""
    cols = [c for c in spec.get("feature_columns", []) if c in present]
    groups: dict[str, list[str]] = {}
    assigned: set[str] = set()
    for spec_key, gname in _NAMED_GROUP_KEYS:
        raw = spec.get(spec_key) or []
        if spec_key == "per_fold_aggregates":
            members = [a.get("name") for a in raw if isinstance(a, dict)]
        else:
            members = list(raw)
        members = [c for c in members if c in present and c not in assigned]
        if members:
            groups[gname] = members
            assigned.update(members)
    rest = [c for c in cols if c not in assigned]
    if rest:
        groups["other_features"] = rest
    return groups


def _round_new_cols(spec: dict, prev_full: dict | None, present: set[str]) -> list[str]:
    """Columns introduced this round = current authored cols minus the prior round's
    full authored cols. Empty when there is no prior spec."""
    if not prev_full:
        return []
    prev = set(prev_full.get("feature_columns", []))
    return [c for c in spec.get("feature_columns", []) if c in present and c not in prev]


def _prune_spec(spec: dict, prune_cols: set[str]) -> dict:
    """Return a copy of the spec with pruned columns removed from feature_columns and
    every named group list, preserving schema so _append_authored_features is happy."""
    out = dict(spec)
    out["feature_columns"] = [c for c in spec.get("feature_columns", []) if c not in prune_cols]
    out["per_fold_aggregates"] = [
        a for a in (spec.get("per_fold_aggregates") or [])
        if not (isinstance(a, dict) and a.get("name") in prune_cols)
    ]
    for key in ("text_svd", "image_features", "datetime_derived", "distribution_shift_interactions"):
        if key in spec:
            out[key] = [c for c in spec[key] if c not in prune_cols]
    out["ablation_pruned_columns"] = sorted(prune_cols)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="agent")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--cv-folds", required=True, help="path to {run_id}_cv_folds.json")
    ap.add_argument("--feature-spec", required=True, help="path to {run_id}_feature_spec.json")
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--prune-margin", type=float, default=None,
                    help="prune a group only if removing it improves OOF by MORE than this "
                         "(an absolute metric delta). Default 0.01*baseline (1%%) — well above "
                         "fold noise, so only clearly-harmful groups (e.g. regressing lags) are "
                         "auto-pruned; marginal/neutral groups are kept for the stronger tuned "
                         "stack + the thin LLM gate to judge.")
    ap.add_argument("--prev-spec", default=None,
                    help="prior feature_spec_full.json for round>1 NEW-column diff; "
                         "defaults to the sibling {run_id}_feature_spec_full.json")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    logs = repo / "outputs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    spec_path = Path(args.feature_spec)
    full_path = spec_path.with_name(spec_path.name.replace("feature_spec", "feature_spec_full"))
    out_json = logs / f"{args.run_id}_feature_ablation.json"

    def _passthrough(reason: str, extra: dict | None = None) -> None:
        """Leave the spec untouched, log a degraded note, exit 0 (never block)."""
        payload = {"run_id": args.run_id, "status": "degraded", "reason": reason,
                   "pruned_groups": [], "kept_groups": [], "pruned_columns": [],
                   "determinism_seed": SEED}
        if extra:
            payload.update(extra)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({"run_id": args.run_id, "ablation": "passthrough", "reason": reason}))

    # Budget-guard / explicit skip: behave like the watchdog/optimizer under the
    # 800k-token / 90-min shortcut — pass features through unpruned.
    if os.environ.get("AWARDB_SKIP_ABLATION"):
        return _passthrough("AWARDB_SKIP_ABLATION set")

    if not spec_path.exists():
        return _passthrough(f"feature spec absent: {spec_path}")
    if not Path(args.cv_folds).exists():
        return _passthrough(f"cv folds absent: {args.cv_folds}")

    try:
        from scripts.run_modeling_agent import _append_authored_features
        from src.data_agent.schema import discover_schema
        from src.data_agent.features import build_feature_bundle
        from src.data_agent.cv import load_canonical_folds

        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        prev_full = None
        if args.round and args.round > 1:
            prev_path = Path(args.prev_spec) if args.prev_spec else full_path
            if prev_path.exists():
                prev_full = json.loads(prev_path.read_text(encoding="utf-8"))

        schema = discover_schema(repo / "data")
        bundle = build_feature_bundle(schema)
        added = _append_authored_features(bundle, spec_path)
        if not added:
            return _passthrough("no authored features joined (nothing to ablate)")

        present = set(bundle.feature_columns)
        valid_mask = pd.to_numeric(bundle.target, errors="coerce").notna().to_numpy()
        cf = load_canonical_folds(args.cv_folds, valid_mask=valid_mask)
        if not cf.folds:
            return _passthrough("no usable canonical folds")
        folds, scored_mask = cf.folds, cf.scored_mask

        baseline_mae, metric_name, model_name = _oof_block_mae(bundle, schema, folds, scored_mask)
        # Conservative auto-prune: only drop a group whose removal IMPROVES OOF by
        # more than ``prune_margin`` (1% of baseline by default). A single untuned
        # HGB is a weak proxy for the tuned multi-model stack, so we never auto-drop
        # merely-neutral/weak groups (within fold noise) — those are left for the
        # stronger stack and the thin LLM gate. Clearly-harmful groups (a regressing
        # lag group improves OOF a lot when removed) are caught decisively.
        prune_margin = args.prune_margin if args.prune_margin is not None else 0.01 * baseline_mae

        all_groups = _feature_groups(spec, present)
        # Round 1: ablate every logical group. Rounds 2-3: ablate ONLY this round's
        # new columns (one cheap extra fit) — diffed against the prior full spec.
        if args.round and args.round > 1:
            new_cols = _round_new_cols(spec, prev_full, present)
            ablate_targets = {"round_new": new_cols} if new_cols else {}
        else:
            ablate_targets = all_groups

        if not ablate_targets:
            return _passthrough("no ablation targets this round", {"baseline_mae": baseline_mae})

        results: dict[str, dict] = {}
        prune_cols: set[str] = set()
        for gname, gcols in ablate_targets.items():
            gcols = [c for c in gcols if c in present]
            if not gcols:
                continue
            mae_wo, _, _ = _oof_block_mae(_pruned_view(bundle, set(gcols)), schema, folds, scored_mask)
            # delta < 0  => removing the group IMPROVES OOF (group hurts).
            # delta > 0  => removing the group WORSENS OOF (group helps).
            # Auto-prune only when removal improves OOF by more than prune_margin.
            delta = mae_wo - baseline_mae
            decision = "prune" if delta < -prune_margin else "keep"
            results[gname] = {"mae_without": round(mae_wo, 6), "delta": round(delta, 6),
                              "decision": decision, "n_cols": len(gcols), "cols": gcols}
            if decision == "prune":
                prune_cols.update(gcols)

        # Persist the original (audit) then overwrite with the pruned spec.
        full_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        pruned = _prune_spec(spec, prune_cols)
        spec_path.write_text(json.dumps(pruned, indent=2), encoding="utf-8")

        payload = {
            "run_id": args.run_id, "status": "ok", "round": args.round,
            "metric_name": metric_name, "ablation_model": model_name,
            "baseline_mae": round(baseline_mae, 6), "prune_margin": round(prune_margin, 8),
            "per_group": results,
            "pruned_groups": sorted(g for g, r in results.items() if r["decision"] == "prune"),
            "kept_groups": sorted(g for g, r in results.items() if r["decision"] == "keep"),
            "pruned_columns": sorted(prune_cols),
            "n_pruned_columns": len(prune_cols),
            "feature_spec_full": str(full_path),
            "feature_spec_pruned": str(spec_path),
            "determinism_seed": SEED,
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps({k: payload[k] for k in
                          ("run_id", "baseline_mae", "pruned_groups", "n_pruned_columns")}))
    except Exception as exc:  # noqa: BLE001 — gate is best-effort, never blocks
        return _passthrough(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
