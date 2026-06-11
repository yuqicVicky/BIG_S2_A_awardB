"""Code-enforced leakage invariant for the *consumed* feature set.

The modeling pipeline computes per-group target statistics **inside** the sklearn
Pipeline (fit per CV fold — see :class:`GroupTargetAggregator`), which is
leakage-safe. The risk this guard defends against is the opposite pattern: a
*static* feature column — one materialised before the CV split — that is the
target, a transform of it, or a full-data target aggregate. Such a column bakes
held-out-fold target signal into training-fold rows and inflates CV.

This guard is **dataset-agnostic**: it takes the resolved target column and the
group keys the engine already discovered, and reads everything else from the
data. It hardcodes no column, file, or class name. It is meant to run at the one
chokepoint every model trains through (``build_feature_bundle``), so any feature
the agent adds is checked the same way.

The result maps onto the shared :class:`~src.data_agent.gates.Verdict` schema so
it can act as the deterministic floor for the ``leakage`` stage; an LLM auditor's
verdict still overrides it (see ``gates.load_llm_verdict``).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

# Default name tokens that mark a column as target/outcome-derived. Kept local so
# this module has no import dependency on ``features`` (which imports this one).
# Callers may pass their own list to stay in sync with the feature engine.
DEFAULT_LEAKAGE_NAME_TOKENS = (
    "target", "label", "truth", "actual", "observed",
    "future", "post", "after", "final", "prediction", "predicted",
)

# A static numeric feature whose |pearson r| with the target exceeds this is, for
# practical purposes, the target itself (or a monotone transform). Group-constant
# aggregates won't trip this — they're caught by the group-reconstruction check.
_CORR_AS_TARGET = 0.98

# Relative tolerance for deciding a static column equals a recomputed group
# target aggregate. Scaled by the target's own spread so it is unit-free.
_AGG_MATCH_RTOL = 1e-3


def _normalize(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name))


def _target_transform_names(target_column: str) -> set[str]:
    """Common deterministic transforms of the target a pipeline might persist."""
    t = str(target_column)
    suffixes = ("log", "log1p", "log10", "sqrt", "boxcox", "scaled", "norm",
                "std", "z", "rank", "expm1", "diff", "pct", "resid")
    out = {t, _normalize(t)}
    for s in suffixes:
        out.add(f"{t}_{s}")
        out.add(_normalize(f"{t}_{s}"))
        out.add(f"{s}_{t}")
    return out


def scan_precomputed_target_leakage(
    train_df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    *,
    group_aggregate_keys: Iterable[Sequence[str]] | None = None,
    name_tokens: Iterable[str] | None = None,
    corr_threshold: float = _CORR_AS_TARGET,
) -> dict[str, Any]:
    """Scan the *static* model feature set for precomputed target leakage.

    Parameters mirror what ``build_feature_bundle`` already has on hand. Returns a
    plain dict ``{status, severity, findings, checked, summary}`` — never raises on
    ordinary data, so it is safe to call on every build.

    A finding is one of:
      * ``target_in_features``    — the target (or a transform) is a model feature.
      * ``leakage_name_token``    — a feature name carries an outcome token.
      * ``target_correlation``    — a static feature is ~equal to the target.
      * ``precomputed_group_aggregate`` — a static feature reconstructs a full-data
        per-group target aggregate (the cross-fold leak that motivates this guard).
    """
    tokens = tuple(name_tokens) if name_tokens is not None else DEFAULT_LEAKAGE_NAME_TOKENS
    transforms = _target_transform_names(target_column)
    feature_columns = [c for c in feature_columns if c in train_df.columns]

    findings: list[dict[str, Any]] = []

    if target_column not in train_df.columns:
        # Nothing to check against — report inconclusive rather than a false pass.
        return {
            "status": "warn", "severity": "low",
            "findings": [{"issue": "target_absent_from_train_frame",
                          "detail": f"target '{target_column}' not in the build frame; guard skipped"}],
            "checked": {"n_features": len(feature_columns), "n_group_keys": 0},
            "summary": "leakage guard skipped: target column absent",
        }

    y = pd.to_numeric(train_df[target_column], errors="coerce")
    y_std = float(y.std(skipna=True)) if y.notna().any() else 0.0

    # ── 1. target / transform present as a model feature ──────────────────────
    for col in feature_columns:
        if col == target_column or col in transforms or _normalize(col) in transforms:
            findings.append({"issue": "target_in_features", "column": col, "severity": "high",
                             "detail": "target column or a transform of it is in the model feature set"})

    # ── 2. outcome name token in a feature name ───────────────────────────────
    for col in feature_columns:
        if col == target_column:
            continue
        ncol = _normalize(col)
        hit = next((tok for tok in tokens if tok in ncol), None)
        if hit is not None:
            findings.append({"issue": "leakage_name_token", "column": col, "token": hit,
                             "severity": "high",
                             "detail": f"feature name contains outcome token '{hit}'"})

    # ── 3. a static feature that is ~the target (monotone transform) ──────────
    if y_std > 0:
        for col in feature_columns:
            if col == target_column:
                continue
            x = pd.to_numeric(train_df[col], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() < 5 or float(x[mask].std()) == 0.0:
                continue
            try:
                r = float(np.corrcoef(x[mask], y[mask])[0, 1])
            except Exception:
                continue
            if np.isfinite(r) and abs(r) >= corr_threshold:
                findings.append({"issue": "target_correlation", "column": col,
                                 "pearson_r": round(r, 4), "severity": "high",
                                 "detail": f"static feature |r| {abs(r):.3f} >= {corr_threshold} with target"})

    # ── 4. static feature reconstructs a full-data per-group target aggregate ──
    # This is the cross-fold leak: a column constant within a discovered group key
    # whose value equals that group's target mean/median computed over ALL rows.
    already = {f.get("column") for f in findings}
    group_keys = [list(k) for k in (group_aggregate_keys or []) if k]
    if y_std > 0 and group_keys:
        tol = max(_AGG_MATCH_RTOL * y_std, 1e-9)
        for keys in group_keys:
            if any(k not in train_df.columns for k in keys):
                continue
            grouped = y.groupby([train_df[k] for k in keys])
            for stat in ("mean", "median"):
                agg = grouped.transform(stat)
                if agg.isna().all():
                    continue
                for col in feature_columns:
                    if col in already or col in keys:
                        continue
                    x = pd.to_numeric(train_df[col], errors="coerce")
                    mask = x.notna() & agg.notna()
                    if mask.sum() < 10:
                        continue
                    if float(np.nanmax(np.abs((x[mask] - agg[mask]).to_numpy()))) <= tol:
                        findings.append({"issue": "precomputed_group_aggregate", "column": col,
                                         "group_keys": keys, "stat": stat, "severity": "high",
                                         "detail": "static feature equals a full-data per-group target "
                                                   f"{stat} over {keys}; recompute inside the CV fold"})
                        already.add(col)

    severity = "high" if findings else "none"
    status = "fail" if findings else "pass"
    n = len(findings)
    summary = ("no precomputed target leakage in the consumed feature set"
               if not findings else
               f"{n} static feature(s) carry precomputed target signal: "
               + ", ".join(sorted({f['column'] for f in findings if f.get('column')})[:8]))
    return {
        "status": status, "severity": severity, "findings": findings,
        "checked": {"n_features": len(feature_columns), "n_group_keys": len(group_keys)},
        "summary": summary,
    }


def guard_to_verdict(result: dict[str, Any], stage: str = "leakage"):
    """Map a guard result onto the shared ``gates.Verdict`` schema (deterministic
    floor for the leakage stage). Imported lazily to avoid an import cycle."""
    from .gates import Verdict

    offending = sorted({f.get("column") for f in result.get("findings", []) if f.get("column")})
    return Verdict(
        stage=stage,
        status=result.get("status", "pass"),
        reasons=[result.get("summary", "")],
        suggested_corrections=({"drop_or_recompute_per_fold": offending} if offending else {}),
        checked={**result.get("checked", {}), "lens": "precomputed_target_leakage"},
        critic="code:leakage_guard",
    )
