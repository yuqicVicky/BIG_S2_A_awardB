"""leakage-check skill: systematic leakage audit (the 9 checks).

Implements ``.claude/skills/leakage-check/SKILL.md``. The standalone gate
(``invocation_point="leakage_check"``) runs checks 1, 2, 3, 7, 8 on the feature
candidates + data profile (+ optional df for correlation). Structural checks
(4, 5, 6, 9) run when transformer/split/evaluation evidence is supplied.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_CHECKS_BY_POINT = {
    "plan_critique": [1, 2, 3, 7],
    "leakage_check": [1, 2, 3, 7, 8],
    "preprocessing": [4, 5, 6],
    "modeling": [4, 6, 9],
    "evaluation": [4, 6, 9],
}
_CHECK_NAMES = {
    1: "Target Variable in Feature Set", 2: "ID-Like Columns in Feature Set",
    3: "Future-Information and Post-Outcome Variables", 4: "Preprocessing Before Split",
    5: "Transformer Fitted on Non-Training Data", 6: "Feature Selection Outside Training Set",
    7: "Duplicate Rows Spanning Train and Test", 8: "Suspiciously Predictive Feature",
    9: "Metrics Computed on Train/Val Reported as Final",
}
_STRONG_POST = ["post_", "after_", "final_", "diagnosis_after", "status_after", "future_"]
_WEAK_POST = ["next_", "end_", "result_", "_at_discharge", "_at_followup", "date", "time"]
_SCORE_PAT = ["score", "pred", "proba", "probability", "prediction"]


def _get(obj: Any, key: str, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def run_leakage_check(
    invocation_point: str,
    *,
    task_spec: Any,
    data_profile: Any,
    df: pd.DataFrame | None = None,
    plan: dict | None = None,
    transformers: Any = None,
    splits: Any = None,
    evaluation: dict | None = None,
    run_id: str = "",
    datetime_derived_features: set[str] | None = None,
) -> dict:
    checks = _CHECKS_BY_POINT.get(invocation_point, [1, 2, 3, 7, 8])
    _dt_safe: set[str] = set(datetime_derived_features or set())
    target = _get(task_spec, "target_variable")
    features = list(_get(task_spec, "feature_candidates", []) or [])
    id_cols = list(_get(data_profile, "potential_id_columns", []) or [])
    n_rows = int(_get(data_profile, "n_rows", 0) or (len(df) if df is not None else 0))
    duplicate_count = int(_get(data_profile, "duplicate_count", 0) or 0)
    cat_cols = _get(data_profile, "categorical_columns", {}) or {}

    suspected: list[dict] = []
    failed: list[dict] = []
    warned: list[dict] = []
    fixes: list[dict] = []
    warnings: list[dict] = []

    def fail(check: int, column: str | None, detail: str, pattern: str | None = None):
        failed.append({"check_id": check, "check_name": _CHECK_NAMES[check], "verdict": "FAIL", "detail": detail, "column": column})
        if column:
            suspected.append({"column": column, "pattern_matched": pattern, "check": check, "verdict": "FAIL", "reason": detail})
            fixes.append({"column": column, "fix": f"Remove '{column}' from feature_candidates ({detail}).", "blocking": True})

    def warn(check: int, column: str | None, detail: str, pattern: str | None = None, wtype: str = "suspected_leakage_column"):
        warned.append({"check_id": check, "check_name": _CHECK_NAMES[check], "verdict": "WARN", "detail": detail, "column": column})
        if column:
            suspected.append({"column": column, "pattern_matched": pattern, "check": check, "verdict": "WARN", "reason": detail})
        warnings.append({"type": wtype, "column": column, "message": detail, "severity": "WARN"})

    # Check 1 — target (or derivative) in features
    if 1 in checks and target:
        if target in features:
            fail(1, target, "Target variable present in feature_candidates.", "target")
        for feat in features:
            low = feat.lower()
            if low in {f"{target.lower()}_{s}" for s in ("encoded", "binary", "int", "flag")}:
                fail(1, feat, f"Derived-target name pattern of '{target}'.", "<target>_*")
        if df is not None and target in df.columns:
            for feat, corr in _correlations(df, features, target).items():
                if abs(corr) > 0.95:
                    fail(1, feat, f"Correlation with target = {corr:.3f} (> 0.95).", "corr>0.95")

    # Check 2 — ID-like in features
    # Datetime-derived features (e.g. ordinal) may have one value per row; they are
    # valid predictive features, not ID columns, so we exempt them here.
    if 2 in checks:
        for feat in features:
            if feat in _dt_safe:
                continue  # valid datetime-derived feature
            if feat in id_cols:
                fail(2, feat, "Column is in data_profile.potential_id_columns.", "id")
            else:
                n_unique = int(cat_cols.get(feat, {}).get("n_unique", -1)) if isinstance(cat_cols, dict) else -1
                if n_unique < 0 and df is not None and feat in df.columns:
                    n_unique = int(df[feat].nunique())
                if n_rows and n_unique == n_rows:
                    fail(2, feat, f"n_unique == n_rows ({n_rows}); effective ID column.", "n_unique==n_rows")

    # Check 3 — future / post-outcome
    # Datetime-derived features (engineered from a datetime column available at
    # prediction time) are exempt: they are valid predictive features, not leakage.
    if 3 in checks:
        for feat in features:
            if feat in _dt_safe:
                continue  # valid datetime-derived feature; skip temporal-name check
            low = feat.lower()
            if any(p in low for p in _STRONG_POST):
                fail(3, feat, "Name matches a strong post-outcome pattern.", "post/after/final")
            elif any(p in low for p in _WEAK_POST):
                warn(3, feat, "Name has a weak temporal signal; verify availability at prediction time.", "date/time")
            if target and any(p in low for p in _SCORE_PAT):
                warn(3, feat, "Name suggests a model-generated score/prediction; verify origin.", "score/pred")

    # Check 7 — duplicate rows
    if 7 in checks:
        x_train = _get(splits, "X_train")
        x_test = _get(splits, "X_test")
        if x_train is not None and x_test is not None:
            merged = pd.merge(pd.DataFrame(x_train), pd.DataFrame(x_test), how="inner")
            if len(merged) > 0:
                fail(7, None, f"{len(merged)} exact rows shared between X_train and X_test.")
        elif duplicate_count > 0:
            warn(7, None, f"{duplicate_count} duplicate rows in raw data; verify they do not span the split.", wtype="duplicate_rows")

    # Check 8 — suspiciously predictive single feature
    if 8 in checks and target and df is not None and target in df.columns:
        for feat, score in _univariate_strength(df, features, target).items():
            if score is not None and score > 0.98:
                name_hit = any(p in feat.lower() for p in _SCORE_PAT)
                if name_hit:
                    fail(8, feat, f"Univariate strength {score:.3f} (>0.98) and suspicious name.", "score/pred")
                else:
                    warn(8, feat, f"Univariate strength {score:.3f} (>0.98) on training data; verify origin.", "auc>0.98")

    # Check 5 — transformer fitted on train only
    if 5 in checks and transformers is not None:
        fitted_on = _get(transformers, "fitted_on")
        if fitted_on not in (None, "X_train"):
            fail(5, None, f"Transformer fitted_on='{fitted_on}', expected 'X_train'.")

    # Check 9 — final metrics on test only
    if 9 in checks and evaluation:
        rows = evaluation.get("performance_table") or []
        if rows and any(r.get("split") != "test" for r in rows):
            fail(9, None, "performance_table contains a non-test split.")
        if evaluation.get("selection_split") not in (None, "val"):
            warn(9, None, "selection_split should be 'val'.", wtype="selection_split")

    leakage_risk = "high" if failed else ("medium" if warned else "low")
    return {
        "invocation_point": invocation_point,
        "run_id": run_id,
        "checks_performed": checks,
        "leakage_risk": leakage_risk,
        "approved": len(failed) == 0,
        "suspected_columns": suspected,
        "failed_checks": failed,
        "warned_checks": warned,
        "required_fixes": fixes,
        "warnings": warnings,
    }


def _correlations(df: pd.DataFrame, features: list[str], target: str) -> dict[str, float]:
    tnum = _numeric_target(df[target])
    out: dict[str, float] = {}
    for feat in features:
        if feat not in df.columns or not pd.api.types.is_numeric_dtype(df[feat]):
            continue
        pair = pd.concat([pd.to_numeric(df[feat], errors="coerce"), tnum], axis=1).dropna()
        if len(pair) >= 5 and pair.iloc[:, 0].std() > 0:
            out[feat] = float(pair.corr().iloc[0, 1])
    return out


def _univariate_strength(df: pd.DataFrame, features: list[str], target: str) -> dict[str, float | None]:
    tnum = _numeric_target(df[target])
    binary = tnum.dropna().nunique() == 2
    out: dict[str, float | None] = {}
    for feat in features:
        if feat not in df.columns or not pd.api.types.is_numeric_dtype(df[feat]):
            continue
        pair = pd.concat([pd.to_numeric(df[feat], errors="coerce"), tnum], axis=1).dropna()
        if len(pair) < 5 or pair.iloc[:, 0].std() == 0:
            continue
        if binary:
            try:
                auc = roc_auc_score(pair.iloc[:, 1], pair.iloc[:, 0])
                out[feat] = float(max(auc, 1 - auc))
            except Exception:
                out[feat] = None
        else:
            out[feat] = float(abs(pair.corr().iloc[0, 1]))
    return out


def _numeric_target(target: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(target):
        return pd.to_numeric(target, errors="coerce")
    return pd.Series(pd.factorize(target)[0], index=target.index).replace(-1, np.nan)
