"""Deep data-pattern analysis for Award B runs.

Analyses train/test schema, time coverage, within-period split patterns,
target distribution, target-by-group aggregations, feature-target correlations,
distribution shift, and validation strategy implications. Generic — no
dataset-specific names or hard-coded column references.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ── public entry point ────────────────────────────────────────────────────────

def run_pattern_analysis(
    schema: Any,
    bundle: Any,
    *,
    repo_root: Path,
    run_id: str,
    period_rank: dict[str, int] | None = None,
    group_keys: list[str] | None = None,
    out_name: str | None = None,
) -> dict[str, Any]:
    """Compute deep data patterns; write the pattern report.

    ``period_rank`` (token→ordinal) lets time detection work when the period
    column is an opaque token that cannot date-parse (resolved upstream from the
    description's date table); ``None`` falls back to date-parsing. ``group_keys``
    are the panel keys (e.g. jurisdiction × category) used for series-length and
    autocorrelation diagnostics. ``out_name`` overrides the output filename
    (default ``{run_id}_data_pattern_report.json``)."""
    train_df: pd.DataFrame = bundle.train_df
    predict_df: pd.DataFrame = bundle.predict_df
    target_col: str = schema.target_column
    row_id_col: str = schema.row_id_column
    time_col: str | None = schema.time_column
    feature_cols: list[str] = bundle.feature_columns
    numeric_cols: list[str] = bundle.numeric_columns
    categorical_cols: list[str] = bundle.categorical_columns
    # Panel group keys for time-series diagnostics: prefer explicit group_keys,
    # else the raw (non-derived) categorical columns present in train.
    if group_keys is None:
        group_keys = [c for c in categorical_cols if "__" not in c and c in train_df.columns]

    report: dict[str, Any] = {"run_id": run_id}

    # ── 1. Schema comparison ─────────────────────────────────────────────────
    train_raw_cols = set(train_df.columns) - {target_col}
    pred_raw_cols = set(predict_df.columns)
    train_only = sorted(train_raw_cols - pred_raw_cols)
    predict_only = sorted(pred_raw_cols - train_raw_cols)
    shared = sorted(train_raw_cols & pred_raw_cols)
    report["schema_comparison"] = {
        "n_train_rows": int(len(train_df)),
        "n_predict_rows": int(len(predict_df)),
        "train_only_columns": train_only,
        "predict_only_columns": predict_only,
        "shared_columns": shared,
        "note": (
            f"{len(train_only)} column(s) only in training data (may include target components); "
            f"{len(predict_only)} column(s) only in prediction data."
        ),
    }

    # ── 2. Time coverage ─────────────────────────────────────────────────────
    time_coverage = _analyze_time_coverage(
        train_df, predict_df, time_col, row_id_col, period_rank=period_rank)
    report["time_coverage"] = time_coverage

    # ── 3. Target distribution ───────────────────────────────────────────────
    report["target_distribution"] = _analyze_target(train_df, target_col)

    # ── 4. Target by time-derived features ───────────────────────────────────
    time_feat_cols = [
        c for c in feature_cols
        if "__" in c
        and any(t in c for t in ["hour", "dayofweek", "month", "day", "weekofyear", "quarter", "year"])
        and c in train_df.columns
    ]
    report["target_by_time_features"] = _target_by_groups(train_df, target_col, time_feat_cols[:8])

    # ── 5. Target by categorical features ────────────────────────────────────
    raw_cat_cols = [
        c for c in categorical_cols
        if "__" not in c and c in train_df.columns
    ]
    report["target_by_categorical"] = _target_by_groups(train_df, target_col, raw_cat_cols[:6])

    # ── 6. Numeric feature-target correlations ────────────────────────────────
    report["feature_target_correlations"] = _feature_correlations(
        train_df, feature_cols, target_col, numeric_cols
    )

    # ── 7. Distribution shift: train vs prediction features ─────────────────
    report["distribution_shift"] = _distribution_shift(
        train_df, predict_df, feature_cols, numeric_cols
    )

    # ── 8. High-cardinality ID risk ──────────────────────────────────────────
    report["high_cardinality_risk"] = _high_cardinality_check(
        train_df, feature_cols, len(train_df)
    )

    # ── 9. Train-only columns: target-component detection ────────────────────
    report["target_component_analysis"] = _target_component_check(
        train_df, target_col, train_only
    )

    # ── 10. Cross-feature × target interactions ─────────────────────────────
    report["cross_feature_target_patterns"] = _cross_feature_target_patterns(
        train_df, target_col, feature_cols
    )

    # ── 11. Validation strategy implications ─────────────────────────────────
    report["validation_implications"] = _validation_implications(
        time_coverage,
        report["target_distribution"],
        report["distribution_shift"],
    )

    # ── 12. Time-series shape (series length) + target autocorrelation ───────
    ts_time_col = time_col or row_id_col
    report["is_timeseries"] = bool(time_coverage.get("has_time"))
    report["time_series_shape"] = _time_series_shape(
        train_df, ts_time_col, group_keys, period_rank)
    report["target_autocorrelation"] = _target_autocorrelation(
        train_df, target_col, group_keys, ts_time_col, period_rank)

    # ── Write ────────────────────────────────────────────────────────────────
    logs_dir = repo_root / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / (out_name or f"{run_id}_data_pattern_report.json")
    out_path.write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )
    return report


# ── time coverage ─────────────────────────────────────────────────────────────

def _parse_dt(series: pd.Series) -> pd.Series:
    """Datetime-parse using the *distinct* values only, then map back.

    Opaque / low-cardinality keys (e.g. a hashed period id) otherwise force
    ``pd.to_datetime`` into a per-row ``dateutil`` fallback over every row — very
    slow on large panels. Parsing the handful of distinct values is equivalent
    and fast, and costs nothing extra when the values really are all distinct.
    """
    s = series.astype(str)
    uniq = pd.unique(s)
    parsed = pd.to_datetime(pd.Series(uniq), errors="coerce")
    mapping = dict(zip(uniq, parsed))
    return pd.to_datetime(s.map(mapping), errors="coerce")


def _analyze_time_coverage(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    time_col: str | None,
    row_id_col: str | None,
    period_rank: dict[str, int] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"has_time": False}
    col = time_col or row_id_col
    if not col:
        return result
    # Opaque-token path: when the period column cannot date-parse but an upstream
    # token→ordinal map is supplied, recognise the chronological structure via the
    # ordinal (fixes the false `has_time=False` on hashed period ids).
    if period_rank and col in train_df.columns:
        result["ordinal_source"] = "period_rank"
        result["temporal_column"] = col
        tr = train_df[col].astype(str).map(period_rank).dropna()
        if len(tr):
            result["has_time"] = True
            result["train_min_rank"] = int(tr.min())
            result["train_max_rank"] = int(tr.max())
            result["train_n_unique_periods"] = int(tr.nunique())
        if col in predict_df.columns:
            pr = predict_df[col].astype(str).map(period_rank).dropna()
            if len(pr):
                result["predict_min_rank"] = int(pr.min())
                result["predict_max_rank"] = int(pr.max())
                result["predict_n_unique_periods"] = int(pr.nunique())
                if len(tr):
                    result["temporal_overlap"] = not (
                        tr.max() < pr.min() or pr.max() < tr.min())
                    result["predict_after_train"] = bool(pr.min() > tr.max())
        return result
    for frame_name, df in [("train", train_df), ("predict", predict_df)]:
        if col not in df.columns:
            continue
        parsed = _parse_dt(df[col])
        if parsed.notna().mean() < 0.6:
            continue
        result["has_time"] = True
        result[f"{frame_name}_min"] = str(parsed.min())
        result[f"{frame_name}_max"] = str(parsed.max())
        result[f"{frame_name}_n_unique_timestamps"] = int(parsed.nunique())

    if not result.get("has_time"):
        return result

    result["temporal_column"] = col

    # Detect overlap between train and predict time ranges
    if "train_min" in result and "predict_min" in result:
        t_min = pd.to_datetime(result["train_min"])
        t_max = pd.to_datetime(result["train_max"])
        p_min = pd.to_datetime(result["predict_min"])
        p_max = pd.to_datetime(result["predict_max"])
        result["temporal_overlap"] = not (t_max < p_min or p_max < t_min)
        result["predict_after_train"] = p_min > t_max

    # Within-period pattern: check if train and predict partition on a sub-unit
    if col in train_df.columns and col in predict_df.columns:
        train_parsed = _parse_dt(train_df[col])
        pred_parsed = _parse_dt(predict_df[col])
        if train_parsed.notna().mean() >= 0.6 and pred_parsed.notna().mean() >= 0.6:
            for attr in ["day", "hour", "dayofweek", "month", "weekofyear"]:
                try:
                    train_vals = set(getattr(train_parsed.dt, attr).dropna().unique().tolist())
                    pred_vals = set(getattr(pred_parsed.dt, attr).dropna().unique().tolist())
                except AttributeError:
                    continue
                overlap = train_vals & pred_vals
                if len(overlap) == 0 and len(train_vals) >= 2 and len(pred_vals) >= 2:
                    tv, pv = sorted(train_vals), sorted(pred_vals)
                    if tv[-1] < pv[0]:
                        result["within_period_pattern"] = {
                            "feature": attr,
                            "train_range": [int(min(tv)), int(max(tv))],
                            "predict_range": [int(min(pv)), int(max(pv))],
                            "direction": "train_lower",
                            "implication": (
                                f"Training covers {attr} {int(min(tv))}–{int(max(tv))}, "
                                f"prediction covers {attr} {int(min(pv))}–{int(max(pv))}. "
                                f"Use within-period holdout: hold out training rows with "
                                f"{attr} ≥ {int(max(tv)) - max(1, (int(max(tv))-int(min(tv)))//5)}."
                            ),
                        }
                        break

    return result


# ── target distribution ───────────────────────────────────────────────────────

def _analyze_target(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    if target_col not in df.columns:
        return {"error": "target column not found"}
    series = pd.to_numeric(df[target_col], errors="coerce").dropna()
    if len(series) == 0:
        return {"error": "no numeric target values"}

    q = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]
    quantiles = {f"p{int(qi*100)}": float(series.quantile(qi)) for qi in q}
    zero_frac = float((series == 0).mean())
    near_zero_frac = float((series.abs() < 1e-6).mean())

    try:
        from scipy import stats as scipy_stats
        skewness = float(scipy_stats.skew(series.values))
        kurtosis = float(scipy_stats.kurtosis(series.values))
    except Exception:
        skewness = float(series.skew())
        kurtosis = float(series.kurt())

    result: dict[str, Any] = {
        "count": int(len(series)),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "skewness": skewness,
        "kurtosis": kurtosis,
        "zero_fraction": zero_frac,
        "near_zero_fraction": near_zero_frac,
        "quantiles": quantiles,
        "is_non_negative": bool(series.min() >= 0),
        "is_right_skewed": bool(skewness > 1.5),
        "interpretation": _skew_interpretation(skewness, zero_frac),
    }
    return result


def _skew_interpretation(skewness: float, zero_frac: float) -> str:
    parts = []
    if skewness > 2.0:
        parts.append("heavily right-skewed — log-transform of target may help")
    elif skewness > 1.0:
        parts.append("moderately right-skewed")
    elif skewness < -1.0:
        parts.append("left-skewed")
    else:
        parts.append("approximately symmetric")
    if zero_frac > 0.20:
        parts.append(f"{zero_frac:.1%} zero values (may indicate zero-inflated distribution)")
    return "; ".join(parts)


# ── target by groups ──────────────────────────────────────────────────────────

def _target_by_groups(
    df: pd.DataFrame, target_col: str, group_cols: list[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if target_col not in df.columns:
        return result
    target = pd.to_numeric(df[target_col], errors="coerce")
    for col in group_cols:
        if col not in df.columns:
            continue
        n_unique = int(df[col].nunique(dropna=True))
        if n_unique < 2 or n_unique > 48:
            continue
        try:
            grouped = (
                pd.DataFrame({"g": df[col].reset_index(drop=True), "t": target.reset_index(drop=True)})
                .dropna()
                .groupby("g")["t"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            mean_range = float(grouped["mean"].max() - grouped["mean"].min())
            result[col] = {
                "n_groups": n_unique,
                "target_mean_range": round(mean_range, 4),
                "target_mean_std_across_groups": round(float(grouped["mean"].std()), 4),
                "is_predictive": mean_range > float(target.std()) * 0.5,
                "top_groups": [
                    {
                        "group": _py(row["g"]),
                        "mean": round(float(row["mean"]), 4),
                        "count": int(row["count"]),
                    }
                    for _, row in grouped.nlargest(5, "mean").iterrows()
                ],
            }
        except Exception:
            pass
    return result


# ── feature-target correlations ───────────────────────────────────────────────

def _feature_correlations(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    numeric_cols: list[str],
) -> dict[str, Any]:
    if target_col not in df.columns:
        return {}
    target = pd.to_numeric(df[target_col], errors="coerce")
    corrs: list[dict] = []
    for col in feature_cols:
        if col not in df.columns or col not in numeric_cols:
            continue
        feat = pd.to_numeric(df[col], errors="coerce")
        pair = pd.concat([feat, target], axis=1).dropna()
        if len(pair) < 10 or pair.iloc[:, 0].std() == 0:
            continue
        r = float(pair.corr().iloc[0, 1])
        if np.isfinite(r):
            corrs.append({"feature": col, "pearson_r": round(r, 4), "abs_r": round(abs(r), 4)})
    corrs.sort(key=lambda x: x["abs_r"], reverse=True)
    return {
        "top_positive": [c for c in corrs if c["pearson_r"] > 0][:10],
        "top_negative": [c for c in corrs if c["pearson_r"] < 0][:5],
        "n_features_computed": len(corrs),
    }


# ── distribution shift ────────────────────────────────────────────────────────

def _distribution_shift(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    numeric_cols: list[str],
) -> dict[str, Any]:
    shifts: list[dict] = []
    for col in feature_cols:
        if col not in train_df.columns or col not in predict_df.columns:
            continue
        if col not in numeric_cols:
            continue
        t_vals = pd.to_numeric(train_df[col], errors="coerce").dropna()
        p_vals = pd.to_numeric(predict_df[col], errors="coerce").dropna()
        if len(t_vals) < 5 or len(p_vals) < 5:
            continue
        t_mean, t_std = float(t_vals.mean()), float(t_vals.std())
        p_mean = float(p_vals.mean())
        if t_std > 0:
            z = abs(t_mean - p_mean) / t_std
            if z > 1.0:
                shifts.append({
                    "feature": col,
                    "train_mean": round(t_mean, 4),
                    "predict_mean": round(p_mean, 4),
                    "z_score": round(z, 4),
                    "severity": "high" if z > 3.0 else ("medium" if z > 2.0 else "low"),
                })
    shifts.sort(key=lambda x: x["z_score"], reverse=True)
    return {
        "shifted_features": shifts[:10],
        "n_features_with_shift": len(shifts),
        "has_notable_shift": any(s["severity"] in ("medium", "high") for s in shifts),
    }


# ── high-cardinality detection ────────────────────────────────────────────────

def _high_cardinality_check(
    df: pd.DataFrame, feature_cols: list[str], n_rows: int
) -> list[dict]:
    risky: list[dict] = []
    for col in feature_cols:
        if col not in df.columns or "__" in col:
            continue
        n_unique = int(df[col].nunique(dropna=True))
        frac = n_unique / max(n_rows, 1)
        if frac > 0.50:
            parsed = _parse_dt(df[col])
            is_dt = parsed.notna().mean() >= 0.6
            if not is_dt:
                risky.append({
                    "feature": col,
                    "n_unique": n_unique,
                    "cardinality_fraction": round(frac, 4),
                    "risk": "high — likely an ID column; remove from feature set",
                })
    return risky


# ── target-component detection ────────────────────────────────────────────────

def _target_component_check(
    df: pd.DataFrame, target_col: str, train_only_cols: list[str]
) -> dict[str, Any]:
    """Identify train-only columns that are likely components of the target."""
    if target_col not in df.columns or not train_only_cols:
        return {"components": [], "note": "No train-only columns to analyze."}

    target = pd.to_numeric(df[target_col], errors="coerce").dropna()
    components: list[dict] = []

    for col in train_only_cols:
        if col not in df.columns:
            continue
        col_data = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(col_data) < 10:
            continue
        pair = pd.concat(
            [col_data.rename("col"), target.rename("tgt")], axis=1
        ).dropna()
        if len(pair) < 10 or pair["col"].std() == 0:
            continue
        r = float(pair.corr().iloc[0, 1])
        if not np.isfinite(r):
            continue

        # Risk tier based on |r| threshold
        abs_r = abs(r)
        if abs_r > 0.80:
            risk_tier = "high"
        elif abs_r > 0.50:
            risk_tier = "medium"
        elif abs_r > 0.30:
            risk_tier = "low"
        else:
            continue  # Not associated enough to report

        is_component = risk_tier in ("high", "medium")

        # Additive component check: does col / target approximate a fraction ≤ 1?
        additive_check = None
        if is_component and pair["tgt"].std() > 0:
            ratio = float((pair["col"] / pair["tgt"].replace(0, np.nan)).dropna().mean())
            additive_check = 0.0 < ratio < 1.01

        # Always absent from prediction data by definition (train-only columns)
        excluded_correctly = True

        entry = {
            "column": col,
            "pearson_r_with_target": round(r, 4),
            "risk_tier": risk_tier,
            "is_component": is_component,
            "likely_additive_component": bool(additive_check) if additive_check is not None else None,
            "excluded_correctly": excluded_correctly,
            "verdict": (
                "EXCLUDE — train-only column with high target correlation; leakage if used as feature"
                if risk_tier == "high" else
                f"CAUTION — train-only column with moderate target correlation (r={r:.4f}); "
                "correctly excluded from model since absent from prediction data"
            ),
        }
        if is_component:
            components.append(entry)
        else:
            # Still record low-risk ones in a separate list for full transparency
            components.append(entry)  # include all tiers; report writer can filter by is_component

    high_medium = [c for c in components if c.get("risk_tier") in ("high", "medium")]
    return {
        "components": components,
        "high_risk_components": high_medium,
        "n_components_detected": len(high_medium),
        "note": (
            f"{len(high_medium)} train-only column(s) detected as likely target components "
            "(|r| > 0.50). "
            "These are correctly excluded from the model feature set since they are absent "
            "from the prediction data."
        ) if high_medium else "No high/medium-risk target-component columns detected among train-only columns.",
    }


# ── cross-feature × target interactions ──────────────────────────────────────

def _cross_feature_target_patterns(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Compute generic interaction patterns: time-derived features × low-cardinality features.

    Detects whether the target's relationship with a time feature differs
    significantly across levels of a binary/low-cardinality feature. Generic —
    uses only feature naming conventions (``__`` = time-derived) and value counts.
    """
    if target_col not in df.columns:
        return {}

    target = pd.to_numeric(df[target_col], errors="coerce")
    if target.isna().all():
        return {}

    # Identify time-derived features in the feature set (have __ in name)
    time_feats = [
        c for c in feature_cols
        if "__" in c and c in df.columns
        and any(t in c for t in ["hour", "dayofweek", "month", "day", "weekofyear", "quarter"])
    ]

    # Identify low-cardinality non-time features (2–10 unique numeric values)
    low_card_feats: list[str] = []
    for c in feature_cols:
        if c in df.columns and "__" not in c:
            series = pd.to_numeric(df[c], errors="coerce").dropna()
            n_unique = int(series.nunique())
            if 2 <= n_unique <= 10:
                low_card_feats.append(c)

    interactions: list[dict] = []
    for tf in time_feats[:6]:  # limit to avoid combinatorial explosion
        tf_series = pd.to_numeric(df[tf], errors="coerce")
        n_tf_unique = int(tf_series.nunique(dropna=True))
        if n_tf_unique < 2 or n_tf_unique > 48:
            continue
        for lc in low_card_feats[:8]:
            lc_series = pd.to_numeric(df[lc], errors="coerce")
            lc_vals = sorted(lc_series.dropna().unique().tolist())
            if len(lc_vals) < 2:
                continue
            try:
                # Compute target mean per (time_feat, low_card_feat) level
                frame = pd.DataFrame({
                    "tf": tf_series.reset_index(drop=True),
                    "lc": lc_series.reset_index(drop=True),
                    "tgt": target.reset_index(drop=True),
                }).dropna()
                if len(frame) < 20:
                    continue
                # Mean target per time-feature value, separately per low-card level
                level_curves: dict[str, list] = {}
                for lc_val in lc_vals:
                    sub = frame[frame["lc"] == lc_val]
                    if len(sub) < 4:
                        continue
                    curve = sub.groupby("tf")["tgt"].mean()
                    if len(curve) >= 2:
                        level_curves[str(lc_val)] = {
                            "mean_range": float(curve.max() - curve.min()),
                            "mean_of_group_means": float(curve.mean()),
                        }
                if len(level_curves) < 2:
                    continue
                ranges = [v["mean_range"] for v in level_curves.values()]
                group_means = [v["mean_of_group_means"] for v in level_curves.values()]
                interaction_strength = float(max(ranges) - min(ranges))
                level_spread = float(max(group_means) - min(group_means))
                if interaction_strength > 1e-6 or level_spread > 1e-6:
                    interactions.append({
                        "time_feature": tf,
                        "binary_feature": lc,
                        "interaction_strength": round(interaction_strength, 4),
                        "level_spread": round(level_spread, 4),
                        "level_curves": level_curves,
                    })
            except Exception:
                pass

    interactions.sort(key=lambda x: x["interaction_strength"] + x["level_spread"], reverse=True)
    return {
        "top_interactions": interactions[:3],
        "n_interactions_tested": len(interactions),
        "note": (
            "Top interaction effects between time features and low-cardinality features. "
            "interaction_strength = max_level_time_range - min_level_time_range; "
            "level_spread = max_level_mean - min_level_mean."
        ),
    }


# ── validation implications ───────────────────────────────────────────────────

def _validation_implications(
    time_coverage: dict,
    target_dist: dict,
    dist_shift: dict,
) -> dict[str, Any]:
    recommendations: list[str] = []

    if time_coverage.get("within_period_pattern"):
        p = time_coverage["within_period_pattern"]
        recommendations.append(
            f"Within-period split detected ({p['feature']}): use within-period holdout "
            f"(hold out last ~20% of training {p['feature']} values) to simulate "
            f"how prediction rows differ from training rows."
        )
    elif time_coverage.get("has_time") and not time_coverage.get("temporal_overlap"):
        recommendations.append(
            "Prediction rows are temporally after training rows: use time-based holdout "
            "(hold out last 20% of training timestamps) rather than random split."
        )
    elif time_coverage.get("has_time"):
        recommendations.append(
            "Temporal overlap detected between train and prediction: use time-ordered "
            "holdout to avoid future leakage in validation."
        )

    if target_dist.get("is_right_skewed"):
        recommendations.append(
            f"Target is right-skewed (skewness={target_dist.get('skewness', '?'):.2f}): "
            "consider log-transform; MAE on raw target will weight large values heavily."
        )

    if dist_shift.get("has_notable_shift"):
        shifted = [s["feature"] for s in dist_shift.get("shifted_features", [])
                   if s["severity"] in ("medium", "high")]
        if shifted:
            recommendations.append(
                f"Distribution shift detected in: {', '.join(shifted[:5])}. "
                "Validation score may not represent out-of-distribution performance."
            )

    chosen_strategy = "within_period_holdout" if time_coverage.get("within_period_pattern") else (
        "time_based_holdout" if time_coverage.get("has_time") else "random_holdout"
    )

    return {
        "recommended_strategy": chosen_strategy,
        "recommendations": recommendations,
    }


# ── time-series shape + autocorrelation (planner-facing) ──────────────────────

def _period_ordinal(df: pd.DataFrame, time_col: str,
                    period_rank: dict[str, int] | None) -> pd.Series | None:
    """Per-row period ordinal: from ``period_rank`` (opaque tokens) when given,
    else a dense rank of the date-parsed column. ``None`` when unavailable."""
    if not time_col or time_col not in df.columns:
        return None
    if period_rank:
        o = df[time_col].astype(str).map(period_rank)
        return o if o.notna().mean() >= 0.5 else None
    parsed = _parse_dt(df[time_col])
    if parsed.notna().mean() >= 0.6:
        return parsed.rank(method="dense")
    return None


def _time_series_shape(
    train_df: pd.DataFrame,
    time_col: str | None,
    group_keys: list[str] | None,
    period_rank: dict[str, int] | None,
) -> dict[str, Any]:
    """Series-length diagnostics so the planner can size lags/rolling windows to
    the data: total periods, rows per period, and per-group series length."""
    if not time_col or time_col not in train_df.columns:
        return {"available": False}
    out: dict[str, Any] = {"available": True}
    n_periods = int(train_df[time_col].nunique(dropna=True))
    out["n_periods"] = n_periods
    out["rows_per_period"] = round(len(train_df) / max(n_periods, 1), 2)
    gk = [g for g in (group_keys or []) if g in train_df.columns]
    if gk:
        try:
            sizes = train_df.groupby(gk)[time_col].nunique()
            out["group_keys"] = gk
            out["per_group_series_length"] = {
                "min": int(sizes.min()),
                "median": float(sizes.median()),
                "max": int(sizes.max()),
                "n_groups": int(len(sizes)),
            }
        except Exception:
            pass
    return out


def _target_autocorrelation(
    train_df: pd.DataFrame,
    target_col: str,
    group_keys: list[str] | None,
    time_col: str | None,
    period_rank: dict[str, int] | None,
    lags: tuple[int, ...] = (1, 3, 6, 12),
) -> dict[str, Any]:
    """Within-group, period-ordered target autocorrelation at representative lags,
    averaged across panel groups. Tells the planner WHICH lag features carry signal
    and how to size rolling windows. Advisory only — never a model feature."""
    if target_col not in train_df.columns or not time_col or time_col not in train_df.columns:
        return {"available": False}
    gk = [g for g in (group_keys or []) if g in train_df.columns]
    if not gk:
        return {"available": False}
    ordi = _period_ordinal(train_df, time_col, period_rank)
    if ordi is None:
        return {"available": False}
    df = train_df[gk].copy()
    df["_ord"] = ordi.to_numpy()
    df["_t"] = pd.to_numeric(train_df[target_col], errors="coerce").to_numpy()
    df = df.dropna(subset=["_ord", "_t"])
    if len(df) < 20:
        return {"available": False}
    # Pre-build each group's period-ordered, gap-filled target series once, so a
    # lag respects real period spacing (NaN at missing periods) and Series.autocorr
    # computes corr(s, s.shift(lag)) correctly.
    series_by_group: list[pd.Series] = []
    for _, g in df.groupby(gk):
        s = g.sort_values("_ord").drop_duplicates("_ord").set_index("_ord")["_t"]
        if len(s) < 4 or s.std(skipna=True) == 0:
            continue
        lo, hi = int(s.index.min()), int(s.index.max())
        series_by_group.append(s.reindex(range(lo, hi + 1)))
    acf: dict[str, float] = {}
    for lag in lags:
        rs: list[float] = []
        for s in series_by_group:
            if s.notna().sum() <= lag + 2:
                continue
            r = s.autocorr(lag)  # corr(s, s.shift(lag)), NaN-aware
            if r is not None and np.isfinite(r):
                rs.append(float(r))
        if rs:
            acf[f"lag{lag}"] = round(float(np.mean(rs)), 4)
    if not acf:
        return {"available": False}
    strongest = sorted(acf.items(), key=lambda kv: abs(kv[1]), reverse=True)
    out: dict[str, Any] = {"available": True, "method": "within_group_pearson",
                           "n_groups_used": int(df.groupby(gk).ngroups)}
    out.update(acf)
    # lags whose mean ACF is materially positive (signal worth a lag feature)
    out["strongest_lags"] = [int(k.replace("lag", "")) for k, v in strongest if v >= 0.2]
    return out


# ── helpers ───────────────────────────────────────────────────────────────────

def _py(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _json_default(value: Any) -> Any:
    import numpy as np
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, pd.NaT.__class__)):
        return str(value)
    return str(value)
