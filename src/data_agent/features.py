"""Data loading and feature frame construction.

Datetime detection and feature extraction are intentionally generic:
any column that can be parsed as a datetime — including the row_id
column and join keys — is treated as a potential source of time
features (hour, dayofweek, day, month, year, is_weekend, quarter,
weekofyear, cyclical encodings, ordinal).  The raw column is NOT
added to the model feature set; only the derived ``col__<field>``
columns are.  This avoids target leakage from ID columns while
preserving temporal information for any unknown future dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import SchemaSpec, read_table
from .task import TaskSpec, resolve_task_spec


@dataclass
class FeatureBundle:
    train_df: pd.DataFrame
    predict_df: pd.DataFrame
    sample_submission: pd.DataFrame
    target: pd.Series
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    task: TaskSpec
    profile: dict[str, Any]


LEAKAGE_NAME_TOKENS = [
    "target",
    "label",
    "truth",
    "actual",
    "observed",
    "future",
    "post",
    "after",
    "final",
    "prediction",
    "predicted",
]

# Minimum fraction of values that must parse as datetime for a column to be
# treated as a datetime source.  Applied to a sample of up to 200 rows.
_DATETIME_PARSE_THRESHOLD = 0.6

# Maximum number of unique values in a time-derived feature to be included in
# the target-signal audit (guards against high-cardinality columns such as
# the full ordinal index).
_AUDIT_MAX_GROUPS = 50


# ── public entry point ────────────────────────────────────────────────────────

def build_feature_bundle(spec: SchemaSpec) -> FeatureBundle:
    target_df = read_table(spec.train_target_file)
    sample_submission = read_table(spec.sample_submission_file)
    train_cov = read_table(spec.train_covariates_file) if spec.train_covariates_file else None
    val_cov = (
        read_table(spec.validation_covariates_file)
        if spec.validation_covariates_file
        else None
    )

    train_df = _merge_train(target_df, train_cov, spec)
    predict_df = _merge_prediction(sample_submission, val_cov, spec)

    # ── datetime feature extraction ───────────────────────────────────────────
    # Detect datetime-like columns in the combined frame (including row_id and
    # join keys that would otherwise be excluded from model features).
    all_dt_cols = _detect_datetime_columns(train_df)
    time_sources = _resolve_time_sources(all_dt_cols, spec)
    train_df, predict_df, time_audit = _extract_time_features(train_df, predict_df, time_sources)

    if spec.target_column not in train_df.columns:
        raise ValueError(f"Target column '{spec.target_column}' not found after training merge.")

    feature_columns, excluded_columns = _choose_feature_columns(train_df, predict_df, spec)
    if not feature_columns:
        raise ValueError("No usable feature columns were inferred.")

    # ── interaction feature generation ───────────────────────────────────────
    # Adds numeric products of time-derived × low-cardinality features. Generic:
    # detection is purely by __ naming and value counts, not column names.
    target_for_skew = train_df.get(spec.target_column)
    skewness = float(pd.to_numeric(target_for_skew, errors="coerce").dropna().skew()) if target_for_skew is not None else 0.0
    train_df, predict_df, new_interaction_cols = _add_interaction_features(
        train_df, predict_df, feature_columns, skewness
    )
    for col in new_interaction_cols:
        if col not in feature_columns:
            feature_columns.append(col)

    train_aligned = train_df.copy()
    predict_aligned = predict_df.copy()
    for col in feature_columns:
        if col not in train_aligned.columns:
            train_aligned[col] = np.nan
        if col not in predict_aligned.columns:
            predict_aligned[col] = np.nan

    numeric_columns = [
        col
        for col in feature_columns
        if pd.api.types.is_numeric_dtype(train_aligned[col])
        and not pd.api.types.is_bool_dtype(train_aligned[col])
    ]
    categorical_columns = [col for col in feature_columns if col not in numeric_columns]

    description_text = _read_description(spec.description_path)
    sample_target = (
        sample_submission[spec.target_column]
        if spec.target_column in sample_submission.columns
        else None
    )
    task = resolve_task_spec(
        description_text,
        train_aligned[spec.target_column],
        sample_target,
        has_block_col=bool(spec.block_column),
        target_column=spec.target_column,
    )

    # ── target-signal audit for time features ─────────────────────────────────
    generated_time_features = [f for fs in time_audit["generated_features"].values() for f in fs]
    time_signal = _audit_time_target_signal(
        train_aligned, train_aligned[spec.target_column], generated_time_features
    )

    profile = {
        "train_rows": int(len(train_aligned)),
        "prediction_rows": int(len(predict_aligned)),
        "sample_submission_rows": int(len(sample_submission)),
        "target_column": spec.target_column,
        "row_id_column": spec.row_id_column,
        "join_keys": spec.join_keys,
        "time_column": spec.time_column,
        "category_column": spec.category_column,
        "block_column": spec.block_column,
        "task_type": task.task_type,
        "metric": task.metric,
        "output_kind": task.output_kind,
        "task_evidence": task.evidence,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "target_summary": _series_summary(train_aligned[spec.target_column]),
        "target_distribution": (
            _class_distribution(train_aligned[spec.target_column]) if task.is_classification else None
        ),
        "missing_rates": {
            col: float(train_aligned[col].isna().mean()) for col in feature_columns
        },
        # ── excluded columns breakdown ────────────────────────────────────────
        "excluded_columns": excluded_columns,
        "train_only_columns": sorted(
            [c for c in train_df.columns if c not in predict_df.columns and c != spec.target_column]
        ),
        # ── datetime / time-feature audit ─────────────────────────────────────
        "feature_audit": {
            "detected_row_id_column": spec.row_id_column,
            "detected_datetime_columns": all_dt_cols,
            "datetime_feature_sources": time_sources,
            "generated_time_features": time_audit["generated_features"],
            "final_feature_columns": feature_columns,
            "time_target_signal": time_signal,
        },
    }

    return FeatureBundle(
        train_df=train_aligned,
        predict_df=predict_aligned,
        sample_submission=sample_submission,
        target=train_aligned[spec.target_column],
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        task=task,
        profile=profile,
    )


# ── datetime detection ────────────────────────────────────────────────────────

def _detect_datetime_columns(df: pd.DataFrame, min_parse_rate: float = _DATETIME_PARSE_THRESHOLD) -> list[str]:
    """Return every column in *df* that can be reliably parsed as a datetime.

    Includes columns that are already datetime64, string columns whose values
    parse as datetime, and object columns with mixed content above the
    threshold.  Pure numeric columns are skipped (epoch integers are not
    treated as datetimes without explicit annotation).
    """
    result: list[str] = []
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            result.append(col)
            continue
        if pd.api.types.is_numeric_dtype(series):
            continue
        # Sample up to 200 non-null values for speed.
        sample = series.dropna().head(200)
        if len(sample) == 0:
            continue
        try:
            parsed = pd.to_datetime(sample, errors="coerce")
            rate = float(parsed.notna().mean())
            if rate >= min_parse_rate:
                result.append(col)
        except Exception:
            pass
    return result


def _resolve_time_sources(detected_dt_cols: list[str], spec: SchemaSpec) -> list[str]:
    """Determine which detected datetime columns should generate time features.

    Priority order:
    1. The explicitly identified time column from schema discovery.
    2. The row_id column (if it is datetime-like — e.g. a ``datetime`` column
       used as the submission row identifier).
    3. Any join key that is datetime-like.
    4. Any remaining detected datetime column that is not the target.

    The raw source column is kept as a *feature source only*; it is never
    added to the model feature set directly (that is enforced by
    ``_choose_feature_columns`` which excludes row_id and target).
    """
    seen: set[str] = set()
    sources: list[str] = []

    def _add(col: str) -> None:
        if col and col not in seen:
            seen.add(col)
            sources.append(col)

    if spec.time_column and spec.time_column in detected_dt_cols:
        _add(spec.time_column)

    # row_id that is also datetime-like → extract time features from it
    if spec.row_id_column and spec.row_id_column in detected_dt_cols:
        _add(spec.row_id_column)

    for key in spec.join_keys:
        if key in detected_dt_cols:
            _add(key)

    # Any remaining datetime column (not the target)
    for col in detected_dt_cols:
        if col != spec.target_column:
            _add(col)

    return sources


# ── time feature extraction ───────────────────────────────────────────────────

def _extract_time_features(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    source_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Extract time features from all identified datetime source columns.

    Returns the augmented DataFrames and an audit dictionary that lists, for
    each source column, the names of the features that were generated.
    """
    audit: dict[str, Any] = {"generated_features": {}}
    train = train_df
    pred = predict_df
    for col in source_cols:
        if col not in train.columns:
            continue
        train, pred, features_added = _add_features_for_datetime_col(train, pred, col)
        audit["generated_features"][col] = features_added
    return train, pred, audit


def _has_time_component(parsed: pd.Series) -> bool:
    """Return True if *parsed* contains sub-day time information (any non-midnight)."""
    non_null = parsed.dropna()
    if len(non_null) == 0:
        return False
    # If ANY value has a non-zero hour, minute, or second → sub-day resolution.
    return bool(
        ((non_null.dt.hour != 0) | (non_null.dt.minute != 0) | (non_null.dt.second != 0)).any()
    )


def _add_features_for_datetime_col(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Extract a comprehensive set of time features from *col* in both frames.

    Generated features (always):
        col__year, col__month, col__month_sin, col__month_cos,
        col__day, col__dayofweek, col__dayofweek_sin, col__dayofweek_cos,
        col__is_weekend, col__quarter, col__weekofyear, col__ordinal

    Generated features (only when sub-day timestamps are detected):
        col__hour, col__hour_sin, col__hour_cos

    The raw *col* value is NOT added to the feature set here — that decision
    belongs to ``_choose_feature_columns``, which excludes row_id and target
    columns but includes the derived ``col__*`` engineered features.
    """
    features_added: list[str] = []

    train_parsed = pd.to_datetime(train_df[col], errors="coerce")
    parse_rate = float(train_parsed.notna().mean()) if len(train_parsed) else 0.0
    if parse_rate < _DATETIME_PARSE_THRESHOLD:
        return train_df, predict_df, features_added

    pred_has_col = col in predict_df.columns
    pred_parsed = pd.to_datetime(predict_df[col], errors="coerce") if pred_has_col else pd.Series(dtype="datetime64[ns]")

    train = train_df.copy()
    pred = predict_df.copy()

    has_sub_day = _has_time_component(train_parsed)

    def _set(frame: pd.DataFrame, parsed: pd.Series, name: str, values: Any) -> None:
        if name not in frame.columns:
            frame[name] = values

    # ── year ──────────────────────────────────────────────────────────────────
    feat = f"{col}__year"
    _set(train, train_parsed, feat, train_parsed.dt.year)
    if pred_has_col:
        _set(pred, pred_parsed, feat, pred_parsed.dt.year)
    features_added.append(feat)

    # ── month (raw + cyclical) ────────────────────────────────────────────────
    feat = f"{col}__month"
    _set(train, train_parsed, feat, train_parsed.dt.month)
    if pred_has_col:
        _set(pred, pred_parsed, feat, pred_parsed.dt.month)
    features_added.append(feat)

    for suffix, values_fn in [
        ("__month_sin", lambda p: np.sin(2 * np.pi * p.dt.month / 12)),
        ("__month_cos", lambda p: np.cos(2 * np.pi * p.dt.month / 12)),
    ]:
        feat = f"{col}{suffix}"
        _set(train, train_parsed, feat, values_fn(train_parsed))
        if pred_has_col:
            _set(pred, pred_parsed, feat, values_fn(pred_parsed))
        features_added.append(feat)

    # ── day of month ──────────────────────────────────────────────────────────
    feat = f"{col}__day"
    _set(train, train_parsed, feat, train_parsed.dt.day)
    if pred_has_col:
        _set(pred, pred_parsed, feat, pred_parsed.dt.day)
    features_added.append(feat)

    # ── day of week (0=Monday, raw + cyclical) ────────────────────────────────
    feat = f"{col}__dayofweek"
    _set(train, train_parsed, feat, train_parsed.dt.dayofweek)
    if pred_has_col:
        _set(pred, pred_parsed, feat, pred_parsed.dt.dayofweek)
    features_added.append(feat)

    for suffix, values_fn in [
        ("__dayofweek_sin", lambda p: np.sin(2 * np.pi * p.dt.dayofweek / 7)),
        ("__dayofweek_cos", lambda p: np.cos(2 * np.pi * p.dt.dayofweek / 7)),
    ]:
        feat = f"{col}{suffix}"
        _set(train, train_parsed, feat, values_fn(train_parsed))
        if pred_has_col:
            _set(pred, pred_parsed, feat, values_fn(pred_parsed))
        features_added.append(feat)

    # ── is_weekend ────────────────────────────────────────────────────────────
    feat = f"{col}__is_weekend"
    _set(train, train_parsed, feat, (train_parsed.dt.dayofweek >= 5).astype(int))
    if pred_has_col:
        _set(pred, pred_parsed, feat, (pred_parsed.dt.dayofweek >= 5).astype(int))
    features_added.append(feat)

    # ── quarter ───────────────────────────────────────────────────────────────
    feat = f"{col}__quarter"
    _set(train, train_parsed, feat, train_parsed.dt.quarter)
    if pred_has_col:
        _set(pred, pred_parsed, feat, pred_parsed.dt.quarter)
    features_added.append(feat)

    # ── week of year ──────────────────────────────────────────────────────────
    feat = f"{col}__weekofyear"
    try:
        train_woy = train_parsed.dt.isocalendar().week.astype(int)
    except AttributeError:
        train_woy = train_parsed.dt.week.astype(int)  # type: ignore[attr-defined]
    _set(train, train_parsed, feat, train_woy)
    if pred_has_col:
        try:
            pred_woy = pred_parsed.dt.isocalendar().week.astype(int)
        except AttributeError:
            pred_woy = pred_parsed.dt.week.astype(int)  # type: ignore[attr-defined]
        _set(pred, pred_parsed, feat, pred_woy)
    features_added.append(feat)

    # ── hour (only when sub-day data present, raw + cyclical) ─────────────────
    if has_sub_day:
        feat = f"{col}__hour"
        _set(train, train_parsed, feat, train_parsed.dt.hour)
        if pred_has_col:
            _set(pred, pred_parsed, feat, pred_parsed.dt.hour)
        features_added.append(feat)

        for suffix, values_fn in [
            ("__hour_sin", lambda p: np.sin(2 * np.pi * p.dt.hour / 24)),
            ("__hour_cos", lambda p: np.cos(2 * np.pi * p.dt.hour / 24)),
        ]:
            feat = f"{col}{suffix}"
            _set(train, train_parsed, feat, values_fn(train_parsed))
            if pred_has_col:
                _set(pred, pred_parsed, feat, values_fn(pred_parsed))
            features_added.append(feat)

    # ── ordinal (temporal ordering proxy) ────────────────────────────────────
    # Built from the union of train and predict values so test timestamps are
    # always in the map.
    feat = f"{col}__ordinal"
    combined_vals = pd.concat(
        [train[[col]], pred[[col]] if pred_has_col else pd.DataFrame({col: []})],
        ignore_index=True,
    )[col]
    ordered = sorted(v for v in combined_vals.dropna().astype(str).unique().tolist())
    ordinal_map = {v: i for i, v in enumerate(ordered)}
    _set(train, train_parsed, feat, train[col].astype(str).map(ordinal_map))
    if pred_has_col:
        _set(pred, pred_parsed, feat, pred[col].astype(str).map(ordinal_map))
    features_added.append(feat)

    return train, pred, features_added


# ── target-signal audit ───────────────────────────────────────────────────────

def _audit_time_target_signal(
    train_df: pd.DataFrame,
    target: pd.Series,
    time_features: list[str],
) -> dict[str, Any]:
    """For each time-derived feature, report target mean/std by group.

    This reveals whether the feature has predictive value (large
    ``target_mean_range``) without fitting any model.  Written to the profile
    ``feature_audit.time_target_signal`` key so the report writer can quote it.
    """
    result: dict[str, Any] = {}
    target_num = pd.to_numeric(target, errors="coerce")
    if target_num.isna().all():
        return result

    for feat in time_features:
        if feat not in train_df.columns:
            continue
        col_data = train_df[feat]
        n_unique = int(col_data.nunique(dropna=True))
        if n_unique < 2 or n_unique > _AUDIT_MAX_GROUPS:
            continue
        try:
            frame = pd.DataFrame({
                "feat": col_data.reset_index(drop=True),
                "target": target_num.reset_index(drop=True),
            }).dropna()
            grouped = frame.groupby("feat")["target"].agg(["mean", "std", "count"]).reset_index()
            result[feat] = {
                "n_groups": n_unique,
                "target_mean_range": float(grouped["mean"].max() - grouped["mean"].min()),
                "target_mean_std_across_groups": float(grouped["mean"].std()),
                "groups": [
                    {
                        "group": _py(row["feat"]),
                        "mean": float(row["mean"]),
                        "std": float(row["std"]) if not np.isnan(row["std"]) else None,
                        "count": int(row["count"]),
                    }
                    for _, row in grouped.iterrows()
                ],
            }
        except Exception:
            pass
    return result


# ── interaction feature generation ────────────────────────────────────────────

def _add_interaction_features(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_columns: list[str],
    skewness: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Generate numeric product interactions: time-derived × low-cardinality features.

    Only uses __ naming detection and value-count detection. No hardcoded column names.
    Returns augmented train, augmented predict, and list of added column names.
    Limited to at most 6 new interaction columns.
    """
    # Detect time-derived features in the current feature set
    time_feats = [
        c for c in feature_columns
        if "__" in c and c in train_df.columns
        and any(t in c for t in ["hour", "dayofweek", "month", "day", "weekofyear", "quarter"])
        and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    # Detect low-cardinality, non-time numeric features (2–10 unique values)
    low_card_feats: list[str] = []
    for c in feature_columns:
        if c in train_df.columns and "__" not in c and pd.api.types.is_numeric_dtype(train_df[c]):
            n_unique = int(train_df[c].nunique(dropna=True))
            if 2 <= n_unique <= 10:
                low_card_feats.append(c)

    if not time_feats or not low_card_feats:
        return train_df, predict_df, []

    added: list[str] = []
    train = train_df.copy()
    pred = predict_df.copy()
    max_interactions = 6

    # Score pairs by estimated variance contribution: std(tf) * std(lc) (generic)
    pairs: list[tuple[float, str, str]] = []
    for tf in time_feats:
        tf_std = float(train[tf].std()) if train[tf].std() > 0 else 0.0
        for lc in low_card_feats:
            lc_std = float(train[lc].std()) if train[lc].std() > 0 else 0.0
            pairs.append((tf_std * lc_std, tf, lc))
    pairs.sort(reverse=True)

    for _, tf, lc in pairs[:max_interactions]:
        col_name = f"{tf}_x_{lc}"
        if col_name in train.columns:
            continue
        try:
            tf_train = pd.to_numeric(train[tf], errors="coerce").fillna(0)
            lc_train = pd.to_numeric(train[lc], errors="coerce").fillna(0)
            train[col_name] = tf_train * lc_train

            if tf in pred.columns and lc in pred.columns:
                tf_pred = pd.to_numeric(pred[tf], errors="coerce").fillna(0)
                lc_pred = pd.to_numeric(pred[lc], errors="coerce").fillna(0)
                pred[col_name] = tf_pred * lc_pred
            else:
                pred[col_name] = np.nan

            added.append(col_name)
        except Exception:
            pass

    return train, pred, added


# ── merge helpers ─────────────────────────────────────────────────────────────

def _merge_train(
    target_df: pd.DataFrame, train_cov: pd.DataFrame | None, spec: SchemaSpec
) -> pd.DataFrame:
    if train_cov is None:
        return target_df.copy()
    keys = [key for key in spec.join_keys if key in target_df.columns and key in train_cov.columns]
    if keys:
        return target_df.merge(train_cov, on=keys, how="left", suffixes=("", "_cov"))
    if len(target_df) == len(train_cov):
        return pd.concat(
            [target_df.reset_index(drop=True), _non_duplicate_columns(train_cov, target_df).reset_index(drop=True)],
            axis=1,
        )
    return target_df.copy()


def _merge_prediction(
    sample_submission: pd.DataFrame, val_cov: pd.DataFrame | None, spec: SchemaSpec
) -> pd.DataFrame:
    prediction = sample_submission.copy()
    if spec.target_column in prediction.columns:
        prediction = prediction.drop(columns=[spec.target_column])
    if val_cov is None:
        return prediction
    keys = [key for key in spec.join_keys if key in prediction.columns and key in val_cov.columns]
    if keys:
        return prediction.merge(val_cov, on=keys, how="left", suffixes=("", "_cov"))
    common = [
        col
        for col in prediction.columns
        if col in val_cov.columns and col not in {spec.row_id_column, spec.target_column}
    ]
    if common:
        return prediction.merge(val_cov, on=common, how="left", suffixes=("", "_cov"))
    if len(prediction) == len(val_cov):
        return pd.concat(
            [prediction.reset_index(drop=True), _non_duplicate_columns(val_cov, prediction).reset_index(drop=True)],
            axis=1,
        )
    return prediction


def _non_duplicate_columns(source: pd.DataFrame, existing: pd.DataFrame) -> pd.DataFrame:
    return source[[col for col in source.columns if col not in existing.columns]].copy()


# ── feature column selection ──────────────────────────────────────────────────

def _choose_feature_columns(
    train_df: pd.DataFrame, predict_df: pd.DataFrame, spec: SchemaSpec
) -> tuple[list[str], dict[str, str]]:
    excluded_map: dict[str, str] = {}

    def _exclude(col: str, reason: str) -> None:
        excluded_map[col] = reason

    _exclude(spec.target_column, "target")
    if spec.row_id_column:
        _exclude(spec.row_id_column, "row_id")

    excluded = set(excluded_map)
    for col in train_df.columns:
        ncol = _norm(col)
        if col in excluded:
            continue
        if ncol == _norm(spec.target_column):
            _exclude(col, "target_derived_name")
            excluded.add(col)
            continue
        if any(token in ncol for token in LEAKAGE_NAME_TOKENS) and col not in spec.join_keys:
            _exclude(col, "leakage_name_token")
            excluded.add(col)

    # Train-only columns (absent from predict_df) are excluded from features
    for col in train_df.columns:
        if col in excluded:
            continue
        if col not in predict_df.columns:
            _exclude(col, "train_only_absent_from_prediction")
            excluded.add(col)

    # Columns present in both frames (excluding leakage / id columns)
    shared = [col for col in train_df.columns if col in predict_df.columns and col not in excluded]

    # Engineered columns (containing ``__``) that may only exist in train_df
    # after feature extraction — include them if they also appear in predict_df.
    engineered = [
        col
        for col in train_df.columns
        if "__" in col and col not in excluded and col not in shared
    ]
    for col in engineered:
        if col in predict_df.columns:
            shared.append(col)

    return shared, {k: v for k, v in excluded_map.items() if k not in (spec.target_column, spec.row_id_column)}


# ── misc helpers ──────────────────────────────────────────────────────────────

def _read_description(description_path: str | None) -> str:
    if not description_path:
        return ""
    try:
        return Path(description_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _series_summary(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    return {
        "count": int(numeric.notna().sum()),
        "missing": int(numeric.isna().sum()),
        "mean": float(numeric.mean()) if numeric.notna().any() else None,
        "std": float(numeric.std()) if numeric.notna().sum() > 1 else None,
        "min": float(numeric.min()) if numeric.notna().any() else None,
        "median": float(numeric.median()) if numeric.notna().any() else None,
        "max": float(numeric.max()) if numeric.notna().any() else None,
    }


def _class_distribution(series: pd.Series) -> dict[str, Any]:
    counts = series.dropna().value_counts()
    total = int(counts.sum())
    classes = []
    for idx, cnt in counts.items():
        label = idx.item() if hasattr(idx, "item") else idx
        classes.append(
            {"label": label, "count": int(cnt), "fraction": float(cnt / total) if total else 0.0}
        )
    return {
        "n_classes": int(len(counts)),
        "majority_class_rate": float(counts.iloc[0] / total) if total else 0.0,
        "classes": classes,
    }


def _norm(value: str) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _py(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value
