"""Data loading and feature frame construction."""

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

    train_df, predict_df = _add_time_features(train_df, predict_df, spec.time_column)

    if spec.target_column not in train_df.columns:
        raise ValueError(f"Target column '{spec.target_column}' not found after training merge.")

    feature_columns = _choose_feature_columns(train_df, predict_df, spec)
    if not feature_columns:
        raise ValueError("No usable feature columns were inferred.")

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


def _read_description(description_path: str | None) -> str:
    if not description_path:
        return ""
    try:
        return Path(description_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


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


def _add_time_features(
    train_df: pd.DataFrame, predict_df: pd.DataFrame, time_col: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not time_col or time_col not in train_df.columns:
        return train_df, predict_df
    train = train_df.copy()
    pred = predict_df.copy()
    combined = pd.concat(
        [train[[time_col]], pred[[time_col]] if time_col in pred.columns else pd.DataFrame({time_col: []})],
        ignore_index=True,
    )
    values = combined[time_col]
    parsed = pd.to_datetime(values, errors="coerce")
    parse_rate = float(parsed.notna().mean()) if len(parsed) else 0.0

    if parse_rate >= 0.6:
        train_parsed = pd.to_datetime(train[time_col], errors="coerce")
        train[f"{time_col}__year"] = train_parsed.dt.year
        train[f"{time_col}__month"] = train_parsed.dt.month
        train[f"{time_col}__month_sin"] = np.sin(2 * np.pi * train_parsed.dt.month / 12)
        train[f"{time_col}__month_cos"] = np.cos(2 * np.pi * train_parsed.dt.month / 12)
        if time_col in pred.columns:
            pred_parsed = pd.to_datetime(pred[time_col], errors="coerce")
            pred[f"{time_col}__year"] = pred_parsed.dt.year
            pred[f"{time_col}__month"] = pred_parsed.dt.month
            pred[f"{time_col}__month_sin"] = np.sin(2 * np.pi * pred_parsed.dt.month / 12)
            pred[f"{time_col}__month_cos"] = np.cos(2 * np.pi * pred_parsed.dt.month / 12)

    ordered_values = sorted(v for v in values.dropna().astype(str).unique().tolist())
    ordinal_map = {value: idx for idx, value in enumerate(ordered_values)}
    train[f"{time_col}__ordinal"] = train[time_col].astype(str).map(ordinal_map)
    if time_col in pred.columns:
        pred[f"{time_col}__ordinal"] = pred[time_col].astype(str).map(ordinal_map)
    return train, pred


def _choose_feature_columns(
    train_df: pd.DataFrame, predict_df: pd.DataFrame, spec: SchemaSpec
) -> list[str]:
    excluded = {spec.target_column, spec.row_id_column}
    for col in train_df.columns:
        ncol = _norm(col)
        if col in excluded:
            continue
        if ncol == _norm(spec.target_column):
            excluded.add(col)
            continue
        if any(token in ncol for token in LEAKAGE_NAME_TOKENS) and col not in spec.join_keys:
            excluded.add(col)

    shared = [col for col in train_df.columns if col in predict_df.columns and col not in excluded]
    engineered = [
        col
        for col in train_df.columns
        if "__" in col and col not in excluded and col not in shared
    ]
    for col in engineered:
        if col in predict_df.columns:
            shared.append(col)
    return shared


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

