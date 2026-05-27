"""Model selection and prediction for generic panel-style regression tasks."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
import pandas as pd

try:  # sklearn is preferred, but baseline models can run without it.
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - minimal evaluation runtimes only
    ColumnTransformer = object  # type: ignore
    SKLEARN_AVAILABLE = False

    def mean_absolute_error(y_true, y_pred):
        return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))

from .features import FeatureBundle


@dataclass
class ModelResult:
    predictions: np.ndarray
    selected_model_name: str
    model_scores: list[dict]
    holdout_strategy: dict
    metric_name: str
    target_clip_min: float | None
    target_clip_max: float | None


class GroupMeanRegressor:
    """Simple target mean baseline with hierarchical fallback."""

    def __init__(self, group_cols: list[str]):
        self.group_cols = group_cols
        self.global_mean_: float = 0.0
        self.mapping_: dict[tuple, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "GroupMeanRegressor":
        y_series = pd.Series(y, index=X.index, name="target")
        frame = pd.concat([X[self.group_cols].reset_index(drop=True), y_series.reset_index(drop=True)], axis=1)
        self.global_mean_ = float(y_series.mean())
        if self.group_cols:
            grouped = frame.groupby(self.group_cols, dropna=False)["target"].mean()
            self.mapping_ = {self._key(key): float(value) for key, value in grouped.items()}
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.group_cols:
            return np.full(len(X), self.global_mean_, dtype=float)
        preds = []
        for _, row in X[self.group_cols].iterrows():
            preds.append(self.mapping_.get(self._key(tuple(row.values)), self.global_mean_))
        return np.array(preds, dtype=float)

    @staticmethod
    def _key(value) -> tuple:
        if not isinstance(value, tuple):
            return (value,)
        return value


def train_and_predict(bundle: FeatureBundle, block_column: str | None, random_state: int = 42) -> ModelResult:
    y = pd.to_numeric(bundle.target, errors="coerce")
    valid_mask = y.notna()
    if valid_mask.sum() < 5:
        raise ValueError("Need at least five non-missing target rows to train a model.")

    train_df = bundle.train_df.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
    X = train_df[bundle.feature_columns].copy()
    X_pred = bundle.predict_df[bundle.feature_columns].copy()

    split = _make_holdout_split(train_df, bundle.feature_columns, y, bundle.profile.get("time_column"), random_state)
    train_idx, holdout_idx, holdout_strategy = split
    X_train, X_holdout = X.iloc[train_idx], X.iloc[holdout_idx]
    y_train, y_holdout = y.iloc[train_idx], y.iloc[holdout_idx]
    holdout_frame = train_df.iloc[holdout_idx].reset_index(drop=True)

    score_fn, metric_name = _score_function(block_column, holdout_frame)
    candidates = _build_candidates(bundle, train_df, random_state)
    scores: list[dict] = []
    fitted_candidates = []

    for name, estimator_factory in candidates:
        try:
            estimator = estimator_factory()
            estimator.fit(X_train, y_train)
            pred = _sanitize_predictions(estimator.predict(X_holdout), y)
            mae = float(mean_absolute_error(y_holdout, pred))
            selection_metric = float(score_fn(y_holdout, pred))
            scores.append(
                {
                    "name": name,
                    "mae": mae,
                    metric_name: selection_metric,
                    "status": "ok",
                }
            )
            fitted_candidates.append((selection_metric, name, estimator_factory))
        except Exception as exc:
            scores.append({"name": name, "status": "failed", "error": str(exc)})

    if not fitted_candidates:
        raise RuntimeError("All candidate models failed; cannot produce predictions.")

    _, selected_name, selected_factory = min(fitted_candidates, key=lambda item: item[0])
    final_model = selected_factory()
    final_model.fit(X, y)
    predictions = _sanitize_predictions(final_model.predict(X_pred), y)

    clip_min = 0.0 if float(y.min()) >= 0 else None
    clip_max = _reasonable_upper_clip(y)
    if clip_min is not None or clip_max is not None:
        predictions = np.clip(
            predictions,
            clip_min if clip_min is not None else -np.inf,
            clip_max if clip_max is not None else np.inf,
        )

    return ModelResult(
        predictions=predictions,
        selected_model_name=selected_name,
        model_scores=scores,
        holdout_strategy=holdout_strategy,
        metric_name=metric_name,
        target_clip_min=clip_min,
        target_clip_max=clip_max,
    )


def block_averaged_mae(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, blocks: pd.Series) -> float:
    frame = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred), "block": blocks.values})
    maes = []
    for _, sub in frame.groupby("block", dropna=False):
        maes.append(mean_absolute_error(sub["y_true"], sub["y_pred"]))
    return float(np.mean(maes)) if maes else float(mean_absolute_error(y_true, y_pred))


def _make_holdout_split(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    y: pd.Series,
    time_column: str | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if time_column and time_column in train_df.columns:
        values = train_df[time_column].astype(str)
        unique_values = sorted(values.dropna().unique().tolist())
        if len(unique_values) >= 5:
            n_holdout_periods = max(1, math.ceil(len(unique_values) * 0.2))
            holdout_values = set(unique_values[-n_holdout_periods:])
            holdout_mask = values.isin(holdout_values).to_numpy()
            if 0 < holdout_mask.sum() < len(train_df):
                holdout_idx = np.flatnonzero(holdout_mask)
                train_idx = np.flatnonzero(~holdout_mask)
                return train_idx, holdout_idx, {
                    "type": "time_holdout",
                    "time_column": time_column,
                    "holdout_values": sorted(holdout_values),
                    "n_train": int(len(train_idx)),
                    "n_holdout": int(len(holdout_idx)),
                }

    indices = np.arange(len(train_df))
    if SKLEARN_AVAILABLE:
        train_idx, holdout_idx = train_test_split(indices, test_size=0.2, random_state=random_state)
    else:
        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(indices)
        n_holdout = max(1, int(round(len(indices) * 0.2)))
        holdout_idx = shuffled[:n_holdout]
        train_idx = shuffled[n_holdout:]
    return np.array(train_idx), np.array(holdout_idx), {
        "type": "random_holdout",
        "random_state": random_state,
        "n_train": int(len(train_idx)),
        "n_holdout": int(len(holdout_idx)),
    }


def _score_function(block_column: str | None, holdout_frame: pd.DataFrame) -> tuple[Callable, str]:
    if block_column and block_column in holdout_frame.columns:
        blocks = holdout_frame[block_column].reset_index(drop=True)
        return lambda y_true, y_pred: block_averaged_mae(y_true, y_pred, blocks), "block_mae"
    return lambda y_true, y_pred: mean_absolute_error(y_true, y_pred), "mae"


def _build_candidates(bundle: FeatureBundle, train_df: pd.DataFrame, random_state: int):
    candidates = []
    group_candidates = _group_candidates(bundle, train_df)
    for name, cols in group_candidates:
        candidates.append((name, lambda cols=cols: GroupMeanRegressor(cols)))

    if not SKLEARN_AVAILABLE:
        if not candidates:
            candidates.append(("baseline_global_mean", lambda: GroupMeanRegressor([])))
        return candidates

    candidates.append(("dummy_mean", lambda: DummyRegressor(strategy="mean")))

    preprocessor = _make_preprocessor(bundle.numeric_columns, bundle.categorical_columns, scale_numeric=False)
    scaled_preprocessor = _make_preprocessor(bundle.numeric_columns, bundle.categorical_columns, scale_numeric=True)

    candidates.append(
        (
            "hist_gradient_boosting",
            lambda: Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", HistGradientBoostingRegressor(random_state=random_state, max_iter=250, learning_rate=0.05)),
                ]
            ),
        )
    )
    candidates.append(
        (
            "extra_trees",
            lambda: Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", ExtraTreesRegressor(n_estimators=250, random_state=random_state, n_jobs=-1, min_samples_leaf=2)),
                ]
            ),
        )
    )
    candidates.append(
        (
            "random_forest",
            lambda: Pipeline(
                [
                    ("preprocess", preprocessor),
                    ("model", RandomForestRegressor(n_estimators=180, random_state=random_state, n_jobs=-1, min_samples_leaf=2)),
                ]
            ),
        )
    )
    candidates.append(
        (
            "ridge",
            lambda: Pipeline([("preprocess", scaled_preprocessor), ("model", Ridge(alpha=1.0))]),
        )
    )
    candidates.append(
        (
            "elastic_net",
            lambda: Pipeline(
                [
                    ("preprocess", scaled_preprocessor),
                    ("model", ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=random_state, max_iter=5000)),
                ]
            ),
        )
    )

    try:
        import lightgbm as lgb  # type: ignore

        candidates.append(
            (
                "lightgbm",
                lambda: Pipeline(
                    [
                        ("preprocess", preprocessor),
                        (
                            "model",
                            lgb.LGBMRegressor(
                                n_estimators=500,
                                learning_rate=0.04,
                                num_leaves=31,
                                subsample=0.85,
                                colsample_bytree=0.85,
                                min_child_samples=10,
                                random_state=random_state,
                                n_jobs=-1,
                                verbose=-1,
                            ),
                        ),
                    ]
                ),
            )
        )
    except Exception:
        pass

    return candidates


def _group_candidates(bundle: FeatureBundle, train_df: pd.DataFrame) -> list[tuple[str, list[str]]]:
    candidates = []
    category = bundle.profile.get("category_column")
    if category and category in bundle.feature_columns:
        candidates.append(("baseline_category_mean", [category]))
    group_cols = [
        col
        for col in bundle.profile.get("join_keys", [])
        if col in bundle.feature_columns and col not in {bundle.profile.get("time_column"), category}
    ]
    if group_cols:
        candidates.append(("baseline_group_mean", group_cols[:2]))
    if category and group_cols and category in bundle.feature_columns:
        candidates.append(("baseline_group_category_mean", (group_cols[:2] + [category])[:3]))
    return candidates


def _make_preprocessor(numeric_columns: list[str], categorical_columns: list[str], scale_numeric: bool) -> ColumnTransformer:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_encoder = _one_hot_encoder()
    transformers = []
    if numeric_columns:
        transformers.append(("num", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", categorical_encoder),
                    ]
                ),
                categorical_columns,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=50)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _sanitize_predictions(pred: np.ndarray, y_train: pd.Series) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    finite_mean = float(pd.to_numeric(y_train, errors="coerce").mean())
    pred = np.where(np.isfinite(pred), pred, finite_mean)
    return pred


def _reasonable_upper_clip(y: pd.Series) -> float | None:
    if not (float(y.min()) >= 0):
        return None
    q99 = float(y.quantile(0.99))
    max_y = float(y.max())
    if not np.isfinite(q99) or not np.isfinite(max_y):
        return None
    return max(max_y * 1.5, q99 * 2.0)
