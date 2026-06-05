"""Task-conditioned model selection and prediction.

The candidate model pool is a function of the resolved task type (see
``task.py``): regressors for regression, classifiers for binary / multiclass
classification. Each pool runs baselines first (per the workflow contract) then
a diverse set of candidates, and the best model is chosen on a held-out split by
the task-appropriate metric. Predictions are formatted to match what the task
requires (continuous values, class labels, or probabilities).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from .task import ACCURACY, BINARY, F1, F1_MACRO, LOG_LOSS, MULTICLASS, REGRESSION, ROC_AUC

try:  # sklearn is preferred, but baseline models can run without it.
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        ExtraTreesRegressor,
        GradientBoostingClassifier,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.svm import SVC

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - minimal evaluation runtimes only
    ColumnTransformer = object  # type: ignore
    SKLEARN_AVAILABLE = False

    def mean_absolute_error(y_true, y_pred):
        return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))

    def accuracy_score(y_true, y_pred):
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


try:
    import xgboost as xgb  # type: ignore
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

from .features import FeatureBundle


@dataclass
class ModelResult:
    predictions: np.ndarray
    selected_model_name: str
    model_scores: list[dict]
    holdout_strategy: dict
    metric_name: str
    greater_is_better: bool
    task_type: str
    output_kind: str
    extra_metrics: dict = field(default_factory=dict)
    target_clip_min: float | None = None
    target_clip_max: float | None = None
    # selected model's holdout arrays (for plotting/evaluation; not used for selection)
    holdout_y_true: Any = None
    holdout_y_pred: Any = None
    holdout_y_proba: Any = None
    holdout_label_classes: list | None = None
    holdout_by_model: Any = None  # name -> pred (regression) or (pred, proba) (classification)
    residual_analysis: dict = field(default_factory=dict)


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


class GroupModeClassifier:
    """Group class-frequency baseline mirroring :class:`GroupMeanRegressor`.

    Fits per-group class probabilities (with a global fallback) so it supports
    both label selection (argmax) and probability-based metrics. Trained on
    integer-encoded labels ``0..n_classes-1``.
    """

    def __init__(self, group_cols: list[str]):
        self.group_cols = group_cols
        self.classes_: np.ndarray = np.array([0])
        self.global_proba_: np.ndarray = np.array([1.0])
        self.mapping_: dict[tuple, np.ndarray] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "GroupModeClassifier":
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        index = {c: i for i, c in enumerate(self.classes_)}
        k = len(self.classes_)
        global_counts = np.zeros(k, dtype=float)
        for value in y_arr:
            global_counts[index[value]] += 1
        self.global_proba_ = global_counts / global_counts.sum()
        if self.group_cols:
            frame = X[self.group_cols].reset_index(drop=True).copy()
            frame["__y"] = y_arr
            for key, sub in frame.groupby(self.group_cols, dropna=False):
                counts = np.zeros(k, dtype=float)
                for value in sub["__y"]:
                    counts[index[value]] += 1
                self.mapping_[self._key(key)] = counts / counts.sum()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.group_cols:
            return np.tile(self.global_proba_, (len(X), 1))
        rows = [
            self.mapping_.get(self._key(tuple(row.values)), self.global_proba_)
            for _, row in X[self.group_cols].iterrows()
        ]
        return np.vstack(rows)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @staticmethod
    def _key(value) -> tuple:
        if not isinstance(value, tuple):
            return (value,)
        return value


class LogTargetRegressor:
    """Wraps any sklearn-compatible regressor; applies log1p to y at fit time and expm1 to predictions.

    Only safe when the target is non-negative. Improves accuracy on right-skewed distributions.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=float)
        self.estimator.fit(X, np.log1p(np.clip(y_arr, 0.0, None)))
        return self

    def predict(self, X):
        return np.expm1(self.estimator.predict(X))


class TopKAverageRegressor:
    """Refit k regressors on full training data and return the mean of their predictions."""

    def __init__(self, factories: list):
        self.factories = factories
        self._models: list = []

    def fit(self, X, y):
        self._models = [f() for f in self.factories]
        for m in self._models:
            m.fit(X, y)
        return self

    def predict(self, X):
        preds = [m.predict(X) for m in self._models]
        return np.mean(preds, axis=0)


class TopKVoteClassifier:
    """Refit k classifiers on full training data and return the average of their class probabilities."""

    def __init__(self, factories: list):
        self.factories = factories
        self._models: list = []
        self.classes_: np.ndarray = np.array([])

    def fit(self, X, y):
        self._models = [f() for f in self.factories]
        for m in self._models:
            m.fit(X, y)
        if self._models:
            self.classes_ = getattr(self._models[0], "classes_", np.unique(y))
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        probas = [np.asarray(m.predict_proba(X)) for m in self._models if hasattr(m, "predict_proba")]
        if not probas:
            return np.zeros((len(X), len(self.classes_)))
        return np.mean(probas, axis=0)


# ── dispatcher ────────────────────────────────────────────────────────────────

def train_and_predict(bundle: FeatureBundle, block_column: str | None = None, random_state: int = 42) -> ModelResult:
    if bundle.task.task_type == REGRESSION:
        return _train_regression(bundle, block_column, random_state)
    return _train_classification(bundle, random_state)


# ── regression path (Award A behaviour, preserved) ─────────────────────────────

def _train_regression(bundle: FeatureBundle, block_column: str | None, random_state: int) -> ModelResult:
    y = pd.to_numeric(bundle.target, errors="coerce")
    valid_mask = y.notna()
    if valid_mask.sum() < 5:
        raise ValueError("Need at least five non-missing target rows to train a model.")

    train_df = bundle.train_df.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
    X = train_df[bundle.feature_columns].copy()
    X_pred = bundle.predict_df[bundle.feature_columns].copy()

    within_period_info = bundle.profile.get("within_period_info")
    split = _make_holdout_split(
        train_df, bundle.feature_columns, y, bundle.profile.get("time_column"), random_state,
        within_period_info=within_period_info,
    )
    train_idx, holdout_idx, holdout_strategy = split
    X_train, X_holdout = X.iloc[train_idx], X.iloc[holdout_idx]
    y_train, y_holdout = y.iloc[train_idx], y.iloc[holdout_idx]
    holdout_frame = train_df.iloc[holdout_idx].reset_index(drop=True)

    score_fn, metric_name = _score_function(block_column, holdout_frame)
    candidates = _build_candidates(bundle, train_df, random_state)
    scores: list[dict] = []
    fitted_candidates = []
    holdout_capture: dict[str, np.ndarray] = {}

    for name, estimator_factory in candidates:
        try:
            estimator = estimator_factory()
            estimator.fit(X_train, y_train)
            pred = _sanitize_predictions(estimator.predict(X_holdout), y)
            holdout_capture[name] = pred
            mae = float(mean_absolute_error(y_holdout, pred))
            selection_metric = float(score_fn(y_holdout, pred))
            scores.append(
                {
                    "name": name,
                    "status": "ok",
                    "score": selection_metric,
                    "detail": {"mae": mae, metric_name: selection_metric},
                }
            )
            fitted_candidates.append((selection_metric, name, estimator_factory))
        except Exception as exc:
            scores.append({"name": name, "status": "failed", "error": str(exc)})

    if not fitted_candidates:
        raise RuntimeError("All candidate models failed; cannot produce predictions.")

    # ── try top-2 ensemble ────────────────────────────────────────────────────
    ensemble_entry = _try_regression_ensemble(fitted_candidates, holdout_capture, score_fn, y_holdout)
    if ensemble_entry is not None:
        e_score, e_name, e_pred, e_factory = ensemble_entry
        holdout_capture[e_name] = e_pred
        mae_e = float(mean_absolute_error(y_holdout, e_pred))
        scores.append({
            "name": e_name, "status": "ok", "score": e_score,
            "detail": {"mae": mae_e, metric_name: e_score},
        })
        fitted_candidates.append((e_score, e_name, e_factory))

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

    # ── multi-split stability ─────────────────────────────────────────────────
    # Evaluate the selected model on 2 additional time-ordered splits to estimate
    # variance. This is generic: splits are computed from the time_column or ordinal index.
    stability_scores = [float(score_fn(y_holdout, holdout_capture[selected_name]))] if selected_name in holdout_capture else []
    try:
        _stability_splits = _compute_additional_splits(train_df, bundle.profile.get("time_column"), random_state)
        _stability_final = selected_factory()
        for _alt_train_idx, _alt_holdout_idx in _stability_splits:
            _x_tr = X.iloc[_alt_train_idx]
            _y_tr = y.iloc[_alt_train_idx]
            _x_hd = X.iloc[_alt_holdout_idx]
            _y_hd = y.iloc[_alt_holdout_idx]
            _hd_frame = train_df.iloc[_alt_holdout_idx].reset_index(drop=True)
            _alt_score_fn, _ = _score_function(block_column, _hd_frame)
            _m = selected_factory()
            _m.fit(_x_tr, _y_tr)
            _p = _sanitize_predictions(_m.predict(_x_hd), y)
            stability_scores.append(float(_alt_score_fn(_y_hd, _p)))
    except Exception:
        pass
    if len(stability_scores) >= 2:
        _stab_arr = np.array(stability_scores, dtype=float)
        holdout_strategy["stability"] = {
            "split_scores": [round(s, 4) for s in stability_scores],
            "cv_mae_mean": round(float(_stab_arr.mean()), 4),
            "cv_mae_std": round(float(_stab_arr.std()), 4),
            "relative_stability": round(float(_stab_arr.std() / _stab_arr.mean()), 4) if _stab_arr.mean() > 0 else None,
        }

    selected_detail = _selected_detail(scores, selected_name)
    y_true_holdout = np.asarray(y_holdout, dtype=float)
    y_pred_holdout = holdout_capture.get(selected_name)
    residual_analysis = _compute_residual_analysis(y_true_holdout, y_pred_holdout) if y_pred_holdout is not None else {}
    return ModelResult(
        predictions=predictions,
        selected_model_name=selected_name,
        model_scores=scores,
        holdout_strategy=holdout_strategy,
        metric_name=metric_name,
        greater_is_better=False,
        task_type=REGRESSION,
        output_kind="value",
        extra_metrics=selected_detail,
        target_clip_min=clip_min,
        target_clip_max=clip_max,
        holdout_y_true=y_true_holdout,
        holdout_y_pred=y_pred_holdout,
        holdout_by_model=holdout_capture,
        residual_analysis=residual_analysis,
    )


# ── classification path ─────────────────────────────────────────────────────────

def _train_classification(bundle: FeatureBundle, random_state: int) -> ModelResult:
    task = bundle.task
    raw_y = bundle.target
    valid_mask = raw_y.notna()
    if valid_mask.sum() < 5:
        raise ValueError("Need at least five non-missing target rows to train a model.")

    train_df = bundle.train_df.loc[valid_mask].reset_index(drop=True)
    y_raw = raw_y.loc[valid_mask].reset_index(drop=True)

    classes = list(task.class_labels) if task.class_labels else sorted(y_raw.unique().tolist(), key=lambda v: str(v))
    label_to_int = {label: idx for idx, label in enumerate(classes)}
    int_to_label = {idx: label for label, idx in label_to_int.items()}
    y = y_raw.map(label_to_int)
    if y.isna().any():  # labels outside the resolved class set (defensive)
        keep = y.notna()
        train_df = train_df.loc[keep.values].reset_index(drop=True)
        y = y.loc[keep].reset_index(drop=True)
    y = y.astype(int)
    n_classes = len(classes)
    if task.task_type == BINARY and task.positive_label in label_to_int:
        positive_index = label_to_int[task.positive_label]
    else:
        positive_index = n_classes - 1

    X = train_df[bundle.feature_columns].copy()
    X_pred = bundle.predict_df[bundle.feature_columns].copy()

    split = _make_holdout_split(
        train_df, bundle.feature_columns, y, bundle.profile.get("time_column"), random_state, stratify_labels=y.to_numpy()
    )
    train_idx, holdout_idx, holdout_strategy = split
    X_train, X_holdout = X.iloc[train_idx], X.iloc[holdout_idx]
    y_train, y_holdout = y.iloc[train_idx], y.iloc[holdout_idx]

    metric = task.metric
    greater = task.greater_is_better
    candidates = _build_classification_candidates(bundle, train_df, random_state, task)
    scores: list[dict] = []
    fitted_candidates = []
    holdout_capture: dict[str, tuple] = {}

    for name, estimator_factory in candidates:
        try:
            estimator = estimator_factory()
            estimator.fit(X_train, y_train)
            pred = np.asarray(estimator.predict(X_holdout))
            proba = _safe_proba(estimator, X_holdout, n_classes)
            holdout_capture[name] = (pred, proba)
            detail = _classification_metrics(y_holdout.to_numpy(), pred, proba, n_classes, positive_index)
            selection_metric = detail.get(metric)
            if selection_metric is None:
                selection_metric = detail.get(ACCURACY, 0.0)
            scores.append(
                {"name": name, "status": "ok", "score": float(selection_metric), "detail": detail}
            )
            fitted_candidates.append((float(selection_metric), name, estimator_factory))
        except Exception as exc:
            scores.append({"name": name, "status": "failed", "error": str(exc)})

    if not fitted_candidates:
        raise RuntimeError("All candidate models failed; cannot produce predictions.")

    # ── try top-2 probability ensemble ───────────────────────────────────────
    ensemble_entry_cls = _try_classification_ensemble(
        fitted_candidates, holdout_capture, y_holdout,
        n_classes, positive_index, metric, greater,
    )
    if ensemble_entry_cls is not None:
        e_score, e_name, e_pred, e_proba, e_factory = ensemble_entry_cls
        holdout_capture[e_name] = (e_pred, e_proba)
        e_detail = _classification_metrics(y_holdout.to_numpy(), e_pred, e_proba, n_classes, positive_index)
        scores.append({"name": e_name, "status": "ok", "score": float(e_score), "detail": e_detail})
        fitted_candidates.append((float(e_score), e_name, e_factory))

    selected_score, selected_name, selected_factory = (
        max(fitted_candidates, key=lambda item: item[0])
        if greater
        else min(fitted_candidates, key=lambda item: item[0])
    )
    final_model = selected_factory()
    final_model.fit(X, y)

    if task.output_kind == "probability" and n_classes == 2:
        proba = _safe_proba(final_model, X_pred, n_classes)
        if proba is None:
            raise RuntimeError("Probability output requested but the selected model lacks predict_proba.")
        predictions = np.clip(proba[:, positive_index], 0.0, 1.0)
        output_kind = "probability"
    else:
        pred_int = np.asarray(final_model.predict(X_pred)).astype(int)
        predictions = np.array([int_to_label[i] for i in pred_int], dtype=object)
        output_kind = "label"

    selected_detail = _selected_detail(scores, selected_name)
    sel_pred, sel_proba = holdout_capture.get(selected_name, (None, None))
    return ModelResult(
        predictions=predictions,
        selected_model_name=selected_name,
        model_scores=scores,
        holdout_strategy=holdout_strategy,
        metric_name=metric,
        greater_is_better=greater,
        task_type=task.task_type,
        output_kind=output_kind,
        extra_metrics=selected_detail,
        target_clip_min=None,
        target_clip_max=None,
        holdout_y_true=np.asarray(y_holdout),
        holdout_y_pred=sel_pred,
        holdout_y_proba=sel_proba,
        holdout_label_classes=list(classes),
        holdout_by_model=holdout_capture,
    )


def _classification_metrics(
    y_true: np.ndarray, pred: np.ndarray, proba: np.ndarray | None, n_classes: int, positive_index: int
) -> dict:
    detail: dict[str, float] = {ACCURACY: float(accuracy_score(y_true, pred))}
    if not SKLEARN_AVAILABLE:
        return detail
    if n_classes == 2:
        y_pos = (np.asarray(y_true) == positive_index).astype(int)
        pred_pos = (np.asarray(pred) == positive_index).astype(int)
        detail[F1] = float(f1_score(y_pos, pred_pos, zero_division=0))
        if proba is not None:
            try:
                detail[ROC_AUC] = float(roc_auc_score(y_pos, proba[:, positive_index]))
            except Exception:
                pass
            try:
                detail[LOG_LOSS] = float(log_loss(y_true, proba, labels=list(range(n_classes))))
            except Exception:
                pass
    else:
        detail[F1_MACRO] = float(f1_score(y_true, pred, average="macro", zero_division=0))
        if proba is not None:
            try:
                detail[ROC_AUC] = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))
            except Exception:
                pass
            try:
                detail[LOG_LOSS] = float(log_loss(y_true, proba, labels=list(range(n_classes))))
            except Exception:
                pass
    return detail


def _safe_proba(estimator, X, n_classes: int) -> np.ndarray | None:
    if not hasattr(estimator, "predict_proba"):
        return None
    try:
        proba = np.asarray(estimator.predict_proba(X), dtype=float)
    except Exception:
        return None
    # Realign columns to the canonical 0..n_classes-1 ordering (a fold may be
    # missing a class, which would otherwise shift probability columns).
    classes = list(getattr(estimator, "classes_", range(proba.shape[1])))
    full = np.zeros((proba.shape[0], n_classes), dtype=float)
    for col, cls in enumerate(classes):
        cls_int = int(cls)
        if 0 <= cls_int < n_classes:
            full[:, cls_int] = proba[:, col]
    return full


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
    stratify_labels: np.ndarray | None = None,
    within_period_info: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    # Within-period holdout: use it when within_period_info is supplied (preferred split strategy).
    # Holds out the top 20% of training sub-period values (e.g. days 16–19 when train has days 1–19),
    # fitting on the remaining values. This simulates the train/predict partition exactly.
    if within_period_info and time_column and time_column in train_df.columns:
        feat_attr = within_period_info.get("feature")  # e.g. "day"
        train_range = within_period_info.get("train_range")  # e.g. [1, 19]
        predict_range = within_period_info.get("predict_range")  # e.g. [20, 31]
        if feat_attr and train_range:
            try:
                parsed = pd.to_datetime(train_df[time_column], errors="coerce")
                sub_values = getattr(parsed.dt, feat_attr)
                all_train_vals = sorted(sub_values.dropna().unique().tolist())
                if len(all_train_vals) >= 4:
                    n_holdout_vals = max(1, math.ceil(len(all_train_vals) * 0.2))
                    holdout_sub_vals = set(all_train_vals[-n_holdout_vals:])
                    fit_sub_vals = set(all_train_vals[:-n_holdout_vals])
                    holdout_mask = sub_values.isin(holdout_sub_vals).to_numpy()
                    if 0 < holdout_mask.sum() < len(train_df):
                        holdout_idx = np.flatnonzero(holdout_mask)
                        train_idx = np.flatnonzero(~holdout_mask)
                        return train_idx, holdout_idx, {
                            "type": "within_period_holdout",
                            "time_column": time_column,
                            "period_feature": feat_attr,
                            "fit_day_distribution": {
                                "min": int(min(fit_sub_vals)),
                                "max": int(max(fit_sub_vals)),
                                "unique_values": sorted(int(v) for v in fit_sub_vals),
                            },
                            "validation_day_distribution": {
                                "min": int(min(holdout_sub_vals)),
                                "max": int(max(holdout_sub_vals)),
                                "unique_values": sorted(int(v) for v in holdout_sub_vals),
                            },
                            "prediction_day_distribution": {
                                "min": int(predict_range[0]),
                                "max": int(predict_range[1]),
                            } if predict_range else {},
                            "n_train": int(len(train_idx)),
                            "n_holdout": int(len(holdout_idx)),
                        }
            except Exception:
                pass  # fall through to time_holdout or random

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
    stratified = False
    if SKLEARN_AVAILABLE:
        if stratify_labels is not None:
            try:
                train_idx, holdout_idx = train_test_split(
                    indices, test_size=0.2, random_state=random_state, stratify=stratify_labels
                )
                stratified = True
            except Exception:
                train_idx, holdout_idx = train_test_split(indices, test_size=0.2, random_state=random_state)
        else:
            train_idx, holdout_idx = train_test_split(indices, test_size=0.2, random_state=random_state)
    else:
        rng = np.random.default_rng(random_state)
        shuffled = rng.permutation(indices)
        n_holdout = max(1, int(round(len(indices) * 0.2)))
        holdout_idx = shuffled[:n_holdout]
        train_idx = shuffled[n_holdout:]
    return np.array(train_idx), np.array(holdout_idx), {
        "type": "stratified_holdout" if stratified else "random_holdout",
        "random_state": random_state,
        "n_train": int(len(train_idx)),
        "n_holdout": int(len(holdout_idx)),
    }


def _score_function(block_column: str | None, holdout_frame: pd.DataFrame) -> tuple[Callable, str]:
    if block_column and block_column in holdout_frame.columns:
        blocks = holdout_frame[block_column].reset_index(drop=True)
        return lambda y_true, y_pred: block_averaged_mae(y_true, y_pred, blocks), "block_mae"
    return lambda y_true, y_pred: mean_absolute_error(y_true, y_pred), "mae"


def _selected_detail(scores: list[dict], selected_name: str) -> dict:
    for score in scores:
        if score.get("name") == selected_name and score.get("status") == "ok":
            return dict(score.get("detail", {}))
    return {}


def _build_candidates(bundle: FeatureBundle, train_df: pd.DataFrame, random_state: int):
    rs = random_state
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
            lambda: Pipeline([
                ("preprocess", preprocessor),
                ("model", HistGradientBoostingRegressor(random_state=rs, max_iter=400, learning_rate=0.05)),
            ]),
        )
    )
    candidates.append(
        (
            "extra_trees",
            lambda: Pipeline([
                ("preprocess", preprocessor),
                ("model", ExtraTreesRegressor(n_estimators=400, random_state=rs, n_jobs=-1, min_samples_leaf=2, max_features="sqrt")),
            ]),
        )
    )
    candidates.append(
        (
            "random_forest",
            lambda: Pipeline([
                ("preprocess", preprocessor),
                ("model", RandomForestRegressor(n_estimators=300, random_state=rs, n_jobs=-1, min_samples_leaf=2)),
            ]),
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
            lambda: Pipeline([
                ("preprocess", scaled_preprocessor),
                ("model", ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=rs, max_iter=5000)),
            ]),
        )
    )
    candidates.append(
        (
            "gradient_boosting",
            lambda: Pipeline([
                ("preprocess", preprocessor),
                ("model", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=rs, subsample=0.85)),
            ]),
        )
    )

    # ── LightGBM variants ─────────────────────────────────────────────────────
    try:
        import lightgbm as lgb  # type: ignore

        # Adapt complexity to dataset size: more leaves only make sense when
        # there are enough training rows to fill them (rough rule: n_rows / 100).
        _n_rows_reg = len(train_df)
        _strong_leaves = min(127, max(31, _n_rows_reg // 100))

        candidates.append(
            (
                "lightgbm",
                lambda: Pipeline([
                    ("preprocess", preprocessor),
                    ("model", lgb.LGBMRegressor(
                        n_estimators=800,
                        learning_rate=0.04,
                        num_leaves=63,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_samples=20,
                        reg_alpha=0.1,
                        reg_lambda=0.1,
                        random_state=rs,
                        n_jobs=-1,
                        verbose=-1,
                    )),
                ]),
            )
        )
        candidates.append(
            (
                "lightgbm_strong",
                lambda _sl=_strong_leaves: Pipeline([
                    ("preprocess", preprocessor),
                    ("model", lgb.LGBMRegressor(
                        n_estimators=1200,
                        learning_rate=0.02,
                        num_leaves=_sl,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_samples=20,
                        reg_alpha=0.05,
                        reg_lambda=0.05,
                        random_state=rs,
                        n_jobs=-1,
                        verbose=-1,
                    )),
                ]),
            )
        )
        # target-transform variants for right-skewed non-negative targets
        target_vals = pd.to_numeric(bundle.target, errors="coerce").dropna()
        if float(target_vals.min()) >= 0 and len(target_vals) > 10:
            skew = float(target_vals.skew())
            if skew > 0.5:
                # sqrt transform: effective for moderate skew (0.5 < skew ≤ 1.5)
                candidates.append(
                    (
                        "hgb_sqrt",
                        lambda: SqrtTargetRegressor(Pipeline([
                            ("preprocess", preprocessor),
                            ("model", HistGradientBoostingRegressor(random_state=rs, max_iter=400, learning_rate=0.05)),
                        ])),
                    )
                )
                candidates.append(
                    (
                        "lightgbm_sqrt",
                        lambda: SqrtTargetRegressor(Pipeline([
                            ("preprocess", preprocessor),
                            ("model", lgb.LGBMRegressor(
                                n_estimators=800,
                                learning_rate=0.04,
                                num_leaves=63,
                                subsample=0.85,
                                colsample_bytree=0.85,
                                min_child_samples=20,
                                reg_alpha=0.1,
                                reg_lambda=0.1,
                                random_state=rs,
                                n_jobs=-1,
                                verbose=-1,
                            )),
                        ])),
                    )
                )
            if skew > 1.5:
                candidates.append(
                    (
                        "lightgbm_log",
                        lambda: LogTargetRegressor(Pipeline([
                            ("preprocess", preprocessor),
                            ("model", lgb.LGBMRegressor(
                                n_estimators=800,
                                learning_rate=0.04,
                                num_leaves=63,
                                subsample=0.85,
                                colsample_bytree=0.85,
                                min_child_samples=20,
                                reg_alpha=0.1,
                                reg_lambda=0.1,
                                random_state=rs,
                                n_jobs=-1,
                                verbose=-1,
                            )),
                        ])),
                    )
                )
                candidates.append(
                    (
                        "hgb_log",
                        lambda: LogTargetRegressor(Pipeline([
                            ("preprocess", preprocessor),
                            ("model", HistGradientBoostingRegressor(random_state=rs, max_iter=400, learning_rate=0.05)),
                        ])),
                    )
                )
    except Exception:
        pass

    # ── XGBoost (optional) ────────────────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        try:
            candidates.append(
                (
                    "xgboost",
                    lambda: Pipeline([
                        ("preprocess", preprocessor),
                        ("model", xgb.XGBRegressor(
                            n_estimators=800,
                            learning_rate=0.04,
                            max_depth=6,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_alpha=0.1,
                            reg_lambda=1.0,
                            random_state=rs,
                            n_jobs=-1,
                            verbosity=0,
                            eval_metric="mae",
                        )),
                    ]),
                )
            )
        except Exception:
            pass

    return candidates


def _build_classification_candidates(bundle: FeatureBundle, train_df: pd.DataFrame, random_state: int, task):
    rs = random_state
    candidates = []
    for name, cols in _group_candidates(bundle, train_df):
        cls_name = name.replace("_mean", "_mode")
        candidates.append((cls_name, lambda cols=cols: GroupModeClassifier(cols)))

    if not SKLEARN_AVAILABLE:
        if not candidates:
            candidates.append(("baseline_global_mode", lambda: GroupModeClassifier([])))
        return candidates

    candidates.append(("dummy_most_frequent", lambda: DummyClassifier(strategy="most_frequent")))
    candidates.append(("dummy_stratified", lambda: DummyClassifier(strategy="stratified", random_state=rs)))

    preprocessor = _make_preprocessor(bundle.numeric_columns, bundle.categorical_columns, scale_numeric=False)
    scaled_preprocessor = _make_preprocessor(bundle.numeric_columns, bundle.categorical_columns, scale_numeric=True)

    candidates.append(
        (
            "logistic_regression",
            lambda: Pipeline([
                ("preprocess", scaled_preprocessor),
                ("model", LogisticRegression(max_iter=1000, random_state=rs, C=1.0)),
            ]),
        )
    )
    candidates.append(
        (
            "random_forest",
            lambda: Pipeline([
                ("preprocess", preprocessor),
                ("model", RandomForestClassifier(n_estimators=400, random_state=rs, n_jobs=-1, min_samples_leaf=2)),
            ]),
        )
    )
    candidates.append(
        (
            "extra_trees",
            lambda: Pipeline([
                ("preprocess", preprocessor),
                ("model", ExtraTreesClassifier(n_estimators=400, random_state=rs, n_jobs=-1, min_samples_leaf=2, max_features="sqrt")),
            ]),
        )
    )
    candidates.append(
        (
            "hist_gradient_boosting",
            lambda: Pipeline([
                ("preprocess", preprocessor),
                ("model", HistGradientBoostingClassifier(random_state=rs, max_iter=500, learning_rate=0.05)),
            ]),
        )
    )
    candidates.append(
        (
            "gradient_boosting",
            lambda: Pipeline([
                ("preprocess", preprocessor),
                ("model", GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=rs, subsample=0.85)),
            ]),
        )
    )
    candidates.append(
        (
            "gaussian_nb",
            lambda: Pipeline([("preprocess", scaled_preprocessor), ("model", GaussianNB())]),
        )
    )
    if task.task_type == BINARY:
        candidates.append(
            (
                "knn",
                lambda: Pipeline([("preprocess", scaled_preprocessor), ("model", KNeighborsClassifier(n_neighbors=15))]),
            )
        )

    # SVM with RBF kernel: excellent for small/medium datasets (≤50k rows)
    if len(train_df) <= 50_000:
        candidates.append(
            (
                "svm_rbf",
                lambda: Pipeline([
                    ("preprocess", scaled_preprocessor),
                    ("model", SVC(kernel="rbf", probability=True, random_state=rs, C=1.0, gamma="scale")),
                ]),
            )
        )

    # ── LightGBM variants ─────────────────────────────────────────────────────
    try:
        import lightgbm as lgb  # type: ignore

        candidates.append(
            (
                "lightgbm",
                lambda: Pipeline([
                    ("preprocess", preprocessor),
                    ("model", lgb.LGBMClassifier(
                        n_estimators=600,
                        learning_rate=0.04,
                        num_leaves=63,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_samples=20,
                        reg_alpha=0.1,
                        reg_lambda=0.1,
                        random_state=rs,
                        n_jobs=-1,
                        verbose=-1,
                    )),
                ]),
            )
        )
        _n_rows_cls = len(train_df)
        _strong_leaves_cls = min(127, max(31, _n_rows_cls // 100))
        candidates.append(
            (
                "lightgbm_strong",
                lambda _sl=_strong_leaves_cls: Pipeline([
                    ("preprocess", preprocessor),
                    ("model", lgb.LGBMClassifier(
                        n_estimators=1000,
                        learning_rate=0.02,
                        num_leaves=_sl,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_samples=20,
                        reg_alpha=0.05,
                        reg_lambda=0.05,
                        random_state=rs,
                        n_jobs=-1,
                        verbose=-1,
                    )),
                ]),
            )
        )
    except Exception:
        pass

    # ── XGBoost (optional) ────────────────────────────────────────────────────
    if XGBOOST_AVAILABLE:
        try:
            candidates.append(
                (
                    "xgboost",
                    lambda: Pipeline([
                        ("preprocess", preprocessor),
                        ("model", xgb.XGBClassifier(
                            n_estimators=600,
                            learning_rate=0.04,
                            max_depth=6,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_alpha=0.1,
                            reg_lambda=1.0,
                            random_state=rs,
                            n_jobs=-1,
                            verbosity=0,
                            use_label_encoder=False,
                            eval_metric="logloss",
                        )),
                    ]),
                )
            )
        except Exception:
            pass

    return candidates


def _try_regression_ensemble(
    fitted_candidates: list,
    holdout_capture: dict,
    score_fn: Callable,
    y_holdout: pd.Series,
) -> tuple | None:
    """Average top-2 regression models if both captured and the ensemble scores better."""
    ok = [(s, n, f) for s, n, f in fitted_candidates if n in holdout_capture]
    if len(ok) < 2:
        return None
    top2 = sorted(ok)[:2]
    best_score = top2[0][0]
    # Skip ensemble if the second model is much worse (> 15% gap) — averaging would hurt
    if top2[1][0] > best_score * 1.15:
        return None
    pred1 = holdout_capture[top2[0][1]]
    pred2 = holdout_capture[top2[1][1]]
    ensemble_pred = (pred1 + pred2) * 0.5
    ensemble_score = float(score_fn(y_holdout, ensemble_pred))
    if ensemble_score >= best_score:
        return None
    name = f"ensemble({top2[0][1]}+{top2[1][1]})"
    f1, f2 = top2[0][2], top2[1][2]
    factory = lambda f1=f1, f2=f2: TopKAverageRegressor([f1, f2])
    return (ensemble_score, name, ensemble_pred, factory)


def _try_classification_ensemble(
    fitted_candidates: list,
    holdout_capture: dict,
    y_holdout: pd.Series,
    n_classes: int,
    positive_index: int,
    metric: str,
    greater: bool,
) -> tuple | None:
    """Soft-vote top-2 classifiers if both have probability outputs and the ensemble scores better."""
    ok = [(s, n, f) for s, n, f in fitted_candidates if n in holdout_capture]
    if len(ok) < 2:
        return None
    top2 = sorted(ok, reverse=greater)[:2]
    best_score = top2[0][0]
    # Both must have probability arrays
    p1 = holdout_capture[top2[0][1]][1]
    p2 = holdout_capture[top2[1][1]][1]
    if p1 is None or p2 is None:
        return None
    avg_proba = (p1 + p2) * 0.5
    avg_pred = np.argmax(avg_proba, axis=1)
    detail = _classification_metrics(y_holdout.to_numpy(), avg_pred, avg_proba, n_classes, positive_index)
    ensemble_score = detail.get(metric, detail.get(ACCURACY, 0.0))
    improved = ensemble_score > best_score if greater else ensemble_score < best_score
    if not improved:
        return None
    name = f"ensemble({top2[0][1]}+{top2[1][1]})"
    f1, f2 = top2[0][2], top2[1][2]
    factory = lambda f1=f1, f2=f2: TopKVoteClassifier([f1, f2])
    return (float(ensemble_score), name, avg_pred, avg_proba, factory)


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


def _compute_residual_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute residual breakdown for heteroscedasticity and tail-region accuracy."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)
    n = len(y_true)
    if n < 8:
        return {}

    result: dict = {}

    # Residuals by predicted-value quartile
    try:
        pred_q = pd.qcut(pd.Series(y_pred), q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"], duplicates="drop")
        by_pred_q: list[dict] = []
        for label in pred_q.cat.categories:
            mask = (pred_q == label).values
            if mask.sum() == 0:
                continue
            by_pred_q.append({
                "pred_quantile": str(label),
                "n": int(mask.sum()),
                "mean_residual": round(float(residuals[mask].mean()), 4),
                "mean_abs_residual": round(float(abs_residuals[mask].mean()), 4),
            })
        result["by_pred_quantile"] = by_pred_q
    except Exception:
        pass

    # Residuals by true-value quartile
    try:
        true_q = pd.qcut(pd.Series(y_true), q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"], duplicates="drop")
        by_true_q: list[dict] = []
        for label in true_q.cat.categories:
            mask = (true_q == label).values
            if mask.sum() == 0:
                continue
            by_true_q.append({
                "true_quantile": str(label),
                "n": int(mask.sum()),
                "mean_residual": round(float(residuals[mask].mean()), 4),
                "mean_abs_residual": round(float(abs_residuals[mask].mean()), 4),
            })
        result["by_true_quantile"] = by_true_q
    except Exception:
        pass

    # Heteroscedasticity: correlation of |residual| with predicted value
    try:
        pair = pd.concat(
            [pd.Series(y_pred, name="pred"), pd.Series(abs_residuals, name="abs_res")], axis=1
        ).dropna()
        if len(pair) >= 5:
            het_corr = float(pair.corr().iloc[0, 1])
            result["heteroscedasticity_corr"] = round(het_corr, 4)
            result["heteroscedasticity_note"] = (
                "strong positive — errors grow with predicted values" if het_corr > 0.50 else
                "moderate positive — some error growth with predicted values" if het_corr > 0.25 else
                "low — errors roughly uniform across predicted value range"
            )
    except Exception:
        pass

    # High-value bias analysis (correctly signed: residual = y_true - y_pred)
    # positive residual → y_true > y_pred → model underpredicts
    # negative residual → y_true < y_pred → model overpredicts
    try:
        top_mask = y_true >= float(np.percentile(y_true, 75))
        if top_mask.sum() >= 4:
            top_resid = residuals[top_mask]  # y_true - y_pred for top-quartile rows
            underpred_frac = float((top_resid > 0).mean())  # fraction where y_true > y_pred
            overpred_frac = float((top_resid < 0).mean())   # fraction where y_true < y_pred
            mean_resid = float(top_resid.mean())

            # Threshold: negligible if |mean_resid| < 1% of Q3
            q3 = float(np.percentile(y_true, 75))
            if abs(mean_resid) < 0.01 * abs(q3) if q3 != 0 else abs(mean_resid) < 1e-6:
                bias_direction = "negligible_bias"
            elif mean_resid > 0:
                bias_direction = "systematic_underprediction"
            else:
                bias_direction = "systematic_overprediction"

            high_value_bias = {
                "n_high_value_rows": int(top_mask.sum()),
                "fraction_underpredicted": round(underpred_frac, 4),
                "fraction_overpredicted": round(overpred_frac, 4),
                "mean_residual_high_values": round(mean_resid, 4),
                "bias_direction": bias_direction,
                "note": (
                    "model systematically underpredicts high values" if bias_direction == "systematic_underprediction" else
                    "model systematically overpredicts high values" if bias_direction == "systematic_overprediction" else
                    "no systematic bias in high-value region"
                ),
            }
            result["high_value_bias"] = high_value_bias
            # Backward-compatibility alias
            result["high_value_underprediction"] = {
                "n_high_value_rows": high_value_bias["n_high_value_rows"],
                "fraction_underpredicted": high_value_bias["fraction_underpredicted"],
                "mean_residual_high_values": high_value_bias["mean_residual_high_values"],
                "note": high_value_bias["note"],
            }
    except Exception:
        pass

    return result


class SqrtTargetRegressor:
    """Wraps any sklearn-compatible regressor; applies sqrt to y at fit time and squares predictions.

    Only safe when the target is non-negative. Useful for moderately right-skewed distributions
    (skew > 0.5), complementing the log-transform variant used for heavier skew.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        y_arr = np.clip(np.asarray(y, dtype=float), 0.0, None)
        self.estimator.fit(X, np.sqrt(y_arr))
        return self

    def predict(self, X):
        return np.square(np.clip(self.estimator.predict(X), 0.0, None))


def _compute_additional_splits(
    train_df: pd.DataFrame,
    time_column: str | None,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return up to 2 alternative (train_idx, holdout_idx) splits for stability estimation.

    Uses chronological ordering on the time column when available, otherwise index order.
    Splits at the 60th and 40th percentile of the ordered series to get earlier holdouts.
    Generic: no dataset-specific column names.
    """
    n = len(train_df)
    if n < 20:
        return []
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    try:
        if time_column and time_column in train_df.columns:
            order = pd.to_datetime(train_df[time_column], errors="coerce").rank(method="first", na_option="bottom")
        else:
            order = pd.Series(np.arange(n), index=train_df.index)
        order_vals = order.values
        for cutoff_pct in [0.80, 0.60]:
            cutoff = np.percentile(order_vals, cutoff_pct * 100)
            fit_mask = order_vals <= cutoff
            hd_mask = ~fit_mask
            if 4 <= hd_mask.sum() < n:
                splits.append((np.flatnonzero(fit_mask), np.flatnonzero(hd_mask)))
    except Exception:
        pass
    return splits[:2]


def _reasonable_upper_clip(y: pd.Series) -> float | None:
    if not (float(y.min()) >= 0):
        return None
    q99 = float(y.quantile(0.99))
    max_y = float(y.max())
    if not np.isfinite(q99) or not np.isfinite(max_y):
        return None
    return max(max_y * 1.5, q99 * 2.0)
