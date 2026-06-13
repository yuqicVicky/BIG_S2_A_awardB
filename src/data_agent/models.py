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
import os
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from .heartbeat import emit
from .task import (
    ACCURACY,
    BINARY,
    BLOCK_MAE,
    F1,
    F1_MACRO,
    LOG_LOSS,
    MAE,
    MULTICLASS,
    REGRESSION,
    RMSE,
    ROC_AUC,
)

try:  # sklearn is preferred, but baseline models can run without it.
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import GroupKFold, KFold, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

    SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - minimal evaluation runtimes only
    ColumnTransformer = object  # type: ignore
    SKLEARN_AVAILABLE = False

    def mean_absolute_error(y_true, y_pred):
        return float(np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))))

    def mean_squared_error(y_true, y_pred):
        d = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
        return float(np.mean(d * d))

    def r2_score(y_true, y_pred):
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)
        ss_res = float(np.sum((yt - yp) ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

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


# Per-group target statistics emitted as features (version2's hist_* signal).
_AGG_STATS = ("mean", "median", "std", "min", "max", "q25", "q75", "count")


class GroupMedianImputer:
    """Leakage-safe per-group median imputation for high-missingness numeric columns.

    ``fit(X, y=None)`` computes, per group key, the median of each target
    numeric column (fit on the training fold only). ``transform`` fills missing
    values in those columns using the per-group median, falling back to the
    global median for groups unseen during fit. Fit *inside* the Pipeline so
    validation/test rows never influence the imputation statistics — no leakage.

    Generic: the group column and columns to impute are selected dynamically
    (highest-cardinality non-aggregate string column; columns with any NaN).
    No column name is hardcoded.
    """

    def __init__(self, group_col: str | None = None, impute_cols: list[str] | None = None):
        self.group_col = group_col          # discovered at fit-time when None
        self.impute_cols = impute_cols      # discovered at fit-time when None
        self._group_col_fit: str | None = None
        self._impute_cols_fit: list[str] = []
        self._medians: dict[str, dict] = {}  # col -> {group -> median}
        self._global: dict[str, float] = {}  # col -> global median fallback

    def get_params(self, deep: bool = True) -> dict:
        return {"group_col": self.group_col, "impute_cols": self.impute_cols}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: pd.DataFrame, y=None) -> "GroupMedianImputer":
        Xr = X.reset_index(drop=True)

        # Resolve the group column: use the caller-supplied one if valid, else
        # pick the string column with the most unique values among short-string
        # (categorical) columns — excluding free-text (mean token length >= 3).
        gc = self.group_col if (self.group_col and self.group_col in Xr.columns) else None
        if gc is None:
            best, best_n = None, 0
            for col in Xr.columns:
                if not (pd.api.types.is_object_dtype(Xr[col]) or pd.api.types.is_string_dtype(Xr[col])):
                    continue
                non_empty = Xr[col].dropna().astype(str)
                non_empty = non_empty[non_empty.str.strip() != ""]
                if len(non_empty) == 0:
                    continue
                # Skip free-text columns (long average token length)
                mean_tokens = float(non_empty.str.split().map(len).mean())
                if mean_tokens >= 3.0:
                    continue
                n = int(Xr[col].nunique(dropna=True))
                if n > best_n:
                    best, best_n = col, n
            gc = best
        self._group_col_fit = gc

        # Resolve columns to impute: caller-supplied list if given, else all
        # numeric columns (fit per-group medians for ALL numeric cols — even those
        # fully present in train — so that transform can fill val/test rows whose
        # missing values were not visible during training).
        if self.impute_cols is not None:
            cols = [c for c in self.impute_cols if c in Xr.columns]
        else:
            cols = [
                c for c in Xr.columns
                if c != gc and pd.api.types.is_numeric_dtype(Xr[c])
                and not pd.api.types.is_bool_dtype(Xr[c])
            ]
        self._impute_cols_fit = cols

        self._global = {}
        self._medians = {}
        for col in cols:
            num = pd.to_numeric(Xr[col], errors="coerce")
            self._global[col] = float(num.median()) if num.notna().any() else 0.0
            if gc and gc in Xr.columns:
                frame = pd.DataFrame({"g": Xr[gc].astype(str), "v": num})
                grp_med = frame.groupby("g", dropna=False)["v"].median().to_dict()
                # Replace NaN group medians with global fallback
                self._medians[col] = {
                    k: float(v) if (v is not None and pd.notna(v)) else self._global[col]
                    for k, v in grp_med.items()
                }
            else:
                self._medians[col] = {}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._impute_cols_fit:
            return X
        out = X.copy()
        gc = self._group_col_fit
        for col in self._impute_cols_fit:
            if col not in out.columns:
                continue
            missing_mask = out[col].isna()
            if not missing_mask.any():
                continue
            if gc and gc in out.columns and self._medians.get(col):
                group_keys = out[gc].astype(str)
                fallback = self._global.get(col, 0.0)
                fill_vals = group_keys.map(
                    lambda k, d=self._medians[col], fb=fallback: d.get(k, fb)
                )
                out.loc[missing_mask, col] = fill_vals[missing_mask]
            else:
                out.loc[missing_mask, col] = self._global.get(col, 0.0)
        return out

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y).transform(X)


def _agg_col_name(keys: list[str], stat: str) -> str:
    return f"tgt_{stat}__" + "_".join(keys)


def _aggregate_feature_names(group_specs) -> list[str]:
    names: list[str] = []
    for keys in (group_specs or []):
        for stat in _AGG_STATS:
            names.append(_agg_col_name(list(keys), stat))
    return names


class GroupTargetAggregator:
    """Leakage-safe per-group target statistics as model features.

    ``fit(X, y)`` computes, on the supplied rows only, target statistics per
    group key; ``transform`` maps them onto rows by group key with a global
    fallback for unseen groups. Fit *inside* the model Pipeline, so during
    cross-validation / holdout it only ever sees the training fold — the encoded
    statistics never leak the row's own target into evaluation. This is the
    dominant signal on a stable panel (a group's own history) and generalises to
    any dataset with overlapping group keys.
    """

    def __init__(self, group_specs, stats=None):
        self.group_specs = group_specs
        self.stats = list(stats) if stats is not None else list(_AGG_STATS)
        self.tables_: dict[tuple, pd.DataFrame] = {}
        self.global_: dict[str, float] = {}

    def get_params(self, deep: bool = True) -> dict:
        return {"group_specs": self.group_specs, "stats": self.stats}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: pd.DataFrame, y) -> "GroupTargetAggregator":
        yv = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
        Xr = X.reset_index(drop=True)
        self.global_ = {
            "mean": float(yv.mean()),
            "median": float(yv.median()),
            "std": float(yv.std()) if len(yv) > 1 else 0.0,
            "min": float(yv.min()),
            "max": float(yv.max()),
            "q25": float(yv.quantile(0.25)),
            "q75": float(yv.quantile(0.75)),
            "count": 0.0,
        }
        frame = Xr.copy()
        frame["__y__"] = yv.values
        self.tables_ = {}
        for keys in (self.group_specs or []):
            keys = list(keys)
            if not all(k in Xr.columns for k in keys):
                continue
            g = frame.groupby(keys, dropna=False)["__y__"]
            self.tables_[tuple(keys)] = pd.DataFrame({
                "mean": g.mean(), "median": g.median(), "std": g.std().fillna(0.0),
                "min": g.min(), "max": g.max(),
                "q25": g.quantile(0.25), "q75": g.quantile(0.75),
                "count": g.count().astype(float),
            })
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out_idx = out.index
        for keys in (self.group_specs or []):
            keys = list(keys)
            tbl = self.tables_.get(tuple(keys))
            have_keys = all(k in out.columns for k in keys)
            if tbl is None or not have_keys:
                # Fill with global fallback — no merge needed
                for stat in self.stats:
                    col = _agg_col_name(keys, stat)
                    fill = 0.0 if stat == "count" else self.global_.get(stat, 0.0)
                    out[col] = fill
                continue

            # Single merge per group spec (not one per stat) — significantly
            # faster on large DataFrames: 8 stat columns joined in one pass.
            valid_stats = [s for s in self.stats if s in tbl.columns]
            merged = out[keys].reset_index(drop=True).merge(
                tbl[valid_stats], how="left", left_on=keys, right_index=True
            )
            for stat in self.stats:
                col = _agg_col_name(keys, stat)
                fill = 0.0 if stat == "count" else self.global_.get(stat, 0.0)
                if stat in valid_stats:
                    out[col] = merged[stat].fillna(fill).to_numpy()
                else:
                    out[col] = fill
        return out

    def fit_transform(self, X, y=None, **kwargs):
        return self.fit(X, y).transform(X)


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

def train_and_predict(
    bundle: FeatureBundle, block_column: str | None = None, random_state: int = 42,
    families: set[str] | None = None,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
    scored_mask: np.ndarray | None = None,
) -> ModelResult:
    """Train candidates and predict. When ``folds`` is given (the canonical shared
    folds from ``cv.load_canonical_folds`` aligned to the valid-target frame), CV
    uses them verbatim instead of building its own — so every Step-6 candidate
    scores OOF on the SAME partition. ``scored_mask`` restricts OOF scoring to the
    held-out rows of single-holdout strategies. Both default ``None`` =
    backward-compatible behaviour."""
    if bundle.task.task_type == REGRESSION:
        return _train_regression(bundle, block_column, random_state, families=families,
                                 folds=folds, scored_mask=scored_mask)
    return _train_classification(bundle, random_state, families=families)


def _candidate_family(name: str) -> str:
    """Coarse family tag for a candidate model, used by the modeling-group
    specialists to train a focused subset. Dataset-agnostic (name-based)."""
    n = name.lower()
    if n.startswith("baseline") or n.startswith("dummy"):
        return "baseline"
    if any(t in n for t in ("lightgbm", "xgboost", "catboost", "hist_gradient", "gradient_boosting")) or n.startswith("hgb"):
        return "gbdt"
    if n.startswith("ridge") or n.startswith("elastic") or "linear" in n or "logistic" in n:
        return "linear"
    if "random_forest" in n or "extra_trees" in n:
        return "trees"
    return "other"


# ── regression path (Award A behaviour, preserved) ─────────────────────────────

def _train_regression(bundle: FeatureBundle, block_column: str | None, random_state: int,
                      families: set[str] | None = None,
                      folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
                      scored_mask: np.ndarray | None = None) -> ModelResult:
    y = pd.to_numeric(bundle.target, errors="coerce")
    valid_mask = y.notna()
    if valid_mask.sum() < 5:
        raise ValueError("Need at least five non-missing target rows to train a model.")

    train_df = bundle.train_df.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].reset_index(drop=True)
    X = train_df[bundle.feature_columns].copy()
    X_pred = bundle.predict_df[bundle.feature_columns].copy()

    metric = getattr(bundle.task, "metric", None)
    greater = bool(getattr(bundle.task, "greater_is_better", False))
    _, metric_name = _score_function(metric, block_column, train_df)

    # ── cross-validation folds (group whole periods out when possible) ─────────
    # Guard against a unique-per-row key (e.g. an hourly timestamp used as the row
    # id) silently collapsing GroupKFold into ordinary KFold: resolve to a sane
    # block, coarsening a too-granular datetime key to whole-period blocks so the
    # CV actually mirrors held-out periods. Dataset-agnostic.
    group_col = bundle.profile.get("time_column") or block_column
    if folds is not None:
        # Canonical shared folds (cv.py) — every Step-6 candidate uses these exact
        # folds, so OOF scores are directly comparable for keep-best.
        cv_desc = {"type": "canonical_external", "n_splits": len(folds), "group_column": group_col}
    else:
        groups, block_reason = _resolve_cv_groups(train_df, group_col, len(X))
        folds, cv_desc = _make_cv_folds(len(X), groups, random_state)
        cv_desc["group_column"] = group_col
        if block_reason:
            cv_desc["block_reason"] = block_reason
    if scored_mask is not None:
        scored_mask = np.asarray(scored_mask, dtype=bool)
        if len(scored_mask) != len(X):
            scored_mask = None  # misaligned — ignore rather than corrupt scoring

    def _score_idx(idx, y_true, y_pred) -> float:
        hf = train_df.iloc[idx].reset_index(drop=True)
        sfn, _ = _score_function(metric, block_column, hf)
        return float(sfn(np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)))

    candidates = _build_candidates(bundle, train_df, random_state)
    if families:
        # A modeling-group specialist trains a focused subset; baselines are
        # always kept so a valid fallback exists.
        keep = set(families) | {"baseline"}
        filtered = [(n, f) for (n, f) in candidates if _candidate_family(n) in keep]
        if any(_candidate_family(n) in families for n, _ in filtered):
            candidates = filtered
    budget = _TimeBudget(_time_budget_seconds())

    scores: list[dict] = []
    oof_preds: dict[str, np.ndarray] = {}
    factories: dict[str, Callable] = {}
    cand_score: dict[str, float] = {}

    # ── out-of-fold cross-validation for every candidate ──────────────────────
    # The wall-clock budget is enforced *between folds* (not only between
    # candidates) and a per-candidate ceiling stops any single runaway model
    # (e.g. a near-OLS ElasticNet on a wide one-hot matrix) from consuming the
    # whole slice. A candidate that cannot finish all folds in time is abandoned
    # rather than registered with a gap-filled OOF that would corrupt selection
    # (the common-OOF NNLS keep-best requires every candidate scored on the same
    # full partition).
    for name, factory in candidates:
        if budget.exhausted():
            scores.append({"name": name, "status": "skipped", "error": "time_budget_exhausted"})
            continue
        # Per-candidate ceiling: at most half the total slice (floored at 90s) so
        # one model can never eat the budget, while legitimate slow models
        # (e.g. RandomForest) still complete. With the per-fold deadline check
        # below this bounds any overrun to ~one fold.
        cand_cap = max(90.0, 0.5 * budget.total)
        cand_start = time.monotonic()
        try:
            oof = np.full(len(X), np.nan, dtype=float)
            fold_scores = []
            aborted = False
            for tr_idx, va_idx in folds:
                if budget.exhausted() or (time.monotonic() - cand_start) > cand_cap:
                    aborted = True
                    break
                est = factory()
                est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                p = _sanitize_predictions(est.predict(X.iloc[va_idx]), y)
                oof[va_idx] = p
                fold_scores.append(_score_idx(va_idx, y.iloc[va_idx], p))
            if aborted:
                # Do not register a partially-evaluated candidate: its OOF has
                # holes and its score is not comparable to full-fold candidates.
                scores.append({"name": name, "status": "skipped",
                               "error": "candidate_time_cap",
                               "detail": {"folds_done": int(len(fold_scores)),
                                          "elapsed_sec": round(time.monotonic() - cand_start, 1)}})
                emit("candidate_timeout", model=name,
                     elapsed_sec=round(time.monotonic() - cand_start, 1),
                     folds_done=int(len(fold_scores)))
                continue
            mask = ~np.isnan(oof)
            if scored_mask is not None:
                mask = mask & scored_mask
            if mask.sum() < 5:
                scores.append({"name": name, "status": "failed", "error": "insufficient_oof_coverage"})
                continue
            overall = _score_idx(np.flatnonzero(mask), y[mask], oof[mask])
            mae = float(mean_absolute_error(y[mask], oof[mask]))
            scores.append({
                "name": name, "status": "ok", "score": overall,
                "detail": {"mae": mae, metric_name: overall,
                           "cv_mean": float(np.mean(fold_scores)),
                           "cv_std": float(np.std(fold_scores)),
                           "cv_scores": [round(float(s), 4) for s in fold_scores]},
            })
            oof_preds[name] = oof
            factories[name] = factory
            cand_score[name] = overall
            emit("candidate_done", model=name, partial_cv=overall,
                 best_so_far=(max if greater else min)(cand_score.values()))
        except Exception as exc:
            scores.append({"name": name, "status": "failed", "error": str(exc)})

    if not factories:
        raise RuntimeError("All candidate models failed; cannot produce predictions.")

    # ── aggressive, budget-bounded hyperparameter tuning of the top models ─────
    _tune_top_models(
        factories, cand_score, oof_preds, scores, X, y, folds,
        _score_idx, metric_name, greater, budget, random_state,
    )

    # ── stacking: convex blend of the best base models' OOF predictions ────────
    stack = _build_stack(oof_preds, cand_score, y, greater, _score_idx, metric_name)
    if stack is not None:
        scores.append(stack["score_entry"])
        cand_score[stack["name"]] = stack["score"]
        oof_preds[stack["name"]] = stack["oof"]  # expose stacked OOF so write_oof finds it

    selected_name = (max if greater else min)(cand_score, key=lambda n: cand_score[n])

    # ── final fit + predict (seed-averaged), stacking-aware ────────────────────
    if stack is not None and selected_name == stack["name"]:
        predictions = _predict_stack(stack, factories, X, y, X_pred, random_state)
    else:
        predictions = _final_predict(factories[selected_name], X, y, X_pred, random_state)
    predictions = _sanitize_predictions(predictions, y)

    clip_min = 0.0 if float(y.min()) >= 0 else None
    clip_max = _reasonable_upper_clip(y)
    if clip_min is not None or clip_max is not None:
        predictions = np.clip(
            predictions,
            clip_min if clip_min is not None else -np.inf,
            clip_max if clip_max is not None else np.inf,
        )

    # ── isotonic post-calibration (distribution shift guard) ─────────────────
    # If the test predictions are substantially shifted relative to training
    # (test median > 1.5x training median) while OOF predictions match training
    # well, the model is extrapolating outside its training range. Fitting an
    # isotonic regressor on (OOF predictions → observed y) and applying it to
    # the test predictions corrects systematic level shifts without overfitting.
    # Dataset-agnostic: the shift ratio is computed purely from prediction arrays.
    predictions = _maybe_isotonic_calibrate(predictions, oof_preds, selected_name, stack, y)

    holdout_strategy = dict(cv_desc)
    selected_detail = _selected_detail(scores, selected_name)
    # Per-fold CV stability for the selected model. A stack's own detail carries
    # no per-fold scores, so fall back to its top base (representative variance).
    _stab_detail = selected_detail
    if "cv_std" not in _stab_detail and stack is not None and selected_name == stack["name"] and stack["bases"]:
        _stab_detail = _selected_detail(scores, stack["bases"][0])
    if "cv_std" in _stab_detail:
        _cm = float(_stab_detail.get("cv_mean", _stab_detail.get(metric_name, 0.0)))
        _cs = float(_stab_detail.get("cv_std", 0.0))
        holdout_strategy["stability"] = {
            "split_scores": _stab_detail.get("cv_scores", []),
            "cv_mae_mean": round(_cm, 4),
            "cv_mae_std": round(_cs, 4),
            "relative_stability": round(_cs / _cm, 4) if _cm > 0 else None,
        }

    # selected model's OOF arrays (full coverage) for plotting / residuals
    sel_oof = oof_preds.get(selected_name)
    if sel_oof is None and stack is not None and selected_name == stack["name"]:
        sel_oof = stack.get("oof")
    if sel_oof is not None:
        m = ~np.isnan(sel_oof)
        if scored_mask is not None:
            m = m & scored_mask
        y_true_holdout = np.asarray(y[m], dtype=float)
        y_pred_holdout = sel_oof[m]
    else:
        y_true_holdout, y_pred_holdout = np.asarray(y, dtype=float), None
    residual_analysis = _compute_residual_analysis(y_true_holdout, y_pred_holdout) if y_pred_holdout is not None else {}

    return ModelResult(
        predictions=predictions,
        selected_model_name=selected_name,
        model_scores=scores,
        holdout_strategy=holdout_strategy,
        metric_name=metric_name,
        greater_is_better=greater,
        task_type=REGRESSION,
        output_kind="value",
        extra_metrics=selected_detail,
        target_clip_min=clip_min,
        target_clip_max=clip_max,
        holdout_y_true=y_true_holdout,
        holdout_y_pred=y_pred_holdout,
        holdout_by_model=oof_preds,
        residual_analysis=residual_analysis,
    )


# ── cross-validation, stacking, seed-averaging helpers ─────────────────────────

def _time_budget_seconds() -> float:
    """Global wall-clock budget for model search (env-overridable). Default 45
    min — keeps the full pipeline inside the Award-B 2-hour cap including I/O and
    report generation (typically 15-20 min). Set AWARDB_TIME_BUDGET_SEC to override."""
    try:
        return float(os.environ.get("AWARDB_TIME_BUDGET_SEC", "2700"))
    except Exception:
        return 2700.0


class _TimeBudget:
    def __init__(self, seconds: float):
        self.total = max(30.0, float(seconds))
        self.deadline = time.monotonic() + self.total

    def exhausted(self) -> bool:
        return time.monotonic() >= self.deadline

    def remaining(self) -> float:
        """Seconds left before the wall-clock deadline (never negative)."""
        return max(0.0, self.deadline - time.monotonic())


def _n_jobs() -> int:
    """Safe parallel-jobs count. Default 1 avoids macOS fork-based deadlocks.
    Set AWARDB_N_JOBS=-1 on Linux for full parallelism."""
    try:
        return int(os.environ.get("AWARDB_N_JOBS", "1"))
    except Exception:
        return 1


def _coarsen_datetime_blocks(series: pd.Series, n: int, k: int = 5):
    """Derive a coarser temporal block from a too-granular datetime-like column so a
    blocked CV holds *whole periods* out instead of one row per group. Tries
    year -> year-month -> ISO-week -> calendar-day and keeps a granularity with a
    healthy block count. Returns ``(keys, label)`` or ``(None, None)``. Dataset-
    agnostic; never coerces a numeric id column into epoch timestamps."""
    if pd.api.types.is_numeric_dtype(series):
        return None, None
    dt = pd.to_datetime(series, errors="coerce")
    if float(dt.notna().mean()) < 0.8:
        return None, None
    min_groups = max(2 * k, 6)
    max_groups = max(min_groups, n // 4)
    grans = [("year", "%Y"), ("year_month", "%Y-%m"), ("year_week", "%G-%V"), ("date", "%Y-%m-%d")]
    counted = []
    for label, fmt in grans:
        keys = dt.dt.strftime(fmt).fillna("NaT").astype(str)
        g = int(keys.nunique())
        counted.append((label, keys, g))
        if min_groups <= g <= max_groups:
            return keys, f"{label} ({g} blocks)"
    eligible = [(lbl, keys, g) for (lbl, keys, g) in counted if g >= min_groups]
    if eligible:  # none in the ideal band: finest with enough blocks for k folds
        lbl, keys, g = eligible[-1]
        return keys, f"{lbl} ({g} blocks)"
    usable = [(lbl, keys, g) for (lbl, keys, g) in counted if g >= 2]
    if usable:  # last resort: coarsest key with at least two blocks
        lbl, keys, g = usable[0]
        return keys, f"{lbl} ({g} blocks)"
    return None, None


def _resolve_cv_groups(train_df: pd.DataFrame, group_col: str | None, n: int, k: int | None = None):
    """Resolve the GroupKFold grouping series, guarding against a unique-per-row key
    (an hourly timestamp, a row id) that would silently collapse GroupKFold into
    ordinary KFold. A too-granular datetime key is coarsened to whole-period blocks.
    Returns ``(groups, reason)``; ``groups is None`` means shuffled KFold."""
    if k is None:
        k = _max_splits()
    if not group_col or group_col not in train_df.columns:
        return None, None
    s = train_df[group_col].reset_index(drop=True)
    nun = int(s.nunique(dropna=False))
    if nun < 3:
        return None, f"'{group_col}' has {nun} distinct value(s); shuffled KFold"
    if nun <= n // 2:
        return s.astype(str), None  # healthy block — hold whole groups out as-is
    coarse, label = _coarsen_datetime_blocks(s, n, k)
    if coarse is not None:
        return coarse, f"coarsened '{group_col}' -> {label} (raw {nun}/{n} ~unique-per-row)"
    return None, f"'{group_col}' too granular ({nun}/{n}); shuffled KFold"


def _make_cv_folds(n: int, groups: pd.Series | None, random_state: int, max_splits: int | None = None):
    """Return ``(folds, description)``. Group whole periods out (GroupKFold) when
    a grouping key is available so validation mirrors the disjoint hidden periods;
    otherwise shuffled KFold. Degrades to manual folds without sklearn."""
    if max_splits is None:
        max_splits = _max_splits()
    if SKLEARN_AVAILABLE and groups is not None:
        ng = int(groups.nunique())
        # Only group when the key yields whole held-out blocks. A near-unique-per-row
        # key (ng ~ n) would collapse GroupKFold into ordinary KFold — fall back
        # honestly instead (callers coarsen such keys upstream via _resolve_cv_groups).
        if 3 <= ng <= max(3, n // 2):
            k = int(min(max_splits, ng))
            folds = list(GroupKFold(n_splits=k).split(np.arange(n), groups=groups.to_numpy()))
            return folds, {"type": "grouped_kfold", "n_splits": k, "n_groups": ng}
    if SKLEARN_AVAILABLE:
        k = int(min(max_splits, max(2, n // 2)))
        folds = list(KFold(n_splits=k, shuffle=True, random_state=random_state).split(np.arange(n)))
        return folds, {"type": "kfold", "n_splits": k}
    k = int(min(max_splits, max(2, n // 2)))
    chunks = np.array_split(np.arange(n), k)
    folds = [
        (np.concatenate([chunks[j] for j in range(k) if j != i]), chunks[i])
        for i in range(k)
    ]
    return folds, {"type": "manual_kfold", "n_splits": k}


def _seedable_param(est) -> str | None:
    try:
        params = est.get_params()
    except Exception:
        return None
    if "model__random_state" in params:
        return "model__random_state"
    if "random_state" in params:
        return "random_state"
    return None


def _seed_count() -> int:
    try:
        return max(1, int(os.environ.get("AWARDB_SEEDS", "2")))
    except Exception:
        return 2


def _max_splits() -> int:
    """Inner CV fold count. A real knob (default 5 → unchanged behaviour) so the
    orchestrator/watchdog can derive a leaner fold count under time pressure."""
    try:
        return max(2, int(os.environ.get("AWARDB_MAX_SPLITS", "5")))
    except Exception:
        return 5


def _final_predict(factory: Callable, X, y, X_pred, random_state: int, n_seeds: int | None = None) -> np.ndarray:
    """Fit on full training data and predict, averaging over several seeds for
    estimators that expose a ``random_state`` (variance reduction). Deterministic
    estimators are fit once."""
    if n_seeds is None:
        n_seeds = _seed_count()
    param = _seedable_param(factory())
    if not param or n_seeds <= 1:
        est = factory()
        est.fit(X, y)
        return np.asarray(est.predict(X_pred), dtype=float)
    preds = []
    for s in range(n_seeds):
        est = factory()
        try:
            est.set_params(**{param: random_state + 100 * (s + 1)})
        except Exception:
            pass
        est.fit(X, y)
        preds.append(np.asarray(est.predict(X_pred), dtype=float))
        emit("seed_done", fold=s)
    return np.mean(preds, axis=0)


def _build_stack(oof_preds, cand_score, y, greater, score_idx, metric_name, top_k: int = 3):
    """Convex (non-negative, sum-to-one) blend of the top base models' OOF
    predictions. Robust by construction — no extrapolation — and only kept when
    it beats the best single base on the same OOF rows."""
    ranked = sorted(cand_score, key=lambda n: cand_score[n], reverse=greater)
    bases = ranked[:top_k]
    if len(bases) < 2:
        return None
    mask = np.ones(len(y), dtype=bool)
    for n in bases:
        mask &= ~np.isnan(oof_preds[n])
    if int(mask.sum()) < 10:
        return None
    M = np.column_stack([oof_preds[n][mask] for n in bases])
    yv = np.asarray(y[mask], dtype=float)
    weights = None
    try:
        from scipy.optimize import nnls
        w, _ = nnls(M, yv)
        if w.sum() > 0:
            weights = w / w.sum()
    except Exception:
        weights = None
    if weights is None:
        weights = np.ones(len(bases)) / len(bases)
    stacked = M @ weights
    best_single = cand_score[bases[0]]
    stacked_score = score_idx(np.flatnonzero(mask), yv, stacked)
    improved = stacked_score > best_single if greater else stacked_score < best_single
    if not improved:
        return None
    full_oof = np.full(len(y), np.nan, dtype=float)
    full_oof[mask] = stacked
    name = "stack(" + "+".join(bases) + ")"
    return {
        "name": name,
        "bases": bases,
        "weights": [float(w) for w in weights],
        "score": stacked_score,
        "oof": full_oof,
        "score_entry": {
            "name": name, "status": "ok", "score": stacked_score,
            "detail": {metric_name: stacked_score,
                       "weights": {b: round(float(w), 3) for b, w in zip(bases, weights)}},
        },
    }


def _predict_stack(stack, factories, X, y, X_pred, random_state) -> np.ndarray:
    total = None
    for name, w in zip(stack["bases"], stack["weights"]):
        p = w * _final_predict(factories[name], X, y, X_pred, random_state)
        total = p if total is None else total + p
    return total


def _tune_iters() -> int:
    # Wall-clock (not tokens) is the binding Award B cap, and tuning dominates it.
    # A leaner default + early-stopping (see _tune_top_models) keeps the cheap wins
    # and drops the long tail of no-improvement iterations that only chase CV noise.
    try:
        return max(0, int(os.environ.get("AWARDB_TUNE_ITER", "6")))
    except Exception:
        return 6


def _py(v):
    return v.item() if isinstance(v, np.generic) else v


def _param_space(model) -> dict:
    """Per-family randomized-search space, keyed by estimator class name.
    Dataset-agnostic — only the model type matters, never any column."""
    cls = type(model).__name__
    if cls == "LGBMRegressor":
        return {"num_leaves": [31, 63, 95, 127], "learning_rate": [0.02, 0.03, 0.05],
                "n_estimators": [600, 1000, 1500], "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0], "min_child_samples": [10, 20, 40],
                "reg_lambda": [0.0, 0.1, 1.0]}
    if cls == "CatBoostRegressor":
        return {"depth": [4, 6, 8], "learning_rate": [0.02, 0.03, 0.05],
                "l2_leaf_reg": [1.0, 3.0, 5.0, 9.0], "iterations": [800, 1200, 2000]}
    if cls == "XGBRegressor":
        return {"max_depth": [4, 6, 8], "learning_rate": [0.02, 0.03, 0.05],
                "n_estimators": [600, 1000, 1500], "subsample": [0.7, 0.85, 1.0],
                "colsample_bytree": [0.7, 0.85, 1.0], "reg_lambda": [0.5, 1.0, 2.0]}
    if cls == "HistGradientBoostingRegressor":
        return {"max_iter": [300, 600, 900], "learning_rate": [0.03, 0.05, 0.08],
                "max_leaf_nodes": [31, 63, 127], "l2_regularization": [0.0, 0.1, 1.0],
                "min_samples_leaf": [10, 20, 40]}
    if cls in ("RandomForestRegressor", "ExtraTreesRegressor"):
        return {"n_estimators": [300, 600], "max_depth": [None, 12, 20],
                "min_samples_leaf": [1, 2, 4], "max_features": ["sqrt", 0.5, 1.0]}
    return {}


def _tune_top_models(factories, cand_score, oof_preds, scores, X, y, folds,
                     score_idx, metric_name, greater, budget, random_state, n_families: int = 2):
    """Randomized search over the top model families using the same CV folds and
    block-MAE scorer. A tuned config is added as a new candidate only when it
    beats its base on OOF. Bounded by the global wall-clock budget so it stays
    inside the 2-hour cap; a tuning failure degrades to the default params."""
    if not SKLEARN_AVAILABLE or _tune_iters() <= 0 or budget.exhausted():
        return
    rng = np.random.default_rng(random_state)
    ranked = sorted(cand_score, key=lambda n: cand_score[n], reverse=greater)
    tuned = 0
    for base_name in ranked:
        if tuned >= n_families or budget.exhausted():
            break
        factory = factories.get(base_name)
        if factory is None:
            continue
        try:
            steps = getattr(factory(), "named_steps", {})
        except Exception:
            continue
        if "model" not in steps:
            continue  # only tune plain Pipeline(preprocess, model) candidates
        space = _param_space(steps["model"])
        if not space:
            continue
        tuned += 1
        best_cfg, best_oof, best_score = None, None, cand_score[base_name]
        # Early-stop a plateaued search: stop this family after `patience`
        # consecutive non-improving draws. Brute-force depth only buys CV noise, so
        # this reclaims wall-clock for the (token-cheap) real subagent layer.
        patience = max(4, _tune_iters() // 3)
        no_improve = 0
        for _ in range(_tune_iters()):
            if budget.exhausted() or no_improve >= patience:
                break
            cfg = {f"model__{k}": _py(rng.choice(np.array(v, dtype=object))) for k, v in space.items()}
            try:
                oof = np.full(len(X), np.nan, dtype=float)
                budget_hit = False
                for tr_idx, va_idx in folds:
                    if budget.exhausted():
                        budget_hit = True
                        break
                    est = factory().set_params(**cfg)
                    est.fit(X.iloc[tr_idx], y.iloc[tr_idx])
                    oof[va_idx] = _sanitize_predictions(est.predict(X.iloc[va_idx]), y)
                if budget_hit:
                    break  # out of time mid-draw: stop tuning this family
                m = ~np.isnan(oof)
                sc = score_idx(np.flatnonzero(m), y[m], oof[m])
            except Exception:
                continue
            if (sc > best_score) if greater else (sc < best_score):
                best_cfg, best_oof, best_score = cfg, oof, sc
                no_improve = 0
            else:
                no_improve += 1
        if best_cfg is not None:
            tname = f"tuned({base_name})"
            factories[tname] = (lambda f=factory, c=dict(best_cfg): f().set_params(**c))
            oof_preds[tname] = best_oof
            cand_score[tname] = best_score
            scores.append({"name": tname, "status": "ok", "score": best_score,
                           "detail": {metric_name: best_score, "tuned_from": base_name,
                                      "params": {k.replace("model__", ""): v for k, v in best_cfg.items()}}})
            emit("tune_done", model=tname, partial_cv=best_score)


# ── classification path ─────────────────────────────────────────────────────────

def _train_classification(bundle: FeatureBundle, random_state: int,
                          families: set[str] | None = None) -> ModelResult:
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
    if families:
        # A modeling-group specialist trains a focused subset; baselines are always
        # kept so a valid fallback exists. Mirrors the regression filter so the
        # keep-best group is task-agnostic. If the requested family is absent from
        # the (lean) classification pool, fall back to the full set.
        keep = set(families) | {"baseline"}
        filtered = [(n, f) for (n, f) in candidates if _candidate_family(n) in keep]
        if any(_candidate_family(n) in families for n, _ in filtered):
            candidates = filtered
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
) -> tuple[np.ndarray, np.ndarray, dict]:
    if time_column and time_column in train_df.columns:
        raw = train_df[time_column]
        values = raw.astype(str)
        # Order unique periods by their *parsed* datetime (or numeric) value so
        # the held-out 20% are genuinely the latest periods (a leakage-safe time
        # holdout). When the key is neither datetime- nor numeric-parseable
        # (e.g. an opaque/hashed period id), do NOT trust lexicographic order —
        # fall through to a grouped-block holdout below.
        parsed = pd.to_datetime(raw, errors="coerce")
        dt_rate = float(parsed.notna().mean())
        order_key = None
        ordering = None
        if dt_rate >= 0.8:
            order_key, ordering = parsed, "datetime"
        else:
            numeric = pd.to_numeric(raw, errors="coerce")
            if float(numeric.notna().mean()) >= 0.8:
                order_key, ordering = numeric, "numeric"
        if order_key is not None:
            _tmp = pd.DataFrame({"orig": values, "key": order_key})
            unique_values = (
                _tmp.dropna(subset=["key"]).drop_duplicates("orig").sort_values("key")["orig"].tolist()
            )
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
                        "ordering": ordering,
                        "holdout_values": [str(v) for v in unique_values[-n_holdout_periods:]],
                        "n_train": int(len(train_idx)),
                        "n_holdout": int(len(holdout_idx)),
                    }
        else:
            # Opaque / unorderable block key: hold out whole periods at random so
            # none appears in both partitions — mirrors the disjoint validation
            # periods of a panel and needs no ordering.
            uniq = values.drop_duplicates().tolist()
            if len(uniq) >= 5:
                rng = np.random.default_rng(random_state)
                n_holdout_periods = max(1, int(round(len(uniq) * 0.2)))
                perm = rng.permutation(np.array(uniq, dtype=object))
                holdout_values = set(perm[:n_holdout_periods].tolist())
                holdout_mask = values.isin(holdout_values).to_numpy()
                if 0 < holdout_mask.sum() < len(train_df):
                    holdout_idx = np.flatnonzero(holdout_mask)
                    train_idx = np.flatnonzero(~holdout_mask)
                    return train_idx, holdout_idx, {
                        "type": "grouped_block_holdout",
                        "time_column": time_column,
                        "n_holdout_periods": int(n_holdout_periods),
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


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _score_function(
    metric: str | None, block_column: str | None, holdout_frame: pd.DataFrame
) -> tuple[Callable, str]:
    """Return the ``(scorer, name)`` pair for the resolved regression metric.

    Award B is graded by block-averaged MAE, so a discovered block column selects
    ``block_mae`` (the resolved default when no other metric is stated). An
    explicit ``rmse`` / ``mae`` in the description still wins. Dataset-agnostic.
    """
    if metric == RMSE:
        return _rmse, "rmse"
    if metric == MAE:
        return (lambda y_true, y_pred: float(mean_absolute_error(y_true, y_pred))), "mae"
    # block_mae (resolved default when a block/category column exists and no
    # explicit metric was given) or any unrecognised metric.
    if block_column and block_column in holdout_frame.columns:
        blocks = holdout_frame[block_column].reset_index(drop=True)
        return (lambda y_true, y_pred: block_averaged_mae(y_true, y_pred, blocks)), "block_mae"
    return (lambda y_true, y_pred: float(mean_absolute_error(y_true, y_pred))), "mae"


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

    _gs = bundle.group_aggregate_keys
    _tx = bundle.text_columns

    # ── group-median imputer (leakage-safe per-group imputation) ──────────────
    # Fit per-group medians for ALL numeric feature columns using the highest-
    # cardinality short-string (non-free-text) column as the grouping key.
    # This fills missing values in both training and prediction frames using
    # fold-level training statistics — no leakage. Especially useful when
    # a numeric column is fully present in training but missing at predict time.
    # Generic: no column name is hardcoded; detection is by dtype + mean token length.
    _gmi_group_col: str | None = None
    _text_col_set = set(bundle.text_columns or [])
    # Identify the best grouping column: highest cardinality short-string column
    # that is NOT a free-text column (text columns have high mean token length).
    _best_gc_n = 0
    for col in bundle.feature_columns:
        if col not in train_df.columns or col in _text_col_set:
            continue
        if not (pd.api.types.is_object_dtype(train_df[col]) or pd.api.types.is_string_dtype(train_df[col])):
            continue
        n = int(train_df[col].nunique(dropna=True))
        if n > _best_gc_n:
            _gmi_group_col, _best_gc_n = col, n

    def _make_preprocessor_fresh(scale: bool) -> object:
        """Return a new preprocessor with freshly instantiated stateful transformers.

        Each call creates independent mutable objects so concurrent CV folds
        and repeated factory calls never share state. GroupTargetAggregator and
        GroupMedianImputer both accumulate per-fold fit state, so they MUST be
        freshly instantiated for every Pipeline construction — sharing them across
        factory calls would cause fold N's fitted statistics to bleed into fold N+1.
        """
        gmi = GroupMedianImputer(group_col=_gmi_group_col) if _gmi_group_col is not None else None
        return _make_preprocessor(
            bundle.numeric_columns, bundle.categorical_columns, scale_numeric=scale,
            text_columns=_tx, group_specs=_gs, group_median_imputer=gmi,
        )

    # Do NOT cache a single preprocessor and close over it in lambdas.
    # Each lambda must call _make_preprocessor_fresh() at call time so that every
    # Pipeline constructed during CV gets its own stateful transformer instances.

    candidates.append(
        (
            "hist_gradient_boosting",
            lambda: Pipeline([
                ("preprocess", _make_preprocessor_fresh(scale=False)),
                ("model", HistGradientBoostingRegressor(random_state=rs, max_iter=150, learning_rate=0.05)),
            ]),
        )
    )
    candidates.append(
        (
            "extra_trees",
            lambda: Pipeline([
                ("preprocess", _make_preprocessor_fresh(scale=False)),
                ("model", ExtraTreesRegressor(n_estimators=200, random_state=rs, n_jobs=_n_jobs(), min_samples_leaf=2, max_features="sqrt")),
            ]),
        )
    )
    candidates.append(
        (
            "random_forest",
            lambda: Pipeline([
                ("preprocess", _make_preprocessor_fresh(scale=False)),
                ("model", RandomForestRegressor(n_estimators=150, random_state=rs, n_jobs=_n_jobs(), min_samples_leaf=2)),
            ]),
        )
    )
    candidates.append(
        (
            "ridge",
            lambda: Pipeline([("preprocess", _make_preprocessor_fresh(scale=True)), ("model", Ridge(alpha=1.0))]),
        )
    )
    candidates.append(
        (
            "elastic_net",
            lambda: Pipeline([
                ("preprocess", _make_preprocessor_fresh(scale=True)),
                # max_iter/tol bounded + randomised coordinate selection so a
                # single fit on a wide one-hot matrix converges fast and can
                # never stall the slice (the previous alpha=0.001/max_iter=5000
                # near-OLS fit could run for many minutes per fold).
                ("model", ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=rs,
                                     max_iter=2000, tol=1e-3, selection="random")),
            ]),
        )
    )
    candidates.append(
        (
            "gradient_boosting",
            lambda: Pipeline([
                ("preprocess", _make_preprocessor_fresh(scale=False)),
                ("model", GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=rs, subsample=0.85)),
            ]),
        )
    )

    # ── target-transform variants (skew-driven) ───────────────────────────────
    # A right-skewed non-negative target (common for rate / count panels) gains
    # from a variance-stabilising transform. The choice is made purely from the
    # target's skew — dataset-agnostic. HistGradientBoosting is used so a
    # transformed candidate exists even when LightGBM is unavailable.
    _target_vals = pd.to_numeric(bundle.target, errors="coerce").dropna()
    _nonneg = bool(len(_target_vals) and float(_target_vals.min()) >= 0)
    add_sqrt = add_log = False
    if _nonneg and len(_target_vals) > 10:
        _skew = float(_target_vals.skew())
        add_sqrt = _skew > 0.5
        add_log = _skew > 1.5

    if add_log:
        candidates.append(
            (
                "hgb_log",
                lambda: LogTargetRegressor(Pipeline([
                    ("preprocess", _make_preprocessor_fresh(scale=False)),
                    ("model", HistGradientBoostingRegressor(random_state=rs, max_iter=150, learning_rate=0.05)),
                ])),
            )
        )
        # Skew-driven log target must reach EVERY family, not just GBDT — otherwise
        # the trees/linear specialists train on the raw skewed target and lose to the
        # floor's lightgbm_log purely on the transform. Give the trees and linear
        # families their own log-target variants so each specialist can pick it.
        candidates.append(
            (
                "extra_trees_log",
                lambda: LogTargetRegressor(Pipeline([
                    ("preprocess", _make_preprocessor_fresh(scale=False)),
                    ("model", ExtraTreesRegressor(n_estimators=200, random_state=rs, n_jobs=_n_jobs(), min_samples_leaf=2, max_features="sqrt")),
                ])),
            )
        )
        candidates.append(
            (
                "ridge_log",
                lambda: LogTargetRegressor(Pipeline([
                    ("preprocess", _make_preprocessor_fresh(scale=True)),
                    ("model", Ridge(alpha=1.0)),
                ])),
            )
        )
    if add_sqrt:
        candidates.append(
            (
                "hgb_sqrt",
                lambda: SqrtTargetRegressor(Pipeline([
                    ("preprocess", _make_preprocessor_fresh(scale=False)),
                    ("model", HistGradientBoostingRegressor(random_state=rs, max_iter=150, learning_rate=0.05)),
                ])),
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
                    ("preprocess", _make_preprocessor_fresh(scale=False)),
                    ("model", lgb.LGBMRegressor(
                        n_estimators=500,
                        learning_rate=0.04,
                        num_leaves=63,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        min_child_samples=20,
                        reg_alpha=0.1,
                        reg_lambda=0.1,
                        random_state=rs,
                        n_jobs=_n_jobs(),
                        verbose=-1,
                    )),
                ]),
            )
        )
        candidates.append(
            (
                "lightgbm_strong",
                lambda _sl=_strong_leaves: Pipeline([
                    ("preprocess", _make_preprocessor_fresh(scale=False)),
                    ("model", lgb.LGBMRegressor(
                        n_estimators=700,
                        learning_rate=0.02,
                        num_leaves=_sl,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        min_child_samples=20,
                        reg_alpha=0.05,
                        reg_lambda=0.05,
                        random_state=rs,
                        n_jobs=_n_jobs(),
                        verbose=-1,
                    )),
                ]),
            )
        )
        # LightGBM target-transform variants (same metric/skew flags as above)
        if add_sqrt:
            candidates.append(
                (
                    "lightgbm_sqrt",
                    lambda: SqrtTargetRegressor(Pipeline([
                        ("preprocess", _make_preprocessor_fresh(scale=False)),
                        ("model", lgb.LGBMRegressor(
                            n_estimators=500,
                            learning_rate=0.04,
                            num_leaves=63,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            min_child_samples=20,
                            reg_alpha=0.1,
                            reg_lambda=0.1,
                            random_state=rs,
                            n_jobs=_n_jobs(),
                            verbose=-1,
                        )),
                    ])),
                )
            )
        if add_log:
            candidates.append(
                (
                    "lightgbm_log",
                    lambda: LogTargetRegressor(Pipeline([
                        ("preprocess", _make_preprocessor_fresh(scale=False)),
                        ("model", lgb.LGBMRegressor(
                            n_estimators=500,
                            learning_rate=0.04,
                            num_leaves=63,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            min_child_samples=20,
                            reg_alpha=0.1,
                            reg_lambda=0.1,
                            random_state=rs,
                            n_jobs=_n_jobs(),
                            verbose=-1,
                        )),
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
                        ("preprocess", _make_preprocessor_fresh(scale=False)),
                        ("model", xgb.XGBRegressor(
                            n_estimators=500,
                            learning_rate=0.04,
                            max_depth=6,
                            subsample=0.85,
                            colsample_bytree=0.85,
                            reg_alpha=0.1,
                            reg_lambda=1.0,
                            random_state=rs,
                            n_jobs=_n_jobs(),
                            verbosity=0,
                            eval_metric="mae",
                        )),
                    ]),
                )
            )
        except Exception:
            pass

    # ── CatBoost (optional) ───────────────────────────────────────────────────
    try:
        from catboost import CatBoostRegressor  # type: ignore

        candidates.append((
            "catboost",
            lambda: Pipeline([
                ("preprocess", _make_preprocessor_fresh(scale=False)),
                ("model", CatBoostRegressor(
                    iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                    loss_function="MAE", random_seed=rs, thread_count=_n_jobs(),
                    allow_writing_files=False, verbose=False,
                )),
            ]),
        ))
        if add_log:
            candidates.append((
                "catboost_log",
                lambda: LogTargetRegressor(Pipeline([
                    ("preprocess", _make_preprocessor_fresh(scale=False)),
                    ("model", CatBoostRegressor(
                        iterations=600, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
                        loss_function="RMSE", random_seed=rs, thread_count=_n_jobs(),
                        allow_writing_files=False, verbose=False,
                    )),
                ])),
            ))
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

    preprocessor = _make_preprocessor(
        bundle.numeric_columns, bundle.categorical_columns, scale_numeric=False,
        text_columns=bundle.text_columns,
    )
    scaled_preprocessor = _make_preprocessor(
        bundle.numeric_columns, bundle.categorical_columns, scale_numeric=True,
        text_columns=bundle.text_columns,
    )

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
            "hist_gradient_boosting",
            lambda: Pipeline([
                ("preprocess", preprocessor),
                ("model", HistGradientBoostingClassifier(random_state=rs, max_iter=500, learning_rate=0.05)),
            ]),
        )
    )
    # Award B is always a regression task (block-averaged MAE), so the
    # classification path is a lean, general-purpose fallback only: baselines +
    # one linear model + one gradient-boosting model. No heavy/slow candidates.
    return candidates


def _try_regression_ensemble(
    fitted_candidates: list,
    holdout_capture: dict,
    score_fn: Callable,
    y_holdout: pd.Series,
    greater: bool = False,
) -> tuple | None:
    """Average top-2 regression models if both captured and the ensemble scores
    better on the selection metric. Honors metric direction (``greater``)."""
    ok = [(s, n, f) for s, n, f in fitted_candidates if n in holdout_capture]
    if len(ok) < 2:
        return None
    top2 = sorted(ok, key=lambda t: t[0], reverse=greater)[:2]
    best_score = top2[0][0]
    second_score = top2[1][0]
    # Skip ensemble if the second model is much worse (> 15% gap) — averaging would
    # hurt. The ratio gate is only meaningful for positive scores (mae/rmse/rmsle).
    if best_score > 0:
        if greater and second_score < best_score * 0.85:
            return None
        if not greater and second_score > best_score * 1.15:
            return None
    pred1 = holdout_capture[top2[0][1]]
    pred2 = holdout_capture[top2[1][1]]
    ensemble_pred = (pred1 + pred2) * 0.5
    ensemble_score = float(score_fn(y_holdout, ensemble_pred))
    improved = ensemble_score > best_score if greater else ensemble_score < best_score
    if not improved:
        return None
    name = f"ensemble({top2[0][1]}+{top2[1][1]})"
    f1, f2 = top2[0][2], top2[1][2]
    factory = lambda f1=f1, f2=f2: TopKAverageRegressor([f1, f2])
    return (ensemble_score, name, ensemble_pred, factory)


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


def _text_to_str(x):
    """Coerce a 1-D column (possibly with NaN) into a clean array of strings."""
    return pd.Series(np.asarray(x).ravel()).fillna("").astype(str).to_numpy()


def _text_pipeline():
    """TF-IDF (1-2 grams) → TruncatedSVD compression for a free-text column.

    sklearn-only and CPU-friendly; produces a small dense block of semantic
    features instead of exploding the text into useless one-hot levels.
    """
    return Pipeline([
        ("tostr", FunctionTransformer(_text_to_str)),
        ("tfidf", TfidfVectorizer(max_features=400, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("svd", TruncatedSVD(n_components=16, random_state=0)),
    ])


def _load_custom_head_transformers() -> list[tuple]:
    """Agent-authored, per-run feature transformers registered into the consumed
    model Pipeline. **No-op by default** — returns ``[]`` unless an optional
    ``custom_features`` module exposing ``build_head_transformers()`` is present.

    This is the seam for *agent-directed, code-enforced* feature engineering: the
    agent decides features per dataset and registers them here as transformers in
    the one pipeline the model trains on — never a standalone matrix. Each item is
    ``(step_name, transformer, [output_numeric_column_names])``:

    * the transformer is sklearn-compatible: ``fit(X: DataFrame, y) -> self`` and
      ``transform(X: DataFrame) -> DataFrame`` (it appends its columns and returns
      the whole frame, like :class:`GroupTargetAggregator`);
    * the output names are registered into the numeric feature set so the
      ``ColumnTransformer`` keeps them;
    * the factory must return **fresh** instances on each call — this runs once per
      fold via ``_make_preprocessor`` so per-fold fit state never leaks across folds.

    Because these run as head steps *inside* the per-fold Pipeline, any target use
    is fit on the training fold only (leakage-safe), and the leakage guard still
    checks their static output. Nothing here is hardcoded: the module is optional
    and resolves its own columns from the data.
    """
    try:
        from . import custom_features  # type: ignore  # optional, absent by default
    except Exception:
        return []
    factory = getattr(custom_features, "build_head_transformers", None)
    if not callable(factory):
        return []
    try:
        items = list(factory())
    except Exception:
        return []
    cleaned: list[tuple] = []
    for it in items:
        try:
            name, transformer, out_names = it
        except Exception:
            continue
        if transformer is None:
            continue
        cleaned.append((str(name), transformer, [str(c) for c in (out_names or [])]))
    return cleaned


def _make_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
    scale_numeric: bool,
    text_columns: list[str] | None = None,
    group_specs=None,
    group_median_imputer: "GroupMedianImputer | None" = None,
):
    """Build the feature preprocessor.

    Numeric (median-imputed, optionally scaled) + categorical (one-hot with
    infrequent grouping) + free-text (TF-IDF→SVD). When ``group_specs`` is given,
    a :class:`GroupTargetAggregator` is prepended so per-group target statistics
    become numeric features — fit per CV fold, hence leakage-safe.

    When ``group_median_imputer`` is given it is placed at the head of the pipeline
    (before ``GroupTargetAggregator``) so that per-group medians are computed on
    the training fold only and then used to fill missing values in the same fold
    before the aggregate statistics are computed — fully leakage-safe.
    """
    text_columns = text_columns or []
    group_specs = group_specs or []
    agg_names = _aggregate_feature_names(group_specs)
    numeric_all = list(numeric_columns) + agg_names

    # Agent-authored per-run transformers (no-op unless a custom_features module is
    # present). Their declared output columns join the numeric feature set so the
    # ColumnTransformer keeps them; the transformers themselves are prepended below.
    custom_head = _load_custom_head_transformers()
    for _cname, _ct, _couts in custom_head:
        for c in _couts:
            if c not in numeric_all:
                numeric_all.append(c)

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = []
    if numeric_all:
        transformers.append(("num", Pipeline(numeric_steps), numeric_all))
    if categorical_columns:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", _one_hot_encoder()),
            ]),
            categorical_columns,
        ))
    for i, col in enumerate(text_columns):
        # name must not contain "__" (sklearn reserves it for param nesting)
        transformers.append((f"text{i}", _text_pipeline(), col))

    column_transform = ColumnTransformer(transformers=transformers, remainder="drop")

    # Build the head steps: group_median_imputer (optional) → group_agg (optional) → columns
    head_steps: list[tuple] = []
    if group_median_imputer is not None:
        head_steps.append(("group_median_impute", group_median_imputer))
    if group_specs:
        head_steps.append(("group_agg", GroupTargetAggregator(group_specs)))
    # Agent-authored transformers run inside the per-fold Pipeline → leakage-safe.
    for _cname, _ct, _couts in custom_head:
        head_steps.append((f"custom_{_cname}", _ct))
    head_steps.append(("columns", column_transform))

    if len(head_steps) == 1:
        return column_transform
    return Pipeline(head_steps)


def _one_hot_encoder() -> OneHotEncoder:
    """One-hot encoder that *groups* rare categories instead of silently
    truncating to the top-N.

    Categories appearing in fewer than 1% of rows fold into a single
    "infrequent" level (``min_frequency``), with a generous hard cap
    (``max_categories``) as a backstop against feature-width explosion on
    very high-cardinality columns. Unknown categories seen at predict time also
    fold into the infrequent bucket. Degrades gracefully across sklearn
    versions that lack these parameters.
    """
    for kwargs in (
        dict(handle_unknown="infrequent_if_exist", sparse_output=False, min_frequency=0.01, max_categories=100),
        dict(handle_unknown="ignore", sparse_output=False, max_categories=50),
        dict(handle_unknown="ignore", sparse=False),
    ):
        try:
            return OneHotEncoder(**kwargs)
        except TypeError:
            continue
    return OneHotEncoder(handle_unknown="ignore")


def _sanitize_predictions(pred: np.ndarray, y_train: pd.Series) -> np.ndarray:
    pred = np.asarray(pred, dtype=float)
    finite_mean = float(pd.to_numeric(y_train, errors="coerce").mean())
    pred = np.where(np.isfinite(pred), pred, finite_mean)
    return pred


def _maybe_isotonic_calibrate(
    predictions: np.ndarray,
    oof_preds: dict,
    selected_name: str,
    stack,
    y: pd.Series,
) -> np.ndarray:
    """Apply isotonic regression post-calibration when test predictions are
    substantially shifted above the training distribution.

    Triggered when test_median / train_median > 1.5.  Fits an isotonic
    regressor on (OOF predictions → y_train), then applies it to the test
    predictions to correct systematic level shifts caused by extrapolation.
    Only applied when enough OOF coverage is available (>= 50 rows).

    Dataset-agnostic: no column names are used.
    """
    if not SKLEARN_AVAILABLE:
        return predictions

    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception:
        return predictions

    try:
        train_median = float(np.median(pd.to_numeric(y, errors="coerce").dropna()))
        test_median = float(np.median(predictions[np.isfinite(predictions)]))
        if train_median <= 0 or test_median <= 0:
            return predictions
        ratio = test_median / train_median
        if ratio <= 1.5:
            return predictions  # no significant shift — skip calibration

        # Collect OOF predictions for the selected model
        sel_oof = oof_preds.get(selected_name)
        if sel_oof is None and stack is not None and selected_name == stack.get("name"):
            sel_oof = stack.get("oof")
        if sel_oof is None:
            return predictions

        mask = np.isfinite(sel_oof) & np.isfinite(np.asarray(y, dtype=float))
        if int(mask.sum()) < 50:
            return predictions  # not enough coverage for reliable calibration

        oof_fit = sel_oof[mask]
        y_fit = np.asarray(y, dtype=float)[mask]

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_fit, y_fit)

        calibrated = iso.predict(predictions)
        calibrated = np.asarray(calibrated, dtype=float)
        # Sanity: calibrated median should not deviate more than 50% from training
        cal_median = float(np.median(calibrated[np.isfinite(calibrated)]))
        if cal_median > 0 and abs(cal_median / train_median - 1.0) < 0.50:
            return calibrated
        return predictions
    except Exception:
        return predictions


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
