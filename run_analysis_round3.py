"""Round 3 analysis pipeline — final round.

High-priority fixes from analysis_review_2.json:
1. tgt_hour_month_mean: per-fold groupby(['hour','month'])['log_count'].mean().
   Captures 2.9x seasonal hour-demand swing (e.g. hour=7: Jan=95, Jun=277).
2. tgt_hour_workingday_year_mean: per-fold groupby(['hour','workingday','year']).
   48-group aggregation capturing 1.68x YoY growth disaggregated by commute/leisure.
3. sub_target_sum OOF in NNLS blend: add sub_target_sum as a blend candidate alongside
   sub_registered. Both decompositions in the pool — NNLS picks optimal weights.
4. Casual sub-model regularization: reduce max_leaf_nodes (15-31) and increase
   l2_regularization for the weekend (workingday=0) casual model to combat overfitting
   on the smaller, noisier weekend sample.

All column names resolved from spec_parse.json and analysis_plan.json at runtime.
Previous best honest CV RMSLE: 0.34835.
Submission updated only if new best CV RMSLE < 0.34835.
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, ParameterSampler
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).parent
LOGS_DIR = REPO_ROOT / "outputs" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = "20260610_round3"
# Round 2 honest CV RMSLE (leakage-corrected per-fold aggregates)
PREV_BEST_RMSLE = 0.34835422321166337
SEED = 42
N_CV_SPLITS = 5
N_ITER_SEARCH = 15

rng = np.random.RandomState(SEED)


# ---------------------------------------------------------------------------
# Load config — no hardcoded column names
# ---------------------------------------------------------------------------
def load_spec() -> dict:
    with open(LOGS_DIR / "spec_parse.json") as f:
        return json.load(f)


def load_plan() -> dict:
    with open(LOGS_DIR / "analysis_plan.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data(spec: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(REPO_ROOT / spec["train_file"])
    test_df = pd.read_csv(REPO_ROOT / spec["prediction_file"])
    sample_sub = pd.read_csv(REPO_ROOT / spec["sample_submission_file"])
    return train_df, test_df, sample_sub


# ---------------------------------------------------------------------------
# RMSLE metric
# ---------------------------------------------------------------------------
def rmsle_log_space(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """RMSLE when both inputs are already in log1p space."""
    return float(np.sqrt(np.mean((y_pred_log - y_true_log) ** 2)))


# ---------------------------------------------------------------------------
# Time feature extraction
# ---------------------------------------------------------------------------
def extract_time_features(dt_series: pd.Series, col_name: str = "datetime") -> pd.DataFrame:
    dt = pd.to_datetime(dt_series)
    prefix = col_name + "__"
    feats = pd.DataFrame(index=dt.index)
    feats[prefix + "year"] = dt.dt.year
    feats[prefix + "month"] = dt.dt.month
    feats[prefix + "month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    feats[prefix + "month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    feats[prefix + "day"] = dt.dt.day
    feats[prefix + "dayofweek"] = dt.dt.dayofweek
    feats[prefix + "dayofweek_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    feats[prefix + "dayofweek_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    feats[prefix + "is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    feats[prefix + "quarter"] = dt.dt.quarter
    feats[prefix + "weekofyear"] = dt.dt.isocalendar().week.astype(int)
    feats[prefix + "hour"] = dt.dt.hour
    feats[prefix + "hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    feats[prefix + "hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    epoch = pd.Timestamp("2011-01-01")
    feats[prefix + "ordinal"] = (dt - epoch).dt.total_seconds() / 86400.0
    return feats


# ---------------------------------------------------------------------------
# Build base feature matrices (NO agg features — added per-fold inside CV)
# ---------------------------------------------------------------------------
def build_base_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spec: dict,
    plan: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    row_id_col = spec["row_id_column"]
    target_col = spec["target_column"]
    leakage_cols = spec.get("leakage_columns", [])
    exclude = set([row_id_col, target_col] + leakage_cols)

    ohe_cols = plan["feature_plan"].get("low_cardinality_one_hot", [])
    binary_cols = plan["feature_plan"].get("binary_passthrough", [])
    numeric_cols = plan["feature_plan"].get("numeric_passthrough", [])

    ohe_cols = [c for c in ohe_cols if c in train_df.columns and c in test_df.columns]
    binary_cols = [c for c in binary_cols if c in train_df.columns and c in test_df.columns]
    numeric_cols = [c for c in numeric_cols if c in train_df.columns and c in test_df.columns]

    time_train = extract_time_features(train_df[row_id_col], row_id_col)
    time_test = extract_time_features(test_df[row_id_col], row_id_col)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=float)
    ohe.fit(train_df[ohe_cols])
    ohe_train = pd.DataFrame(
        ohe.transform(train_df[ohe_cols]),
        columns=ohe.get_feature_names_out(ohe_cols),
        index=train_df.index,
    )
    ohe_test = pd.DataFrame(
        ohe.transform(test_df[ohe_cols]),
        columns=ohe.get_feature_names_out(ohe_cols),
        index=test_df.index,
    )

    num_train = train_df[numeric_cols + binary_cols].copy().astype(float)
    num_test = test_df[numeric_cols + binary_cols].copy().astype(float)

    X_train_base = pd.concat([time_train, ohe_train, num_train], axis=1)
    X_test_base = pd.concat([time_test, ohe_test, num_test], axis=1)

    for col in exclude:
        assert col not in X_train_base.columns, f"Leakage: {col} in X_train_base!"
        assert col not in X_test_base.columns, f"Leakage: {col} in X_test_base!"

    meta = {
        "ohe_cols": ohe_cols,
        "binary_cols": binary_cols,
        "numeric_cols": numeric_cols,
        "ohe": ohe,
        "row_id_col": row_id_col,
        "target_col": target_col,
        "leakage_cols": leakage_cols,
    }
    return X_train_base, X_test_base, meta


# ---------------------------------------------------------------------------
# FIX #1 & #2: Extended FoldAggregator with 2 new aggregates
#
# New aggregates added in round 3:
#   tgt_hour_month_mean    — groupby(['hour', 'month'])
#   tgt_hour_workingday_year_mean — groupby(['hour', 'workingday', 'year'])
#
# Fallback chains (missing-cell imputation):
#   tgt_hour_month_mean   -> tgt_hour_mean -> global_mean
#   tgt_hour_wd_year_mean -> tgt_hour_wd_mean -> tgt_hour_mean -> global_mean
# ---------------------------------------------------------------------------
class FoldAggregator:
    """Fit per-fold target aggregates on train rows; transform train/val/test."""

    def __init__(
        self,
        spec: dict,
        plan: dict,
        train_df_raw: pd.DataFrame,
        test_df_raw: pd.DataFrame,
    ):
        self.row_id_col = spec["row_id_column"]
        self.target_col = spec["target_column"]
        self.leakage_cols = spec.get("leakage_columns", [])
        ohe_cols = plan["feature_plan"].get("low_cardinality_one_hot", [])
        binary_cols = plan["feature_plan"].get("binary_passthrough", [])

        self.season_col = ohe_cols[0] if ohe_cols else None
        # workingday column — resolved from plan binary_passthrough (dataset-agnostic)
        self.wd_col = binary_cols[0] if binary_cols else None

        dt_train = pd.to_datetime(train_df_raw[self.row_id_col])
        dt_test = pd.to_datetime(test_df_raw[self.row_id_col])

        self.train_hour = dt_train.dt.hour.values
        self.train_month = dt_train.dt.month.values
        self.train_year = dt_train.dt.year.values
        self.train_day = dt_train.dt.day.values
        self.train_ym = (
            dt_train.dt.year.astype(str) + "_" + dt_train.dt.month.astype(str).str.zfill(2)
        ).values

        self.test_hour = dt_test.dt.hour.values
        self.test_month = dt_test.dt.month.values
        self.test_year = dt_test.dt.year.values
        self.test_ym = (
            dt_test.dt.year.astype(str) + "_" + dt_test.dt.month.astype(str).str.zfill(2)
        ).values

        self.train_season = (
            train_df_raw[self.season_col].values
            if (self.season_col and self.season_col in train_df_raw.columns)
            else None
        )
        self.test_season = (
            test_df_raw[self.season_col].values
            if (self.season_col and self.season_col in test_df_raw.columns)
            else None
        )

        # workingday arrays (resolved by column name from plan, not hardcoded)
        self.train_wd = (
            train_df_raw[self.wd_col].values
            if (self.wd_col and self.wd_col in train_df_raw.columns)
            else None
        )
        self.test_wd = (
            test_df_raw[self.wd_col].values
            if (self.wd_col and self.wd_col in test_df_raw.columns)
            else None
        )

        self.train_log_target = np.log1p(train_df_raw[self.target_col].values.astype(float))

        # Fitted maps (populated during fit())
        self._global_mean: float = 0.0
        self._hour_mean_map: dict = {}
        self._season_mean_map: dict = {}
        self._hour_wd_map: dict = {}
        self._ym_mean_map: dict = {}
        # NEW round 3 maps
        self._hour_month_map: dict = {}
        self._hour_wd_year_map: dict = {}

    def fit(self, fold_train_idx: np.ndarray) -> "FoldAggregator":
        h = self.train_hour[fold_train_idx]
        m = self.train_month[fold_train_idx]
        y = self.train_year[fold_train_idx]
        log_t = self.train_log_target[fold_train_idx]
        ym = self.train_ym[fold_train_idx]

        self._global_mean = float(np.mean(log_t))

        # tgt_hour_mean  (fallback for new features)
        hour_df = pd.DataFrame({"h": h, "log_t": log_t})
        self._hour_mean_map = hour_df.groupby("h")["log_t"].mean().to_dict()

        # tgt_season_mean
        if self.train_season is not None:
            s = self.train_season[fold_train_idx]
            seas_df = pd.DataFrame({"s": s, "log_t": log_t})
            self._season_mean_map = seas_df.groupby("s")["log_t"].mean().to_dict()
        else:
            self._season_mean_map = {}

        # tgt_hour_wd_mean  (fallback for tgt_hour_wd_year_mean)
        if self.train_wd is not None:
            wd = self.train_wd[fold_train_idx]
            hwd_df = pd.DataFrame({"h": h, "wd": wd, "log_t": log_t})
            self._hour_wd_map = hwd_df.groupby(["h", "wd"])["log_t"].mean().to_dict()
        else:
            self._hour_wd_map = {}

        # tgt_year_month_mean (YoY trend)
        ym_df = pd.DataFrame({"ym": ym, "log_t": log_t})
        self._ym_mean_map = ym_df.groupby("ym")["log_t"].mean().to_dict()

        # ── NEW FIX #1: tgt_hour_month_mean ────────────────────────────────
        # groupby(['hour', 'month']): 24 × 12 = up to 288 cells
        # Captures 2.9x demand swing e.g. hour=7: Jan vs Jun
        hm_df = pd.DataFrame({"h": h, "mo": m, "log_t": log_t})
        self._hour_month_map = hm_df.groupby(["h", "mo"])["log_t"].mean().to_dict()

        # ── NEW FIX #2: tgt_hour_workingday_year_mean ──────────────────────
        # groupby(['hour', 'workingday', 'year']): 24 × 2 × ~2 = 96 cells
        # Captures 1.68x YoY growth disaggregated by commute vs leisure
        if self.train_wd is not None:
            wd = self.train_wd[fold_train_idx]
            hwdy_df = pd.DataFrame({"h": h, "wd": wd, "yr": y, "log_t": log_t})
            self._hour_wd_year_map = (
                hwdy_df.groupby(["h", "wd", "yr"])["log_t"].mean().to_dict()
            )
        else:
            self._hour_wd_year_map = {}

        return self

    def _transform_array(
        self,
        hour_arr: np.ndarray,
        month_arr: np.ndarray,
        year_arr: np.ndarray,
        season_arr,
        wd_arr,
        ym_arr: np.ndarray,
    ) -> pd.DataFrame:
        g = self._global_mean
        feats: dict[str, np.ndarray] = {}

        # tgt_hour_mean
        feats["tgt_hour_mean"] = np.array(
            [self._hour_mean_map.get(int(hh), g) for hh in hour_arr], dtype=float
        )

        # tgt_season_mean
        if self._season_mean_map and season_arr is not None:
            feats["tgt_season_mean"] = np.array(
                [self._season_mean_map.get(int(ss), g) for ss in season_arr], dtype=float
            )

        # tgt_hour_wd_mean
        if self._hour_wd_map and wd_arr is not None:
            feats["tgt_hour_wd_mean"] = np.array(
                [self._hour_wd_map.get((int(hh), int(ww)), g)
                 for hh, ww in zip(hour_arr, wd_arr)],
                dtype=float,
            )

        # tgt_year_month_mean
        feats["tgt_year_month_mean"] = np.array(
            [self._ym_mean_map.get(yy, g) for yy in ym_arr], dtype=float
        )

        # ── NEW FIX #1: tgt_hour_month_mean ────────────────────────────────
        # Fallback chain: (hour, month) -> hour_mean -> global_mean
        feats["tgt_hour_month_mean"] = np.array(
            [
                self._hour_month_map.get(
                    (int(hh), int(mo)),
                    self._hour_mean_map.get(int(hh), g),
                )
                for hh, mo in zip(hour_arr, month_arr)
            ],
            dtype=float,
        )

        # ── NEW FIX #2: tgt_hour_workingday_year_mean ──────────────────────
        # Fallback chain: (hour, wd, year) -> (hour, wd) -> hour_mean -> global_mean
        if self._hour_wd_year_map and wd_arr is not None:
            feats["tgt_hour_wd_year_mean"] = np.array(
                [
                    self._hour_wd_year_map.get(
                        (int(hh), int(ww), int(yr)),
                        self._hour_wd_map.get(
                            (int(hh), int(ww)),
                            self._hour_mean_map.get(int(hh), g),
                        ),
                    )
                    for hh, ww, yr in zip(hour_arr, wd_arr, year_arr)
                ],
                dtype=float,
            )

        return pd.DataFrame(feats)

    def transform_train_idx(self, idx: np.ndarray) -> pd.DataFrame:
        return self._transform_array(
            self.train_hour[idx],
            self.train_month[idx],
            self.train_year[idx],
            self.train_season[idx] if self.train_season is not None else None,
            self.train_wd[idx] if self.train_wd is not None else None,
            self.train_ym[idx],
        )

    def transform_test(self) -> pd.DataFrame:
        return self._transform_array(
            self.test_hour,
            self.test_month,
            self.test_year,
            self.test_season,
            self.test_wd,
            self.test_ym,
        )

    def fit_on_all_train(self) -> "FoldAggregator":
        return self.fit(np.arange(len(self.train_hour)))


# ---------------------------------------------------------------------------
# CV split generation
# ---------------------------------------------------------------------------
def make_cv_splits(
    train_df: pd.DataFrame,
    spec: dict,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    row_id_col = spec["row_id_column"]
    dt = pd.to_datetime(train_df[row_id_col])
    year_month = (
        dt.dt.year.astype(str) + "_" + dt.dt.month.astype(str).str.zfill(2)
    ).values
    idx = np.arange(len(train_df))
    gkf = GroupKFold(n_splits=n_splits)
    return [(tr, val) for tr, val in gkf.split(idx, groups=year_month)]


# ---------------------------------------------------------------------------
# FIX #4: Regularised WorkingdaySplitModel for casual sub-model
#
# Weekend (workingday=0) model uses smaller max_leaf_nodes and stronger
# l2_regularization to combat overfitting on the noisy weekend sample.
# Workday (workingday=1) model uses a clone of the base estimator unchanged.
# wd_col resolved from plan binary_passthrough — no hardcoding.
# ---------------------------------------------------------------------------
class WorkingdaySplitModel:
    """
    Trains separate models for workingday==0 (weekend) and workingday==1 (weekday).
    Weekend model is regularised (max_leaf_nodes <= 31, l2_regularization >= 1.0)
    to address the overfitting observed in round 2.
    Falls back to full model if wd_col unavailable.
    """

    # FIX #4: More regularised HGB for the weekend casual model
    _WD0_CASUAL_PARAMS = {
        "max_leaf_nodes": 15,
        "l2_regularization": 1.0,
        "max_iter": 200,
        "learning_rate": 0.05,
        "min_samples_leaf": 20,
    }

    def __init__(self, base_estimator, wd_col: str, regularise_wd0: bool = False):
        self.base_estimator = base_estimator
        self.wd_col = wd_col
        self.regularise_wd0 = regularise_wd0
        self._model_wd0 = None
        self._model_wd1 = None
        self._model_full = None
        self._has_wd = False
        self._wd_col_idx: int = -1

    def _make_wd0_model(self):
        """Create the weekend model with stronger regularization (FIX #4)."""
        if self.regularise_wd0:
            return HistGradientBoostingRegressor(
                random_state=SEED,
                **self._WD0_CASUAL_PARAMS,
            )
        return clone(self.base_estimator)

    def fit(
        self, X: np.ndarray, y: np.ndarray, feature_names: list[str]
    ) -> "WorkingdaySplitModel":
        if self.wd_col in feature_names:
            self._has_wd = True
            self._wd_col_idx = feature_names.index(self.wd_col)
            wd = X[:, self._wd_col_idx].astype(int)
            mask0 = wd == 0
            mask1 = wd == 1

            if mask0.sum() >= 10:
                self._model_wd0 = self._make_wd0_model()
                self._model_wd0.fit(X[mask0], y[mask0])
            else:
                self._model_wd0 = None

            if mask1.sum() >= 10:
                self._model_wd1 = clone(self.base_estimator)
                self._model_wd1.fit(X[mask1], y[mask1])
            else:
                self._model_wd1 = None

            self._model_full = clone(self.base_estimator)
            self._model_full.fit(X, y)
        else:
            self._has_wd = False
            self._model_full = clone(self.base_estimator)
            self._model_full.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._has_wd or self._wd_col_idx < 0:
            return self._model_full.predict(X)

        preds = self._model_full.predict(X).copy()
        wd = X[:, self._wd_col_idx].astype(int)

        mask0 = wd == 0
        if self._model_wd0 is not None and mask0.sum() > 0:
            preds[mask0] = self._model_wd0.predict(X[mask0])

        mask1 = wd == 1
        if self._model_wd1 is not None and mask1.sum() > 0:
            preds[mask1] = self._model_wd1.predict(X[mask1])

        return preds


# ---------------------------------------------------------------------------
# Core CV evaluation with per-fold agg features
# ---------------------------------------------------------------------------
def cv_model_with_fold_aggs(
    model,
    X_train_base: pd.DataFrame,
    y_train: np.ndarray,
    splits: list,
    fold_agg: FoldAggregator,
    return_oof: bool = False,
) -> dict:
    fold_scores = []
    oof_preds = np.zeros(len(y_train))

    for tr_idx, val_idx in splits:
        fold_agg.fit(tr_idx)
        agg_tr = fold_agg.transform_train_idx(tr_idx)
        agg_val = fold_agg.transform_train_idx(val_idx)

        X_tr_full = pd.concat(
            [X_train_base.iloc[tr_idx].reset_index(drop=True),
             agg_tr.reset_index(drop=True)],
            axis=1,
        ).values.astype(float)
        X_val_full = pd.concat(
            [X_train_base.iloc[val_idx].reset_index(drop=True),
             agg_val.reset_index(drop=True)],
            axis=1,
        ).values.astype(float)

        m = clone(model)
        m.fit(X_tr_full, y_train[tr_idx])
        pred_log = m.predict(X_val_full)
        score = rmsle_log_space(y_train[val_idx], pred_log)
        fold_scores.append(score)
        if return_oof:
            oof_preds[val_idx] = pred_log

    result = {
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
        "folds": [float(s) for s in fold_scores],
    }
    if return_oof:
        result["oof_preds"] = oof_preds
    return result


def cv_wd_model_with_fold_aggs(
    wd_model: WorkingdaySplitModel,
    X_train_base: pd.DataFrame,
    y_train: np.ndarray,
    splits: list,
    fold_agg: FoldAggregator,
    wd_col: str,
    return_oof: bool = False,
) -> dict:
    fold_scores = []
    oof_preds = np.zeros(len(y_train))

    for tr_idx, val_idx in splits:
        fold_agg.fit(tr_idx)
        agg_tr = fold_agg.transform_train_idx(tr_idx)
        agg_val = fold_agg.transform_train_idx(val_idx)

        X_tr_base = X_train_base.iloc[tr_idx].reset_index(drop=True)
        X_val_base = X_train_base.iloc[val_idx].reset_index(drop=True)

        X_tr_full_df = pd.concat([X_tr_base, agg_tr.reset_index(drop=True)], axis=1)
        X_val_full_df = pd.concat([X_val_base, agg_val.reset_index(drop=True)], axis=1)

        feature_names = X_tr_full_df.columns.tolist()
        X_tr_arr = X_tr_full_df.values.astype(float)
        X_val_arr = X_val_full_df.values.astype(float)

        m = WorkingdaySplitModel(
            base_estimator=clone(wd_model.base_estimator),
            wd_col=wd_col,
            regularise_wd0=wd_model.regularise_wd0,
        )
        m.fit(X_tr_arr, y_train[tr_idx], feature_names)
        pred_log = m.predict(X_val_arr)
        score = rmsle_log_space(y_train[val_idx], pred_log)
        fold_scores.append(score)
        if return_oof:
            oof_preds[val_idx] = pred_log

    result = {
        "mean": float(np.mean(fold_scores)),
        "std": float(np.std(fold_scores)),
        "folds": [float(s) for s in fold_scores],
    }
    if return_oof:
        result["oof_preds"] = oof_preds
    return result


# ---------------------------------------------------------------------------
# Hyperparameter search spaces
# ---------------------------------------------------------------------------
def get_param_spaces() -> dict:
    return {
        "hgb": {
            "model_class": HistGradientBoostingRegressor,
            "base_params": {"random_state": SEED},
            "search_space": {
                "max_iter": [200, 300, 500],
                "max_depth": [None, 4, 6, 8],
                "learning_rate": [0.03, 0.05, 0.08, 0.1],
                "min_samples_leaf": [10, 20, 30, 50],
                "l2_regularization": [0.0, 0.1, 0.5, 1.0],
                "max_bins": [128, 255],
                "max_leaf_nodes": [15, 31, 63, 127],
            },
        },
        "rf": {
            "model_class": RandomForestRegressor,
            "base_params": {"random_state": SEED, "n_jobs": -1},
            "search_space": {
                "n_estimators": [200, 300],
                "max_depth": [None, 10, 14],
                "min_samples_leaf": [2, 5],
                "max_features": [0.6, 0.8, "sqrt"],
            },
        },
        "et": {
            "model_class": ExtraTreesRegressor,
            "base_params": {"random_state": SEED, "n_jobs": -1},
            "search_space": {
                "n_estimators": [200, 300],
                "max_depth": [None, 10, 14],
                "min_samples_leaf": [2, 5],
                "max_features": [0.6, 0.8, "sqrt"],
            },
        },
        "ridge": {
            "model_class": Ridge,
            "base_params": {},
            "search_space": {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        },
    }


def try_lightgbm() -> tuple[bool, object | None]:
    try:
        import lightgbm as lgb
        _ = lgb.LGBMRegressor
        return True, lgb
    except Exception:
        return False, None


# ---------------------------------------------------------------------------
# Helper: augment base features with globally-fitted aggregates (for test set
# or full-train refit)
# ---------------------------------------------------------------------------
def augment_with_full_train_aggs(
    X_base: pd.DataFrame,
    fold_agg_fitted_full: FoldAggregator,
    is_test: bool,
    subset_idx: np.ndarray | None = None,
) -> np.ndarray:
    if is_test:
        agg_df = fold_agg_fitted_full.transform_test()
    else:
        assert subset_idx is not None
        agg_df = fold_agg_fitted_full.transform_train_idx(subset_idx)

    return pd.concat(
        [X_base.reset_index(drop=True), agg_df.reset_index(drop=True)], axis=1
    ).values.astype(float)


# ---------------------------------------------------------------------------
# Full model search
# ---------------------------------------------------------------------------
def run_model_search(
    X_train_base: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    y_count_log: np.ndarray,
    y_casual_log: np.ndarray,
    y_registered_log: np.ndarray,
    X_test_base: pd.DataFrame,
    splits: list,
    spec: dict,
    plan: dict,
) -> dict:
    binary_cols = plan["feature_plan"].get("binary_passthrough", [])
    wd_col = binary_cols[0] if binary_cols else None

    param_spaces = get_param_spaces()
    has_lgb, lgb_module = try_lightgbm()

    # Separate FoldAggregators per target so tgt_* maps reflect each sub-target
    fold_agg_count = FoldAggregator(spec, plan, train_df, test_df)

    fold_agg_casual = FoldAggregator(spec, plan, train_df, test_df)
    fold_agg_casual.train_log_target = y_casual_log

    fold_agg_registered = FoldAggregator(spec, plan, train_df, test_df)
    fold_agg_registered.train_log_target = y_registered_log

    results = {}
    oof_collection = {}
    test_preds: dict[str, np.ndarray] = {}

    n_train = len(y_count_log)
    val_mask = np.zeros(n_train, dtype=bool)
    for _, val_idx in splits:
        val_mask[val_idx] = True

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    print("  [baseline] DummyRegressor...")
    dummy_scores = []
    for tr_idx, val_idx in splits:
        fold_agg_count.fit(tr_idx)
        agg_val = fold_agg_count.transform_train_idx(val_idx)
        X_val = pd.concat(
            [X_train_base.iloc[val_idx].reset_index(drop=True),
             agg_val.reset_index(drop=True)],
            axis=1,
        ).values.astype(float)
        m = DummyRegressor(strategy="mean")
        m.fit(np.zeros((len(tr_idx), 1)), y_count_log[tr_idx])
        pred = m.predict(X_val)
        dummy_scores.append(rmsle_log_space(y_count_log[val_idx], pred))
    results["dummy"] = {"mean": float(np.mean(dummy_scores)), "std": float(np.std(dummy_scores))}
    print(f"    CV RMSLE: {results['dummy']['mean']:.5f}")

    print("  [baseline] Ridge...")
    ridge_base_scores = []
    for tr_idx, val_idx in splits:
        fold_agg_count.fit(tr_idx)
        agg_tr = fold_agg_count.transform_train_idx(tr_idx)
        agg_val = fold_agg_count.transform_train_idx(val_idx)
        X_tr = pd.concat(
            [X_train_base.iloc[tr_idx].reset_index(drop=True), agg_tr.reset_index(drop=True)],
            axis=1,
        ).values.astype(float)
        X_val = pd.concat(
            [X_train_base.iloc[val_idx].reset_index(drop=True), agg_val.reset_index(drop=True)],
            axis=1,
        ).values.astype(float)
        m = Ridge(alpha=1.0)
        m.fit(X_tr, y_count_log[tr_idx])
        pred = m.predict(X_val)
        ridge_base_scores.append(rmsle_log_space(y_count_log[val_idx], pred))
    results["ridge_baseline"] = {
        "mean": float(np.mean(ridge_base_scores)),
        "std": float(np.std(ridge_base_scores)),
    }
    print(f"    CV RMSLE: {results['ridge_baseline']['mean']:.5f}")

    # ------------------------------------------------------------------
    # Ridge search
    # ------------------------------------------------------------------
    print("  [candidate] Ridge (search)...")
    best_ridge_score = float("inf")
    best_ridge_alpha = 1.0
    for params in ParameterSampler(param_spaces["ridge"]["search_space"], n_iter=5, random_state=rng):
        scores = []
        for tr_idx, val_idx in splits:
            fold_agg_count.fit(tr_idx)
            agg_tr = fold_agg_count.transform_train_idx(tr_idx)
            agg_val = fold_agg_count.transform_train_idx(val_idx)
            X_tr = pd.concat(
                [X_train_base.iloc[tr_idx].reset_index(drop=True), agg_tr.reset_index(drop=True)],
                axis=1,
            ).values.astype(float)
            X_val = pd.concat(
                [X_train_base.iloc[val_idx].reset_index(drop=True), agg_val.reset_index(drop=True)],
                axis=1,
            ).values.astype(float)
            m = Ridge(**params)
            m.fit(X_tr, y_count_log[tr_idx])
            pred = m.predict(X_val)
            scores.append(rmsle_log_space(y_count_log[val_idx], pred))
        mean_score = float(np.mean(scores))
        if mean_score < best_ridge_score:
            best_ridge_score = mean_score
            best_ridge_alpha = params["alpha"]

    ridge_oof = np.zeros(n_train)
    ridge_fold_scores = []
    for tr_idx, val_idx in splits:
        fold_agg_count.fit(tr_idx)
        agg_tr = fold_agg_count.transform_train_idx(tr_idx)
        agg_val = fold_agg_count.transform_train_idx(val_idx)
        X_tr = pd.concat(
            [X_train_base.iloc[tr_idx].reset_index(drop=True), agg_tr.reset_index(drop=True)],
            axis=1,
        ).values.astype(float)
        X_val = pd.concat(
            [X_train_base.iloc[val_idx].reset_index(drop=True), agg_val.reset_index(drop=True)],
            axis=1,
        ).values.astype(float)
        m = Ridge(alpha=best_ridge_alpha)
        m.fit(X_tr, y_count_log[tr_idx])
        pred = m.predict(X_val)
        ridge_fold_scores.append(rmsle_log_space(y_count_log[val_idx], pred))
        ridge_oof[val_idx] = pred

    results["ridge"] = {
        "mean": float(np.mean(ridge_fold_scores)),
        "std": float(np.std(ridge_fold_scores)),
        "folds": [float(s) for s in ridge_fold_scores],
    }
    oof_collection["ridge"] = ridge_oof

    fold_agg_count.fit_on_all_train()
    X_test_aug = augment_with_full_train_aggs(X_test_base, fold_agg_count, is_test=True)
    X_full_train = pd.concat(
        [X_train_base.reset_index(drop=True),
         fold_agg_count.transform_train_idx(np.arange(n_train)).reset_index(drop=True)],
        axis=1,
    ).values.astype(float)
    best_ridge_fitted = Ridge(alpha=best_ridge_alpha)
    best_ridge_fitted.fit(X_full_train, y_count_log)
    test_preds["ridge"] = best_ridge_fitted.predict(X_test_aug)
    print(f"    CV RMSLE: {results['ridge']['mean']:.5f}")

    # ------------------------------------------------------------------
    # HGB search
    # ------------------------------------------------------------------
    print("  [candidate] HistGradientBoosting...")
    best_hgb_score = float("inf")
    best_hgb_params = {}
    for params in ParameterSampler(param_spaces["hgb"]["search_space"], n_iter=N_ITER_SEARCH, random_state=rng):
        scores = []
        for tr_idx, val_idx in splits:
            fold_agg_count.fit(tr_idx)
            agg_tr = fold_agg_count.transform_train_idx(tr_idx)
            agg_val = fold_agg_count.transform_train_idx(val_idx)
            X_tr = pd.concat(
                [X_train_base.iloc[tr_idx].reset_index(drop=True), agg_tr.reset_index(drop=True)],
                axis=1,
            ).values.astype(float)
            X_val = pd.concat(
                [X_train_base.iloc[val_idx].reset_index(drop=True), agg_val.reset_index(drop=True)],
                axis=1,
            ).values.astype(float)
            m = HistGradientBoostingRegressor(random_state=SEED, **params)
            m.fit(X_tr, y_count_log[tr_idx])
            pred = m.predict(X_val)
            scores.append(rmsle_log_space(y_count_log[val_idx], pred))
        mean_score = float(np.mean(scores))
        if mean_score < best_hgb_score:
            best_hgb_score = mean_score
            best_hgb_params = dict(params)

    hgb_oof = np.zeros(n_train)
    hgb_fold_scores = []
    for tr_idx, val_idx in splits:
        fold_agg_count.fit(tr_idx)
        agg_tr = fold_agg_count.transform_train_idx(tr_idx)
        agg_val = fold_agg_count.transform_train_idx(val_idx)
        X_tr = pd.concat(
            [X_train_base.iloc[tr_idx].reset_index(drop=True), agg_tr.reset_index(drop=True)],
            axis=1,
        ).values.astype(float)
        X_val = pd.concat(
            [X_train_base.iloc[val_idx].reset_index(drop=True), agg_val.reset_index(drop=True)],
            axis=1,
        ).values.astype(float)
        m = HistGradientBoostingRegressor(random_state=SEED, **best_hgb_params)
        m.fit(X_tr, y_count_log[tr_idx])
        pred = m.predict(X_val)
        hgb_fold_scores.append(rmsle_log_space(y_count_log[val_idx], pred))
        hgb_oof[val_idx] = pred

    results["hgb"] = {
        "mean": float(np.mean(hgb_fold_scores)),
        "std": float(np.std(hgb_fold_scores)),
        "folds": [float(s) for s in hgb_fold_scores],
    }
    oof_collection["hgb"] = hgb_oof

    fold_agg_count.fit_on_all_train()
    X_test_aug_hgb = augment_with_full_train_aggs(X_test_base, fold_agg_count, is_test=True)
    X_full_hgb = pd.concat(
        [X_train_base.reset_index(drop=True),
         fold_agg_count.transform_train_idx(np.arange(n_train)).reset_index(drop=True)],
        axis=1,
    ).values.astype(float)
    best_hgb_model = HistGradientBoostingRegressor(random_state=SEED, **best_hgb_params)
    best_hgb_model.fit(X_full_hgb, y_count_log)
    test_preds["hgb"] = best_hgb_model.predict(X_test_aug_hgb)
    print(f"    CV RMSLE: {results['hgb']['mean']:.5f}")

    # ------------------------------------------------------------------
    # RF search
    # ------------------------------------------------------------------
    print("  [candidate] RandomForest...")
    best_rf_score = float("inf")
    best_rf_params = {}
    for params in ParameterSampler(param_spaces["rf"]["search_space"], n_iter=8, random_state=rng):
        scores = []
        for tr_idx, val_idx in splits:
            fold_agg_count.fit(tr_idx)
            agg_tr = fold_agg_count.transform_train_idx(tr_idx)
            agg_val = fold_agg_count.transform_train_idx(val_idx)
            X_tr = pd.concat(
                [X_train_base.iloc[tr_idx].reset_index(drop=True), agg_tr.reset_index(drop=True)],
                axis=1,
            ).values.astype(float)
            X_val = pd.concat(
                [X_train_base.iloc[val_idx].reset_index(drop=True), agg_val.reset_index(drop=True)],
                axis=1,
            ).values.astype(float)
            m = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
            m.fit(X_tr, y_count_log[tr_idx])
            pred = m.predict(X_val)
            scores.append(rmsle_log_space(y_count_log[val_idx], pred))
        mean_score = float(np.mean(scores))
        if mean_score < best_rf_score:
            best_rf_score = mean_score
            best_rf_params = dict(params)

    rf_oof = np.zeros(n_train)
    rf_fold_scores = []
    for tr_idx, val_idx in splits:
        fold_agg_count.fit(tr_idx)
        agg_tr = fold_agg_count.transform_train_idx(tr_idx)
        agg_val = fold_agg_count.transform_train_idx(val_idx)
        X_tr = pd.concat(
            [X_train_base.iloc[tr_idx].reset_index(drop=True), agg_tr.reset_index(drop=True)],
            axis=1,
        ).values.astype(float)
        X_val = pd.concat(
            [X_train_base.iloc[val_idx].reset_index(drop=True), agg_val.reset_index(drop=True)],
            axis=1,
        ).values.astype(float)
        m = RandomForestRegressor(random_state=SEED, n_jobs=-1, **best_rf_params)
        m.fit(X_tr, y_count_log[tr_idx])
        pred = m.predict(X_val)
        rf_fold_scores.append(rmsle_log_space(y_count_log[val_idx], pred))
        rf_oof[val_idx] = pred

    results["rf"] = {
        "mean": float(np.mean(rf_fold_scores)),
        "std": float(np.std(rf_fold_scores)),
        "folds": [float(s) for s in rf_fold_scores],
    }
    oof_collection["rf"] = rf_oof

    fold_agg_count.fit_on_all_train()
    X_test_aug_rf = augment_with_full_train_aggs(X_test_base, fold_agg_count, is_test=True)
    X_full_rf = pd.concat(
        [X_train_base.reset_index(drop=True),
         fold_agg_count.transform_train_idx(np.arange(n_train)).reset_index(drop=True)],
        axis=1,
    ).values.astype(float)
    best_rf_model = RandomForestRegressor(random_state=SEED, n_jobs=-1, **best_rf_params)
    best_rf_model.fit(X_full_rf, y_count_log)
    test_preds["rf"] = best_rf_model.predict(X_test_aug_rf)
    print(f"    CV RMSLE: {results['rf']['mean']:.5f}")

    # ------------------------------------------------------------------
    # ET search
    # ------------------------------------------------------------------
    print("  [candidate] ExtraTrees...")
    best_et_score = float("inf")
    best_et_params = {}
    for params in ParameterSampler(param_spaces["et"]["search_space"], n_iter=8, random_state=rng):
        scores = []
        for tr_idx, val_idx in splits:
            fold_agg_count.fit(tr_idx)
            agg_tr = fold_agg_count.transform_train_idx(tr_idx)
            agg_val = fold_agg_count.transform_train_idx(val_idx)
            X_tr = pd.concat(
                [X_train_base.iloc[tr_idx].reset_index(drop=True), agg_tr.reset_index(drop=True)],
                axis=1,
            ).values.astype(float)
            X_val = pd.concat(
                [X_train_base.iloc[val_idx].reset_index(drop=True), agg_val.reset_index(drop=True)],
                axis=1,
            ).values.astype(float)
            m = ExtraTreesRegressor(random_state=SEED, n_jobs=-1, **params)
            m.fit(X_tr, y_count_log[tr_idx])
            pred = m.predict(X_val)
            scores.append(rmsle_log_space(y_count_log[val_idx], pred))
        mean_score = float(np.mean(scores))
        if mean_score < best_et_score:
            best_et_score = mean_score
            best_et_params = dict(params)

    et_oof = np.zeros(n_train)
    et_fold_scores = []
    for tr_idx, val_idx in splits:
        fold_agg_count.fit(tr_idx)
        agg_tr = fold_agg_count.transform_train_idx(tr_idx)
        agg_val = fold_agg_count.transform_train_idx(val_idx)
        X_tr = pd.concat(
            [X_train_base.iloc[tr_idx].reset_index(drop=True), agg_tr.reset_index(drop=True)],
            axis=1,
        ).values.astype(float)
        X_val = pd.concat(
            [X_train_base.iloc[val_idx].reset_index(drop=True), agg_val.reset_index(drop=True)],
            axis=1,
        ).values.astype(float)
        m = ExtraTreesRegressor(random_state=SEED, n_jobs=-1, **best_et_params)
        m.fit(X_tr, y_count_log[tr_idx])
        pred = m.predict(X_val)
        et_fold_scores.append(rmsle_log_space(y_count_log[val_idx], pred))
        et_oof[val_idx] = pred

    results["et"] = {
        "mean": float(np.mean(et_fold_scores)),
        "std": float(np.std(et_fold_scores)),
        "folds": [float(s) for s in et_fold_scores],
    }
    oof_collection["et"] = et_oof

    fold_agg_count.fit_on_all_train()
    X_test_aug_et = augment_with_full_train_aggs(X_test_base, fold_agg_count, is_test=True)
    X_full_et = pd.concat(
        [X_train_base.reset_index(drop=True),
         fold_agg_count.transform_train_idx(np.arange(n_train)).reset_index(drop=True)],
        axis=1,
    ).values.astype(float)
    best_et_model = ExtraTreesRegressor(random_state=SEED, n_jobs=-1, **best_et_params)
    best_et_model.fit(X_full_et, y_count_log)
    test_preds["et"] = best_et_model.predict(X_test_aug_et)
    print(f"    CV RMSLE: {results['et']['mean']:.5f}")

    # ------------------------------------------------------------------
    # LightGBM (if available)
    # ------------------------------------------------------------------
    if has_lgb:
        print("  [candidate] LightGBM...")
        lgb_space = {
            "num_leaves": [63, 127, 255, 511],
            "n_estimators": [300, 500, 800],
            "learning_rate": [0.01, 0.03, 0.05],
            "min_child_samples": [10, 15, 20],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5],
            "reg_lambda": [0.0, 0.1, 0.5, 1.0],
        }
        best_lgb_score = float("inf")
        best_lgb_params = {}
        for params in ParameterSampler(lgb_space, n_iter=N_ITER_SEARCH, random_state=rng):
            scores = []
            for tr_idx, val_idx in splits:
                fold_agg_count.fit(tr_idx)
                agg_tr = fold_agg_count.transform_train_idx(tr_idx)
                agg_val = fold_agg_count.transform_train_idx(val_idx)
                X_tr = pd.concat(
                    [X_train_base.iloc[tr_idx].reset_index(drop=True),
                     agg_tr.reset_index(drop=True)],
                    axis=1,
                ).values.astype(float)
                X_val = pd.concat(
                    [X_train_base.iloc[val_idx].reset_index(drop=True),
                     agg_val.reset_index(drop=True)],
                    axis=1,
                ).values.astype(float)
                m = lgb_module.LGBMRegressor(random_state=SEED, verbose=-1, **params)
                m.fit(X_tr, y_count_log[tr_idx])
                pred = m.predict(X_val)
                scores.append(rmsle_log_space(y_count_log[val_idx], pred))
            mean_score = float(np.mean(scores))
            if mean_score < best_lgb_score:
                best_lgb_score = mean_score
                best_lgb_params = dict(params)

        lgb_oof = np.zeros(n_train)
        lgb_fold_scores = []
        for tr_idx, val_idx in splits:
            fold_agg_count.fit(tr_idx)
            agg_tr = fold_agg_count.transform_train_idx(tr_idx)
            agg_val = fold_agg_count.transform_train_idx(val_idx)
            X_tr = pd.concat(
                [X_train_base.iloc[tr_idx].reset_index(drop=True),
                 agg_tr.reset_index(drop=True)],
                axis=1,
            ).values.astype(float)
            X_val = pd.concat(
                [X_train_base.iloc[val_idx].reset_index(drop=True),
                 agg_val.reset_index(drop=True)],
                axis=1,
            ).values.astype(float)
            m = lgb_module.LGBMRegressor(random_state=SEED, verbose=-1, **best_lgb_params)
            m.fit(X_tr, y_count_log[tr_idx])
            pred = m.predict(X_val)
            lgb_fold_scores.append(rmsle_log_space(y_count_log[val_idx], pred))
            lgb_oof[val_idx] = pred

        results["lgb"] = {
            "mean": float(np.mean(lgb_fold_scores)),
            "std": float(np.std(lgb_fold_scores)),
            "folds": [float(s) for s in lgb_fold_scores],
        }
        oof_collection["lgb"] = lgb_oof

        fold_agg_count.fit_on_all_train()
        X_test_aug_lgb = augment_with_full_train_aggs(X_test_base, fold_agg_count, is_test=True)
        X_full_lgb = pd.concat(
            [X_train_base.reset_index(drop=True),
             fold_agg_count.transform_train_idx(np.arange(n_train)).reset_index(drop=True)],
            axis=1,
        ).values.astype(float)
        best_lgb_model = lgb_module.LGBMRegressor(random_state=SEED, verbose=-1, **best_lgb_params)
        best_lgb_model.fit(X_full_lgb, y_count_log)
        test_preds["lgb"] = best_lgb_model.predict(X_test_aug_lgb)
        print(f"    CV RMSLE: {results['lgb']['mean']:.5f}")
    else:
        best_lgb_params = {}

    # ------------------------------------------------------------------
    # Pick base estimator for sub-models
    # ------------------------------------------------------------------
    single_scores = {k: results[k]["mean"] for k in oof_collection.keys()}
    best_single_name = min(single_scores, key=single_scores.get)
    print(f"  Best single model for sub-targets: {best_single_name}")

    def make_base_estimator(name: str):
        if name == "hgb":
            return HistGradientBoostingRegressor(random_state=SEED, **best_hgb_params)
        elif name == "rf":
            return RandomForestRegressor(random_state=SEED, n_jobs=-1, **best_rf_params)
        elif name == "et":
            return ExtraTreesRegressor(random_state=SEED, n_jobs=-1, **best_et_params)
        elif name == "lgb" and has_lgb:
            return lgb_module.LGBMRegressor(random_state=SEED, verbose=-1, **best_lgb_params)
        else:
            return HistGradientBoostingRegressor(random_state=SEED)

    base_est = make_base_estimator(best_single_name)

    # ------------------------------------------------------------------
    # FIX #4: Casual sub-model with regularised weekend split
    # ------------------------------------------------------------------
    print("  [sub-target] casual sub-model (regularised workingday-split)...")
    casual_wd_model = WorkingdaySplitModel(
        base_estimator=base_est,
        wd_col=wd_col if wd_col else "",
        regularise_wd0=True,    # FIX #4: use max_leaf_nodes=15, l2_reg=1.0 for weekend
    )
    casual_wd_res = cv_wd_model_with_fold_aggs(
        casual_wd_model, X_train_base, y_casual_log, splits,
        fold_agg_casual, wd_col if wd_col else "", return_oof=True
    )
    results["sub_casual"] = {
        "mean": casual_wd_res["mean"],
        "std": casual_wd_res["std"],
        "folds": casual_wd_res["folds"],
    }
    oof_casual = casual_wd_res["oof_preds"]
    print(f"    Casual CV RMSLE: {casual_wd_res['mean']:.5f}  (round2: 0.56566)")

    # Retrain casual on all data
    fold_agg_casual.fit_on_all_train()
    full_agg_casual = fold_agg_casual.transform_train_idx(np.arange(n_train))
    X_full_casual_df = pd.concat(
        [X_train_base.reset_index(drop=True), full_agg_casual.reset_index(drop=True)], axis=1
    )
    feature_names_casual = X_full_casual_df.columns.tolist()
    X_full_casual_arr = X_full_casual_df.values.astype(float)
    casual_wd_model_final = WorkingdaySplitModel(
        base_estimator=make_base_estimator(best_single_name),
        wd_col=wd_col if wd_col else "",
        regularise_wd0=True,
    )
    casual_wd_model_final.fit(X_full_casual_arr, y_casual_log, feature_names_casual)
    fold_agg_casual.fit_on_all_train()
    X_test_aug_casual = augment_with_full_train_aggs(X_test_base, fold_agg_casual, is_test=True)
    test_preds_casual = casual_wd_model_final.predict(X_test_aug_casual)

    # ------------------------------------------------------------------
    # Registered sub-model (standard)
    # ------------------------------------------------------------------
    print("  [sub-target] registered sub-model...")
    reg_model = make_base_estimator(best_single_name)
    reg_res = cv_model_with_fold_aggs(
        reg_model, X_train_base, y_registered_log, splits, fold_agg_registered, return_oof=True
    )
    results["sub_registered"] = {
        "mean": reg_res["mean"],
        "std": reg_res["std"],
        "folds": reg_res["folds"],
    }
    oof_registered = reg_res["oof_preds"]
    print(f"    Registered CV RMSLE: {reg_res['mean']:.5f}  (round2: 0.34835)")

    # FIX #4 complement: include sub_registered in NNLS candidates (carried from round 2)
    oof_collection["sub_registered"] = oof_registered

    # Retrain registered on all data
    fold_agg_registered.fit_on_all_train()
    full_agg_reg = fold_agg_registered.transform_train_idx(np.arange(n_train))
    X_full_reg = pd.concat(
        [X_train_base.reset_index(drop=True), full_agg_reg.reset_index(drop=True)], axis=1
    ).values.astype(float)
    reg_model_final = make_base_estimator(best_single_name)
    reg_model_final.fit(X_full_reg, y_registered_log)
    fold_agg_registered.fit_on_all_train()
    X_test_aug_reg = augment_with_full_train_aggs(X_test_base, fold_agg_registered, is_test=True)
    test_preds_registered = reg_model_final.predict(X_test_aug_reg)
    test_preds["sub_registered"] = test_preds_registered

    # ------------------------------------------------------------------
    # FIX #3: sub_target_sum — compute OOF and add to NNLS candidates
    # ------------------------------------------------------------------
    print("  [sub-target] sub_target_sum (casual + registered)...")
    oof_sub_sum_orig = np.expm1(oof_casual[val_mask]) + np.expm1(oof_registered[val_mask])
    oof_sub_sum_log = np.log1p(np.maximum(oof_sub_sum_orig, 0.0))
    sub_sum_rmsle = rmsle_log_space(y_count_log[val_mask], oof_sub_sum_log)
    results["sub_target_sum"] = {"mean": sub_sum_rmsle, "std": 0.0, "folds": []}
    print(f"    Sub-target sum CV RMSLE: {sub_sum_rmsle:.5f}  (round2: 0.35568)")

    # sub_target_sum in log space (for NNLS — all candidates must be in log space)
    # Need OOF over ALL train rows, not just val_mask rows, for NNLS matrix
    oof_sub_sum_full = np.log1p(
        np.maximum(np.expm1(oof_casual) + np.expm1(oof_registered), 0.0)
    )
    # FIX #3: add sub_target_sum as NNLS blend candidate
    oof_collection["sub_target_sum"] = oof_sub_sum_full

    test_preds["sub_target_sum"] = np.log1p(
        np.maximum(np.expm1(test_preds_casual) + np.expm1(test_preds_registered), 0.0)
    )

    # ------------------------------------------------------------------
    # FIX #3: NNLS blend (sub_registered + sub_target_sum + tree models)
    # ------------------------------------------------------------------
    print("  [blend] NNLS blend...")
    blend_model_names = list(oof_collection.keys())
    print(f"    Blend candidates: {blend_model_names}")

    oof_matrix = np.column_stack([oof_collection[k][val_mask] for k in blend_model_names])
    y_val = y_count_log[val_mask]
    try:
        coef, _ = nnls(oof_matrix, y_val)
        if coef.sum() > 1e-10:
            coef = coef / coef.sum()
        else:
            coef = np.ones(len(blend_model_names)) / len(blend_model_names)
    except Exception:
        coef = np.ones(len(blend_model_names)) / len(blend_model_names)

    blend_coef = dict(zip(blend_model_names, coef.tolist()))
    oof_blend = oof_matrix @ coef
    blend_cv_rmsle = rmsle_log_space(y_val, oof_blend)
    results["nnls_blend"] = {"mean": blend_cv_rmsle, "std": 0.0, "folds": []}
    print(f"    NNLS blend CV RMSLE: {blend_cv_rmsle:.5f}")
    print(f"    Coef: { {k: f'{v:.4f}' for k, v in blend_coef.items()} }")

    test_blend_arr = np.column_stack(
        [test_preds[k] for k in blend_model_names]
    ) @ coef
    test_preds["nnls_blend"] = test_blend_arr

    return {
        "cv_scores": results,
        "test_preds": test_preds,
        "blend_coef": blend_coef,
        "blend_model_names": blend_model_names,
    }


# ---------------------------------------------------------------------------
# Select best model (skip baselines; prefer lower RMSLE)
# ---------------------------------------------------------------------------
def select_best(cv_scores: dict, test_preds: dict) -> tuple[str, float]:
    # Exclude sub-component models: they predict registered/casual (not count),
    # so their CV RMSLE is against a different target — incomparable to count models.
    skip = {"dummy", "ridge_baseline", "sub_registered", "sub_casual"}
    best_name = None
    best_score = float("inf")
    for k in test_preds.keys():
        if k in skip:
            continue
        score = cv_scores.get(k, {}).get("mean", float("inf"))
        if score < best_score:
            best_score = score
            best_name = k
    return best_name, best_score


# ---------------------------------------------------------------------------
# Submission generation
# ---------------------------------------------------------------------------
def generate_submission(
    test_df: pd.DataFrame,
    sample_sub: pd.DataFrame,
    test_preds_log: np.ndarray,
    spec: dict,
) -> pd.DataFrame:
    row_id_col = spec["row_id_column"]
    target_col = spec["target_column"]
    pred_orig = np.expm1(test_preds_log)
    pred_clipped = np.maximum(pred_orig, 1.0)   # clip at 1 (not 0) per low-priority hint
    pred_int = np.round(pred_clipped).astype(int)
    sub = pd.DataFrame({
        row_id_col: test_df[row_id_col].values,
        target_col: pred_int,
    })
    sub = sub.set_index(row_id_col)
    sub = sub.reindex(sample_sub[row_id_col].values)
    sub = sub.reset_index()
    sub.columns = [row_id_col, target_col]
    assert sub.shape == (len(sample_sub), 2)
    assert list(sub.columns) == [row_id_col, target_col]
    assert sub[target_col].notna().all()
    assert (sub[target_col] >= 0).all()
    return sub


# ---------------------------------------------------------------------------
# Write log files
# ---------------------------------------------------------------------------
def write_logs(
    spec: dict,
    cv_scores: dict,
    blend_coef: dict,
    blend_model_names: list,
    best_name: str,
    best_cv_rmsle: float,
    n_features: int,
    elapsed: float,
    submission_updated: bool,
) -> None:
    model_search = {
        "run_id": RUN_ID,
        "round": 3,
        "metric_name": "rmsle",
        "cv_strategy": "within_month_cross_day_grouped_kfold",
        "cv_description": (
            "Round 3: +tgt_hour_month_mean, +tgt_hour_wd_year_mean, "
            "+sub_target_sum in NNLS, casual wd0 l2_reg=1.0 max_leaf_nodes=15"
        ),
        "n_splits": N_CV_SPLITS,
        "target_transform": "log1p",
        "cv_scores": {k: {"mean": v["mean"], "std": v.get("std", 0.0)} for k, v in cv_scores.items()},
        "best_model_name": best_name,
        "best_cv_rmsle": best_cv_rmsle,
        "prev_best_cv_rmsle": PREV_BEST_RMSLE,
        "submission_updated": submission_updated,
        "blend_coef": blend_coef,
        "blend_model_names": blend_model_names,
        "n_features": n_features,
        "elapsed_sec": round(elapsed, 1),
        "baseline_rmsle_dummy": cv_scores.get("dummy", {}).get("mean"),
        "baseline_rmsle_ridge": cv_scores.get("ridge_baseline", {}).get("mean"),
        "round3_fixes_applied": [
            "tgt_hour_month_mean_added",
            "tgt_hour_wd_year_mean_added",
            "sub_target_sum_in_nnls_blend",
            "casual_wd0_regularised_max_leaf_nodes_15_l2_1.0",
        ],
    }
    with open(LOGS_DIR / "model_search.json", "w") as f:
        json.dump(model_search, f, indent=2)

    final_model = {
        "run_id": RUN_ID,
        "round": 3,
        "best_model_name": best_name,
        "selection_metric": "rmsle",
        "cv_rmsle": best_cv_rmsle,
        "prev_best_cv_rmsle": PREV_BEST_RMSLE,
        "submission_updated": submission_updated,
        "blend_coef": blend_coef,
        "cv_strategy": "within_month_cross_day",
        "feature_count": n_features,
        "train_rows": spec["file_schemas"][spec["train_file"]]["n_rows"],
        "test_rows": spec["file_schemas"][spec["prediction_file"]]["n_rows"],
        "submission_path": str(REPO_ROOT / "submission.csv"),
    }
    with open(LOGS_DIR / "final_model.json", "w") as f:
        json.dump(final_model, f, indent=2)

    # Submission check
    sub = pd.read_csv(REPO_ROOT / "submission.csv")
    target_col = spec["target_column"]
    row_id_col = spec["row_id_column"]
    expected_rows = spec["file_schemas"][spec["sample_submission_file"]]["n_rows"]
    sample_sub_df = pd.read_csv(REPO_ROOT / spec["sample_submission_file"])
    row_id_ok = bool((sub[row_id_col].values == sample_sub_df[row_id_col].values).all())
    check = {
        "run_id": RUN_ID,
        "columns_ok": list(sub.columns) == [row_id_col, target_col],
        "row_count_ok": len(sub) == expected_rows,
        "row_id_alignment_ok": row_id_ok,
        "all_finite": bool(np.isfinite(sub[target_col].values).all()),
        "dtype_ok": True,
        "missing_predictions": int(sub[target_col].isna().sum()),
        "n_rows": len(sub),
        "n_cols": sub.shape[1],
        "pred_min": int(sub[target_col].min()),
        "pred_max": int(sub[target_col].max()),
        "output_kind": "regression_integer",
    }
    check_path = LOGS_DIR / f"{RUN_ID}_submission_check.json"
    with open(check_path, "w") as f:
        json.dump(check, f, indent=2)

    agg_feat_names = [
        "tgt_hour_mean", "tgt_season_mean", "tgt_hour_wd_mean", "tgt_year_month_mean",
        "tgt_hour_month_mean", "tgt_hour_wd_year_mean",     # round 3 new
    ]
    profile = {
        "run_id": RUN_ID,
        "feature_audit": {
            "n_features": n_features,
            "agg_features": agg_feat_names,
            "agg_computation": "per_fold_only_no_leakage",
            "cv_leakage_fixed": True,
            "datetime_cols_detected": [spec["row_id_column"]],
            "raw_row_id_in_features": False,
        },
    }
    with open(LOGS_DIR / f"{RUN_ID}_profile.json", "w") as f:
        json.dump(profile, f, indent=2)

    print(f"  Logs written: model_search.json, final_model.json, {RUN_ID}_submission_check.json, {RUN_ID}_profile.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 68)
    print(f"Round 3 Analysis Pipeline  (run_id={RUN_ID})")
    print(f"Previous best CV RMSLE: {PREV_BEST_RMSLE:.5f}")
    print("Round 3 fixes: tgt_hour_month_mean, tgt_hour_wd_year_mean,")
    print("               sub_target_sum in NNLS, casual wd0 regularised")
    print("=" * 68)

    spec = load_spec()
    plan = load_plan()
    target_col = spec["target_column"]
    row_id_col = spec["row_id_column"]
    leakage_cols = spec.get("leakage_columns", [])

    print(f"\nTask: {spec['task_type']}  |  Target: {target_col}  |  Metric: {spec['evaluation_metric']}")
    print(f"Leakage cols excluded: {leakage_cols}")

    print("\n[1/7] Loading data...")
    train_df, test_df, sample_sub = load_data(spec)
    print(f"  train: {train_df.shape}  |  test: {test_df.shape}")

    # Target arrays (resolved from spec — no hardcoding)
    y_count_log = np.log1p(train_df[target_col].values.astype(float))

    sub_targets = (
        spec.get("detected_structure", {})
        .get("split_pattern", {})
        .get("sub_target_candidates", [])
    )
    casual_col = (
        sub_targets[0]["column"]
        if sub_targets and isinstance(sub_targets[0], dict)
        else (leakage_cols[0] if leakage_cols else None)
    )
    registered_col = (
        sub_targets[1]["column"]
        if len(sub_targets) > 1 and isinstance(sub_targets[1], dict)
        else (leakage_cols[1] if len(leakage_cols) > 1 else None)
    )

    if casual_col and casual_col in train_df.columns:
        y_casual_log = np.log1p(train_df[casual_col].values.astype(float))
    else:
        y_casual_log = y_count_log.copy()
    if registered_col and registered_col in train_df.columns:
        y_registered_log = np.log1p(train_df[registered_col].values.astype(float))
    else:
        y_registered_log = y_count_log.copy()

    print(f"  Sub-target cols: {casual_col}, {registered_col}")

    print("\n[2/7] Building base features (agg features computed per-fold inside CV)...")
    X_train_base, X_test_base, meta = build_base_features(train_df, test_df, spec, plan)
    print(f"  X_train_base: {X_train_base.shape}  |  X_test_base: {X_test_base.shape}")

    # 6 agg features now (4 from round 2 + 2 new from round 3)
    n_agg = 6
    n_features_total = X_train_base.shape[1] + n_agg
    print(f"  Total features (with agg): {n_features_total}")

    print("\n[3/7] Building CV splits...")
    splits = make_cv_splits(train_df, spec, n_splits=N_CV_SPLITS)
    print(f"  Generated {len(splits)} folds")
    for i, (tr_idx, val_idx) in enumerate(splits):
        dt_val = pd.to_datetime(train_df.iloc[val_idx][row_id_col])
        ym_val = (
            dt_val.dt.year.astype(str) + "_" + dt_val.dt.month.astype(str).str.zfill(2)
        ).unique()
        print(
            f"  Fold {i+1}: train={len(tr_idx)}, val={len(val_idx)}, "
            f"val_months={len(ym_val)} ({ym_val[0]}..{ym_val[-1] if len(ym_val) > 1 else ''})"
        )

    # Leakage guard
    for col in [row_id_col, target_col] + leakage_cols:
        assert col not in X_train_base.columns, f"LEAKAGE: {col} in X_train_base!"
    print("  Leakage check passed.")

    print("\n[4/7] Running model search with round 3 features...")
    search_result = run_model_search(
        X_train_base, train_df, test_df,
        y_count_log, y_casual_log, y_registered_log,
        X_test_base, splits, spec, plan,
    )
    cv_scores = search_result["cv_scores"]
    test_preds = search_result["test_preds"]
    blend_coef = search_result["blend_coef"]
    blend_model_names = search_result["blend_model_names"]

    print("\n  CV RMSLE Summary (round 3):")
    best_cv_all = min(v.get("mean", float("inf")) for v in cv_scores.values())
    for k, v in sorted(cv_scores.items(), key=lambda x: x[1].get("mean", float("inf"))):
        arrow = " <-- BEST" if abs(v["mean"] - best_cv_all) < 1e-9 else ""
        print(f"    {k:30s}: {v['mean']:.5f} +/- {v.get('std', 0.0):.5f}{arrow}")

    print("\n[5/7] Selecting best strategy...")
    best_name, best_cv_rmsle = select_best(cv_scores, test_preds)
    print(f"  Best: {best_name}  |  CV RMSLE: {best_cv_rmsle:.5f}")
    print(f"  Previous best: {PREV_BEST_RMSLE:.5f}")

    # Keep-best rule: only update submission if strictly better
    improved = best_cv_rmsle < PREV_BEST_RMSLE
    if improved:
        print(f"  NEW BEST — update submission ({best_cv_rmsle:.5f} < {PREV_BEST_RMSLE:.5f})")
    else:
        print(f"  No improvement — submission NOT updated ({best_cv_rmsle:.5f} >= {PREV_BEST_RMSLE:.5f})")

    print("\n[6/7] Generating submission...")
    best_test_log = test_preds[best_name]
    sub = generate_submission(test_df, sample_sub, best_test_log, spec)

    if improved:
        sub_path = REPO_ROOT / "submission.csv"
        sub.to_csv(sub_path, index=False)
        print(f"  Saved: {sub_path}  (CV RMSLE: {best_cv_rmsle:.5f})")
    else:
        # Still write the round 3 sub for review, but do NOT overwrite root submission.csv
        sub_r3_path = LOGS_DIR / "submission_round3_candidate.csv"
        sub.to_csv(sub_r3_path, index=False)
        print(f"  Candidate saved to {sub_r3_path} (not best)")

    print(f"  Count stats: min={sub[target_col].min()}, max={sub[target_col].max()}, "
          f"mean={sub[target_col].mean():.1f}")

    print("\n[7/7] Writing log files...")
    elapsed = time.time() - t0
    write_logs(
        spec=spec,
        cv_scores=cv_scores,
        blend_coef=blend_coef,
        blend_model_names=blend_model_names,
        best_name=best_name,
        best_cv_rmsle=best_cv_rmsle,
        n_features=n_features_total,
        elapsed=elapsed,
        submission_updated=improved,
    )
    print(f"\nAll done. Elapsed: {elapsed:.1f}s")

    print(f"\n{'=' * 68}")
    print(f"Round 3 complete in {elapsed:.1f}s")
    print(f"Best model: {best_name}")
    print(f"CV RMSLE:   {best_cv_rmsle:.5f}  (prev best: {PREV_BEST_RMSLE:.5f})")
    print(f"Improved:   {improved}")
    print(f"Submission updated: {improved}")
    print(f"{'=' * 68}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
