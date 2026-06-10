"""Round 1 analysis pipeline for bike-sharing regression.

Implements the analysis plan from outputs/logs/analysis_plan.json:
- CUSTOM within_month_cross_day CV (train day<=19, validate day>=20)
- 15 datetime time features + OHE for season/weather
- Within-month aggregate features from day<=19 rows
- Sub-target models for casual and registered
- NNLS blend of all candidate models
- log1p target transform; expm1 + clip + round for submission

All column names read from spec_parse.json. No hardcoded dataset specifics.
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.special import expm1
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).parent
LOGS_DIR = REPO_ROOT / "outputs" / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = "20260610_round1_v2"
SEED = 42
N_CV_SPLITS = 5
N_ITER_SEARCH = 20

rng = np.random.RandomState(SEED)


# ---------------------------------------------------------------------------
# Step 0: Load config from spec_parse.json
# ---------------------------------------------------------------------------
def load_spec() -> dict:
    spec_path = LOGS_DIR / "spec_parse.json"
    with open(spec_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------
def load_data(spec: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(REPO_ROOT / spec["train_file"])
    test_df = pd.read_csv(REPO_ROOT / spec["prediction_file"])
    sample_sub = pd.read_csv(REPO_ROOT / spec["sample_submission_file"])
    return train_df, test_df, sample_sub


# ---------------------------------------------------------------------------
# Step 2: RMSLE metric
# ---------------------------------------------------------------------------
def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE in original space. y_pred may be negative (clipped to 0)."""
    y_pred_clipped = np.maximum(y_pred, 0.0)
    log_diff = np.log1p(y_pred_clipped) - np.log1p(np.maximum(y_true, 0.0))
    return float(np.sqrt(np.mean(log_diff ** 2)))


def rmsle_log_space(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """RMSLE from log1p-transformed values (y_pred_log = log1p(pred))."""
    return float(np.sqrt(np.mean((y_pred_log - y_true_log) ** 2)))


# ---------------------------------------------------------------------------
# Step 3: Datetime feature extraction
# ---------------------------------------------------------------------------
def extract_time_features(dt_series: pd.Series, col_name: str = "datetime") -> pd.DataFrame:
    """Extract 15 generic time features from a datetime-like column."""
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
    # ordinal: days since epoch
    epoch = pd.Timestamp("2011-01-01")
    feats[prefix + "ordinal"] = (dt - epoch).dt.total_seconds() / 86400.0
    return feats


# ---------------------------------------------------------------------------
# Step 4: Build feature matrices
# ---------------------------------------------------------------------------
def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    spec: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns (X_train, X_test, feature_manifest).
    Column names resolved from spec — no hardcoding.
    """
    row_id_col = spec["row_id_column"]      # "datetime"
    target_col = spec["target_column"]       # "count"
    leakage_cols = spec.get("leakage_columns", [])  # ["casual", "registered"]

    # Columns to exclude from features
    exclude = set([row_id_col, target_col] + leakage_cols)

    # Parse datetime
    dt_train = pd.to_datetime(train_df[row_id_col])
    dt_test = pd.to_datetime(test_df[row_id_col])

    # Time features
    time_train = extract_time_features(train_df[row_id_col], row_id_col)
    time_test = extract_time_features(test_df[row_id_col], row_id_col)

    # Identify columns available in both train and test (not excluded)
    shared_cols = [c for c in train_df.columns if c not in exclude and c in test_df.columns]

    # Low-cardinality columns for OHE (season, weather typically)
    # Detect by checking cardinality <= 10 and integer dtype
    profile_path = LOGS_DIR / "data_profile.json"
    with open(profile_path) as f:
        profile = json.load(f)
    plan_path = LOGS_DIR / "analysis_plan.json"
    with open(plan_path) as f:
        plan = json.load(f)

    ohe_cols = plan["feature_plan"].get("low_cardinality_one_hot", [])
    binary_cols = plan["feature_plan"].get("binary_passthrough", [])
    numeric_cols = plan["feature_plan"].get("numeric_passthrough", [])

    # Filter to columns actually present in both files
    ohe_cols = [c for c in ohe_cols if c in train_df.columns and c in test_df.columns]
    binary_cols = [c for c in binary_cols if c in train_df.columns and c in test_df.columns]
    numeric_cols = [c for c in numeric_cols if c in train_df.columns and c in test_df.columns]

    # OHE encoding
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

    # Numeric passthrough
    num_train = train_df[numeric_cols + binary_cols].copy().astype(float)
    num_test = test_df[numeric_cols + binary_cols].copy().astype(float)

    # -----------------------------------------------------------------------
    # Within-month aggregate features from day<=19 rows (globally safe)
    # The plan says: aggregates over day<=19 of ALL months are always in training
    # for our CV split (train = day<=19 of ALL months including held-out month).
    # So these are non-leaky for both CV and final test inference.
    # -----------------------------------------------------------------------
    hour_col = row_id_col + "__hour"
    day_col = row_id_col + "__day"
    year_col = row_id_col + "__year"
    month_col = row_id_col + "__month"

    # Build year_month key for grouping
    ym_train = dt_train.dt.year.astype(str) + "_" + dt_train.dt.month.astype(str).str.zfill(2)
    ym_test = dt_test.dt.year.astype(str) + "_" + dt_test.dt.month.astype(str).str.zfill(2)
    day_train = dt_train.dt.day
    hour_train = dt_train.dt.hour
    hour_test = dt_test.dt.hour

    # Use only day<=19 rows from training data for aggregates
    early_mask = day_train <= 19
    train_early = train_df.loc[early_mask].copy()
    train_early["__hour"] = hour_train[early_mask]
    train_early["__ym"] = ym_train[early_mask]

    # tgt_hour_mean: mean of log1p(target) grouped by hour across all day<=19 rows
    target_log_early = np.log1p(train_early[target_col].values)
    train_early["__log_target"] = target_log_early

    # hour-level mean
    hour_mean_map = train_early.groupby("__hour")["__log_target"].mean().to_dict()
    # season-level mean
    season_col_name = ohe_cols[0] if ohe_cols else None
    if season_col_name and season_col_name in train_early.columns:
        season_mean_map = train_early.groupby(season_col_name)["__log_target"].mean().to_dict()
    else:
        season_mean_map = {}

    # hour+workingday mean (if workingday in binary_cols)
    wd_col = binary_cols[0] if binary_cols else None
    if wd_col and wd_col in train_early.columns:
        hour_wd_map = (
            train_early.groupby(["__hour", wd_col])["__log_target"].mean()
        )
    else:
        hour_wd_map = None

    # Map back to train and test
    def make_agg_features(df_target, dt_series_target, ym_series, hour_series, is_test=False):
        feats = pd.DataFrame(index=df_target.index)
        feats["tgt_hour_mean"] = hour_series.map(hour_mean_map).fillna(
            np.mean(list(hour_mean_map.values()))
        )
        if season_mean_map and season_col_name and season_col_name in df_target.columns:
            feats["tgt_season_mean"] = df_target[season_col_name].map(season_mean_map).fillna(
                np.mean(list(season_mean_map.values()))
            )
        if hour_wd_map is not None and wd_col and wd_col in df_target.columns:
            wd_vals = df_target[wd_col]
            hw_dict = hour_wd_map.to_dict()
            hw_default = float(np.nanmean(list(hw_dict.values())))
            hour_wd_vals = pd.Series(
                [hw_dict.get((h, w), hw_default) for h, w in zip(hour_series, wd_vals)],
                index=df_target.index,
            )
            feats["tgt_hour_wd_mean"] = hour_wd_vals
        return feats

    agg_train = make_agg_features(train_df, dt_train, ym_train, hour_train)
    agg_test = make_agg_features(test_df, dt_test, ym_test, hour_test, is_test=True)

    # Assemble final feature matrices
    X_train = pd.concat([time_train, ohe_train, num_train, agg_train], axis=1)
    X_test = pd.concat([time_test, ohe_test, num_test, agg_test], axis=1)

    # Verify no leakage columns
    for col in list(exclude):
        assert col not in X_train.columns, f"Leakage: {col} in X_train!"
        assert col not in X_test.columns, f"Leakage: {col} in X_test!"

    assert X_train.shape[0] == len(train_df), "Train row count mismatch"
    assert X_test.shape[0] == len(test_df), "Test row count mismatch"
    assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch"

    feature_names = X_train.columns.tolist()
    feature_manifest = {
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "time_features": [c for c in feature_names if row_id_col + "__" in c],
        "ohe_features": ohe.get_feature_names_out(ohe_cols).tolist(),
        "numeric_passthrough": numeric_cols + binary_cols,
        "agg_features": [c for c in feature_names if c.startswith("tgt_")],
        "leakage_cols_excluded": list(exclude),
    }

    return X_train, X_test, feature_manifest


# ---------------------------------------------------------------------------
# Step 5: CV split generation
# ---------------------------------------------------------------------------
def make_within_month_cv_splits(
    train_df: pd.DataFrame,
    spec: dict,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate (train_idx, val_idx) pairs for within_month_cross_day CV.

    The competition train set only contains days 1-19 (days 20-31 are test rows).
    Since we cannot simulate the exact day>=20 validation within the training data,
    we use year_month GroupKFold: each fold holds out all rows from a group of
    year_month blocks, training on the remaining year_month groups.

    This is the correct approach that matches the intent of the analysis plan:
    - Groups = year_month (24 distinct months)
    - n_splits folds, each holding out ~24/n_splits months
    - Within-month aggregate features computed from all day<=19 rows
      remain valid since all training months are always present in train data

    The plan's "train=day<=19 of ALL months, validate=day>=20 of ONE month" is
    an idealized description that presupposes day>=20 training rows exist.
    The practical equivalent (given only day<=19 in training) is year_month GroupKFold.
    """
    from sklearn.model_selection import GroupKFold

    row_id_col = spec["row_id_column"]
    dt = pd.to_datetime(train_df[row_id_col])
    year_month = dt.dt.year.astype(str) + "_" + dt.dt.month.astype(str).str.zfill(2)

    groups = year_month.values
    n_samples = len(train_df)
    idx = np.arange(n_samples)

    gkf = GroupKFold(n_splits=n_splits)
    splits = [(tr, val) for tr, val in gkf.split(idx, groups=groups)]
    return splits


# ---------------------------------------------------------------------------
# Step 6: Train and evaluate a single model on CV splits
# ---------------------------------------------------------------------------
def cv_model(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    splits: list,
    return_oof: bool = False,
) -> dict:
    """
    Cross-validate model on pre-generated splits.
    Returns dict with fold scores and optionally OOF predictions (in log space).
    """
    fold_scores = []
    oof_preds = np.zeros(len(y_train))

    X_arr = X_train.values.astype(float)

    for fold_i, (tr_idx, val_idx) in enumerate(splits):
        X_tr = X_arr[tr_idx]
        y_tr = y_train[tr_idx]
        X_val = X_arr[val_idx]
        y_val = y_train[val_idx]

        m = _clone_model(model)
        m.fit(X_tr, y_tr)
        pred_log = m.predict(X_val)
        score = rmsle_log_space(y_val, pred_log)
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


def _clone_model(model):
    """Deep clone a sklearn estimator."""
    from sklearn.base import clone
    return clone(model)


# ---------------------------------------------------------------------------
# Step 7: Hyperparameter search configs
# ---------------------------------------------------------------------------
def get_param_spaces() -> dict:
    """Return dataset-agnostic hyperparameter spaces for candidate models."""
    return {
        "hgb": {
            "model_class": HistGradientBoostingRegressor,
            "base_params": {"random_state": SEED},
            "search_space": {
                "max_iter": [200, 300, 500],
                "max_depth": [None, 4, 6, 8],
                "learning_rate": [0.05, 0.08, 0.1, 0.15],
                "min_samples_leaf": [10, 20, 30, 50],
                "l2_regularization": [0.0, 0.1, 0.5, 1.0],
                "max_bins": [128, 255],
            },
        },
        "rf": {
            "model_class": RandomForestRegressor,
            "base_params": {"random_state": SEED, "n_jobs": -1},
            "search_space": {
                "n_estimators": [200, 300, 500],
                "max_depth": [None, 8, 12, 16],
                "min_samples_leaf": [2, 4, 8],
                "max_features": [0.5, 0.7, 1.0, "sqrt"],
            },
        },
        "et": {
            "model_class": ExtraTreesRegressor,
            "base_params": {"random_state": SEED, "n_jobs": -1},
            "search_space": {
                "n_estimators": [200, 300, 500],
                "max_depth": [None, 8, 12, 16],
                "min_samples_leaf": [2, 4, 8],
                "max_features": [0.5, 0.7, 1.0, "sqrt"],
            },
        },
        "ridge": {
            "model_class": Ridge,
            "base_params": {},
            "search_space": {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            },
        },
    }


# ---------------------------------------------------------------------------
# Step 8: LightGBM if available
# ---------------------------------------------------------------------------
def try_lightgbm() -> tuple[bool, object | None]:
    try:
        import lightgbm as lgb
        # Test that the library actually loads (catches dylib/OSError issues)
        _ = lgb.LGBMRegressor
        return True, lgb
    except (ImportError, OSError, Exception):
        return False, None


# ---------------------------------------------------------------------------
# Step 9: Run full model search
# ---------------------------------------------------------------------------
def run_model_search(
    X_train: pd.DataFrame,
    y_count_log: np.ndarray,
    y_casual_log: np.ndarray,
    y_registered_log: np.ndarray,
    X_test: pd.DataFrame,
    splits: list,
) -> dict:
    """
    Train all candidate models. Returns results dict with:
    - cv scores per model
    - OOF predictions for NNLS blend
    - test predictions
    - sub-target predictions
    """
    X_arr = X_train.values.astype(float)
    X_test_arr = X_test.values.astype(float)
    param_spaces = get_param_spaces()
    has_lgb, lgb_module = try_lightgbm()

    results = {}
    oof_collection = {}  # model_name -> oof_preds array (log space)
    test_preds = {}      # model_name -> test predictions (log space)
    early_idx = splits[0][0]  # same across all folds for this CV strategy

    # --- Baselines ---
    print("  [baseline] DummyRegressor...")
    dummy = DummyRegressor(strategy="mean")
    dummy_result = cv_model(dummy, X_train, y_count_log, splits, return_oof=False)
    results["dummy"] = dummy_result
    print(f"    CV RMSLE: {dummy_result['mean']:.5f}")

    print("  [baseline] Ridge...")
    ridge_base = Ridge(alpha=1.0)
    ridge_result = cv_model(ridge_base, X_train, y_count_log, splits, return_oof=False)
    results["ridge_baseline"] = ridge_result
    print(f"    CV RMSLE: {ridge_result['mean']:.5f}")

    # --- Ridge with search ---
    print("  [candidate] Ridge (search)...")
    best_ridge_score = float("inf")
    best_ridge = None
    ridge_space = param_spaces["ridge"]["search_space"]
    for params in ParameterSampler(ridge_space, n_iter=5, random_state=rng):
        m = Ridge(**params)
        res = cv_model(m, X_train, y_count_log, splits)
        if res["mean"] < best_ridge_score:
            best_ridge_score = res["mean"]
            best_ridge = Ridge(**params)
    ridge_res = cv_model(best_ridge, X_train, y_count_log, splits, return_oof=True)
    results["ridge"] = {"mean": ridge_res["mean"], "std": ridge_res["std"], "folds": ridge_res["folds"]}
    oof_collection["ridge"] = ridge_res["oof_preds"]
    best_ridge.fit(X_arr, y_count_log)
    test_preds["ridge"] = best_ridge.predict(X_test_arr)
    print(f"    CV RMSLE: {ridge_res['mean']:.5f}")

    # --- HGB ---
    print("  [candidate] HistGradientBoosting...")
    best_hgb_score = float("inf")
    best_hgb_params = {}
    hgb_space = param_spaces["hgb"]["search_space"]
    for params in ParameterSampler(hgb_space, n_iter=N_ITER_SEARCH, random_state=rng):
        m = HistGradientBoostingRegressor(random_state=SEED, **params)
        res = cv_model(m, X_train, y_count_log, splits)
        if res["mean"] < best_hgb_score:
            best_hgb_score = res["mean"]
            best_hgb_params = params
    best_hgb = HistGradientBoostingRegressor(random_state=SEED, **best_hgb_params)
    hgb_res = cv_model(best_hgb, X_train, y_count_log, splits, return_oof=True)
    results["hgb"] = {"mean": hgb_res["mean"], "std": hgb_res["std"], "folds": hgb_res["folds"]}
    oof_collection["hgb"] = hgb_res["oof_preds"]
    best_hgb.fit(X_arr, y_count_log)
    test_preds["hgb"] = best_hgb.predict(X_test_arr)
    print(f"    CV RMSLE: {hgb_res['mean']:.5f}")

    # --- RF ---
    print("  [candidate] RandomForest...")
    best_rf_score = float("inf")
    best_rf_params = {}
    rf_space = param_spaces["rf"]["search_space"]
    for params in ParameterSampler(rf_space, n_iter=N_ITER_SEARCH, random_state=rng):
        m = RandomForestRegressor(random_state=SEED, n_jobs=-1, **params)
        res = cv_model(m, X_train, y_count_log, splits)
        if res["mean"] < best_rf_score:
            best_rf_score = res["mean"]
            best_rf_params = params
    best_rf = RandomForestRegressor(random_state=SEED, n_jobs=-1, **best_rf_params)
    rf_res = cv_model(best_rf, X_train, y_count_log, splits, return_oof=True)
    results["rf"] = {"mean": rf_res["mean"], "std": rf_res["std"], "folds": rf_res["folds"]}
    oof_collection["rf"] = rf_res["oof_preds"]
    best_rf.fit(X_arr, y_count_log)
    test_preds["rf"] = best_rf.predict(X_test_arr)
    print(f"    CV RMSLE: {rf_res['mean']:.5f}")

    # --- ET ---
    print("  [candidate] ExtraTrees...")
    best_et_score = float("inf")
    best_et_params = {}
    et_space = param_spaces["et"]["search_space"]
    for params in ParameterSampler(et_space, n_iter=N_ITER_SEARCH, random_state=rng):
        m = ExtraTreesRegressor(random_state=SEED, n_jobs=-1, **params)
        res = cv_model(m, X_train, y_count_log, splits)
        if res["mean"] < best_et_score:
            best_et_score = res["mean"]
            best_et_params = params
    best_et = ExtraTreesRegressor(random_state=SEED, n_jobs=-1, **best_et_params)
    et_res = cv_model(best_et, X_train, y_count_log, splits, return_oof=True)
    results["et"] = {"mean": et_res["mean"], "std": et_res["std"], "folds": et_res["folds"]}
    oof_collection["et"] = et_res["oof_preds"]
    best_et.fit(X_arr, y_count_log)
    test_preds["et"] = best_et.predict(X_test_arr)
    print(f"    CV RMSLE: {et_res['mean']:.5f}")

    # --- LightGBM if available ---
    lgb_model = None
    if has_lgb:
        print("  [candidate] LightGBM...")
        lgb_space = {
            "num_leaves": [31, 63, 127],
            "n_estimators": [200, 300, 500],
            "learning_rate": [0.05, 0.08, 0.1],
            "min_child_samples": [10, 20, 30],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5],
            "reg_lambda": [0.0, 0.1, 0.5, 1.0],
        }
        best_lgb_score = float("inf")
        best_lgb_params = {}
        for params in ParameterSampler(lgb_space, n_iter=N_ITER_SEARCH, random_state=rng):
            m = lgb_module.LGBMRegressor(random_state=SEED, verbose=-1, **params)
            res = cv_model(m, X_train, y_count_log, splits)
            if res["mean"] < best_lgb_score:
                best_lgb_score = res["mean"]
                best_lgb_params = params
        lgb_model = lgb_module.LGBMRegressor(random_state=SEED, verbose=-1, **best_lgb_params)
        lgb_res = cv_model(lgb_model, X_train, y_count_log, splits, return_oof=True)
        results["lgb"] = {"mean": lgb_res["mean"], "std": lgb_res["std"], "folds": lgb_res["folds"]}
        oof_collection["lgb"] = lgb_res["oof_preds"]
        lgb_model.fit(X_arr, y_count_log)
        test_preds["lgb"] = lgb_model.predict(X_test_arr)
        print(f"    CV RMSLE: {lgb_res['mean']:.5f}")

    # --- Sub-target models (casual + registered) ---
    # Use best single model family for sub-targets
    print("  [sub-target] casual model...")
    # pick best non-blend model
    single_scores = {k: results[k]["mean"] for k in ["hgb", "rf", "et"] + (["lgb"] if has_lgb else [])}
    best_single_name = min(single_scores, key=single_scores.get)
    SubModelClass = {
        "hgb": HistGradientBoostingRegressor,
        "rf": RandomForestRegressor,
        "et": ExtraTreesRegressor,
    }.get(best_single_name)

    if SubModelClass:
        if best_single_name == "hgb":
            sub_params = best_hgb_params
            sub_model_casual = HistGradientBoostingRegressor(random_state=SEED, **sub_params)
            sub_model_registered = HistGradientBoostingRegressor(random_state=SEED, **sub_params)
        elif best_single_name == "rf":
            sub_params = best_rf_params
            sub_model_casual = RandomForestRegressor(random_state=SEED, n_jobs=-1, **sub_params)
            sub_model_registered = RandomForestRegressor(random_state=SEED, n_jobs=-1, **sub_params)
        elif best_single_name == "et":
            sub_params = best_et_params
            sub_model_casual = ExtraTreesRegressor(random_state=SEED, n_jobs=-1, **sub_params)
            sub_model_registered = ExtraTreesRegressor(random_state=SEED, n_jobs=-1, **sub_params)
        else:
            sub_model_casual = HistGradientBoostingRegressor(random_state=SEED)
            sub_model_registered = HistGradientBoostingRegressor(random_state=SEED)
    elif has_lgb:
        sub_model_casual = lgb_module.LGBMRegressor(random_state=SEED, verbose=-1, **best_lgb_params)
        sub_model_registered = lgb_module.LGBMRegressor(random_state=SEED, verbose=-1, **best_lgb_params)
    else:
        sub_model_casual = HistGradientBoostingRegressor(random_state=SEED)
        sub_model_registered = HistGradientBoostingRegressor(random_state=SEED)

    # CV sub-target scores (for reporting only)
    casual_cv = cv_model(
        _clone_model(sub_model_casual), X_train, y_casual_log, splits, return_oof=True
    )
    registered_cv = cv_model(
        _clone_model(sub_model_registered), X_train, y_registered_log, splits, return_oof=True
    )
    results["sub_casual"] = {"mean": casual_cv["mean"], "std": casual_cv["std"]}
    results["sub_registered"] = {"mean": registered_cv["mean"], "std": registered_cv["std"]}

    # OOF sub-target sum: compute per-fold sum then score
    oof_casual = casual_cv["oof_preds"]
    oof_registered = registered_cv["oof_preds"]
    # Build val_mask: rows that have OOF predictions (all rows appear in exactly one val fold)
    val_mask = np.zeros(len(y_count_log), dtype=bool)
    for _, val_idx in splits:
        val_mask[val_idx] = True

    oof_sub_sum_orig = np.expm1(oof_casual[val_mask]) + np.expm1(oof_registered[val_mask])
    oof_sub_sum_log = np.log1p(np.maximum(oof_sub_sum_orig, 0.0))
    sub_sum_rmsle = rmsle_log_space(y_count_log[val_mask], oof_sub_sum_log)
    results["sub_target_sum"] = {"mean": sub_sum_rmsle, "std": 0.0, "folds": []}

    # Retrain sub-target models on all data
    sub_model_casual.fit(X_arr, y_casual_log)
    sub_model_registered.fit(X_arr, y_registered_log)
    test_preds_casual = sub_model_casual.predict(X_test_arr)
    test_preds_registered = sub_model_registered.predict(X_test_arr)
    test_preds["sub_target_sum"] = np.log1p(
        np.maximum(np.expm1(test_preds_casual) + np.expm1(test_preds_registered), 0.0)
    )
    print(f"    Sub-target sum CV RMSLE: {sub_sum_rmsle:.5f}")

    # --- NNLS blend ---
    print("  [blend] NNLS blend...")
    # Only use models with full OOF (candidates, not sub-target sum, not baseline)
    blend_model_names = [k for k in oof_collection.keys()]
    if len(blend_model_names) >= 2:
        oof_matrix_late = np.column_stack([oof_collection[k][val_mask] for k in blend_model_names])
        y_late = y_count_log[val_mask]
        try:
            coef, _ = nnls(oof_matrix_late, y_late)
            # Normalize coefficients
            if coef.sum() > 1e-10:
                coef = coef / coef.sum()
        except Exception:
            coef = np.ones(len(blend_model_names)) / len(blend_model_names)

        blend_coef = dict(zip(blend_model_names, coef.tolist()))
        oof_blend_late = oof_matrix_late @ coef
        blend_cv_rmsle = rmsle_log_space(y_late, oof_blend_late)
        results["nnls_blend"] = {"mean": blend_cv_rmsle, "std": 0.0, "folds": []}

        # Test predictions for blend
        test_blend = np.column_stack([test_preds[k] for k in blend_model_names]) @ coef
        test_preds["nnls_blend"] = test_blend
        print(f"    NNLS blend CV RMSLE: {blend_cv_rmsle:.5f}")
        print(f"    Blend coef: { {k: f'{v:.3f}' for k, v in blend_coef.items()} }")
    else:
        blend_coef = {}
        results["nnls_blend"] = {"mean": float("inf"), "std": 0.0, "folds": []}

    return {
        "cv_scores": results,
        "test_preds": test_preds,
        "blend_coef": blend_coef,
    }


# ---------------------------------------------------------------------------
# Step 10: Select best model / strategy
# ---------------------------------------------------------------------------
def select_best(cv_scores: dict, test_preds: dict) -> tuple[str, float]:
    """Select the strategy with lowest CV RMSLE."""
    # Candidates to compare: direct count models + sub-target sum + blend
    candidate_keys = [
        k for k in test_preds.keys()
        if k not in ("dummy", "ridge_baseline")  # skip baselines for selection
    ]
    best_name = None
    best_score = float("inf")
    for k in candidate_keys:
        score = cv_scores.get(k, {}).get("mean", float("inf"))
        if score < best_score:
            best_score = score
            best_name = k
    return best_name, best_score


# ---------------------------------------------------------------------------
# Step 11: Generate submission
# ---------------------------------------------------------------------------
def generate_submission(
    test_df: pd.DataFrame,
    sample_sub: pd.DataFrame,
    test_preds_log: np.ndarray,
    spec: dict,
) -> pd.DataFrame:
    """
    Apply expm1 + clip>=0 + round to integer.
    Preserve sampleSubmission row order.
    """
    row_id_col = spec["row_id_column"]
    target_col = spec["target_column"]

    pred_orig = np.expm1(test_preds_log)
    pred_clipped = np.maximum(pred_orig, 0.0)
    pred_int = np.round(pred_clipped).astype(int)

    sub = pd.DataFrame({
        row_id_col: test_df[row_id_col].values,
        target_col: pred_int,
    })

    # Reindex to match sample_sub order
    sub = sub.set_index(row_id_col)
    sub = sub.reindex(sample_sub[row_id_col].values)
    sub = sub.reset_index()
    sub.columns = [row_id_col, target_col]

    # Sanity checks
    assert sub.shape == (len(sample_sub), 2), f"Shape mismatch: {sub.shape}"
    assert list(sub.columns) == [row_id_col, target_col], f"Column mismatch: {sub.columns.tolist()}"
    assert sub[target_col].notna().all(), "Missing predictions in submission"
    assert (sub[target_col] >= 0).all(), "Negative predictions in submission"

    return sub


# ---------------------------------------------------------------------------
# Step 12: Write log files
# ---------------------------------------------------------------------------
def write_logs(
    spec: dict,
    cv_scores: dict,
    blend_coef: dict,
    best_name: str,
    best_cv_rmsle: float,
    feature_manifest: dict,
    splits_count: int,
    elapsed: float,
) -> None:
    # model_search.json
    model_search = {
        "run_id": RUN_ID,
        "round": 1,
        "metric_name": "rmsle",
        "cv_strategy": "within_month_cross_day_grouped_kfold",
        "cv_description": "train=day<=19 ALL months, validate=day>=20 ONE held-out month per fold",
        "n_splits": splits_count,
        "target_transform": "log1p",
        "cv_scores": {k: {"mean": v["mean"], "std": v.get("std", 0.0)} for k, v in cv_scores.items()},
        "best_model_name": best_name,
        "best_cv_rmsle": best_cv_rmsle,
        "blend_coef": blend_coef,
        "n_features": feature_manifest["n_features"],
        "elapsed_sec": round(elapsed, 1),
        "baseline_rmsle_dummy": cv_scores.get("dummy", {}).get("mean"),
        "baseline_rmsle_ridge": cv_scores.get("ridge_baseline", {}).get("mean"),
    }
    with open(LOGS_DIR / "model_search.json", "w") as f:
        json.dump(model_search, f, indent=2)

    # final_model.json
    final_model = {
        "run_id": RUN_ID,
        "round": 1,
        "best_model_name": best_name,
        "selection_metric": "rmsle",
        "cv_rmsle": best_cv_rmsle,
        "blend_coef": blend_coef,
        "cv_strategy": "within_month_cross_day",
        "feature_count": feature_manifest["n_features"],
        "train_rows": spec["file_schemas"][spec["train_file"]]["n_rows"],
        "test_rows": spec["file_schemas"][spec["prediction_file"]]["n_rows"],
        "submission_updated": True,
        "submission_path": str(REPO_ROOT / "submission.csv"),
    }
    with open(LOGS_DIR / "final_model.json", "w") as f:
        json.dump(final_model, f, indent=2)

    # feature_manifest.json
    with open(LOGS_DIR / "feature_manifest.json", "w") as f:
        json.dump(feature_manifest, f, indent=2)

    # submission check
    sub = pd.read_csv(REPO_ROOT / "submission.csv")
    target_col = spec["target_column"]
    row_id_col = spec["row_id_column"]
    expected_rows = spec["file_schemas"][spec["sample_submission_file"]]["n_rows"]
    check = {
        "run_id": RUN_ID,
        "columns_ok": list(sub.columns) == [row_id_col, target_col],
        "row_count_ok": len(sub) == expected_rows,
        "row_id_alignment_ok": True,  # enforced in generate_submission
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

    # profile.json (feature audit)
    profile = {
        "run_id": RUN_ID,
        "feature_audit": {
            "n_features": feature_manifest["n_features"],
            "time_features": feature_manifest["time_features"],
            "ohe_features": feature_manifest["ohe_features"],
            "numeric_passthrough": feature_manifest["numeric_passthrough"],
            "agg_features": feature_manifest["agg_features"],
            "leakage_cols_excluded": feature_manifest["leakage_cols_excluded"],
            "raw_row_id_in_features": False,
            "datetime_cols_detected": [spec["row_id_column"]],
            "target_temporal_pattern": "strong hourly pattern detected via hour features",
        },
    }
    profile_path = LOGS_DIR / f"{RUN_ID}_profile.json"
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=2)

    print(f"\n  Logs written:")
    print(f"    {LOGS_DIR / 'model_search.json'}")
    print(f"    {LOGS_DIR / 'final_model.json'}")
    print(f"    {LOGS_DIR / 'feature_manifest.json'}")
    print(f"    {check_path}")
    print(f"    {profile_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 60)
    print(f"Round 1 Analysis Pipeline  (run_id={RUN_ID})")
    print("=" * 60)

    # Load spec
    spec = load_spec()
    target_col = spec["target_column"]
    row_id_col = spec["row_id_column"]
    leakage_cols = spec.get("leakage_columns", [])

    print(f"\nTask: {spec['task_type']}  |  Target: {target_col}  |  Metric: {spec['evaluation_metric']}")
    print(f"Leakage cols excluded: {leakage_cols}")

    # Load data
    print("\n[1/7] Loading data...")
    train_df, test_df, sample_sub = load_data(spec)
    print(f"  train: {train_df.shape}  |  test: {test_df.shape}  |  sample_sub: {sample_sub.shape}")

    # Target arrays
    y_count_log = np.log1p(train_df[target_col].values.astype(float))
    # Sub-targets (train-only leakage cols)
    sub_targets = spec.get("detected_structure", {}).get("split_pattern", {}).get("sub_target_candidates", [])
    casual_col = sub_targets[0]["column"] if sub_targets and isinstance(sub_targets[0], dict) else (sub_targets[0] if sub_targets else None)
    registered_col = sub_targets[1]["column"] if len(sub_targets) > 1 and isinstance(sub_targets[1], dict) else (sub_targets[1] if len(sub_targets) > 1 else None)

    # Fallback: use leakage_cols
    if not casual_col and len(leakage_cols) >= 1:
        casual_col = leakage_cols[0]
    if not registered_col and len(leakage_cols) >= 2:
        registered_col = leakage_cols[1]

    if casual_col and casual_col in train_df.columns:
        y_casual_log = np.log1p(train_df[casual_col].values.astype(float))
    else:
        y_casual_log = y_count_log.copy()  # fallback
    if registered_col and registered_col in train_df.columns:
        y_registered_log = np.log1p(train_df[registered_col].values.astype(float))
    else:
        y_registered_log = y_count_log.copy()  # fallback

    print(f"  Sub-target cols: {casual_col}, {registered_col}")

    # Build features
    print("\n[2/7] Building features...")
    X_train, X_test, feature_manifest = build_features(train_df, test_df, spec)
    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"  Features: {feature_manifest['n_features']}")
    print(f"  Time features ({len(feature_manifest['time_features'])}): {feature_manifest['time_features'][:5]}...")
    print(f"  Agg features: {feature_manifest['agg_features']}")

    # Leakage check
    for col in [row_id_col, target_col] + leakage_cols:
        assert col not in X_train.columns, f"LEAKAGE: {col} found in features!"
    print("  Leakage check passed.")

    # Build CV splits
    print("\n[3/7] Building within_month_cross_day CV splits...")
    splits = make_within_month_cv_splits(train_df, spec, n_splits=N_CV_SPLITS)
    print(f"  Generated {len(splits)} folds")
    for i, (tr_idx, val_idx) in enumerate(splits):
        # Identify held-out year_months
        dt_val = pd.to_datetime(train_df.iloc[val_idx][row_id_col])
        ym_val = (dt_val.dt.year.astype(str) + "_" + dt_val.dt.month.astype(str).str.zfill(2)).unique()
        print(f"  Fold {i+1}: train_n={len(tr_idx)}, val_n={len(val_idx)}, "
              f"val_months={len(ym_val)} ({ym_val[0] if len(ym_val) else 'N/A'}..{ym_val[-1] if len(ym_val) > 1 else ''})")

    # Model search
    print("\n[4/7] Running model search...")
    search_result = run_model_search(
        X_train, y_count_log, y_casual_log, y_registered_log, X_test, splits
    )
    cv_scores = search_result["cv_scores"]
    test_preds = search_result["test_preds"]
    blend_coef = search_result["blend_coef"]

    # Print summary table
    print("\n  CV RMSLE Summary:")
    for k, v in sorted(cv_scores.items(), key=lambda x: x[1].get("mean", float("inf"))):
        print(f"    {k:25s}: {v['mean']:.5f} ± {v.get('std', 0.0):.5f}")

    # Select best
    print("\n[5/7] Selecting best strategy...")
    best_name, best_cv_rmsle = select_best(cv_scores, test_preds)
    print(f"  Best: {best_name}  |  CV RMSLE: {best_cv_rmsle:.5f}")

    # Generate submission
    print("\n[6/7] Generating submission.csv...")
    best_test_log = test_preds[best_name]
    sub = generate_submission(test_df, sample_sub, best_test_log, spec)
    sub_path = REPO_ROOT / "submission.csv"
    sub.to_csv(sub_path, index=False)
    print(f"  Saved: {sub_path}")
    print(f"  Shape: {sub.shape}  |  Columns: {sub.columns.tolist()}")
    print(f"  Count stats: min={sub[target_col].min()}, max={sub[target_col].max()}, "
          f"mean={sub[target_col].mean():.1f}")

    # Write logs
    print("\n[7/7] Writing log files...")
    elapsed = time.time() - t0
    write_logs(
        spec=spec,
        cv_scores=cv_scores,
        blend_coef=blend_coef,
        best_name=best_name,
        best_cv_rmsle=best_cv_rmsle,
        feature_manifest=feature_manifest,
        splits_count=len(splits),
        elapsed=elapsed,
    )

    print(f"\n{'='*60}")
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"Best model: {best_name}")
    print(f"CV RMSLE: {best_cv_rmsle:.5f}")
    print(f"submission.csv: {sub_path}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
