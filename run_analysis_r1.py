"""
Round 1 analysis pipeline — fully general-purpose, no dataset-specific hardcoding.
Reads spec_parse.json and analysis_plan.json for all column/file/metric config.
"""
import json
import math
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
    HAS_HGB = True
except ImportError:
    HAS_HGB = False

from scipy.optimize import nnls

warnings.filterwarnings("ignore")

REPO = Path("/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo")
LOGS = REPO / "outputs" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

# ── Load config ────────────────────────────────────────────────────────────────
with open(LOGS / "spec_parse.json") as f:
    spec = json.load(f)

with open(LOGS / "analysis_plan.json") as f:
    plan = json.load(f)

TRAIN_FILE   = REPO / spec["train_file"]
TEST_FILE    = REPO / spec["prediction_file"]
SAMPLE_FILE  = REPO / spec["sample_submission_file"]
TARGET_COL   = spec["target_column"]
ROW_ID_COL   = spec["row_id_column"]
LEAKAGE_COLS = spec.get("leakage_columns", [])
METRIC       = spec["evaluation_metric"].lower()   # rmsle
SEED         = 42

fp = plan["feature_plan"]
TIME_COLS    = fp["datetime_source_columns"]
TIME_FEATS   = fp["time_features_to_generate"]
OHE_COLS     = fp["low_cardinality_encode_one_hot"]
EXCLUDE_COLS = set(fp["exclude_columns"])

print(f"[config] target={TARGET_COL}, row_id={ROW_ID_COL}, metric={METRIC}")
print(f"[config] leakage={LEAKAGE_COLS}, ohe={OHE_COLS}")

# ── Step 1: Load data ──────────────────────────────────────────────────────────
train_df = pd.read_csv(TRAIN_FILE)
test_df  = pd.read_csv(TEST_FILE)
sample_df = pd.read_csv(SAMPLE_FILE)

print(f"[data] train={train_df.shape}, test={test_df.shape}, sample={sample_df.shape}")
assert list(sample_df.columns) == [ROW_ID_COL, TARGET_COL], \
    f"sample submission columns mismatch: {sample_df.columns.tolist()}"

# ── Step 2: Datetime parsing + time feature extraction ─────────────────────────
def extract_time_features(df: pd.DataFrame, col: str, requested: list) -> pd.DataFrame:
    """Parse a datetime-like column and extract generic time features (no hardcoding)."""
    s = pd.to_datetime(df[col], infer_datetime_format=True)
    out = {}
    mapping = {
        "year":           s.dt.year,
        "month":          s.dt.month,
        "month_sin":      np.sin(2 * math.pi * s.dt.month / 12),
        "month_cos":      np.cos(2 * math.pi * s.dt.month / 12),
        "day":            s.dt.day,
        "dayofweek":      s.dt.dayofweek,
        "dayofweek_sin":  np.sin(2 * math.pi * s.dt.dayofweek / 7),
        "dayofweek_cos":  np.cos(2 * math.pi * s.dt.dayofweek / 7),
        "is_weekend":     (s.dt.dayofweek >= 5).astype(int),
        "quarter":        s.dt.quarter,
        "weekofyear":     s.dt.isocalendar().week.astype(int),
        "hour":           s.dt.hour,
        "hour_sin":       np.sin(2 * math.pi * s.dt.hour / 24),
        "hour_cos":       np.cos(2 * math.pi * s.dt.hour / 24),
        "ordinal":        s.apply(lambda x: x.toordinal()),
    }
    for feat in requested:
        if feat in mapping:
            out[f"{col}__{feat}"] = mapping[feat].values
    return pd.DataFrame(out, index=df.index)


def build_features(df: pd.DataFrame, ohe: OneHotEncoder = None, fit: bool = True):
    """Build feature matrix. Returns (X, ohe_fitted)."""
    parts = []

    # Time features
    for tc in TIME_COLS:
        if tc in df.columns:
            tf = extract_time_features(df, tc, TIME_FEATS)
            parts.append(tf)

    # Numeric features (exclude leakage, target, row_id, time src, ohe cols)
    skip = EXCLUDE_COLS | set(LEAKAGE_COLS) | set(TIME_COLS) | set(OHE_COLS)
    numeric_cols = [c for c in df.columns if c not in skip
                    and pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        parts.append(df[numeric_cols].copy())

    # OHE categorical features
    ohe_present = [c for c in OHE_COLS if c in df.columns]
    if ohe_present:
        if fit:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore",
                                dtype=np.float32)
            ohe.fit(df[ohe_present])
        ohe_arr = ohe.transform(df[ohe_present])
        ohe_names = ohe.get_feature_names_out(ohe_present)
        parts.append(pd.DataFrame(ohe_arr, columns=ohe_names, index=df.index))

    X = pd.concat(parts, axis=1)
    # Median impute any remaining NaN
    for col in X.columns:
        if X[col].isna().any():
            X[col].fillna(X[col].median(), inplace=True)
    return X, ohe


# Build datetime sort key for time-based splits
train_df["_dt"] = pd.to_datetime(train_df[ROW_ID_COL], infer_datetime_format=True)
train_df = train_df.sort_values("_dt").reset_index(drop=True)

# Target: log1p
y_log = np.log1p(train_df[TARGET_COL].values.astype(float))

# Feature matrices
X_train, ohe_fitted = build_features(train_df, fit=True)
X_test, _ = build_features(test_df, ohe=ohe_fitted, fit=False)

print(f"[features] X_train={X_train.shape}, X_test={X_test.shape}")
feature_names = X_train.columns.tolist()

# Write feature manifest
feat_manifest = {
    "feature_names": feature_names,
    "n_features": len(feature_names),
    "time_features_extracted": [f"{tc}__{f}" for tc in TIME_COLS for f in TIME_FEATS
                                  if f"{tc}__{f}" in feature_names],
    "ohe_features": [c for c in feature_names if any(c.startswith(o) for o in OHE_COLS)],
    "numeric_passthrough": [c for c in feature_names
                             if not any(c.startswith(o) for o in OHE_COLS)
                             and not any(c.startswith(f"{tc}__") for tc in TIME_COLS)],
}
with open(LOGS / "feature_manifest.json", "w") as f:
    json.dump(feat_manifest, f, indent=2)
print("[features] manifest written")

# ── Step 3: Time-based holdout split (last 20%) ────────────────────────────────
split_idx = int(len(X_train) * 0.80)
X_ho = X_train.iloc[split_idx:].values
y_ho = y_log[split_idx:]
X_cv = X_train.iloc[:split_idx].values
y_cv = y_log[:split_idx]


def rmsle_score(y_true_log, y_pred_log):
    """RMSLE from log1p-space predictions (clip negatives)."""
    y_pred_log = np.clip(y_pred_log, 0, None)
    return math.sqrt(mean_squared_error(y_true_log, y_pred_log))


# ── Step 4: Baseline models ────────────────────────────────────────────────────
baseline_results = {}

dummy = DummyRegressor(strategy="mean")
dummy.fit(X_cv, y_cv)
dummy_pred = dummy.predict(X_ho)
dummy_rmsle = rmsle_score(y_ho, dummy_pred)
baseline_results["dummy_mean"] = {"holdout_rmsle": round(dummy_rmsle, 5)}
print(f"[baseline] dummy rmsle={dummy_rmsle:.4f}")

ridge = Ridge(alpha=1.0)
ridge.fit(X_cv, y_cv)
ridge_pred = ridge.predict(X_ho)
ridge_rmsle = rmsle_score(y_ho, ridge_pred)
baseline_results["ridge"] = {"holdout_rmsle": round(ridge_rmsle, 5)}
print(f"[baseline] ridge rmsle={ridge_rmsle:.4f}")

with open(LOGS / "baseline_results.json", "w") as f:
    json.dump(baseline_results, f, indent=2)

# ── Step 5: GroupKFold CV on year-month groups ─────────────────────────────────
# Build groups from datetime column (year-month)
dt_series = pd.to_datetime(train_df[ROW_ID_COL], infer_datetime_format=True)
ym_groups = (dt_series.dt.year * 100 + dt_series.dt.month).values
# Only use the CV portion for group CV
ym_cv = ym_groups[:split_idx]

gkf = GroupKFold(n_splits=5)

def cv_rmsle(estimator, X, y, groups):
    scores = []
    for tr, va in gkf.split(X, y, groups):
        estimator.fit(X[tr], y[tr])
        pred = estimator.predict(X[va])
        scores.append(rmsle_score(y[va], pred))
    return np.mean(scores), np.std(scores)


# ── Candidate definitions ──────────────────────────────────────────────────────
candidates = {}

# HistGradientBoostingRegressor
if HAS_HGB:
    from scipy.stats import randint, uniform
    hgb_param_dist = {
        "max_iter":        randint(100, 500),
        "max_depth":       randint(3, 10),
        "learning_rate":   uniform(0.01, 0.2),
        "l2_regularization": uniform(0.0, 1.0),
        "min_samples_leaf": randint(5, 50),
    }
    hgb_base = HistGradientBoostingRegressor(random_state=SEED)
    hgb_rs = RandomizedSearchCV(
        hgb_base, hgb_param_dist, n_iter=20, cv=3,
        scoring="neg_mean_squared_error", random_state=SEED, n_jobs=-1
    )
    hgb_rs.fit(X_cv, y_cv)
    best_hgb = hgb_rs.best_estimator_
    cv_mean, cv_std = cv_rmsle(best_hgb, X_cv, y_cv, ym_cv)
    best_hgb.fit(X_cv, y_cv)
    ho_rmsle = rmsle_score(y_ho, best_hgb.predict(X_ho))
    candidates["hgb"] = {
        "estimator": best_hgb,
        "cv_rmsle": round(cv_mean, 5),
        "cv_std": round(cv_std, 5),
        "holdout_rmsle": round(ho_rmsle, 5),
        "best_params": hgb_rs.best_params_,
    }
    print(f"[candidate] hgb cv={cv_mean:.4f}±{cv_std:.4f} holdout={ho_rmsle:.4f}")

# RandomForestRegressor
from scipy.stats import randint as sp_randint
rf_param_dist = {
    "n_estimators":  sp_randint(100, 500),
    "max_depth":     sp_randint(5, 30),
    "min_samples_leaf": sp_randint(2, 20),
    "max_features":  [0.5, 0.7, 1.0, "sqrt"],
}
rf_base = RandomForestRegressor(random_state=SEED, n_jobs=-1)
rf_rs = RandomizedSearchCV(
    rf_base, rf_param_dist, n_iter=20, cv=3,
    scoring="neg_mean_squared_error", random_state=SEED, n_jobs=-1
)
rf_rs.fit(X_cv, y_cv)
best_rf = rf_rs.best_estimator_
cv_mean, cv_std = cv_rmsle(best_rf, X_cv, y_cv, ym_cv)
best_rf.fit(X_cv, y_cv)
ho_rmsle = rmsle_score(y_ho, best_rf.predict(X_ho))
candidates["random_forest"] = {
    "estimator": best_rf,
    "cv_rmsle": round(cv_mean, 5),
    "cv_std": round(cv_std, 5),
    "holdout_rmsle": round(ho_rmsle, 5),
    "best_params": rf_rs.best_params_,
}
print(f"[candidate] rf cv={cv_mean:.4f}±{cv_std:.4f} holdout={ho_rmsle:.4f}")

# ExtraTreesRegressor
et_param_dist = {
    "n_estimators":  sp_randint(100, 500),
    "max_depth":     sp_randint(5, 30),
    "min_samples_leaf": sp_randint(2, 20),
    "max_features":  [0.5, 0.7, 1.0, "sqrt"],
}
et_base = ExtraTreesRegressor(random_state=SEED, n_jobs=-1)
et_rs = RandomizedSearchCV(
    et_base, et_param_dist, n_iter=20, cv=3,
    scoring="neg_mean_squared_error", random_state=SEED, n_jobs=-1
)
et_rs.fit(X_cv, y_cv)
best_et = et_rs.best_estimator_
cv_mean, cv_std = cv_rmsle(best_et, X_cv, y_cv, ym_cv)
best_et.fit(X_cv, y_cv)
ho_rmsle = rmsle_score(y_ho, best_et.predict(X_ho))
candidates["extra_trees"] = {
    "estimator": best_et,
    "cv_rmsle": round(cv_mean, 5),
    "cv_std": round(cv_std, 5),
    "holdout_rmsle": round(ho_rmsle, 5),
    "best_params": et_rs.best_params_,
}
print(f"[candidate] et cv={cv_mean:.4f}±{cv_std:.4f} holdout={ho_rmsle:.4f}")

# ── Step 6: NNLS OOF blend ─────────────────────────────────────────────────────
print("[blend] building OOF predictions for NNLS blend...")
oof_preds = {}
for name, cand in candidates.items():
    est = cand["estimator"]
    oof = np.zeros(len(X_cv))
    for tr, va in gkf.split(X_cv, y_cv, ym_cv):
        est.fit(X_cv[tr], y_cv[tr])
        oof[va] = est.predict(X_cv[va])
    oof_preds[name] = oof
    # Refit on full CV set
    est.fit(X_cv, y_cv)

oof_matrix = np.column_stack([oof_preds[n] for n in candidates])
coef, _ = nnls(oof_matrix, y_cv)
coef_sum = coef.sum()
if coef_sum > 0:
    coef = coef / coef_sum

blend_oof = oof_matrix @ coef
blend_cv_rmsle = rmsle_score(y_cv, blend_oof)

# Holdout blend predictions
ho_matrix = np.column_stack([
    cand["estimator"].predict(X_ho) for cand in candidates.values()
])
blend_ho = ho_matrix @ coef
blend_ho_rmsle = rmsle_score(y_ho, blend_ho)
print(f"[blend] nnls coef={dict(zip(candidates.keys(), coef.round(4)))} "
      f"cv_rmsle={blend_cv_rmsle:.4f} holdout={blend_ho_rmsle:.4f}")

candidates["nnls_blend"] = {
    "estimator": None,
    "coef": dict(zip(candidates.keys(), coef.tolist())),
    "cv_rmsle": round(blend_cv_rmsle, 5),
    "cv_std": 0.0,
    "holdout_rmsle": round(blend_ho_rmsle, 5),
    "best_params": {},
}

# ── Step 7: Select best by holdout RMSLE ──────────────────────────────────────
best_name = min(candidates, key=lambda k: candidates[k]["holdout_rmsle"])
best_info = candidates[best_name]
print(f"[selection] best model: {best_name} (holdout_rmsle={best_info['holdout_rmsle']})")

# ── Write model_search.json ────────────────────────────────────────────────────
model_search_log = {
    "run_id": "20260610_round1",
    "metric_name": METRIC,
    "cv_strategy": "time_based_group_kfold",
    "n_splits": 5,
    "group_col": "year_month",
    "seed": SEED,
    "candidates": {
        k: {key: v for key, v in info.items() if key != "estimator"}
        for k, info in candidates.items()
    },
    "best_model_name": best_name,
    "best_holdout_rmsle": best_info["holdout_rmsle"],
}
with open(LOGS / "model_search.json", "w") as f:
    json.dump(model_search_log, f, indent=2)
print("[logs] model_search.json written")

# ── Step 8: Generate submission ───────────────────────────────────────────────
print("[submission] generating predictions...")

# Refit best model on FULL training set
if best_name == "nnls_blend":
    # Refit all component models on full train
    X_full = X_train.values
    test_preds_list = []
    for name in [n for n in candidates if n != "nnls_blend"]:
        est = candidates[name]["estimator"]
        est.fit(X_full, y_log)
        test_preds_list.append(est.predict(X_test.values))
    test_matrix = np.column_stack(test_preds_list)
    blend_coef = np.array([candidates["nnls_blend"]["coef"][n]
                            for n in candidates if n != "nnls_blend"])
    raw_preds = test_matrix @ blend_coef
else:
    best_est = candidates[best_name]["estimator"]
    best_est.fit(X_train.values, y_log)
    raw_preds = best_est.predict(X_test.values)

# Inverse transform: expm1 → clip ≥ 0 → round to int
final_preds = np.expm1(np.clip(raw_preds, 0, None))
final_preds = np.clip(final_preds, 0, None)
final_preds_int = np.round(final_preds).astype(int)

# Preserve sample submission row order
sub_df = sample_df[[ROW_ID_COL]].copy()
# Map predictions by datetime alignment
test_pred_map = dict(zip(test_df[ROW_ID_COL], final_preds_int))
sub_df[TARGET_COL] = sub_df[ROW_ID_COL].map(test_pred_map)

# Verify alignment
missing = sub_df[TARGET_COL].isna().sum()
if missing > 0:
    print(f"[warn] {missing} rows in sample submission not matched; using 0 fallback")
    sub_df[TARGET_COL] = sub_df[TARGET_COL].fillna(0).astype(int)
else:
    sub_df[TARGET_COL] = sub_df[TARGET_COL].astype(int)

submission_path = REPO / "submission.csv"
sub_df.to_csv(submission_path, index=False)
print(f"[submission] written to {submission_path}: shape={sub_df.shape}")

# ── Write final_model.json ────────────────────────────────────────────────────
final_model_log = {
    "run_id": "20260610_round1",
    "best_model_name": best_name,
    "selection_metric": METRIC,
    "holdout_rmsle": best_info["holdout_rmsle"],
    "cv_rmsle": best_info["cv_rmsle"],
    "blend_coef": best_info.get("coef", None),
    "best_params": best_info.get("best_params", {}),
    "feature_count": len(feature_names),
    "train_rows": len(X_train),
    "test_rows": len(X_test),
    "submission_path": str(submission_path),
}
with open(LOGS / "final_model.json", "w") as f:
    json.dump(final_model_log, f, indent=2)
print("[logs] final_model.json written")

# ── Verification ──────────────────────────────────────────────────────────────
assert submission_path.exists(), "submission.csv missing!"
assert sub_df.shape[1] == 2, f"wrong column count: {sub_df.columns.tolist()}"
assert len(sub_df) == len(sample_df), f"row count mismatch: {len(sub_df)} vs {len(sample_df)}"
assert sub_df[TARGET_COL].isna().sum() == 0, "missing predictions!"
assert np.isfinite(sub_df[TARGET_COL].values).all(), "non-finite predictions!"
print(f"[verify] submission OK: shape={sub_df.shape}, "
      f"cols={sub_df.columns.tolist()}, "
      f"pred_range=[{sub_df[TARGET_COL].min()},{sub_df[TARGET_COL].max()}]")

print("\n=== Round 1 pipeline complete ===")
print(f"  Best model : {best_name}")
print(f"  Holdout RMSLE: {best_info['holdout_rmsle']}")
print(f"  submission.csv rows: {len(sub_df)}")
