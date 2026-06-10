"""Round 2 pipeline: adds hour-level target-aggregate features + rolling-mean lag features.

High-priority suggestions from analysis_review_1.json:
1. Hour-level target-aggregate features (mean log-count by hour, hour+workingday, hour+season)
   computed inside each CV fold to prevent leakage.
2. 7-day and 30-day rolling mean of log-count per hour group, shifted 1 position.

Key design choices matching round 1:
- Target transformed to log1p(count) for training (RMSLE optimization).
- Models predict in log space; predictions expm1'd back.
- Same GroupKFold (n_splits=5, year-month groups, seed=42).

Keep-best: overwrites submission.csv only if new holdout RMSLE < 0.2996 (current best).
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = Path("/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo")
LOGS = BASE / "outputs" / "logs"
TRAIN_FILE = BASE / "data" / "train" / "train.csv"
TEST_FILE = BASE / "data" / "test" / "test.csv"
SAMPLE_SUB = BASE / "data" / "test" / "sampleSubmission.csv"
SUBMISSION_OUT = BASE / "submission.csv"
PREV_BEST_RMSLE = 0.29963
RUN_ID = "20260610_round2"

t0 = time.monotonic()
print(f"[Round 2] Starting at {datetime.now().isoformat()}")

# ── load config ────────────────────────────────────────────────────────────────
spec = json.loads((LOGS / "spec_parse.json").read_text())
TARGET = spec["target_column"]          # "count"
ROW_ID = spec["row_id_column"]          # "datetime"
LEAKAGE = spec.get("leakage_columns", [])   # ["casual", "registered"]
print(f"[Round 2] Target={TARGET!r} row_id={ROW_ID!r} leakage={LEAKAGE}")

# ── load data ──────────────────────────────────────────────────────────────────
train_raw = pd.read_csv(TRAIN_FILE)
test_raw = pd.read_csv(TEST_FILE)
sample_sub = pd.read_csv(SAMPLE_SUB)

# Sort train chronologically
dt_train = pd.to_datetime(train_raw[ROW_ID], errors="coerce")
sort_order = dt_train.argsort().values
train_raw = train_raw.iloc[sort_order].reset_index(drop=True)
dt_train = pd.to_datetime(train_raw[ROW_ID], errors="coerce")

dt_test = pd.to_datetime(test_raw[ROW_ID], errors="coerce")
sort_order_test = dt_test.argsort().values
test_raw = test_raw.iloc[sort_order_test].reset_index(drop=True)
dt_test = pd.to_datetime(test_raw[ROW_ID], errors="coerce")

# ── datetime feature extraction ────────────────────────────────────────────────
def extract_time_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    dt = pd.to_datetime(df[col], errors="coerce")
    p = f"{col}__"
    out = pd.DataFrame(index=df.index)
    out[f"{p}year"] = dt.dt.year
    out[f"{p}month"] = dt.dt.month
    out[f"{p}month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
    out[f"{p}month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
    out[f"{p}day"] = dt.dt.day
    out[f"{p}dayofweek"] = dt.dt.dayofweek
    out[f"{p}dayofweek_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    out[f"{p}dayofweek_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    out[f"{p}is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    out[f"{p}quarter"] = dt.dt.quarter
    try:
        out[f"{p}weekofyear"] = dt.dt.isocalendar().week.astype(int)
    except AttributeError:
        out[f"{p}weekofyear"] = dt.dt.week.astype(int)
    out[f"{p}hour"] = dt.dt.hour
    out[f"{p}hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    out[f"{p}hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    out[f"{p}ordinal"] = dt.apply(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    return out

# Build base feature matrices
EXCL = set(LEAKAGE) | {TARGET, ROW_ID}

# Low-cardinality cols for hour-agg grouping (season, workingday, etc.)
# Detected generically from both train and test
def _find_low_card_cols(df_tr, df_te, max_unique=6, exclude=None):
    exclude = exclude or set()
    result = []
    for c in df_tr.columns:
        if c not in df_te.columns or c in exclude:
            continue
        if not pd.api.types.is_numeric_dtype(df_tr[c]):
            continue
        nu = int(df_tr[c].nunique(dropna=True))
        if 2 <= nu <= max_unique:
            result.append(c)
    return result

low_card_raw = _find_low_card_cols(train_raw, test_raw, exclude=EXCL)
# Exclude leakage cols
agg_partner_cols = [c for c in low_card_raw if c not in EXCL][:3]
print(f"[Round 2] Hour-agg partner cols (raw): {agg_partner_cols}")

train_time_feats = extract_time_features(train_raw, ROW_ID)
test_time_feats = extract_time_features(test_raw, ROW_ID)

# Raw non-target numeric features
raw_num_cols = [c for c in train_raw.columns if c not in EXCL
                and pd.api.types.is_numeric_dtype(train_raw[c])
                and c in test_raw.columns]
raw_cat_cols = [c for c in train_raw.columns if c not in EXCL
                and not pd.api.types.is_numeric_dtype(train_raw[c])
                and c in test_raw.columns]

# Assemble base frames
train_base = pd.concat([
    train_raw[raw_num_cols + raw_cat_cols].reset_index(drop=True),
    train_time_feats.reset_index(drop=True),
], axis=1)
test_base = pd.concat([
    test_raw[raw_num_cols + raw_cat_cols].reset_index(drop=True),
    test_time_feats.reset_index(drop=True),
], axis=1)

HOUR_COL = f"{ROW_ID}__hour"    # "datetime__hour"
ORDINAL_COL = f"{ROW_ID}__ordinal"

# ── target (log1p transformed, matching round 1) ───────────────────────────────
y_raw = train_raw[TARGET].values.astype(float)
y_log = np.log1p(y_raw)   # train on log1p; predict → expm1

# ── RMSLE scorer (operating on original-space values) ──────────────────────────
def rmsle(y_true, y_pred):
    """Standard RMSLE: sqrt(mean((log1p(y_pred) - log1p(y_true))^2))."""
    yt = np.maximum(np.asarray(y_true, dtype=float), 0.0)
    yp = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    return float(np.sqrt(np.mean((np.log1p(yt) - np.log1p(yp)) ** 2)))

def rmsle_log_space(y_true_log, y_pred_log):
    """RMSLE when both inputs are already in log1p space."""
    return float(np.sqrt(np.mean((y_pred_log - y_true_log) ** 2)))

# ── CV setup (year-month GroupKFold matching round 1) ──────────────────────────
ym_groups = (dt_train.dt.year * 100 + dt_train.dt.month).values
N_SPLITS = 5
gkf = GroupKFold(n_splits=N_SPLITS)

# Round 1 used first 80% as CV portion and last 20% as holdout
# We replicate the same time holdout for fair comparison
split_idx = int(len(train_base) * 0.80)
y_cv_log = y_log[:split_idx]
y_ho_raw = y_raw[split_idx:]
y_ho_log = y_log[split_idx:]
ym_cv = ym_groups[:split_idx]

print(f"[Round 2] CV set: {split_idx} rows, Holdout: {len(train_base) - split_idx} rows")
print(f"[Round 2] GroupKFold: {N_SPLITS} splits over {pd.Series(ym_cv).nunique()} year-month groups")

# ── rolling-mean features (per-fold, leakage-safe) ─────────────────────────────
# Computed using log1p(count) as the rolling target for consistency with model target
ROLLING_WINDOWS = [("7d", 7 * 24), ("30d", 30 * 24)]
ROLLING_COLS = [f"rolling_{label}_log_mean_by_hour" for label, _ in ROLLING_WINDOWS]

def compute_rolling_features_for_split(
    X_tr: pd.DataFrame,
    y_tr_log: np.ndarray,
    X_va: pd.DataFrame,
    X_test_frame: pd.DataFrame,
    hour_col: str,
    ordinal_col: str,
    windows: list,
    feat_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-fold rolling-mean of log1p(count) by hour group.

    - Training rows: rolling(window).mean().shift(1) — no self-leakage.
    - Val/Test rows: last known rolling mean per hour from training.
    - Sorting by ordinal ensures chronological order within the fold.
    """
    X_tr_out = X_tr.copy()
    X_va_out = X_va.copy()
    X_test_out = X_test_frame.copy()

    n_tr = len(X_tr)
    # Build sorted working frame using original positions (not iloc index)
    pos = np.arange(n_tr)
    ordinals = X_tr[ordinal_col].values.astype(float) if ordinal_col in X_tr.columns else pos.astype(float)
    hours = X_tr[hour_col].values

    sort_idx = np.argsort(ordinals)  # sorted order (chronological)
    # Map sorted-position → original-position
    orig_pos_of_sorted = pos[sort_idx]

    sorted_y = y_tr_log[sort_idx]
    sorted_hours = hours[sort_idx]

    for feat_name, (label, window_rows) in zip(feat_names, windows):
        feat_sorted = np.full(n_tr, np.nan, dtype=float)
        last_rolling_by_hour: dict = {}

        # Per-hour group: compute rolling on sorted rows
        unique_hours = np.unique(sorted_hours)
        for h in unique_hours:
            h_mask = sorted_hours == h
            h_sorted_idx = np.flatnonzero(h_mask)
            s = pd.Series(sorted_y[h_mask])
            rolled = s.rolling(window=window_rows, min_periods=1).mean()
            rolled_shifted = rolled.shift(1).ffill()
            feat_sorted[h_sorted_idx] = rolled_shifted.values
            last_val = float(rolled.dropna().iloc[-1]) if len(rolled.dropna()) > 0 else float(s.mean())
            last_rolling_by_hour[h] = last_val

        global_fallback = float(np.nanmean(feat_sorted)) if not np.all(np.isnan(feat_sorted)) else 0.0

        # Map back to original (unsorted) positions
        result_orig = np.full(n_tr, global_fallback, dtype=float)
        for sorted_i, orig_p in enumerate(orig_pos_of_sorted):
            v = feat_sorted[sorted_i]
            result_orig[orig_p] = v if not np.isnan(v) else global_fallback
        X_tr_out[feat_name] = result_orig

        # Val and test: last known rolling mean per hour
        for out_f, in_f in [(X_va_out, X_va), (X_test_out, X_test_frame)]:
            if hour_col in in_f.columns:
                out_f[feat_name] = [
                    last_rolling_by_hour.get(h, global_fallback)
                    for h in in_f[hour_col].values
                ]
            else:
                out_f[feat_name] = global_fallback

    return X_tr_out, X_va_out, X_test_out


# ── hour-level target-aggregate features (per-fold, leakage-safe) ──────────────
# Groups: [hour], [hour, extra1], [hour, extra2] where extras are low-card raw cols
# These pairs need to be in the base frame (available in both train and test)
hour_agg_group_specs = [[HOUR_COL]] + [[HOUR_COL, c] for c in agg_partner_cols if c in train_base.columns and c in test_base.columns]
HOUR_AGG_COLS = ["hagg_" + "_".join(spec) for spec in hour_agg_group_specs]
print(f"[Round 2] Hour-agg features: {HOUR_AGG_COLS}")

def compute_hour_agg_for_split(
    X_tr: pd.DataFrame,
    y_tr_log: np.ndarray,
    X_va: pd.DataFrame,
    X_test_frame: pd.DataFrame,
    group_specs: list[list[str]],
    feat_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-fold hour-level target aggregate (mean log1p(count)) by group.

    Fit on training fold only; transform val/test via lookup with global fallback.
    """
    X_tr_out = X_tr.copy()
    X_va_out = X_va.copy()
    X_test_out = X_test_frame.copy()

    global_mean = float(np.mean(y_tr_log))

    for spec_keys, feat_name in zip(group_specs, feat_names):
        valid_keys = [k for k in spec_keys if k in X_tr.columns]
        if len(valid_keys) != len(spec_keys):
            for f in [X_tr_out, X_va_out, X_test_out]:
                f[feat_name] = global_mean
            continue

        df = X_tr[valid_keys].copy()
        df["__y"] = y_tr_log
        tbl = df.groupby(valid_keys, dropna=False)["__y"].mean()
        tbl_dict = {k: float(v) for k, v in tbl.items()}

        if len(valid_keys) == 1:
            k0 = valid_keys[0]
            X_tr_out[feat_name] = X_tr[k0].map(lambda v, t=tbl_dict, g=global_mean: t.get(v, g))
            X_va_out[feat_name] = X_va[k0].map(lambda v, t=tbl_dict, g=global_mean: t.get(v, g))
            if k0 in X_test_frame.columns:
                X_test_out[feat_name] = X_test_frame[k0].map(lambda v, t=tbl_dict, g=global_mean: t.get(v, g))
            else:
                X_test_out[feat_name] = global_mean
        else:
            def _lookup(row, keys, tbl, g):
                key = tuple(row[k] for k in keys)
                return tbl.get(key, g)
            X_tr_out[feat_name] = [_lookup(r, valid_keys, tbl_dict, global_mean) for _, r in X_tr[valid_keys].iterrows()]
            X_va_out[feat_name] = [_lookup(r, valid_keys, tbl_dict, global_mean) for _, r in X_va[valid_keys].iterrows()]
            test_keys_present = all(k in X_test_frame.columns for k in valid_keys)
            if test_keys_present:
                X_test_out[feat_name] = [_lookup(r, valid_keys, tbl_dict, global_mean) for _, r in X_test_frame[valid_keys].iterrows()]
            else:
                X_test_out[feat_name] = global_mean

    return X_tr_out, X_va_out, X_test_out


# ── initialize new feature columns in base frames ──────────────────────────────
for col in ROLLING_COLS + HOUR_AGG_COLS:
    if col not in train_base.columns:
        train_base[col] = np.nan
    if col not in test_base.columns:
        test_base[col] = np.nan

# ── all feature columns ────────────────────────────────────────────────────────
ALL_FEAT_COLS = [c for c in train_base.columns if c in test_base.columns]
NUM_FEAT_COLS = [c for c in ALL_FEAT_COLS if pd.api.types.is_numeric_dtype(train_base[c])]
CAT_FEAT_COLS = [c for c in ALL_FEAT_COLS if c not in NUM_FEAT_COLS]
print(f"[Round 2] Total features: {len(ALL_FEAT_COLS)} ({len(NUM_FEAT_COLS)} num + {len(CAT_FEAT_COLS)} cat)")

# ── split base frames AFTER new columns are initialized ────────────────────────
X_cv_base = train_base.iloc[:split_idx].copy()
X_ho_base = train_base.iloc[split_idx:].copy()

# ── preprocessor ────────────────────────────────────────────────────────────────
def make_ohe():
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

def make_preprocessor():
    transformers = []
    if NUM_FEAT_COLS:
        transformers.append(("num", SimpleImputer(strategy="median"), NUM_FEAT_COLS))
    if CAT_FEAT_COLS:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", make_ohe()),
            ]),
            CAT_FEAT_COLS,
        ))
    if not transformers:
        from sklearn.preprocessing import FunctionTransformer
        return FunctionTransformer()
    return ColumnTransformer(transformers, remainder="drop")


# ── apply all per-fold features ────────────────────────────────────────────────
def apply_fold_features(X_tr, y_tr_log, X_va, X_test_dummy):
    # Step 1: rolling
    X_tr, X_va, X_test_dummy = compute_rolling_features_for_split(
        X_tr, y_tr_log, X_va, X_test_dummy,
        HOUR_COL, ORDINAL_COL, ROLLING_WINDOWS, ROLLING_COLS
    )
    # Step 2: hour-agg
    X_tr, X_va, X_test_dummy = compute_hour_agg_for_split(
        X_tr, y_tr_log, X_va, X_test_dummy,
        hour_agg_group_specs, HOUR_AGG_COLS
    )
    return X_tr, X_va, X_test_dummy


# ── candidates ─────────────────────────────────────────────────────────────────
CANDIDATES = {
    "hgb": lambda: HistGradientBoostingRegressor(
        random_state=42, max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
        l2_regularization=0.1, min_samples_leaf=20,
    ),
    "random_forest": lambda: RandomForestRegressor(
        n_estimators=400, random_state=42, n_jobs=-1, min_samples_leaf=2, max_features="sqrt",
    ),
    "extra_trees": lambda: ExtraTreesRegressor(
        n_estimators=400, random_state=42, n_jobs=-1, min_samples_leaf=2, max_features="sqrt",
    ),
}

try:
    import lightgbm as lgb
    CANDIDATES["lightgbm"] = lambda: lgb.LGBMRegressor(
        n_estimators=800, learning_rate=0.04, num_leaves=63,
        subsample=0.85, colsample_bytree=0.85, min_child_samples=20,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, n_jobs=-1, verbose=-1,
    )
    print("[Round 2] LightGBM available")
except ImportError:
    print("[Round 2] LightGBM unavailable")

# Dummy test frame for fold feature computation (only used to pass structure)
X_test_dummy_global = test_base[ALL_FEAT_COLS].copy()

# ── OOF cross-validation on CV portion (first 80%) ────────────────────────────
X_cv_df = X_cv_base[ALL_FEAT_COLS].copy()
y_cv_log_arr = y_cv_log.copy()
y_cv_raw = y_raw[:split_idx]

folds_cv = list(gkf.split(np.arange(len(X_cv_df)), groups=ym_cv))
print(f"[Round 2] CV folds: {N_SPLITS}, CV rows: {len(X_cv_df)}")

oof_log_preds: dict[str, np.ndarray] = {}
cand_cv_scores: dict[str, float] = {}
cand_details: dict[str, dict] = {}

for name, factory in CANDIDATES.items():
    print(f"[Round 2] Training {name}...", end="", flush=True)
    oof_log = np.full(len(X_cv_df), np.nan, dtype=float)
    fold_rmsles_log = []

    for tr_idx, va_idx in folds_cv:
        X_tr_fold = X_cv_df.iloc[tr_idx].copy()
        X_va_fold = X_cv_df.iloc[va_idx].copy()
        y_tr_log_f = y_cv_log_arr[tr_idx]
        y_va_log_f = y_cv_log_arr[va_idx]

        # Compute rolling + hour-agg features for this fold
        X_tr_aug, X_va_aug, _ = apply_fold_features(
            X_tr_fold, y_tr_log_f, X_va_fold, X_test_dummy_global.copy()
        )

        # Preprocess
        prep = make_preprocessor()
        X_tr_pp = prep.fit_transform(X_tr_aug[ALL_FEAT_COLS])
        X_va_pp = prep.transform(X_va_aug[ALL_FEAT_COLS])

        est = factory()
        est.fit(X_tr_pp, y_tr_log_f)
        preds_log = np.maximum(est.predict(X_va_pp), 0.0)
        oof_log[va_idx] = preds_log
        fold_rmsles_log.append(rmsle_log_space(y_va_log_f, preds_log))

    mask = ~np.isnan(oof_log)
    cv_rmsle_log = rmsle_log_space(y_cv_log_arr[mask], oof_log[mask])
    cv_std = float(np.std(fold_rmsles_log))
    oof_log_preds[name] = oof_log
    cand_cv_scores[name] = cv_rmsle_log
    cand_details[name] = {
        "cv_rmsle": round(cv_rmsle_log, 5),
        "cv_std": round(cv_std, 5),
        "fold_rmsles": [round(f, 5) for f in fold_rmsles_log],
    }
    print(f" CV RMSLE={cv_rmsle_log:.5f} ± {cv_std:.5f}")


# ── NNLS blend ─────────────────────────────────────────────────────────────────
print("[Round 2] Computing NNLS blend...")
cand_names = list(oof_log_preds.keys())
mask_all = np.ones(len(y_cv_log_arr), dtype=bool)
for n in cand_names:
    mask_all &= ~np.isnan(oof_log_preds[n])

M = np.column_stack([oof_log_preds[n][mask_all] for n in cand_names])
y_masked = y_cv_log_arr[mask_all]
w, _ = nnls(M, y_masked)
if w.sum() > 0:
    w = w / w.sum()
else:
    w = np.ones(len(cand_names)) / len(cand_names)

blend_oof_log = M @ w
blend_rmsle_log = rmsle_log_space(y_masked, blend_oof_log)
print(f"[Round 2] NNLS blend CV RMSLE={blend_rmsle_log:.5f}, weights={dict(zip(cand_names, w.round(4)))}")
cand_cv_scores["nnls_blend"] = blend_rmsle_log
blend_coef = {n: float(wi) for n, wi in zip(cand_names, w)}

# ── select best model ──────────────────────────────────────────────────────────
best_name = min(cand_cv_scores, key=lambda n: cand_cv_scores[n])
best_cv = cand_cv_scores[best_name]
print(f"[Round 2] Best CV: {best_name} = {best_cv:.5f}")
print(f"[Round 2] Previous round 1 best CV: 0.35256 (nnls_blend)")


# ── holdout RMSLE (last 20% of training data) ──────────────────────────────────
def predict_holdout_log(name: str) -> np.ndarray:
    """Fit on CV portion (first 80%), predict holdout (last 20%), in log space."""
    X_tr_full = X_cv_base[ALL_FEAT_COLS].copy()
    X_va_full = X_ho_base[ALL_FEAT_COLS].copy()
    y_tr_full = y_cv_log

    X_tr_aug, X_va_aug, _ = apply_fold_features(
        X_tr_full, y_tr_full, X_va_full, X_test_dummy_global.copy()
    )
    prep = make_preprocessor()
    X_tr_pp = prep.fit_transform(X_tr_aug[ALL_FEAT_COLS])
    X_va_pp = prep.transform(X_va_aug[ALL_FEAT_COLS])

    if name == "nnls_blend":
        preds_log = np.zeros(len(X_va_pp), dtype=float)
        for cn, wi in blend_coef.items():
            if wi < 1e-6:
                continue
            est = CANDIDATES[cn]()
            est.fit(X_tr_pp, y_tr_full)
            preds_log += wi * np.maximum(est.predict(X_va_pp), 0.0)
    else:
        est = CANDIDATES[name]()
        est.fit(X_tr_pp, y_tr_full)
        preds_log = np.maximum(est.predict(X_va_pp), 0.0)

    return preds_log

print("[Round 2] Computing holdout RMSLE...")
ho_log_preds = predict_holdout_log(best_name)
# Holdout RMSLE in original space (using expm1 of log predictions)
ho_preds_orig = np.expm1(np.maximum(ho_log_preds, 0.0))
holdout_rmsle_r2 = rmsle(y_ho_raw, ho_preds_orig)
holdout_rmsle_logspace = rmsle_log_space(y_ho_log, ho_log_preds)
print(f"[Round 2] Holdout RMSLE (original space) = {holdout_rmsle_r2:.5f}  (prev best = {PREV_BEST_RMSLE:.5f})")
print(f"[Round 2] Holdout RMSLE (log space) = {holdout_rmsle_logspace:.5f}  (prev best = 0.29963)")

# ── keep-best rule ─────────────────────────────────────────────────────────────
# Compare in log-space since round 1 holdout was measured in log-space
keep_best = holdout_rmsle_logspace < PREV_BEST_RMSLE
print(f"[Round 2] Keep-best: {keep_best}")

# ── final predictions on test ──────────────────────────────────────────────────
print("[Round 2] Generating final test predictions...")

# Fit on FULL training data
X_tr_final = train_base[ALL_FEAT_COLS].copy()
X_te_final = test_base[ALL_FEAT_COLS].copy()
y_tr_final_log = y_log  # all training rows

X_tr_aug_final, _, X_te_aug_final = apply_fold_features(
    X_tr_final, y_tr_final_log, X_tr_final.copy(), X_te_final
)
prep_final = make_preprocessor()
X_tr_final_pp = prep_final.fit_transform(X_tr_aug_final[ALL_FEAT_COLS])
X_te_final_pp = prep_final.transform(X_te_aug_final[ALL_FEAT_COLS])

if best_name == "nnls_blend":
    test_log_preds = np.zeros(len(X_te_final_pp), dtype=float)
    for cn, wi in blend_coef.items():
        if wi < 1e-6:
            continue
        seed_preds_log = []
        for seed in [42, 142, 242]:
            est = CANDIDATES[cn]()
            try:
                est.set_params(random_state=seed)
            except Exception:
                pass
            est.fit(X_tr_final_pp, y_tr_final_log)
            seed_preds_log.append(np.maximum(est.predict(X_te_final_pp), 0.0))
        test_log_preds += wi * np.mean(seed_preds_log, axis=0)
else:
    seed_preds_log = []
    for seed in [42, 142, 242]:
        est = CANDIDATES[best_name]()
        try:
            est.set_params(random_state=seed)
        except Exception:
            pass
        est.fit(X_tr_final_pp, y_tr_final_log)
        seed_preds_log.append(np.maximum(est.predict(X_te_final_pp), 0.0))
    test_log_preds = np.mean(seed_preds_log, axis=0)

# Convert back to original space
test_preds_orig = np.expm1(np.maximum(test_log_preds, 0.0))
test_preds_orig = np.maximum(test_preds_orig, 0.0)

# ── rebuild submission in sample_submission order ──────────────────────────────
# test_raw is sorted by datetime; sample_sub may be in different order
test_dt_sorted = test_raw[ROW_ID].astype(str).values
test_dt_to_pred = dict(zip(test_dt_sorted, test_preds_orig))
sample_sub_order = sample_sub[ROW_ID].astype(str).tolist()
sub_preds = np.array([test_dt_to_pred.get(dt, np.nan) for dt in sample_sub_order], dtype=float)

if np.isnan(sub_preds).any():
    median_val = float(np.nanmedian(sub_preds))
    sub_preds = np.where(np.isnan(sub_preds), median_val, sub_preds)
sub_preds = np.round(sub_preds).astype(int)
sub_preds = np.maximum(sub_preds, 0)

sub_df = pd.DataFrame({ROW_ID: sample_sub_order, TARGET: sub_preds})
assert len(sub_df) == len(sample_sub), f"Row count mismatch: {len(sub_df)} vs {len(sample_sub)}"
assert sub_df.shape[1] == 2

if keep_best:
    sub_df.to_csv(SUBMISSION_OUT, index=False)
    print(f"[Round 2] submission.csv updated with {len(sub_df)} rows")
else:
    print(f"[Round 2] submission.csv NOT updated (round 2 log-space RMSLE {holdout_rmsle_logspace:.5f} >= prev {PREV_BEST_RMSLE:.5f})")

elapsed = time.monotonic() - t0
print(f"[Round 2] Elapsed: {elapsed:.1f}s")

# ── write logs ─────────────────────────────────────────────────────────────────
model_search_r2 = {
    "run_id": RUN_ID,
    "round": 2,
    "metric_name": "rmsle",
    "cv_strategy": "time_based_group_kfold",
    "n_splits": N_SPLITS,
    "group_col": "year_month",
    "seed": 42,
    "target_transform": "log1p",
    "new_features": {
        "hour_target_agg": HOUR_AGG_COLS,
        "rolling_lag": ROLLING_COLS,
    },
    "candidates": {
        **{n: {
            "cv_rmsle": cand_cv_scores[n],
            "cv_std": cand_details.get(n, {}).get("cv_std", 0.0),
            "fold_rmsles": cand_details.get(n, {}).get("fold_rmsles", []),
        } for n in CANDIDATES},
        "nnls_blend": {
            "coef": blend_coef,
            "cv_rmsle": blend_rmsle_log,
            "cv_std": 0.0,
        },
    },
    "best_model_name": best_name,
    "best_cv_rmsle": cand_cv_scores[best_name],
    "holdout_rmsle": holdout_rmsle_logspace,
    "holdout_rmsle_original_space": holdout_rmsle_r2,
    "prev_best_holdout_rmsle": PREV_BEST_RMSLE,
    "submission_updated": keep_best,
    "elapsed_sec": round(elapsed, 1),
}
(LOGS / "model_search.json").write_text(json.dumps(model_search_r2, indent=2))
print("[Round 2] Written model_search.json")

final_model_r2 = {
    "run_id": RUN_ID,
    "round": 2,
    "best_model_name": best_name,
    "selection_metric": "rmsle",
    "holdout_rmsle": holdout_rmsle_logspace,
    "holdout_rmsle_original_space": holdout_rmsle_r2,
    "cv_rmsle": cand_cv_scores[best_name],
    "blend_coef": blend_coef if best_name == "nnls_blend" else {},
    "feature_count": len(ALL_FEAT_COLS),
    "train_rows": len(train_base),
    "test_rows": len(test_base),
    "submission_updated": keep_best,
    "submission_path": str(SUBMISSION_OUT),
}
(LOGS / "final_model.json").write_text(json.dumps(final_model_r2, indent=2))
print("[Round 2] Written final_model.json")

feature_manifest = {
    "run_id": RUN_ID,
    "round": 2,
    "total_features": len(ALL_FEAT_COLS),
    "feature_columns": ALL_FEAT_COLS,
    "new_in_round_2": {
        "hour_target_agg_features": HOUR_AGG_COLS,
        "rolling_lag_features": ROLLING_COLS,
    },
    "numeric_features": NUM_FEAT_COLS,
    "categorical_features": CAT_FEAT_COLS,
}
(LOGS / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2))
print("[Round 2] Written feature_manifest.json")

analysis_review_2 = {
    "round": 2,
    "reviewer": "analysis-programmer-round2",
    "current_best_cv_score": round(cand_cv_scores[best_name], 5),
    "current_best_holdout_score": round(holdout_rmsle_logspace, 5),
    "prev_best_holdout_score": PREV_BEST_RMSLE,
    "metric": "rmsle",
    "target_transform": "log1p",
    "submission_updated": keep_best,
    "new_features_added": HOUR_AGG_COLS + ROLLING_COLS,
    "candidate_scores": {n: round(cand_cv_scores[n], 5) for n in cand_cv_scores},
    "round1_best_cv": 0.35256,
    "round1_best_holdout": 0.29963,
    "approved_for_final": keep_best,
    "summary": (
        f"Round 2: Added hour-level target-agg features {HOUR_AGG_COLS} "
        f"and rolling-mean features {ROLLING_COLS}. "
        f"Best CV RMSLE={cand_cv_scores[best_name]:.5f} (model={best_name}), "
        f"holdout RMSLE={holdout_rmsle_logspace:.5f} (log space). "
        f"Submission {'updated' if keep_best else 'NOT updated (no improvement)'}."
    ),
}
(LOGS / "analysis_review_2.json").write_text(json.dumps(analysis_review_2, indent=2))
print("[Round 2] Written analysis_review_2.json")
print("[Round 2] Done.")
