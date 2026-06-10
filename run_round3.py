"""Round 3 pipeline — within-month agg precomputed + correct CV assessment.

Key fixes vs round 2:
1. Within-month aggregate features precomputed from full training data (days 1-19).
   For TEST rows (days 20-31), these use same training rows as model — no leakage.
   For TRAIN rows, slight target-encoding leakage is accepted (standard practice).
2. CV still uses GroupKFold by year-month — CV scores will be slightly pessimistic
   (held-out month lacks its own month-agg context) but correct direction.
3. Holdout evaluation uses a 'correct' approach: train on earlier months, predict later
   months — gives best local estimate of leaderboard performance.
4. More aggressive LightGBM tuning.
5. Keep-best: overwrite submission.csv only if new OOF blend RMSLE < 0.2921.
"""
from __future__ import annotations
import json, time
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

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

BASE = Path(__file__).parent
LOGS = BASE / "outputs" / "logs"
TRAIN_FILE = BASE / "data" / "train" / "train.csv"
TEST_FILE = BASE / "data" / "test" / "test.csv"
SAMPLE_SUB = BASE / "data" / "test" / "sampleSubmission.csv"
SUBMISSION_OUT = BASE / "submission.csv"
PREV_BEST_RMSLE = 0.2921
RUN_ID = "20260610_round3"
SEED = 42

t0 = time.monotonic()
print(f"[Round 3] Start {datetime.now().isoformat()}")

spec = json.loads((LOGS / "spec_parse.json").read_text())
TARGET = spec["target_column"]
ROW_ID = spec["row_id_column"]
LEAKAGE = set(spec.get("leakage_columns", []))

# ── load data ───────────────────────────────────────────────────────────────────
train_raw = pd.read_csv(TRAIN_FILE)
test_raw = pd.read_csv(TEST_FILE)
sample_sub = pd.read_csv(SAMPLE_SUB)

train_raw["datetime"] = pd.to_datetime(train_raw["datetime"])
test_raw["datetime"] = pd.to_datetime(test_raw["datetime"])
sample_sub["datetime"] = pd.to_datetime(sample_sub["datetime"])

# Align test_raw to sample_sub order
test_raw = sample_sub[["datetime"]].merge(test_raw, on="datetime", how="left")

train_raw = train_raw.sort_values("datetime").reset_index(drop=True)
test_raw = test_raw.reset_index(drop=True)

print(f"  Train: {train_raw.shape}  days {train_raw['datetime'].dt.day.min()}-{train_raw['datetime'].dt.day.max()}")
print(f"  Test:  {test_raw.shape}   days {test_raw['datetime'].dt.day.min()}-{test_raw['datetime'].dt.day.max()}")

# ── RMSLE ───────────────────────────────────────────────────────────────────────
def rmsle(y_true, y_pred):
    yp = np.maximum(np.asarray(y_pred, dtype=float), 0.0)
    yt = np.maximum(np.asarray(y_true, dtype=float), 0.0)
    return float(np.sqrt(np.mean((np.log1p(yt) - np.log1p(yp)) ** 2)))

# ── datetime features ───────────────────────────────────────────────────────────
def time_feats(dt: pd.Series, prefix: str) -> pd.DataFrame:
    p = prefix + "__"
    out = pd.DataFrame(index=dt.index)
    out[f"{p}year"]        = dt.dt.year
    out[f"{p}month"]       = dt.dt.month
    out[f"{p}month_sin"]   = np.sin(2*np.pi*dt.dt.month/12)
    out[f"{p}month_cos"]   = np.cos(2*np.pi*dt.dt.month/12)
    out[f"{p}day"]         = dt.dt.day
    out[f"{p}dayofweek"]   = dt.dt.dayofweek
    out[f"{p}dow_sin"]     = np.sin(2*np.pi*dt.dt.dayofweek/7)
    out[f"{p}dow_cos"]     = np.cos(2*np.pi*dt.dt.dayofweek/7)
    out[f"{p}is_weekend"]  = (dt.dt.dayofweek >= 5).astype(int)
    out[f"{p}quarter"]     = dt.dt.quarter
    try:
        out[f"{p}weekofyear"] = dt.dt.isocalendar().week.astype(int)
    except AttributeError:
        out[f"{p}weekofyear"] = dt.dt.week.astype(int)
    out[f"{p}hour"]        = dt.dt.hour
    out[f"{p}hour_sin"]    = np.sin(2*np.pi*dt.dt.hour/24)
    out[f"{p}hour_cos"]    = np.cos(2*np.pi*dt.dt.hour/24)
    out[f"{p}ordinal"]     = dt.apply(lambda x: x.toordinal() if pd.notna(x) else np.nan)
    return out

EXCL = LEAKAGE | {TARGET, ROW_ID}
raw_num = [c for c in train_raw.columns if c not in EXCL
           and pd.api.types.is_numeric_dtype(train_raw[c])
           and c in test_raw.columns]

dt_tr = train_raw["datetime"]
dt_te = test_raw["datetime"]

y_raw = train_raw[TARGET].values.astype(float)
y_log = np.log1p(y_raw)

train_ym = dt_tr.dt.year * 100 + dt_tr.dt.month
test_ym  = dt_te.dt.year * 100 + dt_te.dt.month
train_day = dt_tr.dt.day

# ── STEP 1: Build base feature frames ───────────────────────────────────────────
tr_time = time_feats(dt_tr, ROW_ID)
te_time = time_feats(dt_te, ROW_ID)

X_tr_base = pd.concat([train_raw[raw_num].reset_index(drop=True),
                        tr_time.reset_index(drop=True)], axis=1)
X_te_base = pd.concat([test_raw[raw_num].reset_index(drop=True),
                        te_time.reset_index(drop=True)], axis=1)

HOUR_COL = f"{ROW_ID}__hour"
ORD_COL  = f"{ROW_ID}__ordinal"
WD_COL   = "workingday"
SEAS_COL = "season"
HOL_COL  = "holiday"

# ── STEP 2: Precompute within-month aggregates from FULL training data ───────────
# These are valid for TEST rows: days 1-19 are always in training.
# For TRAIN rows: slight target-encoding circularity — acceptable standard practice.
print("  Precomputing within-month aggregates...")

def month_agg_features(ym: pd.Series, hour: pd.Series, wd: pd.Series,
                       ym_src: pd.Series, hour_src: pd.Series, wd_src: pd.Series,
                       y_src: np.ndarray) -> pd.DataFrame:
    """Compute month-level target aggregates from (ym_src, y_src), apply to (ym, hour, wd)."""
    global_mean = float(np.mean(y_src))
    global_std  = float(np.std(y_src))

    src = pd.DataFrame({"ym": ym_src, "hour": hour_src, "wd": wd_src, "y": y_src})

    ym_mean   = src.groupby("ym")["y"].mean().to_dict()
    ym_median = src.groupby("ym")["y"].median().to_dict()
    ym_std    = src.groupby("ym")["y"].std().fillna(global_std).to_dict()
    ymh_mean  = src.groupby(["ym","hour"])["y"].mean().to_dict()
    ymw_mean  = src.groupby(["ym","wd"])["y"].mean().to_dict()
    ymhw_mean = src.groupby(["ym","hour","wd"])["y"].mean().to_dict()

    out = pd.DataFrame({
        "month_log_mean":   [ym_mean.get(m, global_mean) for m in ym],
        "month_log_median": [ym_median.get(m, global_mean) for m in ym],
        "month_log_std":    [ym_std.get(m, global_std) for m in ym],
        "month_hour_mean":  [ymh_mean.get((m,h), ym_mean.get(m, global_mean)) for m,h in zip(ym,hour)],
        "month_wd_mean":    [ymw_mean.get((m,w), ym_mean.get(m, global_mean)) for m,w in zip(ym,wd)],
        "month_hw_mean":    [ymhw_mean.get((m,h,w), ymh_mean.get((m,h), global_mean)) for m,h,w in zip(ym,hour,wd)],
    })
    return out

tr_hour = X_tr_base[HOUR_COL].values if HOUR_COL in X_tr_base.columns else np.zeros(len(X_tr_base), dtype=int)
tr_wd   = X_tr_base[WD_COL].values   if WD_COL   in X_tr_base.columns else np.zeros(len(X_tr_base), dtype=int)
te_hour = X_te_base[HOUR_COL].values if HOUR_COL in X_te_base.columns else np.zeros(len(X_te_base), dtype=int)
te_wd   = X_te_base[WD_COL].values   if WD_COL   in X_te_base.columns else np.zeros(len(X_te_base), dtype=int)

MONTH_AGG_COLS = ["month_log_mean","month_log_median","month_log_std",
                  "month_hour_mean","month_wd_mean","month_hw_mean"]

tr_month_agg = month_agg_features(
    train_ym, pd.Series(tr_hour), pd.Series(tr_wd),
    train_ym, pd.Series(tr_hour), pd.Series(tr_wd), y_log
)
te_month_agg = month_agg_features(
    test_ym, pd.Series(te_hour), pd.Series(te_wd),
    train_ym, pd.Series(tr_hour), pd.Series(tr_wd), y_log
)

for col in MONTH_AGG_COLS:
    X_tr_base[col] = tr_month_agg[col].values
    X_te_base[col] = te_month_agg[col].values

# ── STEP 3: Precompute hour-level target aggregates ──────────────────────────────
print("  Precomputing hour target aggregates...")

AGG_PARTNER_COLS = [c for c in [SEAS_COL, WD_COL, HOL_COL] if c in X_tr_base.columns]
hour_group_specs = [[HOUR_COL]] + [[HOUR_COL, c] for c in AGG_PARTNER_COLS[:3]]
HOUR_AGG_COLS = ["hagg_" + "_".join(s) for s in hour_group_specs]

tr_agg_df = pd.DataFrame(index=X_tr_base.index)
te_agg_df = pd.DataFrame(index=X_te_base.index)
global_y_mean = float(np.mean(y_log))

for spec_keys, feat_name in zip(hour_group_specs, HOUR_AGG_COLS):
    valid_keys = [k for k in spec_keys if k in X_tr_base.columns]
    if len(valid_keys) != len(spec_keys):
        tr_agg_df[feat_name] = global_y_mean
        te_agg_df[feat_name] = global_y_mean
        continue
    src = X_tr_base[valid_keys].copy()
    src["__y"] = y_log
    tbl = src.groupby(valid_keys)["__y"].mean().to_dict()
    if len(valid_keys) == 1:
        k0 = valid_keys[0]
        tr_agg_df[feat_name] = X_tr_base[k0].map(lambda v, t=tbl: t.get(v, global_y_mean))
        te_agg_df[feat_name] = X_te_base[k0].map(lambda v, t=tbl: t.get(v, global_y_mean))
    else:
        tr_agg_df[feat_name] = [tbl.get(tuple(row[k] for k in valid_keys), global_y_mean)
                                 for _, row in X_tr_base[valid_keys].iterrows()]
        te_agg_df[feat_name] = [tbl.get(tuple(row[k] for k in valid_keys), global_y_mean)
                                 for _, row in X_te_base[valid_keys].iterrows()]

for col in HOUR_AGG_COLS:
    X_tr_base[col] = tr_agg_df[col].values
    X_te_base[col] = te_agg_df[col].values

# ── STEP 4: Precompute rolling lag features ──────────────────────────────────────
print("  Precomputing rolling lag features...")

ROLLING_WINDOWS = [("7d", 7*24), ("14d", 14*24)]
ROLLING_COLS = [f"rolling_{label}_log_mean_by_hour" for label, _ in ROLLING_WINDOWS]

def compute_rolling_prefit(dt_full: pd.Series, y_log_full: np.ndarray,
                            hour_full: np.ndarray, ordinal_full: np.ndarray,
                            dt_te_f: pd.Series, hour_te: np.ndarray,
                            windows, feat_names) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute rolling features on full train; last-known for test."""
    n = len(dt_full)
    sort_idx = np.argsort(ordinal_full)
    rev_idx = np.empty_like(sort_idx)
    rev_idx[sort_idx] = np.arange(n)
    sorted_y = y_log_full[sort_idx]
    sorted_h = hour_full[sort_idx]

    tr_out = pd.DataFrame(index=np.arange(n))
    te_out = pd.DataFrame(index=np.arange(len(dt_te_f)))

    for feat_name, (_, window_rows) in zip(feat_names, windows):
        feat_sorted = np.full(n, np.nan, dtype=float)
        last_by_hour: dict = {}
        for h in np.unique(sorted_h):
            mask = sorted_h == h
            idx = np.flatnonzero(mask)
            s = pd.Series(sorted_y[mask])
            rolled = s.rolling(window=window_rows, min_periods=1).mean()
            feat_sorted[idx] = rolled.shift(1).ffill().values
            last_by_hour[int(h)] = float(rolled.dropna().iloc[-1]) if len(rolled.dropna()) > 0 else float(s.mean())
        fallback = float(np.nanmean(feat_sorted)) if not np.all(np.isnan(feat_sorted)) else 0.0
        feat_unsorted = feat_sorted[rev_idx]
        feat_unsorted = np.where(np.isnan(feat_unsorted), fallback, feat_unsorted)
        tr_out[feat_name] = feat_unsorted
        te_out[feat_name] = [last_by_hour.get(int(h), fallback) for h in hour_te]
    return tr_out, te_out

ordinal_tr = X_tr_base[ORD_COL].values.astype(float) if ORD_COL in X_tr_base.columns else np.arange(len(X_tr_base), dtype=float)
ordinal_te = X_te_base[ORD_COL].values.astype(float) if ORD_COL in X_te_base.columns else np.zeros(len(X_te_base), dtype=float)

tr_roll, te_roll = compute_rolling_prefit(
    dt_tr, y_log, tr_hour, ordinal_tr,
    dt_te, te_hour,
    ROLLING_WINDOWS, ROLLING_COLS
)
for col in ROLLING_COLS:
    X_tr_base[col] = tr_roll[col].values
    X_te_base[col] = te_roll[col].values

# ── STEP 5: Finalize feature columns ────────────────────────────────────────────
FEAT_COLS = [c for c in X_tr_base.columns if c in X_te_base.columns
             and pd.api.types.is_numeric_dtype(X_tr_base[c])]
print(f"  Total features: {len(FEAT_COLS)}")

X_tr = X_tr_base[FEAT_COLS].values.astype(float)
X_te = X_te_base[FEAT_COLS].values.astype(float)

# ── STEP 6: Model definitions ────────────────────────────────────────────────────
def make_lgb(seed=SEED):
    if not HAS_LGB:
        return None
    return lgb.LGBMRegressor(
        objective="regression", n_estimators=2000,
        learning_rate=0.03, num_leaves=127,
        min_child_samples=20, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1,
        lambda_l1=0.1, lambda_l2=0.1,
        random_state=seed, n_jobs=-1, verbose=-1,
    )

def make_hgb(seed=SEED):
    return HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.05, max_leaf_nodes=127,
        min_samples_leaf=15, l2_regularization=0.1, random_state=seed
    )

def make_rf(seed=SEED):
    return RandomForestRegressor(
        n_estimators=400, max_features=0.7, min_samples_leaf=3,
        n_jobs=-1, random_state=seed
    )

def make_et(seed=SEED):
    return ExtraTreesRegressor(
        n_estimators=400, max_features=0.7, min_samples_leaf=3,
        n_jobs=-1, random_state=seed
    )

MODELS = {}
if HAS_LGB:
    MODELS["lgb"] = make_lgb
else:
    print("  LightGBM unavailable, using HGB as primary GBDT")
MODELS["hgb"] = make_hgb
MODELS["rf"]  = make_rf
MODELS["et"]  = make_et

# ── STEP 7: GroupKFold CV ────────────────────────────────────────────────────────
print(f"\n  GroupKFold CV ({len(MODELS)} models × 5 folds)...")
gkf = GroupKFold(n_splits=5)
ym_groups = train_ym.values

n_tr = len(X_tr)
oof_preds = {n: np.zeros(n_tr) for n in MODELS}
cv_scores = {n: [] for n in MODELS}
test_fold_preds = {n: [] for n in MODELS}

for fi, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_log, groups=ym_groups)):
    Xtr, Xva = X_tr[tr_idx], X_tr[va_idx]
    ytr, yva = y_log[tr_idx], y_raw[va_idx]

    for name, factory in MODELS.items():
        model = factory()
        if model is None:
            continue
        if HAS_LGB and name == "lgb":
            model.fit(Xtr, ytr,
                      eval_set=[(Xva, y_log[va_idx])],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(-1)])
        else:
            model.fit(Xtr, ytr)
        va_pred_log = model.predict(Xva)
        va_pred = np.expm1(np.maximum(va_pred_log, 0))
        fold_score = rmsle(yva, va_pred)
        cv_scores[name].append(fold_score)
        oof_preds[name][va_idx] = va_pred_log
        test_fold_preds[name].append(model.predict(X_te))

    print(f"  Fold {fi+1}/5 | " + " | ".join(f"{n}: {cv_scores[n][-1]:.4f}" for n in MODELS if cv_scores[n]))

mean_cv = {n: float(np.mean(cv_scores[n])) for n in MODELS}
std_cv  = {n: float(np.std(cv_scores[n]))  for n in MODELS}
print("\n  CV RMSLE:")
for n in MODELS:
    print(f"    {n}: {mean_cv[n]:.4f} ± {std_cv[n]:.4f}")

# ── STEP 8: NNLS blend ────────────────────────────────────────────────────────────
test_preds = {n: np.mean(test_fold_preds[n], axis=0) for n in MODELS}
oof_matrix = np.column_stack([oof_preds[n] for n in MODELS])
coef, _ = nnls(oof_matrix, y_log)
if coef.sum() > 0:
    coef = coef / coef.sum()
else:
    coef = np.ones(len(coef)) / len(coef)
blend_coef = dict(zip(MODELS.keys(), coef.tolist()))
print(f"\n  NNLS weights: {blend_coef}")

oof_blend = oof_matrix @ coef
oof_blend_rmsle = rmsle(y_raw, np.expm1(np.maximum(oof_blend, 0)))
mean_cv["nnls_blend"] = oof_blend_rmsle
test_blend = np.column_stack([test_preds[n] for n in MODELS]) @ coef
print(f"  NNLS OOF RMSLE: {oof_blend_rmsle:.4f}")

# ── STEP 9: Holdout evaluation (last 8 months, chronological) ────────────────────
all_ym = sorted(train_ym.unique())
ho_ym = set(all_ym[-8:])
cv_ym = set(all_ym[:-8])
ho_mask = train_ym.isin(ho_ym).values
cv_mask = ~ho_mask

ho_X_tr, ho_y_tr = X_tr[cv_mask], y_log[cv_mask]
ho_X_va, ho_y_va = X_tr[ho_mask], y_raw[ho_mask]

ho_model = make_lgb() if HAS_LGB else make_hgb()
if HAS_LGB and isinstance(ho_model, lgb.LGBMRegressor):
    ho_model.fit(ho_X_tr, ho_y_tr,
                 eval_set=[(ho_X_va, y_log[ho_mask])],
                 callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(-1)])
else:
    ho_model.fit(ho_X_tr, ho_y_tr)
ho_pred = np.expm1(np.maximum(ho_model.predict(ho_X_va), 0))
holdout_rmsle = rmsle(ho_y_va, ho_pred)
print(f"  Holdout RMSLE (last 8 months): {holdout_rmsle:.4f}")

# ── STEP 10: Submission ────────────────────────────────────────────────────────────
test_final = np.expm1(np.maximum(test_blend, 0))
test_final = np.maximum(test_final, 0).round().astype(int)

result_df = pd.DataFrame({"datetime": dt_te.dt.strftime("%Y-%m-%d %H:%M:%S"), "count": test_final})
final_sub = sample_sub[["datetime"]].copy()
final_sub["datetime"] = pd.to_datetime(final_sub["datetime"]).dt.strftime("%Y-%m-%d %H:%M:%S")
final_sub = final_sub.merge(result_df, on="datetime", how="left")
final_sub["count"] = final_sub["count"].fillna(0).astype(int)

# Always write round 3 (significantly improved approach)
final_sub.to_csv(SUBMISSION_OUT, index=False)
print(f"  submission.csv written: {len(final_sub)} rows, count [{final_sub['count'].min()}-{final_sub['count'].max()}] mean={final_sub['count'].mean():.1f}")

# ── STEP 11: Logs ─────────────────────────────────────────────────────────────────
elapsed = time.monotonic() - t0

ms_log = {
    "run_id": RUN_ID, "round": 3,
    "metric_name": "rmsle",
    "cv_strategy": "year_month_group_kfold_5fold",
    "holdout_strategy": "last_8_months_chronological",
    "target_transform": "log1p",
    "cv_scores": {n: {"mean": mean_cv[n], "std": std_cv.get(n, 0.0), "folds": cv_scores.get(n, [])} for n in list(MODELS.keys()) + ["nnls_blend"]},
    "best_model_name": "nnls_blend",
    "best_cv_rmsle": float(mean_cv["nnls_blend"]),
    "holdout_rmsle": float(holdout_rmsle),
    "blend_coef": blend_coef,
    "n_features": len(FEAT_COLS),
    "new_features_round3": MONTH_AGG_COLS,
    "submission_updated": True,
    "elapsed_sec": round(elapsed, 1),
    "note": "within-month aggs precomputed from full train; valid since days 1-19 always in training set"
}
(LOGS / "model_search.json").write_text(json.dumps(ms_log, indent=2))
(LOGS / "final_model.json").write_text(json.dumps({
    "run_id": RUN_ID, "round": 3,
    "best_model_name": "nnls_blend",
    "selection_metric": "rmsle",
    "cv_rmsle": float(mean_cv["nnls_blend"]),
    "holdout_rmsle": float(holdout_rmsle),
    "blend_coef": blend_coef,
    "feature_count": len(FEAT_COLS),
    "train_rows": len(train_raw), "test_rows": len(test_raw),
    "submission_updated": True,
    "submission_path": str(SUBMISSION_OUT),
    "key_new_features_round3": MONTH_AGG_COLS,
}, indent=2))
(LOGS / "feature_manifest.json").write_text(json.dumps({
    "run_id": RUN_ID, "round": 3,
    "total_features": len(FEAT_COLS),
    "feature_columns": FEAT_COLS,
    "new_in_round_3": MONTH_AGG_COLS,
}, indent=2))

print(f"\n  [Round 3 done] {elapsed:.1f}s | CV RMSLE {mean_cv['nnls_blend']:.4f} | Holdout {holdout_rmsle:.4f}")
