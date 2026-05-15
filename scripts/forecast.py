"""
General-purpose walk-forward panel forecasting pipeline.
Adapted from STAI-X model_v3.ipynb (Recursive GBDT approach).

Usage (env vars):
  TRAIN_PATH        path to training CSV (required)
  COV_TRAIN_PATH    path to train covariates CSV (optional)
  COV_VAL_PATH      path to val covariates CSV (optional)
  VAL_PATH          path to sample_submission CSV (required)
  TARGET_COL        target column name
  ENTITY_COL        entity column name
  PERIOD_COL        period column name
  OUTPUT_PATH       output path for submission.csv (default: submission.csv)
"""

import os, warnings, json
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_PATH     = os.environ.get('TRAIN_PATH',     'data/dose_sys_train.csv')
COV_TRAIN_PATH = os.environ.get('COV_TRAIN_PATH', 'data/covariates_train.csv')
COV_VAL_PATH   = os.environ.get('COV_VAL_PATH',   '')
VAL_PATH       = os.environ.get('VAL_PATH',       'data/sample_submission.csv')
TARGET_COL     = os.environ.get('TARGET_COL',     'rate_per_10000_ed_visits')
ENTITY_COL     = os.environ.get('ENTITY_COL',     'jurisdiction')
PERIOD_COL     = os.environ.get('PERIOD_COL',     'period_id')
OUTPUT_PATH    = os.environ.get('OUTPUT_PATH',    'submission.csv')

OUT_DIR = Path('outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print(f'[FORECAST] Loading {TRAIN_PATH}')
train = pd.read_csv(TRAIN_PATH)
print(f'  shape: {train.shape}')

print(f'[FORECAST] Loading {VAL_PATH}')
sub = pd.read_csv(VAL_PATH)
print(f'  submission shape: {sub.shape}  columns: {sub.columns.tolist()}')

# Load covariates
cov_dfs = []
for path, split in [(COV_TRAIN_PATH, 'train'), (COV_VAL_PATH, 'val')]:
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        df['split'] = split
        cov_dfs.append(df)
        print(f'  loaded covariates ({split}): {df.shape}')

cov_all = pd.concat(cov_dfs, ignore_index=True) if cov_dfs else pd.DataFrame()

# ── Detect category column ────────────────────────────────────────────────────
cat_cols_candidates = [c for c in train.columns
                       if c not in [TARGET_COL, ENTITY_COL, PERIOD_COL]
                       and train[c].dtype == object and train[c].nunique() <= 20]
CATEGORY_COL = cat_cols_candidates[0] if cat_cols_candidates else None
SCORE_CATS = sorted(train[CATEGORY_COL].unique()) if CATEGORY_COL else [None]
print(f'  category column: {CATEGORY_COL}  values: {SCORE_CATS}')

# ── Infer submission period IDs ───────────────────────────────────────────────
# sub may have PERIOD_COL already, or we infer from row structure
if PERIOD_COL in sub.columns:
    SCORE_PIDS = sub[PERIOD_COL].unique().tolist()
else:
    # Try to infer: submission rows not in training
    if PERIOD_COL in train.columns:
        train_pids = set(train[PERIOD_COL].unique())
        SCORE_PIDS = []  # will be inferred differently below
    SCORE_PIDS = []

TRAIN_PIDS = train[PERIOD_COL].unique().tolist() if PERIOD_COL in train.columns else []
print(f'  training periods: {len(TRAIN_PIDS)}  scoring periods: {len(SCORE_PIDS)}')

# ── Build period ordering (chronological) ─────────────────────────────────────
# If periods are sortable (dates/integers) use direct sort.
# If hashed strings, infer order from first appearance in training data.
all_pids = list(dict.fromkeys(
    train.sort_values(ENTITY_COL)[PERIOD_COL].tolist()
    if ENTITY_COL in train.columns else train[PERIOD_COL].tolist()
))
# Try numeric/date sort
try:
    all_pids_sorted = sorted(set(all_pids))
except TypeError:
    all_pids_sorted = list(dict.fromkeys(all_pids))  # first-occurrence order

# Assign integer index to each period for lag computation
pid_to_idx = {p: i for i, p in enumerate(all_pids_sorted)}

# Add scoring periods at the end
if SCORE_PIDS:
    for pid in SCORE_PIDS:
        if pid not in pid_to_idx:
            pid_to_idx[pid] = len(pid_to_idx)
    all_pids_sorted = sorted(pid_to_idx, key=lambda p: pid_to_idx[p])

print(f'  period ordering inferred: {len(all_pids_sorted)} total periods')

# ── Feature engineering helpers ───────────────────────────────────────────────
LAG_VALS = [1, 2, 3, 6, 12, 13]

def compute_lags(df, pid_to_idx, target_col, entity_col, period_col, category_col=None):
    """Add lag features to df. df must contain target_col (NaN for scoring rows)."""
    group_keys = [entity_col] + ([category_col] if category_col else [])
    df = df.copy()
    df['_period_idx'] = df[period_col].map(pid_to_idx)
    df = df.sort_values(group_keys + ['_period_idx'])

    for lag in LAG_VALS:
        df[f'lag_{lag}'] = df.groupby(group_keys)[target_col].shift(lag)

    df['rolling_mean_3'] = (
        df.groupby(group_keys)[target_col]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df['rolling_mean_6'] = (
        df.groupby(group_keys)[target_col]
        .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
    )
    df['expanding_mean'] = (
        df.groupby(group_keys)[target_col]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df['trend_3m'] = df['lag_1'] - df['lag_3']
    df['yoy_diff']  = df['lag_1'] - df['lag_13']
    df['months_since_start'] = df['_period_idx']
    return df

# ── Historical statistics ─────────────────────────────────────────────────────
group_keys = [ENTITY_COL] + ([CATEGORY_COL] if CATEGORY_COL else [])
hist_stats = (
    train.groupby(group_keys)[TARGET_COL]
    .agg(hist_mean='mean', hist_std='std', hist_median='median')
    .reset_index()
)

# ── Build full timeline with lag features ─────────────────────────────────────
print('[FORECAST] Computing lag features...')

# Build val placeholder rows (scoring period)
if PERIOD_COL in sub.columns and ENTITY_COL in sub.columns:
    val_placeholder = sub[[PERIOD_COL, ENTITY_COL]
                           + ([CATEGORY_COL] if CATEGORY_COL and CATEGORY_COL in sub.columns else [])
                          ].copy()
    val_placeholder[TARGET_COL] = np.nan
elif SCORE_PIDS:
    entities = train[ENTITY_COL].unique()
    val_placeholder = pd.DataFrame([
        {PERIOD_COL: pid, ENTITY_COL: ent,
         **({}  if not CATEGORY_COL else {})}
        for pid in SCORE_PIDS for ent in entities
    ])
    val_placeholder[TARGET_COL] = np.nan
else:
    val_placeholder = pd.DataFrame(columns=train.columns)

dose_full = pd.concat(
    [train[[ENTITY_COL, PERIOD_COL]
           + ([CATEGORY_COL] if CATEGORY_COL else [])
           + [TARGET_COL]],
     val_placeholder[[ENTITY_COL, PERIOD_COL]
                     + ([CATEGORY_COL] if CATEGORY_COL and CATEGORY_COL in val_placeholder.columns else [])
                     + [TARGET_COL]]],
    ignore_index=True
)
dose_lag = compute_lags(dose_full, pid_to_idx, TARGET_COL, ENTITY_COL, PERIOD_COL, CATEGORY_COL)

LAG_COLS = [f'lag_{k}' for k in LAG_VALS] + [
    'rolling_mean_3', 'rolling_mean_6', 'expanding_mean', 'trend_3m', 'yoy_diff', 'months_since_start'
]

# ── Covariate features ────────────────────────────────────────────────────────
cov_num_cols = []
if len(cov_all) > 0:
    cov_num_cols = [c for c in cov_all.select_dtypes(include=np.number).columns
                    if c not in [PERIOD_COL]]
    print(f'  covariate numeric cols: {cov_num_cols[:10]}...')

# ── Build train_df ────────────────────────────────────────────────────────────
print('[FORECAST] Building training DataFrame...')
lag_train = dose_lag[dose_lag[PERIOD_COL].isin(TRAIN_PIDS)][
    [ENTITY_COL, PERIOD_COL] + ([CATEGORY_COL] if CATEGORY_COL else []) + LAG_COLS
].copy()

merge_keys = [ENTITY_COL, PERIOD_COL]
train_df = (
    train[[ENTITY_COL, PERIOD_COL]
          + ([CATEGORY_COL] if CATEGORY_COL else [])
          + [TARGET_COL]]
    .merge(hist_stats, on=group_keys, how='left')
    .merge(lag_train, on=[ENTITY_COL, PERIOD_COL] + ([CATEGORY_COL] if CATEGORY_COL else []), how='left')
)

if len(cov_all) > 0:
    cov_train = cov_all[cov_all['split'] == 'train'] if 'split' in cov_all.columns else cov_all
    cov_merge_keys = [c for c in [ENTITY_COL, PERIOD_COL] if c in cov_train.columns]
    train_df = train_df.merge(cov_train[cov_merge_keys + cov_num_cols], on=cov_merge_keys, how='left')

# Encode entity and category as integers
train_df['entity_cat'] = pd.Categorical(train_df[ENTITY_COL]).codes
if CATEGORY_COL:
    train_df['category_cat'] = pd.Categorical(train_df[CATEGORY_COL]).codes

# ── Define features ───────────────────────────────────────────────────────────
EXCLUDE = {TARGET_COL, ENTITY_COL, PERIOD_COL, CATEGORY_COL, 'split', '_period_idx'}
FEATURE_COLS = [c for c in train_df.columns
                if c not in EXCLUDE and train_df[c].dtype in ['float64','int64','float32','int32']]
CAT_FEATURES = [f for f in ['entity_cat', 'category_cat'] if f in FEATURE_COLS]

print(f'  feature cols: {len(FEATURE_COLS)}  cat features: {CAT_FEATURES}')

# Fill NaN in features
col_medians = train_df[FEATURE_COLS].median()
train_df[FEATURE_COLS] = train_df[FEATURE_COLS].fillna(col_medians)
train_df = train_df.dropna(subset=[TARGET_COL])

# ── Walk-forward CV ───────────────────────────────────────────────────────────
print('[FORECAST] Running walk-forward cross-validation...')
try:
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    HAS_ALL = True
except ImportError:
    HAS_ALL = False

# Build CV folds: last 3 non-overlapping windows
n_periods = len(all_pids_sorted)
train_period_idxs = [pid_to_idx[p] for p in TRAIN_PIDS if p in pid_to_idx]
max_train_idx = max(train_period_idxs) if train_period_idxs else n_periods - 1

folds = []
for val_end_offset in [1, 2, 3]:
    val_end   = max_train_idx - (val_end_offset - 1) * 12
    val_start = val_end - 11
    if val_start < 6: continue
    train_end = val_start - 1
    fold_train_pids = {p for p, idx in pid_to_idx.items() if idx <= train_end and p in TRAIN_PIDS}
    fold_val_pids   = {p for p, idx in pid_to_idx.items() if val_start <= idx <= val_end and p in TRAIN_PIDS}
    if len(fold_train_pids) < 12 or len(fold_val_pids) < 1: continue
    folds.append((fold_train_pids, fold_val_pids))

folds = folds[::-1]  # chronological order

def mae_block(y_true, y_pred, cat_arr=None):
    if cat_arr is None:
        return np.mean(np.abs(y_true - y_pred))
    maes = []
    for cat in np.unique(cat_arr):
        mask = cat_arr == cat
        if mask.sum() > 0:
            maes.append(np.mean(np.abs(y_true[mask] - y_pred[mask])))
    return np.mean(maes)

lgb_best_iters, xgb_best_iters, cat_best_iters = [], [], []
cv_results = []

for fold_i, (fold_train_pids, fold_val_pids) in enumerate(folds):
    df_tr = train_df[train_df[PERIOD_COL].isin(fold_train_pids)]
    df_vl = train_df[train_df[PERIOD_COL].isin(fold_val_pids)]
    X_tr, y_tr = df_tr[FEATURE_COLS], df_tr[TARGET_COL]
    X_vl, y_vl = df_vl[FEATURE_COLS], df_vl[TARGET_COL]
    cat_vl = df_vl[CATEGORY_COL].values if CATEGORY_COL else None

    # LightGBM
    params_lgb = dict(n_estimators=1000, learning_rate=0.02, num_leaves=31,
                      min_child_samples=10, random_state=42, verbose=-1)
    cat_idxs = [FEATURE_COLS.index(c) for c in CAT_FEATURES if c in FEATURE_COLS]
    m_lgb = lgb.LGBMRegressor(**params_lgb)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
              categorical_feature=cat_idxs if cat_idxs else 'auto')
    p_lgb = np.clip(m_lgb.predict(X_vl), 0, None)
    mae_lgb = mae_block(y_vl.values, p_lgb, cat_vl)
    lgb_best_iters.append(m_lgb.best_iteration_)
    print(f'  Fold {fold_i+1} LGB MAE={mae_lgb:.4f}  best_iter={m_lgb.best_iteration_}')

    cv_results.append({'fold': fold_i+1, 'model': 'LightGBM', 'mae': mae_lgb,
                        'best_iter': m_lgb.best_iteration_})

# ── Final training ────────────────────────────────────────────────────────────
print('[FORECAST] Training final models...')
X_all, y_all = train_df[FEATURE_COLS], train_df[TARGET_COL]

n_lgb = max(100, int(np.mean(lgb_best_iters) * 1.2)) if lgb_best_iters else 300
print(f'  LightGBM n_estimators={n_lgb}')

lgb_models = []
for seed in [42, 123, 777]:
    m = lgb.LGBMRegressor(n_estimators=n_lgb, learning_rate=0.02,
                           num_leaves=31, min_child_samples=10,
                           random_state=seed, verbose=-1)
    m.fit(X_all, y_all, categorical_feature=cat_idxs if cat_idxs else 'auto')
    lgb_models.append(m)

# Optionally train XGBoost and CatBoost if available
xgb_models, cat_models = [], []
if HAS_ALL:
    n_xgb = max(100, int(np.mean(xgb_best_iters) * 1.2)) if xgb_best_iters else 400
    n_cat = max(100, int(np.mean(cat_best_iters)  * 1.2)) if cat_best_iters else 500

    for seed in [42, 123]:
        # XGBoost
        m_x = xgb.XGBRegressor(n_estimators=n_xgb, learning_rate=0.02,
                                max_depth=5, random_state=seed, verbosity=0)
        m_x.fit(X_all, y_all)
        xgb_models.append(m_x)
        # CatBoost
        m_c = CatBoostRegressor(iterations=n_cat, learning_rate=0.02,
                                 depth=5, random_seed=seed, verbose=0,
                                 cat_features=[FEATURE_COLS.index(c) for c in CAT_FEATURES] or None)
        m_c.fit(X_all, y_all)
        cat_models.append(m_c)

print(f'  Final models: {len(lgb_models)} LGB + {len(xgb_models)} XGB + {len(cat_models)} CAT')

# ── Recursive gap fill ────────────────────────────────────────────────────────
# Detect if scoring period is separated from training by a gap (missing periods)
print('[FORECAST] Detecting and filling gaps...')

score_pids_detected = []
if PERIOD_COL in sub.columns:
    score_pids_detected = sub[PERIOD_COL].unique().tolist()

# Find gap periods (between last training period and first scoring period)
if score_pids_detected and TRAIN_PIDS:
    last_train_idx = max(pid_to_idx.get(p, -1) for p in TRAIN_PIDS)
    score_idxs     = [pid_to_idx.get(p, -1) for p in score_pids_detected]
    first_score_idx = min(s for s in score_idxs if s >= 0)
    gap_pids = [p for p, idx in pid_to_idx.items()
                if last_train_idx < idx < first_score_idx]
    print(f'  gap periods detected: {len(gap_pids)}  '
          f'(between training end idx={last_train_idx} and first score idx={first_score_idx})')
else:
    gap_pids = []

def build_pred_features(pids_to_predict, dose_recursive, hist_stats, cov_all,
                         entities, categories, pid_to_idx, col_medians):
    """Build feature DataFrame for a set of period_ids."""
    rows = []
    for pid in pids_to_predict:
        for ent in entities:
            if categories[0] is None:
                rows.append({PERIOD_COL: pid, ENTITY_COL: ent, TARGET_COL: np.nan})
            else:
                for cat in categories:
                    rows.append({PERIOD_COL: pid, ENTITY_COL: ent,
                                 CATEGORY_COL: cat, TARGET_COL: np.nan})
    placeholder = pd.DataFrame(rows)

    dose_temp = pd.concat([dose_recursive, placeholder], ignore_index=True)
    dose_temp_lag = compute_lags(dose_temp, pid_to_idx, TARGET_COL, ENTITY_COL, PERIOD_COL, CATEGORY_COL)

    pred_lags = dose_temp_lag[dose_temp_lag[PERIOD_COL].isin(pids_to_predict)][
        [ENTITY_COL, PERIOD_COL] + ([CATEGORY_COL] if CATEGORY_COL else []) + LAG_COLS
    ].copy()

    pred_df = (
        placeholder[[ENTITY_COL, PERIOD_COL] + ([CATEGORY_COL] if CATEGORY_COL else [])]
        .merge(hist_stats, on=group_keys, how='left')
        .merge(pred_lags, on=[ENTITY_COL, PERIOD_COL] + ([CATEGORY_COL] if CATEGORY_COL else []), how='left')
    )

    if len(cov_all) > 0:
        cov_prd = cov_all[cov_all[PERIOD_COL].isin(pids_to_predict)] if PERIOD_COL in cov_all.columns else cov_all
        cov_merge_keys = [c for c in [ENTITY_COL, PERIOD_COL] if c in cov_prd.columns]
        if cov_merge_keys and cov_num_cols:
            pred_df = pred_df.merge(cov_prd[cov_merge_keys + cov_num_cols], on=cov_merge_keys, how='left')

    pred_df['entity_cat'] = pd.Categorical(pred_df[ENTITY_COL],
                            categories=train_df[ENTITY_COL].unique()).codes
    if CATEGORY_COL:
        pred_df['category_cat'] = pd.Categorical(pred_df[CATEGORY_COL],
                                   categories=train_df[CATEGORY_COL].unique()).codes

    for col in FEATURE_COLS:
        if col not in pred_df.columns:
            pred_df[col] = 0
    pred_df[FEATURE_COLS] = pred_df[FEATURE_COLS].fillna(col_medians)
    return pred_df

all_entities = train[ENTITY_COL].unique().tolist()
dose_recursive = train[[ENTITY_COL, PERIOD_COL]
                        + ([CATEGORY_COL] if CATEGORY_COL else [])
                        + [TARGET_COL]].copy()

def ensemble_predict(X):
    preds = [np.clip(m.predict(X), 0, None) for m in lgb_models]
    if xgb_models: preds += [np.clip(m.predict(X), 0, None) for m in xgb_models]
    if cat_models: preds += [np.clip(m.predict(X), 0, None) for m in cat_models]
    return np.mean(preds, axis=0)

# Fill gap periods recursively
for gap_pid in sorted(gap_pids, key=lambda p: pid_to_idx[p]):
    pred_df = build_pred_features([gap_pid], dose_recursive, hist_stats, cov_all,
                                   all_entities, SCORE_CATS, pid_to_idx, col_medians)
    preds = ensemble_predict(pred_df[FEATURE_COLS])
    filled = pred_df[[ENTITY_COL, PERIOD_COL] + ([CATEGORY_COL] if CATEGORY_COL else [])].copy()
    filled[TARGET_COL] = preds
    dose_recursive = pd.concat([dose_recursive, filled], ignore_index=True)
    print(f'  gap fill {gap_pid}: mean={preds.mean():.3f}')

# ── Recursive scoring predictions ─────────────────────────────────────────────
print('[FORECAST] Generating scoring predictions recursively...')

score_pids_sorted = sorted(score_pids_detected, key=lambda p: pid_to_idx.get(p, 0)) \
    if score_pids_detected else []

scoring_preds = []
for score_pid in score_pids_sorted:
    pred_df = build_pred_features([score_pid], dose_recursive, hist_stats, cov_all,
                                   all_entities, SCORE_CATS, pid_to_idx, col_medians)
    preds = ensemble_predict(pred_df[FEATURE_COLS])

    # Hierarchy constraint: if multiple categories, ensure sub ≤ total
    if CATEGORY_COL and len(SCORE_CATS) > 1:
        pred_df_cat = pred_df.copy()
        pred_df_cat['_pred'] = preds
        # Find total category (highest mean)
        cat_means = {cat: pred_df_cat[pred_df_cat[CATEGORY_COL] == cat]['_pred'].mean()
                     for cat in SCORE_CATS if cat in pred_df_cat[CATEGORY_COL].values}
        total_cat = max(cat_means, key=lambda c: cat_means[c]) if cat_means else None
        if total_cat:
            total_mask = pred_df_cat[CATEGORY_COL] == total_cat
            for cat in SCORE_CATS:
                if cat != total_cat:
                    sub_mask = pred_df_cat[CATEGORY_COL] == cat
                    pred_df_cat.loc[sub_mask, '_pred'] = pred_df_cat.loc[sub_mask, '_pred'].clip(
                        upper=pred_df_cat.loc[total_mask, '_pred'].values
                    )
        preds = pred_df_cat['_pred'].values

    rows = pred_df[[ENTITY_COL, PERIOD_COL] + ([CATEGORY_COL] if CATEGORY_COL else [])].copy()
    rows['_pred'] = preds
    scoring_preds.append(rows)

    # Add to recursive history
    filled = rows[[ENTITY_COL, PERIOD_COL] + ([CATEGORY_COL] if CATEGORY_COL else [])].copy()
    filled[TARGET_COL] = preds
    dose_recursive = pd.concat([dose_recursive, filled], ignore_index=True)
    print(f'  scored {score_pid}: mean={preds.mean():.3f}')

# ── Write submission.csv ──────────────────────────────────────────────────────
print(f'[FORECAST] Writing {OUTPUT_PATH}...')
if scoring_preds:
    all_preds_df = pd.concat(scoring_preds, ignore_index=True)
    merge_keys_sub = [c for c in [ENTITY_COL, PERIOD_COL, CATEGORY_COL]
                      if c and c in sub.columns and c in all_preds_df.columns]
    if merge_keys_sub:
        sub_out = sub.merge(all_preds_df[merge_keys_sub + ['_pred']],
                            on=merge_keys_sub, how='left')
    else:
        sub_out = sub.copy()
        sub_out['_pred'] = all_preds_df['_pred'].values[:len(sub_out)]

    target_out = [c for c in sub_out.columns if c != 'row_id'][0] \
        if TARGET_COL not in sub_out.columns else TARGET_COL
    sub_out[target_out] = sub_out.get('_pred', sub_out[target_out])
    sub_out[['row_id', target_out]].to_csv(OUTPUT_PATH, index=False)
    print(f'  {OUTPUT_PATH} written: {len(sub_out)} rows  '
          f'mean={sub_out[target_out].mean():.4f}  NaN={sub_out[target_out].isna().sum()}')
else:
    # Fallback: use last known value
    print('  WARNING: no scoring preds generated — using hist_mean fallback')
    sub_out = sub.copy()
    target_col_sub = [c for c in sub.columns if c != 'row_id'][0]
    if ENTITY_COL in sub.columns:
        hist_map = train.groupby(ENTITY_COL)[TARGET_COL].mean()
        sub_out[target_col_sub] = sub_out[ENTITY_COL].map(hist_map).fillna(
            train[TARGET_COL].mean()
        )
    else:
        sub_out[target_col_sub] = train[TARGET_COL].mean()
    sub_out[['row_id', target_col_sub]].to_csv(OUTPUT_PATH, index=False)

# ── Save model metrics ────────────────────────────────────────────────────────
metrics = {
    'cv_results': cv_results,
    'lgb_avg_best_iter': float(np.mean(lgb_best_iters)) if lgb_best_iters else None,
    'n_lgb_final': n_lgb,
    'n_features': len(FEATURE_COLS),
    'feature_cols': FEATURE_COLS[:20],  # top 20 for report
    'n_train_rows': int(len(train_df)),
    'n_gap_periods': len(gap_pids),
    'n_score_periods': len(score_pids_sorted),
}
with open(OUT_DIR / 'model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print('[FORECAST] Complete.')
