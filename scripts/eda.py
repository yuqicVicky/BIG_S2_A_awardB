"""
General-purpose EDA for longitudinal panel data.
Adapted from STAI-X correlation analysis notebook (check_v2.ipynb).

Usage (env vars):
  TRAIN_PATH        path to training CSV (required)
  COV_TRAIN_PATH    path to train covariates CSV (optional)
  COV_VAL_PATH      path to val covariates CSV (optional)
  TARGET_COL        name of target column (default: rate_per_10000_ed_visits)
  ENTITY_COL        name of entity/jurisdiction column (default: jurisdiction)
  PERIOD_COL        name of period/time column (default: period_id)
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sps
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Config from env ───────────────────────────────────────────────────────────
TRAIN_PATH     = os.environ.get('TRAIN_PATH',     'data/dose_sys_train.csv')
COV_TRAIN_PATH = os.environ.get('COV_TRAIN_PATH', 'data/covariates_train.csv')
COV_VAL_PATH   = os.environ.get('COV_VAL_PATH',   '')
TARGET_COL     = os.environ.get('TARGET_COL',     'rate_per_10000_ed_visits')
ENTITY_COL     = os.environ.get('ENTITY_COL',     'jurisdiction')
PERIOD_COL     = os.environ.get('PERIOD_COL',     'period_id')

OUT_DIR = Path('outputs/eda')
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'[EDA] Loading training data from {TRAIN_PATH}')
train = pd.read_csv(TRAIN_PATH)
print(f'  shape: {train.shape}  columns: {train.columns.tolist()}')

# Detect category column (optional — if data has multiple categories per entity/period)
cat_cols_candidates = [c for c in train.columns
                       if c not in [TARGET_COL, ENTITY_COL, PERIOD_COL]
                       and train[c].dtype == object and train[c].nunique() <= 20]
CATEGORY_COL = cat_cols_candidates[0] if cat_cols_candidates else None
print(f'  category column detected: {CATEGORY_COL}')

# ── 1. Basic statistics ───────────────────────────────────────────────────────
print('[EDA] Computing per-entity statistics...')
hist_stats = (
    train.groupby([ENTITY_COL] + ([CATEGORY_COL] if CATEGORY_COL else []))[TARGET_COL]
    .agg(hist_mean='mean', hist_std='std', hist_median='median',
         hist_min='min', hist_max='max', n_periods='count')
    .reset_index()
)
hist_stats.to_csv(OUT_DIR / 'summary_stats.csv', index=False)
print(f'  saved summary_stats.csv  ({hist_stats.shape[0]} rows)')

# ── 2. Time series panel plot ─────────────────────────────────────────────────
print('[EDA] Plotting time series by entity...')

# Use the primary category if available (first value alphabetically)
if CATEGORY_COL:
    primary_cat = sorted(train[CATEGORY_COL].dropna().unique())[0]
    ts_data = train[train[CATEGORY_COL] == primary_cat].copy()
else:
    ts_data = train.copy()

# Sort periods by appearance frequency (proxy for chronological order if hashed)
period_order = ts_data.groupby(PERIOD_COL)[TARGET_COL].mean().index.tolist()
try:
    period_order = sorted(period_order)
except TypeError:
    pass  # keep original order if not sortable

ts_data[PERIOD_COL] = pd.Categorical(ts_data[PERIOD_COL], categories=period_order, ordered=True)
ts_pivot = ts_data.pivot_table(index=PERIOD_COL, columns=ENTITY_COL, values=TARGET_COL)

n_entities = ts_pivot.shape[1]
fig_cols = min(4, n_entities)
fig_rows = max(1, (n_entities + fig_cols - 1) // fig_cols)

# Aggregate: mean ± std across all entities
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
mu = ts_pivot.mean(axis=1)
sigma = ts_pivot.std(axis=1)
x = range(len(mu))

axes[0].fill_between(x, mu - sigma, mu + sigma, alpha=0.25, color='steelblue', label='±1 std')
axes[0].plot(x, mu, color='steelblue', lw=2, label='mean across entities')
for col in ts_pivot.columns:
    axes[0].plot(x, ts_pivot[col].values, lw=0.4, alpha=0.3, color='steelblue')
axes[0].set_title(f'All entities — {TARGET_COL}')
axes[0].set_xlabel('Period index')
axes[0].set_ylabel(TARGET_COL)
axes[0].legend(fontsize=8)
tick_step = max(1, len(x) // 10)
axes[0].set_xticks(x[::tick_step])
axes[0].set_xticklabels([str(period_order[i]) for i in x[::tick_step]],
                         rotation=45, ha='right', fontsize=7)

# High/medium/low entities
hist_mean_series = ts_pivot.mean()
q33 = hist_mean_series.quantile(0.333)
q67 = hist_mean_series.quantile(0.667)
tier_colors = {'Low': 'steelblue', 'Medium': 'darkorange', 'High': 'crimson'}
for ent in ts_pivot.columns:
    v = hist_mean_series[ent]
    tier = 'Low' if v <= q33 else ('Medium' if v <= q67 else 'High')
    axes[1].plot(x, ts_pivot[ent].values, lw=0.8, alpha=0.5,
                 color=tier_colors[tier])
for tier, color in tier_colors.items():
    axes[1].plot([], [], color=color, lw=2, label=tier)
axes[1].set_title(f'Entities colored by baseline tier')
axes[1].set_xlabel('Period index')
axes[1].legend(fontsize=8)
axes[1].set_xticks(x[::tick_step])
axes[1].set_xticklabels([str(period_order[i]) for i in x[::tick_step]],
                          rotation=45, ha='right', fontsize=7)

plt.tight_layout()
plt.savefig(OUT_DIR / 'ts_by_entity.png', dpi=100, bbox_inches='tight')
plt.close()
print('  saved ts_by_entity.png')

# ── 3. Entity-to-entity correlation heatmap ───────────────────────────────────
print('[EDA] Computing entity correlation matrix...')
entity_corr = ts_pivot.corr()

fig, ax = plt.subplots(figsize=(max(8, n_entities // 3), max(7, n_entities // 3)))
mask = np.zeros_like(entity_corr, dtype=bool)
np.fill_diagonal(mask, True)
sns.heatmap(entity_corr, mask=mask, cmap='RdYlGn', vmin=-0.3, vmax=1.0,
            ax=ax, xticklabels=True, yticklabels=True,
            annot=(n_entities <= 20), fmt='.2f', linewidths=0.3 if n_entities <= 30 else 0)
ax.set_title(f'Entity × Entity correlation — {TARGET_COL}\n(category: {primary_cat if CATEGORY_COL else "all"})')
ax.tick_params(axis='x', labelsize=max(5, 9 - n_entities // 15), rotation=90)
ax.tick_params(axis='y', labelsize=max(5, 9 - n_entities // 15))
plt.tight_layout()
plt.savefig(OUT_DIR / 'corr_heatmap.png', dpi=100, bbox_inches='tight')
plt.close()
print('  saved corr_heatmap.png')

# ── 4. Pairwise correlation analysis ─────────────────────────────────────────
print('[EDA] Pairwise correlation analysis...')
entities = entity_corr.columns.tolist()
pair_rs = []
for i, ea in enumerate(entities):
    for j, eb in enumerate(entities):
        if j <= i: continue
        pair_rs.append(entity_corr.loc[ea, eb])

pair_rs = np.array(pair_rs)
print(f'  pairwise r: mean={pair_rs.mean():.4f}  std={pair_rs.std():.4f}  '
      f'median={np.median(pair_rs):.4f}')

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(pair_rs, bins=40, color='steelblue', edgecolor='white')
axes[0].set_title('Distribution of pairwise entity correlations')
axes[0].set_xlabel('Pearson r')
axes[0].set_ylabel('Count')
axes[0].axvline(pair_rs.mean(), color='crimson', lw=2, label=f'mean={pair_rs.mean():.3f}')
axes[0].legend()

# Tier-based within/between analysis
hist_mean_map = ts_pivot.mean()
def get_tier(v): return 'Low' if v <= q33 else ('Medium' if v <= q67 else 'High')
tier_map = {e: get_tier(hist_mean_map[e]) for e in entities}
within, between = [], []
for i, ea in enumerate(entities):
    for j, eb in enumerate(entities):
        if j <= i: continue
        r = entity_corr.loc[ea, eb]
        if tier_map[ea] == tier_map[eb]: within.append(r)
        else: between.append(r)

axes[1].boxplot([within, between], labels=['Within-tier', 'Between-tier'])
t_stat, p_val = sps.ttest_ind(within, between, equal_var=False)
axes[1].set_title(f'Within vs between-tier corr  (t={t_stat:.2f}, p={p_val:.3f})')
axes[1].set_ylabel('Pearson r')
plt.tight_layout()
plt.savefig(OUT_DIR / 'tier_corr_analysis.png', dpi=100, bbox_inches='tight')
plt.close()
print('  saved tier_corr_analysis.png')

# ── 5. Covariate correlation with target ──────────────────────────────────────
cov_dfs = []
for path_var, split_label in [(COV_TRAIN_PATH, 'train'), (COV_VAL_PATH, 'val')]:
    if path_var and os.path.exists(path_var):
        df = pd.read_csv(path_var)
        df['split'] = split_label
        cov_dfs.append(df)

if cov_dfs:
    print('[EDA] Analyzing covariate correlations...')
    cov_all = pd.concat(cov_dfs, ignore_index=True)
    # Merge covariates with training target
    merge_keys = [c for c in [ENTITY_COL, PERIOD_COL] if c in cov_all.columns and c in train.columns]
    merged = train[[ENTITY_COL, PERIOD_COL, TARGET_COL]].merge(cov_all, on=merge_keys, how='inner')
    num_cov_cols = [c for c in merged.select_dtypes(include=np.number).columns
                    if c != TARGET_COL and merged[c].std() > 0]

    cov_corr_rows = []
    for col in num_cov_cols:
        valid = merged[[TARGET_COL, col]].dropna()
        if len(valid) < 30: continue
        r, p = sps.pearsonr(valid[TARGET_COL], valid[col])
        cov_corr_rows.append({'feature': col, 'r': r, 'p': p, 'abs_r': abs(r)})
    cov_corr_df = pd.DataFrame(cov_corr_rows).sort_values('abs_r', ascending=False)
    cov_corr_df.to_csv(OUT_DIR / 'covariate_corr.csv', index=False)

    top_n = min(20, len(cov_corr_df))
    if top_n > 0:
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
        colors = ['steelblue' if r >= 0 else 'crimson'
                  for r in cov_corr_df.head(top_n)['r']]
        ax.barh(cov_corr_df.head(top_n)['feature'],
                cov_corr_df.head(top_n)['r'], color=colors)
        ax.set_title(f'Top {top_n} covariate correlations with {TARGET_COL}')
        ax.set_xlabel('Pearson r')
        ax.axvline(0, color='black', lw=0.8)
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'covariate_corr.png', dpi=100, bbox_inches='tight')
        plt.close()
        print(f'  saved covariate_corr.png  ({top_n} features)')
else:
    print('[EDA] No covariate files found — skipping covariate analysis')
    cov_corr_df = pd.DataFrame()

# ── 6. Seasonality check ──────────────────────────────────────────────────────
print('[EDA] Checking for seasonality...')
monthly_avg = ts_data.groupby(PERIOD_COL)[TARGET_COL].mean().reset_index()
monthly_avg = monthly_avg.rename(columns={PERIOD_COL: 'period', TARGET_COL: 'mean_rate'})

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(len(monthly_avg)), monthly_avg['mean_rate'], marker='o', lw=1.5,
        color='steelblue', markersize=4)
ax.set_title(f'Global mean {TARGET_COL} over time')
ax.set_xlabel('Period index')
ax.set_ylabel(f'Mean {TARGET_COL}')
tick_step = max(1, len(monthly_avg) // 12)
ax.set_xticks(range(0, len(monthly_avg), tick_step))
ax.set_xticklabels([str(p) for p in monthly_avg['period'].iloc[::tick_step]],
                    rotation=45, ha='right', fontsize=7)
plt.tight_layout()
plt.savefig(OUT_DIR / 'global_trend.png', dpi=100, bbox_inches='tight')
plt.close()
print('  saved global_trend.png')

# ── 7. Save EDA summary JSON (used by generate_report.py) ────────────────────
eda_summary = {
    'n_entities':        int(train[ENTITY_COL].nunique()),
    'n_periods':         int(train[PERIOD_COL].nunique()),
    'n_rows':            int(len(train)),
    'target_col':        TARGET_COL,
    'entity_col':        ENTITY_COL,
    'period_col':        PERIOD_COL,
    'category_col':      CATEGORY_COL,
    'target_mean':       float(train[TARGET_COL].mean()),
    'target_std':        float(train[TARGET_COL].std()),
    'target_min':        float(train[TARGET_COL].min()),
    'target_max':        float(train[TARGET_COL].max()),
    'pairwise_r_mean':   float(pair_rs.mean()),
    'pairwise_r_std':    float(pair_rs.std()),
    'within_tier_r':     float(np.mean(within)) if within else None,
    'between_tier_r':    float(np.mean(between)) if between else None,
    'tier_ttest_p':      float(p_val) if within else None,
    'top_covariate':     cov_corr_df.iloc[0]['feature'] if len(cov_corr_df) > 0 else None,
    'top_covariate_r':   float(cov_corr_df.iloc[0]['r']) if len(cov_corr_df) > 0 else None,
}

with open(OUT_DIR / 'eda_summary.json', 'w') as f:
    json.dump(eda_summary, f, indent=2)

print(f'\n[EDA] Complete. Outputs saved to {OUT_DIR}/')
print(f'  Entities: {eda_summary["n_entities"]}  '
      f'Periods: {eda_summary["n_periods"]}  '
      f'Target mean: {eda_summary["target_mean"]:.3f}')
