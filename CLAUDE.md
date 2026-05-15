# Data Analysis Agent — Standard Operating Procedure

## Trigger
This SOP activates when the user says **"Do the data analysis"** or any similar instruction.
Follow every phase below in order. Do not stop early. Do not ask for clarification.

---

## Phase 0 — Read the Task Description

1. Read `data/DATA_DESCRIPTION.md` carefully.
2. Extract and record:
   - **training file path** (the file with known target values)
   - **covariates file path(s)** (train and/or val splits)
   - **validation/test file path** (the file whose target you must predict)
   - **target column name** (e.g. `rate_per_10000_ed_visits`)
   - **entity column name** (e.g. `jurisdiction`, `state`, `entity_id`)
   - **period column name** (e.g. `period_id`, `date`, `month`)
   - **submission format**: exact column names required in `submission.csv`
   - Any **evaluation metric** mentioned (default: MAE)
3. List all files in `data/` to confirm they exist before proceeding.

---

## Phase 1 — Exploratory Data Analysis

Run `python scripts/eda.py` passing the paths you found in Phase 0.
The script accepts environment variables — set them before calling:

```bash
TRAIN_PATH=data/train.csv \
COV_TRAIN_PATH=data/covariates_train.csv \
COV_VAL_PATH=data/covariates_val.csv \
TARGET_COL=rate_per_10000_ed_visits \
ENTITY_COL=jurisdiction \
PERIOD_COL=period_id \
python scripts/eda.py
```

If the script fails or the env vars don't match the actual file schema,
**do not retry blindly** — inspect the first few rows of each file with
`head -5 data/<filename>`, adjust the variable names, then re-run.

EDA outputs (saved to `outputs/eda/`):
- `summary_stats.csv` — per-entity descriptive statistics
- `ts_by_entity.png` — time series panel plot
- `corr_heatmap.png` — entity-to-entity correlation heatmap
- `covariate_corr.png` — covariate vs target correlation
- `eda_summary.json` — key statistics for the report

---

## Phase 2 — Feature Engineering and Forecasting

Run `python scripts/forecast.py` with the same environment variables plus:

```bash
TRAIN_PATH=data/train.csv \
COV_TRAIN_PATH=data/covariates_train.csv \
COV_VAL_PATH=data/covariates_val.csv \
VAL_PATH=data/sample_submission.csv \
TARGET_COL=rate_per_10000_ed_visits \
ENTITY_COL=jurisdiction \
PERIOD_COL=period_id \
OUTPUT_PATH=submission.csv \
python scripts/forecast.py
```

The script will:
1. Infer the temporal ordering of period_ids from the training data
2. Build lag features (lag_1, lag_2, lag_3, lag_6, lag_12, lag_13)
3. Build trend features (trend_3m = lag_1 - lag_3, yoy_diff = lag_1 - lag_13)
4. Build rolling mean features (rolling_3, rolling_6)
5. Add date-derived features (month, quarter, months_since_start) if dates are available
6. Add covariates (all numeric columns from the covariates file)
7. Run 3-fold walk-forward cross-validation with LightGBM
8. Train final ensemble (LightGBM + XGBoost + CatBoost)
9. Apply recursive gap fill if there is a gap between training end and scoring start
10. Write `submission.csv` to the repo root

If the script fails, **read the traceback**, identify the root cause, fix the script
(or write a corrected inline version), and re-run. Common issues:
- Column name mismatch: use `pd.read_csv(path).columns.tolist()` to inspect
- Period ordering: if period_ids are hashed strings, sort by first-seen order in training data
- Missing covariates for val period: fill with training period medians

Verify `submission.csv` exists and has the correct schema before proceeding:
```python
import pandas as pd
sub = pd.read_csv('submission.csv')
print(sub.shape, sub.columns.tolist(), sub.isna().sum())
```

---

## Phase 3 — Generate report.pdf

Run `python scripts/generate_report.py` after both Phase 1 and Phase 2 complete.

```bash
TARGET_COL=rate_per_10000_ed_visits \
ENTITY_COL=jurisdiction \
python scripts/generate_report.py
```

The report must include:
1. **Dataset overview** — shape, entity count, period count, target distribution
2. **EDA findings** — key correlations, seasonal patterns, entity-level variation
3. **Feature engineering** — list of features used
4. **Model architecture** — CV results (MAE per fold), ensemble weights
5. **Prediction summary** — mean/std/min/max of predictions by entity group

If reportlab is not installed: `pip install reportlab --break-system-packages --quiet`

---

## Phase 4 — Final Verification

Run this checklist before finishing:

```python
import pandas as pd, os

# 1. submission.csv exists and is valid
sub = pd.read_csv('submission.csv')
assert sub.shape[1] == 2, "submission.csv must have exactly 2 columns"
assert sub.isnull().sum().sum() == 0, "submission.csv has NaN values"
print(f"submission.csv: {sub.shape[0]} rows, columns={sub.columns.tolist()}")

# 2. report.pdf exists
assert os.path.exists('report.pdf'), "report.pdf missing"
print(f"report.pdf: {os.path.getsize('report.pdf') / 1024:.1f} KB")

print("All checks passed.")
```

If either file is missing or invalid, return to the appropriate phase and fix it.

---

## Fallback — If Scripts Fail Completely

If `scripts/forecast.py` fails beyond repair, write the entire forecasting pipeline
inline as a Python script. The logic should be:

```python
# Minimal fallback forecasting pipeline
import pandas as pd, numpy as np
import lightgbm as lgb

# 1. Load data (adapt paths from DATA_DESCRIPTION.md)
train = pd.read_csv('data/<train_file>')
cov   = pd.read_csv('data/<covariates_file>')
sub   = pd.read_csv('data/<submission_file>')

# 2. Sort by entity + period
train = train.sort_values([ENTITY_COL, PERIOD_COL])

# 3. Lag features
for lag in [1, 2, 3, 6, 12, 13]:
    train[f'lag_{lag}'] = (
        train.groupby(ENTITY_COL)[TARGET_COL].shift(lag)
    )
train['trend_3m'] = train['lag_1'] - train['lag_3']
train['yoy_diff']  = train['lag_1'] - train['lag_13']
train['rolling_3'] = (
    train.groupby(ENTITY_COL)[TARGET_COL]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)

# 4. Add hist_mean per entity
hist_mean = train.groupby(ENTITY_COL)[TARGET_COL].mean().rename('hist_mean')
train = train.merge(hist_mean, on=ENTITY_COL)

# 5. Drop NaN rows and train LightGBM
feature_cols = [c for c in train.columns
                if c not in [TARGET_COL, ENTITY_COL, PERIOD_COL]
                and train[c].dtype in ['float64','int64','float32','int32']]
df_train = train.dropna(subset=feature_cols + [TARGET_COL])
X = df_train[feature_cols]
y = df_train[TARGET_COL]

model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                           num_leaves=31, random_state=42)
model.fit(X, y)

# 6. Build prediction features for val period
# (merge covariates, fill lags with last known value, predict)
# ... adapt as needed from DATA_DESCRIPTION.md ...

# 7. Write submission
sub['<target_col>'] = predictions
sub[['row_id', '<target_col>']].to_csv('submission.csv', index=False)
```

---

## Important Notes

- **Never hardcode period_ids** — always derive them from the input files at runtime.
- **Never assume the domain** — the held-out dataset may not be about drug overdoses.
- **Hierarchy constraints**: if the DATA_DESCRIPTION.md mentions nested categories
  (e.g., subcategory ≤ total), enforce them: clip subcategory predictions to ≤ total.
- **Clipping**: never predict negative rates. Use `np.clip(preds, 0, None)`.
- **submission.csv goes to repo root** (`./submission.csv`), not `data/`.
- **report.pdf goes to repo root** (`./report.pdf`), not `outputs/`.
