---
name: feature-engineer
description: Use this skill to split the dataset into train/test sets and build a preprocessing pipeline (median imputation + one-hot encoding) that is fit on the training split only. Never fit on validation or test data. Returns the fitted transformer and split arrays for use in modeling.
---

# Skill: feature-engineer

## When to Use

Use this skill after leakage check passes and before any model is trained. It must run in this exact order:

1. Split raw data into train/test
2. Identify numeric and categorical feature columns
3. Build the `ColumnTransformer` pipeline
4. **Fit only on train split**
5. Transform train, validation (if applicable), and test using the fitted transformer

Never call this skill on the full dataset — only on the training split.

## Python Function

```python
from src.data_agent.skills.preprocessing import build_preprocessing_pipeline

pipeline = build_preprocessing_pipeline(df, numeric_cols, categorical_cols)
# Returns an UNFITTED ColumnTransformer
# Caller is responsible for: pipeline.fit(X_train) then pipeline.transform(X_test)
```

Train/test splitting and end-to-end preprocessing are handled inside `train_and_evaluate_models`:

```python
from src.data_agent.skills.modeling import train_and_evaluate_models

result = train_and_evaluate_models(
    df=state.raw_data,
    target_col=state.task.target_variable,
    task_type=state.task.task_type,
    feature_cols=state.task.feature_candidates,
    test_size=0.2,
    random_state=42,
)
```

## Inputs

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `df` | `pd.DataFrame` | Yes | Raw, unmodified data (not yet split) |
| `target_col` | `str` | Yes | Column to predict; excluded from features |
| `task_type` | `TaskType` | Yes | Determines metric selection downstream |
| `feature_cols` | `list[str]` | Yes | Columns to use; ID and leakage columns must be excluded before passing |
| `test_size` | `float` | No | Default 0.2 |
| `random_state` | `int` | No | Default 42; must be recorded in state |

## Outputs Written to State

| State field | Content |
|-------------|---------|
| `state.split_metadata` | `{n_train, n_test, test_size, random_state, target_col}` |
| `state.transformer_record` | Fitted `ColumnTransformer` object |
| `state.model_results` | Baseline and candidate model metrics |

## Pipeline Components

| Step | Numeric Columns | Categorical Columns |
|------|----------------|-------------------|
| Imputation | `SimpleImputer(strategy="median")` | `SimpleImputer(strategy="most_frequent")` |
| Encoding | `StandardScaler()` | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` |

## Critical Constraints

- **Fit on train only.** `pipeline.fit(X_train)` — never `pipeline.fit(X_full)` or `pipeline.fit(X_test)`.
- **Apply to test via fitted object.** `pipeline.transform(X_test)` — not re-fit.
- **Record the random seed.** Every split must record `random_state` in `state.split_metadata`.
- **Exclude before split.** Any leakage columns or ID columns must be removed from `feature_cols` before this skill is called. The skill does not perform leakage detection.

## Inspector Checks

After this step the Inspector must verify:

1. `state.split_metadata` contains `n_train`, `n_test`, `test_size`, `random_state`.
2. Transformer was fit on train rows only (n_train rows).
3. No test data touched during fitting.
4. `n_train + n_test == total rows` (within rounding for stratified splits).

## Error Conditions

| Condition | Required Action |
|-----------|----------------|
| `target_col` not in DataFrame | Raise `ValueError` |
| `feature_cols` is empty after exclusions | Raise `ValueError("No usable features remain after exclusions")` |
| Categorical column with all-null values | Drop column and log warning to state |

## Notes

- The pipeline returns transformed numpy arrays. Column names are not preserved in the transformed output; feature name mapping is stored in the `ColumnTransformer` object.
- For datasets with >50 unique categories in a single column, OHE will produce many columns. V1 does not apply cardinality capping.
