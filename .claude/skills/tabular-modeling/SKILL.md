---
name: tabular-modeling
description: Use this skill when training, validating, or inspecting tabular machine learning models for classification or regression. It covers train/test split, preprocessing pipelines, baseline models, candidate models, leakage avoidance, prediction generation, and artifact saving.
---

# Skill: tabular-modeling

## When to Use

Use this skill after `leakage-check` has returned `approved: true` and preprocessing is confirmed safe. It covers everything from feature construction through model training and held-out evaluation.

Trigger this skill when:
- The `modeling` step in the approved plan begins execution.
- A baseline model needs to be established before a candidate model is trained.
- Predictions need to be generated on the test set and saved as artifacts.
- Model artifacts need to be serialized for report generation.

Do **not** run this skill before:
- `leakage-check` has completed with `approved: true` for the preprocessing stage.
- `state.splits` contains `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`.
- `state.task.task_type` is one of `binary_classification`, `multiclass_classification`, `regression`.

This skill does not apply to `descriptive` tasks. If `task_type` is `descriptive`, halt and return an error.

> **Implementation note (Award-B mode).** When the analysis is schema-driven (a hidden
> `data/DATA_DESCRIPTION.md` plus a sample submission), the proven engine
> `models.train_and_predict` uses a single internal stratified/time/random holdout to
> select the model, then refits the selected model on all training rows before
> predicting the unlabeled competition test set. In that mode `state.splits` carries the
> internal holdout, and the `val`/`test` metrics in `state.evaluation` reference that
> same held-out split (the Kaggle test labels are hidden, so only predictions are
> produced for them). The leakage rules below are unchanged.

---

## Inputs

| Field | Source | Required |
|-------|--------|----------|
| `task.task_type` | `state.task.task_type` | Yes |
| `task.target_variable` | `state.task.target_variable` | Yes |
| `task.feature_candidates` | `state.task.feature_candidates` | Yes |
| `task.excluded_from_features` | `state.task.excluded_from_features` | Yes |
| `splits.X_train` | `state.splits.X_train` | Yes |
| `splits.X_val` | `state.splits.X_val` | Yes |
| `splits.X_test` | `state.splits.X_test` | Yes |
| `splits.y_train` | `state.splits.y_train` | Yes |
| `splits.y_val` | `state.splits.y_val` | Yes |
| `splits.y_test` | `state.splits.y_test` | Yes |
| `splits.random_seed` | `state.splits.random_seed` | Yes |
| `leakage_audit.approved` | `state.leakage_audit.approved` | Yes — must be `true` |
| `data_profile.categorical_columns` | `state.data_profile.categorical_columns` | Yes |
| `data_profile.numeric_columns` | `state.data_profile.numeric_columns` | Yes |
| `run_id` | `state.run_id` | Yes |

If `leakage_audit.approved` is `false` or missing, halt immediately. Do not proceed with any model training.

---

## Required Workflow

Execute these steps in exact order. No step may be skipped. No step may be executed before the step that precedes it.

### Step 1 — Identify Target Variable

Read `state.task.target_variable`. This is the only column that may be used as `y`. Record it in `state.models.target_variable`.

Verify:
- `target_variable` exists in the original DataFrame.
- `target_variable` is **not** in `state.task.feature_candidates`.
- If it is in `feature_candidates` for any reason, halt with error `"target_in_features"` and do not proceed.

### Step 2 — Exclude Target from Features

Construct `X` as `df[state.task.feature_candidates]`. The target column must not appear in `X` at any point.

Explicitly verify: `assert target_variable not in X.columns`.

If the assertion fails, halt with `"target_in_features"`.

### Step 3 — Exclude ID-Like and Leakage-Risk Columns

Remove from `feature_candidates` any column that:
- Appears in `state.task.excluded_from_features`.
- Appears in `state.data_profile.potential_id_columns`.
- Was flagged in `state.leakage_audit.suspected_columns` with verdict `FAIL`.

Record the final feature list as `state.models.features_used`. Any column removed at this step must be logged in `state.models.exclusions` with the reason.

### Step 4 — Confirm Split Exists

Verify that `state.splits.X_train`, `state.splits.X_val`, and `state.splits.X_test` are all non-empty DataFrames.

Verify that the union of train + val + test row indices covers all rows in the original dataset with no overlap between sets:
- `len(X_train) + len(X_val) + len(X_test) == len(X_original)`
- `set(X_train.index) ∩ set(X_test.index) == ∅`
- `set(X_train.index) ∩ set(X_val.index) == ∅`

If any check fails, halt with `"split_integrity_error"`. Do not proceed with modeling on a compromised split.

### Step 5 — Build Preprocessing Pipeline (fit on X_train only)

Construct a `sklearn.pipeline.Pipeline` or `ColumnTransformer` that handles:

**Numeric columns:**
- Imputation: `SimpleImputer(strategy='median')`
- Scaling: `StandardScaler()`

**Categorical columns:**
- Imputation: `SimpleImputer(strategy='most_frequent')`
- Encoding: `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`

**High-cardinality categorical columns** (`n_unique > 20`):
- Use `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` instead of OHE.
- Record in `state.models.high_cardinality_columns_encoded` which columns used ordinal encoding.

Fitting rule — strictly enforced:

```python
preprocessor.fit(X_train)           # ONLY this is permitted
X_train_t = preprocessor.transform(X_train)
X_val_t   = preprocessor.transform(X_val)
X_test_t  = preprocessor.transform(X_test)   # saved but NOT inspected until evaluation step
```

`fit_transform(X_train)` is permitted as equivalent to `fit(X_train); transform(X_train)`.  
`fit_transform` on any data other than `X_train` is never permitted.

Record:
- `state.transformers.preprocessor`: the fitted preprocessor object.
- `state.transformers.fitted_on`: `"X_train"`.
- `state.transformers.n_samples_fit`: `len(X_train)`.
- `state.artifacts.<run_id>_preprocessor.joblib`: serialized preprocessor.

### Step 6 — Inline Leakage Check on Preprocessing

Before proceeding to model training, verify:

1. `state.transformers.fitted_on == "X_train"`.
2. `state.transformers.n_samples_fit == len(X_train)`.
3. No column from `state.leakage_audit.suspected_columns` (FAIL verdict) is present in `X_train_t`.

If any check fails, halt with `"preprocessing_leakage_detected"`. Log the failure in `state.execution_log.modeling.leakage_check`.

### Step 7 — Train Baseline Model

Train the baseline model specified for the current `task_type` (see Section 5). The baseline model must:

- Be fitted on `(X_train_t, y_train)` only.
- Be evaluated on `(X_val_t, y_val)` to produce `val_metrics`.
- Not access `X_test_t` or `y_test` at any point during this step.
- Have all required metrics recorded in `state.models.baseline.val_metrics` with `split: "val"`.

Record:
- `state.models.baseline.model_name`
- `state.models.baseline.hyperparameters`
- `state.models.baseline.val_metrics` (with `split: "val"`)
- `state.artifacts.<run_id>_baseline_model.joblib`

### Step 8 — Train Candidate Model

Train the candidate model specified for the current `task_type` (see Section 5). The candidate model must:

- Be fitted on `(X_train_t, y_train)` only.
- Use cross-validation within `X_train` for hyperparameter selection if tuning is applied. `X_val` and `X_test` must not participate in cross-validation.
- Be evaluated on `(X_val_t, y_val)` to produce `val_metrics`.
- Not access `X_test_t` or `y_test` at any point during this step.
- Have val metrics compared against baseline val metrics. If candidate does not improve over baseline on the primary metric, issue a WARN — do not automatically halt.

Record:
- `state.models.candidate.model_name`
- `state.models.candidate.hyperparameters`
- `state.models.candidate.cv_strategy` (if CV was used)
- `state.models.candidate.val_metrics` (with `split: "val"`)
- `state.models.candidate.baseline_comparison` (delta on primary metric)
- `state.artifacts.<run_id>_candidate_model.joblib`

### Step 9 — Generate Predictions on Test Set

After both baseline and candidate models are trained and evaluated on the validation set, generate predictions on the test set:

```python
y_pred_baseline  = baseline_pipeline.predict(X_test_t)
y_pred_candidate = candidate_pipeline.predict(X_test_t)

# For classifiers only:
y_proba_baseline  = baseline_pipeline.predict_proba(X_test_t)
y_proba_candidate = candidate_pipeline.predict_proba(X_test_t)
```

Save predictions as artifacts:
- `state.artifacts.<run_id>_test_predictions.csv`: columns `[y_true, y_pred_baseline, y_pred_candidate]`
- For classifiers: also include `y_proba_candidate` columns (one per class).

### Step 10 — Compute Test Metrics and Record

Compute all required metrics for both baseline and candidate on the test set. All metrics must carry `split: "test"`.

See Section 5 for required metrics per task type.

Record:
- `state.models.baseline.test_metrics` (with `split: "test"`)
- `state.models.candidate.test_metrics` (with `split: "test"`)
- `state.evaluation.baseline_vs_candidate`: comparison table with delta per metric.

### Step 11 — Save All Artifacts

Verify all artifact paths are written and accessible. Record the full manifest in `state.artifacts`.

See Section 6 for the complete required artifact list.

---

## Preprocessing Rules

These rules are absolute. Violation of any rule is a `FAIL` that halts execution.

| Rule | Description |
|------|-------------|
| **Split before fit** | `train_test_split` must execute before any call to `preprocessor.fit()`. No exceptions. |
| **Fit on train only** | `preprocessor.fit()` or `preprocessor.fit_transform()` may only receive `X_train` as input. |
| **Transform all splits** | After fitting, `preprocessor.transform()` is applied to `X_train`, `X_val`, and `X_test` separately. |
| **Seed required** | `train_test_split` must use `random_state=state.splits.random_seed`. The seed must be a fixed integer, never `None`. |
| **Stratify classification** | For `binary_classification` and `multiclass_classification`, use `stratify=y` in `train_test_split`. |
| **Default split ratios** | Train 70%, Val 15%, Test 15%. Deviations require justification recorded in `state.splits.split_rationale`. |
| **No target encoding pre-split** | Target encoding (mean encoding, leave-one-out encoding) must be performed inside a CV fold or after split, fitted on train fold only. |
| **No SMOTE pre-split** | Oversampling must be applied only to `X_train`, never before splitting. |
| **No scaling on target** | `y_train`, `y_val`, `y_test` must not be scaled or transformed unless the task explicitly requires it. If target scaling is applied, record the inverse transform in `state.transformers`. |
| **Preserve test set** | `X_test` and `y_test` must not be accessed, inspected, or used in any fitting or selection decision until Step 9. |

---

## Model Rules by Task Type

### `binary_classification`

**Baseline model:**

```python
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy='stratified', random_state=seed)
baseline.fit(X_train_t, y_train)
```

**Candidate model:**

```python
from sklearn.ensemble import RandomForestClassifier

candidate = RandomForestClassifier(
    n_estimators=100,
    random_state=seed,
    class_weight='balanced'   # required if class_imbalance_detected == True
)
candidate.fit(X_train_t, y_train)
```

**Required metrics (both baseline and candidate, both val and test splits):**

| Metric | Function | Notes |
|--------|----------|-------|
| Accuracy | `accuracy_score` | Always report alongside class base rate |
| F1 (weighted) | `f1_score(average='weighted')` | Primary for imbalanced data |
| AUC-ROC | `roc_auc_score` | Primary overall metric |
| Precision (weighted) | `precision_score(average='weighted')` | — |
| Recall (weighted) | `recall_score(average='weighted')` | — |

**Required artifacts:** confusion matrix plot, ROC curve plot.

**Class imbalance rule:** If `state.task.class_imbalance_detected == true`, set `class_weight='balanced'` in the candidate model. Record in `state.models.candidate.hyperparameters.class_weight: "balanced"`. Do not use default `class_weight=None` when imbalance is detected.

---

### `multiclass_classification`

**Baseline model:**

```python
from sklearn.dummy import DummyClassifier

baseline = DummyClassifier(strategy='most_frequent', random_state=seed)
baseline.fit(X_train_t, y_train)
```

**Candidate model:**

```python
from sklearn.ensemble import RandomForestClassifier

candidate = RandomForestClassifier(
    n_estimators=100,
    random_state=seed,
    class_weight='balanced'
)
candidate.fit(X_train_t, y_train)
```

**Required metrics (both baseline and candidate, both val and test splits):**

| Metric | Function | Notes |
|--------|----------|-------|
| F1 (weighted) | `f1_score(average='weighted')` | Primary metric |
| F1 (macro) | `f1_score(average='macro')` | Required — penalizes majority-class bias |
| Accuracy | `accuracy_score` | Report per-class base rate alongside |
| Precision (macro) | `precision_score(average='macro')` | — |
| Recall (macro) | `recall_score(average='macro')` | — |

**Required artifacts:** confusion matrix plot (multi-class, normalized by true label).

---

### `regression`

**Baseline model:**

```python
from sklearn.dummy import DummyRegressor

baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train_t, y_train)
```

**Candidate model:**

```python
from sklearn.ensemble import RandomForestRegressor

candidate = RandomForestRegressor(
    n_estimators=100,
    random_state=seed
)
candidate.fit(X_train_t, y_train)
```

**Required metrics (both baseline and candidate, both val and test splits):**

| Metric | Function | Notes |
|--------|----------|-------|
| RMSE | `np.sqrt(mean_squared_error(...))` | Primary metric |
| MAE | `mean_absolute_error` | Robust to outliers; compare with RMSE |
| R² | `r2_score` | Proportion of variance explained |
| MAPE | `mean_absolute_percentage_error` | Only if target has no zero values |

**Required artifacts:** residual plot (y_pred − y_true vs y_pred), prediction-vs-actual scatter plot.

**Outlier rule:** If `state.task.warnings` contains `target_outliers`, record both RMSE and MAE and note their divergence in `state.models.candidate.notes`. Do not remove outliers without explicit justification in the plan.

---

## Artifact Rules

Every artifact must be written before the modeling step is marked complete. File names must use the `run_id` prefix. No artifact path may be a placeholder string.

### Required Artifacts

| Artifact | Path Pattern | Produced In |
|----------|-------------|-------------|
| Fitted preprocessor | `outputs/artifacts/<run_id>_preprocessor.joblib` | Step 5 |
| Baseline model | `outputs/artifacts/<run_id>_baseline_model.joblib` | Step 7 |
| Candidate model | `outputs/artifacts/<run_id>_candidate_model.joblib` | Step 8 |
| Test predictions CSV | `outputs/artifacts/<run_id>_test_predictions.csv` | Step 9 |
| Confusion matrix plot | `outputs/artifacts/<run_id>_confusion_matrix.png` | Step 10 (classification) |
| ROC curve plot | `outputs/artifacts/<run_id>_roc_curve.png` | Step 10 (binary only) |
| Residual plot | `outputs/artifacts/<run_id>_residuals.png` | Step 10 (regression) |
| Prediction vs actual | `outputs/artifacts/<run_id>_pred_vs_actual.png` | Step 10 (regression) |
| Feature importance CSV | `outputs/artifacts/<run_id>_feature_importance.csv` | Step 10 |
| Feature importance plot | `outputs/artifacts/<run_id>_feature_importance.png` | Step 10 |
| Run log | `outputs/logs/<run_id>_run.json` | Continuously updated |

After all artifacts are written, verify each path with `os.path.exists(path)`. Any path that returns `False` must be recorded as a FAIL in `state.execution_log.modeling.artifact_check`.

---

## Output Format

Write results to `state.models` and `state.evaluation`.

```json
{
  "models": {
    "target_variable": "Survived",
    "features_used": ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"],
    "exclusions": {
      "PassengerId": "id_column",
      "Name": "high_cardinality",
      "Ticket": "high_cardinality",
      "Cabin": "critical_missing"
    },
    "high_cardinality_columns_encoded": [],
    "baseline": {
      "model_name": "DummyClassifier",
      "hyperparameters": {"strategy": "stratified", "random_state": 42},
      "val_metrics":  {"split": "val",  "accuracy": 0.508, "f1_weighted": 0.501, "roc_auc": 0.500},
      "test_metrics": {"split": "test", "accuracy": 0.511, "f1_weighted": 0.503, "roc_auc": 0.501}
    },
    "candidate": {
      "model_name": "RandomForestClassifier",
      "hyperparameters": {"n_estimators": 100, "random_state": 42, "class_weight": null},
      "cv_strategy": "none",
      "val_metrics":  {"split": "val",  "accuracy": 0.821, "f1_weighted": 0.819, "roc_auc": 0.876},
      "test_metrics": {"split": "test", "accuracy": 0.810, "f1_weighted": 0.808, "roc_auc": 0.864},
      "baseline_comparison": {
        "metric": "roc_auc",
        "split": "test",
        "baseline": 0.501,
        "candidate": 0.864,
        "delta": 0.363,
        "improved": true
      },
      "notes": ""
    }
  },
  "evaluation": {
    "primary_metric": "roc_auc",
    "split": "test",
    "baseline_vs_candidate": [
      {"metric": "accuracy",    "split": "test", "baseline": 0.511, "candidate": 0.810, "delta": 0.299},
      {"metric": "f1_weighted", "split": "test", "baseline": 0.503, "candidate": 0.808, "delta": 0.305},
      {"metric": "roc_auc",     "split": "test", "baseline": 0.501, "candidate": 0.864, "delta": 0.363}
    ],
    "artifacts": {
      "confusion_matrix": "outputs/artifacts/run_20240115_a3f7_confusion_matrix.png",
      "roc_curve":        "outputs/artifacts/run_20240115_a3f7_roc_curve.png",
      "feature_importance_csv":  "outputs/artifacts/run_20240115_a3f7_feature_importance.csv",
      "feature_importance_plot": "outputs/artifacts/run_20240115_a3f7_feature_importance.png",
      "test_predictions":        "outputs/artifacts/run_20240115_a3f7_test_predictions.csv"
    }
  },
  "transformers": {
    "preprocessor": "outputs/artifacts/run_20240115_a3f7_preprocessor.joblib",
    "fitted_on": "X_train",
    "n_samples_fit": 623
  }
}
```

### Metric Object Schema

Every metric value, wherever it appears in `state`, must follow this schema:

```json
{
  "metric_name": "roc_auc",
  "value": 0.864,
  "split": "test"
}
```

A metric object without a `split` field is invalid. The Inspector will reject it.

---

## Do Not

- **Do not fit any preprocessor or transformer on the full dataset before splitting.** This is the single most common leakage source. `preprocessor.fit()` takes `X_train` as input. No other form is acceptable.
- **Do not skip the baseline model.** The baseline must be trained, evaluated, and recorded before the candidate model. A `state.models` object with only `candidate` and no `baseline` is invalid.
- **Do not use the target variable, ID columns, or leakage-flagged columns as features.** The Step 2 and Step 3 exclusion checks must be performed before any data reaches a model. If a column slips through, halt immediately.
- **Do not report only training performance.** `state.models.*.train_metrics` may be recorded for diagnostic purposes, but the final reported metrics in `state.evaluation` must be from the test split (`split: "test"`). Validation metrics (`split: "val"`) are used for model selection only.
- **Do not access the test set before Step 9.** `X_test` and `y_test` must not be passed to any model's `fit()`, `fit_transform()`, cross-validation, or hyperparameter search. They are sealed until the evaluation step.
- **Do not omit split labels from any metric object.** Every numeric metric value must include its `split` field. A bare number with no split context is meaningless and will be flagged by the Inspector and the Report Reviewer.
- **Do not proceed if `leakage_audit.approved` is false.** No model training — not even the baseline — may begin before the leakage audit has returned `approved: true`.
- **Do not report a metric on a set that was used for any fitting or selection decision.** If `X_val` was used to select hyperparameters, it cannot serve as the held-out evaluation set. Use `X_test` for all final reported performance.
