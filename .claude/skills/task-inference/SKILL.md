---
name: task-inference
description: Use this skill when deciding the analysis task type from a user request and data profile, including target variable inference, target type detection, descriptive analysis, classification, regression, forecasting, survival analysis, and recommended metrics.
---

# Skill: task-inference

## When to Use

Use this skill after `data-profiling` completes and before `initial_planning` begins. It must run on:
- `state.user_request.goal`: the raw user-provided analysis goal.
- `state.data_profile`: the completed profile from `data-profiling`.

Trigger this skill whenever:
- The Orchestrator needs to determine what kind of analysis to perform.
- The user's goal is ambiguous and must be resolved before a plan can be written.
- `plan_critique` questions the task type chosen during planning.

Do **not** run this skill before `data-profiling` is complete. Task inference without a data profile is guesswork.

---

## Inputs

| Field | Source | Required |
|-------|--------|----------|
| `user_request.goal` | `state.user_request.goal` | Yes |
| `data_profile.columns` | `state.data_profile.columns` | Yes |
| `data_profile.dtypes` | `state.data_profile.dtypes` | Yes |
| `data_profile.simplified_types` | `state.data_profile.simplified_types` | Yes |
| `data_profile.numeric_columns` | `state.data_profile.numeric_columns` | Yes |
| `data_profile.categorical_columns` | `state.data_profile.categorical_columns` | Yes |
| `data_profile.target_like_columns` | `state.data_profile.target_like_columns` | Yes |
| `data_profile.n_rows` | `state.data_profile.n_rows` | Yes |
| `data_profile.warnings` | `state.data_profile.warnings` | Yes |

If any required input is missing, return a `task_type` of `unknown` with a `confidence` of `0.0` and record the missing field in `warnings`.

---

## Required Checks

Run every check in order. Each check informs the final `task_type` and `confidence` score.

### Check 1 — Explicit Target Variable in User Request

Scan `user_request.goal` for explicit target naming patterns:

- Direct naming: `"predict X"`, `"classify X"`, `"forecast X"`, `"model X"`, `"regress on X"`, `"target is X"`, `"outcome: X"`
- Implicit naming: `"whether X"`, `"probability of X"`, `"likelihood of X"`, `"risk of X"`

If a target column name is explicitly mentioned and exists in `data_profile.columns`: set `target_explicit = true`, record `target_variable`.

If a target is mentioned but does not exist in the data: set `target_explicit = false`, produce a WARNING `target_not_found`.

### Check 2 — Target Column Existence in Data

If `target_explicit = true`: verify the named column is present in `data_profile.columns`.

If `target_explicit = false`: search for candidate targets using:
1. `data_profile.target_like_columns` (from data-profiling check 14).
2. Column names matching: `target`, `label`, `y`, `outcome`, `class`, `survived`, `churn`, `default`, `fraud`, `result`, `flag`, `status`, `response`.
3. Binary numeric columns (exactly 2 unique values: 0/1 or True/False).

If exactly one strong candidate is found: record as `target_variable` with `target_inferred = true`.  
If multiple candidates are found: record all in `target_candidates`, set `target_variable = null`, produce a WARNING `ambiguous_target`.  
If no candidate is found and task appears supervised: produce a WARNING `no_target_found`.

### Check 3 — Target Type Detection

If a `target_variable` has been resolved, determine its type:

| Condition | `target_type` |
|-----------|--------------|
| Numeric column, `n_unique > 20` or continuous distribution | `continuous` |
| Numeric or boolean, `n_unique == 2`, values in {0,1} or {True,False} | `binary` |
| Categorical or numeric, `3 <= n_unique <= 20` | `multiclass` |
| Categorical, `n_unique > 20` | `high_cardinality_categorical` → produce WARNING |
| Cannot be determined | `unknown` → produce WARNING |

### Check 4 — Descriptive Analysis Detection

The task is `descriptive` if any of the following are true:
- User goal contains: `"describe"`, `"summarize"`, `"explore"`, `"understand"`, `"profile"`, `"distribution"`, `"overview"`, `"visualize"`, `"EDA"`, `"patterns in"`, `"how does X vary"`.
- No target variable is present or inferable.
- User explicitly says: `"no prediction"`, `"not predicting"`, `"just analysis"`.

If descriptive is detected and no target exists, set `task_type = descriptive` with high confidence.  
If descriptive language is present but a target also exists, prefer supervised task type and note the ambiguity in `warnings`.

### Check 5 — Classification Detection

The task is likely classification if:
- `target_type` is `binary` or `multiclass`, **and**
- User goal contains: `"predict"`, `"classify"`, `"identify"`, `"detect"`, `"will X"`, `"is X"`, `"whether"`, `"probability of"`, `"risk of"`, `"flag"`.
- Or: No explicit verb, but `target_type` is `binary` or `multiclass`.

Map to:
- `binary_classification` if `target_type == binary`
- `multiclass_classification` if `target_type == multiclass`

### Check 6 — Regression Detection

The task is likely regression if:
- `target_type` is `continuous`, **and**
- User goal contains: `"predict"`, `"estimate"`, `"forecast"` (non-temporal), `"how much"`, `"what will X be"`, `"model X"`.
- Or: No explicit verb, but `target_type` is `continuous`.

Map to: `regression`

### Check 7 — Forecasting Detection

The task may be forecasting if:
- A datetime column exists in `data_profile.datetime_columns`, **or**
- Column names suggest time series: `date`, `time`, `period`, `month`, `year`, `week`, `quarter`, `timestamp`.
- User goal contains: `"forecast"`, `"next period"`, `"future"`, `"time series"`, `"trend"`, `"seasonal"`.

**V1 support status:** `forecasting` is **not supported** in V1. If detected, record in `unsupported_but_detected_task_types` with rationale. Do not set `task_type = forecasting`. Inform the user.

### Check 8 — Survival Analysis Detection

The task may be survival analysis if:
- A duration or time-to-event column exists: names containing `duration`, `tenure`, `days_to`, `time_to`, `survival_time`, `follow_up`.
- An event indicator column exists: binary column named `event`, `died`, `churned`, `converted`, `censored`.
- User goal contains: `"survival"`, `"time to event"`, `"churn duration"`, `"hazard"`, `"Kaplan"`.

**V1 support status:** `survival_analysis` is **not supported** in V1. Record in `unsupported_but_detected_task_types`. Do not set `task_type = survival_analysis`. Inform the user.

### Check 9 — Causal Inference Detection

The task may be causal inference if:
- User goal contains: `"effect of"`, `"impact of"`, `"does X cause"`, `"causal"`, `"treatment effect"`, `"A/B"`, `"experiment"`, `"counterfactual"`, `"what if we changed"`.

**V1 support status:** `causal_inference` is **not supported** in V1. Record in `unsupported_but_detected_task_types`. Do not set `task_type = causal_inference`.

**Important:** If causal language is detected but the task is otherwise classification or regression, do **not** upgrade the task type. Record a WARNING `causal_language_in_non_causal_task` and note it in the report limitations.

---

## Task Type Rules

Final `task_type` selection follows this priority:

1. If the user explicitly states a task type (e.g., `"run a regression"`) → use that, verify it is supported.
2. If `target_type` is `binary` or `multiclass` → `binary_classification` or `multiclass_classification`.
3. If `target_type` is `continuous` → `regression`.
4. If no target can be identified → `descriptive`.
5. If an unsupported task type is strongly detected and no supported type applies → `unknown` with `confidence < 0.5`.

### V1 Supported Task Types

| `task_type` | Supported | Notes |
|-------------|-----------|-------|
| `descriptive` | Yes | No target required |
| `binary_classification` | Yes | Requires binary target |
| `multiclass_classification` | Yes | Requires 3–20 class target |
| `regression` | Yes | Requires continuous target |
| `forecasting` | No | Detect and report; do not execute |
| `survival_analysis` | No | Detect and report; do not execute |
| `causal_inference` | No | Detect and report; do not execute |

If the inferred task type is unsupported, set `task_type = unknown`, add the detected type to `unsupported_but_detected_task_types`, and produce a user-facing message explaining V1 scope.

---

## Recommended Metrics

Write recommended evaluation metrics to `state.task.recommended_metrics` based on `task_type` and dataset characteristics.

### `descriptive`

| Metric / Output | Notes |
|-----------------|-------|
| Distribution plots | All numeric and categorical columns |
| Missing value heatmap | — |
| Correlation matrix | Numeric columns |
| Top-N value counts | Categorical columns |

No predictive metrics.

### `binary_classification`

| Metric | Priority | Notes |
|--------|----------|-------|
| AUC-ROC | Primary | Insensitive to class imbalance |
| F1 (weighted) | Primary | — |
| Accuracy | Secondary | Report alongside base rate |
| Precision / Recall | Secondary | Report per class |
| Confusion matrix | Required | — |

If class imbalance detected (`majority_class_rate > 0.80`): add WARNING `class_imbalance`, promote AUC-ROC as the single primary metric, and note that accuracy is misleading.

### `multiclass_classification`

| Metric | Priority | Notes |
|--------|----------|-------|
| F1 (weighted) | Primary | — |
| F1 (macro) | Primary | Penalizes majority-class bias |
| Accuracy | Secondary | Report alongside per-class base rate |
| Confusion matrix | Required | — |

### `regression`

| Metric | Priority | Notes |
|--------|----------|-------|
| RMSE | Primary | Penalizes large errors |
| MAE | Primary | Robust to outliers |
| R² | Primary | Proportion of variance explained |
| Residual plot | Required | — |
| Prediction vs actual plot | Required | — |

If target has outliers (values beyond `mean ± 3 * std`): add WARNING `target_outliers`, recommend reporting both RMSE and MAE and noting their divergence.

---

## Output Format

Write the result to `state.task`. All fields are required.

```json
{
  "task_type": "binary_classification",
  "target_variable": "Survived",
  "target_type": "binary",
  "target_explicit": false,
  "target_inferred": true,
  "target_candidates": [],
  "feature_candidates": [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"
  ],
  "excluded_from_features": [
    "PassengerId", "Name", "Ticket", "Cabin"
  ],
  "exclusion_reasons": {
    "PassengerId": "potential_id_column",
    "Name":        "high_cardinality",
    "Ticket":      "high_cardinality",
    "Cabin":       "critical_missing"
  },
  "confidence": 0.92,
  "recommended_metrics": ["auc_roc", "f1_weighted", "accuracy", "confusion_matrix"],
  "class_imbalance_detected": false,
  "unsupported_but_detected_task_types": [],
  "warnings": [
    {
      "type": "target_inferred",
      "column": "Survived",
      "message": "Target variable was not explicitly named in the user request. Inferred from column name and binary value distribution (0/1). Please confirm.",
      "severity": "WARN"
    }
  ],
  "rationale": "User goal 'predict passenger survival' implies a supervised binary prediction task. 'Survived' is the only binary column and matches target-like naming heuristics. Non-informative columns excluded based on ID and cardinality checks from data profile."
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `task_type` | `string` | One of the V1 supported types, or `unknown` |
| `target_variable` | `string \| null` | Resolved target column name, or `null` if unresolved |
| `target_type` | `string` | `binary`, `multiclass`, `continuous`, `unknown` |
| `target_explicit` | `boolean` | Whether the target was named by the user |
| `target_inferred` | `boolean` | Whether the target was inferred from data profile |
| `target_candidates` | `list[string]` | All candidate columns considered; empty if resolved |
| `feature_candidates` | `list[string]` | Columns recommended for use as features |
| `excluded_from_features` | `list[string]` | Columns excluded with reasons |
| `exclusion_reasons` | `dict` | Maps excluded column name → exclusion reason |
| `confidence` | `float` | 0.0–1.0; confidence in `task_type` assignment |
| `recommended_metrics` | `list[string]` | Metrics to evaluate against |
| `class_imbalance_detected` | `boolean` | Classification only; `false` for other task types |
| `unsupported_but_detected_task_types` | `list[string]` | Detected but unsupported task types |
| `warnings` | `list[object]` | See warning schema below |
| `rationale` | `string` | Plain-English explanation of how `task_type` was reached |

### Warning Schema

```json
{
  "type": "string",
  "column": "string | null",
  "message": "string",
  "severity": "WARN | FAIL"
}
```

Use `severity: FAIL` only for `target_not_found` when the task appears strongly supervised (user explicitly requests prediction but no target exists). All other warnings use `severity: WARN`.

---

## Uncertainty Handling

When confidence is low or the target is ambiguous, record the uncertainty explicitly. Do not resolve ambiguity by silent assumption.

### Confidence Thresholds

| `confidence` | Meaning | Required Action |
|---|---|---|
| `>= 0.85` | High — proceed | Write `task_type` and continue to `initial_planning` |
| `0.60 – 0.84` | Moderate — proceed with caveat | Write `task_type`, add `low_confidence` WARNING, note in report limitations |
| `0.40 – 0.59` | Low — ask user | Write `task_type = unknown`, list top two candidates in `warnings`, **ask the user to confirm** before planning |
| `< 0.40` | Very low — cannot proceed | Write `task_type = unknown`, `confidence`, and a clear message. Halt and request clarification |

### Ambiguous Target Handling

If `target_candidates` contains more than one column and none is strongly preferred:
- Do not pick one arbitrarily.
- List all candidates in `target_candidates`.
- Set `target_variable = null`.
- Set `confidence` according to the threshold table.
- If `confidence < 0.60`: produce a user-facing clarification request naming the candidates and asking the user to specify.

### Unsupported Task Type Handling

If the inferred task type is unsupported:
- Set `task_type = unknown`.
- Add the detected type to `unsupported_but_detected_task_types`.
- Write a `unsupported_task_type` WARNING with a clear explanation of V1 scope and what the user would need to do to use a supported type instead (e.g., simplify forecasting to regression by ignoring temporal order).

---

## Do Not

- **Do not train any model.** Task inference is a reasoning step, not a computation step.
- **Do not generate a full analysis plan.** Write `state.task` only. Planning belongs to `initial_planning`.
- **Do not force-assign a task type when the target is unclear.** If the evidence is insufficient, record `task_type = unknown` and surface the uncertainty. Silent guessing introduces leakage and metric confusion downstream.
- **Do not use causal language.** Even if the user's goal contains causal framing, record the task as `classification` or `regression` and flag `causal_language_in_non_causal_task` in warnings. Do not upgrade the task type to `causal_inference`.
- **Do not exclude columns from features without recording the reason.** Every excluded column must appear in `excluded_from_features` with an entry in `exclusion_reasons`.
- **Do not assume class balance.** Always check `data_profile.categorical_columns[target_variable].top_values` for class distribution and record `class_imbalance_detected` accurately.
