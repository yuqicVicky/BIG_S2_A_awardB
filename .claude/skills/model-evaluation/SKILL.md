---
name: model-evaluation
description: Use this skill when selecting, computing, interpreting, or reviewing model metrics for classification and regression tasks. It ensures metrics match the task type, class imbalance is considered, and evaluation is performed only on held-out data.
---

# Skill: model-evaluation

## When to Use

Use this skill at three distinct points in the workflow, each with a different scope:

| Invocation Point | When | Scope |
|-----------------|------|-------|
| During `modeling` (inline) | After baseline and candidate are trained | Compute val metrics; compare candidate vs baseline |
| During `evaluation` step | After both models are finalized | Compute test metrics; produce evaluation artifacts |
| During `report_review` | Before report is approved | Verify metric values in report match `state.evaluation` exactly |

Do **not** use this skill to compute metrics on training data and report them as final performance. Training metrics are diagnostic only.

Do **not** invoke this skill before `state.splits` is confirmed non-empty and `state.leakage_audit.approved == true`.

---

## Metric Rules by Task Type

### `binary_classification`

All metrics must be computed on the validation set during model selection and on the test set for final evaluation. Every metric object must carry a `split` field.

#### Required Metrics

| Metric | Function | When Primary |
|--------|----------|-------------|
| **ROC-AUC** | `roc_auc_score(y_true, y_proba[:, 1])` | Always — threshold-independent |
| **PR-AUC** | `average_precision_score(y_true, y_proba[:, 1])` | When positive class is rare (`positive_rate < 0.20`) |
| **F1 (weighted)** | `f1_score(y_true, y_pred, average='weighted')` | When class imbalance is present |
| **F1 (binary)** | `f1_score(y_true, y_pred, average='binary')` | When positive class is the primary interest |
| **Sensitivity (Recall)** | `recall_score(y_true, y_pred)` | When false negatives are costly (medical, fraud) |
| **Specificity** | `TN / (TN + FP)` | Reported alongside sensitivity |
| **Precision** | `precision_score(y_true, y_pred)` | When false positives are costly |
| **Accuracy** | `accuracy_score(y_true, y_pred)` | Secondary — always report with class base rate |
| **Confusion Matrix** | `confusion_matrix(y_true, y_pred)` | Required artifact — not a scalar metric |

#### Class Imbalance Rules

Compute the majority class rate: `majority_rate = max(class_counts) / n_samples`.

| `majority_rate` | Imbalance Level | Required Action |
|-----------------|-----------------|-----------------|
| `< 0.70` | Balanced | Report accuracy as primary alongside ROC-AUC |
| `0.70 – 0.80` | Mild imbalance | Report ROC-AUC as primary; demote accuracy to secondary |
| `0.80 – 0.90` | Moderate imbalance | Report PR-AUC alongside ROC-AUC; add WARNING; do not use accuracy alone |
| `> 0.90` | Severe imbalance | Report PR-AUC as co-primary; add FAIL-level WARNING if report uses accuracy as the sole or primary metric |

When imbalance is detected, record `state.evaluation.class_imbalance_detected: true` and `state.evaluation.majority_class_rate: <value>`.

#### Required Artifacts

- Confusion matrix plot (normalized by true label).
- ROC curve plot with AUC annotated.
- PR curve plot (if `positive_rate < 0.20` or `majority_rate > 0.80`).

---

### `multiclass_classification`

#### Required Metrics

| Metric | Function | Notes |
|--------|----------|-------|
| **F1 (weighted)** | `f1_score(y_true, y_pred, average='weighted')` | Primary — accounts for class frequency |
| **F1 (macro)** | `f1_score(y_true, y_pred, average='macro')` | Required — penalizes majority-class bias |
| **Balanced Accuracy** | `balanced_accuracy_score(y_true, y_pred)` | Required — average recall across classes |
| **Accuracy** | `accuracy_score(y_true, y_pred)` | Secondary — report per-class base rate alongside |
| **Per-class Precision** | `precision_score(..., average=None)` | Required — one value per class |
| **Per-class Recall** | `recall_score(..., average=None)` | Required — one value per class |
| **Per-class F1** | `f1_score(..., average=None)` | Required — one value per class |
| **Confusion Matrix** | `confusion_matrix(y_true, y_pred)` | Required artifact |

#### Class Imbalance in Multiclass

Compute per-class frequency. If any class has `class_rate < 0.10` and `n_classes >= 3`:
- Flag `minority_classes` in `state.evaluation.warnings`.
- Promote macro-F1 over weighted-F1 as primary metric.
- Produce a WARNING: `"Minority class detected. Macro-F1 and per-class recall are more informative than accuracy."`

#### Required Artifacts

- Confusion matrix plot (normalized by true label, all classes labeled).

---

### `regression`

#### Required Metrics

| Metric | Function | Notes |
|--------|----------|-------|
| **RMSE** | `np.sqrt(mean_squared_error(y_true, y_pred))` | Primary — penalizes large errors |
| **MAE** | `mean_absolute_error(y_true, y_pred)` | Primary — interpretable in target units |
| **R²** | `r2_score(y_true, y_pred)` | Required — proportion of variance explained |
| **MAPE** | `mean_absolute_percentage_error(y_true, y_pred)` | Conditional: only if `min(y_true) > 0` |
| **Residual summary** | `y_pred - y_true` statistics | Mean, std, min, max, p5, p95 of residuals |

#### RMSE vs MAE Divergence Rule

If `RMSE / MAE > 2.0`, this indicates the presence of large prediction errors on a subset of samples. Both metrics must be reported and the divergence flagged:

```
WARNING: RMSE/MAE ratio = <value>. Large residuals detected on a subset of samples.
Report both RMSE and MAE. Do not report only one.
```

#### Negative R² Rule

If `R² < 0`, the model performs worse than a mean predictor baseline. This must be flagged:
- Set `state.evaluation.worse_than_baseline: true`.
- Add a FAIL-level WARNING: `"Candidate model R² < 0. Model performs worse than the mean-predictor baseline. Do not report as a useful model."`
- The report must not present a negative-R² model as a successful result.

#### Required Artifacts

- Residual plot: residuals (`y_pred − y_true`) on y-axis vs `y_pred` on x-axis. Include a horizontal reference line at 0.
- Prediction vs actual scatter plot: `y_pred` on y-axis vs `y_true` on x-axis. Include a 45-degree reference line.

---

### `descriptive`

For descriptive tasks, there are no predictive metrics. The following are required instead:

| Output | Description |
|--------|-------------|
| **Summary statistics** | Per-column: count, mean, std, min, p25, median, p75, max (numeric); count, n_unique, top, freq (categorical) |
| **Missingness summary** | Missing count and rate per column |
| **Distribution plots** | Histogram or KDE for numeric columns; bar chart for categorical columns |
| **Correlation matrix** | Pearson for numeric columns; Cramér's V or association stats for categorical |
| **Key findings** | 3–7 bullet points summarizing notable patterns, anomalies, and relationships |

No model performance table is produced for `descriptive` tasks. `state.evaluation.performance_table` must be `null` and `state.evaluation.best_model` must be `null`.

---

## Evaluation Validity Checks

Run all applicable checks before writing `state.evaluation`. A check that fails must be recorded in `failed_checks` and, if FAIL-severity, sets `approved: false`.

### Check A — Held-Out Data Only

**Rule:** All metrics in `state.evaluation` must be computed on `X_test` / `y_test`. No metric computed on training or validation data may appear in the final evaluation output.

**How to verify:**
- Every metric object must have `"split": "test"`.
- The execution log must show that test data was first accessed in the `evaluate_test` step.
- If any metric carries `"split": "train"` or `"split": "val"`, it must not appear in `performance_table`.

**Severity:** FAIL if any final metric is not on the test set.

### Check B — Metric Matches Task Type

**Rule:** The metrics used must be appropriate for `state.task.task_type`.

**Violations:**
- Reporting accuracy as the sole metric for a classification task with `majority_rate > 0.80`.
- Reporting R² for a classification task.
- Reporting AUC-ROC for a regression task.
- Reporting RMSE for a classification task.

**Severity:** FAIL for metric-task mismatch. WARN for sole accuracy on imbalanced binary data.

### Check C — Baseline Comparison Present

**Rule:** The performance table must include both baseline and candidate metrics side-by-side. A candidate-only table is invalid.

**Severity:** FAIL if baseline metrics are absent from `performance_table`.

### Check D — Selection Metric Matches User Goal

**Rule:** The metric used to select the best model must align with the user's stated goal.

| User Goal Contains | Recommended Selection Metric |
|-------------------|------------------------------|
| "minimize false negatives", "catch all cases", "recall" | Sensitivity (Recall) |
| "minimize false positives", "precision" | Precision |
| "fraud", "anomaly", "rare event" | PR-AUC |
| "general performance", "balanced" | ROC-AUC (classification) or RMSE (regression) |
| No specific preference stated | Default: ROC-AUC (binary), F1 weighted (multiclass), RMSE (regression) |

If the selection metric does not match the user goal, produce a WARN and note the mismatch in `state.evaluation.warnings`.

**Severity:** WARN (does not block approval, but must be recorded and surfaced in the report).

### Check E — Split Labels Present on All Metrics

**Rule:** Every scalar metric value in `state.evaluation` must have an associated `split` field. A metric without a split label is unverifiable.

**Severity:** FAIL if any metric is missing its `split` field.

### Check F — No Training Metrics Presented as Final

**Rule:** If `state.models.*.train_metrics` exists, it must be clearly labeled as `split: "train"` and must not appear in `performance_table` alongside test metrics without a clear label distinguishing them.

**Severity:** WARN if train metrics are present in the output without clear labeling. FAIL if train metrics are the only metrics reported.

---

## Model Selection Rules

After computing val metrics for both baseline and candidate models, apply these rules to select the final model.

1. **Primary metric governs selection.** The model with the higher primary metric on the validation set is selected. The primary metric for each task type:

   | Task Type | Primary Metric | Higher = Better |
   |-----------|---------------|-----------------|
   | `binary_classification` | ROC-AUC | Yes |
   | `binary_classification` (imbalanced, `majority_rate > 0.80`) | PR-AUC | Yes |
   | `multiclass_classification` | F1 (weighted) | Yes |
   | `multiclass_classification` (minority class present) | F1 (macro) | Yes |
   | `regression` | RMSE | No (lower = better) |

2. **Tie-breaking.** If primary metrics are within 0.005 (classification) or 1% of the RMSE range (regression), prefer the simpler model (baseline over candidate).

3. **Candidate must beat baseline.** If the candidate model does not improve over the baseline on the primary metric, do not discard the candidate — but record a WARNING: `"Candidate model does not improve over baseline on primary metric. Consider returning to modeling with a different algorithm or feature set."`. The decision to proceed rests with the Orchestrator, not this skill.

4. **Model selection must use val metrics only.** The test set must not influence model selection. If the selection was based on test metrics, this is a data leakage violation — flag it with a FAIL in `failed_checks`.

5. **Record selection rationale.** `state.evaluation.selection_rationale` must explain which metric, which split, and which value led to the selection.

---

## Output Format

Write results to `state.evaluation`.

```json
{
  "task_type": "binary_classification",
  "split_used_for_final_evaluation": "test",
  "class_imbalance_detected": false,
  "majority_class_rate": 0.616,
  "worse_than_baseline": false,
  "metrics_used": ["roc_auc", "f1_weighted", "accuracy", "precision_weighted", "recall_weighted"],
  "selection_metric": "roc_auc",
  "selection_split": "val",
  "selection_rationale": "Candidate RandomForestClassifier achieved val ROC-AUC of 0.876 vs baseline 0.500. Candidate selected.",
  "performance_table": [
    {
      "model":        "DummyClassifier (baseline)",
      "split":        "test",
      "roc_auc":      0.501,
      "f1_weighted":  0.503,
      "accuracy":     0.511,
      "precision_w":  0.498,
      "recall_w":     0.511
    },
    {
      "model":        "RandomForestClassifier (candidate)",
      "split":        "test",
      "roc_auc":      0.864,
      "f1_weighted":  0.808,
      "accuracy":     0.810,
      "precision_w":  0.812,
      "recall_w":     0.810
    }
  ],
  "best_model": "RandomForestClassifier",
  "best_model_key": "candidate",
  "delta_vs_baseline": {
    "roc_auc":     {"baseline": 0.501, "candidate": 0.864, "delta": 0.363, "split": "test"},
    "f1_weighted": {"baseline": 0.503, "candidate": 0.808, "delta": 0.305, "split": "test"},
    "accuracy":    {"baseline": 0.511, "candidate": 0.810, "delta": 0.299, "split": "test"}
  },
  "confusion_matrix": {
    "labels": [0, 1],
    "matrix": [[93, 15], [19, 51]],
    "split": "test"
  },
  "residual_summary": null,
  "approved": true,
  "failed_checks": [],
  "warned_checks": [],
  "warnings": [],
  "artifacts": {
    "confusion_matrix_plot": "outputs/artifacts/run_20240115_a3f7_confusion_matrix.png",
    "roc_curve_plot":        "outputs/artifacts/run_20240115_a3f7_roc_curve.png",
    "pr_curve_plot":         null,
    "residual_plot":         null,
    "pred_vs_actual_plot":   null
  }
}
```

### Regression Output Additions

For regression tasks, replace `confusion_matrix` and ROC/PR artifact fields with:

```json
{
  "residual_summary": {
    "split": "test",
    "mean":  0.23,
    "std":   4.11,
    "min":  -18.4,
    "p5":    -7.2,
    "p95":    7.8,
    "max":   22.1
  },
  "artifacts": {
    "residual_plot":       "outputs/artifacts/<run_id>_residuals.png",
    "pred_vs_actual_plot": "outputs/artifacts/<run_id>_pred_vs_actual.png"
  }
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `task_type` | `string` | Copied from `state.task.task_type` |
| `split_used_for_final_evaluation` | `string` | Must be `"test"` |
| `class_imbalance_detected` | `boolean` | Classification only |
| `majority_class_rate` | `float \| null` | Classification only; null for regression/descriptive |
| `worse_than_baseline` | `boolean` | True if candidate primary metric < baseline primary metric |
| `metrics_used` | `list[string]` | All metric names computed |
| `selection_metric` | `string` | Metric used to pick best model |
| `selection_split` | `string` | Must be `"val"` — never `"test"` |
| `selection_rationale` | `string` | Plain-English explanation of model selection |
| `performance_table` | `list[object]` | One row per model, all metrics, `split: "test"` |
| `best_model` | `string \| null` | Name of selected model; null for descriptive |
| `best_model_key` | `string \| null` | `"baseline"` or `"candidate"`; null for descriptive |
| `delta_vs_baseline` | `dict \| null` | Per-metric delta; null for descriptive |
| `confusion_matrix` | `object \| null` | Classification only |
| `residual_summary` | `object \| null` | Regression only |
| `approved` | `boolean` | `true` only if `failed_checks` is empty |
| `failed_checks` | `list[object]` | Must be empty for `approved: true` |
| `warned_checks` | `list[object]` | Does not block approval |
| `warnings` | `list[object]` | All warnings including warn-level check failures |
| `artifacts` | `dict` | All artifact paths; null for artifacts not applicable to this task type |

---

## Do Not

- **Do not compute metrics on training data and present them as final.** Train metrics are for diagnostic purposes only. They must be labeled `split: "train"` if recorded and must not appear in `performance_table`.
- **Do not use accuracy as the sole or primary metric for imbalanced binary classification.** If `majority_class_rate > 0.80`, accuracy is misleading. The report must use ROC-AUC or PR-AUC as the primary framing metric. An evaluation that leads with accuracy on an imbalanced dataset will be flagged as FAIL by the Report Reviewer.
- **Do not omit the baseline from the performance table.** A `performance_table` with only the candidate model is incomplete. Both baseline and candidate must appear in every comparison.
- **Do not select the best model using test set metrics.** Model selection must use validation metrics (`split: "val"`). Using test performance for selection is a form of leakage. Set `selection_split: "val"` always.
- **Do not report metrics without split labels.** Every metric value must carry its `split` field. A number without a split context cannot be verified and will be rejected during report review.
- **Do not suppress or skip Check A (held-out data only).** This check is mandatory at every invocation. It may not be waived for any reason.
- **Do not report R² as the only regression metric.** R² without RMSE and MAE lacks interpretability in the target's units. All three must be reported together.
- **Do not present a negative-R² model as successful.** If `R² < 0`, the evaluation must explicitly state that the model is worse than a mean predictor baseline, regardless of whether the user's original goal implied this level of scrutiny.
