---
name: report-writing
description: Use this skill when generating a data analysis report from structured analysis outputs, EDA findings, model results, predictions, figures, and inspection logs. It ensures claims are evidence-based and avoids unsupported causal language.
---

# Skill: report-writing

## When to Use

Use this skill during the `report_generate` step, after all execution steps have completed and `state.execution_log` shows no unresolved FAIL verdicts.

**Pre-conditions — all must be true before writing begins:**

| Condition | State Field to Check |
|-----------|---------------------|
| Data profiling complete | `state.data_profile` is non-null |
| Task type resolved | `state.task.task_type` is one of the four supported values |
| Plan approved | `state.plan.approved.status == "approved"` |
| Leakage audit passed | `state.leakage_audit.approved == true` |
| EDA results present | `state.eda_results` is non-null |
| Evaluation complete (supervised) | `state.evaluation` is non-null and `state.evaluation.approved == true` |
| All execution steps passed | `state.execution_log` contains no unresolved FAIL entries |

If any pre-condition is not met, halt and return a `ReportWritingError` naming the missing field. Do not attempt to write a report from incomplete state.

---

## Inputs

| Field | Source | Required |
|-------|--------|----------|
| `user_request.goal` | `state.user_request.goal` | Yes |
| `task` | `state.task` | Yes |
| `data_profile` | `state.data_profile` | Yes |
| `eda_results` | `state.eda_results` | Yes |
| `evaluation` | `state.evaluation` | Conditional — required for supervised tasks |
| `models.baseline` | `state.models.baseline` | Conditional — required for supervised tasks |
| `models.candidate` | `state.models.candidate` | Conditional — required for supervised tasks |
| `interpretation` | `state.interpretation` | Yes |
| `leakage_audit` | `state.leakage_audit` | Yes |
| `execution_log` | `state.execution_log` | Yes |
| `artifacts` | `state.artifacts` | Yes |
| `run_id` | `state.run_id` | Yes |
| `splits` | `state.splits` | Conditional — required for supervised tasks |

---

## Default Report Structure

Every report must contain all sections below, in this order. A section may be abbreviated if the task type makes it inapplicable (e.g., `Modeling and Prediction Results` is replaced by a one-line note for descriptive tasks), but it must still be present. No section may be omitted entirely.

---

### Section 1 — Executive Summary

**Required content:**
- One paragraph (4–8 sentences) summarizing the entire analysis from objective through key conclusions.
- State the task type and target variable (if any).
- State the single most important finding or model performance result.
- State the primary limitation.

**Evidence required:** None — this section summarizes Sections 2–10. Numbers cited here must match their source sections exactly.

**Length:** 100–200 words.

---

### Section 2 — Objective

**Required content:**
- Restate the user's original analysis goal verbatim (in a quote block).
- Describe the analytical interpretation of that goal: what task type was inferred and why.
- State what the analysis does and does not attempt to answer.

**Evidence required:** `state.user_request.goal`, `state.task.task_type`, `state.task.rationale`.

---

### Section 3 — Data Overview

**Required content:**
- Source file name and format.
- Number of rows and columns.
- Column names and types (summarized in a table).
- Date of analysis.
- Split sizes, random seed, and split rationale (supervised tasks only).

**Evidence required:** `state.data_profile.n_rows`, `state.data_profile.n_columns`, `state.data_profile.dtypes`, `state.splits` (supervised).

**Format:** At minimum one markdown table listing column name, simplified type, and n_unique or range.

---

### Section 4 — Data Quality Assessment

**Required content:**
- Missing values: table of columns with `missing_rate > 0`, showing count and percentage.
- Duplicate rows: count and percentage.
- Identified potential ID columns and their disposition (excluded from features).
- High-cardinality columns and how they were handled.
- Any `critical_missing` columns (`missing_rate >= 0.50`) must be explicitly named and their handling described.
- Leakage audit summary: verdict and any columns flagged.

**Evidence required:** `state.data_profile.missing_summary`, `state.data_profile.duplicate_count`, `state.data_profile.potential_id_columns`, `state.leakage_audit`.

**Format:** Missingness table required. Leakage audit verdict must be stated explicitly (PASS / WARN with detail).

---

### Section 5 — Methods

**Required content:**
- Preprocessing steps applied (imputation strategy, encoding strategy, scaling).
- Feature selection approach and final feature list.
- Train/val/test split ratios and random seed (supervised tasks only).
- Model(s) trained with algorithm names and key hyperparameters.
- Evaluation metrics selected and rationale for selection.
- Any special handling: class imbalance treatment, outlier strategy, high-cardinality encoding.

**Evidence required:** `state.plan.approved`, `state.transformers`, `state.models.baseline.hyperparameters`, `state.models.candidate.hyperparameters`, `state.evaluation.selection_metric`.

---

### Section 6 — Exploratory Findings

**Required content:**
- 3–7 key observations from EDA, each supported by a specific figure or statistic.
- Distribution summary for the target variable (supervised tasks).
- Correlation or association findings relevant to the analysis goal.
- Any anomalies, outliers, or unexpected patterns found during EDA.
- All figures referenced must be listed in `referenced_artifacts` with their paths.

**Evidence required:** `state.eda_results`, artifact paths in `state.artifacts`.

**Language rule:** Every claim in this section must cite a figure, table, or computed statistic. No observation may be stated as "appears to" or "seems to" without a backing number.

---

### Section 7 — Modeling and Prediction Results

**For supervised tasks (`binary_classification`, `multiclass_classification`, `regression`):**

**Required content:**
- Baseline model name, algorithm, and test-set metrics (all required metrics for task type).
- Candidate model name, algorithm, and test-set metrics.
- A comparison table showing both models side-by-side on all metrics, labeled with `split: test`.
- Primary metric improvement: delta between candidate and baseline, as an absolute value and percentage.
- Confusion matrix (classification) or residual summary (regression).
- A one-sentence plain-English interpretation of the performance result.

**Evidence required:** `state.evaluation.performance_table`, `state.evaluation.delta_vs_baseline`, `state.evaluation.confusion_matrix` or `state.evaluation.residual_summary`.

**Comparison table format:**

```
| Model                         | Split | ROC-AUC | F1 (weighted) | Accuracy |
|-------------------------------|-------|---------|---------------|----------|
| DummyClassifier (baseline)    | test  | 0.501   | 0.503         | 0.511    |
| RandomForestClassifier (cand) | test  | 0.864   | 0.808         | 0.810    |
| Delta                         | —     | +0.363  | +0.305        | +0.299   |
```

Every metric value in the table must match `state.evaluation.performance_table` exactly. No rounding beyond two decimal places.

**For descriptive tasks:**

Replace this section with a summary of key descriptive statistics. No model comparison table. Write: *"This analysis is descriptive. No predictive model was trained."*

---

### Section 8 — Interpretation

**Required content:**
- Feature importance or key drivers (supervised tasks): table of top features with importance scores, accompanied by the feature importance plot.
- Plain-English description of what the model has learned, stated as associations or predictive patterns — not causes.
- For regression: direction and magnitude of top features' relationship with the target.
- For classification: which features most distinguish the predicted classes.
- For descriptive: the most important patterns and relationships found.

**Evidence required:** `state.interpretation.feature_importance` (supervised), `state.eda_results.key_findings`.

**Language rule — see Section 5 (Language Rules).** This section is the highest-risk section for causal language. Every sentence must be reviewed before finalizing.

---

### Section 9 — Limitations

**Required content — all applicable items must be present:**

| Limitation Category | What to Address |
|--------------------|-----------------|
| **Data quality** | Missing data columns and rates; potential impact on findings |
| **Sample size** | Number of rows; whether sample is sufficient for the task |
| **Missingness** | Specific columns with `missing_rate > 0.10` and potential bias introduced |
| **Temporal scope** | Whether data is from a specific time period that may limit generalization |
| **Feature coverage** | Variables that were excluded and why; variables that might have been useful but were unavailable |
| **Modeling assumptions** | What the chosen model assumes (linearity, feature independence, stationarity) and whether those hold |
| **Generalizability** | Whether the model is expected to perform equally on data from different distributions, time periods, or populations |
| **Causal interpretation** | Explicit statement that findings are associative/predictive, not causal |

This section must always be substantive. A one-line "limitations are minimal" is not acceptable.

**Evidence required:** `state.data_profile.missing_summary`, `state.data_profile.n_rows`, `state.task.warnings`, `state.evaluation.warnings` (if any).

---

### Section 10 — Recommendations

**Required content:**
- 2–5 concrete, actionable recommendations based directly on the findings.
- Each recommendation must cite the specific finding that motivates it.
- Distinguish between data recommendations (collect more data, address missingness) and analytical recommendations (further investigation, different model type) and business recommendations (if the goal warrants them).

**Language rule:** Recommendations must be stated conditionally or as suggestions — not as guaranteed outcomes. Do not write "doing X will result in Y" unless a causal method was applied.

---

### Section 11 — Appendix

**Required content:**
- Full list of features used in modeling with their types.
- Full preprocessing pipeline description (transformers, parameters).
- All artifact paths generated in this run, formatted as a table.
- Run metadata: `run_id`, analysis date, model versions used.

**Format:**

```
| Artifact | Path |
|----------|------|
| Confusion matrix | outputs/artifacts/<run_id>_confusion_matrix.png |
| ROC curve        | outputs/artifacts/<run_id>_roc_curve.png        |
| Feature importance | outputs/artifacts/<run_id>_feature_importance.png |
| Test predictions | outputs/artifacts/<run_id>_test_predictions.csv   |
| Preprocessor     | outputs/artifacts/<run_id>_preprocessor.joblib    |
```

---

## Evidence Rules

Every factual claim, number, or performance statement in the report must be traceable to a specific field in `state` or to a named artifact. Fabricated or estimated numbers are not permitted.

### Number Traceability

| Claim Type | Required Source |
|-----------|----------------|
| Row/column counts | `state.data_profile.n_rows`, `state.data_profile.n_columns` |
| Missing rates | `state.data_profile.missing_summary[col].missing_rate` |
| Model metrics | `state.evaluation.performance_table` (must match exactly) |
| Feature importance scores | `state.interpretation.feature_importance` |
| Split sizes | `state.splits` |
| EDA statistics | `state.eda_results` |

If a number appears in the report that cannot be traced to one of the above, it must be removed or replaced with a traceable value.

### Figure Traceability

Every figure referenced in the report (e.g., "Figure 1: Confusion Matrix") must:
- Have a path in `state.artifacts`.
- Be listed in `referenced_artifacts` in the output.
- Have `os.path.exists(path) == True` at time of report writing.

A figure that does not exist as an artifact must not be referenced in the report. Write around missing figures — do not use placeholder references like "see figure below (to be added)".

### Limitations Evidence

The Limitations section must address each item in `state.task.warnings`, `state.data_profile.warnings`, and `state.evaluation.warnings`. Every warning generated during the analysis must either appear in the Limitations section or be explicitly noted as resolved with a justification.

---

## Language Rules

These rules are absolute. The Report Reviewer will check every sentence in Sections 6, 7, 8, and 10 against them.

### Rule 1 — No Causal Language for Observational Findings

Do not use causal language when reporting observational findings, model outputs, or feature importance.

**Prohibited phrases:**

| Prohibited | Use Instead |
|-----------|-------------|
| "X causes Y" | "X is associated with Y" / "X predicts Y" |
| "X leads to Y" | "higher X is associated with higher Y" |
| "the effect of X on Y" | "the relationship between X and Y" |
| "X results in Y" | "X is a strong predictor of Y" |
| "due to X, Y increases" | "when X is higher, Y tends to be higher" |
| "X drives Y" | "X is among the top predictors of Y" |
| "treating X will change Y" | "the model associates X with Y" |

The only exception: if a formal causal inference method (instrumental variables, difference-in-differences, propensity score matching, etc.) was applied and documented in the Methods section, the resulting effect estimates may be described in causal language, clearly scoped to the causal method used.

### Rule 2 — No Over-Claiming Generalization

Do not imply that a model trained on this dataset will generalize to populations, time periods, or distributions not represented in the data.

**Prohibited:**
- "This model can predict X for any patient / customer / user."
- "These results apply to the general population."
- "The model is production-ready."

**Permitted:**
- "Within the study dataset, the model achieves…"
- "Generalization to other populations should be validated before deployment."
- "Results may not generalize outside the conditions represented in this dataset."

### Rule 3 — No Hidden Limitations

If any of the following are true, the Limitations section must explicitly say so:
- Any column had `missing_rate > 0.10`.
- `n_rows < 1000` for a supervised task.
- Any WARN-level finding in `state.leakage_audit`.
- Candidate model did not improve over baseline.
- Class imbalance was detected.
- Target was inferred (not explicitly named by the user).

Omitting a known limitation when writing the report is a violation. The Report Reviewer will check `state.task.warnings` and `state.data_profile.warnings` against the Limitations section.

### Rule 4 — Uncertainty Must Be Expressed

If the task confidence was below 0.85, or if the primary metric improvement over baseline was less than 0.05 (absolute), or if `n_rows < 500`, the report must include explicit uncertainty language:

- "Given the small sample size (N=XXX), these results should be interpreted with caution."
- "The model's improvement over baseline is modest (ΔAUC = 0.03), and results may not be stable across different samples."
- "Task type was inferred with moderate confidence; findings should be reviewed in the context of the original goal."

---

## Output Format

Write the completed report to `state.report_draft`, convert it to PDF, and save both files to disk.

**Conversion steps (required):**
1. Write the full Markdown text to `outputs/reports/<run_id>_report.md`
2. Call `scripts/md_to_pdf.py` via `from md_to_pdf import convert as md_to_pdf; md_to_pdf(md_path, pdf_path)` — this uses `reportlab` Platypus to produce a styled, paginated PDF with tables, headings, bullet lists, and code blocks
3. Verify the PDF file exists on disk (`os.path.exists`) before returning

If PDF conversion fails, log the error in `state.execution_log` and fall back to Markdown only, setting `report_format` to `"markdown"` and recording a WARN.

**Dependencies:** `reportlab` (pre-installed). The `scripts/md_to_pdf.py` utility handles Unicode sanitization, inline markup (bold/italic/code), table rendering, horizontal rules, blockquotes, and numbered/bullet lists.

```json
{
  "report_draft": "# Data Analysis Report\n\n## 1. Executive Summary\n...",
  "report_path": "outputs/reports/<run_id>_report.pdf",
  "report_format": "pdf",
  "referenced_artifacts": [
    {
      "label": "Figure 1: Confusion Matrix",
      "path": "outputs/artifacts/<run_id>_confusion_matrix.png",
      "section": "Section 7"
    },
    {
      "label": "Figure 2: ROC Curve",
      "path": "outputs/artifacts/<run_id>_roc_curve.png",
      "section": "Section 7"
    },
    {
      "label": "Figure 3: Feature Importance",
      "path": "outputs/artifacts/<run_id>_feature_importance.png",
      "section": "Section 8"
    },
    {
      "label": "Table A1: Test Predictions",
      "path": "outputs/artifacts/<run_id>_test_predictions.csv",
      "section": "Appendix"
    }
  ],
  "sections_present": [
    "executive_summary", "objective", "data_overview", "data_quality",
    "methods", "exploratory_findings", "modeling_results",
    "interpretation", "limitations", "recommendations", "appendix"
  ],
  "word_count": 1842,
  "warnings_addressed": [
    "critical_missing:Cabin",
    "potential_id_column:PassengerId",
    "target_inferred:Survived"
  ]
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `report_draft` | `string` | Full report text in Markdown |
| `report_path` | `string` | Path where the PDF report is saved (`.pdf`); falls back to `.md` if PDF conversion fails |
| `report_format` | `string` | `"pdf"` (primary); `"markdown"` only if PDF conversion fails |
| `referenced_artifacts` | `list[object]` | All figures and tables cited in the report, with paths and section labels |
| `sections_present` | `list[string]` | All 11 section identifiers; must have all 11 entries |
| `word_count` | `int` | Approximate word count of the full report |
| `warnings_addressed` | `list[string]` | Each warning from state that was addressed in the Limitations section, in format `"type:column"` |

---

## Do Not

- **Do not write any metric value that is not present in `state.evaluation`.** If a metric was not computed, do not estimate or approximate it. Write around the gap or acknowledge the limitation.
- **Do not present training-set performance as model performance.** If only training metrics are available, state that explicitly and flag it as a limitation. Do not present them as predictive accuracy.
- **Do not omit the Limitations section or make it trivial.** Every warning produced during the analysis must be acknowledged. A report with a one-line limitations section will be rejected by the Report Reviewer.
- **Do not reference figures that do not exist as artifact files.** Verify every path in `referenced_artifacts` before writing the report. A broken figure reference is grounds for FAIL during report review.
- **Do not write "X causes Y" or any equivalent for observational findings.** Association and prediction language is required everywhere in the report except for formally documented causal methods.
- **Do not claim the model generalizes beyond the training distribution.** All performance claims are scoped to the dataset used. Generalization is always stated as requiring additional validation.
- **Do not write the report before all pre-conditions are met.** Starting report generation with an unresolved FAIL in `state.execution_log` produces a report that cannot be validated. Return a `ReportWritingError` instead.
- **Do not use placeholder values or "[TBD]" text.** Every field in the report must be filled with actual values from state. If a value is genuinely unavailable, explain why in the text — do not mark it as pending.
