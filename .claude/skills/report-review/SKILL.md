---
name: report-review
description: Use this skill when reviewing final data analysis reports for factual consistency, supported claims, metric correctness, limitation coverage, and avoidance of unsupported causal conclusions.
---

# Skill: report-review

## When to Use

Use this skill during the `report_review` step, after `report-writing` has produced `state.report_draft` and saved the report file to disk.

This skill performs an independent audit of the report. It must be run as if the reviewer has not seen the analysis — reading the report against `state` and artifacts, not from memory of what was computed.

**Pre-conditions — all must be true before review begins:**

| Condition | Field to Check |
|-----------|---------------|
| Report draft exists | `state.report_draft` is non-null |
| Report file exists on disk | `os.path.exists(state.report_path)` |
| Evaluation complete (supervised) | `state.evaluation.approved == true` |
| Leakage audit complete | `state.leakage_audit` is non-null |
| Execution log complete | `state.execution_log` has entries for all plan steps |
| All artifacts referenced in report exist | Verify each path in `state.report_draft` against `state.artifacts` |

If any pre-condition fails, return a `ReviewError` naming the missing item. Do not issue a PASS or FAIL on an incomplete report.

---

## Required Checks

Run every check in order. Each check produces one or more entries in the output. A check may produce a PASS, WARN, or FAIL verdict. All findings must be recorded — nothing may be silently skipped.

---

### Check 1 — Numeric Values Match State

**Question:** Does every numeric value in the report match its source in `state` or an artifact?

**What to check:**
- Every metric value in Section 7 (Modeling and Prediction Results) against `state.evaluation.performance_table`. Match must be exact to two decimal places.
- Every row/column count in Section 3 (Data Overview) against `state.data_profile.n_rows` and `state.data_profile.n_columns`.
- Every missing rate or missing count in Section 4 (Data Quality) against `state.data_profile.missing_summary`.
- Every split size in Sections 3 or 5 against `state.splits`.
- Every feature importance score in Section 8 (Interpretation) against `state.interpretation.feature_importance`.
- Any percentage, count, or statistic cited in Section 6 (Exploratory Findings) against `state.eda_results`.

**How to check:** For each number found in the report, identify its claimed source, retrieve the value from state, and compare. Flag any discrepancy greater than 0.01 (absolute) as a metric error.

**Severity:** FAIL if any model performance metric does not match `state.evaluation`. WARN for other numeric discrepancies.

---

### Check 2 — Model Metrics Match Artifacts

**Question:** Do the model metrics reported in the text and tables match the evaluation artifacts?

**What to check:**
- The performance comparison table in Section 7 must match `state.evaluation.performance_table` exactly.
- Baseline metrics must match `state.models.baseline.test_metrics`.
- Candidate metrics must match `state.models.candidate.test_metrics`.
- Every metric in the report carries a `split` label (visible as a column header or parenthetical) indicating which dataset it was computed on.
- The `split` label for every final metric must be `test`, not `train` or `val`.

**Severity:** FAIL if any metric value does not match its source record. FAIL if any metric is missing a split label. FAIL if a metric is labeled as `train` or `val` but presented as final performance.

---

### Check 3 — Conclusions Are Supported by Results

**Question:** Is every conclusion in Sections 1, 8, and 10 grounded in findings from Sections 6 or 7?

**What to check:**
- The Executive Summary's stated "key finding" must be traceable to a specific result in Section 6 or 7.
- Each recommendation in Section 10 must cite a specific finding. A recommendation without a finding citation is unsupported.
- Any claim about feature importance ("X is the most important predictor") must appear in `state.interpretation.feature_importance` with a supporting score.
- Any conclusion about model performance ("the model significantly outperforms the baseline") must be grounded in `state.evaluation.delta_vs_baseline`.
- Claims about data patterns ("class A customers tend to have higher X") must cite a statistic from `state.eda_results`.

**How to check:** For each declarative claim in Sections 1, 8, and 10, identify the supporting evidence in the report body. If no evidence is cited or traceable within the report, flag as `unsupported_claim`.

**Severity:** FAIL if any conclusion is unsupported by reported evidence. WARN if evidence is present but not clearly cited.

---

### Check 4 — Causal Language Audit

**Question:** Does the report contain causal claims for observational findings?

**What to check:** Scan every sentence in the report for prohibited causal phrases. Check all sections, with particular attention to Sections 6, 8, and 10.

**Prohibited phrases and patterns:**

| Pattern | Example in Report | Severity |
|---------|------------------|----------|
| `causes` | "Age causes survival rate to decrease" | FAIL |
| `leads to` | "Higher Fare leads to better outcomes" | FAIL |
| `effect of X on Y` | "the effect of class on survival" | FAIL |
| `results in` | "lower Pclass results in lower survival" | FAIL |
| `due to` (causal sense) | "survival improved due to the feature" | FAIL |
| `drives` | "Sex drives the prediction" | FAIL |
| `influences` | "Embarked influences the outcome" | WARN |
| `impacts` | "Pclass impacts survival" | WARN |
| `determines` | "Age determines the result" | FAIL |
| `predicts causally` | any explicit causal prediction claim | FAIL |

Distinguish between:
- Causal claim (prohibited): "X causes Y" / "X leads to Y" — FAIL.
- Predictive claim (permitted): "X predicts Y" / "X is associated with Y" / "the model uses X to predict Y".
- Feature importance claim (permitted): "X is among the top predictors of Y in this model".

**Exception:** If the Methods section documents the use of a formal causal inference method (e.g., IV, DiD, PSM, RCT), causal language is permitted in sentences explicitly scoped to that method's output.

**Severity:** FAIL for any phrase in the prohibited FAIL list. WARN for WARN-level phrases. All instances must be listed in `unsupported_claims` with the exact sentence, the phrase that triggered the flag, and the suggested replacement.

---

### Check 5 — Limitations Section Coverage

**Question:** Does the Limitations section address every significant warning and risk identified during the analysis?

**What to check:**
Cross-reference `state.data_profile.warnings`, `state.task.warnings`, and `state.evaluation.warnings` against the Limitations section.

**Required coverage — each item must be present in Limitations if the condition is met:**

| Condition | Must Be in Limitations |
|-----------|----------------------|
| Any `missing_rate > 0.10` | Named column(s), rate, and potential impact |
| `duplicate_count > 0` | Count and whether duplicates were removed |
| `n_rows < 1000` (supervised task) | Sample size warning |
| `class_imbalance_detected == true` | Imbalance extent and handling |
| `target_inferred == true` | Statement that target was inferred, not specified |
| Any WARN in `leakage_audit.warned_checks` | Addressed or confirmed resolved |
| Candidate did not improve over baseline | Explicitly stated |
| Any `potential_id_column` excluded from features | Noted as excluded |
| Task confidence < 0.85 | Noted as potential task misclassification |

**Severity:** FAIL if `missing_rate > 0.50` for any column is not mentioned. WARN for other uncovered items. Each uncovered item must appear in `missing_limitations` in the output.

---

### Check 6 — Metrics Are Explained

**Question:** Are the reported metrics explained in plain language, not just listed as numbers?

**What to check:**
- Section 7 must include at least one sentence interpreting the primary metric in plain language (e.g., "An AUC-ROC of 0.86 means the model correctly ranks a positive case above a negative case 86% of the time.").
- For regression: RMSE must be described in the units of the target variable ("an average prediction error of X units").
- For multiclass: the difference between macro-F1 and weighted-F1 must be acknowledged if both are reported.
- The confusion matrix, if present, must have at least one sentence describing what the cells represent in the context of the analysis goal.

**Severity:** WARN if any required metric interpretation is missing. Does not block approval, but every missing interpretation must appear in `optional_improvements`.

---

### Check 7 — Recommendations Are Supported

**Question:** Is every recommendation in Section 10 traceable to a specific finding in the analysis?

**What to check:**
- Each recommendation must name the finding that motivates it.
- Recommendations must not promise outcomes ("this will improve X by Y%") unless directly supported by a quantified result.
- At minimum, one recommendation must address the primary limitation identified in Section 9.
- Recommendations must be scoped to what the analysis actually shows — not extended to populations, time periods, or conditions not covered by the data.

**Severity:** FAIL if any recommendation is directly contradicted by the findings. WARN if a recommendation lacks a named motivating finding. WARN if a recommendation overstates the expected impact.

---

### Check 8 — Internal Contradictions

**Question:** Are there any statements in the report that contradict each other?

**What to check:**
- A metric value cited in the Executive Summary must match the same metric in Section 7.
- The stated `task_type` in Section 2 must match `state.task.task_type`.
- The described preprocessing steps in Section 5 must match `state.plan.approved`.
- The stated train/val/test split sizes and seed must be consistent between Sections 3 and 5.
- A finding described as "significant" in Section 8 must not be described as "marginal" in Section 9, or vice versa, without explanation.
- The leakage audit outcome in Section 4 must match `state.leakage_audit.approved`.

**How to check:** Extract the key factual claims from each section and compare against each other and against `state`. Flag any inconsistency.

**Severity:** FAIL for metric contradictions between sections. WARN for language-level inconsistencies (e.g., "significant" vs "marginal" applied to the same result).

---

### Check 9 — Referenced Figures and Tables Exist

**Question:** Does every figure and table referenced in the report exist as a real artifact file?

**What to check:**
- Every inline figure reference (e.g., "Figure 1", "see confusion matrix plot") must correspond to a path in `state.artifacts`.
- For each path, verify `os.path.exists(path) == True`.
- The Appendix artifact table must list all paths in `state.artifacts`. No path in `state.artifacts` may be absent from the Appendix without explanation.
- No figure may be referenced by a path that is not in `state.artifacts`.

**Severity:** FAIL if any referenced figure does not exist on disk. WARN if a figure in `state.artifacts` is not referenced anywhere in the report.

---

## Failure Criteria

These conditions unconditionally produce `approved: false`. They cannot be waived.

| Condition | Check | Rationale |
|-----------|-------|-----------|
| Any prohibited causal phrase present in report | 4 | Observational analysis misrepresented as causal |
| Any model metric in report does not match `state.evaluation` | 1, 2 | Report is factually incorrect |
| Any metric missing its split label | 2 | Metric is unverifiable |
| Any final metric labeled `train` or `val` | 2 | Performance is overstated |
| Any conclusion with no supporting evidence traceable in the report | 3 | Report makes unsupported claims |
| Any referenced figure does not exist on disk | 9 | Report references nonexistent evidence |
| `missing_rate > 0.50` column not mentioned in Limitations | 5 | Critical data quality issue hidden |
| Any recommendation directly contradicted by findings | 7 | Report gives conflicting guidance |
| Metric value in Executive Summary contradicts Section 7 | 8 | Internal contradiction in final output |

If the report fails review, set `approved: false`, populate `required_revisions` with one entry per FAIL, and return the output to the `report_generate` step. The Report Writer must address every entry in `required_revisions` before resubmitting.

Maximum revision cycles: 2. If the report still fails after 2 cycles, halt execution and escalate to the user.

---

## Output Format

Write the review result to `state.report_review`.

```json
{
  "approved": false,
  "review_date": "2024-01-15T14:32:00Z",
  "run_id": "run_20240115_a3f7",
  "checks_performed": [1, 2, 3, 4, 5, 6, 7, 8, 9],
  "unsupported_claims": [
    {
      "section": "Section 8",
      "sentence": "Sex drives the predicted survival outcome.",
      "issue": "Causal language ('drives') used for an observational/predictive finding.",
      "triggered_by_phrase": "drives",
      "severity": "FAIL",
      "suggested_replacement": "Sex is among the top predictors of the predicted survival outcome in this model."
    }
  ],
  "metric_errors": [
    {
      "section": "Section 7",
      "metric": "f1_weighted",
      "model": "RandomForestClassifier",
      "split": "test",
      "value_in_report": 0.82,
      "value_in_state": 0.808,
      "delta": 0.012,
      "severity": "FAIL",
      "required_fix": "Replace 0.82 with 0.808 in the performance table."
    }
  ],
  "contradictions": [
    {
      "location_a": "Executive Summary",
      "location_b": "Section 7, Table 1",
      "description": "Executive Summary states ROC-AUC = 0.87; Section 7 table shows 0.864.",
      "severity": "FAIL",
      "required_fix": "Use 0.864 consistently. Executive Summary must match Section 7 exactly."
    }
  ],
  "missing_limitations": [
    {
      "warning_source": "state.data_profile.warnings",
      "warning_type": "critical_missing",
      "column": "Cabin",
      "missing_rate": 0.771,
      "severity": "FAIL",
      "required_fix": "Add a limitation noting that Cabin is 77.1% missing and was excluded from features. Discuss potential impact."
    }
  ],
  "required_revisions": [
    {
      "revision_id": "R1",
      "check": 4,
      "severity": "FAIL",
      "location": "Section 8, paragraph 2",
      "description": "Causal phrase 'drives' used for observational finding.",
      "required_action": "Replace 'Sex drives the predicted survival outcome' with 'Sex is among the top predictors of the predicted survival outcome in this model.'"
    },
    {
      "revision_id": "R2",
      "check": 1,
      "severity": "FAIL",
      "location": "Section 7, Table 1",
      "description": "F1 (weighted) for RandomForestClassifier is reported as 0.82 but state.evaluation shows 0.808.",
      "required_action": "Correct F1 (weighted) to 0.808 in the performance table."
    }
  ],
  "optional_improvements": [
    {
      "check": 6,
      "severity": "WARN",
      "location": "Section 7",
      "description": "AUC-ROC is reported (0.864) but not explained in plain language.",
      "suggestion": "Add: 'An AUC-ROC of 0.864 means the model correctly ranks a positive case above a negative case 86.4% of the time.'"
    }
  ],
  "summary": {
    "total_checks": 9,
    "passed": 6,
    "warned": 1,
    "failed": 2,
    "required_revisions_count": 2,
    "optional_improvements_count": 1
  }
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `approved` | `boolean` | `true` only if `required_revisions` is empty |
| `review_date` | `string` | ISO 8601 timestamp of this review |
| `checks_performed` | `list[int]` | Always `[1, 2, 3, 4, 5, 6, 7, 8, 9]` — all checks run every time |
| `unsupported_claims` | `list[object]` | Claims without supporting evidence; includes causal language findings |
| `metric_errors` | `list[object]` | Per-metric discrepancies between report and state |
| `contradictions` | `list[object]` | Internal inconsistencies between report sections or between report and state |
| `missing_limitations` | `list[object]` | Warnings from state not addressed in Limitations section |
| `required_revisions` | `list[object]` | One entry per FAIL; must be resolved before `approved: true` |
| `optional_improvements` | `list[object]` | WARN-level findings that do not block approval |
| `summary` | `object` | Counts of passed/warned/failed checks and revision totals |

### Verdict Mapping

| Condition | `approved` |
|-----------|-----------|
| `required_revisions` is empty | `true` |
| `required_revisions` has one or more entries | `false` |
| Any pre-condition failed | `false` (ReviewError, not a standard verdict) |

`optional_improvements` entries do not affect `approved`. A report with only WARN-level findings may be approved.

---

## Do Not

- **Do not approve a report that contains prohibited causal language.** A single causal phrase in an observational finding is a FAIL regardless of how accurate the rest of the report is. Every instance must appear in `unsupported_claims` with the exact sentence and required replacement.
- **Do not approve a report where any metric does not match `state.evaluation`.** The review is the last line of defense against numeric errors. Accept no rounding inconsistencies beyond 0.01 for model metrics.
- **Do not issue a verdict without running all nine checks.** `checks_performed` must always contain all nine check IDs. Skipping a check because it "seems unlikely to be violated" is not permitted.
- **Do not treat WARN-level findings as FAIL unless they also meet a FAIL criterion.** A missing metric interpretation (Check 6) is a WARN, not a FAIL. It goes in `optional_improvements`, not `required_revisions`. Do not inflate WARN findings to block approval.
- **Do not verify findings from memory of the analysis.** Every verification must be performed by reading the current values in `state.*` fields and comparing against the report text. The reviewer must not rely on recollection of what was computed.
- **Do not approve a report with a broken artifact reference.** If `os.path.exists(path)` is `False` for any figure cited in the report, that is a FAIL. There is no exception for figures that "should" exist.
- **Do not issue `approved: true` if `required_revisions` is non-empty.** The `approved` field is derived mechanically: `approved = len(required_revisions) == 0`. No override is permitted.
- **Do not accept a revised report that fixes some but not all required revisions.** On resubmission, re-run all nine checks. If any previously identified FAIL persists, it must remain in `required_revisions` for the new output. Do not carry forward only the unresolved items — run all checks fresh.
