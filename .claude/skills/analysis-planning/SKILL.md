---
name: analysis-planning
description: Use this skill when creating or reviewing a structured data analysis plan. It converts a user request, data profile, task specification, and risk assessment into executable plan steps with required skills and success criteria.
---

# Skill: analysis-planning

## When to Use

Use this skill after `task-inference` completes and before any execution step begins. It produces `state.plan.draft`, which must be reviewed by `plan_critique` before execution.

Trigger this skill whenever:
- The Orchestrator needs to generate a new analysis plan from scratch.
- `plan_critique` returns a FAIL verdict and the plan must be rewritten.
- The user modifies the analysis goal after task inference and the existing plan is invalidated.

Do **not** run this skill before `task-inference` and `data-profiling` are both complete. A plan generated without a task specification or data profile cannot be validated.

---

## Inputs

| Field | Source | Required |
|-------|--------|----------|
| `user_request.goal` | `state.user_request.goal` | Yes |
| `task.task_type` | `state.task.task_type` | Yes |
| `task.target_variable` | `state.task.target_variable` | Conditional — required if task is supervised |
| `task.feature_candidates` | `state.task.feature_candidates` | Conditional — required if task is supervised |
| `task.excluded_from_features` | `state.task.excluded_from_features` | Yes |
| `task.recommended_metrics` | `state.task.recommended_metrics` | Yes |
| `data_profile.n_rows` | `state.data_profile.n_rows` | Yes |
| `data_profile.missing_summary` | `state.data_profile.missing_summary` | Yes |
| `data_profile.warnings` | `state.data_profile.warnings` | Yes |
| `data_profile.datetime_columns` | `state.data_profile.datetime_columns` | Yes |
| `data_profile.potential_id_columns` | `state.data_profile.potential_id_columns` | Yes |
| `data_profile.high_cardinality_columns` | `state.data_profile.high_cardinality_columns` | Yes |
| `task.confidence` | `state.task.confidence` | Yes |
| `task.warnings` | `state.task.warnings` | Yes |

If `task.task_type` is missing or `unknown`, do not generate a plan. Return a `PlanningError` with the message: `"Cannot generate plan: task type is unresolved. Run task-inference first."`.

---

## Required Plan Components

Every plan, regardless of task type, must include these mandatory phases in this order. No phase may be omitted.

| Phase | Required For | step_id Prefix |
|-------|-------------|----------------|
| Data Validation | All tasks | `validate_` |
| EDA | All tasks | `eda_` |
| Leakage Check | All tasks | `leakage_check` |
| Preprocessing | Supervised tasks only | `preprocess_` |
| Baseline Model | Supervised tasks only | `baseline_` |
| Candidate Model | Supervised tasks only | `model_` |
| Evaluation | Supervised tasks only | `evaluate_` |
| Interpretation | All tasks | `interpret_` |
| Report Generation | All tasks | `report_generate` |
| Report Review | All tasks | `report_review` |

For `descriptive` tasks: phases Preprocessing, Baseline Model, Candidate Model, and Evaluation are omitted. All other phases are required.

For supervised tasks (`binary_classification`, `multiclass_classification`, `regression`): all ten phases are required.

**The Baseline Model phase is non-negotiable for all supervised tasks.** A plan that jumps directly to a candidate model without a baseline will be rejected by `plan_critique` with a FAIL verdict.

---

## PlanStep Schema

Every step in the plan must conform to this schema. Every field is required. No field may be null or omitted.

```json
{
  "step_id": "string",
  "name": "string",
  "goal": "string",
  "inputs": ["string"],
  "outputs": ["string"],
  "success_criteria": ["string"],
  "risks": ["string"],
  "required_skills": ["string"],
  "responsible_agent": "python | claude | python+claude"
}
```

### Field Definitions

| Field | Type | Rules |
|-------|------|-------|
| `step_id` | `string` | Unique within the plan. Use `snake_case`. Must match a recognized prefix from the Required Plan Components table. |
| `name` | `string` | Short human-readable step name (5–10 words). |
| `goal` | `string` | One sentence stating what this step must accomplish. Must be specific enough to evaluate — no vague goals like "process the data". |
| `inputs` | `list[string]` | State fields or artifact paths this step reads. Must include at least one entry. |
| `outputs` | `list[string]` | State fields or artifact paths this step writes. Must include at least one entry. |
| `success_criteria` | `list[string]` | Verifiable conditions that Inspector will check. Must include at least two criteria. Criteria must be falsifiable — no criteria like "step runs without error". |
| `risks` | `list[string]` | Known risks or failure modes for this step. Must include at least one risk. |
| `required_skills` | `list[string]` | Skills from the Skill Registry (see skill-routing.md). Must be non-empty for all executable steps. |
| `responsible_agent` | `string` | One of `python`, `claude`, `python+claude`. Use `python+claude` when Claude performs post-execution inspection. |

---

## Required Skills Rule

`required_skills` must be populated for every step before the plan is submitted. Cross-reference against `skill-routing.md` Section 1 when populating.

### Mandatory skills per phase

| Phase | Mandatory `required_skills` |
|-------|-----------------------------|
| Data Validation | `data-loader`, `data-profiler` |
| EDA | `eda-analyzer`, `plot-generator` |
| Leakage Check | `leakage-check` |
| Preprocessing | `data-splitter`, `leakage-check` |
| Preprocessing (with transforms) | `data-splitter`, `feature-engineer`, `leakage-check` |
| Baseline Model | `tabular-modeling`, `model-evaluation` |
| Candidate Model | `tabular-modeling`, `leakage-check`, `model-evaluation` |
| Evaluation | `model-evaluation`, `plot-generator` |
| Interpretation | `plot-generator` |
| Report Generation | `report-writing` |
| Report Review | `report-review`, `model-evaluation`, `leakage-check` |

If `required_skills` is empty for any executable step, the plan is invalid. Do not submit it.

---

## Planning Rules by Task Type

### All Task Types

1. The first step must validate the data (confirm file loaded, shape matches profile, no unexpected nulls introduced).
2. EDA must produce at least: distribution summary for all features, correlation or association analysis, missing-value visualization.
3. `leakage_check` must appear as a standalone step between EDA and Preprocessing (for supervised tasks) or between EDA and Interpretation (for descriptive tasks). It must never be merged into another step.
4. Interpretation must not contain causal language. If the user's goal contains causal framing, note it as out of scope in the interpretation step's `risks` field.
5. Report generation must reference specific artifact paths and metric values from earlier steps — not placeholders.
6. Report review must be the final step before `final_delivery`. It must independently verify claims in the report against the execution log.

### `descriptive`

7. No `data-splitter`, `tabular-modeling`, or `model-evaluation` skills may appear anywhere in the plan.
8. EDA must be more extensive: include at minimum pairwise correlation, distribution plots per column, and a missing-value heatmap.
9. Interpretation must summarize key patterns, anomalies, and relationships found during EDA. It must not predict outcomes.

### `binary_classification`

10. Preprocessing step must include train/val/test split with an explicit random seed.
11. Baseline step must train a `DummyClassifier` (stratified or most-frequent strategy). Baseline metrics must include accuracy, F1 (weighted), and AUC-ROC.
12. Candidate model step must compare against the baseline in its `success_criteria`.
13. Evaluation step must produce a confusion matrix and ROC curve as artifacts.
14. If `class_imbalance_detected = true` (from `state.task`): the preprocessing step must include a class-balance strategy (oversampling, undersampling, or `class_weight` parameter). This must appear in the step's `goal` and `required_skills` as `feature-engineer`.

### `multiclass_classification`

15. Preprocessing step must include train/val/test split with an explicit random seed.
16. Baseline step must train a `DummyClassifier`. Baseline metrics must include F1 (weighted), F1 (macro), and accuracy.
17. Candidate model step must compare against baseline. `success_criteria` must include: candidate F1 (weighted) > baseline F1 (weighted).
18. Evaluation step must produce a confusion matrix (multi-class) as an artifact.

### `regression`

19. Preprocessing step must include train/val/test split with an explicit random seed.
20. Baseline step must train a mean-predictor (`DummyRegressor`, strategy `mean`). Baseline metrics must include RMSE, MAE, and R².
21. Candidate model step must compare against baseline. `success_criteria` must include: candidate RMSE < baseline RMSE.
22. Evaluation step must produce a residual plot and a prediction-vs-actual scatter plot as artifacts.
23. If the target has outliers (flagged in `state.task.warnings`): the preprocessing step must include outlier handling strategy. Document the chosen strategy in the step's `goal`.

---

## Output Format

Write the complete plan to `state.plan.draft`. The plan object schema:

```json
{
  "plan_id": "plan_<run_id>",
  "task_type": "binary_classification",
  "target_variable": "Survived",
  "created_at": "<ISO timestamp>",
  "status": "draft",
  "steps": [
    {
      "step_id": "validate_data",
      "name": "Validate Loaded Data",
      "goal": "Confirm the loaded DataFrame matches the profiled schema: shape, dtypes, and column names are consistent with state.data_profile.",
      "inputs": ["state.raw_data", "state.data_profile"],
      "outputs": ["state.validation_result"],
      "success_criteria": [
        "DataFrame shape matches data_profile.n_rows and data_profile.n_columns",
        "All column names in raw_data match data_profile.columns",
        "No new null values introduced during loading"
      ],
      "risks": [
        "File re-read may differ from profiled version if file was modified between profiling and validation"
      ],
      "required_skills": ["data-loader", "data-profiler"],
      "responsible_agent": "python+claude"
    },
    {
      "step_id": "eda_exploration",
      "name": "Exploratory Data Analysis",
      "goal": "Produce distribution plots, correlation heatmap, missing-value chart, and class balance summary for the Survived target.",
      "inputs": ["state.raw_data", "state.task"],
      "outputs": ["state.eda_results", "state.artifacts.eda_plots"],
      "success_criteria": [
        "Distribution plot produced for every numeric feature in state.task.feature_candidates",
        "Correlation heatmap artifact written to state.artifacts",
        "Class balance for Survived recorded in state.eda_results"
      ],
      "risks": [
        "High-cardinality columns (Name, Ticket) may produce unusable plots — cap at top-20 values",
        "Missing values in Age and Cabin may skew distribution plots"
      ],
      "required_skills": ["eda-analyzer", "plot-generator"],
      "responsible_agent": "python+claude"
    },
    {
      "step_id": "leakage_check",
      "name": "Pre-Modeling Leakage Audit",
      "goal": "Audit feature candidates, column names, and the draft preprocessing plan for data leakage vectors before any split or transformation.",
      "inputs": ["state.raw_data", "state.task", "state.plan.draft"],
      "outputs": ["state.leakage_audit"],
      "success_criteria": [
        "All potential ID columns confirmed excluded from feature_candidates",
        "No future-looking or post-target columns in feature_candidates",
        "Leakage audit verdict is PASS"
      ],
      "risks": [
        "PassengerId may have been accidentally retained in feature set",
        "Encoded target column may appear under a different name"
      ],
      "required_skills": ["leakage-check"],
      "responsible_agent": "claude"
    },
    {
      "step_id": "preprocess_split_transform",
      "name": "Split Data and Fit Transformers",
      "goal": "Split into train (70%), val (15%), test (15%) with seed=42. Fit imputer and encoder on X_train only. Apply to val and test via fitted transformers.",
      "inputs": ["state.raw_data", "state.task", "state.leakage_audit"],
      "outputs": [
        "state.splits.X_train", "state.splits.X_val", "state.splits.X_test",
        "state.splits.y_train", "state.splits.y_val", "state.splits.y_test",
        "state.transformers", "state.artifacts.transformers"
      ],
      "success_criteria": [
        "Split performed before any transformer is fitted",
        "Transformer fit called only on X_train",
        "Split sizes and seed recorded in state.splits",
        "Fitted transformers serialized to state.artifacts.transformers"
      ],
      "risks": [
        "fit_transform on full dataset instead of fit on train → data leakage",
        "Random seed not set → non-reproducible splits"
      ],
      "required_skills": ["data-splitter", "feature-engineer", "leakage-check"],
      "responsible_agent": "python+claude"
    },
    {
      "step_id": "baseline_dummy_classifier",
      "name": "Train Baseline DummyClassifier",
      "goal": "Establish performance floor using DummyClassifier (strategy=stratified). Record accuracy, F1 (weighted), and AUC-ROC on the validation set.",
      "inputs": ["state.splits"],
      "outputs": ["state.models.baseline", "state.artifacts.baseline_model"],
      "success_criteria": [
        "DummyClassifier trained on X_train, y_train only",
        "Validation metrics recorded with split label 'val'",
        "All three baseline metrics (accuracy, F1, AUC-ROC) present in state.models.baseline.val_metrics"
      ],
      "risks": [
        "Evaluating baseline on test set at this stage — must use val only"
      ],
      "required_skills": ["tabular-modeling", "model-evaluation"],
      "responsible_agent": "python+claude"
    },
    {
      "step_id": "model_candidate",
      "name": "Train Candidate Model",
      "goal": "Train a gradient-boosted classifier. Tune hyperparameters using cross-validation on X_train. Compare val metrics against baseline.",
      "inputs": ["state.splits", "state.models.baseline"],
      "outputs": ["state.models.candidate", "state.artifacts.candidate_model"],
      "success_criteria": [
        "Model trained and evaluated on val set only (test set not touched)",
        "Val AUC-ROC > baseline val AUC-ROC",
        "Hyperparameters and CV strategy recorded in state.models.candidate",
        "Leakage check confirms test set not used during tuning"
      ],
      "risks": [
        "Hyperparameter search may accidentally use test data if not isolated",
        "Overfitting to val set if too many iterations"
      ],
      "required_skills": ["tabular-modeling", "leakage-check", "model-evaluation"],
      "responsible_agent": "python+claude"
    },
    {
      "step_id": "evaluate_test",
      "name": "Final Evaluation on Test Set",
      "goal": "Evaluate the selected candidate model on the held-out test set. Produce confusion matrix and ROC curve. Compare all metrics against baseline.",
      "inputs": ["state.models.candidate", "state.models.baseline", "state.splits"],
      "outputs": [
        "state.evaluation",
        "state.artifacts.confusion_matrix",
        "state.artifacts.roc_curve"
      ],
      "success_criteria": [
        "Metrics computed on X_test, y_test only",
        "All metrics labeled with split 'test'",
        "Confusion matrix artifact written",
        "ROC curve artifact written",
        "Candidate vs baseline comparison table present in state.evaluation"
      ],
      "risks": [
        "Using val metrics as final metrics — must recompute on test set",
        "Reporting training accuracy as final accuracy"
      ],
      "required_skills": ["model-evaluation", "plot-generator"],
      "responsible_agent": "python+claude"
    },
    {
      "step_id": "interpret_results",
      "name": "Interpret Feature Importance and Key Drivers",
      "goal": "Compute and plot feature importances from the candidate model. Identify top predictors. List caveats where results are uncertain or sample is small.",
      "inputs": ["state.models.candidate", "state.evaluation", "state.task"],
      "outputs": [
        "state.interpretation",
        "state.artifacts.feature_importance_plot",
        "state.artifacts.feature_importance_csv"
      ],
      "success_criteria": [
        "Feature importance computed and sorted descending",
        "Feature importance plot artifact written",
        "At least one caveat recorded in state.interpretation.caveats",
        "No causal language used in interpretation text"
      ],
      "risks": [
        "Feature importance for tree models reflects training-set patterns, not causal relationships",
        "Correlated features may split importance artificially"
      ],
      "required_skills": ["model-evaluation", "plot-generator"],
      "responsible_agent": "python+claude"
    },
    {
      "step_id": "report_generate",
      "name": "Generate Final Analysis Report",
      "goal": "Write a structured Markdown report covering all nine required sections, referencing actual artifact paths and metric values from the execution log.",
      "inputs": [
        "state.data_profile", "state.task", "state.eda_results",
        "state.evaluation", "state.interpretation", "state.execution_log"
      ],
      "outputs": ["state.report_draft", "state.artifacts.report_md"],
      "success_criteria": [
        "All nine required report sections present",
        "Every metric in the report includes its split label",
        "All artifact references resolve to existing files",
        "No causal language present",
        "Baseline comparison table included in Model Performance section"
      ],
      "risks": [
        "Metric values copied incorrectly from state",
        "Artifact paths written as placeholders rather than actual paths"
      ],
      "required_skills": ["report-writing"],
      "responsible_agent": "claude"
    },
    {
      "step_id": "report_review",
      "name": "Independent Report Review",
      "goal": "Audit the report draft against the execution log, evaluation metrics, and leakage audit. Verify every claim and flag any unsupported statement.",
      "inputs": [
        "state.report_draft", "state.execution_log",
        "state.evaluation", "state.leakage_audit", "state.artifacts"
      ],
      "outputs": ["state.report_review"],
      "success_criteria": [
        "Every metric in report matches state.evaluation exactly",
        "No causal language detected",
        "All artifact references verified as existing files",
        "Leakage audit outcome correctly described in report",
        "Review verdict is PASS"
      ],
      "risks": [
        "Report may reference stale metric values from intermediate runs",
        "Subtle causal framing may pass without careful reading"
      ],
      "required_skills": ["report-review", "model-evaluation", "leakage-check"],
      "responsible_agent": "claude"
    }
  ],
  "plan_skill_summary": {
    "all_skills_used": [
      "data-loader", "data-profiler", "eda-analyzer", "plot-generator",
      "leakage-check", "data-splitter", "feature-engineer",
      "tabular-modeling", "model-evaluation", "report-writing", "report-review"
    ],
    "steps_missing_required_skills": []
  },
  "plan_warnings": [
    {
      "type": "class_imbalance_not_detected",
      "message": "Class balance check passed. No additional resampling step required.",
      "severity": "INFO"
    }
  ]
}
```

### Top-Level Plan Fields

| Field | Type | Description |
|-------|------|-------------|
| `plan_id` | `string` | `plan_<run_id>` |
| `task_type` | `string` | Copied from `state.task.task_type` |
| `target_variable` | `string \| null` | Copied from `state.task.target_variable` |
| `created_at` | `string` | ISO 8601 timestamp |
| `status` | `string` | `draft` on creation; updated to `approved` by `plan_critique` |
| `steps` | `list[PlanStep]` | Ordered list of all plan steps |
| `plan_skill_summary.all_skills_used` | `list[string]` | Deduplicated skills across all steps |
| `plan_skill_summary.steps_missing_required_skills` | `list[string]` | Must be empty before submission |
| `plan_warnings` | `list[object]` | Planner-level notes; does not block approval |

---

## Do Not

- **Do not execute any code.** This skill produces a plan object in `state.plan.draft`. It does not run Python, call external tools, or produce data artifacts.
- **Do not train any model.** Model training belongs to the `tabular-modeling` skill during the `modeling` step.
- **Do not skip the leakage check step.** `leakage_check` must appear as an explicit, standalone step in every plan. It may not be merged into EDA, preprocessing, or any other step.
- **Do not omit the baseline model for supervised tasks.** A plan with a candidate model but no baseline is invalid. `plan_critique` will reject it with a FAIL verdict. The baseline step must appear before the candidate model step.
- **Do not write vague success criteria.** Every criterion must be falsifiable and specific enough for the Inspector to verify programmatically or by reading state fields. Criteria like "the step runs without error" or "the output looks reasonable" are not acceptable.
- **Do not use placeholder values in step outputs.** Every output field must name a real `state.*` path or a concrete artifact path under `outputs/artifacts/<run_id>_*`.
- **Do not generate a plan if task type is `unknown`.** An unresolved task type means task inference is incomplete. Return a `PlanningError` and wait for the task type to be resolved.
- **Do not preprocess before the leakage check.** The `leakage_check` step must complete with a PASS verdict before `preprocess_split_transform` is executed. In the plan ordering, `leakage_check` always precedes any preprocessing step.
