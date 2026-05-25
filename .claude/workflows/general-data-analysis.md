# Workflow: General Data Analysis

**version:** 1.0  
**scope:** CSV/XLSX · tabular data · descriptive analysis · classification · regression  
**entry_point:** `user_intake`  
**exit_point:** `final_delivery`

---

## Architecture Overview

```
user_intake
    │
    ▼
data_loading ──► data_profiling ──► task_inference
                                         │
                              ┌──────────▼──────────┐
                              │    initial_planning  │
                              └──────────┬──────────┘
                                         │
                              ┌──────────▼──────────┐
                              │    plan_critique     │◄── loop until PASS
                              └──────────┬──────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │              eda               │
                         └───────────────┬───────────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │         leakage_check          │◄── FAIL → halt
                         └───────────────┬───────────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │         preprocessing          │
                         └───────────────┬───────────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │           modeling             │
                         └───────────────┬───────────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │           evaluation           │
                         └───────────────┬───────────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │         interpretation         │
                         └───────────────┬───────────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │       report_generation        │
                         └───────────────┬───────────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │         report_review          │◄── FAIL → back to report_generation
                         └───────────────┬───────────────┘
                                         │
                         ┌───────────────▼───────────────┐
                         │         final_delivery         │
                         └───────────────────────────────┘
```

---

## Steps

---

### Step 1 — `user_intake`

| Field | Value |
|-------|-------|
| **step_id** | `user_intake` |
| **goal** | Receive and validate the user's data file path and analysis requirement. Confirm inputs are sufficient before any computation begins. |
| **input** | Raw user message containing: file path(s), analysis goal or question |
| **output** | `state.user_request`: validated file path, raw goal string, detected file format (`csv`/`xlsx`) |
| **required_skills** | `file-validation` |
| **responsible_agent** | Claude (Orchestrator) |
| **success_criteria** | File exists and is readable; goal string is non-empty; format is supported |
| **failure_criteria** | File not found; unsupported format; goal is absent or ambiguous beyond resolution |

**Notes:**  
If the goal is ambiguous, Claude must ask one clarifying question before proceeding. Do not proceed with an unresolved ambiguity.

---

### Step 2 — `data_loading`

| Field | Value |
|-------|-------|
| **step_id** | `data_loading` |
| **goal** | Load the raw data file into a DataFrame. Record shape, column names, dtypes, and a sample of raw rows. |
| **input** | `state.user_request.file_path`, `state.user_request.file_format` |
| **output** | `state.raw_data` (DataFrame), `state.load_metadata`: shape, columns, dtypes, head(5), encoding used |
| **required_skills** | `data-loader` |
| **responsible_agent** | Python (Programmer) |
| **success_criteria** | DataFrame loads without error; shape > (0, 0); all columns readable; metadata written to state |
| **failure_criteria** | Parse error; zero rows or columns; encoding failure; file is empty |

**Notes:**  
Do not modify the data during loading. `state.raw_data` must be an unmodified copy of the file contents.

---

### Step 3 — `data_profiling`

| Field | Value |
|-------|-------|
| **step_id** | `data_profiling` |
| **goal** | Produce a comprehensive statistical profile of the raw data to inform task inference and planning. |
| **input** | `state.raw_data` |
| **output** | `state.data_profile`: per-column stats (dtype, missing rate, unique count, min/max/mean/std for numeric, top values for categorical), correlation summary, duplicate row count |
| **required_skills** | `data-profiler` |
| **responsible_agent** | Python (Programmer) → Claude (Inspector) |
| **success_criteria** | Profile covers all columns; missing rates computed; numeric and categorical columns handled separately; Inspector verdict is PASS or WARN |
| **failure_criteria** | Profile missing any column; missing rates not computed; Inspector verdict is FAIL |

---

### Step 4 — `task_inference`

| Field | Value |
|-------|-------|
| **step_id** | `task_inference` |
| **goal** | Infer the analytical task type from the user goal and data profile. Identify the target variable (if any), feature candidates, and any domain constraints. |
| **input** | `state.user_request.goal`, `state.data_profile` |
| **output** | `state.task`: `task_type` (one of `descriptive`, `classification`, `regression`), `target_column` (or null), `feature_candidates`, `constraints`, `rationale` |
| **required_skills** | — (Claude reasoning only) |
| **responsible_agent** | Claude (Orchestrator) |
| **success_criteria** | `task_type` is one of the three supported values; rationale references evidence from `data_profile`; target column exists in the data if task is supervised |
| **failure_criteria** | Task type is unsupported; target column does not exist; no rationale provided |

---

### Step 5 — `initial_planning`

| Field | Value |
|-------|-------|
| **step_id** | `initial_planning` |
| **goal** | Generate a step-by-step execution plan tailored to the inferred task. Each plan step must name the skill it will invoke and the expected output. |
| **input** | `state.task`, `state.data_profile`, `state.user_request.goal` |
| **output** | `state.plan.draft`: ordered list of plan steps, each with `step_name`, `skill`, `rationale`, `expected_output` |
| **required_skills** | — (Claude reasoning only) |
| **responsible_agent** | Claude (Planner) |
| **success_criteria** | Plan covers EDA → preprocessing → modeling (if supervised) → evaluation → interpretation; each step has a named skill; baseline step is present for supervised tasks |
| **failure_criteria** | Plan skips EDA; plan has no baseline for supervised tasks; plan preprocesses before split; any step lacks a named skill |

---

### Step 6 — `plan_critique`

| Field | Value |
|-------|-------|
| **step_id** | `plan_critique` |
| **goal** | Critically review the draft plan for logical errors, data leakage risks, missing steps, and unsupported scope. Approve or revise. |
| **input** | `state.plan.draft`, `state.task`, `state.data_profile` |
| **output** | `state.plan.approved`: final approved plan with any revisions documented; `state.plan.critique`: list of issues found and how each was resolved |
| **required_skills** | `leakage-check` (as a review lens) |
| **responsible_agent** | Claude (Inspector) |
| **success_criteria** | All critique issues are resolved; approved plan has no preprocessing-before-split; baseline present for supervised tasks; verdict is PASS |
| **failure_criteria** | Unresolved FAIL-level issues remain in the critique |

**Loop condition:** If critique finds FAIL-level issues, return to `initial_planning`. Maximum 3 iterations before halting with an error.

---

### Step 7 — `eda`

| Field | Value |
|-------|-------|
| **step_id** | `eda` |
| **goal** | Execute exploratory data analysis per the approved plan. Produce distribution plots, correlation heatmap, missing-value summary, and class balance (for classification). |
| **input** | `state.raw_data`, `state.plan.approved`, `state.task` |
| **output** | `state.eda_results`: summary statistics, key findings; `state.artifacts`: paths to all generated plots |
| **required_skills** | `eda-analyzer`, `plot-generator` |
| **responsible_agent** | Python (Programmer) → Claude (Inspector) |
| **success_criteria** | All planned EDA steps executed; at least one plot artifact per planned visualization; Inspector verdict is PASS or WARN; findings written to state |
| **failure_criteria** | Any planned EDA step not executed; artifact paths missing or broken; Inspector verdict is FAIL |

---

### Step 8 — `leakage_check`

| Field | Value |
|-------|-------|
| **step_id** | `leakage_check` |
| **goal** | Before any split or transformation, audit the data and plan for potential sources of data leakage. This is a mandatory gate. |
| **input** | `state.raw_data`, `state.task`, `state.plan.approved`, `state.data_profile` |
| **output** | `state.leakage_audit`: list of checked leakage vectors with verdict per vector; overall verdict (`PASS` / `FAIL`) |
| **required_skills** | `leakage-check` |
| **responsible_agent** | Claude (Inspector) |
| **success_criteria** | All leakage vectors checked (temporal leakage, target encoding before split, ID columns in features, future-looking features); overall verdict is PASS |
| **failure_criteria** | Any leakage vector is unresolved; overall verdict is FAIL |

**Hard stop:** A FAIL verdict halts execution. The plan must be revised before proceeding.

---

### Step 9 — `preprocessing`

| Field | Value |
|-------|-------|
| **step_id** | `preprocessing` |
| **goal** | Split data into train/validation/test sets, then fit all transformers on the training split only and apply to validation and test via the fitted transformer. |
| **input** | `state.raw_data`, `state.task`, `state.plan.approved` |
| **output** | `state.splits`: `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test`, split sizes, random seed; `state.transformers`: fitted transformer objects; `state.artifacts`: saved transformer files |
| **required_skills** | `data-splitter`, `feature-engineer`, `leakage-check` |
| **responsible_agent** | Python (Programmer) → Claude (Inspector) |
| **success_criteria** | Split performed before any fitting; transformer fit only on `X_train`; split sizes and seed recorded; leakage check on transformed data passes; Inspector verdict is PASS |
| **failure_criteria** | Any transformer fit on validation or test data; split sizes not recorded; seed not set; Inspector verdict is FAIL |

---

### Step 10 — `modeling`

| Field | Value |
|-------|-------|
| **step_id** | `modeling` |
| **goal** | Train a baseline model followed by one or more candidate models. Record all hyperparameters, training details, and training-set metrics. |
| **input** | `state.splits`, `state.task`, `state.plan.approved` |
| **output** | `state.models`: per-model record with `model_name`, `hyperparameters`, `train_metrics`, `val_metrics`, fitted model object; `state.artifacts`: serialized model files |
| **required_skills** | `tabular-modeling`, `leakage-check`, `model-evaluation` |
| **responsible_agent** | Python (Programmer) → Claude (Inspector) |
| **success_criteria** | Baseline model trained and recorded; at least one candidate model trained; both train and val metrics recorded per model; no test data touched during training; leakage check passes; Inspector verdict is PASS |
| **failure_criteria** | No baseline recorded; test data used during training or hyperparameter tuning; metrics missing split labels; Inspector verdict is FAIL |

---

### Step 11 — `evaluation`

| Field | Value |
|-------|-------|
| **step_id** | `evaluation` |
| **goal** | Evaluate the final selected model on the held-out test set. Compare against baseline. Produce all required evaluation artifacts. |
| **input** | `state.models`, `state.splits` |
| **output** | `state.evaluation`: test metrics per model with split label, baseline comparison, confusion matrix or residual plot path; `state.artifacts`: evaluation plots |
| **required_skills** | `model-evaluation`, `plot-generator` |
| **responsible_agent** | Python (Programmer) → Claude (Inspector) |
| **success_criteria** | Test metrics computed on `X_test` only; all metrics labeled with split; baseline comparison present; appropriate metric for task type used (accuracy/F1/AUC for classification; RMSE/MAE/R² for regression); Inspector verdict is PASS |
| **failure_criteria** | Test metrics computed on train or val data; no baseline comparison; metric not appropriate for task type; Inspector verdict is FAIL |

---

### Step 12 — `interpretation`

| Field | Value |
|-------|-------|
| **step_id** | `interpretation` |
| **goal** | Interpret model results: feature importance, key drivers, notable patterns. Flag any findings that require caution (small sample, class imbalance, near-random performance). |
| **input** | `state.models`, `state.evaluation`, `state.eda_results`, `state.task` |
| **output** | `state.interpretation`: feature importance table, top drivers, caveats list, `state.artifacts`: feature importance plot |
| **required_skills** | `model-evaluation`, `plot-generator` |
| **responsible_agent** | Claude (Orchestrator) + Python (Programmer) |
| **success_criteria** | Feature importance computed and plotted; caveats section present; no causal language used; all numerical claims reference a specific metric or artifact |
| **failure_criteria** | Causal language present; numerical claim without supporting metric; no caveats section |

---

### Step 13 — `report_generation`

| Field | Value |
|-------|-------|
| **step_id** | `report_generation` |
| **goal** | Write a structured final report covering: objective, data summary, methodology, key findings, model performance, interpretation, limitations, and conclusions. |
| **input** | `state.data_profile`, `state.task`, `state.plan.approved`, `state.eda_results`, `state.evaluation`, `state.interpretation` |
| **output** | `state.report_draft`: full report text in Markdown; `state.artifacts`: report file path |
| **required_skills** | `report-writing` |
| **responsible_agent** | Claude (Report Writer) |
| **success_criteria** | All required sections present; every metric includes its split label; no unsupported causal claims; uncertainty expressed where sample is small or metrics are near baseline; all artifact paths valid |
| **failure_criteria** | Any required section missing; causal language present; metric without split label; unsupported numerical claim |

**Required report sections:**

1. Executive Summary
2. Data Overview
3. Task & Objective
4. Methodology
5. Key EDA Findings
6. Model Performance (with baseline comparison table)
7. Feature Importance & Drivers
8. Limitations & Caveats
9. Conclusions

---

### Step 14 — `report_review`

| Field | Value |
|-------|-------|
| **step_id** | `report_review` |
| **goal** | Independently review the report draft against the execution log, metrics, and artifacts. Verify all claims are supported and no prohibited language is present. |
| **input** | `state.report_draft`, `state.execution_log`, `state.evaluation`, `state.leakage_audit`, `state.artifacts` |
| **output** | `state.report_review`: verdict (`PASS` / `FAIL`), list of issues with severity and required correction per issue |
| **required_skills** | `report-review`, `model-evaluation`, `leakage-check` |
| **responsible_agent** | Claude (Report Reviewer) |
| **success_criteria** | All metrics in report match `state.evaluation`; no causal language; all artifact references resolve; baseline comparison present; verdict is PASS |
| **failure_criteria** | Any metric in report does not match execution log; causal language present; broken artifact reference; verdict is FAIL |

**Loop condition:** A FAIL verdict returns to `report_generation` with the issue list. Maximum 2 revision cycles.

---

### Step 15 — `final_delivery`

| Field | Value |
|-------|-------|
| **step_id** | `final_delivery` |
| **goal** | Package and deliver all outputs: approved report, artifacts, logs, and a machine-readable run summary. |
| **input** | `state.report_draft` (reviewed), `state.artifacts`, `state.execution_log`, `state.run_id` |
| **output** | `outputs/reports/<run_id>_report.md`, `outputs/artifacts/<run_id>_*`, `outputs/logs/<run_id>_run.json`, `state.delivery_manifest` |
| **required_skills** | `file-writer` |
| **responsible_agent** | Python (Programmer) |
| **success_criteria** | All artifact paths in manifest exist; report file written; run log written; manifest written to state |
| **failure_criteria** | Any artifact path in manifest is broken; report file not written |

---

## Skill Registry

| Skill ID | Used In |
|----------|---------|
| `file-validation` | user_intake |
| `data-loader` | data_loading |
| `data-profiler` | data_profiling |
| `eda-analyzer` | eda |
| `plot-generator` | eda, evaluation, interpretation |
| `leakage-check` | plan_critique, leakage_check, preprocessing, modeling, report_review |
| `data-splitter` | preprocessing |
| `feature-engineer` | preprocessing |
| `tabular-modeling` | modeling |
| `model-evaluation` | modeling, evaluation, interpretation, report_review |
| `report-writing` | report_generation |
| `report-review` | report_review |
| `file-writer` | final_delivery |

---

## Agent Registry

| Agent Role | Steps | Responsibility |
|------------|-------|----------------|
| Orchestrator (Claude) | user_intake, task_inference, interpretation | High-level reasoning and judgment |
| Planner (Claude) | initial_planning | Draft plan generation |
| Inspector (Claude) | plan_critique, leakage_check, + post-execution inspection in steps 3/7/8/9/10/11 | Verification and gate-keeping |
| Programmer (Python) | data_loading, data_profiling, eda, preprocessing, modeling, evaluation, final_delivery | Deterministic computation |
| Report Writer (Claude) | report_generation | Narrative generation |
| Report Reviewer (Claude) | report_review | Independent report audit |

---

## State Completeness Gate

Before `report_generation` begins, the following `AnalysisState` fields must all be non-null:

```
state.data_profile
state.task
state.plan.approved
state.leakage_audit           (verdict == PASS)
state.splits                  (supervised tasks only)
state.models                  (supervised tasks only)
state.evaluation              (supervised tasks only)
state.eda_results
state.interpretation
state.execution_log           (all steps with PASS or WARN)
```

If any required field is missing, execution must halt and report the missing field.
