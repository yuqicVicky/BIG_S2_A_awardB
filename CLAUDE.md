# CLAUDE.md — Award B: General Data Analysis AI Agent

## Project Goal

Build a General Data Analysis AI Agent for the STAI-X Award B AI Automation challenge.

Given a user-provided data file and analysis requirement, the agent autonomously:
- reads and profiles the data
- infers the task type
- plans, critiques, and executes the analysis
- builds and evaluates models
- generates a final report

**V1 scope:** CSV/XLSX input, tabular data, descriptive analysis, classification, regression.

---

## Core Workflow

Every run follows these stages in order. No stage may be skipped.

| # | Stage | Owner |
|---|-------|-------|
| 1 | **Data Intake** | Python |
| 2 | **Data Profiling** | Python + Claude |
| 3 | **Task Inference** | Claude |
| 4 | **Initial Planning** | Claude |
| 5 | **Plan Critique & Optimization** | Claude |
| 6 | **Step-by-step Execution** | Python (Programmer) |
| 7 | **Step-by-step Inspection** | Claude (Inspector) |
| 8 | **Modeling & Prediction** | Python (Programmer) |
| 9 | **Report Generation** | Claude |
| 10 | **Report Review** | Claude (Report Reviewer) |

---

## Division of Responsibility

### Claude is responsible for:
- Reasoning and judgment (task type, plan quality, claim validity)
- Planning and plan critique
- Interpreting inspection results and deciding whether to proceed, retry, or halt
- Writing the natural-language report
- Reviewing the report before it is finalized

### Python functions are responsible for:
- All deterministic, reproducible computation (loading, transforming, fitting, scoring, plotting)
- Writing results into `AnalysisState`
- Producing log entries and artifacts at every step

Claude must **not** perform computation that should be deterministic.  
Python functions must **not** make narrative judgments.

---

## AnalysisState Contract

Every stage must read from and write to `AnalysisState`. No stage may produce output that is not recorded in state.

Required fields that must be populated before the report stage:
- `data_profile` — schema, dtypes, shape, missing rates, basic statistics
- `task_type` — one of `descriptive`, `classification`, `regression`
- `plan` — approved step list with rationale
- `execution_log` — per-step: inputs, outputs, pass/fail, inspector verdict
- `artifacts` — paths to all generated files (plots, CSVs, model files)
- `metrics` — final evaluation metrics with split details
- `report_draft` — generated report text
- `report_review` — reviewer verdict and any required corrections

---

## Execution Rules

### Data integrity
- **No preprocessing on the full dataset before train/test split.** All transformations (imputation, scaling, encoding) must be fit on the training split only and applied to test via the fitted transformer. Violation = data leakage; the Inspector must catch and fail this.

### Modeling
- **Baseline first, then candidate model.** Every modeling stage must establish a baseline (e.g., `DummyClassifier`, mean predictor) before training any candidate model. Metrics must compare candidate against baseline.
- Report must include: train/validation/test sizes, random seed, CV strategy if used.

### Language and claims
- **Prediction and association are not causality.** The report must not use causal language (`causes`, `leads to`, `effect of`, `due to`) for observational findings unless a causal inference method was explicitly applied and documented.
- Uncertainty must be expressed where metrics are close or samples are small.
- No metric may be reported without stating the dataset split it was computed on.

---

## Inspector Rules

The Inspector runs after every execution step and after the report is drafted. The Inspector must check:

1. **Data leakage** — was any transformation fit on data that includes the test set?
2. **Metrics integrity** — are all metrics labeled with the correct split? Are train and test metrics both reported?
3. **Artifacts presence** — does every claimed artifact exist at the stated path?
4. **Unsupported claims** — does the report contain causal claims, numerical claims without supporting metrics, or conclusions not supported by the execution log?
5. **Baseline comparison** — is a baseline reported alongside every model metric?

Inspector verdict must be one of `PASS`, `WARN`, or `FAIL`. A `FAIL` at any step halts execution until the issue is resolved.

---

## File Conventions

```
outputs/
├── artifacts/   # plots, feature importance CSVs, model serializations
├── logs/        # one JSON log file per run, append-only
└── reports/     # final PDF or Markdown report per run
```

Each run is identified by a `run_id` (timestamp + short hash). All artifacts and logs for a run use this prefix.

---

## What Claude Must Not Do

- Do not skip the Inspector after any execution step.
- Do not approve a plan that lacks a baseline modeling step for supervised tasks.
- Do not write a report before all Inspector verdicts are `PASS` or `WARN` (with documented rationale).
- Do not interpret model output as causal without documented causal methodology.
- Do not report a metric without its associated split label.
- Do not preprocess before splitting.
