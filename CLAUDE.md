# CLAUDE.md — STAI-X Award B Automation Agent

## Primary Instruction

When the user prompt is:

```text
Do the data analysis
```

run the analysis pipeline (an agent orchestration over `src/data_agent/`, with a
deterministic fallback):

```bash
python main.py
```

If Python reports missing packages, install the committed core dependencies and
rerun:

```bash
pip install -r requirements.txt
python main.py
```

Do not manually craft a submission. Do not rely on any overdose-specific field
names. The only authority for the hidden evaluation dataset is
`data/DATA_DESCRIPTION.md` plus the files placed under `data/`.

## Required Outputs

The run must produce both files in the repository root:

- `submission.csv`
- `report.pdf`

`submission.csv` must contain exactly two columns:

```text
<row_id_column>,<target_column>
```

Both column names come from the sample submission / `DATA_DESCRIPTION.md` (e.g.
`PassengerId,Survived` for Titanic). The output must preserve the sample submission's
row order, and values must match the required output format (integer class labels,
string labels, or continuous/probability values).

## Workflow Contract

`main.py` drives `orchestrator.run_orchestrated_analysis`, which threads an
`AnalysisState` through the staged agent pipeline below — each `.claude/` agent maps to
a stage and is backed by a real skill in `src/data_agent/skills/`. On any stage failure
it falls back to the deterministic `runner.run_analysis`, which performs the same core
steps. The pipeline performs these stages:

1. Parse and log the hidden dataset schema.
2. Infer the task type (regression / binary / multiclass classification), the
   evaluation metric, and the required output format from `DATA_DESCRIPTION.md`
   (primary authority) with a data-driven fallback.
3. Load the target, covariates, validation, and sample submission tables.
4. Build train and prediction feature matrices.
5. Evaluate baselines and candidate models on an internal holdout, using the
   model pool that matches the task type.
6. Select the best model by the task-appropriate holdout metric (MAE or
   block-averaged MAE for regression; accuracy / ROC-AUC / F1 for
   classification).
7. Refit the selected model on all available training rows.
8. Write and validate `submission.csv`, formatting values to match the required
   output (continuous values, class labels, or probabilities).
9. Generate a dynamic analysis report as `report.pdf`, then run an independent
   report review.

Plot artifacts are written under `outputs/artifacts/`, the full run state under
`outputs/logs/{run_id}_state.json`, and per-run logs + a manifest under `outputs/logs/`.

## Modeling Rules

- Detect the task type before modeling; use regressors for regression and
  classifiers for classification. The candidate pool is a function of the task.
- Baseline models must be evaluated before candidate models.
- Candidate selection is data-driven; no single algorithm is always preferred.
- If a category or block column is discovered in a regression task, use
  block-averaged MAE for model selection.
- For classification, prefer a stratified holdout; if a time column is
  discovered, prefer a time-based holdout. Otherwise use a random holdout with a
  fixed seed.
- Submission values must match what `DATA_DESCRIPTION.md` and the sample
  submission require (e.g. integer class labels for accuracy-scored tasks).
- Never use validation targets or future target values during feature
  engineering.
- If LightGBM is unavailable, continue with scikit-learn fallback models.

## Prohibited Assumptions

Do not hardcode:

- `rate_per_10000_ed_visits`
- `overdose_category`
- `all_drugs`, `all_opioids`, or `all_stimulants`
- `918` rows
- any fixed period-id map
- any local absolute path from a developer machine

The original Award A scripts are preserved only as reference material under
`scripts/award_a_reference/`; they are not the Award B execution path.

## Inspection Checklist

Before considering the run complete, verify:

- `submission.csv` exists in the repo root.
- `report.pdf` exists in the repo root.
- `submission.csv` has exactly two columns.
- row ids match the sample submission in order.
- predictions are finite and non-missing.
- logs were written under `outputs/logs/`.
- the report describes the actual current run rather than a fixed prior dataset.
