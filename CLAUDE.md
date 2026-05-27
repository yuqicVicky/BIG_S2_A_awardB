# CLAUDE.md — STAI-X Award B Automation Agent

## Primary Instruction

When the user prompt is:

```text
Do the data analysis
```

run the deterministic pipeline:

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
row_id,<target_column_name>
```

The target column name must be inferred from `DATA_DESCRIPTION.md` and the sample
submission. The output must preserve the row order of the sample submission.

## Workflow Contract

The Python pipeline performs these stages:

1. Parse and log the hidden dataset schema.
2. Load the target, covariates, validation, and sample submission tables.
3. Build train and prediction feature matrices.
4. Evaluate baselines and candidate models on an internal holdout.
5. Select the best model by holdout MAE or block-averaged MAE.
6. Refit the selected model on all available training rows.
7. Write and validate `submission.csv`.
8. Generate a dynamic analysis report as `report.pdf`.

## Modeling Rules

- Baseline models must be evaluated before candidate models.
- Candidate selection is data-driven; no single algorithm is always preferred.
- If a category or block column is discovered, use block-averaged MAE for model
  selection.
- If a time column is discovered, prefer a time-based holdout. Otherwise use a
  random holdout with a fixed seed.
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
