# STAI-X Challenge 2026 — Award B Automation Agent

This repository contains a Claude Code automation pipeline for the STAI-X
Award B evaluation. The evaluation harness places a hidden dataset under
`data/`, adds `data/DATA_DESCRIPTION.md`, opens this repository in Claude Code,
and issues one prompt:

```text
Do the data analysis
```

The agent should then run the deterministic Python entry point:

```bash
python main.py
```

The run writes two required files to the repository root:

- `submission.csv` with exactly `row_id,<target_column_name>`
- `report.pdf` describing the data, modeling procedure, metrics, and checks

## How The Pipeline Works

1. Parse `data/DATA_DESCRIPTION.md` and inspect files under `data/`.
2. Infer the train target table, covariates tables, sample submission, target
   column, row id column, join keys, time column, and category/block column.
3. Build train and prediction feature frames without using validation targets.
4. Train baselines and a small candidate model pool.
5. Select the best model by internal holdout MAE or block-averaged MAE.
6. Refit the selected model on all training data and generate predictions.
7. Validate the submission schema and generate a dynamic PDF report.

## Repository Structure

```text
.
├── main.py                    # single deterministic entry point
├── src/data_agent/            # generic Award B pipeline implementation
├── scripts/award_a_reference/ # original Award A scripts kept as reference only
├── data/                      # empty at submission; organizers populate
├── outputs/                   # runtime artifacts/logs/reports
├── CLAUDE.md                  # Claude Code operating instructions
├── .claude/                   # Claude Code agent config
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

For the optional LightGBM candidate:

```bash
pip install -r requirements-optional.txt
```

If LightGBM is unavailable, the pipeline falls back to scikit-learn models. If
scikit-learn is unavailable, it still produces a valid submission using
baseline mean models.

## Local Reproduction

Place a hidden-style dataset in `data/`:

```text
data/
├── DATA_DESCRIPTION.md
├── ... training target table ...
├── ... training covariates table ...
├── ... validation covariates table ...
└── ... sample submission table ...
```

Then run:

```bash
python main.py
```

Expected outputs:

```text
submission.csv
report.pdf
outputs/logs/<run_id>_*.json
outputs/reports/<run_id>_report.md
outputs/reports/<run_id>_report.pdf
```

## Design Notes

- The implementation does not assume overdose-specific column names.
- The target column and output schema come from `DATA_DESCRIPTION.md` and the
  sample submission.
- Row count is never hardcoded.
- Baseline models are evaluated before candidate models.
- Candidate models are selected by internal holdout performance, not by a fixed
  preferred algorithm.
- No external data is used.
