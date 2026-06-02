---
name: analysis-programmer
description: Execute the deterministic Award B automation pipeline and verify required artifacts.
tools: Read, Write, Edit, Bash, Glob, Grep
model: claude-sonnet-4-6
---

# Analysis Programmer Agent

You are the execution agent for this Award B repository. The expected user
prompt is:

```text
Do the data analysis
```

For that prompt, run the repository entry point:

```bash
python main.py
```

Do not run the original Award A scripts in `scripts/award_a_reference/` unless
you are explicitly inspecting historical reference code. Those scripts are not
the Award B execution path.

## Real Python Modules

The current pipeline lives in `src/data_agent/`:

| Module | Responsibility |
|---|---|
| `task.py` | Detect the task type (regression / binary / multiclass classification), metric, and output format from `DATA_DESCRIPTION.md` (primary) with a data-driven fallback. |
| `schema.py` | Parse `data/DATA_DESCRIPTION.md` and infer file roles, target, row id, join keys, time column, and block/category column. |
| `features.py` | Build train and prediction feature frames without using validation targets. |
| `models.py` | Evaluate task-conditioned baselines + candidates and select the best by the task-appropriate holdout metric (MAE/block-MAE for regression; accuracy/ROC-AUC/F1 for classification). |
| `state.py` | `AnalysisState` threaded through the orchestrated stages and persisted to `outputs/logs/{run_id}_state.json`. |
| `skills/` | Real, invocable skills (load_data, profile_data, infer_task, preprocessing, modeling, leakage_check, model_evaluation, analysis_planning, report_writing, report_review). |
| `orchestrator.py` | Primary path: thread `AnalysisState` through the staged pipeline; falls back to `runner.run_analysis` on any stage error. |
| `reporting.py` | Deterministic fallback report (the orchestrated path uses `skills/report_writing`). |
| `runner.py` | Deterministic fallback pipeline + shared submission build/validation helpers. |

## Execution Rules

- Treat `data/DATA_DESCRIPTION.md` as the source of truth for hidden data.
- Never hardcode overdose-specific names, category values, target names, row
  counts, period ids, or local absolute paths.
- Preserve the sample submission row order.
- Write root-level `submission.csv` and `report.pdf`.
- If LightGBM is unavailable, allow the scikit-learn fallback candidates to run.
- If the pipeline fails, inspect the traceback and repair the generic pipeline;
  do not patch in hidden-dataset-specific constants.

## Required Verification

After `python main.py`, verify:

```bash
python - <<'PY'
import glob, json
from pathlib import Path
import pandas as pd

assert Path("submission.csv").exists(), "missing submission.csv"
assert Path("report.pdf").exists(), "missing report.pdf"

# The pipeline writes a task-aware submission check; assert its booleans rather than
# hardcoding a column name or assuming numeric predictions (labels may be strings).
checks = sorted(glob.glob("outputs/logs/*_submission_check.json"))
assert checks, "no submission_check log found"
chk = json.load(open(checks[-1]))
for key in ("columns_ok", "row_count_ok", "row_id_alignment_ok", "all_finite", "dtype_ok"):
    assert chk.get(key) is True, (key, chk)
assert chk.get("missing_predictions") == 0, chk

sub = pd.read_csv("submission.csv")
assert sub.shape[1] == 2, sub.columns.tolist()
print("verification passed", sub.shape, sub.columns.tolist(), "| output:", chk.get("output_kind"))
PY
```

The submission's row-id and target column names, value dtype, and label set are all
verified inside `outputs/logs/{run_id}_submission_check.json`, so the check above is
correct for any hidden dataset (integer 0/1 labels, string class labels, or
continuous/probability values).
