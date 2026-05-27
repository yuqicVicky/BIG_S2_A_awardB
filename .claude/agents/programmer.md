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
| `schema.py` | Parse `data/DATA_DESCRIPTION.md` and infer file roles, target, row id, join keys, time column, and block/category column. |
| `features.py` | Build train and prediction feature frames without using validation targets. |
| `models.py` | Evaluate baselines and candidate models, then select the best model by holdout MAE or block-averaged MAE. |
| `reporting.py` | Generate the dynamic Markdown/PDF report for the current run. |
| `runner.py` | Orchestrate the full end-to-end pipeline. |

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
from pathlib import Path
import pandas as pd
import numpy as np

assert Path("submission.csv").exists(), "missing submission.csv"
assert Path("report.pdf").exists(), "missing report.pdf"
sub = pd.read_csv("submission.csv")
assert sub.shape[1] == 2, sub.columns.tolist()
assert sub.columns[0] == "row_id", sub.columns.tolist()
pred = pd.to_numeric(sub.iloc[:, 1], errors="coerce")
assert pred.notna().all(), "missing predictions"
assert np.isfinite(pred).all(), "non-finite predictions"
print("verification passed", sub.shape, sub.columns.tolist())
PY
```

If `DATA_DESCRIPTION.md` specifies a different row-id column name, use that
schema instead of the literal `row_id` in the verification.
