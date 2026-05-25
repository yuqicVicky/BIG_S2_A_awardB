---
name: data-loader
description: Use this skill to load a CSV or XLSX data file into a DataFrame without modification. Returns shape, column names, dtypes, encoding used, and any load warnings. This is the mandatory first step before any profiling or analysis.
---

# Skill: data-loader

## When to Use

Use this skill as the very first step of every analysis run, immediately after receiving a valid file path from the user. All subsequent steps depend on the raw DataFrame produced here.

## Python Function

```python
from data_agent.skills.load_data import load_dataset

result = load_dataset(file_path)
# result.df          — raw, unmodified DataFrame
# result.n_rows      — integer row count
# result.n_columns   — integer column count
# result.file_format — "csv" or "xlsx"
# result.encoding    — encoding detected or used
# result.warnings    — list of non-fatal load issues
```

## Inputs

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `file_path` | `str` | Yes | Absolute or relative path to a `.csv` or `.xlsx` file |

## Outputs Written to State

| State field | Content |
|-------------|---------|
| `state.load_metadata` | `{n_rows, n_columns, file_format, encoding, load_warnings}` |
| `state.raw_data` | The raw, unmodified `DataFrame` |

## Constraints

- **Do not modify the data.** The returned DataFrame must be an exact in-memory copy of the file contents.
- **Do not infer or impute.** Missing values must remain as-is.
- **Supported formats:** CSV (any delimiter auto-detected) and XLSX (first sheet by default).
- **Unsupported formats** (JSON, Parquet, etc.) must raise an error immediately; do not attempt to load.

## Error Conditions

| Condition | Required Action |
|-----------|----------------|
| File not found | Raise `FileNotFoundError`; halt pipeline |
| Unsupported extension | Raise `ValueError("Unsupported file format: {ext}")` |
| Zero rows after load | Raise `ValueError("Dataset is empty")` |
| Parse failure | Raise original exception with file path in message |

## Inspector Checks

After this step, the Inspector must verify:

1. `state.load_metadata` is populated with all four required keys.
2. `state.raw_data` has `n_rows > 0` and `n_columns > 0`.
3. No transformation has been applied to the data (dtypes match raw file).

## Example Output

```json
{
  "n_rows": 891,
  "n_columns": 12,
  "file_format": "csv",
  "encoding": "utf-8",
  "load_warnings": []
}
```

## Notes

- Large files (>500 MB) will be loaded in full. There is no chunked-read mode in V1.
- Multi-sheet XLSX files: only the first sheet is read. Sheet name is included in `load_metadata` as a warning if more than one sheet is detected.
