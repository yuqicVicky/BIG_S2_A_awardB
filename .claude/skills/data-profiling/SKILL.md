---
name: data-profiling
description: Use this skill when loading or inspecting a tabular dataset before analysis. It covers file format, rows, columns, dtypes, missingness, duplicates, numeric columns, categorical columns, datetime-like columns, potential ID columns, constant columns, high-cardinality columns, and data quality warnings.
---

# Skill: data-profiling

## When to Use

Use this skill immediately after `data-loader` completes and before any analysis, planning, or modeling begins. It must run on the **raw, unmodified** DataFrame exactly as loaded from disk.

Trigger this skill whenever:
- A new data file has been loaded for the first time.
- The user asks to "understand", "inspect", or "describe" a dataset.
- `task_inference` needs a data profile to reason about task type.
- `plan_critique` requires evidence about data quality before approving a plan.

Do **not** defer this skill to after preprocessing — it must always see the raw data.

---

## Required Checks

Run every check in this list, in order. No check may be skipped. Each check produces one or more fields in the output profile.

### 1. File Format

Detect and record the source file format from the file extension or MIME type.

- Accepted values: `csv`, `xlsx`
- Record delimiter (for CSV), sheet name (for XLSX), and encoding used.

### 2. Row Count

Record the total number of rows in the loaded DataFrame (`len(df)`). Flag if row count is zero.

### 3. Column Count

Record the total number of columns (`len(df.columns)`). Flag if column count is zero or equals one (likely a parsing error).

### 4. Column Names

Record the full list of column names in original order. Flag any of:
- Unnamed columns (e.g., `Unnamed: 0`) — likely an artifact of index export.
- Duplicate column names.
- Column names containing only whitespace.
- Column names that are purely numeric strings.

### 5. Dtypes

Record the pandas dtype for every column. Map to a simplified type category:
- `int` / `float` → `numeric`
- `object` / `string` → `categorical`
- `datetime64` / `timedelta` → `datetime`
- `bool` → `boolean`
- `category` → `categorical`

### 6. Missingness by Column

For every column compute:
- `missing_count`: number of null values.
- `missing_rate`: `missing_count / n_rows`, expressed as a float 0–1.

Classify:
- `missing_rate == 0` → `complete`
- `0 < missing_rate < 0.05` → `low_missing`
- `0.05 <= missing_rate < 0.20` → `moderate_missing`
- `0.20 <= missing_rate < 0.50` → `high_missing`
- `missing_rate >= 0.50` → `critical_missing` → **produce a WARNING**

### 7. Duplicated Rows

Count the number of fully duplicated rows (`df.duplicated().sum()`). If `duplicate_count > 0`, produce a WARNING with the count and the percentage of total rows.

### 8. Numeric Columns

For each numeric column record:
- `min`, `max`, `mean`, `std`, `median`
- `p25`, `p75` (25th and 75th percentiles)
- `n_zeros`: count of zero values
- `n_negatives`: count of negative values

Flag if `std == 0` (constant column — see check 12).

### 9. Categorical Columns

For each categorical column record:
- `n_unique`: number of distinct values
- `top_values`: top 5 most frequent values with their counts
- `mode`: single most frequent value

Flag if `n_unique == 1` (constant column — see check 12).

### 10. Datetime-Like Columns

Identify columns that appear to represent dates or times even if stored as strings or objects. Detection heuristics:
- Column name contains: `date`, `time`, `dt`, `year`, `month`, `day`, `timestamp`, `created`, `updated`.
- A sample of values successfully parses with `pd.to_datetime(..., errors='coerce')` with fewer than 10% parse failures.

Record:
- `detected_as_datetime`: `true` / `false`
- `inferred_format`: best-guess format string if parseable
- `min_date`, `max_date` if parseable

Produce a WARNING if a datetime-like column is currently typed as `object` — it should be converted before analysis.

### 11. Potential ID Columns

Flag a column as a potential ID column if it meets any of:
- Column name contains: `id`, `_id`, `uuid`, `key`, `index`, `code`, `ref`.
- `n_unique == n_rows` (every value is unique).
- Values are purely numeric integers with no semantic range (e.g., sequential integers starting at 0 or 1).

Produce a WARNING for each potential ID column: it must be excluded from features to avoid data leakage.

### 12. Constant Columns

Flag a column as constant if:
- `n_unique == 1` (after ignoring nulls), or
- `std == 0` for numeric columns.

Produce a WARNING for each constant column: it carries no information and should be dropped in preprocessing.

### 13. High-Cardinality Categorical Columns

Flag a categorical column as high-cardinality if `n_unique / n_rows > 0.50` **and** `n_unique > 20`.

Produce a WARNING: high-cardinality columns may require special encoding (target encoding, hashing) or should be reviewed for accidental ID status.

### 14. Suspicious Target-Like or Leakage-Like Columns

Heuristically identify columns that may be the target variable or may cause leakage:

**Target-like columns** — column name contains any of: `target`, `label`, `outcome`, `y`, `response`, `churn`, `survived`, `result`, `flag`, `class`, `status`.

**Leakage-like columns** — column name contains any of: `future`, `post_`, `after_`, `next_`, `leaked`, `score` (when paired with a classification target), or values that appear to be model outputs (e.g., probabilities in [0, 1] with near-perfect correlation to another column).

Produce a WARNING for each suspicious column. These must be reviewed by `leakage-check` before any modeling.

---

## Output Format

The profile must be written to `state.data_profile` as a structured object. The canonical JSON schema:

```json
{
  "data_type": "tabular",
  "file_format": "csv",
  "encoding": "utf-8",
  "delimiter": ",",
  "n_rows": 891,
  "n_columns": 12,
  "columns": [
    "PassengerId", "Survived", "Pclass", "Name", "Sex",
    "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"
  ],
  "dtypes": {
    "PassengerId": "int64",
    "Survived": "int64",
    "Pclass": "int64",
    "Name": "object",
    "Sex": "object",
    "Age": "float64",
    "SibSp": "int64",
    "Parch": "int64",
    "Ticket": "object",
    "Fare": "float64",
    "Cabin": "object",
    "Embarked": "object"
  },
  "simplified_types": {
    "PassengerId": "numeric",
    "Survived": "numeric",
    "Pclass": "numeric",
    "Name": "categorical",
    "Sex": "categorical",
    "Age": "numeric",
    "SibSp": "numeric",
    "Parch": "numeric",
    "Ticket": "categorical",
    "Fare": "numeric",
    "Cabin": "categorical",
    "Embarked": "categorical"
  },
  "numeric_columns": {
    "Age": {
      "min": 0.42, "max": 80.0, "mean": 29.70, "std": 14.53,
      "median": 28.0, "p25": 20.12, "p75": 38.0,
      "n_zeros": 0, "n_negatives": 0
    },
    "Fare": {
      "min": 0.0, "max": 512.33, "mean": 32.20, "std": 49.69,
      "median": 14.45, "p25": 7.91, "p75": 31.0,
      "n_zeros": 15, "n_negatives": 0
    }
  },
  "categorical_columns": {
    "Sex": {
      "n_unique": 2,
      "top_values": {"male": 577, "female": 314},
      "mode": "male"
    },
    "Embarked": {
      "n_unique": 3,
      "top_values": {"S": 644, "C": 168, "Q": 77},
      "mode": "S"
    }
  },
  "datetime_columns": [],
  "potential_id_columns": ["PassengerId"],
  "constant_columns": [],
  "high_cardinality_columns": ["Name", "Ticket", "Cabin"],
  "target_like_columns": ["Survived"],
  "leakage_like_columns": [],
  "missing_summary": {
    "Age":      {"missing_count": 177, "missing_rate": 0.1987, "severity": "high_missing"},
    "Cabin":    {"missing_count": 687, "missing_rate": 0.7710, "severity": "critical_missing"},
    "Embarked": {"missing_count": 2,   "missing_rate": 0.0022, "severity": "low_missing"}
  },
  "duplicate_count": 0,
  "warnings": [
    {
      "type": "critical_missing",
      "column": "Cabin",
      "message": "77.1% of values are missing. Consider dropping or using as binary indicator.",
      "severity": "WARN"
    },
    {
      "type": "potential_id_column",
      "column": "PassengerId",
      "message": "All 891 values are unique. Must be excluded from features.",
      "severity": "WARN"
    },
    {
      "type": "high_cardinality",
      "column": "Name",
      "message": "891 unique values out of 891 rows (100%). Likely an identifier.",
      "severity": "WARN"
    },
    {
      "type": "target_like_column",
      "column": "Survived",
      "message": "Column name suggests it may be the target variable. Confirm before modeling.",
      "severity": "WARN"
    }
  ]
}
```

All fields are required. If a field has no entries, record it as an empty list `[]` or empty object `{}`, never as `null`.

---

## Warnings to Produce

Every warning must be written to `profile.warnings` as an object with:
- `type`: one of the types below
- `column`: the column name (or `null` for dataset-level warnings)
- `message`: a human-readable explanation
- `severity`: always `"WARN"` for profiling (profiling never produces `FAIL`)

| Warning Type | Trigger Condition |
|---|---|
| `critical_missing` | `missing_rate >= 0.50` for any column |
| `high_missing` | `0.20 <= missing_rate < 0.50` for any column |
| `zero_rows` | `n_rows == 0` |
| `single_column` | `n_columns == 1` |
| `unnamed_column` | Column name matches `Unnamed: \d+` |
| `duplicate_column_names` | Any two columns share the same name |
| `duplicate_rows` | `duplicate_count > 0` |
| `potential_id_column` | Column meets ID detection criteria (check 11) |
| `constant_column` | `n_unique == 1` or `std == 0` (check 12) |
| `high_cardinality` | `n_unique / n_rows > 0.50` and `n_unique > 20` (check 13) |
| `datetime_as_object` | Datetime-like column is typed as `object` (check 10) |
| `target_like_column` | Column name suggests it is the target (check 14) |
| `leakage_like_column` | Column name or values suggest leakage risk (check 14) |

Profiling does not suppress or deduplicate warnings. Every detected issue produces its own warning entry.

---

## Do Not

- **Do not train any model.** Profiling is read-only analysis of the raw data.
- **Do not modify the original DataFrame.** No column drops, type casts, fills, or transformations. Write results to `state.data_profile` only.
- **Do not delete any column** from the DataFrame, even if it appears useless. Recording its existence is the profiler's job. Deletion decisions belong to `feature-engineer`.
- **Do not write final conclusions** about the dataset (e.g., "this dataset is ready for modeling", "Age should be imputed with median"). Record facts and warnings. Interpretation belongs to `task_inference` and `plan_critique`.
- **Do not infer the task type.** That is the responsibility of `task_inference`. The profiler records what is in the data, not what should be done with it.
- **Do not access the test set.** If a split already exists in state, profile only the raw pre-split data.
