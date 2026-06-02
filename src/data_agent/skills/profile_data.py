"""data-profiling skill: structured profile of a raw, unmodified DataFrame.

Implements every check in ``.claude/skills/data-profiling/SKILL.md`` and emits the
canonical ``DataProfile`` schema. Read-only: never mutates the DataFrame.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

# keyword sets (matched against name tokens, see _tokens)
_ID_TOKENS = {"id", "uuid", "key", "index", "code", "ref"}
_DATETIME_TOKENS = {"date", "time", "dt", "year", "month", "day", "timestamp", "created", "updated"}
_TARGET_TOKENS = {"target", "label", "outcome", "y", "response", "churn", "survived", "result", "flag", "class", "status"}
_LEAKAGE_TOKENS = {"future", "post", "after", "next", "leaked", "score"}


class DataProfile(BaseModel):
    data_type: str = "tabular"
    file_format: str = "csv"
    encoding: str = "utf-8"
    delimiter: str | None = ","
    n_rows: int = 0
    n_columns: int = 0
    columns: list[str] = []
    dtypes: dict[str, str] = {}
    simplified_types: dict[str, str] = {}
    numeric_columns: dict[str, dict] = {}
    categorical_columns: dict[str, dict] = {}
    datetime_columns: list[dict] = []
    potential_id_columns: list[str] = []
    constant_columns: list[str] = []
    high_cardinality_columns: list[str] = []
    target_like_columns: list[str] = []
    leakage_like_columns: list[str] = []
    missing_summary: dict[str, dict] = {}
    duplicate_count: int = 0
    warnings: list[dict] = []


def _tokens(name: str) -> set[str]:
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return {p.lower() for p in re.split(r"[^A-Za-z0-9]+", spaced) if p}


def _simplified_type(dtype: Any) -> str:
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
        return "datetime"
    if pd.api.types.is_numeric_dtype(dtype):
        return "numeric"
    return "categorical"


def _missing_severity(rate: float) -> str:
    if rate == 0:
        return "complete"
    if rate < 0.05:
        return "low_missing"
    if rate < 0.20:
        return "moderate_missing"
    if rate < 0.50:
        return "high_missing"
    return "critical_missing"


def _py(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def profile_tabular_data(df: pd.DataFrame, file_path: str) -> DataProfile:
    n_rows = int(len(df))
    n_columns = int(df.shape[1])
    ext = Path(file_path).suffix.lower()
    file_format = "xlsx" if ext in {".xlsx", ".xls"} else "csv"
    columns = [str(c) for c in df.columns]
    warnings: list[dict] = []

    def warn(wtype: str, column: str | None, message: str) -> None:
        warnings.append({"type": wtype, "column": column, "message": message, "severity": "WARN"})

    if n_rows == 0:
        warn("zero_rows", None, "Dataset has zero rows.")
    if n_columns == 1:
        warn("single_column", None, "Only one column detected; possible parsing error.")

    # 4. column-name anomalies
    seen: set[str] = set()
    for col in columns:
        if re.match(r"^Unnamed: \d+$", col):
            warn("unnamed_column", col, "Unnamed column; likely an index-export artifact.")
        if col in seen:
            warn("duplicate_column_names", col, "Duplicate column name detected.")
        seen.add(col)

    # 5. dtypes / simplified types
    dtypes = {c: str(df[c].dtype) for c in columns}
    simplified = {c: _simplified_type(df[c].dtype) for c in columns}

    numeric_columns: dict[str, dict] = {}
    categorical_columns: dict[str, dict] = {}
    datetime_columns: list[dict] = []
    potential_id_columns: list[str] = []
    constant_columns: list[str] = []
    high_cardinality_columns: list[str] = []
    target_like_columns: list[str] = []
    leakage_like_columns: list[str] = []
    missing_summary: dict[str, dict] = {}

    for col in columns:
        series = df[col]
        non_null = series.dropna()
        n_unique = int(non_null.nunique())
        tokens = _tokens(col)

        # 6. missingness
        missing_count = int(series.isna().sum())
        rate = float(missing_count / n_rows) if n_rows else 0.0
        severity = _missing_severity(rate)
        missing_summary[col] = {"missing_count": missing_count, "missing_rate": round(rate, 4), "severity": severity}
        if severity == "critical_missing":
            warn("critical_missing", col, f"{rate:.1%} of values are missing.")
        elif severity == "high_missing":
            warn("high_missing", col, f"{rate:.1%} of values are missing.")

        kind = simplified[col]

        # 8. numeric stats
        if kind == "numeric" and len(non_null):
            numeric = pd.to_numeric(non_null, errors="coerce").dropna()
            std = float(numeric.std()) if len(numeric) > 1 else 0.0
            numeric_columns[col] = {
                "min": _py(numeric.min()), "max": _py(numeric.max()),
                "mean": float(numeric.mean()), "std": std,
                "median": float(numeric.median()),
                "p25": float(numeric.quantile(0.25)), "p75": float(numeric.quantile(0.75)),
                "n_zeros": int((numeric == 0).sum()), "n_negatives": int((numeric < 0).sum()),
            }

        # 9. categorical stats
        if kind in {"categorical", "boolean"} and len(non_null):
            counts = non_null.value_counts()
            top = {str(_py(k)): int(v) for k, v in counts.head(5).items()}
            categorical_columns[col] = {
                "n_unique": n_unique,
                "top_values": top,
                "mode": str(_py(counts.index[0])) if len(counts) else None,
            }

        # 10. datetime-like
        looks_dt = bool(tokens & _DATETIME_TOKENS)
        if simplified[col] == "datetime":
            datetime_columns.append({"column": col, "detected_as_datetime": True, "inferred_format": None})
        elif kind == "categorical" and len(non_null) and (looks_dt or _is_stringy(non_null)):
            parsed = _try_parse_dates(non_null)
            if parsed is not None and parsed.notna().any() and parsed.notna().mean() >= 0.90:
                datetime_columns.append({
                    "column": col, "detected_as_datetime": True, "inferred_format": None,
                    "min_date": str(parsed.min()), "max_date": str(parsed.max()),
                })
                warn("datetime_as_object", col, "Datetime-like column stored as object; convert before analysis.")

        # 11. potential ID (name pattern, or unique integer key)
        is_int_like = kind == "numeric" and pd.api.types.is_integer_dtype(series.dropna().infer_objects().dtype if len(non_null) else series.dtype)
        if (tokens & _ID_TOKENS) or (n_unique == n_rows and (is_int_like or pd.api.types.is_integer_dtype(series.dtype))):
            potential_id_columns.append(col)
            warn("potential_id_column", col, f"All {n_unique} values unique / id-like name. Exclude from features.")

        # 12. constant
        std_zero = col in numeric_columns and numeric_columns[col]["std"] == 0
        if n_unique <= 1 or std_zero:
            constant_columns.append(col)
            warn("constant_column", col, "Column is constant; carries no information.")

        # 13. high-cardinality categorical
        if kind == "categorical" and n_rows and (n_unique / n_rows) > 0.50 and n_unique > 20:
            high_cardinality_columns.append(col)
            warn("high_cardinality", col, f"{n_unique} unique of {n_rows} rows; may be an identifier or need special encoding.")

        # 14. target-like / leakage-like
        if tokens & _TARGET_TOKENS:
            target_like_columns.append(col)
            warn("target_like_column", col, "Name suggests it may be the target variable.")
        if tokens & _LEAKAGE_TOKENS:
            leakage_like_columns.append(col)
            warn("leakage_like_column", col, "Name suggests possible leakage; review before modeling.")

    # 7. duplicate rows
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        warn("duplicate_rows", None, f"{duplicate_count} duplicated rows ({duplicate_count / n_rows:.1%}).")

    return DataProfile(
        data_type="tabular",
        file_format=file_format,
        encoding="utf-8",
        delimiter="," if file_format == "csv" else None,
        n_rows=n_rows,
        n_columns=n_columns,
        columns=columns,
        dtypes=dtypes,
        simplified_types=simplified,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        potential_id_columns=potential_id_columns,
        constant_columns=constant_columns,
        high_cardinality_columns=high_cardinality_columns,
        target_like_columns=target_like_columns,
        leakage_like_columns=leakage_like_columns,
        missing_summary=missing_summary,
        duplicate_count=duplicate_count,
        warnings=warnings,
    )


def _is_stringy(series: pd.Series) -> bool:
    sample = series.head(200)
    return bool(len(sample)) and bool(sample.map(lambda v: isinstance(v, str)).mean() > 0.5)


def _try_parse_dates(series: pd.Series) -> pd.Series | None:
    import warnings as _w

    sample = series.head(200)
    if not len(sample):
        return None
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        return pd.to_datetime(sample, errors="coerce")
