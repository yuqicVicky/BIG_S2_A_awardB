"""data-loader skill: load a CSV/XLSX file into a DataFrame, unmodified.

Implements the contract in ``.claude/skills/data-loader/SKILL.md``: CSV and XLSX
only, no transformation/imputation, explicit error conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

_SUPPORTED = {".csv": "csv", ".xlsx": "xlsx", ".xls": "xlsx"}


@dataclass
class LoadResult:
    df: pd.DataFrame
    n_rows: int
    n_columns: int
    file_format: str
    encoding: str
    warnings: list[dict] = field(default_factory=list)

    def metadata(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "file_format": self.file_format,
            "encoding": self.encoding,
            "load_warnings": self.warnings,
        }


def load_dataset(file_path: str) -> LoadResult:
    path = Path(file_path)
    ext = path.suffix.lower()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if ext not in _SUPPORTED:
        raise ValueError(f"Unsupported file format: {ext}")

    warnings: list[dict] = []
    encoding = "utf-8"

    if ext == ".csv":
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except UnicodeDecodeError:
            encoding = "latin-1"
            df = pd.read_csv(path, encoding="latin-1")
            warnings.append({
                "type": "encoding_fallback", "column": None,
                "message": "utf-8 decode failed; loaded with latin-1.", "severity": "WARN",
            })
        except Exception as exc:  # pragma: no cover - surfaced with path
            raise type(exc)(f"Parse failure for {file_path}: {exc}") from exc
    else:
        try:
            excel = pd.ExcelFile(path)
            if len(excel.sheet_names) > 1:
                warnings.append({
                    "type": "multi_sheet", "column": None,
                    "message": f"Multiple sheets {excel.sheet_names}; read first '{excel.sheet_names[0]}'.",
                    "severity": "WARN",
                })
            df = pd.read_excel(path, sheet_name=excel.sheet_names[0])
            encoding = "binary"
        except Exception as exc:  # pragma: no cover
            raise type(exc)(f"Parse failure for {file_path}: {exc}") from exc

    if len(df) == 0:
        raise ValueError("Dataset is empty")

    return LoadResult(
        df=df,
        n_rows=int(len(df)),
        n_columns=int(df.shape[1]),
        file_format=_SUPPORTED[ext],
        encoding=encoding,
        warnings=warnings,
    )
