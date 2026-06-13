"""Schema discovery for Award B hidden datasets.

The competition harness adds ``data/DATA_DESCRIPTION.md`` at evaluation time.
The description is the authority, but hidden descriptions may vary in wording,
so this module combines lightweight text parsing with file inspection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import re
from typing import Iterable

import pandas as pd


SUPPORTED_TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet"}


@dataclass
class SchemaSpec:
    data_dir: str
    description_path: str
    train_target_file: str
    train_covariates_file: str | None
    validation_covariates_file: str | None
    sample_submission_file: str
    row_id_column: str
    target_column: str
    join_keys: list[str] = field(default_factory=list)
    time_column: str | None = None
    category_column: str | None = None
    block_column: str | None = None
    inferred_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def discover_schema(data_dir: Path) -> SchemaSpec:
    data_dir = data_dir.resolve()
    description_path = data_dir / "DATA_DESCRIPTION.md"
    if not description_path.exists():
        raise FileNotFoundError(
            "Expected data/DATA_DESCRIPTION.md. Award B evaluation provides this file."
        )

    description = description_path.read_text(encoding="utf-8", errors="replace")
    table_files = _find_table_files(data_dir)
    if not table_files:
        raise FileNotFoundError("No supported tabular files found under data/.")

    profiles = [_profile_table(path) for path in table_files]
    sample_profile = _choose_sample_submission(profiles, description)
    if sample_profile is None:
        raise ValueError("Could not identify a sample submission file.")

    row_id_col = _choose_row_id(sample_profile["columns"], description)
    target_col = _choose_target_column(sample_profile["columns"], row_id_col, description)

    target_profile = _choose_train_target(profiles, sample_profile, target_col, description)
    if target_profile is None:
        raise ValueError(
            f"Could not find a training target file containing target column '{target_col}'."
        )

    train_cov_profile = _choose_covariates(
        profiles=profiles,
        target_profile=target_profile,
        sample_profile=sample_profile,
        target_col=target_col,
        description=description,
        role="train",
    )
    val_cov_profile = _choose_covariates(
        profiles=profiles,
        target_profile=target_profile,
        sample_profile=sample_profile,
        target_col=target_col,
        description=description,
        role="validation",
    )

    join_keys = _infer_join_keys(
        target_profile=target_profile,
        train_cov_profile=train_cov_profile,
        val_cov_profile=val_cov_profile,
        sample_profile=sample_profile,
        row_id_col=row_id_col,
        target_col=target_col,
        description=description,
    )
    # Columns shared between target file and sample submission but absent from
    # covariates (so not join keys) may still be category/block dimensions.
    non_key_shared = [
        col for col in target_profile["columns"]
        if col in sample_profile["columns"]
        and col not in join_keys
        and col not in {row_id_col, target_col}
    ]
    time_col = _infer_special_column(join_keys, "time", description)
    # Fallback: if no join-key was flagged as the time axis, check whether the
    # row_id column itself is temporal (e.g. a "datetime" or "timestamp" ID).
    # This enables time-based holdout splitting for datasets where the row
    # identifier is the only time column.
    if time_col is None and row_id_col:
        _time_patterns = ["period", "date", "month", "week", "time", "year", "quarter", "stamp"]
        if any(p in _norm(row_id_col) for p in _time_patterns):
            time_col = row_id_col
    category_col = _infer_special_column(join_keys, "category", description, extra_candidates=non_key_shared)
    block_col = _infer_block_column(join_keys, category_col, description, extra_candidates=non_key_shared)
    # Block-averaged MAE on a temporal panel blocks by the forecast PERIOD: the
    # validation rows are whole held-out periods, so per-period MAE averaging
    # mirrors the evaluation (and the blocked CV holds whole periods out). Prefer
    # the time column as the block when one exists; otherwise keep the
    # category/group block. Dataset-agnostic — no column names are hardcoded.
    if time_col:
        block_col = time_col

    notes = []
    for label, profile in [
        ("train_target_file", target_profile),
        ("train_covariates_file", train_cov_profile),
        ("validation_covariates_file", val_cov_profile),
        ("sample_submission_file", sample_profile),
    ]:
        if profile:
            notes.append(f"{label}: {profile['path'].relative_to(data_dir)}")
    if not join_keys:
        notes.append("No explicit join keys inferred; row-order fallback may be used.")

    return SchemaSpec(
        data_dir=str(data_dir),
        description_path=str(description_path),
        train_target_file=str(target_profile["path"]),
        train_covariates_file=str(train_cov_profile["path"]) if train_cov_profile else None,
        validation_covariates_file=str(val_cov_profile["path"]) if val_cov_profile else None,
        sample_submission_file=str(sample_profile["path"]),
        row_id_column=row_id_col,
        target_column=target_col,
        join_keys=join_keys,
        time_column=time_col,
        category_column=category_col,
        block_column=block_col,
        inferred_notes=notes,
    )


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_schema_json(spec: SchemaSpec, path: Path) -> None:
    path.write_text(json.dumps(spec.to_dict(), indent=2), encoding="utf-8")


def _find_table_files(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and path.name != "DATA_DESCRIPTION.md"
        and path.suffix.lower() in SUPPORTED_TABLE_SUFFIXES
    )


def _profile_table(path: Path) -> dict:
    try:
        df = read_table(path)
    except Exception as exc:  # pragma: no cover - profile captures bad files
        return {"path": path, "columns": [], "n_rows": 0, "error": str(exc)}
    return {
        "path": path,
        "columns": [str(c) for c in df.columns],
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
    }


def _choose_sample_submission(profiles: list[dict], description: str) -> dict | None:
    mentioned = _mentioned_files(description)
    ranked = []
    for profile in profiles:
        name = _norm(profile["path"].name)
        columns = [_norm(c) for c in profile["columns"]]
        score = 0
        if profile["path"].name in mentioned:
            score += 4
        if "sample" in name:
            score += 3
        if "submission" in name or "submit" in name:
            score += 4
        if "row_id" in columns or "rowid" in columns:
            score += 3
        if profile.get("n_columns", 0) >= 2:
            score += 1
        if score:
            ranked.append((score, profile))
    return max(ranked, key=lambda item: item[0])[1] if ranked else None


def _choose_row_id(columns: list[str], description: str) -> str:
    direct = _direct_column_match(description, ["row id column", "row_id column", "row id", "identifier column"])
    if direct and direct in columns:
        return direct
    exact = _find_named_column(columns, ["row_id", "rowid", "id"])
    if exact:
        return exact
    described = _description_column(description, ["row_id", "row id", "identifier"])
    if described and described in columns:
        return described
    return columns[0]


def _choose_target_column(columns: list[str], row_id_col: str, description: str) -> str:
    # Strongest signal: explicit verb directive like "Predict `column`" or "Fill `column`"
    for verb in ["predict", "fill", "estimate", "forecast"]:
        m = re.search(rf'\b{verb}\b[^.`]{{0,30}}`([^`]+)`', description, re.I)
        if m:
            col = m.group(1).strip()
            if col in columns and col != row_id_col:
                return col

    direct = _direct_column_match(
        description,
        ["target column", "target", "prediction column", "outcome column", "response column"],
    )
    if direct and direct in columns and direct != row_id_col:
        return direct
    described = _description_column(
        description,
        ["target", "prediction", "must look like"],
    )
    if described and described in columns and described != row_id_col:
        return described
    non_id = [col for col in columns if col != row_id_col]
    if len(non_id) == 1:
        return non_id[0]
    for col in non_id:
        ncol = _norm(col)
        if any(token in ncol for token in ["target", "rate", "value", "score", "count"]):
            return col
    return non_id[-1]


def _choose_train_target(
    profiles: list[dict], sample_profile: dict, target_col: str, description: str
) -> dict | None:
    ranked = []
    for profile in profiles:
        if profile["path"] == sample_profile["path"]:
            continue
        if target_col not in profile["columns"]:
            continue
        name = _norm(str(profile["path"]))
        score = 5
        if "train" in name:
            score += 3
        if any(token in name for token in ["target", "label", "truth", "outcome"]):
            score += 3
        if any(token in name for token in ["submission", "sample"]):
            score -= 4
        ranked.append((score, profile))
    return max(ranked, key=lambda item: item[0])[1] if ranked else None


def _choose_covariates(
    profiles: list[dict],
    target_profile: dict,
    sample_profile: dict,
    target_col: str,
    description: str,
    role: str,
) -> dict | None:
    ranked = []
    target_cols = set(target_profile["columns"])
    sample_cols = set(sample_profile["columns"])
    for profile in profiles:
        if profile["path"] in {target_profile["path"], sample_profile["path"]}:
            continue
        if target_col in profile["columns"]:
            continue
        columns = set(profile["columns"])
        common_with_target = len((columns & target_cols) - {target_col})
        common_with_sample = len((columns & sample_cols) - {target_col})
        name = _norm(str(profile["path"]))
        score = common_with_target + common_with_sample
        if any(token in name for token in ["covariate", "feature", "x_"]):
            score += 3
        if role == "train":
            if "train" in name:
                score += 4
            if any(token in name for token in ["val", "valid", "test", "holdout"]):
                score -= 4
        else:
            if any(token in name for token in ["val", "valid", "test", "holdout", "predict"]):
                score += 4
            if "train" in name:
                score -= 4
            if columns & sample_cols:
                score += 2
        if score > 0:
            ranked.append((score, profile))
    return max(ranked, key=lambda item: item[0])[1] if ranked else None


def _infer_join_keys(
    target_profile: dict,
    train_cov_profile: dict | None,
    val_cov_profile: dict | None,
    sample_profile: dict,
    row_id_col: str,
    target_col: str,
    description: str,
) -> list[str]:
    excluded = {row_id_col, target_col}
    described = _description_columns(description)
    tables: list[Iterable[str]] = [target_profile["columns"], sample_profile["columns"]]
    if train_cov_profile:
        tables.append(train_cov_profile["columns"])
    if val_cov_profile:
        tables.append(val_cov_profile["columns"])

    common = set(tables[0])
    for cols in tables[1:]:
        common &= set(cols)
    keys = [col for col in target_profile["columns"] if col in common and col not in excluded]

    if not keys:
        target_sample_common = (
            set(target_profile["columns"]) & set(sample_profile["columns"]) - excluded
        )
        keys = [col for col in sample_profile["columns"] if col in target_sample_common]

    described_keys = [col for col in described if col in set().union(*(set(t) for t in tables))]
    for col in described_keys:
        if col not in keys and col not in excluded:
            keys.append(col)
    return keys


def _infer_special_column(
    keys: list[str], kind: str, description: str, extra_candidates: list[str] | None = None
) -> str | None:
    if kind == "time":
        patterns = ["period", "date", "month", "week", "time", "year", "quarter"]
    else:
        patterns = ["category", "type", "group", "class", "segment", "series", "measure"]
    all_candidates = list(keys) + (extra_candidates or [])
    for col in all_candidates:
        ncol = _norm(col)
        if any(pattern in ncol for pattern in patterns):
            return col
    described = _description_column(description, patterns)
    if described in set(all_candidates):
        return described
    return None


def _infer_block_column(
    keys: list[str], category_col: str | None, description: str, extra_candidates: list[str] | None = None
) -> str | None:
    all_candidates = list(keys) + (extra_candidates or [])
    described = _description_column(description, ["block", "category", "group"])
    if described in set(all_candidates):
        return described
    return category_col


def _mentioned_files(description: str) -> set[str]:
    names = set()
    for match in re.finditer(r"[\w./-]+\.(?:csv|tsv|txt|xlsx|xls|parquet)", description, re.I):
        names.add(Path(match.group(0)).name)
    return names


def parse_period_date_map(data_dir: Path) -> dict[str, str]:
    """Parse the period_id → date string mapping from DATA_DESCRIPTION.md.

    Returns an OrderedDict-like plain dict sorted chronologically (earliest first),
    covering ALL periods (training + validation).  Keys are period_id strings;
    values are ISO date strings (e.g. '2019-01-31').

    Falls back to an empty dict if the mapping table is absent or unparseable.
    """
    desc_path = Path(data_dir) / "DATA_DESCRIPTION.md"
    if not desc_path.exists():
        return {}
    text = desc_path.read_text(encoding="utf-8", errors="replace")
    # Match markdown table rows: | date | period_id |  (or reversed order)
    # The table in DATA_DESCRIPTION.md looks like:
    #   | `2019-01-31` | `uTjgI1Sv` |
    date_pid: list[tuple[str, str]] = []
    for m in re.finditer(
        r"\|\s*`?(\d{4}-\d{2}-\d{2})`?\s*\|\s*`?([A-Za-z0-9]{6,12})`?\s*\|",
        text,
    ):
        date_pid.append((m.group(1), m.group(2)))
    if not date_pid:
        return {}
    # Sort by date to guarantee chronological order
    date_pid.sort(key=lambda x: x[0])
    return {pid: date for date, pid in date_pid}


def _description_columns(description: str) -> list[str]:
    found: list[str] = []
    for match in re.finditer(r"`([^`]+)`", description):
        token = match.group(1).strip()
        if token and "/" not in token and "." not in token and token not in found:
            found.append(token)
    return found


def _description_column(description: str, context_terms: list[str]) -> str | None:
    columns = _description_columns(description)
    if not columns:
        return None
    lower_description = description.lower()
    for col in columns:
        idx = lower_description.find(col.lower())
        if idx < 0:
            continue
        window = lower_description[max(0, idx - 120) : idx + 120]
        if any(term.lower() in window for term in context_terms):
            return col
    return None


def _direct_column_match(description: str, labels: list[str]) -> str | None:
    for label in labels:
        escaped = re.escape(label)
        patterns = [
            rf"{escaped}\s*(?:is|:|=)\s*`([^`]+)`",
            rf"{escaped}\s*(?:is|:|=)\s*([A-Za-z_][\w]*)",
        ]
        for pattern in patterns:
            match = re.search(pattern, description, flags=re.I)
            if match:
                return match.group(1).strip()
    return None


def _find_named_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {_norm(col): col for col in columns}
    for candidate in candidates:
        if _norm(candidate) in normalized:
            return normalized[_norm(candidate)]
    return None


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
