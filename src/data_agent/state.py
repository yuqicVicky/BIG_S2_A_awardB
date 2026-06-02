"""Shared analysis state threaded through the orchestrated pipeline.

`AnalysisState` is the single object every stage reads from and writes to, and is
persisted to ``outputs/logs/{run_id}_state.json`` after each stage. The field set
covers every ``state.*`` reference in the `.claude/` agent and skill definitions;
where the docs use two names for the same concept, a canonical field is backed by a
read-only property alias (e.g. ``state.task`` -> ``task_spec``).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class UserRequest(BaseModel):
    goal: str = ""
    file_path: str = ""
    raw_message: str | None = None


class AnalysisState(BaseModel):
    # arbitrary_types_allowed: hold DataFrames / fitted estimators as plain attrs.
    # protected_namespaces=(): allow the ``model_results`` field name.
    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    # identity / request
    run_id: str
    created_at: str = ""
    user_request: UserRequest = Field(default_factory=UserRequest)
    data_path: str = ""

    # stage 1 — intake
    load_metadata: dict | None = None
    raw_data: Any = Field(default=None, exclude=True)

    # stage 2 — profile
    data_profile: dict | None = None

    # stage 3 — task
    task_spec: dict | None = None

    # stage 4 / 5 — plan
    initial_plan: dict | None = None
    optimized_plan: dict | None = None

    # stage 6 / 7 — execution / inspection
    execution_logs: dict = Field(default_factory=dict)
    inspection_logs: dict = Field(default_factory=dict)

    # leakage + split + transform
    leakage_audit: dict | None = None
    split_metadata: dict | None = None
    splits: Any = Field(default=None, exclude=True)
    transformers: Any = Field(default=None, exclude=True)
    transformer_record: dict | None = None

    # modeling + evaluation
    model_results: dict | None = None
    raw_modeling_results: dict | None = None
    evaluation: dict | None = None
    eda_results: dict | None = None
    interpretation: dict | None = None

    # report + review
    report_draft: str | None = None
    report_path: str | None = None
    report_review: dict | None = None

    # artifacts + run summary
    artifacts: dict = Field(default_factory=dict)
    summary: dict | None = None

    # ── read-only aliases for the docs' alternate names ──────────────────────
    @property
    def task(self) -> SimpleNamespace:
        # Attribute-accessible view of task_spec (docs use state.task.task_type, etc.).
        return SimpleNamespace(**(self.task_spec or {}))

    @property
    def plan(self) -> SimpleNamespace:
        return SimpleNamespace(draft=self.initial_plan, approved=self.optimized_plan)

    @property
    def execution_log(self) -> dict:
        return self.execution_logs

    @property
    def models(self) -> SimpleNamespace:
        return SimpleNamespace(**(self.model_results or {}))

    # ── helpers ──────────────────────────────────────────────────────────────
    def set_raw_data(self, df: Any) -> None:
        self.raw_data = df

    def register_artifact(self, label: str, path: str | Path) -> None:
        self.artifacts[label] = str(path)

    def log_event(self, step: str, **entry: Any) -> None:
        self.execution_logs[step] = {"step": step, **entry}

    def to_json(self) -> str:
        # Robust against numpy scalars nested inside dict fields; runtime objects
        # (raw_data/splits/transformers) are excluded via Field(exclude=True).
        return json.dumps(self.model_dump(), default=_json_default, indent=2)

    def persist(self, logs_dir: str | Path) -> Path:
        logs_dir = Path(logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        path = logs_dir / f"{self.run_id}_state.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path
