"""Per-run output directory helpers (single source of truth for run paths).

Every run's intermediate artifacts live under ``outputs/runs/<run_id>/`` with
``logs/`` and ``scratch/`` subdirectories, so concurrent or successive runs never
collide and **filenames carry no ``{run_id}_`` prefix** — the directory provides the
isolation instead. This removes the "agent invented a placeholder timestamp in the
filename" failure mode entirely: callers only need the run_id to resolve the dir.

Not covered here (intentionally flat archives / repo-root deliverables):
- ``outputs/reports/<run_id>_report.{md,pdf}`` — a browseable per-run report archive.
- repo-root ``submission.csv`` / ``report.pdf`` — the required deliverables.
"""
from __future__ import annotations

from pathlib import Path


def run_root(repo_root: str | Path, run_id: str) -> Path:
    """``outputs/runs/<run_id>`` (not created)."""
    return Path(repo_root) / "outputs" / "runs" / str(run_id)


def run_logs_dir(repo_root: str | Path, run_id: str) -> Path:
    """``outputs/runs/<run_id>/logs`` — created if absent."""
    d = run_root(repo_root, run_id) / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_scratch_dir(repo_root: str | Path, run_id: str) -> Path:
    """``outputs/runs/<run_id>/scratch`` — created if absent."""
    d = run_root(repo_root, run_id) / "scratch"
    d.mkdir(parents=True, exist_ok=True)
    return d
