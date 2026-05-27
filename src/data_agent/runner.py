"""End-to-end Award B runner."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import build_feature_bundle
from .models import train_and_predict
from .reporting import write_report
from .schema import discover_schema, write_schema_json


def run_analysis(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    run_id = _run_id(repo_root)
    outputs_dir = repo_root / "outputs"
    artifacts_dir = outputs_dir / "artifacts"
    logs_dir = outputs_dir / "logs"
    reports_dir = outputs_dir / "reports"
    for directory in [artifacts_dir, logs_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    _remove_stale_outputs(repo_root)

    print(f"=== Award B automated analysis: {run_id} ===")
    schema = discover_schema(repo_root / "data")
    write_schema_json(schema, logs_dir / f"{run_id}_schema.json")
    print(f"Target column: {schema.target_column}")
    print(f"Join keys: {schema.join_keys}")

    bundle = build_feature_bundle(schema)
    _write_json(bundle.profile, logs_dir / f"{run_id}_profile.json")
    print(f"Training rows: {bundle.profile['train_rows']}")
    print(f"Prediction rows: {bundle.profile['prediction_rows']}")
    print(f"Features: {len(bundle.feature_columns)}")

    model_result = train_and_predict(bundle, block_column=schema.block_column)
    model_result_dict = {
        "selected_model_name": model_result.selected_model_name,
        "model_scores": model_result.model_scores,
        "holdout_strategy": model_result.holdout_strategy,
        "metric_name": model_result.metric_name,
        "target_clip_min": model_result.target_clip_min,
        "target_clip_max": model_result.target_clip_max,
    }
    _write_json(model_result_dict, logs_dir / f"{run_id}_model_selection.json")
    print(f"Selected model: {model_result.selected_model_name}")

    submission_path = repo_root / "submission.csv"
    submission = _build_submission(bundle, schema.row_id_column, schema.target_column, model_result.predictions)
    submission.to_csv(submission_path, index=False)
    submission_check = _validate_submission(bundle.sample_submission, submission, schema.row_id_column, schema.target_column)
    _write_json(submission_check, logs_dir / f"{run_id}_submission_check.json")
    print(f"Submission written: {submission_path}")

    md_path, pdf_path = write_report(
        repo_root=repo_root,
        run_id=run_id,
        schema=schema.to_dict(),
        profile=bundle.profile,
        model_result=model_result_dict,
        submission_check=submission_check,
    )
    print(f"Report markdown: {md_path}")
    print(f"Report PDF: {pdf_path}")

    manifest = {
        "run_id": run_id,
        "submission_path": str(submission_path),
        "report_pdf_path": str(pdf_path),
        "report_md_path": str(md_path),
        "schema": schema.to_dict(),
        "model": model_result_dict,
        "submission_check": submission_check,
    }
    _write_json(manifest, logs_dir / f"{run_id}_manifest.json")
    print("=== Analysis complete ===")
    return manifest


def _build_submission(
    bundle,
    row_id_column: str,
    target_column: str,
    predictions: np.ndarray,
) -> pd.DataFrame:
    sample = bundle.sample_submission.copy()
    if row_id_column not in sample.columns:
        raise ValueError(f"Row id column '{row_id_column}' not found in sample submission.")
    if len(predictions) != len(sample):
        raise ValueError(
            f"Prediction count {len(predictions)} does not match sample submission rows {len(sample)}."
        )
    output = pd.DataFrame(
        {
            row_id_column: sample[row_id_column].values,
            target_column: predictions,
        }
    )
    return output


def _validate_submission(
    sample: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column: str,
    target_column: str,
) -> dict[str, Any]:
    expected_columns = [row_id_column, target_column]
    checks = {
        "expected_columns": expected_columns,
        "actual_columns": list(submission.columns),
        "n_rows_sample": int(len(sample)),
        "n_rows_submission": int(len(submission)),
        "columns_ok": list(submission.columns) == expected_columns,
        "row_count_ok": len(submission) == len(sample),
        "row_id_alignment_ok": submission[row_id_column].tolist() == sample[row_id_column].tolist(),
        "all_finite": bool(np.isfinite(pd.to_numeric(submission[target_column], errors="coerce")).all()),
        "missing_predictions": int(pd.to_numeric(submission[target_column], errors="coerce").isna().sum()),
    }
    if not all(
        [
            checks["columns_ok"],
            checks["row_count_ok"],
            checks["row_id_alignment_ok"],
            checks["all_finite"],
            checks["missing_predictions"] == 0,
        ]
    ):
        raise ValueError(f"Submission validation failed: {checks}")
    return checks


def _remove_stale_outputs(repo_root: Path) -> None:
    for name in ["submission.csv", "report.pdf"]:
        path = repo_root / name
        if path.exists():
            path.unlink()


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _run_id(repo_root: Path) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha256(str(repo_root).encode("utf-8")).hexdigest()[:8]
    return f"{stamp}_{digest}"

