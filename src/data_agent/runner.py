"""End-to-end Award B runner."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import audit_summary_text, extract_dynamic_terms, run_audit, write_audit_log
from .features import build_feature_bundle, _read_description
from .gates import write_verdict
from .leakage_guard import guard_to_verdict
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

    # ── PRE-RUN anti-hardcoding audit ────────────────────────────────────────
    # Runs immediately after schema discovery, before any feature engineering.
    dynamic_terms = extract_dynamic_terms(repo_root / "data")
    pre_audit = run_audit(repo_root, run_id, phase="pre", dynamic_terms=dynamic_terms)
    write_audit_log(pre_audit, logs_dir)
    _print_audit_summary("PRE-RUN", pre_audit)

    bundle = build_feature_bundle(schema)
    _write_json(bundle.profile, logs_dir / f"{run_id}_profile.json")
    print(f"Training rows: {bundle.profile['train_rows']}")
    print(f"Prediction rows: {bundle.profile['prediction_rows']}")
    print(f"Features: {len(bundle.feature_columns)}")

    # Code-enforced leakage floor over the static feature set (deterministic; an
    # LLM auditor verdict still overrides via {run_id}_llm_gate_leakage.json).
    write_verdict(guard_to_verdict(bundle.leakage_guard), logs_dir, run_id)
    if bundle.leakage_guard.get("status") == "fail":
        print(f"[leakage-guard] FAIL: {bundle.leakage_guard.get('summary')}")

    print(f"Task type: {bundle.task.task_type} | metric: {bundle.task.metric} | output: {bundle.task.output_kind}")

    model_result = train_and_predict(bundle, block_column=schema.block_column)
    model_result_dict = {
        "task_type": model_result.task_type,
        "output_kind": model_result.output_kind,
        "selected_model_name": model_result.selected_model_name,
        "metric_name": model_result.metric_name,
        "greater_is_better": model_result.greater_is_better,
        "selected_metrics": model_result.extra_metrics,
        "model_scores": model_result.model_scores,
        "holdout_strategy": model_result.holdout_strategy,
        "target_clip_min": model_result.target_clip_min,
        "target_clip_max": model_result.target_clip_max,
    }
    _write_json(model_result_dict, logs_dir / f"{run_id}_model_selection.json")
    print(f"Selected model: {model_result.selected_model_name} (metric {model_result.metric_name})")
    selected_score = _selected_score(model_result)
    if selected_score is not None:
        print(f"Holdout {model_result.metric_name}: {selected_score:.4f}")

    submission_path = repo_root / "submission.csv"
    _desc_text = _read_description(schema.description_path)
    model_result.predictions, _mono = apply_monotonic_constraints(
        model_result.predictions, bundle.sample_submission, schema, _desc_text
    )
    if _mono.get("applied"):
        _write_json(_mono, logs_dir / f"{run_id}_monotonic_constraints.json")
        print(f"[monotonic] {_mono['parent']} >= children; rows adjusted: {_mono['rows_adjusted']}")
    submission = _build_submission(
        bundle, schema.row_id_column, schema.target_column, model_result.predictions, model_result.output_kind
    )
    submission.to_csv(submission_path, index=False)
    submission_check = _validate_submission(
        bundle.sample_submission, submission, schema.row_id_column, schema.target_column, model_result.output_kind
    )
    _write_json(submission_check, logs_dir / f"{run_id}_submission_check.json")
    print(f"Submission written: {submission_path}")

    # ── POST-RUN anti-hardcoding audit ───────────────────────────────────────
    # Runs before final output; also scans any generated report markdown.
    report_md_candidate = repo_root / "outputs" / "reports" / f"{run_id}_report.md"
    extra_files = [report_md_candidate] if report_md_candidate.exists() else []
    post_audit = run_audit(
        repo_root, run_id, phase="post",
        dynamic_terms=dynamic_terms,
        extra_files=extra_files,
    )
    write_audit_log(post_audit, logs_dir)
    _print_audit_summary("POST-RUN", post_audit)

    pre_audit_summary = audit_summary_text(pre_audit)
    post_audit_summary = audit_summary_text(post_audit)

    md_path, pdf_path = write_report(
        repo_root=repo_root,
        run_id=run_id,
        schema=schema.to_dict(),
        profile=bundle.profile,
        model_result=model_result_dict,
        submission_check=submission_check,
        pre_audit_summary=pre_audit_summary,
        post_audit_summary=post_audit_summary,
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
        "pre_audit": {
            "verdict": pre_audit.verdict,
            "n_risky": pre_audit.n_risky,
            "n_unacceptable": pre_audit.n_unacceptable,
        },
        "post_audit": {
            "verdict": post_audit.verdict,
            "n_risky": post_audit.n_risky,
            "n_unacceptable": post_audit.n_unacceptable,
        },
    }
    _write_json(manifest, logs_dir / f"{run_id}_manifest.json")
    print("=== Analysis complete ===")
    return manifest


def _selected_score(model_result) -> float | None:
    for score in model_result.model_scores:
        if score.get("name") == model_result.selected_model_name and score.get("status") == "ok":
            return score.get("score")
    return None


def apply_monotonic_constraints(predictions, sample_submission, schema, description_text):
    """Enforce a parent>=child ordering among category values when the dataset
    description declares the categories nested / non-exclusive / totals.

    Dataset-agnostic — no category names are hardcoded. The 'parent' (superset /
    total) is inferred as the category value with the largest typical prediction;
    within each (other-key) group it is raised to at least its children. Returns
    ``(possibly-adjusted predictions, info)``. No-op unless a hierarchy cue is
    present in the description and a category column with >=2 values exists.
    """
    info: dict = {"applied": False, "reason": ""}
    preds = np.asarray(predictions, dtype=float).copy()
    cat = getattr(schema, "category_column", None)
    if not cat or cat not in sample_submission.columns:
        info["reason"] = "no category column"
        return preds, info
    cats = sample_submission[cat]
    uniq = list(pd.unique(cats.dropna()))
    if len(uniq) < 2:
        info["reason"] = "fewer than two categories"
        return preds, info
    t = (description_text or "").lower()
    cues = ["nested", "non-exclusive", "nonexclusive", "subset", "superset", "includes",
            "contained", "aggregate", "never sum", "hierarch", "total", "overall"]
    if not any(c in t for c in cues):
        info["reason"] = "no hierarchy cue in description"
        return preds, info
    other_keys = [k for k in (getattr(schema, "join_keys", []) or [])
                  if k in sample_submission.columns and k != cat]
    if not other_keys:
        info["reason"] = "no grouping keys"
        return preds, info
    cats_arr = cats.to_numpy()
    means = {c: float(np.nanmean(preds[cats_arr == c])) for c in uniq}
    parent = max(means, key=means.get)
    children = [c for c in uniq if c != parent]
    work = sample_submission[other_keys].copy()
    work["__cat"] = cats_arr
    work["__i"] = np.arange(len(work))
    n_adj = 0
    for _, sub in work.groupby(other_keys, dropna=False, sort=False):
        pm = (sub["__cat"] == parent).to_numpy()
        if not pm.any():
            continue
        child_idx = sub.loc[sub["__cat"].isin(children), "__i"].to_numpy()
        if len(child_idx) == 0:
            continue
        child_max = float(np.max(preds[child_idx]))
        p_i = int(sub.loc[pm, "__i"].iloc[0])
        if preds[p_i] < child_max:
            preds[p_i] = child_max
            n_adj += 1
    info = {"applied": True, "parent": str(parent), "children": [str(c) for c in children],
            "group_keys": other_keys, "rows_adjusted": int(n_adj),
            "parent_inferred_by": "largest_mean_prediction"}
    return preds, info


def _build_submission(
    bundle,
    row_id_column: str,
    target_column: str,
    predictions: np.ndarray,
    output_kind: str,
) -> pd.DataFrame:
    sample = bundle.sample_submission.copy()
    if row_id_column not in sample.columns:
        raise ValueError(f"Row id column '{row_id_column}' not found in sample submission.")
    if len(predictions) != len(sample):
        raise ValueError(
            f"Prediction count {len(predictions)} does not match sample submission rows {len(sample)}."
        )
    values = _format_predictions(predictions, output_kind, sample, target_column)
    output = pd.DataFrame(
        {
            row_id_column: sample[row_id_column].values,
            target_column: values,
        }
    )
    return output


def _format_predictions(
    predictions: np.ndarray, output_kind: str, sample: pd.DataFrame, target_column: str
) -> np.ndarray:
    """Coerce predictions to the format/dtype the submission requires.

    Class labels are cast to the sample submission's target dtype (e.g. integer
    0/1 for accuracy-scored tasks); values and probabilities stay float.
    """
    if output_kind == "label":
        series = pd.Series(predictions)
        if target_column in sample.columns:
            target_dtype = sample[target_column].dtype
            try:
                if pd.api.types.is_integer_dtype(target_dtype):
                    return pd.to_numeric(series, errors="coerce").round().astype("int64").to_numpy()
                return series.astype(target_dtype).to_numpy()
            except Exception:
                return series.to_numpy()
        return series.to_numpy()
    return np.asarray(predictions, dtype=float)


def _validate_submission(
    sample: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column: str,
    target_column: str,
    output_kind: str,
) -> dict[str, Any]:
    expected_columns = [row_id_column, target_column]
    target = submission[target_column]
    checks: dict[str, Any] = {
        "expected_columns": expected_columns,
        "actual_columns": list(submission.columns),
        "output_kind": output_kind,
        "n_rows_sample": int(len(sample)),
        "n_rows_submission": int(len(submission)),
        "columns_ok": list(submission.columns) == expected_columns,
        "row_count_ok": len(submission) == len(sample),
        "row_id_alignment_ok": submission[row_id_column].tolist() == sample[row_id_column].tolist(),
        "missing_predictions": int(target.isna().sum()),
    }

    if output_kind == "label":
        checks["all_finite"] = bool(target.notna().all())
        if target_column in sample.columns and pd.api.types.is_integer_dtype(sample[target_column].dtype):
            checks["dtype_ok"] = bool(pd.api.types.is_integer_dtype(target.dtype))
        else:
            checks["dtype_ok"] = True
        predicted_labels = set(target.dropna().tolist())
        checks["distinct_predicted_labels"] = sorted(_py(v) for v in predicted_labels)[:25]
        if target_column in sample.columns:
            sample_labels = set(sample[target_column].dropna().tolist())
            checks["labels_subset_of_sample"] = bool(predicted_labels.issubset(sample_labels)) if sample_labels else None
    else:
        numeric = pd.to_numeric(target, errors="coerce")
        checks["all_finite"] = bool(np.isfinite(numeric).all())
        checks["dtype_ok"] = True

    if not all(
        [
            checks["columns_ok"],
            checks["row_count_ok"],
            checks["row_id_alignment_ok"],
            checks["all_finite"],
            checks["dtype_ok"],
            checks["missing_predictions"] == 0,
        ]
    ):
        raise ValueError(f"Submission validation failed: {checks}")
    return checks


def _py(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


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


def _print_audit_summary(phase: str, result) -> None:
    verdict = result.verdict.upper()
    print(
        f"[audit:{phase}] verdict={verdict} "
        f"risky={result.n_risky} unacceptable={result.n_unacceptable}"
    )
    for f in result.findings:
        if f.classification == "unacceptable":
            print(f"  [UNACCEPTABLE] {f.file_path}:{f.line_number} term='{f.term}' — {f.reason}")

