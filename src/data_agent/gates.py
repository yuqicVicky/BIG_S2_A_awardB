"""Closed-loop supervision: the verdict protocol, the stage-gate runner, and the
deterministic critics that give the adversarial supervisor real teeth.

Every gate emits one JSON *verdict* (see :class:`Verdict`) to ``outputs/logs``.
:func:`run_stage_with_gate` runs a stage, critiques its output, and — on a
``fail`` verdict — re-runs the stage once with a correction hint, bounded by a
hard retry cap and a no-progress guard so the loop can never spin.

Two layers share this one schema:

* **Deterministic critics** (this module) always run, even headless — the
  Award-B evaluator executes ``python main.py`` with no Claude session and no
  API, so these must be able to close the loop entirely on their own. They never
  raise; an uncorrectable problem degrades to a logged ``fail`` verdict and the
  deliverable is still produced.
* **LLM critic subagents** (the ``.claude/agents`` specs) write verdicts in this
  same schema to a sibling file. When such a file is present it *takes
  precedence* over the deterministic verdict for that stage — see
  :func:`load_llm_verdict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import task as T

# ── verdict statuses ──────────────────────────────────────────────────────────
PASS = "pass"
WARN = "warn"
FAIL = "fail"

_RANK = {PASS: 0, WARN: 1, FAIL: 2}

# Stages that carry a stable, fixed-name alias log (the CLAUDE.md inspection
# checklist promises these exact filenames regardless of run_id). The report
# review writes its own report_review.json (with an ``approved`` field), so it
# is intentionally not aliased here.
STAGE_ALIASES = {
    "prediction_sanity": "prediction_sanity.json",
    "supervisor": "supervisor_gatekeeper.json",
}


def _worst(a: str, b: str) -> str:
    """Return the more severe of two statuses (pass < warn < fail)."""
    return a if _RANK[a] >= _RANK[b] else b


@dataclass
class Verdict:
    """One critic's judgement of one stage, in the shared cross-layer schema."""

    stage: str
    status: str = PASS
    reasons: list[str] = field(default_factory=list)
    suggested_corrections: dict = field(default_factory=dict)
    checked: dict = field(default_factory=dict)
    critic: str = "deterministic"
    run_id: str | None = None

    @property
    def failed(self) -> bool:
        return self.status == FAIL

    @property
    def ok(self) -> bool:
        return self.status != FAIL

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "run_id": self.run_id,
            "status": self.status,
            "reasons": list(self.reasons),
            "suggested_corrections": dict(self.suggested_corrections),
            "checked": _jsonable(self.checked),
            "critic": self.critic,
        }


# ── serialization ──────────────────────────────────────────────────────────────

def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of nested numpy/pandas scalars for JSON logging."""
    try:
        return json.loads(json.dumps(obj, default=_json_default))
    except Exception:
        return {"_unserializable": str(obj)}


def write_verdict(verdict: Verdict, logs_dir: str | Path, run_id: str) -> dict:
    """Persist a verdict to ``gate_{stage}.json`` (+ a fixed-name alias for the
    stages the inspection checklist names explicitly). ``logs_dir`` is already the
    per-run dir (``outputs/runs/<run_id>/logs``), so no run_id prefix is needed."""
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    verdict.run_id = verdict.run_id or run_id
    payload = verdict.to_dict()
    _dump(payload, logs_dir / f"gate_{verdict.stage}.json")
    alias = STAGE_ALIASES.get(verdict.stage)
    if alias:
        _dump(payload, logs_dir / alias)
    return payload


def _dump(obj: Any, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def load_llm_verdict(logs_dir: str | Path, run_id: str, stage: str) -> Verdict | None:
    """Load an LLM critic's verdict for ``stage`` if one was written.

    The Claude-driven path lets a critic subagent drop a verdict file named
    ``llm_gate_{stage}.json`` in the run dir (same schema). When present it
    overrides the deterministic verdict. Returns ``None`` when absent or
    unreadable, so the headless deterministic path is unaffected.
    """
    path = Path(logs_dir) / f"llm_gate_{stage}.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    status = str(data.get("status", PASS)).lower()
    if status not in _RANK:
        status = PASS
    return Verdict(
        stage=str(data.get("stage", stage)),
        status=status,
        reasons=list(data.get("reasons", []) or []),
        suggested_corrections=dict(data.get("suggested_corrections", {}) or {}),
        checked=dict(data.get("checked", {}) or {}),
        critic=str(data.get("critic", f"llm:{stage}")),
        run_id=data.get("run_id", run_id),
    )


# ── the stage-gate runner ───────────────────────────────────────────────────────

def run_stage_with_gate(
    name: str,
    produce_fn: Callable[[], Any],
    critic_fn: Callable[[Any], Verdict],
    apply_correction_fn: Callable[[dict], Any] | None = None,
    *,
    max_retries: int = 1,
    logs_dir: str | Path | None = None,
    run_id: str | None = None,
    llm_verdict_loader: Callable[[str], Verdict | None] | None = None,
) -> tuple[Any, Verdict]:
    """Run a stage, critique it, and re-run once with a correction on ``fail``.

    Parameters
    ----------
    produce_fn:
        Runs the stage and returns its output (any type).
    critic_fn:
        ``critic_fn(output) -> Verdict``. The deterministic critic.
    apply_correction_fn:
        ``apply_correction_fn(corrections) -> bool|None``. Mutates the inputs
        that ``produce_fn`` closes over so the next ``produce_fn()`` reflects the
        correction. Returning ``False`` signals "cannot apply" and stops retries.
    max_retries:
        Hard cap on corrective reruns (loop guard).
    llm_verdict_loader:
        Optional ``loader(stage) -> Verdict|None``. When it returns a verdict, it
        takes precedence over the deterministic one (the LLM critic layer).

    Returns ``(output, verdict)``; never raises for ordinary gate logic, so the
    deliverable is never lost. ``verdict`` reflects the final state after any
    corrective reruns.
    """
    output = produce_fn()
    verdict = _resolve(name, output, critic_fn, llm_verdict_loader)

    attempts = 0
    seen: set[tuple] = set()
    while verdict.failed and attempts < max_retries:
        if apply_correction_fn is None or not verdict.suggested_corrections:
            break
        reason_key = tuple(sorted(verdict.reasons))
        if reason_key in seen:  # no-progress guard
            verdict.reasons.append("no-progress: identical failure recurred after correction; stopping retries")
            break
        seen.add(reason_key)
        try:
            applied = apply_correction_fn(dict(verdict.suggested_corrections))
        except Exception as exc:  # corrections must never crash the pipeline
            verdict.reasons.append(f"correction raised {type(exc).__name__}: {exc}; stopping retries")
            break
        if applied is False:
            verdict.reasons.append("correction could not be applied; stopping retries")
            break
        attempts += 1
        output = produce_fn()
        verdict = _resolve(name, output, critic_fn, llm_verdict_loader)

    verdict.checked = {**verdict.checked, "corrective_reruns": attempts}
    if logs_dir is not None and run_id is not None:
        write_verdict(verdict, logs_dir, run_id)
    return output, verdict


def _resolve(
    name: str,
    output: Any,
    critic_fn: Callable[[Any], Verdict],
    llm_verdict_loader: Callable[[str], Verdict | None] | None,
) -> Verdict:
    if llm_verdict_loader is not None:
        try:
            llm = llm_verdict_loader(name)
        except Exception:
            llm = None
        if llm is not None:
            return llm
    return critic_fn(output)


# ── deterministic critics ────────────────────────────────────────────────────────

def _sample_value_format(series: pd.Series | None) -> str:
    """Classify a sample-submission target column's value format.

    Returns one of ``prob01`` (numeric, all in [0,1] and not all-integer),
    ``int``, ``float``, ``string``, or ``unknown``.
    """
    if series is None:
        return "unknown"
    st = pd.Series(series).dropna()
    if not len(st):
        return "unknown"
    num = pd.to_numeric(st, errors="coerce")
    if num.notna().all():
        vals = num.to_numpy(dtype=float)
        all_int = bool(np.all(np.mod(vals, 1) == 0))
        if all_int:
            return "int"
        if float(np.nanmin(vals)) >= 0.0 and float(np.nanmax(vals)) <= 1.0:
            return "prob01"
        return "float"
    return "string"


def check_task_consistency(
    *,
    task_type: str,
    metric: str | None,
    output_kind: str | None,
    target_series: pd.Series,
    sample_target_series: pd.Series | None = None,
    description_text: str | None = None,
    run_id: str | None = None,
) -> Verdict:
    """Cross-check the resolved task against the data, the description metric, and
    the sample-submission value format — and *surface* contradictions loudly.

    This is the gate that would have caught the original House-Prices bug: a
    regression metric coexisting with a fabricated multiclass ``task_type``. On a
    hard contradiction it returns ``fail`` with a ``force_task_type`` correction
    so the loop can re-resolve the task. Entirely dataset-agnostic.
    """
    data = T._infer_from_data(target_series, sample_target_series)
    desc_metric = T._parse_description_metric(description_text or "")
    reasons: list[str] = []
    corrections: dict = {}
    status = PASS

    is_class = task_type in (T.BINARY, T.MULTICLASS)
    is_reg = task_type == T.REGRESSION

    def forced_class() -> str:
        return T.BINARY if data["nunique"] == 2 else T.MULTICLASS

    # (A) resolved metric vs task family — a direct internal contradiction.
    if metric in T._CLASS_METRICS and is_reg:
        reasons.append(f"resolved metric '{metric}' is a classification metric but task_type is regression")
        corrections["force_task_type"] = forced_class()
        status = FAIL
    elif metric in T._REG_METRICS and is_class:
        reasons.append(f"resolved metric '{metric}' is a regression metric but task_type is '{task_type}'")
        corrections["force_task_type"] = T.REGRESSION
        status = FAIL

    # (B) description metric vs task family — independent of the resolved metric;
    #     catches a task_type that drifted away from what the brief says it scores.
    if desc_metric in T._REG_METRICS and is_class:
        reasons.append(f"description metric '{desc_metric}' implies regression but task_type is '{task_type}'")
        corrections["force_task_type"] = T.REGRESSION
        status = FAIL
    elif desc_metric in T._CLASS_METRICS and is_reg:
        reasons.append(f"description metric '{desc_metric}' implies classification but task_type is regression")
        corrections.setdefault("force_task_type", forced_class())
        status = FAIL

    # (C) task vs data shape.
    if is_class and data["task"] == T.REGRESSION:
        reasons.append(
            "task_type is classification but the target is continuous / high-cardinality numeric "
            f"({data['nunique']} distinct values); data implies regression"
        )
        corrections["force_task_type"] = T.REGRESSION
        status = FAIL
    elif is_reg and data["nunique"] == 2:
        reasons.append("task_type is regression but the target has only 2 distinct values; data may imply binary classification")
        corrections.setdefault("force_task_type", T.BINARY)
        status = _worst(status, WARN)

    # (D) output_kind vs sample-submission value format (advisory).
    if sample_target_series is not None and output_kind:
        fmt = _sample_value_format(sample_target_series)
        if output_kind == "probability" and fmt not in ("prob01", "float"):
            reasons.append(f"output_kind is 'probability' but sample-submission values look like '{fmt}'")
            status = _worst(status, WARN)
        elif output_kind == "value" and is_class:
            reasons.append("output_kind 'value' on a classification task is inconsistent")
            status = _worst(status, WARN)

    checked = {
        "task_type": task_type,
        "resolved_metric": metric,
        "output_kind": output_kind,
        "description_metric": desc_metric,
        "data_task": data["task"],
        "n_unique_target": data["nunique"],
        "target_is_numeric": data["is_numeric"],
        "sample_is_float": data["sample_is_float"],
        "sample_is_int": data["sample_is_int"],
    }
    return Verdict(
        stage="task_inference",
        status=status,
        reasons=reasons,
        suggested_corrections=corrections,
        checked=checked,
        run_id=run_id,
    )


def check_schema(
    *,
    sample_submission: pd.DataFrame | None,
    row_id_column: str | None,
    target_column: str | None,
    train_columns: list[str] | None = None,
    run_id: str | None = None,
) -> Verdict:
    """Validate the discovered schema: a usable sample submission, and row-id /
    target columns that actually exist where they must."""
    reasons: list[str] = []
    status = PASS
    cols = list(sample_submission.columns) if sample_submission is not None else []

    if sample_submission is None or len(sample_submission) == 0:
        reasons.append("no usable sample submission was resolved from the dataset")
        status = FAIL
    if not row_id_column:
        reasons.append("no row-id column was resolved")
        status = FAIL
    elif cols and row_id_column not in cols:
        reasons.append(f"row-id column '{row_id_column}' is not present in the sample submission")
        status = _worst(status, WARN)
    if target_column and cols and target_column not in cols:
        reasons.append(f"target column '{target_column}' is not present in the sample submission")
        status = _worst(status, WARN)
    if row_id_column and train_columns is not None and row_id_column not in train_columns:
        reasons.append(f"row-id column '{row_id_column}' is not present in the training columns")
        status = _worst(status, WARN)

    checked = {
        "sample_submission_columns": cols,
        "row_id_column": row_id_column,
        "target_column": target_column,
        "sample_rows": int(len(sample_submission)) if sample_submission is not None else 0,
    }
    return Verdict(stage="schema", status=status, reasons=reasons, checked=checked, run_id=run_id)


def check_prediction_sanity(
    *,
    predictions: Any,
    train_target: Any,
    task_type: str,
    holdout_y_true: Any = None,
    holdout_y_pred: Any = None,
    metric_name: str | None = None,
    baseline_score: float | None = None,
    candidate_score: float | None = None,
    greater_is_better: bool = False,
    clip_fraction: float | None = None,
    run_id: str | None = None,
) -> Verdict:
    """Sanity-check final predictions for the failure modes that slip past a
    green model-selection score: degenerate (near-constant) output, heavy
    clipping, a large distribution shift vs the training target, suspiciously
    perfect holdout fit (leakage/overfit), and a model that fails to beat its
    own baseline. Dataset-agnostic; thresholds are relative, not absolute."""
    reasons: list[str] = []
    corrections: dict = {}
    status = PASS
    preds = np.asarray(predictions)
    n = int(len(preds))
    is_reg = task_type == T.REGRESSION

    if is_reg:
        fp = pd.to_numeric(pd.Series(preds), errors="coerce").to_numpy(dtype=float)
        if n and not np.all(np.isfinite(fp)):
            reasons.append("final predictions contain non-finite values")
            corrections["sanitize_predictions"] = True
            status = FAIL
        tt = pd.to_numeric(pd.Series(train_target), errors="coerce").dropna().to_numpy(dtype=float)
        finite_preds = fp[np.isfinite(fp)] if n else fp
        if len(tt) > 1 and len(finite_preds) > 1:
            pstd = float(np.std(finite_preds))
            tstd = float(np.std(tt)) or 1.0
            if pstd / (abs(tstd) + 1e-9) < 0.01:
                reasons.append(f"predictions are near-constant (pred std {pstd:.4g} vs target std {tstd:.4g})")
                corrections["reexamine_model_pool"] = True
                status = FAIL
            pmed = float(np.median(finite_preds))
            tmed = float(np.median(tt))
            if abs(tmed) > 1e-9:
                ratio = pmed / tmed
                if ratio > 3.0 or ratio < 1 / 3:
                    reasons.append(f"prediction median {pmed:.4g} is far from train-target median {tmed:.4g} (ratio {ratio:.2f})")
                    status = _worst(status, WARN)
    else:
        if n and len({str(v) for v in preds.tolist()}) == 1:
            reasons.append("every prediction is the same class label")
            status = _worst(status, WARN)

    if clip_fraction is not None and clip_fraction > 0.10:
        reasons.append(f"{clip_fraction:.0%} of predictions were clipped to the allowed range")
        status = _worst(status, FAIL if clip_fraction > 0.30 else WARN)
        corrections.setdefault("widen_or_transform", True)

    if is_reg and holdout_y_true is not None and holdout_y_pred is not None:
        yt = np.asarray(holdout_y_true, dtype=float)
        yp = np.asarray(holdout_y_pred, dtype=float)
        if len(yt) and len(yt) == len(yp) and np.allclose(yt, yp, rtol=1e-3, atol=1e-6):
            reasons.append("holdout predictions almost exactly equal the holdout targets (possible leakage / overfit)")
            corrections.setdefault("prefer_regularized", True)
            status = _worst(status, WARN)

    if baseline_score is not None and candidate_score is not None:
        better = candidate_score > baseline_score if greater_is_better else candidate_score < baseline_score
        if not better:
            reasons.append(f"selected model ({candidate_score:.4g}) does not beat the baseline ({baseline_score:.4g}) on {metric_name}")
            status = _worst(status, WARN)

    checked = {
        "n_predictions": n,
        "task_type": task_type,
        "metric": metric_name,
        "clip_fraction": clip_fraction,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
    }
    return Verdict(
        stage="prediction_sanity",
        status=status,
        reasons=reasons,
        suggested_corrections=corrections,
        checked=checked,
        run_id=run_id,
    )
