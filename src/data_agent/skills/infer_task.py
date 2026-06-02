"""task-inference skill: produce a rich, doc-shaped TaskSpec.

The rich pydantic :class:`TaskSpec` is a *superset* of the engine dataclass in
``task.py``. Detection (task_type / metric / output_kind / class labels) is delegated
to the proven :func:`task.resolve_task_spec` so behaviour matches the deterministic
pipeline exactly; the extra doc fields (confidence, recommended_metrics, class
imbalance, feature candidates, warnings, rationale) are additive.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from ..task import BINARY, MULTICLASS, REGRESSION, resolve_task_spec

_SUPERVISED_VERBS = ["predict", "classify", "forecast", "model", "estimate", "regress on", "detect", "identify"]
_DESCRIPTIVE_WORDS = ["describe", "summarize", "summarise", "explore", "understand", "profile",
                      "distribution", "overview", "visualize", "visualise", "eda", "patterns in"]
_CAUSAL_WORDS = ["effect of", "impact of", "does ", " cause", "causal", "treatment effect", "a/b", "counterfactual", "what if we changed"]


class TaskSpec(BaseModel):
    # ── doc-contract fields (task-inference SKILL.md "Output Format") ──
    task_type: str = "unknown"
    target_variable: str | None = None
    target_type: str = "unknown"
    target_explicit: bool = False
    target_inferred: bool = False
    target_candidates: list[str] = []
    feature_candidates: list[str] = []
    excluded_from_features: list[str] = []
    exclusion_reasons: dict[str, str] = {}
    confidence: float = 0.0
    recommended_metrics: list[str] = []
    class_imbalance_detected: bool = False
    majority_class_rate: float | None = None
    unsupported_but_detected_task_types: list[str] = []
    warnings: list[dict] = []
    rationale: str = ""
    # ── engine fields consumed by models.py / runner.py (same object) ──
    metric: str = "mae"
    greater_is_better: bool = True
    output_kind: str = "value"
    class_labels: list | None = None
    positive_label: Any = None
    evidence: dict = {}


def infer_task_spec(
    user_goal: str,
    df: pd.DataFrame,
    profile: Any,
    *,
    description_text: str | None = None,
    schema_target: str | None = None,
    sample_submission: pd.DataFrame | None = None,
    has_block_col: bool = False,
) -> TaskSpec:
    profile_dict = profile.model_dump() if isinstance(profile, BaseModel) else dict(profile or {})
    columns = profile_dict.get("columns") or [str(c) for c in df.columns]
    goal = (user_goal or "").lower()
    warnings: list[dict] = []

    # ── target resolution (Checks 1–2) ──
    target_variable: str | None = None
    target_explicit = False
    target_inferred = False
    target_candidates: list[str] = []

    if schema_target and schema_target in columns:
        target_variable, target_explicit = schema_target, True
    else:
        named = _target_from_goal(goal, columns)
        if named:
            target_variable, target_explicit = named, True
        else:
            cands = _candidate_targets(df, profile_dict, columns)
            if len(cands) == 1:
                target_variable, target_inferred = cands[0], True
            elif len(cands) > 1:
                target_candidates = cands
                warnings.append(_warn("ambiguous_target", None, f"Multiple candidate targets: {cands}. Please specify."))

    descriptive = any(w in goal for w in _DESCRIPTIVE_WORDS) or any(
        p in goal for p in ["no prediction", "not predicting", "just analysis"]
    )

    # ── task_type / engine fields ──
    metric, greater_is_better, output_kind = "mae", True, "value"
    class_labels, positive_label, evidence = None, None, {}
    target_type = "unknown"

    if target_variable and target_variable in df.columns:
        sample_target = (
            sample_submission[target_variable]
            if sample_submission is not None and target_variable in getattr(sample_submission, "columns", [])
            else None
        )
        engine = resolve_task_spec(
            description_text or user_goal or "",
            df[target_variable],
            sample_target,
            has_block_col=has_block_col,
            target_column=target_variable,
        )
        task_type = engine.task_type
        metric, greater_is_better, output_kind = engine.metric, engine.greater_is_better, engine.output_kind
        class_labels, positive_label, evidence = engine.class_labels, engine.positive_label, engine.evidence
        target_type = _target_type(df[target_variable], task_type)
        if descriptive:
            warnings.append(_warn("descriptive_and_supervised", None,
                                  "Descriptive language detected but a target exists; treating as supervised."))
    elif descriptive:
        task_type = "descriptive"
    else:
        task_type = "unknown"
        warnings.append(_warn("no_target_found", None, "No target could be resolved and the goal is not descriptive.",
                              severity="FAIL" if any(v in goal for v in _SUPERVISED_VERBS) else "WARN"))

    # ── class imbalance (classification only) ──
    class_imbalance_detected = False
    majority_class_rate: float | None = None
    if task_type in (BINARY, MULTICLASS) and target_variable in df.columns:
        counts = df[target_variable].dropna().value_counts(normalize=True)
        if len(counts):
            majority_class_rate = float(counts.iloc[0])
            class_imbalance_detected = majority_class_rate > 0.80
            if class_imbalance_detected:
                warnings.append(_warn("class_imbalance", target_variable,
                                      f"Majority class is {majority_class_rate:.1%}; prefer ROC-AUC over accuracy."))

    # ── feature candidates / exclusions ──
    feature_candidates, excluded, exclusion_reasons = _features_and_exclusions(columns, target_variable, profile_dict)

    # ── recommended metrics ──
    recommended_metrics = _recommended_metrics(task_type, class_imbalance_detected)

    # ── target_inferred warning ──
    if target_inferred:
        warnings.append(_warn("target_inferred", target_variable,
                              "Target not explicitly named; inferred from data profile. Please confirm."))

    # ── unsupported task types (detect, do not execute) ──
    unsupported = _detect_unsupported(goal, profile_dict)
    if unsupported:
        warnings.append(_warn("unsupported_task_type", None,
                              f"Detected unsupported task(s) {unsupported}; V1 handles regression/classification/descriptive only."))
    if any(c in goal for c in _CAUSAL_WORDS) and task_type in (BINARY, MULTICLASS, REGRESSION):
        warnings.append(_warn("causal_language_in_non_causal_task", None,
                              "Causal phrasing detected; treated as predictive, not causal. Note in limitations."))

    confidence = _confidence(task_type, target_explicit, target_inferred, target_candidates)

    return TaskSpec(
        task_type=task_type,
        target_variable=target_variable,
        target_type=target_type,
        target_explicit=target_explicit,
        target_inferred=target_inferred,
        target_candidates=target_candidates,
        feature_candidates=feature_candidates,
        excluded_from_features=excluded,
        exclusion_reasons=exclusion_reasons,
        confidence=confidence,
        recommended_metrics=recommended_metrics,
        class_imbalance_detected=class_imbalance_detected,
        majority_class_rate=majority_class_rate,
        unsupported_but_detected_task_types=unsupported,
        warnings=warnings,
        rationale=_rationale(task_type, target_variable, target_explicit, target_inferred),
        metric=metric,
        greater_is_better=greater_is_better,
        output_kind=output_kind,
        class_labels=class_labels,
        positive_label=positive_label,
        evidence=evidence,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _warn(wtype: str, column: str | None, message: str, severity: str = "WARN") -> dict:
    return {"type": wtype, "column": column, "message": message, "severity": severity}


def _target_from_goal(goal: str, columns: list[str]) -> str | None:
    has_verb = any(v in goal for v in _SUPERVISED_VERBS) or "target is" in goal or "whether" in goal
    if not has_verb:
        return None
    matches = [c for c in columns if re.search(rf"\b{re.escape(c.lower())}\b", goal)]
    return max(matches, key=len) if matches else None


def _candidate_targets(df: pd.DataFrame, profile: dict, columns: list[str]) -> list[str]:
    cands: list[str] = list(profile.get("target_like_columns") or [])
    for col in columns:
        if col in cands:
            continue
        series = df[col].dropna()
        if series.nunique() == 2:  # binary numeric/boolean
            num = pd.to_numeric(series, errors="coerce")
            if num.notna().all() or pd.api.types.is_bool_dtype(df[col]):
                cands.append(col)
    return cands


def _target_type(series: pd.Series, task_type: str) -> str:
    if task_type == REGRESSION:
        return "continuous"
    if task_type == BINARY:
        return "binary"
    if task_type == MULTICLASS:
        return "multiclass"
    nun = int(series.dropna().nunique())
    if nun == 2:
        return "binary"
    if 3 <= nun <= 20:
        return "multiclass"
    return "unknown"


def _features_and_exclusions(columns: list[str], target: str | None, profile: dict):
    features: list[str] = []
    excluded: list[str] = []
    reasons: dict[str, str] = {}
    for col in columns:
        if col == target:
            continue
        reason = _exclusion_reason(col, profile)
        if reason:
            excluded.append(col)
            reasons[col] = reason
        else:
            features.append(col)
    return features, excluded, reasons


def _exclusion_reason(col: str, profile: dict) -> str | None:
    if col in (profile.get("potential_id_columns") or []):
        return "potential_id_column"
    if col in (profile.get("leakage_like_columns") or []):
        return "leakage_like_column"
    if (profile.get("missing_summary") or {}).get(col, {}).get("severity") == "critical_missing":
        return "critical_missing"
    if col in (profile.get("constant_columns") or []):
        return "constant_column"
    if col in (profile.get("high_cardinality_columns") or []):
        return "high_cardinality"
    return None


def _recommended_metrics(task_type: str, imbalance: bool) -> list[str]:
    if task_type == BINARY:
        return ["roc_auc", "f1_weighted", "accuracy", "confusion_matrix"]
    if task_type == MULTICLASS:
        return ["f1_weighted", "f1_macro", "accuracy", "confusion_matrix"]
    if task_type == REGRESSION:
        return ["rmse", "mae", "r2"]
    return []


def _detect_unsupported(goal: str, profile: dict) -> list[str]:
    out: list[str] = []
    if any(w in goal for w in ["forecast", "next period", "time series", "seasonal"]) or profile.get("datetime_columns"):
        if any(w in goal for w in ["forecast", "time series", "seasonal", "trend"]):
            out.append("forecasting")
    if any(w in goal for w in ["survival", "time to event", "hazard", "kaplan"]):
        out.append("survival_analysis")
    if any(c in goal for c in _CAUSAL_WORDS):
        out.append("causal_inference")
    return out


def _confidence(task_type: str, explicit: bool, inferred: bool, candidates: list[str]) -> float:
    if task_type == "unknown":
        return 0.5 if candidates else 0.3
    if explicit:
        return 0.95
    if inferred:
        return 0.9
    if task_type == "descriptive":
        return 0.85
    return 0.8


def _rationale(task_type: str, target: str | None, explicit: bool, inferred: bool) -> str:
    how = "explicitly named" if explicit else ("inferred from the data profile" if inferred else "not resolved")
    if task_type == "descriptive":
        return "No target was resolved and the goal is descriptive; performing exploratory analysis."
    if target:
        return f"Resolved a supervised {task_type} task on target '{target}' ({how}); non-informative columns excluded via profile checks."
    return f"Task type '{task_type}'; target {how}."
