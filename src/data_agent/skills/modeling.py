"""tabular-modeling skill: train + evaluate task-conditioned models.

Award-B (bundle) mode delegates to the proven ``models.train_and_predict`` unchanged,
so model selection, baselines, and the final predictions/submission are identical to
the deterministic pipeline. Generic (single-file) mode builds a FeatureBundle from a
DataFrame and reuses the same engine. Results are repackaged into the doc-shaped
``model_results.baseline``/``.candidate``; the selected model's holdout arrays are
returned for plotting/evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..features import FeatureBundle
from ..models import ModelResult, train_and_predict
from ..task import BINARY, MULTICLASS, REGRESSION


@dataclass
class ModelingResult:
    model_results: dict
    predictions: np.ndarray
    model_result_obj: ModelResult
    holdout: dict


def train_and_evaluate_models(
    *,
    bundle: FeatureBundle | None = None,
    df: pd.DataFrame | None = None,
    target_col: str | None = None,
    task_type: str | None = None,
    feature_cols: list[str] | None = None,
    task_spec=None,
    block_column: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    families: set[str] | None = None,
    folds=None,
    scored_mask=None,
) -> ModelingResult:
    eval_truth = None
    if bundle is None:
        if df is None or target_col is None:
            raise ValueError("Provide either a FeatureBundle or (df, target_col).")
        bundle, eval_truth = _bundle_from_single_df(
            df, target_col, feature_cols, task_spec, random_state, test_size=test_size
        )

    model_result = train_and_predict(bundle, block_column, random_state, families=families,
                                     folds=folds, scored_mask=scored_mask)
    packaged = _repackage(model_result)
    return ModelingResult(
        model_results=packaged,
        predictions=model_result.predictions,
        model_result_obj=model_result,
        holdout=_holdout_dict(model_result, bundle, eval_truth, packaged),
    )


def _repackage(mr: ModelResult) -> dict:
    ok = [s for s in mr.model_scores if s.get("status") == "ok"]

    def is_baseline(name: str) -> bool:
        return name.startswith("dummy") or name.startswith("baseline")

    baselines = [s for s in ok if is_baseline(s["name"])]
    best_base = None
    if baselines:
        best_base = (max if mr.greater_is_better else min)(baselines, key=lambda s: s["score"])
    candidate = next((s for s in ok if s["name"] == mr.selected_model_name), None)

    def pack(score: dict | None) -> dict | None:
        if not score:
            return None
        return {
            "model_name": score["name"],
            "val_metrics": dict(score.get("detail", {})),
            "selection_score": score["score"],
        }

    result = {
        "baseline": pack(best_base),
        "candidate": pack(candidate),
        "selected_model_name": mr.selected_model_name,
        "metric_name": mr.metric_name,
        "greater_is_better": mr.greater_is_better,
        "all_scores": ok,
    }
    if best_base and candidate:
        result["baseline_comparison"] = {
            "metric": mr.metric_name,
            "baseline": best_base["score"],
            "candidate": candidate["score"],
            "delta": candidate["score"] - best_base["score"],
        }
    return result


def _positive_index(task, classes: list | None) -> int:
    if classes and getattr(task, "positive_label", None) in classes:
        return classes.index(task.positive_label)
    return (len(classes) - 1) if classes else 0


def _holdout_dict(mr: ModelResult, bundle: FeatureBundle, eval_truth, packaged: dict) -> dict:
    task = bundle.task
    if eval_truth is None:  # Award-B: use the captured proper holdout (shared y_true across models)
        by_model = mr.holdout_by_model or {}
        base_name = (packaged.get("baseline") or {}).get("model_name")
        base_pred = base_proba = None
        if base_name in by_model:
            captured = by_model[base_name]
            base_pred, base_proba = captured if isinstance(captured, tuple) else (captured, None)
        return {
            "task_type": mr.task_type,
            "output_kind": mr.output_kind,
            "y_true": mr.holdout_y_true,
            "y_pred": mr.holdout_y_pred,
            "y_proba": mr.holdout_y_proba,
            "baseline_name": base_name,
            "baseline_y_pred": base_pred,
            "baseline_y_proba": base_proba,
            "label_classes": mr.holdout_label_classes,
            "positive_index": _positive_index(task, mr.holdout_label_classes),
        }
    # Generic mode: predictions are on the eval split; align with eval_truth.
    if mr.task_type == REGRESSION:
        return {
            "task_type": mr.task_type, "output_kind": mr.output_kind,
            "y_true": np.asarray(eval_truth, dtype=float),
            "y_pred": np.asarray(mr.predictions, dtype=float),
            "y_proba": None, "label_classes": None, "positive_index": 0,
        }
    classes = mr.holdout_label_classes or getattr(task, "class_labels", None) or sorted(
        pd.Series(eval_truth).dropna().unique().tolist(), key=str
    )
    l2i = {label: i for i, label in enumerate(classes)}
    y_true = np.array([l2i.get(v, -1) for v in eval_truth])
    y_pred = np.array([l2i.get(v, -1) for v in mr.predictions])
    return {
        "task_type": mr.task_type, "output_kind": mr.output_kind,
        "y_true": y_true, "y_pred": y_pred, "y_proba": None,
        "label_classes": classes, "positive_index": _positive_index(task, classes),
    }


def _bundle_from_single_df(df, target_col, feature_cols, task_spec, random_state, *, test_size: float = 0.2):
    """Build a FeatureBundle from one labelled DataFrame (generic single-file mode)."""
    from sklearn.model_selection import train_test_split

    if task_spec is None:
        from .infer_task import infer_task_spec
        from .profile_data import profile_tabular_data

        profile = profile_tabular_data(df, "data.csv")
        task_spec = infer_task_spec("", df, profile, schema_target=target_col)

    feats = list(feature_cols) if feature_cols else list(task_spec.feature_candidates) or [
        c for c in df.columns if c != target_col
    ]
    feats = [c for c in feats if c in df.columns and c != target_col]

    work = df.dropna(subset=[target_col]).reset_index(drop=True)
    stratify = work[target_col] if task_spec.task_type in (BINARY, MULTICLASS) else None
    train_part, test_part = train_test_split(
        work, test_size=test_size, random_state=random_state, stratify=stratify
    )
    train_part = train_part.reset_index(drop=True)
    test_part = test_part.reset_index(drop=True)

    numeric = [
        c for c in feats
        if pd.api.types.is_numeric_dtype(train_part[c]) and not pd.api.types.is_bool_dtype(train_part[c])
    ]
    categorical = [c for c in feats if c not in numeric]

    sample_submission = pd.DataFrame({"__row_id__": np.arange(len(test_part)), target_col: 0})
    profile = {"time_column": None, "category_column": None, "block_column": None, "join_keys": []}
    bundle = FeatureBundle(
        train_df=train_part,
        predict_df=test_part,
        sample_submission=sample_submission,
        target=train_part[target_col],
        feature_columns=feats,
        numeric_columns=numeric,
        categorical_columns=categorical,
        task=task_spec,
        profile=profile,
    )
    return bundle, test_part[target_col]
