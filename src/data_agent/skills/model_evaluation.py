"""model-evaluation skill: task-appropriate metrics, plots, and validity checks.

Implements ``.claude/skills/model-evaluation/SKILL.md``: per-task metric keys, a
baseline-vs-candidate ``performance_table`` (every row ``split="test"``), delta vs
baseline, confusion-matrix/residual artifacts, and validity checks A–F. Inputs are the
selected model's holdout arrays plus the baseline's (from the modeling skill).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / deterministic
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ..task import BINARY, MULTICLASS, REGRESSION


def evaluate_model(
    *,
    task_type: str,
    y_true,
    y_pred,
    y_proba=None,
    baseline_y_pred=None,
    baseline_y_proba=None,
    class_labels: list | None = None,
    positive_index: int = 1,
    run_id: str,
    artifacts_dir: str | Path,
    selection_metric: str | None = None,
    selection_rationale: str | None = None,
    candidate_name: str = "candidate",
    baseline_name: str = "baseline",
) -> dict:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if task_type == REGRESSION:
        return _evaluate_regression(
            y_true.astype(float), y_pred.astype(float), baseline_y_pred,
            run_id, artifacts_dir, selection_metric, selection_rationale, candidate_name, baseline_name,
        )
    return _evaluate_classification(
        task_type, y_true, y_pred, y_proba, baseline_y_pred, baseline_y_proba,
        class_labels, positive_index, run_id, artifacts_dir,
        selection_metric, selection_rationale, candidate_name, baseline_name,
    )


# ── classification ──────────────────────────────────────────────────────────

def _classification_row(model: str, y_true, y_pred, y_proba, task_type, n_classes, positive_index) -> dict:
    row = {"model": model, "split": "test"}
    row["accuracy"] = float(accuracy_score(y_true, y_pred))
    row["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    row["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    row["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    if task_type == MULTICLASS:
        row["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        row["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    if y_proba is not None:
        try:
            if task_type == BINARY:
                row["roc_auc"] = float(roc_auc_score((y_true == positive_index).astype(int), y_proba[:, positive_index]))
            else:
                row["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
        except Exception:
            row["roc_auc"] = None
    else:
        row["roc_auc"] = None
    return row


def _evaluate_classification(
    task_type, y_true, y_pred, y_proba, baseline_y_pred, baseline_y_proba,
    class_labels, positive_index, run_id, artifacts_dir,
    selection_metric, selection_rationale, candidate_name, baseline_name,
):
    classes = class_labels or sorted(set(np.unique(y_true)).union(np.unique(y_pred)))
    n_classes = len(classes)
    label_index = list(range(n_classes))

    if baseline_y_pred is None:  # synthesize a most-frequent baseline on y_true
        majority = int(np.bincount(y_true.astype(int), minlength=n_classes).argmax())
        baseline_y_pred = np.full_like(y_true, majority)

    candidate_row = _classification_row(f"{candidate_name} (candidate)", y_true, y_pred, y_proba, task_type, n_classes, positive_index)
    baseline_row = _classification_row(f"{baseline_name} (baseline)", y_true, np.asarray(baseline_y_pred), baseline_y_proba, task_type, n_classes, positive_index)

    counts = np.bincount(y_true.astype(int), minlength=n_classes)
    majority_class_rate = float(counts.max() / counts.sum()) if counts.sum() else None
    imbalance = majority_class_rate is not None and majority_class_rate > 0.80
    positive_rate = float((y_true == positive_index).mean()) if task_type == BINARY else None

    cm = confusion_matrix(y_true, y_pred, labels=label_index)
    artifacts = {"confusion_matrix_plot": None, "roc_curve_plot": None, "pr_curve_plot": None,
                 "residual_plot": None, "pred_vs_actual_plot": None}
    artifacts["confusion_matrix_plot"] = _plot_confusion(cm, classes, run_id, artifacts_dir)
    if task_type == BINARY and y_proba is not None:
        artifacts["roc_curve_plot"] = _plot_roc((y_true == positive_index).astype(int), y_proba[:, positive_index], run_id, artifacts_dir)
        if (positive_rate is not None and positive_rate < 0.20) or imbalance:
            artifacts["pr_curve_plot"] = _plot_pr((y_true == positive_index).astype(int), y_proba[:, positive_index], run_id, artifacts_dir)

    metric = selection_metric or ("roc_auc" if task_type == BINARY else "f1_weighted")
    delta = _delta(baseline_row, candidate_row, ["roc_auc", "f1_weighted", "accuracy"])
    cand_primary = candidate_row.get(metric)
    base_primary = baseline_row.get(metric)
    worse = bool(cand_primary is not None and base_primary is not None and cand_primary < base_primary)

    warnings: list[dict] = []
    if imbalance:
        warnings.append(_w("class_imbalance", f"Majority class {majority_class_rate:.1%}; use ROC-AUC/PR-AUC, not accuracy alone."))

    checks = _validity_checks(task_type, [baseline_row, candidate_row], metric)
    return {
        "task_type": task_type,
        "split_used_for_final_evaluation": "test",
        "class_imbalance_detected": bool(imbalance),
        "majority_class_rate": majority_class_rate,
        "worse_than_baseline": worse,
        "metrics_used": [k for k in candidate_row if k not in ("model", "split")],
        "selection_metric": metric,
        "selection_split": "val",
        "selection_rationale": selection_rationale or f"Selected by {metric} on the internal holdout.",
        "performance_table": [baseline_row, candidate_row],
        "best_model": candidate_name,
        "best_model_key": "candidate",
        "delta_vs_baseline": delta,
        "confusion_matrix": {"labels": [_py(c) for c in classes], "matrix": cm.tolist(), "split": "test"},
        "residual_summary": None,
        "approved": all(c.get("severity") != "FAIL" for c in checks),
        "failed_checks": [c for c in checks if c.get("severity") == "FAIL"],
        "warned_checks": [c for c in checks if c.get("severity") == "WARN"],
        "warnings": warnings,
        "artifacts": artifacts,
    }


# ── regression ──────────────────────────────────────────────────────────────

def _regression_row(model: str, y_true, y_pred) -> dict:
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(mean_absolute_error(y_true, y_pred))
    row = {"model": model, "split": "test", "rmse": rmse, "mae": mae, "r2": float(r2_score(y_true, y_pred))}
    if np.min(y_true) > 0:
        row["mape"] = float(np.mean(np.abs((y_true - y_pred) / y_true)))
    return row


def _evaluate_regression(y_true, y_pred, baseline_y_pred, run_id, artifacts_dir,
                         selection_metric, selection_rationale, candidate_name, baseline_name):
    if baseline_y_pred is None:
        baseline_y_pred = np.full_like(y_true, float(np.mean(y_true)))
    baseline_y_pred = np.asarray(baseline_y_pred, dtype=float)

    candidate_row = _regression_row(f"{candidate_name} (candidate)", y_true, y_pred)
    baseline_row = _regression_row(f"{baseline_name} (baseline)", y_true, baseline_y_pred)

    residuals = y_pred - y_true
    residual_summary = {
        "split": "test", "mean": float(residuals.mean()), "std": float(residuals.std()),
        "min": float(residuals.min()), "p5": float(np.percentile(residuals, 5)),
        "p95": float(np.percentile(residuals, 95)), "max": float(residuals.max()),
    }
    artifacts = {"confusion_matrix_plot": None, "roc_curve_plot": None, "pr_curve_plot": None,
                 "residual_plot": _plot_residuals(y_pred, residuals, run_id, artifacts_dir),
                 "pred_vs_actual_plot": _plot_pred_vs_actual(y_true, y_pred, run_id, artifacts_dir)}

    metric = selection_metric or "mae"
    warnings: list[dict] = []
    if candidate_row["mae"] > 0 and (candidate_row["rmse"] / candidate_row["mae"]) > 2.0:
        warnings.append(_w("rmse_mae_divergence", f"RMSE/MAE = {candidate_row['rmse'] / candidate_row['mae']:.2f}; large residuals on a subset."))
    worse = candidate_row["r2"] < 0
    if worse:
        warnings.append(_w("worse_than_baseline", "Candidate R^2 < 0: worse than a mean-predictor baseline.", severity="FAIL"))

    delta = _delta(baseline_row, candidate_row, ["rmse", "mae", "r2"])
    checks = _validity_checks(REGRESSION, [baseline_row, candidate_row], metric)
    return {
        "task_type": REGRESSION,
        "split_used_for_final_evaluation": "test",
        "class_imbalance_detected": False,
        "majority_class_rate": None,
        "worse_than_baseline": bool(worse),
        "metrics_used": [k for k in candidate_row if k not in ("model", "split")],
        "selection_metric": metric,
        "selection_split": "val",
        "selection_rationale": selection_rationale or f"Selected by {metric} on the internal holdout.",
        "performance_table": [baseline_row, candidate_row],
        "best_model": candidate_name,
        "best_model_key": "candidate",
        "delta_vs_baseline": delta,
        "confusion_matrix": None,
        "residual_summary": residual_summary,
        "approved": all(c.get("severity") != "FAIL" for c in checks) and not worse,
        "failed_checks": [c for c in checks if c.get("severity") == "FAIL"],
        "warned_checks": [c for c in checks if c.get("severity") == "WARN"],
        "warnings": warnings,
        "artifacts": artifacts,
    }


# ── validity checks (A–F) ─────────────────────────────────────────────────────

def _validity_checks(task_type: str, rows: list[dict], metric: str) -> list[dict]:
    checks: list[dict] = []
    if not all(r.get("split") == "test" for r in rows):  # Check A / E
        checks.append({"check": "held_out_only", "severity": "FAIL", "message": "Non-test split in performance_table."})
    if len(rows) < 2:  # Check C
        checks.append({"check": "baseline_present", "severity": "FAIL", "message": "Baseline missing from performance_table."})
    reg_keys, cls_keys = {"rmse", "mae", "r2"}, {"roc_auc", "f1_weighted", "accuracy"}
    present = set(rows[-1].keys())
    if task_type == REGRESSION and not (reg_keys & present):  # Check B
        checks.append({"check": "metric_matches_task", "severity": "FAIL", "message": "No regression metric present."})
    if task_type in (BINARY, MULTICLASS) and not (cls_keys & present):
        checks.append({"check": "metric_matches_task", "severity": "FAIL", "message": "No classification metric present."})
    return checks


# ── plot helpers ──────────────────────────────────────────────────────────────

def _save(fig, run_id: str, name: str, artifacts_dir: Path) -> str:
    path = artifacts_dir / f"{run_id}_{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_confusion(cm: np.ndarray, classes: list, run_id: str, artifacts_dir: Path) -> str:
    norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([str(c) for c in classes])
    ax.set_yticklabels([str(c) for c in classes])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.2f}", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, run_id, "confusion_matrix", artifacts_dir)


def _plot_roc(y_true_pos, score, run_id: str, artifacts_dir: Path) -> str:
    fpr, tpr, _ = roc_curve(y_true_pos, score)
    auc = roc_auc_score(y_true_pos, score)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, color="#1a3a5c", label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return _save(fig, run_id, "roc_curve", artifacts_dir)


def _plot_pr(y_true_pos, score, run_id: str, artifacts_dir: Path) -> str:
    precision, recall, _ = precision_recall_curve(y_true_pos, score)
    ap = average_precision_score(y_true_pos, score)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(recall, precision, color="#4a9fd4", label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    return _save(fig, run_id, "pr_curve", artifacts_dir)


def _plot_residuals(y_pred, residuals, run_id: str, artifacts_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_pred, residuals, s=10, alpha=0.5, color="#4a9fd4")
    ax.axhline(0, color="#1a3a5c", lw=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (pred - true)")
    ax.set_title("Residual Plot")
    return _save(fig, run_id, "residuals", artifacts_dir)


def plot_feature_importance(importances: dict, run_id: str, artifacts_dir: str | Path) -> str | None:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    items = sorted(importances.items(), key=lambda kv: abs(kv[1]), reverse=True)[:15]
    if not items:
        return None
    labels = [k for k, _ in items][::-1]
    values = [abs(v) for _, v in items][::-1]
    fig, ax = plt.subplots(figsize=(5.5, max(2.5, 0.35 * len(labels) + 1)))
    ax.barh(labels, values, color="#4a9fd4")
    ax.set_xlabel("Importance (|association|)")
    ax.set_title("Feature Importance")
    return _save(fig, run_id, "feature_importance", artifacts_dir)


def _plot_pred_vs_actual(y_true, y_pred, run_id: str, artifacts_dir: Path) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, s=10, alpha=0.5, color="#4a9fd4")
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], "--", color="#1a3a5c")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    return _save(fig, run_id, "pred_vs_actual", artifacts_dir)


# ── small helpers ─────────────────────────────────────────────────────────────

def _delta(baseline_row: dict, candidate_row: dict, keys: list[str]) -> dict:
    out: dict = {}
    for key in keys:
        b, c = baseline_row.get(key), candidate_row.get(key)
        if isinstance(b, (int, float)) and isinstance(c, (int, float)):
            out[key] = {"baseline": b, "candidate": c, "delta": c - b, "split": "test"}
    return out


def _w(wtype: str, message: str, severity: str = "WARN") -> dict:
    return {"type": wtype, "column": None, "message": message, "severity": severity}


def _py(value):
    return value.item() if hasattr(value, "item") else value
