"""Task-type, metric, and output-format resolution for Award B.

The competition harness places ``data/DATA_DESCRIPTION.md`` next to the data at
evaluation time. That description is the *authority* for what the task is, how it
is scored, and what the submission must contain. Hidden descriptions vary in
wording, so this module combines:

1. lightweight text parsing of the description (primary), and
2. data-driven inference from the training target and the sample submission
   (fallback + feasibility check).

The resolved :class:`TaskSpec` becomes the single source of truth that the model
selection, submission-formatting, and reporting stages branch on, so the
pipeline adapts to regression, binary classification, and multiclass
classification without any dataset-specific hardcoding.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

import numpy as np
import pandas as pd


# ── task types ───────────────────────────────────────────────────────────────
REGRESSION = "regression"
BINARY = "binary_classification"
MULTICLASS = "multiclass_classification"

# ── metrics ──────────────────────────────────────────────────────────────────
ACCURACY = "accuracy"
ROC_AUC = "roc_auc"
F1 = "f1"
F1_MACRO = "f1_macro"
LOG_LOSS = "log_loss"
MAE = "mae"
RMSE = "rmse"
R2 = "r2"
BLOCK_MAE = "block_mae"

_GREATER_IS_BETTER = {
    ACCURACY: True,
    ROC_AUC: True,
    F1: True,
    F1_MACRO: True,
    R2: True,
    MAE: False,
    RMSE: False,
    LOG_LOSS: False,
    BLOCK_MAE: False,
}

_CLASS_METRICS = {ACCURACY, ROC_AUC, F1, F1_MACRO, LOG_LOSS}
_REG_METRICS = {MAE, RMSE, R2, BLOCK_MAE}

# Generic classification keyword used only as an intermediate signal before the
# binary/multiclass distinction is resolved from the data.
_CLASSIFICATION = "classification"

# Maximum number of distinct integer-valued target levels treated as multiclass
# when the description is silent. Guards against integer-valued *regression*
# targets such as counts, which usually take many more levels than this.
_MAX_CLASS_LEVELS = 20


@dataclass
class TaskSpec:
    task_type: str
    metric: str
    greater_is_better: bool
    output_kind: str  # "value" | "label" | "probability"
    class_labels: list | None = None
    positive_label: Any = None
    evidence: dict = field(default_factory=dict)

    @property
    def is_classification(self) -> bool:
        return self.task_type in (BINARY, MULTICLASS)

    def to_dict(self) -> dict:
        data = asdict(self)
        if self.class_labels is not None:
            data["class_labels"] = [_to_py(v) for v in self.class_labels]
        data["positive_label"] = _to_py(self.positive_label)
        return data


def resolve_task_spec(
    description_text: str | None,
    target_series: pd.Series,
    sample_target_series: pd.Series | None = None,
    has_block_col: bool = False,
    target_column: str | None = None,
) -> TaskSpec:
    """Resolve the task type, metric, and output format.

    Description hints win when present *and* feasible against the data; otherwise
    the data-driven inference is used. All decisions and their sources are
    recorded in ``evidence`` for logging and the report.
    """
    description_text = description_text or ""
    target_column = target_column or (str(target_series.name) if target_series.name is not None else "target")

    desc_task = _parse_description_task(description_text)
    desc_metric = _parse_description_metric(description_text)
    desc_output = _parse_description_output(description_text, target_column)

    data = _infer_from_data(target_series, sample_target_series)

    # ── task type ────────────────────────────────────────────────────────────
    if desc_task == _CLASSIFICATION:
        resolved_task = BINARY if data["nunique"] == 2 else MULTICLASS
        task_source = "description+data"
    elif desc_task in (BINARY, MULTICLASS, REGRESSION) and _task_feasible(desc_task, data):
        resolved_task = desc_task
        task_source = "description"
    else:
        resolved_task = data["task"]
        task_source = "data(description_infeasible)" if desc_task else "data"

    # ── output kind ──────────────────────────────────────────────────────────
    if resolved_task == REGRESSION:
        output_kind = "value"
    else:
        output_kind = desc_output or data["output_kind"]
        if output_kind not in ("label", "probability"):
            output_kind = "label"

    # ── metric ───────────────────────────────────────────────────────────────
    metric = desc_metric
    if metric == F1 and resolved_task == MULTICLASS:
        metric = F1_MACRO
    if metric is None or not _metric_matches_task(metric, resolved_task):
        metric = _default_metric(resolved_task, output_kind, has_block_col)

    # ── class labels / positive class ────────────────────────────────────────
    class_labels = data["class_labels"] if resolved_task in (BINARY, MULTICLASS) else None
    positive_label = None
    if resolved_task == BINARY and class_labels:
        positive_label = _choose_positive_label(class_labels)

    evidence = {
        "task_source": task_source,
        "description_task": desc_task,
        "description_metric": desc_metric,
        "description_output": desc_output,
        "data_task": data["task"],
        "n_unique_target": data["nunique"],
        "target_is_numeric": data["is_numeric"],
        "target_integer_like": data["integer_like"],
        "sample_is_integer": data["sample_is_int"],
        "sample_is_float": data["sample_is_float"],
    }

    return TaskSpec(
        task_type=resolved_task,
        metric=metric,
        greater_is_better=_GREATER_IS_BETTER[metric],
        output_kind=output_kind,
        class_labels=class_labels,
        positive_label=positive_label,
        evidence=evidence,
    )


# ── description parsing ───────────────────────────────────────────────────────

def _parse_description_task(text: str) -> str | None:
    t = text.lower()
    # Order matters: "binary classification" also contains "classif", so the
    # more specific phrases are tested first.
    if re.search(r"binary[\s-]+classification", t) or re.search(r"predict[^.]{0,40}\bbinary\b", t):
        return BINARY
    if re.search(r"multi[\s-]?class", t) or "multinomial" in t:
        return MULTICLASS
    if re.search(r"\bclassif", t):  # classification / classify / classifier
        return _CLASSIFICATION
    if re.search(r"\bregression\b", t) or re.search(r"predict[^.]{0,40}(continuous|numeric|real[\s-]?valued)", t):
        return REGRESSION
    return None


def _parse_description_metric(text: str) -> str | None:
    t = text.lower()
    # Prefer a window around an evaluation cue ("the evaluation metric is ...").
    windows = [
        t[max(0, m.start() - 40): m.start() + 120]
        for m in re.finditer(r"(evaluat|metric|scored|judged|score is|measured by|ranked by)", t)
    ]
    for scope in windows + [t]:
        hit = _match_metric(scope)
        if hit:
            return hit
    return None


def _match_metric(t: str) -> str | None:
    if re.search(r"roc[\s-]?auc", t) or "area under" in t or re.search(r"\bauc\b", t):
        return ROC_AUC
    if re.search(r"log[\s-]?loss", t) or "cross-entropy" in t or "cross entropy" in t:
        return LOG_LOSS
    if re.search(r"\bf1\b", t) or "f1-score" in t or "f1 score" in t:
        return F1
    if "accuracy" in t:
        return ACCURACY
    if "rmse" in t or "root mean squared" in t or "root-mean-squared" in t:
        return RMSE
    if re.search(r"\bmae\b", t) or "mean absolute error" in t:
        return MAE
    if re.search(r"\br2\b", t) or "r-squared" in t or "r^2" in t or "coefficient of determination" in t:
        return R2
    return None


def _parse_description_output(text: str, target_column: str) -> str | None:
    t = text.lower()
    tc = re.escape(target_column.lower())

    prob = re.search(r"probabilit", t)
    if prob:
        window = t[max(0, prob.start() - 80): prob.start() + 80]
        if any(k in window for k in [target_column.lower(), "submit", "submission", "output", "predict", "score", "column"]):
            return "probability"

    # Schema-table or inline type declaration for the target, e.g.
    # "| `Survived` | int |" or "Survived: int" or "`Survived` int".
    if re.search(rf"`?{tc}`?\s*[:|]?\s*\|?\s*int", t):
        return "label"
    if re.search(r"\b0 or 1\b", t) or "binary value" in t or "class label" in t or "predicted label" in t:
        return "label"
    return None


# ── data-driven inference ─────────────────────────────────────────────────────

def _infer_from_data(target: pd.Series, sample_target: pd.Series | None) -> dict:
    s = target.dropna()
    nunique = int(s.nunique())
    numeric = pd.to_numeric(s, errors="coerce")
    is_numeric = bool(len(s) > 0 and numeric.notna().all())
    if is_numeric:
        vals = numeric.to_numpy(dtype=float)
        integer_like = bool(np.all(np.isfinite(vals)) and np.all(np.mod(vals, 1) == 0))
    else:
        integer_like = False

    sample_is_int = False
    sample_is_float = False
    sample_is_numeric = True
    if sample_target is not None:
        st = sample_target.dropna()
        if len(st):
            st_num = pd.to_numeric(st, errors="coerce")
            if st_num.notna().all():
                svals = st_num.to_numpy(dtype=float)
                sample_is_int = bool(np.all(np.mod(svals, 1) == 0))
                sample_is_float = not sample_is_int
            else:
                sample_is_numeric = False

    if not is_numeric:
        task = BINARY if nunique == 2 else MULTICLASS
    elif nunique == 2:
        task = BINARY
    elif integer_like and 2 < nunique <= _MAX_CLASS_LEVELS and (sample_is_int or sample_target is None):
        task = MULTICLASS
    else:
        task = REGRESSION

    if task == REGRESSION:
        output_kind = "value"
    elif not sample_is_numeric:
        output_kind = "label"
    elif sample_is_float:
        output_kind = "probability"
    else:
        output_kind = "label"

    class_labels = sorted(s.unique().tolist(), key=lambda x: str(x)) if task != REGRESSION else None

    return {
        "task": task,
        "output_kind": output_kind,
        "nunique": nunique,
        "is_numeric": is_numeric,
        "integer_like": integer_like,
        "sample_is_int": sample_is_int,
        "sample_is_float": sample_is_float,
        "class_labels": class_labels,
    }


def _task_feasible(desc_task: str, data: dict) -> bool:
    if desc_task == BINARY:
        return data["nunique"] == 2
    if desc_task == MULTICLASS:
        return data["nunique"] >= 2 and (not data["is_numeric"] or data["integer_like"])
    if desc_task == REGRESSION:
        return data["is_numeric"]
    return True


def _default_metric(task: str, output_kind: str, has_block_col: bool) -> str:
    if task == REGRESSION:
        return BLOCK_MAE if has_block_col else MAE
    if task == BINARY:
        return ROC_AUC if output_kind == "probability" else ACCURACY
    if task == MULTICLASS:
        return ACCURACY
    return MAE


def _metric_matches_task(metric: str, task: str) -> bool:
    if task == REGRESSION:
        return metric in _REG_METRICS
    return metric in _CLASS_METRICS


def _choose_positive_label(labels: list) -> Any:
    label_set = set(labels)
    if 1 in label_set:
        return 1
    if True in label_set:
        return True
    for token in ("yes", "true", "survived", "positive", "1", "y"):
        for label in labels:
            if str(label).strip().lower() == token:
                return label
    return sorted(labels, key=lambda x: str(x))[-1]


def _to_py(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value
