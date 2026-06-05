"""report-review skill: independent audit of the generated report.

Implements ``.claude/skills/report-review/SKILL.md``: verify metric values match
``state.evaluation``, split labels present, no causal/over-claiming language, all
sections present, figures exist, no placeholders. ``approved`` only when no FAIL.
"""

from __future__ import annotations

import os
import re
from typing import Any

# Causal-claim phrases that should not appear in observational findings. Chosen to
# avoid matching the report's own disclaimers ("not causal", "not causes").
_CAUSAL_PHRASES = ["leads to", "results in", "caused by", "the effect of", "due to ", " drives "]
_OVERCLAIM_PHRASES = ["production-ready", "production ready", "any patient", "any customer",
                      "general population", "will generalize", "guaranteed"]
_PLACEHOLDERS = ["[tbd]", "to be added", "placeholder", "todo", "see figure below (to be added)"]
_SECTION_MARKERS = [f"## {i}." for i in range(1, 16)]
_SUPERVISED = {"binary_classification", "multiclass_classification", "regression"}


def review_report(*, state: Any) -> dict:
    report = state.report_draft or ""
    low = report.lower()
    evaluation = state.evaluation or {}
    task_type = (state.task_spec or {}).get("task_type", "unknown")
    supervised = task_type in _SUPERVISED

    required_revisions: list[dict] = []
    warnings: list[dict] = []

    def fail(check: str, detail: str):
        required_revisions.append({"check": check, "detail": detail, "severity": "FAIL"})

    def warn(check: str, detail: str):
        warnings.append({"check": check, "detail": detail, "severity": "WARN"})

    # 1 — all 11 sections present
    missing = [m for m in _SECTION_MARKERS if m not in report]
    if missing:
        fail("sections_present", f"Missing section markers: {missing}")

    # 2 — metric values match evaluation (candidate primary)
    pt = evaluation.get("performance_table") or []
    metric = evaluation.get("selection_metric")
    if supervised and pt and metric:
        val = pt[-1].get(metric)
        if isinstance(val, (int, float)) and f"{val:.4f}" not in report:
            warn("metric_match", f"Candidate {metric}={val:.4f} not found verbatim in report.")

    # 3 — split labels present
    if supervised and "split" not in low and "test" not in low:
        fail("split_labels", "No split label (test) found in report.")

    # 4 — causal language
    for phrase in _CAUSAL_PHRASES:
        if phrase in low:
            fail("causal_language", f"Causal phrase '{phrase.strip()}' present.")
            break

    # 5 — over-claiming generalization
    for phrase in _OVERCLAIM_PHRASES:
        if phrase in low:
            fail("overclaim", f"Over-claiming phrase '{phrase}' present.")
            break

    # 6 — limitations substantive
    lim = report.split("## 13. Limitations", 1)
    if len(lim) < 2 or len(lim[1].split("## 14", 1)[0].split()) < 20:
        fail("limitations", "Limitations section is missing or too brief.")

    # 7 — baseline comparison present (supervised)
    if supervised and "baseline" not in low:
        fail("baseline_comparison", "No baseline comparison found in report.")

    # 8 — referenced figures exist on disk
    for alt, path in re.findall(r"!\[(.*?)\]\((.+?)\)", report):
        if not os.path.exists(path):
            fail("figure_exists", f"Referenced figure missing on disk: {path}")

    # 9 — no placeholders
    for ph in _PLACEHOLDERS:
        if ph in low:
            fail("placeholder", f"Placeholder text '{ph}' present.")
            break

    approved = len(required_revisions) == 0
    return {
        "approved": approved,
        "verdict": "PASS" if approved else "FAIL",
        "required_revisions": required_revisions,
        "warnings": warnings,
        "checks_performed": ["sections", "metric_match", "split_labels", "causal_language",
                             "overclaim", "limitations", "baseline_comparison", "figure_exists", "placeholder"],
    }
