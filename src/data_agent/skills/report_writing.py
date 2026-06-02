"""report-writing skill: build the 11-section report from AnalysisState.

Generates the Markdown report, embeds existing plot artifacts, converts to PDF via
``scripts/md_to_pdf.convert``, and copies the PDF to the repo-root ``report.pdf``.
Every number is pulled from state; no causal language; limitations cover the run's
warnings. (The deterministic ``reporting.write_report`` remains the fallback.)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_SUPERVISED = {"binary_classification", "multiclass_classification", "regression"}
_SECTIONS = [
    "executive_summary", "objective", "data_overview", "data_quality", "methods",
    "exploratory_findings", "modeling_results", "interpretation", "limitations",
    "recommendations", "appendix",
]
_FIGURE_SECTION = {
    "confusion_matrix_plot": ("Figure: Confusion Matrix", "Section 7"),
    "roc_curve_plot": ("Figure: ROC Curve", "Section 7"),
    "pr_curve_plot": ("Figure: Precision-Recall Curve", "Section 7"),
    "residual_plot": ("Figure: Residuals", "Section 7"),
    "pred_vs_actual_plot": ("Figure: Predicted vs Actual", "Section 7"),
    "feature_importance_plot": ("Figure: Feature Importance", "Section 8"),
}


def write_analysis_report(*, state: Any, repo_root: str | Path = ".") -> dict:
    repo_root = Path(repo_root).resolve()
    task = state.task_spec or {}
    profile = state.data_profile or {}
    evaluation = state.evaluation or {}
    interpretation = state.interpretation or {}
    leakage = state.leakage_audit or {}
    artifacts = dict(state.artifacts or {})
    task_type = task.get("task_type", "unknown")
    supervised = task_type in _SUPERVISED

    figures = _collect_figures(evaluation, artifacts)
    referenced: list[dict] = [{"label": lab, "path": p, "section": sec} for lab, p, sec in figures]

    md = _build_markdown(state, task, profile, evaluation, interpretation, leakage, artifacts, figures, supervised)

    reports_dir = repo_root / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / f"{state.run_id}_report.md"
    md_path.write_text(md, encoding="utf-8")

    pdf_path = reports_dir / f"{state.run_id}_report.pdf"
    report_format = "pdf"
    warnings: list[dict] = []
    try:
        # md_to_pdf.py lives next to the code (<repo>/scripts), not the output dir.
        scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from md_to_pdf import convert as md_to_pdf  # type: ignore

        md_to_pdf(str(md_path), str(pdf_path))
        if not pdf_path.exists():
            raise RuntimeError("PDF not produced")
        (repo_root / "report.pdf").write_bytes(pdf_path.read_bytes())
        final_path = str(repo_root / "report.pdf")
    except Exception as exc:  # fall back to markdown
        report_format = "markdown"
        final_path = str(md_path)
        warnings.append({"type": "pdf_conversion_failed", "message": str(exc), "severity": "WARN"})

    state.report_draft = md
    state.report_path = final_path

    return {
        "report_path": final_path,
        "report_format": report_format,
        "referenced_artifacts": referenced,
        "sections_present": list(_SECTIONS),
        "word_count": len(md.split()),
        "warnings_addressed": _warnings_addressed(task, profile),
        "warnings": warnings,
    }


# ── markdown assembly ─────────────────────────────────────────────────────────

def _build_markdown(state, task, profile, evaluation, interpretation, leakage, artifacts, figures, supervised) -> str:
    task_type = task.get("task_type", "unknown")
    target = task.get("target_variable")
    goal = (state.user_request.goal if state.user_request else "") or "Do the data analysis"
    n_rows = profile.get("n_rows", "—")
    n_cols = profile.get("n_columns", "—")
    metric = evaluation.get("selection_metric", task.get("metric", "—"))
    pt = evaluation.get("performance_table") or []
    cand = pt[-1] if pt else {}
    primary_val = cand.get(metric) if isinstance(cand, dict) else None
    primary_str = f"{primary_val:.4f}" if isinstance(primary_val, (int, float)) else "—"

    lines: list[str] = []
    A = lines.append

    A("# Automated Data Analysis Report")
    A("")
    A("## 1. Executive Summary")
    A("")
    if supervised:
        A(f"This analysis addressed a **{task_type}** task on target `{target}`. "
          f"The selected model achieved a held-out `{metric}` of {primary_str}. "
          f"Metrics are reported on a held-out split; all findings are associative/predictive, not causal. "
          f"The primary limitation is that results are scoped to the supplied dataset and require external validation before deployment.")
    else:
        A(f"This analysis was **descriptive**. No predictive model was trained. "
          f"It summarizes the structure and quality of a {n_rows}-row, {n_cols}-column dataset.")
    A("")

    A("## 2. Objective")
    A("")
    A(f"> {goal}")
    A("")
    A(f"Interpreted task type: `{task_type}`. {task.get('rationale', '')}")
    A("")

    A("## 3. Data Overview")
    A("")
    A(f"Rows: {n_rows} | Columns: {n_cols} | Analysis date: {state.created_at or '—'}")
    A("")
    A("| Column | Type | Role |")
    A("|---|---|---|")
    simplified = profile.get("simplified_types", {})
    excluded = set(task.get("excluded_from_features", []) or [])
    for col in (profile.get("columns") or [])[:30]:
        role = "target" if col == target else ("excluded" if col in excluded else "feature")
        A(f"| {col} | {simplified.get(col, '—')} | {role} |")
    A("")
    if supervised and state.split_metadata:
        sm = state.split_metadata
        A(f"Split: train={sm.get('n_train')}, test={sm.get('n_test')}, seed={sm.get('random_state')}, stratified={sm.get('stratified')}.")
        A("")

    A("## 4. Data Quality Assessment")
    A("")
    miss = profile.get("missing_summary", {})
    miss_rows = [(c, m) for c, m in miss.items() if m.get("missing_rate", 0) > 0]
    if miss_rows:
        A("| Column | Missing % | Severity |")
        A("|---|---|---|")
        for c, m in sorted(miss_rows, key=lambda kv: -kv[1].get("missing_rate", 0))[:15]:
            A(f"| {c} | {m.get('missing_rate', 0):.1%} | {m.get('severity', '—')} |")
        A("")
    A(f"Duplicate rows: {profile.get('duplicate_count', 0)}. "
      f"Potential ID columns (excluded): {', '.join(profile.get('potential_id_columns', []) or []) or 'none'}. "
      f"High-cardinality columns: {', '.join(profile.get('high_cardinality_columns', []) or []) or 'none'}.")
    A("")
    A(f"Leakage audit: **{(leakage.get('leakage_risk', 'n/a') or 'n/a').upper()}** "
      f"(approved={leakage.get('approved')}). "
      f"Flagged columns: {', '.join(s['column'] for s in (leakage.get('suspected_columns') or []) if s.get('column')) or 'none'}.")
    A("")

    A("## 5. Methods")
    A("")
    A("Numeric features were median-imputed (and standardized for linear/distance models); "
      "categorical features were most-frequent-imputed and one-hot encoded with unknown categories ignored. "
      f"Models were compared on a held-out split and selected by `{metric}`. "
      "Baselines were evaluated before candidate models.")
    A("")

    A("## 6. Exploratory Findings")
    A("")
    findings = (state.eda_results or {}).get("key_findings") if state.eda_results else None
    if findings:
        for f in findings[:7]:
            A(f"- {f}")
    else:
        dist = profile.get("target_distribution") or {}
        if supervised and task_type != "regression" and dist.get("classes"):
            parts = ", ".join(f"{c['label']}={c['fraction']:.1%}" for c in dist["classes"][:6])
            A(f"- Target `{target}` class balance: {parts} (majority {dist.get('majority_class_rate', 0):.1%}).")
        A(f"- {len(profile.get('numeric_columns', {}))} numeric and {len(profile.get('categorical_columns', {}))} categorical columns profiled.")
        if miss_rows:
            A(f"- {len(miss_rows)} columns contain missing values; highest is "
              f"{max(miss_rows, key=lambda kv: kv[1]['missing_rate'])[0]}.")
    A("")

    A("## 7. Modeling and Prediction Results")
    A("")
    if supervised and pt:
        A(_perf_table_md(task_type, evaluation))
        A("")
        delta = evaluation.get("delta_vs_baseline") or {}
        if metric in delta:
            d = delta[metric]
            A(f"On `{metric}`, the candidate scores {d['candidate']:.4f} vs baseline {d['baseline']:.4f} "
              f"(delta {d['delta']:+.4f}).")
        A("")
        for key in ("confusion_matrix_plot", "roc_curve_plot", "residual_plot", "pred_vs_actual_plot"):
            p = _fig_path(evaluation, artifacts, key)
            if p:
                A(f"![{_FIGURE_SECTION[key][0]}]({p})")
        A("")
    else:
        A("This analysis is descriptive. No predictive model was trained.")
        A("")

    A("## 8. Interpretation")
    A("")
    importances = interpretation.get("feature_importance") or {}
    if importances:
        A("| Feature | Importance |")
        A("|---|---|")
        for feat, score in list(importances.items())[:10]:
            A(f"| {feat} | {score:.4f} |")
        A("")
        fip = _fig_path(evaluation, artifacts, "feature_importance_plot")
        if fip:
            A(f"![{_FIGURE_SECTION['feature_importance_plot'][0]}]({fip})")
            A("")
    A("Features are reported as associations/predictors, not causes.")
    A("")

    A("## 9. Limitations")
    A("")
    for lim in _limitations(task, profile, evaluation, leakage, supervised):
        A(f"- {lim}")
    A("")

    A("## 10. Recommendations")
    A("")
    A("- Validate these results on data from outside this dataset before any deployment decision.")
    if miss_rows:
        A("- Improve collection for high-missing columns to reduce imputation reliance.")
    A("- Consider additional feature engineering or alternative algorithms if higher performance is required.")
    A("")

    A("## 11. Appendix")
    A("")
    A(f"Run id: `{state.run_id}` | Task: `{task_type}` | Target: `{target}`.")
    A("")
    A("| Artifact | Path |")
    A("|---|---|")
    for label, path in artifacts.items():
        A(f"| {label} | {path} |")
    for label, path, _ in figures:
        A(f"| {label} | {path} |")
    A("")
    return "\n".join(lines)


def _perf_table_md(task_type: str, evaluation: dict) -> str:
    if task_type == "regression":
        cols = ["rmse", "mae", "r2"]
    else:
        cols = ["roc_auc", "f1_weighted", "accuracy"]
    header = "| Model | Split | " + " | ".join(c.upper() for c in cols) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(cols)) + "|"
    rows = [header, sep]
    for r in evaluation.get("performance_table", []):
        vals = " | ".join(f"{r.get(c):.4f}" if isinstance(r.get(c), (int, float)) else "—" for c in cols)
        rows.append(f"| {r.get('model', '—')} | {r.get('split', 'test')} | {vals} |")
    delta = evaluation.get("delta_vs_baseline") or {}
    if delta:
        vals = " | ".join(f"{delta[c]['delta']:+.4f}" if c in delta else "—" for c in cols)
        rows.append(f"| Delta | — | {vals} |")
    return "\n".join(rows)


def _collect_figures(evaluation: dict, artifacts: dict) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    eval_art = evaluation.get("artifacts") or {}
    for key, (label, section) in _FIGURE_SECTION.items():
        path = eval_art.get(key) or artifacts.get(key)
        if path and os.path.exists(path):
            out.append((label, path, section))
    return out


def _fig_path(evaluation: dict, artifacts: dict, key: str) -> str | None:
    path = (evaluation.get("artifacts") or {}).get(key) or artifacts.get(key)
    return path if path and os.path.exists(path) else None


def _limitations(task, profile, evaluation, leakage, supervised) -> list[str]:
    out: list[str] = []
    miss = profile.get("missing_summary", {})
    high_miss = [c for c, m in miss.items() if m.get("missing_rate", 0) > 0.10]
    if high_miss:
        out.append(f"Columns with >10% missing values may bias results: {', '.join(high_miss[:8])}.")
    n_rows = profile.get("n_rows", 0) or 0
    if supervised and n_rows < 1000:
        out.append(f"Sample size is modest (N={n_rows}); results may not be stable across samples.")
    if task.get("class_imbalance_detected"):
        out.append("Class imbalance was detected; accuracy alone is misleading — ROC-AUC/PR-AUC are emphasized.")
    if task.get("target_inferred"):
        out.append("The target variable was inferred, not explicitly named; confirm it matches intent.")
    if (leakage.get("leakage_risk") == "medium"):
        out.append("The leakage audit raised WARN-level findings; review flagged columns.")
    if evaluation.get("worse_than_baseline"):
        out.append("The candidate did not clearly beat the baseline on the primary metric.")
    out.append("Findings are associative/predictive, not causal, and are scoped to this dataset.")
    out.append("Modeling assumes the supplied features are available at prediction time.")
    return out


def _warnings_addressed(task, profile) -> list[str]:
    out: list[str] = []
    for w in (task.get("warnings") or []):
        out.append(f"{w.get('type')}:{w.get('column')}")
    for w in (profile.get("warnings") or []):
        out.append(f"{w.get('type')}:{w.get('column')}")
    return out
