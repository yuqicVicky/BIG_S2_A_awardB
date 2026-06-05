"""report-writing skill: build the comprehensive analysis report from AnalysisState.

Generates Markdown, embeds existing plot artifacts, converts to PDF via
scripts/md_to_pdf.convert, and copies PDF to repo-root report.pdf.
All numbers come from state; no causal language; paths are relative to repo_root.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_SUPERVISED = {"binary_classification", "multiclass_classification", "regression"}
_SECTIONS = [
    "executive_summary", "objective", "data_overview", "data_pattern_understanding",
    "data_quality", "methods", "exploratory_findings", "modeling_results",
    "residual_analysis", "interpretation", "leakage_and_feature_audit",
    "validation_strategy", "limitations", "recommendations", "appendix",
]
_FIGURE_SECTION = {
    "confusion_matrix_plot": ("Figure: Confusion Matrix", "Section 8"),
    "roc_curve_plot": ("Figure: ROC Curve", "Section 8"),
    "pr_curve_plot": ("Figure: Precision-Recall Curve", "Section 8"),
    "residual_plot": ("Figure: Residuals", "Section 9"),
    "pred_vs_actual_plot": ("Figure: Predicted vs Actual", "Section 9"),
    "feature_importance_plot": ("Figure: Feature Importance", "Section 10"),
}


def write_analysis_report(*, state: Any, repo_root: str | Path = ".") -> dict:
    repo_root = Path(repo_root).resolve()
    task = state.task_spec or {}
    profile = state.data_profile or {}
    evaluation = state.evaluation or {}
    interpretation = state.interpretation or {}
    leakage = state.leakage_audit or {}
    artifacts = dict(state.artifacts or {})
    residual_analysis = state.residual_analysis or {}
    data_pattern = state.data_pattern_report or {}
    task_type = task.get("task_type", "unknown")
    supervised = task_type in _SUPERVISED

    figures = _collect_figures(evaluation, artifacts)
    referenced: list[dict] = [{"label": lab, "path": p, "section": sec} for lab, p, sec in figures]

    md = _build_markdown(
        state, task, profile, evaluation, interpretation, leakage,
        artifacts, figures, supervised, residual_analysis, data_pattern, repo_root,
    )

    reports_dir = repo_root / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / f"{state.run_id}_report.md"
    md_path.write_text(md, encoding="utf-8")

    pdf_path = reports_dir / f"{state.run_id}_report.pdf"
    report_format = "pdf"
    warnings: list[dict] = []
    try:
        scripts_dir = str(Path(__file__).resolve().parents[3] / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from md_to_pdf import convert as md_to_pdf  # type: ignore

        md_to_pdf(str(md_path), str(pdf_path))
        if not pdf_path.exists():
            raise RuntimeError("PDF not produced")
        (repo_root / "report.pdf").write_bytes(pdf_path.read_bytes())
        final_path = str(repo_root / "report.pdf")
    except Exception as exc:
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

def _build_markdown(
    state, task, profile, evaluation, interpretation, leakage,
    artifacts, figures, supervised, residual_analysis, data_pattern, repo_root,
) -> str:
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
    all_model_scores = (state.raw_modeling_results or {}).get("all_scores", [])

    lines: list[str] = []
    A = lines.append

    # ── 1. Executive Summary ──────────────────────────────────────────────────
    A("# Automated Data Analysis Report")
    A("")
    A("## 1. Executive Summary")
    A("")
    if supervised:
        A(f"This analysis addressed a **{task_type}** task on target `{target}`. "
          f"The selected model achieved a held-out `{metric}` of {primary_str}. "
          f"All findings are associative/predictive, not causal. "
          f"The primary limitation is that results are scoped to the supplied training data "
          f"and require external validation before deployment.")
    else:
        A(f"This analysis is **descriptive**. No predictive model was trained. "
          f"It summarizes the structure and quality of a {n_rows}-row, {n_cols}-column dataset.")
    A("")

    # ── 2. Objective ──────────────────────────────────────────────────────────
    A("## 2. Objective")
    A("")
    A(f"> {goal}")
    A("")
    A(f"Interpreted task type: `{task_type}`. {task.get('rationale', '')}")
    A("")

    # ── 3. Data Overview ──────────────────────────────────────────────────────
    A("## 3. Data Overview")
    A("")
    A(f"Rows: {n_rows} | Columns: {n_cols} | Analysis date: {state.created_at or '—'}")
    A("")

    # Schema comparison from pattern report
    schema_cmp = data_pattern.get("schema_comparison", {})
    if schema_cmp:
        A(f"**Training rows:** {schema_cmp.get('n_train_rows', n_rows)} | "
          f"**Prediction rows:** {schema_cmp.get('n_predict_rows', '—')}")
        A("")
        train_only = schema_cmp.get("train_only_columns", [])
        if train_only:
            A(f"**Train-only columns** (absent from prediction data — correctly excluded from model): "
              f"`{'`, `'.join(train_only[:10])}`")
            A("")
        pred_only = schema_cmp.get("predict_only_columns", [])
        if pred_only:
            A(f"**Prediction-only columns:** `{'`, `'.join(pred_only[:5])}`")
            A("")

    A("| Column | Type | Role |")
    A("|---|---|---|")
    simplified = profile.get("simplified_types", {})
    train_only_set = set(schema_cmp.get("train_only_columns", []))
    # Use the authoritative model feature set (from bundle.feature_columns stored in model_results)
    final_model_features = set((state.model_results or {}).get("final_feature_columns", []))
    row_id_col = task.get("row_id_column") or ""
    for col in (profile.get("columns") or [])[:40]:
        if col == target:
            role = "target"
        elif col in train_only_set:
            role = "excluded (train-only)"
        elif col == row_id_col:
            role = "row id"
        elif final_model_features and col in final_model_features:
            role = "feature"
        elif final_model_features:
            # Column is known but not in the actual model — excluded for some reason
            role = "excluded"
        else:
            # No feature set info: fall back to conservative labeling
            role = "feature"
        A(f"| {col} | {simplified.get(col, '—')} | {role} |")
    A("")
    if supervised and state.split_metadata:
        sm = state.split_metadata
        A(f"Split type: `{sm.get('type', '—')}` | "
          f"Train rows: {sm.get('n_train', '—')} | "
          f"Holdout rows: {sm.get('n_holdout', sm.get('n_test', '—'))}")
        if sm.get("time_column"):
            A(f"Time column used for holdout: `{sm.get('time_column')}`")
        A("")

    # ── 4. Data Pattern Understanding ─────────────────────────────────────────
    A("## 4. Data Pattern Understanding")
    A("")
    _write_data_pattern_section(lines, data_pattern, target)

    # ── 5. Data Quality Assessment ───────────────────────────────────────────
    A("## 5. Data Quality Assessment")
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
      f"Potential ID columns: {', '.join(profile.get('potential_id_columns', []) or []) or 'none'}. "
      f"High-cardinality columns: {', '.join(profile.get('high_cardinality_columns', []) or []) or 'none'}.")
    A("")

    # ── 6. Methods ───────────────────────────────────────────────────────────
    A("## 6. Methods")
    A("")
    val_strat = (data_pattern.get("validation_implications") or {}).get("recommended_strategy", "")
    A("Numeric features were median-imputed (and standardized for linear/distance models); "
      "categorical features were most-frequent-imputed and one-hot encoded with unknown categories ignored. "
      f"Models were compared on a `{val_strat or 'held-out'}` split and selected by `{metric}`. "
      "Baselines were evaluated before candidate models. "
      "Train-only columns absent from prediction data were automatically excluded from the feature set.")
    time_cov = data_pattern.get("time_coverage", {})
    sm_section6 = state.split_metadata or {}
    if time_cov.get("has_time"):
        within = time_cov.get("within_period_pattern")
        if within and sm_section6.get("type") == "within_period_holdout":
            fit_dist = sm_section6.get("fit_day_distribution", {})
            val_dist = sm_section6.get("validation_day_distribution", {})
            pred_dist = sm_section6.get("prediction_day_distribution", {})
            feat = sm_section6.get("period_feature", within.get("feature", "sub-period"))
            A(f"\nUsing within-period holdout: fit on {feat} {fit_dist.get('min', '—')}–{fit_dist.get('max', '—')}, "
              f"validate on {feat} {val_dist.get('min', '—')}–{val_dist.get('max', '—')}, "
              f"prediction set uses {feat} {pred_dist.get('min', '—')}–{pred_dist.get('max', '—')}.")
        elif within:
            A(f"\nWithin-period pattern detected: training `{within['feature']}` range "
              f"{within['train_range']}, prediction `{within['feature']}` range {within['predict_range']}. "
              f"Chronological time-holdout was applied.")
        else:
            A(f"\nTime-based holdout: last ~20% of training timestamps held out as validation "
              f"(train: {time_cov.get('train_min','?')[:10]} – {time_cov.get('train_max','?')[:10]}).")
    A("")

    # ── 7. Exploratory Findings ───────────────────────────────────────────────
    A("## 7. Exploratory Findings")
    A("")
    _write_exploratory_findings(lines, state, task, profile, data_pattern, target, task_type, supervised)

    # ── 8. Modeling and Prediction Results ────────────────────────────────────
    A("## 8. Modeling and Prediction Results")
    A("")
    if supervised:
        if all_model_scores:
            A("### All models evaluated")
            A("")
            A(f"| Model | Status | {metric.upper()} |")
            A("|---|---|---|")
            for m in all_model_scores:
                status = m.get("status", "—")
                score = m.get("score")
                score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "—"
                selected_marker = " ← **selected**" if m.get("name") == (state.model_results or {}).get("selected_model_name") else ""
                A(f"| {m.get('name', '—')} | {status} | {score_str}{selected_marker} |")
            A("")
        elif pt:
            A(_perf_table_md(task_type, evaluation))
            A("")
        delta = evaluation.get("delta_vs_baseline") or {}
        if metric in delta:
            d = delta[metric]
            A(f"Selected model scored {d['candidate']:.4f} on `{metric}` vs baseline {d['baseline']:.4f} "
              f"(delta {d['delta']:+.4f}).")
            A("")
        for key in ("confusion_matrix_plot", "roc_curve_plot", "residual_plot", "pred_vs_actual_plot"):
            p = _fig_path(evaluation, artifacts, key)
            if p:
                rel_p = _make_relative(p, repo_root)
                A(f"![{_FIGURE_SECTION[key][0]}]({rel_p})")
        A("")
    else:
        A("This analysis is descriptive. No predictive model was trained.")
        A("")

    # ── 9. Residual Analysis ─────────────────────────────────────────────────
    A("## 9. Residual Analysis")
    A("")
    _write_residual_section(lines, residual_analysis, task_type)

    # ── 10. Interpretation ───────────────────────────────────────────────────
    A("## 10. Interpretation")
    A("")
    importances = interpretation.get("feature_importance") or {}
    if importances:
        A("Feature associations with target (|Pearson r|; model features only):")
        A("")
        A("| Feature | |Pearson r| |")
        A("|---|---|")
        for feat, score in list(importances.items())[:15]:
            A(f"| {feat} | {score:.4f} |")
        A("")
        fip = _fig_path(evaluation, artifacts, "feature_importance_plot")
        if fip:
            rel_fip = _make_relative(fip, repo_root)
            A(f"![{_FIGURE_SECTION['feature_importance_plot'][0]}]({rel_fip})")
            A("")
    A("Features are reported as statistical associations, not causes. "
      "Train-only columns (including any target-component columns) are excluded from this table.")
    A("")

    # ── 11. Leakage and Feature Audit ────────────────────────────────────────
    A("## 11. Leakage and Feature Audit")
    A("")
    _write_leakage_section(lines, leakage, data_pattern)

    # ── 12. Validation Strategy ───────────────────────────────────────────────
    A("## 12. Validation Strategy")
    A("")
    _write_validation_strategy_section(lines, state, data_pattern, metric)

    # ── 13. Limitations ──────────────────────────────────────────────────────
    A("## 13. Limitations")
    A("")
    for lim in _limitations(task, profile, evaluation, leakage, supervised, data_pattern, residual_analysis):
        A(f"- {lim}")
    A("")

    # ── 14. Recommendations ──────────────────────────────────────────────────
    A("## 14. Recommendations")
    A("")
    A("- Validate these results on data from outside this dataset before any deployment decision.")
    if miss_rows:
        A("- Improve collection for high-missing columns to reduce imputation reliance.")
    val_recs = (data_pattern.get("validation_implications") or {}).get("recommendations", [])
    for rec in val_recs[:3]:
        A(f"- {rec}")
    ra = residual_analysis.get("high_value_bias") or residual_analysis.get("high_value_underprediction", {})
    if ra.get("bias_direction") == "systematic_underprediction" and ra.get("fraction_underpredicted", 0) > 0.65:
        A("- Model systematically underpredicts high-value targets; consider log-transforming the target "
          "or using a model with explicit leaf-value boosting.")
    elif ra.get("bias_direction") == "systematic_overprediction" and ra.get("fraction_overpredicted", 0) > 0.65:
        A("- Model systematically overpredicts high-value targets; consider regularization or target clipping.")
    A("- Consider additional feature engineering or alternative algorithms if higher performance is required.")
    A("")

    # ── 15. Appendix ─────────────────────────────────────────────────────────
    A("## 15. Appendix")
    A("")
    A(f"Run id: `{state.run_id}` | Task: `{task_type}` | Target: `{target}`")
    A("")
    A("| Artifact | Relative path |")
    A("|---|---|")
    for label, path in artifacts.items():
        rel = _make_relative(str(path), repo_root)
        A(f"| {label} | {rel} |")
    for label, path, _ in figures:
        rel = _make_relative(str(path), repo_root)
        A(f"| {label} | {rel} |")
    A("")
    return "\n".join(lines)


# ── section helpers ───────────────────────────────────────────────────────────

def _write_data_pattern_section(lines: list, data_pattern: dict, target: str | None) -> None:
    A = lines.append

    if not data_pattern:
        A("_Data pattern analysis not available for this run._")
        A("")
        return

    # Time coverage
    tc = data_pattern.get("time_coverage", {})
    if tc.get("has_time"):
        A("### Temporal Coverage")
        A("")
        A(f"Temporal column: `{tc.get('temporal_column', '—')}`")
        A("")
        A("| Split | Min | Max | Unique timestamps |")
        A("|---|---|---|---|")
        A(f"| Train | {str(tc.get('train_min', '—'))[:19]} | {str(tc.get('train_max', '—'))[:19]} "
          f"| {tc.get('train_n_unique_timestamps', '—')} |")
        A(f"| Predict | {str(tc.get('predict_min', '—'))[:19]} | {str(tc.get('predict_max', '—'))[:19]} "
          f"| {tc.get('predict_n_unique_timestamps', '—')} |")
        A("")
        wpp = tc.get("within_period_pattern")
        if wpp:
            A(f"**Within-period split detected:** Training data has `{wpp['feature']}` values "
              f"{wpp['train_range'][0]}–{wpp['train_range'][1]}, prediction data has "
              f"{wpp['predict_range'][0]}–{wpp['predict_range'][1]}. "
              f"These are non-overlapping partitions of the same time period — "
              f"use within-period holdout to simulate this gap.")
            A("")
        elif tc.get("temporal_overlap") is False:
            A("Prediction rows are temporally **after** all training rows — "
              "time-based holdout is appropriate.")
            A("")

    # Target distribution
    td = data_pattern.get("target_distribution", {})
    if td and not td.get("error"):
        A("### Target Distribution")
        A("")
        A(f"Mean: {td.get('mean', '—'):.3f} | Std: {td.get('std', '—'):.3f} | "
          f"Min: {td.get('min', '—'):.3f} | Max: {td.get('max', '—'):.3f}")
        A("")
        A(f"Skewness: {td.get('skewness', '—'):.3f} | "
          f"Zero fraction: {td.get('zero_fraction', 0):.1%}")
        A("")
        A(f"*{td.get('interpretation', '')}*")
        A("")
        q = td.get("quantiles", {})
        if q:
            q_keys = ["p5", "p25", "p50", "p75", "p95"]
            A("| " + " | ".join(q_keys) + " |")
            A("|" + "|".join(["---"] * len(q_keys)) + "|")
            A("| " + " | ".join(f"{q.get(k, '—'):.2f}" if isinstance(q.get(k), (int, float)) else "—" for k in q_keys) + " |")
            A("")

    # Target by time features
    tbt = data_pattern.get("target_by_time_features", {})
    predictive_time = {k: v for k, v in tbt.items() if v.get("is_predictive")}
    if predictive_time:
        A("### Target Variation by Time Features")
        A("")
        A("| Feature | n groups | Target mean range | Predictive? |")
        A("|---|---|---|---|")
        for feat, info in list(predictive_time.items())[:6]:
            A(f"| {feat} | {info.get('n_groups', '—')} | "
              f"{info.get('target_mean_range', '—'):.3f} | yes |")
        A("")

    # Target by categorical
    tbc = data_pattern.get("target_by_categorical", {})
    predictive_cat = {k: v for k, v in tbc.items() if v.get("is_predictive")}
    if predictive_cat:
        A("### Target Variation by Categorical Features")
        A("")
        A("| Feature | n groups | Target mean range |")
        A("|---|---|---|")
        for feat, info in list(predictive_cat.items())[:5]:
            A(f"| {feat} | {info.get('n_groups', '—')} | "
              f"{info.get('target_mean_range', '—'):.3f} |")
        A("")

    # Target-component detection (uses high_risk_components list if available)
    tca = data_pattern.get("target_component_analysis", {})
    comps = tca.get("high_risk_components") or [c for c in tca.get("components", []) if c.get("is_component")]
    if comps:
        A("### Target-Component Columns (Train-Only — Correctly Excluded)")
        A("")
        A(f"The following train-only columns were detected as likely target components "
          f"(|r| > 0.50 with target) and are **absent from prediction data**, "
          f"so they were automatically excluded from the model:")
        A("")
        A("| Column | r with target | Risk tier | Likely additive component? |")
        A("|---|---|---|---|")
        for c in comps:
            additive = "yes" if c.get("likely_additive_component") else ("no" if c.get("likely_additive_component") is False else "—")
            risk = c.get("risk_tier", "high")
            A(f"| {c['column']} | {c['pearson_r_with_target']:.4f} | {risk} | {additive} |")
        A("")

    # Distribution shift
    ds = data_pattern.get("distribution_shift", {})
    if ds.get("has_notable_shift"):
        A("### Distribution Shift (Train vs Prediction)")
        A("")
        A("| Feature | Train mean | Predict mean | z-score | Severity |")
        A("|---|---|---|---|---|")
        for s in ds.get("shifted_features", [])[:8]:
            if s["severity"] in ("medium", "high"):
                A(f"| {s['feature']} | {s['train_mean']:.3f} | {s['predict_mean']:.3f} "
                  f"| {s['z_score']:.2f} | {s['severity']} |")
        A("")


def _write_exploratory_findings(
    lines, state, task, profile, data_pattern, target, task_type, supervised
) -> None:
    A = lines.append
    findings = (state.eda_results or {}).get("key_findings") if state.eda_results else None

    # Custom findings from pattern report
    td = data_pattern.get("target_distribution", {})
    tc = data_pattern.get("time_coverage", {})
    tbt = data_pattern.get("target_by_time_features", {})

    custom_findings: list[str] = []

    if supervised and td and not td.get("error"):
        custom_findings.append(
            f"Target `{target}`: mean={td.get('mean', '—'):.2f}, "
            f"std={td.get('std', '—'):.2f}, "
            f"skewness={td.get('skewness', '—'):.2f}. "
            f"{td.get('interpretation', '')}"
        )

    if tc.get("has_time"):
        wpp = tc.get("within_period_pattern")
        if wpp:
            custom_findings.append(
                f"Within-period split: train covers `{wpp['feature']}` "
                f"{wpp['train_range'][0]}–{wpp['train_range'][1]}, "
                f"prediction covers {wpp['predict_range'][0]}–{wpp['predict_range'][1]}."
            )
        else:
            custom_findings.append(
                f"Temporal data detected: train range "
                f"{str(tc.get('train_min','?'))[:10]}–{str(tc.get('train_max','?'))[:10]}."
            )

    if tbt:
        predictive = [k for k, v in tbt.items() if v.get("is_predictive")]
        if predictive:
            custom_findings.append(
                f"Predictive time features (high target mean-range): {', '.join(predictive[:5])}."
            )

    ds = data_pattern.get("distribution_shift", {})
    if ds.get("has_notable_shift"):
        shifted = [s["feature"] for s in ds.get("shifted_features", []) if s["severity"] in ("medium", "high")]
        if shifted:
            custom_findings.append(
                f"Distribution shift detected between train and prediction for: {', '.join(shifted[:4])}."
            )

    hcr = data_pattern.get("high_cardinality_risk", [])
    if hcr:
        custom_findings.append(
            f"High-cardinality columns (ID-risk) excluded from features: "
            f"{', '.join(c['feature'] for c in hcr[:3])}."
        )

    all_findings = custom_findings + (findings or [])
    if not all_findings:
        dist = profile.get("target_distribution") or {}
        if supervised and task_type != "regression" and dist.get("classes"):
            parts = ", ".join(f"{c['label']}={c['fraction']:.1%}" for c in dist["classes"][:6])
            all_findings.append(f"Target `{target}` class balance: {parts}.")
        all_findings.append(
            f"{len(profile.get('numeric_columns', {}))} numeric and "
            f"{len(profile.get('categorical_columns', {}))} categorical columns profiled."
        )

    miss = profile.get("missing_summary", {})
    miss_rows = [(c, m) for c, m in miss.items() if m.get("missing_rate", 0) > 0]
    if miss_rows:
        all_findings.append(
            f"{len(miss_rows)} columns contain missing values; highest is "
            f"{max(miss_rows, key=lambda kv: kv[1]['missing_rate'])[0]}."
        )

    for f in all_findings[:10]:
        A(f"- {f}")
    A("")


def _write_residual_section(lines: list, residual_analysis: dict, task_type: str) -> None:
    A = lines.append
    if not residual_analysis or task_type != "regression":
        A("_Residual analysis is available for regression tasks only._" if task_type != "regression"
          else "_Residual analysis was not computed for this run._")
        A("")
        return

    # By predicted quantile
    by_pred = residual_analysis.get("by_pred_quantile", [])
    if by_pred:
        A("### Residuals by Predicted-Value Quartile")
        A("")
        A("| Predicted quartile | n | Mean residual | Mean |residual| |")
        A("|---|---|---|---|")
        for row in by_pred:
            A(f"| {row.get('pred_quantile')} | {row.get('n')} "
              f"| {row.get('mean_residual', '—'):.4f} | {row.get('mean_abs_residual', '—'):.4f} |")
        A("")

    # By true-value quantile
    by_true = residual_analysis.get("by_true_quantile", [])
    if by_true:
        A("### Residuals by True-Value Quartile")
        A("")
        A("| True-value quartile | n | Mean residual | Mean |residual| |")
        A("|---|---|---|---|")
        for row in by_true:
            A(f"| {row.get('true_quantile')} | {row.get('n')} "
              f"| {row.get('mean_residual', '—'):.4f} | {row.get('mean_abs_residual', '—'):.4f} |")
        A("")

    # Heteroscedasticity
    het_corr = residual_analysis.get("heteroscedasticity_corr")
    het_note = residual_analysis.get("heteroscedasticity_note", "")
    if het_corr is not None:
        A(f"**Heteroscedasticity:** correlation of |residual| with predicted value = {het_corr:.4f}. "
          f"{het_note}.")
        A("")

    # High-value bias (uses corrected high_value_bias field; falls back to high_value_underprediction)
    hv = residual_analysis.get("high_value_bias") or residual_analysis.get("high_value_underprediction", {})
    if hv:
        bias_dir = hv.get("bias_direction", "")
        underpred_frac = hv.get("fraction_underpredicted", 0)
        overpred_frac = hv.get("fraction_overpredicted")
        mean_resid = hv.get("mean_residual_high_values", 0)
        if bias_dir == "systematic_underprediction":
            direction_text = f"systematic underprediction of high values ({underpred_frac:.1%} of top-quartile rows have y_true > y_pred)"
        elif bias_dir == "systematic_overprediction":
            direction_text = f"systematic overprediction of high values ({overpred_frac:.1%} of top-quartile rows have y_pred > y_true)"
        else:
            direction_text = f"{underpred_frac:.1%} of top-quartile rows underpredicted"
        A(f"**High-value prediction bias:** {direction_text}. "
          f"Mean residual (y_true − y_pred) in top quartile: {mean_resid:.4f}. "
          f"{hv.get('note', '')}")
        A("")


def _write_leakage_section(lines: list, leakage: dict, data_pattern: dict) -> None:
    A = lines.append
    risk = (leakage.get("leakage_risk") or "n/a").upper()
    A(f"Leakage audit: **{risk}** (approved={leakage.get('approved')}).")
    A("")

    # True leakage (FAIL-level)
    failed = leakage.get("failed_checks", [])
    warned = leakage.get("warned_checks", [])

    if failed:
        A("**True leakage findings (FAIL):**")
        A("")
        for f in failed:
            col = f.get("column") or "—"
            A(f"- {f.get('check_name')}: column `{col}` — {f.get('detail')}")
        A("")

    if warned:
        A("**Caution-level findings (WARN):**")
        A("")
        for w in warned:
            col = w.get("column") or "—"
            A(f"- {w.get('check_name')}: column `{col}` — {w.get('detail')}")
        A("")

    # Target-component columns (valid exclusion, not flagged as leakage)
    tca = data_pattern.get("target_component_analysis", {})
    comps = tca.get("high_risk_components") or [c for c in tca.get("components", []) if c.get("is_component")]
    if comps:
        A("**Target-component columns (excluded, not leakage):** "
          f"{', '.join(c['column'] for c in comps)} are absent from prediction data and "
          "were automatically excluded from the model feature set. "
          "They are not classified as leakage since the exclusion mechanism is data-driven.")
        A("")

    if not failed and not warned and not comps:
        A("No leakage findings. All features are available at prediction time.")
        A("")


def _write_validation_strategy_section(lines: list, state: Any, data_pattern: dict, metric: str) -> None:
    A = lines.append
    sm = state.split_metadata or {}
    vi = data_pattern.get("validation_implications", {})

    A(f"**Strategy:** `{sm.get('type', vi.get('recommended_strategy', '—'))}`")
    A("")

    if sm.get("type") == "within_period_holdout":
        feat = sm.get("period_feature", "sub-period unit")
        fit_dist = sm.get("fit_day_distribution", {})
        val_dist = sm.get("validation_day_distribution", {})
        pred_dist = sm.get("prediction_day_distribution", {})
        A(f"Within-period holdout on `{sm.get('time_column', '—')}` by `{feat}`.")
        A("")
        A(f"| Partition | {feat} range | n rows |")
        A("|---|---|---|")
        A(f"| Fit (model training) | {fit_dist.get('min', '—')}–{fit_dist.get('max', '—')} "
          f"| {sm.get('n_train', '—')} |")
        A(f"| Validation (holdout) | {val_dist.get('min', '—')}–{val_dist.get('max', '—')} "
          f"| {sm.get('n_holdout', '—')} |")
        A(f"| Prediction set | {pred_dist.get('min', '—')}–{pred_dist.get('max', '—')} "
          f"| — |")
        A("")
        A(f"This simulates the hidden evaluation: the holdout `{feat}` range "
          f"({val_dist.get('min', '—')}–{val_dist.get('max', '—')}) is drawn from the same "
          f"labeled training data, while the prediction set uses `{feat}` values "
          f"({pred_dist.get('min', '—')}–{pred_dist.get('max', '—')}) absent from training labels.")
    elif sm.get("type") == "time_holdout":
        tc = data_pattern.get("time_coverage", {})
        A(f"Time column: `{sm.get('time_column', '—')}`. "
          f"Train: {sm.get('n_train', '—')} rows | Holdout: {sm.get('n_holdout', '—')} rows.")
        wpp = tc.get("within_period_pattern")
        if wpp:
            A(f"\nNote: a within-period pattern was detected ({wpp.get('feature')} "
              f"train={wpp.get('train_range')} vs predict={wpp.get('predict_range')}) "
              f"but chronological time-holdout was used instead.")
        else:
            A("\nTime-based ordering ensures no future data leaks into training.")
    elif sm.get("type") in ("random_holdout", "stratified_holdout"):
        A(f"Random seed: {sm.get('random_state', 42)}. "
          f"Train: {sm.get('n_train', '—')} rows | Holdout: {sm.get('n_holdout', '—')} rows.")

    # Stability information
    stability = sm.get("stability", {})
    if stability:
        A("")
        A("**Multi-split stability:**")
        split_scores = stability.get("split_scores", [])
        if split_scores:
            scores_str = ", ".join(f"{s:.4f}" for s in split_scores)
            A(f"Split scores: {scores_str} | "
              f"Mean: {stability.get('cv_mae_mean', '—'):.4f} | "
              f"Std: {stability.get('cv_mae_std', '—'):.4f} | "
              f"Relative std: {stability.get('relative_stability', '—'):.4f}")

    recs = vi.get("recommendations", [])
    if recs:
        A("")
        A("**Validation implications:**")
        for r in recs[:3]:
            A(f"- {r}")
    A("")


# ── other helpers ─────────────────────────────────────────────────────────────

def _perf_table_md(task_type: str, evaluation: dict) -> str:
    if task_type == "regression":
        cols = ["rmse", "mae", "r2"]
    else:
        cols = ["roc_auc", "f1_weighted", "accuracy"]
    header = "| Model | Split | " + " | ".join(c.upper() for c in cols) + " |"
    sep = "|---|---|" + "|".join(["---"] * len(cols)) + "|"
    rows = [header, sep]
    for r in evaluation.get("performance_table", []):
        vals = " | ".join(
            f"{r.get(c):.4f}" if isinstance(r.get(c), (int, float)) else "—" for c in cols
        )
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
        if path and os.path.exists(str(path)):
            out.append((label, str(path), section))
    return out


def _fig_path(evaluation: dict, artifacts: dict, key: str) -> str | None:
    path = (evaluation.get("artifacts") or {}).get(key) or artifacts.get(key)
    return str(path) if path and os.path.exists(str(path)) else None


def _make_relative(path_str: str, repo_root: Path) -> str:
    try:
        return str(Path(path_str).relative_to(repo_root))
    except (ValueError, TypeError):
        return path_str


def _limitations(task, profile, evaluation, leakage, supervised, data_pattern, residual_analysis) -> list[str]:
    out: list[str] = []
    miss = profile.get("missing_summary", {})
    high_miss = [c for c, m in miss.items() if m.get("missing_rate", 0) > 0.10]
    if high_miss:
        out.append(f"Columns with >10% missing values may bias results: {', '.join(high_miss[:8])}.")
    n_rows = profile.get("n_rows", 0) or 0
    if supervised and n_rows < 1000:
        out.append(f"Sample size is modest (N={n_rows}); results may not be stable.")
    if task.get("class_imbalance_detected"):
        out.append("Class imbalance detected; accuracy alone is misleading — ROC-AUC/PR-AUC emphasized.")
    if task.get("target_inferred"):
        out.append("The target variable was inferred, not explicitly named; confirm it matches intent.")
    if leakage.get("leakage_risk") == "medium":
        out.append("The leakage audit raised WARN-level findings; review flagged columns.")
    if evaluation.get("worse_than_baseline"):
        out.append("The candidate did not clearly beat the baseline on the primary metric.")
    ds = data_pattern.get("distribution_shift", {})
    if ds.get("has_notable_shift"):
        out.append("Distribution shift detected between training and prediction features; "
                   "held-out validation score may not represent true out-of-distribution performance.")
    td = data_pattern.get("target_distribution", {})
    if td.get("is_right_skewed"):
        out.append(f"Target is right-skewed (skewness={td.get('skewness', '?'):.2f}); "
                   "MAE on raw target weights large values heavily.")
    hv = residual_analysis.get("high_value_bias") or residual_analysis.get("high_value_underprediction", {})
    if hv.get("bias_direction") == "systematic_underprediction" and hv.get("fraction_underpredicted", 0) > 0.60:
        out.append("Model systematically underpredicts high target values; "
                   "predictions for extreme values should be treated with caution.")
    elif hv.get("bias_direction") == "systematic_overprediction" and hv.get("fraction_overpredicted", 0) > 0.60:
        out.append("Model systematically overpredicts high target values; "
                   "predictions for extreme values should be treated with caution.")
    out.append("Findings are associative/predictive, not causal, and are scoped to this dataset.")
    out.append("The pipeline assumes the supplied features are available at prediction time; "
               "train-only columns were automatically excluded.")
    return out


def _warnings_addressed(task, profile) -> list[str]:
    out: list[str] = []
    for w in (task.get("warnings") or []):
        out.append(f"{w.get('type')}:{w.get('column')}")
    for w in (profile.get("warnings") or []):
        out.append(f"{w.get('type')}:{w.get('column')}")
    return out
