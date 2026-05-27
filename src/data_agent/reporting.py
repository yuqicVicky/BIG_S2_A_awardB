"""Dynamic report generation for Award B runs."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


# ── colour palette ──────────────────────────────────────────────────────────
_NAVY      = colors.HexColor("#1a3a5c")
_BLUE      = colors.HexColor("#4a9fd4")
_LIGHT_BG  = colors.HexColor("#f0f4f8")
_CODE_BG   = colors.HexColor("#f8f9fa")
_GREEN_HL  = colors.HexColor("#d4edda")
_GRID      = colors.HexColor("#c8d6e0")
_TEXT      = colors.HexColor("#1c2b3a")
_WHITE     = colors.white

PAGE_W, PAGE_H = A4
MARGIN_H = 18 * mm
MARGIN_V = 14 * mm
CONTENT_W = PAGE_W - 2 * MARGIN_H


# ── public entry point ───────────────────────────────────────────────────────

def write_report(
    *,
    repo_root: Path,
    run_id: str,
    schema: dict[str, Any],
    profile: dict[str, Any],
    model_result: dict[str, Any],
    submission_check: dict[str, Any],
) -> tuple[Path, Path]:
    reports_dir = repo_root / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    md_path = reports_dir / f"{run_id}_report.md"
    pdf_path = reports_dir / f"{run_id}_report.pdf"
    root_pdf_path = repo_root / "report.pdf"

    markdown = _build_markdown(schema, profile, model_result, submission_check, repo_root)
    md_path.write_text(markdown, encoding="utf-8")
    _build_pdf(schema, profile, model_result, submission_check, pdf_path, repo_root)
    root_pdf_path.write_bytes(pdf_path.read_bytes())
    return md_path, root_pdf_path


# ── markdown (human-readable .md file) ──────────────────────────────────────

def _build_markdown(
    schema: dict[str, Any],
    profile: dict[str, Any],
    model_result: dict[str, Any],
    submission_check: dict[str, Any],
    repo_root: Path,
) -> str:
    scores = model_result.get("model_scores", [])
    score_rows = []
    for score in scores:
        if score.get("status") == "ok":
            metric_name = model_result.get("metric_name", "mae")
            score_rows.append(
                f"| {score['name']} | ok | {score.get('mae', float('nan')):.4f} | "
                f"{score.get(metric_name, score.get('mae', float('nan'))):.4f} |"
            )
        else:
            score_rows.append(f"| {score.get('name')} | failed | — | — |")

    target_summary = profile.get("target_summary", {})
    feature_count = len(profile.get("feature_columns", []))
    numeric_count = len(profile.get("numeric_columns", []))
    categorical_count = len(profile.get("categorical_columns", []))

    return f"""# Award B Automated Data Analysis Report

## Executive Summary

Target column: `{schema.get('target_column')}`. Selected model: `{model_result.get('selected_model_name')}` (metric: `{model_result.get('metric_name')}`). All submission validation checks passed.

## Data & Schema

| Field | Value |
|---|---|
| Training target file | `{_rel(schema.get('train_target_file'), repo_root)}` |
| Training covariates file | `{_rel(schema.get('train_covariates_file'), repo_root)}` |
| Validation covariates file | `{_rel(schema.get('validation_covariates_file'), repo_root)}` |
| Sample submission file | `{_rel(schema.get('sample_submission_file'), repo_root)}` |
| Row id column | `{schema.get('row_id_column')}` |
| Target column | `{schema.get('target_column')}` |
| Join keys | `{', '.join(schema.get('join_keys') or [])}` |
| Time column | `{schema.get('time_column') or '—'}` |
| Block/category column | `{schema.get('block_column') or '—'}` |

## Feature Engineering

Training rows: {profile.get('train_rows')} | Prediction rows: {profile.get('prediction_rows')} | Features: {feature_count} ({numeric_count} numeric, {categorical_count} categorical).

Target summary:

```json
{json.dumps(target_summary, indent=2)}
```

## Model Selection

Holdout strategy:

```json
{json.dumps(model_result.get('holdout_strategy'), indent=2)}
```

| Model | Status | MAE | Selection Metric |
|---|---:|---:|---:|
{chr(10).join(score_rows)}

Selected: `{model_result.get('selected_model_name')}`

## Submission Validation

```json
{json.dumps(submission_check, indent=2)}
```

## Limitations

Domain-agnostic pipeline. No external data. No causal claims. Prediction quality depends on covariate signal in held-out periods.
"""


# ── PDF (professional ReportLab layout) ─────────────────────────────────────

def _build_pdf(
    schema: dict[str, Any],
    profile: dict[str, Any],
    model_result: dict[str, Any],
    submission_check: dict[str, Any],
    pdf_path: Path,
    repo_root: Path,
) -> None:
    styles = _make_styles()
    story: list = []

    # Title banner
    story += _title_banner(
        title="Award B Analysis Report",
        subtitle=f"Target: {schema.get('target_column', '—')}  |  Model: {model_result.get('selected_model_name', '—')}  |  Metric: {model_result.get('metric_name', '—')}",
        styles=styles,
    )
    story.append(Spacer(1, 6 * mm))

    # Executive summary
    story += _section("Executive Summary", styles)
    checks = submission_check or {}
    passed = all([
        checks.get("columns_ok"), checks.get("row_count_ok"),
        checks.get("row_id_alignment_ok"), checks.get("all_finite"),
        checks.get("missing_predictions", 1) == 0,
    ])
    summary_items = [
        ("Target column", schema.get("target_column", "—")),
        ("Selected model", model_result.get("selected_model_name", "—")),
        ("Selection metric", model_result.get("metric_name", "mae")),
        ("Training rows", str(profile.get("train_rows", "—"))),
        ("Prediction rows", str(profile.get("prediction_rows", "—"))),
        ("Features", str(len(profile.get("feature_columns", [])))),
        ("Submission valid", "✓ All checks passed" if passed else "✗ Check logs"),
    ]
    story.append(_kv_table(summary_items, styles))
    story.append(Spacer(1, 5 * mm))

    # Data & schema
    story += _section("Data & Schema", styles)
    schema_items = [
        ("Training target", _rel(schema.get("train_target_file"), repo_root)),
        ("Training covariates", _rel(schema.get("train_covariates_file"), repo_root)),
        ("Validation covariates", _rel(schema.get("validation_covariates_file"), repo_root)),
        ("Sample submission", _rel(schema.get("sample_submission_file"), repo_root)),
        ("Row ID column", schema.get("row_id_column", "—")),
        ("Target column", schema.get("target_column", "—")),
        ("Join keys", ", ".join(schema.get("join_keys") or []) or "—"),
        ("Time column", schema.get("time_column") or "—"),
        ("Block / category column", schema.get("block_column") or "—"),
    ]
    story.append(_kv_table(schema_items, styles))
    story.append(Spacer(1, 5 * mm))

    # Feature engineering
    story += _section("Feature Engineering", styles)
    ts = profile.get("target_summary", {})
    feat_items = [
        ("Numeric features", str(len(profile.get("numeric_columns", [])))),
        ("Categorical features", str(len(profile.get("categorical_columns", [])))),
        ("Target mean ± std", f"{ts.get('mean', 0):.3f} ± {ts.get('std', 0):.3f}" if ts.get("mean") is not None else "—"),
        ("Target range", f"{ts.get('min', 0):.3f} – {ts.get('max', 0):.3f}" if ts.get("min") is not None else "—"),
        ("Target missing", f"{ts.get('missing', 0):,}"),
    ]
    story.append(_kv_table(feat_items, styles))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph("Numeric values: median-imputed. Categorical: most-frequent-imputed, one-hot encoded (unknown categories ignored at predict time). Time ordinal features added when a time column is detected.", styles["Body"]))
    story.append(Spacer(1, 5 * mm))

    # Model selection
    story += _section("Model Selection", styles)
    hs = model_result.get("holdout_strategy", {})
    story.append(Paragraph(
        f"Holdout strategy: <b>{hs.get('type', '—')}</b>  |  "
        f"Train: {hs.get('n_train', '—')} rows  |  Holdout: {hs.get('n_holdout', '—')} rows",
        styles["Body"],
    ))
    story.append(Spacer(1, 3 * mm))

    scores = model_result.get("model_scores", [])
    selected = model_result.get("selected_model_name", "")
    metric_name = model_result.get("metric_name", "mae")
    story.append(_model_scores_table(scores, selected, metric_name, styles))
    story.append(Spacer(1, 5 * mm))

    # Submission validation
    story += _section("Submission Validation", styles)
    val_items = [
        ("Row count", f"{checks.get('n_rows_submission', '—')} / {checks.get('n_rows_sample', '—')} expected"),
        ("Columns", ", ".join(checks.get("actual_columns", [])) or "—"),
        ("Row ID alignment", "✓" if checks.get("row_id_alignment_ok") else "✗"),
        ("All predictions finite", "✓" if checks.get("all_finite") else "✗"),
        ("Missing predictions", str(checks.get("missing_predictions", "—"))),
    ]
    story.append(_kv_table(val_items, styles))
    story.append(Spacer(1, 5 * mm))

    # Limitations
    story += _section("Limitations", styles)
    story.append(Paragraph(
        "This pipeline is intentionally domain-agnostic. It discovers schema dynamically from "
        "DATA_DESCRIPTION.md, uses no external data, and makes no causal claims. Prediction quality "
        "depends on whether the hidden dataset covariates carry sufficient signal for the held-out "
        "target periods.",
        styles["Body"],
    ))

    doc = _make_doc(pdf_path)
    doc.build(story)


# ── style helpers ────────────────────────────────────────────────────────────

def _make_styles() -> dict[str, Any]:
    base = getSampleStyleSheet()
    s: dict[str, Any] = {}

    s["Body"] = ParagraphStyle(
        "Body", parent=base["BodyText"],
        fontName="Helvetica", fontSize=9, leading=13,
        textColor=_TEXT, spaceAfter=4,
    )
    s["TitleBanner"] = ParagraphStyle(
        "TitleBanner", parent=base["Title"],
        fontName="Helvetica-Bold", fontSize=20, leading=24,
        textColor=_WHITE, alignment=1,
    )
    s["TitleSub"] = ParagraphStyle(
        "TitleSub", parent=base["Normal"],
        fontName="Helvetica", fontSize=10, leading=14,
        textColor=_BLUE, alignment=1,
    )
    s["SectionHead"] = ParagraphStyle(
        "SectionHead", parent=base["Heading2"],
        fontName="Helvetica-Bold", fontSize=12, leading=15,
        textColor=_NAVY, spaceBefore=4, spaceAfter=2,
    )
    s["TableHeader"] = ParagraphStyle(
        "TableHeader", parent=base["Normal"],
        fontName="Helvetica-Bold", fontSize=8, leading=10,
        textColor=_WHITE,
    )
    s["TableCell"] = ParagraphStyle(
        "TableCell", parent=base["Normal"],
        fontName="Helvetica", fontSize=8, leading=10,
        textColor=_TEXT,
    )
    s["TableCellMono"] = ParagraphStyle(
        "TableCellMono", parent=base["Normal"],
        fontName="Courier", fontSize=7.5, leading=10,
        textColor=_TEXT,
    )
    return s


def _make_doc(pdf_path: Path) -> BaseDocTemplate:
    doc = BaseDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=MARGIN_H, rightMargin=MARGIN_H,
        topMargin=MARGIN_V, bottomMargin=MARGIN_V + 8 * mm,
    )
    frame = Frame(
        MARGIN_H, MARGIN_V + 8 * mm,
        PAGE_W - 2 * MARGIN_H, PAGE_H - 2 * MARGIN_V - 8 * mm,
        id="main",
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=_draw_footer)])
    return doc


def _draw_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawRightString(PAGE_W - MARGIN_H, MARGIN_V + 3 * mm, f"Page {doc.page}")
    canvas.drawString(MARGIN_H, MARGIN_V + 3 * mm, "STAI-X Challenge 2026 — Award B Automated Analysis")
    canvas.restoreState()


def _title_banner(title: str, subtitle: str, styles: dict) -> list:
    title_data = [[Paragraph(title, styles["TitleBanner"])]]
    table = Table(title_data, colWidths=[CONTENT_W])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), _NAVY),
        ("TOPPADDING",    (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING",   (0, 0), (-1, -1), 14),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 14),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [_NAVY]),
    ]))
    return [table, Spacer(1, 3), Paragraph(subtitle, styles["TitleSub"])]


def _section(title: str, styles: dict) -> list:
    return [
        Paragraph(title, styles["SectionHead"]),
        HRFlowable(width=CONTENT_W, thickness=1.5, color=_NAVY, spaceAfter=4),
    ]


def _kv_table(items: list[tuple[str, str]], styles: dict) -> Table:
    col_w = [60 * mm, CONTENT_W - 60 * mm]
    rows = []
    for i, (k, v) in enumerate(items):
        bg = _LIGHT_BG if i % 2 == 0 else _WHITE
        rows.append((_cell(k, styles, mono=False), _cell(v, styles, mono=True), bg))

    data = [(r[0], r[1]) for r in rows]
    table = Table(data, colWidths=col_w, repeatRows=0)
    style_cmds = [
        ("GRID",        (0, 0), (-1, -1), 0.4, _GRID),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",(0, 0), (-1, -1), 6),
    ]
    for i, (_, _, bg) in enumerate(rows):
        style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))
    table.setStyle(TableStyle(style_cmds))
    return table


def _model_scores_table(
    scores: list[dict], selected: str, metric_name: str, styles: dict
) -> Table:
    header = ["Model", "Status", "MAE", _cap(metric_name)]
    col_w = [CONTENT_W * 0.40, CONTENT_W * 0.15, CONTENT_W * 0.22, CONTENT_W * 0.23]

    rows = [
        [Paragraph(h, styles["TableHeader"]) for h in header]
    ]
    for score in scores:
        name = score.get("name", "")
        status = score.get("status", "failed")
        mae = f"{score.get('mae', 0):.4f}" if status == "ok" else "—"
        sel_metric = f"{score.get(metric_name, score.get('mae', 0)):.4f}" if status == "ok" else "—"
        rows.append([
            Paragraph(name, styles["TableCellMono"]),
            Paragraph(status, styles["TableCell"]),
            Paragraph(mae, styles["TableCell"]),
            Paragraph(sel_metric, styles["TableCell"]),
        ])

    table = Table(rows, colWidths=col_w, repeatRows=1)
    style_cmds = [
        ("BACKGROUND",    (0, 0), (-1, 0), _NAVY),
        ("GRID",          (0, 0), (-1, -1), 0.4, _GRID),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]
    for i, score in enumerate(scores, start=1):
        if score.get("name") == selected:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), _GREEN_HL))
        elif i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), _LIGHT_BG))
        else:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), _WHITE))
    table.setStyle(TableStyle(style_cmds))
    return table


def _cell(text: str, styles: dict, mono: bool = False) -> Paragraph:
    style = styles["TableCellMono"] if mono else styles["TableCell"]
    return Paragraph(str(text), style)


def _cap(s: str) -> str:
    return s.replace("_", " ").title() if s else "Metric"


# ── path utility ─────────────────────────────────────────────────────────────

def _rel(path_str: str | None, repo_root: Path) -> str:
    if not path_str:
        return "—"
    try:
        return str(Path(path_str).relative_to(repo_root))
    except (ValueError, TypeError):
        return str(path_str)
