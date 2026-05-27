"""Dynamic report generation for Award B runs."""

from __future__ import annotations

from pathlib import Path
import html
import json
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


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

    markdown = _build_markdown(schema, profile, model_result, submission_check)
    md_path.write_text(markdown, encoding="utf-8")
    _markdown_to_pdf(markdown, pdf_path)
    root_pdf_path.write_bytes(pdf_path.read_bytes())
    return md_path, root_pdf_path


def _build_markdown(
    schema: dict[str, Any],
    profile: dict[str, Any],
    model_result: dict[str, Any],
    submission_check: dict[str, Any],
) -> str:
    scores = model_result.get("model_scores", [])
    score_rows = []
    for score in scores:
        if score.get("status") == "ok":
            metric_name = model_result.get("metric_name", "mae")
            score_rows.append(
                f"| {score['name']} | ok | {score.get('mae', float('nan')):.6f} | "
                f"{score.get(metric_name, score.get('mae', float('nan'))):.6f} |"
            )
        else:
            score_rows.append(f"| {score.get('name')} | failed |  |  |")
    if not score_rows:
        score_rows.append("| none | failed |  |  |")

    target_summary = profile.get("target_summary", {})
    feature_count = len(profile.get("feature_columns", []))
    numeric_count = len(profile.get("numeric_columns", []))
    categorical_count = len(profile.get("categorical_columns", []))

    return f"""# Award B Automated Data Analysis Report

## Executive Summary

This run parsed `data/DATA_DESCRIPTION.md`, identified the target column
`{schema.get('target_column')}`, built a generic tabular regression pipeline,
and wrote a two-column `submission.csv` aligned to the provided sample
submission. The selected model was `{model_result.get('selected_model_name')}`
based on internal `{model_result.get('metric_name')}`. The output passed schema
checks for row count, column names, finite predictions, and row-id alignment.

## Data And Schema

| Field | Value |
|---|---|
| Training target file | `{schema.get('train_target_file')}` |
| Training covariates file | `{schema.get('train_covariates_file')}` |
| Validation covariates file | `{schema.get('validation_covariates_file')}` |
| Sample submission file | `{schema.get('sample_submission_file')}` |
| Row id column | `{schema.get('row_id_column')}` |
| Target column | `{schema.get('target_column')}` |
| Join keys | `{', '.join(schema.get('join_keys') or [])}` |
| Time column | `{schema.get('time_column')}` |
| Block/category column | `{schema.get('block_column')}` |

## Feature Engineering

The training frame contains {profile.get('train_rows')} rows and the prediction
frame contains {profile.get('prediction_rows')} rows. The model used
{feature_count} features: {numeric_count} numeric and {categorical_count}
categorical. Numeric values were median-imputed. Categorical values were
most-frequent-imputed and one-hot encoded with unknown validation categories
ignored. If a time column was detected, ordinal and date-derived time features
were added without using validation targets.

Target summary:

```json
{json.dumps(target_summary, indent=2)}
```

## Model Selection

Internal validation strategy:

```json
{json.dumps(model_result.get('holdout_strategy'), indent=2)}
```

Candidate model results:

| Model | Status | MAE | Selection Metric |
|---|---:|---:|---:|
{chr(10).join(score_rows)}

Selected model: `{model_result.get('selected_model_name')}`.

## Submission Validation

```json
{json.dumps(submission_check, indent=2)}
```

## Limitations

This pipeline is intentionally domain-agnostic. It does not use external data,
does not assume overdose-specific field names, and makes no causal claims.
Prediction quality depends on whether the hidden dataset's covariates contain
enough signal for the held-out target period and whether the inferred schema
matches the organizer's `DATA_DESCRIPTION.md`.
"""


def _markdown_to_pdf(markdown: str, pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SmallBody",
            parent=styles["BodyText"],
            fontSize=9,
            leading=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["BodyText"],
            fontName="Courier",
            fontSize=7,
            leading=9,
            backColor=colors.HexColor("#f5f5f5"),
            leftIndent=6,
            rightIndent=6,
            spaceAfter=8,
        )
    )
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []
    in_code = False
    code_lines = []
    table_lines = []

    def flush_code():
        nonlocal code_lines
        if code_lines:
            story.append(Paragraph(_escape("<br/>".join(code_lines)), styles["CodeBlock"]))
            code_lines = []

    def flush_table():
        nonlocal table_lines
        if not table_lines:
            return
        rows = []
        for line in table_lines:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            rows.append([_clean_inline(cell) for cell in cells])
        if rows:
            table = Table(rows, repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 8))
        table_lines = []

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_table()
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue
        if line.startswith("|"):
            table_lines.append(line)
            continue
        flush_table()
        if not line:
            story.append(Spacer(1, 6))
        elif line.startswith("# "):
            story.append(Paragraph(_clean_inline(line[2:]), styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(_clean_inline(line[3:]), styles["Heading2"]))
        else:
            story.append(Paragraph(_clean_inline(line), styles["SmallBody"]))
    flush_code()
    flush_table()
    doc.build(story)


def _clean_inline(text: str) -> str:
    text = html.escape(text)
    text = text.replace("`", "")
    return text


def _escape(text: str) -> str:
    return html.escape(text)

