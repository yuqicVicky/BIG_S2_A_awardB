"""Convert a Markdown file to PDF using reportlab Platypus."""
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph,
    Spacer, Table, TableStyle, HRFlowable, PageBreak,
)

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm

REPLACEMENTS = [
    ("—", "--"),
    ("–", "-"),
    ("‘", "'"),
    ("’", "'"),
    ("“", '"'),
    ("”", '"'),
    ("•", "-"),
    ("×", "x"),
    ("²", "2"),
    ("√", "sqrt"),
    ("≥", ">="),
    ("≤", "<="),
]


def sanitize(text: str) -> str:
    for src, dst in REPLACEMENTS:
        text = text.replace(src, dst)
    # Escape reportlab XML special chars
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text


def inline_markup(text: str) -> str:
    """Convert **bold**, *italic*, `code` to reportlab XML.

    Apply code first to protect backtick spans, then bold/italic using
    only * delimiters (not _) to avoid mangling snake_case identifiers.
    """
    # Protect inline code spans first (replace with placeholder, restore after)
    code_spans = []
    def save_code(m):
        code_spans.append(m.group(1))
        return f"\x00CODE{len(code_spans)-1}\x00"

    text = re.sub(r"`([^`]+?)`", save_code, text)

    # Links — strip URL, keep label
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    # Bold ** only (not __)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # Italic * only (not _) — require non-space neighbors
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<i>\1</i>", text)

    # Restore code spans
    def restore_code(m):
        idx = int(m.group(1))
        return f'<font name="Courier" size="9">{code_spans[idx]}</font>'
    text = re.sub(r"\x00CODE(\d+)\x00", restore_code, text)

    return text


def build_styles():
    base = getSampleStyleSheet()
    styles = {}
    styles["h1"] = ParagraphStyle(
        "h1", parent=base["Heading1"], fontSize=18, spaceAfter=6,
        textColor=colors.HexColor("#111111"),
    )
    styles["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"], fontSize=14, spaceAfter=4,
        textColor=colors.HexColor("#222222"),
    )
    styles["h3"] = ParagraphStyle(
        "h3", parent=base["Heading3"], fontSize=11, spaceAfter=3,
        textColor=colors.HexColor("#333333"),
    )
    styles["body"] = ParagraphStyle(
        "body", parent=base["Normal"], fontSize=10, leading=14,
        spaceAfter=4, textColor=colors.HexColor("#1a1a1a"),
    )
    styles["bullet"] = ParagraphStyle(
        "bullet", parent=styles["body"], leftIndent=12, firstLineIndent=0,
        bulletIndent=0, spaceAfter=2,
    )
    styles["code"] = ParagraphStyle(
        "code", parent=base["Code"], fontSize=8, leading=11,
        textColor=colors.HexColor("#333333"),
        backColor=colors.HexColor("#f5f5f5"),
        leftIndent=10, rightIndent=10, spaceAfter=4,
    )
    styles["blockquote"] = ParagraphStyle(
        "blockquote", parent=styles["body"],
        leftIndent=20, textColor=colors.HexColor("#555555"),
        backColor=colors.HexColor("#eeeeee"), italic=1,
    )
    return styles


TABLE_HEADER_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("WORDWRAP", (0, 0), (-1, -1), True),
])


def parse_markdown(md_text: str, styles):
    """Parse Markdown lines into a list of reportlab Flowable objects."""
    story = []
    lines = md_text.splitlines()
    i = 0
    usable_w = PAGE_W - 2 * MARGIN

    while i < len(lines):
        line = lines[i]

        # Fenced code block
        if line.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(sanitize(lines[i]))
                i += 1
            i += 1  # consume closing ```
            code_text = "<br/>".join(code_lines) if code_lines else "&nbsp;"
            story.append(Paragraph(code_text, styles["code"]))
            story.append(Spacer(1, 4))
            continue

        # Markdown table — collect all consecutive | lines
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            # Filter out separator rows (only -, :, |, space)
            rows = []
            for tl in table_lines:
                cells = [c.strip() for c in tl.split("|")[1:-1]]
                if all(re.match(r"^[-: ]+$", c) for c in cells):
                    continue
                rows.append(cells)
            if rows:
                # Normalize column count
                n_cols = max(len(r) for r in rows)
                norm = [r + [""] * (n_cols - len(r)) for r in rows]
                # Wrap cells in Paragraph for word-wrap
                cell_style = ParagraphStyle(
                    "tc", parent=styles["body"], fontSize=8, leading=10,
                )
                data = [
                    [Paragraph(inline_markup(sanitize(c)), cell_style) for c in row]
                    for row in norm
                ]
                col_w = usable_w / n_cols
                tbl = Table(data, colWidths=[col_w] * n_cols, repeatRows=1)
                tbl.setStyle(TABLE_HEADER_STYLE)
                story.append(tbl)
                story.append(Spacer(1, 8))
            continue

        # Horizontal rule
        if re.match(r"^[-_]{3,}$", line.strip()):
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
            story.append(Spacer(1, 6))
            i += 1
            continue

        # Headings
        if line.startswith("# "):
            story.append(Spacer(1, 10))
            story.append(Paragraph(sanitize(line[2:]), styles["h1"]))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#444444")))
            story.append(Spacer(1, 4))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Spacer(1, 8))
            story.append(Paragraph(sanitize(line[3:]), styles["h2"]))
            story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#aaaaaa")))
            story.append(Spacer(1, 3))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Spacer(1, 5))
            story.append(Paragraph(sanitize(line[4:]), styles["h3"]))
            story.append(Spacer(1, 2))
            i += 1
            continue

        # Blockquote
        if line.startswith("> "):
            story.append(Paragraph(
                inline_markup(sanitize(line[2:])), styles["blockquote"]
            ))
            i += 1
            continue

        # Bullet list
        m = re.match(r"^[-*] (.+)", line)
        if m:
            story.append(Paragraph(
                "- " + inline_markup(sanitize(m.group(1))), styles["bullet"]
            ))
            i += 1
            continue

        # Numbered list
        m = re.match(r"^\d+\. (.+)", line)
        if m:
            story.append(Paragraph(
                inline_markup(sanitize(m.group(1))), styles["bullet"]
            ))
            i += 1
            continue

        # Empty line
        if line.strip() == "":
            story.append(Spacer(1, 5))
            i += 1
            continue

        # Normal paragraph
        story.append(Paragraph(inline_markup(sanitize(line)), styles["body"]))
        i += 1

    return story


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(PAGE_W / 2, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def convert(md_path: str, pdf_path: str) -> None:
    with open(md_path, encoding="utf-8") as f:
        md_text = f.read()

    styles = build_styles()
    story = parse_markdown(md_text, styles)

    frame = Frame(MARGIN, MARGIN, PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN, id="main")
    template = PageTemplate(id="main", frames=[frame], onPage=header_footer)
    doc = BaseDocTemplate(
        pdf_path,
        pagesize=A4,
        pageTemplates=[template],
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN + 0.5 * cm,
    )
    doc.build(story)
    size_kb = os.path.getsize(pdf_path) // 1024
    print(f"PDF written: {pdf_path}  ({size_kb} KB)")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "outputs/reports/report.md"
    dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".md", ".pdf")
    convert(src, dst)
