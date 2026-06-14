"""Convert a Markdown file to PDF using reportlab Platypus."""
import os
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, PageTemplate, Paragraph,
    Spacer, Table, TableStyle, HRFlowable, PageBreak,
)

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm

# ── Visual palette ───────────────────────────────────────────────────────────
ACCENT       = colors.HexColor("#1F6F78")   # teal — headings, table header, rules
ACCENT_DARK  = colors.HexColor("#11424A")   # title + h2 text
ACCENT_LIGHT = colors.HexColor("#E7F1F2")   # callout / card background
INK          = colors.HexColor("#1A1A1A")   # body text
INK_SOFT     = colors.HexColor("#444444")   # secondary text
RULE_SOFT    = colors.HexColor("#C9D6D8")   # hairline rules
ZEBRA        = colors.HexColor("#F4F8F8")   # alternate table row

# Helvetica (WinAnsi) renders em/en dashes, smart quotes, ×, ² natively, so we keep
# those glyphs. Only substitute characters genuinely outside the base font encoding.
REPLACEMENTS = [
    ("•", "-"),
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
        "h1", parent=base["Heading1"], fontName="Helvetica-Bold",
        fontSize=21, leading=25, spaceBefore=2, spaceAfter=6, textColor=ACCENT_DARK,
    )
    styles["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"], fontName="Helvetica-Bold",
        fontSize=14, leading=18, spaceBefore=8, spaceAfter=3, textColor=ACCENT_DARK,
    )
    styles["h3"] = ParagraphStyle(
        "h3", parent=base["Heading3"], fontName="Helvetica-Bold",
        fontSize=10.5, leading=13, spaceBefore=4, spaceAfter=2, textColor=ACCENT,
    )
    styles["body"] = ParagraphStyle(
        "body", parent=base["Normal"], fontSize=10, leading=15,
        spaceAfter=5, textColor=INK,
    )
    styles["bullet"] = ParagraphStyle(
        "bullet", parent=styles["body"], leftIndent=14, firstLineIndent=0,
        bulletIndent=0, spaceAfter=3,
    )
    styles["code"] = ParagraphStyle(
        "code", parent=base["Code"], fontSize=8, leading=11,
        textColor=INK_SOFT, backColor=colors.HexColor("#F4F6F6"),
        borderColor=RULE_SOFT, borderWidth=0.5, borderPadding=6,
        leftIndent=4, rightIndent=4, spaceBefore=2, spaceAfter=6,
    )
    # Callout text style; the colored card is drawn around it in parse_markdown.
    styles["callout"] = ParagraphStyle(
        "callout", parent=styles["body"], fontSize=10.5, leading=15,
        textColor=ACCENT_DARK, spaceAfter=0,
    )
    return styles


TABLE_HEADER_STYLE = TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("LINEBELOW", (0, 0), (-1, 0), 0.6, ACCENT_DARK),
    ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE_SOFT),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ZEBRA]),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
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
            story.append(Spacer(1, 8))
            story.append(Paragraph(sanitize(line[2:]), styles["h1"]))
            story.append(HRFlowable(width="100%", thickness=2, color=ACCENT,
                                    spaceBefore=2, spaceAfter=2))
            story.append(Spacer(1, 6))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Spacer(1, 9))
            story.append(Paragraph(sanitize(line[3:]), styles["h2"]))
            story.append(HRFlowable(width="100%", thickness=0.8, color=ACCENT,
                                    spaceBefore=1, spaceAfter=2))
            story.append(Spacer(1, 3))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Spacer(1, 5))
            story.append(Paragraph(sanitize(line[4:]), styles["h3"]))
            story.append(Spacer(1, 1))
            i += 1
            continue

        # Image: ![alt](path) — embed if the file exists, else skip silently
        m = re.match(r"^!\[(.*?)\]\((.+?)\)\s*$", line)
        if m:
            alt, img_path = m.group(1), m.group(2)
            if os.path.exists(img_path):
                try:
                    iw, ih = ImageReader(img_path).getSize()
                    scale = min(1.0, (usable_w * 0.78) / iw)
                    story.append(Image(img_path, width=iw * scale, height=ih * scale))
                    if alt:
                        story.append(Paragraph(sanitize(alt), styles["body"]))
                    story.append(Spacer(1, 6))
                except Exception:
                    pass
            i += 1
            continue

        # Blockquote → callout card with a left accent bar
        if line.startswith("> "):
            quote_lines = []
            while i < len(lines) and lines[i].startswith(">"):
                quote_lines.append(lines[i].lstrip(">").strip())
                i += 1
            text = inline_markup(sanitize(" ".join(l for l in quote_lines if l)))
            para = Paragraph(text, styles["callout"])
            card = Table([["", para]], colWidths=[4, usable_w - 4])
            card.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (0, 0), ACCENT),
                ("BACKGROUND", (1, 0), (1, 0), ACCENT_LIGHT),
                ("LEFTPADDING", (1, 0), (1, 0), 10),
                ("RIGHTPADDING", (1, 0), (1, 0), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 0),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(Spacer(1, 2))
            story.append(card)
            story.append(Spacer(1, 6))
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
    # Hairline rule above the footer
    canvas.setStrokeColor(RULE_SOFT)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 1.35 * cm, PAGE_W - MARGIN, 1.35 * cm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(INK_SOFT)
    canvas.drawString(MARGIN, 1.0 * cm, "Analysis Report")
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(PAGE_W - MARGIN, 1.0 * cm, f"Page {doc.page}")
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
