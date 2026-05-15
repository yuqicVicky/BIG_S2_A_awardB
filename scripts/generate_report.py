"""
Generate report.pdf from EDA outputs and model metrics.
Uses reportlab (pip install reportlab).

Usage (env vars):
  TARGET_COL    target column name
  ENTITY_COL    entity column name
"""

import os, json
import numpy as np
from pathlib import Path
from datetime import datetime

TARGET_COL = os.environ.get('TARGET_COL', 'rate_per_10000_ed_visits')
ENTITY_COL = os.environ.get('ENTITY_COL', 'jurisdiction')

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, Image, PageBreak, HRFlowable)
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'reportlab',
                    '--break-system-packages', '-q'], check=True)
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                     TableStyle, Image, PageBreak, HRFlowable)
    from reportlab.lib.enums import TA_LEFT, TA_CENTER

EDA_DIR = Path('outputs/eda')
OUT_DIR  = Path('outputs')
REPORT_PATH = 'report.pdf'

# ── Load JSON summaries ───────────────────────────────────────────────────────
eda_summary = {}
model_metrics = {}

eda_path = EDA_DIR / 'eda_summary.json'
if eda_path.exists():
    with open(eda_path) as f:
        eda_summary = json.load(f)

model_path = OUT_DIR / 'model_metrics.json'
if model_path.exists():
    with open(model_path) as f:
        model_metrics = json.load(f)

# ── Styles ────────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()
DARK_BLUE  = colors.HexColor('#1B3A6B')
MID_BLUE   = colors.HexColor('#2E75B6')
LIGHT_GREY = colors.HexColor('#F2F2F2')
MID_GREY   = colors.HexColor('#D9D9D9')

title_style = ParagraphStyle('Title', parent=styles['Title'],
    fontSize=22, textColor=DARK_BLUE, spaceAfter=6)
h1_style = ParagraphStyle('H1', parent=styles['Heading1'],
    fontSize=14, textColor=DARK_BLUE, spaceBefore=16, spaceAfter=6,
    borderPad=4, leading=18)
h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
    fontSize=11, textColor=MID_BLUE, spaceBefore=10, spaceAfter=4)
body_style = ParagraphStyle('Body', parent=styles['Normal'],
    fontSize=10, leading=14, spaceAfter=6)
small_style = ParagraphStyle('Small', parent=styles['Normal'],
    fontSize=9, leading=12, textColor=colors.HexColor('#555555'))
caption_style = ParagraphStyle('Caption', parent=styles['Normal'],
    fontSize=9, leading=12, textColor=colors.HexColor('#666666'),
    alignment=TA_CENTER, spaceAfter=8)
mono_style = ParagraphStyle('Mono', parent=styles['Code'],
    fontSize=9, leading=13, backColor=LIGHT_GREY,
    borderPad=6, leftIndent=12)

def section_line():
    return HRFlowable(width='100%', thickness=1, color=MID_BLUE, spaceAfter=6)

def kv_table(rows, col_widths=(2.5*inch, 4.0*inch)):
    """Two-column key-value table."""
    data = [[Paragraph(f'<b>{k}</b>', small_style),
             Paragraph(str(v), small_style)] for k, v in rows]
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), LIGHT_GREY),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, LIGHT_GREY]),
        ('GRID', (0,0), (-1,-1), 0.5, MID_GREY),
        ('TOPPADDING',    (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING',   (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    return tbl

def img_if_exists(path, width=6.5*inch):
    if Path(path).exists():
        try:
            from PIL import Image as PILImage
            w, h = PILImage.open(path).size
            aspect = h / w
            return Image(str(path), width=width, height=width * aspect)
        except Exception:
            return Image(str(path), width=width)
    return Paragraph(f'<i>[Image not found: {path}]</i>', small_style)

# ── Build document ────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    REPORT_PATH, pagesize=letter,
    leftMargin=0.85*inch, rightMargin=0.85*inch,
    topMargin=0.9*inch, bottomMargin=0.85*inch,
)
story = []

# ── Cover ─────────────────────────────────────────────────────────────────────
story.append(Spacer(1, 0.5*inch))
story.append(Paragraph('Data Analysis Report', title_style))
story.append(Paragraph(f'Automated Pipeline — {datetime.now().strftime("%Y-%m-%d")}', small_style))
story.append(Spacer(1, 0.2*inch))
story.append(section_line())

# Quick summary table
n_entities = eda_summary.get('n_entities', 'N/A')
n_periods  = eda_summary.get('n_periods',  'N/A')
t_mean     = eda_summary.get('target_mean', 'N/A')
t_std      = eda_summary.get('target_std',  'N/A')
story.append(kv_table([
    ('Target variable',   TARGET_COL),
    ('Entity column',     ENTITY_COL),
    ('Entities',          n_entities),
    ('Training periods',  n_periods),
    ('Target mean ± std', f'{t_mean:.3f} ± {t_std:.3f}' if isinstance(t_mean, float) else 'N/A'),
    ('Target range',      f'[{eda_summary.get("target_min", "?"):.2f}, {eda_summary.get("target_max", "?"):.2f}]'
                          if isinstance(eda_summary.get("target_min"), float) else 'N/A'),
]))
story.append(Spacer(1, 0.3*inch))

# ── Section 1: Dataset Overview ───────────────────────────────────────────────
story.append(Paragraph('1. Dataset Overview', h1_style))
story.append(section_line())
story.append(Paragraph(
    f'The training dataset contains <b>{n_entities}</b> entities observed over '
    f'<b>{n_periods}</b> time periods, yielding {eda_summary.get("n_rows", "N/A")} total rows. '
    f'The forecasting target is <b>{TARGET_COL}</b>. '
    f'The global mean is {t_mean:.3f} with standard deviation {t_std:.3f}.'
    if isinstance(t_mean, float)
    else f'Dataset: {n_entities} entities, {n_periods} periods.',
    body_style))

# ── Section 2: Time Series Analysis ──────────────────────────────────────────
story.append(Paragraph('2. Time Series Analysis', h1_style))
story.append(section_line())
story.append(Paragraph(
    'The figure below shows the temporal evolution of the target variable across all entities. '
    'Entities are stratified into Low / Medium / High tiers by historical mean.',
    body_style))
story.append(img_if_exists(EDA_DIR / 'ts_by_entity.png'))
story.append(Paragraph('Figure 1. Time series by entity (left: aggregate ±1 std; right: colored by tier).', caption_style))
story.append(img_if_exists(EDA_DIR / 'global_trend.png'))
story.append(Paragraph('Figure 2. Global mean target over time.', caption_style))

# ── Section 3: Correlation Structure ─────────────────────────────────────────
story.append(Paragraph('3. Cross-Entity Correlation Structure', h1_style))
story.append(section_line())
pr_mean = eda_summary.get('pairwise_r_mean')
pr_std  = eda_summary.get('pairwise_r_std')
w_r     = eda_summary.get('within_tier_r')
b_r     = eda_summary.get('between_tier_r')
p_val   = eda_summary.get('tier_ttest_p')

story.append(Paragraph(
    f'The mean pairwise Pearson correlation across all entity pairs is '
    f'<b>r = {pr_mean:.4f}</b> (std = {pr_std:.4f}), indicating '
    f'{"moderate" if pr_mean and pr_mean > 0.3 else "weak"} synchrony in the target series.'
    if isinstance(pr_mean, float) else 'Pairwise correlation not computed.',
    body_style))

if isinstance(w_r, float) and isinstance(b_r, float):
    sig = 'significantly higher' if (p_val and p_val < 0.05) else 'not significantly different'
    story.append(Paragraph(
        f'Within-tier correlation (r = {w_r:.4f}) is <b>{sig}</b> from '
        f'between-tier correlation (r = {b_r:.4f}; Welch t-test p = {p_val:.3f}). '
        f'{"Baseline-level tier is a useful grouping variable." if p_val and p_val < 0.05 else ""}',
        body_style))

story.append(img_if_exists(EDA_DIR / 'corr_heatmap.png'))
story.append(Paragraph('Figure 3. Entity × Entity correlation heatmap.', caption_style))
story.append(img_if_exists(EDA_DIR / 'tier_corr_analysis.png'))
story.append(Paragraph('Figure 4. Within-tier vs between-tier pairwise correlation.', caption_style))

# ── Section 4: Covariate Analysis ────────────────────────────────────────────
top_cov = eda_summary.get('top_covariate')
if top_cov:
    story.append(Paragraph('4. Covariate Correlation Analysis', h1_style))
    story.append(section_line())
    story.append(Paragraph(
        f'The strongest covariate predictor is <b>{top_cov}</b> '
        f'(r = {eda_summary.get("top_covariate_r", "?"):.4f}). '
        f'All significant covariates are incorporated as model features.',
        body_style))
    story.append(img_if_exists(EDA_DIR / 'covariate_corr.png'))
    story.append(Paragraph('Figure 5. Covariate correlations with target (top features).', caption_style))
    sec_offset = 1
else:
    sec_offset = 0

# ── Section 5: Modeling Approach ──────────────────────────────────────────────
story.append(Paragraph(f'{4 + sec_offset}. Feature Engineering and Modeling', h1_style))
story.append(section_line())

n_features = model_metrics.get('n_features', 'N/A')
story.append(Paragraph('<b>Feature engineering:</b>', h2_style))
story.append(kv_table([
    ('Autoregressive (AR)',  'lag_1, lag_2, lag_3, lag_6, lag_12, lag_13'),
    ('Trend',                'trend_3m = lag_1 − lag_3; yoy_diff = lag_1 − lag_13'),
    ('Moving average (MA)',  'rolling_mean_3, rolling_mean_6, expanding_mean'),
    ('Temporal',             'months_since_start (linear trend)'),
    ('Entity statistics',    'hist_mean, hist_std, hist_median per entity'),
    ('Exogenous covariates', 'All numeric columns from covariates files'),
    ('Total features',       str(n_features)),
]))

story.append(Spacer(1, 0.15*inch))
story.append(Paragraph('<b>Model architecture:</b>', h2_style))
story.append(Paragraph(
    'An ensemble of gradient boosting models (LightGBM, XGBoost, CatBoost) is trained '
    'with walk-forward cross-validation to prevent temporal leakage. '
    'If a gap exists between the training period end and the scoring period start, '
    'it is filled via <b>recursive prediction</b>: each gap month is predicted and added '
    'to the history before computing lags for the next month.',
    body_style))

# CV results table
cv_results = model_metrics.get('cv_results', [])
if cv_results:
    story.append(Paragraph('<b>Walk-forward CV results (LightGBM):</b>', h2_style))
    lgb_cv = [r for r in cv_results if r.get('model') == 'LightGBM']
    if lgb_cv:
        header = ['Fold', 'Val period', 'MAE', 'Best iter.']
        rows_data = [header]
        for r in lgb_cv:
            rows_data.append([str(r['fold']), '—', f'{r["mae"]:.4f}', str(r.get('best_iter','—'))])
        maes = [r['mae'] for r in lgb_cv]
        rows_data.append(['Mean', '', f'{np.mean(maes):.4f} ± {np.std(maes):.4f}', ''])

        tbl = Table(rows_data, colWidths=[0.8*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), DARK_BLUE),
            ('TEXTCOLOR',  (0,0), (-1,0), colors.white),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, LIGHT_GREY]),
            ('GRID', (0,0), (-1,-1), 0.5, MID_GREY),
            ('TOPPADDING',    (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING',   (0,0), (-1,-1), 6),
            ('FONTNAME', (0,-1), (-1,-1), 'Helvetica-Bold'),
        ]))
        story.append(tbl)

# ── Section 6: Predictions ────────────────────────────────────────────────────
story.append(Paragraph(f'{5 + sec_offset}. Prediction Results', h1_style))
story.append(section_line())

import pandas as pd
if Path('submission.csv').exists():
    sub_df = pd.read_csv('submission.csv')
    target_out = [c for c in sub_df.columns if c != 'row_id'][0]
    story.append(kv_table([
        ('submission.csv rows',  str(len(sub_df))),
        ('Prediction mean',      f'{sub_df[target_out].mean():.4f}'),
        ('Prediction std',       f'{sub_df[target_out].std():.4f}'),
        ('Prediction min / max', f'{sub_df[target_out].min():.4f} / {sub_df[target_out].max():.4f}'),
        ('NaN count',            str(sub_df[target_out].isna().sum())),
    ]))
else:
    story.append(Paragraph('<i>submission.csv not found.</i>', body_style))

# ── Section 7: Methodology Summary ───────────────────────────────────────────
story.append(Paragraph(f'{6 + sec_offset}. Methodology Summary', h1_style))
story.append(section_line())
story.append(Paragraph(
    'The pipeline follows a four-phase structure: '
    '(1) <b>EDA</b> — data loading, entity correlation analysis, covariate exploration; '
    '(2) <b>Feature engineering</b> — lag/trend/rolling features derived from temporal ordering; '
    '(3) <b>Modeling</b> — gradient boosting ensemble with walk-forward CV, '
    'recursive gap fill for non-contiguous scoring windows; '
    '(4) <b>Submission</b> — predictions aligned to sample_submission.csv row order, '
    'clipped to [0, ∞) and hierarchy-constrained where applicable. '
    'The pipeline is domain-agnostic: all column names and file paths are '
    'inferred from DATA_DESCRIPTION.md at runtime.',
    body_style))

# ── Build ─────────────────────────────────────────────────────────────────────
doc.build(story)
print(f'[REPORT] report.pdf written ({Path(REPORT_PATH).stat().st_size / 1024:.1f} KB)')
