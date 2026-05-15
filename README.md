# STAI-X Challenge 2026 — Award B Submission

## Pipeline Overview

This repository implements a **general-purpose AI automation pipeline** for
longitudinal panel forecasting. When invoked with the single prompt
`"Do the data analysis"`, the agent autonomously:

1. Reads `data/DATA_DESCRIPTION.md` to understand the task schema
2. Runs exploratory data analysis (correlation, time series structure)
3. Engineers temporal features and trains a gradient boosting ensemble
4. Applies recursive gap filling for multi-step-ahead forecasting
5. Writes `submission.csv` and `report.pdf` to the repo root

No human intervention is required after the initial prompt.

---

## Architecture

```
User prompt: "Do the data analysis"
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │  CLAUDE.md  (Agent Standard Operating Procedure)    │
  │                                                     │
  │  Phase 0: Parse DATA_DESCRIPTION.md                 │
  │     → infer file paths, column names, target        │
  │                                                     │
  │  Phase 1: scripts/eda.py                            │
  │     → time series plots, correlation heatmaps       │
  │     → covariate analysis, outputs/ saved            │
  │                                                     │
  │  Phase 2: scripts/forecast.py                       │
  │     → lag/trend features, walk-forward CV           │
  │     → LightGBM + XGBoost + CatBoost ensemble        │
  │     → recursive gap fill → submission.csv           │
  │                                                     │
  │  Phase 3: scripts/generate_report.py                │
  │     → collect EDA figures + model metrics           │
  │     → write report.pdf                              │
  │                                                     │
  │  Phase 4: Verification                              │
  │     → assert submission.csv + report.pdf valid      │
  └─────────────────────────────────────────────────────┘
```

### Agent Components

| Component | Implementation |
|---|---|
| Brain / LLM | Claude Sonnet 4.6 (medium effort) |
| Memory | CLAUDE.md as persistent SOP; `outputs/eda/eda_summary.json` stores EDA facts across phases |
| Planning | CLAUDE.md decomposes task into 4 sequential phases; each phase has explicit success criteria |
| Action | Bash execution of Python scripts; inline Python when scripts need repair |
| Execution | Claude Code with `--dangerously-skip-permissions`; sandboxed Linux environment |
| Observation | Script stdout/stderr inspected after each run; verification assertions in Phase 4 |

---

## Repository Structure

```
.
├── CLAUDE.md                   ← Agent SOP (read this first)
├── README.md                   ← This file
├── data/                       ← Empty at submission; organizers populate
│   └── .gitkeep
├── scripts/
│   ├── eda.py                  ← General-purpose EDA for panel data
│   ├── forecast.py             ← Walk-forward CV + ensemble forecasting
│   └── generate_report.py      ← report.pdf generation (reportlab)
├── .claude/
│   └── settings.json           ← Claude Code permissions config
└── outputs/                    ← Created at runtime (not committed)
    └── eda/
```

---

## How to Run (Reproduction)

### Setup
```bash
git clone https://github.com/[org]/[repo]
cd [repo]
pip install lightgbm xgboost catboost reportlab pandas numpy scipy scikit-learn matplotlib seaborn --break-system-packages
```

### Populate data directory
```bash
cp /path/to/competition/data/* data/
cp /path/to/DATA_DESCRIPTION.md data/
```

### Run the agent
```bash
claude --dangerously-skip-permissions
# Then type: Do the data analysis
```

### Expected outputs
- `./submission.csv` — predictions with schema `(row_id, <target_col>)`
- `./report.pdf` — data analysis report

---

## Design Principles

**Domain-agnostic**: The pipeline infers all column names and file paths from
`DATA_DESCRIPTION.md` at runtime. It makes no assumptions about the data domain.

**Robust to errors**: CLAUDE.md includes explicit fallback instructions. If a script
fails, the agent reads the traceback, diagnoses the issue, and either repairs the
script or writes an inline replacement.

**Temporal integrity**: Walk-forward cross-validation respects the time axis — no
future data leaks into training folds. Recursive gap filling handles the case where
lag features reference periods with no observed target values.

**Hierarchy-aware**: If nested categories are described (e.g., subcategory ≤ total),
predictions are clipped to enforce the constraint.
