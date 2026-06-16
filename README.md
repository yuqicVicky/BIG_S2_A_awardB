# STAI-X Challenge 2026 — Award B Automation Agent

A self-contained, multi-agent system that turns a hidden dataset into a scored
submission and a written analysis — fully automated, from a single natural-language
instruction.

The evaluation harness drops a hidden dataset into `data/`, adds a
`DATA_DESCRIPTION.md`, opens the repo in Claude Code, and says *"Do the data
analysis."* The system runs end to end and writes two files to the repo root:

- **`submission.csv`** — `row_id,<target>` in the sample-submission row order.
- **`report.pdf`** — a standalone memo covering the data, method, validation,
  results, and limitations.

Everything — target, keys, metric, output schema — is discovered at runtime.
No dataset-specific column names, file names, or row counts are hardcoded.

---

## How it works

Two layers, designed so the deliverable is never lost:

1. **A deterministic Python floor** (`python main.py`) — a robust, reproducible
   pipeline that always produces a valid scored submission, even with no agent layer
   and no internet.
2. **An agent orchestration on top** (`CLAUDE.md`) — a team of specialist agents that
   plan, engineer features, search models, audit for leakage, write the report, and
   gate the release. It only ever *improves* on the floor; if it's cut short, the floor
   result still ships.

```bash
python main.py        # the deterministic floor, standalone
```

Agents never call each other. They communicate only through JSON logs in
`outputs/runs/{run_id}/logs/`, and reviewers never edit what they review — a doer is
re-dispatched when a reviewer asks for changes.

---

## The 8-step workflow

Orchestrated by `CLAUDE.md`; each agent lives in `.claude/agents/`.

| # | Step | Agents | Output |
|---|------|--------|--------|
| 1 | **Convert** | `data-format-converter` | any non-CSV input → CSV |
| 2 | **Task + profile** | `task-inference-agent`, `data-profiler` | task/target/metric; stats, missingness, imputation plan |
| 3 | **Pre-run setup** | `validation-and-schema-guardian`, `data-pattern-analyzer` | CV strategy + folds; feature-influence map (advisory) |
| 4 | **Feature plan** | `analysis-planner` | leakage-aware feature blueprint + `modeling_mode` |
| 5 | **Plan gate** | `plan-reviewer` | adversarial plan review (≤2 rounds) |
| 6 | **Execution** | programmer → modeling group → reviewers | features + trained models + consolidated verdict (≤2 rounds) |
| 7 | **Format + report** | `validation-and-schema-guardian`, `report-writer` | validated `submission.csv`; `report.pdf` |
| 8 | **Final review** | `report-reviewer`, `supervisor-gatekeeper` | release gate |

### Inside Step 6 (the execution loop)

- **`analysis-programmer`** builds the planned, leakage-safe features and seeds the
  deterministic floor submission.
- A **feature-ablation gate** + **`feature-engineering-reviewer`** prune feature groups
  to what actually earns its place.
- **Modeling** branches on `modeling_mode`:
  - *general* → **`model-search-agent`** (baselines first, then robust model search).
  - *specialist* → **`modeling-specialist`** ×2 (`gbdt`, `linear`) in parallel, under
    a live **`modeling-watchdog`**, combined by **`ensemble-meta`** (keep-best / NNLS
    blend, only kept when it strictly beats the best single candidate).
- **Reviewers** run independently: **`model-performance-reviewer`** always, plus
  risk-gated **`feature-leakage-reviewer`** and **`generalization-reviewer`**. The
  performance reviewer (lead) consolidates them into one verdict that drives the loop.

---

## Cross-cutting supervision

- **`modeling-watchdog`** (efficiency) — shares each specialist's progress heartbeat,
  derives a time slice from the wall-clock that remains, and kills/restarts a run
  projected to overrun. It never regresses the floor submission.
- **Budget guard** — the orchestrator checks tokens and elapsed time after every step;
  on pressure it skips straight to formatting + report so a valid deliverable always
  ships inside the 2-hour cap.
- **Closed-loop gates** — each stage emits a `pass | warn | fail` verdict; a `fail`
  triggers one bounded, targeted correction.

---

## Feature engineering: agent-directed, code-enforced

The agent decides *which* features to build for the dataset at hand, but registers them
as transformers in the **single pipeline the model actually trains on**
(`src/data_agent/features.py` + `models.py`) — never a standalone matrix the model would
ignore. The code then guarantees the invariants:

- **One consumed pipeline.** Floor and every specialist build features through
  `build_feature_bundle` — a single source of truth.
- **Leakage-safe by construction.** Per-group target statistics and group-median
  imputation are fit **inside each CV fold**, so held-out targets never reach training.
- **Code-enforced leakage guard** (`src/data_agent/leakage_guard.py`) — a
  dataset-agnostic check that fails the build if any feature is the target, a transform
  of it, or a precomputed full-data target aggregate (including disguised ones).

---

## The deterministic floor

`python main.py` → `models.train_and_predict` is the always-available result:

- **Blocked GroupKFold CV** — whole periods held out per fold, mirroring disjoint hidden
  validation periods; a near-unique time key is coarsened so CV never collapses to random
  KFold.
- **Leakage-safe per-fold group/target-aggregate features** (the dominant signal on a
  stable panel) + **TF-IDF → TruncatedSVD** for free text.
- **Early-stopped randomized tuning**, a **convex out-of-fold stack**, **seed-averaging**,
  and a **cross-family NNLS blend** — kept only when it strictly beats the best single
  candidate.
- **Incremental checkpointing** — a baseline submission is written before the heavy search
  and overwritten only when beaten, so a valid `submission.csv` always exists.
- **Monotonic post-processing** when the description declares nested categories
  (parent ≥ child), with the parent inferred from the data.

---

## Repository structure

```text
.
├── main.py                       # deterministic entry point (the floor)
├── CLAUDE.md                     # orchestrator: the 8-step workflow
├── src/data_agent/               # generic pipeline: schema, features, models,
│                                 #   CV, gates, leakage guard, reporting
├── scripts/                      # modeling runner, ablation gate, report→PDF, audits
├── tests/                        # dataset-agnostic unit tests
├── .claude/
│   ├── agents/                   # the specialist agents
│   ├── policy/                   # modeling rules, anti-hardcoding, datetime, checklist
│   └── skills/                   # reusable analysis skills (e.g. missingness audit)
├── data/                         # empty at submission; the harness populates it
├── outputs/                      # per-run logs/reports (generated, not shipped)
├── requirements.txt
└── requirements-optional.txt     # lightgbm / xgboost / catboost (optional)
```

`data/` and `outputs/` hold no committed content — the harness provides the data and the
pipeline generates the outputs.

---

## Setup & run

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt   # optional GBMs; degrades gracefully without

python main.py                             # produces submission.csv + report.pdf
```

No GPU required. If LightGBM/XGBoost/CatBoost are unavailable the pipeline falls back to
scikit-learn; if scikit-learn is unavailable it still produces a valid baseline
submission.

To reproduce locally, place a hidden-style dataset under `data/` (with a
`DATA_DESCRIPTION.md` and a sample submission) and run `python main.py`. Outputs land at
the repo root, with per-run logs and report sources under `outputs/runs/{run_id}/`.

---

## Design principles

- **Nothing is tuned to one dataset.** Target, keys, category/block dimension, and output
  schema are discovered from `DATA_DESCRIPTION.md` and the sample submission.
- **Robust selection.** Blocked GroupKFold scored by the official metric — not a single
  holdout — immune to an opaque or un-orderable time key.
- **Never regress.** Every accuracy lever (stacking, tuning, blending) is kept only when
  it strictly beats the floor; a checkpointed baseline guarantees a valid deliverable.
- **Bounded and safe.** The whole run stays inside a 2-hour cap; the leakage guard and the
  final supervisor gate inspect every artifact before release. No external data is used.
```