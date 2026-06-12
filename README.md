# STAI-X Challenge 2026 — Award B Automation Agent

This repository is a self-contained, multi-agent automation system for the STAI-X
Award B evaluation. The evaluation harness places a hidden dataset under `data/`,
adds `data/DATA_DESCRIPTION.md`, opens the repository in Claude Code, and issues a
single natural-language instruction to perform the analysis. The system then runs
end to end and writes two files to the repository root:

- `submission.csv` — exactly `row_id,<target_column_name>`, in the sample-submission
  row order.
- `report.pdf` — a standalone analysis describing the data, methodology, validation,
  results, and limitations.

Everything is discovered from `DATA_DESCRIPTION.md` and the sample submission at
runtime; no dataset-specific column names, file names, or row counts are hardcoded.

## How it runs

A deterministic Python pipeline is the always-available floor:

```bash
python main.py
```

On top of that floor, a top-level orchestrator (`CLAUDE.md`) dispatches a team of
specialist agents that plan, build features, search models, audit for leakage and
hardcoding, write the report, and gate the final release. The deterministic engine
guarantees a valid scored submission even if the agent layer is cut short.

## Workflow

The orchestrator in `CLAUDE.md` runs an eight-step workflow; each step is a
specialist agent under `.claude/agents/`. Agents never call each other — they
communicate only through JSON logs in `outputs/logs/`.

1. **data-format-converter** — converts any non-CSV inputs to CSV.
2. **data-profiler** (+ missingness skill) — descriptive stats, missingness
   mechanism, and an imputation plan.
3. **task-inference-agent** — resolves task type, target, keys, block dimension,
   and metric from `DATA_DESCRIPTION.md`.
4. **analysis-planner** — produces a modeling plan and self-critiques it for
   leakage, validation, and hardcoding.
5. **plan-reviewer** — adversarial plan review (bounded loop, accept on pass).
6. **analysis-programmer + modeling group** — the programmer implements the planned
   features *inside the model's feature engine*; the `modeling-specialist` (one
   parametrized agent, run once per `gbdt`/`trees`/`linear` family) searches models in
   parallel; `ensemble-meta`
   keeps the best candidate or a leakage-safe blend. A `hardcoding-and-feature-auditor`
   and a `results-reviewer` audit and steer each round.
7. **report-writer-reviewer + validation-and-schema-guardian** — generate and
   self-review `report.pdf`; validate the submission schema, row order, and value
   ranges.
8. **supervisor-gatekeeper** — final release gate: inspects every artifact, confirms
   the submission and report, and triggers a bounded targeted repair if needed.

## Cross-cutting supervision

Two supervisors run alongside the workflow, plus a closed-loop gate at every stage:

- **modeling-watchdog** (efficiency) — shares each modeling specialist's progress
  heartbeat, derives a time slice from the wall-clock that remains, and kills/restarts
  a run projected to overrun, so a round never blows the budget. It never regresses
  the deterministic floor submission.
- **optimizer** (quality) — an independent critic at the step boundaries the
  deterministic gates do not cover, plus one bounded high-value redo. It grounds every
  judgment in the artifact the model actually consumes.
- **closed-loop gates** (`src/data_agent/gates.py`) — each stage emits a verdict
  (`pass | warn | fail`); a `fail` triggers one bounded, targeted correction. An LLM
  critic's verdict for a stage overrides the deterministic one.

## Feature engineering: agent-directed, code-enforced

Feature engineering is **agent-directed but code-enforced**. The agent decides *which*
features to build for the dataset at hand, but it registers them as transformers in
the single feature pipeline the model actually trains on (`src/data_agent/features.py`
+ `models.py`) — never as a standalone matrix that the model would ignore. This keeps
the flexibility of an agent while the code guarantees the invariants:

- **One consumed pipeline.** Every model (the floor and the parallel specialists)
  builds features through `build_feature_bundle`, so there is a single source of truth.
- **Leakage-safe by construction.** Per-group target statistics and group-median
  imputation are fit **inside each cross-validation fold**, so held-out target values
  never enter a training fold.
- **Code-enforced leakage guard** (`src/data_agent/leakage_guard.py`). A
  dataset-agnostic check runs at the feature chokepoint and fails the build if any
  static model feature is the target, a transform of it, or a precomputed full-data
  target aggregate — including disguised aggregates that name and correlation checks
  would miss. Its verdict becomes the deterministic `leakage` gate.
- **Registration seam** (`build_head_transformers` in an optional `custom_features`
  module). Agent-authored, per-run transformers are inserted into the consumed
  per-fold pipeline automatically — flexible *and* leakage-safe, with no standalone
  matrix.

## Heavy-compute floor + parallel modeling group

The deterministic engine (`python main.py` → `models.train_and_predict`) is the floor —
a robust, reproducible result that always exists:

- **Blocked GroupKFold cross-validation** — whole periods are held out per fold,
  mirroring the disjoint hidden validation periods. A near unique-per-row period key
  (e.g. an hourly timestamp) is coarsened to whole-period blocks so the CV never
  silently collapses into random KFold.
- **Leakage-safe group/target-aggregate features** fit inside each fold (the dominant
  signal on a stable panel), plus **TF-IDF → TruncatedSVD** features for free text.
- **Early-stopped randomized tuning**, a **convex out-of-fold stack**, **seed-averaging**
  of the final fit, and a **cross-family convex (NNLS) blend** of the floor and the
  modeling-group specialists — kept only when it strictly beats the best single
  candidate, so it never regresses.
- **Incremental checkpointing** — a baseline submission is written before the heavy
  search and overwritten only when beaten, so a valid scored `submission.csv` always
  exists even if the run is cut off.
- **Monotonic post-processing** — when the description declares nested/total categories,
  a parent ≥ child ordering is enforced, with the parent inferred from the data.

When budget remains, the orchestrator dispatches the modeling group as real subagents
(`scripts/run_modeling_agent.py --approach <family>`) with a reserved per-family
wall-clock slice; `ensemble-meta` keep-bests by the cross-validated official metric.
The in-process Python path remains the always-on safety net.

## Agent design and architecture

| Component | What it is here |
|---|---|
| **Brain / LLM** | Claude drives the top-level orchestrator and every specialist (perception, planning, the modeling group, audit, supervision). |
| **Memory** | `outputs/logs/{run_id}_*.json` (schema, profile, plan, gates, candidates) + the shared verdict schema as the inter-agent contract. |
| **Planning** | `analysis-planner` decomposes the task and self-critiques for leakage/validation; the orchestrator assigns the modeling group's division of labor. |
| **Action** | Tested Python skills in `src/data_agent/` (schema discovery, the feature engine, the CV/stacking/tuning engine, submission, report) invoked via Bash. |
| **Execution** | A deterministic CPU-only floor (`python main.py`, bounded by `AWARDB_TIME_BUDGET_SEC` < 2 h) plus a bounded, additive agent team. |
| **Observation** | Closed-loop gates plus `validation-and-schema-guardian`, `hardcoding-and-feature-auditor`, and `supervisor-gatekeeper` inspect every artifact and keep-best the submission. |

## Repository structure

```text
.
├── main.py                         # deterministic entry point (the floor)
├── src/data_agent/                 # generic pipeline: schema, features, models, gates, leakage guard
├── scripts/
│   ├── run_modeling_agent.py       # modeling-group specialist runner (family-filtered)
│   └── md_to_pdf.py                # report Markdown → PDF
├── tests/                          # dataset-agnostic unit tests
├── data/                           # empty at submission; the harness populates it
├── outputs/                        # run artifacts/logs/reports (generated, not shipped)
├── CLAUDE.md                       # top-level orchestrator: the 8-step workflow
├── .claude/
│   ├── agents/                     # specialist agents (core pipeline + modeling group + supervisors)
│   ├── policy/                     # modeling rules, anti-hardcoding, datetime, inspection checklist
│   └── skills/                     # reusable analysis skills (e.g. missingness audit)
├── requirements.txt
└── requirements-optional.txt       # lightgbm / xgboost / catboost (optional)
```

`data/` and `outputs/` hold no committed content — the harness provides the data and
the pipeline generates the outputs.

## Setup

```bash
pip install -r requirements.txt
```

For the optional gradient-boosting candidates (LightGBM / XGBoost / CatBoost):

```bash
pip install -r requirements-optional.txt
```

If these are unavailable the pipeline falls back to scikit-learn models; if
scikit-learn is unavailable it still produces a valid submission from baseline mean
models. No GPU is required.

## Local reproduction

Place a hidden-style dataset under `data/`:

```text
data/
├── DATA_DESCRIPTION.md
├── ... training target table ...
├── ... training covariates table ...
├── ... validation covariates table ...
└── ... sample submission table ...
```

Then run:

```bash
python main.py
```

Expected outputs: `submission.csv` and `report.pdf` at the repository root, plus
per-run logs and report sources under `outputs/`.

## Design notes

- Nothing is tuned to a single dataset. The target, keys, category/block dimension, and
  output schema are discovered from `DATA_DESCRIPTION.md` and the sample submission;
  row counts and period maps are never hardcoded.
- Selection uses **blocked GroupKFold cross-validation** scored by the official metric
  (e.g. block-averaged MAE), not a single holdout — robust on a panel and immune to an
  opaque or un-orderable time key.
- **Leakage-safe per-fold group/target-aggregate features** are the dominant signal;
  free-text columns get TF-IDF→SVD features; a convex OOF stack, early-stopped tuning,
  seed-averaging, and a cross-family NNLS blend push accuracy without ever regressing
  the floor.
- Baselines are evaluated before candidates; the candidate pool is task-driven, with
  optional LightGBM/XGBoost/CatBoost that degrade gracefully to scikit-learn.
- A baseline submission is checkpointed before the heavy search and overwritten only
  when beaten; the whole search is bounded to stay inside the 2-hour cap.
- Monotonic/hierarchy constraints are enforced only when the description implies them,
  with the parent category inferred from the data. No external data is used.
```
