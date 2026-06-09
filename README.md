# STAI-X Challenge 2026 — Award B Automation Agent

This repository contains a Claude Code automation pipeline for the STAI-X
Award B evaluation. The evaluation harness places a hidden dataset under
`data/`, adds `data/DATA_DESCRIPTION.md`, opens this repository in Claude Code,
and issues one prompt:

```text
Do the data analysis
```

The agent should then run the deterministic Python entry point:

```bash
python main.py
```

The run writes two required files to the repository root:

- `submission.csv` with exactly `row_id,<target_column_name>`
- `report.pdf` describing the data, modeling procedure, metrics, and checks

## How The Pipeline Works

The pipeline runs a 14-phase hub-and-spoke agent workflow controlled by the
`analysis-orchestrator`. Each phase is handled by a specialist agent:

1. **task-inference-agent** — parses `DATA_DESCRIPTION.md`; writes `spec_parse.json`.
2. **validation-and-schema-guardian** — chooses validation strategy; writes `validation_strategy.json`.
3. **hardcoding-and-feature-auditor** — pre-run audit; writes `hardcoding_audit_pre.json`.
4. **data-profiler** — profiles all data files; writes `data_profile.json`.
5. **analysis-planner** — creates and self-critiques the modeling plan; writes `analysis_plan.json`.
6. **analysis-programmer** — runs `python main.py`; writes `submission.csv`.
7. **model-search-agent** — trains baselines and candidates; selects best; writes `model_search.json` and `final_model.json`.
8. **validation-and-schema-guardian** — validates `submission.csv`; writes `submission_validation.json`.
9. **hardcoding-and-feature-auditor** — feature engineering audit; writes `feature_audit_review.json`.
10. **supervisor-gatekeeper** — inspects all logs; decides if repair is needed; writes `supervisor_gatekeeper.json`.
11. *(Conditional repair rerun — at most once)*
12. **hardcoding-and-feature-auditor** — post-run audit; writes `hardcoding_audit_post.json`.
13. **report-writer-reviewer** — generates and self-reviews `report.pdf`; writes `report_review.json`.
14. **supervisor-gatekeeper** — final gate confirmation.

All agents communicate through the orchestrator (hub-and-spoke). No agent calls
another agent directly.

## Heavy-Compute Floor + Parallel Modeling Group

The deterministic engine (`python main.py` → `models.train_and_predict`) is the
**floor** — a robust, reproducible result that always exists:

- **Blocked GroupKFold cross-validation** — whole periods are held out per fold,
  mirroring the disjoint hidden validation periods. A near unique-per-row period
  key (e.g. an hourly timestamp) is **coarsened to whole-period blocks** by
  `_resolve_cv_groups` so the CV never silently collapses into random KFold.
- **Leakage-safe group/target-aggregate features** — per-group target
  mean/median/std/quantiles fit *inside* each CV fold (the dominant signal on a
  stable panel; this is the generic form of the hand-crafted `hist_*` features).
- **Free-text features** — TF-IDF → TruncatedSVD on detected free-text columns.
- **Early-stopped randomized hyperparameter tuning**, a **convex out-of-fold
  stack**, **seed-averaging** of the final fit, and a **cross-family convex
  (NNLS) blend** of the floor + modeling-group specialists (kept only when it
  strictly beats the best single — never regresses). Tuning is leaner by default
  because wall-clock, not tokens, is the binding Award-B cap.
- **Incremental checkpointing** — a baseline submission is written *before* the
  heavy search, then overwritten, so a valid scored `submission.csv` always
  exists even if the run is cut off ("partial output is scored").
- **Monotonic post-processing** — when the description declares nested/total
  categories, a parent ≥ child ordering is enforced (parent inferred from the
  data, never hardcoded).

When wall-clock/token budget remains, the orchestrator dispatches a **parallel
modeling group** with a clear division of labor, **diverse-first**:
`linear-encoding-specialist`, `trees-specialist`, then `gbdt-specialist` (which
overlaps the floor's GBDT stack most, so it runs last and is first to be skipped
under budget) — each running `scripts/run_modeling_agent.py --approach <family>`
with a reserved per-family wall-clock slice. `ensemble-meta` picks the best
candidate or a convex NNLS blend by the cross-validated official metric, and
`supervisor-gatekeeper` overwrites `submission.csv` **only if it strictly beats the
current one** (keep-best). For Award B this group also runs as **real subagents**
(Step 2 in `CLAUDE.md`) so the multi-agent architecture genuinely executes — the
in-process Python path remains the always-on safety net.

## Agent Design and Architecture

| Component | What it is here |
|---|---|
| **Brain / LLM** | Claude Sonnet 4.6 drives `analysis-orchestrator` + the specialist agents (perception, planning, the modeling group, audit, supervisor). |
| **Memory** | `outputs/logs/{run_id}_*.json` (schema, profile, model selection, gates, candidates) + `{run_id}_state.json`; the shared verdict schema is the inter-agent contract. |
| **Planning** | `analysis-planner` decomposes the task and self-critiques for leakage/validation; the orchestrator assigns the modeling group's division of labor. |
| **Action** | Tested Python skills in `src/data_agent/` (schema discovery, feature bundle, CV/stacking/tuning engine, submission, report) invoked via Bash. |
| **Execution** | A deterministic floor (`python main.py`, CPU-only, bounded by `AWARDB_TIME_BUDGET_SEC` < 2 h) plus a bounded, additive agent team. |
| **Observation** | Closed-loop gates (`gates.py`) + `validation-and-schema-guardian`, `hardcoding-and-feature-auditor`, and `supervisor-gatekeeper` inspect every artifact and keep-best the submission. |

## Repository Structure

```text
.
├── main.py                        # deterministic entry point (the floor)
├── src/data_agent/                # generic Award B pipeline implementation
├── scripts/run_modeling_agent.py  # modeling-group specialist runner (family-filtered)
├── data/                          # empty at submission; organizers populate
├── outputs/                       # runtime artifacts/logs/reports
├── CLAUDE.md                      # Claude Code operating instructions
├── .claude/agents/                # specialist agents (core pipeline + modeling group)
│   ├── orchestrator.md            # hub controller — workflow + modeling group
│   ├── task_inference.md          # schema parsing from DATA_DESCRIPTION.md
│   ├── data_profiler.md           # data quality profiling
│   ├── planner.md                 # plan generation + self-critique
│   ├── programmer.md              # pipeline execution
│   ├── model_search_agent.md      # baseline + candidate model search
│   ├── validation_schema_guardian.md  # validation strategy + submission check
│   ├── hardcoding_feature_auditor.md  # anti-hardcoding + feature audit
│   ├── report_writer_reviewer.md  # report generation + self-review
│   ├── supervisor_gatekeeper.md   # final release gate + keep-best
│   ├── gbdt_specialist.md         # modeling group: gradient-boosted trees
│   ├── linear_encoding_specialist.md  # modeling group: linear + encodings
│   ├── trees_specialist.md        # modeling group: bagged trees
│   └── ensemble_meta.md           # modeling group: best/blend + hand-off
├── requirements.txt
└── requirements-optional.txt      # lightgbm / xgboost / catboost (optional)
```

## Setup

```bash
pip install -r requirements.txt
```

For the optional gradient-boosting candidates (LightGBM / XGBoost / CatBoost):

```bash
pip install -r requirements-optional.txt
```

If these are unavailable, the pipeline falls back to scikit-learn models. If
scikit-learn is unavailable, it still produces a valid submission using baseline
mean models. No GPU is required.

## Local Reproduction

Place a hidden-style dataset in `data/`:

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

Expected outputs:

```text
submission.csv
report.pdf
outputs/logs/<run_id>_*.json
outputs/reports/<run_id>_report.md
outputs/reports/<run_id>_report.pdf
```

## Design Notes

- The implementation does not assume overdose-specific column names; nothing is
  tuned to a single dataset. The target, keys, category/block dimension, and
  output schema are all discovered from `DATA_DESCRIPTION.md` + the sample
  submission. Row counts and period maps are never hardcoded.
- Selection uses **blocked GroupKFold cross-validation** scored by
  **block-averaged MAE** (the Award-B metric), not a single holdout — far more
  robust on a panel and immune to an opaque/un-orderable time key.
- **Leakage-safe group/target-aggregate features** (fit per CV fold) are the
  dominant signal; **free-text** columns get TF-IDF→SVD features; a **convex OOF
  stack**, **early-stopped tuning**, **seed-averaging**, and a **cross-family NNLS
  blend** of the floor + specialists push accuracy (the blend never regresses the
  floor — it is kept only when it strictly wins on CV).
- Baselines are evaluated before candidates; the candidate pool is task-driven
  (regressors for regression), with optional LightGBM/XGBoost/CatBoost that
  degrade gracefully to scikit-learn when absent (no GPU required).
- A baseline submission is **checkpointed before** the heavy search and only
  overwritten when beaten (so partial output is always scored); the whole search
  is bounded by `AWARDB_TIME_BUDGET_SEC` to stay inside the 2-hour cap.
- Monotonic/hierarchy constraints are enforced only when the description implies
  them, with the parent category inferred from the data. No external data is used.
