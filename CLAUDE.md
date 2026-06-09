# CLAUDE.md — STAI-X Award B Automation Agent

## Primary Instruction

When the user prompt is:

```text
Do the data analysis
```

**Run the deterministic pipeline, then dispatch the real agent layer.** The Python
pipeline always produces a complete, principled `submission.csv` + `report.pdf` by itself;
the Claude subagents then independently verify and refine it under a keep-best gate that
**can never regress the deliverable**. Wall-clock (≈2h) is the binding cap and is spent on
the Python compute; the 1M-token budget is spent almost entirely by the subagent layer
(`python main.py` itself spends **0 tokens**).

### Step 1 — run the deterministic pipeline (always)

```bash
python main.py
```

If Python reports missing packages, install dependencies and rerun:

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt   # lightgbm / xgboost / catboost (optional, graceful fallback)
python main.py
```

`main.py` runs a heavy-compute pipeline (**blocked GroupKFold cross-validation that holds
whole periods out** — a too-granular datetime key is coarsened to period blocks so the CV
never collapses to random KFold, leakage-safe group/target-aggregate + TF-IDF features,
**early-stopped** randomized tuning, an OOF stack, seed-averaging, and a cross-family
convex **blend**) and **always writes a valid `submission.csv` (+ a pre-search baseline
checkpoint) and `report.pdf`**. It runs the in-process modeling group + blend itself,
keeping the best by cross-validated official metric (`block_mae` for the regression panel;
`accuracy`/etc. for the classification fallback). **This alone is a complete, robust,
principled submission and the safety net for everything below.** Writes
`outputs/logs/{run_id}_ensemble_meta.json`, `{run_id}_model_selection.json`, and the
per-stage deterministic gate verdicts. Disable the in-process group with
`AWARDB_MODELING_GROUP=0`.

### Step 2 — dispatch the real modeling group + critic gates (when budget remains)

After `main.py` finishes, the floor + report already exist, so the run is safe no matter
what happens next. Using the ample remaining budget, dispatch the agent layer **as real
subagents via the Task tool**, in this order. Every step is **keep-best and idempotent**:
a failed, skipped, or worse subagent **leaves the current `submission.csv` untouched**.

1. **Parallel modeling group — diverse-first.** Dispatch the three specialists in parallel
   (gbdt overlaps the floor's GBDT stack most, so it is the most expendable under budget):
   `linear-encoding-specialist`, `trees-specialist`, `gbdt-specialist`. Each runs
   `python scripts/run_modeling_agent.py --approach <linear|trees|gbdt> --run-id <run_id>`
   and writes `outputs/logs/{run_id}_agent_<approach>.json` +
   `outputs/logs/{run_id}_cand_<approach>.csv`. This is genuine independent exploration
   that confirms or challenges the in-process result.
2. **`ensemble-meta`.** Reads every `{run_id}_agent_*.json` plus the floor's
   `{run_id}_model_selection.json` and the in-process `{run_id}_ensemble_meta.json`,
   compares candidates + the blend by the cross-validated official metric, and records the
   best choice. Never selects anything worse than the current submission.
3. **`supervisor-gatekeeper`.** Final keep-best gate: overwrite repo-root `submission.csv`
   **only if** the chosen candidate strictly beats the current one by CV. Writes
   `outputs/logs/supervisor_gatekeeper.json` (+ `{run_id}_llm_gate_supervisor.json`).
4. **Adversarial critic gates.** Dispatch critics for `leakage`, `prediction_sanity`, and
   `report`; each inspects the logs and writes `outputs/logs/{run_id}_llm_gate_{stage}.json`
   in the shared verdict schema. An LLM verdict **takes precedence** over the deterministic
   one (`gates.load_llm_verdict` / `orchestrator._emit_gate`).

A bare `python main.py` with no Claude layer is already complete (the Python pipeline does
all of the above in-process); Step 2 makes the multi-agent architecture **actually
execute** as real subagents for Award B and can only improve the deliverable. See
`.claude/agents/orchestrator.md` → "Parallel modeling group".

Do not manually craft a submission. Do not rely on any overdose-specific field
names. The only authority for the hidden evaluation dataset is
`data/DATA_DESCRIPTION.md` plus the files placed under `data/`.

### Runtime knobs (env vars, all optional with safe defaults)

| Var | Default | Effect |
|---|---|---|
| `AWARDB_TIME_BUDGET_SEC` | 5400 | wall-clock budget for floor + group search (kept < 2h cap) |
| `AWARDB_TUNE_ITER` | 12 | randomized-search iterations per tuned family, **early-stopped** on plateau (0 disables tuning) |
| `AWARDB_SEEDS` | 3 | seed-averaging count for the final fit |
| `AWARDB_MODELING_GROUP` | 1 | auto-run Step 2 modeling group when budget remains (`0` disables) |
| `AWARDB_GROUP_TUNE_ITER` | max(`AWARDB_TUNE_ITER`, 24) | per-specialist tuning iterations |

## Required Outputs

The run must produce both files in the repository root:

- `submission.csv`
- `report.pdf`

`submission.csv` must contain exactly two columns:

```text
<row_id_column>,<target_column>
```

Both column names come from the sample submission / `DATA_DESCRIPTION.md` (e.g.
`PassengerId,Survived` for Titanic). The output must preserve the sample submission's
row order, and values must match the required output format (integer class labels,
string labels, or continuous/probability values).

## Workflow Contract

`main.py` drives `orchestrator.run_orchestrated_analysis`, which threads an
`AnalysisState` through the staged agent pipeline below — each `.claude/agents/` agent
maps to a phase and is backed by a real skill in `src/data_agent/skills/`. On any stage
failure it falls back to the deterministic `runner.run_analysis`. The pipeline
performs these 14 phases via a hub-and-spoke architecture controlled by
`analysis-orchestrator`:

| Phase | Agent | Log file |
|-------|-------|----------|
| 1 | `task-inference-agent` | `spec_parse.json` |
| 2 | `validation-and-schema-guardian` (schema review) | `validation_strategy.json` |
| 3 | `hardcoding-and-feature-auditor` (pre-run) | `hardcoding_audit_pre.json` |
| 4 | `data-profiler` | `data_profile.json` |
| 5 | `analysis-planner` | `analysis_plan.json` |
| 6 | `analysis-programmer` (first pass) | `submission.csv` |
| 7 | `model-search-agent` | `model_search.json`, `final_model.json` |
| 8 | `validation-and-schema-guardian` (validation) | `submission_validation.json` |
| 9 | `hardcoding-and-feature-auditor` (feature audit + overfitting audit) | `feature_audit_review.json`, `overfitting_leakage_audit.json` |
| 10 | `supervisor-gatekeeper` (first review) | `supervisor_gatekeeper.json`, `prediction_sanity.json` |
| 11 | Conditional repair rerun (programmer → model-search → guardian) | — |
| 12 | `hardcoding-and-feature-auditor` (post-run) | `hardcoding_audit_post.json` |
| 13 | `report-writer-reviewer` | `report.pdf`, `report_review.json` |
| 14 | `supervisor-gatekeeper` (final gate) | `supervisor_gatekeeper.json` |

Agents communicate only through the orchestrator (hub-and-spoke). No agent calls
another agent directly. All inter-agent data passes through JSON log files.

**Deterministic floor (what Phase 6–7 actually run).** The engine
`models.train_and_predict` performs: blocked **GroupKFold cross-validation** (whole
periods held out, mirroring the disjoint validation periods — a **too-granular datetime
key is coarsened to whole-period blocks** by `_resolve_cv_groups` so the CV never collapses
to random KFold), **leakage-safe group/target-aggregate features** (per-group target
mean/median/std/quantiles fit inside the CV, the dominant panel signal), **free-text
TF-IDF→SVD** features, **early-stopped randomized hyperparameter tuning** of the top
families, a **convex OOF stack**, and **seed-averaging**. It writes a pre-search **baseline
checkpoint** submission first, then overwrites with the full result — so a valid, scored
`submission.csv` always exists ("partial output is scored").

**Parallel modeling group (Phase 7 enhancement).** Runs in-process by default and, for
Award B, **also** as real subagents (Step 2 of the Primary Instruction). When budget
remains the orchestrator dispatches family specialists **diverse-first**
(`linear-encoding-specialist`, `trees-specialist`, `gbdt-specialist` — gbdt overlaps the
floor most, so it runs last) in parallel via `scripts/run_modeling_agent.py --approach <X>`,
then `ensemble-meta` (best **or convex NNLS blend** by CV official metric) and
`supervisor-gatekeeper` (overwrite `submission.csv` only if it strictly beats the current
one — keep-best). The in-process group reserves a per-family wall-clock slice so the
diverse families are never starved. See `.claude/agents/orchestrator.md`.

Plot artifacts are written under `outputs/artifacts/`, the full run state under
`outputs/logs/{run_id}_state.json`, and per-run logs + a manifest under `outputs/logs/`.

## Modeling Rules

- Detect the task type before modeling. Award B is always a **block-averaged-MAE
  panel regression**, so the regression path is primary; a lean classifier
  fallback (HistGradientBoosting + LogisticRegression + baselines) is kept so the
  pipeline stays general-purpose. The candidate pool is a function of the task.
- Baseline models must be evaluated before candidate models.
- Candidate selection is data-driven; no single algorithm is always preferred.
- **Selection uses blocked GroupKFold cross-validation** (whole periods held out)
  scored by the resolved official metric. Metric resolution is `block_mae`
  (primary, when a category/block column exists) with `mae`/`rmse` as fallbacks
  when the description names a plain error metric. Metric direction
  (`greater_is_better`) is honored. A `log1p`-target model variant is added when
  the target is right-skewed (skew-driven, not metric-driven).
- The block for block-averaged MAE is the **period/time column** when one exists
  (per-period MAE averaged), matching the held-out-period structure. A near
  unique-per-row period key is **coarsened to whole-period blocks**
  (`_resolve_cv_groups`) so the blocked CV holds genuine periods out rather than
  collapsing to random KFold.
- Regression accuracy levers (all dataset-agnostic): leakage-safe
  group/target-aggregate features (fit per CV fold), free-text TF-IDF→SVD,
  **early-stopped** randomized hyperparameter tuning of the top families, a convex
  OOF stack, seed-averaging, and a **cross-family convex (NNLS) blend** of the floor
  + modeling-group specialists (kept only when it strictly beats the best single —
  never regresses). See the engine `models.train_and_predict` and
  `modeling_group.run_modeling_group`.
- For the classification fallback, prefer a stratified/grouped holdout.
- Submission values must match what `DATA_DESCRIPTION.md` and the sample
  submission require; the output is always exactly two columns `row_id,<target>`.
- Never use validation targets or future target values during feature
  engineering (target aggregates are fit inside each CV fold to stay leakage-safe).
- LightGBM / XGBoost / CatBoost are optional; continue with scikit-learn fallback
  models when any is unavailable. No GPU is required.

## Prohibited Assumptions

Do not hardcode:

- `rate_per_10000_ed_visits`
- `overdose_category`
- `all_drugs`, `all_opioids`, or `all_stimulants`
- `918` rows
- any fixed period-id map
- any local absolute path from a developer machine

## Mandatory Anti-Hardcoding Audit

This audit runs automatically on every pipeline execution and can also be run manually.  It is dataset-agnostic: it works on any future hidden dataset.

### When it runs

| Phase | Trigger | What is scanned |
|---|---|---|
| **Pre-run** | Immediately after `discover_schema()`, before any feature engineering | All `.py` files under `src/` and `scripts/` |
| **Post-run** | Before finalising `submission.csv` and `report.pdf` | Same as pre-run, plus the generated report `.md` |

### What it searches for

**Static suspicious terms** (always searched, regardless of dataset):

- Award A field names: `rate_per_10000_ed_visits`, `overdose_category`, `all_drugs`, `all_opioids`, `all_stimulants`
- Known competition-specific column names: `survived`, `passengerid`, `casual`, `registered`, `saleprice`, etc.
- Assumed file names: `train.csv`, `test.csv`, `sampleSubmission.csv`, etc.
- Domain vocabulary: `overdose`, `opioid`, `stimulant`, `titanic`, etc.
- Magic numbers: `918` (previous hardcoded row count)

**Dynamic suspicious terms** (extracted from the current dataset at runtime):

- Backtick-quoted identifiers in `DATA_DESCRIPTION.md`
- Column names from the training file header
- Column names from the sample submission header

Dynamic terms that are generic Python/pandas identifiers (`mean`, `std`, `count`, `min`, `max`, `id`, `value`, etc.) are automatically excluded to reduce noise.

### Classification rules

| Classification | Condition |
|---|---|
| **acceptable** | In `tests/`, comments (`#`), docstrings (`"""`), or markdown prose; OR term appears in a list of 3+ candidate fallbacks |
| **risky** | Term in a 1–2 item list, `in`-operator check, or `!=` comparison |
| **unacceptable** | Direct column access `df["term"]`, variable assignment `var = "term"`, equality check `== "term"`, file loading with hardcoded name |

### Verdict

- `pass` — zero unacceptable findings
- `warn` — zero unacceptable but one or more risky findings
- `fail` — one or more unacceptable findings

A `fail` verdict is logged as a warning but does **not** halt the pipeline.  Fix the unacceptable findings before the next submission.

### Output

Each audit phase writes to:
```
outputs/logs/hardcoding_audit_{pre|post}_{run_id}.json
```

The final report includes a brief audit summary section.

### Manual execution

```bash
# Quick pre-run audit with dynamic terms from data/
python scripts/audit_hardcoding.py

# Verbose output showing all findings
python scripts/audit_hardcoding.py --verbose

# Post-run audit
python scripts/audit_hardcoding.py --phase post

# Skip dynamic term extraction
python scripts/audit_hardcoding.py --no-dynamic
```

Exit codes: `0` = pass, `1` = warn, `2` = fail.

### Implementation files

| File | Role |
|---|---|
| `src/data_agent/audit.py` | Audit engine: scanning, classification, term extraction, log serialisation |
| `scripts/audit_hardcoding.py` | CLI entry point |
| `src/data_agent/runner.py` | Pre/post audit integration in the pipeline |
| `tests/test_audit.py` | 40 regression tests — run before modifying `audit.py` |

```bash
python -m pytest tests/test_audit.py -v
```

## Datetime Feature Engineering — Invariant Behaviour

This rule applies to every unknown future dataset, not only the current one.

### What the agent MUST do

1. **Scan every column for datetime parseability**, including the row_id column
   and join keys. A column that is used as a submission row identifier (e.g.
   `datetime`, `timestamp`, `date`, `period`) still contains temporal
   information that must be extracted as derived features.

2. **Never add the raw row_id / join key to the model feature set** — only the
   derived `col__<field>` columns may be used as model inputs.

3. **Extract the full set of generic time features** from every detected
   datetime source column:
   - Always: `year`, `month`, `month_sin`, `month_cos`, `day`, `dayofweek`,
     `dayofweek_sin`, `dayofweek_cos`, `is_weekend`, `quarter`,
     `weekofyear`, `ordinal`
   - When sub-day timestamps are present: `hour`, `hour_sin`, `hour_cos`

4. **Write a feature audit** to `outputs/logs/<run_id>_profile.json` (under
   `feature_audit`) that records:
   - `detected_datetime_columns`
   - `datetime_feature_sources`
   - `generated_time_features` (per source column)
   - `final_feature_columns`
   - `time_target_signal` (target mean / std per time-feature group)

5. **Mention time feature detection in the report**: which columns were
   recognised as datetime sources, which features were generated, and whether
   the target-signal audit shows a meaningful diurnal / weekly / seasonal
   pattern.

### What is FORBIDDEN

- Hard-coding any dataset-specific column names (`datetime`, `casual`,
  `registered`, `season`, `count`, …) in the feature engineering logic.
- Skipping time feature extraction because a datetime column happens to be
  the row_id column.
- Adding the raw datetime string column as a model feature.

### Implementation files

| File | Role |
|---|---|
| `src/data_agent/features.py` | `_detect_datetime_columns`, `_resolve_time_sources`, `_add_features_for_datetime_col`, `_extract_time_features`, `_audit_time_target_signal` |
| `tests/test_datetime_features.py` | 23 regression tests — run before modifying `features.py` |

Run tests with:
```bash
python -m pytest tests/test_datetime_features.py -v
```

## Agent Architecture

The `.claude/agents/` directory contains 10 core pipeline agents plus a 4-agent
parallel **modeling group**. Each core agent corresponds to a distinct failure mode:

| Agent file | Role | Failure mode it guards |
|------------|------|------------------------|
| `orchestrator.md` | Hub controller | Workflow ordering; 2-hour cap; dispatches the modeling group |
| `task_inference.md` | Schema parsing | Wrong task type; bad target column |
| `data_profiler.md` | Data quality | Missed data issues; schema drift |
| `planner.md` | Plan + self-critique | Leakage in plan; missing baselines |
| `programmer.md` | Pipeline execution | Runtime errors; hardcoded columns |
| `model_search_agent.md` | Model search | Suboptimal model; missing baseline |
| `validation_schema_guardian.md` | Validation + submission | Wrong split; bad submission format |
| `hardcoding_feature_auditor.md` | Audit (3 modes) | Hardcoded columns; missing features |
| `report_writer_reviewer.md` | Report + self-review | Stale conclusions; metric errors |
| `supervisor_gatekeeper.md` | Final gate + keep-best | Undetected critical issues; group regressions |

**Parallel modeling group** (division of labor, dispatched by the orchestrator when
budget remains; each calls `scripts/run_modeling_agent.py --approach <family>`):

| Agent file | Division of labor |
|------------|-------------------|
| `gbdt_specialist.md` | gradient-boosted trees (LightGBM/XGBoost/CatBoost/HGB) + tuning |
| `linear_encoding_specialist.md` | regularized linear models on group/target encodings |
| `trees_specialist.md` | bagged trees (RandomForest/ExtraTrees) |
| `ensemble_meta.md` | best/blend candidates by CV block-MAE; hand off to supervisor (keep-best) |

**Removed / merged agents** (no longer in `.claude/agents/`):
- `plan_critic.md` → merged into `analysis-planner` (self-critique section)
- `report_writer.md` → merged into `report-writer-reviewer`
- `report_reviewer.md` → merged into `report-writer-reviewer`
- `inspector.md` → upgraded to `supervisor-gatekeeper`

## Closed-Loop Supervision (Gates)

The adversarial supervision is implemented as a two-layer closed loop sharing
one verdict schema, defined in `src/data_agent/gates.py`:

```json
{
  "stage": "schema | task_inference | leakage | prediction_sanity | submission | report | supervisor",
  "run_id": "...",
  "status": "pass | warn | fail",
  "reasons": ["..."],
  "suggested_corrections": {"force_task_type": "regression"},
  "checked": { "...evidence..." },
  "critic": "deterministic | llm:<agent>"
}
```

- **Deterministic critics always run** (headless `python main.py`):
  `check_task_consistency`, `check_schema`, `check_prediction_sanity`, plus the
  elevated leakage / submission / report gates. `run_stage_with_gate()` re-runs
  a stage **once** with a `suggested_corrections` hint on `fail`, bounded by a
  hard retry cap **and** a no-progress guard.
- **LLM critic layer** (Claude-driven path): a critic subagent may write
  `outputs/logs/{run_id}_llm_gate_{stage}.json` in the same schema; it **takes
  precedence** over the deterministic verdict (`load_llm_verdict` / `_emit_gate`).
- Each stage writes `outputs/logs/{run_id}_gate_{stage}.json`; the
  `prediction_sanity` and `supervisor` stages also write the fixed-name aliases
  `prediction_sanity.json` and `supervisor_gatekeeper.json`. A `fail` is logged
  and surfaced but **never halts** — the deliverable is always produced.

Closed-loop auto-fixes that re-run a stage: task-consistency
(`force_task_type` → re-resolve + rebuild bundle), leakage (`drop_columns` →
drop + re-run model selection), prediction-sanity (drop remaining suspected
features → re-run model selection).

## Inspection Checklist

Before considering the run complete, verify:

- `submission.csv` exists in the repo root.
- `report.pdf` exists in the repo root.
- `submission.csv` has exactly two columns.
- row ids match the sample submission in order.
- predictions are finite and non-missing.
- logs were written under `outputs/logs/`.
- the report describes the actual current run rather than a fixed prior dataset.
- `outputs/logs/supervisor_gatekeeper.json` final gate is written.
- `outputs/logs/report_review.json` shows `approved: true`.
- `outputs/logs/overfitting_leakage_audit.json` written by Phase 9.
- `outputs/logs/prediction_sanity.json` written by Phase 10.
- per-stage gate verdicts `outputs/logs/{run_id}_gate_{stage}.json` written for
  `schema`, `task_inference`, `leakage`, `prediction_sanity`, `submission`,
  `report`, and `supervisor`.
- for a regression panel, `model_selection.json` shows `metric_name: block_mae`
  and a `grouped_kfold` (or `time_holdout`) `holdout_strategy.type` — selection is
  blocked cross-validation, not a single holdout.
- group/target-aggregate features (`tgt_*`) and any detected text columns appear
  in the feature set; the opaque block/period key is NOT a raw model feature.
- a baseline `submission.csv` is checkpointed before the heavy search; the final
  file beats or equals it (keep-best) and has exactly two columns `row_id,<target>`.
- report Section 8 (Overfitting and Generalization Controls) is present.
