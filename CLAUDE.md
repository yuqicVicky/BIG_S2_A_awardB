# CLAUDE.md — STAI-X Award B Orchestrator (lean dispatcher)

## How you operate

On the prompt `Do the data analysis`, **you (Claude) are the orchestrator** — there is no
orchestrator agent. Generate one `run_id = {YYYYMMDD_HHMMSS}`, persist it across all steps,
and dispatch the 8 steps below **in order** via the Task tool.

Your job is to **dispatch agents and follow each reviewer's `next_action` mechanically** — you
make no analytical judgment of your own. The **only** decision you make yourself is the budget
guard. Agents communicate **only** through JSON files in `outputs/logs/`; never call one agent
from another. Every step is idempotent — a failed step logs and the workflow continues.

Only the top-level instance can dispatch, so all control-flow lives here. Keep this file lean:
static domain policy lives in `.claude/policy/` (linked at the bottom) and each agent reads the
policy it needs.

Per dispatch: keep prompts ≤150 words, tell the agent "compact JSON only, no prose", skip EDA plots.

---

## Reviewer independence

No agent is ever both a doer and the reviewer of its own work. A **reviewer** reads the
artifact fresh, writes **only** its own `*_review.json`, never edits what it reviews, and
owns its loop `next_action`. When a reviewer requires changes, you re-dispatch the **doer**
that owns the artifact — the reviewer never fixes it itself.

| Agent | Role | Judges / produces |
|-------|------|-------------------|
| `data-format-converter`, `data-profiler`, `task-inference-agent`, `analysis-planner`, `analysis-programmer`, `model-search-agent`, `modeling-specialist` (one parametrized agent, run once per `gbdt`/`trees`/`linear` family), `ensemble-meta`, `report-writer` | doer | produce analysis artifacts |
| `validation-and-schema-guardian` | doer | picks CV strategy (3) + formats/validates submission (7) — never reviews another agent's work |
| `plan-reviewer` | reviewer | the feature plan — data coverage, feature reasonableness, completeness constraints |
| `feature-engineering-reviewer` | reviewer | the Step-6A′ ablation evidence — finalizes which authored feature groups enter the model (utility), complementary to the leakage axis |
| `model-performance-reviewer` (+ Step-6 **lead**) | reviewer | modeling result; **lead** consolidates the 3 Step-6 reports + owns the loop `next_action` |
| `feature-leakage-reviewer` | reviewer | features + CV-level leakage |
| `generalization-reviewer` | reviewer | CV validity, calibration, prediction sanity |
| `hardcoding-and-feature-auditor` | reviewer | hardcoded terms (pre Step 3, post Step 8) |
| `report-reviewer` | reviewer | report.pdf |
| `supervisor-gatekeeper` | reviewer | final gate over all artifacts |

---

## Hard constraints + budget guard

| Constraint | Limit | If exceeded |
|---|---|---|
| Token budget | 1,000,000 | skip to Step 7 |
| Wall-clock | 2 hours | skip to Step 7 |
| Feature-plan/completeness gate | ≤2 | accept plan after round 2 |
| Improvement rounds | ≤3 | ship best after round 3 |

**Budget guard (your only self-check):** after any step, if cumulative tokens > 800,000 OR
elapsed > 90 min, skip directly to Step 7. Under this shortcut also set `AWARDB_SKIP_ABLATION=1`
so the Step-6A′ ablation gate passes features through unpruned (it never blocks the floor). Every
other branch is read from a reviewer's `next_action`.

---

## Cross-cutting supervisors

Neither can ever regress the deterministic floor; both are **skipped under the 800k-token /
90-min shortcut**:

- **`modeling-watchdog`** — *efficiency*, **wired into Step 6B `specialist` mode** (see there). It
  runs **concurrently** with the background-launched specialist trainings (not a sequential step):
  it tails their shared heartbeat files live, projects each run's total from the wall-clock that
  actually remains, and `pkill`s / recommends one leaner restart for any run projected to overrun
  its slice, so the round stays inside its time budget. Per-process memory is bounded separately by
  the `ulimit -v` cap on each background launch.
- **`optimizer`** — *quality + full-pipeline critic* (**available, not currently wired into the
  8-step body**). An independent `continue|revise|stop` discriminator at step boundaries the
  reviewers and the `llm_gate` files do not cover, plus a one-shot bounded redo of a suboptimal
  step. The reviewers catch what is *wrong*; the optimizer finds what is *suboptimal* or
  *directionally off*. Dispatch it manually when budget allows.

---

## Failure handling — no step halts the workflow

A doer failure never terminates the run. On any doer failure:
1. **Repair-retry (≤2 attempts):** re-dispatch the **same** agent with `repair_mode=true` and the
   captured error/traceback; it diagnoses and fixes its own authored code (no dataset-specific
   hardcoding to dodge the error). The agent reports `repair_exhausted` after its 2nd failed try.
2. **Minimal deterministic fallback:** if still failing, synthesize the step's required artifact
   from the frozen `src/data_agent/` building blocks (per-step recipe below), log a `degraded`
   warning, and **continue** to the next step.

The only hard exit is the **budget guard** (skip to Step 7). Reviewers still own their loop
`next_action`; this rule governs only *doer crashes*, not reviewer verdicts.

---

## The 8 steps

Inputs/outputs per step are the **I/O contract table** below — it is the single source of file
lineage (every input must be produced by an earlier step; the only external inputs are `data/`
and `DATA_DESCRIPTION.md`). The prose here gives the **control logic** only.

**Step 1 — Convert to CSV.** Dispatch `data-format-converter` (scans `data/`, converts non-CSV → CSV). On failure: log & continue.

**Step 2 — Task + profile.** Dispatch sequentially in dependency order: `task-inference-agent` first (task type, target, row_id, metric, file paths — `DATA_DESCRIPTION.md` is authority → `spec_parse.json`), **then** `data-profiler` (descriptive stats, missingness skill, target skewness + log-transform flag — it consumes `spec_parse.json` to resolve file roles). On failure of either: **repair-retry ≤2, then minimal fallback** (per *Failure handling*) — never halt. Minimal fallback: build `spec_parse.json` via `src/data_agent/schema.discover_schema()` + `task` detection, and `data_profile.json` via `features.build_feature_bundle().profile`; log `degraded` and continue.

**Step 3 — Pre-run setup.** Dispatch `validation-and-schema-guardian` (schema-review → picks the CV strategy in `validation_strategy.json` **and** emits `{run_id}_cv_folds.json`, the single canonical fold assignment every Step-6 candidate scores OOF on) and `hardcoding-and-feature-auditor` (pre → scans `src/`+`scripts/`+`outputs/scratch/` for hardcoded terms). Advisory; on failure log & continue.

**Step 4 — Feature plan.** Dispatch `analysis-planner` → `analysis_plan.json` (`modeling_mode` + `data_coverage` + `feature_plan` + `completeness_constraints` — a feature-engineering contract, **not** model architecture; the model pool is fixed in code and CV is owned by the guardian). Its self-critique is a draft, **not** an approval. On failure: **repair-retry ≤2, then minimal fallback** — never halt. Minimal fallback: synthesize a default `analysis_plan.json` from `spec_parse.json` + `data_profile.json` (`modeling_mode="general"`, `feature_plan` = profile-driven defaults, `data_coverage` mapping every `file_schemas` column, `completeness_constraints` from `spec_parse`); log `degraded` and continue to Step 5.

**Step 5 — Feature-plan + completeness gate (≤2 rounds).** Each round dispatch `plan-reviewer` (judges data coverage, feature reasonableness, completeness); follow its `next_action`:
- `accept_plan` → go to Step 6.
- `revise_plan` → dispatch `analysis-planner` (revision mode, pass the `plan_review_{r}.json` path), then re-dispatch `plan-reviewer` for round 2 and accept unconditionally.

**Step 6 — Execution + improvement loop (≤3 rounds).** Division of labor: **`analysis-programmer` = features** (authors leakage-safe feature code at runtime), the **modeling agents = models**. Every candidate scores OOF on the **canonical shared folds** (`{run_id}_cv_folds.json`) so keep-best is apples-to-apples. Each round:
- **A.** Dispatch `analysis-programmer` (**features only**): round 1 seeds the floor **once** — launch `AWARDB_RUN_ID=$RUN_ID AWARDB_SKIP_REPORT=1 AWARDB_TIME_BUDGET_SEC=1200 python main.py` **in the background** (Bash `run_in_background`), then **immediately and concurrently** author `outputs/scratch/{run_id}/feature_pipeline.py` (it only needs `spec_parse.json` + `data_profile.json`, not main.py's output). After main.py finishes, execute `feature_pipeline.py` to write `{run_id}_features_train/pred.parquet` + `{run_id}_feature_spec.json`. Rounds 2-3: set `AWARDB_SKIP_FLOOR=1 AWARDB_KEEP_OUTPUTS=1` — if `{run_id}_oof_floor.csv` already exists, orchestrator skips main.py instantly; apply only **feature-related** `merged_high_impact_suggestions` to the script, rebuild. `AWARDB_TIME_BUDGET_SEC=1200` limits the floor search to 20 min (specialists do the real tuning). No modeling, no candidate, no `submission.csv` after floor. Outputs: `submission.csv` baseline, `{run_id}_model_selection.json`, `{run_id}_oof_floor.csv`, `{run_id}_cv_folds.json` (if absent), `{run_id}_features_train/pred.parquet`, `{run_id}_feature_spec.json`.
- **A′. Feature ablation gate (prove before inclusion).** After the programmer writes `{run_id}_feature_spec.json`, run the deterministic gate so only value-adding feature groups reach the specialists (declaration → ablation-proven). Run `python scripts/run_feature_ablation_gate.py --run-id {run_id} --cv-folds outputs/logs/{run_id}_cv_folds.json --feature-spec outputs/logs/{run_id}_feature_spec.json --round {r}` (round 1 ablates all logical groups; rounds 2-3 pass `--round {r}` so it ablates only this round's NEW columns vs the prior `feature_spec_full.json` — cheap, one fast HGB per group on the canonical folds). It saves the original to `{run_id}_feature_spec_full.json` and overwrites `{run_id}_feature_spec.json` with the survivors (auto-prunes only groups whose removal improves OOF by >1% — a regressing lag group is caught here, *before* ~100 min of specialist training). Then dispatch `feature-engineering-reviewer` (reads `{run_id}_feature_ablation.json` + the full spec, reconciles with any prior `feature_audit_review.json`, finalizes the pruned `{run_id}_feature_spec.json`, writes `{run_id}_feature_gate.json`). Strictly additive: on any failure or under the budget shortcut (set `AWARDB_SKIP_ABLATION=1`) it passes features through unpruned — never blocks the floor. Only the **pruned** spec flows into B; `_append_authored_features` consumes it unchanged.
- **B.** Branch on `analysis_plan.json.modeling_mode` (no judgment of your own). All modeling consumes `{run_id}_cv_folds.json` + the authored `{run_id}_feature_spec.json` and emits `{run_id}_oof_<name>.csv`:
  - `general` → dispatch `model-search-agent` (trains the full pool on canonical folds + authored features; **owns `submission.csv` keep-best** via common-OOF NNLS over floor + search).
  - `specialist` → **background-launch + live watchdog** (the watchdog supervises *concurrently*, it is **not** a sequential step). For each family in {gbdt, trees, linear}, the orchestrator starts its training as a **background** process (Bash `run_in_background`) so all three run at once and the heartbeat is live:
    ```
    AWARDB_HEARTBEAT_PATH=outputs/logs/{run_id}_<fam>-specialist_progress.jsonl \
      bash -c 'if [ "$(uname)" = "Linux" ]; then ulimit -v <mem_kb>; fi; \
        python scripts/run_modeling_agent.py --approach <fam> --round {r} \
        --run-id {run_id} --cv-folds outputs/logs/{run_id}_cv_folds.json \
        --feature-spec outputs/logs/{run_id}_feature_spec.json'
    ```
    Pass `--round {r}` so the search **deepens each round** (rounds 2-3 raise tuning iterations +
    seed-averaging within the fixed pool) — this gives the improvement loop a real model-side lever
    instead of repeating round 1's identical search. An explicit `AWARDB_TUNE_ITER`/`AWARDB_SEEDS`
    env still overrides the per-round default.
    `ulimit -v` is a **Linux-only** virtual-memory hard cap — on macOS it limits the virtual address space which Python exhausts at startup, causing immediate SIGKILL (exit 144). The `uname` guard applies the cap only on Linux; on macOS the OS kernel handles memory pressure. Derive `<mem_kb>` from available RAM (e.g. a fraction of total / number of parallel roles) — never a fixed literal. **In the same batch**, concurrently dispatch `modeling-watchdog` (prompt: `run_id`, `round`, `rounds_left`, `remaining_wall_clock_sec`, `roles="gbdt trees linear"`). It tails the shared heartbeats **while the runs are live**, projects each run's total from its per-unit cost, and `pkill`s any projected to overrun its time slice, writing `{run_id}_<fam>-specialist_watchdog.json` + a `budget_pressure` signal. For each role it killed, **relaunch once** in the background with that file's `recommended_budget`. After all roles emit `final_done` (or are killed), dispatch `ensemble-meta` (common-OOF NNLS over floor + surviving specialists, **promotes `submission.csv` keep-best**). If `scripts/run_modeling_agent.py` is absent or `pkill` is unavailable, fall back to plain parallel Task dispatch of `modeling-specialist` (one instance per family, `family` passed in the prompt; no watchdog) — the floor is the safety net.
  - The mode owner also writes `prediction_sanity.json`, `{run_id}_promotion.json`, and `{run_id}_prior_best.csv`.
- **C.** Dispatch the **three reviewers in parallel** (one batch, 3 Task calls): `model-performance-reviewer` (review mode), `feature-leakage-reviewer` (reads `{run_id}_feature_spec.json` + `feature_pipeline.py`), `generalization-reviewer`. Each writes only its own report.
- **D.** After all three finish, re-dispatch `model-performance-reviewer` (**lead** mode): it reads the 3 reports, merges their high-impact suggestions (tagged by `source_reviewer`), may set `revert_promotion`+`blacklist_candidate`, writes `analysis_review_{r}.json`, and **owns `next_action`**.
- **E.** Follow `next_action`: if `revert_promotion == true` → restore `submission.csv` from `{run_id}_prior_best.csv` and pass `blacklist_candidate` into the next round's mode owner (excluded from its NNLS pool); then `continue_round` → next round; `stop_and_report` → Step 7.

**Keep-best (common-OOF NNLS):** every candidate's OOF on `{run_id}_cv_folds.json` is scored on the one official metric; the keep-best owner (`model-search-agent` general / `ensemble-meta` specialist) NNLS-blends across candidates (weights applied to test preds) and overwrites `submission.csv` only on a strict improvement, recording `{run_id}_promotion.json` (+ `{run_id}_prior_best.csv` for rollback). The floor seeded in 6A guarantees a deliverable. After Step 6 `submission.csv` holds the best prediction; `supervisor-gatekeeper` re-checks in Step 8.

**Step 7 — Report + format.** Dispatch **sequentially** (formatter first so the report can read its output):
1. `validation-and-schema-guardian` (submission-validation → validates/reformats `submission.csv`). Then read `next_action`: `proceed` → step 2; `run_deterministic_fallback` → run the fallback, then re-dispatch this agent.
2. `report-writer` → `report.pdf` (author only; does **not** self-review).

**Step 8 — Final review.** Dispatch **sequentially**:
1. `report-reviewer` → `report_review.json`; follow `next_action`: `approve_report` → step 2; `revise_report` → re-dispatch `report-writer`, then re-dispatch `report-reviewer`.
2. `supervisor-gatekeeper` → final gate; follow `next_action`: `deliver` → done; `repair_first` → dispatch the one repair doer it names in `issues`, then re-dispatch it; `run_deterministic_fallback` → run the fallback, then re-enter at Step 7.

**Deterministic fallback** (the only command you run directly — every other step is a subagent
dispatch; the Phase-6A `python main.py` is run *by* `analysis-programmer`). Run it **only** when
a subagent emits `next_action: run_deterministic_fallback`:
```bash
pip install -r requirements.txt && pip install -r requirements-optional.txt   # if needed
python main.py
```
It produces `submission.csv` + `report.pdf` via the in-process pipeline; then re-enter at Step 7.

---

## I/O contract (file lineage)

| Step | Agent(s) | Inputs | Outputs (in `outputs/logs/` unless noted) |
|------|----------|--------|-------------------------------------------|
| 1 | `data-format-converter` | `data/` | `data_conversion.json` |
| 2a | `task-inference-agent` | `DATA_DESCRIPTION.md`, data CSVs, `data/` (sidecar scan) | `spec_parse.json` (incl. `file_schemas` + `file_sidecars` for non-tabular modalities e.g. images) |
| 2b | `data-profiler` | data CSVs, `spec_parse.json` | `data_profile.json`, `missingness_profile.json`, `imputation_plan.json` |
| 3a | `validation-and-schema-guardian` (schema-review) | `spec_parse.json`, `data_profile.json` | `validation_strategy.json`, `{run_id}_cv_folds.json` |
| 3b | `hardcoding-and-feature-auditor` (pre) | repo `src/`+`scripts/`, `DATA_DESCRIPTION.md`, headers | `hardcoding_audit_pre.json` |
| 4 | `analysis-planner` | `spec_parse.json`, `data_profile.json`, `validation_strategy.json` | `analysis_plan.json` (`modeling_mode` + `data_coverage` + `feature_plan` + `completeness_constraints`) |
| 5 | `plan-reviewer` ↔ `analysis-planner` | `analysis_plan.json`, `spec_parse.json`, `data_profile.json` | `plan_review_{1,2}.json` |
| 6A | `analysis-programmer` (features) | `analysis_plan.json`, `spec_parse.json`, `data_profile.json`, `{run_id}_cv_folds.json` (+ `analysis_review_{r-1}.json` round>1) | floor (round 1): `{run_id}_profile/state/submission_check.json`, `{run_id}_model_selection.json`, `{run_id}_oof_floor.csv`, `submission.csv`; features: `{run_id}_features_train/pred.parquet`, `{run_id}_feature_spec.json`, `outputs/scratch/{run_id}/feature_pipeline.py` |
| 6A′ | `run_feature_ablation_gate.py` (script) → `feature-engineering-reviewer` | `{run_id}_feature_spec.json`, `{run_id}_cv_folds.json`, `data/` (+ prior `{run_id}_feature_spec_full.json` round>1) | `{run_id}_feature_ablation.json`, `{run_id}_feature_spec_full.json` (original backup), pruned `{run_id}_feature_spec.json`, `{run_id}_feature_gate.json`, `{run_id}_llm_gate_feature_gate.json` |
| 6B | `model-search-agent` (general) **or** `modeling-specialist` (×3: gbdt/trees/linear)+`modeling-watchdog`+`ensemble-meta` (specialist) | `{run_id}_cv_folds.json`, `{run_id}_feature_spec.json`, `{run_id}_oof_floor.csv`, `{run_id}_model_selection.json`, `spec_parse.json`, `validation_strategy.json` | general: `model_search.json`+`final_model.json`; specialist: `{run_id}_<fam>-specialist_progress.jsonl` (heartbeats) + `{run_id}_<fam>-specialist_watchdog.json` (watchdog verdicts) + `{run_id}_ensemble_meta.json`+`{run_id}_meta_choice.csv`+`{run_id}_cand_*.csv`+`{run_id}_agent_*.json`; both: `{run_id}_oof_*.csv`, `prediction_sanity.json`, `{run_id}_promotion.json`, `{run_id}_prior_best.csv`, `submission.csv` |
| 6C | `model-performance-reviewer` (review); `feature-leakage-reviewer`; `generalization-reviewer` — **parallel** | modeling outputs (6B), `{run_id}_profile.json`, pruned `{run_id}_feature_spec.json`+`feature_pipeline.py`+`{run_id}_feature_gate.json` (feature reviewer), `validation_strategy.json`, `prediction_sanity.json`, `{run_id}_promotion.json`, `submission.csv` | `model_performance_review.json`; `feature_audit_review.json`; `overfitting_leakage_audit.json` |
| 6D | `model-performance-reviewer` (lead) ↔ `analysis-programmer` | the 3 reports above + `analysis_review_{r-1}.json` | `analysis_review_{1,2,3}.json` (owns `next_action`) |
| 7a | `validation-and-schema-guardian` (submission-validation) | `submission.csv`, `data/sample_submission.csv`, `spec_parse.json` | `submission_validation.json` + formatted `submission.csv` |
| 7b | `report-writer` | all logs above | `report.pdf` (repo root) + `outputs/reports/{run_id}_report.{md,pdf}` |
| 8a | `report-reviewer` ↔ `report-writer` | `report.pdf` + logs | `report_review.json` |
| 8b | `supervisor-gatekeeper` | all logs, `submission.csv`, `report.pdf`, `DATA_DESCRIPTION.md`, sample submission | `supervisor_gatekeeper.json` |

**Required deliverables (repo root):** `submission.csv` (exactly two columns
`<row_id>,<target>`, sample-submission row order, finite values matching the required format)
and `report.pdf`.

---

## Closed-loop verdict protocol

Some agents emit a lightweight machine verdict to `outputs/logs/{run_id}_llm_gate_{stage}.json`:
```json
{"run_id": "...", "stage": "task_inference|leakage|prediction_sanity|schema|submission|report|supervisor",
 "status": "pass|fail", "reasons": ["..."]}
```
Emit `fail` (with ≥1 concrete reason) when the stage's own check fails, else `pass`. The
`supervisor` stage takes the **worst** status across all gate files. These are advisory
diagnostics — they inform a reviewer's `next_action` but never halt the workflow themselves.

---

## Global modeling & safety rules

- **Detect task type before modeling** (from `task-inference-agent`); evaluate a baseline before candidates; selection is data-driven (no fixed favourite algorithm).
- **CV = the canonical shared folds** in `{run_id}_cv_folds.json` (derived once by `validation-and-schema-guardian` from `validation_strategy.json`; structure-aware — within-period / group-time / time / group / stratified / blocked GroupKFold). **Every Step-6 candidate scores OOF on these exact folds** so keep-best is apples-to-apples (`src/data_agent/cv.py`). Metric `block_mae` (primary, when a block column exists) else `mae`/`rmse`. Coarsen a near-unique period key to whole-period blocks. Classification → stratified/grouped holdout.
- Accuracy levers (dataset-agnostic): leakage-safe group/target aggregates **fit per fold**, TF-IDF→SVD on free text, early-stopped randomized tuning, convex OOF stack, seed-averaging, cross-family NNLS blend.
- **Never** use validation/future targets in feature engineering. LightGBM/XGBoost/CatBoost optional — fall back to scikit-learn.

- **No hardcoding.** Resolve every column name, file path, task type, and metric at runtime from `spec_parse.json` / `data_profile.json` / `DATA_DESCRIPTION.md` — never literal. This applies equally to **LLM-authored code** under `outputs/scratch/{run_id}/` (e.g. `programmer_pipeline.py`) and to the embedded snippets agents write at runtime. Banned literals include `rate_per_10000_ed_visits`, `overdose_category`, `all_drugs/all_opioids/all_stimulants`, `918`, any fixed period-id map, any dev absolute path, or any term not derived from the data at runtime. `hardcoding-and-feature-auditor` enforces this (pre Step 3, post Step 8) over `src/`, `scripts/`, **and `outputs/scratch/`**; its search terms and classification rules live in its agent file.

- **Datetime features (every dataset).** Scan all columns (incl. row_id/join keys) for datetime parseability; **never** add the raw row_id/join-key/datetime string as a model feature — only derived `col__<field>` columns (year, month, sin/cos, day, dayofweek, is_weekend, quarter, weekofyear, ordinal; + hour fields when sub-day). Record detection under `{run_id}_profile.json → feature_audit` and mention it in the report.

- **Sidecar modalities (images etc.).** When `spec_parse.json.file_sidecars` is non-empty, the planner plans `feature_plan.image_features` and the programmer extracts dependency-light image features (colormap-inversion → scalar-field summaries via numpy+PIL+matplotlib; **no torch**), joined by the sidecar `key_columns`, written to the `image_features` group of `{run_id}_feature_spec.json`. They are a **static per-key observation — not target-derived** (no per-fold leakage), validated like any group by the Step-6A′ ablation gate. Pillow/sidecar absent → gracefully skipped; the floor still ships.

---

## Policy references (read by the agents that need them)

| File | Covers |
|------|--------|
| `.claude/policy/modeling_rules.md` | Task detection, baseline-first, GroupKFold/block-MAE, accuracy levers, runtime budget knobs |
| `.claude/policy/prohibited_assumptions.md` | No hardcoded columns / files / counts / budgets — resolve dynamically |
| `.claude/policy/hardcoding_audit.md` | Anti-hardcoding audit modes, search terms, classification rules |
| `.claude/policy/datetime_invariants.md` | Datetime scan + feature extraction invariants (period_id is opaque, non-datetime) |
| `.claude/policy/inspection_checklist.md` | Pre-completion verification (incl. watchdog/optimizer artifacts) |

---

## Definition of done

- `submission.csv` + `report.pdf` exist in the repo root; submission has two columns, sample-submission row order, finite predictions.
- Each `*_review.json` was written by its **reviewer** (not a doer), and no reviewer edited the artifact it reviewed.
- Step-6A′ ran the feature ablation gate (`{run_id}_feature_ablation.json` + `{run_id}_feature_gate.json`) so the specialists trained on ablation-proven features (or `degraded`/skipped pass-through under the budget shortcut).
- Step-6 ran the 3 parallel reviewer reports + the lead's `analysis_review_{r}.json` each round; `prediction_sanity.json` exists.
- `supervisor_gatekeeper.json` final gate says `deliver`; `report_review.json` is approved.
- No orphan inputs (every file read was produced by an earlier step).
