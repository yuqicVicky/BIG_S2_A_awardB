# CLAUDE.md — STAI-X Award B Orchestrator

## How you operate

On `Do the data analysis`, **you are the orchestrator**. Generate `run_id = {YYYYMMDD_HHMMSS}`,
create `outputs/runs/{run_id}/{logs,scratch}/`, execute the 8 steps in order.

**Your only decisions:** (1) budget guard after every step; (2) four token-saving rules before
each dispatch; (3) follow each reviewer's `next_action` mechanically.

All I/O goes through `outputs/runs/{run_id}/logs/` (unprefixed filenames). Per dispatch: include
`run_id` in every prompt; ≤150 words; "compact JSON only, no prose".

---

## Reviewer independence

| Agent | Role |
|-------|------|
| `data-format-converter`, `data-profiler`, `data-pattern-analyzer`, `task-inference-agent`, `analysis-planner`, `analysis-programmer`, `model-search-agent`, `modeling-specialist` (gbdt/linear), `ensemble-meta`, `report-writer` | doer |
| `validation-and-schema-guardian` | doer — CV strategy (Step 3) + submission validation (Step 7) |
| `plan-reviewer` | reviewer — feature plan |
| `feature-engineering-reviewer` | reviewer — ablation evidence |
| `model-performance-reviewer` (review + lead) | reviewer — modeling result + round consolidation |
| `feature-leakage-reviewer` | reviewer — features + CV-level leakage (risk-surface-gated) |
| `generalization-reviewer` | reviewer — CV validity, calibration, prediction sanity (risk-surface-gated) |
| `report-reviewer` | reviewer — report.pdf |
| `supervisor-gatekeeper` | reviewer — final gate |

Reviewers never edit what they review. Re-dispatch the **doer** when a reviewer requires changes.

---

## Hard constraints + budget guard

| Constraint | Limit | If exceeded |
|---|---|---|
| Token budget | 1,000,000 | skip to Step 7 |
| Wall-clock | 2 hours | skip to Step 7 |
| Feature-plan gate | ≤2 rounds | accept after round 2 |
| Improvement rounds | ≤2 total | ship best; guard truncates earlier |

**Guard (check after every step):** tokens > 800k OR elapsed > 90 min → skip to Step 7; set
`AWARDB_SKIP_ABLATION=1`.

---

## Token-saving rules

1. **Step 3c skip:** `split_structure == "i.i.d."` AND total columns ≤ 20 → write stub
   `feature_influence.json {"skipped":true}`, skip dispatch.
2. **Round-2 specialist skip:** read `ensemble_meta.json` — families with NNLS weight < 0.05
   → reuse existing `cand_{fam}.csv` / `oof_{fam}.csv`, skip retraining.
3. **Round-2 conditional-reviewer skip:** if round-1 `feature_audit_review.json` and
   `overfitting_leakage_audit.json` both have no HIGH findings → skip both; dispatch only
   `model-performance-reviewer` (review mode).
4. **Report figures skip:** dispatch `report-writer` with `no_figures=true` by default.
   Re-enable only if `report-reviewer` cites missing visualisations.

---

## Failure handling

1. **Repair-retry (≤2):** re-dispatch same agent with `repair_mode=true` + error text.
   `analysis-programmer` repairs `scratch/feature_pipeline.py` only — never `src/data_agent/`.
2. **Checkpoint recovery:** on post-training failure, `orchestrator.py` reloads
   `scratch/predictions_checkpoint.npy` → writes `submission.csv` without retraining.
3. **Minimal fallback:** synthesize artifact from `src/data_agent/` building blocks, log
   `degraded`, continue.

Budget guard is the only hard exit. This rule covers doer crashes only.

---

## The 8 steps

`logs/` = `outputs/runs/{run_id}/logs/` · `root` = repo root · `?` = optional

### Step 1 — Convert to CSV
| | |
|---|---|
| Dispatch | `data-format-converter` |
| In | `data/` |
| Out | `logs/data_conversion.json` |
| Failure | log & continue |

### Step 2 — Task + profile *(sequential)*
| | 2a `task-inference-agent` | 2b `data-profiler` |
|---|---|---|
| In | `DATA_DESCRIPTION.md`, `data/*.csv` | `logs/spec_parse.json`, `data/*.csv` |
| Out | `logs/spec_parse.json`, `logs/llm_gate_task_inference.json` | `logs/data_profile.json`, `logs/missingness_profile.json`, `logs/imputation_plan.json` |

Failure: repair-retry ≤2, then minimal fallback (`schema.discover_schema()` + `build_feature_bundle().profile`).

### Step 3 — Pre-run setup *(sequential)*
| | 3a `validation-and-schema-guardian` | 3c `data-pattern-analyzer` |
|---|---|---|
| In | `logs/spec_parse.json`, `logs/data_profile.json` | `logs/spec_parse.json`, `logs/data_profile.json`, `logs/validation_strategy.json`, `data/*.csv` |
| Out | `logs/validation_strategy.json`, `logs/cv_folds.json`, `logs/llm_gate_schema.json` | `logs/feature_influence.json` |

**Apply token-saving rule 1 before 3c.** Failure: log & continue (advisory).

### Step 4 — Feature plan
| | |
|---|---|
| Dispatch | `analysis-planner` |
| In | `logs/spec_parse.json`, `logs/data_profile.json`, `logs/validation_strategy.json`, `logs/feature_influence.json`? |
| Out | `logs/analysis_plan.json` |
| Failure | repair-retry ≤2, then minimal fallback (`modeling_mode="general"`, profile-driven defaults) |

### Step 5 — Feature-plan gate *(≤2 rounds)*
| | |
|---|---|
| Dispatch | `plan-reviewer` |
| In | `logs/analysis_plan.json`, `logs/spec_parse.json`, `logs/data_profile.json` |
| Out | `logs/plan_review_{r}.json` |

`accept_plan` → Step 6. `revise_plan` → re-dispatch `analysis-planner` (pass `plan_review_1.json`), re-dispatch `plan-reviewer` → accept unconditionally at round 2.

### Step 6 — Execution *(≤2 rounds, budget-gated)*

All modeling agents read `logs/analysis_plan.json.modeling_hints` before training.

**6A — Features + floor** · Dispatch: `analysis-programmer`

| | Round 1 | Round 2 |
|---|---|---|
| In | `logs/analysis_plan.json`, `logs/spec_parse.json`, `logs/data_profile.json`, `logs/cv_folds.json` | same + `logs/analysis_review_1.json` |
| Out (floor, round 1 only) | `root/submission.csv`, `logs/model_selection.json`, `logs/oof_floor.csv` | — |
| Out (every round) | `logs/features_train.parquet`, `logs/features_pred.parquet`, `logs/feature_spec.json`, `scratch/feature_pipeline.py` | same |

Round 1: run `AWARDB_RUN_ID={run_id} AWARDB_SKIP_REPORT=1 AWARDB_TIME_BUDGET_SEC=600 python main.py` in **foreground** before authoring features. Floor writes `logs/validation_strategy.json` only when absent.

**6A′ — Feature ablation gate**

Run: `python scripts/run_feature_ablation_gate.py --run-id {run_id} --cv-folds logs/cv_folds.json --feature-spec logs/feature_spec.json --round {r}`
Then dispatch `feature_gate`:

| In | `logs/feature_ablation.json`, `logs/feature_spec.json`, `logs/feature_spec_full.json`? |
|---|---|
| Out | `logs/feature_ablation.json`, `logs/feature_spec_full.json`, `logs/feature_spec.json` (pruned), `logs/feature_gate.json`, `logs/llm_gate_feature_gate.json` |

On failure: `AWARDB_SKIP_ABLATION=1` (pass-through).

**6B — Modeling** · Branch on `logs/analysis_plan.json.modeling_mode`

*`general`:* Dispatch `model-search-agent`

| In | `logs/cv_folds.json`, `logs/feature_spec.json`, `logs/oof_floor.csv`, `logs/model_selection.json`, `logs/spec_parse.json`, `logs/validation_strategy.json`, `logs/features_train.parquet`, `logs/features_pred.parquet` |
|---|---|
| Out | `logs/model_search.json`, `logs/final_model.json`, `logs/oof_*.csv`, `logs/prediction_sanity.json`, `logs/promotion.json`, `logs/prior_best.csv`, `root/submission.csv` |

*`specialist`:* **(Apply token-saving rule 2 in round 2.)** Dispatch in parallel: `modeling-specialist` × 2 (gbdt, linear)

| Each specialist In | `logs/cv_folds.json`, `logs/feature_spec.json`, `logs/spec_parse.json`, `logs/validation_strategy.json`, `logs/features_train.parquet`, `logs/features_pred.parquet` |
|---|---|
| Each specialist Out | `logs/agent_{fam}.json`, `logs/cand_{fam}.csv` (float proba), `logs/oof_{fam}.csv` (float proba), `logs/{fam}-specialist_progress.jsonl` |
| Set per specialist | `AWARDB_TIME_BUDGET_SEC`≈1000s, `AWARDB_HEARTBEAT_PATH=logs/{fam}-specialist_progress.jsonl` |

Then dispatch `ensemble-meta`:

| In | `logs/cand_*.csv`, `logs/oof_*.csv`, `logs/cv_folds.json`, `logs/spec_parse.json`, `logs/oof_floor.csv`, `logs/model_selection.json` |
|---|---|
| Out | `logs/ensemble_meta.json`, `logs/meta_choice.csv`, `logs/prediction_sanity.json`, `logs/promotion.json`, `logs/prior_best.csv`, `root/submission.csv` (keep-best) |

**6C — Review** · **(Apply token-saving rule 3 in round 2.)**

Always dispatch `model-performance-reviewer` (review):

| In | `logs/agent_*.json`, `logs/ensemble_meta.json`, `logs/promotion.json`, `logs/prediction_sanity.json`, `logs/model_selection.json`, `logs/feature_spec.json`, `logs/validation_strategy.json` |
|---|---|
| Out | `logs/model_performance_review.json` |

`feature-leakage-reviewer` fires if: (a) `chosen_strategy` ∉ `{stratified_kfold, random_holdout}` OR (b) `file_sidecars` non-empty OR (c) `per_fold_aggregates` non-empty:

| In | `logs/feature_spec.json`, `logs/feature_gate.json`, `logs/feature_ablation.json`, `logs/cv_folds.json`, `logs/validation_strategy.json`, `logs/spec_parse.json`, `scratch/feature_pipeline.py` |
|---|---|
| Out | `logs/feature_audit_review.json` |

`generalization-reviewer` fires if leakage condition fires OR `task_type` ∈ `{binary_classification, multiclass_classification}`:

| In | `logs/validation_strategy.json`, `logs/prediction_sanity.json`, `logs/promotion.json`, `logs/ensemble_meta.json`, `logs/oof_*.csv`, `logs/spec_parse.json`, `logs/agent_*.json` |
|---|---|
| Out | `logs/overfitting_leakage_audit.json` |

Conservative default: `validation_strategy.json` missing → dispatch both.

**6D — Lead review** · Dispatch `model-performance-reviewer` (lead):

| In | `logs/model_performance_review.json`, `logs/feature_audit_review.json`?, `logs/overfitting_leakage_audit.json`? |
|---|---|
| Out | `logs/analysis_review_{r}.json` (owns `next_action`) |

**6E — Budget gate**
- Guard passes AND `next_action = revise_and_rerun` → one improvement round (6A→6A′→6B→6C→6D); force `stop_and_report` after round 2.
- Guard fails OR `next_action = stop_and_report` → Step 7.

### Step 7 — Format + report *(sequential)*

**7a** · Dispatch `validation-and-schema-guardian` (submission-validation):

| In | `root/submission.csv`, sample submission CSV (`data/`), `logs/spec_parse.json` |
|---|---|
| Out | `logs/submission_validation.json`, `logs/llm_gate_submission.json`; reformats `root/submission.csv` in-place |

`proceed` → 7b. `run_deterministic_fallback` → run fallback, re-dispatch.

**7b** · Dispatch `report-writer` **(apply token-saving rule 4 — pass `no_figures=true`):**

| In | all `logs/` files |
|---|---|
| Out | `root/report.pdf`, `outputs/reports/{run_id}_report.md`, `outputs/reports/{run_id}_report.pdf` |

### Step 8 — Final review *(sequential)*

**8a** · Dispatch `report-reviewer`:

| In | `root/report.pdf`, all `logs/` files |
|---|---|
| Out | `logs/report_review.json` |

`approve_report` → 8b. `revise_report` → re-dispatch `report-writer` (re-enable figures if cited), re-dispatch `report-reviewer`.

**8b** · Dispatch `supervisor-gatekeeper`:

| In | all `logs/`, `root/submission.csv`, `root/report.pdf`, `DATA_DESCRIPTION.md`, sample submission |
|---|---|
| Out | `logs/supervisor_gatekeeper.json`, `logs/llm_gate_supervisor.json` |

`deliver` → done. `repair_first` → dispatch named doer, re-dispatch gatekeeper. `run_deterministic_fallback` → run fallback, re-enter Step 7.

**Deterministic fallback** (only when a subagent emits `next_action: run_deterministic_fallback`):
```bash
pip install -r requirements.txt && pip install -r requirements-optional.txt && python main.py
```

---

## Closed-loop verdict protocol

Each gate agent writes `logs/llm_gate_{stage}.json`:
```json
{"run_id":"...","stage":"task_inference|leakage|prediction_sanity|schema|submission|report|supervisor","status":"pass|fail","reasons":["..."]}
```
`supervisor` takes the worst status across all gate files. Advisory only — never halts the workflow.

---

## Policy references

| File | Covers | Read by |
|------|--------|---------|
| `.claude/policy/modeling_rules.md` | Task detection, baseline-first, CV/metric authority, keep-best NNLS, accuracy levers, runtime budget knobs, sidecar modalities | modeling agents, planner, ensemble-meta |
| `.claude/policy/prohibited_assumptions.md` | No hardcoded columns/files/counts/budgets | all coding agents |
| `.claude/policy/hardcoding_audit.md` | Audit modes, search terms, classification rules | `hardcoding-and-feature-auditor` |
| `.claude/policy/datetime_invariants.md` | Datetime scan + feature extraction invariants | `analysis-programmer`, `analysis-planner` |
| `.claude/policy/inspection_checklist.md` | Definition of done + pre-completion verification | `supervisor-gatekeeper` |
