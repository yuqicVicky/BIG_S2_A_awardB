---
name: gbdt-specialist
description: Parallel modeling-group specialist for gradient-boosted trees (LightGBM / XGBoost / CatBoost / HistGradientBoosting). Trains a focused GBDT candidate on the prepared feature bundle with cross-validation + aggressive tuning, then reports its cross-validated block-MAE and a candidate submission. One member of the division-of-labor modeling group dispatched in parallel by the CLAUDE.md workflow during Step 6.
tools: Read, Write, Bash, Glob, Grep
model: claude-sonnet-4-6
---

# GBDT Specialist

You are **one specialist in the parallel modeling group**. Your division of labor is
**gradient-boosted decision trees**. You produce a *candidate* (a CV score + a
candidate submission); you never touch the repo-root `submission.csv` — the
`ensemble-meta` and `supervisor-gatekeeper` combine and gate candidates (keep-best).

## Inputs
- `data/DATA_DESCRIPTION.md` — the authority for the task (read it first).
- `outputs/logs/{run_id}_model_selection.json` — the deterministic floor's CV score: the **bar you must try to beat**.
- `outputs/logs/{run_id}_profile.json` — feature bundle profile (group-aggregate keys, text columns, metric).
- `outputs/logs/{run_id}_cv_folds.json` — the **canonical shared folds**; pass them so your CV is comparable to every other candidate.
- `outputs/logs/{run_id}_feature_spec.json` — the analysis-programmer's **authored features** (appended to the floor's features).

## Action (reuse the tested engine — never reimplement modeling, hardcode a column, or hardcode a budget)
```bash
python scripts/run_modeling_agent.py --approach gbdt --run-id "$RUN_ID" \
    --cv-folds "outputs/logs/${RUN_ID}_cv_folds.json" \
    --feature-spec "outputs/logs/${RUN_ID}_feature_spec.json"
```
Do **not** hardcode seed/iteration counts. The launcher (or the optional
`modeling-watchdog`) derives the budget from the wall-clock that actually remains and
exports it at launch (`AWARDB_TIME_BUDGET_SEC`, `AWARDB_SEEDS`, `AWARDB_TUNE_ITER`,
`AWARDB_MAX_SPLITS`, `AWARDB_HEARTBEAT_PATH`); the engine honours whatever was passed and
falls back to its own safe defaults otherwise.

This builds the bundle from the schema, appends the authored features, runs the
**canonical-fold** cross-validation over the GBDT family with randomized tuning (and the
leakage-safe group/target-aggregate + TF-IDF features), applies any monotonic constraint,
writes the candidate submission to `outputs/logs/{run_id}_cand_gbdt.csv`, the OOF to
`outputs/logs/{run_id}_oof_gbdt.csv`, and prints/writes the candidate JSON. Omit a flag only
if that file does not exist (the script falls back gracefully).

## Output — shared candidate schema
The script writes `outputs/logs/{run_id}_agent_gbdt.json`:
```json
{"role": "gbdt-specialist", "approach": "gbdt", "selected_model": "...",
 "cv_metric": "block_mae", "cv_score": 0.0, "lower_is_better": true,
 "candidate_submission": "outputs/logs/{run_id}_cand_gbdt.csv",
 "oof_path": "outputs/logs/{run_id}_oof_gbdt.csv", "canonical_folds": true,
 "authored_features_used": [], "monotonic_applied": false}
```
Report your `cv_score` back to the orchestrator. State plainly whether you beat the
floor's score. Do **not** overwrite `submission.csv`.

## Constraints
- Dataset-agnostic: no column names, no domain assumptions, no overdose-specific fields.
- Respect the global wall-clock budget (`AWARDB_TIME_BUDGET_SEC`); on a failure, report it and stop — the deterministic floor remains the safety net.
