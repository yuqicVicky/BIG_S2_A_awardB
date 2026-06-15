# Policy — Inspection Checklist

> Reference for the STAI-X Award B workflow. Extracted from CLAUDE.md so the dispatcher stays lean.
> Read by `supervisor-gatekeeper` (Step 7) and used by the orchestrator before declaring the run done.

Before considering the run complete, verify:

- `submission.csv` exists in the repo root.
- `report.pdf` exists in the repo root.
- `submission.csv` has exactly two columns.
- Row ids match the sample submission in order.
- Predictions are finite and non-missing.
- Logs were written under `outputs/runs/{run_id}/logs/`.
- The report describes the actual current run (not a fixed prior dataset).
- `outputs/runs/{run_id}/logs/supervisor_gatekeeper.json` final gate is written.
- `outputs/runs/{run_id}/logs/report_review.json` shows `approved: true`.
- `outputs/runs/{run_id}/logs/feature_audit_review.json` written during Step 5.
- `outputs/runs/{run_id}/logs/submission_validation.json` written during Step 6.
- `outputs/runs/{run_id}/logs/plan_review_*.json` written during Step 4.
- `outputs/runs/{run_id}/logs/analysis_review_*.json` written during Step 5.
- For a regression panel, `model_search.json` shows `metric_name: block_mae` and a `grouped_kfold` (or `time_holdout`) holdout strategy.
- Group/target-aggregate features (`tgt_*`) appear in the feature set; the opaque block/period key is NOT a raw model feature.
- Report contains a section on Overfitting and Generalization Controls.
- `outputs/runs/{run_id}/logs/data_conversion.json` written during Step 1.
- `outputs/runs/{run_id}/logs/missingness_profile.json` and `imputation_plan.json` written during Step 2.

## Restructure additions (watchdog + optimizer/critic)

- For each modeling round, the per-round wall-clock stayed within its derived slice (no round
  blew the budget). If a specialist was killed, `outputs/runs/{run_id}/logs/{role}_watchdog.json`
  records the kill + the leaner restart budget (and a `budget_pressure`), and the floor
  `submission.csv` never regressed.
- The `optimizer` critic (`critic_checkpoint` mode) may write
  `outputs/runs/{run_id}/logs/llm_gate_{stage}.json` for any of the five gap boundaries
  (`conversion, profiling, planning, feature_pipeline, modeling`), each carrying a `critic_action`
  (`continue|revise|stop`); at most one `revise` per boundary and at most one `stop` for the whole
  run were honoured. Any `optimization`-mode redo wrote exactly one
  `outputs/runs/{run_id}/logs/optimizer_{step}.json` and ran at most once.
