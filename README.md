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

## Repository Structure

```text
.
├── main.py                        # single deterministic entry point
├── src/data_agent/                # generic Award B pipeline implementation
├── scripts/award_a_reference/     # original Award A scripts kept as reference only
├── data/                          # empty at submission; organizers populate
├── outputs/                       # runtime artifacts/logs/reports
├── CLAUDE.md                      # Claude Code operating instructions
├── .claude/agents/                # 10 core specialist agents
│   ├── orchestrator.md            # hub controller — 14-phase workflow
│   ├── task_inference.md          # schema parsing from DATA_DESCRIPTION.md
│   ├── data_profiler.md           # data quality profiling
│   ├── planner.md                 # plan generation + self-critique
│   ├── programmer.md              # pipeline execution
│   ├── model_search_agent.md      # baseline + candidate model search
│   ├── validation_schema_guardian.md  # validation strategy + submission check
│   ├── hardcoding_feature_auditor.md  # anti-hardcoding + feature audit
│   ├── report_writer_reviewer.md  # report generation + self-review
│   └── supervisor_gatekeeper.md   # final release gate
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

For the optional LightGBM candidate:

```bash
pip install -r requirements-optional.txt
```

If LightGBM is unavailable, the pipeline falls back to scikit-learn models. If
scikit-learn is unavailable, it still produces a valid submission using
baseline mean models.

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

- The implementation does not assume overdose-specific column names.
- The target column and output schema come from `DATA_DESCRIPTION.md` and the
  sample submission.
- Row count is never hardcoded.
- Baseline models are evaluated before candidate models.
- Candidate models are selected by internal holdout performance, not by a fixed
  preferred algorithm.
- No external data is used.
