# CLAUDE.md — STAI-X Award B Automation Agent

## Primary Instruction

When the user prompt is:

```text
Do the data analysis
```

run the analysis pipeline (an agent orchestration over `src/data_agent/`, with a
deterministic fallback):

```bash
python main.py
```

If Python reports missing packages, install the committed core dependencies and
rerun:

```bash
pip install -r requirements.txt
python main.py
```

Do not manually craft a submission. Do not rely on any overdose-specific field
names. The only authority for the hidden evaluation dataset is
`data/DATA_DESCRIPTION.md` plus the files placed under `data/`.

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

Plot artifacts are written under `outputs/artifacts/`, the full run state under
`outputs/logs/{run_id}_state.json`, and per-run logs + a manifest under `outputs/logs/`.

## Modeling Rules

- Detect the task type before modeling; use regressors for regression and
  classifiers for classification. The candidate pool is a function of the task.
- Baseline models must be evaluated before candidate models.
- Candidate selection is data-driven; no single algorithm is always preferred.
- If a category or block column is discovered in a regression task, use
  block-averaged MAE for model selection.
- For classification, prefer a stratified holdout; if a time column is
  discovered, prefer a time-based holdout. Otherwise use a random holdout with a
  fixed seed.
- Submission values must match what `DATA_DESCRIPTION.md` and the sample
  submission require (e.g. integer class labels for accuracy-scored tasks).
- Never use validation targets or future target values during feature
  engineering.
- If LightGBM is unavailable, continue with scikit-learn fallback models.

## Prohibited Assumptions

Do not hardcode:

- `rate_per_10000_ed_visits`
- `overdose_category`
- `all_drugs`, `all_opioids`, or `all_stimulants`
- `918` rows
- any fixed period-id map
- any local absolute path from a developer machine

The original Award A scripts are preserved only as reference material under
`scripts/award_a_reference/`; they are not the Award B execution path.

## Mandatory Anti-Hardcoding Audit

This audit runs automatically on every pipeline execution and can also be run manually.  It is dataset-agnostic: it works on any future hidden dataset.

### When it runs

| Phase | Trigger | What is scanned |
|---|---|---|
| **Pre-run** | Immediately after `discover_schema()`, before any feature engineering | All `.py` files under `src/` and `scripts/` (excluding `award_a_reference/`) |
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
| **acceptable** | In `tests/`, `scripts/award_a_reference/`, comments (`#`), docstrings (`"""`), or markdown prose; OR term appears in a list of 3+ candidate fallbacks |
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

The `.claude/agents/` directory contains exactly 10 core agents. Each corresponds to a
distinct failure mode:

| Agent file | Role | Failure mode it guards |
|------------|------|------------------------|
| `orchestrator.md` | Hub controller | Workflow ordering; 2-hour cap |
| `task_inference.md` | Schema parsing | Wrong task type; bad target column |
| `data_profiler.md` | Data quality | Missed data issues; schema drift |
| `planner.md` | Plan + self-critique | Leakage in plan; missing baselines |
| `programmer.md` | Pipeline execution | Runtime errors; hardcoded columns |
| `model_search_agent.md` | Model search | Suboptimal model; missing baseline |
| `validation_schema_guardian.md` | Validation + submission | Wrong split; bad submission format |
| `hardcoding_feature_auditor.md` | Audit (3 modes) | Hardcoded columns; missing features |
| `report_writer_reviewer.md` | Report + self-review | Stale conclusions; metric errors |
| `supervisor_gatekeeper.md` | Final gate | Undetected critical issues |

**Removed / merged agents** (no longer in `.claude/agents/`):
- `plan_critic.md` → merged into `analysis-planner` (self-critique section)
- `report_writer.md` → merged into `report-writer-reviewer`
- `report_reviewer.md` → merged into `report-writer-reviewer`
- `inspector.md` → upgraded to `supervisor-gatekeeper`

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
- report Section 8 (Overfitting and Generalization Controls) is present.
