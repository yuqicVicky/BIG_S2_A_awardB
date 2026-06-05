# Workflow: Award B General Data Analysis

**version:** 2.0
**scope:** Any tabular dataset placed under `data/` with a `DATA_DESCRIPTION.md`
**entry_point:** `analysis-orchestrator`
**exit_point:** `supervisor-gatekeeper` (final gate)

---

## Architecture Overview

Hub-and-spoke: all control flows through `analysis-orchestrator`. Agents do not call each other directly.

```
analysis-orchestrator
        │
        ├─ Phase 1  ──► task-inference-agent
        ├─ Phase 2  ──► validation-and-schema-guardian (schema review)
        ├─ Phase 3  ──► hardcoding-and-feature-auditor (pre-run)
        ├─ Phase 4  ──► data-profiler
        ├─ Phase 5  ──► analysis-planner
        ├─ Phase 6  ──► analysis-programmer (first pass)
        ├─ Phase 7  ──► model-search-agent
        ├─ Phase 8  ──► validation-and-schema-guardian (validation)
        ├─ Phase 9  ──► hardcoding-and-feature-auditor (feature audit)
        ├─ Phase 10 ──► supervisor-gatekeeper (first review)
        ├─ Phase 11 ──► [conditional repair: programmer → model-search → guardian]
        ├─ Phase 12 ──► hardcoding-and-feature-auditor (post-run)
        ├─ Phase 13 ──► report-writer-reviewer
        └─ Phase 14 ──► supervisor-gatekeeper (final gate)
```

---

## Phase Specifications

### Phase 1 — Task Inference

| Field | Value |
|-------|-------|
| **Agent** | `task-inference-agent` |
| **Input** | `data/DATA_DESCRIPTION.md`, all files under `data/` |
| **Output** | `outputs/logs/spec_parse.json` |
| **Gate** | File exists; `task_type != "unknown"` |
| **Failure action** | Halt — task type cannot be resolved |

### Phase 2 — Initial Schema Review

| Field | Value |
|-------|-------|
| **Agent** | `validation-and-schema-guardian` (mode: schema_review) |
| **Input** | `spec_parse.json`, data file paths |
| **Output** | `outputs/logs/validation_strategy.json` |
| **Gate** | File exists |
| **Failure action** | Log warning; continue with default random holdout |

### Phase 3 — Pre-Run Hardcoding Audit

| Field | Value |
|-------|-------|
| **Agent** | `hardcoding-and-feature-auditor` (mode: pre) |
| **Input** | `spec_parse.json`; all `.py` files under `src/` and `scripts/` |
| **Output** | `outputs/logs/hardcoding_audit_pre.json` |
| **Gate** | File exists |
| **Failure action** | Log findings; do not halt — programmer must fix before submission |

### Phase 4 — Data Profiling

| Field | Value |
|-------|-------|
| **Agent** | `data-profiler` |
| **Input** | All files under `data/`; `spec_parse.json` |
| **Output** | `outputs/logs/data_profile.json` |
| **Gate** | File exists; `train.n_rows > 0` |
| **Failure action** | Halt — cannot model without data profile |

### Phase 5 — Analysis Planning

| Field | Value |
|-------|-------|
| **Agent** | `analysis-planner` |
| **Input** | `spec_parse.json`, `data_profile.json` |
| **Output** | `outputs/logs/analysis_plan.json` |
| **Gate** | File exists; `critique.verdict == "PASS"` or `"WARN"` |
| **Failure action** | Halt — plan could not be approved |

### Phase 6 — First Implementation Pass

| Field | Value |
|-------|-------|
| **Agent** | `analysis-programmer` |
| **Input** | `spec_parse.json`, `data_profile.json`, `analysis_plan.json` |
| **Output** | `submission.csv` in repo root; pipeline logs |
| **Gate** | `submission.csv` exists |
| **Failure action** | One repair attempt; if still failing, halt |

### Phase 7 — Model Search

| Field | Value |
|-------|-------|
| **Agent** | `model-search-agent` |
| **Input** | `spec_parse.json`, `data_profile.json`, `analysis_plan.json` |
| **Output** | `outputs/logs/model_search.json`, `outputs/logs/final_model.json` |
| **Gate** | Both files exist; `best_val_score` is finite |
| **Failure action** | Fall back to best baseline; record `used_baseline: true` |

### Phase 8 — Validation & Submission Check

| Field | Value |
|-------|-------|
| **Agent** | `validation-and-schema-guardian` (mode: submission_validation) |
| **Input** | `spec_parse.json`, `submission.csv`, `model_search.json` |
| **Output** | `outputs/logs/submission_validation.json` |
| **Gate** | File exists |
| **Failure action** | Log FAIL; proceed to supervisor for repair decision |

### Phase 9 — Feature Audit + Overfitting Audit

| Field | Value |
|-------|-------|
| **Agent** | `hardcoding-and-feature-auditor` (modes: feature-audit, overfitting-audit) |
| **Input** | `spec_parse.json`, `data_profile.json`, `final_model.json`, `model_search.json`, `validation_strategy.json`, pipeline profile logs |
| **Output** | `outputs/logs/feature_audit_review.json`, `outputs/logs/overfitting_leakage_audit.json` |
| **Gate** | `feature_audit_review.json` exists |
| **Failure action** | Log findings; continue |

### Phase 10 — First Supervisor Review

| Field | Value |
|-------|-------|
| **Agent** | `supervisor-gatekeeper` |
| **Input** | All log files; `submission.csv` |
| **Output** | `outputs/logs/supervisor_gatekeeper.json`, `outputs/logs/prediction_sanity.json` |
| **Gate** | `supervisor_gatekeeper.json` exists |
| **Decision** | `repair_needed` flag determines Phase 11 |

### Phase 11 — Conditional Repair Rerun

**Only if** `supervisor_gatekeeper.repair_needed == true`. Maximum one repair cycle.

Sub-agents (in order, controlled by orchestrator):
1. `analysis-programmer` — apply general-purpose fixes only
2. `model-search-agent` — only if `model_rerun_required == true`
3. `validation-and-schema-guardian` — rerun submission validation
4. `supervisor-gatekeeper` — second review

### Phase 12 — Post-Run Hardcoding Audit

| Field | Value |
|-------|-------|
| **Agent** | `hardcoding-and-feature-auditor` (mode: post) |
| **Input** | `spec_parse.json`; all `.py` files; generated report Markdown |
| **Output** | `outputs/logs/hardcoding_audit_post.json` |
| **Gate** | File exists |
| **Failure action** | Log findings; do not halt |

### Phase 13 — Report Generation & Review

| Field | Value |
|-------|-------|
| **Agent** | `report-writer-reviewer` |
| **Input** | All log files; `run_id` |
| **Output** | `report.pdf` (repo root), `outputs/reports/{run_id}_report.pdf`, `outputs/logs/report_review.json` |
| **Gate** | `report.pdf` exists in repo root |
| **Failure action** | Deliver Markdown fallback if PDF conversion fails |

### Phase 14 — Final Supervisor Gate

| Field | Value |
|-------|-------|
| **Agent** | `supervisor-gatekeeper` (final_gate: true) |
| **Input** | All log files; `submission.csv`; `report.pdf` |
| **Output** | Updated `outputs/logs/supervisor_gatekeeper.json` |
| **Gate** | Always completes; verdict delivered to user |
| **Delivery** | Deliver `submission.csv` and `report.pdf` regardless of verdict |

---

## Agent Registry

| Agent | File | Phases |
|-------|------|--------|
| `analysis-orchestrator` | `orchestrator.md` | All (hub) |
| `task-inference-agent` | `task_inference.md` | 1 |
| `validation-and-schema-guardian` | `validation_schema_guardian.md` | 2, 8, 11 |
| `hardcoding-and-feature-auditor` | `hardcoding_feature_auditor.md` | 3, 9, 12 |
| `data-profiler` | `data_profiler.md` | 4 |
| `analysis-planner` | `planner.md` | 5 |
| `analysis-programmer` | `programmer.md` | 6, 11 |
| `model-search-agent` | `model_search_agent.md` | 7, 11 |
| `supervisor-gatekeeper` | `supervisor_gatekeeper.md` | 10, 11, 14 |
| `report-writer-reviewer` | `report_writer_reviewer.md` | 13 |

---

## Log File Registry

All logs are written to `outputs/logs/`:

| File | Written by | Read by |
|------|-----------|---------|
| `spec_parse.json` | task-inference-agent | all agents |
| `validation_strategy.json` | validation-and-schema-guardian | analysis-planner, report-writer-reviewer |
| `hardcoding_audit_pre.json` | hardcoding-and-feature-auditor | supervisor-gatekeeper |
| `data_profile.json` | data-profiler | analysis-planner, model-search-agent, report-writer-reviewer |
| `analysis_plan.json` | analysis-planner | analysis-programmer, model-search-agent |
| `model_search.json` | model-search-agent | validation-and-schema-guardian, supervisor-gatekeeper, report-writer-reviewer |
| `final_model.json` | model-search-agent | supervisor-gatekeeper, report-writer-reviewer |
| `submission_validation.json` | validation-and-schema-guardian | supervisor-gatekeeper, report-writer-reviewer |
| `feature_audit_review.json` | hardcoding-and-feature-auditor | supervisor-gatekeeper, report-writer-reviewer |
| `overfitting_leakage_audit.json` | hardcoding-and-feature-auditor | supervisor-gatekeeper, report-writer-reviewer |
| `hardcoding_audit_post.json` | hardcoding-and-feature-auditor | supervisor-gatekeeper, report-writer-reviewer |
| `supervisor_gatekeeper.json` | supervisor-gatekeeper | orchestrator |
| `prediction_sanity.json` | supervisor-gatekeeper | report-writer-reviewer |
| `report_review.json` | report-writer-reviewer | supervisor-gatekeeper |

---

## Constraints

- **Hub-and-spoke only.** No agent may call another agent directly.
- **No more than one repair rerun.** Phase 11 executes at most once.
- **2-hour wall-clock limit.** Orchestrator enforces this.
- **`DATA_DESCRIPTION.md` is the primary authority.** No schema assumption may be made without reading it first.
- **No hardcoded column names, file names, or metrics** anywhere in agent logic.
- **Deliver `submission.csv` and `report.pdf`** regardless of supervisor verdict — a partial result is better than no result.
