# Skill Routing Rules

**version:** 2.0
**applies_to:** `general-data-analysis` workflow (v2 — 14-phase architecture)
**enforced_by:** `analysis-planner` (self-critique), `analysis-orchestrator`, all subagents

---

## 1. Workflow Step → Required Skills

Every workflow step has a fixed set of required skills. Skills marked **HARD** are mandatory and cannot be omitted under any circumstance. Skills marked **COND** are required only when the stated condition is true.

| step_id | Required Skills | Type | Condition |
|---------|----------------|------|-----------|
| `user_intake` | `file-validation` | HARD | — |
| `data_loading` | `data-loader` | HARD | — |
| `data_profiling` | `data-profiler` | HARD | — |
| `task_inference` | _(Claude reasoning only — no skill invocation)_ | — | — |
| `initial_planning` | _(Claude reasoning only — no skill invocation)_ | — | — |
| `plan_critique` | `leakage-check` | HARD | — |
| `eda` | `eda-analyzer` | HARD | — |
| `eda` | `plot-generator` | HARD | — |
| `leakage_check` | `leakage-check` | HARD | — |
| `preprocessing` | `data-splitter` | HARD | — |
| `preprocessing` | `feature-engineer` | COND | any encoding, scaling, or imputation in plan |
| `preprocessing` | `leakage-check` | HARD | — |
| `modeling` | `tabular-modeling` | HARD | — |
| `modeling` | `leakage-check` | HARD | — |
| `modeling` | `model-evaluation` | HARD | — |
| `evaluation` | `model-evaluation` | HARD | — |
| `evaluation` | `plot-generator` | HARD | — |
| `interpretation` | `model-evaluation` | COND | supervised task |
| `interpretation` | `plot-generator` | HARD | — |
| `report_generation` | `report-writing` | HARD | — |
| `report_review` | `report-review` | HARD | — |
| `report_review` | `model-evaluation` | HARD | — |
| `report_review` | `leakage-check` | HARD | — |
| `final_delivery` | `file-writer` | HARD | — |

### Hard-Stop Steps

The following steps must **never** proceed without their HARD skills present. A missing HARD skill at these steps is a `FAIL` — execution halts.

| step_id | Rationale |
|---------|-----------|
| `leakage_check` | Without `leakage-check`, the gate cannot function and leakage goes undetected |
| `preprocessing` | Without `data-splitter`, the train/test contract cannot be enforced |
| `modeling` | Without all three skills, baseline comparison and leakage verification cannot happen |
| `evaluation` | Without `model-evaluation`, no valid test metrics can be produced |
| `report_generation` | Without `report-writing`, the report cannot be structured or claim-checked |
| `report_review` | Without all three skills, the review cannot be independent or complete |

---

## 2. Task Type → Required Skills

Skills required depend on the inferred `task_type`. These are applied on top of the step-level requirements above.

### 2a. All Task Types

| Skill | Steps Where Required |
|-------|---------------------|
| `data-loader` | `data_loading` |
| `data-profiler` | `data_profiling` |
| `eda-analyzer` | `eda` |
| `plot-generator` | `eda`, `evaluation`, `interpretation` |
| `file-writer` | `final_delivery` |
| `report-writing` | `report_generation` |
| `report-review` | `report_review` |

### 2b. `descriptive` only

| Skill | Steps Where Required |
|-------|---------------------|
| `eda-analyzer` | `eda` |
| `plot-generator` | `eda` |

No modeling skills required. `modeling`, `evaluation`, `interpretation` steps are skipped. The following skills must **not** appear in a descriptive plan: `tabular-modeling`, `data-splitter`, `model-evaluation`.

### 2c. `classification` only

| Skill | Steps Where Required |
|-------|---------------------|
| `data-splitter` | `preprocessing` |
| `feature-engineer` | `preprocessing` |
| `leakage-check` | `plan_critique`, `leakage_check`, `preprocessing`, `modeling`, `report_review` |
| `tabular-modeling` | `modeling` |
| `model-evaluation` | `modeling`, `evaluation`, `interpretation`, `report_review` |
| `plot-generator` | `evaluation` (confusion matrix, ROC curve) |

**Mandatory metrics:** accuracy, F1 (weighted), AUC-ROC. Report must include all three.

### 2d. `regression` only

| Skill | Steps Where Required |
|-------|---------------------|
| `data-splitter` | `preprocessing` |
| `feature-engineer` | `preprocessing` |
| `leakage-check` | `plan_critique`, `leakage_check`, `preprocessing`, `modeling`, `report_review` |
| `tabular-modeling` | `modeling` |
| `model-evaluation` | `modeling`, `evaluation`, `interpretation`, `report_review` |
| `plot-generator` | `evaluation` (residual plot, prediction vs actual) |

**Mandatory metrics:** RMSE, MAE, R². Report must include all three.

---

## 3. PlanStep `required_skills` Field Contract

Every `PlanStep` object in `state.plan.draft` and `state.plan.approved` **must** include a `required_skills` field. No exceptions.

```
PlanStep:
  step_id:         str           # must match a step_id from general-data-analysis.md
  step_name:       str
  description:     str
  required_skills: list[str]     # REQUIRED — must be non-empty for all executable steps
  expected_output: str
  responsible:     "python" | "claude"
```

Rules:
- `required_skills` must be a list, never null, never an empty string.
- For `task_inference` and `initial_planning` (Claude-only steps), `required_skills` must be set to `[]` explicitly — not omitted.
- For all other steps, `required_skills` must contain at least one skill.
- Skill names must exactly match an entry in the Skill Registry (Section 5).

---

## 4. analysis-planner Rules

When generating the plan draft, `analysis-planner` must:

1. **Populate `required_skills` for every step** before the plan is considered complete.
2. Cross-reference each step's `required_skills` against Tables 1 and 2 above.
3. For supervised tasks, include `leakage-check` in `required_skills` for `plan_critique`, `leakage_check`, `preprocessing`, `modeling`, and `report_review`.
4. Never omit `tabular-modeling`, `leakage-check`, and `model-evaluation` from the `modeling` step for supervised tasks.
5. Never include modeling skills (`tabular-modeling`, `data-splitter`, `model-evaluation`) in a plan whose `task_type` is `descriptive`.
6. If a step requires a COND skill and the condition is true, treat it as HARD for that run.

**Output format** — plan must include a skill summary block:

```
plan_skill_summary:
  all_skills_used: [<deduplicated list of all skills across all steps>]
  steps_missing_required_skills: []   # must be empty before plan is submitted
```

---

## 5. analysis-planner Self-Critique Rules

The self-critique section of `analysis-planner` must verify the following before issuing a PASS verdict:

| Check | Severity if Failed |
|-------|--------------------|
| Every executable step has a non-empty `required_skills` list | FAIL |
| `required_skills` for each step matches Table 1 (all HARD skills present) | FAIL |
| COND skills are included when their condition is met | FAIL |
| Modeling step includes `tabular-modeling`, `leakage-check`, `model-evaluation` (supervised only) | FAIL |
| Preprocessing step includes `data-splitter` and `leakage-check` (supervised only) | FAIL |
| Report review step includes `report-review`, `model-evaluation`, `leakage-check` | FAIL |
| No modeling skills appear in a `descriptive` plan | FAIL |
| All skill names in the plan exist in the Skill Registry | WARN |
| `plan_skill_summary.steps_missing_required_skills` is empty | FAIL |

PlanCriticAgent must record each check result in `state.plan.critique` with:
- `check`: name of the check
- `verdict`: PASS / WARN / FAIL
- `detail`: what was found

---

## 6. analysis-orchestrator Dispatch Rules

When the Orchestrator dispatches a task to a subagent, it must pass `required_skills` explicitly as part of the task payload. The dispatch payload schema is:

```
TaskDispatch:
  step_id:         str
  step_name:       str
  description:     str
  required_skills: list[str]    # copied directly from state.plan.approved[step_id]
  inputs:          dict         # relevant state fields
  run_id:          str
```

Rules:
1. The Orchestrator must never dispatch a task without `required_skills` in the payload.
2. `required_skills` in the dispatch must match `state.plan.approved[step_id].required_skills` exactly.
3. If `required_skills` is empty for an executable step (indicating a planner error), the Orchestrator must halt and raise `SkillRoutingError` before dispatching.
4. The Orchestrator must log the dispatch payload (including `required_skills`) to `state.execution_log[step_id].dispatch`.

---

## 7. Subagent Fallback Inference

If a subagent receives a task where `required_skills` is missing or empty (due to an upstream error), it may infer the skills from `step_id` using Table 1. This is a **fallback only** — it does not excuse the upstream omission.

**Fallback procedure:**

1. Check `required_skills` field in the received `TaskDispatch`.
2. If missing or empty: look up `step_id` in Table 1 and infer the HARD skills.
3. Record a warning in `state.execution_log[step_id].warnings`:
   ```
   {
     "type": "skill_routing_fallback",
     "step_id": "<step_id>",
     "message": "required_skills was missing from dispatch. Inferred from step_id fallback: [<inferred skills>].",
     "severity": "WARN"
   }
   ```
4. Proceed with the inferred skills.
5. If the `step_id` is also missing or unrecognized, the subagent must return a `FAIL` result — it cannot safely infer skills without a known step.

**Fallback is never silent.** Every fallback invocation must appear in the warnings list.

---

## 8. Non-Negotiable Skills for Critical Steps

The following steps are **blocked from execution** if their mandatory skill set is incomplete. The subagent must return `status: FAIL` and must not attempt partial execution.

### `modeling`

| Mandatory Skill | Why Non-Negotiable |
|----------------|--------------------|
| `tabular-modeling` | Without this, no model can be trained |
| `leakage-check` | Without this, train/test contamination cannot be detected during fitting |
| `model-evaluation` | Without this, no metrics can be computed; baseline comparison is impossible |

If any of the three is missing: **halt, do not train**.

### `preprocessing`

| Mandatory Skill | Why Non-Negotiable |
|----------------|--------------------|
| `data-splitter` | Without this, the train/test boundary cannot be enforced |
| `leakage-check` | Without this, fit-before-split errors go undetected |

If either is missing: **halt, do not transform**.

### `evaluation`

| Mandatory Skill | Why Non-Negotiable |
|----------------|--------------------|
| `model-evaluation` | Without this, test metrics cannot be computed |
| `plot-generator` | Without this, evaluation artifacts cannot be produced |

If either is missing: **halt, do not evaluate on test set**.

### `report_generation`

| Mandatory Skill | Why Non-Negotiable |
|----------------|--------------------|
| `report-writing` | Without this, the report cannot be structured or claim-checked |

If missing: **halt, do not write the report**.

### `report_review`

| Mandatory Skill | Why Non-Negotiable |
|----------------|--------------------|
| `report-review` | Without this, the review has no audit framework |
| `model-evaluation` | Without this, metric claims in the report cannot be verified |
| `leakage-check` | Without this, leakage findings in the report cannot be verified |

If any of the three is missing: **halt, do not issue a review verdict**.

---

## 5. Skill Registry

Canonical list of all valid skill IDs. Any skill name not in this list is invalid and must trigger a WARN in PlanCriticAgent.

| Skill ID | Owned By | Description |
|----------|----------|-------------|
| `file-validation` | Python | Check file existence, format, and readability |
| `data-loader` | Python | Load CSV/XLSX into DataFrame |
| `data-profiler` | Python | Compute per-column statistics and data profile |
| `eda-analyzer` | Python | Correlation, distribution, missing-value analysis |
| `plot-generator` | Python | Generate and save matplotlib/seaborn figures |
| `leakage-check` | Claude | Audit for data leakage vectors |
| `data-splitter` | Python | Train/val/test split with seed recording |
| `feature-engineer` | Python | Imputation, encoding, scaling (fit on train only) |
| `tabular-modeling` | Python | Train baseline and candidate tabular models |
| `model-evaluation` | Python + Claude | Compute metrics, compare to baseline, verify split labels |
| `report-writing` | Claude | Write structured narrative report |
| `report-review` | Claude | Audit report claims against execution log |
| `file-writer` | Python | Write output files and delivery manifest |
