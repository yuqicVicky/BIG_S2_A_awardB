# CLAUDE.md — STAI-X Award B Orchestrator (lean dispatcher)

## Primary Instruction

When the user prompt is:

```text
Do the data analysis
```

**You (Claude) are the orchestrator.** Dispatch each specialist agent via the Task tool in the
order below. There is no separate orchestrator agent — only the top-level instance can dispatch,
so all control-flow lives here. Keep this file lean: static domain policy lives in `.claude/policy/`
(linked at the bottom) and each agent reads the policy it needs.

Generate a `run_id` at the start and persist it across all steps:

```
run_id = "{YYYYMMDD_HHMMSS}"
```

Every step is idempotent. A failed or skipped step writes a partial log and the workflow continues.

---

## Hard Constraints

| Constraint | Limit | Action if exceeded |
|---|---|---|
| Total token budget | **1,000,000** | Skip optional rounds; go to Step 6 |
| Wall-clock time | **2 hours** | Skip optional rounds; go to Step 6 |
| Plan review rounds | max 3 | Accept plan after round 3 |
| Analysis improvement rounds | max 3 | Ship best result after round 3 |

**Token efficiency (every dispatch):** prompts ≤150 words, no preamble; tell each agent "Be concise,
compact JSON only, no narrative prose"; skip EDA plots. **If cumulative tokens > 800,000 or elapsed
> 90 min after any step, skip directly to Step 6.**

---

## Two cross-cutting supervisors

- **`modeling-watchdog`** — *efficiency*, Step 5 only. Shares the worker's live heartbeat file,
  derives each specialist's time slice from the wall-clock that remains, and kills/restarts a runaway
  run so each round stays < 30 min. Publishes a `budget_pressure` (`low|med|high`) the critic reads.
  See the Step 5 handshake.
- **`optimizer`** — *quality + full-pipeline critic*. Two modes: **(a) `critic_checkpoint`** — an
  independent `continue|revise|stop` discriminator at the five boundaries the existing `llm_gate` gates
  do not cover; **(b) `optimization`** — direct one bounded high-impact redo of a step. The reviewers
  catch what is *wrong*; the optimizer finds what is *suboptimal* or *directionally off*.

The pipeline already emits a unified `{run_id}_llm_gate_{stage}.json` verdict (schema in
`src/data_agent/gates.py`) at **seven** stages — `task_inference, schema, submission, leakage,
prediction_sanity, report, supervisor`. The two hooks below make every boundary either critic-gated or
gate-acted; **both are skipped under the 800k-token / 90-min shortcut.**

**Hook A — Critic checkpoint (five gap boundaries).** Boundaries with no existing gate, and the
immediate upstream a `stop` rolls back to: `conversion`(1, upstream none) · `profiling`(2a→1) ·
`planning`(3→2) · `feature_pipeline`(5A→3) · `modeling`(5B→5A).
```
After boundary B produces its artifact → dispatch
    optimizer(mode=critic_checkpoint, stage=B, run_id, artifact_paths, grounding_paths,
              remaining_wall_clock_sec, budget_pressure)
  → read outputs/logs/{run_id}_llm_gate_{B}.json → switch critic_action:
     continue → proceed
     revise  (NOT revised[B] AND remaining_sec ≥ max_cost_sec):
        revised[B]=true; re-dispatch the SAME producer once with prompt_override (minimal scope)
     stop    (NOT stopped_global AND remaining_sec ≥ max_cost_sec):
        stopped_global=true; re-dispatch B's IMMEDIATE UPSTREAM step once with prompt_override,
        then resume forward from that step
     else → log advisory; proceed
  Budget downgrade: remaining_sec tight OR budget_pressure=="high" → act only on status=="fail"
    for correctness/schema/leakage; demote quality warns to advisory.
```
Guards: `revised[B]` ≤1 per boundary; `stopped_global` ≤1 per **whole run**; budget gate
`remaining_sec ≥ max_cost_sec`. For `modeling` the critic **defers to `results-reviewer` inside the
Step-5 round loop** and runs once after the loop exits (the leftover-win role of the old optimizer);
`feature_pipeline` is critiqued after Phase A (round 1, and later rounds only when new high-impact
features were added).

**Hook B — act on the seven existing gates.** After each stage that writes a gate
(`task_inference`=2b, `leakage`=5C, `prediction_sanity`=5B, `schema`+`submission`=6b, `report`=6a,
`supervisor`=7): read its `{run_id}_llm_gate_{stage}.json`; IF `status=="fail"` AND fixable AND NOT
`repaired[stage]`: apply ONE targeted repair from that gate's own `suggested_corrections` (e.g.
leakage→`drop_columns`, prediction_sanity→`prefer_regularized`/`sanitize_predictions`), set
`repaired[stage]`, proceed. One repair per stage; the deterministic floor is never regressed.

---

## 8-Step Workflow

| Step | Agent(s) | Key outputs |
|------|----------|-------------|
| 1 | `data-format-converter` | `data_conversion.json` |
| 2a | `data-profiler` (invokes the missingness-audit skill) | `data_profile.json`, `missingness_profile.json`, `imputation_plan.json` |
| 2b | `task-inference-agent` (reads `DATA_DESCRIPTION.md` as authority) | `spec_parse.json` |
| 3 | `analysis-planner` (built-in 12-point self-critique) | `analysis_plan.json` |

**Step 1** is best-effort (log + continue on failure). **Step 2** halts the workflow if either agent
fails — downstream cannot proceed. **Step 3** halts on failure (no plan, no analysis).

### Step 4 — Plan Review Loop (efficiency-first, max 3 rounds)

Read `analysis_plan.json → critique.verdict`:
- **PASS** → accept immediately, skip the loop.
- **WARN** → dispatch `plan-reviewer` once (round 1). Accept if `approved == true` or zero FAIL findings.
- **FAIL** → enter the loop:
```
FOR round = 1..3:
  dispatch plan-reviewer (≤80 words, FAIL-severity only: leakage, wrong CV, missing baseline,
    executable-code hardcoding) → plan_review_{round}.json
  IF approved == true: BREAK
  IF zero FAIL findings (only WARNs): log WARNs, accept, BREAK
  IF FAIL findings: dispatch analysis-planner in revision mode ("Fix only the FAIL items in
    plan_review_{round}.json; do not rework WARN-only sections.") → overwrites analysis_plan.json
  IF round == 3: accept regardless, log outstanding issues.
```

### Step 5 — Analysis Execution + Improvement Loop (max 3 rounds)

Track elapsed time after each round; if > 90 min, skip to Step 6.

```
FOR round = 1..3:

  ── Phase A: Feature pipeline (agent-directed, code-enforced) ──
  Dispatch analysis-programmer (≤120 words): "Implement the analysis_plan.json features INSIDE the
    feature engine the models actually train on (the build_feature_bundle path that the Phase-B
    modeling entrypoint consumes) — as fold-safe in-pipeline transformers, NOT a standalone feature
    matrix. Decide which features from the plan/profile at runtime; resolve every column dynamically.
    Do not train models. Emit the feature-pipeline checkpoint describing the registered transformers
    and asserting they live in the consumed pipeline. Round {round}.
    {IF round>1: Implement ONLY the expected_impact='high' suggestions in analysis_review_{round-1}.json;
     do not re-implement items already in place.}"
  The model-consumed engine is the single source of truth for features; a feature the modeling path
  does not read is an unconsumed decoy, not the model's feature set.

  ── Phase B: Model search (live watchdog) ──
  Derive this round's remaining wall-clock slice (dynamically — see modeling_rules.md; no fixed
  numbers). IF scripts/run_modeling_agent.py EXISTS:
    1. For each family in {gbdt, trees, linear}: LAUNCH its training in the BACKGROUND (Bash
       run_in_background), exporting AWARDB_HEARTBEAT_PATH=outputs/logs/{run_id}_{role}_progress.jsonl
       and a budget env scaled to the slice (AWARDB_TIME_BUDGET_SEC, and leaner AWARDB_SEEDS /
       AWARDB_MAX_SPLITS / AWARDB_TUNE_ITER when the slice is tight, fuller when ample):
         python scripts/run_modeling_agent.py --approach <fam> --run-id {run_id}
    2. Dispatch modeling-watchdog (prompt: run_id, round, rounds_left, remaining_wall_clock_sec,
       roles="gbdt trees linear"). It tails the heartbeats and kills any run projected to overrun,
       writing {run_id}_{role}_watchdog.json.
    3. For each role it killed: relaunch ONCE in the background with watchdog.recommended_budget.
       Otherwise await final_done.
    4. Dispatch ensemble-meta: combine candidate CSVs + floor, keep-best → {run_id}_meta_choice.csv,
       ensemble_meta.json.
  ELSE: dispatch model-search-agent (general mode, ≤120 words): blocked GroupKFold CV, metric from
    spec_parse.json, NNLS blend, keep-best → model_search.json, final_model.json, submission.csv.

  ── Phase C: Feature audit (every round — never skip) ──
  Dispatch hardcoding-and-feature-auditor (mode "feature-audit") → feature_audit_review.json.
  Mandatory each round because new features are added per round (catches CV-level target leakage).

  ── Phase D: Review ──
  Dispatch results-reviewer (≤80 words, compact JSON, suggestions ≤50 words each)
    → analysis_review_{round}.json. ONLY results-reviewer writes this file; if the programmer wrote
    it as a side-effect, re-dispatch results-reviewer to overwrite before reading approved_for_final.

  ── Phase E: Continue? ──
  Read analysis_review_{round}.json:
    round 1: BREAK if approved_for_final OR zero high-impact suggestions; else continue.
    round 2: BREAK if approved_for_final; improvement_pct=(prev-cur)/prev*100; BREAK if <0.5%
             OR zero high-impact suggestions; else continue.
    round 3: BREAK (ship best).
```

**Keep-best:** `submission.csv` is overwritten only if the new CV score strictly beats the previous
best (enforced by the model-search / ensemble-meta layer and the supervisor). The floor `submission.csv`
never regresses, so the watchdog killing a runaway specialist can never lose the deliverable.

### Step 6 — Report + Submission Validation (parallel)

- `report-writer-reviewer` → `report.pdf` (repo root) + `report_review.json` (self-review for factual
  accuracy + metric correctness).
- `validation-and-schema-guardian` (validation mode) → `submission_validation.json` (exactly 2 columns,
  row count + order match the sample submission, finite values in plausible range).

### Step 7 — Final Gate + Format Correction

`supervisor-gatekeeper` (final-gate mode) reads every log, inspects `submission.csv` + `report.pdf`
against `DATA_DESCRIPTION.md` + the inspection checklist, and triggers targeted repairs if needed:
submission mismatch → re-dispatch `validation-and-schema-guardian`; report section missing →
`report-writer-reviewer` for that section; critical prediction issue → `model-search-agent` one repair
pass. Writes `supervisor_gatekeeper.json` (final verdict).

---

## Deterministic fallback

If the workflow fails at or before Step 5 round 1 (no `submission.csv`), run:

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt   # if needed
python main.py
```

This produces a complete `submission.csv` + `report.pdf` via the in-process pipeline. Then re-enter at
Step 6 (report writing is still performed by the agent layer).

---

## Required Outputs

Both files in the repo root: `submission.csv` and `report.pdf`. `submission.csv` has exactly two
columns — `<row_id_column>,<target_column>` (names from the sample submission / `DATA_DESCRIPTION.md`),
the sample submission's row order, and values in the required format (class labels / strings /
continuous). Agents communicate **only through JSON log files** in `outputs/logs/`.

---

## Agent Architecture

All agents in `.claude/agents/`. The 8-step workflow is orchestrated by this file.

| Agent | Role | Step |
|-------|------|------|
| `data-format-converter` | Convert non-CSV → CSV | 1 |
| `data-profiler` | Descriptive stats + missingness | 2a |
| `task-inference-agent` | Task/target/metric resolution | 2b |
| `analysis-planner` | Plan + self-critique + revision | 3, 4 |
| `plan-reviewer` | Adversarial plan review | 4 |
| `analysis-programmer` | Feature pipeline + error repair | 5A |
| `gbdt` / `trees` / `linear-encoding` specialists, `ensemble-meta` | Parallel model search (script path) | 5B |
| `model-search-agent` | Model search + keep-best (fallback path) | 5B |
| **`modeling-watchdog`** | **Live efficiency supervisor: budget + kill/restart** | **5B** |
| `hardcoding-and-feature-auditor` | Feature/hardcoding/leakage audit | 5C, 7 |
| `results-reviewer` | Per-round improvement suggestions | 5D |
| **`optimizer`** | **Cross-cutting quality + full-pipeline critic: `critic_checkpoint` (continue/revise/stop) at 5 gap boundaries; one bounded redo** | **all** |
| `report-writer-reviewer` | Report generation + self-review | 6a |
| `validation-and-schema-guardian` | Submission schema validation | 6b |
| `supervisor-gatekeeper` | Final gate + format correction | 7 |

Deprecated (not called): `orchestrator.md` — replaced by this file.

---

## Policy references (read by the agents that need them)

| File | Covers |
|------|--------|
| `.claude/policy/modeling_rules.md` | Task detection, baseline-first, GroupKFold/block-MAE, accuracy levers, runtime budget knobs |
| `.claude/policy/prohibited_assumptions.md` | No hardcoded columns / files / counts / budgets — resolve dynamically |
| `.claude/policy/hardcoding_audit.md` | Anti-hardcoding audit modes, search terms, classification rules |
| `.claude/policy/datetime_invariants.md` | Datetime scan + feature extraction invariants (period_id is opaque, non-datetime) |
| `.claude/policy/inspection_checklist.md` | Pre-completion verification (incl. watchdog/optimizer artifacts) |
