# CLAUDE.md — STAI-X Award B Analysis Agent

## Primary Instruction

When the user prompt is:

```text
Do the data analysis
```

**You (Claude) are the orchestrator.** Execute the 8-step agent workflow below by dispatching each specialist agent via the Task tool in the order defined. There is no separate orchestrator agent — all workflow coordination is defined here.

Every step is idempotent. A failed or skipped step writes a partial log and allows the workflow to continue.

Generate a `run_id` at the start and persist it across all steps:

```
run_id = "{YYYYMMDD_HHMMSS}"
```

---

## Hard Constraints (enforced throughout)

| Constraint | Limit | Action if exceeded |
|---|---|---|
| Total token budget | **1,000,000 tokens** | Skip optional rounds; go directly to Step 6 |
| Wall-clock time | **2 hours** | Skip optional rounds; go directly to Step 6 |
| Plan review rounds | max 3 | Accept plan after round 3 regardless |
| Analysis improvement rounds | max 3 | Ship best result after round 3 |

**Token efficiency rules (apply to every agent dispatch):**
- Write prompts in ≤150 words. No preamble, no restating what the agent already knows.
- Tell every agent: "Be concise. Write compact JSON only. No narrative prose in outputs."
- Skip EDA plot generation to save I/O time.
- If cumulative tokens already exceed 800,000 or elapsed time exceeds 90 minutes after any step, skip directly to Step 6 (report + submission).

---

## 8-Step Workflow

### Step 1 — Data Format Conversion

**Agent:** `data-format-converter`

Dispatch this agent first. It scans `data/` for non-CSV files (Excel, Parquet, JSON, TSV, etc.) and converts them to CSV so all downstream agents work with a uniform format.

**Output required:** `outputs/logs/data_conversion.json`

**On failure:** log the error and continue — the step is best-effort; subsequent agents will discover files themselves.

---

### Step 2 — Data Profiling + Missingness Audit + Task Inference

Dispatch these two agents **sequentially** (data-profiler first, then task-inference-agent):

**Agent A: `data-profiler`**

Runs descriptive statistics over every data file, invokes the `missingness-audit-planner` skill to produce a column-level imputation plan, and writes a `descriptive_summary` block for use in the report.

**Outputs required:**
- `outputs/logs/data_profile.json` (shape, dtypes, target distribution, descriptive stats, schema diff)
- `outputs/logs/missingness_profile.json` (from the skill)
- `outputs/logs/imputation_plan.json` (from the skill)

**Agent B: `task-inference-agent`**

Reads `DATA_DESCRIPTION.md` as primary authority, reads `data_profile.json`, and resolves: task type, target column, row-id column, evaluation metric, train/prediction/submission file paths.

**Output required:** `outputs/logs/spec_parse.json`

**On failure of either agent:** halt the workflow — downstream agents cannot proceed without these outputs.

---

### Step 3 — Initial Analysis Planning

**Agent:** `analysis-planner`

Reads `spec_parse.json` and `data_profile.json`, produces a concrete modeling plan covering all applicable model families (tree-based, linear, ensemble/blend) plus their cross-validation strategy, and runs a built-in 12-point self-critique. Outputs the first approved draft of the plan.

**Output required:** `outputs/logs/analysis_plan.json`

**On failure:** halt — no plan means no analysis.

---

### Step 4 — Plan Review Loop (efficiency-first, max 3 rounds)

**Before dispatching any reviewer:** read `analysis_plan.json → critique.verdict`.
- `PASS`: accept the plan immediately. Skip the entire review loop.
- `WARN`: dispatch the plan-reviewer **once** (round 1 only) with the abbreviated
  prompt below. If the reviewer returns `approved == true` or zero FAIL findings,
  accept the plan. Do NOT iterate further on WARN plans — only FAIL findings trigger
  revision. This one-pass check catches structural errors the 12-point self-critique
  missed (e.g. wrong CV strategy) without burning tokens on style issues.
- `FAIL`: enter the full review loop below.

```
FOR round = 1 TO 3:
  1. Dispatch plan-reviewer with prompt (≤80 words):
       "Review analysis_plan.json. Round {round}. Focus on FAIL-severity issues
        only: leakage, wrong CV strategy, missing baseline, executable-code
        hardcoding. Be concise — compact JSON only."
     → writes outputs/logs/plan_review_{round}.json

  2. Read plan_review_{round}.json:
     a. IF approved == true: BREAK (plan accepted)
     b. IF zero FAIL findings (only WARNs): log the WARNs, accept the plan, BREAK
        (WARNs are advisory — do not trigger revision, do not block progression)
     c. IF FAIL findings exist: dispatch analysis-planner in revision mode:
          "Fix only the FAIL-severity items in plan_review_{round}.json.
           Do not rework sections with only WARN findings."
        → overwrites outputs/logs/analysis_plan.json

  3. IF round == 3: accept plan regardless, log any outstanding issues.
```

**Revision scope rule:** the planner must fix FAIL items only. Style improvements,
annotation changes, and WARN-only sections must NOT be revised (wasted tokens).

**Outputs:** `outputs/logs/plan_review_{1,2,3}.json` (only rounds that actually run)

---

### Step 5 — Analysis Execution + Improvement Loop (max 3 rounds)

Track elapsed time after each round. If elapsed time exceeds **90 minutes** after any
round, skip remaining rounds and go directly to Step 6.

```
FOR round = 1 TO 3:

  ── Phase A: Feature pipeline ──
  Dispatch analysis-programmer (prompt ≤120 words):
    "Implement the feature pipeline from analysis_plan.json.
     Write feature matrices to outputs/logs/. Do not train models.
     This is round {round}.
     {IF round > 1: Implement ONLY the high-priority suggestions listed
      in outputs/logs/analysis_review_{round-1}.json
      (suggestions with expected_impact='high' only).
      Do not re-implement items already in place.}"

  ── Phase B: Model search ──

  Check whether `scripts/run_modeling_agent.py` exists:

  **If the script EXISTS** — dispatch these 4 agents in parallel (specialist mode):
    1. `gbdt-specialist`: "Train GBDT candidate for round {round}. Beat the floor in outputs/logs/{run_id}_model_selection.json."
    2. `trees-specialist`: "Train bagged-trees candidate for round {round}. Beat the floor."
    3. `linear-encoding-specialist`: "Train linear candidate for round {round}. Beat the floor."
    After all three complete, dispatch:
    4. `ensemble-meta`: "Combine all candidates + floor. Write {run_id}_meta_choice.csv and ensemble_meta.json."

  **If the script DOES NOT EXIST** — dispatch model-search-agent (general mode):
    Dispatch model-search-agent (prompt ≤120 words):
      "Train candidate models on the prepared feature matrices.
       Use blocked GroupKFold CV. Detect competition metric from spec_parse.json;
       use RMSLE if specified, multi-metric ranking if not.
       Apply NNLS convex blend across model families.
       Write model_search.json, final_model.json, and submission.csv (keep-best).
       This is round {round}."

  ── Phase C: Feature audit (every round) ──
  Dispatch hardcoding-and-feature-auditor (mode "feature-audit")
  → writes outputs/logs/feature_audit_review.json (overwritten each round)

  The audit must run every round because new features are added in each round.
  Skipping it in round 2+ was a bug: it allowed CV-level target leakage from
  precomputed aggregate features to go undetected.

  ── Phase D: Review ──
  Dispatch results-reviewer (prompt ≤80 words):
    "Review round {round} results. Write compact JSON only.
     Keep suggestion text ≤50 words each."
  → writes outputs/logs/analysis_review_{round}.json

  IMPORTANT: analysis_review_{round}.json must be written ONLY by results-reviewer.
  If the programmer wrote a file with this name as a side-effect, the orchestrator
  must dispatch results-reviewer to overwrite it before reading approved_for_final.

  ── Phase E: Decide whether to continue ──
  Read analysis_review_{round}.json:

  IF round == 1:
    IF approved_for_final == true: BREAK
    IF zero suggestions with expected_impact == "high": BREAK
       (plateau — further rounds will not improve score meaningfully)
    ELSE: continue to round 2

  IF round == 2:
    IF approved_for_final == true: BREAK
    improvement_pct = (prev_cv_score - current_cv_score) / prev_cv_score * 100
    IF improvement_pct < 0.5: BREAK (diminishing returns)
    IF zero suggestions with expected_impact == "high": BREAK
    ELSE: continue to round 3

  IF round == 3: BREAK (always — final round, ship best result)
```

**Keep-best rule:** `submission.csv` is overwritten only if the new CV score strictly
beats the previous best. The model-search agent enforces this internally.

**Outputs:**
- `outputs/logs/model_search.json`, `outputs/logs/final_model.json`
- `submission.csv` (repo root, keep-best)
- `outputs/logs/feature_audit_review.json` (every round — overwritten)
- `outputs/logs/analysis_review_{1,2,3}.json` (only rounds that run)

---

### Step 6 — Report Writing + Submission Formatting

Dispatch these two agents in **parallel**:

**Agent A: `report-writer-reviewer`**

Reads all log files produced so far (spec_parse, data_profile, analysis_plan, model_search, final_model, feature_audit, submission_validation, hardcoding audit, prediction_sanity). Generates `report.pdf` dynamically from actual run data, then immediately self-reviews it for factual accuracy and metric correctness.

**Output required:** `report.pdf` (repo root), `outputs/logs/report_review.json`

**Agent B: `validation-and-schema-guardian`** (validation mode)

Validates the final `submission.csv`: column count (exactly 2), row count (must match sample submission), row-id order, no missing values, predictions are finite and in a plausible range.

**Output required:** `outputs/logs/submission_validation.json`

---

### Step 7 — Final Output Review + Format Correction

**Agent:** `supervisor-gatekeeper` (final-gate mode)

Reads every log file under `outputs/logs/`, inspects `submission.csv` and `report.pdf`, verifies they meet the requirements in `DATA_DESCRIPTION.md` and the sample submission. If format issues are found, the supervisor triggers targeted repairs:

- **Submission format mismatch:** re-dispatch `validation-and-schema-guardian` to reformat.
- **Report missing required section:** re-dispatch `report-writer-reviewer` for that section only.
- **Critical prediction issue:** re-dispatch `model-search-agent` for one repair pass.

The supervisor writes the final verdict and confirms the run is complete.

**Output required:** `outputs/logs/supervisor_gatekeeper.json`

---

### Deterministic fallback

If the agent workflow fails at or before Step 5 round 1 (no `submission.csv` produced), run:

```bash
python main.py
```

Install dependencies first if needed:

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt
python main.py
```

This produces a complete `submission.csv` + `report.pdf` via the in-process pipeline. Then re-enter the workflow at Step 6 (report writing is still performed by the agent layer).

---

## Required Outputs

The run must produce both files in the repository root:

- `submission.csv`
- `report.pdf`

`submission.csv` must contain exactly two columns:

```text
<row_id_column>,<target_column>
```

Both column names come from the sample submission / `DATA_DESCRIPTION.md`. The output must preserve the sample submission's row order, and values must match the required output format (integer class labels, string labels, or continuous values).

---

## Workflow Contract

Every agent writes its outputs to `outputs/logs/`. Agents communicate **only through JSON log files** — no agent calls another agent directly. The 8-step workflow above is the sole coordination mechanism.

| Step | Agent(s) | Key log file(s) |
|------|----------|-----------------|
| 1 | `data-format-converter` | `data_conversion.json` |
| 2a | `data-profiler` | `data_profile.json`, `missingness_profile.json`, `imputation_plan.json` |
| 2b | `task-inference-agent` | `spec_parse.json` |
| 3 | `analysis-planner` | `analysis_plan.json` |
| 4 | `plan-reviewer` ↔ `analysis-planner` | `plan_review_{1,2,3}.json` |
| 5 | `analysis-programmer` + `model-search-agent` ↔ `results-reviewer` | `model_search.json`, `final_model.json`, `analysis_review_{1,2,3}.json` |
| 5 (audit) | `hardcoding-and-feature-auditor` | `feature_audit_review.json` |
| 6a | `report-writer-reviewer` | `report.pdf`, `report_review.json` |
| 6b | `validation-and-schema-guardian` | `submission_validation.json` |
| 7 | `supervisor-gatekeeper` | `supervisor_gatekeeper.json` |

---

## Modeling Rules

- Detect the task type before modeling. Use the output of `task-inference-agent` — never assume.
- Baseline models must be evaluated before candidate models.
- Candidate selection is data-driven. No single algorithm is always preferred.
- **Selection uses blocked GroupKFold cross-validation** (whole periods held out).
  Metric resolution: `block_mae` (primary, when a category/block column exists) with `mae`/`rmse` as fallbacks.
- The block for block-averaged MAE is the **period/time column** when one exists.
  A near-unique-per-row period key is **coarsened to whole-period blocks** so the blocked CV
  holds genuine periods out rather than collapsing to random KFold.
- Regression accuracy levers (all dataset-agnostic): leakage-safe group/target-aggregate features
  (fit per CV fold), free-text TF-IDF→SVD, early-stopped randomized hyperparameter tuning,
  a convex OOF stack, seed-averaging, and a **cross-family convex (NNLS) blend**.
- For classification, prefer a stratified/grouped holdout.
- Submission values must match what `DATA_DESCRIPTION.md` and the sample submission require.
- Never use validation targets or future target values during feature engineering.
- LightGBM / XGBoost / CatBoost are optional; continue with scikit-learn fallbacks when unavailable.

---

## Prohibited Assumptions

Do not hardcode:

- `rate_per_10000_ed_visits`
- `overdose_category`
- `all_drugs`, `all_opioids`, or `all_stimulants`
- `918` rows
- any fixed period-id map
- any local absolute path from a developer machine
- any column name, file name, or domain term not derived from `DATA_DESCRIPTION.md` or the data files at runtime

All column names, file paths, task types, and metric names must be resolved dynamically by the agents from the actual data and description files. Every agent that references a column name must read it from `spec_parse.json` or `data_profile.json` — never hardcode it.

---

## Mandatory Anti-Hardcoding Audit

The `hardcoding-and-feature-auditor` agent runs in three modes during the workflow:

| When | Mode | Trigger |
|------|------|---------|
| Step 5, each round | `feature-audit` | After programming + model search |
| Step 7 (if supervisor flags it) | `post` | Before finalising outputs |

### What it searches for

**Static suspicious terms** (always searched):
- Award A field names: `rate_per_10000_ed_visits`, `overdose_category`, `all_drugs`, `all_opioids`, `all_stimulants`
- Known competition-specific column names: `survived`, `passengerid`, `casual`, `registered`, `saleprice`, etc.
- Assumed file names: `train.csv`, `test.csv`, `sampleSubmission.csv`, etc.
- Domain vocabulary: `overdose`, `opioid`, `stimulant`, `titanic`, etc.
- Magic numbers: `918`

**Dynamic suspicious terms** (extracted from the current dataset at runtime):
- Backtick-quoted identifiers in `DATA_DESCRIPTION.md`
- Column names from the training file header
- Column names from the sample submission header

### Classification rules

| Classification | Condition |
|---|---|
| **acceptable** | In `tests/`, comments (`#`), docstrings, or markdown prose; OR term appears in a list of 3+ candidate fallbacks |
| **risky** | Term in a 1–2 item list, `in`-operator check, or `!=` comparison |
| **unacceptable** | Direct column access `df["term"]`, variable assignment `var = "term"`, equality check `== "term"`, file loading with hardcoded name |

A `fail` verdict is logged and surfaced but does **not** halt the workflow.

### Manual execution

```bash
python scripts/audit_hardcoding.py           # pre-run
python scripts/audit_hardcoding.py --verbose
python scripts/audit_hardcoding.py --phase post
```

---

## Datetime Feature Engineering — Invariant Behaviour

This rule applies to every unknown future dataset, not only the current one.

### What the agent MUST do

1. **Scan every column for datetime parseability**, including the row_id column and join keys.
2. **Never add the raw row_id / join key to the model feature set** — only derived `col__<field>` columns.
3. **Extract the full set of generic time features** from every detected datetime source column:
   - Always: `year`, `month`, `month_sin`, `month_cos`, `day`, `dayofweek`, `dayofweek_sin`, `dayofweek_cos`, `is_weekend`, `quarter`, `weekofyear`, `ordinal`
   - When sub-day timestamps are present: `hour`, `hour_sin`, `hour_cos`
4. **Write a feature audit** to `outputs/logs/<run_id>_profile.json` under `feature_audit`.
5. **Mention time feature detection in the report**: which columns were recognised, which features were generated, whether the target-signal audit shows a meaningful temporal pattern.

### What is FORBIDDEN

- Hard-coding any dataset-specific column names in the feature engineering logic.
- Skipping time feature extraction because a datetime column happens to be the row_id column.
- Adding the raw datetime string column as a model feature.

---

## Agent Architecture

All agents are in `.claude/agents/`. The 8-step workflow is orchestrated by **CLAUDE.md** (this file) — there is no separate orchestrator agent.

### Core pipeline agents

| Agent file | Role | Step |
|------------|------|------|
| `data_format_converter.md` | Convert non-CSV files to CSV | 1 |
| `data_profiler.md` | Descriptive stats + missingness audit | 2a |
| `task_inference.md` | Task type + target + metric resolution | 2b |
| `planner.md` | Analysis plan + self-critique + revision | 3, 4 |
| `plan_reviewer.md` | Independent adversarial plan review | 4 |
| `programmer.md` | Pipeline execution + error repair | 5 |
| `model_search_agent.md` | Model search + CV + keep-best | 5 |
| `results_reviewer.md` | Results review + improvement suggestions | 5 |
| `hardcoding_feature_auditor.md` | Audit: feature-audit + post modes | 5, 7 |
| `validation_schema_guardian.md` | Submission schema validation | 6b |
| `report_writer_reviewer.md` | Report generation + self-review | 6a |
| `supervisor_gatekeeper.md` | Final gate + format correction | 7 |

### Modeling specialist agents (dispatched inside model-search-agent or Step 5)

| Agent file | Division of labor |
|------------|-------------------|
| `gbdt_specialist.md` | Gradient-boosted trees (LightGBM/XGBoost/CatBoost/HGB) |
| `linear_encoding_specialist.md` | Regularized linear models on group/target encodings |
| `trees_specialist.md` | Bagged trees (RandomForest/ExtraTrees) |
| `ensemble_meta.md` | Best/blend by CV block-MAE; keep-best |

### Deprecated agents (kept for reference only, not called in the workflow)

- `orchestrator.md` — replaced by this file (CLAUDE.md)

---

## Inspection Checklist

Before considering the run complete, verify:

- `submission.csv` exists in the repo root.
- `report.pdf` exists in the repo root.
- `submission.csv` has exactly two columns.
- Row ids match the sample submission in order.
- Predictions are finite and non-missing.
- Logs were written under `outputs/logs/`.
- The report describes the actual current run (not a fixed prior dataset).
- `outputs/logs/supervisor_gatekeeper.json` final gate is written.
- `outputs/logs/report_review.json` shows `approved: true`.
- `outputs/logs/feature_audit_review.json` written during Step 5.
- `outputs/logs/submission_validation.json` written during Step 6.
- `outputs/logs/plan_review_*.json` written during Step 4.
- `outputs/logs/analysis_review_*.json` written during Step 5.
- For a regression panel, `model_search.json` shows `metric_name: block_mae` and a `grouped_kfold` (or `time_holdout`) holdout strategy.
- Group/target-aggregate features (`tgt_*`) appear in the feature set; the opaque block/period key is NOT a raw model feature.
- Report contains a section on Overfitting and Generalization Controls.
- `outputs/logs/data_conversion.json` written during Step 1.
- `outputs/logs/missingness_profile.json` and `imputation_plan.json` written during Step 2.
