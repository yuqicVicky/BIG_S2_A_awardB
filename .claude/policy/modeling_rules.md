# Policy — Modeling Rules

> Reference for the STAI-X Award B workflow. Extracted from CLAUDE.md so the dispatcher stays lean.
> Read by the modeling specialists, `model-search-agent`, `analysis-planner`, and `ensemble-meta`.

- Detect the task type before modeling. Use the output of `task-inference-agent` — never assume.
- Baseline models must be evaluated before candidate models.
- Candidate selection is data-driven. No single algorithm is always preferred.
- **Selection uses blocked GroupKFold cross-validation** (whole periods held out).
  Metric resolution: the **planner is the single metric authority** — it reads the
  task-inference output and writes `analysis_plan.json.metric_decision.primary_metric`, which the
  ablation gate, modeling specialists, ensemble keep-best and CV all read (consumed in
  `build_feature_bundle` → `apply_metric_decision`). Defaults when the description names no metric:
  regression → `block_mae` (a category/block column exists) else `mae`, **preferring the MAE family
  over RMSE/R²**; classification → `accuracy` (`roc_auc` for probability output, `f1` under strong
  imbalance). `prefer_mae_family` is regression-only — a classification metric never scores a
  regression task or vice versa.
- The block for block-averaged MAE is the **period/time column** when one exists.
  A near-unique-per-row period key is **coarsened to whole-period blocks** so the blocked CV
  holds genuine periods out rather than collapsing to random KFold.
- **Scoring subset.** When the submission scores only a subset of the target's categories
  (`spec_parse.json.scoring_subset` non-null), OOF block_mae is computed over the **scoring
  categories only** — train-only sub-categories still train and produce OOF (for lag features)
  but do **not** count toward the metric. The guardian restricts `cv_folds.json.scored_rows`
  accordingly; every consumer inherits it via `scored_mask`. Expect a **higher, leaderboard-
  aligned** CV than scoring all categories — a jump from this alignment is correct, **not** a
  regression. Check `cv_folds.json.scoring_restricted` to confirm it is active.
- Regression accuracy levers (all dataset-agnostic): leakage-safe group/target-aggregate features
  (fit per CV fold), free-text TF-IDF→SVD, early-stopped randomized hyperparameter tuning,
  a convex OOF stack, seed-averaging, and a **cross-family convex (NNLS) blend**.
- For classification, prefer a stratified/grouped holdout.
- Submission values must match what `DATA_DESCRIPTION.md` and the sample submission require.
- Never use validation targets or future target values during feature engineering.
- LightGBM / XGBoost / CatBoost are optional; continue with scikit-learn fallbacks when unavailable.

## Runtime budget knobs (derived, never hardcoded)

The orchestrator / `modeling-watchdog` derive these from the wall-clock that actually remains and
export them at launch; the engine (`src/data_agent/models.py`, `modeling_group.py`) honours them
exactly and is otherwise unchanged at the defaults:

| Env var | Default | Effect |
|---|---|---|
| `AWARDB_TIME_BUDGET_SEC` | 5400 | shared wall-clock cap (floor + group) |
| `AWARDB_SEEDS` | 3 | seed-averaging count for the final fit |
| `AWARDB_TUNE_ITER` / `AWARDB_GROUP_TUNE_ITER` | 12 / = TUNE_ITER | randomized-search iterations (honoured as-is — never silently raised) |
| `AWARDB_MAX_SPLITS` | 5 | inner CV fold count |
| `AWARDB_HEARTBEAT_PATH` | unset | when set, the engine streams per-candidate/seed progress for the watchdog |

Specialists never set these themselves — no fixed seed/iteration counts in any agent.

## Keep-best (common-OOF NNLS)

Every candidate's OOF on `cv_folds.json` is scored on the one official metric. The keep-best
owner (`model-search-agent` in general mode / `ensemble-meta` in specialist mode) performs a
**non-negative least-squares (NNLS) blend** across all candidate OOF vectors, then applies those
weights to test predictions. `submission.csv` is overwritten **only on a strict improvement**
over the current best; the prior best is saved to `prior_best.csv` for rollback.
The floor seeded in Step 6A guarantees a deliverable even if all specialists fail.

## Sidecar modalities (images, etc.)

When `spec_parse.json.file_sidecars` is non-empty:
- The planner includes `feature_plan.image_features` in `analysis_plan.json`.
- The programmer extracts dependency-light image features (colormap-inversion → scalar summaries
  via numpy + PIL + matplotlib; **no torch**), joined by `key_columns`, written to the
  `image_features` group of `feature_spec.json`.
- Image features are a **static per-key observation** — not target-derived, no per-fold leakage.
- Validated like any group by the Step-6A′ ablation gate.
- If Pillow or the sidecar is absent: gracefully skip; the floor still ships.
