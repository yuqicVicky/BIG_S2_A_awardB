"""Step 2 — the budget-bounded, keep-best parallel modeling group.

After the deterministic floor (Phase 6–7) writes a scored ``submission.csv``, this
stage spends the *remaining* wall-clock budget training focused per-family
specialists (gbdt / linear / trees) — harder-tuned than the floor — then keeps the
best by cross-validated score. It runs **automatically whenever budget remains**
(no prompt), and is strictly additive:

* on any error or exhausted budget it returns the floor unchanged;
* it swaps to a specialist **only when that specialist strictly beats the floor's
  CV score** (keep-best), so the deliverable can never regress.

Dataset-agnostic and task-agnostic: families are filtered by model-name family
tags, the comparison honours the resolved metric direction (``greater_is_better``),
and only families actually present in the floor's pool are dispatched. Works for
the regression panel (block-MAE) and the lean classification fallback (accuracy)
alike.

Knobs (all optional):

| Env var | Default | Effect |
|---|---|---|
| ``AWARDB_MODELING_GROUP`` | ``1`` | set ``0``/``false`` to disable Step 2 |
| ``AWARDB_TIME_BUDGET_SEC`` | ``5400`` | shared wall-clock cap (floor + group) |
| ``AWARDB_GROUP_TUNE_ITER`` | ``AWARDB_TUNE_ITER`` | per-specialist tuning iterations (honoured as-is — never silently raised) |
| ``AWARDB_SEEDS`` | ``3`` | seed-averaging count for the final fit |
| ``AWARDB_MAX_SPLITS`` | ``5`` | inner CV fold count |

The orchestrator/``modeling-watchdog`` derive these from the wall-clock that remains
and pass them at launch; the engine honours them exactly.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import ModelResult, _candidate_family
from .runner import _build_submission, _write_json, apply_monotonic_constraints
from .skills.modeling import ModelingResult, train_and_evaluate_models

# Diverse families first: linear + bagged trees carry a different inductive bias to
# the floor's GBDT stack, so they add the most blend value. gbdt overlaps the floor
# most, so it runs LAST and is the first to be skipped when wall-clock runs low.
_GROUP_FAMILIES = ("linear", "trees", "gbdt")
# Don't start a specialist unless at least this much wall-clock remains.
_MIN_SPECIALIST_SECONDS = 15.0


def _enabled() -> bool:
    return os.environ.get("AWARDB_MODELING_GROUP", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


def _global_budget_seconds() -> float:
    try:
        return float(os.environ.get("AWARDB_TIME_BUDGET_SEC", "5400"))
    except (TypeError, ValueError):
        return 5400.0


def _group_tune_iter() -> int:
    """Per-specialist tuning iterations. The orchestrator/watchdog owns the time
    budget, so an explicit ``AWARDB_GROUP_TUNE_ITER`` (or ``AWARDB_TUNE_ITER``) is
    honoured **exactly** — never silently raised to a floor. Respects a
    globally-disabled tuner (``AWARDB_TUNE_ITER<=0``)."""
    try:
        base = int(os.environ.get("AWARDB_TUNE_ITER", "24"))
    except (TypeError, ValueError):
        base = 24
    if base <= 0:  # tuning disabled globally → honour that
        return 0
    try:
        return max(0, int(os.environ.get("AWARDB_GROUP_TUNE_ITER", str(base))))
    except (TypeError, ValueError):
        return base


@contextlib.contextmanager
def _specialist_env(remaining_seconds: float):
    """Bound a specialist's internal search to the wall-clock that actually remains
    and give it a harder per-family tuning budget. Restores the environment after."""
    saved = {k: os.environ.get(k) for k in ("AWARDB_TIME_BUDGET_SEC", "AWARDB_TUNE_ITER")}
    os.environ["AWARDB_TIME_BUDGET_SEC"] = str(max(1, int(remaining_seconds)))
    tune = _group_tune_iter()
    if tune > 0:
        os.environ["AWARDB_TUNE_ITER"] = str(tune)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _selected_cv_score(mr: ModelResult) -> float | None:
    """The selected model's cross-validated score; falls back to the best OK score."""
    for score in mr.model_scores:
        if score.get("name") == mr.selected_model_name and score.get("status") == "ok":
            return score.get("score")
    ok = [s for s in mr.model_scores if s.get("status") == "ok" and s.get("score") is not None]
    if not ok:
        return None
    return (max if mr.greater_is_better else min)(ok, key=lambda s: s["score"]).get("score")


def _is_better(challenger: float | None, incumbent: float | None,
               greater_is_better: bool, *, eps: float = 1e-9) -> bool:
    """Strictly-better test honouring metric direction. Ties keep the incumbent."""
    if challenger is None or not np.isfinite(challenger):
        return False
    if incumbent is None or not np.isfinite(incumbent):
        return True
    if greater_is_better:
        return challenger > incumbent + eps
    return challenger < incumbent - eps


def _available_families(floor_mr: ModelResult) -> list[str]:
    """Only dispatch specialists for families the floor actually fielded — avoids
    re-running the full pool for a family that doesn't exist in this task."""
    seen = {
        _candidate_family(s["name"])
        for s in floor_mr.model_scores
        if s.get("status") == "ok"
    }
    return [fam for fam in _GROUP_FAMILIES if fam in seen]


def _finite_predictions(mr: ModelResult) -> bool:
    preds = np.asarray(mr.predictions)
    if preds.dtype.kind in "biufc":
        return bool(np.all(np.isfinite(preds.astype(float))))
    # object/label predictions: reject None / NaN entries
    return not any(
        x is None or (isinstance(x, float) and not np.isfinite(x))
        for x in preds.tolist()
    )


def _oof_pair(mr: ModelResult):
    """Aligned ``(y_true, y_pred)`` out-of-fold arrays for a regression result, or
    ``(None, None)``. Floor + specialists share data, seed, and deterministic folds,
    so their OOF rows align by position — the basis for a CV-scored cross-family
    blend (no extra holdout needed)."""
    yt = getattr(mr, "holdout_y_true", None)
    yp = getattr(mr, "holdout_y_pred", None)
    if yt is None or yp is None:
        return None, None
    yt = np.asarray(yt, dtype=float)
    yp = np.asarray(yp, dtype=float)
    if yt.ndim != 1 or yt.shape != yp.shape or len(yt) < 5:
        return None, None
    if not (np.all(np.isfinite(yt)) and np.all(np.isfinite(yp))):
        return None, None
    return yt, yp


def _metric_score(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str,
                  blocks: np.ndarray | None) -> float:
    """Score a prediction vector on the resolved official metric (matches the floor)."""
    err = np.abs(y_true - y_pred)
    if metric_name == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if metric_name == "block_mae" and blocks is not None and len(blocks) == len(y_true):
        return float(pd.DataFrame({"e": err, "b": blocks}).groupby("b")["e"].mean().mean())
    return float(np.mean(err))  # mae (and block_mae fallback when no usable block)


def _aligned_blocks(bundle, schema, n_oof: int, scored_mask=None) -> np.ndarray | None:
    """Block labels aligned to the floor's OOF rows, for block_mae scoring of the
    blend. ``None`` when unavailable or misaligned. When OOF scoring is restricted to
    the submission's scoring subset (``scored_mask``), the OOF pair length is the
    restricted count, so the block vector must be subselected to the SAME rows —
    otherwise the length guard fails and ``_metric_score`` silently falls back to
    plain MAE, making the blend keep-best decision inconsistent with the per-model
    block_mae scores."""
    bc = getattr(schema, "block_column", None)
    if not bc:
        return None
    try:
        y_full = pd.to_numeric(bundle.target, errors="coerce")
        tdf = bundle.train_df.loc[y_full.notna()].reset_index(drop=True)
        if bc not in tdf.columns:
            return None
        blk = tdf[bc].astype(str).to_numpy()
        if scored_mask is not None and len(scored_mask) == len(blk):
            blk = blk[np.asarray(scored_mask, dtype=bool)]
        if len(blk) == n_oof:
            return blk
    except Exception:
        return None
    return None


def _try_blend(floor_modeling, cand_modelings, *, bundle, schema, description,
               metric_name, greater, best_score, run_id, logs_dir, scored_mask=None):
    """Convex (NNLS, sum-to-one) blend of the floor + specialist OOF predictions,
    scored on the official metric. Returns ``(blended_modeling, name, cv, weights)``
    only when the blend STRICTLY beats ``best_score``; otherwise ``None`` (the caller
    keeps the incumbent — the blend can never regress the deliverable). Pure
    aggregation: it adds no new model, just reweights existing predictions."""
    try:
        from scipy.optimize import nnls
    except Exception:
        return None
    floor_mr = floor_modeling.model_result_obj
    fy, fp = _oof_pair(floor_mr)
    if fy is None:
        return None
    cols = [("floor", fp, np.asarray(floor_mr.predictions, dtype=float))]
    for key, cm in cand_modelings:
        cmr = cm.model_result_obj
        cy, cp = _oof_pair(cmr)
        if cy is None or cy.shape != fy.shape or not np.allclose(cy, fy, rtol=0, atol=1e-9):
            continue  # candidate OOF rows not aligned to the floor's — skip
        tp = np.asarray(cmr.predictions, dtype=float)
        if tp.shape != cols[0][2].shape:  # align to the floor's TEST predictions
            continue
        cols.append((key, cp, tp))
    if len(cols) < 2:
        return None  # need the floor + at least one aligned specialist to blend
    M_oof = np.column_stack([c[1] for c in cols])
    try:
        w, _ = nnls(M_oof, fy)
    except Exception:
        return None
    if not np.isfinite(w).all() or w.sum() <= 0:
        return None
    w = w / w.sum()
    blocks = _aligned_blocks(bundle, schema, len(fy), scored_mask=scored_mask)
    blend_cv = _metric_score(fy, M_oof @ w, metric_name, blocks)
    if not _is_better(blend_cv, best_score, greater):
        return None  # blend doesn't strictly help → keep the incumbent
    # Same convex weights on the test predictions; convexity keeps them in range.
    blended_test = np.column_stack([c[2] for c in cols]) @ w
    try:
        blended_test, _ = apply_monotonic_constraints(
            blended_test, bundle.sample_submission, schema, description)
        blended_test = np.asarray(blended_test, dtype=float)
    except Exception:
        pass
    blend_name = f"blend({'+'.join(c[0] for c in cols)})"
    blended_mr = dataclasses.replace(
        floor_mr, predictions=blended_test, selected_model_name=blend_name)
    blended_modeling = dataclasses.replace(
        floor_modeling, predictions=blended_test, model_result_obj=blended_mr)
    weights = {c[0]: round(float(wi), 4) for c, wi in zip(cols, w)}
    return blended_modeling, blend_name, blend_cv, weights


def run_modeling_group(
    *,
    bundle,
    schema,
    description: str,
    floor_modeling: ModelingResult,
    repo_root: Path,
    run_id: str,
    logs_dir: Path,
    t_start: float,
    random_state: int = 42,
    folds=None,
    scored_mask=None,
) -> tuple[ModelingResult, dict[str, Any]]:
    """Run the keep-best modeling group and return ``(best_modeling, meta)``.

    ``best_modeling`` is either the floor (unchanged) or a specialist that strictly
    beat it on cross-validated score. Always writes
    the run dir's ``ensemble_meta.json``. Never raises.
    """
    floor_mr = floor_modeling.model_result_obj
    greater = bool(floor_mr.greater_is_better)
    floor_score = _selected_cv_score(floor_mr)
    meta: dict[str, Any] = {
        "run_id": run_id,
        "metric": floor_mr.metric_name,
        "greater_is_better": greater,
        "candidates": {"floor": floor_score},
        "choice": "floor",
        "chosen_cv_score": floor_score,
        "chosen_submission": "submission.csv",
        "specialists": {},
    }

    def _finish(best_modeling: ModelingResult, note: str | None = None):
        if note:
            meta["note"] = note
        try:
            _write_json(meta, logs_dir / "ensemble_meta.json")
        except Exception:
            pass
        return best_modeling, meta

    if not _enabled():
        return _finish(floor_modeling, note="modeling group disabled (AWARDB_MODELING_GROUP=0)")

    families = _available_families(floor_mr)
    if not families:
        return _finish(floor_modeling, note="no recognised model families in the floor pool")

    global_budget = _global_budget_seconds()
    best_modeling = floor_modeling
    best_score = floor_score
    ok_candidates: list[tuple[str, ModelingResult]] = []

    n_fam = len(families)
    for i, fam in enumerate(families):
        remaining = global_budget - (time.monotonic() - t_start)
        if remaining < _MIN_SPECIALIST_SECONDS:
            meta["specialists"][fam] = {
                "status": "skipped", "reason": "budget exhausted",
                "remaining_s": round(remaining, 1),
            }
            continue
        # Reserve a fair share of the remaining wall-clock so the diverse families
        # (run first) are never starved by an earlier specialist overrunning.
        fams_left = n_fam - i
        slice_budget = min(remaining, max(_MIN_SPECIALIST_SECONDS, remaining / fams_left))
        try:
            with _specialist_env(slice_budget):
                cand_modeling = train_and_evaluate_models(
                    bundle=bundle,
                    block_column=schema.block_column,
                    random_state=random_state,
                    families={fam},
                    folds=folds,
                    scored_mask=scored_mask,
                )
            cand_mr = cand_modeling.model_result_obj
            # A family with no real candidate falls back to the full pool — that is
            # just the floor again, so ignore it (keeps the comparison honest).
            if _candidate_family(cand_mr.selected_model_name) != fam:
                meta["specialists"][fam] = {
                    "status": "unavailable", "selected": cand_mr.selected_model_name,
                }
                continue
            cand_score = _selected_cv_score(cand_mr)
            meta["candidates"][fam] = cand_score
            entry: dict[str, Any] = {
                "status": "ok",
                "selected_model": cand_mr.selected_model_name,
                "cv_score": cand_score,
            }
            # Persist a candidate submission for traceability. NEVER the repo-root
            # submission.csv — only the keep-best swap below changes the deliverable.
            try:
                cand_preds, _ = apply_monotonic_constraints(
                    cand_mr.predictions, bundle.sample_submission, schema, description)
                cand_sub = _build_submission(
                    bundle, schema.row_id_column, schema.target_column,
                    cand_preds, cand_mr.output_kind)
                cand_csv = logs_dir / f"cand_{fam}.csv"
                cand_sub.to_csv(cand_csv, index=False)
                entry["candidate_submission"] = str(cand_csv)
            except Exception:
                pass
            meta["specialists"][fam] = entry

            if _finite_predictions(cand_mr):
                ok_candidates.append((fam, cand_modeling))
                if _is_better(cand_score, best_score, greater):
                    best_modeling, best_score = cand_modeling, cand_score
                    meta["choice"], meta["chosen_cv_score"] = fam, cand_score
        except Exception as exc:  # a specialist must never break the run
            meta["specialists"][fam] = {
                "status": "error", "error": f"{type(exc).__name__}: {exc}",
            }

    # ── cross-family convex blend (diversity, not just best-single) ────────────
    # Decorrelated members can beat every single candidate by CV; keep-best safety
    # means the blend is adopted ONLY when it strictly beats the incumbent score.
    try:
        blended = _try_blend(
            floor_modeling, ok_candidates, bundle=bundle, schema=schema,
            description=description, metric_name=floor_mr.metric_name,
            greater=greater, best_score=best_score, run_id=run_id, logs_dir=logs_dir,
            scored_mask=scored_mask)
    except Exception as exc:
        blended = None
        meta.setdefault("blend", {})["error"] = f"{type(exc).__name__}: {exc}"
    if blended is not None:
        blended_modeling, blend_name, blend_cv, weights = blended
        if _finite_predictions(blended_modeling.model_result_obj):
            best_modeling, best_score = blended_modeling, blend_cv
            meta["choice"], meta["chosen_cv_score"] = blend_name, blend_cv
            meta["candidates"][blend_name] = blend_cv
            meta["blend"] = {"members": list(weights.keys()), "weights": weights,
                             "cv_score": blend_cv}
            try:  # traceability only — the orchestrator writes the real submission.csv
                bsub = _build_submission(
                    bundle, schema.row_id_column, schema.target_column,
                    blended_modeling.model_result_obj.predictions,
                    blended_modeling.model_result_obj.output_kind)
                bsub.to_csv(logs_dir / "cand_blend.csv", index=False)
            except Exception:
                pass

    return _finish(best_modeling)
