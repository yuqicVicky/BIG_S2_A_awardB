"""Canonical cross-validation folds — the single source of truth for Step-6 CV.

Every Step-6 modeling candidate (the deterministic floor, the gbdt/trees/linear
specialists, and the model-search agent) must score its out-of-fold (OOF)
predictions on the SAME fold assignment so keep-best comparisons are valid
(apples-to-apples). That assignment is derived ONCE from the LLM-chosen
``validation_strategy.json`` (structure-aware: within-period / group-time / time /
group / stratified / random) and persisted to ``{run_id}_cv_folds.json``.

The fold file is indexed by **raw train-file row order** (length = full train rows).
A consumer that drops rows (e.g. ``models._train_regression`` filters NaN targets)
loads the file with its own ``valid_mask`` so the folds re-align by position.

This module is dataset-agnostic: every column name comes from the strategy /
schema at runtime; nothing is hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import _make_cv_folds, _resolve_cv_groups, SKLEARN_AVAILABLE

if SKLEARN_AVAILABLE:  # pragma: no cover - import guard
    from sklearn.model_selection import StratifiedKFold


@dataclass
class CanonicalFolds:
    """Concrete fold assignment shared by every candidate.

    ``folds`` are ``(train_idx, val_idx)`` positional-index pairs into whatever
    frame the loader was given (full train, or a valid-mask subset). ``scored_mask``
    marks the rows that enter OOF scoring (all rows for k-fold strategies; only the
    held-out rows for single-holdout strategies).
    """

    folds: list[tuple[np.ndarray, np.ndarray]]
    scored_mask: np.ndarray
    strategy: str
    n_folds: int
    description: dict[str, Any] = field(default_factory=dict)


# ── building (single CV owner: validation-and-schema-guardian, Step 3) ─────────

def build_canonical_folds(
    train_df: pd.DataFrame,
    *,
    validation_strategy: dict,
    target: pd.Series | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Map ``validation_strategy.json`` to a concrete fold assignment over the FULL
    train rows. Returns ``(fold_assignment, scored_rows, description)`` where:

    - ``fold_assignment[i]`` = the fold whose VALIDATION set contains row ``i``;
      ``-1`` means row ``i`` is never validated (single-holdout training rows).
    - ``scored_rows[i]`` = row ``i`` enters OOF scoring (``fold_assignment[i] != -1``).

    Deterministic given ``random_state``. Falls back to a shuffled KFold whenever a
    structured strategy cannot be realised on this data.
    """
    n = len(train_df)
    strategy = (validation_strategy or {}).get("chosen_strategy") or "random_holdout"
    params = (validation_strategy or {}).get("holdout_parameters") or {}
    rs = int(params.get("random_state") or random_state)
    fa = np.full(n, -1, dtype=int)
    desc: dict[str, Any] = {"requested_strategy": strategy}

    def _kfold_from_groups(group_col: str | None, label: str) -> tuple[np.ndarray, dict] | None:
        groups, reason = _resolve_cv_groups(train_df, group_col, n)
        folds, cv_desc = _make_cv_folds(n, groups, rs)
        assign = np.full(n, -1, dtype=int)
        for f, (_tr, va) in enumerate(folds):
            assign[va] = f
        cv_desc["group_column"] = group_col
        cv_desc["realised_as"] = label
        if reason:
            cv_desc["block_reason"] = reason
        return assign, cv_desc

    def _single_holdout(val_mask: np.ndarray, info: dict) -> tuple[np.ndarray, dict]:
        assign = np.full(n, -1, dtype=int)
        assign[np.asarray(val_mask, dtype=bool)] = 0
        return assign, info

    # ── structured strategies (first realisable wins) ─────────────────────────
    # Forward expanding-window time-series CV: train strictly on past periods,
    # validate on future blocks — mirrors a competition whose test is later periods
    # than all training (no forward peeking, unlike GroupKFold over random periods).
    # Realised via an explicit per-row period rank + per-fold val rank boundaries
    # stored in ``fold_spec`` so ``load_canonical_folds`` can rebuild each fold's
    # train set as "all rows in periods strictly before this fold's val block".
    if strategy == "forward_expanding_time":
        out = _forward_expanding_folds(train_df, params, n)
        if out is not None:
            fa, fold_spec, info = out
            desc.update(info)
            desc["fold_spec"] = fold_spec
            return _finalise(fa, desc)
        # else: fall through — guardian's fallback (e.g. group_time_split / has_panel)

    if strategy in ("group_time_split", "group_split") or validation_strategy.get("detected_structure", {}).get("has_panel"):
        group_col = params.get("group_column") or params.get("time_column")
        if strategy == "group_split":
            held = _hold_out_groups(train_df, group_col, params.get("holdout_fraction", 0.2), rs)
            if held is not None:
                fa, info = _single_holdout(held, {"strategy": "group_split", "group_column": group_col})
                desc.update(info)
                return _finalise(fa, desc)
        out = _kfold_from_groups(group_col, "group_time_split")
        if out is not None and int((out[0] >= 0).sum()) == n:
            fa, info = out
            desc.update(info)
            return _finalise(fa, desc)

    if strategy == "within_period_holdout":
        held = _within_period_mask(train_df, params, rs)
        if held is not None:
            fa, info = _single_holdout(held, {"strategy": "within_period_holdout", **{k: params.get(k) for k in ("sub_period_feature", "holdout_threshold", "holdout_fraction", "time_column")}})
            desc.update(info)
            return _finalise(fa, desc)

    if strategy == "time_based_holdout":
        held = _time_tail_mask(train_df, params.get("time_column"), params.get("holdout_fraction", 0.2))
        if held is not None:
            fa, info = _single_holdout(held, {"strategy": "time_based_holdout", "time_column": params.get("time_column")})
            desc.update(info)
            return _finalise(fa, desc)

    if strategy == "stratified_kfold" and SKLEARN_AVAILABLE:
        strat = _stratify_labels(train_df, params.get("stratify_column"), target)
        if strat is not None:
            k = int(params.get("n_splits") or 5)
            try:
                skf = StratifiedKFold(n_splits=max(2, k), shuffle=True, random_state=rs)
                assign = np.full(n, -1, dtype=int)
                for f, (_tr, va) in enumerate(skf.split(np.arange(n), strat)):
                    assign[va] = f
                desc.update({"strategy": "stratified_kfold", "n_splits": max(2, k), "stratify_column": params.get("stratify_column")})
                return _finalise(assign, desc)
            except Exception:
                pass

    # ── default: shuffled KFold (full OOF coverage) ──────────────────────────
    folds, cv_desc = _make_cv_folds(n, None, rs)
    assign = np.full(n, -1, dtype=int)
    for f, (_tr, va) in enumerate(folds):
        assign[va] = f
    cv_desc["realised_as"] = "random_kfold"
    desc.update(cv_desc)
    return _finalise(assign, desc)


def _finalise(fold_assignment: np.ndarray, desc: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    scored = fold_assignment >= 0
    desc["n_folds"] = int(fold_assignment.max()) + 1 if fold_assignment.max() >= 0 else 0
    desc["n_scored_rows"] = int(scored.sum())
    return fold_assignment, scored, desc


def _forward_expanding_folds(train_df: pd.DataFrame, params: dict, n: int):
    """Build expanding-window forward folds. Returns ``(fold_assignment, fold_spec,
    info)`` or ``None`` when it cannot be realised (caller then falls back).

    ``fold_assignment[i]`` = the fold whose *validation* block contains row ``i``
    (``-1`` = never validated, i.e. an early period used only for training). Each
    fold's validation block is ``horizon`` consecutive periods; blocks are taken
    non-overlapping from the most recent period backwards (so the last fold's val =
    the final ``horizon`` periods, the closest analogue to the hidden test). The
    fold's *training* set is reconstructed downstream as every row in a period
    strictly earlier than its block — hence ``fold_spec`` carries the per-row
    chronological ``period_rank`` and each fold's ``[val_lo, val_hi)`` rank range."""
    time_col = params.get("time_column") or params.get("group_column")
    period_order = params.get("period_order") or []
    if not time_col or time_col not in train_df.columns or not period_order:
        return None
    horizon = int(params.get("horizon") or 0)
    k = int(params.get("n_folds") or 5)
    rank_of = {str(p): i for i, p in enumerate(period_order)}
    col = train_df[time_col].astype(str)
    present = [p for p in period_order if p in set(col.unique())]
    n_periods = len(present)
    if horizon < 1 or n_periods < horizon + 1:
        return None  # not enough history to validate even one future block
    # per-row chronological rank (index into period_order); rows whose period is
    # absent from the order map are ineligible (rank -1 → never train/val).
    period_rank = col.map(rank_of).fillna(-1).astype(int).to_numpy()
    # present-period ranks, sorted, to lay out non-overlapping val blocks from the end
    present_ranks = sorted(rank_of[p] for p in present)
    hi_idx = len(present_ranks)
    blocks: list[tuple[int, int]] = []  # (val_lo_rank, val_hi_rank) exclusive-hi
    for _f in range(k):
        lo_idx = hi_idx - horizon
        if lo_idx < 1:  # need ≥1 earlier period to train on
            break
        val_lo = present_ranks[lo_idx]
        val_hi = present_ranks[hi_idx - 1] + 1
        blocks.append((val_lo, val_hi))
        hi_idx = lo_idx
    if not blocks:
        return None
    blocks.reverse()  # chronological: fold 0 = earliest val block
    fa = np.full(n, -1, dtype=int)
    fold_ranges = []
    for f, (lo, hi) in enumerate(blocks):
        fa[(period_rank >= lo) & (period_rank < hi)] = f
        fold_ranges.append({"val_lo": int(lo), "val_hi": int(hi)})
    fold_spec = {"type": "expanding", "period_rank": [int(x) for x in period_rank.tolist()],
                 "folds": fold_ranges}
    info = {"strategy": "forward_expanding_time", "time_column": time_col,
            "group_column": time_col, "n_splits": len(blocks), "horizon": horizon,
            "n_periods": n_periods, "realised_as": "forward_expanding_time"}
    return fa, fold_spec, info


def _within_period_mask(train_df: pd.DataFrame, params: dict, rs: int) -> np.ndarray | None:
    time_col = params.get("time_column")
    feat = params.get("sub_period_feature") or "day"
    if not time_col or time_col not in train_df.columns:
        return None
    parsed = pd.to_datetime(train_df[time_col], errors="coerce")
    if float(parsed.notna().mean()) < 0.6:
        return None
    sp = getattr(parsed.dt, feat, None)
    if sp is None or not sp.notna().any():
        return None
    threshold = params.get("holdout_threshold")
    if threshold is None:
        frac = float(params.get("holdout_fraction") or 0.2)
        threshold = float(sp.quantile(1.0 - frac))
    mask = (sp >= float(threshold)).to_numpy()
    if 0 < int(mask.sum()) < len(train_df):
        return mask
    return None


def _time_tail_mask(train_df: pd.DataFrame, time_col: str | None, frac: float) -> np.ndarray | None:
    if not time_col or time_col not in train_df.columns:
        return None
    parsed = pd.to_datetime(train_df[time_col], errors="coerce")
    if float(parsed.notna().mean()) < 0.6:
        return None
    n = len(train_df)
    order = np.argsort(parsed.fillna(parsed.max()).to_numpy())
    n_hold = max(1, int(round(n * float(frac or 0.2))))
    mask = np.zeros(n, dtype=bool)
    mask[order[-n_hold:]] = True
    if 0 < int(mask.sum()) < n:
        return mask
    return None


def _hold_out_groups(train_df: pd.DataFrame, group_col: str | None, frac: float, rs: int) -> np.ndarray | None:
    if not group_col or group_col not in train_df.columns:
        return None
    groups = pd.unique(train_df[group_col].dropna())
    if len(groups) < 3:
        return None
    rng = np.random.default_rng(rs)
    perm = rng.permutation(np.asarray(groups, dtype=object))
    n_hold = max(1, int(round(len(groups) * float(frac or 0.2))))
    held = set(perm[:n_hold].tolist())
    mask = train_df[group_col].isin(held).to_numpy()
    if 0 < int(mask.sum()) < len(train_df):
        return mask
    return None


def _stratify_labels(train_df: pd.DataFrame, strat_col: str | None, target: pd.Series | None) -> np.ndarray | None:
    if strat_col and strat_col in train_df.columns:
        return train_df[strat_col].astype(str).to_numpy()
    if target is not None and len(target) == len(train_df):
        t = pd.Series(target)
        if t.nunique(dropna=True) <= max(20, len(t) // 50):
            return t.astype(str).to_numpy()
    return None


# ── persistence ──────────────────────────────────────────────────────────────

def write_cv_folds_json(
    path: str | Path,
    *,
    run_id: str,
    fold_assignment: np.ndarray,
    scored_rows: np.ndarray,
    description: dict,
) -> None:
    payload = {
        "run_id": run_id,
        "strategy": description.get("strategy") or description.get("requested_strategy") or "random_kfold",
        "row_order": "raw_train_file_order",
        "n_train_rows": int(len(fold_assignment)),
        "n_folds": int(description.get("n_folds", int(fold_assignment.max()) + 1 if len(fold_assignment) and fold_assignment.max() >= 0 else 0)),
        "fold_assignment": [int(x) for x in np.asarray(fold_assignment).tolist()],
        "scored_rows": [bool(x) for x in np.asarray(scored_rows).tolist()],
        "group_column": description.get("group_column"),
        "description": description,
    }
    # Forward/expanding CV needs explicit per-fold train semantics (train = periods
    # strictly before each fold's val block), which the plain fa==f/fa!=f rebuild
    # cannot express. Persist it top-level so load_canonical_folds can honour it.
    if description.get("fold_spec"):
        payload["fold_spec"] = description["fold_spec"]
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_canonical_folds(path: str | Path, *, valid_mask: np.ndarray | None = None) -> CanonicalFolds:
    """Reconstruct ``CanonicalFolds`` from ``{run_id}_cv_folds.json``. When the
    consumer filtered rows (e.g. dropped NaN targets), pass ``valid_mask`` (a bool
    array over the FULL train rows) so the folds re-align to the filtered frame."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fa_full = np.asarray(data["fold_assignment"], dtype=int)
    scored_full = np.asarray(data["scored_rows"], dtype=bool)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if len(vm) != len(fa_full):
            raise ValueError(f"valid_mask length {len(vm)} != fold_assignment length {len(fa_full)}")
        fa = fa_full[vm]
        scored = scored_full[vm]
    else:
        fa, scored = fa_full, scored_full
    k = int(data.get("n_folds") or (int(fa.max()) + 1 if len(fa) and fa.max() >= 0 else 0))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    fold_spec = data.get("fold_spec") or {}
    if fold_spec.get("type") == "expanding":
        # Forward expanding window: each fold trains on rows whose period rank is
        # strictly BEFORE its validation block (no forward peeking). Plain fa!=f
        # would wrongly pull later folds' val periods into training, so rebuild
        # train/val from the per-row period rank + per-fold [val_lo, val_hi).
        pr_full = np.asarray(fold_spec.get("period_rank", []), dtype=int)
        pr = pr_full[np.asarray(valid_mask, dtype=bool)] if valid_mask is not None else pr_full
        if len(pr) == len(fa):
            for fr in fold_spec.get("folds", []):
                lo, hi = int(fr["val_lo"]), int(fr["val_hi"])
                val_idx = np.flatnonzero((pr >= lo) & (pr < hi))
                tr_idx = np.flatnonzero((pr >= 0) & (pr < lo))
                if len(val_idx) >= 1 and len(tr_idx) >= 1:
                    folds.append((tr_idx, val_idx))
    if not folds:  # k-fold strategies (or expanding rebuild unavailable)
        for f in range(max(k, 1)):
            val_idx = np.flatnonzero(fa == f)
            tr_idx = np.flatnonzero(fa != f)
            if len(val_idx) >= 1 and len(tr_idx) >= 1:
                folds.append((tr_idx, val_idx))
    return CanonicalFolds(
        folds=folds,
        scored_mask=scored,
        strategy=str(data.get("strategy", "")),
        n_folds=len(folds),
        description=data.get("description", {}),
    )


def write_oof(path: str | Path, oof: np.ndarray, scored_mask: np.ndarray | None = None) -> None:
    """Persist a candidate's OOF predictions aligned to the (filtered) train row
    order as ``[row_index, oof_pred]``. Rows outside ``scored_mask`` / NaN are
    written as NaN so the keep-best owner aligns candidates by row_index."""
    oof = np.asarray(oof, dtype=float)
    idx = np.arange(len(oof))
    out = pd.DataFrame({"row_index": idx, "oof_pred": oof})
    if scored_mask is not None:
        out.loc[~np.asarray(scored_mask, dtype=bool), "oof_pred"] = np.nan
    out.to_csv(path, index=False)


# ── disk-level common-OOF NNLS keep-best (for the LLM keep-best owners) ─────────

def nnls_keep_best(
    candidates: list[dict],
    y_true: np.ndarray,
    *,
    metric_name: str,
    greater_is_better: bool,
    blocks: np.ndarray | None = None,
) -> dict:
    """Score every candidate's OOF on ONE metric, then NNLS-blend across candidates
    (sum-to-one weights applied to their test predictions). ``candidates`` is a list
    of ``{name, oof: 1d array, test: 1d array}`` already aligned to ``y_true`` rows.

    Returns ``{scores, best_single, blend_weights, blend_oof_score, choice,
    chosen_test, chosen_score}``. The blend is chosen only when it strictly beats the
    best single candidate; otherwise the best single wins (never regresses).
    """
    y_true = np.asarray(y_true, dtype=float)
    # A candidate is usable if its OOF aligns to y_true and has *some* finite rows.
    # Forward/expanding CV scores only the held-out future blocks, so OOF is NaN
    # elsewhere — we must NOT require all-finite (that would drop every candidate).
    usable = [c for c in candidates
              if c.get("oof") is not None and len(c["oof"]) == len(y_true)
              and np.isfinite(np.asarray(c["oof"], dtype=float)).any()]
    if not usable:
        return {"scores": {}, "choice": None, "chosen_test": None, "chosen_score": None}
    # Score every candidate on the COMMON finite rows (intersection of all OOF finite
    # masks) so best-single vs blend is apples-to-apples on the same row set. Under a
    # shared canonical scored_mask these masks coincide; the intersection is the safe
    # general rule.
    oofs = {c["name"]: np.asarray(c["oof"], dtype=float) for c in usable}
    common = np.isfinite(y_true)
    for arr in oofs.values():
        common &= np.isfinite(arr)
    if int(common.sum()) < 5:
        return {"scores": {}, "choice": None, "chosen_test": None, "chosen_score": None}
    yc = y_true[common]
    bc = (np.asarray(blocks)[common] if blocks is not None and len(blocks) == len(y_true) else None)
    scores = {c["name"]: _score(yc, oofs[c["name"]][common], metric_name, bc) for c in usable}
    best_single = (max if greater_is_better else min)(scores, key=lambda n: scores[n])
    result = {
        "scores": {k: round(float(v), 6) for k, v in scores.items()},
        "best_single": best_single,
        "blend_weights": None,
        "blend_oof_score": None,
        "choice": best_single,
        "chosen_test": next(c["test"] for c in usable if c["name"] == best_single),
        "chosen_score": float(scores[best_single]),
        "n_common_oof_rows": int(common.sum()),
    }
    if len(usable) < 2:
        return result
    try:
        from scipy.optimize import nnls
    except Exception:
        return result
    M = np.column_stack([oofs[c["name"]][common] for c in usable])
    try:
        w, _ = nnls(M, yc)
    except Exception:
        return result
    if not np.isfinite(w).all() or w.sum() <= 0:
        return result
    w = w / w.sum()
    blend_score = _score(yc, M @ w, metric_name, bc)
    better = (blend_score > scores[best_single]) if greater_is_better else (blend_score < scores[best_single])
    result["blend_weights"] = {c["name"]: round(float(wi), 6) for c, wi in zip(usable, w)}
    result["blend_oof_score"] = float(blend_score)
    if better:
        T = np.column_stack([np.asarray(c["test"], dtype=float) for c in usable])
        result["choice"] = "blend(" + "+".join(c["name"] for c in usable) + ")"
        result["chosen_test"] = T @ w
        result["chosen_score"] = float(blend_score)
    return result


def _score(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str, blocks: np.ndarray | None) -> float:
    err = np.abs(y_true - y_pred)
    if metric_name == "rmse":
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if metric_name == "block_mae" and blocks is not None and len(blocks) == len(y_true):
        return float(pd.DataFrame({"e": err, "b": blocks}).groupby("b")["e"].mean().mean())
    return float(np.mean(err))
