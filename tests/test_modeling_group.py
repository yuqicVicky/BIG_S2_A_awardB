"""Regression tests for the auto-run, keep-best modeling group (Step 2).

Covered:
- the per-family CV-score keep-best logic (swap only on a strict improvement),
- the wall-clock budget guard (specialists skipped when budget is exhausted),
- the disable toggle (AWARDB_MODELING_GROUP=0),
- metric-direction handling for both lower-is-better and greater-is-better,
- the `families` filter wired through the *classification* path (parity with
  regression), verified on a synthetic, dataset-agnostic bundle.

All synthetic — no dependence on whatever happens to sit under data/.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data_agent import modeling_group as MG
from src.data_agent.features import FeatureBundle
from src.data_agent.models import ModelResult, _candidate_family, train_and_predict
from src.data_agent.skills.modeling import ModelingResult
from src.data_agent.task import BINARY, TaskSpec


# ── fakes ──────────────────────────────────────────────────────────────────────

def _fake_modeling(name: str, score: float, *, greater: bool, n_pred: int = 6,
                   metric: str = "block_mae", output_kind: str = "value"):
    mr = SimpleNamespace(
        selected_model_name=name,
        model_scores=[{"name": name, "status": "ok", "score": score}],
        greater_is_better=greater,
        metric_name=metric,
        predictions=np.arange(n_pred, dtype=float),
        output_kind=output_kind,
        holdout_strategy={"type": "grouped_kfold"},
        residual_analysis={},
    )
    return SimpleNamespace(
        model_result_obj=mr,
        model_results={"all_scores": mr.model_scores, "selected_model_name": name},
        predictions=mr.predictions,
        holdout={},
    )


def _floor_with_all_families(selected: str, score: float, *, greater: bool):
    """A floor whose pool spans gbdt/linear/trees so all three are dispatched. The
    *selected* model carries exactly ``score``; the others are strictly worse."""
    worse = (lambda d: score - d) if greater else (lambda d: score + d)
    pool = {"lightgbm": worse(0.5), "ridge": worse(0.3), "random_forest": worse(0.7)}
    pool[selected] = score  # selected gets exactly the floor score
    fm = _fake_modeling(selected, score, greater=greater)
    fm.model_result_obj.model_scores = [
        {"name": name, "status": "ok", "score": value} for name, value in pool.items()
    ]
    fm.model_result_obj.selected_model_name = selected
    return fm


def _schema():
    return SimpleNamespace(block_column=None, row_id_column="id", target_column="y")


def _bundle():
    return SimpleNamespace(sample_submission=pd.DataFrame({"id": range(6), "y": 0}))


def _run(tmp_path, floor, patch_map, *, t_start=None, monkeypatch=None):
    """Drive run_modeling_group with train_and_evaluate_models patched to return a
    canned candidate per requested family."""
    def fake_train(*, bundle, block_column, random_state, families):
        fam = next(iter(families))
        if fam not in patch_map:
            # mimic a family that fell back to the full pool (no real candidate)
            return _fake_modeling("ridge", 999.0, greater=floor.model_result_obj.greater_is_better)
        name, score = patch_map[fam]
        return _fake_modeling(name, score, greater=floor.model_result_obj.greater_is_better)

    monkeypatch.setattr(MG, "train_and_evaluate_models", fake_train)
    return MG.run_modeling_group(
        bundle=_bundle(), schema=_schema(), description="",
        floor_modeling=floor, repo_root=tmp_path, run_id="t",
        logs_dir=tmp_path, t_start=t_start if t_start is not None else time.monotonic(),
    )


# ── pure helpers ───────────────────────────────────────────────────────────────

class TestHelpers:
    def test_is_better_lower(self):
        assert MG._is_better(0.8, 0.9, False)
        assert not MG._is_better(0.9, 0.8, False)
        assert not MG._is_better(0.9, 0.9, False)  # tie keeps incumbent

    def test_is_better_greater(self):
        assert MG._is_better(0.9, 0.8, True)
        assert not MG._is_better(0.8, 0.9, True)

    def test_is_better_handles_nonfinite(self):
        assert not MG._is_better(float("nan"), 0.9, False)
        assert MG._is_better(0.5, None, False)  # any finite beats a missing incumbent

    def test_selected_cv_score_prefers_selected(self):
        mr = SimpleNamespace(
            selected_model_name="b", greater_is_better=False,
            model_scores=[{"name": "a", "status": "ok", "score": 1.0},
                          {"name": "b", "status": "ok", "score": 2.0}])
        assert MG._selected_cv_score(mr) == 2.0

    def test_selected_cv_score_fallback_to_best(self):
        mr = SimpleNamespace(
            selected_model_name="missing", greater_is_better=False,
            model_scores=[{"name": "a", "status": "ok", "score": 1.0},
                          {"name": "b", "status": "ok", "score": 2.0}])
        assert MG._selected_cv_score(mr) == 1.0  # lower-is-better best

    def test_available_families(self):
        mr = SimpleNamespace(model_scores=[
            {"name": "lightgbm", "status": "ok", "score": 1.0},
            {"name": "ridge", "status": "ok", "score": 1.0},
            {"name": "dummy_mean", "status": "ok", "score": 9.0}])
        # diverse-first order (linear before gbdt); trees absent from this pool
        assert MG._available_families(mr) == ["linear", "gbdt"]

    def test_finite_predictions(self):
        ok = SimpleNamespace(predictions=np.array([1.0, 2.0, 3.0]))
        bad = SimpleNamespace(predictions=np.array([1.0, np.nan]))
        assert MG._finite_predictions(ok)
        assert not MG._finite_predictions(bad)


# ── keep-best behaviour ─────────────────────────────────────────────────────────

class TestKeepBest:
    def test_swaps_to_strictly_better_specialist(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AWARDB_MODELING_GROUP", raising=False)
        floor = _floor_with_all_families("ridge", 0.90, greater=False)
        best, meta = _run(tmp_path, floor, {
            "gbdt": ("lightgbm", 0.80),   # better
            "linear": ("ridge", 0.95),    # worse
            "trees": ("random_forest", 1.10),  # worse
        }, monkeypatch=monkeypatch)
        assert meta["choice"] == "gbdt"
        assert meta["chosen_cv_score"] == 0.80
        assert best.model_result_obj.selected_model_name == "lightgbm"
        assert (tmp_path / "t_ensemble_meta.json").exists()

    def test_keeps_floor_when_no_specialist_improves(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AWARDB_MODELING_GROUP", raising=False)
        floor = _floor_with_all_families("ridge", 0.90, greater=False)
        best, meta = _run(tmp_path, floor, {
            "gbdt": ("lightgbm", 0.95),
            "linear": ("ridge", 0.91),
            "trees": ("random_forest", 1.10),
        }, monkeypatch=monkeypatch)
        assert meta["choice"] == "floor"
        assert best is floor

    def test_greater_is_better_direction(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AWARDB_MODELING_GROUP", raising=False)
        floor = _floor_with_all_families("ridge", 0.80, greater=True)  # accuracy
        best, meta = _run(tmp_path, floor, {
            "gbdt": ("lightgbm", 0.85),   # higher is better → wins
            "linear": ("ridge", 0.78),
            "trees": ("random_forest", 0.70),
        }, monkeypatch=monkeypatch)
        assert meta["choice"] == "gbdt" and meta["chosen_cv_score"] == 0.85

    def test_budget_exhausted_skips_all(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AWARDB_MODELING_GROUP", raising=False)
        monkeypatch.setenv("AWARDB_TIME_BUDGET_SEC", "5400")
        floor = _floor_with_all_families("ridge", 0.90, greater=False)
        best, meta = _run(tmp_path, floor, {"gbdt": ("lightgbm", 0.10)},
                          t_start=time.monotonic() - 10_000, monkeypatch=monkeypatch)
        assert meta["choice"] == "floor"  # nothing ran despite a "better" candidate
        assert all(v.get("status") == "skipped" for v in meta["specialists"].values())

    def test_disabled_toggle(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AWARDB_MODELING_GROUP", "0")
        floor = _floor_with_all_families("ridge", 0.90, greater=False)
        best, meta = _run(tmp_path, floor, {"gbdt": ("lightgbm", 0.10)},
                          monkeypatch=monkeypatch)
        assert meta["choice"] == "floor" and best is floor
        assert "disabled" in meta.get("note", "")

    def test_ignores_specialist_that_falls_back_to_full_pool(self, tmp_path, monkeypatch):
        # When a requested family has no real candidate, the engine returns the full
        # pool (selected model from another family). The group must not count it.
        monkeypatch.delenv("AWARDB_MODELING_GROUP", raising=False)
        floor = _floor_with_all_families("ridge", 0.90, greater=False)
        # gbdt resolves to a linear model (ridge) → "unavailable", not counted.
        best, meta = _run(tmp_path, floor, {
            "gbdt": ("ridge", 0.10),  # better score but WRONG family → ignored
        }, monkeypatch=monkeypatch)
        assert meta["specialists"]["gbdt"]["status"] == "unavailable"
        assert meta["choice"] == "floor"


# ── families filter wired through the classification path ───────────────────────

def _clf_bundle(n: int = 160) -> FeatureBundle:
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = (x1 + 0.5 * x2 + rng.normal(scale=0.3, size=n) > 0).astype(int)
    train = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    predict = pd.DataFrame({"x1": rng.normal(size=24), "x2": rng.normal(size=24)})
    sample = pd.DataFrame({"id": range(24), "y": 0})
    task = TaskSpec(task_type=BINARY, metric="accuracy", greater_is_better=True,
                    output_kind="label", class_labels=[0, 1], positive_label=1)
    return FeatureBundle(
        train_df=train, predict_df=predict, sample_submission=sample, target=train["y"],
        feature_columns=["x1", "x2"], numeric_columns=["x1", "x2"], categorical_columns=[],
        task=task, profile={}, text_columns=[], group_aggregate_keys=[])


class TestClassificationFamilyFilter:
    def test_linear_family_excludes_gbdt(self):
        mr = train_and_predict(_clf_bundle(), families={"linear"})
        names = {s["name"] for s in mr.model_scores}
        assert "logistic_regression" in names
        assert "hist_gradient_boosting" not in names
        assert _candidate_family(mr.selected_model_name) in ("linear", "baseline")

    def test_gbdt_family_excludes_linear(self):
        mr = train_and_predict(_clf_bundle(), families={"gbdt"})
        names = {s["name"] for s in mr.model_scores}
        assert "hist_gradient_boosting" in names
        assert "logistic_regression" not in names
        assert _candidate_family(mr.selected_model_name) in ("gbdt", "baseline")

    def test_absent_family_falls_back_to_full_pool(self):
        # "trees" has no candidate in the lean classification pool → no filtering.
        mr = train_and_predict(_clf_bundle(), families={"trees"})
        names = {s["name"] for s in mr.model_scores}
        assert {"logistic_regression", "hist_gradient_boosting"} <= names


# ── cross-family convex blend ────────────────────────────────────────────────────

def _mk_modeling(y, oof, test_pred, name):
    """A minimal ModelingResult carrying aligned OOF arrays for blend tests."""
    mr = ModelResult(
        predictions=np.asarray(test_pred, dtype=float), selected_model_name=name,
        model_scores=[], holdout_strategy={}, metric_name="mae",
        greater_is_better=False, task_type="regression", output_kind="value",
        holdout_y_true=np.asarray(y, dtype=float), holdout_y_pred=np.asarray(oof, dtype=float))
    return ModelingResult(model_results={}, predictions=mr.predictions,
                          model_result_obj=mr, holdout={})


class TestBlend:
    def _bundle_schema(self, y, m):
        bundle = SimpleNamespace(
            target=pd.Series(y), train_df=pd.DataFrame({"x": y}),
            sample_submission=pd.DataFrame({"id": range(m), "t": range(m)}))
        schema = SimpleNamespace(block_column=None, row_id_column="id", target_column="t")
        return bundle, schema

    def test_blend_beats_floor_when_decorrelated(self):
        rng = np.random.default_rng(0)
        n, m = 200, 40
        y = rng.normal(50, 10, n)
        floor_oof = y + rng.normal(0, 6, n)
        cand_oof = y + rng.normal(0, 6, n)            # independent errors → blend helps
        floor = _mk_modeling(y, floor_oof, rng.normal(50, 10, m), "stack(floor)")
        cand = _mk_modeling(y, cand_oof, rng.normal(50, 10, m), "ridge")
        bundle, schema = self._bundle_schema(y, m)
        best = float(np.mean(np.abs(y - floor_oof)))
        res = MG._try_blend(floor, [("linear", cand)], bundle=bundle, schema=schema,
                            description="", metric_name="mae", greater=False,
                            best_score=best, run_id="t", logs_dir=None)
        assert res is not None
        blended_modeling, name, cv, w = res
        assert name.startswith("blend(") and cv < best          # strictly better by CV
        assert abs(sum(w.values()) - 1.0) < 1e-6                 # convex (sum-to-one)
        preds = blended_modeling.model_result_obj.predictions
        assert len(preds) == m and np.all(np.isfinite(preds))

    def test_blend_none_when_no_improvement(self):
        # A candidate identical to the floor can't beat it → keep-best returns None
        # (the blend can never regress the deliverable).
        rng = np.random.default_rng(1)
        n, m = 200, 40
        y = rng.normal(50, 10, n)
        floor_oof = y + rng.normal(0, 5, n)
        floor = _mk_modeling(y, floor_oof, rng.normal(50, 10, m), "stack(floor)")
        cand = _mk_modeling(y, floor_oof.copy(), rng.normal(50, 10, m), "ridge")
        bundle, schema = self._bundle_schema(y, m)
        best = float(np.mean(np.abs(y - floor_oof)))
        res = MG._try_blend(floor, [("linear", cand)], bundle=bundle, schema=schema,
                            description="", metric_name="mae", greater=False,
                            best_score=best, run_id="t", logs_dir=None)
        assert res is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
