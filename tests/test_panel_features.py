"""Regression tests for the Award-B panel upgrades: leakage-safe group/target
aggregates, GroupKFold CV, free-text detection/routing, group-key discovery,
convex stacking, candidate family tags, and monotonic post-processing.

All fixtures are tiny + synthetic and dataset-agnostic (no overdose names), so
they double as the "different-domain panel" generalization guard.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.data_agent import features as F
from src.data_agent import models as M
from src.data_agent.runner import apply_monotonic_constraints


class TestGroupTargetAggregator:
    def test_adds_columns_and_global_fallback(self):
        X = pd.DataFrame({"g": ["a", "a", "b", "b", "c"], "x": [1, 2, 3, 4, 5]})
        y = pd.Series([10.0, 12.0, 20.0, 22.0, 30.0])
        agg = M.GroupTargetAggregator([["g"]]).fit(X, y)
        out = agg.transform(pd.DataFrame({"g": ["a", "b", "zzz"], "x": [0, 0, 0]}))
        mean_col = M._agg_col_name(["g"], "mean")
        assert mean_col in out.columns
        vals = out[mean_col].tolist()
        assert abs(vals[0] - 11.0) < 1e-6 and abs(vals[1] - 21.0) < 1e-6
        assert abs(vals[2] - float(y.mean())) < 1e-6  # unseen group -> global mean
        # count for an unseen group folds to 0
        assert out[M._agg_col_name(["g"], "count")].tolist()[2] == 0.0

    def test_leakage_safe_when_fit_on_subset(self):
        # Fit on a subset only; transform must reflect just that subset's target.
        X = pd.DataFrame({"g": ["a"] * 10})
        y = pd.Series(np.arange(10, dtype=float))
        agg = M.GroupTargetAggregator([["g"]]).fit(X.iloc[:5], y.iloc[:5])
        out = agg.transform(X.iloc[5:])
        assert abs(out[M._agg_col_name(["g"], "mean")].iloc[0] - 2.0) < 1e-6  # mean(0..4)

    def test_aggregate_feature_names_count(self):
        names = M._aggregate_feature_names([["g"], ["g", "h"]])
        assert len(names) == 2 * len(M._AGG_STATS)


class TestCVFolds:
    def test_grouped_kfold_holds_whole_groups_out(self):
        groups = pd.Series(np.repeat(np.arange(10), 5).astype(str))
        folds, desc = M._make_cv_folds(50, groups, 0)
        assert desc["type"] == "grouped_kfold"
        for tr, va in folds:
            assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[va]))

    def test_kfold_when_no_groups(self):
        _, desc = M._make_cv_folds(50, None, 0)
        assert desc["type"] == "kfold"

    def test_make_cv_folds_rejects_unique_per_row(self):
        # A near-unique-per-row key (an hourly timestamp / row id) must NOT yield a
        # GroupKFold — that silently collapses into ordinary KFold. Fall back honestly.
        n = 60
        uniq = pd.Series([f"id{i}" for i in range(n)])  # nunique == n
        _, desc = M._make_cv_folds(n, uniq, 0)
        assert desc["type"] == "kfold"

    def test_resolve_coarsens_unique_per_row_datetime(self):
        # A unique-per-row datetime is coarsened to whole-period blocks so the blocked
        # CV actually holds periods out (the Award-B generalization guard).
        n = 180
        dt = pd.date_range("2021-01-01", periods=n, freq="2D")  # ~12 months, 1 row/value
        df = pd.DataFrame({"ts": dt.astype(str), "x": np.arange(n)})
        groups, reason = M._resolve_cv_groups(df, "ts", n)
        assert groups is not None and reason and "coarsened" in reason
        assert 3 <= groups.nunique() <= n // 2           # whole-period blocks, not 1/row
        folds, desc = M._make_cv_folds(n, groups, 0)
        assert desc["type"] == "grouped_kfold" and desc["n_groups"] < n
        for tr, va in folds:                              # whole blocks held out
            assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[va]))

    def test_resolve_keeps_healthy_block_as_is(self):
        # A moderate-cardinality category block is used directly (no coarsening).
        n = 100
        df = pd.DataFrame({"cat": (["a", "b", "c", "d"] * 25), "x": np.arange(n)})
        groups, reason = M._resolve_cv_groups(df, "cat", n)
        assert groups is not None and reason is None and groups.nunique() == 4


class TestTextDetection:
    def test_detects_multiword_text(self):
        df = pd.DataFrame({
            "txt": [f"press release number {i} announces a new policy update" for i in range(20)],
            "cat": (["x", "y"] * 10),
        })
        assert F._detect_text_columns(df, ["txt", "cat"]) == ["txt"]

    def test_detects_text_even_when_duplicated_by_join(self):
        # Many distinct multi-word values, each duplicated across rows (low unique
        # *ratio* but high unique *count*) — the join-duplication case. Still text.
        base = [f"state health department release {i} about treatment access programs" for i in range(15)]
        df = pd.DataFrame({"txt": base * 8})  # 120 rows, 15 distinct, each x8
        assert F._detect_text_columns(df, ["txt"]) == ["txt"]

    def test_ignores_short_categorical(self):
        df = pd.DataFrame({"cat": ["red", "blue", "green"] * 10})
        assert F._detect_text_columns(df, ["cat"]) == []


class TestGroupKeyDiscovery:
    def test_excludes_disjoint_block_key(self):
        train = pd.DataFrame({
            "period": [f"p{i}" for i in range(10) for _ in range(3)],
            "jur": ["A", "B", "C"] * 10,
            "cat": ["m", "n", "o"] * 10,
            "t": range(30),
        })
        predict = pd.DataFrame({
            "period": [f"q{i}" for i in range(2) for _ in range(3)],  # disjoint periods
            "jur": ["A", "B", "C"] * 2,
            "cat": ["m", "n", "o"] * 2,
        })
        schema = SimpleNamespace(join_keys=["period", "jur"], category_column="cat",
                                 target_column="t", row_id_column="row_id", time_column="period")
        keys = F._discover_group_keys(train, predict, schema, block_key="period")
        flat = [k for grp in keys for k in grp]
        assert "period" not in flat            # disjoint block key never a group key
        assert ["jur"] in keys and ["cat"] in keys


class TestStacking:
    def test_convex_blend_beats_best_single(self):
        n = 200
        rng = np.random.default_rng(0)
        y = pd.Series(rng.normal(10, 3, n))
        oof = {"a": y.values + rng.normal(0, 1.0, n),
               "b": y.values + rng.normal(0, 1.4, n),
               "c": y.values + rng.normal(0, 6.0, n)}
        cs = {k: float(np.mean(np.abs(y.values - v))) for k, v in oof.items()}
        sidx = lambda idx, yt, yp: float(np.mean(np.abs(np.asarray(yt) - np.asarray(yp))))
        st = M._build_stack(oof, cs, y, greater=False, score_idx=sidx, metric_name="mae")
        assert st is not None
        assert abs(sum(st["weights"]) - 1.0) < 1e-6
        assert all(w >= -1e-9 for w in st["weights"])         # non-negative (convex)
        assert st["score"] <= min(cs.values()) + 1e-9          # beats best single


class TestCandidateFamily:
    def test_family_tags(self):
        assert M._candidate_family("lightgbm_strong") == "gbdt"
        assert M._candidate_family("catboost_log") == "gbdt"
        assert M._candidate_family("hist_gradient_boosting") == "gbdt"
        assert M._candidate_family("ridge") == "linear"
        assert M._candidate_family("elastic_net") == "linear"
        assert M._candidate_family("random_forest") == "trees"
        assert M._candidate_family("baseline_group_mean") == "baseline"


class TestMonotonic:
    def _frames(self):
        ss = pd.DataFrame({
            "period": ["p1"] * 3 + ["p2"] * 3,
            "jur": ["A"] * 6,
            "cat": ["all", "x", "y", "all", "x", "y"],
        })
        schema = SimpleNamespace(category_column="cat", join_keys=["period", "jur"],
                                 target_column="t", row_id_column="row_id")
        # 'all' has the largest mean (6) so it is inferred as the parent; p1 violates.
        preds = np.array([2.0, 5.0, 1.0, 10.0, 3.0, 4.0])
        return ss, schema, preds

    def test_enforces_parent_when_cue_present(self):
        ss, schema, preds = self._frames()
        out, info = apply_monotonic_constraints(
            preds, ss, schema, "Categories are nested and non-exclusive — never sum across them.")
        assert info["applied"] is True
        assert info["parent"] == "all"
        assert info["rows_adjusted"] == 1
        assert out[0] == 5.0          # p1 'all' raised to max(child)=5
        assert out[3] == 10.0         # p2 already satisfied, unchanged

    def test_noop_without_cue(self):
        ss, schema, preds = self._frames()
        out, info = apply_monotonic_constraints(preds, ss, schema, "A plain regression task.")
        assert info["applied"] is False
        assert np.allclose(out, preds)
