"""Multi-dataset generalization harness + closed-loop gate tests.

Tiny synthetic fixtures spanning regression / log-scale-regression / binary /
multiclass / imbalanced / the-original-bug shape, plus direct tests of the
deterministic critics and the stage-gate runner in ``gates.py``. The point is to
prove the agent's decisions generalize *without* any dataset-specific names — so
every fixture is built in-test and none of them is House Prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_agent import task as T
from src.data_agent.gates import (
    FAIL,
    PASS,
    WARN,
    Verdict,
    check_prediction_sanity,
    check_schema,
    check_task_consistency,
    load_llm_verdict,
    run_stage_with_gate,
    write_verdict,
)
from src.data_agent.skills.infer_task import infer_task_spec


# ── task generalization across dataset shapes ────────────────────────────────

class TestTaskGeneralization:
    def test_regression_rmse(self):
        y = pd.Series(np.random.default_rng(0).normal(50, 12, 300), name="y")
        spec = T.resolve_task_spec("Regression task scored by RMSE.", y, None, target_column="y")
        assert (spec.task_type, spec.metric, spec.output_kind) == (T.REGRESSION, T.RMSE, "value")

    def test_regression_log_scale(self):
        y = pd.Series(np.random.default_rng(1).integers(10_000, 900_000, size=300), name="price")
        sample = pd.Series(np.random.default_rng(2).integers(10_000, 900_000, size=120))
        spec = T.resolve_task_spec(
            "Regression task. Scored on root mean squared error of log-transformed prices.",
            y, sample, target_column="price")
        assert spec.task_type == T.REGRESSION
        # log-scale (RMSLE) special-casing was removed for Award B (always
        # block-MAE); a "root mean squared error …" description now resolves to RMSE.
        assert spec.metric == T.RMSE

    def test_binary(self):
        y = pd.Series([0, 1] * 120, name="y")
        sample = pd.Series([0, 1] * 60)
        spec = T.resolve_task_spec("Binary classification scored by accuracy.", y, sample, target_column="y")
        assert spec.task_type == T.BINARY
        assert spec.output_kind == "label"

    def test_binary_probability_output(self):
        y = pd.Series([0, 1] * 120, name="y")
        sample = pd.Series(np.random.default_rng(3).uniform(0, 1, 60))  # float probs
        spec = T.resolve_task_spec(
            "Binary classification. Submit the predicted probability. Scored by ROC AUC.",
            y, sample, target_column="y")
        assert spec.task_type == T.BINARY
        assert spec.metric == T.ROC_AUC
        assert spec.output_kind == "probability"

    def test_multiclass(self):
        y = pd.Series([0, 1, 2, 3, 4] * 40, name="y")
        sample = pd.Series([0, 1, 2, 3, 4] * 10)
        spec = T.resolve_task_spec("Multiclass classification of the label.", y, sample, target_column="y")
        assert spec.task_type == T.MULTICLASS

    def test_imbalanced_binary_flag(self):
        df = pd.DataFrame({"f1": range(100), "y": [0] * 90 + [1] * 10})
        spec = infer_task_spec("predict y", df, {"columns": ["f1", "y"]}, schema_target="y")
        assert spec.task_type == T.BINARY
        assert spec.class_imbalance_detected is True
        assert abs(spec.majority_class_rate - 0.90) < 1e-9
        assert "roc_auc" in spec.recommended_metrics

    def test_the_original_bug_high_card_with_incidental_classification_prose(self):
        # A regression brief whose *feature* prose contains the word
        # "classification" — the exact shape that used to flip the task to a
        # hundreds-of-classes multiclass problem.
        rng = np.random.default_rng(7)
        y = pd.Series(rng.integers(35_000, 760_000, size=400), name="SalePrice")  # ~continuous int
        sample = pd.Series(rng.integers(35_000, 760_000, size=200))
        desc = (
            "This is a supervised tabular regression task. Predict the continuous SalePrice. "
            "The MSZoning feature gives the general zoning classification of the property. "
            "Official scoring: root mean squared error on log-transformed sale prices."
        )
        spec = T.resolve_task_spec(desc, y, sample, target_column="SalePrice")
        assert spec.task_type == T.REGRESSION, spec.evidence
        assert spec.metric == T.RMSE
        # and the consistency gate is happy with the correct resolution
        v = check_task_consistency(
            task_type=spec.task_type, metric=spec.metric, output_kind=spec.output_kind,
            target_series=y, sample_target_series=sample, description_text=desc)
        assert v.status == PASS, v.reasons


# ── deterministic task-consistency gate ──────────────────────────────────────

class TestConsistencyGate:
    def _reg_target(self):
        return pd.Series(np.random.default_rng(0).normal(100, 30, 300), name="y")

    def test_clean_regression_passes(self):
        y = self._reg_target()
        v = check_task_consistency(task_type=T.REGRESSION, metric=T.RMSE, output_kind="value",
                                   target_series=y, description_text="regression scored by rmse")
        assert v.status == PASS and not v.suggested_corrections

    def test_clean_binary_passes(self):
        y = pd.Series([0, 1] * 100, name="y")
        v = check_task_consistency(task_type=T.BINARY, metric=T.ACCURACY, output_kind="label",
                                   target_series=y, sample_target_series=pd.Series([0, 1] * 50),
                                   description_text="binary classification scored by accuracy")
        assert v.status == PASS

    def test_regression_metric_with_classification_task_fails(self):
        # the original-bug signature: regression metric, multiclass task_type
        y = pd.Series(np.random.default_rng(1).integers(0, 600, size=400), name="y")
        v = check_task_consistency(task_type=T.MULTICLASS, metric=T.RMSE, output_kind="label",
                                   target_series=y, description_text="scored by rmse")
        assert v.status == FAIL
        assert v.suggested_corrections.get("force_task_type") == T.REGRESSION

    def test_classification_metric_with_regression_task_fails(self):
        y = pd.Series([0, 1] * 150, name="y")
        v = check_task_consistency(task_type=T.REGRESSION, metric=T.MAE, output_kind="value",
                                   target_series=y, description_text="scored by accuracy")
        # description metric accuracy implies classification while task is regression
        assert v.status == FAIL
        assert v.suggested_corrections.get("force_task_type") in (T.BINARY, T.MULTICLASS)

    def test_classification_task_on_continuous_data_fails(self):
        y = pd.Series(np.random.default_rng(2).normal(0, 1, 400), name="y")  # continuous floats
        v = check_task_consistency(task_type=T.MULTICLASS, metric=T.ACCURACY, output_kind="label",
                                   target_series=y)
        assert v.status == FAIL
        assert v.suggested_corrections.get("force_task_type") == T.REGRESSION

    def test_probability_output_vs_label_sample_warns(self):
        y = pd.Series([0, 1] * 100, name="y")
        v = check_task_consistency(task_type=T.BINARY, metric=T.ROC_AUC, output_kind="probability",
                                   target_series=y, sample_target_series=pd.Series([0, 1] * 50))
        assert v.status == WARN

    def test_closed_loop_converges_on_contradiction(self):
        # The orchestrator's closed loop: a contradictory spec (multiclass task +
        # regression metric) fails the gate, the force_task_type correction is
        # applied via resolve_task_spec, and the re-checked verdict then passes.
        y = pd.Series(np.random.default_rng(0).integers(0, 500, 400), name="y")
        v1 = check_task_consistency(task_type=T.MULTICLASS, metric=T.RMSE, output_kind="label",
                                    target_series=y, description_text="scored by rmse")
        assert v1.status == FAIL
        forced = v1.suggested_corrections["force_task_type"]
        spec2 = T.resolve_task_spec("scored by rmse", y, None, target_column="y", force_task_type=forced)
        assert spec2.task_type == T.REGRESSION
        v2 = check_task_consistency(task_type=spec2.task_type, metric=spec2.metric,
                                    output_kind=spec2.output_kind, target_series=y,
                                    description_text="scored by rmse")
        assert v2.status == PASS  # loop converged, no further correction needed


# ── prediction sanity gate ───────────────────────────────────────────────────

class TestPredictionSanity:
    def test_clean_regression_passes(self):
        tt = np.random.default_rng(0).normal(100, 20, 300)
        preds = np.random.default_rng(1).normal(100, 18, 120)
        v = check_prediction_sanity(predictions=preds, train_target=tt, task_type=T.REGRESSION,
                                    baseline_score=10.0, candidate_score=5.0, metric_name="rmse")
        assert v.status == PASS

    def test_near_constant_fails(self):
        tt = np.random.default_rng(0).normal(100, 20, 300)
        preds = np.full(120, 100.0)
        v = check_prediction_sanity(predictions=preds, train_target=tt, task_type=T.REGRESSION)
        assert v.status == FAIL
        assert v.suggested_corrections.get("reexamine_model_pool") is True

    def test_non_finite_fails(self):
        tt = np.random.default_rng(0).normal(100, 20, 300)
        preds = np.array([1.0, np.nan, 3.0, np.inf] * 5)
        v = check_prediction_sanity(predictions=preds, train_target=tt, task_type=T.REGRESSION)
        assert v.status == FAIL
        assert v.suggested_corrections.get("sanitize_predictions") is True

    def test_no_baseline_lift_warns(self):
        tt = np.random.default_rng(0).normal(100, 20, 300)
        preds = np.random.default_rng(1).normal(100, 18, 120)
        v = check_prediction_sanity(predictions=preds, train_target=tt, task_type=T.REGRESSION,
                                    baseline_score=5.0, candidate_score=6.0, metric_name="rmse")
        assert v.status == WARN

    def test_classification_all_same_label_warns(self):
        v = check_prediction_sanity(predictions=np.array(["a"] * 50), train_target=["a", "b"] * 25,
                                    task_type=T.BINARY)
        assert v.status == WARN


# ── schema gate ──────────────────────────────────────────────────────────────

class TestSchemaGate:
    def test_good_schema_passes(self):
        ss = pd.DataFrame({"Id": [1, 2, 3], "y": [0, 0, 0]})
        v = check_schema(sample_submission=ss, row_id_column="Id", target_column="y",
                         train_columns=["Id", "f1", "y"])
        assert v.status == PASS

    def test_missing_sample_submission_fails(self):
        v = check_schema(sample_submission=None, row_id_column="Id", target_column="y")
        assert v.status == FAIL

    def test_rowid_not_in_sample_warns(self):
        ss = pd.DataFrame({"row": [1, 2], "y": [0, 0]})
        v = check_schema(sample_submission=ss, row_id_column="Id", target_column="y")
        assert v.status == WARN


# ── the stage-gate runner ────────────────────────────────────────────────────

class TestStageGateRunner:
    def test_pass_through_no_rerun(self):
        out, v = run_stage_with_gate(
            "s", produce_fn=lambda: "ok",
            critic_fn=lambda o: Verdict("s", PASS), apply_correction_fn=None)
        assert out == "ok" and v.status == PASS and v.checked["corrective_reruns"] == 0

    def test_fail_then_correct_reruns_to_pass(self):
        state = {"fixed": False}

        def produce():
            return "good" if state["fixed"] else "bad"

        def critic(o):
            return Verdict("s", PASS) if o == "good" else Verdict(
                "s", FAIL, reasons=["bad"], suggested_corrections={"fix": True})

        def apply(corr):
            state["fixed"] = True
            return True

        out, v = run_stage_with_gate("s", produce, critic, apply, max_retries=1)
        assert out == "good" and v.status == PASS and v.checked["corrective_reruns"] == 1

    def test_correction_declined_stops(self):
        out, v = run_stage_with_gate(
            "s", produce_fn=lambda: "bad",
            critic_fn=lambda o: Verdict("s", FAIL, reasons=["x"], suggested_corrections={"k": 1}),
            apply_correction_fn=lambda corr: False, max_retries=1)
        assert v.status == FAIL and v.checked["corrective_reruns"] == 0

    def test_no_progress_guard(self):
        out, v = run_stage_with_gate(
            "s", produce_fn=lambda: "bad",
            critic_fn=lambda o: Verdict("s", FAIL, reasons=["same"], suggested_corrections={"k": 1}),
            apply_correction_fn=lambda corr: True, max_retries=2)
        assert v.checked["corrective_reruns"] == 1
        assert any("no-progress" in r for r in v.reasons)

    def test_llm_verdict_takes_precedence(self):
        out, v = run_stage_with_gate(
            "s", produce_fn=lambda: "x",
            critic_fn=lambda o: Verdict("s", PASS),  # deterministic says pass
            apply_correction_fn=None, max_retries=0,
            llm_verdict_loader=lambda stage: Verdict("s", FAIL, reasons=["llm"], critic="llm:test"))
        assert v.status == FAIL and v.critic == "llm:test"

    def test_verdict_written_to_disk(self, tmp_path):
        run_stage_with_gate(
            "s", produce_fn=lambda: "x", critic_fn=lambda o: Verdict("s", WARN, reasons=["note"]),
            apply_correction_fn=None, logs_dir=tmp_path, run_id="RID")
        assert (tmp_path / "RID_gate_s.json").exists()


# ── verdict persistence + aliases + LLM round-trip ───────────────────────────

class TestVerdictIO:
    def test_alias_written_for_prediction_sanity(self, tmp_path):
        write_verdict(Verdict("prediction_sanity", PASS), tmp_path, "RID")
        assert (tmp_path / "RID_gate_prediction_sanity.json").exists()
        assert (tmp_path / "prediction_sanity.json").exists()  # fixed-name alias

    def test_llm_verdict_roundtrip(self, tmp_path):
        import json
        payload = {"stage": "report", "status": "fail", "reasons": ["stale metric"],
                   "suggested_corrections": {}, "checked": {}, "critic": "llm:report-writer-reviewer"}
        (tmp_path / "RID_llm_gate_report.json").write_text(json.dumps(payload))
        v = load_llm_verdict(tmp_path, "RID", "report")
        assert v is not None and v.status == FAIL and v.critic == "llm:report-writer-reviewer"

    def test_llm_verdict_absent_returns_none(self, tmp_path):
        assert load_llm_verdict(tmp_path, "RID", "schema") is None


# ── M5: modeling / feature generalization knobs ──────────────────────────────

class TestModelingKnobs:
    def test_time_holdout_orders_chronologically_not_lexicographically(self):
        from src.data_agent import models as M
        n = 200
        period = np.repeat(np.arange(1, 11), 20)  # integer periods 1..10
        df = pd.DataFrame({"period": period, "x": np.random.default_rng(0).normal(size=n)})
        y = pd.Series(np.random.default_rng(1).normal(size=n))
        _, _, strat = M._make_holdout_split(df, ["x"], y, "period", 42)
        assert strat["type"] == "time_holdout"
        assert strat["ordering"] in ("numeric", "datetime")
        hv = set(strat["holdout_values"])
        # latest periods held out: a chronological order keeps 10; the old
        # lexicographic sort ("1","10","2",...,"9") would have kept "8" instead.
        assert "10" in hv and "8" not in hv

    def test_ohe_groups_rare_and_handles_unseen(self):
        from src.data_agent.models import SKLEARN_AVAILABLE, _one_hot_encoder
        if not SKLEARN_AVAILABLE:
            import pytest
            pytest.skip("sklearn unavailable")
        enc = _one_hot_encoder()
        X = pd.DataFrame({"c": ["A"] * 300 + ["B"] * 200 + [f"r{i}" for i in range(100)]})
        enc.fit(X)
        width = enc.transform(X).shape[1]
        assert 2 <= width <= 51  # grouped/capped — never the 102 distinct levels
        # an unseen category transforms without error and keeps the same width
        r = enc.transform(pd.DataFrame({"c": ["unseen_xyz"]}))
        assert r.shape == (1, width)

    def test_datetime_detected_when_values_appear_after_first_rows(self):
        from src.data_agent.features import _detect_datetime_columns
        # first 250 non-null values are non-date tokens; real dates follow — a
        # head(200) sample would miss the column, a spread sample catches it.
        col = [f"tok{i}" for i in range(250)] + [f"2021-{(i % 12) + 1:02d}-15" for i in range(750)]
        df = pd.DataFrame({"d": col, "x": range(len(col))})
        assert "d" in _detect_datetime_columns(df)

    def test_interactions_ranked_by_target_correlation(self):
        from src.data_agent.features import _add_interaction_features
        rng = np.random.default_rng(0)
        n = 400
        hour = rng.integers(0, 24, n)
        cols = {"t__hour": hour}
        for i in range(8):  # 8 low-card noise features (similar variance)
            cols[f"lc{i}"] = rng.integers(0, 3, n)
        signal = rng.integers(0, 3, n)
        cols["lc_signal"] = signal
        target = pd.Series(hour * signal * 4.0 + rng.normal(0, 1, n), name="y")
        train = pd.DataFrame(cols)
        pred = train.iloc[:40].copy()
        _, _, added = _add_interaction_features(train, pred, list(cols), 0.0, target=target)
        assert len(added) == 6  # capped at max_interactions
        # 9 candidate pairs > 6, so ranking decides; the target-correlated pair
        # must be among those kept (variance ranking would not reliably keep it).
        assert "t__hour_x_lc_signal" in added
