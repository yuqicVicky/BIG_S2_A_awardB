"""Staged orchestration — the primary execution path.

`run_orchestrated_analysis` threads an :class:`AnalysisState` through the workflow
stages (profile -> task -> plan -> critique -> eda -> leakage -> model -> evaluate ->
interpret -> submission -> report -> review), invoking the real skills and persisting
state after each stage. The Award-B submission is produced only via the proven
``runner`` helpers, and any stage failure falls back to the deterministic
``runner.run_analysis`` so the deliverable can never be lost.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from .features import build_feature_bundle
from .gates import (
    FAIL,
    PASS,
    WARN,
    Verdict,
    _worst,
    check_prediction_sanity,
    check_schema,
    check_task_consistency,
    load_llm_verdict,
    run_stage_with_gate,
    write_verdict,
)
from .pattern_analysis import run_pattern_analysis
from .runner import (
    _build_submission,
    _remove_stale_outputs,
    _run_id,
    _validate_submission,
    _write_json,
    apply_monotonic_constraints,
    run_analysis,
)
from .schema import discover_schema, read_table, write_schema_json
from .skills.analysis_planning import build_analysis_plan, critique_and_optimize_plan
from .skills.infer_task import infer_task_spec
from .skills.leakage_check import run_leakage_check
from .skills.model_evaluation import evaluate_model, plot_feature_importance
from .skills.modeling import train_and_evaluate_models
from .skills.profile_data import profile_tabular_data
from .skills.report_review import review_report
from .skills.report_writing import write_analysis_report
from .state import AnalysisState, UserRequest
from .task import BINARY, MULTICLASS, REGRESSION


def _baseline_checkpoint(bundle, schema):
    """Fast group-mean baseline predictions for the pre-search checkpoint
    submission. Uses overlapping group keys (jurisdiction/category-like) + the
    category dimension; dataset-agnostic. Returns ``None`` if not feasible."""
    from .models import GroupMeanRegressor

    y = pd.to_numeric(bundle.target, errors="coerce")
    mask = y.notna()
    if int(mask.sum()) < 1:
        return None
    keys = [
        k for k in (schema.join_keys or [])
        if k in bundle.train_df.columns and k in bundle.predict_df.columns and k != schema.time_column
    ]
    if schema.category_column and schema.category_column in bundle.predict_df.columns:
        keys = keys + [schema.category_column]
    keys = keys[:3]
    gm = GroupMeanRegressor(keys)
    gm.fit(bundle.train_df.loc[mask].reset_index(drop=True), y[mask].reset_index(drop=True))
    return np.asarray(gm.predict(bundle.predict_df), dtype=float)


def run_orchestrated_analysis(
    repo_root: Path,
    *,
    goal: str | None = None,
    file_path: str | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    run_id = _run_id(repo_root)
    logs_dir = repo_root / "outputs" / "logs"
    artifacts_dir = repo_root / "outputs" / "artifacts"
    for directory in (logs_dir, artifacts_dir, repo_root / "outputs" / "reports"):
        directory.mkdir(parents=True, exist_ok=True)

    data_dir = repo_root / "data"
    award_b = file_path is None and (data_dir / "DATA_DESCRIPTION.md").exists()

    try:
        if award_b:
            return _run_award_b(repo_root, run_id, logs_dir, artifacts_dir, goal, random_state)
        return _run_generic(repo_root, run_id, logs_dir, artifacts_dir, goal, file_path, random_state)
    except Exception as exc:  # safety net — never lose the deliverable
        print(f"[orchestrator] staged run failed ({type(exc).__name__}: {exc}); "
              f"falling back to deterministic runner.run_analysis")
        return run_analysis(repo_root)


# ── Award-B mode (schema-driven; produces submission.csv + report.pdf) ────────

def _clip_fraction(mr) -> float | None:
    """Fraction of final predictions sitting exactly on a clip bound (a heavy
    fraction signals the model is fighting the allowed range)."""
    try:
        p = np.asarray(mr.predictions, dtype=float)
        if not len(p):
            return None
        hits = np.zeros(len(p), dtype=bool)
        if mr.target_clip_min is not None:
            hits |= np.isclose(p, float(mr.target_clip_min))
        if mr.target_clip_max is not None:
            hits |= np.isclose(p, float(mr.target_clip_max))
        return float(hits.mean())
    except Exception:
        return None


def _emit_gate(logs_dir, run_id, stage, det_verdict):
    """Persist the effective verdict for a stage, preferring an LLM critic's
    verdict (``{run_id}_llm_gate_{stage}.json``) over the deterministic one when
    the Claude-driven path produced one. The headless path simply uses the
    deterministic verdict. Returns the effective verdict."""
    llm = load_llm_verdict(logs_dir, run_id, stage)
    effective = llm if llm is not None else det_verdict
    write_verdict(effective, logs_dir, run_id)
    return effective


def _run_award_b(repo_root, run_id, logs_dir, artifacts_dir, goal, random_state) -> dict:
    print(f"=== Award B orchestrated analysis: {run_id} ===")
    t_start = time.monotonic()  # wall-clock anchor for the Step-2 budget guard
    _remove_stale_outputs(repo_root)
    data_dir = repo_root / "data"
    description = (data_dir / "DATA_DESCRIPTION.md").read_text(encoding="utf-8", errors="replace")
    goal = goal or _goal_from_description(description)

    state = AnalysisState(
        run_id=run_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        user_request=UserRequest(goal=goal, file_path=str(data_dir)),
        data_path=str(data_dir),
    )

    # Stage 1 — schema + intake
    schema = discover_schema(data_dir)
    write_schema_json(schema, logs_dir / f"{run_id}_schema.json")
    bundle = build_feature_bundle(schema)
    train_df = bundle.train_df
    state.load_metadata = {
        "n_rows": int(len(train_df)), "n_columns": int(train_df.shape[1]),
        "file_format": Path(schema.train_target_file).suffix.lstrip("."), "encoding": "utf-8", "load_warnings": [],
    }
    state.set_raw_data(train_df)
    print(f"Target column: {schema.target_column} | task: {bundle.task.task_type}")

    # Gate — schema sanity (loud failure on a missing sample submission / bad ids)
    _schema_v = check_schema(
        sample_submission=bundle.sample_submission, row_id_column=schema.row_id_column,
        target_column=schema.target_column, train_columns=list(train_df.columns), run_id=run_id)
    _schema_v = _emit_gate(logs_dir, run_id, "schema", _schema_v)
    if _schema_v.failed:
        print(f"[gate:schema] FAIL: {_schema_v.reasons}")
    state.persist(logs_dir)

    # Stage 1b — data pattern analysis (deep EDA)
    try:
        pattern_report = run_pattern_analysis(
            schema, bundle, repo_root=repo_root, run_id=run_id
        )
        state.data_pattern_report = pattern_report
    except Exception as _pa_exc:
        print(f"[orchestrator] data pattern analysis failed ({_pa_exc}); continuing")
        state.data_pattern_report = None
    state.persist(logs_dir)

    # Stage 2 — profile
    profile = profile_tabular_data(train_df, schema.train_target_file)
    state.data_profile = profile.model_dump()
    state.persist(logs_dir)

    # Stage 3 — task inference (rich spec; engine subset == bundle.task)
    spec = infer_task_spec(
        goal, train_df, profile,
        description_text=description, schema_target=schema.target_column,
        sample_submission=bundle.sample_submission, has_block_col=bool(schema.block_column),
    )
    state.task_spec = spec.model_dump()
    # Augment task_spec with schema fields used by the report writer
    state.task_spec["row_id_column"] = schema.row_id_column
    print(f"Task: {spec.task_type} | metric: {spec.metric} | output: {spec.output_kind}")

    # Gate — task consistency (closed loop, bounded to one re-resolution). Cross-
    # checks the resolved task against the data, the description metric, and the
    # sample-submission format; on a hard contradiction it re-resolves the task
    # (and rebuilds the feature bundle modeling consumes) with a forced type.
    def _sample_target():
        ss = bundle.sample_submission
        return ss[schema.target_column] if schema.target_column in getattr(ss, "columns", []) else None

    def _consistency_verdict():
        return check_task_consistency(
            task_type=spec.task_type, metric=spec.metric, output_kind=spec.output_kind,
            target_series=train_df[schema.target_column], sample_target_series=_sample_target(),
            description_text=description, run_id=run_id)

    _tc = _consistency_verdict()
    if _tc.failed and _tc.suggested_corrections.get("force_task_type"):
        forced = _tc.suggested_corrections["force_task_type"]
        print(f"[gate:task_inference] FAIL → re-resolving with force_task_type={forced}: {_tc.reasons}")
        try:
            bundle = build_feature_bundle(schema, force_task_type=forced)
            train_df = bundle.train_df
            spec = infer_task_spec(
                goal, train_df, profile, description_text=description,
                schema_target=schema.target_column, sample_submission=bundle.sample_submission,
                has_block_col=bool(schema.block_column), force_task_type=forced)
            state.task_spec = spec.model_dump()
            state.task_spec["row_id_column"] = schema.row_id_column
            print(f"[gate:task_inference] re-resolved → task: {spec.task_type} | metric: {spec.metric}")
            _tc = _consistency_verdict()
            _tc.checked["corrective_reruns"] = 1
        except Exception as _tc_exc:
            print(f"[gate:task_inference] correction failed ({type(_tc_exc).__name__}: {_tc_exc}); keeping original spec")
            _tc.reasons.append(f"correction failed: {_tc_exc}")
    _tc = _emit_gate(logs_dir, run_id, "task_inference", _tc)
    state.persist(logs_dir)

    # Stage 4/5 — plan + critique
    state.initial_plan = build_analysis_plan(
        run_id=run_id, user_goal=goal, task_spec=state.task_spec, data_profile=state.data_profile)
    critique, optimized = critique_and_optimize_plan(
        initial_plan=state.initial_plan, task_spec=state.task_spec, data_profile=state.data_profile, user_goal=goal)
    state.optimized_plan = optimized
    state.persist(logs_dir)

    # Stage 7 — EDA summary
    state.eda_results = _eda_summary(profile.model_dump(), spec.model_dump())
    state.persist(logs_dir)

    # Stage 8 — leakage gate (does not abort Award-B: features come from the proven pipeline)
    # Scope the audit to bundle.feature_columns — the actual model feature set — so that
    # train-only columns (casual/registered/etc.) excluded by _choose_feature_columns are
    # not incorrectly flagged as leakage. Also exempt datetime-derived features from the
    # temporal-name check (they are valid prediction-time features, not leakage).
    _dt_features: set[str] = set()
    _generated = (bundle.profile.get("feature_audit") or {}).get("generated_time_features") or {}
    for feats in _generated.values():
        _dt_features.update(feats if isinstance(feats, list) else [])
    _leakage_spec = dict(state.task_spec or {})
    _leakage_spec["feature_candidates"] = bundle.feature_columns
    state.leakage_audit = run_leakage_check(
        "leakage_check", task_spec=_leakage_spec, data_profile=state.data_profile,
        df=train_df, run_id=run_id, datetime_derived_features=_dt_features,
    )
    # Gate — leakage (closed loop): a blocking failure naming a model feature drops
    # that feature before modeling. Non-blocking suspects are remembered so the
    # prediction-sanity gate can drop them on a degenerate-prediction rerun.
    _leak = state.leakage_audit
    _leak_suspect_cols = [
        s.get("column") for s in (_leak.get("suspected_columns") or [])
        if s.get("column") in bundle.feature_columns
    ]
    _leak_drop = [
        f.get("column") for f in (_leak.get("required_fixes") or [])
        if f.get("blocking") and f.get("column") in bundle.feature_columns
    ]
    _leak_reasons = [f"{f.get('check_name')}: {f.get('detail')}" for f in (_leak.get("failed_checks") or [])]
    _leak_status = FAIL if not _leak.get("approved") else (WARN if _leak.get("warned_checks") else PASS)
    if _leak_drop and len(bundle.feature_columns) - len(_leak_drop) >= 1:
        bundle.feature_columns = [c for c in bundle.feature_columns if c not in _leak_drop]
        bundle.numeric_columns = [c for c in bundle.numeric_columns if c in bundle.feature_columns]
        bundle.categorical_columns = [c for c in bundle.categorical_columns if c in bundle.feature_columns]
        bundle.text_columns = [c for c in bundle.text_columns if c in bundle.feature_columns]
        _leak_reasons.append(f"auto-dropped leakage-suspected columns before modeling: {_leak_drop}")
        _leak_status = WARN  # corrected
        print(f"[gate:leakage] auto-dropped {_leak_drop}")
    _leak_v = _emit_gate(logs_dir, run_id, "leakage", Verdict(
        stage="leakage", status=_leak_status, reasons=_leak_reasons,
        suggested_corrections={"drop_columns": _leak_drop} if _leak_drop else {},
        checked={"leakage_risk": _leak.get("leakage_risk"), "dropped": _leak_drop,
                 "suspects": _leak_suspect_cols}, run_id=run_id))
    _leak_status = _leak_v.status
    if _leak_status == FAIL:
        print(f"[gate:leakage] FAIL: {_leak.get('leakage_risk')} (continuing on remaining features)")
    state.persist(logs_dir)

    # Stage 9b — wire within-period info into bundle.profile so _make_holdout_split can use it
    if state.data_pattern_report:
        wpp = (state.data_pattern_report.get("time_coverage") or {}).get("within_period_pattern")
        if wpp:
            bundle.profile["within_period_info"] = wpp
            print(f"[orchestrator] within-period pattern detected: {wpp.get('feature')} "
                  f"train={wpp.get('train_range')} predict={wpp.get('predict_range')}")

    # Stage 10 — modeling under the prediction-sanity gate (closed loop, bounded
    # to one rerun). The critic flags degenerate / non-finite / no-lift / overfit
    # predictions; on a fail it drops remaining leakage-suspected features and
    # re-runs model selection once. An LLM critic verdict, if present, wins.

    # ── checkpoint: write a fast baseline submission up-front so a valid, scored
    # submission.csv always exists even if the heavy/aggressive search is later
    # cut off by the 2-hour cap ("partial output is scored"). Overwritten below
    # by the full model's prediction. Regression panels only.
    if bundle.task.task_type == REGRESSION:
        try:
            _ck = _baseline_checkpoint(bundle, schema)
            if _ck is not None:
                _ck, _ = apply_monotonic_constraints(_ck, bundle.sample_submission, schema, description)
                _ck_sub = _build_submission(bundle, schema.row_id_column, schema.target_column, _ck, "value")
                _ck_sub.to_csv(repo_root / "submission.csv", index=False)
                print("[checkpoint] baseline submission written (pre-search safety net)")
        except Exception as _ck_exc:
            print(f"[checkpoint] baseline checkpoint skipped: {_ck_exc}")

    _feat_state = {"columns": list(bundle.feature_columns)}

    def _produce_modeling():
        cols = _feat_state["columns"]
        bundle.feature_columns = cols
        bundle.numeric_columns = [c for c in bundle.numeric_columns if c in cols]
        bundle.categorical_columns = [c for c in bundle.categorical_columns if c in cols]
        bundle.text_columns = [c for c in bundle.text_columns if c in cols]
        return train_and_evaluate_models(
            bundle=bundle, block_column=schema.block_column, random_state=random_state)

    def _sanity_critic(modeling_):
        m = modeling_.model_result_obj
        base = (modeling_.model_results.get("baseline") or {}).get("selection_score")
        cand = (modeling_.model_results.get("candidate") or {}).get("selection_score")
        return check_prediction_sanity(
            predictions=m.predictions, train_target=bundle.target, task_type=m.task_type,
            holdout_y_true=m.holdout_y_true, holdout_y_pred=m.holdout_y_pred,
            metric_name=m.metric_name, baseline_score=base, candidate_score=cand,
            greater_is_better=m.greater_is_better, clip_fraction=_clip_fraction(m), run_id=run_id)

    def _sanity_correction(_corr):
        drop = [c for c in _leak_suspect_cols if c in _feat_state["columns"]]
        if not drop or len(_feat_state["columns"]) - len(drop) < 1:
            return False
        _feat_state["columns"] = [c for c in _feat_state["columns"] if c not in drop]
        print(f"[gate:prediction_sanity] retrying without suspected features: {drop}")
        return True

    modeling, _sanity_v = run_stage_with_gate(
        "prediction_sanity", _produce_modeling, _sanity_critic, _sanity_correction,
        max_retries=1, logs_dir=logs_dir, run_id=run_id,
        llm_verdict_loader=lambda s: load_llm_verdict(logs_dir, run_id, s))
    if not _sanity_v.ok:
        print(f"[gate:prediction_sanity] {_sanity_v.status.upper()}: {_sanity_v.reasons}")
    mr = modeling.model_result_obj
    state.model_results = modeling.model_results
    state.model_results["final_feature_columns"] = bundle.feature_columns
    state.raw_modeling_results = {"all_scores": modeling.model_results.get("all_scores", [])}
    state.split_metadata = mr.holdout_strategy
    state.residual_analysis = mr.residual_analysis or {}
    print(f"Selected model: {mr.selected_model_name} ({mr.metric_name})")

    # ── Step 2 — parallel modeling group (auto-run when budget remains) ─────────
    # Checkpoint the floor submission first so a valid, scored deliverable exists
    # on disk before the additional search (keep-best safety net), then dispatch
    # the family specialists. This runs automatically — no prompt — and is bounded
    # by AWARDB_TIME_BUDGET_SEC; a worse or interrupted group run keeps the floor.
    try:
        _floor_sub = _build_submission(
            bundle, schema.row_id_column, schema.target_column, mr.predictions, mr.output_kind)
        _floor_sub.to_csv(repo_root / "submission.csv", index=False)
        print("[modeling-group] floor submission checkpointed (keep-best safety net)")
    except Exception as _fl_exc:
        print(f"[modeling-group] floor checkpoint skipped: {_fl_exc}")
    try:
        from .modeling_group import run_modeling_group
        modeling, _meta = run_modeling_group(
            bundle=bundle, schema=schema, description=description,
            floor_modeling=modeling, repo_root=repo_root, run_id=run_id,
            logs_dir=logs_dir, t_start=t_start, random_state=random_state)
        state.model_results["modeling_group"] = _meta
        if _meta.get("choice") != "floor":
            mr = modeling.model_result_obj
            state.model_results = modeling.model_results
            state.model_results["final_feature_columns"] = bundle.feature_columns
            state.model_results["modeling_group"] = _meta
            state.raw_modeling_results = {"all_scores": modeling.model_results.get("all_scores", [])}
            state.split_metadata = mr.holdout_strategy
            state.residual_analysis = mr.residual_analysis or {}
            print(f"[modeling-group] keep-best → {_meta['choice']} beats floor "
                  f"({mr.metric_name}={_meta.get('chosen_cv_score')}); model={mr.selected_model_name}")
        else:
            print(f"[modeling-group] floor kept as best "
                  f"({mr.metric_name}={_meta.get('chosen_cv_score')}); "
                  f"candidates={_meta.get('candidates')}")
    except Exception as _mg_exc:
        print(f"[modeling-group] skipped ({type(_mg_exc).__name__}: {_mg_exc}); keeping floor")

    # Write validation_strategy.json
    _write_json(mr.holdout_strategy, logs_dir / f"{run_id}_validation_strategy.json")

    # Write model_stability_by_split.json and overfitting_audit.json
    stability = mr.holdout_strategy.get("stability", {})
    _write_json({
        "run_id": run_id,
        "selected_model": mr.selected_model_name,
        "primary_split_type": mr.holdout_strategy.get("type"),
        "stability": stability,
        "n_splits_evaluated": len(stability.get("split_scores", [])) if stability else 0,
    }, logs_dir / f"{run_id}_model_stability_by_split.json")

    # Write overfitting_audit.json: primary holdout score vs additional splits
    _overfit_audit = {
        "run_id": run_id,
        "selected_model": mr.selected_model_name,
        "metric": mr.metric_name,
        "primary_holdout_score": stability.get("split_scores", [None])[0] if stability else None,
        "stability": stability,
        "assessment": (
            "stable" if stability and stability.get("relative_stability") is not None
            and stability["relative_stability"] < 0.15 else
            "moderate_variance" if stability and stability.get("relative_stability") is not None
            and stability["relative_stability"] < 0.30 else
            "high_variance"
        ),
    }
    _write_json(_overfit_audit, logs_dir / f"{run_id}_overfitting_audit.json")
    # Phase-9 combined overfitting + leakage audit (named by the inspection
    # checklist). Generic: merges the stability assessment with the leakage gate.
    _write_json({
        **_overfit_audit,
        "leakage_risk": (state.leakage_audit or {}).get("leakage_risk"),
        "leakage_approved": (state.leakage_audit or {}).get("approved"),
        "leakage_suspected_columns": (state.leakage_audit or {}).get("suspected_columns", []),
    }, logs_dir / f"{run_id}_overfitting_leakage_audit.json")

    state.persist(logs_dir)

    # Stage 11 — evaluation (+ plots)
    hd = modeling.holdout
    state.evaluation = evaluate_model(
        task_type=hd["task_type"], y_true=hd["y_true"], y_pred=hd["y_pred"], y_proba=hd.get("y_proba"),
        baseline_y_pred=hd.get("baseline_y_pred"), baseline_y_proba=hd.get("baseline_y_proba"),
        class_labels=hd.get("label_classes"), positive_index=hd.get("positive_index", 1),
        run_id=run_id, artifacts_dir=artifacts_dir,
        selection_metric=mr.metric_name, candidate_name=mr.selected_model_name, baseline_name=hd.get("baseline_name", "baseline"),
    )
    for label, path in (state.evaluation.get("artifacts") or {}).items():
        if path:
            state.register_artifact(label, path)
    state.persist(logs_dir)

    # Stage 12 — interpretation (associative importance + plot)
    # Use bundle.feature_columns — the actual model feature set — not spec.feature_candidates,
    # which includes all train columns and may include train-only target-component columns.
    importances = _associative_importance(train_df, bundle.feature_columns, schema.target_column)
    fip = plot_feature_importance(importances, run_id, artifacts_dir) if importances else None
    if fip:
        state.register_artifact("feature_importance_plot", fip)
    state.interpretation = {
        "feature_importance": importances,
        "top_features": sorted(importances, key=lambda k: importances[k], reverse=True)[:10],
        "caveats": ["Importance is associative (|correlation| for numeric features), not causal."],
    }

    # Write feature_importance.json and profile.json (feature_audit)
    _write_json({
        "run_id": run_id,
        "final_model_features": bundle.feature_columns,
        "feature_importance": importances,
        "excluded_features": bundle.profile.get("excluded_columns", {}),
        "invariant_check": {
            "importance_subset_of_model": all(f in bundle.feature_columns for f in importances),
        },
    }, logs_dir / f"{run_id}_feature_importance.json")

    # Write _profile.json (feature_audit) — required by verification script
    _write_json(bundle.profile, logs_dir / f"{run_id}_profile.json")

    state.persist(logs_dir)

    # Monotonic/hierarchy post-processing (no-op unless the description declares
    # nested/total categories) — generic, no column names hardcoded.
    mr.predictions, _mono = apply_monotonic_constraints(
        mr.predictions, bundle.sample_submission, schema, description)
    if _mono.get("applied"):
        _write_json(_mono, logs_dir / f"{run_id}_monotonic_constraints.json")
        print(f"[monotonic] {_mono['parent']} >= children; rows adjusted: {_mono['rows_adjusted']}")

    # Build + validate submission (proven path)
    submission = _build_submission(bundle, schema.row_id_column, schema.target_column, mr.predictions, mr.output_kind)
    submission_path = repo_root / "submission.csv"
    submission.to_csv(submission_path, index=False)
    # Gate — submission format. _validate_submission raises on a hard failure
    # (which the outer handler turns into the proven deterministic fallback); we
    # log a verdict either way so the gate trail is complete.
    try:
        submission_check = _validate_submission(bundle.sample_submission, submission, schema.row_id_column, schema.target_column, mr.output_kind)
        write_verdict(Verdict(stage="submission", status=PASS,
                              checked={k: submission_check.get(k) for k in
                                       ("columns_ok", "row_count_ok", "row_id_alignment_ok", "all_finite", "dtype_ok")},
                              run_id=run_id), logs_dir, run_id)
    except Exception as _sub_exc:
        write_verdict(Verdict(stage="submission", status=FAIL, reasons=[str(_sub_exc)], run_id=run_id), logs_dir, run_id)
        raise
    _write_json(submission_check, logs_dir / f"{run_id}_submission_check.json")
    print(f"Submission written: {submission_path}")

    # Write prediction_distribution.json
    try:
        y_train_num = pd.to_numeric(bundle.target, errors="coerce").dropna()
        val_preds = mr.holdout_y_pred
        final_preds = mr.predictions
        def _quantiles(arr) -> dict:
            a = np.asarray(arr, dtype=float)
            a = a[np.isfinite(a)]
            if len(a) == 0:
                return {}
            return {f"p{int(q*100)}": float(np.percentile(a, q*100)) for q in [0.05, 0.25, 0.50, 0.75, 0.95]}
        _pred_dist: dict = {
            "run_id": run_id,
            "train_target_quantiles": _quantiles(y_train_num),
            "validation_predictions_quantiles": _quantiles(val_preds) if val_preds is not None else {},
            "final_predictions_quantiles": _quantiles(final_preds) if final_preds is not None else {},
        }
        # KS-like range comparison (generic, no scipy required)
        try:
            _tr_q = _quantiles(y_train_num)
            _fp_q = _quantiles(final_preds)
            if _tr_q and _fp_q:
                _range_diff = abs(_fp_q.get("p50", 0) - _tr_q.get("p50", 0)) / (abs(_tr_q.get("p50", 1)) + 1e-9)
                _pred_dist["distribution_similarity"] = {
                    "final_vs_train_median_ratio": round(float(_fp_q.get("p50", 0) / max(abs(_tr_q.get("p50", 1)), 1e-9)), 4),
                    "final_vs_train_p50_delta": round(float(_fp_q.get("p50", 0) - _tr_q.get("p50", 0)), 4),
                    "final_vs_train_p95_delta": round(float(_fp_q.get("p95", 0) - _tr_q.get("p95", 0)), 4),
                }
        except Exception:
            pass
        _write_json(_pred_dist, logs_dir / f"{run_id}_prediction_distribution.json")
    except Exception:
        pass

    # Report + review (gate at WARN; persist report_review.json with `approved`)
    report_out = write_analysis_report(state=state, repo_root=repo_root)
    state.report_review = review_report(state=state)
    _rr = state.report_review or {}
    print(f"Report: {state.report_path} ({report_out['report_format']}) | review: {_rr.get('verdict')}")
    _rr_status = PASS if _rr.get("approved") else (FAIL if str(_rr.get("verdict", "")).upper() == "FAIL" else WARN)
    _rr_reasons = list(_rr.get("required_revisions") or [])
    _emit_gate(logs_dir, run_id, "report", Verdict(
        stage="report", status=_rr_status, reasons=_rr_reasons,
        checked={"verdict": _rr.get("verdict"), "approved": bool(_rr.get("approved"))},
        critic="deterministic", run_id=run_id))
    _write_json(_rr, logs_dir / "report_review.json")
    state.persist(logs_dir)

    # Summary + manifest
    summary = {
        "run_id": run_id, "task_type": spec.task_type, "selected_model": mr.selected_model_name,
        "metric": mr.metric_name, "output_kind": mr.output_kind,
        "submission_valid": submission_check.get("dtype_ok") and submission_check.get("all_finite"),
        "report_review": state.report_review["verdict"],
    }
    state.summary = summary
    _write_json(_model_selection_log(mr), logs_dir / f"{run_id}_model_selection.json")
    manifest = {
        "run_id": run_id, "submission_path": str(submission_path), "report_path": state.report_path,
        "schema": schema.to_dict(), "summary": summary, "submission_check": submission_check,
        "artifacts": state.artifacts, "state_path": str(logs_dir / f"{run_id}_state.json"),
    }
    _write_json(manifest, logs_dir / f"{run_id}_manifest.json")

    # Final supervisor gate — aggregate every stage verdict into one release
    # judgement (supervisor_gatekeeper.json). Per the contract a FAIL is logged
    # and surfaced but does not halt: the deliverable is always produced.
    _stage_verdicts = []
    for _st in ("schema", "task_inference", "leakage", "prediction_sanity", "submission", "report"):
        _p = logs_dir / f"{run_id}_gate_{_st}.json"
        if _p.exists():
            try:
                _stage_verdicts.append(json.loads(_p.read_text(encoding="utf-8")))
            except Exception:
                pass
    _sup_status = PASS
    _sup_reasons: list[str] = []
    for _vd in _stage_verdicts:
        _sup_status = _worst(_sup_status, _vd.get("status", PASS))
        for _r in _vd.get("reasons", []) or []:
            _sup_reasons.append(f"{_vd.get('stage')}: {_r}")
    _sup_v = _emit_gate(logs_dir, run_id, "supervisor", Verdict(
        stage="supervisor", status=_sup_status, reasons=_sup_reasons,
        checked={"gates": [v.get("stage") for v in _stage_verdicts],
                 "submission_valid": bool(summary.get("submission_valid")),
                 "report_review": summary.get("report_review"),
                 "submission_path": str(submission_path), "report_path": state.report_path},
        critic="deterministic", run_id=run_id))
    print(f"[gate:supervisor] {_sup_v.status.upper()} | gates={[v.get('stage') for v in _stage_verdicts]}")
    state.persist(logs_dir)
    print("=== Analysis complete ===")
    return manifest


# ── generic single-file mode (no submission; analysis + report) ───────────────

def _run_generic(repo_root, run_id, logs_dir, artifacts_dir, goal, file_path, random_state) -> dict:
    from .skills.load_data import load_dataset

    print(f"=== Generic orchestrated analysis: {run_id} ({file_path}) ===")
    goal = goal or "Analyze the dataset"
    state = AnalysisState(
        run_id=run_id, created_at=datetime.now(timezone.utc).isoformat(),
        user_request=UserRequest(goal=goal, file_path=str(file_path)), data_path=str(file_path),
    )
    load = load_dataset(file_path)
    state.load_metadata = load.metadata()
    state.set_raw_data(load.df)
    profile = profile_tabular_data(load.df, file_path)
    state.data_profile = profile.model_dump()
    spec = infer_task_spec(goal, load.df, profile)
    state.task_spec = spec.model_dump()
    state.persist(logs_dir)

    if spec.task_type in (BINARY, MULTICLASS, "regression") and spec.target_variable:
        modeling = train_and_evaluate_models(
            df=load.df, target_col=spec.target_variable, task_spec=spec, random_state=random_state)
        mr = modeling.model_result_obj
        state.model_results = modeling.model_results
        hd = modeling.holdout
        state.evaluation = evaluate_model(
            task_type=hd["task_type"], y_true=hd["y_true"], y_pred=hd["y_pred"], y_proba=hd.get("y_proba"),
            baseline_y_pred=hd.get("baseline_y_pred"), class_labels=hd.get("label_classes"),
            positive_index=hd.get("positive_index", 1), run_id=run_id, artifacts_dir=artifacts_dir,
            selection_metric=mr.metric_name, candidate_name=mr.selected_model_name, baseline_name=hd.get("baseline_name", "baseline"))
        for label, path in (state.evaluation.get("artifacts") or {}).items():
            if path:
                state.register_artifact(label, path)
    state.eda_results = _eda_summary(profile.model_dump(), spec.model_dump())
    state.leakage_audit = run_leakage_check("leakage_check", task_spec=state.task_spec, data_profile=state.data_profile, df=load.df, run_id=run_id)
    report_out = write_analysis_report(state=state, repo_root=repo_root)
    state.report_review = review_report(state=state)
    state.persist(logs_dir)
    manifest = {"run_id": run_id, "report_path": state.report_path, "report_format": report_out["report_format"],
                "task_type": spec.task_type, "state_path": str(logs_dir / f"{run_id}_state.json")}
    _write_json(manifest, logs_dir / f"{run_id}_manifest.json")
    print("=== Analysis complete ===")
    return manifest


# ── helpers ───────────────────────────────────────────────────────────────────

def _goal_from_description(description: str) -> str:
    for line in description.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return s[:300]
    return "Do the data analysis"


def _eda_summary(profile: dict, task: dict) -> dict:
    findings: list[str] = []
    dist = profile.get("target_distribution") or {}
    if dist.get("classes"):
        parts = ", ".join(f"{c['label']}={c['fraction']:.1%}" for c in dist["classes"][:6])
        findings.append(f"Target class balance: {parts}.")
    findings.append(f"{len(profile.get('numeric_columns', {}))} numeric and {len(profile.get('categorical_columns', {}))} categorical columns.")
    miss = {c: m for c, m in (profile.get("missing_summary") or {}).items() if m.get("missing_rate", 0) > 0}
    if miss:
        worst = max(miss.items(), key=lambda kv: kv[1]["missing_rate"])
        findings.append(f"{len(miss)} columns have missing values; highest is {worst[0]} ({worst[1]['missing_rate']:.1%}).")
    return {"key_findings": findings, "n_warnings": len(profile.get("warnings") or [])}


def _associative_importance(df: pd.DataFrame, features: list[str], target: str) -> dict:
    if target not in df.columns:
        return {}
    tgt = df[target]
    if pd.api.types.is_numeric_dtype(tgt):
        tnum = pd.to_numeric(tgt, errors="coerce")
    else:
        tnum = pd.Series(pd.factorize(tgt)[0], index=tgt.index).replace(-1, np.nan)
    out: dict[str, float] = {}
    for feat in features:
        if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
            pair = pd.concat([pd.to_numeric(df[feat], errors="coerce"), tnum], axis=1).dropna()
            if len(pair) >= 5 and pair.iloc[:, 0].std() > 0:
                out[feat] = float(abs(pair.corr().iloc[0, 1]))
    return out


def _model_selection_log(mr) -> dict:
    return {
        "task_type": mr.task_type, "output_kind": mr.output_kind,
        "selected_model_name": mr.selected_model_name, "metric_name": mr.metric_name,
        "greater_is_better": mr.greater_is_better, "selected_metrics": mr.extra_metrics,
        "model_scores": mr.model_scores, "holdout_strategy": mr.holdout_strategy,
    }
