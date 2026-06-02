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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import build_feature_bundle
from .runner import (
    _build_submission,
    _remove_stale_outputs,
    _run_id,
    _validate_submission,
    _write_json,
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
from .task import BINARY, MULTICLASS


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

def _run_award_b(repo_root, run_id, logs_dir, artifacts_dir, goal, random_state) -> dict:
    print(f"=== Award B orchestrated analysis: {run_id} ===")
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
    print(f"Task: {spec.task_type} | metric: {spec.metric} | output: {spec.output_kind}")
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
    state.leakage_audit = run_leakage_check(
        "leakage_check", task_spec=state.task_spec, data_profile=state.data_profile, df=train_df, run_id=run_id)
    if not state.leakage_audit["approved"]:
        print(f"[orchestrator] leakage WARN/FAIL: {state.leakage_audit['leakage_risk']} (continuing on proven features)")
    state.persist(logs_dir)

    # Stage 10 — modeling (reuses models.train_and_predict)
    modeling = train_and_evaluate_models(bundle=bundle, block_column=schema.block_column, random_state=random_state)
    mr = modeling.model_result_obj
    state.model_results = modeling.model_results
    state.raw_modeling_results = {"all_scores": modeling.model_results.get("all_scores", [])}
    state.split_metadata = mr.holdout_strategy
    print(f"Selected model: {mr.selected_model_name} ({mr.metric_name})")
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
    importances = _associative_importance(train_df, spec.feature_candidates, schema.target_column)
    fip = plot_feature_importance(importances, run_id, artifacts_dir) if importances else None
    if fip:
        state.register_artifact("feature_importance_plot", fip)
    state.interpretation = {
        "feature_importance": importances,
        "top_features": sorted(importances, key=lambda k: importances[k], reverse=True)[:10],
        "caveats": ["Importance is associative (|correlation| for numeric features), not causal."],
    }
    state.persist(logs_dir)

    # Build + validate submission (proven path)
    submission = _build_submission(bundle, schema.row_id_column, schema.target_column, mr.predictions, mr.output_kind)
    submission_path = repo_root / "submission.csv"
    submission.to_csv(submission_path, index=False)
    submission_check = _validate_submission(bundle.sample_submission, submission, schema.row_id_column, schema.target_column, mr.output_kind)
    _write_json(submission_check, logs_dir / f"{run_id}_submission_check.json")
    print(f"Submission written: {submission_path}")

    # Report + review
    report_out = write_analysis_report(state=state, repo_root=repo_root)
    state.report_review = review_report(state=state)
    print(f"Report: {state.report_path} ({report_out['report_format']}) | review: {state.report_review['verdict']}")
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
