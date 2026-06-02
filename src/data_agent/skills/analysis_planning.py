"""analysis-planning skill: deterministic plan generator + critic.

Produces ``state.plan.draft`` (analysis-planning SKILL) and the optimized/approved
plan (plan_critic agent). Execution is orchestrator-driven; the plan is recorded for
fidelity, inspection, and the report. Generators are deterministic so the orchestrator
can produce a valid plan without the LLM, while the agent docs describe the same
contract for the LLM-driven path.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .leakage_check import run_leakage_check

_SUPERVISED = {"binary_classification", "multiclass_classification", "regression"}
_CAUSAL = ["effect of", "impact of", "causes", "causal", "treatment effect", "counterfactual"]


def _get(obj: Any, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _step(step_id, name, goal, inputs, outputs, success, risks, skills, agent="python+claude") -> dict:
    return {
        "step_id": step_id, "name": name, "goal": goal,
        "inputs": inputs, "outputs": outputs, "success_criteria": success,
        "risks": risks, "required_skills": skills, "responsible_agent": agent,
    }


def _baseline_spec(task_type: str) -> tuple[str, list[str], str]:
    if task_type == "regression":
        return ("DummyRegressor (mean)", ["rmse", "mae", "r2"], "candidate val RMSE < baseline val RMSE")
    if task_type == "multiclass_classification":
        return ("DummyClassifier (most_frequent)", ["f1_weighted", "f1_macro", "accuracy"],
                "candidate val F1 (weighted) > baseline val F1 (weighted)")
    return ("DummyClassifier (stratified)", ["accuracy", "f1_weighted", "roc_auc"],
            "candidate val ROC-AUC > baseline val ROC-AUC")


def _eval_artifacts(task_type: str) -> list[str]:
    if task_type == "regression":
        return ["state.artifacts.residual_plot", "state.artifacts.pred_vs_actual"]
    if task_type == "multiclass_classification":
        return ["state.artifacts.confusion_matrix"]
    return ["state.artifacts.confusion_matrix", "state.artifacts.roc_curve"]


def build_analysis_plan(*, run_id: str, user_goal: str, task_spec: Any, data_profile: Any,
                        leakage_audit: dict | None = None) -> dict:
    task_type = _get(task_spec, "task_type", "unknown")
    if not task_type or task_type == "unknown":
        raise ValueError("Cannot generate plan: task type is unresolved. Run task-inference first.")
    target = _get(task_spec, "target_variable")
    imbalance = bool(_get(task_spec, "class_imbalance_detected", False))

    steps: list[dict] = [
        _step("validate_data", "Validate Loaded Data",
              "Confirm the loaded DataFrame matches the profiled schema (shape, dtypes, column names).",
              ["state.raw_data", "state.data_profile"], ["state.validation_result"],
              ["DataFrame shape matches data_profile.n_rows and n_columns",
               "All column names match data_profile.columns"],
              ["File re-read may differ from the profiled version"],
              ["data-loader", "data-profiler"]),
        _step("eda_exploration", "Exploratory Data Analysis",
              "Produce distribution summaries, correlation/association analysis, and a missing-value overview.",
              ["state.raw_data", "state.task"], ["state.eda_results", "state.artifacts.eda_plots"],
              ["Distribution summary produced for numeric features",
               "Missing-value summary recorded in state.eda_results"],
              ["High-cardinality columns may produce unusable plots"],
              ["eda-analyzer", "plot-generator"]),
        _step("leakage_check", "Pre-Modeling Leakage Audit",
              "Audit feature candidates and column names for leakage before any split or transformation.",
              ["state.raw_data", "state.task", "state.plan.draft"], ["state.leakage_audit"],
              ["All potential ID columns confirmed excluded from feature_candidates",
               "Leakage audit verdict is PASS"],
              ["An ID or encoded-target column may have been retained in features"],
              ["leakage-check"], agent="claude"),
    ]

    if task_type in _SUPERVISED:
        baseline_name, baseline_metrics, candidate_criterion = _baseline_spec(task_type)
        preprocess_skills = ["data-splitter", "feature-engineer", "leakage-check"] if imbalance else ["data-splitter", "leakage-check"]
        preprocess_goal = "Split into train/val/test with seed=42; fit imputer/encoder on X_train only."
        if imbalance:
            preprocess_goal += " Apply a class-balance strategy (class_weight/resampling)."
        steps += [
            _step("preprocess_split_transform", "Split Data and Fit Transformers", preprocess_goal,
                  ["state.raw_data", "state.task", "state.leakage_audit"],
                  ["state.splits.X_train", "state.splits.X_test", "state.transformers"],
                  ["Split performed before any transformer is fitted",
                   "Transformer fit called only on X_train", "Split seed recorded in state.split_metadata"],
                  ["fit_transform on full data → leakage", "Unset seed → non-reproducible splits"],
                  preprocess_skills),
            _step("baseline_model", f"Train Baseline {baseline_name}",
                  f"Establish a performance floor with {baseline_name}; record {', '.join(baseline_metrics)} on validation.",
                  ["state.splits"], ["state.models.baseline", "state.artifacts.baseline_model"],
                  ["Baseline trained on X_train only",
                   f"Baseline metrics ({', '.join(baseline_metrics)}) recorded with split 'val'"],
                  ["Evaluating baseline on the test set instead of val"],
                  ["tabular-modeling", "model-evaluation"]),
            _step("model_candidate", "Train Candidate Model",
                  "Train task-conditioned candidate models and select the best on validation.",
                  ["state.splits", "state.models.baseline"], ["state.models.candidate", "state.artifacts.candidate_model"],
                  ["Model selected on validation only (test untouched)", candidate_criterion],
                  ["Tuning may leak test data if not isolated"],
                  ["tabular-modeling", "leakage-check", "model-evaluation"]),
            _step("evaluate_test", "Final Evaluation",
                  "Evaluate the selected model on held-out data; produce the required artifacts and a baseline comparison.",
                  ["state.models.candidate", "state.models.baseline", "state.splits"],
                  ["state.evaluation", *_eval_artifacts(task_type)],
                  ["All metrics labeled with split 'test'", "Candidate vs baseline comparison present"],
                  ["Reporting val metrics as final"],
                  ["model-evaluation", "plot-generator"]),
        ]

    steps += [
        _step("interpret_results", "Interpret Drivers",
              "Compute feature importance / key patterns; list caveats. No causal language.",
              ["state.evaluation", "state.task"], ["state.interpretation", "state.artifacts.feature_importance_plot"],
              ["Feature importance computed and sorted", "At least one caveat recorded"],
              ["Importance reflects associations, not causation"],
              ["plot-generator"]),
        _step("report_generate", "Generate Final Report",
              "Write the structured report referencing real artifact paths and metric values.",
              ["state.evaluation", "state.interpretation", "state.execution_log"],
              ["state.report_draft", "state.artifacts.report_md"],
              ["All required sections present", "Every metric includes its split label"],
              ["Metric values copied incorrectly from state"],
              ["report-writing"], agent="claude"),
        _step("report_review", "Independent Report Review",
              "Audit the report against state; verify claims, metrics, and artifact references.",
              ["state.report_draft", "state.evaluation", "state.leakage_audit", "state.artifacts"],
              ["state.report_review"],
              ["Every metric in report matches state.evaluation", "Review verdict is PASS"],
              ["Subtle causal framing may pass without careful reading"],
              ["report-review", "model-evaluation", "leakage-check"], agent="claude"),
    ]

    all_skills = sorted({s for step in steps for s in step["required_skills"]})
    return {
        "plan_id": f"plan_{run_id}",
        "task_type": task_type,
        "target_variable": target,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "draft",
        "steps": steps,
        "plan_skill_summary": {"all_skills_used": all_skills, "steps_missing_required_skills": []},
        "plan_warnings": [],
    }


def critique_and_optimize_plan(*, initial_plan: dict, task_spec: Any, data_profile: Any,
                               user_goal: str = "") -> tuple[dict, dict]:
    steps = initial_plan.get("steps", [])
    step_ids = [s["step_id"] for s in steps]
    task_type = initial_plan.get("task_type")
    failed: list[dict] = []
    warned: list[dict] = []

    def check(ok: bool, name: str, detail: str, severity: str = "FAIL"):
        if not ok:
            (failed if severity == "FAIL" else warned).append({"check": name, "detail": detail, "severity": severity})

    # structural checks (subset of plan_critic's 15)
    required = {"validate_data", "eda_exploration", "leakage_check", "interpret_results", "report_generate", "report_review"}
    check(required.issubset(set(step_ids)), "required_phases", f"Missing phases: {required - set(step_ids)}")
    check("leakage_check" in step_ids, "leakage_standalone", "leakage_check must be a standalone step")
    check(all(s["required_skills"] for s in steps), "skills_present", "Every step needs required_skills")
    check(not any(any(c in s["goal"].lower() for c in _CAUSAL) for s in steps), "no_causal_language",
          "Plan goals must avoid causal language", severity="WARN")
    if task_type in _SUPERVISED:
        check("baseline_model" in step_ids and "model_candidate" in step_ids,
              "baseline_required", "Supervised plans need a baseline and a candidate")
        if "baseline_model" in step_ids and "model_candidate" in step_ids:
            check(step_ids.index("baseline_model") < step_ids.index("model_candidate"),
                  "baseline_before_candidate", "Baseline must precede candidate")
        check("preprocess_split_transform" in step_ids, "split_present", "Supervised plans need a split step")
        if "leakage_check" in step_ids and "preprocess_split_transform" in step_ids:
            check(step_ids.index("leakage_check") < step_ids.index("preprocess_split_transform"),
                  "leakage_before_preprocess", "Leakage check must precede preprocessing")

    # embed a structural leakage audit at critique (checks 1,2,3,7)
    leakage_at_critique = run_leakage_check(
        "plan_critique", task_spec=task_spec if isinstance(task_spec, dict) else task_spec.model_dump(),
        data_profile=data_profile if isinstance(data_profile, dict) else data_profile.model_dump(),
        run_id=initial_plan.get("plan_id", ""),
    )
    if not leakage_at_critique["approved"]:
        failed.append({"check": "leakage_audit", "detail": "Structural leakage FAIL at critique", "severity": "FAIL"})

    n_checks = 8
    quality = round(max(0.0, (n_checks - len(failed) - 0.5 * len(warned)) / n_checks), 3)
    verdict = "PASS" if not failed else "FAIL"

    optimized = dict(initial_plan)
    optimized["status"] = "approved" if verdict == "PASS" else "draft"
    optimized["leakage_audit_at_critique"] = leakage_at_critique

    critique = {
        "verdict": verdict,
        "plan_quality_score": quality,
        "failed_checks": failed,
        "warned_checks": warned,
        "applied_fixes": [],
    }
    return critique, optimized
