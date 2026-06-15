#!/usr/bin/env python3
"""check_run.py — independent post-hoc verifier for a STAI-X Award B run.

Agents in this pipeline communicate *only* through JSON/CSV artifacts under
``outputs/logs/`` (+ a couple of repo-root deliverables). So "did subagent X run
correctly?" reduces to "is X's expected artifact present and well-formed?". This
script encodes the CLAUDE.md I/O contract and checks every step's artifact for a
given ``run_id``. It imports/runs **no** agent and no modeling code — pure file
inspection — so it still works even if the orchestration died half-way (incl.
before the supervisor-gatekeeper agent could run its own check).

It is tolerant of the doc/code naming drift: an artifact named ``foo.json`` is
looked up as both ``{run_id}_foo.json`` and ``foo.json``; gate files are matched
under both ``{run_id}_gate_*.json`` and ``{run_id}_llm_gate_*.json``.

Usage:
    python scripts/check_run.py <run_id>
    python scripts/check_run.py --latest
    python scripts/check_run.py <run_id> --logs outputs/logs

Exit code: 0 if every REQUIRED artifact is present and valid, else 1.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

OK, INVALID, MISSING, NA, INFO = "ok", "invalid", "missing", "n/a", "info"
MARK = {OK: "OK   ", INVALID: "INVAL", MISSING: "MISS ", NA: "--   ", INFO: "info "}


# ── artifact resolution ──────────────────────────────────────────────────────
# Two on-disk layouts are supported: the current per-run directory
# ``outputs/runs/<rid>/{logs,scratch}`` with UNPREFIXED filenames, and the legacy
# ``outputs/logs/<rid>_X`` + ``outputs/scratch/<rid>/`` layout. Resolution tries the
# new layout first and falls back to the legacy one so old runs stay inspectable.
def _bare(name: str, rid: str) -> str:
    """Logical name with any ``{rid}_`` template/prefix stripped to the bare name."""
    return name.format(rid=rid).replace(f"{rid}_", "")


def _scratch_dirs(rid: str) -> list[Path]:
    return [REPO / "outputs" / "runs" / rid / "scratch",   # new
            REPO / "outputs" / "scratch" / rid]            # legacy


def _candidates(name: str, rid: str, logs: Path, loc: str) -> list[Path]:
    """All on-disk paths a logical artifact name could resolve to (new + legacy)."""
    if loc == "root":
        return [REPO / name]
    if loc == "reports":
        return [REPO / "outputs" / "reports" / name.format(rid=rid)]
    if loc == "scratch":
        return [d / name for d in _scratch_dirs(rid)]
    # default: logs. ``logs`` is already the per-run dir when it exists.
    bare = _bare(name, rid)
    legacy = REPO / "outputs" / "logs"
    # new layout (unprefixed in run dir) → legacy prefixed → legacy bare.
    plan = [(logs, bare), (legacy, name.format(rid=rid)),
            (legacy, f"{rid}_{bare}"), (legacy, bare)]
    out: list[Path] = []
    for base, fname in plan:
        if "*" in fname:
            out += [Path(p) for p in glob.glob(str(base / fname))]
        else:
            out.append(base / fname)
    return out


def _valid(path: Path) -> bool:
    """Present + well-formed. JSON must parse; other files must be non-empty."""
    if not path.is_file():
        return False
    if path.suffix == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8"))
            return True
        except Exception:
            return False
    return path.stat().st_size > 0


def resolve(name: str, rid: str, logs: Path, loc: str):
    """Return (status, found_path|None) for a single artifact name."""
    found = None
    for c in _candidates(name, rid, logs, loc):
        if c.is_file():
            found = c
            return (OK if _valid(c) else INVALID, c)
    return (MISSING, None)


def check_group(artifacts: list[tuple[str, str]], rid: str, logs: Path, mode: str):
    """artifacts = list of (name, loc). mode 'all' → every one must be OK;
    'any' → at least one OK. Returns (status, detail_str)."""
    results = [(n, *resolve(n, rid, logs, loc)) for n, loc in artifacts]
    statuses = [r[1] for r in results]
    if mode == "any":
        agg = OK if OK in statuses else (INVALID if INVALID in statuses else MISSING)
    else:  # all
        agg = OK if all(s == OK for s in statuses) else (
            INVALID if INVALID in statuses else MISSING)
    # human detail: show the resolved/searched names + per-name mark
    bits = []
    for n, st, p in results:
        tag = {OK: "", INVALID: " (invalid json)", MISSING: " (missing)"}[st]
        shown = p.name if p else n.format(rid=rid) if "{rid}" in n else n
        bits.append(shown + tag)
    return agg, ("  |  ".join(bits) if mode == "all" else " OR ".join(bits))


# ── modeling-mode detection ──────────────────────────────────────────────────
def detect_mode(rid: str, logs: Path) -> str:
    st, p = resolve("analysis_plan.json", rid, logs, "logs")
    if st == OK:
        try:
            mm = json.loads(p.read_text(encoding="utf-8")).get("modeling_mode")
            if mm in ("general", "specialist"):
                return mm
        except Exception:
            pass
    # infer from produced files
    if (resolve("{rid}_ensemble_meta.json", rid, logs, "logs")[0] == OK
            or glob.glob(str(logs / "cand_*.csv"))
            or glob.glob(str(REPO / "outputs" / "logs" / f"{rid}_cand_*.csv"))):
        return "specialist"
    if (resolve("model_search.json", rid, logs, "logs")[0] == OK
            or resolve("final_model.json", rid, logs, "logs")[0] == OK):
        return "general"
    return "unknown"


# ── the I/O contract as a check table ────────────────────────────────────────
# (step, agent, artifacts[(name, loc)], satisfy 'all'|'any', required, mode_filter)
def build_checks(mode: str):
    M = mode
    C = [
        ("1",  "data-format-converter",        [("data_conversion.json", "logs")], "all", False, None),
        ("2a", "task-inference-agent",         [("spec_parse.json", "logs")], "all", True,  None),
        ("2b", "data-profiler",                [("data_profile.json", "logs")], "all", True,  None),
        ("2b", "data-profiler (missingness)",  [("missingness_profile.json", "logs"),
                                                ("imputation_plan.json", "logs")], "all", True, None),
        ("3a", "validation-schema-guardian",   [("validation_strategy.json", "logs"),
                                                ("{rid}_cv_folds.json", "logs")], "all", True, None),
        ("3b", "hardcoding-feature-auditor",   [("hardcoding_audit_pre.json", "logs")], "all", True, None),
        ("4",  "analysis-planner",             [("analysis_plan.json", "logs")], "all", True, None),
        ("5",  "plan-reviewer",                [("plan_review_1.json", "logs"),
                                                ("plan_review_2.json", "logs")], "any", True, None),
        ("6A", "analysis-programmer (floor)",  [("{rid}_model_selection.json", "logs"),
                                                ("{rid}_oof_floor.csv", "logs")], "all", True, None),
        ("6A", "analysis-programmer (feats)",  [("{rid}_feature_spec.json", "logs")], "all", True, None),
        ("6A", "analysis-programmer (pipe)",   [("feature_pipeline.py", "scratch")], "all", False, None),
        ("6A", "analysis-programmer (matrix)", [("{rid}_features_train.parquet", "logs"),
                                                ("{rid}_features_train.csv", "logs")], "any", False, None),
        ("6B", "model-search-agent [general]", [("model_search.json", "logs"),
                                                ("final_model.json", "logs")], "all", True, "general"),
        ("6B", "modeling-specialist [spec]",   [("{rid}_cand_*.csv", "logs"),
                                                ("{rid}_agent_*.json", "logs")], "all", True, "specialist"),
        ("6B", "ensemble-meta [specialist]",   [("{rid}_ensemble_meta.json", "logs")], "all", True, "specialist"),
        ("6B", "modeling-watchdog [spec]",     [("{rid}_*-specialist_watchdog.json", "logs")], "any", False, "specialist"),
        ("6B", "keep-best (common)",           [("prediction_sanity.json", "logs"),
                                                ("{rid}_promotion.json", "logs")], "all", False, None),
        ("6B", "OOF candidates (common)",      [("{rid}_oof_*.csv", "logs")], "any", False, None),
        ("6C", "model-performance-reviewer",   [("model_performance_review.json", "logs")], "all", True, None),
        ("6C", "feature-leakage-reviewer",     [("feature_audit_review.json", "logs")], "all", True, None),
        ("6C", "generalization-reviewer",      [("overfitting_leakage_audit.json", "logs")], "all", True, None),
        ("6D", "model-performance (lead)",     [("analysis_review_1.json", "logs"),
                                                ("analysis_review_2.json", "logs"),
                                                ("analysis_review_3.json", "logs")], "any", True, None),
        ("7a", "validation-schema-guardian",   [("submission_validation.json", "logs")], "all", True, None),
        ("7b", "report-writer (pdf)",          [("report.pdf", "root")], "all", True, None),
        ("7b", "report-writer (logs)",         [("{rid}_report.pdf", "reports"),
                                                ("{rid}_report.md", "reports")], "any", False, None),
        ("8a", "report-reviewer",              [("report_review.json", "logs")], "all", True, None),
        ("8b", "supervisor-gatekeeper",        [("supervisor_gatekeeper.json", "logs")], "all", True, None),
        ("--", "deliverable: submission.csv",  [("submission.csv", "root")], "all", True, None),
    ]
    # keep a check only if its mode_filter is None or matches (unknown → keep both, marked n/a handled below)
    return C


# ── gate + watchdog summaries ────────────────────────────────────────────────
def gate_summary(rid: str, logs: Path):
    legacy = REPO / "outputs" / "logs"
    files = sorted(set(glob.glob(str(logs / "gate_*.json"))
                       + glob.glob(str(logs / "llm_gate_*.json"))
                       + glob.glob(str(legacy / f"{rid}_gate_*.json"))
                       + glob.glob(str(legacy / f"{rid}_llm_gate_*.json"))))
    rows = []
    for f in files:
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            rows.append((d.get("stage", Path(f).stem), d.get("status", "?"),
                         d.get("reasons", [])))
        except Exception:
            rows.append((Path(f).name, "unreadable", []))
    return rows


def watchdog_summary(rid: str, logs: Path):
    rows = []
    legacy = REPO / "outputs" / "logs"
    files = sorted(set(glob.glob(str(logs / "*-specialist_watchdog.json"))
                       + glob.glob(str(legacy / f"{rid}_*-specialist_watchdog.json"))))
    for f in files:
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            rows.append((d.get("role", Path(f).stem), d.get("killed"),
                         d.get("reason", ""), d.get("budget_pressure", "")))
        except Exception:
            rows.append((Path(f).name, "?", "unreadable", ""))
    return rows


# ── main ─────────────────────────────────────────────────────────────────────
def latest_run_id(logs: Path) -> str | None:
    candidates: list[tuple[float, str]] = []  # (mtime, rid)
    # new layout: outputs/runs/<rid>/logs/
    runs_root = REPO / "outputs" / "runs"
    if runs_root.is_dir():
        for d in runs_root.iterdir():
            if d.is_dir() and (d / "logs").is_dir():
                candidates.append((d.stat().st_mtime, d.name))
    # legacy layout: flat outputs/logs/<rid>_<anchor>
    for suf in ("_state.json", "_manifest.json", "_feature_spec.json", "_model_selection.json"):
        for p in glob.glob(str(logs / f"*{suf}")):
            candidates.append((os.path.getmtime(p), Path(p).name[: -len(suf)]))
    if not candidates:
        return None
    return max(candidates)[1]


def _effective_logs(rid: str, cli_logs: Path) -> Path:
    """Per-run dir when present (new layout), else the CLI/legacy logs dir."""
    run_logs = REPO / "outputs" / "runs" / rid / "logs"
    return run_logs if run_logs.is_dir() else cli_logs


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify all subagent artifacts for a run.")
    ap.add_argument("run_id", nargs="?", help="the run_id to check")
    ap.add_argument("--latest", action="store_true", help="auto-pick the newest run_id")
    ap.add_argument("--logs", default="outputs/logs", help="logs dir (default outputs/logs)")
    args = ap.parse_args()

    cli_logs = (REPO / args.logs).resolve() if not os.path.isabs(args.logs) else Path(args.logs)
    rid = args.run_id
    if args.latest or not rid:
        rid = latest_run_id(cli_logs)
        if not rid:
            print(f"no run found under outputs/runs/ or {cli_logs} "
                  f"(need a run dir or *_state.json anchor)")
            return 1
        print(f"[--latest] using run_id = {rid}")

    # Prefer the per-run directory (new layout); fall back to the flat logs dir.
    logs = _effective_logs(rid, cli_logs)
    if not logs.is_dir():
        print(f"logs dir not found: {logs}")
        return 1

    mode = detect_mode(rid, logs)
    checks = build_checks(mode)

    print(f"\nRun: {rid}   |   logs: {logs}   |   modeling_mode: {mode}\n")
    print(f"{'STEP':<4} {'STATUS':<6} {'AGENT':<34} ARTIFACT(S)")
    print("-" * 100)

    missing_required = 0
    for step, agent, arts, satisfy, required, mfilter in checks:
        # mode gating
        if mfilter and mode != "unknown" and mode != mfilter:
            status, detail = NA, f"(not applicable in '{mode}' mode)"
        else:
            status, detail = check_group(arts, rid, logs, satisfy)
            if status == NA:
                pass
            elif status != OK:
                if not required:
                    status = INFO if status == MISSING else status
                elif mfilter == "unknown":
                    pass
                if required and status in (MISSING, INVALID):
                    missing_required += 1
        flag = MARK[status]
        req = "" if required else " (optional)"
        print(f"{step:<4} [{flag}] {agent:<34} {detail}{req}")

    # gates
    gates = gate_summary(rid, logs)
    print("\n" + "-" * 100)
    if gates:
        worst = "pass"
        order = {"pass": 0, "warn": 1, "fail": 2, "unreadable": 3, "?": 1}
        print("GATE VERDICTS:")
        for stage, st, reasons in gates:
            if order.get(st, 1) > order.get(worst, 0):
                worst = st
            rs = ("  — " + "; ".join(reasons)) if reasons else ""
            print(f"   {stage:<18} {st}{rs}")
        print(f"   → worst gate status: {worst.upper()}")
    else:
        print("GATE VERDICTS: (none found — gate files not written this run)")

    # watchdog
    wd = watchdog_summary(rid, logs)
    if wd:
        print("\nWATCHDOG (specialist mode):")
        for role, killed, reason, bp in wd:
            k = "KILLED" if killed else "ok"
            print(f"   {role:<22} {k:<7} pressure={bp or '-'}  {reason}")

    # verdict
    print("\n" + "=" * 100)
    if missing_required == 0:
        print("VERDICT: ✅ all REQUIRED subagent artifacts present and valid.")
        return 0
    print(f"VERDICT: ❌ {missing_required} required artifact(s) MISSING/INVALID "
          f"— those subagents did not run or did not complete.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
