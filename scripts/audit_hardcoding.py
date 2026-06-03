#!/usr/bin/env python3
"""CLI entry point for the anti-hardcoding audit.

Usage
-----
Run from the repository root:

    python scripts/audit_hardcoding.py
    python scripts/audit_hardcoding.py --phase pre
    python scripts/audit_hardcoding.py --phase post --run-id my_run_id
    python scripts/audit_hardcoding.py --no-dynamic   # skip dynamic term extraction
    python scripts/audit_hardcoding.py --verbose       # print every finding

The audit log is written to outputs/logs/hardcoding_audit_{phase}_{run_id}.json.
Exit code: 0=pass, 1=warn, 2=fail.
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

# Ensure the package is importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_agent.audit import (
    audit_summary_text,
    extract_dynamic_terms,
    run_audit,
    write_audit_log,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anti-hardcoding audit for the Award B AutoML agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        choices=["pre", "post"],
        default="pre",
        help="Audit phase label (default: pre).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID to embed in the log filename.  Defaults to a timestamp.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to the repository root (default: current directory).",
    )
    parser.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Skip dynamic term extraction from DATA_DESCRIPTION.md and data files.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every finding, not just unacceptable ones.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_id = args.run_id or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S") + "_manual"
    logs_dir = repo_root / "outputs" / "logs"

    print(f"[audit] repo_root = {repo_root}")
    print(f"[audit] phase     = {args.phase}")
    print(f"[audit] run_id    = {run_id}")

    dynamic_terms: dict | None = None
    if not args.no_dynamic:
        data_dir = repo_root / "data"
        if data_dir.exists():
            dynamic_terms = extract_dynamic_terms(data_dir)
            n_dynamic = sum(len(v) for v in dynamic_terms.values())
            print(f"[audit] dynamic terms extracted: {n_dynamic} from {list(dynamic_terms.keys())}")
        else:
            print("[audit] data/ directory not found — skipping dynamic term extraction")

    result = run_audit(
        repo_root=repo_root,
        run_id=run_id,
        phase=args.phase,
        dynamic_terms=dynamic_terms,
    )

    log_path = write_audit_log(result, logs_dir)
    print(f"[audit] log written → {log_path.relative_to(repo_root)}")
    print()
    print(f"  Files scanned  : {len(result.scanned_files)}")
    print(f"  Acceptable     : {result.n_acceptable}")
    print(f"  Risky          : {result.n_risky}")
    print(f"  Unacceptable   : {result.n_unacceptable}")
    print(f"  Verdict        : {result.verdict.upper()}")
    print()

    if args.verbose:
        for f in result.findings:
            marker = {"acceptable": "  OK", "risky": "RISK", "unacceptable": "FAIL"}[f.classification]
            print(f"  [{marker}] {f.file_path}:{f.line_number}  term='{f.term}'  {f.reason}")
            print(f"        {f.line_content.strip()[:100]}")
    else:
        unacceptable = [f for f in result.findings if f.classification == "unacceptable"]
        risky = [f for f in result.findings if f.classification == "risky"]
        for f in unacceptable:
            print(f"  [FAIL] {f.file_path}:{f.line_number}  term='{f.term}'")
            print(f"         reason : {f.reason}")
            print(f"         line   : {f.line_content.strip()[:100]}")
        if risky and not unacceptable:
            print(f"  {len(risky)} risky finding(s) — use --verbose to see details")

    print()
    print(audit_summary_text(result))

    exit_codes = {"pass": 0, "warn": 1, "fail": 2}
    return exit_codes.get(result.verdict, 2)


if __name__ == "__main__":
    sys.exit(main())
