"""Anti-hardcoding audit for the Award B AutoML agent.

Two-pass design
---------------
pre-run  : scans src/ and scripts/ against static + dynamic suspicious terms,
           invoked after schema discovery but before feature engineering.
post-run : same scan plus the generated report markdown, invoked before
           finalising submission.csv and report.pdf.

Output: outputs/logs/hardcoding_audit_{phase}_{run_id}.json

Classification
--------------
  acceptable   — found only in tests, comments, docstrings, documentation,
                 reference scripts, or negative-example lists.
  risky        — appears in source code in a multi-item candidate/fallback
                 list, or in a non-critical context that *could* affect
                 runtime behaviour.
  unacceptable — directly controls column selection, file loading, feature
                 engineering, target assignment, or report content via a
                 hardcoded string.

Verdict
-------
  pass  — zero unacceptable findings
  warn  — zero unacceptable but one or more risky findings
  fail  — one or more unacceptable findings
"""

from __future__ import annotations

import json
import re
import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


# ── static suspicious-term catalogue ─────────────────────────────────────────
# These are always searched regardless of the current dataset.
# Keys are human-readable categories; values are the terms to look for as
# quoted string literals in Python source files.

STATIC_TERMS: dict[str, list[str]] = {
    # Previously hardcoded Award A / A1 field names
    "award_a_field_names": [
        "rate_per_10000_ed_visits",
        "overdose_category",
        "all_drugs",
        "all_opioids",
        "all_stimulants",
    ],
    # Known competition-specific column names (Titanic, housing, etc.)
    "known_competition_columns": [
        "survived",
        "passengerid",
        "pclass",
        "sibsp",
        "parch",
        "embarked",
        "saleprice",
        "lotarea",
        "yearbuilt",
        "casual",
        "registered",
    ],
    # Assumed file names (should be discovered, not hardcoded)
    "assumed_file_names": [
        "train.csv",
        "test.csv",
        "samplesubmission.csv",
        "sampleSubmission.csv",
        "val.csv",
        "train.parquet",
        "test.parquet",
        "submission.csv",
    ],
    # Domain vocabulary from past challenges
    "domain_vocabulary": [
        "overdose",
        "opioid",
        "stimulant",
        "titanic",
        "shipwreck",
    ],
    # Magic numbers documented as forbidden in CLAUDE.md
    "magic_numbers": [
        "918",
    ],
}

# ── paths configuration ───────────────────────────────────────────────────────

# Directories/files to scan (relative to repo root, as strings to match).
# outputs/scratch/ holds LLM-authored runtime scripts (e.g. programmer_pipeline.py);
# they must obey the same no-hardcoding rules as the committed pipeline.
SCAN_TARGETS = ["src/", "main.py", "scripts/", "outputs/scratch/"]

# Paths that make every finding in them acceptable regardless of content
ALWAYS_ACCEPTABLE_PATH_FRAGMENTS = [
    "tests/",
    "__pycache__/",
]

# File extensions to scan
SCANNABLE_EXTENSIONS = {".py", ".md"}

# ── context-pattern helpers ───────────────────────────────────────────────────

# Patterns that indicate a string literal *controls* logic (unacceptable)
# {t} is replaced by the regex-escaped term.
_UNACCEPTABLE_PATTERNS = [
    # Direct DataFrame/dict column access: df["term"], frame['term']
    r"""\w+\s*\[\s*['"]{t}['"]\s*\]""",
    # Direct assignment to a schema-critical variable
    r"""(?:target_col(?:umn)?|row_id(?:_col(?:umn)?)?|id_col(?:umn)?|datetime_col|time_col)\s*=\s*['"]{t}['"]""",
    # Equality comparison: == "term" or "term" ==
    r"""==\s*['"]{t}['"]""",
    r"""['"]{t}['"]\s*==""",
    # File loading with hardcoded path
    r"""(?:open|read_csv|read_table|read_excel|read_parquet|Path)\s*\(\s*['"]{t}['"]""",
    # Any variable assignment: VAR = "term" or var = "term"
    # Covers module-level constants, local variables, and keyword defaults.
    r"""\b[\w_]+\s*=\s*['"]{t}['"]""",
]

# Patterns that indicate a risky-but-possibly-acceptable context
_RISKY_PATTERNS = [
    # Single-item list: ["term"] — a single forced candidate
    r"""\[\s*['"]{t}['"]\s*\]""",
    # in-operator check: "term" in var or var in ["term"]
    r"""['"]{t}['"]\s+in\b""",
    r"""\bin\s+['"]{t}['"]""",
    # Inequality: != "term"
    r"""!=\s*['"]{t}['"]""",
]


# ── data structures ───────────────────────────────────────────────────────────

@dataclass
class AuditFinding:
    file_path: str
    line_number: int
    line_content: str
    term: str
    term_category: str
    term_source: str           # "static" | "dynamic:<source>"
    classification: str        # "acceptable" | "risky" | "unacceptable"
    reason: str


@dataclass
class AuditResult:
    phase: str                 # "pre" | "post"
    run_id: str
    timestamp: str
    scanned_files: list[str]
    static_terms_searched: dict[str, list[str]]
    dynamic_terms_searched: dict[str, list[str]]
    findings: list[AuditFinding]
    n_acceptable: int = 0
    n_risky: int = 0
    n_unacceptable: int = 0
    verdict: str = "pass"      # "pass" | "warn" | "fail"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["findings"] = [asdict(f) for f in self.findings]
        return d


# ── public API ────────────────────────────────────────────────────────────────

def run_audit(
    repo_root: Path,
    run_id: str,
    phase: str = "pre",
    dynamic_terms: dict[str, list[str]] | None = None,
    extra_files: list[Path] | None = None,
) -> AuditResult:
    """Run the anti-hardcoding audit.

    Args:
        repo_root:      Repository root directory.
        run_id:         Current pipeline run ID (used in log file naming).
        phase:          "pre" (before modeling) or "post" (before final output).
        dynamic_terms:  Additional terms extracted from the current dataset's
                        DATA_DESCRIPTION.md and column names.  Keyed by source
                        label (e.g. ``"train_columns"``, ``"description_tokens"``).
        extra_files:    Additional files to scan (e.g. the generated report .md).

    Returns:
        AuditResult with all findings and a ``verdict``.
    """
    all_terms: dict[str, tuple[str, str]] = {}  # term → (category, source)

    for category, terms in STATIC_TERMS.items():
        for t in terms:
            all_terms[t.lower()] = (category, "static")

    for source_label, terms in (dynamic_terms or {}).items():
        for t in terms:
            key = t.lower()
            if key not in all_terms and _is_worth_auditing(t):
                all_terms[key] = (f"dynamic_{source_label}", f"dynamic:{source_label}")

    files_to_scan = _collect_scan_targets(repo_root) + (extra_files or [])

    all_findings: list[AuditFinding] = []
    scanned_paths: list[str] = []

    for file_path in files_to_scan:
        if not file_path.exists():
            continue
        scanned_paths.append(str(file_path.relative_to(repo_root)))
        findings = _scan_file(file_path, all_terms, repo_root)
        all_findings.extend(findings)

    n_acceptable = sum(1 for f in all_findings if f.classification == "acceptable")
    n_risky = sum(1 for f in all_findings if f.classification == "risky")
    n_unacceptable = sum(1 for f in all_findings if f.classification == "unacceptable")

    if n_unacceptable > 0:
        verdict = "fail"
    elif n_risky > 0:
        verdict = "warn"
    else:
        verdict = "pass"

    dynamic_searched: dict[str, list[str]] = {}
    for source_label, terms in (dynamic_terms or {}).items():
        dynamic_searched[source_label] = [t for t in terms if _is_worth_auditing(t)]

    return AuditResult(
        phase=phase,
        run_id=run_id,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        scanned_files=scanned_paths,
        static_terms_searched=STATIC_TERMS,
        dynamic_terms_searched=dynamic_searched,
        findings=all_findings,
        n_acceptable=n_acceptable,
        n_risky=n_risky,
        n_unacceptable=n_unacceptable,
        verdict=verdict,
    )


def extract_dynamic_terms(data_dir: Path) -> dict[str, list[str]]:
    """Extract suspicious terms from the current dataset's files.

    Sources:
    - DATA_DESCRIPTION.md: backtick-quoted identifiers and file names
    - Train file header: column names
    - Sample submission header: column names

    Returns a dict keyed by source label.
    """
    result: dict[str, list[str]] = {}
    data_dir = data_dir.resolve()

    desc_path = data_dir / "DATA_DESCRIPTION.md"
    if desc_path.exists():
        text = desc_path.read_text(encoding="utf-8", errors="replace")
        tokens = _extract_description_tokens(text)
        if tokens:
            result["description_tokens"] = tokens

    for rel in _find_table_files(data_dir):
        role = _guess_file_role(rel.name)
        try:
            cols = _read_header(rel)
            if cols:
                result[f"{role}_columns"] = cols
        except Exception:
            pass

    return result


def write_audit_log(result: AuditResult, logs_dir: Path) -> Path:
    """Serialise the AuditResult to JSON and return the written path."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"hardcoding_audit_{result.phase}_{result.run_id}.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def audit_summary_text(result: AuditResult) -> str:
    """Return a one-paragraph human-readable summary for reports."""
    lines = [
        f"Anti-hardcoding audit ({result.phase}-run): "
        f"{len(result.scanned_files)} files scanned. "
        f"Verdict: {result.verdict.upper()}. "
        f"Findings — acceptable: {result.n_acceptable}, "
        f"risky: {result.n_risky}, "
        f"unacceptable: {result.n_unacceptable}."
    ]
    unacceptable = [f for f in result.findings if f.classification == "unacceptable"]
    if unacceptable:
        lines.append("Unacceptable findings:")
        for f in unacceptable[:5]:
            lines.append(
                f"  • {Path(f.file_path).name}:{f.line_number} — term '{f.term}': {f.reason}"
            )
    risky = [f for f in result.findings if f.classification == "risky"]
    if risky and not unacceptable:
        lines.append(f"Risky findings (review recommended): {len(risky)} occurrence(s).")
    if result.verdict == "pass":
        lines.append("No hardcoding concerns detected.")
    return " ".join(lines)


# ── file collection ───────────────────────────────────────────────────────────

def _collect_scan_targets(repo_root: Path) -> list[Path]:
    collected: list[Path] = []
    for target in SCAN_TARGETS:
        candidate = repo_root / target
        if candidate.is_file() and candidate.suffix in SCANNABLE_EXTENSIONS:
            collected.append(candidate)
        elif candidate.is_dir():
            for path in sorted(candidate.rglob("*")):
                if path.is_file() and path.suffix in SCANNABLE_EXTENSIONS:
                    rel = str(path.relative_to(repo_root))
                    if not any(frag in rel for frag in ALWAYS_ACCEPTABLE_PATH_FRAGMENTS):
                        collected.append(path)
    return collected


# ── file scanning ─────────────────────────────────────────────────────────────

def _scan_file(
    file_path: Path,
    all_terms: dict[str, tuple[str, str]],
    repo_root: Path,
) -> list[AuditFinding]:
    """Scan a single file for all suspicious terms and return findings."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    rel_path = str(file_path.relative_to(repo_root))
    is_always_acceptable = any(frag in rel_path for frag in ALWAYS_ACCEPTABLE_PATH_FRAGMENTS)
    is_markdown = file_path.suffix == ".md"
    is_test = "test" in rel_path.lower()

    lines = text.splitlines()
    docstring_state = _DocstringTracker()
    findings: list[AuditFinding] = []

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        in_docstring = docstring_state.update(stripped)
        is_comment = stripped.startswith("#")

        for term_lower, (category, source) in all_terms.items():
            # Quick pre-check: is the term (quoted) even on this line?
            if f'"{term_lower}"' not in line.lower() and f"'{term_lower}'" not in line.lower():
                continue

            # More precise match: the term as a complete quoted string literal,
            # case-insensitive (column names may be mixed-case).
            if not re.search(r"""['"]""" + re.escape(term_lower) + r"""['"]""", line, re.IGNORECASE):
                continue

            classification, reason = _classify(
                line=line,
                term=term_lower,
                rel_path=rel_path,
                is_always_acceptable=is_always_acceptable,
                is_markdown=is_markdown,
                is_test=is_test,
                is_comment=is_comment,
                in_docstring=in_docstring,
            )

            findings.append(AuditFinding(
                file_path=rel_path,
                line_number=lineno,
                line_content=line.rstrip()[:200],
                term=term_lower,
                term_category=category,
                term_source=source,
                classification=classification,
                reason=reason,
            ))

    return findings


# ── classification logic ──────────────────────────────────────────────────────

def _classify(
    line: str,
    term: str,
    rel_path: str,
    is_always_acceptable: bool,
    is_markdown: bool,
    is_test: bool,
    is_comment: bool,
    in_docstring: bool,
) -> tuple[str, str]:
    """Return (classification, reason) for one term occurrence on one line."""

    if is_always_acceptable:
        return "acceptable", "in always-acceptable path (tests/reference/cache)"
    if is_test:
        return "acceptable", "in test file"
    if is_comment:
        return "acceptable", "appears in comment"
    if in_docstring:
        return "acceptable", "appears in docstring"
    if is_markdown:
        # In documentation, flag only if it looks like a code snippet (backtick context)
        if _in_markdown_code_fence(line):
            return "risky", "quoted term in markdown code fence (may be template leakage)"
        return "acceptable", "appears in documentation prose"

    t = re.escape(term)

    # ── unacceptable patterns ────────────────────────────────────────────────
    for raw_pattern in _UNACCEPTABLE_PATTERNS:
        pattern = raw_pattern.replace("{t}", t)
        if re.search(pattern, line, re.IGNORECASE):
            label = raw_pattern.split("{t}")[0].strip()
            return "unacceptable", f"hardcoded string in control-flow pattern: {label!r}"

    # ── risky patterns ───────────────────────────────────────────────────────
    for raw_pattern in _RISKY_PATTERNS:
        pattern = raw_pattern.replace("{t}", t)
        if re.search(pattern, line, re.IGNORECASE):
            return "risky", f"hardcoded string in risky pattern: {raw_pattern!r}"

    # ── multi-item candidate list → acceptable heuristic ────────────────────
    # If the term appears inside a list literal that has 3+ quoted items,
    # it is treated as a generic fallback heuristic rather than hardcoding.
    if _in_multi_item_list(line, term, min_items=3):
        return "acceptable", "term is one of 3+ candidates in a fallback heuristic list"

    # ── two-item list → risky ────────────────────────────────────────────────
    if _in_multi_item_list(line, term, min_items=2):
        return "risky", "term is in a short (2-item) candidate list — review"

    # ── unclassified string literal in source ────────────────────────────────
    return "risky", "quoted string literal in source code (context unclear)"


# ── helper: docstring tracker ─────────────────────────────────────────────────

class _DocstringTracker:
    """Simple state machine to track whether we are inside a docstring."""

    def __init__(self) -> None:
        self._in = False
        self._marker: str | None = None

    def update(self, stripped: str) -> bool:
        """Process the stripped line and return whether we are inside a docstring."""
        for marker in ('"""', "'''"):
            if stripped.startswith(marker):
                if not self._in:
                    self._in = True
                    self._marker = marker
                    # Closed on the same line?
                    rest = stripped[3:]
                    if marker in rest:
                        self._in = False
                        self._marker = None
                    return self._in
                elif self._marker == marker:
                    self._in = False
                    self._marker = None
                    return False
        if self._in and self._marker and self._marker in stripped:
            self._in = False
            self._marker = None
        return self._in


# ── helper: list context detection ───────────────────────────────────────────

def _in_multi_item_list(line: str, term: str, min_items: int = 3) -> bool:
    """Return True if *term* appears inside a list literal with >= min_items quoted strings."""
    t = re.escape(term)
    # Find all quoted strings on this line
    all_quoted = re.findall(r"""['"][^'"]{0,120}['"]""", line)
    if len(all_quoted) < min_items:
        return False
    # Check that our term is among them
    for q in all_quoted:
        inner = q[1:-1]
        if inner.lower() == term.lower():
            return True
    return False


def _in_markdown_code_fence(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("```") or stripped.startswith("`")


# ── helper: dynamic term extraction ──────────────────────────────────────────

def _extract_description_tokens(text: str) -> list[str]:
    """Extract backtick-quoted identifiers and file names from a description."""
    tokens: list[str] = []
    # Backtick-quoted names
    for m in re.finditer(r"`([^`]{1,80})`", text):
        token = m.group(1).strip()
        if _is_worth_auditing(token) and token not in tokens:
            tokens.append(token)
    # File name references
    for m in re.finditer(r"\b([\w.-]+\.(?:csv|tsv|xlsx|parquet|txt))\b", text, re.IGNORECASE):
        token = m.group(1)
        if token not in tokens:
            tokens.append(token)
    return tokens


def _is_worth_auditing(term: str) -> bool:
    """Return True if the term is specific enough to be worth auditing.

    Filters out:
    - Very short terms (single characters, common 2-letter words)
    - Generic Python / pandas identifiers that would produce huge false-positive noise
    - Common English words unlikely to be column names
    """
    if len(term) < 3:
        return False
    lower = term.lower()
    # Generic identifiers that are fine to appear in any codebase
    generic = {
        "id", "val", "row", "col", "key", "num", "str", "int", "float",
        "true", "false", "none", "null", "nan", "inf",
        "name", "type", "date", "time", "year", "month", "week",
        "datetime", "timestamp", "date_time",          # Python type / module names
        "mean", "std", "min", "max", "sum", "count",  # pandas aggregation names
        "train", "test", "data", "value", "label",    # very generic ML terms
        "target", "feature", "index", "column", "row",
        "left", "right", "inner", "outer",            # merge operations
        "ignore", "raise", "coerce",                  # pandas error_handling
    }
    if lower in generic:
        return False
    # Terms that are substrings of common builtins
    if lower in {"len", "set", "map", "zip", "any", "all", "not"}:
        return False
    return True


def _find_table_files(data_dir: Path) -> list[Path]:
    exts = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}
    return sorted(
        p for p in data_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def _guess_file_role(filename: str) -> str:
    lower = filename.lower()
    if "train" in lower:
        return "train"
    if "test" in lower or "val" in lower:
        return "validation"
    if "sample" in lower or "submission" in lower:
        return "sample_submission"
    return "data"


def _read_header(path: Path) -> list[str]:
    """Read only the header row from a tabular file."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix == ".tsv" else ","
        with path.open(encoding="utf-8", errors="replace") as fh:
            header = fh.readline().strip()
        return [c.strip().strip('"').strip("'") for c in header.split(sep) if c.strip()]
    try:
        import pandas as pd
        if suffix in {".xlsx", ".xls"}:
            return list(pd.read_excel(path, nrows=0).columns)
        if suffix == ".parquet":
            return list(pd.read_parquet(path, columns=[]).columns)
    except Exception:
        pass
    return []
