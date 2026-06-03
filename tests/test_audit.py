"""Regression tests for the anti-hardcoding audit module.

Key invariants tested:
1. Known bad patterns are flagged as unacceptable.
2. Legitimate schema-discovery heuristics (multi-item candidate lists,
   comments, docstrings, test files) are classified as acceptable.
3. Dynamic terms extracted from a toy DATA_DESCRIPTION.md are found.
4. The verdict maps correctly to finding classifications.
5. The audit does not produce false positives on the live codebase.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from src.data_agent.audit import (
    AuditResult,
    _classify,
    _DocstringTracker,
    _extract_description_tokens,
    _in_multi_item_list,
    _is_worth_auditing,
    extract_dynamic_terms,
    run_audit,
    write_audit_log,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_result(risky: int = 0, unacceptable: int = 0) -> AuditResult:
    from src.data_agent.audit import AuditFinding
    findings = []
    for _ in range(risky):
        findings.append(AuditFinding(
            file_path="src/foo.py", line_number=1, line_content="",
            term="x", term_category="test", term_source="static",
            classification="risky", reason="test",
        ))
    for _ in range(unacceptable):
        findings.append(AuditFinding(
            file_path="src/foo.py", line_number=2, line_content="",
            term="y", term_category="test", term_source="static",
            classification="unacceptable", reason="test",
        ))
    return AuditResult(
        phase="pre", run_id="test", timestamp="",
        scanned_files=[], static_terms_searched={}, dynamic_terms_searched={},
        findings=findings,
        n_acceptable=0, n_risky=risky, n_unacceptable=unacceptable,
        verdict="fail" if unacceptable else ("warn" if risky else "pass"),
    )


# ── _classify ─────────────────────────────────────────────────────────────────

class TestClassify:
    def _cls(self, line: str, term: str = "count", **kwargs) -> str:
        defaults = dict(
            rel_path="src/foo.py",
            is_always_acceptable=False,
            is_markdown=False,
            is_test=False,
            is_comment=False,
            in_docstring=False,
        )
        defaults.update(kwargs)
        classification, _ = _classify(line=line, term=term, **defaults)
        return classification

    # ── acceptable contexts ───────────────────────────────────────────────────

    def test_comment_line_is_acceptable(self):
        assert self._cls('    # target_col = "count"', is_comment=True) == "acceptable"

    def test_docstring_line_is_acceptable(self):
        assert self._cls('    "count"', in_docstring=True) == "acceptable"

    def test_test_file_is_acceptable(self):
        assert self._cls('    df["count"]', is_test=True) == "acceptable"

    def test_always_acceptable_path(self):
        assert self._cls('    df["count"]', is_always_acceptable=True) == "acceptable"

    def test_markdown_prose_is_acceptable(self):
        assert self._cls("The target column is `count`.", is_markdown=True) == "acceptable"

    def test_multi_item_candidate_list_acceptable(self):
        # 3+ items in a candidate list → acceptable heuristic
        line = '    cols = ["row_id", "rowid", "count", "id"]'
        assert self._cls(line, term="count") == "acceptable"

    # ── unacceptable patterns ─────────────────────────────────────────────────

    def test_direct_column_access_is_unacceptable(self):
        assert self._cls('    val = df["count"]') == "unacceptable"

    def test_target_col_assignment_is_unacceptable(self):
        assert self._cls('    target_col = "count"') == "unacceptable"

    def test_target_column_assignment_is_unacceptable(self):
        assert self._cls('    target_column = "count"') == "unacceptable"

    def test_equality_check_is_unacceptable(self):
        assert self._cls('    if col == "count":') == "unacceptable"

    def test_read_csv_hardcoded_is_unacceptable(self):
        assert self._cls('    df = pd.read_csv("train.csv")', term="train.csv") == "unacceptable"

    # ── risky patterns ────────────────────────────────────────────────────────

    def test_single_item_list_is_risky(self):
        assert self._cls('    keys = ["count"]') == "risky"

    def test_two_item_list_is_risky(self):
        line = '    candidates = ["count", "total"]'
        assert self._cls(line) == "risky"

    def test_in_operator_check_is_risky(self):
        # `if "term" in list:` — membership check, risky but not unacceptable
        assert self._cls('    if "fare_amount" in columns:', term="fare_amount") == "risky"


# ── _DocstringTracker ─────────────────────────────────────────────────────────

class TestDocstringTracker:
    def test_enters_and_exits_docstring(self):
        tracker = _DocstringTracker()
        assert tracker.update('"""') is True       # open
        assert tracker.update("some text") is True  # inside
        assert tracker.update('"""') is False       # close

    def test_single_line_docstring(self):
        tracker = _DocstringTracker()
        # Opens and closes on same line
        assert tracker.update('"""short docstring"""') is False

    def test_not_in_docstring_by_default(self):
        tracker = _DocstringTracker()
        assert tracker.update("x = 1") is False

    def test_single_quote_docstring(self):
        tracker = _DocstringTracker()
        assert tracker.update("'''") is True
        assert tracker.update("inside") is True
        assert tracker.update("'''") is False


# ── _in_multi_item_list ───────────────────────────────────────────────────────

class TestInMultiItemList:
    def test_three_item_list_detected(self):
        line = '    _find_named_column(cols, ["row_id", "rowid", "count"])'
        assert _in_multi_item_list(line, "count", min_items=3) is True

    def test_single_item_list_not_detected_at_min3(self):
        line = '    keys = ["count"]'
        assert _in_multi_item_list(line, "count", min_items=3) is False

    def test_two_item_list_detected_at_min2(self):
        line = '    keys = ["count", "total"]'
        assert _in_multi_item_list(line, "count", min_items=2) is True

    def test_term_not_in_list(self):
        line = '    keys = ["row_id", "rowid", "identifier"]'
        assert _in_multi_item_list(line, "count", min_items=3) is False


# ── _is_worth_auditing ────────────────────────────────────────────────────────

class TestIsWorthAuditing:
    def test_generic_pandas_terms_excluded(self):
        for term in ["mean", "std", "count", "min", "max", "sum"]:
            assert _is_worth_auditing(term) is False, f"'{term}' should be excluded"

    def test_dataset_specific_terms_included(self):
        for term in ["rate_per_10000_ed_visits", "casual", "registered", "saleprice"]:
            assert _is_worth_auditing(term) is True, f"'{term}' should be included"

    def test_very_short_terms_excluded(self):
        assert _is_worth_auditing("id") is False
        assert _is_worth_auditing("a") is False

    def test_specific_column_name_included(self):
        assert _is_worth_auditing("overdose_category") is True


# ── _extract_description_tokens ───────────────────────────────────────────────

class TestExtractDescriptionTokens:
    def test_extracts_backtick_tokens(self):
        text = "Predict `fare_amount`. The row ID is `trip_id`. See `train.csv`."
        tokens = _extract_description_tokens(text)
        assert "fare_amount" in tokens
        assert "trip_id" in tokens
        assert "train.csv" in tokens

    def test_deduplicates(self):
        text = "Columns: `price`, `price`, `volume`."
        tokens = _extract_description_tokens(text)
        assert tokens.count("price") == 1

    def test_skips_too_long_tokens(self):
        long = "x" * 90
        text = f"See `{long}`."
        tokens = _extract_description_tokens(text)
        assert long not in tokens


# ── run_audit integration ─────────────────────────────────────────────────────

class TestRunAudit:
    def test_detects_hardcoded_column_access(self, tmp_path):
        """A Python file with df["fare_amount"] must produce an unacceptable finding."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        bad_file = src_dir / "bad.py"
        # Use a dataset-specific term that passes _is_worth_auditing
        bad_file.write_text('target = df["fare_amount"]\n', encoding="utf-8")

        result = run_audit(
            repo_root=tmp_path,
            run_id="test001",
            phase="pre",
            dynamic_terms={"train_columns": ["fare_amount"]},
        )
        unacceptable = [f for f in result.findings if f.classification == "unacceptable"]
        assert len(unacceptable) >= 1
        assert any(f.term == "fare_amount" for f in unacceptable)
        assert result.verdict == "fail"

    def test_ignores_test_files(self, tmp_path):
        """The same pattern in a tests/ file must be acceptable."""
        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_something.py"
        test_file.write_text('val = df["count"]\n', encoding="utf-8")

        result = run_audit(
            repo_root=tmp_path,
            run_id="test002",
            phase="pre",
            dynamic_terms={"train_columns": ["count"]},
        )
        unacceptable = [f for f in result.findings if f.classification == "unacceptable"]
        assert len(unacceptable) == 0

    def test_accepts_multi_item_candidate_list(self, tmp_path):
        """A fallback heuristic list with 3+ items must not be flagged as unacceptable."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        good_file = src_dir / "schema.py"
        good_file.write_text(
            'cols = _find_named_column(columns, ["row_id", "rowid", "id", "count"])\n',
            encoding="utf-8",
        )
        result = run_audit(
            repo_root=tmp_path,
            run_id="test003",
            phase="pre",
            dynamic_terms={"train_columns": ["count"]},
        )
        unacceptable = [f for f in result.findings if f.classification == "unacceptable"]
        assert len(unacceptable) == 0

    def test_comments_not_flagged(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        file = src_dir / "features.py"
        file.write_text(
            '# Do not hardcode "count" or "datetime" column names\n',
            encoding="utf-8",
        )
        result = run_audit(
            repo_root=tmp_path, run_id="test004", phase="pre",
            dynamic_terms={"train_columns": ["count"]},
        )
        unacceptable = [f for f in result.findings if f.classification == "unacceptable"]
        assert len(unacceptable) == 0

    def test_static_term_award_a_flagged(self, tmp_path):
        """A static forbidden term from Award A must be flagged."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        bad_file = src_dir / "runner.py"
        bad_file.write_text(
            'TARGET = "rate_per_10000_ed_visits"\n',
            encoding="utf-8",
        )
        result = run_audit(repo_root=tmp_path, run_id="test005", phase="pre")
        unacceptable = [f for f in result.findings if f.classification == "unacceptable"]
        assert any(f.term == "rate_per_10000_ed_visits" for f in unacceptable)

    def test_pass_verdict_on_clean_code(self, tmp_path):
        """Clean code with no suspicious terms gets a pass verdict."""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        clean_file = src_dir / "clean.py"
        clean_file.write_text(
            textwrap.dedent("""\
                def add(a, b):
                    return a + b
            """),
            encoding="utf-8",
        )
        result = run_audit(repo_root=tmp_path, run_id="test006", phase="pre")
        assert result.verdict == "pass"
        assert result.n_unacceptable == 0

    def test_verdict_fail_when_unacceptable(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "bad.py").write_text('val = df["survived"]\n', encoding="utf-8")
        result = run_audit(repo_root=tmp_path, run_id="test007", phase="pre")
        assert result.verdict == "fail"

    def test_verdict_warn_when_only_risky(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "risky.py").write_text(
            'keys = ["casual", "registered"]\n', encoding="utf-8"
        )
        result = run_audit(repo_root=tmp_path, run_id="test008", phase="pre")
        assert result.verdict in ("warn", "fail")


# ── write_audit_log ───────────────────────────────────────────────────────────

class TestWriteAuditLog:
    def test_writes_valid_json(self, tmp_path):
        result = _make_result(risky=1)
        path = write_audit_log(result, tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert "findings" in data
        assert "verdict" in data
        assert data["verdict"] == "warn"

    def test_filename_includes_phase_and_run_id(self, tmp_path):
        result = _make_result()
        result.phase = "post"
        result.run_id = "abc123"
        path = write_audit_log(result, tmp_path)
        assert "post" in path.name
        assert "abc123" in path.name


# ── live codebase smoke test ──────────────────────────────────────────────────

class TestLiveCodebase:
    def test_no_award_a_terms_in_src(self):
        """The live src/ must not contain any Award A field names at runtime paths."""
        repo_root = Path(__file__).parent.parent
        result = run_audit(repo_root=repo_root, run_id="live_test", phase="pre")
        award_a_unacceptable = [
            f for f in result.findings
            if f.classification == "unacceptable"
            and f.term_category == "award_a_field_names"
        ]
        assert award_a_unacceptable == [], (
            f"Live src/ contains Award A field names: "
            f"{[(f.file_path, f.line_number, f.term) for f in award_a_unacceptable]}"
        )
