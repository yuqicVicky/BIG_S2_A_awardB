"""Regression tests for generic datetime feature extraction.

These tests verify the core invariant: any column that can be parsed as a
datetime — including the row_id column — generates rich time features (hour,
dayofweek, month, year, is_weekend, …) even when the raw column is excluded
from the model feature set.  No dataset-specific field names are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_agent.features import (
    _detect_datetime_columns,
    _resolve_time_sources,
    _add_features_for_datetime_col,
    _extract_time_features,
    _has_time_component,
    _audit_time_target_signal,
)
from src.data_agent.schema import SchemaSpec


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_spec(row_id_col: str, target_col: str, time_col: str | None = None) -> SchemaSpec:
    return SchemaSpec(
        data_dir=".",
        description_path=None,
        train_target_file=".",
        train_covariates_file=None,
        validation_covariates_file=None,
        sample_submission_file=".",
        row_id_column=row_id_col,
        target_column=target_col,
        join_keys=[],
        time_column=time_col,
    )


def _hourly_train_df(n: int = 48) -> pd.DataFrame:
    """Toy hourly dataframe whose row_id is a datetime string."""
    times = pd.date_range("2023-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp": times.strftime("%Y-%m-%d %H:%M:%S"),
        "feature_a": rng.normal(size=n),
        "target": rng.integers(0, 200, size=n).astype(float),
    })


def _daily_train_df(n: int = 30) -> pd.DataFrame:
    """Toy daily dataframe whose row_id is a date string."""
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "feature_b": rng.normal(size=n),
        "target": rng.normal(size=n),
    })


# ── _detect_datetime_columns ──────────────────────────────────────────────────

class TestDetectDatetimeColumns:
    def test_detects_string_datetime_column(self):
        df = _hourly_train_df()
        found = _detect_datetime_columns(df)
        assert "timestamp" in found

    def test_skips_numeric_columns(self):
        df = _hourly_train_df()
        found = _detect_datetime_columns(df)
        assert "feature_a" not in found
        assert "target" not in found

    def test_detects_already_parsed_datetime64(self):
        df = pd.DataFrame({"ts": pd.date_range("2023-01-01", periods=5, freq="h"), "val": range(5)})
        found = _detect_datetime_columns(df)
        assert "ts" in found

    def test_ignores_non_datetime_string_column(self):
        df = pd.DataFrame({"name": ["alice", "bob", "carol"], "val": [1, 2, 3]})
        found = _detect_datetime_columns(df)
        assert "name" not in found

    def test_daily_date_strings_detected(self):
        df = _daily_train_df()
        found = _detect_datetime_columns(df)
        assert "date" in found


# ── _resolve_time_sources ────────────────────────────────────────────────────

class TestResolveTimeSources:
    def test_row_id_datetime_is_included(self):
        df = _hourly_train_df()
        detected = _detect_datetime_columns(df)
        spec = _make_spec(row_id_col="timestamp", target_col="target")
        sources = _resolve_time_sources(detected, spec)
        assert "timestamp" in sources

    def test_explicit_time_column_first(self):
        df = _hourly_train_df()
        detected = _detect_datetime_columns(df)
        spec = _make_spec(row_id_col="timestamp", target_col="target", time_col="timestamp")
        sources = _resolve_time_sources(detected, spec)
        assert sources[0] == "timestamp"
        assert sources.count("timestamp") == 1  # no duplicates

    def test_target_column_excluded(self):
        # Even if target column were a datetime string, it must not be a source.
        df = pd.DataFrame({
            "ts": pd.date_range("2023-01-01", periods=5, freq="h").strftime("%Y-%m-%d %H:%M:%S"),
            "outcome": pd.date_range("2024-01-01", periods=5, freq="h").strftime("%Y-%m-%d %H:%M:%S"),
        })
        detected = _detect_datetime_columns(df)
        spec = _make_spec(row_id_col="ts", target_col="outcome")
        sources = _resolve_time_sources(detected, spec)
        assert "outcome" not in sources


# ── _has_time_component ───────────────────────────────────────────────────────

class TestHasTimeComponent:
    def test_hourly_data_has_time_component(self):
        parsed = pd.to_datetime(pd.Series(["2023-01-01 08:00:00", "2023-01-01 09:00:00"]))
        assert _has_time_component(parsed) is True

    def test_daily_data_has_no_time_component(self):
        parsed = pd.to_datetime(pd.Series(["2023-01-01", "2023-01-02", "2023-01-03"]))
        assert _has_time_component(parsed) is False

    def test_empty_series_returns_false(self):
        assert _has_time_component(pd.Series(dtype="datetime64[ns]")) is False


# ── _add_features_for_datetime_col ───────────────────────────────────────────

class TestAddFeaturesForDatetimeCol:
    def test_generates_core_time_features(self):
        train = _hourly_train_df()
        pred = _hourly_train_df(n=10)
        train_out, pred_out, added = _add_features_for_datetime_col(train, pred, "timestamp")

        required = [
            "timestamp__year", "timestamp__month", "timestamp__month_sin", "timestamp__month_cos",
            "timestamp__day", "timestamp__dayofweek", "timestamp__dayofweek_sin", "timestamp__dayofweek_cos",
            "timestamp__is_weekend", "timestamp__quarter", "timestamp__weekofyear", "timestamp__ordinal",
        ]
        for feat in required:
            assert feat in train_out.columns, f"Missing feature: {feat}"
            assert feat in pred_out.columns, f"Missing feature in pred: {feat}"
            assert feat in added, f"Not in added list: {feat}"

    def test_generates_hour_features_for_sub_day_data(self):
        train = _hourly_train_df()
        pred = _hourly_train_df(n=10)
        _, _, added = _add_features_for_datetime_col(train, pred, "timestamp")
        assert "timestamp__hour" in added
        assert "timestamp__hour_sin" in added
        assert "timestamp__hour_cos" in added

    def test_no_hour_features_for_daily_data(self):
        train = _daily_train_df()
        pred = _daily_train_df(n=5)
        _, _, added = _add_features_for_datetime_col(train, pred, "date")
        assert "date__hour" not in added

    def test_hour_values_are_correct(self):
        train = _hourly_train_df(n=24)
        pred = train.copy()
        train_out, _, _ = _add_features_for_datetime_col(train, pred, "timestamp")
        expected_hours = pd.to_datetime(train["timestamp"]).dt.hour.values
        actual_hours = train_out["timestamp__hour"].values
        np.testing.assert_array_equal(actual_hours, expected_hours)

    def test_is_weekend_correct_values(self):
        # 2023-01-07 is Saturday, 2023-01-02 is Monday
        train = pd.DataFrame({
            "ts": ["2023-01-07 10:00:00", "2023-01-02 10:00:00"],
            "target": [1.0, 2.0],
        })
        train_out, _, _ = _add_features_for_datetime_col(train, train.copy(), "ts")
        assert train_out["ts__is_weekend"].iloc[0] == 1  # Saturday
        assert train_out["ts__is_weekend"].iloc[1] == 0  # Monday

    def test_ordinal_maps_both_train_and_pred(self):
        """Ordinal index must cover prediction timestamps not seen in training."""
        train = _hourly_train_df(n=24)
        pred = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-02", periods=6, freq="h").strftime("%Y-%m-%d %H:%M:%S"),
            "x": range(6),
        })
        train_out, pred_out, _ = _add_features_for_datetime_col(train, pred, "timestamp")
        # Ordinals in pred must be non-null (timestamps were in combined map)
        assert pred_out["timestamp__ordinal"].notna().all()

    def test_raw_column_not_overwritten(self):
        """The original column must be preserved unchanged."""
        train = _hourly_train_df()
        original_vals = train["timestamp"].copy()
        train_out, _, _ = _add_features_for_datetime_col(train, train.copy(), "timestamp")
        pd.testing.assert_series_equal(train_out["timestamp"], original_vals)


# ── _extract_time_features (orchestrator) ────────────────────────────────────

class TestExtractTimeFeatures:
    def test_row_id_datetime_generates_features_end_to_end(self):
        """Core regression test: row_id = datetime string → features extracted."""
        train = _hourly_train_df()
        pred = _hourly_train_df(n=10)
        spec = _make_spec(row_id_col="timestamp", target_col="target")

        detected = _detect_datetime_columns(train)
        sources = _resolve_time_sources(detected, spec)
        train_out, pred_out, audit = _extract_time_features(train, pred, sources)

        assert "timestamp" in audit["generated_features"]
        assert len(audit["generated_features"]["timestamp"]) > 0
        assert "timestamp__hour" in train_out.columns
        assert "timestamp__dayofweek" in train_out.columns
        assert "timestamp__month" in train_out.columns
        assert "timestamp__year" in train_out.columns

    def test_no_datetime_columns_produces_no_features(self):
        df = pd.DataFrame({"id": [1, 2, 3], "x": [0.1, 0.2, 0.3], "y": [4, 5, 6]})
        detected = _detect_datetime_columns(df)
        sources = _resolve_time_sources(detected, _make_spec("id", "y"))
        _, _, audit = _extract_time_features(df, df.copy(), sources)
        assert all(len(v) == 0 for v in audit["generated_features"].values()) or not audit["generated_features"]


# ── _audit_time_target_signal ─────────────────────────────────────────────────

class TestAuditTimeTargetSignal:
    def test_hour_shows_signal_for_cyclic_target(self):
        """Hourly target with a strong diurnal pattern → large mean range."""
        n = 24 * 30
        times = pd.date_range("2023-01-01", periods=n, freq="h")
        # Target is high during business hours, low at night.
        target = pd.Series(np.where((times.hour >= 8) & (times.hour <= 18), 300.0, 20.0))
        df = pd.DataFrame({"hour": times.hour, "target": target})

        signal = _audit_time_target_signal(df, target, ["hour"])
        assert "hour" in signal
        assert signal["hour"]["target_mean_range"] > 100

    def test_returns_empty_for_non_numeric_target(self):
        df = pd.DataFrame({"hour": [0, 1, 2], "target": ["a", "b", "c"]})
        signal = _audit_time_target_signal(df, df["target"], ["hour"])
        assert signal == {}

    def test_high_cardinality_feature_skipped(self):
        """Features with > 50 unique values are skipped to avoid noise."""
        n = 200
        df = pd.DataFrame({"ordinal": range(n), "target": np.random.default_rng(0).normal(size=n)})
        signal = _audit_time_target_signal(df, df["target"], ["ordinal"])
        assert "ordinal" not in signal
