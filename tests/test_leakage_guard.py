"""Tests for the code-enforced precomputed-target-leakage guard.

The guard runs at the one feature chokepoint every model trains through
(``build_feature_bundle``). These tests are dataset-agnostic: they synthesise
panels rather than depend on ``data/``.
"""

import numpy as np
import pandas as pd

from src.data_agent.leakage_guard import scan_precomputed_target_leakage


def _panel(seed=0, n=600, n_groups=20):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_groups, n)
    y = g * 1.0 + rng.normal(0, 1.0, n)
    return pd.DataFrame({"g": g, "y": y, "benign_noise": rng.normal(0, 1, n)})


def test_clean_feature_set_passes():
    df = _panel()
    # Only benign, non-target-derived features.
    res = scan_precomputed_target_leakage(df, ["benign_noise"], "y",
                                          group_aggregate_keys=[["g"]])
    assert res["status"] == "pass"
    assert res["findings"] == []


def test_catches_monotone_target_transform():
    df = _panel()
    df["y_times_two"] = df["y"] * 2.0
    res = scan_precomputed_target_leakage(df, ["benign_noise", "y_times_two"], "y")
    assert res["status"] == "fail"
    assert any(f["column"] == "y_times_two" for f in res["findings"])


def test_catches_outcome_name_token():
    df = _panel()
    df["target_leak"] = df["benign_noise"]
    res = scan_precomputed_target_leakage(df, ["target_leak"], "y")
    assert res["status"] == "fail"
    assert any(f["issue"] == "leakage_name_token" for f in res["findings"])


def test_catches_disguised_precomputed_group_aggregate():
    """The unique value-add: a full-data per-group target mean with an innocuous
    name and low *global* correlation (large within-group noise) — invisible to
    the name-token and correlation checks, caught by reconstruction."""
    rng = np.random.default_rng(7)
    n = 1200
    g = rng.integers(0, 40, n)
    y = rng.normal(0, 0.3, 40)[g] + rng.normal(0, 5.0, n)
    df = pd.DataFrame({"region": g, "y": y})
    df["region_profile"] = df.groupby("region")["y"].transform("mean")

    glob_r = abs(float(np.corrcoef(df["region_profile"], df["y"])[0, 1]))
    assert glob_r < 0.98  # evades the correlation check

    res = scan_precomputed_target_leakage(df, ["region_profile"], "y",
                                          group_aggregate_keys=[["region"]])
    assert res["status"] == "fail"
    assert any(f["issue"] == "precomputed_group_aggregate" for f in res["findings"])


def test_benign_group_aggregate_of_nontarget_not_flagged():
    df = _panel()
    df["grp_noise_mean"] = df.groupby("g")["benign_noise"].transform("mean")
    res = scan_precomputed_target_leakage(df, ["grp_noise_mean"], "y",
                                          group_aggregate_keys=[["g"]])
    assert res["status"] == "pass"


def test_missing_target_is_warn_not_false_pass():
    df = _panel().drop(columns=["y"])
    res = scan_precomputed_target_leakage(df, ["benign_noise"], "y")
    assert res["status"] == "warn"


def test_guard_to_verdict_schema():
    df = _panel()
    df["y_copy"] = df["y"]
    res = scan_precomputed_target_leakage(df, ["y_copy"], "y")
    from src.data_agent.leakage_guard import guard_to_verdict
    v = guard_to_verdict(res)
    assert v.stage == "leakage"
    assert v.status == "fail"
    assert "y_copy" in v.suggested_corrections.get("drop_or_recompute_per_fold", [])
    assert v.to_dict()["critic"] == "code:leakage_guard"
