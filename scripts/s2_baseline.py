"""
S2: Baseline Model
Compute baseline predictions on internal holdout (8 most recent training periods).
Baseline: mean rate per (jurisdiction, overdose_category) from non-holdout training periods only.
"""
import os
import json
import numpy as np
import pandas as pd

REPO = "/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo"
RUN_ID = "20260525_001143_cc9cbec2"
CATS = ["all_drugs", "all_opioids", "all_stimulants"]
N_HOLDOUT = 8  # most recent training periods


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def main():
    print("=== S2: Baseline Model ===")

    train_feat = pd.read_parquet(f"{REPO}/outputs/artifacts/train_features.parquet")

    # Identify holdout periods: 8 most recent by temporal_index
    period_tidx = (
        train_feat[["period_id", "temporal_index"]]
        .drop_duplicates()
        .sort_values("temporal_index", ascending=False)
    )
    holdout_periods = period_tidx.head(N_HOLDOUT)["period_id"].tolist()
    train_periods = period_tidx.tail(len(period_tidx) - N_HOLDOUT)["period_id"].tolist()

    print(f"Holdout periods ({N_HOLDOUT}): {holdout_periods}")
    print(f"Train periods for baseline: {len(train_periods)}")

    # Split
    holdout_df = train_feat[train_feat["period_id"].isin(holdout_periods)].copy()
    train_df = train_feat[train_feat["period_id"].isin(train_periods)].copy()

    print(f"Holdout rows: {len(holdout_df)}, Train rows: {len(train_df)}")

    # LEAKAGE CHECK: hist_mean in train_df was computed from full training data.
    # For the baseline, we recompute mean from ONLY the non-holdout rows.
    baseline_stats = (
        train_df.groupby(["jurisdiction", "overdose_category"])["rate_per_10000_ed_visits"]
        .mean()
        .reset_index()
        .rename(columns={"rate_per_10000_ed_visits": "baseline_mean"})
    )

    holdout_df = holdout_df.merge(baseline_stats, on=["jurisdiction", "overdose_category"], how="left")

    # Fallback for any missing jurisdiction-category combo
    global_mean = train_df["rate_per_10000_ed_visits"].mean()
    holdout_df["baseline_mean"] = holdout_df["baseline_mean"].fillna(global_mean)

    print(f"\nBaseline metrics on HOLDOUT split:")
    baseline_metrics = {}
    for cat in CATS:
        sub = holdout_df[holdout_df["overdose_category"] == cat]
        y_true = sub["rate_per_10000_ed_visits"].values
        y_pred = sub["baseline_mean"].values
        m = {
            "split": "holdout",
            "n_rows": len(sub),
            "mae": mae(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
        }
        baseline_metrics[cat] = m
        print(f"  {cat}: MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f} (n={m['n_rows']})")

    # Also compute global baseline (mean over all training, for reference)
    all_train_global = train_feat[train_feat["period_id"].isin(train_periods)]["rate_per_10000_ed_visits"].mean()
    print(f"\nGlobal baseline mean from non-holdout train: {all_train_global:.4f}")

    summary = {
        "holdout_periods": holdout_periods,
        "n_train_periods": len(train_periods),
        "n_holdout_periods": N_HOLDOUT,
        "leakage_check": {
            "baseline_computed_from": "non-holdout training periods only",
            "holdout_data_excluded_from_baseline": True,
        },
        "baseline_metrics_by_category": baseline_metrics,
        "global_train_mean": float(all_train_global),
    }

    os.makedirs(f"{REPO}/outputs/artifacts", exist_ok=True)
    with open(f"{REPO}/outputs/artifacts/baseline_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nS2 COMPLETE - baseline_metrics.json written")
    return summary


if __name__ == "__main__":
    main()
