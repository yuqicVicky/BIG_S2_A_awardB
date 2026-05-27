"""
S1: Feature Engineering
Builds train and val feature matrices with no leakage.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd

REPO = "/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo"
RUN_ID = "20260525_001143_cc9cbec2"

PERIOD_MAP = {
    "2019-01-31": "uTjgI1Sv", "2019-02-28": "wf016pk5", "2019-03-31": "BkTW58Ff",
    "2019-04-30": "shDD7wDP", "2019-05-31": "aZFXT65l", "2019-06-30": "fizSTkFs",
    "2019-07-31": "wQLd1SNL", "2019-08-31": "kmxcVN2e", "2019-09-30": "x24Jbzaz",
    "2019-10-31": "FuLb1kk4", "2019-11-30": "1Tl9271R", "2019-12-31": "Fj7ebbrB",
    "2020-01-31": "h9Re4kM3", "2020-02-29": "gqVDbZc7", "2020-03-31": "UKjQnuej",
    "2020-04-30": "TedmliP4", "2020-05-31": "PJu8Wb2C", "2020-06-30": "mZcpe0Ud",
    "2020-07-31": "NOtvYKB9", "2020-08-31": "YBlSTfgc", "2020-09-30": "rX3aMRGn",
    "2020-10-31": "44RA6kMl", "2020-11-30": "88NtGYTF", "2020-12-31": "tVb8fHGc",
    "2021-01-31": "aHT3VIho", "2021-02-28": "a239r7U4", "2021-03-31": "kTaI18at",
    "2021-04-30": "FZxIVFvr", "2021-05-31": "27DK2m8F", "2021-06-30": "y5ysDDpd",
    "2021-07-31": "KDy1VIvO", "2021-08-31": "N81HwK1a", "2021-09-30": "OIpwoBOI",
    "2021-10-31": "S2Qn2n8u", "2021-11-30": "iIi2mgES", "2021-12-31": "lSdEh765",
    "2022-01-31": "eVeAG5UX", "2022-02-28": "Hy8SBtar", "2022-03-31": "4MVfmuye",
    "2022-04-30": "OqDkgaDk", "2022-05-31": "CDQGTxV0", "2022-06-30": "rle4IZEn",
    "2022-07-31": "OugqP9RF", "2022-08-31": "BhtGJhRU", "2022-09-30": "dp3VfN8B",
    "2022-10-31": "xtjIUpyk", "2022-11-30": "omhpgEVm", "2022-12-31": "S1xSdqr5",
    "2023-01-31": "LALpfR23", "2023-02-28": "UJgFAh3i", "2023-03-31": "Kk6iVNym",
    "2023-04-30": "4VCAqmuO", "2023-05-31": "WB9kCj4E", "2023-06-30": "j1dZWmlF",
    "2023-07-31": "MZ0ENeKD", "2023-08-31": "tLoy7Zpr", "2023-09-30": "QpWgWZqu",
    "2023-10-31": "nICRHvl9", "2023-11-30": "56aULHvm", "2023-12-31": "63zxcdKZ",
    "2024-01-31": "ePA08XXo", "2024-02-29": "i9aSkhZb", "2024-03-31": "9FQthr9A",
    "2024-04-30": "DpR0556d", "2024-07-31": "9Dp3l3qq", "2024-08-31": "7cCeqHbf",
    "2024-09-30": "lfTz14iT", "2024-11-30": "k4mmkR0U", "2024-12-31": "NWU8bRHI",
}
ID_TO_DATE = {v: k for k, v in PERIOD_MAP.items()}

CATS = ["all_drugs", "all_opioids", "all_stimulants"]
NUMERIC_COVS = [
    "unemployment_rate", "labor_force", "temp_avg_f", "precip_in",
    "gtrends_overdose", "gtrends_fentanyl", "gtrends_naloxone",
    "gtrends_opioid", "gtrends_methamphetamine",
]


def make_temporal_index(period_ids_all, known_map):
    """
    Assign a monotone integer temporal index to period IDs.
    Known periods get index from their calendar date; unknown get assigned
    stable negative indices (they likely predate the known range).
    Returns dict: period_id -> temporal_index.
    """
    known = {pid: known_map[pid] for pid in period_ids_all if pid in known_map}
    unknown = [pid for pid in period_ids_all if pid not in known_map]

    # Sort known by date
    sorted_known = sorted(known.items(), key=lambda x: x[1])
    # Build: date string -> rank
    known_ranked = {pid: rank for rank, (pid, _) in enumerate(sorted_known, start=len(unknown))}

    # Unknown get 0..n_unknown-1 (they get lower indices; stable sort)
    unknown_ranked = {pid: i for i, pid in enumerate(sorted(unknown))}

    result = {}
    result.update(unknown_ranked)
    result.update(known_ranked)
    return result


def get_month_features(temporal_index_map, id_to_date):
    """Return dict period_id -> (sin_month, cos_month, month). Unknown get NaN."""
    month_feats = {}
    for pid, tidx in temporal_index_map.items():
        if pid in id_to_date:
            date_str = id_to_date[pid]
            month = int(date_str[5:7])
            sin_m = np.sin(2 * np.pi * month / 12)
            cos_m = np.cos(2 * np.pi * month / 12)
        else:
            month = np.nan
            sin_m = np.nan
            cos_m = np.nan
        month_feats[pid] = (sin_m, cos_m, month)
    return month_feats


def main():
    print("=== S1: Feature Engineering ===")

    # Load data
    train_target = pd.read_csv(f"{REPO}/data/train/dose_sys_train.csv")
    train_cov = pd.read_csv(f"{REPO}/data/train/covariates.csv")
    val_cov = pd.read_csv(f"{REPO}/data/val/covariates.csv")
    sub_template = pd.read_csv(f"{REPO}/data/sample_submission.csv")

    # Filter target to 3 categories
    train_target = train_target[train_target["overdose_category"].isin(CATS)].copy()

    # Build temporal index over training period IDs
    all_train_period_ids = train_cov["period_id"].unique().tolist()
    temporal_index_map = make_temporal_index(all_train_period_ids, ID_TO_DATE)
    print(f"Training temporal index: {len(temporal_index_map)} periods, range [{min(temporal_index_map.values())}, {max(temporal_index_map.values())}]")

    # Build temporal index for val (val gets higher indices than all training)
    val_period_ids = val_cov["period_id"].unique().tolist()
    # All val periods are after training (true temporal holdout), assign them >max_train
    max_train_idx = max(temporal_index_map.values())
    val_temporal_map = {}
    for i, pid in enumerate(sorted(val_period_ids)):
        val_temporal_map[pid] = max_train_idx + 1 + i
    print(f"Val temporal index: {val_temporal_map}")

    # Month features
    month_feats_train = get_month_features(temporal_index_map, ID_TO_DATE)
    month_feats_val = get_month_features(val_temporal_map, ID_TO_DATE)

    # Jurisdiction encoding - fit on training
    all_jurisdictions = sorted(train_cov["jurisdiction"].unique().tolist())
    jur_to_code = {j: i for i, j in enumerate(all_jurisdictions)}
    print(f"Jurisdictions: {len(all_jurisdictions)}")

    # Imputer: fit median on training numeric covariates only
    train_medians = {}
    for col in NUMERIC_COVS:
        train_medians[col] = float(train_cov[col].median())
    print(f"Training medians computed for imputation: {list(train_medians.keys())}")

    # Merge train target with covariates
    train_merged = train_target.merge(
        train_cov[["period_id", "jurisdiction"] + NUMERIC_COVS],
        on=["period_id", "jurisdiction"],
        how="left",
    )
    print(f"After merge: {train_merged.shape}")
    missing_after_merge = train_merged["unemployment_rate"].isnull().sum()
    print(f"Missing unemployment_rate after merge: {missing_after_merge}")

    # Apply median imputation (training median, fitted on training only)
    for col in NUMERIC_COVS:
        train_merged[col] = train_merged[col].fillna(train_medians[col])

    # Add engineered features
    train_merged["log_labor_force"] = np.log1p(train_merged["labor_force"])
    train_merged["temporal_index"] = train_merged["period_id"].map(temporal_index_map)
    train_merged["sin_month"] = train_merged["period_id"].map(lambda p: month_feats_train.get(p, (np.nan, np.nan, np.nan))[0])
    train_merged["cos_month"] = train_merged["period_id"].map(lambda p: month_feats_train.get(p, (np.nan, np.nan, np.nan))[1])
    train_merged["month"] = train_merged["period_id"].map(lambda p: month_feats_train.get(p, (np.nan, np.nan, np.nan))[2])
    train_merged["jurisdiction_code"] = train_merged["jurisdiction"].map(jur_to_code)

    # For unknown period months, impute sin/cos with mean of known values
    sin_mean = float(np.nanmean(train_merged["sin_month"]))
    cos_mean = float(np.nanmean(train_merged["cos_month"]))
    month_mean = float(np.nanmean(train_merged["month"]))
    train_merged["sin_month"] = train_merged["sin_month"].fillna(sin_mean)
    train_merged["cos_month"] = train_merged["cos_month"].fillna(cos_mean)
    train_merged["month"] = train_merged["month"].fillna(month_mean)

    # ====================================================================
    # Historical stats: per (jurisdiction, overdose_category)
    # LEAKAGE CHECK: computed ONLY on training data (not holdout, not val)
    # We compute overall historical mean/std from all training rows here.
    # In modeling steps, when computing holdout predictions, the historical
    # stats must exclude holdout periods. We pre-compute the full-training
    # version here for the refit; holdout-safe version computed in S3.
    # ====================================================================
    hist_stats = (
        train_merged
        .groupby(["jurisdiction", "overdose_category"])["rate_per_10000_ed_visits"]
        .agg(hist_mean="mean", hist_std="std")
        .reset_index()
    )
    hist_stats["hist_std"] = hist_stats["hist_std"].fillna(0.0)
    train_merged = train_merged.merge(hist_stats, on=["jurisdiction", "overdose_category"], how="left")

    # log1p target
    train_merged["log_rate"] = np.log1p(train_merged["rate_per_10000_ed_visits"])

    print(f"Final train feature shape: {train_merged.shape}")
    print(f"NaN check - any NaN in feature cols: {train_merged[NUMERIC_COVS + ['log_labor_force','temporal_index','jurisdiction_code']].isnull().any().any()}")

    # ====================================================================
    # Build val feature matrix
    # ====================================================================
    # Cross-join val covariates with overdose categories
    val_cats = pd.DataFrame({"overdose_category": CATS})
    val_cov_expanded = val_cov.assign(key=1).merge(val_cats.assign(key=1), on="key").drop("key", axis=1)

    # Apply training-fitted median imputation to val
    for col in NUMERIC_COVS:
        val_cov_expanded[col] = val_cov_expanded[col].fillna(train_medians[col])

    val_cov_expanded["log_labor_force"] = np.log1p(val_cov_expanded["labor_force"])
    val_cov_expanded["temporal_index"] = val_cov_expanded["period_id"].map(val_temporal_map)
    val_cov_expanded["sin_month"] = val_cov_expanded["period_id"].map(
        lambda p: month_feats_val.get(p, (sin_mean, cos_mean, month_mean))[0]
    )
    val_cov_expanded["cos_month"] = val_cov_expanded["period_id"].map(
        lambda p: month_feats_val.get(p, (sin_mean, cos_mean, month_mean))[1]
    )
    val_cov_expanded["month"] = val_cov_expanded["period_id"].map(
        lambda p: month_feats_val.get(p, (sin_mean, cos_mean, month_mean))[2]
    )
    val_cov_expanded["jurisdiction_code"] = val_cov_expanded["jurisdiction"].map(jur_to_code)

    # For val jurisdiction codes: any unseen jurisdiction gets -1
    val_cov_expanded["jurisdiction_code"] = val_cov_expanded["jurisdiction_code"].fillna(-1).astype(int)

    # Historical stats for val: use full-training hist_stats (no val data)
    val_cov_expanded = val_cov_expanded.merge(hist_stats, on=["jurisdiction", "overdose_category"], how="left")
    val_cov_expanded["hist_mean"] = val_cov_expanded["hist_mean"].fillna(val_cov_expanded["hist_mean"].mean())
    val_cov_expanded["hist_std"] = val_cov_expanded["hist_std"].fillna(0.0)

    print(f"Final val feature shape: {val_cov_expanded.shape}")
    print(f"NaN check val - any NaN in key cols: {val_cov_expanded[NUMERIC_COVS + ['log_labor_force','temporal_index','jurisdiction_code']].isnull().any().any()}")

    # Save artifacts
    os.makedirs(f"{REPO}/outputs/artifacts", exist_ok=True)
    train_merged.to_parquet(f"{REPO}/outputs/artifacts/train_features.parquet", index=False)
    val_cov_expanded.to_parquet(f"{REPO}/outputs/artifacts/val_features.parquet", index=False)

    # Save encoders for inspection
    encoders = {
        "jur_to_code": jur_to_code,
        "train_medians": train_medians,
        "temporal_index_map": temporal_index_map,
        "val_temporal_map": val_temporal_map,
        "sin_mean": sin_mean,
        "cos_mean": cos_mean,
        "month_mean": month_mean,
    }
    with open(f"{REPO}/outputs/artifacts/encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    # Save hist_stats for inspection
    hist_stats.to_csv(f"{REPO}/outputs/artifacts/hist_stats_full_train.csv", index=False)

    summary = {
        "train_feature_shape": list(train_merged.shape),
        "val_feature_shape": list(val_cov_expanded.shape),
        "feature_columns": [c for c in train_merged.columns if c not in ["period_id", "jurisdiction", "overdose_category", "rate_per_10000_ed_visits", "log_rate"]],
        "temporal_index_range_train": [int(min(temporal_index_map.values())), int(max(temporal_index_map.values()))],
        "val_temporal_index_range": [int(min(val_temporal_map.values())), int(max(val_temporal_map.values()))],
        "leakage_check": {
            "hist_stats_computed_from": "all training rows only (no val data)",
            "imputer_fitted_from": "training covariates only",
            "val_uses_training_medians": True,
            "val_temporal_index_gt_train_max": True,
        },
        "n_nan_train": int(train_merged[NUMERIC_COVS + ["log_labor_force", "temporal_index", "jurisdiction_code"]].isnull().sum().sum()),
        "n_nan_val": int(val_cov_expanded[NUMERIC_COVS + ["log_labor_force", "temporal_index", "jurisdiction_code"]].isnull().sum().sum()),
    }

    with open(f"{REPO}/outputs/artifacts/s1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nS1 COMPLETE")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
