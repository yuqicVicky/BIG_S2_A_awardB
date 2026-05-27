"""
S3: LightGBM Model Training and Evaluation
Train one LightGBM model per overdose category. Evaluate on holdout.
Compare against baseline. All metrics labeled with split.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

REPO = "/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo"
RUN_ID = "20260525_001143_cc9cbec2"
CATS = ["all_drugs", "all_opioids", "all_stimulants"]
N_HOLDOUT = 8
RANDOM_SEED = 42

FEATURE_COLS = [
    "unemployment_rate", "labor_force", "temp_avg_f", "precip_in",
    "gtrends_overdose", "gtrends_fentanyl", "gtrends_naloxone",
    "gtrends_opioid", "gtrends_methamphetamine",
    "log_labor_force", "temporal_index", "sin_month", "cos_month", "month",
    "jurisdiction_code", "hist_mean", "hist_std",
]

LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def r2(y_true, y_pred):
    ss_res = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
    ss_tot = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def main():
    print("=== S3: LightGBM Training ===")

    train_feat = pd.read_parquet(f"{REPO}/outputs/artifacts/train_features.parquet")
    with open(f"{REPO}/outputs/artifacts/baseline_metrics.json") as f:
        baseline = json.load(f)

    holdout_periods = baseline["holdout_periods"]
    train_periods = [p for p in train_feat["period_id"].unique() if p not in holdout_periods]

    holdout_df = train_feat[train_feat["period_id"].isin(holdout_periods)].copy()
    train_df = train_feat[train_feat["period_id"].isin(train_periods)].copy()

    print(f"Training rows: {len(train_df)}, Holdout rows: {len(holdout_df)}")
    print(f"Holdout periods: {len(holdout_periods)}")

    # LEAKAGE CHECK: hist_mean/hist_std in train_df was computed from FULL training data.
    # For the holdout-only evaluation we must recompute hist_stats from non-holdout rows only.
    hist_stats_train_only = (
        train_df.groupby(["jurisdiction", "overdose_category"])["rate_per_10000_ed_visits"]
        .agg(hist_mean_safe="mean", hist_std_safe="std")
        .reset_index()
    )
    hist_stats_train_only["hist_std_safe"] = hist_stats_train_only["hist_std_safe"].fillna(0.0)

    # Recompute hist features for holdout using only non-holdout data
    holdout_df = holdout_df.drop(columns=["hist_mean", "hist_std"])
    holdout_df = holdout_df.merge(
        hist_stats_train_only, on=["jurisdiction", "overdose_category"], how="left"
    )
    holdout_df = holdout_df.rename(columns={"hist_mean_safe": "hist_mean", "hist_std_safe": "hist_std"})
    holdout_df["hist_mean"] = holdout_df["hist_mean"].fillna(train_df["rate_per_10000_ed_visits"].mean())
    holdout_df["hist_std"] = holdout_df["hist_std"].fillna(0.0)

    # Also recompute for train_df (they will still use non-holdout stats here — same)
    # train_df already has hist_mean computed from all training; re-derive from non-holdout for consistency
    train_df = train_df.drop(columns=["hist_mean", "hist_std"])
    train_df = train_df.merge(
        hist_stats_train_only, on=["jurisdiction", "overdose_category"], how="left"
    )
    train_df = train_df.rename(columns={"hist_mean_safe": "hist_mean", "hist_std_safe": "hist_std"})

    models = {}
    holdout_metrics = {}
    train_metrics = {}

    for cat in CATS:
        print(f"\n--- Category: {cat} ---")
        tr = train_df[train_df["overdose_category"] == cat].copy()
        ho = holdout_df[holdout_df["overdose_category"] == cat].copy()

        X_tr = tr[FEATURE_COLS]
        y_tr = tr["log_rate"].values  # log1p-transformed target

        X_ho = ho[FEATURE_COLS]
        y_ho_orig = ho["rate_per_10000_ed_visits"].values  # original scale

        # Sanity: no NaN in features
        assert X_tr.isnull().sum().sum() == 0, f"NaN in train features for {cat}"
        assert X_ho.isnull().sum().sum() == 0, f"NaN in holdout features for {cat}"

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_tr, y_tr)
        models[cat] = model

        # Holdout predictions (convert back from log1p)
        y_ho_pred_log = model.predict(X_ho)
        y_ho_pred = np.expm1(y_ho_pred_log)
        y_ho_pred = np.clip(y_ho_pred, 0.01, None)

        # Train predictions
        y_tr_pred_log = model.predict(X_tr)
        y_tr_pred = np.expm1(y_tr_pred_log)
        y_tr_pred = np.clip(y_tr_pred, 0.01, None)
        y_tr_orig = np.expm1(y_tr)

        h_mae = mae(y_ho_orig, y_ho_pred)
        h_rmse = rmse(y_ho_orig, y_ho_pred)
        h_r2 = r2(y_ho_orig, y_ho_pred)

        t_mae = mae(y_tr_orig, y_tr_pred)
        t_rmse = rmse(y_tr_orig, y_tr_pred)
        t_r2 = r2(y_tr_orig, y_tr_pred)

        baseline_mae = baseline["baseline_metrics_by_category"][cat]["mae"]
        beats_baseline = h_mae < baseline_mae

        holdout_metrics[cat] = {
            "split": "holdout",
            "n_rows": int(len(ho)),
            "mae": h_mae,
            "rmse": h_rmse,
            "r2": h_r2,
            "baseline_mae": baseline_mae,
            "beats_baseline": beats_baseline,
        }
        train_metrics[cat] = {
            "split": "train",
            "n_rows": int(len(tr)),
            "mae": t_mae,
            "rmse": t_rmse,
            "r2": t_r2,
        }

        print(f"  Train  | MAE={t_mae:.4f}, RMSE={t_rmse:.4f}, R2={t_r2:.4f} (n={len(tr)})")
        print(f"  Holdout| MAE={h_mae:.4f}, RMSE={h_rmse:.4f}, R2={h_r2:.4f} (n={len(ho)})")
        print(f"  Baseline MAE={baseline_mae:.4f} | LightGBM beats baseline: {beats_baseline}")

        # Save model
        model_path = f"{REPO}/outputs/artifacts/lgbm_{cat.replace('_','')}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Model saved to: {model_path}")

    # Verify all categories beat baseline
    all_beat = all(holdout_metrics[cat]["beats_baseline"] for cat in CATS)
    print(f"\nAll categories beat baseline on holdout MAE: {all_beat}")

    # Save metrics
    full_metrics = {
        "holdout_metrics": holdout_metrics,
        "train_metrics": train_metrics,
        "all_beat_baseline": all_beat,
        "lgbm_params": LGBM_PARAMS,
        "n_holdout_periods": N_HOLDOUT,
        "holdout_periods": holdout_periods,
        "leakage_check": {
            "hist_stats_recomputed_from_non_holdout": True,
            "no_holdout_data_in_training": True,
            "log1p_target_applied": True,
            "expm1_prediction_applied": True,
        },
    }

    os.makedirs(f"{REPO}/outputs/artifacts", exist_ok=True)
    with open(f"{REPO}/outputs/artifacts/holdout_metrics.json", "w") as f:
        json.dump(full_metrics, f, indent=2)
    with open(f"{REPO}/outputs/artifacts/train_metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)

    # Save feature importance
    feat_imp = {}
    for cat in CATS:
        model = models[cat]
        imp = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        feat_imp[cat] = imp.to_dict(orient="records")
        imp.to_csv(f"{REPO}/outputs/artifacts/feat_importance_{cat}.csv", index=False)

    print("\nS3 COMPLETE")
    return full_metrics, models


if __name__ == "__main__":
    main()
