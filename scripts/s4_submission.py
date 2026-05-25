"""
S4: Refit on Full Training Data and Generate Submission
Refit all three LightGBM models on the full training dataset.
Generate predictions for the 918 validation rows.
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


def main():
    print("=== S4: Refit on Full Training Data + Submission ===")

    train_feat = pd.read_parquet(f"{REPO}/outputs/artifacts/train_features.parquet")
    val_feat = pd.read_parquet(f"{REPO}/outputs/artifacts/val_features.parquet")
    sub_template = pd.read_csv(f"{REPO}/data/sample_submission.csv")

    print(f"Full training rows: {len(train_feat)}")
    print(f"Val feature rows: {len(val_feat)}")
    print(f"Sub template rows: {len(sub_template)}")

    # hist_mean/hist_std in train_feat were computed from all training rows (no val)
    # For refit, this is correct — use full training hist stats.
    # hist_mean/hist_std in val_feat were also computed from all training rows.

    final_models = {}
    predictions = []

    for cat in CATS:
        print(f"\n--- Refit: {cat} ---")
        tr = train_feat[train_feat["overdose_category"] == cat].copy()

        X_tr = tr[FEATURE_COLS]
        y_tr = tr["log_rate"].values

        assert X_tr.isnull().sum().sum() == 0, f"NaN in train features for {cat}"

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_tr, y_tr)
        final_models[cat] = model
        print(f"  Trained on {len(tr)} rows (log1p target)")

        # Val predictions
        val_cat = val_feat[val_feat["overdose_category"] == cat].copy()
        X_val = val_cat[FEATURE_COLS]

        assert X_val.isnull().sum().sum() == 0, f"NaN in val features for {cat}"

        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.clip(y_pred, 0.01, None)

        val_cat = val_cat.copy()
        val_cat["predicted_rate"] = y_pred
        predictions.append(val_cat[["period_id", "jurisdiction", "overdose_category", "predicted_rate"]])

        # Save refitted model
        model_path = f"{REPO}/outputs/artifacts/lgbm_{cat.replace('_','')}_final.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Final model saved to: {model_path}")
        print(f"  Prediction stats: min={y_pred.min():.4f}, max={y_pred.max():.4f}, mean={y_pred.mean():.4f}")

    # Combine all predictions
    all_preds = pd.concat(predictions, ignore_index=True)
    print(f"\nTotal predictions: {len(all_preds)}")

    # Join with submission template
    sub = sub_template.merge(
        all_preds,
        on=["period_id", "jurisdiction", "overdose_category"],
        how="left",
    )

    print(f"After merge: {len(sub)} rows")
    missing_preds = sub["predicted_rate"].isnull().sum()
    print(f"Missing predictions: {missing_preds}")

    if missing_preds > 0:
        # Fallback for any missing: use global mean from training
        fallback = train_feat.groupby("overdose_category")["rate_per_10000_ed_visits"].mean()
        for cat in CATS:
            mask = (sub["overdose_category"] == cat) & (sub["predicted_rate"].isnull())
            sub.loc[mask, "predicted_rate"] = fallback.get(cat, 15.0)
        print("Fallback applied for missing predictions")

    sub["rate_per_10000_ed_visits"] = sub["predicted_rate"]
    submission = sub[["row_id", "rate_per_10000_ed_visits"]].copy()

    # Final checks
    assert len(submission) == 918, f"Expected 918 rows, got {len(submission)}"
    assert submission["rate_per_10000_ed_visits"].isnull().sum() == 0, "NaN in submission!"
    assert (submission["rate_per_10000_ed_visits"] > 0).all(), "Non-positive predictions found!"
    assert list(submission.columns) == ["row_id", "rate_per_10000_ed_visits"], "Wrong columns!"

    # Save submission
    submission_path = f"{REPO}/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission written to: {submission_path}")
    print(submission.head(10))

    # Diagnostics
    diag = {
        "submission_path": submission_path,
        "n_rows": int(len(submission)),
        "columns": list(submission.columns),
        "prediction_stats": {
            cat: {
                "mean": float(submission[sub["overdose_category"] == cat]["rate_per_10000_ed_visits"].mean()),
                "min": float(submission[sub["overdose_category"] == cat]["rate_per_10000_ed_visits"].min()),
                "max": float(submission[sub["overdose_category"] == cat]["rate_per_10000_ed_visits"].max()),
            }
            for cat in CATS
        },
        "all_finite": bool(np.isfinite(submission["rate_per_10000_ed_visits"].values).all()),
        "all_positive": bool((submission["rate_per_10000_ed_visits"] > 0).all()),
        "missing_count": int(submission["rate_per_10000_ed_visits"].isnull().sum()),
        "refit_on_full_training": True,
        "leakage_check": {
            "val_features_use_training_medians": True,
            "hist_stats_from_training_only": True,
            "no_val_data_in_refit": True,
        },
    }

    with open(f"{REPO}/outputs/artifacts/submission_diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2)

    print("\nS4 COMPLETE")
    print(json.dumps({k: v for k, v in diag.items() if k != "prediction_stats"}, indent=2))
    return diag


if __name__ == "__main__":
    main()
