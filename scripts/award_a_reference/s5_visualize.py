"""
S5: Visualization and EDA
Generate diagnostic plots: predicted vs actual, feature importance, submission distributions.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo"
RUN_ID = "20260525_001143_cc9cbec2"
CATS = ["all_drugs", "all_opioids", "all_stimulants"]
N_HOLDOUT = 8

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
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}


def main():
    print("=== S5: Visualization ===")
    os.makedirs(f"{REPO}/outputs/artifacts", exist_ok=True)

    train_feat = pd.read_parquet(f"{REPO}/outputs/artifacts/train_features.parquet")
    val_feat = pd.read_parquet(f"{REPO}/outputs/artifacts/val_features.parquet")
    sub = pd.read_csv(f"{REPO}/submission.csv")

    with open(f"{REPO}/outputs/artifacts/baseline_metrics.json") as f:
        baseline = json.load(f)

    holdout_periods = baseline["holdout_periods"]
    train_periods = [p for p in train_feat["period_id"].unique() if p not in holdout_periods]

    # Rebuild holdout df with safe hist stats (same as S3)
    train_df = train_feat[train_feat["period_id"].isin(train_periods)].copy()
    holdout_df = train_feat[train_feat["period_id"].isin(holdout_periods)].copy()

    hist_stats_train_only = (
        train_df.groupby(["jurisdiction", "overdose_category"])["rate_per_10000_ed_visits"]
        .agg(hist_mean_safe="mean", hist_std_safe="std")
        .reset_index()
    )
    hist_stats_train_only["hist_std_safe"] = hist_stats_train_only["hist_std_safe"].fillna(0.0)

    holdout_df = holdout_df.drop(columns=["hist_mean", "hist_std"])
    holdout_df = holdout_df.merge(
        hist_stats_train_only, on=["jurisdiction", "overdose_category"], how="left"
    )
    holdout_df = holdout_df.rename(columns={"hist_mean_safe": "hist_mean", "hist_std_safe": "hist_std"})
    holdout_df["hist_mean"] = holdout_df["hist_mean"].fillna(train_df["rate_per_10000_ed_visits"].mean())
    holdout_df["hist_std"] = holdout_df["hist_std"].fillna(0.0)

    plot_paths = {}

    for cat in CATS:
        # Load holdout model
        model_path = f"{REPO}/outputs/artifacts/lgbm_{cat.replace('_', '')}.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        ho = holdout_df[holdout_df["overdose_category"] == cat].copy()
        X_ho = ho[FEATURE_COLS]
        y_ho_pred = np.expm1(model.predict(X_ho))
        y_ho_pred = np.clip(y_ho_pred, 0.01, None)
        y_ho_true = ho["rate_per_10000_ed_visits"].values

        # --- Plot 1: Predicted vs Actual (holdout split) ---
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y_ho_true, y_ho_pred, alpha=0.4, s=20, color="#2b7bba")
        lim = (0, max(y_ho_true.max(), y_ho_pred.max()) * 1.05)
        ax.plot(lim, lim, "r--", lw=1.2, label="Perfect prediction")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("Actual rate (per 10,000 ED visits)")
        ax.set_ylabel("Predicted rate (per 10,000 ED visits)")
        ax.set_title(f"Predicted vs Actual - {cat}\n(Holdout split, n={len(ho)})")
        ax.legend(fontsize=8)
        mae_val = float(np.mean(np.abs(y_ho_true - y_ho_pred)))
        ax.text(0.05, 0.92, f"MAE = {mae_val:.3f}", transform=ax.transAxes, fontsize=9, color="#333333")
        plt.tight_layout()
        path = f"{REPO}/outputs/artifacts/pred_vs_actual_{cat}.png"
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"Saved: {path}")
        plot_paths[f"pred_vs_actual_{cat}"] = path

        # --- Plot 2: Feature Importance ---
        imp_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(imp_df["feature"], imp_df["importance"], color="#4daf4a")
        ax.set_xlabel("Feature Importance (LightGBM gain)")
        ax.set_title(f"Feature Importance - {cat}\n(Holdout model)")
        plt.tight_layout()
        path = f"{REPO}/outputs/artifacts/feature_importance_{cat}.png"
        plt.savefig(path, dpi=120)
        plt.close()
        print(f"Saved: {path}")
        plot_paths[f"feature_importance_{cat}"] = path

    # --- Plot 3: Submission distribution per category ---
    sub_template = pd.read_csv(f"{REPO}/data/sample_submission.csv")
    sub_full = sub_template.merge(sub, on="row_id")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    colors = {"all_drugs": "#e41a1c", "all_opioids": "#377eb8", "all_stimulants": "#4daf4a"}
    for ax, cat in zip(axes, CATS):
        vals = sub_full[sub_full["overdose_category"] == cat]["rate_per_10000_ed_visits"]
        ax.hist(vals, bins=20, color=colors[cat], alpha=0.75, edgecolor="k", linewidth=0.5)
        ax.set_title(f"{cat}\n(n={len(vals)})", fontsize=10)
        ax.set_xlabel("Predicted rate (per 10,000)")
        ax.set_ylabel("Count")
        ax.axvline(vals.mean(), color="k", linestyle="--", lw=1.2, label=f"Mean={vals.mean():.1f}")
        ax.legend(fontsize=8)
    fig.suptitle("Submission Prediction Distribution by Overdose Category\n(Validation periods)", fontsize=11)
    plt.tight_layout()
    dist_path = f"{REPO}/outputs/artifacts/submission_distribution.png"
    plt.savefig(dist_path, dpi=120)
    plt.close()
    print(f"Saved: {dist_path}")
    plot_paths["submission_distribution"] = dist_path

    # --- Plot 4: Training trend per category (mean rate over time) ---
    train_trend = (
        train_feat.groupby(["temporal_index", "overdose_category"])["rate_per_10000_ed_visits"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for cat, color in colors.items():
        sub_trend = train_trend[train_trend["overdose_category"] == cat]
        ax.plot(sub_trend["temporal_index"], sub_trend["rate_per_10000_ed_visits"],
                label=cat, color=color, linewidth=1.5, alpha=0.85)
    ax.axvline(x=train_feat["temporal_index"].max() - N_HOLDOUT + 0.5,
               color="gray", linestyle="--", lw=1, label="Holdout start")
    ax.set_xlabel("Temporal index (ordered training periods)")
    ax.set_ylabel("Mean rate per 10,000 ED visits")
    ax.set_title("Mean Overdose Rate Over Training Periods by Category")
    ax.legend(fontsize=9)
    plt.tight_layout()
    trend_path = f"{REPO}/outputs/artifacts/training_trend.png"
    plt.savefig(trend_path, dpi=120)
    plt.close()
    print(f"Saved: {trend_path}")
    plot_paths["training_trend"] = trend_path

    print(f"\nTotal plots generated: {len(plot_paths)}")

    summary = {
        "plots": plot_paths,
        "plot_count": len(plot_paths),
        "all_plots_exist": all(os.path.exists(p) for p in plot_paths.values()),
    }

    with open(f"{REPO}/outputs/artifacts/s5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nS5 COMPLETE")
    return summary


if __name__ == "__main__":
    main()
