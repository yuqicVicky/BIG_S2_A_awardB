"""
Stage 9: Report Generation
Write the analysis report as Markdown and convert to PDF.
"""
import os
import sys
import json

REPO = "/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo"
RUN_ID = "20260525_001143_cc9cbec2"

sys.path.insert(0, f"{REPO}/scripts")
from md_to_pdf import convert as md_to_pdf


REPORT_TEXT = """# STAI-X Award B: Overdose Rate Prediction Report

**Run ID:** 20260525_001143_cc9cbec2
**Report Date:** 2026-05-25
**Task Type:** Regression
**Author:** STAI-X General Data Analysis AI Agent

---

## 1. Executive Summary

This report documents the end-to-end regression analysis conducted to predict monthly emergency-department overdose rates (`rate_per_10000_ed_visits`) for 918 rows spanning 6 validation period-IDs, 51 U.S. jurisdictions, and three overdose categories: `all_drugs`, `all_opioids`, and `all_stimulants`.

Three separate LightGBM regressors -- one per overdose category -- were trained on 77 period-ID x 51 jurisdiction panel data from the DOSE-SYS training set (31,416 total rows; 11,781 rows restricted to the three target categories). A log1p transformation was applied to the target before fitting. Internal validation used a time-based holdout of the 8 most recent training period-IDs (February 2024 through December 2024, 1,224 holdout rows). All three candidate models outperformed the jurisdiction-category historical mean baseline on holdout mean absolute error (MAE). The final submission file (`submission.csv`) contains 918 rows with all predictions positive and finite.

---

## 2. Dataset Overview

| Dataset | Rows | Columns | Notes |
|---------|------|---------|-------|
| Training target (dose_sys_train) | 31,416 | 4 | 77 periods x 51 jurisdictions x 8 categories |
| Training target (3 categories) | 11,781 | 4 | all_drugs, all_opioids, all_stimulants |
| Training covariates | 3,927 | 12 | 77 periods x 51 jurisdictions |
| Validation covariates | 357 | 12 | 7 periods x 51 jurisdictions |
| Submission template | 918 | 5 | 6 periods x 51 jurisdictions x 3 categories |

**Covariate features:** `unemployment_rate`, `labor_force`, `temp_avg_f`, `precip_in`, `gtrends_overdose`, `gtrends_fentanyl`, `gtrends_naloxone`, `gtrends_opioid`, `gtrends_methamphetamine`, `state_doh_release` (text, excluded from modeling).

**Missing values in training covariates:** `temp_avg_f` and `precip_in` missing for 231 rows (~5.9%); `state_doh_release` text missing for 1,162 rows (~29.6%). Median imputation (fitted on training data only) was applied for numeric columns.

**Target distribution by category (training, 3,927 rows each):**

| Category | Mean | Std | Min | Median | Max | Skewness |
|----------|------|-----|-----|--------|-----|----------|
| all_drugs | 22.82 | 9.91 | 6.37 | 20.24 | 79.99 | 1.65 |
| all_opioids | 10.30 | 4.47 | 2.96 | 9.26 | 41.00 | 1.81 |
| all_stimulants | 5.22 | 2.38 | 1.20 | 4.67 | 20.16 | 1.67 |

All three distributions are right-skewed, motivating the log1p target transformation.

---

## 3. Feature Engineering

Features were constructed as follows. All transformations were fitted on the training split only; the same fitted encoders were applied to the validation set.

**Numeric covariates (9):** Used as-is after median imputation.

**Engineered features:**
- `log_labor_force`: natural log of `labor_force + 1` (reduces scale heterogeneity)
- `temporal_index`: monotone integer index assigned by calendar ordering of known period-IDs; unknown period-IDs receive lower indices (0-7); validation periods receive indices 77-83 (strictly greater than any training index)
- `sin_month`, `cos_month`, `month`: cyclical month encoding for seasonality; validation periods with unknown calendar dates received training-mean imputation (sin_mean = 0.0053, cos_mean = 0.0104)
- `jurisdiction_code`: integer encoding of U.S. state/territory (51 values)
- `hist_mean`, `hist_std`: historical mean and standard deviation of the target per (jurisdiction, overdose_category) pair, computed from training data only

**Leakage prevention:**
- Median imputers for numeric covariates were fitted on training covariates only and applied to validation
- Historical statistics (hist_mean, hist_std) were computed from the non-holdout training rows when evaluating on holdout; from all training rows when refitting for submission
- Validation period temporal indices are strictly greater than all training period indices
- No validation observations were used in any feature engineering step

---

## 4. Train/Holdout Split

- **Training split:** 69 period-IDs (temporal indices 0-68), 10,557 rows (per 3-category subset)
- **Holdout split:** 8 most recent period-IDs (2024-02 through 2024-12), 1,224 rows (408 per category)
- **Validation set (submission):** 6 period-IDs not present in training, 918 rows
- **Random seed:** 42
- **Holdout period-IDs:** i9aSkhZb, 9FQthr9A, DpR0556d, 9Dp3l3qq, 7cCeqHbf, lfTz14iT, k4mmkR0U, NWU8bRHI
- **Corresponding calendar months:** Feb 2024, Mar 2024, Apr 2024, Jul 2024, Aug 2024, Sep 2024, Nov 2024, Dec 2024

No cross-validation was used; time-based holdout was the sole internal validation strategy to respect the temporal ordering of the data.

---

## 5. Baseline Model

The baseline assigns each (jurisdiction, overdose_category) pair the mean rate observed across all non-holdout training periods. For any combination not present in training, the global non-holdout training mean was used as a fallback.

**Baseline metrics (holdout split, n=408 per category):**

| Category | MAE (holdout) | RMSE (holdout) |
|----------|--------------|----------------|
| all_drugs | 3.1907 | 4.3885 |
| all_opioids | 1.4601 | 2.0831 |
| all_stimulants | 1.1224 | 1.4513 |

---

## 6. LightGBM Model

Three separate LightGBM regressors were trained, one per overdose category. Target variable was log1p-transformed before fitting; predictions were back-transformed via expm1 and clipped to a minimum of 0.01.

**Hyperparameters (identical across all three models):**

| Parameter | Value |
|-----------|-------|
| n_estimators | 500 |
| learning_rate | 0.05 |
| num_leaves | 31 |
| min_child_samples | 10 |
| subsample | 0.80 |
| colsample_bytree | 0.80 |
| random_state | 42 |

**Training metrics (train split, log1p target back-transformed):**

| Category | MAE (train) | RMSE (train) | R2 (train) | n rows |
|----------|------------|-------------|-----------|--------|
| all_drugs | 1.168 | 1.570 | 0.975 | 3,519 |
| all_opioids | 0.538 | 0.717 | 0.974 | 3,519 |
| all_stimulants | 0.269 | 0.368 | 0.976 | 3,519 |

**Holdout metrics vs. baseline (holdout split, n=408 per category):**

| Category | LightGBM MAE | Baseline MAE | LightGBM RMSE | LightGBM R2 | Beats Baseline |
|----------|-------------|-------------|--------------|------------|----------------|
| all_drugs | 3.129 | 3.191 | 4.188 | 0.823 | Yes |
| all_opioids | 1.429 | 1.460 | 1.982 | 0.814 | Yes |
| all_stimulants | 0.806 | 1.122 | 1.099 | 0.800 | Yes |

All three models outperform the historical-mean baseline on holdout MAE. The gap is most pronounced for `all_stimulants` (0.806 vs 1.122, a 28.2% reduction) and smallest for `all_drugs` (3.129 vs 3.191, a 1.9% reduction).

The relatively high train R2 (~0.975) compared to holdout R2 (~0.81) indicates some degree of overfitting, typical of tree models on panel data where jurisdiction-level fixed effects dominate. The holdout corresponds to the most recent 8 months of training data, which may represent distributional shift not captured in training.

---

## 7. Feature Importance

Top 5 features by LightGBM split importance (holdout model):

**all_drugs:** temp_avg_f (1659), precip_in (1657), gtrends_overdose (1218), gtrends_naloxone (1183), gtrends_methamphetamine (1161)

**all_opioids:** temp_avg_f (1676), precip_in (1650), gtrends_methamphetamine (1227), unemployment_rate (1185), gtrends_overdose (1150)

**all_stimulants:** precip_in (1599), temp_avg_f (1591), gtrends_opioid (1395), gtrends_overdose (1198), unemployment_rate (1143)

Weather variables (`temp_avg_f`, `precip_in`) consistently rank among the top features across all categories. Google Trends search indices for overdose-related terms show substantial importance. Unemployment rate is also a consistently important predictor. The `hist_mean` and `hist_std` features (jurisdiction-category historical means and standard deviations from training data) provide strong baseline anchoring.

Note: feature importances from tree models reflect the frequency and improvement from splits on each variable, not causal pathways. The association between weather conditions and overdose rates observed here is consistent with prior public health literature on seasonal patterns in substance-use-related emergency department visits, but no causal mechanism is established by this analysis.

---

## 8. Submission Predictions

The final submission was generated by refitting all three LightGBM models on the full training dataset (all 77 training period-IDs, 3,927 rows per category) and predicting for the 918 submission rows.

**Submission prediction statistics (validation periods, 306 rows per category):**

| Category | Mean predicted rate | Min | Max |
|----------|--------------------|----|-----|
| all_drugs | 22.52 | 12.78 | 52.20 |
| all_opioids | 10.18 | 5.73 | 21.88 |
| all_stimulants | 6.24 | 3.55 | 14.53 |

Submission predictions are within the plausible range observed in training (all_drugs training range: 6.37 to 79.99; all_opioids: 2.96 to 41.00; all_stimulants: 1.20 to 20.16). All 918 predictions are positive and finite.

---

## 9. Data Integrity and Leakage Audit

| Check | Status |
|-------|--------|
| Validation period-IDs not in training data | PASS |
| Temporal index: val > max(train) | PASS |
| Median imputers fitted on training data only | PASS |
| Historical stats (hist_mean, hist_std) computed from training only | PASS |
| Holdout hist stats recomputed from non-holdout rows only | PASS |
| No NaN in training or validation feature matrices | PASS |
| log1p applied to target before fit; expm1 applied after prediction | PASS |
| All submission predictions positive and finite | PASS |

---

## 10. Inspector Verdicts

| Step | Description | Verdict | Notes |
|------|-------------|---------|-------|
| S1 | Feature Engineering | PASS (WARN) | sin/cos month NaN in initial val features; fixed by filling with training means |
| S2 | Baseline Model | PASS | Baseline from non-holdout data; split labeled correctly |
| S3 | LightGBM Training | PASS | All models beat baseline; split labels verified; leakage checks clear |
| S4 | Submission Generation | PASS | 918 rows, correct columns, all positive/finite, row_ids match template |
| S5 | Visualization | PASS (WARN) | Minor script bug (merge column conflict) fixed inline; all 8 plots generated |

---

## 11. Artifacts

| Artifact | Path |
|----------|------|
| Training features | outputs/artifacts/train_features.parquet |
| Validation features | outputs/artifacts/val_features.parquet |
| Baseline metrics | outputs/artifacts/baseline_metrics.json |
| Holdout metrics | outputs/artifacts/holdout_metrics.json |
| LightGBM (all_drugs, holdout model) | outputs/artifacts/lgbm_alldrugs.pkl |
| LightGBM (all_opioids, holdout model) | outputs/artifacts/lgbm_allopioids.pkl |
| LightGBM (all_stimulants, holdout model) | outputs/artifacts/lgbm_allstimulants.pkl |
| LightGBM (all_drugs, final) | outputs/artifacts/lgbm_alldrugs_final.pkl |
| LightGBM (all_opioids, final) | outputs/artifacts/lgbm_allopioids_final.pkl |
| LightGBM (all_stimulants, final) | outputs/artifacts/lgbm_allstimulants_final.pkl |
| Submission | submission.csv (repo root) |
| Predicted vs Actual plots | outputs/artifacts/pred_vs_actual_*.png |
| Feature importance plots | outputs/artifacts/feature_importance_*.png |
| Submission distribution | outputs/artifacts/submission_distribution.png |
| Training trend | outputs/artifacts/training_trend.png |

---

## 12. Limitations and Uncertainty

- **Holdout improvement margins for all_drugs are modest** (1.9% MAE improvement over baseline), indicating the jurisdiction-category historical mean is already a strong predictor for that category. Uncertainty is higher in the holdout-to-validation generalization.
- **Unknown period-IDs in training** (8 of 77) could not be assigned calendar dates and received lower temporal indices; their month-seasonality features were imputed with training means.
- **Validation period-IDs are entirely out-of-time**: the model has no overlap with validation periods. Prediction quality on these periods cannot be assessed without ground-truth labels.
- **Weather data (temp_avg_f, precip_in)** was missing for one jurisdiction in each period (likely DC, which lacks state-level weather data). Training median imputation was applied; this may marginally reduce accuracy for that jurisdiction.
- **No causal inferences** are drawn from any association identified in this analysis. All findings are observational and reflect statistical associations in the training data only.

---

*Report generated automatically by the STAI-X General Data Analysis AI Agent pipeline.*
*Run ID: 20260525_001143_cc9cbec2*
"""


def main():
    print("=== Stage 9: Report Generation ===")

    out_dir = f"{REPO}/outputs/reports"
    os.makedirs(out_dir, exist_ok=True)

    md_path = f"{out_dir}/{RUN_ID}_report.md"
    pdf_path = f"{out_dir}/{RUN_ID}_report.pdf"

    # Write Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(REPORT_TEXT)
    print(f"Markdown written: {md_path} ({os.path.getsize(md_path)//1024} KB)")

    # Convert to PDF
    md_to_pdf(md_path, pdf_path)
    assert os.path.exists(pdf_path), f"PDF not created at {pdf_path}"
    print(f"Report PDF: {pdf_path} ({os.path.getsize(pdf_path)//1024} KB)")

    return md_path, pdf_path


if __name__ == "__main__":
    main()
