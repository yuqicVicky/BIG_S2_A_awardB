import json
import numpy as np
import pandas as pd
from scipy import stats

spec = json.load(open("outputs/logs/spec_parse.json"))
target_col = spec["target_column"]
period_id_col = spec["detected_structure"]["time_columns"][0]

df_target = pd.read_csv(spec["train_file"])           # dose_sys_train.csv
df_train_cov = pd.read_csv("data/train/covariates.csv")
df_pred_cov = pd.read_csv(spec["prediction_file"])

# Merge target with covariates on join keys for mechanism/target-signal analysis
join_keys = spec["join_keys"]
df_merged = df_target.merge(df_train_cov, on=join_keys, how="left")

results_columns = {}
mechanism_columns = {}
imputation_columns = {}
structural_flags = {}

n = len(df_merged)

def severity(rate):
    if rate == 0:
        return "none"
    if rate < 0.05:
        return "low"
    if rate < 0.25:
        return "moderate"
    if rate < 0.8:
        return "high"
    return "severe"

numeric_cols = df_merged.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in df_merged.columns if c not in numeric_cols]

for col in df_merged.columns:
    n_miss = int(df_merged[col].isna().sum())
    rate = n_miss / n
    sev = severity(rate)
    results_columns[col] = {"n_missing": n_miss, "missing_rate": round(rate, 6), "severity": sev, "dtype": str(df_merged[col].dtype)}

    if n_miss == 0:
        mechanism_columns[col] = {"mechanism_label": "no missingness", "target_signal": False}
        imputation_columns[col] = {"strategy": "no_imputation_needed", "add_missing_indicator": False}
        continue

    miss_mask = df_merged[col].isna()

    # target signal: correlate missingness indicator with target
    target_signal = False
    if col != target_col and target_col in df_merged.columns:
        t = df_merged[target_col]
        valid = t.notna()
        if valid.sum() > 20 and miss_mask[valid].nunique() > 1:
            try:
                corr = np.corrcoef(miss_mask[valid].astype(int), t[valid])[0, 1]
                if abs(corr) > 0.1:
                    target_signal = True
            except Exception:
                pass

    # group-dependent missingness via jurisdiction
    group_dependent = False
    group_col_used = None
    if "jurisdiction" in df_merged.columns and col != "jurisdiction":
        rates_by_group = df_merged.groupby("jurisdiction")[col].apply(lambda s: s.isna().mean())
        if rates_by_group.max() - rates_by_group.min() >= 0.20:
            group_dependent = True
            group_col_used = "jurisdiction"

    # MAR-like: correlation with other numeric features' missingness/values
    mar_like = False
    max_corr = 0.0
    other_numeric = [c for c in numeric_cols if c != col]
    if other_numeric and n_miss >= 20:
        for oc in other_numeric:
            valid = df_merged[oc].notna()
            if valid.sum() > 20 and miss_mask[valid].nunique() > 1:
                try:
                    c = abs(np.corrcoef(miss_mask[valid].astype(int), df_merged[oc][valid])[0, 1])
                    max_corr = max(max_corr, c)
                    if c > 0.3:
                        mar_like = True
                except Exception:
                    pass

    if n_miss < 20:
        label = "insufficient evidence"
    elif target_signal:
        label = "target-associated missingness"
    elif group_dependent:
        label = "group-dependent missingness"
    elif mar_like:
        label = "MAR-like evidence"
    elif col in cat_cols and df_merged[col].nunique(dropna=True) > 20:
        label = "high-cardinality text/category missingness"
    else:
        label = "MCAR-compatible"

    mechanism_columns[col] = {"mechanism_label": label, "target_signal": target_signal,
                               "max_covariate_correlation": round(float(max_corr), 4),
                               "group_dependent": group_dependent}

    # Collins thresholds
    collins_exceeded = (rate >= 0.25) or target_signal or (max_corr >= 0.4)
    mi_upgrade = collins_exceeded

    # strategy decision
    if rate >= 0.8:
        strategy = "drop_column"
        add_ind = False
    elif col in numeric_cols:
        if collins_exceeded:
            strategy = "model_based_imputation_optional"
            add_ind = True
        elif group_dependent:
            strategy = "groupwise_numeric_median_plus_indicator"
            add_ind = True
        elif rate >= 0.10 or target_signal:
            strategy = "numeric_median_plus_indicator"
            add_ind = True
        else:
            strategy = "numeric_median"
            add_ind = False
    else:
        if rate >= 0.10 or df_merged[col].nunique(dropna=True) > 20:
            strategy = "categorical_missing_token_plus_indicator"
            add_ind = True
        else:
            strategy = "categorical_missing_token"
            add_ind = False

    entry = {"strategy": strategy, "add_missing_indicator": add_ind,
             "mi_upgrade_recommended": mi_upgrade,
             "mar_robustness_note": {"likely_robust_per_collins2001": not collins_exceeded}}
    if strategy == "groupwise_numeric_median_plus_indicator":
        entry["group_col"] = group_col_used
    imputation_columns[col] = entry

# Structural missingness check: state_doh_release empty string vs NaN numeric features
structural_pairs = []
if "state_doh_release" in df_merged.columns:
    empty_text = (df_merged["state_doh_release"].fillna("") == "").mean()
    structural_flags["state_doh_release"] = {
        "empty_string_rate": round(float(empty_text), 6),
        "note": "empty string indicates no qualifying press release for that (state, period) -- not true missingness; treat as a valid category/zero-text state for TF-IDF."
    }

# columns with missing, sorted by rate desc
cols_with_missing = sorted([c for c, v in results_columns.items() if v["n_missing"] > 0],
                            key=lambda c: results_columns[c]["missing_rate"], reverse=True)

missingness_profile = {
    "run_id": spec["run_id"],
    "n_rows": n,
    "columns_with_missing": cols_with_missing,
    "columns": results_columns,
    "summary": {
        "n_columns_with_missing": len(cols_with_missing),
        "max_missing_rate": round(max([v["missing_rate"] for v in results_columns.values()]), 6)
    }
}

mechanism_audit = {
    "run_id": spec["run_id"],
    "columns": {c: mechanism_columns[c] for c in cols_with_missing}
}

structural_audit = {
    "run_id": spec["run_id"],
    "column_flags": structural_flags
}

imputation_plan = {
    "run_id": spec["run_id"],
    "columns": imputation_columns,
    "summary": {
        "columns_needing_missing_indicator": [c for c, e in imputation_columns.items() if e["add_missing_indicator"]],
        "columns_to_drop": [c for c, e in imputation_columns.items() if e["strategy"] == "drop_column"],
        "columns_needing_mi_upgrade": [c for c, e in imputation_columns.items() if e.get("mi_upgrade_recommended")]
    }
}

leakage_safe_check = {
    "run_id": spec["run_id"],
    "global_protocol": {
        "rule": "All imputation statistics (median, mode, group medians, IterativeImputer parameters) are fit on the train split only (per-fold for CV), then applied via transform/fillna to validation/prediction data without re-fitting."
    },
    "per_column_risk": {c: ("low" if e["strategy"] in ("no_imputation_needed",) else "controlled_by_global_protocol")
                         for c, e in imputation_columns.items()}
}

with open("outputs/logs/missingness_profile.json", "w") as f:
    json.dump(missingness_profile, f, indent=2, default=str)
with open("outputs/logs/missingness_mechanism_audit.json", "w") as f:
    json.dump(mechanism_audit, f, indent=2, default=str)
with open("outputs/logs/structural_missingness_audit.json", "w") as f:
    json.dump(structural_audit, f, indent=2, default=str)
with open("outputs/logs/imputation_plan.json", "w") as f:
    json.dump(imputation_plan, f, indent=2, default=str)
with open("outputs/logs/leakage_safe_imputation_check.json", "w") as f:
    json.dump(leakage_safe_check, f, indent=2, default=str)

print("columns_with_missing:", cols_with_missing)
for c in cols_with_missing:
    print(c, "->", imputation_columns[c]["strategy"], "| mechanism:", mechanism_columns[c]["mechanism_label"], "| rate:", results_columns[c]["missing_rate"])
