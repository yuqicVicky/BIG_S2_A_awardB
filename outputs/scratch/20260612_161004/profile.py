import json, re, sys
from datetime import datetime
import numpy as np
import pandas as pd

RUN_ID = "20260612_161004"
spec = json.load(open("outputs/logs/spec_parse.json"))

target_col = spec["target_column"]
row_id_col = spec["row_id_column"]
join_keys = spec["join_keys"]
metric = spec["evaluation_metric"]
train_path = spec["train_file"]
pred_path = spec["prediction_file"]
ss_path = spec.get("sample_submission_file")

# extra covariates file for train (long panel keyed on period_id, jurisdiction)
train_cov_path = "data/train/covariates.csv"

period_id_col = None
for c in spec["detected_structure"]["time_columns"]:
    period_id_col = c
    break

# Parse period_id -> month-end date mapping from Data_Description.md
desc = open("data/Data_Description.md").read()
period_map = {}
for m in re.finditer(r"\|\s*`?(\d{4}-\d{2}-\d{2})`?\s*\|\s*`?([A-Za-z0-9]+)`?\s*\|", desc):
    date_str, pid = m.group(1), m.group(2)
    period_map[pid] = date_str

def load(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

df_target = load(train_path)          # dose_sys_train.csv (long, target)
df_train_cov = load(train_cov_path)    # train covariates
df_pred_cov = load(pred_path)          # val covariates
df_ss = load(ss_path)                  # sample submission

def is_id_like(s, n):
    return s.nunique(dropna=True) >= 0.98 * n and n > 1

def numeric_summary(s):
    s2 = s.dropna()
    if s2.empty:
        return None
    desc = s2.describe(percentiles=[0.01,0.05,0.25,0.5,0.75,0.95,0.99])
    return {
        "min": float(s2.min()), "max": float(s2.max()), "mean": float(s2.mean()),
        "std": float(s2.std()), "p1": float(desc.get("1%", np.nan)),
        "p5": float(desc.get("5%", np.nan)), "p25": float(desc["25%"]),
        "p50": float(desc["50%"]), "p75": float(desc["75%"]),
        "p95": float(desc.get("95%", np.nan)), "p99": float(desc.get("99%", np.nan)),
        "n_unique": int(s.nunique(dropna=True))
    }

def categorical_summary(s, top_n=10):
    vc = s.value_counts(dropna=True)
    return {
        "n_unique": int(s.nunique(dropna=True)),
        "top_values": {str(k): int(v) for k, v in vc.head(top_n).items()}
    }

def try_parse_datetime(s):
    s2 = s.dropna().astype(str)
    if s2.empty:
        return False, 0.0
    sample = s2.sample(min(len(s2), 200), random_state=0)
    try:
        parsed = pd.to_datetime(sample, errors="coerce")
        rate = parsed.notna().mean()
        return rate > 0.95, float(rate)
    except Exception:
        return False, 0.0

def free_text_flag(s, mean_len_thresh=30, word_thresh=5):
    s2 = s.dropna().astype(str)
    if s2.empty:
        return False, 0.0
    lens = s2.str.split().apply(len)
    return float(lens.mean()) > word_thresh, float(lens.mean())

def profile_file(df, path, name):
    n_rows, n_cols = df.shape
    cols = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in cols}
    missing = {}
    for c in cols:
        n_miss = int(df[c].isna().sum())
        missing[c] = {"n_missing": n_miss, "missing_rate": round(n_miss / n_rows, 6)}

    numeric_stats = {}
    categorical_stats = {}
    datetime_like = []
    potential_id = []
    constant_cols = []
    near_constant_cols = []
    high_card_cols = []
    text_like_cols = []
    warnings_list = []

    # cardinality threshold: data-driven -- columns with n_unique/n_rows > 0.5 (but not id-like all-unique)
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            ns = numeric_summary(s)
            if ns:
                numeric_stats[c] = ns
            nun = s.nunique(dropna=True)
            if nun <= 1:
                constant_cols.append(c)
            else:
                vc = s.value_counts(dropna=True, normalize=True)
                if not vc.empty and vc.iloc[0] >= 0.99:
                    near_constant_cols.append(c)
            if is_id_like(s, n_rows):
                potential_id.append(c)
        else:
            cat = categorical_summary(s)
            categorical_stats[c] = cat
            nun = s.nunique(dropna=True)
            if nun <= 1:
                constant_cols.append(c)
            else:
                vc = s.value_counts(dropna=True, normalize=True)
                if not vc.empty and vc.iloc[0] >= 0.99:
                    near_constant_cols.append(c)
            if is_id_like(s, n_rows):
                potential_id.append(c)
            # cardinality check (skip the known opaque period_id which is treated specially)
            if c != period_id_col and nun > 0 and (nun / n_rows) > 0.5 and nun > 20:
                high_card_cols.append(c)
            # free text check
            is_text, mean_words = free_text_flag(s)
            if is_text:
                text_like_cols.append(c)
            # datetime parse check (skip opaque ids / known text)
            if c != period_id_col and not is_text:
                is_dt, rate = try_parse_datetime(s)
                if is_dt:
                    datetime_like.append(c)

    # period_id special handling: opaque id mapping to month-end dates per Data_Description.md
    if period_id_col in cols:
        mapped = df[period_id_col].map(period_map)
        map_rate = mapped.notna().mean()
        if map_rate > 0.95:
            datetime_like.append(f"{period_id_col} (via Data_Description period map, ordinal/categorical)")

    dup_rows = int(df.duplicated().sum())

    # missing rate cutoff: data-driven. Use a cutoff at the gap-based threshold; default 0.05 as MAR-relevant floor.
    miss_rates = {c: missing[c]["missing_rate"] for c in cols}
    nonzero_rates = sorted([r for r in miss_rates.values() if r > 0])
    if nonzero_rates:
        cutoff = max(0.05, nonzero_rates[0])
    else:
        cutoff = 0.05

    for c in cols:
        if missing[c]["missing_rate"] > cutoff:
            warnings_list.append(f"{c}: missing_rate={missing[c]['missing_rate']:.4f} exceeds cutoff {cutoff:.4f}")
        if missing[c]["missing_rate"] >= 0.8:
            warnings_list.append(f"{c}: missing_rate>=0.8 -- consider drop_column")

    for c in constant_cols:
        warnings_list.append(f"{c}: constant column")

    if dup_rows > 0:
        warnings_list.append(f"{dup_rows} duplicate rows detected")

    return {
        "path": path,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "columns": cols,
        "dtypes": dtypes,
        "missing": missing,
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
        "datetime_like_columns": datetime_like,
        "potential_id_columns": potential_id,
        "constant_columns": constant_cols,
        "near_constant_columns": near_constant_cols,
        "high_cardinality_columns": high_card_cols,
        "text_like_columns": text_like_cols,
        "duplicate_rows": dup_rows,
        "warnings": warnings_list,
        "missing_rate_cutoff_used": cutoff
    }

profiles = {}
profiles["train_target"] = profile_file(df_target, train_path, "train_target") if df_target is not None else None
profiles["train_covariates"] = profile_file(df_train_cov, train_cov_path, "train_covariates") if df_train_cov is not None else None
profiles["prediction"] = profile_file(df_pred_cov, pred_path, "prediction") if df_pred_cov is not None else None
profiles["sample_submission"] = profile_file(df_ss, ss_path, "sample_submission") if df_ss is not None else None

# ---- Schema diff between train (combined columns across the two train files) and prediction ----
train_cols = set()
if df_target is not None:
    train_cols |= set(df_target.columns)
if df_train_cov is not None:
    train_cols |= set(df_train_cov.columns)
pred_cols = set(df_pred_cov.columns) if df_pred_cov is not None else set()

in_train_only = sorted(train_cols - pred_cols)
in_pred_only = sorted(pred_cols - train_cols)
in_both = sorted(train_cols & pred_cols)

schema_diff = {
    "in_train_only": in_train_only,
    "in_prediction_only": in_pred_only,
    "in_both": in_both
}

# ---- Split structure ----
# Use detected_structure from spec_parse + verify period coverage
train_periods = set()
if df_target is not None and period_id_col in df_target.columns:
    train_periods |= set(df_target[period_id_col].dropna().unique())
if df_train_cov is not None and period_id_col in df_train_cov.columns:
    train_periods |= set(df_train_cov[period_id_col].dropna().unique())

pred_periods = set(df_pred_cov[period_id_col].dropna().unique()) if df_pred_cov is not None and period_id_col in df_pred_cov.columns else set()

overlap = train_periods & pred_periods
split_type = spec["detected_structure"]["split_pattern"]["split_type"]
if split_type == "chronological" and len(overlap) == 0:
    resolved_type = "chronological_no_overlap"
elif len(overlap) > 0:
    resolved_type = "mixed_overlap"
else:
    resolved_type = "chronological_no_overlap"

def period_to_date(pid):
    return period_map.get(pid)

train_dates = sorted([period_to_date(p) for p in train_periods if period_to_date(p)])
pred_dates = sorted([period_to_date(p) for p in pred_periods if period_to_date(p)])

split_structure = {
    "type": resolved_type,
    "cv_recommendation": spec["detected_structure"]["split_pattern"]["cv_recommendation"],
    "train_day_range": [train_dates[0], train_dates[-1]] if train_dates else None,
    "test_day_range": [pred_dates[0], pred_dates[-1]] if pred_dates else None,
    "sub_target_candidates": spec["detected_structure"]["split_pattern"].get("sub_target_candidates", [])
}

# ---- Target distribution ----
target_dist = {"type": "numeric", "recommend_log_transform": False, "log_transform_reason": "none"}
if df_target is not None and target_col in df_target.columns:
    t = df_target[target_col].dropna()
    n_missing = int(df_target[target_col].isna().sum())
    if pd.api.types.is_numeric_dtype(df_target[target_col]):
        from scipy.stats import skew, kurtosis
        sk = float(skew(t))
        kt = float(kurtosis(t))  # excess kurtosis (fisher=True default)
        log1p_sk = None
        if (t >= 0).all():
            log1p_sk = float(skew(np.log1p(t)))
        desc = t.describe(percentiles=[0.25,0.5,0.75])
        # decision: recommend log if skew > 1 (materially right-skewed) and log1p improves skew, OR metric is log-scale
        metric_lower = metric.lower()
        log_metric = any(k in metric_lower for k in ["rmsle", "rmspe", "log"])
        reason_parts = []
        recommend = False
        if sk > 1.0:
            reason_parts.append(f"raw skewness={sk:.3f} > 1.0 (materially right-skewed)")
            if log1p_sk is not None and abs(log1p_sk) < abs(sk):
                reason_parts.append(f"log1p skewness={log1p_sk:.3f} closer to 0 than raw")
                recommend = True
        if log_metric:
            reason_parts.append(f"evaluation_metric='{metric}' is error-on-log scale")
            recommend = True
        if not reason_parts:
            reason_parts.append(f"raw skewness={sk:.3f} <= 1.0 and metric '{metric}' is not log-scale; no transform needed")
        target_dist.update({
            "type": "numeric",
            "recommend_log_transform": recommend,
            "log_transform_reason": "; ".join(reason_parts),
            "min": float(t.min()), "max": float(t.max()), "mean": float(t.mean()), "std": float(t.std()),
            "p25": float(desc["25%"]), "p50": float(desc["50%"]), "p75": float(desc["75%"]),
            "skewness": sk, "excess_kurtosis": kt, "log1p_skewness": log1p_sk, "n_missing": n_missing
        })
    else:
        vc = df_target[target_col].value_counts(dropna=True)
        target_dist.update({
            "type": "categorical",
            "recommend_log_transform": False,
            "log_transform_reason": "categorical target",
            "class_counts": {str(k): int(v) for k, v in vc.items()},
            "n_missing": n_missing
        })

# ---- Descriptive summary ----
all_numeric_cols = set()
all_cat_cols = set()
all_dt_cols = set()
for p in profiles.values():
    if p is None:
        continue
    all_numeric_cols |= set(p["numeric_stats"].keys())
    all_cat_cols |= set(p["categorical_stats"].keys())
    all_dt_cols |= set(p["datetime_like_columns"])

cols_with_missing = set()
max_missing_rate = 0.0
high_card_all = set()
constant_all = set()
for p in profiles.values():
    if p is None:
        continue
    for c, m in p["missing"].items():
        if m["n_missing"] > 0:
            cols_with_missing.add(c)
        max_missing_rate = max(max_missing_rate, m["missing_rate"])
    high_card_all |= set(p["high_cardinality_columns"])
    constant_all |= set(p["constant_columns"])

descriptive_summary = {
    "train_target_n_rows": profiles["train_target"]["n_rows"] if profiles["train_target"] else None,
    "train_target_n_cols": profiles["train_target"]["n_cols"] if profiles["train_target"] else None,
    "train_covariates_n_rows": profiles["train_covariates"]["n_rows"] if profiles["train_covariates"] else None,
    "train_covariates_n_cols": profiles["train_covariates"]["n_cols"] if profiles["train_covariates"] else None,
    "prediction_n_rows": profiles["prediction"]["n_rows"] if profiles["prediction"] else None,
    "prediction_n_cols": profiles["prediction"]["n_cols"] if profiles["prediction"] else None,
    "target_column": target_col,
    "target_type": target_dist["type"],
    "target_mean": target_dist.get("mean"),
    "target_std": target_dist.get("std"),
    "target_min": target_dist.get("min"),
    "target_max": target_dist.get("max"),
    "target_skewness": target_dist.get("skewness"),
    "recommend_log_transform": target_dist["recommend_log_transform"],
    "log_transform_reason": target_dist["log_transform_reason"],
    "columns_with_missing": sorted(cols_with_missing),
    "max_missing_rate": round(max_missing_rate, 6),
    "n_numeric_columns": len(all_numeric_cols),
    "n_categorical_columns": len(all_cat_cols),
    "n_datetime_like_columns": len(all_dt_cols),
    "duplicate_rows_train_target": profiles["train_target"]["duplicate_rows"] if profiles["train_target"] else None,
    "duplicate_rows_train_covariates": profiles["train_covariates"]["duplicate_rows"] if profiles["train_covariates"] else None,
    "high_cardinality_columns": sorted(high_card_all),
    "constant_columns": sorted(constant_all)
}

# ---- summary_warnings ----
summary_warnings = []
for key, p in profiles.items():
    if p is None:
        summary_warnings.append(f"{key}: file not found / not loaded")
        continue
    for w in p["warnings"]:
        summary_warnings.append(f"[{key}] {w}")

if in_pred_only:
    summary_warnings.append(f"prediction columns absent from train: {in_pred_only}")
if in_train_only:
    summary_warnings.append(f"train-only columns (possible sub-target/leakage candidates): {in_train_only}")

# free-text flag note
state_doh_col = "state_doh_release"
summary_warnings.append(f"{state_doh_col}: free-text press-release column -- flagged for TF-IDF/SVD feature extraction (not raw categorical).")

# period_id ordinal/categorical note
summary_warnings.append(f"{period_id_col}: opaque id maps to month-end dates via Data_Description.md table -- treat as ordinal/categorical period feature (sin/cos month, year, ordinal index), not a raw datetime string feature.")

# split structure note
if resolved_type == "chronological_no_overlap":
    summary_warnings.append("Split structure: chronological, no period overlap between train and prediction periods -- use chronological/group-time CV blocked on period_id (block_mae).")
elif resolved_type == "mixed_overlap":
    summary_warnings.append(f"Split structure: overlap detected between train and prediction periods ({sorted(overlap)}) -- review for within-period cross-sub-period structure.")

summary_warnings.append(f"val/covariates.csv extra period not in sample_submission: {spec['detected_structure']['split_pattern'].get('val_covariates_extra_period')}")

out = {
    "run_id": RUN_ID,
    "profiled_at": datetime.utcnow().isoformat() + "Z",
    "train": profiles["train_target"],
    "train_covariates": profiles["train_covariates"],
    "prediction": profiles["prediction"],
    "sample_submission": profiles["sample_submission"],
    "schema_diff": schema_diff,
    "split_structure": split_structure,
    "target_distribution": target_dist,
    "descriptive_summary": descriptive_summary,
    "summary_warnings": summary_warnings
}

with open("outputs/logs/data_profile.json", "w") as f:
    json.dump(out, f, indent=2, default=str)

print("WROTE data_profile.json")
print("train_target rows/cols:", profiles["train_target"]["n_rows"], profiles["train_target"]["n_cols"])
print("train_covariates rows/cols:", profiles["train_covariates"]["n_rows"], profiles["train_covariates"]["n_cols"])
print("prediction rows/cols:", profiles["prediction"]["n_rows"], profiles["prediction"]["n_cols"])
print("recommend_log_transform:", target_dist["recommend_log_transform"], "|", target_dist["log_transform_reason"])
print("split type:", split_structure["type"])
print("train period date range:", split_structure["train_day_range"])
print("pred period date range:", split_structure["test_day_range"])
print("overlap periods:", overlap)
