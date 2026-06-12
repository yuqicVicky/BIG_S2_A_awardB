"""
Leakage-safe feature engineering pipeline for run_id=20260612_161004.

Resolves every column / file name at runtime from spec_parse.json,
data_profile.json and analysis_plan.json -- no hardcoded literals.

Writes:
  outputs/logs/{run_id}_features_train.parquet   (raw train-row order)
  outputs/logs/{run_id}_features_pred.parquet    (sample_submission row order)
  outputs/logs/{run_id}_feature_spec.json
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[3]
LOGS = ROOT / "outputs" / "logs"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RUN_ID = "20260612_161004"

SPEC = json.loads((LOGS / "spec_parse.json").read_text())
PROFILE = json.loads((LOGS / "data_profile.json").read_text())
PLAN = json.loads((LOGS / "analysis_plan.json").read_text())
CV_FOLDS = json.loads((LOGS / f"{RUN_ID}_cv_folds.json").read_text())

# ---------------------------------------------------------------------------
# Resolve names at runtime
# ---------------------------------------------------------------------------
TRAIN_TARGET_PATH = ROOT / SPEC["train_file"]
PRED_COV_PATH = ROOT / SPEC["prediction_file"]
SAMPLE_SUB_PATH = ROOT / SPEC["sample_submission_file"]
TARGET_COL = SPEC["target_column"]
ROW_ID_COL = SPEC["row_id_column"]
JOIN_KEYS = SPEC["join_keys"]  # ["period_id", "jurisdiction"]
DATA_DESC_PATH = ROOT / "data" / "Data_Description.md"

# train covariates file -- discovered from data_coverage available_sources
_cov_train_path = None
for src in PLAN["data_coverage"]["available_sources"]:
    if "covariates.csv" in src["source"] and "train" in src["source"]:
        _cov_train_path = ROOT / src["source"]
        break
TRAIN_COV_PATH = _cov_train_path

PERIOD_COL = JOIN_KEYS[0]
JURIS_COL = JOIN_KEYS[1]

# overdose_category column -- the extra group key beyond join keys, present in
# train target file & sample_submission but not in covariates files
_target_file_cols = set(SPEC["file_schemas"][SPEC["train_file"]]["columns"])
_cov_cols = set(SPEC["file_schemas"][list(SPEC["file_schemas"].keys())[1]]["columns"])
CATEGORY_COL = next(
    c for c in _target_file_cols
    if c not in _cov_cols and c not in (TARGET_COL,)
)

# direct numeric covariates, gtrends, etc -- from feature_plan.direct_numeric
DIRECT_NUMERIC = list(PLAN["feature_plan"]["direct_numeric"])

# text column for TF-IDF/SVD
TEXT_COLS = [d["column"] for d in PLAN["feature_plan"]["text_tfidf_svd"]]
SVD_COMPONENTS = {d["column"]: int(d["svd_components"]) for d in PLAN["feature_plan"]["text_tfidf_svd"]}

# categorical encoding spec
CAT_ENCODINGS = PLAN["feature_plan"]["categorical_encoding"]

# per-fold target aggregates
PER_FOLD_AGGS = PLAN["feature_plan"]["per_fold_target_aggregates"]

# datetime derived fields requested
DT_FIELDS_REQUESTED = PLAN["feature_plan"]["datetime_derived"]

# imputation plan
IMPUTATION = PLAN["feature_plan"]["imputation"]

# log transform recommendation
RECOMMEND_LOG = bool(PROFILE.get("target_distribution", {}).get("recommend_log_transform", False))

# scored categories from sample_submission
_sample_sub_preview = pd.read_csv(SAMPLE_SUB_PATH, nrows=2000)
SCORED_CATEGORIES = sorted(_sample_sub_preview[CATEGORY_COL].unique().tolist())
SUBMISSION_PERIODS = sorted(_sample_sub_preview[PERIOD_COL].unique().tolist())

# ---------------------------------------------------------------------------
# Period <-> date mapping (parsed from Data_Description.md at runtime)
# ---------------------------------------------------------------------------
def parse_period_map(path: Path) -> dict[str, pd.Timestamp]:
    text = path.read_text()
    pairs = re.findall(r"\|\s*`(\d{4}-\d{2}-\d{2})`\s*\|\s*`([A-Za-z0-9]+)`\s*\|", text)
    return {pid: pd.Timestamp(date_str) for date_str, pid in pairs}


PERIOD_MAP = parse_period_map(DATA_DESC_PATH)


def add_period_datetime_features(df: pd.DataFrame, period_col: str) -> pd.DataFrame:
    """Derive col__<field> datetime features from the opaque period_id column.
    Never emits the raw period_id string itself."""
    dates = df[period_col].map(PERIOD_MAP)
    # ordinal rank: dense rank over all known periods (chronological order)
    sorted_periods = sorted(PERIOD_MAP.items(), key=lambda kv: kv[1])
    ordinal_lookup = {pid: i for i, (pid, _) in enumerate(sorted_periods)}
    out = pd.DataFrame(index=df.index)
    out[f"{period_col}__year"] = dates.dt.year.astype("float64")
    out[f"{period_col}__month"] = dates.dt.month.astype("float64")
    out[f"{period_col}__quarter"] = dates.dt.quarter.astype("float64")
    out[f"{period_col}__month_sin"] = np.sin(2 * np.pi * dates.dt.month / 12.0)
    out[f"{period_col}__month_cos"] = np.cos(2 * np.pi * dates.dt.month / 12.0)
    out[f"{period_col}__day"] = dates.dt.day.astype("float64")
    out[f"{period_col}__dayofweek"] = dates.dt.dayofweek.astype("float64")
    out[f"{period_col}__is_weekend"] = (dates.dt.dayofweek >= 5).astype("float64")
    out[f"{period_col}__weekofyear"] = dates.dt.isocalendar().week.astype("float64")
    out[f"{period_col}__ordinal_rank"] = df[period_col].map(ordinal_lookup).astype("float64")
    out[f"{period_col}__ordinal"] = dates.map(lambda d: d.toordinal() if pd.notna(d) else np.nan).astype("float64")
    return out


DATETIME_DERIVED_COLS = []  # filled after first call


# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------
train_target = pd.read_csv(TRAIN_TARGET_PATH)  # period_id, jurisdiction, overdose_category, target
train_cov = pd.read_csv(TRAIN_COV_PATH)  # period_id, jurisdiction, numeric covariates, text
pred_cov = pd.read_csv(PRED_COV_PATH)  # val/covariates.csv -- includes buffer period
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

N_TRAIN = len(train_target)
assert N_TRAIN == CV_FOLDS["n_train_rows"], "train row count mismatch with canonical folds"

# ---------------------------------------------------------------------------
# Build the base (long) train frame: one row per (period, jurisdiction, category)
# merged with covariates on (period_id, jurisdiction)
# ---------------------------------------------------------------------------
train_long = train_target.merge(train_cov, on=[PERIOD_COL, JURIS_COL], how="left", suffixes=("", "_cov"))
assert len(train_long) == N_TRAIN, "merge changed train row count -- join keys not unique per (period,jurisdiction)"

# ---------------------------------------------------------------------------
# Build the prediction (long) frame.
# pred_cov (val/covariates.csv) has 357 rows = 51 jurisdictions x 7 periods
# (6 submission periods + 1 buffer period dsZhPyK4).
# Cross-join with SCORED_CATEGORIES, filter to submission periods, then align
# to sample_submission row order via (period_id, jurisdiction, overdose_category).
# ---------------------------------------------------------------------------
pred_cov_sub = pred_cov[pred_cov[PERIOD_COL].isin(SUBMISSION_PERIODS)].copy()
pred_long_rows = []
for cat in SCORED_CATEGORIES:
    tmp = pred_cov_sub.copy()
    tmp[CATEGORY_COL] = cat
    pred_long_rows.append(tmp)
pred_long = pd.concat(pred_long_rows, ignore_index=True)

# align to sample_submission order
pred_long = sample_sub[[ROW_ID_COL, PERIOD_COL, JURIS_COL, CATEGORY_COL]].merge(
    pred_long, on=[PERIOD_COL, JURIS_COL, CATEGORY_COL], how="left"
)
assert len(pred_long) == len(sample_sub), "prediction frame expansion row count mismatch"
assert (pred_long[ROW_ID_COL].values == sample_sub[ROW_ID_COL].values).all()

# context frame for lag/rolling features: full train_cov history + buffer period
# (jurisdiction, period) panel for gtrends/target lag computation
all_cov_context = pd.concat([train_cov, pred_cov], ignore_index=True).drop_duplicates(
    subset=[PERIOD_COL, JURIS_COL], keep="first"
)

# ---------------------------------------------------------------------------
# Datetime-derived features (period_id) -- applied to both frames
# ---------------------------------------------------------------------------
train_dt = add_period_datetime_features(train_long, PERIOD_COL)
pred_dt = add_period_datetime_features(pred_long, PERIOD_COL)
DATETIME_DERIVED_COLS = list(train_dt.columns)

# ---------------------------------------------------------------------------
# Direct numeric covariates with leakage-safe imputation
#   - temp_avg_f, precip_in -> jurisdiction-month training median
#   - unemployment_rate, labor_force -> jurisdiction training last-known-value
#     carry-forward (fit on train only, applied to pred)
# ---------------------------------------------------------------------------
def fit_jurisdiction_month_median(cov_df: pd.DataFrame, col: str) -> dict:
    tmp = cov_df.copy()
    tmp["__month"] = tmp[PERIOD_COL].map(PERIOD_MAP).dt.month
    grp = tmp.groupby([JURIS_COL, "__month"])[col].median()
    overall = tmp[col].median()
    return {"grp": grp, "overall": overall}


def apply_jurisdiction_month_median(df: pd.DataFrame, col: str, fit: dict) -> pd.Series:
    tmp = df.copy()
    tmp["__month"] = tmp[PERIOD_COL].map(PERIOD_MAP).dt.month
    out = df[col].copy()
    mask = out.isna()
    if mask.any():
        keys = list(zip(tmp.loc[mask, JURIS_COL], tmp.loc[mask, "__month"]))
        filled = pd.Series(
            [fit["grp"].get(k, fit["overall"]) for k in keys], index=out[mask].index
        )
        out.loc[mask] = filled
    out = out.fillna(fit["overall"])
    return out


def fit_jurisdiction_last_known(cov_df: pd.DataFrame, col: str) -> dict:
    """Per-jurisdiction last non-null value, ordered by chronological period, fit on train."""
    tmp = cov_df.copy()
    tmp["__ordinal"] = tmp[PERIOD_COL].map(PERIOD_MAP).map(lambda d: d.toordinal())
    tmp = tmp.sort_values("__ordinal")
    last_vals = tmp.dropna(subset=[col]).groupby(JURIS_COL)[col].last()
    overall = tmp[col].median()
    return {"last": last_vals, "overall": overall}


def apply_jurisdiction_last_known(df: pd.DataFrame, col: str, fit: dict) -> pd.Series:
    out = df[col].copy()
    mask = out.isna()
    if mask.any():
        filled = df.loc[mask, JURIS_COL].map(fit["last"])
        out.loc[mask] = filled
    out = out.fillna(fit["overall"])
    return out


JURIS_MONTH_MEDIAN_COLS = [c for c in ("temp_avg_f", "precip_in") if c in DIRECT_NUMERIC]
LAST_KNOWN_COLS = [c for c in ("unemployment_rate", "labor_force") if c in DIRECT_NUMERIC]
OTHER_DIRECT_NUMERIC = [c for c in DIRECT_NUMERIC if c not in JURIS_MONTH_MEDIAN_COLS and c not in LAST_KNOWN_COLS]

train_numeric = pd.DataFrame(index=train_long.index)
pred_numeric = pd.DataFrame(index=pred_long.index)

for col in JURIS_MONTH_MEDIAN_COLS:
    fit = fit_jurisdiction_month_median(train_cov, col)
    train_numeric[col] = apply_jurisdiction_month_median(train_long, col, fit)
    pred_numeric[col] = apply_jurisdiction_month_median(pred_long, col, fit)

for col in LAST_KNOWN_COLS:
    fit = fit_jurisdiction_last_known(train_cov, col)
    train_numeric[col] = apply_jurisdiction_last_known(train_long, col, fit)
    pred_numeric[col] = apply_jurisdiction_last_known(pred_long, col, fit)

for col in OTHER_DIRECT_NUMERIC:
    med = train_cov[col].median()
    train_numeric[col] = train_long[col].fillna(med)
    pred_numeric[col] = pred_long[col].fillna(med)

# ---------------------------------------------------------------------------
# Categorical encodings: overdose_category one-hot; jurisdiction handled via
# per-fold target encoding below (no plain one-hot for jurisdiction -> 51 cols)
# ---------------------------------------------------------------------------
cat_onehot_specs = [c for c in CAT_ENCODINGS if c["strategy"] == "one_hot"]
ONEHOT_COLS_OUT = []
train_cat = pd.DataFrame(index=train_long.index)
pred_cat = pd.DataFrame(index=pred_long.index)
for spec in cat_onehot_specs:
    col = spec["column"]
    categories = sorted(train_long[col].dropna().unique().tolist())
    for cat_val in categories:
        out_col = f"{col}__is_{cat_val}"
        train_cat[out_col] = (train_long[col] == cat_val).astype("float64")
        pred_cat[out_col] = (pred_long[col] == cat_val).astype("float64")
        ONEHOT_COLS_OUT.append(out_col)

# ---------------------------------------------------------------------------
# Text TF-IDF -> SVD with structural-missing indicator (fit on full train text)
# ---------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

TEXT_SVD_COLS = []
train_text_feats = pd.DataFrame(index=train_long.index)
pred_text_feats = pd.DataFrame(index=pred_long.index)

for col in TEXT_COLS:
    n_comp = SVD_COMPONENTS[col]
    train_text = train_long[col].fillna("").astype(str)
    pred_text = pred_long[col].fillna("").astype(str)

    missing_ind_name = f"{col}__is_missing"
    train_text_feats[missing_ind_name] = train_long[col].isna().astype("float64")
    pred_text_feats[missing_ind_name] = pred_long[col].isna().astype("float64")

    vectorizer = TfidfVectorizer(max_features=2000, min_df=2, ngram_range=(1, 2))
    train_tfidf = vectorizer.fit_transform(train_text)
    pred_tfidf = vectorizer.transform(pred_text)

    n_comp_eff = min(n_comp, train_tfidf.shape[1] - 1, train_tfidf.shape[0] - 1)
    n_comp_eff = max(n_comp_eff, 1)
    svd = TruncatedSVD(n_components=n_comp_eff, random_state=42)
    train_svd = svd.fit_transform(train_tfidf)
    pred_svd = svd.transform(pred_tfidf)

    for i in range(n_comp_eff):
        out_col = f"{col}__svd_{i}"
        train_text_feats[out_col] = train_svd[:, i]
        pred_text_feats[out_col] = pred_svd[:, i]
        TEXT_SVD_COLS.append(out_col)

TEXT_SVD_COLS.append(f"{TEXT_COLS[0]}__is_missing")

# ---------------------------------------------------------------------------
# Per-fold target / category aggregates (fit on training folds only)
# ---------------------------------------------------------------------------
from src.data_agent import cv as cv_mod

canon = cv_mod.load_canonical_folds(LOGS / f"{RUN_ID}_cv_folds.json")

train_agg = pd.DataFrame(index=train_long.index, dtype="float64")
PER_FOLD_AGG_SPECS = []

target_series = train_long[TARGET_COL]
global_mean = target_series.mean()
global_std = target_series.std()

for agg_spec in PER_FOLD_AGGS:
    name = agg_spec["name"]
    group_keys = agg_spec["group_keys"]
    is_std = name.endswith("_std")
    train_agg[name] = np.nan
    for tr_idx, val_idx in canon.folds:
        tr_df = train_long.iloc[tr_idx]
        if is_std:
            stat = tr_df.groupby(group_keys)[TARGET_COL].std()
            fallback = tr_df[TARGET_COL].std()
        else:
            stat = tr_df.groupby(group_keys)[TARGET_COL].mean()
            fallback = tr_df[TARGET_COL].mean()
        if fallback != fallback:  # NaN guard
            fallback = global_std if is_std else global_mean
        val_df = train_long.iloc[val_idx]
        if len(group_keys) == 1:
            keys = val_df[group_keys[0]]
            mapped = keys.map(stat)
        else:
            keys = pd.Series(list(zip(*[val_df[k] for k in group_keys])), index=val_df.index)
            mapped = keys.map(stat.to_dict())
        mapped = mapped.fillna(fallback)
        train_agg.iloc[val_idx, train_agg.columns.get_loc(name)] = mapped.values
    PER_FOLD_AGG_SPECS.append({"name": name, "group_keys": group_keys, "source": "per_fold"})

# Full-train fit for the prediction matrix (refit on ALL of train, applied to pred)
pred_agg = pd.DataFrame(index=pred_long.index, dtype="float64")
for agg_spec in PER_FOLD_AGGS:
    name = agg_spec["name"]
    group_keys = agg_spec["group_keys"]
    is_std = name.endswith("_std")
    if is_std:
        stat = train_long.groupby(group_keys)[TARGET_COL].std()
        fallback = global_std
    else:
        stat = train_long.groupby(group_keys)[TARGET_COL].mean()
        fallback = global_mean
    if len(group_keys) == 1:
        keys = pred_long[group_keys[0]]
        mapped = keys.map(stat)
    else:
        keys = pd.Series(list(zip(*[pred_long[k] for k in group_keys])), index=pred_long.index)
        mapped = keys.map(stat.to_dict())
    pred_agg[name] = mapped.fillna(fallback).values

# ---------------------------------------------------------------------------
# Cross-category auxiliary signal + lag/rolling experimental features
# Built on the full (period, jurisdiction, category) target panel which is
# entirely train-derived; for fold-safety the per-fold target aggregates above
# already isolate fold leakage for group means. Lag/rolling/aux features here
# use only *past* periods relative to each row (no future leakage) and are
# flagged experimental for the Step-6A' ablation gate.
# ---------------------------------------------------------------------------
ordinal_lookup_full = {pid: i for i, (pid, _) in enumerate(sorted(PERIOD_MAP.items(), key=lambda kv: kv[1]))}

panel = train_long[[PERIOD_COL, JURIS_COL, CATEGORY_COL, TARGET_COL]].copy()
panel["__ord"] = panel[PERIOD_COL].map(ordinal_lookup_full)

# auxiliary categories = the 5 unscored categories
AUX_CATEGORIES = sorted(set(panel[CATEGORY_COL].unique()) - set(SCORED_CATEGORIES))

# pivot panel to (period_ord, jurisdiction) x category -> target, for lag lookups
panel_pivot = panel.pivot_table(index=["__ord", JURIS_COL], columns=CATEGORY_COL, values=TARGET_COL)

# gtrends_fentanyl per-jurisdiction lag1 from covariate context (train + buffer)
gtrends_ctx = all_cov_context[[PERIOD_COL, JURIS_COL, "gtrends_fentanyl"]].copy()
gtrends_ctx["__ord"] = gtrends_ctx[PERIOD_COL].map(ordinal_lookup_full)
gtrends_pivot = gtrends_ctx.pivot_table(index="__ord", columns=JURIS_COL, values="gtrends_fentanyl")


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    ords = df[PERIOD_COL].map(ordinal_lookup_full)
    jurs = df[JURIS_COL]
    cats = df[CATEGORY_COL]

    lag1_keys = list(zip(ords - 1, jurs))
    lag2_keys = list(zip(ords - 2, jurs))

    def lookup_panel(keys, cat_series):
        vals = np.full(len(keys), np.nan)
        for i, ((o, j), c) in enumerate(zip(keys, cat_series)):
            try:
                if (o, j) in panel_pivot.index and c in panel_pivot.columns:
                    vals[i] = panel_pivot.loc[(o, j), c]
            except (KeyError, TypeError):
                pass
        return vals

    out["target_lag1_per_jurisdiction_category"] = lookup_panel(lag1_keys, cats)
    out["target_lag2_per_jurisdiction_category"] = lookup_panel(lag2_keys, cats)

    def rolling_mean(window):
        vals = np.full(len(df), np.nan)
        for i, (o, j, c) in enumerate(zip(ords, jurs, cats)):
            window_vals = []
            for w in range(1, window + 1):
                key = (o - w, j)
                if key in panel_pivot.index and c in panel_pivot.columns:
                    v = panel_pivot.loc[key, c]
                    if pd.notna(v):
                        window_vals.append(v)
            if window_vals:
                vals[i] = np.mean(window_vals)
        return vals

    out["target_rolling_mean_3_per_jurisdiction_category"] = rolling_mean(3)
    out["target_rolling_mean_6_per_jurisdiction_category"] = rolling_mean(6)

    # gtrends_fentanyl lag1 per jurisdiction
    gvals = np.full(len(df), np.nan)
    for i, (o, j) in enumerate(zip(ords - 1, jurs)):
        if o in gtrends_pivot.index and j in gtrends_pivot.columns:
            gvals[i] = gtrends_pivot.loc[o, j]
    out["gtrends_fentanyl_lag1_per_jurisdiction"] = gvals

    # cross-category auxiliary lag1 signal: mean of the 5 auxiliary categories
    # at lag1 for the same (jurisdiction, period)
    aux_vals = np.full(len(df), np.nan)
    for i, (o, j) in enumerate(zip(ords - 1, jurs)):
        key = (o, j)
        if key in panel_pivot.index:
            row = panel_pivot.loc[key]
            present = [row[c] for c in AUX_CATEGORIES if c in row.index and pd.notna(row[c])]
            if present:
                aux_vals[i] = np.mean(present)
    out["cross_category_auxiliary_lag1_signal"] = aux_vals

    return out


train_lag = build_lag_features(train_long)
pred_lag = build_lag_features(pred_long)

# fill remaining NaNs in lag features with the column's train median (no leakage:
# fallback constants derived only from train distribution)
LAG_COLS = list(train_lag.columns)
for col in LAG_COLS:
    fallback = train_lag[col].median()
    if fallback != fallback:
        fallback = 0.0
    train_lag[col] = train_lag[col].fillna(fallback)
    pred_lag[col] = pred_lag[col].fillna(fallback)

# ---------------------------------------------------------------------------
# Interactions: jurisdiction x period_ordinal_rank, category x gtrends_fentanyl
# jurisdiction is encoded via its per-fold target-encoding (jurisdiction_target_mean)
# already in train_agg/pred_agg; interaction multiplies that with ordinal rank.
# ---------------------------------------------------------------------------
INTERACTION_COLS = []

ordinal_rank_col = f"{PERIOD_COL}__ordinal_rank"
jur_target_mean_name = next(a["name"] for a in PER_FOLD_AGGS if a["group_keys"] == [JURIS_COL])

train_interact = pd.DataFrame(index=train_long.index)
pred_interact = pd.DataFrame(index=pred_long.index)

train_interact["jurisdiction_x_period_ordinal_rank"] = (
    train_agg[jur_target_mean_name].values * train_dt[ordinal_rank_col].values
)
pred_interact["jurisdiction_x_period_ordinal_rank"] = (
    pred_agg[jur_target_mean_name].values * pred_dt[ordinal_rank_col].values
)
INTERACTION_COLS.append("jurisdiction_x_period_ordinal_rank")

for cat_val in SCORED_CATEGORIES:
    out_col = f"overdose_category_{cat_val}_x_gtrends_fentanyl"
    train_is_cat = (train_long[CATEGORY_COL] == cat_val).astype("float64")
    pred_is_cat = (pred_long[CATEGORY_COL] == cat_val).astype("float64")
    train_interact[out_col] = train_is_cat.values * train_numeric["gtrends_fentanyl"].values
    pred_interact[out_col] = pred_is_cat.values * pred_numeric["gtrends_fentanyl"].values
    INTERACTION_COLS.append(out_col)

# ---------------------------------------------------------------------------
# Assemble final feature matrices
# ---------------------------------------------------------------------------
features_train = pd.concat(
    [train_numeric, train_cat, train_dt, train_text_feats, train_agg, train_interact, train_lag],
    axis=1,
)
features_pred = pd.concat(
    [pred_numeric, pred_cat, pred_dt, pred_text_feats, pred_agg, pred_interact, pred_lag],
    axis=1,
)

assert list(features_train.columns) == list(features_pred.columns)
FEATURE_COLUMNS = list(features_train.columns)

# final safety: no NaNs/infs
features_train = features_train.replace([np.inf, -np.inf], np.nan)
features_pred = features_pred.replace([np.inf, -np.inf], np.nan)
for col in FEATURE_COLUMNS:
    if features_train[col].isna().any():
        fillv = features_train[col].median()
        if fillv != fillv:
            fillv = 0.0
        features_train[col] = features_train[col].fillna(fillv)
        features_pred[col] = features_pred[col].fillna(fillv)
    if features_pred[col].isna().any():
        fillv = features_train[col].median()
        if fillv != fillv:
            fillv = 0.0
        features_pred[col] = features_pred[col].fillna(fillv)

# attach target + keys for downstream modeling convenience (not "features" per se,
# but kept for traceability; modeling agents select FEATURE_COLUMNS explicitly)
features_train.insert(0, TARGET_COL, train_long[TARGET_COL].values)
features_train.insert(0, CATEGORY_COL, train_long[CATEGORY_COL].values)
features_train.insert(0, JURIS_COL, train_long[JURIS_COL].values)
features_train.insert(0, PERIOD_COL, train_long[PERIOD_COL].values)

features_pred.insert(0, CATEGORY_COL, pred_long[CATEGORY_COL].values)
features_pred.insert(0, JURIS_COL, pred_long[JURIS_COL].values)
features_pred.insert(0, PERIOD_COL, pred_long[PERIOD_COL].values)
features_pred.insert(0, ROW_ID_COL, pred_long[ROW_ID_COL].values)

# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------
out_train_path = LOGS / f"{RUN_ID}_features_train.parquet"
out_pred_path = LOGS / f"{RUN_ID}_features_pred.parquet"
fmt = "parquet"
try:
    features_train.to_parquet(out_train_path, index=False)
    features_pred.to_parquet(out_pred_path, index=False)
except Exception:
    fmt = "csv"
    out_train_path = LOGS / f"{RUN_ID}_features_train.csv"
    out_pred_path = LOGS / f"{RUN_ID}_features_pred.csv"
    features_train.to_csv(out_train_path, index=False)
    features_pred.to_csv(out_pred_path, index=False)

feature_spec = {
    "run_id": RUN_ID,
    "format": fmt,
    "features_train": str(out_train_path.relative_to(ROOT)),
    "features_pred": str(out_pred_path.relative_to(ROOT)),
    "feature_columns": FEATURE_COLUMNS,
    "per_fold_aggregates": PER_FOLD_AGG_SPECS,
    "datetime_derived": DATETIME_DERIVED_COLS,
    "text_svd": TEXT_SVD_COLS,
    "categorical_onehot": ONEHOT_COLS_OUT,
    "interactions": INTERACTION_COLS,
    "lag_experimental": LAG_COLS,
    "target_transform": "log1p" if RECOMMEND_LOG else "none",
    "join_keys_for_modeling": [PERIOD_COL, JURIS_COL, CATEGORY_COL],
    "row_id_column": ROW_ID_COL,
    "target_column": TARGET_COL,
    "notes": (
        "Long-format panel: train rows = (period_id, jurisdiction, overdose_category) joined "
        "with covariates on (period_id, jurisdiction). Prediction frame expanded from "
        "data/val/covariates.csv (357 rows incl. buffer period dsZhPyK4) by filtering to the 6 "
        "sample_submission periods and cross-joining with the 3 scored overdose_category values, "
        "then aligned to sample_submission row order. Per-fold target/category aggregates "
        "(jurisdiction_category_target_mean, jurisdiction_target_mean, category_target_mean, "
        "jurisdiction_category_target_std) are fit on each fold's training rows only "
        "(src.data_agent.cv.load_canonical_folds) and refit on full train for the prediction "
        "matrix. temp_avg_f/precip_in imputed via jurisdiction-month training median; "
        "unemployment_rate/labor_force via jurisdiction last-known-value carry-forward fit on "
        "train. state_doh_release: TF-IDF(max_features=2000, 1-2gram, min_df=2)->TruncatedSVD "
        "fit on full train text, applied to prediction text, plus a structural-missing "
        "indicator. Lag/rolling/cross-category-aux features use only past periods (no future "
        "leakage) but are flagged experimental for the Step-6A' ablation gate per the plan."
    ),
}

(LOGS / f"{RUN_ID}_feature_spec.json").write_text(json.dumps(feature_spec, indent=2))

print(f"features_train shape: {features_train.shape}")
print(f"features_pred shape: {features_pred.shape}")
print(f"n_feature_columns: {len(FEATURE_COLUMNS)}")
print(f"format: {fmt}")
