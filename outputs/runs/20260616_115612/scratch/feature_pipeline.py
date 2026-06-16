"""
Feature pipeline for run 20260616_115612.
Resolves all column/file/task names at runtime from spec_parse.json and analysis_plan.json.
Leakage-safe: all target-derived features are computed per canonical fold.
"""
import json
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path("/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo")
RUN_ID = "20260616_115612"
LOGS_DIR = REPO_ROOT / "outputs" / "runs" / RUN_ID / "logs"

# ---------------------------------------------------------------------------
# Load contracts
# ---------------------------------------------------------------------------
spec = json.loads((LOGS_DIR / "spec_parse.json").read_text())
plan = json.loads((LOGS_DIR / "analysis_plan.json").read_text())
cv_raw = json.loads((LOGS_DIR / "cv_folds.json").read_text())

TRAIN_FILE = REPO_ROOT / spec["train_file"]
PRED_FILE = REPO_ROOT / spec["prediction_file"]
TARGET_COL = spec["target_column"]
ROW_ID_COL = spec["row_id_column"]

# feature plan constants (resolved from plan)
DIRECT_NUM = plan["feature_plan"]["direct_numeric"]
IMPUTATION_PLAN = plan["feature_plan"]["imputation"]

# ---------------------------------------------------------------------------
# Load raw data
# ---------------------------------------------------------------------------
train_raw = pd.read_csv(TRAIN_FILE)
pred_raw = pd.read_csv(PRED_FILE)

print(f"Loaded train: {train_raw.shape}, pred: {pred_raw.shape}")

# Canonical fold assignment from cv_folds.json
fold_assignment = np.array(cv_raw["fold_assignment"])  # length = n_train_rows
n_folds = cv_raw["n_folds"]

# ---------------------------------------------------------------------------
# Helper: extract title from Name
# ---------------------------------------------------------------------------
TITLE_MAP = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
    "Don": "Rare", "Rev": "Rare", "Dr": "Rare", "Mme": "Mrs",
    "Major": "Rare", "Lady": "Rare", "Sir": "Rare", "Mlle": "Miss",
    "Col": "Rare", "Capt": "Rare", "Countess": "Rare", "Jonkheer": "Rare",
    "Dona": "Rare",
}
TITLE_ORDINAL = {"Mr": 0, "Mrs": 1, "Miss": 2, "Master": 3, "Rare": 4}


def extract_title(name_series: pd.Series) -> pd.Series:
    """Extract and map title from Name column."""
    extracted = name_series.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    mapped = extracted.map(TITLE_MAP).fillna("Rare")
    return mapped


def extract_deck(cabin_series: pd.Series) -> pd.Series:
    """Extract deck letter from Cabin column; NaN → 'Unknown'."""
    deck = cabin_series.str.extract(r"^([A-Z])", expand=False)
    return deck.fillna("Unknown")


def extract_ticket_prefix(ticket_series: pd.Series) -> pd.Series:
    """Extract alpha prefix from Ticket; numeric-only → 'NUM'."""
    def get_prefix(t):
        t = str(t).strip().replace(".", "").replace("/", "").replace(" ", "")
        alpha = re.sub(r"\d+$", "", t).strip()
        return alpha if alpha else "NUM"
    return ticket_series.apply(get_prefix)


# ---------------------------------------------------------------------------
# Base feature extraction (fold-safe — no target used here)
# ---------------------------------------------------------------------------

def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build non-target-derived features. Safe to apply to any frame."""
    out = pd.DataFrame(index=df.index)

    # Direct numerics (filled with -1 as placeholder; per-fold median imputation
    # is applied for Age/Fare when computing per-fold features for train)
    for col in DIRECT_NUM:
        if col in df.columns:
            out[col] = df[col].values
        else:
            out[col] = np.nan

    # Sex one-hot
    sex_dummies = pd.get_dummies(df["Sex"], prefix="Sex", drop_first=False)
    for c in sex_dummies.columns:
        out[c] = sex_dummies[c].values

    # Embarked — mode impute will be done per fold; for base we fill with "S" temporarily
    embarked = df["Embarked"].fillna("S")
    emb_dummies = pd.get_dummies(embarked, prefix="Emb", drop_first=False)
    for c in emb_dummies.columns:
        out[c] = emb_dummies[c].values

    # Title (ordinal)
    title_str = extract_title(df["Name"])
    out["title"] = title_str.map(TITLE_ORDINAL).fillna(4).astype(int)

    # Deck (ordinal)
    deck_str = extract_deck(df["Cabin"])
    deck_cats = ["A", "B", "C", "D", "E", "F", "G", "T", "Unknown"]
    deck_ord = {d: i for i, d in enumerate(deck_cats)}
    out["deck"] = deck_str.map(deck_ord).fillna(len(deck_cats)).astype(int)

    # has_cabin flag
    out["has_cabin"] = df["Cabin"].notna().astype(int)

    # ticket_prefix (string for later target encoding)
    out["ticket_prefix_str"] = extract_ticket_prefix(df["Ticket"])

    # family_size interaction
    out["family_size"] = df["SibSp"] + df["Parch"] + 1

    # is_alone flag: no siblings, spouse, parents, or children
    out["is_alone"] = ((df["SibSp"] + df["Parch"]) == 0).astype(int)

    # is_large_family flag (family_size >= 4, i.e. SibSp+Parch >= 3)
    out["is_large_family"] = (out["family_size"] >= 4).astype(int)

    # Sex_x_Pclass interaction (ordinal product)
    sex_int = (df["Sex"] == "female").astype(int)
    out["Sex_x_Pclass"] = sex_int * df["Pclass"]

    # Sex_x_Age (female=1 * Age)
    out["Sex_x_Age"] = sex_int * df["Age"].fillna(df["Age"].median())

    # Pclass_x_Fare interaction
    out["Pclass_x_Fare"] = df["Pclass"] * df["Fare"].fillna(df["Fare"].median())

    return out


# ---------------------------------------------------------------------------
# Per-fold pipeline
# ---------------------------------------------------------------------------
# Columns that are target-derived (computed inside fold loop):
# tgt_mean_Sex, tgt_mean_Pclass, tgt_mean_Embarked, tgt_mean_Sex_Pclass,
# ticket_group_size, Age_imputed, Fare_imputed, ticket_prefix_enc,
# Embarked (re-encoded properly after fold-mode imputation)

PER_FOLD_AGG_SPECS = [
    {"name": "tgt_mean_Sex", "group_keys": ["Sex"]},
    {"name": "tgt_mean_Pclass", "group_keys": ["Pclass"]},
    {"name": "tgt_mean_Embarked", "group_keys": ["Embarked"]},
    {"name": "tgt_mean_Sex_Pclass", "group_keys": ["Sex", "Pclass"]},
    {"name": "tgt_mean_title", "group_keys": ["_title"]},
]

# Initialize arrays for OOF aggregates
n_train = len(train_raw)
oof_arrays = {s["name"]: np.full(n_train, np.nan) for s in PER_FOLD_AGG_SPECS}
oof_arrays["ticket_group_size"] = np.full(n_train, np.nan)
oof_arrays["ticket_prefix_enc"] = np.full(n_train, np.nan)
oof_arrays["Age_imputed"] = train_raw["Age"].values.copy().astype(float)
oof_arrays["Fare_imputed"] = train_raw["Fare"].values.copy().astype(float)
oof_arrays["Embarked_mode"] = train_raw["Embarked"].values.copy()
# tgt_mean_title is in PER_FOLD_AGG_SPECS (group_keys=["_title"]) so it's already allocated above

# Accumulators for full-train statistics (for prediction frame)
full_tgt_agg = {}
full_ticket_group = {}
full_ticket_prefix_enc = {}
full_age_by_title_median = {}
full_fare_median = None
full_embarked_mode = None

print("Running per-fold feature engineering...")

for fold_k in range(n_folds):
    val_mask = (fold_assignment == fold_k)
    train_mask = ~val_mask
    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    fold_train = train_raw.iloc[train_idx].copy()
    fold_val = train_raw.iloc[val_idx].copy()

    # --- Age imputation: median by title group from fold_train ---
    fold_train["_title"] = extract_title(fold_train["Name"])
    fold_val["_title"] = extract_title(fold_val["Name"])

    age_by_title = fold_train.groupby("_title")["Age"].median()
    global_age_med = fold_train["Age"].median()

    def impute_age(row):
        if pd.isna(row["Age"]):
            return age_by_title.get(row["_title"], global_age_med)
        return row["Age"]

    oof_arrays["Age_imputed"][val_idx] = fold_val.apply(impute_age, axis=1).values

    # Accumulate full-train age medians (last fold overwrites are fine — we do a final full-train fit below)
    # --- Fare imputation: fold_train median ---
    fare_med = fold_train["Fare"].median()
    fare_val = fold_val["Fare"].fillna(fare_med).values
    oof_arrays["Fare_imputed"][val_idx] = fare_val

    # --- Embarked mode from fold_train ---
    emb_mode = fold_train["Embarked"].mode()[0]
    emb_val_filled = fold_val["Embarked"].fillna(emb_mode).values
    oof_arrays["Embarked_mode"][val_idx] = emb_val_filled

    # --- Target aggregates ---
    for agg_spec in PER_FOLD_AGG_SPECS:
        gkeys = agg_spec["group_keys"]
        agg_name = agg_spec["name"]
        global_mean = fold_train[TARGET_COL].mean()

        if len(gkeys) == 1:
            g = gkeys[0]
            # _title is a derived column already added to fold_train/fold_val above
            if g not in fold_train.columns:
                oof_arrays[agg_name][val_idx] = global_mean
                continue
            grp_mean = fold_train.groupby(g)[TARGET_COL].mean()
            vals = fold_val[g].map(grp_mean).fillna(global_mean).values
        else:
            # Multi-key: create composite key
            fold_train["_ckey"] = fold_train[gkeys].astype(str).agg("__".join, axis=1)
            fold_val["_ckey"] = fold_val[gkeys].astype(str).agg("__".join, axis=1)
            grp_mean = fold_train.groupby("_ckey")[TARGET_COL].mean()
            vals = fold_val["_ckey"].map(grp_mean).fillna(global_mean).values

        oof_arrays[agg_name][val_idx] = vals

    # --- Ticket group size (count from fold_train only) ---
    ticket_counts = fold_train["Ticket"].value_counts()
    oof_arrays["ticket_group_size"][val_idx] = fold_val["Ticket"].map(ticket_counts).fillna(1).values

    # --- Ticket prefix target encoding (per fold) ---
    fold_train["_tpfx"] = extract_ticket_prefix(fold_train["Ticket"])
    fold_val["_tpfx"] = extract_ticket_prefix(fold_val["Ticket"])
    global_mean_pfx = fold_train[TARGET_COL].mean()
    pfx_mean = fold_train.groupby("_tpfx")[TARGET_COL].mean()
    oof_arrays["ticket_prefix_enc"][val_idx] = fold_val["_tpfx"].map(pfx_mean).fillna(global_mean_pfx).values

print("Per-fold done. Computing full-train statistics for prediction frame...")

# ---------------------------------------------------------------------------
# Full-train statistics (for prediction frame)
# ---------------------------------------------------------------------------
train_raw["_title_ft"] = extract_title(train_raw["Name"])
full_age_by_title_median = train_raw.groupby("_title_ft")["Age"].median()
full_global_age_med = train_raw["Age"].median()
full_fare_median = train_raw["Fare"].median()
full_embarked_mode = train_raw["Embarked"].mode()[0]

for agg_spec in PER_FOLD_AGG_SPECS:
    gkeys = agg_spec["group_keys"]
    agg_name = agg_spec["name"]
    global_mean = train_raw[TARGET_COL].mean()
    if len(gkeys) == 1:
        g = gkeys[0]
        if g == "_title":
            # Use the already-derived _title_ft column
            full_tgt_agg[agg_name] = train_raw.groupby("_title_ft")[TARGET_COL].mean()
        elif g in train_raw.columns:
            full_tgt_agg[agg_name] = train_raw.groupby(g)[TARGET_COL].mean()
        else:
            full_tgt_agg[agg_name] = pd.Series(dtype=float)
    else:
        train_raw["_ckey_ft"] = train_raw[gkeys].astype(str).agg("__".join, axis=1)
        full_tgt_agg[agg_name] = train_raw.groupby("_ckey_ft")[TARGET_COL].mean()

full_ticket_group = train_raw["Ticket"].value_counts()
train_raw["_tpfx_ft"] = extract_ticket_prefix(train_raw["Ticket"])
full_ticket_prefix_enc = train_raw.groupby("_tpfx_ft")[TARGET_COL].mean()
full_global_pfx_mean = train_raw[TARGET_COL].mean()

# ---------------------------------------------------------------------------
# Assemble train feature matrix
# ---------------------------------------------------------------------------
print("Assembling train feature matrix...")
feat_train = build_base_features(train_raw)

# Overwrite Age, Fare with fold-imputed values
feat_train["Age"] = oof_arrays["Age_imputed"]
feat_train["Fare"] = oof_arrays["Fare_imputed"]

# is_child flag: Age < 12 using fold-imputed age
feat_train["is_child"] = (oof_arrays["Age_imputed"] < 12).astype(int)

# fare_per_person: Fare / family_size (clipped at 1 to avoid divide-by-zero)
feat_train["fare_per_person"] = oof_arrays["Fare_imputed"] / feat_train["family_size"].clip(lower=1).values

# Overwrite Sex_x_Age and Pclass_x_Fare with properly imputed values
sex_int_tr = (train_raw["Sex"] == "female").astype(int)
feat_train["Sex_x_Age"] = sex_int_tr.values * oof_arrays["Age_imputed"]
feat_train["Pclass_x_Fare"] = train_raw["Pclass"].values * oof_arrays["Fare_imputed"]

# Re-do Embarked one-hot using fold-imputed mode
embarked_filled = pd.Series(oof_arrays["Embarked_mode"], index=train_raw.index)
emb_dum = pd.get_dummies(embarked_filled, prefix="Emb", drop_first=False)
for c in emb_dum.columns:
    feat_train[c] = emb_dum[c].values

# Append per-fold target aggregates
for agg_name in [s["name"] for s in PER_FOLD_AGG_SPECS]:
    feat_train[agg_name] = oof_arrays[agg_name]

feat_train["ticket_group_size"] = oof_arrays["ticket_group_size"]
feat_train["ticket_prefix_enc"] = oof_arrays["ticket_prefix_enc"]

# Drop ticket_prefix_str helper column (not a model feature)
if "ticket_prefix_str" in feat_train.columns:
    feat_train.drop(columns=["ticket_prefix_str"], inplace=True)

print(f"Train feature matrix: {feat_train.shape}")

# ---------------------------------------------------------------------------
# Assemble prediction feature matrix
# ---------------------------------------------------------------------------
print("Assembling prediction feature matrix...")
feat_pred = build_base_features(pred_raw)

# Age imputation: by title group median from full train
pred_raw["_title_pred"] = extract_title(pred_raw["Name"])
def impute_age_pred(row):
    if pd.isna(row["Age"]):
        return full_age_by_title_median.get(row["_title_pred"], full_global_age_med)
    return row["Age"]
feat_pred["Age"] = pred_raw.apply(impute_age_pred, axis=1).values

# Fare imputation
feat_pred["Fare"] = pred_raw["Fare"].fillna(full_fare_median).values

# is_child flag using imputed age
feat_pred["is_child"] = (feat_pred["Age"].values < 12).astype(int)

# fare_per_person: Fare / family_size
feat_pred["fare_per_person"] = feat_pred["Fare"].values / feat_pred["family_size"].clip(lower=1).values

# Embarked: fill with full-train mode, redo one-hot
embarked_pred_filled = pred_raw["Embarked"].fillna(full_embarked_mode)
emb_dum_pred = pd.get_dummies(embarked_pred_filled, prefix="Emb", drop_first=False)

# Ensure same columns as train
for c in [col for col in feat_train.columns if col.startswith("Emb_")]:
    if c not in emb_dum_pred.columns:
        emb_dum_pred[c] = 0
for c in emb_dum_pred.columns:
    feat_pred[c] = emb_dum_pred[c].values

# Recalculate interaction features with properly imputed values
sex_int_pred = (pred_raw["Sex"] == "female").astype(int)
feat_pred["Sex_x_Age"] = sex_int_pred.values * feat_pred["Age"].values
feat_pred["Pclass_x_Fare"] = pred_raw["Pclass"].values * feat_pred["Fare"].values

# Per-fold target aggregates (from full train)
for agg_spec in PER_FOLD_AGG_SPECS:
    gkeys = agg_spec["group_keys"]
    agg_name = agg_spec["name"]
    global_mean = train_raw[TARGET_COL].mean()
    if len(gkeys) == 1:
        g = gkeys[0]
        if g == "_title":
            # Use _title_pred derived column
            feat_pred[agg_name] = pred_raw["_title_pred"].map(full_tgt_agg[agg_name]).fillna(global_mean).values
        elif g in pred_raw.columns:
            feat_pred[agg_name] = pred_raw[g].map(full_tgt_agg[agg_name]).fillna(global_mean).values
        else:
            feat_pred[agg_name] = global_mean
    else:
        pred_raw["_ckey_pred"] = pred_raw[gkeys].astype(str).agg("__".join, axis=1)
        feat_pred[agg_name] = pred_raw["_ckey_pred"].map(full_tgt_agg[agg_name]).fillna(global_mean).values

# Ticket group size from full train counts
feat_pred["ticket_group_size"] = pred_raw["Ticket"].map(full_ticket_group).fillna(1).values

# Ticket prefix target encoding from full train
pred_raw["_tpfx_pred"] = extract_ticket_prefix(pred_raw["Ticket"])
feat_pred["ticket_prefix_enc"] = pred_raw["_tpfx_pred"].map(full_ticket_prefix_enc).fillna(full_global_pfx_mean).values

# Drop ticket_prefix_str helper column
if "ticket_prefix_str" in feat_pred.columns:
    feat_pred.drop(columns=["ticket_prefix_str"], inplace=True)

print(f"Pred feature matrix: {feat_pred.shape}")

# ---------------------------------------------------------------------------
# Align columns between train and pred
# ---------------------------------------------------------------------------
train_cols = set(feat_train.columns)
pred_cols = set(feat_pred.columns)

# Add any missing columns to pred (fill with 0)
for c in sorted(train_cols - pred_cols):
    feat_pred[c] = 0

# Add any missing columns to train (fill with 0)
for c in sorted(pred_cols - train_cols):
    feat_train[c] = 0

# Ensure same column order
all_cols = sorted(feat_train.columns)
feat_train = feat_train[all_cols]
feat_pred = feat_pred[all_cols]

print(f"Final shapes — train: {feat_train.shape}, pred: {feat_pred.shape}")
print(f"Feature columns ({len(all_cols)}): {all_cols[:10]}...")

# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------
train_out = LOGS_DIR / "features_train.parquet"
pred_out = LOGS_DIR / "features_pred.parquet"

try:
    feat_train.to_parquet(train_out, index=False)
    feat_pred.to_parquet(pred_out, index=False)
    output_format = "parquet"
    print(f"Wrote parquet files: {train_out}, {pred_out}")
except Exception as e:
    print(f"Parquet write failed ({e}), falling back to CSV")
    train_out = LOGS_DIR / "features_train.csv"
    pred_out = LOGS_DIR / "features_pred.csv"
    feat_train.to_csv(train_out, index=False)
    feat_pred.to_csv(pred_out, index=False)
    output_format = "csv"

# ---------------------------------------------------------------------------
# Write feature_spec.json
# ---------------------------------------------------------------------------
per_fold_agg_spec = [
    {"name": s["name"], "group_keys": s["group_keys"], "source": "per_fold"}
    for s in PER_FOLD_AGG_SPECS
] + [
    {"name": "ticket_group_size", "group_keys": ["Ticket"], "source": "per_fold"},
    {"name": "ticket_prefix_enc", "group_keys": ["Ticket"], "source": "per_fold"},
]

spec_out = {
    "run_id": RUN_ID,
    "format": output_format,
    "features_train": str(train_out),
    "features_pred": str(pred_out),
    "feature_columns": all_cols,
    "per_fold_aggregates": per_fold_agg_spec,
    "datetime_derived": [],
    "text_svd": [],
    "image_features": [],
    "target_transform": "none",
    "notes": (
        "Round 2. Binary classification (Titanic). Features: direct numerics (Pclass/Age/SibSp/Parch/Fare), "
        "Sex one-hot, Embarked one-hot (mode-imputed per fold on train), title ordinal (from Name), "
        "deck ordinal (from Cabin), has_cabin flag, family_size, is_alone (SibSp+Parch==0), "
        "is_large_family (family_size>=4), is_child (Age<12 post fold-imputation), "
        "fare_per_person (Fare/family_size), interactions (Sex_x_Pclass, Sex_x_Age, Pclass_x_Fare). "
        "Per-fold target aggregates: tgt_mean_Sex, tgt_mean_Pclass, tgt_mean_Embarked, "
        "tgt_mean_Sex_Pclass, tgt_mean_title; ticket_group_size and ticket_prefix_enc also per-fold. "
        "Age imputed by title-group median from fold-train; Fare by fold-train median. "
        "No capability gaps. representation_strategy=tabular_ml."
    ),
}

spec_path = LOGS_DIR / "feature_spec.json"
spec_path.write_text(json.dumps(spec_out, indent=2))
print(f"Wrote feature_spec.json: {spec_path}")
print("Feature pipeline complete.")
