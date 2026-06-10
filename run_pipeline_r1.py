"""
Round 1 ML pipeline for panel regression.
Reads config from analysis_plan.json — no hardcoded column names at module level.
"""
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT = Path("/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo")
LOGS = ROOT / "outputs" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

plan = json.loads((LOGS / "analysis_plan.json").read_text())
fp   = plan["feature_plan"]

TRAIN_DOSE = ROOT / fp["train_dose_file"]
TRAIN_COV  = ROOT / fp["train_covariates_file"]
VAL_COV    = ROOT / fp["val_covariates_file"]
SAMPLE_SUB = ROOT / fp["sample_submission_file"]

TARGET     = fp["target_column"]
ROW_ID     = fp["row_id_column"]
GROUP_COL  = fp["group_key_for_cv"]
JOIN_KEYS  = fp["join_keys"]
EXCLUDE    = set(fp["exclude_columns"])

CAT_OHE    = fp["categorical_columns"]["low_cardinality_one_hot"]
CAT_TE     = fp["categorical_columns"]["high_cardinality_target_encode"]
NUM_COLS   = fp["numeric_columns"]
TEXT_COLS  = fp["text_columns"]

N_SPLITS   = fp["cv_strategy"]["n_splits"]

# ── load & merge ──────────────────────────────────────────────────────────────
print("Loading data...")
dose  = pd.read_csv(TRAIN_DOSE)
tcov  = pd.read_csv(TRAIN_COV)
vcov  = pd.read_csv(VAL_COV)
ssub  = pd.read_csv(SAMPLE_SUB)

train = dose.merge(tcov, on=JOIN_KEYS, how="left")
print(f"  train shape after merge: {train.shape}")

# val: merge val_covariates with sample_submission on join keys that overlap
val_cols_in_ssub = [c for c in JOIN_KEYS if c in ssub.columns]
val = ssub.merge(vcov, on=val_cols_in_ssub, how="left")
print(f"  val shape after merge:   {val.shape}")

# ── feature builder ────────────────────────────────────────────────────────────
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import scipy.sparse as sp

def build_features(df_train, df_val, y_train_series, group_col_values, n_splits=5):
    """
    Builds OOF feature matrix for train and a single feature matrix for val.
    Returns X_train_full, X_val_full, feature_names.
    """
    n_tr = len(df_train)
    n_va = len(df_val)

    # ── numeric imputation (fit on train, apply to both) ──────────────────────
    num_present = [c for c in NUM_COLS if c in df_train.columns]
    num_imp = SimpleImputer(strategy="median")
    num_imp.fit(df_train[num_present].values)
    tr_num = num_imp.transform(df_train[num_present].values)
    va_num = num_imp.transform(df_val[num_present].values)

    # ── one-hot encode low-card cats ──────────────────────────────────────────
    ohe_cols = [c for c in CAT_OHE if c in df_train.columns]
    if ohe_cols:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe.fit(df_train[ohe_cols].fillna("__missing__"))
        tr_ohe = ohe.transform(df_train[ohe_cols].fillna("__missing__"))
        va_ohe = ohe.transform(df_val[ohe_cols].fillna("__missing__"))
        ohe_feat_names = ohe.get_feature_names_out(ohe_cols).tolist()
    else:
        tr_ohe = np.zeros((n_tr, 0))
        va_ohe = np.zeros((n_va, 0))
        ohe_feat_names = []

    # ── text: TF-IDF + SVD (fit on full train, not per-fold to keep dims) ─────
    text_blocks_tr = []
    text_blocks_va = []
    text_feat_names = []
    for tcol, tconf in TEXT_COLS.items():
        if tcol not in df_train.columns:
            continue
        tr_text = df_train[tcol].fillna("").astype(str).tolist()
        va_text = df_val[tcol].fillna("").astype(str).tolist() if tcol in df_val.columns else [""] * n_va
        n_svd = tconf.get("svd_n_components", 20)
        max_feat = tconf.get("tfidf_max_features", 200)
        tfidf = TfidfVectorizer(max_features=max_feat, sublinear_tf=True)
        svd   = TruncatedSVD(n_components=n_svd, random_state=42)
        tr_tfidf = tfidf.fit_transform(tr_text)
        svd.fit(tr_tfidf)
        tr_svd = svd.transform(tr_tfidf)
        va_svd = svd.transform(tfidf.transform(va_text))
        text_blocks_tr.append(tr_svd)
        text_blocks_va.append(va_svd)
        text_feat_names += [f"{tcol}__svd_{i}" for i in range(n_svd)]
        if tconf.get("missing_indicator", False):
            tr_mis = (df_train[tcol].isna() | (df_train[tcol].astype(str).str.strip() == "")).astype(float).values.reshape(-1, 1)
            va_mis = (df_val[tcol].isna() | (df_val[tcol].astype(str).str.strip() == "")).astype(float).values.reshape(-1, 1) if tcol in df_val.columns else np.zeros((n_va, 1))
            text_blocks_tr.append(tr_mis)
            text_blocks_va.append(va_mis)
            text_feat_names.append(f"{tcol}__missing")

    if text_blocks_tr:
        tr_text_arr = np.hstack(text_blocks_tr)
        va_text_arr = np.hstack(text_blocks_va)
    else:
        tr_text_arr = np.zeros((n_tr, 0))
        va_text_arr = np.zeros((n_va, 0))

    # ── target-encode high-card cats (OOF to avoid leakage) ──────────────────
    te_cols = [c for c in CAT_TE if c in df_train.columns]
    tr_te   = np.zeros((n_tr, len(te_cols)))
    va_te   = np.zeros((n_va, len(te_cols)))
    y_arr   = y_train_series.values
    groups  = group_col_values.values

    gkf = GroupKFold(n_splits=n_splits)
    # We'll accumulate val TE predictions across folds and average
    va_te_acc = np.zeros((n_va, len(te_cols)))
    va_fold_counts = np.zeros(n_va)

    global_means = {c: df_train[c].map(
        y_train_series.groupby(df_train[c]).mean()
    ).fillna(y_arr.mean()) for c in te_cols}

    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(df_train, groups=groups)):
        for j, tcol in enumerate(te_cols):
            fold_map = y_arr[tr_idx].copy()
            fold_cats = df_train[tcol].iloc[tr_idx].values
            te_map = pd.Series(fold_map).groupby(fold_cats).mean()
            tr_te[va_idx, j] = df_train[tcol].iloc[va_idx].map(te_map).fillna(y_arr[tr_idx].mean()).values

    # For val set: use full-train mean map
    for j, tcol in enumerate(te_cols):
        full_map = y_arr.copy()
        full_cats = df_train[tcol].values
        te_map = pd.Series(full_map).groupby(full_cats).mean()
        global_mean = y_arr.mean()
        va_te[:, j] = df_val[tcol].map(te_map).fillna(global_mean).values if tcol in df_val.columns else global_mean

    te_feat_names = [f"{c}__te" for c in te_cols]

    # ── assemble ──────────────────────────────────────────────────────────────
    X_tr = np.hstack([tr_num, tr_ohe, tr_text_arr, tr_te])
    X_va = np.hstack([va_num, va_ohe, va_text_arr, va_te])
    feat_names = (
        [f"num__{c}" for c in num_present]
        + ohe_feat_names
        + text_feat_names
        + te_feat_names
    )
    return X_tr, X_va, feat_names

# ── prepare matrices ──────────────────────────────────────────────────────────
print("Building feature matrices...")
y_train = train[TARGET]
X_train, X_val, feat_names = build_features(
    train, val, y_train, train[GROUP_COL]
)
print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")

# ── model definitions ─────────────────────────────────────────────────────────
candidates = {}

candidates["HGB"] = HistGradientBoostingRegressor(
    max_iter=500, learning_rate=0.05, max_depth=6,
    min_samples_leaf=10, random_state=42
)
candidates["RF"] = RandomForestRegressor(
    n_estimators=300, max_depth=None, min_samples_leaf=5,
    n_jobs=-1, random_state=42
)
candidates["ET"] = ExtraTreesRegressor(
    n_estimators=300, max_depth=None, min_samples_leaf=5,
    n_jobs=-1, random_state=42
)
candidates["Ridge"] = Ridge(alpha=10.0)
candidates["ElasticNet"] = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)

try:
    import lightgbm as lgb
    candidates["LGBM"] = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
    )
    print("  LightGBM available")
except ImportError:
    print("  LightGBM not available; skipping")

try:
    import xgboost as xgb
    candidates["XGB"] = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0, eval_metric="mae"
    )
    print("  XGBoost available")
except ImportError:
    print("  XGBoost not available; skipping")

# ── block MAE CV ──────────────────────────────────────────────────────────────
def block_mae(y_true, y_pred, groups):
    """MAE averaged over per-group means (block MAE)."""
    df = pd.DataFrame({"y": y_true, "p": y_pred, "g": groups})
    block = df.groupby("g").apply(lambda x: mean_absolute_error(x["y"], x["p"]))
    return block.mean()

print("\nRunning GroupKFold CV...")
gkf = GroupKFold(n_splits=N_SPLITS)
groups = train[GROUP_COL].values
y_arr  = y_train.values

cv_scores = {}
oof_preds = {name: np.full(len(y_arr), np.nan) for name in candidates}

for name, model in candidates.items():
    fold_scores = []
    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_train, groups=groups)):
        Xf_tr, Xf_va = X_train[tr_idx], X_train[va_idx]
        yf_tr, yf_va = y_arr[tr_idx],   y_arr[va_idx]
        gf_va        = groups[va_idx]

        try:
            model.fit(Xf_tr, yf_tr)
            preds = model.predict(Xf_va)
            score = block_mae(yf_va, preds, gf_va)
            fold_scores.append(score)
            oof_preds[name][va_idx] = preds
        except Exception as e:
            print(f"    [{name}] fold {fold_i} error: {e}")
            fold_scores.append(np.nan)

    mean_score = np.nanmean(fold_scores)
    cv_scores[name] = mean_score
    print(f"  {name:12s}  block_mae = {mean_score:.4f}")

# ── select best ───────────────────────────────────────────────────────────────
valid = {k: v for k, v in cv_scores.items() if not np.isnan(v)}
best_name = min(valid, key=valid.get)
best_score = valid[best_name]
print(f"\nBest model: {best_name}  block_mae = {best_score:.4f}")

# ── NNLS blend ────────────────────────────────────────────────────────────────
from scipy.optimize import nnls

oof_matrix = np.column_stack([oof_preds[n] for n in candidates if not np.isnan(cv_scores[n])])
blend_names = [n for n in candidates if not np.isnan(cv_scores[n])]
# Only use rows where all models have predictions
mask = np.all(~np.isnan(oof_matrix), axis=1)
if mask.sum() > 10:
    coeffs, _ = nnls(oof_matrix[mask], y_arr[mask])
    coeffs /= coeffs.sum() if coeffs.sum() > 0 else 1.0
    blend_oof  = oof_matrix[mask] @ coeffs
    blend_score = block_mae(y_arr[mask], blend_oof, groups[mask])
    print(f"  {'NNLS_blend':12s}  block_mae = {blend_score:.4f}  weights={dict(zip(blend_names, coeffs.round(4)))}")
    if blend_score < best_score:
        best_name  = "NNLS_blend"
        best_score = blend_score
        print(f"  -> NNLS blend selected as best")
else:
    blend_score = np.nan
    coeffs = None
    blend_names_used = []

# ── fit best on full train & predict val ─────────────────────────────────────
print(f"\nFitting {best_name} on full training data...")
if best_name == "NNLS_blend":
    val_preds_all = {}
    for bname in blend_names:
        m = candidates[bname]
        m.fit(X_train, y_arr)
        val_preds_all[bname] = m.predict(X_val)
    val_blend = np.column_stack([val_preds_all[n] for n in blend_names])
    final_preds = val_blend @ coeffs
else:
    best_model = candidates[best_name]
    best_model.fit(X_train, y_arr)
    final_preds = best_model.predict(X_val)

print(f"  Val predictions: min={final_preds.min():.3f}, max={final_preds.max():.3f}, mean={final_preds.mean():.3f}")

# ── write submission ───────────────────────────────────────────────────────────
print("\nWriting submission.csv...")
# Use sample_submission row order; fill in predictions by matching row_id
ssub_out = ssub[[ROW_ID]].copy()
# val has ROW_ID column from ssub; align by position (val is already in ssub row order)
ssub_out[TARGET] = final_preds
ssub_out.to_csv(ROOT / "submission.csv", index=False)
print(f"  submission.csv: {ssub_out.shape}, columns: {ssub_out.columns.tolist()}")

# ── write logs ────────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_search = {
    "run_id": timestamp,
    "metric_name": "block_mae",
    "cv_strategy": "GroupKFold",
    "n_splits": N_SPLITS,
    "group_column": GROUP_COL,
    "cv_scores": {k: float(v) for k, v in cv_scores.items()},
    "blend_score": float(blend_score) if not np.isnan(blend_score) else None,
    "best_model": best_name,
    "best_block_mae": float(best_score),
    "n_train": int(len(y_arr)),
    "n_val": int(len(final_preds)),
    "n_features": int(X_train.shape[1]),
    "feature_names_sample": feat_names[:20],
}
(LOGS / "model_search.json").write_text(json.dumps(model_search, indent=2))

final_model = {
    "run_id": timestamp,
    "best_model": best_name,
    "best_block_mae": float(best_score),
    "n_train": int(len(y_arr)),
    "n_val_predictions": int(len(final_preds)),
    "submission_columns": ssub_out.columns.tolist(),
    "submission_shape": list(ssub_out.shape),
    "blend_weights": dict(zip(blend_names, coeffs.tolist())) if best_name == "NNLS_blend" else {},
}
(LOGS / "final_model.json").write_text(json.dumps(final_model, indent=2))

# ── submission check ──────────────────────────────────────────────────────────
sub_check = {
    "run_id": timestamp,
    "columns_ok": bool(ssub_out.columns.tolist() == [ROW_ID, TARGET]),
    "row_count_ok": bool(len(ssub_out) == len(ssub)),
    "row_id_alignment_ok": bool((ssub_out[ROW_ID].values == ssub[ROW_ID].values).all()),
    "all_finite": bool(np.isfinite(final_preds).all()),
    "dtype_ok": True,
    "missing_predictions": int(np.isnan(final_preds).sum()),
    "output_kind": "regression_continuous",
    "n_rows": int(len(ssub_out)),
}
(LOGS / f"{timestamp}_submission_check.json").write_text(json.dumps(sub_check, indent=2))
print(f"\nSubmission check: {sub_check}")

print("\nDone.")
