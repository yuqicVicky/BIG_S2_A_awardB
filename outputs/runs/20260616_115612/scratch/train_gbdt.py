"""
GBDT specialist training script for run 20260616_115612.
Binary classification (Titanic survival), metric=accuracy.
LightGBM primary, XGBoost/CatBoost fallback.
Canonical 5-fold stratified CV from cv_folds.json.
"""
import json
import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RUN_ID = "20260616_115612"
LOGS = f"/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo/outputs/runs/{RUN_ID}/logs"
HEARTBEAT_PATH = os.environ.get(
    "AWARDB_HEARTBEAT_PATH",
    f"{LOGS}/gbdt-specialist_progress.jsonl"
)
TIME_BUDGET = int(os.environ.get("AWARDB_TIME_BUDGET_SEC", "900"))
START_TIME = time.time()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def emit_heartbeat(msg: dict):
    msg["ts"] = time.time() - START_TIME
    with open(HEARTBEAT_PATH, "a") as f:
        f.write(json.dumps(msg, cls=NpEncoder) + "\n")


def jdump(obj, fp, **kwargs):
    json.dump(obj, fp, cls=NpEncoder, **kwargs)


# ── Load canonical folds ──────────────────────────────────────────────────────
with open(f"{LOGS}/cv_folds.json") as f:
    cv_folds_data = json.load(f)

fold_assignment = np.array(cv_folds_data["fold_assignment"])
n_folds = cv_folds_data["n_folds"]

# ── Load feature parquets ─────────────────────────────────────────────────────
with open(f"{LOGS}/feature_spec.json") as f:
    feat_spec = json.load(f)

feat_cols = feat_spec["feature_columns"]
X_train_full = pd.read_parquet(feat_spec["features_train"])
X_test_full  = pd.read_parquet(feat_spec["features_pred"])

# Keep only feature columns (intersect with available)
available_cols = [c for c in feat_cols if c in X_train_full.columns]
X_train = X_train_full[available_cols].copy()
X_test  = X_test_full[available_cols].copy()

# ── Load raw train for target ─────────────────────────────────────────────────
train_raw = pd.read_csv("/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo/data/train.csv")
test_raw  = pd.read_csv("/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo/data/test.csv")

# Read target col + row_id from spec
with open(f"{LOGS}/spec_parse.json") as f:
    spec = json.load(f)

target_col = spec["target_column"]
row_id_col = spec["row_id_column"]

y = train_raw[target_col].values
train_ids = train_raw[row_id_col].values
test_ids  = test_raw[row_id_col].values

emit_heartbeat({"status": "data_loaded", "n_train": int(len(y)), "n_test": int(len(test_ids)), "n_features": int(len(available_cols))})

# ── Modeling hints ────────────────────────────────────────────────────────────
# planner recommended gbdt as PRIMARY family → confirmed
# native_missing_handling_preferred=True → LightGBM handles NaN natively
# prefer_regularized=False → standard params
# apply_log1p_hint=False → no transform
emit_heartbeat({"status": "hints_applied", "primary_family": "gbdt", "native_nan": True, "regularized": False, "log1p": False})

# ── Floor bar ─────────────────────────────────────────────────────────────────
with open(f"{LOGS}/model_selection.json") as f:
    floor_data = json.load(f)
floor_score = float(floor_data["selected_metrics"]["accuracy"])
emit_heartbeat({"status": "floor_bar", "floor_accuracy": floor_score})

# ── LightGBM CV ───────────────────────────────────────────────────────────────
import lightgbm as lgb
from sklearn.metrics import accuracy_score

# Randomized hyperparameter search
rng = np.random.RandomState(42)

# Generate candidate configs
configs = []
for i in range(30):
    configs.append({
        "n_estimators": int(rng.randint(200, 800)),
        "learning_rate": float(rng.choice([0.01, 0.02, 0.05, 0.1, 0.15])),
        "num_leaves": int(rng.randint(15, 63)),
        "max_depth": int(rng.randint(3, 8)),
        "min_child_samples": int(rng.randint(10, 50)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_alpha": float(rng.choice([0.0, 0.01, 0.1, 1.0])),
        "reg_lambda": float(rng.choice([0.0, 0.1, 1.0, 5.0])),
        "seed": i,
    })

best_score = -np.inf
best_config = None
best_oof = None
best_test_preds = None

emit_heartbeat({"status": "tuning_start", "n_configs": len(configs)})

for cfg_idx, cfg in enumerate(configs):
    elapsed = time.time() - START_TIME
    if elapsed > TIME_BUDGET * 0.75:
        emit_heartbeat({"status": "time_cap_reached", "cfg_idx": cfg_idx, "elapsed": round(elapsed, 1)})
        break

    oof_proba = np.zeros(len(y))
    test_proba_list = []
    fold_accs = []

    for fold in range(n_folds):
        tr_idx = np.where(fold_assignment != fold)[0]
        va_idx = np.where(fold_assignment == fold)[0]

        X_tr, y_tr = X_train.iloc[tr_idx], y[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y[va_idx]

        model = lgb.LGBMClassifier(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            num_leaves=cfg["num_leaves"],
            max_depth=cfg["max_depth"],
            min_child_samples=cfg["min_child_samples"],
            subsample=cfg["subsample"],
            colsample_bytree=cfg["colsample_bytree"],
            reg_alpha=cfg["reg_alpha"],
            reg_lambda=cfg["reg_lambda"],
            random_state=cfg["seed"],
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        proba = model.predict_proba(X_va)[:, 1]
        oof_proba[va_idx] = proba
        fold_accs.append(float(accuracy_score(y_va, (proba >= 0.5).astype(int))))
        test_proba_list.append(model.predict_proba(X_test)[:, 1])

    cv_acc = float(np.mean(fold_accs))
    if cv_acc > best_score:
        best_score = cv_acc
        best_config = cfg
        best_oof = oof_proba.copy()
        best_test_preds = np.mean(test_proba_list, axis=0)
        emit_heartbeat({"status": "new_best", "cfg_idx": cfg_idx, "cv_accuracy": round(cv_acc, 6), "config": cfg})

emit_heartbeat({"status": "tuning_done", "best_cv_accuracy": round(best_score, 6), "best_config": best_config})

# ── Full-train refit with best config ────────────────────────────────────────
elapsed = time.time() - START_TIME
if elapsed < TIME_BUDGET * 0.9:
    emit_heartbeat({"status": "full_refit_start"})
    # seed averaging: refit with 3 seeds, average test preds
    test_proba_refit = []
    for seed in [42, 123, 777]:
        m = lgb.LGBMClassifier(
            n_estimators=best_config["n_estimators"],
            learning_rate=best_config["learning_rate"],
            num_leaves=best_config["num_leaves"],
            max_depth=best_config["max_depth"],
            min_child_samples=best_config["min_child_samples"],
            subsample=best_config["subsample"],
            colsample_bytree=best_config["colsample_bytree"],
            reg_alpha=best_config["reg_alpha"],
            reg_lambda=best_config["reg_lambda"],
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )
        m.fit(X_train, y)
        test_proba_refit.append(m.predict_proba(X_test)[:, 1])
    best_test_preds = np.mean(test_proba_refit, axis=0)
    emit_heartbeat({"status": "full_refit_done", "seeds": [42, 123, 777]})

# ── Output files ──────────────────────────────────────────────────────────────
# OOF CSV: PassengerId + probability (continuous for NNLS blend)
oof_df = pd.DataFrame({
    row_id_col: train_ids,
    target_col: best_oof,   # probabilities
})
oof_df.to_csv(f"{LOGS}/oof_gbdt.csv", index=False)

# Candidate submission: hard labels (threshold 0.5)
test_labels = (best_test_preds >= 0.5).astype(int)
cand_df = pd.DataFrame({
    row_id_col: test_ids,
    target_col: test_labels,
})
cand_df.to_csv(f"{LOGS}/cand_gbdt.csv", index=False)

# ── Prediction sanity ─────────────────────────────────────────────────────────
oof_preds_hard = (best_oof >= 0.5).astype(int)
oof_acc = float(accuracy_score(y, oof_preds_hard))
pred_pos_rate = float(test_labels.mean())
train_pos_rate = float(y.mean())

sanity = {
    "run_id": RUN_ID,
    "family": "gbdt",
    "oof_accuracy": round(oof_acc, 6),
    "train_positive_rate": round(train_pos_rate, 4),
    "pred_positive_rate": round(pred_pos_rate, 4),
    "rate_ratio": round(pred_pos_rate / train_pos_rate, 4) if train_pos_rate > 0 else None,
    "oof_proba_min": round(float(best_oof.min()), 4),
    "oof_proba_max": round(float(best_oof.max()), 4),
    "oof_proba_mean": round(float(best_oof.mean()), 4),
    "checks": {
        "oof_proba_in_01": bool((best_oof >= 0).all() and (best_oof <= 1).all()),
        "pred_rate_reasonable": bool(0.2 <= pred_pos_rate <= 0.6),
        "oof_acc_above_floor": bool(oof_acc > floor_score),
    }
}
with open(f"{LOGS}/prediction_sanity.json", "w") as f:
    jdump(sanity, f, indent=2)

# ── Agent JSON ────────────────────────────────────────────────────────────────
agent_out = {
    "role": "gbdt-specialist",
    "approach": "gbdt",
    "selected_model": "LightGBMClassifier",
    "cv_metric": "accuracy",
    "cv_score": round(float(best_score), 6),
    "lower_is_better": False,
    "beats_floor": bool(best_score > floor_score),
    "floor_score": round(floor_score, 6),
    "candidate_submission": f"{LOGS}/cand_gbdt.csv",
    "oof_path": f"{LOGS}/oof_gbdt.csv",
    "canonical_folds": True,
    "n_folds": int(n_folds),
    "best_params": best_config,
    "authored_features_used": available_cols,
    "monotonic_applied": False,
    "hints_applied": {
        "planner_recommended_primary": True,
        "native_missing_handling_preferred": True,
        "prefer_regularized": False,
        "apply_log1p_hint": False,
    },
    "elapsed_sec": round(time.time() - START_TIME, 1),
}
with open(f"{LOGS}/agent_gbdt.json", "w") as f:
    jdump(agent_out, f, indent=2)

# ── Promotion check ───────────────────────────────────────────────────────────
promotion = {
    "run_id": RUN_ID,
    "family": "gbdt",
    "cv_metric": "accuracy",
    "cv_score": round(float(best_score), 6),
    "floor_score": round(floor_score, 6),
    "promoted": bool(best_score > floor_score),
    "promotion_delta": round(float(best_score - floor_score), 6),
    "candidate_submission": f"{LOGS}/cand_gbdt.csv",
}
with open(f"{LOGS}/promotion.json", "w") as f:
    jdump(promotion, f, indent=2)

emit_heartbeat({"status": "done", "cv_accuracy": round(float(best_score), 6), "promoted": bool(best_score > floor_score)})

print(f"GBDT specialist done. CV accuracy={best_score:.6f}, floor={floor_score:.6f}, promoted={best_score > floor_score}")
print(f"OOF: {LOGS}/oof_gbdt.csv")
print(f"Candidate: {LOGS}/cand_gbdt.csv")
print(f"Agent JSON: {LOGS}/agent_gbdt.json")
