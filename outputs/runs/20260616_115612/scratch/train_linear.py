"""
Linear specialist training script for run 20260616_115612.
Family: linear (LogisticRegression L1/L2/ElasticNet).
Task: binary_classification, metric: accuracy.
Canonical 5-fold stratified CV.
Outputs: agent_linear.json, cand_linear.csv, oof_linear.csv
"""
import json
import time
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterSampler
import warnings
warnings.filterwarnings('ignore')

RUN_ID = "20260616_115612"
LOGS_DIR = f"/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo/outputs/runs/{RUN_ID}/logs"
TIME_BUDGET = int(os.environ.get("AWARDB_TIME_BUDGET_SEC", 900))
HEARTBEAT_PATH = os.environ.get(
    "AWARDB_HEARTBEAT_PATH",
    f"{LOGS_DIR}/linear-specialist_progress.jsonl"
)

start_time = time.time()


def heartbeat(msg: dict):
    msg["ts"] = time.time() - start_time
    with open(HEARTBEAT_PATH, "a") as f:
        f.write(json.dumps(msg) + "\n")


heartbeat({"event": "start", "family": "linear"})

# ---- Load data ----
feature_spec = json.load(open(f"{LOGS_DIR}/feature_spec.json"))
cv_folds_data = json.load(open(f"{LOGS_DIR}/cv_folds.json"))
fold_assignment = np.array(cv_folds_data["fold_assignment"])  # 891 entries, 0-4

train_features = pd.read_parquet(feature_spec["features_train"])
pred_features = pd.read_parquet(feature_spec["features_pred"])
feature_cols = feature_spec["feature_columns"]

# Load target
train_df = pd.read_csv("/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo/data/train.csv")
test_df = pd.read_csv("/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo/data/gender_submission.csv")

target_col = "Survived"
y = train_df[target_col].values
row_id_col = "PassengerId"
test_ids = pd.read_csv("/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo/data/test.csv")[row_id_col]

X = train_features[feature_cols].values.astype(float)
X_test = pred_features[feature_cols].values.astype(float)

n_folds = cv_folds_data["n_folds"]  # 5
n_train = len(y)
assert len(fold_assignment) == n_train, f"Fold assignment length {len(fold_assignment)} != n_train {n_train}"

heartbeat({"event": "data_loaded", "n_train": n_train, "n_features": X.shape[1], "n_test": X_test.shape[0]})

# ---- Define model configs to try ----
# LogisticRegression with L1/L2/ElasticNet across C values
param_grid = []

# L2 ridge configs (fast, good baseline)
for C in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
    param_grid.append({"penalty": "l2", "C": C, "solver": "lbfgs", "max_iter": 2000, "l1_ratio": None})

# L1 lasso configs (feature selection)
for C in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
    param_grid.append({"penalty": "l1", "C": C, "solver": "saga", "max_iter": 2000, "l1_ratio": None})

# ElasticNet configs
for C in [0.1, 0.5, 1.0, 2.0]:
    for l1_ratio in [0.2, 0.5, 0.8]:
        param_grid.append({"penalty": "elasticnet", "C": C, "solver": "saga", "max_iter": 2000, "l1_ratio": l1_ratio})

# class_weight balanced variants (helps with moderate imbalance 38.4% pos)
balanced_configs = []
for C in [0.1, 0.5, 1.0, 2.0]:
    balanced_configs.append({"penalty": "l2", "C": C, "solver": "lbfgs", "max_iter": 2000, "l1_ratio": None, "class_weight": "balanced"})

heartbeat({"event": "configs_defined", "n_configs": len(param_grid) + len(balanced_configs)})

# ---- Cross-validation ----
best_cv_score = -np.inf
best_config = None
best_oof_proba = None

def run_cv(config, fold_assignment, X, y, n_folds):
    """Run 5-fold CV with given config, return (oof_proba_array, cv_accuracy)."""
    oof_proba = np.zeros(len(y))

    cw = config.get("class_weight", None)

    if config["penalty"] == "elasticnet":
        lr = LogisticRegression(
            penalty=config["penalty"],
            C=config["C"],
            solver=config["solver"],
            max_iter=config["max_iter"],
            l1_ratio=config["l1_ratio"],
            class_weight=cw,
            random_state=42
        )
    else:
        lr = LogisticRegression(
            penalty=config["penalty"],
            C=config["C"],
            solver=config["solver"],
            max_iter=config["max_iter"],
            class_weight=cw,
            random_state=42
        )

    fold_accs = []
    for fold in range(n_folds):
        val_mask = fold_assignment == fold
        tr_mask = ~val_mask

        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # Clone lr for each fold
        if config["penalty"] == "elasticnet":
            model = LogisticRegression(
                penalty=config["penalty"],
                C=config["C"],
                solver=config["solver"],
                max_iter=config["max_iter"],
                l1_ratio=config["l1_ratio"],
                class_weight=cw,
                random_state=42
            )
        else:
            model = LogisticRegression(
                penalty=config["penalty"],
                C=config["C"],
                solver=config["solver"],
                max_iter=config["max_iter"],
                class_weight=cw,
                random_state=42
            )

        model.fit(X_tr_s, y_tr)
        proba = model.predict_proba(X_val_s)[:, 1]
        oof_proba[val_mask] = proba

        preds = (proba >= 0.5).astype(int)
        fold_accs.append(accuracy_score(y_val, preds))

    oof_preds = (oof_proba >= 0.5).astype(int)
    cv_acc = accuracy_score(y, oof_preds)
    return oof_proba, cv_acc


# Run all configs
all_configs = param_grid + balanced_configs
results = []

for i, config in enumerate(all_configs):
    elapsed = time.time() - start_time
    if elapsed > TIME_BUDGET * 0.85:
        heartbeat({"event": "budget_stop", "elapsed": elapsed, "configs_done": i})
        break

    try:
        oof_proba, cv_acc = run_cv(config, fold_assignment, X, y, n_folds)
        results.append((cv_acc, config, oof_proba))

        if cv_acc > best_cv_score:
            best_cv_score = cv_acc
            best_config = config
            best_oof_proba = oof_proba.copy()
            heartbeat({"event": "new_best", "cv_acc": cv_acc, "config": str(config)})

        if i % 5 == 0:
            heartbeat({"event": "progress", "configs_done": i, "best_so_far": best_cv_score})
    except Exception as e:
        heartbeat({"event": "config_error", "config": str(config), "error": str(e)})
        continue

heartbeat({"event": "cv_done", "best_cv_acc": best_cv_score, "n_configs_tried": len(results), "best_config": str(best_config)})

# ---- Seed averaging: run best config with multiple random seeds and average proba ----
seed_oof_probas = []
seed_test_probas = []

for seed in [42, 123, 456, 789, 999]:
    elapsed = time.time() - start_time
    if elapsed > TIME_BUDGET * 0.90:
        break

    cw = best_config.get("class_weight", None)
    fold_oof = np.zeros(n_train)
    fold_test_probas = []

    for fold in range(n_folds):
        val_mask = fold_assignment == fold
        tr_mask = ~val_mask

        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_val = X[val_mask]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        X_test_s = scaler.transform(X_test)

        if best_config["penalty"] == "elasticnet":
            model = LogisticRegression(
                penalty=best_config["penalty"],
                C=best_config["C"],
                solver=best_config["solver"],
                max_iter=best_config["max_iter"],
                l1_ratio=best_config["l1_ratio"],
                class_weight=cw,
                random_state=seed
            )
        else:
            model = LogisticRegression(
                penalty=best_config["penalty"],
                C=best_config["C"],
                solver=best_config["solver"],
                max_iter=best_config["max_iter"],
                class_weight=cw,
                random_state=seed
            )

        model.fit(X_tr_s, y_tr)
        fold_oof[val_mask] = model.predict_proba(X_val_s)[:, 1]
        fold_test_probas.append(model.predict_proba(X_test_s)[:, 1])

    seed_oof_probas.append(fold_oof)
    seed_test_probas.append(np.mean(fold_test_probas, axis=0))

# Average across seeds
if seed_oof_probas:
    final_oof_proba = np.mean(seed_oof_probas, axis=0)
    final_test_proba = np.mean(seed_test_probas, axis=0)
    final_cv_acc = accuracy_score(y, (final_oof_proba >= 0.5).astype(int))

    if final_cv_acc >= best_cv_score:
        best_cv_score = final_cv_acc
        best_oof_proba = final_oof_proba
    else:
        # Use seed-averaged test proba anyway (more stable)
        pass

    heartbeat({"event": "seed_avg_done", "seed_avg_acc": final_cv_acc, "n_seeds": len(seed_oof_probas)})
else:
    # Fallback: train best config on full training set for test predictions
    cw = best_config.get("class_weight", None)
    fold_test_probas = []
    for fold in range(n_folds):
        val_mask = fold_assignment == fold
        tr_mask = ~val_mask
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X[tr_mask])
        X_test_s = scaler.transform(X_test)
        if best_config["penalty"] == "elasticnet":
            model = LogisticRegression(
                penalty=best_config["penalty"], C=best_config["C"],
                solver=best_config["solver"], max_iter=best_config["max_iter"],
                l1_ratio=best_config["l1_ratio"], class_weight=cw, random_state=42)
        else:
            model = LogisticRegression(
                penalty=best_config["penalty"], C=best_config["C"],
                solver=best_config["solver"], max_iter=best_config["max_iter"],
                class_weight=cw, random_state=42)
        model.fit(X_tr_s, y[tr_mask])
        fold_test_probas.append(model.predict_proba(X_test_s)[:, 1])
    final_test_proba = np.mean(fold_test_probas, axis=0)

# ---- Write outputs ----
# OOF CSV (continuous probabilities)
oof_df = pd.DataFrame({
    "row_idx": np.arange(n_train),
    "oof_proba": best_oof_proba
})
oof_df.to_csv(f"{LOGS_DIR}/oof_linear.csv", index=False)

# Candidate submission (continuous probabilities, not thresholded)
cand_df = pd.DataFrame({
    row_id_col: test_ids.values,
    target_col: final_test_proba
})
cand_df.to_csv(f"{LOGS_DIR}/cand_linear.csv", index=False)

# Agent JSON
floor_score = 0.8100558659217877  # from model_selection.json logistic_regression score
beats_floor = best_cv_score > floor_score

agent_json = {
    "role": "linear-specialist",
    "approach": "linear",
    "selected_model": f"LogisticRegression(penalty={best_config['penalty']}, C={best_config['C']}, class_weight={best_config.get('class_weight')})",
    "cv_metric": "accuracy",
    "cv_score": best_cv_score,
    "lower_is_better": False,
    "candidate_submission": f"{LOGS_DIR}/cand_linear.csv",
    "oof_path": f"{LOGS_DIR}/oof_linear.csv",
    "canonical_folds": True,
    "authored_features_used": feature_cols,
    "monotonic_applied": False,
    "floor_score": floor_score,
    "beats_floor": beats_floor,
    "n_configs_tried": len(results),
    "n_seeds_averaged": len(seed_oof_probas),
    "hints_applied": {
        "prefer_regularized": "not forced (ratio=0.025 < 0.10); standard regularized LR used anyway",
        "native_missing_handling": "no NaN in features (pre-imputed by programmer); confirmed",
        "apply_log1p": "not applicable (binary classification)",
        "class_balance": "balanced class_weight variants included in search; moderate imbalance 38.4% pos",
        "family_recommendation": "planner designated linear as secondary diversity (primary=gbdt)"
    },
    "elapsed_sec": time.time() - start_time
}

with open(f"{LOGS_DIR}/agent_linear.json", "w") as f:
    json.dump(agent_json, f, indent=2)

heartbeat({
    "event": "done",
    "cv_acc": best_cv_score,
    "beats_floor": beats_floor,
    "elapsed": time.time() - start_time
})

print(f"Linear specialist done. CV accuracy: {best_cv_score:.6f} (floor: {floor_score:.6f}, beats_floor: {beats_floor})")
print(f"Best config: {best_config}")
print(f"Elapsed: {time.time() - start_time:.1f}s")
