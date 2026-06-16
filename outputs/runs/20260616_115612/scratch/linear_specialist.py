"""
Linear specialist for run 20260616_115612.
Binary classification (Titanic survival), metric=accuracy.
Family: linear — LogisticRegression with L1/L2/ElasticNet regularization + seed averaging.
"""
import json
import time
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

RUN_ID = "20260616_115612"
BASE = "/Users/yuqic/Documents/Claude/Projects/STAI-X challenge/award-b-repo"
LOGS = f"{BASE}/outputs/runs/{RUN_ID}/logs"
HEARTBEAT_PATH = f"{LOGS}/linear-specialist_progress.jsonl"
TIME_BUDGET = int(os.environ.get("AWARDB_TIME_BUDGET_SEC", "900"))

t0 = time.time()

def heartbeat(msg: dict):
    msg["ts"] = time.time() - t0
    with open(HEARTBEAT_PATH, "a") as f:
        f.write(json.dumps(msg) + "\n")

heartbeat({"event": "start", "family": "linear", "budget_sec": TIME_BUDGET})

# Load spec
with open(f"{LOGS}/spec_parse.json") as f:
    spec = json.load(f)
target_col   = spec["target_column"]
row_id_col   = spec["row_id_column"]
train_file   = f"{BASE}/{spec['train_file']}"
pred_file    = f"{BASE}/{spec['prediction_file']}"

# Load feature parquets
train_feats = pd.read_parquet(f"{LOGS}/features_train.parquet")
pred_feats  = pd.read_parquet(f"{LOGS}/features_pred.parquet")

with open(f"{LOGS}/feature_spec.json") as f:
    fspec = json.load(f)
feature_cols = fspec["feature_columns"]

# Load target from raw train CSV
train_raw = pd.read_csv(train_file)
y_train = train_raw[target_col].values.astype(int)
passenger_ids_train = train_raw[row_id_col].values

# Load test PassengerIds
test_raw = pd.read_csv(pred_file)
passenger_ids_test = test_raw[row_id_col].values

# Load canonical folds
with open(f"{LOGS}/cv_folds.json") as f:
    cv_folds_data = json.load(f)
fold_assignment = np.array(cv_folds_data["fold_assignment"])
n_folds = cv_folds_data["n_folds"]

X_train = train_feats[feature_cols].copy()
X_test  = pred_feats[feature_cols].copy()

# Fill any remaining NaNs (fallback safety)
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_test  = X_test.fillna(train_medians)

heartbeat({"event": "data_loaded", "n_train": len(X_train), "n_test": len(X_test), "n_features": len(feature_cols)})

# --- Hyperparameter grid ---
param_grid = [
    # L2 (Ridge) — various C
    {"solver": "lbfgs",  "penalty": "l2",          "C": 0.01,  "max_iter": 2000},
    {"solver": "lbfgs",  "penalty": "l2",          "C": 0.1,   "max_iter": 2000},
    {"solver": "lbfgs",  "penalty": "l2",          "C": 1.0,   "max_iter": 2000},
    {"solver": "lbfgs",  "penalty": "l2",          "C": 5.0,   "max_iter": 2000},
    {"solver": "lbfgs",  "penalty": "l2",          "C": 10.0,  "max_iter": 2000},
    {"solver": "lbfgs",  "penalty": "l2",          "C": 50.0,  "max_iter": 2000},
    # L1 (Lasso)
    {"solver": "saga",   "penalty": "l1",          "C": 0.1,   "max_iter": 3000},
    {"solver": "saga",   "penalty": "l1",          "C": 1.0,   "max_iter": 3000},
    {"solver": "saga",   "penalty": "l1",          "C": 5.0,   "max_iter": 3000},
    {"solver": "saga",   "penalty": "l1",          "C": 10.0,  "max_iter": 3000},
    # ElasticNet
    {"solver": "saga",   "penalty": "elasticnet",  "C": 0.1,   "l1_ratio": 0.5, "max_iter": 3000},
    {"solver": "saga",   "penalty": "elasticnet",  "C": 1.0,   "l1_ratio": 0.5, "max_iter": 3000},
    {"solver": "saga",   "penalty": "elasticnet",  "C": 5.0,   "l1_ratio": 0.5, "max_iter": 3000},
    {"solver": "saga",   "penalty": "elasticnet",  "C": 1.0,   "l1_ratio": 0.2, "max_iter": 3000},
    {"solver": "saga",   "penalty": "elasticnet",  "C": 1.0,   "l1_ratio": 0.8, "max_iter": 3000},
]

seeds = [42, 7, 123, 17, 99]

best_score       = -np.inf
best_params      = None
best_oof_proba   = None
best_test_proba  = None
results          = []

for pi, params in enumerate(param_grid):
    if time.time() - t0 > TIME_BUDGET * 0.80:
        heartbeat({"event": "budget_reached", "params_tried": pi})
        break

    sum_oof_proba  = np.zeros(len(X_train))
    sum_test_proba = np.zeros(len(X_test))

    for seed in seeds:
        oof_proba = np.zeros(len(X_train))

        for fold_id in range(n_folds):
            tr_idx = np.where(fold_assignment != fold_id)[0]
            va_idx = np.where(fold_assignment == fold_id)[0]

            X_tr, y_tr = X_train.iloc[tr_idx].values, y_train[tr_idx]
            X_va       = X_train.iloc[va_idx].values

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)

            kw = dict(params)
            kw["random_state"] = seed
            clf = LogisticRegression(**kw)
            clf.fit(X_tr_s, y_tr)
            oof_proba[va_idx] = clf.predict_proba(X_va_s)[:, 1]

        sum_oof_proba += oof_proba

        # Refit on full train for test preds
        scaler_full = StandardScaler()
        X_all_s = scaler_full.fit_transform(X_train.values)
        X_tst_s = scaler_full.transform(X_test.values)
        kw2 = dict(params)
        kw2["random_state"] = seed
        clf_full = LogisticRegression(**kw2)
        clf_full.fit(X_all_s, y_train)
        sum_test_proba += clf_full.predict_proba(X_tst_s)[:, 1]

    avg_oof   = sum_oof_proba  / len(seeds)
    avg_tests = sum_test_proba / len(seeds)

    oof_preds = (avg_oof >= 0.5).astype(int)
    score = accuracy_score(y_train, oof_preds)
    results.append({"params": params, "accuracy": score})

    heartbeat({"event": "param_eval", "param_idx": pi, "params": params, "accuracy": round(score, 5)})

    if score > best_score:
        best_score       = score
        best_params      = params
        best_oof_proba   = avg_oof.copy()
        best_test_proba  = avg_tests.copy()

heartbeat({"event": "search_done", "best_accuracy": round(best_score, 6), "best_params": best_params})

# --- Write OOF CSV (continuous probability) ---
oof_df = pd.DataFrame({
    row_id_col:  passenger_ids_train,
    "oof_proba": best_oof_proba
})
oof_df.to_csv(f"{LOGS}/oof_linear.csv", index=False)

# --- Write candidate submission CSV (continuous probability for NNLS blend) ---
cand_df = pd.DataFrame({
    row_id_col:    passenger_ids_test,
    target_col:    best_test_proba   # continuous proba; ensemble-meta thresholds
})
cand_df.to_csv(f"{LOGS}/cand_linear.csv", index=False)

# --- Write agent JSON ---
agent_out = {
    "role": "linear-specialist",
    "approach": "linear",
    "selected_model": (
        f"LogisticRegression(penalty={best_params['penalty']}, "
        f"C={best_params['C']}, solver={best_params['solver']})"
    ),
    "cv_metric": "accuracy",
    "cv_score": round(best_score, 6),
    "lower_is_better": False,
    "candidate_submission": f"{LOGS}/cand_linear.csv",
    "oof_path": f"{LOGS}/oof_linear.csv",
    "canonical_folds": True,
    "authored_features_used": feature_cols,
    "monotonic_applied": False,
    "best_params": best_params,
    "seeds_averaged": seeds,
    "n_configs_tried": len(results),
    "hint_alignment": {
        "model_family_recommendation": "secondary (linear) — planner designated this family as secondary diversity",
        "prefer_regularized": "false — standard regularized linear already appropriate; no parameter change needed",
        "native_missing_handling_preferred": "confirmed — features already imputed in parquet by analysis-programmer",
        "apply_log1p_hint": "false — binary classification, no log transform applicable"
    }
}

with open(f"{LOGS}/agent_linear.json", "w") as f:
    json.dump(agent_out, f, indent=2)

heartbeat({
    "event": "done",
    "cv_score": round(best_score, 6),
    "outputs_written": ["oof_linear.csv", "cand_linear.csv", "agent_linear.json"]
})

print(f"Linear specialist done. CV accuracy={best_score:.6f}. Best params: {best_params}")
