"""feature-engineer skill: preprocessing pipeline + train/val/test split.

The pipeline is built by the proven ``models._make_preprocessor`` (median impute +
StandardScaler for numeric; most-frequent impute + OneHotEncoder(handle_unknown=
"ignore") for categorical). Returns an UNFITTED transformer — the caller fits on the
training split only (no leakage).
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from .. import models as _models
from ..task import BINARY, MULTICLASS


def build_preprocessing_pipeline(
    df: pd.DataFrame | None,
    numeric_cols: list[str],
    categorical_cols: list[str],
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    return _models._make_preprocessor(list(numeric_cols), list(categorical_cols), scale_numeric)


def split_dataset(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    feature_cols: list[str],
    *,
    test_size: float = 0.2,
    val_size: float = 0.0,
    random_state: int = 42,
) -> tuple[SimpleNamespace, dict]:
    work = df.dropna(subset=[target_col]).reset_index(drop=True)
    feats = [c for c in feature_cols if c in work.columns]
    X, y = work[feats], work[target_col]
    stratify = y if task_type in (BINARY, MULTICLASS) else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    X_val = y_val = None
    if val_size and val_size > 0:
        rel = val_size / (1.0 - test_size)
        strat2 = y_train if stratify is not None else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=rel, random_state=random_state, stratify=strat2
        )

    splits = SimpleNamespace(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        random_seed=random_state,
    )
    metadata = {
        "n_train": int(len(X_train)),
        "n_val": int(0 if X_val is None else len(X_val)),
        "n_test": int(len(X_test)),
        "test_size": test_size,
        "val_size": val_size,
        "random_state": random_state,
        "target_col": target_col,
        "stratified": stratify is not None,
        "feature_cols": feats,
    }
    return splits, metadata
