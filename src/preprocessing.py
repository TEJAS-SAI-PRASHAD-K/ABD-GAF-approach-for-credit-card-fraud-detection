import random
from typing import Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
TARGET_COLUMN = "Class"


def set_global_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _sort_split_by_time(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    ordered_idx = X.sort_values("Time").index
    X_sorted = X.loc[ordered_idx].reset_index(drop=True)
    y_sorted = y.loc[ordered_idx].reset_index(drop=True)
    return X_sorted, y_sorted


def load_and_preprocess(path: str = "data/creditcard.csv", return_raw: bool = False):
    """
    Load and preprocess the Kaggle credit card fraud dataset.

    Workflow:
    1) Load CSV, validate schema and nulls.
    2) Sort by Time to preserve transaction chronology.
    3) Stratified split into train/val/test = 70/15/15.
    4) Scale only Time and Amount (fit on train only).
    5) Apply SMOTE on train only.

    Returns:
        X_train_smote, X_val, X_test, y_train_smote, y_val, y_test, scaler

    If return_raw=True, also returns:
        X_train_raw, y_train_raw
    """
    set_global_seed(RANDOM_STATE)

    df = pd.read_csv(path)

    if df.isnull().sum().sum() != 0:
        raise AssertionError("Dataset contains null values, which violates dataset constraints.")

    required_cols = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = required_cols.difference(df.columns)
    if missing:
        raise AssertionError(f"Missing required columns: {sorted(missing)}")

    df = df.sort_values("Time").reset_index(drop=True)

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    X_train_raw, X_temp, y_train_raw, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    X_train_raw, y_train_raw = _sort_split_by_time(X_train_raw, y_train_raw)
    X_val_raw, y_val = _sort_split_by_time(X_val_raw, y_val)
    X_test_raw, y_test = _sort_split_by_time(X_test_raw, y_test)

    scaler = StandardScaler()
    scaler.fit(X_train_raw[["Time", "Amount"]])

    X_train_scaled = X_train_raw.copy()
    X_val = X_val_raw.copy()
    X_test = X_test_raw.copy()

    X_train_scaled[["Time", "Amount"]] = scaler.transform(X_train_scaled[["Time", "Amount"]])
    X_val[["Time", "Amount"]] = scaler.transform(X_val[["Time", "Amount"]])
    X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote_arr, y_train_smote_arr = smote.fit_resample(X_train_scaled, y_train_raw)

    X_train_smote = pd.DataFrame(X_train_smote_arr, columns=FEATURE_COLUMNS)
    y_train_smote = pd.Series(y_train_smote_arr, name=TARGET_COLUMN).astype(int)

    # Keep deterministic order after SMOTE for reproducibility.
    X_train_smote = X_train_smote.reset_index(drop=True)
    y_train_smote = y_train_smote.reset_index(drop=True)

    if return_raw:
        return (
            X_train_smote,
            X_val.reset_index(drop=True),
            X_test.reset_index(drop=True),
            y_train_smote,
            y_val.reset_index(drop=True),
            y_test.reset_index(drop=True),
            scaler,
            X_train_scaled.reset_index(drop=True),
            y_train_raw.reset_index(drop=True),
        )

    return (
        X_train_smote,
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train_smote,
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True),
        scaler,
    )
