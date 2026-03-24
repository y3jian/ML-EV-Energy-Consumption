"""
DS2 — EV car specs. Target: **Efficiency** (Wh/km).
Features: Range, Accel, TopSpeed, PowerTrain, FastCharge, PlugType, Segment.

Source: `data/processed/2-ElectricCarData_processed.csv`. Splits 60/20/20, `random_state=42`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common_metrics import project_root

TARGET_COL = "Efficiency"
FEATURE_COLS = [
    "Range",
    "Accel",
    "TopSpeed",
    "PowerTrain",
    "FastCharge",
    "PlugType",
    "Segment",
]
_NUMERIC = ["Range", "Accel", "TopSpeed", "FastCharge"]
_REQUIRED_NUMERIC = ["Range", "Accel", "TopSpeed"]
_CATEGORICAL = ["PowerTrain", "PlugType", "Segment"]
_PROCESSED_CSV = "2-ElectricCarData_processed.csv"
TARGET_DESC = "Efficiency (Wh/km)"


def _first_float(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip()
    if s in ("-", "", "nan", "NaN"):
        return np.nan
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s.replace(",", ""))
    return float(m.group()) if m else np.nan


def load_parsed_frame(csv_path: Path | None = None) -> pd.DataFrame:
    path = csv_path or (project_root() / "data" / "processed" / _PROCESSED_CSV)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    for c in _NUMERIC + [TARGET_COL]:
        df[c] = df[c].map(_first_float)
    for c in _CATEGORICAL:
        df[c] = df[c].astype(str).str.strip()
    return df


def load_splits(
    *,
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    df = load_parsed_frame()
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL]

    ok = y.notna() & X[_REQUIRED_NUMERIC].notna().all(axis=1)
    X = X.loc[ok].reset_index(drop=True)
    y = y.loc[ok].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    fc_med = X_train["FastCharge"].median()
    for part in (X_train, X_val, X_test):
        part["FastCharge"] = part["FastCharge"].fillna(fc_med)

    X_tr = pd.get_dummies(X_train, columns=_CATEGORICAL, drop_first=True)
    X_va = pd.get_dummies(X_val, columns=_CATEGORICAL, drop_first=True)
    X_te = pd.get_dummies(X_test, columns=_CATEGORICAL, drop_first=True)

    X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    return X_tr, X_va, X_te, y_train, y_val, y_test
