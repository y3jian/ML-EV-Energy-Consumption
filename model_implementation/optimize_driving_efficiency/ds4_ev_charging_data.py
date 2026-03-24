"""
DS4 — EV charging sessions. Target: **Charging Cost (USD)**.

Features (per spec): Energy Consumed (kWh), Charging Duration, Charging Rate,
State of Charge start/end, Time of Day, Day of Week, Charger Type,
Charging Station Location, User Type.

Uses pre-built splits: `data/processed/4-ev_charging_X_{train,val,test}.csv` and `y_*`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common_metrics import project_root

PREFIX = "4-ev_charging"
TARGET_COL = "Charging Cost (USD)"
TARGET_DESC = "Charging Cost (USD)"

_NUMERIC_FEATURES = [
    "Energy Consumed (kWh)",
    "Charging Duration (hours)",
    "Charging Rate (kW)",
    "State of Charge (Start %)",
    "State of Charge (End %)",
]
_OHE_PREFIXES = (
    "Charging Station Location_",
    "Time of Day_",
    "Day of Week_",
    "Charger Type_",
    "User Type_",
)


def _select_feature_columns(columns: list[str]) -> list[str]:
    selected: list[str] = []
    for c in columns:
        if c in _NUMERIC_FEATURES:
            selected.append(c)
        elif any(c.startswith(p) for p in _OHE_PREFIXES):
            selected.append(c)
    return selected


def _read_y(path: Path) -> pd.Series:
    s = pd.read_csv(path).iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    base = project_root() / "data" / "processed"
    X_train = pd.read_csv(base / f"{PREFIX}_X_train.csv")
    cols = _select_feature_columns(list(X_train.columns))
    if not cols:
        raise ValueError("No DS4 feature columns matched — check processed CSV.")

    X_train = X_train[cols]
    X_val = pd.read_csv(base / f"{PREFIX}_X_val.csv")[cols]
    X_test = pd.read_csv(base / f"{PREFIX}_X_test.csv")[cols]

    y_train = _read_y(base / f"{PREFIX}_y_train.csv")
    y_val = _read_y(base / f"{PREFIX}_y_val.csv")
    y_test = _read_y(base / f"{PREFIX}_y_test.csv")

    return X_train, X_val, X_test, y_train, y_val, y_test
