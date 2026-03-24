"""
DS5 — EV energy consumption while driving. Target: **Energy_Consumption_kWh**.

Features (per spec): Speed_kmh, Acceleration_ms2, Slope_%, Driving_Mode_*,
Traffic_Condition_*, Weather_Condition_*, Tire_Pressure_psi, Vehicle_Weight_kg, Road_Type_*.

Uses pre-built splits: `data/processed/5-EV_energy_consumption_X_{train,val,test}.csv` and `y_*`.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common_metrics import project_root

PREFIX = "5-EV_energy_consumption"
TARGET_DESC = "Energy_Consumption_kWh"

_EXACT_FEATURES = {
    "Speed_kmh",
    "Acceleration_ms2",
    "Slope_%",
    "Tire_Pressure_psi",
    "Vehicle_Weight_kg",
}
_PREFIXES = (
    "Driving_Mode_",
    "Road_Type_",
    "Traffic_Condition_",
    "Weather_Condition_",
)


def _select_feature_columns(columns: list[str]) -> list[str]:
    selected: list[str] = []
    for c in columns:
        if c in _EXACT_FEATURES:
            selected.append(c)
        elif any(c.startswith(p) for p in _PREFIXES):
            selected.append(c)
    return selected


def _read_y(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    col = "Energy_Consumption_kWh" if "Energy_Consumption_kWh" in df.columns else df.columns[0]
    return pd.to_numeric(df[col], errors="coerce")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    base = project_root() / "data" / "processed"
    X_train = pd.read_csv(base / f"{PREFIX}_X_train.csv")
    cols = _select_feature_columns(list(X_train.columns))
    if not cols:
        raise ValueError("No DS5 feature columns matched — check processed CSV.")

    X_train = X_train[cols]
    X_val = pd.read_csv(base / f"{PREFIX}_X_val.csv")[cols]
    X_test = pd.read_csv(base / f"{PREFIX}_X_test.csv")[cols]

    y_train = _read_y(base / f"{PREFIX}_y_train.csv")
    y_val = _read_y(base / f"{PREFIX}_y_val.csv")
    y_test = _read_y(base / f"{PREFIX}_y_test.csv")

    return X_train, X_val, X_test, y_train, y_val, y_test
