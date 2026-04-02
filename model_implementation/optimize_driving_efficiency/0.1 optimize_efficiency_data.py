"""
Load train/val/test for the three optimize-driving-efficiency datasets.

Unlike `predict_ev_range` (uniform `*_X_train.csv` + simple preprocessing), DS2 is built from
`2-ElectricCarData_processed.csv` with string parsing and OHE; DS4/DS5 use processed splits but only
a defined feature subset from the wide X matrices.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


# --- Dataset 2 (EV specs → Efficiency Wh/km) ---

_DS2_TARGET = "Efficiency"
_DS2_FEATURES = [
    "Range",
    "Accel",
    "TopSpeed",
    "PowerTrain",
    "FastCharge",
    "PlugType",
    "Segment",
]
_DS2_NUMERIC = ["Range", "Accel", "TopSpeed", "FastCharge"]
_DS2_REQUIRED = ["Range", "Accel", "TopSpeed"]
_DS2_CATEGORICAL = ["PowerTrain", "PlugType", "Segment"]


def _ds2_first_float(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip()
    if s in ("-", "", "nan", "NaN"):
        return np.nan
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s.replace(",", ""))
    return float(m.group()) if m else np.nan


def load_dataset_2(
    *,
    random_state: int = 42,
    test_size: float = 0.2,
    val_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    path = project_root() / "data" / "processed" / "2-ElectricCarData_processed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    df = pd.read_csv(path)
    for c in _DS2_NUMERIC + [_DS2_TARGET]:
        df[c] = df[c].map(_ds2_first_float)
    for c in _DS2_CATEGORICAL:
        df[c] = df[c].astype(str).str.strip()

    X = df[_DS2_FEATURES].copy()
    y = df[_DS2_TARGET]
    ok = y.notna() & X[_DS2_REQUIRED].notna().all(axis=1)
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

    X_tr = pd.get_dummies(X_train, columns=_DS2_CATEGORICAL, drop_first=True)
    X_va = pd.get_dummies(X_val, columns=_DS2_CATEGORICAL, drop_first=True)
    X_te = pd.get_dummies(X_test, columns=_DS2_CATEGORICAL, drop_first=True)
    X_va = X_va.reindex(columns=X_tr.columns, fill_value=0)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)

    return X_tr, X_va, X_te, y_train, y_val, y_test


# --- Dataset 4 (EV charging → Charging Cost USD) ---

_DS4_PREFIX = "4-ev_charging"
_DS4_NUMERIC = [
    "Energy Consumed (kWh)",
    "Charging Duration (hours)",
    "Charging Rate (kW)",
    "State of Charge (Start %)",
    "State of Charge (End %)",
]
_DS4_OHE_PREFIXES = (
    "Charging Station Location_",
    "Time of Day_",
    "Day of Week_",
    "Charger Type_",
    "User Type_",
)


def _ds4_columns(columns: list[str]) -> list[str]:
    out: list[str] = []
    for c in columns:
        if c in _DS4_NUMERIC:
            out.append(c)
        elif any(c.startswith(p) for p in _DS4_OHE_PREFIXES):
            out.append(c)
    return out


def _read_y_first_col(path: Path) -> pd.Series:
    return pd.to_numeric(pd.read_csv(path).iloc[:, 0], errors="coerce")


def load_dataset_4() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    base = project_root() / "data" / "processed"
    X_train = pd.read_csv(base / f"{_DS4_PREFIX}_X_train.csv")
    cols = _ds4_columns(list(X_train.columns))
    if not cols:
        raise ValueError("DS4: no feature columns matched.")

    X_train = X_train[cols]
    X_val = pd.read_csv(base / f"{_DS4_PREFIX}_X_val.csv")[cols]
    X_test = pd.read_csv(base / f"{_DS4_PREFIX}_X_test.csv")[cols]

    y_train = _read_y_first_col(base / f"{_DS4_PREFIX}_y_train.csv")
    y_val = _read_y_first_col(base / f"{_DS4_PREFIX}_y_val.csv")
    y_test = _read_y_first_col(base / f"{_DS4_PREFIX}_y_test.csv")

    return X_train, X_val, X_test, y_train, y_val, y_test


# --- Dataset 5 (EV energy → Energy_Consumption_kWh) ---

_DS5_PREFIX = "5-EV_energy_consumption"
_DS5_EXACT = {
    "Speed_kmh",
    "Acceleration_ms2",
    "Slope_%",
    "Tire_Pressure_psi",
    "Vehicle_Weight_kg",
}
_DS5_PREFIXES = (
    "Driving_Mode_",
    "Road_Type_",
    "Traffic_Condition_",
    "Weather_Condition_",
)


def _ds5_columns(columns: list[str]) -> list[str]:
    out: list[str] = []
    for c in columns:
        if c in _DS5_EXACT:
            out.append(c)
        elif any(c.startswith(p) for p in _DS5_PREFIXES):
            out.append(c)
    return out


def _read_y_ds5(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    col = "Energy_Consumption_kWh" if "Energy_Consumption_kWh" in df.columns else df.columns[0]
    return pd.to_numeric(df[col], errors="coerce")


def load_dataset_5() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    base = project_root() / "data" / "processed"
    X_train = pd.read_csv(base / f"{_DS5_PREFIX}_X_train.csv")
    cols = _ds5_columns(list(X_train.columns))
    if not cols:
        raise ValueError("DS5: no feature columns matched.")

    X_train = X_train[cols]
    X_val = pd.read_csv(base / f"{_DS5_PREFIX}_X_val.csv")[cols]
    X_test = pd.read_csv(base / f"{_DS5_PREFIX}_X_test.csv")[cols]

    y_train = _read_y_ds5(base / f"{_DS5_PREFIX}_y_train.csv")
    y_val = _read_y_ds5(base / f"{_DS5_PREFIX}_y_val.csv")
    y_test = _read_y_ds5(base / f"{_DS5_PREFIX}_y_test.csv")

    return X_train, X_val, X_test, y_train, y_val, y_test
