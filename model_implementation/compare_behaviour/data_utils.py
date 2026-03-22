"""
Compare driving behaviors — pooled **processed** features from datasets **4** and **5**.

Reads pre-built train/val/test splits (same 60/20/20 logic as the prep notebooks):

- `data/processed/4-ev_charging_patterns_X_{train,val,test}.csv` + `y_*`
- `data/processed/5-EV_energy_consumption_X_{train,val,test}.csv` + `y_*`

Each split stacks dataset-5 rows above dataset-4 rows. Feature columns are the **union** of both
`X` frames; missing columns for a source are filled with **0** (that source never had that feature).

Targets are unified under the name **Energy_Consumption_kWh** (dataset 4’s CSV column is still
`Energy Consumed (kWh)` — values are the same physical quantity).

Run `4-ev_charging_patterns_data_prep.ipynb` and `5-EV_energy_consumption_data_prep.ipynb` first
if these CSVs are missing.

*Emission Level* (dataset 1) is not used here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Unified target name for metrics / reporting
TARGET_COL = "Energy_Consumption_kWh"
Y_COL_DATASET_5 = "Energy_Consumption_kWh"
Y_COL_DATASET_4 = "Energy Consumed (kWh)"

PREFIX_4 = "4-ev_charging_patterns"
PREFIX_5 = "5-EV_energy_consumption"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_y(path: Path, preferred_col: str) -> pd.Series:
    df = pd.read_csv(path)
    if preferred_col in df.columns:
        s = df[preferred_col]
    else:
        s = df.iloc[:, 0]
    return pd.to_numeric(s, errors="coerce")


def _align_and_stack(
    X_a: pd.DataFrame,
    y_a: pd.Series,
    X_b: pd.DataFrame,
    y_b: pd.Series,
    all_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    X_a = X_a.reindex(columns=all_columns, fill_value=0)
    X_b = X_b.reindex(columns=all_columns, fill_value=0)
    X = pd.concat([X_a, X_b], axis=0, ignore_index=True)
    y = pd.concat(
        [y_a.reset_index(drop=True), y_b.reset_index(drop=True)],
        axis=0,
        ignore_index=True,
    )
    y.name = TARGET_COL
    valid = y.notna() & np.isfinite(y)
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)


def load_compare_behaviour_splits(
    processed_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Returns X_train, X_val, X_test, y_train, y_val, y_test from **processed** CSVs.

    Splits are **not** re-shuffled: each split pools the corresponding files from
    dataset 5 then dataset 4 (same random_state=42 splits as in each prep notebook).
    """
    base = processed_dir or (project_root() / "data" / "processed")

    def paths(split: str) -> tuple[Path, Path, Path, Path]:
        return (
            base / f"{PREFIX_5}_X_{split}.csv",
            base / f"{PREFIX_5}_y_{split}.csv",
            base / f"{PREFIX_4}_X_{split}.csv",
            base / f"{PREFIX_4}_y_{split}.csv",
        )

    for split in ("train", "val", "test"):
        p5x, p5y, p4x, p4y = paths(split)
        for p in (p5x, p5y, p4x, p4y):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing processed file: {p}\n"
                    "Run 4-ev_charging_patterns_data_prep.ipynb and "
                    "5-EV_energy_consumption_data_prep.ipynb (export cells)."
                )

    # Union of feature names (sorted for stable column order)
    X5_tr = pd.read_csv(base / f"{PREFIX_5}_X_train.csv")
    X4_tr = pd.read_csv(base / f"{PREFIX_4}_X_train.csv")
    all_columns = sorted(set(X5_tr.columns) | set(X4_tr.columns))

    outs: list[tuple[pd.DataFrame, pd.Series]] = []
    for split in ("train", "val", "test"):
        p5x, p5y, p4x, p4y = paths(split)
        X5 = pd.read_csv(p5x)
        X4 = pd.read_csv(p4x)
        y5 = _read_y(p5y, Y_COL_DATASET_5)
        y4 = _read_y(p4y, Y_COL_DATASET_4)
        outs.append(_align_and_stack(X5, y5, X4, y4, all_columns))

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = outs
    return X_train, X_val, X_test, y_train, y_val, y_test


def print_regression_metrics(
    model_name: str,
    y_true,
    y_pred,
    *,
    split_name: str = "Test",
) -> None:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{'=' * 60}")
    print(f"{model_name} — {split_name} set (target: {TARGET_COL})")
    print(f"{'=' * 60}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R²:   {r2:.6f}")
