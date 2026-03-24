"""Shared regression metrics and repo root helper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def print_metrics(
    model_name: str,
    y_true,
    y_pred,
    *,
    split_name: str,
    target_desc: str,
) -> None:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{'=' * 60}")
    print(f"{model_name} — {split_name} (target: {target_desc})")
    print(f"{'=' * 60}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
