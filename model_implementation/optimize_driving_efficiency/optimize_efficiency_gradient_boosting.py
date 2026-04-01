import re
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(_PKG))

from optimize_efficiency_data import load_dataset_2, load_dataset_4, load_dataset_5
from optimize_efficiency_fit_diagnostics import diagnose_regression_fit


def safe_mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-8
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def regression_errors(y_true, y_pred) -> tuple[float, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, r2


def slugify(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.lower())).strip("_")


def plot_actual_vs_pred(y_true, y_pred, title: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="none")
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi], "--", linewidth=1.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_residuals(y_true, y_pred, title: str, out_path: Path) -> None:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=30, alpha=0.8)
    plt.axvline(0.0, linestyle="--", linewidth=1.2)
    plt.xlabel("Residual (Actual - Predicted)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_bar(df: pd.DataFrame, y_col: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.bar(df["Dataset"], df[y_col])
    plt.ylabel(y_col)
    plt.title(title)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


datasets = {
    "Dataset 2 (EV specs -> Efficiency)": load_dataset_2,
    "Dataset 4 (EV charging -> Charging Cost)": load_dataset_4,
    "Dataset 5 (EV energy -> kWh)": load_dataset_5,
}

results = []
output_dir = _PKG / "results" / "gradient_boosting"
output_dir.mkdir(parents=True, exist_ok=True)

print(
    "Regression metrics: train / validation / test RMSE & R2 for fit checks; "
    "MAE & MAPE on test only."
)

for name, loader in datasets.items():
    print(f"\nProcessing {name}...")

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = loader()

        model = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
        )

        t0 = perf_counter()
        model.fit(X_train, y_train)
        train_s = perf_counter() - t0

        t1 = perf_counter()
        y_pred_tr = model.predict(X_train)
        y_pred_va = model.predict(X_val)
        y_pred = model.predict(X_test)
        pred_s = perf_counter() - t1
        total_s = train_s + pred_s

        tr_rmse, tr_r2 = regression_errors(y_train, y_pred_tr)
        va_rmse, va_r2 = regression_errors(y_val, y_pred_va)
        te_rmse, te_r2 = regression_errors(y_test, y_pred)
        mae = float(mean_absolute_error(y_test, y_pred))
        mape = safe_mape(y_test, y_pred)
        fit_note = diagnose_regression_fit(
            train_rmse=tr_rmse,
            val_rmse=va_rmse,
            test_rmse=te_rmse,
            train_r2=tr_r2,
            val_r2=va_r2,
            test_r2=te_r2,
        )

        results.append(
            {
                "Dataset": name,
                "Train_RMSE": tr_rmse,
                "Val_RMSE": va_rmse,
                "Test_RMSE": te_rmse,
                "Train_R2": tr_r2,
                "Val_R2": va_r2,
                "Test_R2": te_r2,
                "Test_MAE": mae,
                "Test_MAPE_pct": mape,
                "Fit_diagnosis": fit_note,
                "Train_s": train_s,
                "Predict_s": pred_s,
                "Total_s": total_s,
            }
        )

        ds = slugify(name)
        plot_actual_vs_pred(
            y_test,
            y_pred,
            f"Gradient Boosting - {name} (Actual vs Predicted)",
            output_dir / f"{ds}_actual_vs_pred.png",
        )
        plot_residuals(
            y_test,
            y_pred,
            f"Gradient Boosting - {name} (Residuals)",
            output_dir / f"{ds}_residuals.png",
        )

        print(f"{name} DONE")
        print(
            f"RMSE  train/val/test: {tr_rmse:.4f} / {va_rmse:.4f} / {te_rmse:.4f} | "
            f"R2 train/val/test: {tr_r2:.4f} / {va_r2:.4f} / {te_r2:.4f}"
        )
        print(f"Test MAE: {mae:.4f} | Test MAPE: {mape:.2f}%")
        print(f"Fit read: {fit_note}")
        print(
            f"Runtime (train/predict/total): {train_s:.4f}s / {pred_s:.4f}s / {total_s:.4f}s"
        )

    except Exception as e:
        print(f"Error in {name}: {e}")

results_df = pd.DataFrame(results)

print("\n=== Gradient Boosting (optimize driving efficiency) ===")
print(results_df)

out_path = Path("optimize_efficiency_gb_results_comparison.csv")
results_df.to_csv(out_path, index=False)
print(f"\nResults saved to {out_path}")

if not results_df.empty:
    plot_bar(
        results_df,
        "Test_RMSE",
        "Gradient Boosting - test RMSE by dataset",
        output_dir / "comparison_rmse.png",
    )
    plot_bar(
        results_df,
        "Test_R2",
        "Gradient Boosting - test R2 by dataset",
        output_dir / "comparison_r2.png",
    )
    plot_bar(
        results_df,
        "Total_s",
        "Gradient Boosting - total runtime by dataset",
        output_dir / "comparison_runtime.png",
    )
    print(f"Plots saved under: {output_dir}")
