# ML-EV-Energy-Consumption

This project applies machine learning to the **electric vehicle (EV)** space, focusing on **real-world energy consumption**, **range estimation**, and **factors that limit or improve efficiency**.

## Project overview

Despite rapid EV adoption, energy use in practice remains hard to predict. It depends on **vehicle design**, **charging context**, **environment**, and **driving behavior**. Many published estimates assume near-ideal conditions that do not match day-to-day use—making planning harder for policymakers and drivers.

We organize the work into **three modeling use cases** across **six prepared datasets (DS1–DS6)**:

1. **Predict EV range** — Regress a range-related target from tabular features (**DS3, DS4, DS5** in the primary scripts).
2. **Performance bottlenecks** — Regression and classification to surface **which variables matter** for energy/emissions-related outcomes on **DS1, DS4, DS5, DS6** (multiple algorithms + summarized metrics).
3. **Optimize driving efficiency** — Regress **efficiency**, **charging cost**, and **trip energy (kWh)** using defined feature sets for **DS2, DS4, DS5**, with train/validation/test metrics, diagnostics, and comparison dashboards.

## Technologies

- Python, Jupyter / JupyterLab  
- **pandas**, NumPy, **scikit-learn**, matplotlib, seaborn  
- **TensorFlow / Keras** (LSTM for predict-EV-range)  
- **imbalanced-learn** (performance-bottleneck pipelines where used)

Install dependencies (pinned versions in the repo):

```bash
pip install -r requirements.txt
```

## Reproducibility

Clone and enter the repository:

```bash
git clone https://github.com/y3jian/ML-EV-Energy-Consumption.git
cd ML-EV-Energy-Consumption
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

Then install from `requirements.txt` as above.

---

## 1. Prepare datasets

Processed train/validation/test CSVs live under `data/processed/`. To rebuild them, open **`EDA/`** and run the notebooks (order below). Start Jupyter from the **repository root** so relative paths resolve.

| Notebook | Role |
|----------|------|
| `EDA/1-vehicle_emission_data_prep.ipynb` | DS1 — vehicle emissions |
| `EDA/3-EV_population_data_prep.ipynb` | DS3 — EV population |
| `EDA/4-ev_charging_patterns_data_prep.ipynb` | DS4 — charging patterns |
| `EDA/5-EV_energy_consumption_data_prep.ipynb` | DS5 — energy consumption |
| `EDA/6-world_gdp_population_co2_emissions_data_prep.ipynb` | DS6 — GDP / CO2 / population |

> **Note:** **DS2** (electric car catalog / efficiency) is consumed as `data/processed/2-ElectricCarData_processed.csv` in the optimize-driving-efficiency pipeline. If that file is missing, add or rebuild it to match what `model_implementation/optimize_driving_efficiency/0.1 optimize_efficiency_data.py` expects.

---

## 2. Run model experiments

Run Python scripts from the **repository root** (`ML-EV-Energy-Consumption`) unless noted, so paths like `data/processed/` and default CSV output names line up.

### Use case A — Predict EV range

**Goal:** Compare **Random Forest**, **Gradient Boosting**, and **LSTM** on **DS3, DS4, DS5**.

| Script | Output CSV (repo root) |
|--------|-------------------------|
| `model_implementation/predict_ev_range/ev_range_random_forest.py` | `ev_range_rf_results_comparison.csv` |
| `model_implementation/predict_ev_range/ev_range_gradient_boost.py` | `ev_range_gb_results_comparison.csv` |
| `model_implementation/predict_ev_range/ev_range_LSTM.py` | `ev_range_lstm_results_comparison.csv` |

**Additional models** in the same folder (optional): `ev_range_KNN.py`, `ev_range_lasso.py`, `ev_range_ridge.py`.

**Analysis / plots**

- `ev_range_comparison_plots.py` — bar charts for RMSE / R² across GB, LSTM, RF (reads the three CSVs above from the current working directory).
- Optional PNGs at repo root: `rmse_comparison.png`, `r2_comparison.png` if you generate them from your workflow.

---

### Use case B — Optimize driving efficiency

**Goal:** **Ridge**, **Random Forest**, and **Gradient Boosting** on:

- **DS2** — EV specs → **efficiency (Wh/km)**  
- **DS4** — charging features → **charging cost (USD)**  
- **DS5** — driving/energy features → **energy (kWh)**  

Train/validation/test splits, **MAE / MAPE** on test, **fit diagnostics** (`0.2 optimize_efficiency_fit_diagnostics.py`).

| Module / script | Purpose |
|-----------------|--------|
| `0.1 optimize_efficiency_data.py` | Loaders for DS2, DS4, DS5 |
| `0.2 optimize_efficiency_fit_diagnostics.py` | `diagnose_regression_fit(...)` text from split metrics |
| `1.1 optimize_efficiency_gradient_boosting.py` | GB across datasets |
| `1.2 optimize_efficiency_random_forest.py` | RF across datasets |
| `1.3 optimize_efficiency_ridge.py` | Ridge across datasets |
| `2 optimize_driving_efficiency_dataset_dashboards.ipynb` | Per-dataset and cross-dataset diagnostic dashboards (reads `optimize_efficiency_*_results_comparison.csv` from **repo root**) |
| `optimize_driving_efficiency_dataset_dashboards.ipynb` | Alternate copy of the same dashboard notebook (use **`2 ...`** if both exist) |

**Run (from repo root):**

```bash
python "model_implementation/optimize_driving_efficiency/1.1 optimize_efficiency_gradient_boosting.py"
python "model_implementation/optimize_driving_efficiency/1.2 optimize_efficiency_random_forest.py"
python "model_implementation/optimize_driving_efficiency/1.3 optimize_efficiency_ridge.py"
```

**Outputs**

- **CSV (default: current working directory):**  
  `optimize_efficiency_rf_results_comparison.csv`,  
  `optimize_efficiency_gb_results_comparison.csv`,  
  `optimize_efficiency_ridge_results_comparison.csv`  
- **Plots per algorithm:** under `model_implementation/optimize_driving_efficiency/results/<ridge|random_forest|gradient_boosting>/` (e.g. actual vs predicted, residuals, bar comparisons).  
- **Aggregated analysis (when generated):**  
  - `model_implementation/optimize_driving_efficiency/results/fit_patterns/` — e.g. `fit_pattern_evidence_table.csv`  
  - `model_implementation/optimize_driving_efficiency/results/model_comparison/` — merged tables, pivots, best-by-metric summaries  
  - `model_implementation/optimize_driving_efficiency/results/dashboards/` — static diagnostic dashboard PNGs  

The dashboard notebook includes **written conclusions** for DS2, DS4, DS5, and cross-dataset takeaways.

> **Matplotlib headless runs:** training scripts use a non-interactive backend; if font-cache warnings appear, you can set `MPLCONFIGDIR` to a writable directory.

---

### Use case C — Performance bottlenecks

**Goal:** Model **DS5** (regression), **DS1** (classification), **DS4** (regression), **DS6** (regression) with **KNN**, **Lasso**, **Gradient Boosting**, and **Random Forest**. Splits are loaded from `data/processed/` using paths resolved from each script’s location (safe to run from repo root).

| Script | Algorithm |
|--------|-----------|
| `model_implementation/performance_bottlenecks/0 performance_bottleneck_knn.py` | **Stub only** (marked “excluded from final implementation”; placeholder CSV/columns—not wired to `data/processed/`) |
| `model_implementation/performance_bottlenecks/1.1 performance_bottlenecks_gradient_boosting.py` | Gradient boosting |
| `model_implementation/performance_bottlenecks/1.2 performance_bottlenecks_lasso.py` | Lasso / logistic variants |
| `model_implementation/performance_bottlenecks/1.3 performance_bottlenecks_random_forest.py` | Random forest |
| `model_implementation/performance_bottlenecks/2 performance_bottlenecks_plots.py` | Plotting utilities / figures |
| `model_implementation/performance_bottlenecks/3 ANALYSIS performance_bottlenecks_plots.ipynb` | Exploratory / presentation analysis |

**Analysis summary**

- `model_implementation/performance_bottlenecks/results.md` — consolidated CV / test metrics, hyperparameters, and notes for the runs documented there.

---

## Repository structure (high level)

```text
ML-EV-Energy-Consumption/
├── data/
│   ├── processed/          # *_X_train.csv, *_y_test.csv, etc.
│   └── ...
├── EDA/                    # Data-prep notebooks (DS1,3,4,5,6)
├── model_implementation/
│   ├── predict_ev_range/
│   ├── optimize_driving_efficiency/
│   │   ├── 0.1 optimize_efficiency_data.py
│   │   ├── 0.2 optimize_efficiency_fit_diagnostics.py
│   │   ├── 1.1 optimize_efficiency_gradient_boosting.py
│   │   ├── 1.2 optimize_efficiency_random_forest.py
│   │   ├── 1.3 optimize_efficiency_ridge.py
│   │   ├── 2 optimize_driving_efficiency_dataset_dashboards.ipynb
│   │   └── results/      # plots, fit_patterns, model_comparison, dashboards
│   └── performance_bottlenecks/
│       ├── 0 performance_bottleneck_knn.py
│       ├── 1.1 performance_bottlenecks_gradient_boosting.py
│       ├── 1.2 performance_bottlenecks_lasso.py
│       ├── 1.3 performance_bottlenecks_random_forest.py
│       ├── 2 performance_bottlenecks_plots.py
│       ├── 3 ANALYSIS performance_bottlenecks_plots.ipynb
│       └── results.md
├── ev_range_comparison_plots.py
├── requirements.txt
└── README.md
```

---

## License / attribution

Use and cite this repository per your course or organization policy. Upstream data use is subject to the original dataset licenses.
