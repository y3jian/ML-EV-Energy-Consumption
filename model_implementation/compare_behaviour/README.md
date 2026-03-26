# Compare driving behaviors (Use case **C**)

## Data source: **processed** datasets 4 + 5

`data_utils.load_compare_behaviour_splits()` loads:

| Split | Dataset 5 | Dataset 4 |
|-------|-----------|-----------|
| Train | `5-EV_energy_consumption_X_train.csv`, `y_train` | `4-ev_charging_X_train.csv`, `y_train` |
| Val   | `*_X_val.csv`, `y_val` | `*_X_val.csv`, `y_val` |
| Test  | `*_X_test.csv`, `y_test` | `*_X_test.csv`, `y_test` |

- Rows are **stacked** (dataset 5 first, then dataset 4) within each split.
- **Feature columns** = sorted **union** of both `X` files; missing columns for a row’s source are **0**.
- **Target** is reported as **`Energy_Consumption_kWh`**; dataset 4’s CSV column is `Energy Consumed (kWh)` (same units).

Prep notebooks (run export cells if files are missing):

- `4-ev_charging_patterns_data_prep.ipynb`
- `5-EV_energy_consumption_data_prep.ipynb`

## Models

| Script | Model |
|--------|--------|
| `compare_behaviour_random_forest.py` | `RandomForestRegressor` |
| `compare_behaviour_gradient_boosting.py` | `GradientBoostingRegressor` |
| `compare_behaviour_knn.py` | `KNeighborsRegressor` + `StandardScaler` |
| `compare_behaviour_lasso.py` | `Lasso` + `StandardScaler` |
| `compare_behaviour_ridge.py` | `Ridge` + `StandardScaler` |

## Run

```bash
python model_implementation/compare_behaviour/compare_behaviour_random_forest.py
```

## Note on splits

Train/val/test indices are **independent** per dataset (each notebook used `random_state=42` on its own table). The pooled split is **not** a single joint shuffle of merged rows; it reuses the official processed splits for reproducibility with the rest of the project.
