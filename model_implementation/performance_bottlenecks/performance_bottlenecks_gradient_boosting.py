import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

BASE_PATH = str(__import__('pathlib').Path(__file__).resolve().parents[2] / "data" / "processed")
RANDOM_STATE = 42

LEAKAGE_COLS = ["CO2 Emissions", "NOx Emissions", "PM2.5 Emissions", "VOC Emissions", "SO2 Emissions"]

PARAM_GRID = {
    "n_estimators":  [100, 200],
    "learning_rate": [0.05, 0.10],
    "max_depth":     [3, 4]
}


def load_splits(prefix):
    X_train = pd.read_csv(f"{BASE_PATH}/{prefix}_X_train.csv")
    X_val   = pd.read_csv(f"{BASE_PATH}/{prefix}_X_val.csv")
    X_test  = pd.read_csv(f"{BASE_PATH}/{prefix}_X_test.csv")
    y_train = pd.read_csv(f"{BASE_PATH}/{prefix}_y_train.csv").squeeze()
    y_val   = pd.read_csv(f"{BASE_PATH}/{prefix}_y_val.csv").squeeze()
    y_test  = pd.read_csv(f"{BASE_PATH}/{prefix}_y_test.csv").squeeze()
    return X_train, y_train, X_val, y_val, X_test, y_test


# --- DS5: EV Energy Consumption (Regression) ---
X_train, y_train, X_val, y_val, X_test, y_test = load_splits("5-EV_energy_consumption")

search5 = GridSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    PARAM_GRID, cv=5, scoring="r2", n_jobs=-1
)
search5.fit(X_train, y_train)
best5 = search5.best_estimator_

print("DS5 — EV Energy Consumption | Gradient Boosting")
print(f"  Best params : {search5.best_params_}")
for split, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
    p = best5.predict(X)
    print(f"  {split:5s} RMSE     : {np.sqrt(mean_squared_error(y, p)):.4f}  R2: {r2_score(y, p):.4f}")


# --- DS1: Vehicle Emissions (Binary: High vs Not High) ---
X_train, y_train, X_val, y_val, X_test, y_test = load_splits("1-vehicle_emission")
X_train = X_train.drop(columns=LEAKAGE_COLS, errors="ignore")
X_val   = X_val.drop(columns=LEAKAGE_COLS, errors="ignore")
X_test  = X_test.drop(columns=LEAKAGE_COLS, errors="ignore")
y_train = (y_train == 2).astype(int)
y_val   = (y_val == 2).astype(int)
y_test  = (y_test == 2).astype(int)

pipe1 = ImbPipeline([
    ("smote", SMOTE(random_state=RANDOM_STATE)),
    ("clf",   GradientBoostingClassifier(random_state=RANDOM_STATE)),
])
param_grid_1 = {f"clf__{k}": v for k, v in PARAM_GRID.items()}
search1 = GridSearchCV(pipe1, param_grid_1, cv=5, scoring="f1_weighted", n_jobs=-1)
search1.fit(X_train, y_train)
best1 = search1.best_estimator_

print("\nDS1 — Vehicle Emissions (High vs Not High) | Gradient Boosting")
print(f"  Best params  : { {k.replace('clf__', ''): v for k, v in search1.best_params_.items()} }")
for split, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
    p = best1.predict(X)
    print(f"  {split:5s} Accuracy : {accuracy_score(y, p):.4f}  Weighted F1: {f1_score(y, p, average='weighted'):.4f}")
