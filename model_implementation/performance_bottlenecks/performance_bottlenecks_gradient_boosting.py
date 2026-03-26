import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

BASE_PATH = str(__import__('pathlib').Path(__file__).resolve().parents[2] / "data" / "processed")
RANDOM_STATE = 42

LEAKAGE_COLS = ["CO2 Emissions", "NOx Emissions", "PM2.5 Emissions", "VOC Emissions", "SO2 Emissions"]
DATE_COLS = ["Charging Start Time", "Charging End Time"]

PARAM_GRID = {
    "n_estimators":  [100, 200],
    "learning_rate": [0.05, 0.10],
    "max_depth":     [3, 4]
}

PARAM_GRID_SMALL = {
    "n_estimators":  [50, 100],
    "learning_rate": [0.05, 0.10],
    "max_depth":     [2, 3]
}


def load_splits(prefix):
    X_train = pd.read_csv(f"{BASE_PATH}/{prefix}_X_train.csv")
    X_val   = pd.read_csv(f"{BASE_PATH}/{prefix}_X_val.csv")
    X_test  = pd.read_csv(f"{BASE_PATH}/{prefix}_X_test.csv")
    y_train = pd.read_csv(f"{BASE_PATH}/{prefix}_y_train.csv").squeeze()
    y_val   = pd.read_csv(f"{BASE_PATH}/{prefix}_y_val.csv").squeeze()
    y_test  = pd.read_csv(f"{BASE_PATH}/{prefix}_y_test.csv").squeeze()
    X_tv = pd.concat([X_train, X_val], ignore_index=True)
    y_tv = pd.concat([y_train, y_val], ignore_index=True)
    return X_tv, y_tv, X_test, y_test


# --- DS5: EV Energy Consumption (Regression) ---
X_tv, y_tv, X_test, y_test = load_splits("5-EV_energy_consumption")

search5 = GridSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    PARAM_GRID, cv=5, scoring="r2", n_jobs=-1
)
search5.fit(X_tv, y_tv)
y_pred = search5.best_estimator_.predict(X_test)

print("DS5 — EV Energy Consumption | Gradient Boosting")
print(f"  Best params : {search5.best_params_}")
print(f"  RMSE        : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"  R2          : {r2_score(y_test, y_pred):.4f}")


# --- DS1: Vehicle Emissions (Classification) ---
X_tv, y_tv, X_test, y_test = load_splits("1-vehicle_emission")
X_tv   = X_tv.drop(columns=LEAKAGE_COLS, errors="ignore")
X_test = X_test.drop(columns=LEAKAGE_COLS, errors="ignore")

search1 = GridSearchCV(
    GradientBoostingClassifier(random_state=RANDOM_STATE),
    PARAM_GRID, cv=5, scoring="accuracy", n_jobs=-1
)
search1.fit(X_tv, y_tv)
y_pred = search1.best_estimator_.predict(X_test)

print("\nDS1 — Vehicle Emissions | Gradient Boosting")
print(f"  Best params  : {search1.best_params_}")
print(f"  Accuracy     : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Weighted F1  : {f1_score(y_test, y_pred, average='weighted'):.4f}")


# --- DS4: EV Charging Patterns (Regression) ---
X_tv, y_tv, X_test, y_test = load_splits("4-ev_charging")
X_tv   = X_tv.drop(columns=DATE_COLS, errors="ignore")
X_test = X_test.drop(columns=DATE_COLS, errors="ignore")

search4 = GridSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    PARAM_GRID, cv=5, scoring="r2", n_jobs=-1
)
search4.fit(X_tv, y_tv)
y_pred = search4.best_estimator_.predict(X_test)

print("\nDS4 — EV Charging Patterns | Gradient Boosting")
print(f"  Best params : {search4.best_params_}")
print(f"  RMSE        : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"  R2          : {r2_score(y_test, y_pred):.4f}")


# --- DS6: World GDP & CO2 Emissions (Regression) ---
X_tv, y_tv, X_test, y_test = load_splits("6-world_gdp_population_co2_emissions")

search6 = GridSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    PARAM_GRID_SMALL, cv=3, scoring="r2", n_jobs=-1
)
search6.fit(X_tv, y_tv)
y_pred = search6.best_estimator_.predict(X_test)

print("\nDS6 — World GDP & CO2 Emissions | Gradient Boosting")
print(f"  Best params : {search6.best_params_}")
print(f"  RMSE        : {np.sqrt(mean_squared_error(y_test, y_pred)):.4e}")
print(f"  R2          : {r2_score(y_test, y_pred):.4f}")
