import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

BASE_PATH = str(__import__('pathlib').Path(__file__).resolve().parents[2] / "data" / "processed")
RANDOM_STATE = 42
N_ESTIMATORS = 200

LEAKAGE_COLS = ["CO2 Emissions", "NOx Emissions", "PM2.5 Emissions", "VOC Emissions", "SO2 Emissions"]
DATE_COLS = ["Charging Start Time", "Charging End Time"]


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

rf5 = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
cv_r2 = cross_val_score(rf5, X_tv, y_tv, cv=5, scoring="r2", n_jobs=-1)
rf5.fit(X_tv, y_tv)
y_pred = rf5.predict(X_test)

print("DS5 — EV Energy Consumption | Random Forest")
print(f"  CV R2 (5-fold)  : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
print(f"  RMSE            : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"  R2              : {r2_score(y_test, y_pred):.4f}")


# --- DS1: Vehicle Emissions (Classification) ---
X_tv, y_tv, X_test, y_test = load_splits("1-vehicle_emission")
X_tv   = X_tv.drop(columns=LEAKAGE_COLS, errors="ignore")
X_test = X_test.drop(columns=LEAKAGE_COLS, errors="ignore")

rf1 = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
cv_acc = cross_val_score(rf1, X_tv, y_tv, cv=5, scoring="accuracy", n_jobs=-1)
rf1.fit(X_tv, y_tv)
y_pred = rf1.predict(X_test)

print("\nDS1 — Vehicle Emissions | Random Forest")
print(f"  CV Accuracy (5-fold) : {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
print(f"  Accuracy             : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Weighted F1          : {f1_score(y_test, y_pred, average='weighted'):.4f}")


# --- DS4: EV Charging Patterns (Regression) ---
X_tv, y_tv, X_test, y_test = load_splits("4-ev_charging")
X_tv   = X_tv.drop(columns=DATE_COLS, errors="ignore")
X_test = X_test.drop(columns=DATE_COLS, errors="ignore")

rf4 = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
cv_r2 = cross_val_score(rf4, X_tv, y_tv, cv=5, scoring="r2", n_jobs=-1)
rf4.fit(X_tv, y_tv)
y_pred = rf4.predict(X_test)

print("\nDS4 — EV Charging Patterns | Random Forest")
print(f"  CV R2 (5-fold)  : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
print(f"  RMSE            : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"  R2              : {r2_score(y_test, y_pred):.4f}")


# --- DS6: World GDP & CO2 Emissions (Regression) ---
X_tv, y_tv, X_test, y_test = load_splits("6-world_gdp_population_co2_emissions")

rf6 = RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
cv_r2 = cross_val_score(rf6, X_tv, y_tv, cv=3, scoring="r2", n_jobs=-1)
rf6.fit(X_tv, y_tv)
y_pred = rf6.predict(X_test)

print("\nDS6 — World GDP & CO2 Emissions | Random Forest")
print(f"  CV R2 (3-fold)  : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
print(f"  RMSE            : {np.sqrt(mean_squared_error(y_test, y_pred)):.4e}")
print(f"  R2              : {r2_score(y_test, y_pred):.4f}")
