import pandas as pd
import numpy as np

from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

BASE_PATH = str(__import__('pathlib').Path(__file__).resolve().parents[2] / "data" / "processed")
RANDOM_STATE = 42
ALPHAS = np.logspace(-4, 1, 60)

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

lasso5 = LassoCV(alphas=ALPHAS, cv=5, random_state=RANDOM_STATE, max_iter=20000)
lasso5.fit(X_tv, y_tv)
y_pred = lasso5.predict(X_test)

print("DS5 — EV Energy Consumption | Lasso")
print(f"  Optimal alpha   : {lasso5.alpha_:.6f}")
print(f"  Non-zero coefs  : {np.sum(lasso5.coef_ != 0)} / {len(lasso5.coef_)}")
print(f"  RMSE            : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"  R2              : {r2_score(y_test, y_pred):.4f}")


# --- DS1: Vehicle Emissions (Classification) ---
X_tv, y_tv, X_test, y_test = load_splits("1-vehicle_emission")
X_tv   = X_tv.drop(columns=LEAKAGE_COLS, errors="ignore")
X_test = X_test.drop(columns=LEAKAGE_COLS, errors="ignore")

logreg = LogisticRegressionCV(
    Cs=np.logspace(-3, 2, 20),
    l1_ratios=(1,),
    solver="saga",
    cv=5,
    random_state=RANDOM_STATE,
    max_iter=5000,
    use_legacy_attributes=False
)
logreg.fit(X_tv, y_tv)
y_pred = logreg.predict(X_test)

print("\nDS1 — Vehicle Emissions | Lasso (L1 Logistic Regression)")
print(f"  Optimal C per class : {logreg.C_}")
print(f"  Accuracy            : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Weighted F1         : {f1_score(y_test, y_pred, average='weighted'):.4f}")


# --- DS4: EV Charging Patterns (Regression) ---
X_tv, y_tv, X_test, y_test = load_splits("4-ev_charging")
X_tv   = X_tv.drop(columns=DATE_COLS, errors="ignore")
X_test = X_test.drop(columns=DATE_COLS, errors="ignore")

lasso4 = LassoCV(alphas=ALPHAS, cv=5, random_state=RANDOM_STATE, max_iter=20000)
lasso4.fit(X_tv, y_tv)
y_pred = lasso4.predict(X_test)

print("\nDS4 — EV Charging Patterns | Lasso")
print(f"  Optimal alpha   : {lasso4.alpha_:.6f}")
print(f"  Non-zero coefs  : {np.sum(lasso4.coef_ != 0)} / {len(lasso4.coef_)}")
print(f"  RMSE            : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"  R2              : {r2_score(y_test, y_pred):.4f}")


# --- DS6: World GDP & CO2 Emissions (Regression) ---
X_tv, y_tv, X_test, y_test = load_splits("6-world_gdp_population_co2_emissions")

lasso6 = LassoCV(alphas=ALPHAS, cv=3, random_state=RANDOM_STATE, max_iter=20000)
lasso6.fit(X_tv, y_tv)
y_pred = lasso6.predict(X_test)

print("\nDS6 — World GDP & CO2 Emissions | Lasso")
print(f"  Optimal alpha   : {lasso6.alpha_:.6f}")
print(f"  Non-zero coefs  : {np.sum(lasso6.coef_ != 0)} / {len(lasso6.coef_)}")
print(f"  RMSE            : {np.sqrt(mean_squared_error(y_test, y_pred)):.4e}")
print(f"  R2              : {r2_score(y_test, y_pred):.4f}")
