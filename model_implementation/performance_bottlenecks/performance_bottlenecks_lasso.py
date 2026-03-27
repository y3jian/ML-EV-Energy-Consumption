import pandas as pd
import numpy as np

from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score

BASE_PATH = str(__import__('pathlib').Path(__file__).resolve().parents[2] / "data" / "processed")
RANDOM_STATE = 42
ALPHAS = np.logspace(-4, 1, 60)


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
