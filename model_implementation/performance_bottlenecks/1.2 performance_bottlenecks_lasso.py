# Lasso implementation for identifying performance bottlenecks

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
    return X_train, y_train, X_val, y_val, X_test, y_test


# --- DS5: EV Energy Consumption (Regression) ---
X_train, y_train, X_val, y_val, X_test, y_test = load_splits("5-EV_energy_consumption")

lasso5 = LassoCV(alphas=ALPHAS, cv=5, random_state=RANDOM_STATE, max_iter=20000)
lasso5.fit(X_train, y_train)

print("DS5 — EV Energy Consumption | Lasso")
print(f"  Optimal alpha   : {lasso5.alpha_:.6f}")
print(f"  Non-zero coefs  : {np.sum(lasso5.coef_ != 0)} / {len(lasso5.coef_)}")
for split, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
    p = lasso5.predict(X)
    print(f"  {split:5s} RMSE     : {np.sqrt(mean_squared_error(y, p)):.4f}  R2: {r2_score(y, p):.4f}")
