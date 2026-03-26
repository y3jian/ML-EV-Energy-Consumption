import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

_PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(_PKG))

from optimize_efficiency_data import load_dataset_2, load_dataset_4, load_dataset_5

datasets = {
    "Dataset 2 (EV specs → Efficiency)": load_dataset_2,
    "Dataset 4 (EV charging → Charging Cost)": load_dataset_4,
    "Dataset 5 (EV energy → kWh)": load_dataset_5,
}

results = []

for name, loader in datasets.items():
    print(f"\nProcessing {name}...")

    try:
        X_train, _X_val, X_test, y_train, _y_val, y_test = loader()

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = Ridge(alpha=1.0)
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)

        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)

        results.append({"Dataset": name, "RMSE": rmse, "R2": r2})

        print(f"{name} DONE")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")

    except Exception as e:
        print(f"Error in {name}: {e}")

results_df = pd.DataFrame(results)

print("\n=== Ridge (optimize driving efficiency) ===")
print(results_df)

out_path = Path("optimize_efficiency_ridge_results_comparison.csv")
results_df.to_csv(out_path, index=False)
print(f"\nResults saved to {out_path}")
