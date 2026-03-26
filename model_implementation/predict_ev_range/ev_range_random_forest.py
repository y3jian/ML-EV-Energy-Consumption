import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

base_path = "data/processed/"

datasets = {
    #"Dataset 2": "2-ElectricCarData",
    "Dataset 3": "3-EV_population_data",
    "Dataset 4": "4-ev_charging",
    "Dataset 5": "5-EV_energy_consumption"
}

results = []

for name, prefix in datasets.items():
    print(f"\nProcessing {name}...")

    try:
        # Loading of data
        X_train = pd.read_csv(base_path + f"{prefix}_X_train.csv")
        X_test  = pd.read_csv(base_path + f"{prefix}_X_test.csv")
        y_train = pd.read_csv(base_path + f"{prefix}_y_train.csv").values.ravel()
        y_test  = pd.read_csv(base_path + f"{prefix}_y_test.csv").values.ravel()

        # Combines for consistent preprocessing
        X_full = pd.concat([X_train, X_test], axis=0)

        drop_cols = ["User ID", "Charging Station ID"]
        X_full = X_full.drop(columns=[c for c in drop_cols if c in X_full.columns], errors="ignore")

        # Converts datetime columns
        for col in X_full.columns:
            if "Time" in col:
                try:
                    X_full[col] = pd.to_datetime(X_full[col])
                    X_full[col + "_hour"] = X_full[col].dt.hour
                    X_full = X_full.drop(columns=[col])
                except:
                    pass

        # Encodes categorical (object) columns
        categorical_cols = X_full.select_dtypes(include=["object"]).columns
        X_full = pd.get_dummies(X_full, columns=categorical_cols, drop_first=True)

        # Splits data
        X_train = X_full.iloc[:len(X_train), :]
        X_test  = X_full.iloc[len(X_train):, :]

        # Model
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Dataset": name,
            "RMSE": rmse,
            "R2": r2
        })

        print(f"{name} DONE")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")

    except Exception as e:
        print(f"Error in {name}: {e}")

# Final results
results_df = pd.DataFrame(results)

print("\n=== Final Comparison ===")
print(results_df)

results_df.to_csv("rf_results_comparison.csv", index=False)
print("\nResults saved to rf_results_comparison.csv")