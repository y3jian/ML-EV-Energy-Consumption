import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

base_path = "data/processed/"

datasets = {
    "Dataset 3": "3-EV_population_data",
    "Dataset 4": "4-ev_charging",
    "Dataset 5": "5-EV_energy_consumption"
}

results = []

for name, prefix in datasets.items():
    print(f"\nProcessing {name}...")

    try:
        X_train = pd.read_csv(base_path + f"{prefix}_X_train.csv")
        X_test  = pd.read_csv(base_path + f"{prefix}_X_test.csv")
        y_train = pd.read_csv(base_path + f"{prefix}_y_train.csv").values.ravel()
        y_test  = pd.read_csv(base_path + f"{prefix}_y_test.csv").values.ravel()

        SAMPLE_FRAC = 0.5  # uses 50% of data

        X_train = X_train.sample(frac=SAMPLE_FRAC, random_state=42)
        y_train = y_train[X_train.index]

        X_test = X_test.sample(frac=SAMPLE_FRAC, random_state=42)
        y_test = y_test[X_test.index]

        X_full = pd.concat([X_train, X_test], axis=0)

        drop_cols = ["User ID", "Charging Station ID"]
        X_full = X_full.drop(columns=[c for c in drop_cols if c in X_full.columns], errors="ignore")

        for col in X_full.columns:
            if "Time" in col:
                try:
                    X_full[col] = pd.to_datetime(X_full[col])
                    X_full[col + "_hour"] = X_full[col].dt.hour
                    X_full = X_full.drop(columns=[col])
                except:
                    pass

        categorical_cols = X_full.select_dtypes(include=["object"]).columns
        X_full = pd.get_dummies(X_full, columns=categorical_cols, drop_first=True)

        X_train = X_full.iloc[:len(X_train), :]
        X_test  = X_full.iloc[len(X_train):, :]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm  = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        model = Sequential([
            Input(shape=(1, X_train_scaled.shape[1])),
            LSTM(16, activation="relu"),
            Dense(1)
        ])

        model.compile(optimizer="adam", loss="mse")

        model.fit(X_train_lstm, y_train, epochs=10, batch_size=16, verbose=0)

        y_pred = model.predict(X_test_lstm, batch_size=16, verbose=0)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({"Dataset": name, "RMSE": rmse, "R2": r2})

        print(f"{name} DONE")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2: {r2:.4f}")

    except Exception as e:
        print(f"Error in {name}: {e}")

results_df = pd.DataFrame(results)
print("\n=== LSTM Comparison ===")
print(results_df)

results_df.to_csv("lstm_results_comparison.csv", index=False)