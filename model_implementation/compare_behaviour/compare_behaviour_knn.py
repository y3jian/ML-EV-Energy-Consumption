import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score

BASE_PATH = str(__import__('pathlib').Path(__file__).resolve().parents[2] / "data" / "processed")
RANDOM_STATE = 42

DS4_X_COLS = [
    "Time of Day_Afternoon", "Time of Day_Evening", "Time of Day_Morning", "Time of Day_Night",
    "Day of Week_Friday", "Day of Week_Monday", "Day of Week_Saturday", "Day of Week_Sunday",
    "Day of Week_Thursday", "Day of Week_Tuesday", "Day of Week_Wednesday",
    "Distance Driven (since last charge) (km)", "Charging Duration (hours)",
    "Charging Station Location_Chicago", "Charging Station Location_Houston",
    "Charging Station Location_Los Angeles", "Charging Station Location_New York",
    "Charging Station Location_San Francisco",
    "State of Charge (Start %)", "State of Charge (End %)",
]
USER_TYPE_COLS = ["User Type_Casual Driver", "User Type_Commuter", "User Type_Long-Distance Traveler"]

DS5_X_COLS = [
    "Speed_kmh", "Acceleration_ms2", "Distance_Travelled_km",
    "Driving_Mode_1", "Driving_Mode_2", "Driving_Mode_3",
    "Road_Type_1", "Road_Type_2", "Road_Type_3",
    "Traffic_Condition_1", "Traffic_Condition_2", "Traffic_Condition_3",
]

DS1_X_COLS = [
    "Speed", "Acceleration", "Mileage",
    "Vehicle Type_bus", "Vehicle Type_car", "Vehicle Type_motorcycle", "Vehicle Type_truck",
    "Road Type_city", "Road Type_highway", "Road Type_rural",
    "Traffic Conditions_free flow", "Traffic Conditions_heavy", "Traffic Conditions_moderate",
]


def load_splits(prefix):
    X_train = pd.read_csv(f"{BASE_PATH}/{prefix}_X_train.csv")
    X_val   = pd.read_csv(f"{BASE_PATH}/{prefix}_X_val.csv")
    X_test  = pd.read_csv(f"{BASE_PATH}/{prefix}_X_test.csv")
    y_train = pd.read_csv(f"{BASE_PATH}/{prefix}_y_train.csv").squeeze()
    y_val   = pd.read_csv(f"{BASE_PATH}/{prefix}_y_val.csv").squeeze()
    y_test  = pd.read_csv(f"{BASE_PATH}/{prefix}_y_test.csv").squeeze()
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_ds3_splits():
    splits = []
    for split in ("train", "val", "test"):
        X = pd.read_csv(f"{BASE_PATH}/3-EV_population_data_X_{split}.csv")
        ev_cols   = [c for c in X.columns if c.startswith("Electric Vehicle Type_")]
        cafv_cols = [c for c in X.columns if c.startswith("Clean Alternative Fuel Vehicle")]
        ld_cols   = [c for c in X.columns if c.startswith("Legislative District_")]
        splits.append(X[ev_cols + cafv_cols + ld_cols])
    X_train, X_val, X_test = splits
    y_train = pd.read_csv(f"{BASE_PATH}/3-EV_population_data_y_train.csv").squeeze()
    y_val   = pd.read_csv(f"{BASE_PATH}/3-EV_population_data_y_val.csv").squeeze()
    y_test  = pd.read_csv(f"{BASE_PATH}/3-EV_population_data_y_test.csv").squeeze()
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_ds4_splits():
    splits = []
    ys = []
    for split in ("train", "val", "test"):
        X = pd.read_csv(f"{BASE_PATH}/4-ev_charging_X_{split}.csv")
        y = X[USER_TYPE_COLS].idxmax(axis=1).str.replace("User Type_", "", regex=False)
        splits.append(X[DS4_X_COLS])
        ys.append(y)
    return splits[0], ys[0], splits[1], ys[1], splits[2], ys[2]


def scale(X_train, X_val, X_test):
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test)


# --- DS4: EV Charging — classify User Type ---
X_train, y_train, X_val, y_val, X_test, y_test = load_ds4_splits()
Xtr, Xva, Xte = scale(X_train, X_val, X_test)

knn4 = KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)
knn4.fit(Xtr, y_train)

print("DS4 — EV Charging Patterns (User Type) | KNN")
for split, X, y in [("Train", Xtr, y_train), ("Val", Xva, y_val), ("Test", Xte, y_test)]:
    p = knn4.predict(X)
    print(f"  {split:5s} Accuracy : {accuracy_score(y, p):.4f}  Weighted F1: {f1_score(y, p, average='weighted'):.4f}")


# --- DS5: EV Energy Consumption — predict Energy_Consumption_kWh ---
X_train, y_train, X_val, y_val, X_test, y_test = load_splits("5-EV_energy_consumption")
X_train = X_train[DS5_X_COLS]
X_val   = X_val[DS5_X_COLS]
X_test  = X_test[DS5_X_COLS]
Xtr, Xva, Xte = scale(X_train, X_val, X_test)

knn5 = KNeighborsRegressor(n_neighbors=7, weights="distance", n_jobs=-1)
knn5.fit(Xtr, y_train)

print("\nDS5 — EV Energy Consumption | KNN")
for split, X, y in [("Train", Xtr, y_train), ("Val", Xva, y_val), ("Test", Xte, y_test)]:
    p = knn5.predict(X)
    print(f"  {split:5s} RMSE     : {np.sqrt(mean_squared_error(y, p)):.4f}  R2: {r2_score(y, p):.4f}")


# --- DS1: Vehicle Emissions — classify Emission Level ---
X_train, y_train, X_val, y_val, X_test, y_test = load_splits("1-vehicle_emission")
X_train = X_train[DS1_X_COLS]
X_val   = X_val[DS1_X_COLS]
X_test  = X_test[DS1_X_COLS]
Xtr, Xva, Xte = scale(X_train, X_val, X_test)

knn1 = KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=-1)
knn1.fit(Xtr, y_train)

print("\nDS1 — Vehicle Emissions (Emission Level) | KNN")
for split, X, y in [("Train", Xtr, y_train), ("Val", Xva, y_val), ("Test", Xte, y_test)]:
    p = knn1.predict(X)
    print(f"  {split:5s} Accuracy : {accuracy_score(y, p):.4f}  Weighted F1: {f1_score(y, p, average='weighted'):.4f}")


# --- DS3: EV Population — predict Electric Range ---
X_train, y_train, X_val, y_val, X_test, y_test = load_ds3_splits()
Xtr, Xva, Xte = scale(X_train, X_val, X_test)

knn3 = KNeighborsRegressor(n_neighbors=7, weights="distance", n_jobs=-1)
knn3.fit(Xtr, y_train)

print("\nDS3 — EV Population (Electric Range) | KNN")
for split, X, y in [("Train", Xtr, y_train), ("Val", Xva, y_val), ("Test", Xte, y_test)]:
    p = knn3.predict(X)
    print(f"  {split:5s} RMSE     : {np.sqrt(mean_squared_error(y, p)):.4f}  R2: {r2_score(y, p):.4f}")
