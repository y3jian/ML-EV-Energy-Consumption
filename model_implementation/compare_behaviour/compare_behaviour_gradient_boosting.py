import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

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

PARAM_GRID_REG = {
    "n_estimators":  [100, 200],
    "learning_rate": [0.05, 0.10],
    "max_depth":     [3, 4],
}

PARAM_GRID_CLF = {
    "clf__n_estimators":  [50, 100],
    "clf__learning_rate": [0.05, 0.10],
    "clf__max_depth":     [2],
}


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


def clf_pipeline():
    return ImbPipeline([
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf",   GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])


# --- DS4: EV Charging — classify User Type ---
X_train, y_train, X_val, y_val, X_test, y_test = load_ds4_splits()

search4 = GridSearchCV(clf_pipeline(), PARAM_GRID_CLF, cv=5, scoring="f1_weighted", n_jobs=-1)
search4.fit(X_train, y_train)
best4 = search4.best_estimator_

print("DS4 — EV Charging Patterns (User Type) | Gradient Boosting")
print(f"  Best params  : { {k.replace('clf__', ''): v for k, v in search4.best_params_.items()} }")
for split, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
    p = best4.predict(X)
    print(f"  {split:5s} Accuracy : {accuracy_score(y, p):.4f}  Weighted F1: {f1_score(y, p, average='weighted'):.4f}")


# --- DS5: EV Energy Consumption — predict Energy_Consumption_kWh ---
X_train, y_train, X_val, y_val, X_test, y_test = load_splits("5-EV_energy_consumption")
X_train = X_train[DS5_X_COLS]
X_val   = X_val[DS5_X_COLS]
X_test  = X_test[DS5_X_COLS]

search5 = GridSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    PARAM_GRID_REG, cv=5, scoring="r2", n_jobs=-1
)
search5.fit(X_train, y_train)
best5 = search5.best_estimator_

print("\nDS5 — EV Energy Consumption | Gradient Boosting")
print(f"  Best params : {search5.best_params_}")
for split, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
    p = best5.predict(X)
    print(f"  {split:5s} RMSE     : {np.sqrt(mean_squared_error(y, p)):.4f}  R2: {r2_score(y, p):.4f}")


# --- DS1: Vehicle Emissions — classify Emission Level ---
X_train, y_train, X_val, y_val, X_test, y_test = load_splits("1-vehicle_emission")
X_train = X_train[DS1_X_COLS]
X_val   = X_val[DS1_X_COLS]
X_test  = X_test[DS1_X_COLS]

search1 = GridSearchCV(clf_pipeline(), PARAM_GRID_CLF, cv=5, scoring="f1_weighted", n_jobs=-1)
search1.fit(X_train, y_train)
best1 = search1.best_estimator_

print("\nDS1 — Vehicle Emissions (Emission Level) | Gradient Boosting")
print(f"  Best params  : { {k.replace('clf__', ''): v for k, v in search1.best_params_.items()} }")
for split, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
    p = best1.predict(X)
    print(f"  {split:5s} Accuracy : {accuracy_score(y, p):.4f}  Weighted F1: {f1_score(y, p, average='weighted'):.4f}")


# --- DS3: EV Population — predict Electric Range ---
X_train, y_train, X_val, y_val, X_test, y_test = load_ds3_splits()

search3 = GridSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    PARAM_GRID_REG, cv=5, scoring="r2", n_jobs=-1
)
search3.fit(X_train, y_train)
best3 = search3.best_estimator_

print("\nDS3 — EV Population (Electric Range) | Gradient Boosting")
print(f"  Best params : {search3.best_params_}")
for split, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
    p = best3.predict(X)
    print(f"  {split:5s} RMSE     : {np.sqrt(mean_squared_error(y, p)):.4f}  R2: {r2_score(y, p):.4f}")
