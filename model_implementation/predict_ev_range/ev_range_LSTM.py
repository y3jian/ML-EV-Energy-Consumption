import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("EV_DATASET.csv")

X = df.drop("ev_range", axis=1)
y = df["ev_range"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_lstm = X_train_scaled.reshape(
    (X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
)

X_test_lstm = X_test_scaled.reshape(
    (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
)

model = Sequential()

model.add(LSTM(50, activation="relu", input_shape=(1, X_train_scaled.shape[1])))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

model.fit(X_train_lstm, y_train, epochs=30, batch_size=32)

y_pred = model.predict(X_test_lstm)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("LSTM Results")
print("RMSE:", rmse)
print("R2:", r2)