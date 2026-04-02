# EXCLUDED FROM FINAL IMPLEMENTATION

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("EVENTUALLY_PUT_CSV_HERE")

X = df["COLUMNS WE WANT TO USE FOR TESTING"]
y = df["WHAT VALUE WE'RE TRYING TO PREDICT"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_model = KNeighborsRegressor(n_neighbors=5, weights="uniform", metric="minkowski")  # Beginning with 5 neighbour analysis

knn_model.fit(X_train_scaled, y_train)

y_pred = knn_model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("KNN Regression Results")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2:  {r2:.4f}")

plt.figure(figsize=(8, 6))
sb.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)
plt.show()