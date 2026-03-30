import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')

teal = "#2a9d8f"
orange = "#f4a261"
blue = "#006994"

gb = pd.read_csv("ev_range_gb_results_comparison.csv")
lstm = pd.read_csv("ev_range_lstm_results_comparison.csv")
rf = pd.read_csv("ev_range_rf_results_comparison.csv")

gb.columns = gb.columns.str.strip()
lstm.columns = lstm.columns.str.strip()
rf.columns = rf.columns.str.strip()

gb_rmse = gb["RMSE"].values
lstm_rmse = lstm["RMSE"].values
rf_rmse = rf["RMSE"].values

gb_r2 = gb["R2"].values
lstm_r2 = lstm["R2"].values
rf_r2 = rf["R2"].values

datasets = gb["Dataset"].values

bar_width = 0.25
x = list(range(len(datasets)))

plt.figure(figsize=(10,6))

plt.bar(x, gb_rmse, width=bar_width, color=teal)
plt.bar([i + bar_width for i in x], lstm_rmse, width=bar_width, color=blue)
plt.bar([i + 2 * bar_width for i in x], rf_rmse, width=bar_width, color=orange)

plt.xlabel("Dataset")
plt.ylabel("RMSE")
plt.title("RMSE Comparison Across Models")

plt.xticks([i + bar_width for i in x], datasets)
plt.legend(["Gradient Boosting", "LSTM", "Random Forest"])

plt.tight_layout()
plt.savefig("rmse_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(10,6))

plt.bar(x, gb_r2, width=bar_width, color=teal)
plt.bar([i + bar_width for i in x], lstm_r2, width=bar_width, color=blue)
plt.bar([i + 2 * bar_width for i in x], rf_r2, width=bar_width, color=orange)

plt.xlabel("Dataset")
plt.ylabel("R² Score")
plt.title("R² Comparison Across Models")

plt.xticks([i + bar_width for i in x], datasets)
plt.legend(["Gradient Boosting", "LSTM", "Random Forest"])

plt.tight_layout()
plt.savefig("r2_comparison.png", dpi=300)
plt.close()