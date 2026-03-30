import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

BASE_PATH = str(Path(__file__).resolve().parents[2] / "data" / "processed")
OUT_PATH  = str(Path(__file__).resolve().parent / "performance_bottlenecks_plots.png")
RANDOM_STATE = 42
ALPHAS = np.logspace(-4, 1, 60)
LEAKAGE_COLS = ["CO2 Emissions", "NOx Emissions", "PM2.5 Emissions",
                "VOC Emissions", "SO2 Emissions"]


def load_splits(prefix):
    def rd(tag):
        return pd.read_csv(f"{BASE_PATH}/{prefix}_{tag}.csv")
    X_train = rd("X_train"); X_val = rd("X_val"); X_test = rd("X_test")
    y_train = rd("y_train").squeeze()
    y_val   = rd("y_val").squeeze()
    y_test  = rd("y_test").squeeze()
    return X_train, y_train, X_val, y_val, X_test, y_test


def r2s(model, X, y):
    return r2_score(y, model.predict(X))

def accs(model, X, y):
    return accuracy_score(y, model.predict(X))


# ══════════════════════════════════════════════════════════════════════════
# 1. FIT DS5 MODELS
# ══════════════════════════════════════════════════════════════════════════
print("Fitting DS5 models…")
X_train5, y_train5, X_val5, y_val5, X_test5, y_test5 = load_splits("5-EV_energy_consumption")

rf5 = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
rf5.fit(X_train5, y_train5)
ds5_rf = [r2s(rf5, X_train5, y_train5), r2s(rf5, X_val5, y_val5), r2s(rf5, X_test5, y_test5)]

search5 = GridSearchCV(
    GradientBoostingRegressor(random_state=RANDOM_STATE),
    {"n_estimators": [100, 200], "learning_rate": [0.05, 0.10], "max_depth": [3, 4]},
    cv=5, scoring="r2", n_jobs=-1,
)
search5.fit(X_train5, y_train5)
gb5 = search5.best_estimator_
ds5_gb = [r2s(gb5, X_train5, y_train5), r2s(gb5, X_val5, y_val5), r2s(gb5, X_test5, y_test5)]

lasso5 = LassoCV(alphas=ALPHAS, cv=5, random_state=RANDOM_STATE, max_iter=20000)
lasso5.fit(X_train5, y_train5)
ds5_las = [r2s(lasso5, X_train5, y_train5), r2s(lasso5, X_val5, y_val5), r2s(lasso5, X_test5, y_test5)]


# ══════════════════════════════════════════════════════════════════════════
# 2. DS1 OVERFITTING ITERATIONS (computed live)
#    Iter 1 — binary, no balancing, no constraints
#    Iter 2 — binary + class_weight / ImbPipeline, no depth constraints
#    Iter 3 — binary + class_weight / ImbPipeline + depth constraints (current)
# ══════════════════════════════════════════════════════════════════════════
print("Computing DS1 overfitting iterations…")
X_train1, y_train1, X_val1, y_val1, X_test1, y_test1 = load_splits("1-vehicle_emission")
for df in (X_train1, X_val1, X_test1):
    df.drop(columns=LEAKAGE_COLS, errors="ignore", inplace=True)
y_train1 = (y_train1 == 2).astype(int)
y_val1   = (y_val1   == 2).astype(int)
y_test1  = (y_test1  == 2).astype(int)


def fit_rf_iter(class_weight, max_depth, min_samples_leaf):
    m = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1,
        class_weight=class_weight, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
    )
    m.fit(X_train1, y_train1)
    return accs(m, X_train1, y_train1), accs(m, X_test1, y_test1)


def fit_gb_iter(use_smote, max_depth, n_estimators=200, lr=0.1):
    if use_smote:
        m = ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf",   GradientBoostingClassifier(
                n_estimators=n_estimators, learning_rate=lr,
                max_depth=max_depth, random_state=RANDOM_STATE)),
        ])
    else:
        m = GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=lr,
            max_depth=max_depth, random_state=RANDOM_STATE,
        )
    m.fit(X_train1, y_train1)
    return accs(m, X_train1, y_train1), accs(m, X_test1, y_test1)


rf_iters = [
    fit_rf_iter(class_weight=None,       max_depth=None, min_samples_leaf=1),
    fit_rf_iter(class_weight="balanced", max_depth=None, min_samples_leaf=1),
    fit_rf_iter(class_weight="balanced", max_depth=5,    min_samples_leaf=5),
]
gb_iters = [
    fit_gb_iter(use_smote=False, max_depth=4, n_estimators=200, lr=0.1),
    fit_gb_iter(use_smote=True,  max_depth=4, n_estimators=200, lr=0.1),
    fit_gb_iter(use_smote=True,  max_depth=2, n_estimators=100, lr=0.1),
]


# ══════════════════════════════════════════════════════════════════════════
# PLOT — 1×2 layout
# ══════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Performance Bottlenecks — Model Diagnostics", fontsize=14, fontweight="bold")

# ── Chart 1: DS5 R² bar chart ─────────────────────────────────────────────
models  = ["Random Forest", "Gradient Boosting", "Lasso"]
r2_data = np.array([ds5_rf, ds5_gb, ds5_las])
splits  = ["Train", "Val", "Test"]
x       = np.arange(len(models))
width   = 0.25
colors  = ["#4C72B0", "#DD8452", "#55A868"]

for i, (split, color) in enumerate(zip(splits, colors)):
    bars = ax1.bar(x + i * width, r2_data[:, i], width, label=split, color=color, alpha=0.85)
    for bar, val in zip(bars, r2_data[:, i]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7.5)

ax1.set_title("DS5 — EV Energy Consumption\nR² by Model & Split", fontsize=11)
ax1.set_xticks(x + width)
ax1.set_xticklabels(models, fontsize=10)
ax1.set_ylabel("R²")
ax1.set_ylim(0.88, 1.03)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax1.legend(fontsize=9)

# ── Chart 2: DS1 overfitting gap across iterations ────────────────────────
iter_labels = ["Iter 1\n(No balancing,\nno constraints)",
               "Iter 2\n(Balanced,\nno constraints)",
               "Iter 3\n(Balanced +\nconstrained)"]
x     = np.arange(len(iter_labels))
width = 0.2

rf_train = [t  for t,  _ in rf_iters]
rf_test  = [te for _,  te in rf_iters]
gb_train = [t  for t,  _ in gb_iters]
gb_test  = [te for _,  te in gb_iters]

ax2.bar(x - 1.5 * width, rf_train, width, label="RF Train",  color="#4C72B0", alpha=0.85)
ax2.bar(x - 0.5 * width, rf_test,  width, label="RF Test",   color="#4C72B0", alpha=0.45, hatch="//")
ax2.bar(x + 0.5 * width, gb_train, width, label="GB Train",  color="#DD8452", alpha=0.85)
ax2.bar(x + 1.5 * width, gb_test,  width, label="GB Test",   color="#DD8452", alpha=0.45, hatch="//")

ax2.set_title("DS1 — Overfitting Gap Across Tuning Iterations\n(Train vs Test Accuracy)", fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(iter_labels, fontsize=9)
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1.15)
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax2.legend(fontsize=9, ncol=2)

for xs, vals in [(x - 1.5*width, rf_train), (x - 0.5*width, rf_test),
                 (x + 0.5*width, gb_train), (x + 1.5*width, gb_test)]:
    for xi, v in zip(xs, vals):
        ax2.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
