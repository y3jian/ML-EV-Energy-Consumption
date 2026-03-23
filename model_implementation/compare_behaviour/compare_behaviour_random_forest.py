"""
Compare driving behaviors — Random Forest regressor.

Target: **Energy_Consumption_kWh** (pooled from dataset 5 + *Energy Consumed* from dataset 4).
Features: user-specified columns from datasets 4 and 5 (see `data_utils.py`).
"""

from pathlib import Path
import sys

from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import load_compare_behaviour_splits, print_regression_metrics


def main() -> None:
    X_train, X_val, X_test, y_train, y_val, y_test = load_compare_behaviour_splits()

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print_regression_metrics(
        "Random Forest (Compare driving behaviors)",
        y_val,
        model.predict(X_val),
        split_name="Validation",
    )
    print_regression_metrics(
        "Random Forest (Compare driving behaviors)",
        y_test,
        model.predict(X_test),
        split_name="Test",
    )


if __name__ == "__main__":
    main()