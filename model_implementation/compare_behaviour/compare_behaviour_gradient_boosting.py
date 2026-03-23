"""
Compare driving behaviors — Gradient Boosting regressor.

Target: **Energy_Consumption_kWh** (pooled datasets 4 & 5). See `data_utils.py`.
"""

from pathlib import Path
import sys

from sklearn.ensemble import GradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import load_compare_behaviour_splits, print_regression_metrics


def main() -> None:
    X_train, X_val, X_test, y_train, y_val, y_test = load_compare_behaviour_splits()

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    print_regression_metrics(
        "Gradient Boosting (Compare driving behaviors)",
        y_val,
        model.predict(X_val),
        split_name="Validation",
    )
    print_regression_metrics(
        "Gradient Boosting (Compare driving behaviors)",
        y_test,
        model.predict(X_test),
        split_name="Test",
    )


if __name__ == "__main__":
    main()