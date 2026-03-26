"""
Compare driving behaviors — Lasso (L1) regression.
"""

from pathlib import Path
import sys

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_utils import load_compare_behaviour_splits, print_regression_metrics


def main() -> None:
    X_train, X_val, X_test, y_train, y_val, y_test = load_compare_behaviour_splits()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_va = scaler.transform(X_val)
    X_te = scaler.transform(X_test)

    model = Lasso(alpha=0.1, max_iter=10000)
    model.fit(X_tr, y_train)

    print_regression_metrics(
        "Lasso (Compare driving behaviors)",
        y_val,
        model.predict(X_va),
        split_name="Validation",
    )
    print_regression_metrics(
        "Lasso (Compare driving behaviors)",
        y_test,
        model.predict(X_te),
        split_name="Test",
    )


if __name__ == "__main__":
    main()
