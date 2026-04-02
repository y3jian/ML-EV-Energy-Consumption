"""Heuristics for train vs validation vs test regression errors — informal, not a statistical test.

Train-set metrics are **in-sample** for tree/ensemble models, so train error can look unrealistically
good; validation and test are both out-of-sample relative to the fitted model’s training rows.
"""


def diagnose_regression_fit(
    *,
    train_rmse: float,
    val_rmse: float,
    test_rmse: float,
    train_r2: float,
    val_r2: float,
    test_r2: float,
) -> str:
    train_val_ratio = val_rmse / max(train_rmse, 1e-12)
    val_test_rel = abs(val_rmse - test_rmse) / max(test_rmse, 1e-12)
    r2_gap = float(train_r2 - val_r2)

    parts: list[str] = []

    both_weak = train_r2 < 0.28 and val_r2 < 0.28 and abs(train_r2 - val_r2) < 0.12
    if both_weak:
        parts.append(
            "underfitting signal: weak R2 on train and validation with little gap between them"
        )
    elif train_val_ratio > 1.8 and r2_gap > 0.18:
        if val_test_rel < 0.22:
            parts.append(
                "overfitting signal: much worse on validation than in-sample train, "
                "while validation ≈ test (typical generalization gap)"
            )
        else:
            parts.append(
                "large train→validation gap; test disagrees with validation — check split variance "
                "or target noise"
            )
    elif val_test_rel < 0.12 and val_r2 < 0.15:
        parts.append(
            "validation and test align but scores are poor — weak model or hard target, "
            "not necessarily 'overfit'"
        )
    else:
        parts.append("no extreme under/overfit pattern by these rules; compare RMSE/R2 across splits")

    parts.append(
        f"[train val test RMSE ratio≈1:{train_val_ratio:.2f}:{test_rmse / max(train_rmse, 1e-12):.2f}; "
        f"val vs test RMSE rel diff={val_test_rel:.2f}; trainR2−valR2={r2_gap:.3f}]"
    )
    return " ".join(parts)
