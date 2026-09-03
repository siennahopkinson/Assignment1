import pandas as pd
import pytest
from modeling import evaluate_model, train_linear_regression, chronological_split


def test_chronological_split():
    """Tests that chronological_split partitions data sequentially by time.

    Creates a synthetic dataframe with sequential dates to verify that the
    split yields the correct train/test row counts and that all test timestamps
    strictly follow train timestamps.
    """
    data = {
        "datetime": pd.date_range(start="2026-01-01", periods=10, freq="D"),
        "value": range(10),
    }
    df = pd.DataFrame(data)
    train_df, test_df = chronological_split(df, train_fraction=0.8)
    assert len(train_df) == 8
    assert len(test_df) == 2
    assert train_df["datetime"].max() < test_df["datetime"].min()


def test_train_linear_regression_and_evaluate():
    """Tests training and evaluating a linear regression model on synthetic linear data.

    Creates a noise-free deterministic target vector to verify that fitting and
    evaluation via train_linear_regression and evaluate_model yields an R² near 1.0.
    """
    X = pd.DataFrame(
        {
            "x1": list(range(20)),
            "x2": [i * 0.5 for i in range(20)],
        }
    )
    y = 2 * X["x1"] + 3 * X["x2"]
    model = train_linear_regression(X, y)
    r2, rmse = evaluate_model(model, X, y)
    assert r2 == pytest.approx(1.0, abs=1e-6)
    assert rmse == pytest.approx(0.0, abs=1e-6)
