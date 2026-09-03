"""End-to-end pipeline for predicting PMV from environmental sensor data.

Loads the raw Env and PMV CSV files, merges and cleans them, splits the
result chronologically, trains both candidate models (linear regression
and a depth-5 decision tree), and prints their evaluation metrics.

Run this script directly to reproduce the project's reported results:
    python main.py
"""

from data_loading import load_env_data, load_pmv_data
from preprocessing import merge_env_pmv, filter_overlap, clean_model_data
from modeling import (
    chronological_split,
    train_linear_regression,
    train_decision_tree,
    evaluate_model,
)

ENV_FILEPATH = "Env_data_MSI_Lab - 30-07-2025_19-12-2025.csv"
PMV_FILEPATH = "PMV_07_25 - 04_26.csv"

FEATURES = ["Pressure (mb)", "Temperature (°C)", "Humidity (RH %)", "CO2 (ppm)"]
TARGET = "col7"


def main():
    """Runs the full pipeline and prints evaluation results for both models."""
    df_env = load_env_data(ENV_FILEPATH)
    df_pmv = load_pmv_data(PMV_FILEPATH)

    merged = merge_env_pmv(df_env, df_pmv)
    merged_overlap = filter_overlap(merged, df_env)
    model_data = clean_model_data(merged_overlap)

    train, test = chronological_split(model_data)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    lr_model = train_linear_regression(X_train, y_train)
    lr_r2, lr_rmse = evaluate_model(lr_model, X_test, y_test)
    print(f"Linear Regression — R²: {lr_r2:.4f}, RMSE: {lr_rmse:.4f}")

    dt_model = train_decision_tree(X_train, y_train, max_depth=5)
    dt_r2, dt_rmse = evaluate_model(dt_model, X_test, y_test)
    print(f"Decision Tree (depth=5) — R²: {dt_r2:.4f}, RMSE: {dt_rmse:.4f}")


if __name__ == "__main__":
    main()
