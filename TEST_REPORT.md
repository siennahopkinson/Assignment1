Test Report

Four automated tests were written using pytest to verify the correctness of the core data-processing and modeling functions used in this project. Each test uses small, synthetic datasets with a known correct answer, rather than the full real sensor data, so that tests run quickly and do not depend on the large CSV files being present.

test_clean_model_data_drops_missing_required_columns (tests/test_preprocessing.py) — verifies that clean_model_data correctly drops rows missing any of the required environmental or PMV columns, using a small dataframe where the correct surviving row is known in advance. This matters because incorrect handling of missing data would silently corrupt the dataset used for model training.

test_filter_overlap (tests/test_preprocessing.py) — verifies that filter_overlap correctly restricts a dataset to only the rows falling within a reference dataset's time range, including the boundary dates themselves. This matters because the model's usable data domain depends entirely on this filtering step being correct.

test_chronological_split (tests/test_modeling.py) — verifies that chronological_split produces the correct train/test row counts for a given split fraction, and that every timestamp in the test set falls after every timestamp in the training set. This is the most important test in the project, since it directly checks the property that fixed the data leakage problem discovered during validation: if this function were broken, the project's central engineering finding would no longer hold.

test_train_linear_regression_and_evaluate (tests/test_modeling.py) — verifies that train_linear_regression and evaluate_model work correctly together, by fitting on a noise-free synthetic dataset with a known linear relationship and confirming the resulting R² is effectively 1.0 and RMSE is effectively 0. This confirms the model-fitting and evaluation functions are implemented correctly, independent of how well the real environmental data fits a linear model.

All four tests pass:

============================= test session starts ==============================
platform linux -- Python 3.13.15, pytest-8.4.2, pluggy-1.6.0 -- /usr/bin/python3
collected 4 items

tests/test_modeling.py::test_chronological_split PASSED                  [ 25%]
tests/test_modeling.py::test_train_linear_regression_and_evaluate PASSED [ 50%]
tests/test_preprocessing.py::test_clean_model_data_drops_missing_required_columns PASSED [ 75%]
tests/test_preprocessing.py::test_filter_overlap PASSED                  [100%]

============================== 4 passed in 2.11s ===============================
