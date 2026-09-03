markdown
# PMV Thermal Comfort Prediction from Environmental Sensor Data

## Purpose

This project predicts the Predicted Mean Vote (PMV) thermal comfort index
from ambient environmental sensor readings (Pressure, Temperature, Humidity,
and CO2) collected in the Monash Smart Infrastructure (MSI) Lab. The goal is
to allow thermal comfort to be estimated from existing environmental sensors
without needing dedicated comfort-sensing equipment in every room.

Two models were compared: a Linear Regression baseline and a Decision Tree
regressor. Model selection was based on honest, chronologically validated
performance rather than a naive random train/test split, which was found to
produce misleadingly optimistic results due to data leakage (see the project
report's Validation section for full details).

## Repository structure

.
├── Assignment_1.ipynb # Original exploratory notebook (data auditing, EDA)
├── data_loading.py # Functions to load and parse the raw CSV files
├── preprocessing.py # Functions to merge, filter, and clean the data
├── modeling.py # Functions to split, train, and evaluate models
├── main.py # End-to-end script reproducing the full pipeline
├── conftest.py # Empty file enabling pytest to import project modules
├── tests/
│ ├── test_preprocessing.py # Unit tests for preprocessing.py
│ └── test_modeling.py # Unit tests for modeling.py
├── TEST_REPORT.md # Summary of test coverage and pytest results
├── LICENSE # MIT License
└── README.md


## Environment and dependencies

This project was developed and run in Google Colab (Python 3.13). It requires:

- pandas
- numpy
- scikit-learn
- pytest (for running the test suite)

These are all pre-installed in Colab. To run locally instead, install them with:

pip install pandas numpy scikit-learn pytest


## Data

This project uses two CSV files from the MSI Lab dataset, obtained from the
unit's shared course materials:

- `Env_data_MSI_Lab - 30-07-2025_19-12-2025.csv` — environmental sensor readings
- `PMV_07_25 - 04_26.csv` — PMV records

These files are not included in this repository due to their size. To
reproduce the results, place both files in the same directory as `main.py`.

## Usage

To run the full pipeline and reproduce the reported results:

python main.py


This loads both CSV files, merges and cleans them, splits the data
chronologically, trains both the Linear Regression and Decision Tree models,
and prints each model's R² and RMSE on the held-out test set. Expected
output is approximately R² = 0.190 for Linear Regression and R² = 0.163 for
the depth-5 Decision Tree.

## Running the tests

pytest tests/ -v


All four tests should pass. See `TEST_REPORT.md` for a description of what
each test verifies.

## License

This project is licensed under the MIT License — see `LICENSE` for details.
