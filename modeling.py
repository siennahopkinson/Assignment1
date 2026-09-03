def chronological_split(model_data, train_fraction=0.8):
    """Split model data chronologically into train and test sets.

    Sorts the data by timestamp and assigns the earliest portion to
    training and the most recent portion to testing, so that each model
    is evaluated on conditions occurring after the training period
    rather than on data drawn from the same time window as training.
    This avoids the data leakage found when using a random split.

    Args:
        model_data (pandas.DataFrame): Cleaned model data as returned by
            clean_model_data, containing a 'datetime' column.
        train_fraction (float): Proportion of rows, by time order, to
            assign to the training set. Defaults to 0.8.

    Returns:
        tuple: (train, test), two pandas.DataFrame objects containing the
            earliest train_fraction of rows and the remaining, most
            recent rows, respectively.
    """
    model_data_sorted = model_data.sort_values("datetime")
    split_index = int(len(model_data_sorted) * train_fraction)
    train = model_data_sorted.iloc[:split_index]
    test = model_data_sorted.iloc[split_index:]
    return train, test

from sklearn.linear_model import LinearRegression


def train_linear_regression(X_train, y_train):
    """Fits a Linear Regression model on training features and targets.

    Initializes and trains an sklearn LinearRegression instance using the
    provided training dataset.

    Args:
        X_train (pandas.DataFrame or numpy.ndarray): Feature matrix for training.
        y_train (pandas.Series or numpy.ndarray): Target vector for training.

    Returns:
        sklearn.linear_model.LinearRegression: The fitted linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor


def train_decision_tree(X_train, y_train, max_depth=5):
    """Fits a Decision Tree regressor on training features and targets.

    Initializes and trains an sklearn DecisionTreeRegressor instance with a
    fixed maximum depth to control model complexity and help prevent
    overfitting.

    Args:
        X_train (pandas.DataFrame or numpy.ndarray): Feature matrix for
            training.
        y_train (pandas.Series or numpy.ndarray): Target vector for training.
        max_depth (int, optional): Maximum depth of the decision tree.
            Defaults to 5.

    Returns:
        sklearn.tree.DecisionTreeRegressor: The fitted decision tree model.
    """
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluates a fitted model on test data using R² and RMSE.

    Generates predictions for the test dataset and calculates coefficient of
    determination (R²) and Root Mean Squared Error (RMSE) performance metrics.

    Args:
        model: Fitted estimator supporting a predict method.
        X_test (pandas.DataFrame or numpy.ndarray): Feature matrix for testing.
        y_test (pandas.Series or numpy.ndarray): Ground truth target values
            for testing.

    Returns:
        tuple: (r2, rmse) — two floats giving the R² and RMSE of the model's
            predictions on the test set.
    """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return r2, rmse
