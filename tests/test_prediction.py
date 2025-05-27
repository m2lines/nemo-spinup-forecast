import pytest
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from lib.forecast import Predictions
from main_forecast import prepare


def test_defineGP_returns_gaussian_process_regressor():
    gp = Predictions.defineGP()
    assert isinstance(gp, GaussianProcessRegressor)


# Use real prepared data via the setup_prediction_class fixture
@pytest.mark.parametrize(
    "setup_prediction_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_forecast_valid_input(setup_prediction_class):
    """
    Valid Input - Standard Case:
    Provide a valid train_len and steps. Check that it returns forecasts, standard deviations, and metrics.
    """
    pred = setup_prediction_class
    # Choose train_len slightly less than full length to generate test data
    train_len = len(pred) - 5
    steps = 3

    y_hats, y_stds, metrics = pred.Forecast(train_len, steps)

    # Check types
    assert isinstance(y_hats, pd.DataFrame), "y_hats should be a DataFrame"
    assert isinstance(y_stds, pd.DataFrame), "y_stds should be a DataFrame"
    assert isinstance(metrics, list), "metrics should be a list"

    # Check shapes
    n_series = pred.data.shape[1]
    assert y_hats.shape == (len(pred) + steps, n_series)
    assert y_stds.shape == (len(pred) + steps, n_series)
    assert len(metrics) == n_series

    # Since train_len < len(data), metrics should be dicts
    for m in metrics:
        assert isinstance(m, dict), (
            "Each metrics entry should be a dict when test data is present"
        )
        # Check presence of expected metric keys
        for key in ["ma_true", "ma_pred", "mse", "dist_max", "std_true"]:
            assert key in m, f"Metric '{key}' missing in metrics dict"


@pytest.mark.parametrize(
    "setup_prediction_class",
    [("toce", "DINO_1y_grid_T.nc"), ("ssh", "DINO_1m_To_1y_grid_T.nc")],
    indirect=True,
)
def test_forecast_no_test_data(setup_prediction_class):
    """
    Forecast with No Test Data (train_len == len(data)):
    Check that metrics are None when no test data.
    """
    pred = setup_prediction_class
    train_len = len(pred)
    steps = 0

    y_hats, y_stds, metrics = pred.Forecast(train_len, steps)

    # All metrics entries should be None
    assert all(m is None for m in metrics), (
        "Metrics should be None when train_len equals data length"
    )
