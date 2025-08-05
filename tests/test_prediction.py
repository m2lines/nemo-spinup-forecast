import pytest
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from lib.forecast import Predictions
from main_forecast import prepare


def test_defineGP_returns_gaussian_process_regressor():
    """Check that defineGP returns a GaussianProcessRegressor instance."""
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
    """Check that Forecast returns predictions, standard deviations, and metrics for valid inputs."""
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
    """Check that metrics are None when forecasting with no test data."""
    pred = setup_prediction_class
    train_len = len(pred)
    steps = 0

    y_hats, y_stds, metrics = pred.Forecast(train_len, steps)

    # All metrics entries should be None
    assert all(m is None for m in metrics), (
        "Metrics should be None when train_len equals data length"
    )


# Use the setup_prediction_class fixture for real data


@pytest.mark.parametrize(
    "setup_prediction_class",
    [("toce", "DINO_1y_grid_T.nc"), ("ssh", "DINO_1m_To_1y_grid_T.nc")],
    indirect=True,
)
def test_forecast_ts_valid_input(setup_prediction_class):
    """Check that forecast_ts returns correct forecast arrays and metrics for valid time-series inputs."""
    sim = setup_prediction_class
    # Use first component, half of the data for training
    n = 1
    train_len = len(sim) // 2
    steps = len(sim) // 2

    y_hat, y_hat_std, metrics = sim.forecast_ts(n, train_len, steps)

    # Forecast and std arrays should be NumPy arrays of expected length
    assert isinstance(y_hat, np.ndarray)
    assert isinstance(y_hat_std, np.ndarray)
    assert y_hat.shape == (len(sim) + steps,)
    assert y_hat_std.shape == y_hat.shape

    # Since train_len < len(sim), metrics should be returned
    assert metrics is not None
    assert isinstance(metrics, dict)


@pytest.mark.parametrize(
    "setup_prediction_class",
    [("ssh", "DINO_1m_To_1y_grid_T.nc"), ("toce", "DINO_1y_grid_T.nc")],
    indirect=True,
)
def test_forecast_ts_no_test_data_no_steps(setup_prediction_class):
    """Check that metrics are None when forecasting zero steps with no test data."""
    sim = setup_prediction_class
    n = 1
    train_len = len(sim)
    steps = 0

    y_hat, y_hat_std, metrics = sim.forecast_ts(n, train_len, steps)

    # No test data => metrics must be None
    assert metrics is None

    # Forecast arrays should still be returned
    assert isinstance(y_hat, np.ndarray)
    assert isinstance(y_hat_std, np.ndarray)
    assert y_hat.shape == (len(sim) + steps + 1,)
    assert y_hat_std.shape == y_hat.shape


@pytest.mark.parametrize(
    "setup_prediction_class",
    [("ssh", "DINO_1m_To_1y_grid_T.nc"), ("toce", "DINO_1y_grid_T.nc")],
    indirect=True,
)
def test_forecast_ts_no_test_data_with_steps(setup_prediction_class):
    """Check that metrics are None when forecasting multiple steps with no test data."""
    sim = setup_prediction_class
    n = 1
    train_len = len(sim)
    steps = 20

    y_hat, y_hat_std, metrics = sim.forecast_ts(n, train_len, steps)

    # No test data => metrics must be None
    assert metrics is None

    # Forecast arrays should still be returned
    assert isinstance(y_hat, np.ndarray)
    assert isinstance(y_hat_std, np.ndarray)
    assert y_hat.shape == (len(sim) + steps,)
    assert y_hat_std.shape == y_hat.shape


@pytest.mark.parametrize(
    "setup_prediction_class",
    [
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_prepare_standard_case(setup_prediction_class):
    """Check that prepare splits the series, normalizes training data, and returns expected arrays."""
    sim: Predictions = setup_prediction_class
    # pick component 1, train_len < len(data)
    n = 1
    train_len = len(sim.data) // 2
    steps = 20

    mean, std, y_train, y_test, x_train, x_pred = sim.prepare(n, train_len, steps)

    # Compute expected mean and std from raw series
    series = sim.data[f"{sim.var}-{n}"]
    expected_mean = np.nanmean(series.iloc[:train_len])
    expected_std = np.nanstd(series.iloc[:train_len])
    assert np.isclose(mean, expected_mean)
    assert np.isclose(std, expected_std)

    # y_train should be normalized: (raw - mean) / (2*std)
    raw_train = series.iloc[:train_len].values
    expected_y_train = (raw_train - expected_mean) / (2 * expected_std)
    assert np.allclose(y_train.flatten(), expected_y_train)

    # y_test should be raw values for remaining points
    raw_test = series.iloc[train_len:].values
    assert np.allclose(y_test.flatten(), raw_test)

    # x_train should be linspace
    expected_x_train = np.linspace(0, 1, train_len).reshape(-1, 1)
    assert np.allclose(x_train, expected_x_train)

    # x_pred should span entire series length + steps, same as original index (assuming uniform pas)
    pas = x_train[1, 0] - x_train[0, 0]
    total_len = len(series) + steps
    expected_x_pred = np.arange(0, total_len * pas, pas).reshape(-1, 1)
    assert np.allclose(x_pred, expected_x_pred)


@pytest.mark.parametrize(
    "setup_prediction_class",
    [("ssh", "DINO_1m_To_1y_grid_T.nc"), ("toce", "DINO_1y_grid_T.nc")],
    indirect=True,
)
def test_predictions_reconstruct(setup_prediction_class):
    """Check that reconstruct rebuilds the time series from PCA components with correct shape."""
    # setup prediction class
    pred = setup_prediction_class

    steps = 20

    # Forecast specified number of steps
    y_hat, y_hat_std, metrics = pred.Forecast(len(pred), steps)

    # Reconstruct with n predicted components
    n = len(pred.info["pca"].components_)
    reconstructed_preds = pred.reconstruct(y_hat, n, begin=len(pred))

    # Check reconstruction produces the correct shape
    assert reconstructed_preds.shape == (steps,) + tuple(pred.info["shape"])
