from abc import ABC, abstractmethod
from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import RollingFeatures

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ExpSineSquared,
    WhiteKernel,
    DotProduct,
)  # , Matérn
import pandas as pd


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasters.
    """

    @abstractmethod
    def apply_forecast(self, y_train, x_train, x_pred):
        """
        Fit the model on training data and predict on new data.
        """
        pass


class DirectForecaster(BaseForecaster):
    """Forecaster that uses a regressor directly for predictions."""

    def __init__(self, regressor):
        """
        Initialize with a scikit-learn compatible regressor.

        Args:
            regressor: A scikit-learn compatible regressor with fit and predict methods
        """
        self.regressor = regressor

    def apply_forecast(self, y_train, x_train, x_pred):
        """
        Fit the regressor and make direct predictions.

        Args:
            y_train (array-like): Training target series
            x_train (array-like): Training features
            x_pred (array-like): Features for prediction

        Returns:
            tuple: y_hat (ndarray), y_hat_std (ndarray) if available, otherwise (y_hat, None)
        """
        self.regressor.fit(x_train, y_train)

        y_hat, y_hat_std = self.regressor.predict(x_pred, return_std=True)


class RecursiveForecaster(BaseForecaster):
    """Forecaster that uses recursive strategy for multi-step predictions."""

    def __init__(self, regressor, lags=10, window_size=10, steps=30):
        """
        Initialize with a scikit-learn compatible regressor and forecasting parameters.

        Args:
            regressor: A scikit-learn compatible regressor
            lags (int): Number of lags to use for forecasting
            window_size (int): Size of the window for rolling features
            steps (int): Number of steps to forecast
        """
        self.regressor = regressor
        self.lags = lags
        self.window_size = window_size
        self.steps = steps

    def apply_forecast(self, y_train, x_train=None, x_pred=None):
        """
        Fit a recursive forecaster using the provided regressor.

        Args:
            y_train (array-like): Training target series
            x_train (array-like): Not used directly in recursive forecasting,
            x_pred (array-like): Not used directly in recursive forecasting

        Returns:
            tuple: y_hat (ndarray), y_hat_std (ndarray) or (y_hat, None) if std not available
        """
        # Create rolling features for the forecaster
        window_features = RollingFeatures(stats=["mean"], window_sizes=self.window_size)

        # Initialize the forecaster with the regressor
        forecaster = ForecasterRecursive(
            regressor=self.regressor,
            lags=self.lags,
            window_features=window_features,
        )

        # Fit on the training series
        forecaster.fit(pd.Series(y_train))

        # Predict the specified steps
        y_hat = forecaster.predict(steps=self.steps)

        # Recursive forecaster does not return standard deviation for each point
        # Use y_hat as a placeholder for std
        y_hat_std = y_hat

        return y_hat, y_hat_std


def create_gp_regressor(
    data_length=45,
    trend_scale=0.1,
    irregularities_scale=10.0,
    noise_scale=2.0,
    optimize_restarts=8,
    random_seed=42,
):
    """
    Create a Gaussian Process regressor with basic parameter configuration.

    Parameters
    ----------
    data_length : int, default=45
        Length of the data, used to scale kernel parameters
    trend_scale : float, default=0.1
        Scaling factor for the trend kernel
    irregularities_scale : float, default=10.0
        Scaling factor for the irregularities kernel
    noise_scale : float, default=2.0
        Scaling factor for the noise kernel
    optimize_restarts : int, default=8
        Number of restarts for the optimizer
    random_seed : int, default=42
        Random seed for reproducibility

    Returns
    -------
    GaussianProcessRegressor
        Configured Gaussian Process regressor
    """
    # Scale factor based on data length
    scale_factor = 5.0 / data_length

    # Long-term trend kernel (linear behavior)
    trend_kernel = trend_scale * DotProduct(sigma_0=0.0)

    # Periodic kernel for irregularities
    irregularities_kernel = irregularities_scale * ExpSineSquared(
        length_scale=scale_factor, periodicity=scale_factor
    )

    # Noise kernel
    noise_kernel = noise_scale * WhiteKernel(noise_level=1.0)

    # Combine kernels
    kernel = irregularities_kernel + noise_kernel + trend_kernel

    return GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=optimize_restarts,
        random_state=random_seed,
    )


GaussianProcessRecursiveForecaster = RecursiveForecaster(
    create_gp_regressor(), lags=10, window_size=10, steps=30
)
GaussianProcessForecaster = DirectForecaster(create_gp_regressor())
