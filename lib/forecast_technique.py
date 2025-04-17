from abc import ABC, abstractmethod
from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import RollingFeatures

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ExpSineSquared,
    RationalQuadratic,
    WhiteKernel,
    DotProduct,
)  # , Matérn
import numpy as np
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

        # Check if the regressor supports returning standard deviation
        if (
            hasattr(self.regressor, "predict")
            and "return_std" in self.regressor.predict.__code__.co_varnames
        ):
            y_hat, y_hat_std = self.regressor.predict(x_pred, return_std=True)
            return y_hat, y_hat_std
        else:
            y_hat = self.regressor.predict(x_pred)
            return y_hat, None


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


def create_gp_regressor():
    """
    Define Gaussian Process regressor with specified kernel. We use :
        - a long term trend kernel that contains a Dot Product with sigma_0 = 0, for the linear behaviour.
        - an irregularities_kernel for periodic patterns CHANGER 5/45 1/len(data)?
        - a noise_kernel
    We also set a n_restarts_optimizer to optimize hyperparameters

    Returns:
        GaussianProcessRegressor: The Gaussian Process regressor.
    """
    # Long-term trend kernel (linear behavior)
    long_term_trend_kernel = 0.1 * DotProduct(
        sigma_0=0.0
    )  # + 0.5*RBF(length_scale=1/2)# +

    # Periodic kernel for irregularities
    irregularities_kernel = (
        10 * ExpSineSquared(length_scale=5 / 45, periodicity=5 / 45)
    )  # 0.5**2 * RationalQuadratic(length_scale=5.0, alpha=1.0) + 10 * ExpSineSquared(length_scale=5.0)

    # Noise kernel
    noise_kernel = 2 * WhiteKernel(
        noise_level=1
    )  # 0.1**2 * RBF(length_scale=0.01) + 2*WhiteKernel(noise_level=1)

    # Combine kernels
    kernel = irregularities_kernel + noise_kernel + long_term_trend_kernel

    return GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=8, random_state=42
    )


GaussianProcessRecursiveForecaster = RecursiveForecaster(
    create_gp_regressor(), lags=10, window_size=10, steps=30
)
GaussianProcessForecaster = DirectForecaster(create_gp_regressor())
