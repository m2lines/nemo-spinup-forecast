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
    def init_technique(self):
        """
        Initialize the forecasting technique (model, parameters, etc.).
        """
        pass

    @abstractmethod
    def apply_forecast(self, y_train, x_train, x_pred):
        """
        Fit the model on training data and predict on new data.
        """
        pass


class GaussianProcessForecaster(BaseForecaster):
    def init_technique(self):
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

    def apply_forecast(self, y_train, x_train, x_pred):
        """
        Fit a recursive forecaster (skforecast) using the Gaussian Process Regressor.

        Args:
            y_train (array-like): Training target series.
            x_train (array-like): Training features (if applicable).
            x_pred (array-like): Future features for which to forecast.

        Returns:
            tuple: y_hat (ndarray), y_hat_std (ndarray)
        """
        # mean, std, y_train, y_test, x_train, x_pred = *dataain, x_pred = data

        #####################
        gp = self.init_technique()

        gp.fit(x_train, y_train)

        y_hat, y_hat_std = gp.predict(x_pred, return_std=True)
        return y_hat, y_hat_std


class GaussianProcessRecursiveForecaster(GaussianProcessForecaster):
    def init_technique(self):
        return super().init_technique()

    def apply_forecast(self, y_train, x_train, x_pred):
        # mean, std, y_train, y_test, x_train, x_pred = *dataain, x_pred = data

        # Initialize the GP regressor
        gp = self.init_technique()

        # Example: ForecasterRecursive from skforecast
        forecaster = ForecasterRecursive(
            regressor=gp,
            lags=10,
            window_features=RollingFeatures(stats=["mean"], window_sizes=10),
        )

        # Fit on the training series
        forecaster.fit(pd.Series(y_train))

        # Predict the next 30 steps (this is arbitrary; adjust as needed)
        y_hat = forecaster.predict(steps=30)

        # For demonstration, return a dummy standard deviation
        y_hat_std = y_hat  # TODO: Calculate the standard deviation for each y_hat

        # If you prefer directly using gp without skforecast:
        # gp.fit(x_train, y_train)
        # y_hat, y_hat_std = gp.predict(x_pred, return_std=True)

        return y_hat, y_hat_std
