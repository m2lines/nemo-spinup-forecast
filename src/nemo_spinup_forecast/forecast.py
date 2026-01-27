# Adapted from code by Maud Tissot (Spinup-NEMO)
# Original source: https://github.com/maudtst/Spinup-NEMO
# Licensed under the MIT License
#
# Modifications in this version by ICCS, 2025
import os
import pickle
import random
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

sys.path.insert(0, "../")

warnings.filterwarnings("ignore")


def load_ts(file_path, var):
    """
    Load time series data from the file where are saved the prepared simulations.

    This function is used the get the prepared data info in order to
    instantiate a prediction class.

    Parameters
    ----------
    file_path : str
        The path to the file containing the time series data.
    var : str
        The variable to be loaded.

    Returns
    -------
    tuple
        df : pandas.DataFrame
            DataFrame containing the time series data.
        dico : dict
            A dictionary containing all informations on
            the simu (pca, mean, std, time_dim)...
    """
    dico = np.load(file_path + f"/{var}.npz", allow_pickle=True)
    dico = {key: dico[key] for key in dico}
    df = pd.DataFrame(
        dico["ts"], columns=[f"{var}-{i + 1}" for i in range(np.shape(dico["ts"])[1])]
    )
    with open(file_path + f"/pca_{var}", "rb") as fp:
        dico["pca"] = pickle.load(fp)  # noqa: S301
    return df, dico


##################################
##                              ##
##   LOAD AND PREPARE A SIMU    ##
##                              ##
##################################


class Simulation:
    """
    A class for loading and preparing simulation data.

    Attributes
    ----------
    path : str
        The path to the simulation data.
    term : str
        The term for the simulation.
    files : list
        List of files related to the simulation.
    start : int
        The start index for data slicing.
    end : int
        The end index for data slicing.
    ye : bool
        Flag indicating whether to use ye or not.
    comp : float
        The comp value for the simulation.
    len : int
        The length of the simulation.
    desc : dict
        A dictionary containing descriptive statistics of the simulation data.
    time_dim : str
        The name of the time dimension.
    y_size : int
        The size of the y dimension.
    x_size : int
        The size of the x dimension.
    z_size : int or None
        The size of the z dimension, if available.
    shape : tuple
        The shape of the simulation data.
    simulation : xarray.DataArray
        The loaded simulation data
    """

    def __init__(
        self,
        path,
        term,
        filename=None,
        start=0,
        end=None,
        comp=None,
        ye=True,
        dimensionality_reduction=None,
    ):  # choose jobs 3 if 2D else 1
        """
        Initialize Simulation with specified parameters.

        Parameters
        ----------
        path : str
            The path to the simulation data.
        term : str
            The term for the simulation.
        filename : str, optional
            Filename pattern to select, by default ``None``.
        start : int, optional
            The start index for data slicing. Defaults to ``0``.
        end : int, optional
            The end index for data slicing. Defaults to ``None``.
        comp : float, optional
            The comp value for the simulation. Defaults to ``0.9``.
            Percentage of explained variance.
        ye : bool, optional
            Flag indicating whether to use ye or not. Defaults to ``True``.
        dimensionality_reduction : object, optional
            Callable/class used for dimensionality reduction, by default ``None``.
        ssca : bool, optional
            Flag indicating whether ssca is used. Defaults to ``False``.
            Not used in this class
        """
        self.path = path
        self.term = term
        self.filename = filename
        self.files = Simulation.get_data(path, term, filename)
        self.start = start
        self.end = end
        self.ye = ye
        self.comp = comp
        self.len = 0
        self.desc = {}
        self.dimensionality_reduction = dimensionality_reduction(comp)
        self.get_attributes()  # self time_dim, y_size, x_size,
        self.get_simu()  # self simulation , desc {"mean","std","min","max"}

    #### Load files and dimensions info ###

    @staticmethod
    def get_data(path, term, filename):  # noqa: ARG004
        """
        Get the files related to the simulation in the right directory.

        Parameters
        ----------
        path : str
            The path to the simulation data.
        term : str
            The term for the simulation.
            zos    -> sea surface height (also ssh) - (t,y,z)
            so     -> salinity - (t,z,y,x)
            thetao -> temperature - (t,z,y,x)
        filename : str
            Filename pattern to filter.

        Returns
        -------
        grid : list
            List of files related to the simulation.
        """
        grid = []
        for file_name in sorted(os.listdir(path)):
            if filename in file_name:
                grid.append(path + "/" + file_name)
        return grid

    def get_attributes(self):
        """Get attributes of the simulation data."""
        array = xr.open_dataset(
            self.files[-1], decode_times=False, chunks={"time": 200, "x": 120}
        )
        self.time_dim = "time_counter"  # TODO: Specify time dimension in config file

        self.y_size = array.sizes["y"]
        self.x_size = array.sizes["x"]
        if "deptht" in array[self.term].dims:
            self.z_size = array.sizes["deptht"]
            self.shape = (self.z_size, self.y_size, self.x_size)
        elif "olevel" in array[self.term].dims:
            self.z_size = array.sizes["olevel"]
            self.shape = (self.z_size, self.y_size, self.x_size)
        else:
            self.z_size = None
            self.shape = (self.y_size, self.x_size)
        # self.getSSCA(array)

    #### Load simulation ###

    def get_simu(self):
        """Load simulation data."""
        array = [
            self.load_file(fp) for fp in self.files if self.len < (self.end or np.inf)
        ]
        array = xr.concat(array, self.time_dim)
        self.desc = {
            "mean": np.nanmean(array),
            "std": np.nanstd(array),
            "min": np.nanmin(array),
            "max": np.nanmax(array),
        }
        self.simulation = array

        del array

    def load_file(self, file_path):
        """
        Load simulation data from a file.

        Stop when the imported simulation date is superior to the attirbute end.
        This is why we cannot use parallelisation to import simulations.

        Parameters
        ----------
        file_path : str
            The path to the file containing the simulation data.

        Returns
        -------
        xarray.DataArray
            The loaded simulation data.
        """
        array = xr.open_dataset(
            file_path, decode_times=False, chunks={"time": 200, "x": 120}
        )
        array = array[self.term]
        self.len = self.len + array.sizes[self.time_dim]

        return array.load()

    #########################
    #  get_simulation_data simulation   #
    #########################

    def get_simulation_data(self, stand=True):
        """
        Prepare the simulation data selecting indices from start to end.

        Update length, obtain statistics, standardize if specified.

        Parameters
        ----------
        stand : bool, optional
            Flag indicating whether to standardize the simulation data.
            Defaults to ``True``.
        """
        if self.end is not None:
            self.simulation = self.simulation[self.start : self.end]
        else:
            self.simulation = self.simulation[self.start :]
        self.len = np.shape(self.simulation)[0]
        # self.removeClosedSeas()
        self.desc.update(
            {
                "mean": np.nanmean(self.simulation),
                "std": np.nanstd(self.simulation),
                "min": np.nanmin(self.simulation),
                "max": np.nanmax(self.simulation),
            }
        )
        if stand:
            self.standardize()
        self.simulation = self.simulation.values

    def get_ssca(self, array):
        """
        Extract the seasonality data from the simulation.

        Not used : we import yearly data

        Parameters
        ----------
        array : xarray.Dataset
            The last dataset containing simulation data in the simulation file.
        """
        array = array[self.term].values
        n = np.shape(array)[0] // 12 * 12
        array = array[-n:]
        ssca = np.array(array).reshape(
            n // 12, 12, *self.shape
        )  # np.array(array[self.term])
        ssca = np.mean(ssca, axis=0)
        ssca_extended = np.tile(ssca, (n // 12, 1, 1))
        self.desc["ssca"] = ssca
        if not self.ye:
            self.simulation = array - ssca_extended

    def remove_closed_seas(self):
        """
        Remove closed seas from the simulation data.

        Not used : we don't have the specific mask to fill with predictions
        """
        array = self.simulation
        y_range = [
            slice(240, 266),
            slice(235, 276),
            slice(160, 201),
        ]  # mer noir, grands lacs, lac victoria
        x_range = [slice(195, 213), slice(330, 351), slice(310, 325)]
        for y, x in zip(y_range, x_range, strict=True):
            array = array.where(
                (array["x"] < x.start)
                | (array["x"] >= x.stop)
                | (array["y"] < y.start)
                | (array["y"] >= y.stop),
                drop=True,
            )
        self.simulation = array

    def standardize(self):
        """Standardize the simulation data."""
        self.simulation = (self.simulation - self.desc["mean"]) / (2 * self.desc["std"])

    ##################
    #  Compute PCA   #
    ##################

    def decompose(self):
        """Apply Principal Component Analysis (PCA) to the simulation data."""
        self.dimensionality_reduction.set_from_simulation(self)

        self.components, self.pca, self.bool_mask = (
            self.dimensionality_reduction.decompose(self.simulation, self.len)
        )

    def get_component(self, n):
        """
        Get principal component map for the specified component.

        Parameters
        ----------
        n : int
            component used for reconstruction.

        Returns
        -------
        ndarray
            The principal component map.
        """
        map_ = self.dimensionality_reduction.get_component(n)

        return map_

    def get_num_components(self):
        """Get the number of components used in the dimensionality reduction.

        Returns
        -------
        int
            Number of components.
        """
        return self.dimensionality_reduction.get_num_components()

    ###################
    #   Reconstruct   #
    ###################

    def reconstruct(
        self,
        predictions,
        n,
        info,
        begin=0,
    ):
        """
        Reconstruct the time series data from predictions.

        Parameters
        ----------
        predictions : pandas.DataFrame
            parallel_forecasted values for each component.
        n : int
            Number of components to consider for reconstruction.
        info : dict
            Additional information needed by the reconstruction routine.
        begin : int, optional
            Starting index for reconstruction. Defaults to ``0``.

        Returns
        -------
        ndarray
            Reconstructed time series data.
        """
        self.int_mask, ts_array = self.dimensionality_reduction.reconstruct_predictions(
            predictions, n, info, begin
        )

        return ts_array

    ############################### NOT USED IN THE MAIN.PY ######################
    def error(self, n):
        """Compute an error metric (e.g., RMSE) between reconstructions and truth."""
        reconstruction, rmse_values, rmse_map = self.dimensionality_reduction.error(n)
        return reconstruction, rmse_values, rmse_map

    ##############################################################################

    ##################
    #   Save in db   #
    ##################

    def make_dico(self):
        """
        Create a dictionary simulation data and information.

        The dictionary contains simulation data, descriptive statistics,
        and other relevant information.

        Returns
        -------
        dico : dict
            A dictionary containing simulation data and information.
        """
        dico = {}
        dico["ts"] = self.components.tolist()
        dico["mask"] = self.bool_mask
        dico["desc"] = self.desc
        dico["cut"] = self.start
        dico["x_size"] = self.x_size
        dico["y_size"] = self.y_size
        if self.z_size is not None:
            dico["z_size"] = self.z_size
        dico["shape"] = self.shape
        return dico

    def save(self, file_path, term):
        """
        Save the simulation data and information to files.

        Parameters
        ----------
        file_path : str
            The path to the directory where the files will be saved.
        term : str
            The term used in the filenames.
        """
        simu_dico = self.make_dico()
        if not os.path.exists(file_path):  # save infos
            os.makedirs(file_path)
        with open(f"{file_path}/{term}/pca_{term}", "wb") as f:
            pickle.dump(self.pca, f)
        np.savez(f"{file_path}/{term}/{term}", **simu_dico)


class Predictions:
    """
    Class for forecasting and reconstructing time series data using Gaussian Processes.

    Attributes
    ----------
    var : str
        The variable name.
    data : pandas.DataFrame
        The time series data.
    info : dict
        Additional information.
    technique : GaussianProcessRegressor
        The Gaussian Process regressor.
    w : int
        Width for moving average and metrics calculation.
    """

    def __init__(
        self,
        var,
        data=None,
        info=None,
        forecast_technique=None,
        dr_technique=None,
        w=12,
    ):
        """
        Initialize the Predictions object.

        Parameters
        ----------
        var : str
            The variable name.
        data : pandas.DataFrame, optional
            The time series data.
        info : dict, optional
            Additional information.
        forecast_technique : object, optional
            The Gaussian Process regressor.
        dr_technique : object, optional
            Dimensionality reduction technique instance.
        w : int, optional
            Width for moving average and metrics calculation.
        """
        self.var = var
        self.forecaster = forecast_technique
        self.dr_technique = dr_technique
        self.w = w
        self.data = data
        self.info = info
        self.info["desc"] = self.info["desc"].item()
        self.len_ = len(self.data)

    def __len__(self):
        return len(self.data)

    ##################
    #    parallel_forecast    #
    ##################

    def parallel_forecast(self, train_len, steps, jobs=1):
        """
        Parallel forecast of time series data/eofs for each time series.

        Parameters
        ----------
        train_len : int
            Length of the training data.
        steps : int
            Number of steps to forecast.
        jobs : int, optional
            Number of parallel jobs to run. Defaults to ``1``.

        Returns
        -------
        tuple
            y_hats : pandas.DataFrame
                parallel_forecasted values.
            y_stds : pandas.DataFrame
                Standard deviations of the forecasts.
            metrics : list of dict
                One dict of metrics by forecast
        """
        r = Parallel(n_jobs=jobs)(
            delayed(self.forecast_single_series)(c, train_len, steps)
            for c in range(1, self.data.shape[1] + 1)
        )
        y_hats = pd.DataFrame(
            np.array([r[i][0] for i in range(len(r))]).T, columns=self.data.columns
        )
        y_stds = pd.DataFrame(
            np.array([r[i][1] for i in range(len(r))]).T, columns=self.data.columns
        )
        metrics = [r[i][2] for i in range(len(r))]
        return y_hats, y_stds, metrics

    def forecast_single_series(self, n, train_len, steps=0):
        """
        Forecast a single time series.

        Parameters
        ----------
        n : int
            Variable index.
        train_len : int
            Length of the training data.
        steps : int, optional
            Number of steps to forecast. Defaults to ``0``.

        Returns
        -------
        tuple
            y_hat : ndarray
                parallel_forecasted values.
            y_hat_std : ndarray
                Standard deviations of the forecasts.
            metrics : dict
                Dictionary of metrics defined in the corresponding function
        """
        random.seed(20)

        if steps == 0:
            return np.array([]), np.array([]), None  # TODO: Review coding style

        mean, std, y_train, y_test, x_train, x_pred = self.train_test_series(
            n, train_len, steps
        )

        y_hat, y_hat_std = self.forecaster.apply_forecast(y_train, x_train, x_pred)

        y_train, y_hat, y_hat_std = (
            y_train * 2 * std + mean,
            y_hat * 2 * std + mean,
            y_hat_std * 2 * std,
        )
        metrics = None
        if y_test is not None and y_hat is not None:
            metrics = Predictions.get_metrics(2, y_hat, y_test[: len(y_hat)])

        return y_hat, y_hat_std, metrics

    def train_test_series(self, n, train_len, steps):
        """
        Prepare data for forecasting.

        Parameters
        ----------
        n : int
            Variable index.
        train_len : int
            Length of the training data.
        steps : int
            Number of steps to forecast.

        Returns
        -------
        tuple
            mean : float
                Mean of the training data.
            std : float
                Standard deviation of the training data.
            y_train : ndarray
                Training data.
            y_test : ndarray
                Test data.
            x_train : ndarray
                Training features.
            x_pred : ndarray
                Prediction features.
        """
        x_train, pas = np.linspace(0, 1, train_len, retstep=True)
        x_train = x_train.reshape(-1, 1)
        # x_pred = np.arange(0, (len(self) + steps) * pas, pas).reshape(-1, 1)
        # TODO: Check this len(self) + steps logic
        x_pred = np.linspace(1 + pas, 1 + steps * pas, steps).reshape(-1, 1)

        y_train = self.data[self.var + "-" + str(n)].iloc[:train_len].to_numpy()
        mean, std = np.nanmean(y_train), np.nanstd(y_train)
        y_train = (y_train - mean) / (2.0 * std)
        y_test = None
        if train_len < len(self):
            y_test = (
                self.data[self.var + "-" + str(1)].iloc[train_len : len(self)].to_numpy()
            )
        return mean, std, y_train, y_test, x_train, x_pred

    def show(self, n, y_hat, y_hat_std, train_len, color="tab:blue"):
        """
        Plot the forecasted time series data.

        Parameters
        ----------
        n : int
            Corresponding time serie/eof
        y_hat : ndarray
            parallel_forecasted values.
        y_hat_std : ndarray
            Standard deviations of the forecasts.
        train_len : int
            Length of the training data.
        color : str, optional
            Color for the plot. Defaults to "tab:blue".
        """
        total_len = train_len + len(y_hat)
        plt.figure(figsize=(7, 3))
        plt.plot(
            self.data[self.var + "-" + str(n)][:train_len],
            linestyle="dashed",
            color="black",
            alpha=0.7,
            label="Train series",
        )
        if train_len < len(self):
            plt.plot(
                self.data[self.var + "-" + str(n)][
                    train_len - 1 :
                ],  # TODO: Check indexing numbers and data
                linestyle="dashed",
                color="black",
                alpha=0.5,
                label="Test series",
            )

        plt.plot(
            np.arange(train_len, total_len), y_hat, color=color, label="GP forecast"
        )
        plt.fill_between(
            np.arange(train_len, train_len + len(y_hat)),
            y_hat + y_hat_std,
            y_hat - y_hat_std,
            color=color,
            alpha=0.2,
        )
        plt.title(f"parallel_forecast of {self.var} {n!s}")
        plt.legend()
        plt.show()
        print()

    @staticmethod
    def get_metrics(w, y_hat, y_test):
        """
        Calculate metrics for evaluating the forecast.

        Parameters
        ----------
        w : int
            Width for moving average.
        y_hat : ndarray
            parallel_forecasted values.
        y_test : ndarray
            True values.

        Returns
        -------
        dict
            Dictionary containing calculated metrics.
        """
        ma_test = np.convolve(y_test / w, np.ones(w), mode="valid")
        ma_pred = np.convolve(y_hat / w, np.ones(w), mode="valid")
        dist = np.convolve((y_hat - y_test) / w, np.ones(w), mode="valid")
        mse = np.convolve(((y_hat - y_test) ** 2) / w, np.ones(w), mode="valid")
        dist_max, std = [], []
        for i in range(w, len(y_hat) + 1):
            window_t = y_test[i - w : i]
            # window_h = y_hat[i - w : i]
            # maxi/mini
            maxi = np.max(window_t) - np.mean(window_t)
            mini = np.mean(window_t) - np.min(window_t)
            dist_max.append(max(maxi, mini))
            # std
            std.append(np.std(window_t, ddof=1))
        return {
            "ma_true": ma_test,
            "ma_pred": ma_pred,
            "dist": dist,
            "dist_max": dist_max,
            "mse": mse,
            "std_true": np.array(std),
        }


#################################
##                             ##
##   parallel_forecast & reconstruct    ##
##                             ##
#################################
