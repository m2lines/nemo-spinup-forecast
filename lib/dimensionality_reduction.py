from abc import ABC, abstractmethod
import numpy as np
from sklearn.decomposition import (
    PCA,
    KernelPCA,
)  # TODO: Import the general decomposition class
import math


class DimensionalityReduction(ABC):
    @abstractmethod
    def decompose(self):
        pass

    @staticmethod
    @abstractmethod
    def reconstruct_predictions(self):
        pass

    @abstractmethod
    def reconstruct_components(self):
        pass

    @abstractmethod
    def get_component(self):
        pass

    @abstractmethod
    def error(self):
        pass


class DimensionalityReductionPCA(DimensionalityReduction):
    def __init__(self, comp):
        self.components = None
        self.comp = comp
        self.pca = None
        self.shape = None
        self.desc = None

    def get_num_components(self):
        return self.pca.n_components_

    def decompose(self, simulation, length):
        """
        Apply Principal Component Analysis (PCA) to the simulation data. TODO: Generalise the applyPCA to apply decomposition
        """
        array = simulation.reshape(length, -1)
        self.bool_mask = np.asarray(np.isfinite(array[0, :]), dtype=bool)
        array_masked = array[:, self.bool_mask]
        print(self.comp)
        pca = PCA(self.comp, whiten=False)
        self.components = pca.fit_transform(array_masked)
        self.pca = pca

        return self.components, self.pca, self.bool_mask

    @staticmethod
    def reconstruct_predictions(predictions, n, info, begin=0):
        """
        Reconstruct the time series data from predictions.

        Parameters:
            predictions (DataFrame) : Forecasted values for each component.
            n (int)                 : Number of components to consider for reconstruction.
            begin (int, optional)   : Starting index for reconstruction. Defaults to 0.

        Returns:
            array: Reconstructed time series data.
        """
        rec = []
        int_mask = info["mask"].astype(np.int32).reshape(info["shape"])
        # print(int_mask)
        for t in range(begin, len(predictions)):
            map_ = np.zeros((info["shape"]), dtype=np.float32)
            arr = np.array(
                list(predictions.iloc[t, :n]) + [0] * (len(info["pca"].components_) - n)
            )
            temp1 = info["pca"].inverse_transform(arr)
            print("arr: ", arr)
            print("temp1 shape: ", temp1.shape)
            map_[int_mask == 1] = info["pca"].inverse_transform(arr)

            map_[int_mask == 0] = np.nan
            rec.append(map_)
            print(np.array(rec).shape)
        return int_mask, np.array(rec) * 2 * info["desc"]["std"] + info["desc"]["mean"]

    def reconstruct_components(self, n):
        """
        Reconstruct data using a specified number of principal components.

        Parameters:
            n (int) : The number of components used for reconstruction.

        Returns:
            (numpy.array) : The reconstructed data.
        """
        rec = []
        # int_mask =   # Convert the boolean mask to int mask once
        self.int_mask = self.bool_mask.astype(np.int32).reshape(
            self.shape
        )  # Reshape to match the shape of map_
        for t in range(len(self.components)):
            map_ = np.zeros(self.shape, dtype=np.float32)
            arr = np.array(
                list(self.components[t, :n]) + [0] * (len(self.pca.components_) - n)
            )
            map_[self.int_mask == 1] = self.pca.inverse_transform(arr)
            map_[self.int_mask == 0] = np.nan
            rec.append(map_)
        return np.array(rec)

    def get_component(self):
        """
        Get principal component map for the specified component.

        Parameters:
            n (int) : component used for reconstruction.

        Returns:
            (numpy.ndarray): The principal component map.
        """
        # map_ = np.zeros((np.product(self.shape)), dtype=float)
        map_ = np.zeros((np.prod(self.shape)), dtype=float)
        map_[~self.bool_mask] = np.nan
        map_[self.bool_mask] = self.pca.components_[n]
        map_ = map_.reshape(self.shape)
        map_ = 2 * map_ * self.desc["std"] + self.desc["mean"]
        return map_

    def error(self, n):
        return self.rmse(n)

    def rmse(self, n):
        reconstruction = self.reconstruct_components(n)
        rmse_values = self.rmseValues(reconstruction) * 2 * self.desc["std"]
        rmse_map = self.rmseMap(reconstruction) * 2 * self.desc["std"]
        return reconstruction, rmse_values, rmse_map

    def rmseValues(self, reconstruction):
        truth = (
            self.simulation
        )  # * 2 * self.desc["std"] + self.desc["mean"] TODO: Self.simulation
        rec = reconstruction  # * 2 * self.desc["std"] + self.desc["mean"]
        if len(np.shape(truth)) == 3:
            n = np.count_nonzero(~np.isnan(truth[0]))
            rmse_values = np.sqrt(np.nansum((truth - rec) ** 2, axis=(1, 2)) / n)
        else:
            n = np.count_nonzero(~np.isnan(self.simulation[0]), axis=(1, 2))
            rmse_values = np.nansum((truth - rec) ** 2, axis=(2, 3))
            for i in range(len(n)):
                rmse_values[:, i] = rmse_values[:, i] / n[i]
            rmse_values = np.sqrt(rmse_values)
        return rmse_values

    def rmseMap(self, reconstruction):
        t = self.len
        truth = self.simulation
        reconstruction = reconstruction
        return np.sqrt(np.sum((self.simulation[:] - reconstruction) ** 2, axis=0) / t)


class DimensionalityReductionKernelPCA(DimensionalityReduction):
    def __init__(self, comp, kernel="rbf", **kwargs):
        # comp is the number of components
        self.comp = 6
        self.components = None  # Transformed (projected) data
        self.pca = None  # Will hold the KernelPCA instance
        self.shape = None  # Shape of the original spatial grid
        self.desc = None  # Dictionary containing metadata (e.g., mean, std)
        self.bool_mask = None  # Mask for valid features
        self.kernel = kernel  # Kernel type for KernelPCA
        self.kwargs = kwargs  # Additional parameters for KernelPCA

    def get_num_components(self):
        return self.comp

    def decompose(self, simulation, length):
        """
        Apply Kernel Principal Component Analysis (KernelPCA) to the simulation data.
        The simulation is reshaped to 2D (time, features) and only the finite features are used.
        """
        # Reshape the simulation data: assume simulation is originally (time, height, width)
        array = simulation.reshape(length, -1)
        # Create a boolean mask of valid (finite) features
        self.bool_mask = np.asarray(np.isfinite(array[0, :]), dtype=bool)
        array_masked = array[:, self.bool_mask]
        # Save the original spatial shape (for later reconstruction)
        self.shape = simulation.shape[1:]
        # Instantiate KernelPCA with inverse transform enabled
        kpca = KernelPCA(
            n_components=self.comp,
            kernel=self.kernel,
            fit_inverse_transform=True,
            **self.kwargs,
        )
        # Fit and transform the masked data
        self.components = kpca.fit_transform(array_masked)
        self.pca = kpca
        return self.components, self.pca, self.bool_mask

    @staticmethod
    def reconstruct_predictions(predictions, n, info, begin=0):
        """
        Reconstruct the time series data from forecasted predictions.

        Parameters:
            predictions (DataFrame): Forecasted values for each component.
            n (int): Number of components to use for reconstruction.
            info (dict): Dictionary containing keys "mask", "shape", "pca", and "desc".
            begin (int): Starting index for reconstruction.

        Returns:
            tuple: (int_mask, reconstructed time series data)
        """
        rec = []
        int_mask = info["mask"].astype(np.int32).reshape(info["shape"])
        print(int_mask.shape)
        for t in range(begin, len(predictions)):
            # Create an array for the t-th prediction;
            # pad with zeros for any missing components
            arr = np.array(
                list(predictions.iloc[t, :n]) + [0] * (info["pca"].n_components - n)
            )
            # Reshape to 2D (sample, n_components) for inverse_transform
            map_ = np.zeros(info["shape"], dtype=np.float32)

            # inverse transform data
            temp1 = info["pca"].inverse_transform(arr.reshape(1, -1)).flatten()
            # print("Arr: ", arr)
            # print("temp1 shape: ", temp1.shape)
            # Check that the inverse transform has the correct number of data points
            assert len(temp1) == math.prod(info["shape"])

            map_[int_mask == 1] = (
                info["pca"].inverse_transform(arr.reshape(1, -1)).flatten()
            )

            map_[int_mask == 0] = np.nan
            rec.append(map_)
            # print(np.array(rec).shape)
        # Scale the reconstruction back using provided descriptors
        return int_mask, np.array(rec) * 2 * info["desc"]["std"] + info["desc"]["mean"]

    def reconstruct_components(self, n):
        """
        Reconstruct the original data using the first n kernel principal components.

        Returns:
            numpy.array: Reconstructed data with the same spatial dimensions.
        """
        rec = []
        # Convert the boolean mask to an integer mask and reshape to match original grid
        self.int_mask = self.bool_mask.astype(np.int32).reshape(self.shape)
        for t in range(len(self.components)):
            arr = np.array(
                list(self.components[t, :n]) + [0] * (self.pca.n_components - n)
            )
            map_ = np.zeros(self.shape, dtype=np.float32)
            # The inverse_transform expects a 2D array; extract the first (and only) row of the result
            map_[self.int_mask == 1] = self.pca.inverse_transform(arr.reshape(1, -1))[0]
            map_[self.int_mask == 0] = np.nan
            rec.append(map_)
        return np.array(rec)

    def get_component(self, n):
        """
        Get an approximate kernel principal component map for component n.
        Note: For non-linear kernels the mapping is implicit, so this is only a proxy.

        Returns:
            numpy.ndarray: A 2D map corresponding to the n-th component.
        """
        # Create a flat map, fill with NaNs for invalid entries
        map_ = np.zeros(np.prod(self.shape), dtype=float)
        map_[~self.bool_mask] = np.nan
        # For linear kernels one might extract an interpretable component;
        # here we use the dual coefficients as a proxy (if available)
        if hasattr(self.pca, "alphas_") and n < self.pca.alphas_.shape[1]:
            map_[self.bool_mask] = self.pca.alphas_[:, n]
        else:
            # Otherwise, default to zeros if the component is not available
            map_[self.bool_mask] = 0
        map_ = map_.reshape(self.shape)
        # Scale the component using the stored descriptor parameters
        map_ = 2 * map_ * self.desc["std"] + self.desc["mean"]
        return map_

    def error(self, n):
        return self.rmse(n)

    def rmse(self, n):
        """
        Compute RMSE of the reconstruction using n components.

        Returns:
            tuple: (reconstructed data, RMSE values per sample, RMSE map)
        """
        reconstruction = self.reconstruct_components(n)
        rmse_values = self.rmseValues(reconstruction) * 2 * self.desc["std"]
        rmse_map = self.rmseMap(reconstruction) * 2 * self.desc["std"]
        return reconstruction, rmse_values, rmse_map

    def rmseValues(self, reconstruction):
        """
        Compute RMSE values for each time sample.
        """
        truth = self.simulation  # Assumes self.simulation is set externally
        rec = reconstruction
        if len(np.shape(truth)) == 3:
            valid_count = np.count_nonzero(~np.isnan(truth[0]))
            rmse_values = np.sqrt(
                np.nansum((truth - rec) ** 2, axis=(1, 2)) / valid_count
            )
        else:
            valid_count = np.count_nonzero(~np.isnan(self.simulation[0]), axis=(1, 2))
            rmse_values = np.nansum((truth - rec) ** 2, axis=(2, 3))
            for i in range(len(valid_count)):
                rmse_values[:, i] = rmse_values[:, i] / valid_count[i]
            rmse_values = np.sqrt(rmse_values)
        return rmse_values

    def rmseMap(self, reconstruction):
        """
        Compute an RMSE map over the spatial grid.
        """
        t = (
            self.len
        )  # Assumes self.len is defined elsewhere (e.g., number of time steps)
        return np.sqrt(np.sum((self.simulation[:] - reconstruction) ** 2, axis=0) / t)
