import os
import pytest
from lib.forecast import Simulation, load_ts
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA

# TODO: Rename "term" variable for test parametrizations


@pytest.mark.parametrize(
    "term, filename, expected_file_pattern,expected_count",
    [
        # Valid terms with their respective file patterns
        ("toce", "DINO_1y_grid_T", "1y_grid_T.nc", 1),  # valid_temperature
        ("soce", "DINO_1y_grid_T", "1y_grid_T.nc", 1),  # valid_salinity
        ("ssh", "DINO_1m_To_1y_grid_T", "1m_To_1y_grid_T.nc", 1),  # valid_ssh
    ],
)
def test_get_data_valid_terms(term, filename, expected_file_pattern, expected_count):
    """Check get_data returns expected files for valid terms."""

    data_path = "tests/data/nemo_data_e3"

    # Run the get_data method
    files = Simulation.get_data(
        data_path, term, filename
    )  # TODO: Change argument names

    # Verify correct number of files found
    assert len(files) == expected_count, (
        f"Expected {expected_count} files, got {len(files)}"
    )

    # Files should be sorted
    assert files == sorted(files)

    # Check if all found files match the expected pattern for this variable
    for i, file in enumerate(files):
        assert os.path.dirname(file) == data_path
        assert expected_file_pattern in os.path.basename(file), (
            f"File {i} doesn't contain pattern {expected_file_pattern}"
        )
        assert filename in os.path.basename(file), (
            f"File {i} doesn't contain term {filename}"
        )


@pytest.mark.parametrize(
    "term, filename, expected_count",
    [
        # Valid term with nonexistent grid
        ("ssh", "nonexistent", 0),  # ssh_nonexistent_grid
    ],
)
def test_get_data_invalid_combinations(term, filename, expected_count):
    """Check get_data returns no files for invalid term-file combinations."""
    data_path = "tests/data/nemo_data_e3"

    # Run the get_data method
    files = Simulation.get_data(
        data_path, term, filename
    )  # TODO: Change argument names

    # Verify no files are found for invalid combinations
    assert len(files) == expected_count, (
        f"Expected no files for invalid term, got {len(files)}"
    )


@pytest.mark.parametrize(
    "setup_simulation_class, term, shape",
    [
        pytest.param(
            ("toce", "DINO_1y_grid_T.nc"),
            ("toce", "DINO_1y_grid_T.nc"),
            (36, 199, 62),
        ),
        pytest.param(
            ("soce", "DINO_1y_grid_T.nc"),
            ("soce", "DINO_1y_grid_T.nc"),
            (36, 199, 62),
        ),
        (
            ("ssh", "DINO_1m_To_1y_grid_T.nc"),
            ("ssh", "DINO_1m_To_1y_grid_T.nc"),
            (199, 62),
        ),
    ],
    indirect=["setup_simulation_class"],
)
# indirect parameterization of setup_simulation_class fixture
def test_get_attributes(setup_simulation_class, term, shape):
    """Check getAttributes returns correct shape, term, and time_dim.

    Notes
    -----
    See this issue for reason for faliure:
    https://github.com/m2lines/nemo-spinup-forecast/issues/58
    """

    simulation = setup_simulation_class

    simulation.get_attributes()

    # Expected spatial dimensions for this dataset
    assert simulation.shape == shape, f"Shape {simulation.shape} != expected {shape}"
    # Term should match the input parameter exactly
    expected_term_file = (simulation.term, simulation.filename)
    assert expected_term_file == term, f"Term {expected_term_file} != expected {term}"
    # Standard time dimension name for NEMO data
    assert simulation.time_dim == "time_counter", (
        f"Time dimension should be 'time_counter', got {simulation.time_dim}"
    )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_getSimu(setup_simulation_class):
    """Check getSimu sets simulation DataArray and descriptive stats.
    Verifies that the simulation instance has a properly configured xarray DataArray
    and that the computed descriptive statistics (mean, std, min, max) match
    independently calculated values from the underlying data.
    """
    simu = setup_simulation_class

    # Check that 'simulation' attribute exists and is an xarray DataArray
    assert hasattr(simu, "simulation"), (
        "Simulation instance should have 'simulation' attribute"
    )
    assert isinstance(simu.simulation, xr.DataArray), (
        "'simulation' should be a xarray.DataArray"
    )
    # Check DataArray name matches variable name
    assert simu.simulation.name == simu.term, (
        f"DataArray name {simu.simulation.name} does not match term {simu.term}"
    )

    # Extract data values for manual computation
    data = simu.simulation.values

    # Compute expected descriptive statistics
    expected_mean = np.nanmean(data)
    expected_std = np.nanstd(data)
    expected_min = np.nanmin(data)
    expected_max = np.nanmax(data)

    # Check that desc dictionary contains correct keys and values
    for key in ["mean", "std", "min", "max"]:
        assert key in simu.desc, f"'{key}' should be in simu.desc"

    # Compare actual vs expected values
    assert np.isclose(simu.desc["mean"], expected_mean), "Mean calculation incorrect"
    assert np.isclose(simu.desc["std"], expected_std), "Std calculation incorrect"
    assert np.isclose(simu.desc["min"], expected_min), "Min calculation incorrect"
    assert np.isclose(simu.desc["max"], expected_max), "Max calculation incorrect"


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_load_file(setup_simulation_class):
    """Check loadFile returns correct DataArray and updates length."""
    simu = setup_simulation_class

    # Use the first file in the simulation's file list
    file_path = simu.files[0]

    # Reset length to zero to isolate this test
    simu.len = 0

    # Call loadFile
    data_array = simu.load_file(file_path)

    # Ensure the return is a loaded xarray.DataArray
    assert isinstance(data_array, xr.DataArray), (
        "loadFile should return an xarray.DataArray"
    )

    # The DataArray name should match the simulation term
    assert data_array.name == simu.term, (
        f"DataArray name {data_array.name} does not match term {simu.term}"
    )

    # After loading, self.len should equal the size along the time dimension
    assert simu.time_dim == "time_counter", (
        "Expected time dimension to be 'time_counter'"
    )
    expected_len = 50  # Known length for test data files
    assert simu.len == expected_len, (
        f"Length after loadFile ({simu.len}) does not match expected ({expected_len})"
    )


@pytest.fixture()
def dummy_simu():
    """Create bare Simulation instance for prepare tests."""
    simu = Simulation.__new__(Simulation)
    return simu


def test_prepare_slices_based_on_start_end(dummy_simu):
    """Check prepare slices data based on start and end indices."""
    # Create a simple DataArray of length 10 with sequential values
    data = xr.DataArray(np.arange(10, dtype=float), dims=("time",))
    dummy_simu.simulation = data
    dummy_simu.start = 3
    dummy_simu.end = 8
    dummy_simu.desc = {}

    dummy_simu.get_simulation_data(stand=False)

    # After slicing, simulation should be numpy array [3,4,5,6,7]
    expected = np.arange(3, 8, dtype=float)
    expected_len = len(expected)

    assert isinstance(dummy_simu.simulation, np.ndarray), (
        "Simulation should be a numpy array"
    )
    assert dummy_simu.len == expected_len, (
        f"Length {dummy_simu.len} != expected {expected_len}"
    )
    np.testing.assert_array_equal(
        dummy_simu.simulation, expected, err_msg="Simulation data != expected data"
    )


def test_prepare_slices_start_specified_end_none(dummy_simu):
    """Check prepare slices data using only start when end is None."""
    data = xr.DataArray(np.arange(10, dtype=float), dims=("time",))
    dummy_simu.simulation = data
    dummy_simu.start = 4
    dummy_simu.end = None
    dummy_simu.desc = {}

    dummy_simu.get_simulation_data(stand=False)

    # After slicing, simulation should be numpy array [4,5,6,7,8,9]
    expected = np.arange(4, 10, dtype=float)
    expected_len = len(expected)

    assert dummy_simu.len == expected_len, (
        f"Length {dummy_simu.len} != expected {expected_len}"
    )
    np.testing.assert_array_equal(dummy_simu.simulation, expected)


def test_prepare_standardisation_applied(dummy_simu):
    """Check prepare applies standardisation when stand=True.

    This normalises the data
    """
    data = xr.DataArray([0.0, 2.0, 4.0, 6.0], dims=("time",))
    dummy_simu.simulation = data
    dummy_simu.start = 0
    dummy_simu.end = None
    dummy_simu.desc = {}

    dummy_simu.get_simulation_data(stand=True)

    # Manually compute expected standardized values: (x - mean) / (2*std)
    mean = np.nanmean(data)
    std = np.nanstd(data)
    expected = ((data - mean) / (2 * std)).values

    np.testing.assert_allclose(
        dummy_simu.simulation,
        expected,
        err_msg="Standardization formula not applied correctly",
    )


def test_prepare_updates_desc_and_simulation(dummy_simu):
    """Check prepare updates simulation and descriptive statistics."""
    data = xr.DataArray([1.0, 2.0, 3.0, 5.0], dims=("time",))
    dummy_simu.simulation = data
    dummy_simu.start = 1
    dummy_simu.end = 4
    dummy_simu.desc = {}

    dummy_simu.get_simulation_data(stand=False)

    # After slicing, raw numpy array should match values[1:4]
    sliced = data.values[1:4]
    assert isinstance(dummy_simu.simulation, np.ndarray), (
        "Simulation should be numpy array"
    )
    np.testing.assert_array_equal(dummy_simu.simulation, sliced)

    # Check descriptive statistics computed on sliced data
    assert np.isclose(dummy_simu.desc["mean"], np.nanmean(sliced)), (
        "Mean calculation incorrect"
    )
    assert np.isclose(dummy_simu.desc["std"], np.nanstd(sliced)), (
        "Std calculation incorrect"
    )
    assert np.isclose(dummy_simu.desc["min"], np.nanmin(sliced)), (
        "Min calculation incorrect"
    )
    assert np.isclose(dummy_simu.desc["max"], np.nanmax(sliced)), (
        "Max calculation incorrect"
    )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_standardize(setup_simulation_class):
    """Check standardize transforms simulation and preserves desc."""
    simu = setup_simulation_class
    simu.len = 0
    # Load simulation data and compute descriptive stats
    simu.get_simu()

    # Copy original data and desc
    original_data = simu.simulation.copy().values
    original_mean = simu.desc["mean"]
    original_std = simu.desc["std"]

    # Apply standardization
    simu.standardize()

    # The simulation attribute should remain an xarray.DataArray
    assert isinstance(simu.simulation, xr.DataArray), (
        "simulation should be an xarray.DataArray after standardize"
    )

    # Flatten arrays for comparison
    standardized_data = simu.simulation.values
    expected = (original_data - original_mean) / (2 * original_std)

    # Check that the data was standardized correctly (accounting for NaNs)

    (np.testing.assert_allclose(standardized_data, expected, equal_nan=True),)
    (("standardize did not correctly transform simulation data"),)


'''
def create_simulation(data, comp):
    """Helper to create Simulation instance for applyPCA tests."""
    sim = Simulation.__new__(Simulation)
    sim.simulation = data
    sim.len = data.shape[0]
    sim.comp = comp
    return sim


#@pytest.mark.parametrize("create_simulation", [(data, 0.9)], indirect=True)
def test_applyPCA_finite_dummy_data():
    """Check applyPCA outputs components with correct dimensions for finite data."""
    rng = np.random.RandomState(0)
    # Create random data: 100 time steps, 20 spatial features
    data = rng.rand(100, 20)
    sim = create_simulation(data, comp=0.9)
    sim.decompose()

    # Check shape of components: (time_steps, n_components)
    components = sim.components
    expected_time_dim = data.shape[0]
    assert components.shape[0] == expected_time_dim, (
        f"Components time dimension {components.shape[0]} != expected {expected_time_dim}"
    )

    n_components = components.shape[1]
    # Number of components should be between 1 and number of features
    max_components = data.shape[1]
    assert 1 <= n_components <= max_components, (
        f"Number of components {n_components} not in valid range [1, {max_components}]"
    )

    # PCA object should be set and have matching component matrix dimensions
    assert isinstance(sim.pca, PCA), "PCA object should be sklearn PCA instance"
    expected_pca_shape = (n_components, data.shape[1])
    assert sim.pca.components_.shape == expected_pca_shape, (
        f"PCA components shape {sim.pca.components_.shape} != expected {expected_pca_shape}"
    )


def test_applyPCA_masks_nans():
    """Check applyPCA masks features with NaNs in first time slice."""
    rng = np.random.RandomState(1)
    data = rng.rand(50, 10)
    nan_indices = [2, 7]
    # Insert NaNs in first time step at specific feature indices
    data[0, nan_indices] = np.nan
    sim = create_simulation(data, comp=None)
    sim.decompose()

    mask = sim.bool_mask

    # Mask length should equal number of spatial features
    expected_mask_len = data.shape[1]
    assert mask.shape == (expected_mask_len,), (
        f"Mask shape {mask.shape} != expected ({expected_mask_len},)"
    )

    # Indices with NaNs should be masked False, others True
    expected_mask = [False if i in nan_indices else True for i in range(10)]
    assert np.array_equal(mask, expected_mask), (
        f"Mask {mask} != expected {expected_mask}"
    )

    # PCA components second dimension should equal number of unmasked features
    n_unmasked = mask.sum()
    assert sim.pca.components_.shape[1] == n_unmasked, (
        f"PCA components feature dim {sim.pca.components_.shape[1]} != unmasked features {n_unmasked}"
    )

    # Components shape should reflect time steps and selected components
    expected_components_shape = (data.shape[0], sim.pca.n_components_)
    assert sim.components.shape == expected_components_shape, (
        f"Components shape {sim.components.shape} != expected {expected_components_shape}"
    )

'''


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_applyPCA_real_data(setup_simulation_class):
    """Check applyPCA works correctly on real data."""
    sim = setup_simulation_class
    # Prepare the data
    sim.get_simulation_data(stand=False)
    # After prepare, simulation attribute should be a NumPy array
    assert isinstance(sim.simulation, np.ndarray), (
        "Simulation should be numpy array after prepare"
    )
    initial_shape = sim.simulation.shape

    sim.decompose()
    components = sim.components

    # Components first dimension should equal time length
    expected_time_dim = sim.len
    assert components.shape[0] == expected_time_dim, (
        f"Components time dimension {components.shape[0]} != expected {expected_time_dim}"
    )

    # Components second dimension should equal number of PCA components
    expected_n_components = sim.pca.n_components_
    assert components.shape[1] == expected_n_components, (
        f"Components feature dimension {components.shape[1]} != expected {expected_n_components}"
    )

    # Boolean mask length should equal total number of spatial features
    feature_count = np.prod(initial_shape[1:])
    expected_mask_shape = (feature_count,)
    assert sim.bool_mask.shape == expected_mask_shape, (
        f"Mask shape {sim.bool_mask.shape} != expected {expected_mask_shape}"
    )

    # PCA components shape should match [n_components, n_features]
    expected_pca_shape = (sim.pca.n_components_, feature_count)
    assert sim.pca.components_.shape == expected_pca_shape, (
        f"PCA components shape {sim.pca.components_.shape} != expected {expected_pca_shape}"
    )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("soce", "DINO_1y_grid_T.nc"),  # 3D case (z,y,x)
        ("toce", "DINO_1y_grid_T.nc"),  # 3D case (z,y,x)
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),  # 2D case (y,x)
    ],
    indirect=True,
)
def test_getPC_real_data(setup_simulation_class):
    """Check getPC returns correct PC map shape, mask, and values for real data."""
    sim = setup_simulation_class

    # prepare real slice and compute PCA
    sim.get_simulation_data(stand=False)
    sim.decompose()

    std = sim.desc["std"]
    mean = sim.desc["mean"]
    mask = sim.bool_mask  # 1D boolean mask over flattened features
    shape = sim.shape  # e.g. (z,y,x) or (y,x)

    # Flattened mask length must equal product of spatial dimensions
    expected_mask_len = np.prod(shape)
    assert mask.shape == (expected_mask_len,), (
        f"Mask length {mask.shape} != expected ({expected_mask_len},)"
    )

    # Test every principal component
    for comp in range(sim.pca.n_components_):
        pc_map = sim.get_component(
            comp
        )  # Contribution of each coordinate to the components

        # Should return a numpy array with correct spatial shape
        assert isinstance(pc_map, np.ndarray), f"PC map {comp} should be numpy array"
        assert pc_map.shape == shape, (
            f"PC map {comp} shape {pc_map.shape} != expected {shape}"
        )

        flat_map = pc_map.ravel()
        comp_vals = sim.pca.components_[comp]

        # Build expected flattened map: transform component values back to original scale
        expected_flat = np.full(mask.shape, np.nan, dtype=float)
        expected_flat[mask] = 2 * comp_vals * std + mean

        np.testing.assert_allclose(
            flat_map,
            expected_flat,
            equal_nan=True,
            err_msg=f"PC map {comp} values incorrect",
        )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("soce", "DINO_1y_grid_T.nc"),  # 3D case (z,y,x)
        ("toce", "DINO_1y_grid_T.nc"),  # 3D case (z,y,x)
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),  # 2D case (y,x)
    ],
    indirect=True,
)
def test_reconstruct_shape_and_mask_real_data(setup_simulation_class):
    """Check reconstruct returns arrays with correct shape, preserves mask, and finite values."""
    sim = setup_simulation_class

    # set up the PCA on the real data
    sim.get_simulation_data(stand=False)
    sim.decompose()

    # Check for a few n values: 1, all components, and beyond
    ns = [1, sim.pca.n_components_]
    for n in ns:
        rec = sim.dimensionality_reduction.reconstruct_components(n)

        # Should return array with correct shape: (time, *spatial_dims)
        expected_shape = (sim.len, *sim.shape)
        assert isinstance(rec, np.ndarray), (
            f"Reconstruction with {n} components should be numpy array"
        )
        assert rec.shape == expected_shape, (
            f"Reconstruction shape {rec.shape} != expected {expected_shape}"
        )

        # Integer mask should be updated to match spatial shape
        int_mask = sim.dimensionality_reduction.int_mask
        assert int_mask.shape == sim.shape, (
            f"Integer mask shape {int_mask.shape} != expected spatial shape {sim.shape}"
        )

        # For each time slice, masked positions should be NaN, unmasked finite
        flat_mask = int_mask.ravel()
        for t in range(rec.shape[0]):
            flat_rec = rec[t].ravel()
            # Masked positions (0) should contain NaN values
            masked_positions = flat_mask == 0
            assert np.all(np.isnan(flat_rec[masked_positions])), (
                f"Time {t}: masked positions should be NaN"
            )
            # Unmasked positions (1) should contain finite values
            unmasked_positions = flat_mask == 1
            assert np.all(np.isfinite(flat_rec[unmasked_positions])), (
                f"Time {t}: unmasked positions should be finite"
            )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("soce", "DINO_1y_grid_T.nc"),
        ("toce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_reconstruct_full_components_recovers_original_data(setup_simulation_class):
    """Check reconstruct with all components recovers original data."""
    sim = setup_simulation_class

    sim.get_simulation_data(stand=False)
    # Use all available components for reconstruction
    sim.dimensionality_reduction.comp = None
    sim.decompose()

    # Reconstruct using all components - should recover original data
    rec_all = sim.dimensionality_reduction.reconstruct_components(sim.pca.n_components_)

    # Original simulation was stored as raw values before PCA
    orig = sim.simulation
    assert isinstance(orig, np.ndarray), "Original simulation should be numpy array"

    # Shapes should match exactly
    assert rec_all.shape == orig.shape, (
        f"Reconstruction shape {rec_all.shape} != original shape {orig.shape}"
    )

    # Values should match up to numerical tolerance for full reconstruction
    np.testing.assert_allclose(
        rec_all,
        orig,
        rtol=1e-5,
        atol=1e-1,
        equal_nan=True,
        err_msg="Full reconstruction should recover original data",
    )


@pytest.fixture
def dummy_sim_array():
    """
    Fixture providing a simple 3-time-step, 2x2 spatial array simulation.
    """
    sim = Simulation.__new__(Simulation)
    # Create a 3x2x2 array with increasing values for predictable RMSE calculations
    sim.simulation = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
            [[3.0, 4.0], [5.0, 6.0]],
        ]
    )
    sim.len = sim.simulation.shape[0]
    return sim


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("soce", "DINO_1y_grid_T.nc"),  # 3D case (z,y,x)
        ("toce", "DINO_1y_grid_T.nc"),  # 3D case (z,y,x)
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),  # 2D case (y,x)
    ],
    indirect=True,
)
def test_rmseMap_zero_for_identical(setup_simulation_class):
    """Check rmseMap returns zeros for identical reconstruction."""
    sim = setup_simulation_class
    sim.get_simulation_data(stand=False)
    sim.dimensionality_reduction.set_from_simulation(sim)
    # Reconstruction identical to the truth - should yield zero very close to RMSE
    reconstruction = sim.simulation.copy()
    rmse_map = sim.dimensionality_reduction.rmse_map(reconstruction)

    # Expected RMSE map should be all zeros for perfect reconstruction
    expected_shape = sim.simulation.shape[1:]  # Spatial dimensions only
    expected = np.zeros(expected_shape)

    # Verify return type and shape
    assert isinstance(rmse_map, np.ndarray), "RMSE map should be numpy array"
    assert rmse_map.shape == expected_shape, (
        f"RMSE map shape {rmse_map.shape} != expected {expected_shape}"
    )

    # All values should be zero for identical reconstruction
    np.testing.assert_allclose(
        rmse_map,
        expected,
        err_msg="RMSE should be close to zero for identical reconstruction",
    )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),  # 3D data (time, z, y, x)
        ("soce", "DINO_1y_grid_T.nc"),  # 3D data (time, z, y, x)
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),  # 2D data (time, y, x)
    ],
    indirect=True,
)
def test_rmseMap_real_data_full_components_zero(setup_simulation_class):
    """Check rmseMap returns zeros for full-component reconstruction on real data."""
    sim = setup_simulation_class
    # Prepare without standardization to retain raw values
    sim.get_simulation_data(stand=False)
    # Ensure simulation data is a NumPy array
    assert isinstance(sim.simulation, np.ndarray), "Simulation should be numpy array"

    # Use all available components to reconstruct full data
    sim.dimensionality_reduction.comp = None  # None defaults to using all components
    sim.decompose()
    rec_all = sim.dimensionality_reduction.reconstruct_components(sim.pca.n_components_)

    # Compute RMSE map between original and reconstructed data
    rmse_map = sim.dimensionality_reduction.rmse_map(rec_all)
    print("max: ", np.max(rec_all))

    # Check return type and shape
    assert isinstance(rmse_map, np.ndarray), "RMSE map should be numpy array"
    assert rmse_map.shape == sim.shape, (
        f"Expected rmse_map shape {sim.shape}, got {rmse_map.shape}"
    )

    # Boolean mask of valid (unmasked) positions, reshaped to spatial dimensions
    mask = sim.bool_mask.reshape(sim.shape)

    # Unmasked positions (True) should have zero RMSE within tolerance for full reconstruction
    (np.testing.assert_allclose(rmse_map[mask], 0.0, atol=1e-1),)
    (("Non-zero RMSE found at unmasked positions for full reconstruction"),)

    # Masked positions (False) should remain NaN (no data to compute RMSE)
    assert np.all(np.isnan(rmse_map[~mask])), (
        "Expected NaN at masked positions in rmse_map"
    )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),  # 3D data (time, z, y, x)
        ("soce", "DINO_1y_grid_T.nc"),  # 3D data (time, z, y, x)
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),  # 2D data (time, y, x)
    ],
    indirect=True,
)
def test_rmseMap_real_data_with_limited_components_positive(setup_simulation_class):
    """Check rmseMap returns finite non-negative values for single-component reconstruction and NaNs are at masked positions."""
    sim = setup_simulation_class
    sim.get_simulation_data(stand=False)

    # Apply PCA retaining the default variance fraction (or set comp explicitly)
    sim.decompose()
    # Reconstruct using only the first principal component - should have some error
    rec_one = sim.dimensionality_reduction.reconstruct_components(1)
    rmse_map = sim.dimensionality_reduction.rmse_map(rec_one)

    # Check return type and shape
    assert isinstance(rmse_map, np.ndarray), "RMSE map should be numpy array"
    assert rmse_map.shape == sim.shape, (
        f"Expected rmse_map shape {sim.shape}, got {rmse_map.shape}"
    )

    mask = sim.bool_mask.reshape(sim.shape)

    # Unmasked positions: RMSE should be >= 0 and at least one should be > 0 (imperfect reconstruction)
    unmasked_vals = rmse_map[mask]
    assert np.all(unmasked_vals >= 0), (
        "Negative RMSE values found at unmasked positions"
    )
    assert np.any(unmasked_vals > 0), (
        "All RMSE values are zero at unmasked positions for limited-component reconstruction"
    )

    # Masked positions should remain NaN (no data available for RMSE calculation)
    assert np.all(np.isnan(rmse_map[~mask])), (
        "Expected NaN at masked positions in rmse_map"
    )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),  # 3D data (time, z, y, x)
        ("soce", "DINO_1y_grid_T.nc"),  # 3D data (time, z, y, x)
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),  # 2D data (time, y, x)
    ],
    indirect=True,
)
def test_rmseValues_zero_for_identical(setup_simulation_class):
    """Check rmseValues returns zeros for identical reconstruction."""
    sim = setup_simulation_class
    sim.get_simulation_data(stand=False)
    sim.dimensionality_reduction.set_from_simulation(sim)
    # Reconstruction identical to the truth - should yield zero RMSE
    reconstruction = sim.simulation.copy()
    rmse_values = sim.dimensionality_reduction.rmse_values(reconstruction)

    # Verify return type
    assert isinstance(rmse_values, np.ndarray), "RMSE values should be numpy array"

    # Check the correct output shape based on data dimensionality
    if sim.z_size is not None:
        # For 3D data, shape should be (time, depth) - RMSE per time step and depth level
        expected_shape = (sim.len, sim.z_size)
    else:
        # For 2D data, shape should be (time,) - RMSE per time step
        expected_shape = (sim.len,)

    assert rmse_values.shape == expected_shape, (
        f"RMSE values shape {rmse_values.shape} != expected {expected_shape}"
    )

    # All values should be zero for identical reconstruction
    np.testing.assert_allclose(
        rmse_values,
        0,
        err_msg="RMSE should be close to zero for identical reconstruction",
    )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),  # 3D data (time, z, y, x)
        ("soce", "DINO_1y_grid_T.nc"),  # 3D data (time, z, y, x)
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),  # 2D data (time, y, x)
    ],
    indirect=True,
)
def test_rmseValues_real_data_full_components_zero(setup_simulation_class):
    """Check rmseValues returns zeros for full-component reconstruction on real data."""
    sim = setup_simulation_class

    # Prepare raw numpy simulation and compute PCA for full reconstruction
    sim.get_simulation_data(stand=False)
    sim.dimensionality_reduction.comp = None
    sim.decompose()
    rec_all = sim.dimensionality_reduction.reconstruct_components(sim.pca.n_components_)

    # Compute RMSE values over time
    rmse_values = sim.dimensionality_reduction.rmse_values(rec_all)

    # Verify return type
    assert isinstance(rmse_values, np.ndarray), "RMSE values should be numpy array"

    # Check the correct output shape based on data dimensionality
    if sim.z_size is not None:
        # For 3D data, shape should be (time, depth) - RMSE per time step and depth level
        expected_shape = (sim.len, sim.z_size)
    else:
        # For 2D data, shape should be (time,) - RMSE per time step
        expected_shape = (sim.len,)

    assert rmse_values.shape == expected_shape, (
        f"RMSE values shape {rmse_values.shape} != expected {expected_shape}"
    )

    # All RMSE values should be effectively zero for full reconstruction
    np.testing.assert_allclose(
        rmse_values,
        0,
        atol=1e-3,
        err_msg="RMSE should be near zero for full reconstruction",
    )


@pytest.mark.parametrize(
    "setup_simulation_class",
    [
        ("toce", "DINO_1y_grid_T.nc"),
        ("soce", "DINO_1y_grid_T.nc"),
        ("ssh", "DINO_1m_To_1y_grid_T.nc"),
    ],
    indirect=True,
)
def test_rmseOfPCA_real_full_zero(setup_simulation_class):
    """Check rmseOfPCA returns zeros RMSE values and map for full reconstruction."""
    sim = setup_simulation_class
    sim.dimensionality_reduction.comp = None
    sim.dimensionality_reduction
    sim.get_simulation_data(stand=False)
    sim.decompose()

    # Use all components for full reconstruction - should be nearly perfect
    n_comp = sim.pca.n_components_
    rec, rmse_values, rmse_map = sim.error(n_comp)

    # Check RMSE values shape and near-zero values
    if sim.z_size is not None:
        expected_values_shape = (sim.len, sim.z_size)
    else:
        expected_values_shape = (sim.len,)

    assert rmse_values.shape == expected_values_shape, (
        f"RMSE values shape {rmse_values.shape} != expected {expected_values_shape}"
    )
    np.testing.assert_allclose(
        rmse_values,
        0,
        atol=5e-1,
        err_msg="RMSE values should be near zero for full reconstruction",
    )

    # Check RMSE map shape and near-zero values
    if sim.z_size is not None:
        expected_map_shape = (sim.z_size, sim.y_size, sim.x_size)
    else:
        expected_map_shape = (sim.y_size, sim.x_size)

    assert rmse_map.shape == expected_map_shape, (
        f"RMSE map shape {rmse_map.shape} != expected {expected_map_shape}"
    )
    np.testing.assert_allclose(
        rmse_map,
        0,
        atol=5e-1,
        err_msg="RMSE map should be near zero for full reconstruction",
    )


@pytest.fixture
def dummy_sim_pca():
    """
    Dummy simulation to test scaling behavior of rmseOfPCA with constant data.
    """
    c = 3.5
    sim = Simulation.__new__(Simulation)
    # Constant simulation: 4 time steps, 2x2 grid all with value c
    sim.simulation = np.full((4, 2, 2), fill_value=c)
    sim.len = sim.simulation.shape[0]
    # Set std=1 for simple scaling
    sim.desc = {"std": 1.0, "mean": 0.0}
    # Mock reconstruction returns zero array, so raw RMSE = c at every point
    sim.reconstruct = lambda n: np.zeros_like(sim.simulation)
    return sim
