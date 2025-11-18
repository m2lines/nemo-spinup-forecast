from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def test_import_forecast():
    """Test that the forecast module can be imported successfully."""
    try:
        import nemo_spinup_forecast.forecast  # noqa F401

        assert True, "forecast imported successfully."
    except ImportError as e:
        pytest.fail(f"Failed to import forecast: {e}")


def create_netcdf4_file(path: Path):
    """Create a sample NetCDF file with random data for testing."""
    rng = np.random.default_rng()
    time = pd.date_range("2023-01-01", periods=1)
    lat = np.linspace(-90, 90, 10)
    lon = np.linspace(-180, 180, 20)

    data = xr.DataArray(
        rng.random((1, 10, 20)),
        dims=["time", "lat", "lon"],
        coords={"time": time, "lat": lat, "lon": lon},
        name="temperature",
    )

    ds = xr.Dataset({"temperature": data})
    ds.to_netcdf(path)


def test_create_and_read_netcdf(tmp_path):
    """Test creating and reading a NetCDF file."""
    path = tmp_path / "test.nc"

    create_netcdf4_file(path)
    assert path.exists()

    ds = xr.open_dataset(path)
    assert "temperature" in ds
    ds.close()
