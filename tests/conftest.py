from importlib.resources import files

import pytest

from nemo_spinup_forecast.dimensionality_reduction import (
    dimensionality_reduction_techniques,
)
from nemo_spinup_forecast.forecast import Predictions, Simulation, load_ts
from nemo_spinup_forecast.forecast_method import forecast_techniques
from nemo_spinup_forecast.utils import (
    create_run_dir,
    get_dr_technique,
    get_forecast_technique,
    prepare,
)

# Resolve packaged config file (no reliance on repo root layout)
tech_cfg = files("nemo_spinup_forecast.configs").joinpath("techniques_config.yaml")

dr_technique = get_dr_technique(tech_cfg, dimensionality_reduction_techniques)
forecast_technique = get_forecast_technique(tech_cfg, forecast_techniques)


@pytest.fixture()
def setup_simulation_class(request):
    """Fixture to set up the simulation class."""
    # Parameters for the simulation class
    path = "tests/data/nemo_data_e3/"
    start = 20  # Start year for the simulation
    end = 50  # End year for the simulation
    ye = True  # Indicates if the simulation is yearly
    comp = 0.9  # Explained variance ratio for PCA
    term, filename = request.param  # Tuple (phycial property/term, file)

    simu = Simulation(
        path=path,
        start=start,
        end=end,
        ye=ye,
        comp=comp,
        term=term,
        filename=filename,
        dimensionality_reduction=dr_technique,
    )

    return simu


@pytest.fixture()
def setup_prediction_class(request):
    """Fixture to set up a prediction class."""
    # create a per-run directory to store results
    run_dir = create_run_dir("tests/data/nemo_data_e3/")
    out_dir = run_dir / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # TODO: Reminder about handling index of tuple term, filename
    term, filename = request.param  # term to forecast, e.g., "ssh", "toce", "soce"
    path = out_dir
    start = 20
    end = 50
    ye = True  # Indicates if the simulation is yearly
    comp = 0.9  # Explained variance ratio for PCA

    # Applies PCA and saves the results to disk
    prepare(term, filename, path, start, end, ye, comp, dr_technique)

    df, infos = load_ts(f"{path}/simu_prepared/{term}", term)

    simu_ts = Predictions(term, df, infos, forecast_technique, dr_technique)

    return simu_ts, infos
