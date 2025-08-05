import pytest
from lib.forecast import Simulation, Predictions, load_ts
from main_forecast import prepare


@pytest.fixture()
def setup_simulation_class(request):
    """
    Fixture to set up the simulation class
    """
    from lib.forecast import Simulation

    # Parameters for the simulation class
    path = "tests/data/nemo_data_e3/"
    start = 20  # Start year for the simulation
    end = 50  # End year for the simulation
    ye = True  # Indicates if the simulation is yearly
    comp = 0.9  # Explained variance ratio for PCA
    term = request.param  # Term to forecast, e.g., "ssh", "toce", "soce"
    # ("toce", "DINO_1y_grid_T.nc")

    simu = Simulation(
        path=path,
        start=start,
        end=end,
        ye=ye,
        comp=comp,
        term=term,
    )

    return simu


@pytest.fixture()
def setup_prediction_class(request):
    """
    Fixture to set up a prediction class
    """
    # TODO: Reminder about handling index of tuple term, filename
    term = request.param  # term to forecast, e.g., "ssh", "toce", "soce"
    path = "tests/data/nemo_data_e3/"
    start = 20
    end = 50
    ye = True  # Indicates if the simulation is yearly
    comp = 0.9  # Explained variance ratio for PCA

    # Applies PCA and saves the results to disk
    prepare(term, path, start, end, ye, comp)

    df, infos = load_ts(f"{path}/simu_prepared/{term[0]}", term[0])

    simu_ts = Predictions(term[0], df, infos)

    return simu_ts
