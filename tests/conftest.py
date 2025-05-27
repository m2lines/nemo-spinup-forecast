import pytest
from lib.forecast import Simulation, Predictions, load_ts
from main_forecast import prepare


@pytest.fixture()
def setup_simulation_class(request):
    """
    Fixture to set up the simulation class
    """
    from lib.forecast import Simulation

    path = "tests/data/nemo_data_e3/"
    start = 20
    end = 50
    ye = True
    comp = 0.9
    term = request.param
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
    term = request.param
    path = "tests/data/nemo_data_e3/"
    start = 20
    end = 50
    ye = True
    comp = 0.9

    prepare(term, path, start, end, ye, comp)

    df, infos = load_ts(f"{path}/simu_prepared/{term[0]}", term[0])

    simu_ts = Predictions(term[0], df, infos)

    return simu_ts
