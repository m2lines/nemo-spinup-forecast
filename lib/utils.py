import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml

from lib.forecast import Simulation


def create_run_dir(base_path: str) -> Path:
    """
    Create a new timestamped run directory and update the `latest` symlink.

    A directory is created under ``<base_path>/forecasts/runs`` with a unique
    timestamp-based name. After creation, the ``latest`` symlink in
    ``<base_path>/forecasts`` is atomically updated to point to the new directory.

    Parameters
    ----------
    base_path : str
        Base path under which the run directories are stored.

    Returns
    -------
    Path
        Path to the newly created run directory.
    """
    base = Path(base_path).expanduser().resolve()
    runs_root = base / "forecasts" / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S.%fZ")
    random_id = uuid.uuid4().hex[:8]  # 8-char unique ID
    run_id = f"{ts}_{random_id}"

    run_dir = runs_root / run_id
    run_dir.mkdir(parents=False, exist_ok=False)

    # Update 'latest' symlink atomically
    _update_symlink_atomic(runs_root.parent, "latest", run_dir)
    return run_dir


def _update_symlink_atomic(base: Path, name: str, target: Path):
    base.mkdir(parents=True, exist_ok=True)
    tmp = base / f"{name}.tmp"
    final = base / name
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    os.symlink(os.path.relpath(target, base), tmp)
    os.replace(tmp, final)


def prepare(term, filename, simu_path, start, end, ye, comp, dr_technique):
    """
    Prepare the simulation for the forecast.

    Args:
        term (str): term to forecast
        simu_path (str): path to the simulation
        start (int): start of the simulation
        end (int): end of the simulation
        ye (bool): transform monthly simulation to yearly simulation
        comp (int or float): explained variance ratio for the pcaA

    Returns
    -------
        simu (Simulation): simulation object

    """
    # Load yearly or monthly simulations

    simu = Simulation(
        path=str(simu_path.parents[3]),
        start=start,
        end=end,
        ye=ye,
        comp=comp,
        term=term,
        filename=filename,
        dimensionality_reduction=dr_technique,
    )
    print(f"{term} loaded")

    simu.get_simulation_data()
    print(f"{term} prepared")

    # Exctract time series through PCA
    simu.decompose()
    print(f"PCA applied on {term}")

    os.makedirs(f"{simu_path}/simu_prepared/{term}", exist_ok=True)
    print(f"{simu_path}/simu_prepared/{term} created")

    # Create dictionary and save:
    simu.save(f"{simu_path}/simu_prepared", term)
    print(f"{term} saved at {simu_path}/simu_repared/{term}")

    return simu


def get_ocean_term(ocean_property):
    """
    Retrieve an ocean-related term from the 'ocean_terms.yaml' file.

    Parameters
    ----------
    ocean_property : str
        The key of the term to retrieve from the YAML under the 'Terms' section.
        For example, 'temperature' to fetch the corresponding ocean temperature term.

    Returns
    -------
    str or None
        The term associated with the given property, or None if the file is missing
        or the property is not defined.

    Raises
    ------
    FileNotFoundError
        If the 'ocean_terms.yaml' file cannot be found.
    KeyError
        If the specified property is not found under 'Terms' in the YAML file.
    """
    try:
        with open("ocean_terms.yaml", "r") as f:
            terms = yaml.safe_load(f)

        # Attempt to retrieve the requested term
        return terms["Terms"][ocean_property]

    except FileNotFoundError:
        print(
            "\nCouldn't find 'ocean_terms.yaml'. "
            "Please create the file with a 'Terms' section.\n"
        )
        return None

    except KeyError:
        print(
            f"\nThe term '{ocean_property}' was not found in the "
            "'Terms' section of 'ocean_terms.yaml'.\n"
        )
        return None


def get_forecast_technique(nemo_directory, forecast_techniques):
    """Retrieve a forecasting technique from the 'techniques_config.yaml' file.

    Parameters
    ----------
    nemo_directory : Path
        The directory where the 'techniques_config.yaml' file is located.
    forecast_techniques : dict
        A dictionary of available forecasting techniques.

    Returns
    -------
    ForecastTechnique
        An instance of the specified forecasting technique.

    Raises
    ------
    KeyError
        If the specified technique is not found in the `forecast_techniques` dictionary.
    """
    with open(nemo_directory / "techniques_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if config["Forecast_technique"]["name"] not in forecast_techniques:
        msg = (
            f"Forecast_technique {config['Forecast_technique']['name']} not found. "
            "Have you specified a valid forecasting technique in the config file?"
        )
        raise KeyError(msg)
    else:
        return forecast_techniques[config["Forecast_technique"]["name"]]


def get_dr_technique(nemo_directory, dimensionality_reduction_techniques):
    """Retrieve a dimensionality reduction technique from a config file.

    Parameters
    ----------
    nemo_directory : Path
        The directory where the 'techniques_config.yaml' file is located.
    dimensionality_reduction_techniques : dict
        A dictionary of available dimensionality reduction techniques.

    Returns
    -------
    DimensionalityReductionTechnique
        An instance of the specified dimensionality reduction technique.

    Raises
    ------
    KeyError
        If the specified technique is not found in the
        `dimensionality_reduction_techniques` dictionary.
    """
    with open(nemo_directory / "techniques_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if config["DR_technique"]["name"] not in dimensionality_reduction_techniques:
        msg = (
            f"DR_technique {config['DR_technique']['name']} not found. "
            "Have you specified a valid dimensionality reduction "
            "technique in the config file?"
        )
        raise KeyError(msg)
    else:
        return dimensionality_reduction_techniques[config["DR_technique"]["name"]]
