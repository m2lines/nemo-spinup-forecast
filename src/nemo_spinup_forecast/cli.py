# Adapted from code by Maud Tissot (Spinup-NEMO)
# Original source: https://github.com/maudtst/Spinup-NEMO
# Licensed under the MIT License
#
# Modifications in this version by ICCS, 2025
import argparse
import os
from pathlib import Path

import numpy as np

from nemo_spinup_forecast.dimensionality_reduction import (
    dimensionality_reduction_techniques,
)
from nemo_spinup_forecast.forecast import Predictions, load_ts
from nemo_spinup_forecast.forecast_method import forecast_techniques
from nemo_spinup_forecast.utils import (
    create_run_dir,
    get_dr_technique,
    get_forecast_technique,
    get_ocean_term,
    prepare,
)


def jump(simu_path, term, steps, simu, forecast_technique, dr_technique):
    """
    Forecast the simulation.

    Args:
        simu_path (str): path to the simulation
        term (str): term to forecast
        steps (int): number of years to forecast
        simu (Simulation): simulation object
    Returns:
        None

    """
    df, infos = load_ts(
        f"{simu_path}/simu_prepared/{term}", term
    )  # load dataframe and infos

    # Create instance of prediction class
    simu_ts = Predictions(term, df, infos, forecast_technique, dr_technique)
    print(f"{term} time series loaded")

    # Forecast
    y_hat, _y_hat_std, _metrics = simu_ts.parallel_forecast(len(simu_ts), steps)
    print(f"{term} time series forcasted")

    # Reconstruct n predicted components
    # n = len(simu_ts.info["pca"].components_) # PCA
    # n = simu_ts.info["pca"].n_components # Kernel PCA

    n = simu.get_num_components()
    print(f"Number of components: {n}")
    predictions_zos = simu.reconstruct(y_hat, n, infos, begin=0)  # len(simu_ts))
    print(f"{term} predictions reconstructed")

    os.makedirs(f"{simu_path}/simu_predicted/", exist_ok=True)
    np.save(f"{simu_path}/simu_predicted/{term}.npy", predictions_zos)  # Save
    print(f"{term} predictions saved at {simu_path}/simu_predicted/{term}.npy")


def emulate(
    simu_path,
    steps,
    ye,
    start,
    end,
    comp,
    dr_technique,
    forecast_technique,
    ocean_terms_path: Path | str | None = None,
):
    """
    Emulate the forecast.

    Parameters
    ----------
    simu_path : str | Path
        Output directory for prepared and predicted data.
    steps : int
        Number of steps to emulate (years to forecast).
    ye : bool
        Transform monthly simulation to yearly simulation.
    start : int
        Start of the training interval.
    end : int
        End of the training interval.
    comp : int | float | None
        Explained variance ratio for the PCA.
    dr_technique : Any
        Selected dimensionality reduction technique.
    forecast_technique : Any
        Selected forecasting technique.
    ocean_terms_path : Path | str | None, optional
        Optional path to an ocean_terms.yaml file. When provided, ocean
        variable names are resolved from this file instead of the
        packaged default.

    Returns
    -------
    None

    """
    run_name = ""  # "kpca_recurGP_2nd_run_"
    # Resolve ocean terms using provided YAML path if given
    dino_data = [
        (
            get_ocean_term("SSH", yaml_path=ocean_terms_path),
            f"DINO_{run_name}1m_To_1y_grid_T.nc",
        ),
        (
            get_ocean_term("Salinity", yaml_path=ocean_terms_path),
            f"DINO_{run_name}1y_grid_T.nc",
        ),
        (
            get_ocean_term("Temperature", yaml_path=ocean_terms_path),
            f"DINO_{run_name}1y_grid_T.nc",
        ),
    ]

    for term, filename in dino_data:
        print(f"Preparing {term}...")
        simu = prepare(term, filename, simu_path, start, end, ye, comp, dr_technique)
        print()
        print(f"Forecasting {term}...")
        jump(simu_path, term, steps, simu, forecast_technique, dr_technique)
        print()


def main(argv=None) -> int:
    """Entry point for the emulator CLI."""
    parser = argparse.ArgumentParser(description="Emulator")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to simulation data to forecast from",
    )
    parser.add_argument(
        "--ye",
        type=bool,
        help="Transform monthly simulation to yearly simulation",
    )
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Start of the training (0 to keep spin up / t to cut the spin up)",
    )
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="End of the training (end-start = train len)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Number of steps to emulate (years to forecast)",
    )
    parser.add_argument(
        "--comp",
        type=str,
        default="None",
        help="Explained variance ratio for the PCA (int, float, or 'None')",
    )
    parser.add_argument(
        "--ocean-terms",
        type=str,
        default=None,
        help="Path to ocean_terms.yaml (overrides the packaged default)",
    )
    parser.add_argument(
        "--techniques-config",
        type=str,
        default=None,
        help="Path to techniques_config.yaml (overrides package default)",
    )

    args = parser.parse_args(argv)

    # Load config file of techniques
    if args.techniques_config:
        techniques_config_path = Path(args.techniques_config).expanduser().resolve()
    else:
        techniques_config_path = (
            Path(os.path.dirname(os.path.abspath(__file__)))
            / "configs/techniques_config.yaml"
        )

    dr_technique = get_dr_technique(
        techniques_config_path, dimensionality_reduction_techniques
    )
    forecast_technique = get_forecast_technique(
        techniques_config_path, forecast_techniques
    )

    # Create a per-run directory to store results
    run_dir = create_run_dir(args.path)
    out_dir = run_dir / "forecasts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert comp to int or float if possible
    if args.comp is not None:
        if args.comp.isdigit():
            args.comp = int(args.comp)
        elif args.comp.replace(".", "", 1).isdigit():
            args.comp = float(args.comp)
        elif args.comp == "None":
            args.comp = None

    emulate(
        simu_path=out_dir,
        steps=args.steps,
        ye=args.ye,
        start=args.start,
        end=args.end,
        comp=args.comp,
        dr_technique=dr_technique,
        forecast_technique=forecast_technique,
        ocean_terms_path=(
            Path(args.ocean_terms).expanduser().resolve() if args.ocean_terms else None
        ),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
