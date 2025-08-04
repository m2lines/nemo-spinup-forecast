import numpy as np
import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, "./lib/")
from forecast import Predictions, Simulation, load_ts
from forecast_method import forecast_techniques
from dimensionality_reduction import dimensionality_reduction_techniques
from utils import get_ocean_term, get_forecast_technique, get_dr_technique


def prepare(term, filename, simu_path, start, end, ye, comp, dr_technique):
    """
    Prepare the simulation for the forecast

    Args:
        term (str): term to forecast
        simu_path (str): path to the simulation
        start (int): start of the simulation
        end (int): end of the simulation
        ye (bool): transform monthly simulation to yearly simulation
        comp (int or float): explained variance ratio for the pcaA

    Returns:
        None

    """

    # Load yearly or monthly simulations
    simu = Simulation(
        path=simu_path,
        start=start,
        end=end,
        ye=ye,
        comp=comp,
        term=term,
        filename=filename,
        dimensionality_reduction=dr_technique,
    )
    print(f"{term} loaded")

    # Prepare simulations : start to end - removeClosedSeas - (removeSSCA) - standardize - to numpy
    simu.get_simulation_data()
    print(f"{term} prepared")

    # Exctract time series through PCA
    simu.decompose()
    print(f"PCA applied on {term}")

    os.makedirs(f"{simu_path}/simu_prepared/{term}", exist_ok=True)
    print(f"{simu_path}/simu_prepared/{term} created")

    # Create dictionary and save:
    # time series - mask - desc -(ssca) - cut(=start) - x_size - y_size - (z_size) - shape
    simu.save(f"{simu_path}/simu_prepared", term)
    print(f"{term} saved at {simu_path}/simu_repared/{term}")

    return simu


def jump(simu_path, term, steps, simu, forecast_technique, dr_technique):
    """
    Forecast the simulation

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
    y_hat, y_hat_std, metrics = simu_ts.parallel_forecast(len(simu_ts), steps)
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


def emulate(simu_path, steps, ye, start, end, comp, dr_technique, forecast_technique):
    """
    Emulate the forecast

    Args:
        simu_path (str): path to the simulation
        steps (int): number of years to forecast
        ye (bool): transform monthly simulation to yearly simulation
        start (int): start of the simulation
        end (int): end of the simulation
        comp (int or float): explained variance ratio for the pca

    Returns:
        None

    """

    run_name = ""  # "kpca_recurGP_2nd_run_"
    dino_data = [
        (get_ocean_term("SSH"), f"DINO_{run_name}1m_To_1y_grid_T.nc"),
        (get_ocean_term("Salinity"), f"DINO_{run_name}1y_grid_T.nc"),
        (get_ocean_term("Temperature"), f"DINO_{run_name}1y_grid_T.nc"),
    ]

    for term, filename in dino_data:
        print(f"Preparing {term}...")
        simu = prepare(term, filename, simu_path, start, end, ye, comp, dr_technique)
        print()
        print(f"Forecasting {term}...")
        jump(simu_path, term, steps, simu, forecast_technique, dr_technique)
        print()


if __name__ == "__main__":
    # Perform forecast

    # Example use
    # python main_forecast.py --ye True --start 25 --end 65 --comp 0.9 --steps 30 --path /path/to/simu/data

    parser = argparse.ArgumentParser(description="Emulator")
    parser.add_argument(
        "--path", type=str, help="Path to simulation data to forecast from"
    )
    parser.add_argument(
        "--ye", type=bool, help="Transform monthly simulation to yearly simulation"
    )  # Transform monthly simulation to yearly simulation
    parser.add_argument(
        "--start", type=int, help="Start of the training"
    )  # Start of the simu : 0 to keep spin up / t to cut the spin up
    parser.add_argument(
        "--end", type=int, help="End of the training"
    )  # End of the simu  (end-strat = train len)
    parser.add_argument(
        "--steps", type=int, help="Number of steps to emulate"
    )  # Number of years you want to forecast
    parser.add_argument(
        "--comp", type=str, help="Explained variance ratio for the pca"
    )  # Explained variance ratio for the pca
    args = parser.parse_args()

    # Load config file of techniques
    path_to_nemo_directory = os.path.dirname(os.path.abspath(__file__))
    path_to_nemo_directory = Path(path_to_nemo_directory)

    dr_technique = get_dr_technique(
        path_to_nemo_directory, dimensionality_reduction_techniques
    )
    forecast_technique = get_forecast_technique(
        path_to_nemo_directory, forecast_techniques
    )

    # Convert comp to int or float if possible
    if args.comp.isdigit():
        args.comp = int(args.comp)
    elif args.comp.replace(".", "", 1).isdigit():
        args.comp = float(args.comp)
    elif args.comp == "None":
        args.comp = None

    emulate(
        simu_path=args.path,
        steps=args.steps,
        ye=args.ye,
        start=args.start,
        end=args.end,
        comp=args.comp,
        dr_technique=dr_technique,
        forecast_technique=forecast_technique,
    )
