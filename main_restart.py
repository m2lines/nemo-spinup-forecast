import sys
import argparse

sys.path.insert(0, "lib/")
from lib.restart import *
import xarray as xr


def update_restart_slice(restart_file, restart_name, mask_file):
    # restart file "/thredds/idris/work/ues27zx/Restarts/"   mask file '/thredds/idris/work/ues27zx/eORCA1.4.2_mesh_mask_modJD.nc'
    """
    Update a restart file with new predictions and related variables.

    Parameters:
        restart_file (str) : Path to the existing restart file.
        restart_name (str) : Name of the restart file.
        file_mask (str)    : Path to the mask file.

    Returns:
        None
    """
    restart_array = xr.open_dataset(
        restart_file + restart_name, decode_times=False
    )  # load restart file
    mask_array = xr.open_dataset(mask_file, decode_times=False)  # load mask file
    zos_new, so_new, thetao_new = (
        restart.load_predictions()
    )  # load ssh, so and thetao predictions
    restart.update_pred(
        restart_array, zos_new, so_new, thetao_new
    )  # update restart with ssh, so and thetao predictions
    e3t_new = restart.update_e3tm(restart_array, mask_array)  # update e3tm and gete e3t
    deptht_new = restart.get_deptht(
        restart_array, mask_array
    )  # get new deptht for density
    restart.update_rhop(restart_array, mask_array, deptht_new)  # update density
    restart.update_v_velocity(
        restart_array, mask_array, e3t_new[0]
    )  # update meridional velocity
    restart.update_u_velocity(
        restart_array, mask_array, e3t_new[0]
    )  # update zonal velocity
    array = array.rename_vars(
        {"xx": "x", "yy": "y"}
    )  # inverse transformation of x and y vars
    # Restart.to_netcdf(restart_file+restart_name)                                  # save file


# PAs EU LE TEMPS D'ESSAYER
def update_Restarts(restarts_file, mask_file, jobs=10):
    """
    Update multiple restart files in parallel.

    Parameters:
        restarts_file (str)  : Path to the directory containing restart files.
        mask_file (str)      : Path to the mask file.
        jobs (int, optional) : Number of parallel jobs to run. default 10.

    Returns:
        None
    """
    restart_names = restart.getRestartFiles(
        restarts_file
    )  # SUPER LONG PEUT ETRE LE FAIR EN BASH OU ERREUR
    Parallel(jobs)(
        delayed(update_restart_slice)(restarts_file, file, mask_file)
        for file in restart_names
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update of restart files")
    parser.add_argument(
        "--restart_path", type=str, help="path of restart file directory"
    )
    parser.add_argument("--radical", type=str, help="radical of restart filename")
    parser.add_argument("--mask_file", type=str, help="adress of mask file")
    parser.add_argument(
        "--prediction_path", type=str, help="path of prediction directory"
    )
    args = parser.parse_args()

    restart = xr.open_dataset(
        getRestartFiles(args.restart_path, args.radical), decode_times=False
    )
    mask = getMaskFile(args.mask_file, restart)
    restart = load_predictions(restart, dirpath=args.prediction_path)
    restart = propagate_pred(restart, mask)
    recordFullRestart(args.restart_path, args.radical, restart)
    recordPiecedRestart(args.restart_path, args.radical, restart)

    print("""All done. Now you just need to :
                - Back transform the coordinates of the pieced restart files using ncks to the original version (see bash script xarray_to_CMIP.sh)
                - Rename/Overwrite the "NEW_" restart files to their old version if you’re happy with them (see other bash script rewrite.sh)
                - Point to the restart directory in your simulation config.card (if all your non-NEMO restart files are also in the restart_path directory, of course).
                  You might need to reorganize them in a ./OCE/Restart/CM....nc structure instead of ./OCE_CM...nc (there’s the rename.sh bash script for that) but normally it should work without.
             You can see the example script Jumper.sh for how to do most of that. See you soon. :) """)

    # update_Restarts(restarts_file=args.restarts_file,mask_file=args.mask_file)

    # update_restart_files

    # python SpinUp/jumper/main/main_restart.py --restart_files '/thredds/idris/work/ues27zx/eORCA1.4.2_mesh_mask_modJD.nc' --mask_file '/thredds/idris/work/ues27zx/eORCA1.4.2_mesh_mask_modJD.nc'
