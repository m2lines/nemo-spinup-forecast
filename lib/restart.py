# Adapted from code by Maud Tissot (Spinup-NEMO)
# Original source: https://github.com/maudtst/Spinup-NEMO
# Licensed under the MIT License
#
# Modifications in this version by ICCS Climate Tools Team, 2025
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import glob
from utils import get_ocean_term


# SUPER LONG PEUT ETRE LE FAIR EN BASH OU ERREUR
def getRestartFiles(path, radical, puzzled=False):
    """
    Get the restart files of the last simulation step

    Parameters:
        path (str): The path to the restarts files
        radical (str): Radical of the file name
                       (e.g. for "OCE_CM65-LR-pi-SpinupRef_19141231_00390.nc", it's "OCE_CM65-LR-pi-SpinupRef_19141231")
        puzzled (bool): Return Complete Restart File (False) or List of windowed files (True)

    Returns:
        list (of str) or str : List of restart puzzled files paths, or unique restart file path
    """
    print("Retrieving Restart File(s)")
    if puzzled == True:
        return glob.glob(path + radical + "_*.nc").sorted()
    else:
        try:
            print("Path: ", path)
            print(path + radical + ".nc")
            return glob.glob(path + radical + ".nc")[0]
        except IndexError:
            print(
                "No Full Restart Found : Use NEMO_REBUILD from NEMO tools if you didn't do it yet."
            )
            raise


def getMaskFile(maskpath, restart):
    """
    Get the mask file and adapt it to fit the restart file coordinate system.

    Parameters:
        maskpath (str): The path to the mask file, name of the file included.
        restart (xarray.Dataset): The full restart file we are modifying.

    Returns:
        mask (xarray.Dataset) : The mask dataset adapted to the restart file.
    """
    mask = xr.open_dataset(maskpath, decode_times=False)
    # Harmonizing the structure of mask with that of restart
    # Isaac, default configuration for DINO mask is the same as the restart file
    # mask = mask.swap_dims(dims_dict={"z": "nav_lev","t":"time_counter"})
    mask["time_counter"] = restart["time_counter"]
    return mask


def recordFullRestart(path, radical, restart):
    """
    Record the Modified Full Restart Dataset to a file in the input directory for analysis.

    Parameters:
        path (str): The path to the restart file directory
        radical (str): Radical of the original restart file name
                       (e.g. for "OCE_CM65-LR-pi-SpinupRef_19141231_restart_00390.nc", it’s "OCE_CM65-LR-pi-SpinupRef_19141231_restart")
        restart (xarray.Dataset): The full restart file we are modifying.

    Returns:
        str : Recording Completion Message
    """
    restart.to_netcdf(path + "NEW_" + radical + ".nc")
    print("Restart saved as : " + path + "NEW_" + radical + ".nc")
    return "Recording Complete"


def recordPiecedRestart(path, radical, restart):
    """
    Record the Modified Puzzled Restart Datasets to files in the input directory for analysis.
    It is done by iterating on the existing puzzled dataset files, and creating new ones by appending "NEW_" in front of the filename.
    If the user want to overwrite the old files, they will need to do it manually (a 4 line bash script is available).

    Parameters:
        path (str): The path to the restart file directory
        radical (str): Radical of the original restart file name
                       (e.g. for "OCE_CM65-LR-pi-SpinupRef_19141231_restart_00390.nc", it’s "OCE_CM65-LR-pi-SpinupRef_19141231_restart")
        restart (xarray.Dataset): The full restart file we are modifying.

    Returns:
        str : Recording Completion Message
    """
    size = len(glob.glob(path + radical + "_*.nc"))
    for index in range(size):
        Restart_NEW = xr.open_dataset(path + radical + "_%04d.nc" % (index))
        x_slice, y_slice = getXYslice(Restart_NEW)
        Restart_NEW["un"] = restart["un"][:, :, y_slice, x_slice]
        Restart_NEW["vn"] = restart["vn"][:, :, y_slice, x_slice]
        Restart_NEW["ub"] = restart["ub"][:, :, y_slice, x_slice]
        Restart_NEW["vb"] = restart["vb"][:, :, y_slice, x_slice]
        Restart_NEW["sn"] = restart["sn"][:, :, y_slice, x_slice]
        Restart_NEW["tn"] = restart["tn"][:, :, y_slice, x_slice]
        Restart_NEW["sb"] = restart["sb"][:, :, y_slice, x_slice]
        Restart_NEW["tb"] = restart["tb"][:, :, y_slice, x_slice]

        Restart_NEW["rhop"] = restart["rhop"][:, :, y_slice, x_slice]

        Restart_NEW["sshn"] = restart["sshn"][:, y_slice, x_slice]
        Restart_NEW["sshb"] = restart["sshb"][:, y_slice, x_slice]

        Restart_NEW["ssv_m"] = restart["ssv_m"][:, y_slice, x_slice]
        Restart_NEW["ssu_m"] = restart["ssu_m"][:, y_slice, x_slice]
        Restart_NEW["sst_m"] = restart["sst_m"][:, y_slice, x_slice]
        Restart_NEW["sss_m"] = restart["sss_m"][:, y_slice, x_slice]
        Restart_NEW["ssh_m"] = restart["ssh_m"][:, y_slice, x_slice]
        Restart_NEW["e3t_m"] = restart["e3t_m"][:, y_slice, x_slice]

        Restart_NEW.to_netcdf(path + "NEW_" + radical + "_%04d.nc" % (index))
        print(
            "Restart Piece saved as : " + path + "NEW_" + radical + "_%04d.nc" % (index)
        )
    return "Recording Complete"


def load_predictions(restart, dirpath="/data/mtissot/simus_predicted"):
    """
    Load predicted data from saved NumPy files into the restart array.
    We use the same prediction step for now and before steps (e.g sshn/sshb).
    We also update the intermediate step surface variables (e.g. sst_m).

    Returns:
        restart (xarray.Dataset)  with the following primary variables modified :
            ssh    (xarray.DataArray) : ssh sea surface height predictions  - (t,y,x)
            so     (xarray.DataArray) : salinity predictions                - (t,z,y,x)
            thetao (xarray.DataArray) : temperature predictions             - (t,z,y,x)
    """
    ## Loading new SSH in directly affected variables
    ## (loading zos.npy, selecting last snapshot, then converting to fitting xarray.DataArray, and cleaning the nans)
    try:
        term = get_ocean_term("SSH")
        if term is not None:
            zos = np.load(dirpath + f"/{term}.npy")[-1:]
        sshn = xr.DataArray(zos)

        # # Check dimensions

        restart["sshn"] = xr.DataArray(
            zos, dims=("time_counter", "y", "x"), name="sshn"
        ).fillna(0)
        restart["sshb"] = restart["sshn"].copy()
        restart["ssh_m"] = restart["sshn"].copy()
    except FileNotFoundError:
        print("Couldn’t find a SSH/ZOS prediction file, keeping the original SSH.")

    ## Loading new SO in directly affected variables
    ## (loading so.npy, selecting last snapshot, then converting to fitting xarray.DataArray, and cleaning the nans)

    try:
        term = get_ocean_term("Salinity")
        if term is not None:
            so = np.load(dirpath + f"/{term}.npy")[-1:]

        restart["sn"] = xr.DataArray(
            so, dims=("time_counter", "nav_lev", "y", "x"), name="sn"
        ).fillna(0)
        restart["sb"] = restart["sn"].copy()
        restart["sss_m"] = restart["sn"].isel(nav_lev=0).copy()
    except FileNotFoundError:
        print("Couldn’t find a SO prediction file, keeping the original SO.")

    ## Loading new THETAO in directly affected variables
    ## (loading thetao.npy, selecting last snapshot, then converting to fitting xarray.DataArray, and cleaning the nans)
    try:
        term = get_ocean_term("Temperature")
        if term is not None:
            thetao = np.load(dirpath + f"/{term}.npy")[-1:]

        restart["tn"] = xr.DataArray(
            thetao, dims=("time_counter", "nav_lev", "y", "x"), name="tn"
        ).fillna(0)
        restart["tb"] = restart["tn"].copy()
        restart["sst_m"] = restart["tn"].isel(nav_lev=0).copy()
    except FileNotFoundError:
        print("Couldn’t find a THETAO prediction file, keeping the original THETAO.")

    return restart


def getXYslice(restart):
    """
    Given a Restart Dataset with 'DOMAIN_position_first' and 'DOMAIN_position_last' attributes,
    this function calculates and returns slices for x and y dimensions.

    Parameters:
        restart (xarray.Dataset) : Restart file

    Returns
        x_slice (slice dtype=float) : range of x positions
        y_slice (slice dtype=float) : range of y positions
    """
    First = restart.DOMAIN_position_first
    Last = restart.DOMAIN_position_last
    x_slice = slice(First[0] - 1, Last[0])
    y_slice = slice(First[1] - 1, Last[1])
    return x_slice, y_slice


def toXarray(var, name, dep=True, fillna=True):
    """
    Converts a numpy array into an xarray DataArray and replace nan values by 0 to obtain the same data format as in restart files.

    Parameters:
        var (numpy array)       : The  array to be converted.
        name (str)              : The name to be assigned to the resulting xarray DataArray.
        dep (bool, optional)    : If True, indicates that the array has a depth dimension.
                                   Defaults to True.
        fillna (bool, optional) : If True, fills NaN values with 0 after conversion.
                                   Defaults to True.

    Returns:
        array (xarray.DataArray): An xarray DataArray object representing the input numpy array.
    """
    if dep:
        if len(np.shape(var)) == 4:
            array = xr.DataArray(
                var, dims=("time_counter", "nav_lev", "y", "x"), name=name
            )
        elif len(np.shape(var)) == 3:
            array = xr.DataArray(var, dims=("nav_lev", "y", "x"), name=name)
    else:
        if len(np.shape(var)) == 3:
            array = xr.DataArray(var, dims=("time_counter", "y", "x"), name=name)
        elif len(np.shape(var)) == 2:
            array = xr.DataArray(var, dims=("y", "x"), name=name)
    return array.fillna(0)


def propagate_pred(restart, mask):
    """
    Update the variables indirectly affected by the prediction on primary variables (e.g. geostrophic velocities and density).


    Parameters:
        restart (xarray.Dataset) : Full Restart file

    Returns:
        restart (xarray.Dataset) : Full Restart file with all variables modified according to the predictions.
    """

    thetao = restart.tn
    so = restart.sn

    deptht = get_deptht(restart, mask)
    rhop_new, _ = get_density(thetao, so, deptht, mask.tmask)

    e3t_new = update_e3t(restart, mask)
    u_new = update_u_velocity(restart, mask, e3t_new).fillna(0)
    v_new = update_v_velocity(restart, mask, e3t_new).fillna(0)

    restart["un"] = u_new.copy()
    restart["vn"] = v_new.copy()
    restart["ub"] = u_new.copy()
    restart["vb"] = v_new.copy()
    restart["rhop"] = rhop_new.fillna(0)
    restart["ssv_m"] = v_new.isel(nav_lev=0)
    restart["ssu_m"] = u_new.isel(nav_lev=0)
    restart["e3t_m"] = e3t_new.isel(nav_lev=0).fillna(0)

    return restart


def update_e3t(restart, mask):
    """
    Update e3t_m : the cell thickness of the top layer.
    Get e3t : the cell thickness for all dimensions, we can use e3t to get the new bathymetry and to update u and velocities
              e3t = e3t_initital * (1+tmask4D*np.expand_dims(np.tile(ssh*ssmask/(bathy+(1-ssmask)),(75,1,1)),axis=0))
    Get deptht : The depth of each cell on grid. We use it to update the density rhop.

    Parameters:
        mask (xarray.Dataset) : Mask Dataset containing tmask values
        restart (xarray.Dataset)      : Restart file

    Returns:
        e3t (numpy.ndarray) : Updated array of z-axis cell thicknesses.
    """
    # e3t_ini = restart.e3t_ini #Changed
    e3t_ini = mask.e3t_0  # initial z axis cell's thickness on grid T - (t,z,y,x)
    ssmask = mask.tmask.max(
        dim="nav_lev"
    )  # continent mask                            - (t,y,x)
    bathy = e3t_ini.sum(
        dim="nav_lev"
    )  # inital Bathymetry                         - (t,y,x)
    ssh = restart.sshn  # Sea Surface Height                        - (t,y,x)
    tmask = mask.tmask  # bathy mask on grid T                      - (t,z,y,x)
    e3t = e3t_ini * (1 + ssh * ssmask / (bathy + 1 - ssmask))  # - (t,y,x)
    return e3t


def get_deptht(restart, mask):
    """
    Calculate the depth of each vertical level on grid T in the 3D grid.

    Parameters:
        restart (xarray.Dataset)     : The dataset containing ocean model variables.
        mask (xarray.Dataset) : The dataset containing mask variables.

    Returns:
        deptht (numpy.array) : The depth of each vertical level.
    """
    ssh = restart.sshn
    e3w_0 = mask.e3w_0  # initial z axis cell's thickness on grid W - (t,z,y,x)
    e3t_0 = mask.e3t_0  # initial z axis cell's thickness on grid T - (t,z,y,x)
    tmask = mask.tmask  # grid T continent mask                     - (t,z,y,x)
    ssmask = tmask[:, 0]  # bathymetry                                - (t,y,x)
    bathy = e3t_0.sum(
        dim="nav_lev"
    )  # initial condition depth 0                 - (t,z,y,x)
    depth_0 = e3w_0.copy()
    depth_0[:, 0] = 0.5 * e3w_0[:, 0]
    depth_0[:, 1:] = depth_0[:, 0:1].data + e3w_0[:, 1:].cumsum(dim="nav_lev")
    deptht = depth_0 * (1 + ssh / (bathy + 1 - ssmask)) * tmask
    return deptht


def update_rhop(restart, mask):
    """
    Update the rhop variable in the array based on temperature (thetao) and salinity (so).

    Parameters:
        restart (xarray.Dataset) : Restart file
        mask    (xarray.Dataset) : Mask file

    Returns:
        rhop (xarray.DataArray)
    """
    x_slice, y_slice = getXYslice(array)
    so = restart["sn"]
    thetao = restart["tn"]
    tmask = mask["tmask"][-1:, :, y_slice, x_slice]
    deptht = get_depth(restart, mask)

    rhop, rho_insitu = get_density(thetao, so, deptht, tmask)
    return rhop


def get_density(thetao, so, depth, tmask):
    """
    Compute potential density referenced at the surface and density anomaly.

    Parameters:
        thetao (numpy.array) : Temperature array - (t,z,y,x).
        so (numpy.array)     : Salinity array    - (t,z,y,x).
        depth (numpy.array)  : Depth array       - (t,z,y,x).
        tmask (numpy.array)  : Mask array        - (t,z,y,x).

    Returns:
        tuple: A tuple containing:
            array: Potential density referenced at the surface.
            array: Density anomaly.
    """
    rdeltaS = 32.0
    r1_S0 = 0.875 / 35.16504
    r1_T0 = 1.0 / 40.0
    r1_Z0 = 1.0e-4

    EOS000 = 8.0189615746e02
    EOS100 = 8.6672408165e02
    EOS200 = -1.7864682637e03
    EOS300 = 2.0375295546e03
    EOS400 = -1.2849161071e03
    EOS500 = 4.3227585684e02
    EOS600 = -6.0579916612e01
    EOS010 = 2.6010145068e01
    EOS110 = -6.5281885265e01
    EOS210 = 8.1770425108e01
    EOS310 = -5.6888046321e01
    EOS410 = 1.7681814114e01
    EOS510 = -1.9193502195
    EOS020 = -3.7074170417e01
    EOS120 = 6.1548258127e01
    EOS220 = -6.0362551501e01
    EOS320 = 2.9130021253e01
    EOS420 = -5.4723692739
    EOS030 = 2.1661789529e01
    EOS130 = -3.3449108469e01
    EOS230 = 1.9717078466e01
    EOS330 = -3.1742946532
    EOS040 = -8.3627885467
    EOS140 = 1.1311538584e01
    EOS240 = -5.3563304045
    EOS050 = 5.4048723791e-01
    EOS150 = 4.8169980163e-01
    EOS060 = -1.9083568888e-01
    EOS001 = 1.9681925209e01
    EOS101 = -4.2549998214e01
    EOS201 = 5.0774768218e01
    EOS301 = -3.0938076334e01
    EOS401 = 6.6051753097
    EOS011 = -1.3336301113e01
    EOS111 = -4.4870114575
    EOS211 = 5.0042598061
    EOS311 = -6.5399043664e-01
    EOS021 = 6.7080479603
    EOS121 = 3.5063081279
    EOS221 = -1.8795372996
    EOS031 = -2.4649669534
    EOS131 = -5.5077101279e-01
    EOS041 = 5.5927935970e-01
    EOS002 = 2.0660924175
    EOS102 = -4.9527603989
    EOS202 = 2.5019633244
    EOS012 = 2.0564311499
    EOS112 = -2.1311365518e-01
    EOS022 = -1.2419983026
    EOS003 = -2.3342758797e-02
    EOS103 = -1.8507636718e-02
    EOS013 = 3.7969820455e-01

    zh = depth * r1_Z0  # depth
    zt = thetao * r1_T0  # temperature
    zs = np.sqrt(np.abs(so + rdeltaS) * r1_S0)  # square root salinity
    ztm = tmask

    zn3 = EOS013 * zt + EOS103 * zs + EOS003
    zn2 = (
        (EOS022 * zt + EOS112 * zs + EOS012) * zt + (EOS202 * zs + EOS102) * zs + EOS002
    )
    zn1 = (
        (
            (
                (EOS041 * zt + EOS131 * zs + EOS031) * zt
                + (EOS221 * zs + EOS121) * zs
                + EOS021
            )
            * zt
            + ((EOS311 * zs + EOS211) * zs + EOS111) * zs
            + EOS011
        )
        * zt
        + (((EOS401 * zs + EOS301) * zs + EOS201) * zs + EOS101) * zs
        + EOS001
    )
    zn0 = (
        (
            (
                (
                    (
                        (EOS060 * zt + EOS150 * zs + EOS050) * zt
                        + (EOS240 * zs + EOS140) * zs
                        + EOS040
                    )
                    * zt
                    + ((EOS330 * zs + EOS230) * zs + EOS130) * zs
                    + EOS030
                )
                * zt
                + (((EOS420 * zs + EOS320) * zs + EOS220) * zs + EOS120) * zs
                + EOS020
            )
            * zt
            + ((((EOS510 * zs + EOS410) * zs + EOS310) * zs + EOS210) * zs + EOS110) * zs
            + EOS010
        )
        * zt
        + (
            ((((EOS600 * zs + EOS500) * zs + EOS400) * zs + EOS300) * zs + EOS200) * zs
            + EOS100
        )
        * zs
        + EOS000
    )

    zn = ((zn3 * zh + zn2) * zh + zn1) * zh + zn0

    rhop = zn0 * ztm  # potential density referenced at the surface
    rho_insitu = zn * ztm  # density anomaly (masked)
    return rhop, rho_insitu


def plot_density_infos(array, e3t_new, min_=1017):
    """
    Plot density (rhop) information: surface density, density as a function of depth, and the difference in density as a function of depth.
    The difference provides insights into density errors, particularly where it decreases instead of increasing

    Parameters:
        array (xarray.Dataset) : Restart file contains density informations
        e3t_new (numpy.array)  : Array representing the new z-axis cell thickness : the distance between two grid points.
        min_ (float, optional) : Minimum value for color scale. Defaults to 1017.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    a = axes[0].pcolor(array["rhop"][0, 0], vmin=min_)
    fig.colorbar(a, ax=axes[0])

    rhop = array["rhop"].where(array["rhop"][0] != 0.0, np.nan)
    diff_rhop = np.diff(rhop.isel(time_counter=0), axis=0) / e3t_new[0, :-1]

    for i in range(array["rhop"].sizes["x"]):
        for j in range(array["rhop"].sizes["y"]):
            rhop.isel(time_counter=0, x=i, y=j).plot(ax=axes[1])
            axes[2].plot(diff_rhop[:, j, i])

    axes[0].set_title("Surface density")
    axes[1].set_title("Density as a function of depth")
    axes[2].set_title("Diff Density as a function of depth")


# FONCTIONNE MAIS A VOIR
def regularize_rho(rho):
    """
    Regularize the rho variable to ensure the density is alway superieur or equal at a lower depth.
    If the value found at k-1 depth is lower than the value found at k. k-1 value is replaces by k value.

    Parameters:
        rho (numpy.ndarray): Array representing density with dimensions (time, depth, latitude, longitude).

    Returns:
        numpy.ndarray : Regularized array of density.
    """
    t, z, y, x = np.shape(rho)
    for i in range(x):
        for j in range(y):
            for k in range(z - 1):
                if rho[0, k, j, i] > rho[0, k + 1, j, i]:
                    rho[0, k + 1, j, i] = rho[0, k, j, i]
    return rho


def update_u_velocity(
    restart, mask, e3t_new
):  # e3t_new             = maskarray["e3t_0"].values[0,:,y_slice,x_slice]
    """
    Update the v-component velocity array.Meridional

    Parameters:
        restart (xarray.Dataset)     : Restart file.
        mask    (xarray.Dataset)     : Mask dataset.
        e3t_new (numpy.ndarray)      : Updated array of z-axis cell thicknesses.

    Returns:
        None
    """
    un = restart.un.copy()  # initial v velocity of the restart         - (t,z,y,x)
    thetao = restart.tn
    so = restart.sn
    deptht = get_deptht(restart, mask)
    e2t = mask.e2t  # initial y axis cell's thickness on grid T - (y,x)
    ff_f = mask.ff_f  # corriolis force                           - (y,x)
    tmask = mask.tmask
    umask = mask.umask
    vmask = mask.vmask

    _, rho_insitu = get_density(thetao, so, deptht, tmask)
    rho_insitu = rho_insitu.where(
        tmask
    )  # updated density                           - (t,z,y,x)

    ind_prof_u = (umask.argmin(dim="nav_lev") - 1) * umask.isel(nav_lev=0)

    diff_y = rho_insitu.roll(y=-1) - rho_insitu  #                - (t,z,y,x)
    u_new = 9.81 / ff_f * (diff_y / rho_insitu * e3t_new / e2t).cumsum(dim="nav_lev")
    u_new = u_new - u_new.isel(nav_lev=ind_prof_u)
    un_new = add_bottom_velocity(un, u_new, umask[0])  # add V_0        - (t,z,y,x)

    return un_new


def update_v_velocity(
    restart, mask, e3t_new
):  # e3t_new             = maskarray["e3t_0"].values[0,:,y_slice,x_slice]
    """
    Update the v-component velocity array.Meridional

    Parameters:
        restart (xarray.Dataset)     : Restart file.
        mask (xarray.Dataset) : Mask dataset.
        e3t_new (numpy.ndarray)    : Updated array of z-axis cell thicknesses.

    Returns:
        New v_velocity
    """
    vn = restart.vn.copy()  # initial v velocity of the restart         - (t,z,y,x)
    thetao = restart.tn
    so = restart.sn
    deptht = get_deptht(restart, mask)
    e1t = mask.e1t  # initial y axis cell's thickness on grid T - (y,x)
    ff_f = mask.ff_f  # corriolis force                           - (y,x)
    tmask = mask.tmask
    vmask = mask.vmask

    _, rho_insitu = get_density(thetao, so, deptht, tmask)
    rho_insitu = rho_insitu.where(
        tmask
    )  # updated density                           - (t,z,y,x)

    ind_prof_v = (vmask.argmin(dim="nav_lev") - 1) * vmask.isel(nav_lev=0)

    diff_x = -rho_insitu.roll(x=1) + rho_insitu  #                - (t,z,y,x)
    v_new = (
        9.81 / ff_f * (diff_x / rho_insitu * e3t_new / e1t).cumsum(dim="nav_lev")
    )  # v without V_0  - (t,z,y,x) C: On intègre vers le fond puis on retire la valeur au fond sur toute la colonne pour avoir v_fond=vo
    v_new = v_new - v_new.isel(nav_lev=ind_prof_v)
    vn_new = add_bottom_velocity(vn, v_new, vmask[0])

    return vn_new


def add_bottom_velocity(v_restart, v_update, mask):
    """
    Add bottom velocity values to the updated velocity array.

    Parameters:
        v_restart (numpy.array) : Restart velocity array                           - (t,z,y,x)
        v_update (numpy.array)  : New velocity array without the initial condition - (t,z,y,x)
        mask (numpy.array)      : Mask array indicating presence of water cells    - (z,y,x)

    Returns:
        v_restart (numpy.array): Velocity array with bottom velocity values added.
    """
    ind_prof = (mask.argmin(dim="nav_lev") - 1) * mask.isel(nav_lev=0)
    v_fond = v_restart.isel(nav_lev=ind_prof, time_counter=0)
    mask_nan_update = np.isnan(v_update)
    v_new = mask_nan_update * v_restart + (1 - mask_nan_update) * (v_fond + v_update)
    return v_restart
