# $1 : Radical
#
#


import numpy as np
import xarray as xr
import sys


def getXYslice(array):
    """
    Given a Restart array with 'DOMAIN_position_first' and 'DOMAIN_position_last' attributes,
    this function calculates and returns slices for x and y dimensions.

    Parameters:
        array (xarray.DataArray) : Restart file

    Returns
        x_string (slice dtype=float) : range of x positions
        y_string (slice dtype=float) : range of y positions
    """
    First = array.DOMAIN_position_first
    Last = array.DOMAIN_position_last
    x_slice = slice(First[0] - 1, Last[0])
    y_slice = slice(First[1] - 1, Last[1])
    return x_slice, y_slice


def density(thetao, so, depth, tmask):
    """
    Compute potential density referenced at the surface and in-situ density.
    C : Add source for the formula ?

    Parameters:
        thetao (numpy.array) : Temperature array - (t,z,y,x).
        so (numpy.array)     : Salinity array    - (t,z,y,x).
        depth (numpy.array)  : Depth array       - (t,z,y,x).
        tmask (numpy.array)  : Mask array        - (t,z,y,x).

    Returns:
        tuple: A tuple containing:
            array: Potential Density referenced at the surface.
            array: In Situ Density.
    """
    rdeltaS = 32.0
    r1_S0 = 0.875 / 35.16504
    r1_T0 = 1.0 / 40.0
    r1_Z0 = 1.0e-4

    # C : What are thode values ?
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
    rho_insitu = zn * ztm  # in-situ density (masked)
    return rhop, rho_insitu


# C : Why do we need to modify inplace ? Can't we just return vn_new ?
# C : What is the difference between vn and vb ?


def update_v_velocity(
    array, maskarray, e3t_new
):  # e3t_new             = maskarray["e3t_0"].values[0,:,y_slice,x_slice]
    """
    Update the v-component velocity array.Meridional

    Parameters:
        array (xarray.Dataset)     : Restart file.
        maskarray (xarray.Dataset) : Mask dataset.
        e3t_new (numpy.ndarray)    : Updated array of z-axis cell thicknesses.

    Returns:
        None
    """
    x_slice, y_slice = getXYslice(array)
    vn = array.copy().variables[
        "vn"
    ]  # initial v velocity of the restart         - (t,z,y,x)
    e1t = maskarray["e1t"].values[
        0, y_slice, x_slice
    ]  # initial y axis cell's thickness on grid T - (y,x)
    vmask = maskarray["vmask"].values[
        0, :, y_slice, x_slice
    ]  # bathy mask on grid V                      - (z,y,x)
    ff_f = maskarray["ff_f"].values[
        0, y_slice, x_slice
    ]  # corriolis force                           - (y,x)
    tmask = maskarray["tmask"].values[0, :, y_slice, x_slice]

    rhop_new = array.variables["rhop"][0]
    rhop_new = rhop_new.where(
        tmask
    ).values  # updated density                           - (t,z,y,x)

    diff_x = -np.roll(rhop_new, shift=1, axis=2) + rhop_new  #                - (t,z,y,x)
    v_new = (
        9.81 / (rhop_new * ff_f) * np.cumsum(diff_x * e3t_new / e1t, axis=1)
    )  # v without V_0  - (t,z,y,x)
    v_new = np.expand_dims(v_new, axis=0)
    vn_new = add_bottom_velocity(vn.values, v_new, vmask)  # add V_0        - (t,z,y,x)

    array["vn"] = toXarray(vn_new, "vn")
    array["vb"] = toXarray(vn_new, "vb")
    array["ssv_m"] = toXarray(vn_new[:, 0], "vb", dep=False)
    # return v_new,vn_new


# C : What is "restart velocity array" ? Is it the source one ?
# C : v_update and v_restart should have a time dimension of 1 no ?
# C : In restart.py Maud have the same function with "ERROR MAUVAIS", what is the difference ?
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
    time, deptht, y, x = np.shape(v_update)
    for i in range(x):
        for j in range(y):
            v0 = False
            for k in range(deptht)[::-1]:  # From the bottom to the top :
                if (
                    mask[k, j, i] == 1 and v0 == False
                ):  #   If first cell of sea in the water column
                    v0 = v_restart[-1, k, j, i]  #      set V0 to the corresponding value
                elif (
                    mask[k, j, i] == 1 and v0 != False
                ):  #   If cell is not in the bottom
                    v_restart[-1, k, j, i] = (
                        v0 + v_update[-1, k, j, i]
                    )  #      cell is equal to new v cell + v0
    return v_restart


if __name__ == "__main__":
    # C : What is Radical here ? -> Name of the file
    # C : Use argparse
    radical = sys.argv[1]

    MASKdataset = xr.open_dataset("../eORCA1.4.2_mesh_mask_modJD.nc", decode_times=False)
    Restart = xr.open_dataset(radical + ".nc", decode_times=False)
    Restart_NEW = Restart.copy()
    # C : In Maskdataset rename vertical axis with "nav_lev" and t in "time_counter"

    x_slice, y_slice = getXYslice(Restart)

    thetao = Restart.tn  # C: Temperature
    so = Restart.sn  # C : Salinity
    ssh = Restart.sshn  # C : SSH
    un = Restart.un.copy()  # C : Zonal velocity (lat)
    vn = Restart.vn.copy()  # C : Meridional Velocity (lon)
    e3t_ini = Restart.e3t_ini  # C : Initial cell thickness ?
    # C : If e3t_ini and e3t_0 are the same, delete and use e3t_0

    ff_f = MASKdataset.ff_f  # C : Coriolis ?

    e2t = MASKdataset.e2t  # C : Zonal cell size  ?
    e1t = MASKdataset.e1t  # C : Meridional cell size ?

    # C: Vertical cell size ?
    e3w_0 = MASKdataset.e3w_0
    e3u_0 = MASKdataset.e3u_0
    e3v_0 = MASKdataset.e3v_0
    e3t_0 = MASKdataset.e3t_0

    tmask = MASKdataset.tmask
    umask = MASKdataset.umask
    vmask = MASKdataset.vmask

    # Same as get_depth in restart.py
    ssmask = tmask[:, 0]  # bathymetry                                - (t,y,x)
    bathy = e3t_0.sum(dim="z")  # initial condition depth 0                 - (t,z,y,x)
    depth_0 = e3w_0.copy()
    depth_0[:, 0] = 0.5 * e3w_0[:, 0]
    depth_0[:, 1:] = depth_0[:, 0] + e3w_0[:, 1:].cumsum(dim="z")
    deptht = depth_0 * (1 + ssh / (bathy + 1 - ssmask)) * tmask
    rhop_new, rho_insitu_new = density(thetao, so, deptht, tmask)

    # C : No Rho regularisation / checking ?

    # C : Here comment also (update_e3tm in restart.py)
    e3t_new = e3t_ini * (1 + (ssh * ssmask / (bathy + 1 - ssmask)))

    rho_insitu = rho_insitu_new.where(tmask)
    diff_y = rhop_new.roll(y=-1) - rhop_new  #                - (t,z,y,x)
    u_new = 9.81 / (rhop_new * ff_f) * (diff_y * e3t_new / e2t).cumsum(dim="z")
    un_new = add_bottom_velocity(un, u_new, umask[0])

    diff_x = -rhop_new.roll(x=1) + rhop_new  #                - (t,z,y,x)
    v_new = (
        9.81 / (rhop_new * ff_f) * (diff_x * e3t_new / e1t).cumsum(dim="z")
    )  # v without V_0  - (t,z,y,x)
    vn_new = add_bottom_velocity(vn, v_new, vmask)

    Restart_NEW["un"] = un_new
    Restart_NEW["vn"] = vn_new
    Restart_NEW["rhop"] = rhop_new
    Restart_NEW["ub"] = un_new
    Restart_NEW["vb"] = vn_new

    # C : To Add e3tm, ssu_m, ssv_m

    # Not modified yet ( no forecast ) but for later
    Restart_NEW["tn"] = thetao
    Restart_NEW["sn"] = so
    Restart_NEW["sshn"] = ssh

    Restart_NEW.to_netcdf(radical + "_NEW.nc")
