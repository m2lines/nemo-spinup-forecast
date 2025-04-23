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
            + ((((EOS510 * zs + EOS410) * zs + EOS310) * zs + EOS210) * zs + EOS110)
            * zs
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
    ind_prof = (mask.argmin(dim="nav_lev") - 1) * mask.isel(nav_lev=0)
    #    vu=v_update.groupby(np.isnan(v_update))
    v_fond = v_restart.isel(nav_lev=ind_prof, time_counter=0)
    #    v_restart=v_restart.groupby(np.isnan(v_update)).map(parser, vu=vu, v_fond=v_fond)
    mask_nan_update = np.isnan(v_update)
    v_new = mask_nan_update * v_restart + (1 - mask_nan_update) * (v_fond + v_update)
    return v_restart

    # for i in range(x):
    #    for j in range(y):
    #        v0=False
    #        print("i,j=",i,' ',j)
    #
    #        if mask[0,j,i]==1:
    #            for k in range(deptht)[::-1]:# From the bottom to the top :
    #                if mask[k,j,i]==1 and v0==False:                    #   If first cell of sea in the water column
    #                    v0 = v_restart[-1,k,j,i]
    #                    print(v0,' ',k) #      set V0 to the corresponding value
    #                elif mask[k,j,i]==1 and v0!=False:                  #   If cell is not in the bottom
    #                    v_restart[-1,k,j,i] = v0 + v_update[-1,k,j,i]   #      cell is equal to new v cell + v0


# def parser(gb_da, vu, v_fond):
#    if gb_da.name="1":
#        return vu["1"]+ v_fond
#    else:
#        return gb_da


if __name__ == "__main__":
    radical = sys.argv[1]

    MASKdataset = xr.open_dataset(
        "../eORCA1.4.2_mesh_mask_modJD.nc", decode_times=False
    )
    Restart = xr.open_dataset(radical + ".nc", decode_times=False)
    MASKdataset = MASKdataset.swap_dims(dims_dict={"z": "nav_lev", "t": "time_counter"})
    MASKdataset["time_counter"] = Restart["time_counter"]
    Restart_NEW = Restart.copy()
    # Part to replace with ML-extrapolated data ###
    thetao = Restart.tn  #
    so = Restart.sn  #
    ssh = Restart.sshn  #
    ##############################################
    un = Restart.un.copy()
    vn = Restart.vn.copy()
    e3t_ini = Restart.e3t_ini

    ff_f = MASKdataset.ff_f

    e2t = MASKdataset.e2t
    e1t = MASKdataset.e1t

    e3w_0 = MASKdataset.e3w_0
    e3u_0 = MASKdataset.e3u_0
    e3v_0 = MASKdataset.e3v_0
    e3t_0 = MASKdataset.e3t_0

    tmask = MASKdataset.tmask
    umask = MASKdataset.umask
    vmask = MASKdataset.vmask

    ssmask = tmask[:, 0]  # bathymetry                                - (t,y,x)
    bathy = e3t_0.sum(
        dim="nav_lev"
    )  # initial condition depth 0                 - (t,z,y,x)
    depth_0 = e3w_0.copy()
    depth_0[:, 0] = 0.5 * e3w_0[:, 0]
    # print(depth_0)
    # print(depth_0[:,0])
    # print(e3w_0[:,1:])
    # print(e3w_0[:,1:].cumsum(dim="nav_lev"))
    # set_trace()
    depth_0[:, 1:] = depth_0[:, 0:1].data + e3w_0[:, 1:].cumsum(dim="nav_lev")

    #    set_trace()
    deptht = depth_0 * (1 + ssh / (bathy + 1 - ssmask)) * tmask
    rhop_new, rho_insitu_new = density(thetao, so, deptht, tmask)

    e3t_new = e3t_ini * (1 + ssh * ssmask / (bathy + 1 - ssmask))

    ind_prof_u = (umask.argmin(dim="nav_lev") - 1) * umask.isel(nav_lev=0)
    ind_prof_v = (vmask.argmin(dim="nav_lev") - 1) * vmask.isel(nav_lev=0)

    rho_insitu = rho_insitu_new.where(tmask)
    diff_y = rho_insitu.roll(y=-1) - rho_insitu  #                - (t,z,y,x)
    u_new = 9.81 / ff_f * (diff_y / rho_insitu * e3t_new / e2t).cumsum(dim="nav_lev")
    u_new = u_new - u_new.isel(nav_lev=ind_prof_u)
    un_new = add_bottom_velocity(un, u_new, umask[0])

    diff_x = -rho_insitu.roll(x=1) + rho_insitu  #                - (t,z,y,x)
    v_new = (
        9.81 / ff_f * (diff_x / rho_insitu * e3t_new / e1t).cumsum(dim="nav_lev")
    )  # v without V_0  - (t,z,y,x) C: On intègre vers le fond puis on retire la valeur au fond sur toute la colonne pour avoir v_fond=vo
    v_new = v_new - v_new.isel(nav_lev=ind_prof_v)
    vn_new = add_bottom_velocity(vn, v_new, vmask[0])

    # Modifying the Global Restart file and recording it for analysis
    Restart["un"] = un_new[:, :]
    Restart["vn"] = vn_new[:, :]
    Restart["ub"] = un_new[:, :]
    Restart["vb"] = vn_new[:, :]
    Restart["sn"] = so[:, :]
    Restart["tn"] = thetao[:, :]
    Restart["sb"] = so[:, :]
    Restart["tb"] = thetao[:, :]
    Restart["sshn"] = ssh[:, :]
    Restart["sshb"] = ssh[:, :]

    Restart["rhop"] = rhop_new[:, :]

    Restart["ssv_m"] = vn_new[:, 0]
    Restart["ssu_m"] = un_new[:, 0]
    Restart["sst_m"] = thetao[:, 0]
    Restart["sss_m"] = so[:, 0]
    Restart["ssh_m"] = ssh[:]
    Restart["e3t_m"] = e3t_new[:, 0]

    Restart.to_netcdf(radical + "_NEW.nc")

    # Modifying the Local Restart files for use in Accelerated Simulation

    for i in range(340):
        Restart_NEW = xr.open_dataset(radical + "_%04d.nc" % (i))
        print(i)
        x_slice, y_slice = getXYslice(Restart_NEW)
        Restart_NEW["un"] = un_new[:, :, y_slice, x_slice]
        Restart_NEW["vn"] = vn_new[:, :, y_slice, x_slice]
        Restart_NEW["ub"] = un_new[:, :, y_slice, x_slice]
        Restart_NEW["vb"] = vn_new[:, :, y_slice, x_slice]
        Restart_NEW["sn"] = so[:, :, y_slice, x_slice]
        Restart_NEW["tn"] = thetao[:, :, y_slice, x_slice]
        Restart_NEW["sb"] = so[:, :, y_slice, x_slice]
        Restart_NEW["tb"] = thetao[:, :, y_slice, x_slice]
        Restart_NEW["sshn"] = ssh[:, :, y_slice, x_slice]
        Restart_NEW["sshb"] = ssh[:, :, y_slice, x_slice]

        Restart_NEW["rhop"] = rhop_new[:, :, y_slice, x_slice]

        Restart_NEW["ssv_m"] = vn_new[:, 0, y_slice, x_slice]
        Restart_NEW["ssu_m"] = un_new[:, 0, y_slice, x_slice]
        Restart_NEW["sst_m"] = thetao[:, 0, y_slice, x_slice]
        Restart_NEW["sss_m"] = so[:, 0, y_slice, x_slice]
        Restart_NEW["ssh_m"] = ssh[:, y_slice, x_slice]
        Restart_NEW["e3t_m"] = e3t_new[:, 0, y_slice, x_slice]

        Restart_NEW.to_netcdf(radical + "_%04d_NEW.nc" % (i))
