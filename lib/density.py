#!/usr/bin/env python
# =======================================================================
#                        General Documentation

"""Equation of state of Sea-water and related utilities."""

# -----------------------------------------------------------------------
#                       Additional Documentation
#
# Modification History:
# - Oct 2007:  Original by Julien Le Sommer, LEGI/CNRS
# - Jun 2008:  Include pressure related utilities, JLS.
# - Jan 2009:  Include potential_energy_anomaly, JLS
# - Feb 2009:  improve dosctrings, JLS
#
# Notes:
# - Written for Python 2.3, tested with Python 2.4 and Python 2.5
#
# Copyright (c) 2007, 2008,2009 by Julien Le Sommer.
# For licensing, distribution conditions, contact information,
# and additional documentation see the URL
# http://www.legi.hmg.inpg.fr/~lesommer/PyDom/doc/.
# =======================================================================


# ---------------- Module General Import and Declarations ---------------

# - Set module version to package version:

# import package_version
# __version__ = package_version.version
# __author__  = package_version.author
# __date__    = package_version.date
# del package_version

import numpy as N
import numpy as npy
# from PyDom.__param__ import *


# ---------------- Core Functions ---------------------------------------
rau0 = 10e3  # We added rau0 constant


def insitu(theta0, S, Z):
    """In-situ density (kg/m**3)

    Compute in-situ density from `insitu_anom` with
    Jackett and McDougall (1995) equation of state.

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    S : numpy.array
        salinity
    Z : numpy.array
        pressure (dB) or depth (m)

    """
    rho = rau0 * insitu_anom(theta0, S, Z) + rau0
    #
    return rho


def sigma_n_old(theta0, S, n):
    """Potential density referenced to pressure n*1000dB (kg/m**3)

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    S : numpy.array
        salinity
    n : int
        reference pressure / 1000 dB

    """
    if n == 0:
        return sig0(theta0, S)
    else:
        theta_n = theta0_2_theta_n(theta0, S, n)
        dep = N.zeros(theta0.shape) + n * 1000
        sig_n = insitu(theta_n, S, dep) - 1000
        #
        return sig_n


def sigma_n(theta0, S, n):
    """Potential density referenced to pressure n*1000dB (kg/m**3)

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    S : numpy.array
        salinity
    n : int
        reference pressure / 1000 dB

    Method
    ------
    A.-M. Treguier

    """
    #
    dpr4 = 4.8314e-4
    dpd = -2.042967e-2
    dprau0 = 1000.0
    pref = n * 1000.0
    dlref = pref
    # sigmai = 0.*theta0
    dlrs = N.sqrt(N.abs(S))
    dlt = theta0
    dls = S
    # Compute the volumic mass of pure water at atmospheric pressure.
    dlr1 = (
        (((6.536332e-9 * dlt - 1.120083e-6) * dlt + 1.001685e-4) * dlt - 9.095290e-3)
        * dlt
        + 6.793952e-2
    ) * dlt + 999.842594e0
    # Compute the seawater volumic mass at atmospheric pressure.
    dlr2 = (
        ((5.3875e-9 * dlt - 8.2467e-7) * dlt + 7.6438e-5) * dlt - 4.0899e-3
    ) * dlt + 0.824493e0

    dlr3 = (-1.6546e-6 * dlt + 1.0227e-4) * dlt - 5.72466e-3
    # Compute the potential volumic mass (referenced to the surface).
    dlrhop = (dpr4 * dls + dlr3 * dlrs + dlr2) * dls + dlr1

    # Compute the compression terms.
    dle = (-3.508914e-8 * dlt - 1.248266e-8) * dlt - 2.595994e-6

    dlbw = (1.296821e-6 * dlt - 5.782165e-9) * dlt + 1.045941e-4

    dlb = dlbw + dle * dls

    dlc = (-7.267926e-5 * dlt + 2.598241e-3) * dlt + 0.1571896e0

    dlaw = ((5.939910e-6 * dlt + 2.512549e-3) * dlt - 0.1028859e0) * dlt - 4.721788e0

    dla = (dpd * dlrs + dlc) * dls + dlaw

    dlb1 = (-0.1909078e0 * dlt + 7.390729e0) * dlt - 55.87545e0

    dla1 = ((2.326469e-3 * dlt + 1.553190e0) * dlt - 65.00517e0) * dlt + 1044.077e0

    dlkw = (
        ((-1.361629e-4 * dlt - 1.852732e-2) * dlt - 30.41638e0) * dlt + 2098.925e0
    ) * dlt + 190925.6e0

    dlk0 = (dlb1 * dlrs + dla1) * dls + dlkw

    # Compute the potential density anomaly.
    sigmai = dlrhop / (1.0e0 - dlref / (dlk0 - dlref * (dla - dlref * dlb))) - dprau0
    return sigmai


def sig0(theta0, S):
    """Surface referenced potential density (kg/m**3)

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    S : numpy.array
        salinity

    Method
    ------
    Use `insitu`

    """
    dep = N.zeros(theta0.shape)
    sig = insitu(theta0, S, dep) - 1000
    return sig


def insitu_anom(theta0, S, Z):
    """In-situ density anomaly.

    In situ density is computed directly as a function of
    potential temperature relative to the surface.

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    S : numpy.array
        salinity
    Z : numpy.array
        pressure (dB) or depth (m)

    Notes
    -----
    We use  Jackett and McDougall (1995)'s [1]_ equation of state.
    the in situ density is computed directly as a function of
    potential temperature relative to the surface (the opa t
    variable), salt and pressure (assuming no pressure variation
    along geopotential surfaces, i.e. the pressure p in decibars
    is approximated by the depth in meters.

    prd(t,s,p) = ( rho(t,s,p) - rau0 ) / rau0

    with:

    - pressure                      p        decibars

    - potential temperature         t        deg celsius

    - salinity                      s        psu

    - reference volumic mass        rau0     kg/m**3

    - in situ volumic mass          rho      kg/m**3

    - in situ density anomaly      prd      no units

    Examples
    --------
    >>> insitu(40,40,10000)
    1060.93298

    References
    ----------
    .. [1]  Jackett, D. R., and T. J. McDougall, Minimal adjustment of
            hydrographic profiles to achieve static stability, J. Atmos.
            Ocean. Technol.,  12(4), 381-389, 1995.

    """

    zsr = N.sqrt(N.abs(S))
    zt = theta0
    zs = S
    zh = Z

    # compute volumic mass pure water at atm pressure
    zr1 = (
        (((6.536332e-9 * zt - 1.120083e-6) * zt + 1.001685e-4) * zt - 9.095290e-3) * zt
        + 6.793952e-2
    ) * zt + 999.842594
    # seawater volumic mass atm pressure
    zr2 = (
        ((5.3875e-9 * zt - 8.2467e-7) * zt + 7.6438e-5) * zt - 4.0899e-3
    ) * zt + 0.824493
    zr3 = (-1.6546e-6 * zt + 1.0227e-4) * zt - 5.72466e-3
    zr4 = 4.8314e-4

    # potential volumic mass (reference to the surface)
    zrhop = (zr4 * zs + zr3 * zsr + zr2) * zs + zr1

    # add the compression terms
    ze = (-3.508914e-8 * zt - 1.248266e-8) * zt - 2.595994e-6
    zbw = (1.296821e-6 * zt - 5.782165e-9) * zt + 1.045941e-4
    zb = zbw + ze * zs

    zd = -2.042967e-2
    zc = (-7.267926e-5 * zt + 2.598241e-3) * zt + 0.1571896
    zaw = ((5.939910e-6 * zt + 2.512549e-3) * zt - 0.1028859) * zt - 4.721788
    za = (zd * zsr + zc) * zs + zaw

    zb1 = (-0.1909078 * zt + 7.390729) * zt - 55.87545
    za1 = ((2.326469e-3 * zt + 1.553190) * zt - 65.00517) * zt + 1044.077
    zkw = (
        ((-1.361629e-4 * zt - 1.852732e-2) * zt - 30.41638) * zt + 2098.925
    ) * zt + 190925.6
    zk0 = (zb1 * zsr + za1) * zs + zkw

    prd = (zrhop / (1.0 - zh / (zk0 - zh * (za - zh * zb))) - rau0) / rau0

    return prd


def delta(t, s, p):
    """Specific volume anomaly.

    Specific volume anomaly with respect to a standart ocean
    of salinity S = Sstd and T = Tstd.

    Parameters
    ----------
    t : numpy.array
        potential temperature
    s : numpy.array
        salinity
    p : numpy.array
        pressure (dB) or depth (m)

    References
    ----------
    see e.g. page 2188 of Watts et al. JPO, 2001.

    """
    delta = 1.0 / insitu(t, s, p) - 1.0 / insitu(Tstd, Sstd, p)
    return delta


def spice(t, s):
    """Spiciness.

    A state variable for characterizing water masses and their
    diffusive stability

    Parameters
    ----------
    t : numpy.array
        potential temperature
    s : numpy.array
        salinity

    Notes
    -----
    following Flament (2002) [1]_. We could also have used [2]_.

    **caution** This state variable is only valid close to the surface.

    See also
    --------
    http://www.satlab.hawaii.edu/spice/spice.html

    References
    ----------
    .. [1] Flament (2002) A state variable for characterizing water
           masses and their diffusive stability: spiciness. Progress
           in Oceanography Volume 54, 2002, Pages 493-501.
    .. [2] Jackett and McDougall, Deep Sea Research, 32A, 1195-1208, 1985.

    Examples
    --------
    >>> spice(15,33)
    0.54458641375

    """
    B = numpy.zeros((7, 6))
    B[1, 1] = 0
    B[1, 2] = 7.7442e-001
    B[1, 3] = -5.85e-003
    B[1, 4] = -9.84e-004
    B[1, 5] = -2.06e-004

    B[2, 1] = 5.1655e-002
    B[2, 2] = 2.034e-003
    B[2, 3] = -2.742e-004
    B[2, 4] = -8.5e-006
    B[2, 5] = 1.36e-005

    B[3, 1] = 6.64783e-003
    B[3, 2] = -2.4681e-004
    B[3, 3] = -1.428e-005
    B[3, 4] = 3.337e-005
    B[3, 5] = 7.894e-006

    B[4, 1] = -5.4023e-005
    B[4, 2] = 7.326e-006
    B[4, 3] = 7.0036e-006
    B[4, 4] = -3.0412e-006
    B[4, 5] = -1.0853e-006

    B[5, 1] = 3.949e-007
    B[5, 2] = -3.029e-008
    B[5, 3] = -3.8209e-007
    B[5, 4] = 1.0012e-007
    B[5, 5] = 4.7133e-008

    B[6, 1] = -6.36e-010
    B[6, 2] = -1.309e-009
    B[6, 3] = 6.048e-009
    B[6, 4] = -1.1409e-009
    B[6, 5] = -6.676e-010
    #
    t = numpy.array(t)
    s = numpy.array(s)
    #
    coefs = B[1:7, 1:6]
    sp = numpy.zeros(t.shape)
    ss = s - 35.0
    bigT = numpy.ones(t.shape)
    for i in range(6):
        bigS = numpy.ones(t.shape)
        for j in range(5):
            sp += coefs[i, j] * bigT * bigS
            bigS *= ss
        bigT *= t
    return sp


def lapse_rate(t, s, p):
    """Adiabatic lapse rate (deg C/dBar).

    Adiabatic lapse rate (deg C/dBar) from salinity (psu),
    temperature (deg C) and pressure (dbar)

    Parameters
    ----------
    t : numpy.array
        temperature (deg C)
    s : numpy.array
        salinity (psu)
    p : numpy.array
        pressure (dbar)


    Notes
    -----
    This calculator is based on an algorithm for the speed of sound
    published by Chen and Millero (1977) [1]_.

    The underlying equations are valid for temperatures from -2 to 35 deg C,
    pressures from 0 to 10,000 dbar,
    and practical salinity from 2 to 42.

    See also
    --------
    http://fermi.jhuapl.edu/denscalc/spdcalc.html

    References
    ----------
    .. [1] Chen and Millero, Speed of sound in seawater at high pressures,
           J. Acoust. Soc. Am., Vol. 62, No. 5, 1129-1135, Nov 1977.

    """
    #
    ds = s - 35.0
    atg = ((-2.1687e-16 * t + 1.8676e-14) * t - 4.6206e-13) * p * p
    atg += (2.7759e-12 * t - 1.1351e-10) * ds * p
    atg += (((-5.4481e-14 * t + 8.7330e-12) * t - 6.7795e-10) * t + 1.8741e-8) * p
    atg += (-4.2393e-8 * t + 1.8932e-6) * ds
    atg += ((6.6228e-10 * t - 6.8360e-8) * t + 8.5258e-6) * t + 3.5803e-5
    return atg


def theta_n(theta0, s, n):
    """Potential temperature.

    Potential temperature with reference pressure n x 1000dB

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    s : numpy.array
        salinity
    n : int
        reference pressure / 1000 dB

    """
    Pr = n * 1000
    theta = theta0_2_theta(theta0, s, Pr)
    return theta


def theta0_2_theta_n(theta0, s, Pr):
    """Potential temperature.

    Potential temperature with reference pressure Pr.

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    s : numpy.array
        salinity
    Pr : float
        reference pressure (dB)

    Notes
    -----
    **caution** This function uses a very rough estimate of potential temperature

    """
    P0 = 0
    dP = Pr - P0
    dT = dP * lapse_rate(theta0, s, (Pr + P0) / 2.0)
    theta_n = theta0 + dT
    ## # # print '********************************************************************'
    ## # print ' theta0_2_theta_n is now deprecated. Please do not use it.'
    ## # print '********************************************************************'
    return theta_n


def beta(theta0, s, p):
    """Haline contraction coeficient beta.

    Haline contraction coeficient (unit : 1/psu) from McDougall (1987) [1]_

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    s : numpy.array
        salinity
    p : numpy.array
        pressure (dB) or depth (m)

    Notes
    -----
    adapted from McDougall (1987) [1]_.

    References
    ----------
    .. [1] McDougall, Neutral surfaces - Journal of Physical Oceanography,
           vol.17,pp.1950-1964, 1987.

    """
    th = theta0
    beta = 0.785567e-3 - 0.301985e-5 * th
    beta += 0.555579e-7 * th**2 - 0.415613e-9 * th**3
    beta += (s - 35.0) * (
        -0.356603e-6 + 0.788212e-8 * th + 0.408195e-10 * p - 0.602281e-15 * p**2
    )
    beta += (s - 35.0) ** 2 * 0.515032e-8 + p * (
        -0.121555e-7 + 0.192867e-9 * th - 0.213127e-11 * th**2
    )
    beta += p**2 * (0.176621e-12 - 0.175379e-14 * th) + p**3 * (0.121551e-17)
    return beta


def alpha_over_beta(theta0, s, p):
    """Thermal expansion to haline contraction ratio.

    Ratio of the thermal expansion coefficient to the saline
    contraction coeficient (unit: psu/deg).

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    s : numpy.array
        salinity
    p : numpy.array
        pressure (dB) or depth (m)

    Notes
    -----
    from McDougall (1987) [1]_.

    References
    ----------
    .. [1] McDougall, Neutral surfaces - Journal of Physical Oceanography,
           vol.17,pp.1950-1964, 1987.

    """
    th = theta0
    ab = 0.665157e-1 + 0.170907e-1 * th
    ab += -0.203814e-3 * th**2 + 0.298357e-5 * th**3 - 0.255019e-7 * th**4
    ab += (s - 35.0) * (
        0.378110e-2 - 0.846960e-4 * th - 0.164759e-6 * p - 0.251520e-11 * p**2
    )
    ab += (s - 35.0) ** 2 * (-0.678662e-5) + p * (
        0.380374e-4 - 0.933746e-6 * th + 0.791325e-8 * th**2
    )
    ab += 0.512857e-12 * p**2 * th**2 - 0.302285e-13 * p**3
    return ab


def alpha(theta0, s, p):
    """Thermal expansion coefficient.

    Thermal expansion coeficient (unit : 1/deg) from McDougall (1987) [1]_

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    s : numpy.array
        salinity
    p : numpy.array
        pressure (dB) or depth (m)

    References
    ----------
    .. [1] McDougall, Neutral surfaces - Journal of Physical Oceanography,
           vol.17,pp.1950-1964, 1987.

    """
    alpha = beta(theta0, s, p) * alpha_over_beta(theta0, s, p)
    return alpha


# ---------------- Derivatives of Core Funtions ---------------------------


def rhoalpha(theta0, s, p):
    """rho x alpha.

    Return the product rho x alpha.

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    s : numpy.array
        salinity
    p : numpy.array
        pressure (dB) or depth (m)

    """
    zrho = insitu(theta0, s, p)
    za = alpha(theta0, s, p)
    return zrho * za


def rhobeta(theta0, s, p):
    """rho x beta.

    Return the product rho x beta.

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    s : numpy.array
        salinity
    p : numpy.array
        pressure (dB) or depth (m)

    """
    zrho = insitu(theta0, s, p)
    zb = beta(theta0, s, p)
    return zrho * zb


def beta_p(theta0, s, p):
    """Partial derivative of beta with respect to pressure.

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    s : numpy.array
        salinity
    p : numpy.array
        pressure (dB) or depth (m)

    Notes
    -----
    analytic derivation of McDougall (1987)'s formula.

    """
    th = theta0
    zbeta_p = (s - 35.0) * (0.408195e-10 - 0.602281e-15 * p * 2)
    zbeta_p += +(-0.121555e-7 + 0.192867e-9 * th - 0.213127e-11 * th**2)
    zbeta_p += 2 * p * (0.176621e-12 - 0.175379e-14 * th) + 3 * p**2 * (0.121551e-17)
    return zbeta_p


def rho_p(theta0, s, p):
    """Partial derivative of in situ density with respect to pressure.

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    s : numpy.array
        salinity
    p : numpy.array
        pressure (dB) or depth (m)

    Notes
    -----
    analytic derivation of Jackett and McDougall (1995)'s formula.

    """
    zsr = N.sqrt(N.abs(s))
    zt = theta0
    zs = s
    zh = p

    # compute volumic mass pure water at atm pressure
    zr1 = (
        (((6.536332e-9 * zt - 1.120083e-6) * zt + 1.001685e-4) * zt - 9.095290e-3) * zt
        + 6.793952e-2
    ) * zt + 999.842594
    # seawater volumic mass atm pressure
    zr2 = (
        ((5.3875e-9 * zt - 8.2467e-7) * zt + 7.6438e-5) * zt - 4.0899e-3
    ) * zt + 0.824493
    zr3 = (-1.6546e-6 * zt + 1.0227e-4) * zt - 5.72466e-3
    zr4 = 4.8314e-4

    # potential volumic mass (reference to the surface)
    zrhop = (zr4 * zs + zr3 * zsr + zr2) * zs + zr1

    # add the compression terms
    ze = (-3.508914e-8 * zt - 1.248266e-8) * zt - 2.595994e-6
    zbw = (1.296821e-6 * zt - 5.782165e-9) * zt + 1.045941e-4
    zb = zbw + ze * zs

    zd = -2.042967e-2
    zc = (-7.267926e-5 * zt + 2.598241e-3) * zt + 0.1571896
    zaw = ((5.939910e-6 * zt + 2.512549e-3) * zt - 0.1028859) * zt - 4.721788
    za = (zd * zsr + zc) * zs + zaw

    zb1 = (-0.1909078 * zt + 7.390729) * zt - 55.87545
    za1 = ((2.326469e-3 * zt + 1.553190) * zt - 65.00517) * zt + 1044.077
    zkw = (
        ((-1.361629e-4 * zt - 1.852732e-2) * zt - 30.41638) * zt + 2098.925
    ) * zt + 190925.6
    zk0 = (zb1 * zsr + za1) * zs + zkw

    # zrho=  zrhop / (  1.0 - zh / ( zk0 - zh * ( za - zh * zb ) )  )
    zk = zk0 - zh * (za - zh * zb)
    zu = zh / zk
    zk_p = -za + 2 * zh * zb
    zu_p = (1 - (zh / zk) * zk_p) / zk
    zrho_p = (zrhop / (1.0 - zu) ** 2) * zu_p
    return zrho_p


def Tb(theta0, s, p):
    """Thermobaric parameter Tb.

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    s : numpy.array
        salinity
    p : numpy.array
        pressure (dB) or depth (m)

    Notes
    -----
    Tb = beta*(alpha/beta)_p = alpha_p - (alpha/beta)*beta_p

    Tb ~ 2.7E-8 K-1.dbar-1 = 2.7E-12 K-1.Pa-1

    """
    # # # # print 'eos.Tb has not been checked yet...'
    return beta(theta0, s, p) * alpha_over_beta_p(theta0, s, p)
    # return alpha_p(theta0,s,p)-alpha_over_beta(theta0,s,p)*beta_p(theta0,s,p)


def alpha_over_beta_p(theta0, s, p):
    """Partial derivative of alpha/beta with respect to pressure.

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    s : numpy.array
        salinity
    p : numpy.array
        pressure (dB) or depth (m)

    Notes
    -----
    analytic derivation of McDougall (1987)'s formula.

    """
    th = theta0
    zab_p = (s - 35.0) * (-0.164759e-6 - 0.251520e-11 * 2 * p)
    zab_p += 0.380374e-4 - 0.933746e-6 * th + 0.791325e-8 * th**2
    zab_p += 0.512857e-12 * 2 * p * th**2 - 0.302285e-13 * 3 * p**2
    return zab_p


def alpha_p(theta0, s, p):
    """Partial derivative of alpha with respect to pressure.

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    s : numpy.array
        salinity
    p : numpy.array
        pressure (dB) or depth (m)

    Notes
    -----
    computes alpha_p = (alpha/beta)_p * beta + beta_p * (alpha/beta)

    """
    zb = beta(theta0, s, p)
    zb_p = beta_p(theta0, s, p)
    zab = alpha_over_beta(theta0, s, p)
    zab_p = alpha_over_beta_p(theta0, s, p)
    return zab_p * zb + zab * zb_p


def rhobeta_p(theta0, s, p):
    """Partial derivative of rho*beta with respect to pressure.

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    s : numpy.array
        salinity
    p : numpy.array
        pressure (dB) or depth (m)

    """
    zrho = insitu(theta0, s, p)
    zbeta = beta(theta0, s, p)
    zbeta_p = beta_p(theta0, s, p)
    zrho_p = rho_p(theta0, s, p)
    return zrho * zbeta_p + zbeta * zrho_p


def rhoalpha_p(theta0, s, p):
    """Partial derivative of rho*alpha with respect to pressure.

    Parameters
    ----------
    theta0 : numpy.array
        potential temperature
    s : numpy.array
        salinity
    p : numpy.array
        pressure (dB) or depth (m)

    """
    zrhobeta = rhobeta(theta0, s, p)
    zab = alpha_over_beta(theta0, s, p)
    zrhobeta_p = rhobeta_p(theta0, s, p)
    zab_p = alpha_over_beta_p(theta0, s, p)
    return zrhobeta * zab_p + zab * zrhobeta_p


# ---------------- Functions involving Derivatives or integrals --------------
# -                ie : "dom" dependent utilities.
def bn2(dom, theta0, S):
    """Brunt-Vaissala frequency.

    Brunt-Vaissala frequency on the w-grid.

    Parameters
    ----------
    dom : OPA_C_Grid instance
        domain
    theta0 : numpy.array
        potential temperature
    s : numpy.array
        salinity

    Notes
    -----
    The stability frequency is computed according to :

    N2 = grav*(alpha d_z theta - beta d_z S).

    """
    ralpha = alpha(theta0, S, dom.depthT_3D)
    rbeta = beta(theta0, S, dom.depthT_3D)
    ra = dom.lamVz(ralpha, dom.dro_z(theta0))
    rb = dom.lamVz(rbeta, dom.dro_z(S))
    gz = grav * (ra - rb)
    mgz = dom.set_mask(gz, dom.wmask)
    return mgz


def isoslope(dom, theta0, S):
    """Slope of isopycnals."""
    ralpha = alpha(theta0, S, dom.depthT_3D)
    rbeta = beta(theta0, S, dom.depthT_3D)
    tx, ty, tz = dom.grad3D(theta0)
    sx, sy, sz = dom.grad3D(S)
    atx, aty, atz = dom.lamV(ralpha, (tx, ty, tz))
    bsx, bsy, bsz = dom.lamV(rbeta, (sx, sy, sz))
    ghsig = (atx - bsx, aty - bsy, 0.0 * atz)
    gzsig = atz - bsz
    ngh = N.sqrt(dom.normV(ghsig))
    denom = dom.gridW_2_gridT(gzsig)
    s = dom.set_mask(ngh / denom, dom.tmask)
    return s


def Gz(dom, theta0, S):
    """Vertical component of the neutral vector.

    Vertical component of the neutral vector on the w-grid.

    Parameters
    ----------
    dom : OPA_C_Grid instance
        domain
    theta0 : numpy.array
        potential temperature
    s : numpy.array
        salinity

    """
    ralpha = rhoalpha(theta0, S, dom.depthT_3D)
    rbeta = rhobeta(theta0, S, dom.depthT_3D)
    ra = dom.lamVz(ralpha, dom.dro_z(theta0))
    rb = dom.lamVz(rbeta, dom.dro_z(S))
    gz = ra - rb
    mgz = dom.set_mask(gz, dom.wmask)
    return mgz


def hydrostatic(dom, T=None, S=None, rho=None, p0=1e5):
    """Hydrostatic pressure (Pa).

    Parameters
    ----------
    dom : OPA_C_Grid instance
        domain
    T : numpy.array
        potential temperature
    S : numpy.array
        salinity
    rho : numpy.array, optional
        in situ density
    p0 : numpy.array
        surface pressure

    Notes
    -----
    according to \partial_z P = - \rho g

    **caution** P unit is Pa (1 dbar = 1e4 Pascal)

    """
    #
    jpk = dom.jpk
    #
    ph = N.zeros(dom.tmask.shape)
    if rho is not None:
        arho = (rho - rau0) / rau0
    else:
        arho = insitu_anom(T, S, dom.depthT_3D)
    #
    ph[..., 0, :, :] = grav * dom.depthT_3D[0, :, :] * arho[..., 0, :, :]
    ph[..., 1:jpk, :, :] = grav * dom.m_k(arho) * dom.e3w_3D[1:jpk, :, :]
    ph = ph.cumsum(axis=-3)
    ph *= rau0
    ph += rau0 * grav * dom.depthT_3D
    ph += p0
    ph = dom.set_mask(ph, dom.tmask)
    return ph


def surface(dom, ssh, rho=rau0):
    """Surface pressure associated with sea surface height.

    Parameters
    ----------
    dom : OPA_C_Grid instance
        domain
    ssh : numpy.array
        sea surface height
    rho : numpy.array, optional
        surface in situ density
        surface pressure


    """
    ps = rho * grav * ssh
    ps = dom.set_mask(ps, dom.tmask[0, :])
    return ps


def pressure(dom, T=None, S=None, ssh=None, rho=None):
    """Pressure.

    Returns the surface + hydrostatic pressure

    Parameters
    ----------
    dom : OPA_C_Grid instance
        domain
    T : numpy.array
        potential temperature
    S : numpy.array
        salinity
    ssh : numpy.array
        sea surface height
    rho : numpy.array, optional
        in situ density


    """
    if ssh is None:
        ssh = 0.0 * T
        rho_s = rau0
    else:
        rho_s = insitu(T[0], S[0], 0.0)
    if rho is not None:
        p = hydrostatic(dom, rho=rho)
    else:
        p = hydrostatic(dom, T, S)
    p += surface(dom, ssh, rho=rho_s)
    p = dom.set_mask(p, dom.tmask)
    return p


def chi(dom, T, S):
    """Return chi = g/rho0 \int (z * rho) dz"""
    z3d = dom.depthT_3D
    rho = insitu(T, S, z3d)
    integral = dom.integrate_dz(rho * z3d, dom.tmask)
    return integral * grav / rau0


def jebar_old(dom, T, S):
    """Return the JEBAR term. (weak signal to noise ratio...)
    see e.g. Mertz and Wright JPO 1992
    """
    dom.get_bottom_depth()
    invH = 1.0 / dom.bottom_depth
    mchi = chi(dom, T, S)
    return dom.jacobian(mchi, invH)


def jebar(dom, T, S):
    """Return the JEBAR term. (not checked yet...)
    see e.g. Mertz and Wright JPO 1992
    """
    dom.get_bottom_depth()
    z3d = dom.depthT_3D
    rho = insitu(T, S, z3d)
    h3d = dom.stretch(dom.bottom_depth)
    locjac = dom.jacobian(rho, h3d)
    mymask = npy.abs(locjac) < 1.0  # not very satisfactory but well...
    integral = dom.integrate_dz(locjac * z3d, where=mymask)
    return (integral * grav / rau0) / (-(dom.bottom_depth**2))


def geopotential_height_anomaly(dom, T, S, z, zref=0.0):
    """Geopotential heigh anomaly.

    Return geopotential heigh anomaly at pressure z (in dbar) with respect
    to pressure zref (in dbar).

    Parameters
    ----------
    dom : OPA_C_Grid instance
        domain
    T : numpy.array
        potential temperature
    S : numpy.array
        salinity
    z : float
    zref : float

    Notes
    -----
    see e.g. Watt et al. (2001) [1]_

    References
    ----------
    .. [1] Watt et al. JPO 2001.

    """
    delt = delta(T, S, dom.depthT_3D)
    dep = dom.depthT_3D
    if z > zref:
        intdom = (dep > zref) * (dep < z)
        int = -dom.integrate_dz(delt, intdom)  # not sure about  the sign here...
    else:
        intdom = (dep < zref) * (dep > z)
        int = dom.integrate_dz(delt, intdom)
    return int * dbar2pascal / grav  # to get dynamic meters...


def potential_energy_anomaly(dom, T, S, z=0, zref=2500.0):
    """Potential energy anomaly (PEA).

    Return PEA integrated between  pressure zref (in dbar) and pressure zref (in dbar).

    Parameters
    ----------
    dom : OPA_C_Grid instance
        domain
    T : numpy.array
        potential temperature
    S : numpy.array
        salinity
    z : float
    zref : float

    Notes
    -----
    see e.g. Rintoul et al. (2002) [1]_

    References
    ----------
    .. [1] Rintoul et al. JGR 2002.

    """
    delt = delta(T, S, dom.depthT_3D)
    pdelt = dom.depthT_3D * delt
    dep = dom.depthT_3D
    if z > zref:
        intdom = (dep > zref) * (dep < z)
        int = -dom.integrate_dz(pdelt, intdom)
    else:
        intdom = (dep < zref) * (dep > z)
        int = dom.integrate_dz(pdelt, intdom)
    return int * dbar2pascal / grav


def geopotential_height_anomaly3D(dom, T, S):
    """Geopotential heigh anomaly.

    Return geopotential heigh anomaly at pressure z (in dbar) with respect
    to pressure zref (in dbar).

    Parameters
    ----------
    dom : OPA_C_Grid instance
        domain
    T : numpy.array
        potential temperature
    S : numpy.array
        salinity
    zref : float

    Notes
    -----
    see e.g. Watt et al. (2001) [1]_

    References
    ----------
    .. [1] Watt et al. JPO 2001.

    """
    name = miscutils.whoami()
    # print name + ' is not implemented yet...'
    return


def montgomery(dom, t, s, ssh=None, href=0.0):
    """Montgomery potential.

    Parameters
    ----------
    dom : OPA_C_Grid instance
        domain
    t : numpy.array
        potential temperature
    s : numpy.array
        salinity
    ssh : numpy.array
        sea surface height
    href : float, optional
        reference depth

    Notes
    -----
    depends on `bernoulli`

    """

    b = pressure(dom, t, s, ssh=ssh) / rau0
    b += grav * (href - dom.depthT_3D)  # should be ok now...
    return dom.set_mask(b, dom.tmask)


def bernoulli(dom, t, s, uv, ssh=None, href=0.0, method="PM07"):
    """Bernoulli potential.

    Parameters
    ----------
    dom : OPA_C_Grid instance
        domain
    t : numpy.array
        potential temperature
    s : numpy.array
        salinity
    uv: tuple of numpy.arrays
        horizontal velocity
    ssh : numpy.array
        sea surface height
    href : float, optional
        reference depth
    method: {'PM07','MJM01','test'}

    Notes
    -----
    Different methods are available. Note that they do not
    provide the same results.

    """
    exec("_bernoulli = _bernoulli_" + method)
    return _bernoulli(dom, t, s, uv, ssh=ssh, href=href)


def _bernoulli_PM07(dom, t, s, uv, ssh=None, href=0.0):
    """Bernoulli potential.

    Parameters
    ----------
    dom : OPA_C_Grid instance
        domain
    t : numpy.array
        potential temperature
    s : numpy.array
        salinity
    uv: tuple of numpy.arrays
        horizontal velocity
    ssh : numpy.array
        sea surface height
    href : float, optional
        reference depth

    Notes
    -----
    following Polton and Marshall (2007) [1]_

    References
    ----------
    .. [1] Polton and Marshall Ocean Science 2007

    """

    uvw = (uv[0], uv[1], 0.0 * uv[0])
    b = pressure(dom, t, s, ssh=ssh) / rau0
    b += dom.dot(uvw, uvw) / 2.0
    b += grav * (href - dom.depthT_3D)
    return dom.set_mask(b, dom.tmask)


def _bernoulli_MJN01(dom, t, s, uv, ssh=None, href=0.0):
    """Bernoulli potential.

    Parameters
    ----------
    dom : OPA_C_Grid instance
        domain
    t : numpy.array
        potential temperature
    s : numpy.array
        salinity
    uv: tuple of numpy.arrays
        horizontal velocity
    ssh : numpy.array
        sea surface height
    href : float, optional
        reference depth

    Notes
    -----
    following Marshall et al. (2001) [1]_

    References
    ----------
    .. [1] Marshall Jamous and Niilson, J. of Phys. Ocean. 2001

    """
    uvw = (uv[0], uv[1], 0.0 * uv[0])
    rho = insitu(t, s, dom.depthT_3D)
    b = pressure(dom, T=t, S=s, rho=rho, ssh=ssh) / rau0
    b += grav * (href - dom.depthT_3D) * rho / rau0
    return dom.set_mask(b, dom.tmask)


def _bernoulli_test(dom, t, s, uv, ssh=None, href=0.0):
    """You should not use this function."""
    uvw = (uv[0], uv[1], 0.0 * uv[0])
    rho = insitu(t, s, dom.depthT_3D)
    b = pressure(dom, rho=rho, ssh=ssh) / rho
    b += grav * (href - dom.depthT_3D)
    return dom.set_mask(b, dom.tmask)


def helicity(dom, T, S):
    """Neutral helicity.

    Parameters
    ----------
    dom : OPA_C_Grid instance
        domain
    T : numpy.array
        potential temperature
    S : numpy.array
        salinity

    Notes
    -----
    H = A . curl A where A = beta * grad S - alpha * grad T

    recompute with the spatial gradients.

    """
    grad = dom.grad3D
    gS = grad(S)
    gT = grad(T)
    # get A
    alpha = alpha(T, S, dom.depthT_3D)
    beta = beta(T, S, dom.depthT_3D)
    ra = dom.lamV(alpha, gT)
    rb = dom.lamV(beta, gS)
    A = (rb[0] - ra[0], rb[1] - ra[1], rb[2] - ra[2])
    # get curl A
    alpha_p = alpha_p(T, S, dom.depthT_3D)
    beta_p = beta_p(T, S, dom.depthT_3D)
    P = pressure(dom, T, S)
    gP = grad(P / dbar2pascal)
    gPxgT = dom.cross(gP, gT, True)
    gPxgS = dom.cross(gP, gS, True)
    ta = dom.lamV(alpha_p, gPxgT, True)
    tb = dom.lamV(beta_p, gPxgS, True)
    cA = (tb[0] - ta[0], tb[1] - ta[1], tb[2] - ta[2])
    # get H
    H = dom.dot(cA, A, stag_grd=True)
    mH = N.core.ma.masked_where(
        (H == 0.0) + (H > 1.0) + (H < -1.0), H
    )  # happy hard-coding...
    #
    return mH


# ===== end file =====
