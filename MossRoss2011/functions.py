"""This file contains source functions to calculate scaling relations presented in Moss and Ross
(2011).

Reference: https://doi.org/10.1785/0120100248
"""

import numpy as np


def _calc_distrib_params_mag_ad(*, magnitude):
    """
    Calculate mu and sigma for the AD=f(M) relation in Moss & Ross (2011) Eqn 8.

    Parameters
    ----------
    magnitude : Union[float, np.ndarray]
        Earthquake moment magnitude.

    Returns
    -------
    Tuple[np.array, np.array]
        mu : Mean prediction in log10 units.
        sigma : Standard deviation in log10 units.

    Notes
    ------
    Mu and sigma are in log10 units
    """

    a, b, sigma = -2.2192, 0.3244, 0.17
    mu = a + b * magnitude

    return mu, np.full(len(mu), sigma)


def _calc_distrib_params_mag_md(*, magnitude):
    """
    Calculate mu and sigma for the MD=f(M) relation in Moss & Ross (2011) Eqn 9.

    Parameters
    ----------
    magnitude : Union[float, np.ndarray]
        Earthquake moment magnitude.

    Returns
    -------
    Tuple[np.array, np.array]
        mu : Mean prediction in log10 units.
        sigma : Standard deviation in log10 units.

    Notes
    ------
    Mu and sigma are in log10 units
    """

    a, b, sigma = -3.1971, 0.5102, 0.31
    mu = a + b * magnitude

    return mu, np.full(len(mu), sigma)
