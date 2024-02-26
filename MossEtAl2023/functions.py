"""This file contains source functions to calculate scaling relations presented in Moss et al. 2022.

Reference: https://doi.org/10.34948/N3F595
"""

import numpy as np


def _calc_distrib_params_mag_ad(*, magnitude):
    """
    Calculate mu and sigma for the AD=f(M) relation in Moss et al. 2022 Table 4.4, "Empirical AD Complete Only.".

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

    magnitude = np.atleast_1d(magnitude)

    a, b, sigma = -2.87, 0.416, 0.2
    mu = a + b * magnitude

    return mu, np.full(len(mu), sigma)


def _calc_distrib_params_mag_md(*, magnitude):
    """
    Calculate mu and sigma for the MD=f(M) relation in Moss et al. 2022 Table 4.4, "Empirical MD Complete Only.".

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
    magnitude = np.atleast_1d(magnitude)

    a, b, sigma = -2.5, 0.415, 0.2
    mu = a + b * magnitude

    return mu, np.full(len(mu), sigma)


def _calc_distrib_params_d_ad(*, location):
    """
    Calculate alpha and beta per Figures 4.3 and 4.4 (top eqns) in Moss et al. 2022 based on location.

    Parameters
    ----------
    location : Union[float, np.ndarray]
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[float, float]
        alpha : Shape parameter for Gamma distribution.
        beta : Scale parameter for Gamma distribution.

    """

    location = np.atleast_1d(location)

    folded_location = np.minimum(location, 1 - location)

    a1, a2 = 4.2797, 1.6216
    b1, b2 = -0.5003, 0.5133

    alpha = a1 * folded_location + a2
    beta = b1 * folded_location + b2

    return alpha, beta


def _calc_distrib_params_d_md(*, location):
    """
    Calculate alpha and beta per Figures 4.3 and 4.4 (bottom eqns) in Moss et al. 2022 based on location.

    Parameters
    ----------
    location : Union[float, np.ndarray]
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[float, float]
        alpha : Shape parameter for Beta distribution.
        beta : Shape parameter for Beta distribution.

    """

    location = np.atleast_1d(location)

    folded_location = np.minimum(location, 1 - location)

    a1, a2 = 1.422, 1.856
    b1, b2 = -0.0832, 0.1994

    alpha = a1 * folded_location + a2
    beta = b1 * folded_location + b2

    return alpha, beta
