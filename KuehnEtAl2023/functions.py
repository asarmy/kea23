# Import python libraries
import pandas as pd
import numpy as np
from pathlib import Path

# Model constants
MAG_BREAK, DELTA = 7.0, 0.1

def func_mode(coefficients, magnitude):
    fm = (
        coefficients["c1"]
        + coefficients["c2"] * (magnitude - MAG_BREAK)
        + (coefficients["c3"] - coefficients["c2"])
        * DELTA
        * np.log(1 + np.exp((magnitude - MAG_BREAK) / DELTA))
    )
    return fm


def func_mu(coefficients, magnitude, location):
    fm = func_mode(coefficients, magnitude=magnitude)

    alpha = coefficients["alpha"]
    beta = coefficients["beta"]
    gamma = coefficients["gamma"]
    
    a = fm - gamma * np.power(alpha / (alpha + beta), alpha) * np.power(
        beta / (alpha + beta), beta
    )
    
    mu = a + gamma * (location ** alpha) * ((1 - location) ** beta)
    return mu


def func_sd_mode_bilinear(coefficients, magnitude):
    # Used only for strike-slip
    sd = (
        coefficients["s_m,s1"]
        + coefficients["s_m,s2"] * (magnitude - coefficients["s_m,s3"])
        - coefficients["s_m,s2"]
        * DELTA
        * np.log(1 + np.exp((magnitude - coefficients["s_m,s3"]) / DELTA))
    )
    return sd


def func_sd_mode_sigmoid(coefficients, magnitude):
    # Used only for normal
    sd = coefficients["s_m,n1"] - coefficients["s_m,n2"] / (
        1 + np.exp(-1 * coefficients["s_m,n3"] * (magnitude - MAG_BREAK))
    )
    return sd


def func_sd_u(coefficients, location):
    # Used only for strike-slip and reverse
    # Column name2 for stdv coefficients "s_" varies for style of faulting, fix that here
    s_1 = (
        coefficients["s_s1"] if "s_s1" in coefficients.columns else coefficients["s_r1"]
    )
    s_2 = (
        coefficients["s_s2"] if "s_s2" in coefficients.columns else coefficients["s_r2"]
    )

    alpha = coefficients["alpha"]
    beta = coefficients["beta"]

    sd = s_1 + s_2 * (location - alpha / (alpha + beta)) ** 2
    return sd


def func_ss(coefficients, magnitude, location):
    # Calculate median prediction
    med = func_mu(coefficients, magnitude, location)

    # Calculate standard deviations
    sd_mode = func_sd_mode_bilinear(coefficients, magnitude)
    sd_u = func_sd_u(coefficients, location)
    sd_total = np.sqrt(sd_mode**2 + sd_u**2)

    return med, sd_total


def func_nm(coefficients, magnitude, location):
    # Calculate median prediction
    med = func_mu(coefficients, magnitude, location)

    # Calculate standard deviations
    sd_mode = func_sd_mode_sigmoid(coefficients, magnitude)
    sd_u = coefficients["sigma"]
    sd_total = np.sqrt(sd_mode**2 + sd_u**2)

    return med, sd_total


def func_rv(coefficients, magnitude, location):
    # Calculate median prediction
    med = func_mu(coefficients, magnitude, location)

    # Calculate standard deviations
    sd_mode = coefficients["s_m,r"]
    sd_u = func_sd_u(coefficients, location)
    sd_total = np.sqrt(sd_mode**2 + sd_u**2)

    return med, sd_total
