# Python imports
import sys
from pathlib import Path

import numpy as np
import pytest

# Add path for module
# FIXME: shouldn't need this with a package install (`__init__` should suffice)
PROJ_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJ_DIR))

# Module imports
from KuehnEtAl2023.data import load_data
from KuehnEtAl2023.functions import (
    func_mu,
    func_mode,
    func_sd_mode_bilinear,
    func_sd_u,
    func_ss,
)

# Test setup
RTOL = 2e-2

# Add path for expected outputs
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))


# Load the expected outputs, run tests
@pytest.fixture
def results_data():
    ffp = SCRIPT_DIR / "expected_output" / "strike-slip_mean-model.csv"
    return np.genfromtxt(ffp, delimiter=",", skip_header=1, dtype=None)


def test_strike_slip_mean_model(results_data):
    coeffs = load_data("strike-slip")
    coeffs = coeffs.mean(axis=0).to_frame().transpose()
    for row in results_data:
        # Inputs
        magnitude = row[0]
        location = row[1]

        # Expected
        mode_expect = row[2]
        mu_expect = row[3]
        sd_mode_expect = row[4]
        sd_u_expect = row[5]
        median_expect = mu_expect
        sd_tot_expect = row[6]

        # Computed
        mode_calc = func_mode(coeffs, magnitude)
        mu_calc = func_mu(coeffs, magnitude, location)
        sd_mode_calc = func_sd_mode_bilinear(coeffs, magnitude)
        sd_u_calc = func_sd_u(coeffs, location)
        median_calc, sd_tot_calc = func_ss(coeffs, magnitude, location)

        # Tests
        func_names = ["mode", "mu", "sd_mode", "sd_u", "func_ss (med)", "func_ss (sd)"]
        expected_values = [
            mode_expect,
            mu_expect,
            sd_mode_expect,
            sd_u_expect,
            median_expect,
            sd_tot_expect,
        ]
        computed_values = [
            mode_calc,
            mu_calc,
            sd_mode_calc,
            sd_u_calc,
            median_calc,
            sd_tot_calc,
        ]

        for (func_name, expected, computed) in zip(func_names, expected_values, computed_values):
            np.testing.assert_allclose(
                expected,
                computed,
                rtol=RTOL,
                err_msg=f"Mag {magnitude}, u-star {location}, {func_name}, Expected: {expected}, Computed: {computed}",
            )
