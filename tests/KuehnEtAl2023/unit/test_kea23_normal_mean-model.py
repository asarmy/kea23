# Python imports
import sys
from pathlib import Path

import numpy as np
import pytest

# Add path for module
# FIXME: shouldn't need this with a package install (`__init__` should suffice?!)
MODEL_DIR = Path(__file__).resolve().parents[3] / "KuehnEtAl2023"
sys.path.append(str(MODEL_DIR))

# Module imports
from data import POSTERIOR_MEAN
from functions import (
    func_mu,
    func_mode,
    func_sd_mode_sigmoid,
    func_nm,
)

# Test setup
RTOL = 2e-2
STYLE = "normal"
FILE = "normal_mean-model.csv"

# Add path for expected outputs
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))


# Load the expected outputs, run tests
@pytest.fixture
def results_data():
    ffp = SCRIPT_DIR / "expected_output" / FILE
    return np.genfromtxt(ffp, delimiter=",", skip_header=1, dtype=None)


def test_normal_mean_model(results_data):
    coeffs = POSTERIOR_MEAN.get(STYLE)
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
        sd_mode_calc = func_sd_mode_sigmoid(coeffs, magnitude)
        sd_u_calc = coeffs["sigma"]
        median_calc, sd_tot_calc = func_nm(coeffs, magnitude, location)

        # Tests
        func_names = ["mode", "mu", "sd_mode", "sd_u", "func_nm (med)", "func_nm (sd)"]
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
