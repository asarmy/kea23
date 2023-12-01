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
from run_displacement_model import run_model

# Test setup
RTOL = 2e-2

# Add path for expected outputs
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))


# Load the expected outputs, run tests
@pytest.fixture
def results_data():
    ffp = SCRIPT_DIR / "expected_output" / "displacement_mean-model.csv"
    dtype = [float, float, float, "U20"] + [float] * 10
    return np.genfromtxt(ffp, delimiter=",", skip_header=1, dtype=dtype)


def test_run_model(results_data):
    for row in results_data:
        # Inputs
        magnitude = row[0]
        location = row[1]
        percentile = row[2]
        style = row[3]

        # Expected
        mu_loc_expect = row[4]
        sd_loc_expect = row[5]
        mu_compl_expect = row[6]
        sd_compl_expect = row[7]
        Y_site_expect = row[8]
        Y_compl_expect = row[9]
        Y_folded_expect = row[10]
        displ_site_expect = row[11]
        displ_compl_expect = row[12]
        displ_folded_expect = row[13]

        # Computed
        results = run_model(
            magnitude=magnitude,
            location=location,
            style=style,
            percentile=percentile,
        )
        mu_loc_calc = results["mu_site"]
        sd_loc_calc = results["sigma_site"]
        mu_compl_calc = results["mu_complement"]
        sd_compl_calc = results["sigma_complement"]
        Y_site_calc = results["Y_site"]
        Y_compl_calc = results["Y_complement"]
        Y_folded_calc = results["Y_folded"]
        displ_site_calc = results["displ_site"]
        displ_compl_calc = results["displ_complement"]
        displ_folded_calc = results["displ_folded"]

        # Tests
        expected_values = [
            mu_loc_expect,
            sd_loc_expect,
            mu_compl_expect,
            sd_compl_expect,
            Y_site_expect,
            Y_compl_expect,
            Y_folded_expect,
            displ_site_expect,
            displ_compl_expect,
            displ_folded_expect,
        ]
        computed_values = [
            mu_loc_calc,
            sd_loc_calc,
            mu_compl_calc,
            sd_compl_calc,
            Y_site_calc,
            Y_compl_calc,
            Y_folded_calc,
            displ_site_calc,
            displ_compl_calc,
            displ_folded_calc,
        ]

        for (expected, computed) in zip(expected_values, computed_values):
            np.testing.assert_allclose(
                expected,
                computed,
                rtol=RTOL,
                err_msg=f"Mag {magnitude}, u-star {location}, style {style}, percentile {percentile}, Expected: {expected}, Computed: {computed}",
            )
