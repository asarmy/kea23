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
from MossRoss2011.functions import _calc_distrib_params_mag_md

# Test setup
RTOL = 2e-2

# Add path for expected outputs
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))


# Load the expected outputs, run tests
@pytest.fixture
def results_data():
    ffp = SCRIPT_DIR / "expected_output" / "moss_ross_params_mag_md.csv"
    dtype = [float, float, float]
    return np.genfromtxt(ffp, delimiter=",", skip_header=1, dtype=dtype)


def test_run_model(results_data):
    for row in results_data:
        # Inputs
        magnitude = row[0]

        # Expected
        mu_expect = row[1]
        sigma_expect = row[2]

        # Computed
        results = _calc_distrib_params_mag_md(magnitude=magnitude)
        mu_calc = results[0]
        sigma_calc = results[1]

        # Tests
        np.testing.assert_allclose(
            mu_expect,
            mu_calc,
            rtol=RTOL,
            err_msg=f"Mag {magnitude}, Expected mu (log10): {mu_expect}, Computed: {mu_calc}",
        )

        np.testing.assert_allclose(
            sigma_expect,
            sigma_calc,
            rtol=RTOL,
            err_msg=f"Mag {magnitude}, Expected sigma (log10): {sigma_expect}, Computed: {sigma_calc}",
        )
