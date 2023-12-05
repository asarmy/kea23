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
from PetersenEtAl2011.run_displacement_model import run_model

# Test setup
RTOL = 1e-2

# Add path for expected outputs
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR.parent))


# Load the expected outputs, run tests
@pytest.fixture
def results_data():
    ffp = SCRIPT_DIR / "expected_output" / "petersen_et_al_2011_displacements.csv"
    dtype = [float, float, float, float, "U20"]
    return np.genfromtxt(ffp, delimiter=",", skip_header=1, dtype=dtype)


def test_run_model(results_data):
    for row in results_data:
        # Inputs
        magnitude = row[0]
        location = row[1]
        percentile = row[2]
        shape = row[4]

        # Expected
        displ_expect = row[3]

        # Computed
        results = run_model(
            magnitude=magnitude,
            location=location,
            percentile=percentile,
            submodel=shape,
        )
        displ_calc = results["displ"]

        # Tests
        np.testing.assert_allclose(
            displ_expect,
            displ_calc,
            rtol=RTOL,
            err_msg=f"Mag {magnitude}, loc {location}, percentile {percentile}, submodel {shape},Expected: {displ_expect}, Computed: {displ_calc}",
        )
