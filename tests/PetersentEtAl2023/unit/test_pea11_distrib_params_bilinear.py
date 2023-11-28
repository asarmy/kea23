"""This file contains tests for the source functions used to calcualte principal fault
displacement using the Petersen et al. (2011) model. The test answers were provided by
Dr. Rui Chen in July 2021.
"""

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
from PetersenEtAl2011 import functions as pea11

# Test setup
FUNCTION = pea11._calc_distrib_params_bilinear
FILE = "petersen_et_al_2011_bilinear_params.csv"
RTOL = 2e-2


@pytest.mark.parametrize("filename", [FILE])
@pytest.mark.filterwarnings("ignore:Value")
def test_calc(load_data_as_recarray):
    # Loop over rows in CSV
    for row in load_data_as_recarray:
        # Set i/o
        mag, loc = row["magnitude"], row["location"]
        expected_mu, expected_sigma = row["mu"], row["sigma"]

        # Perform calculation
        mu, sigma = FUNCTION(magnitude=mag, location=loc)

        # Check mu
        np.testing.assert_allclose(
            expected_mu,
            mu,
            rtol=RTOL,
            err_msg=(
                f"***For (magnitude, location) of {(mag, loc)}: "
                f"Expected mu: {expected_mu}, Computed mu: {mu}***"
            ),
        )

        # Check sigma
        np.testing.assert_allclose(
            expected_sigma,
            sigma,
            rtol=RTOL,
            err_msg=(
                f"***For (magnitude, location) of {(mag, loc)}: "
                f"Expected sigma: {expected_sigma}, Computed sigma: {sigma}***"
            ),
        )
