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
RTOL = 1e-2


@pytest.mark.parametrize("filename", [FILE])
def test_calc(load_data_as_recarray):
    # Define inputs and expected outputs
    magnitudes = load_data_as_recarray["magnitude"]
    locations = load_data_as_recarray["location"]
    mu_expect = load_data_as_recarray["mu"]
    sigma_expect = load_data_as_recarray["sigma"]

    # Perform calculations
    results = FUNCTION(magnitude=magnitudes, location=locations)
    mu_calc, sigma_calc = results

    # Comparing exepcted and calculated
    np.testing.assert_allclose(mu_expect, mu_calc, rtol=RTOL, err_msg="Discrepancy in mu values")
    np.testing.assert_allclose(
        sigma_expect, sigma_calc, rtol=RTOL, err_msg="Discrepancy in sigma values"
    )
