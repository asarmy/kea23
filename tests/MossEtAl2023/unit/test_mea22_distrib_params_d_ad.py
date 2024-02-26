"""This file contains tests for the internal functions used to calculate principal fault
displacement using the Moss and Ross (2011) model. The basis for the test answers were provided by
Dr. Robb Moss in August 2021, and the results were computed by Alex Sarmiento in October 2022.
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
from MossEtAl2023.functions import _calc_distrib_params_d_ad

# Test setup
FUNCTION = _calc_distrib_params_d_ad
FILE = "moss_et_al_params_d_ad.csv"
RTOL = 1e-2


@pytest.mark.parametrize("filename", [FILE])
def test_calc(load_data_as_recarray):
    # Define inputs and expected outputs
    locations = load_data_as_recarray["location"]
    alpha_expect = load_data_as_recarray["alpha"]
    beta_expect = load_data_as_recarray["beta"]

    # Perform calculations
    results = FUNCTION(location=locations)
    alpha_calc, beta_calc = results

    # Comparing exepcted and calculated
    np.testing.assert_allclose(
        alpha_expect, alpha_calc, rtol=RTOL, err_msg="Discrepancy in alpha values"
    )
    np.testing.assert_allclose(
        beta_expect, beta_calc, rtol=RTOL, err_msg="Discrepancy in beta values"
    )
