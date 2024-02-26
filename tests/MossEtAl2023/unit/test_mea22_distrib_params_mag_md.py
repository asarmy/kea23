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
from MossEtAl2023.functions import _calc_distrib_params_mag_md

# Test setup
FUNCTION = _calc_distrib_params_mag_md
FILE = "moss_et_al_params_mag_md.csv"
RTOL = 1e-2


@pytest.mark.parametrize("filename", [FILE])
def test_calc(load_data_as_recarray):
    # Define inputs and expected outputs
    magnitudes = load_data_as_recarray["magnitude"]
    mu_expect = load_data_as_recarray["mu"]
    sigma_expect = load_data_as_recarray["sigma"]

    # Perform calculations
    results = FUNCTION(magnitude=magnitudes)
    mu_calc, sigma_calc = results

    # Comparing exepcted and calculated
    np.testing.assert_allclose(mu_expect, mu_calc, rtol=RTOL, err_msg="Discrepancy in mu values")
    np.testing.assert_allclose(
        sigma_expect, sigma_calc, rtol=RTOL, err_msg="Discrepancy in sigma values"
    )
