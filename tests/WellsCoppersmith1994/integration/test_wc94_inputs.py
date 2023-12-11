"""This file contains tests for the errors or warnings that should be raised with the user
functions for the Wells and Coppersmith (1994) models.
"""

# Python imports
import sys
from pathlib import Path

import pytest

# Add path for module
# FIXME: shouldn't need this with a package install (`__init__` should suffice)
PROJ_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJ_DIR))

# Module imports
from WellsCoppersmith1994.run_average_displacement import run_ad
from WellsCoppersmith1994.run_max_displacement import run_md


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_input_style():
    mag = [6.5, 7]

    # Test with appropriate styles, case-insensitive; no exception should be raised
    sof = ["Strike-Slip", "strike-slip"]
    run_ad(magnitude=mag, style=sof)
    run_md(magnitude=mag, style=sof)

    # Test with invalid style
    sof = ["Strike-Slip", "meow"]
    with pytest.raises(ValueError):
        run_ad(magnitude=mag, style=sof)
        run_md(magnitude=mag, style=sof)

    # Test with reverse style
    sof = ["normal", "Reverse"]
    with pytest.warns(UserWarning):
        run_ad(magnitude=mag, style=sof)
        run_md(magnitude=mag, style=sof)
