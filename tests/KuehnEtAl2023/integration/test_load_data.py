import sys
from pathlib import Path

import pandas as pd
import pytest

PROJ_DIR = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJ_DIR))

from KuehnEtAl2023.data import load_data


# Test cases
def test_load_data_valid_styles():
    # Test loading data for valid styles
    supported_styles = ["strike-slip", "reverse", "normal"]

    for style in supported_styles:
        data = load_data(style)
        assert isinstance(data, pd.DataFrame)


def test_load_data_invalid_style():
    # Test loading data for an unsupported style
    with pytest.raises(ValueError):
        style = "meow"
        load_data(style)
