"""This script is called to load the model coefficients for the KEA23 displacement model.
- The coefficient files vary based on style of faulting.
- The coefficients are returned in a pandas dataframe.
- Command-line use is supported; try `python data.py --help`
- Module use is supported; try `from data import load_data`

# NOTE: This script is called in the main function in `run_model()`.

Reference: https://doi.org/10.1177/ToBeAssigned
"""

# Python imports
import argparse
from pathlib import Path

import pandas as pd

# Filepath for model coefficients
DIR_DATA = Path(__file__).parents[1] / "data" / "KuehnEtAl2023"

# Filenames for model coefficients
FILENAMES = {
    "strike-slip": "coefficients_posterior_SS_powtr.csv",
    "reverse": "coefficients_posterior_REV_powtr.csv",
    "normal": "coefficients_posterior_NM_powtr.csv",
}

# Constant error message for unsupported style
MSG = "Unsupported style '{style}'. Supported styles are 'strike-slip', 'reverse', and 'normal' (case-sensitive)."


def load_data(style):
    """
    Load model coefficients based on style of faulting.

    Parameters
    ----------
    style : str
        Style of faulting (case-sensitive).
        Valid options are 'strike-slip', 'reverse', or 'normal'.

    Returns
    -------
    DataFrame
        A pandas DataFrame containing the loaded data.

    Raises
    ------
    ValueError
        If the provided `style` is not one of the supported styles.

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `python data.py --style normal`
        Run `python data.py --help`
        #NOTE: CLI is not useful for this application (loading coeffs) but included for completeness.
    """

    if style in FILENAMES:
        filename = FILENAMES[style]
        filepath = DIR_DATA / filename
        df = pd.read_csv(filepath)
        df = df.rename(columns={"Unnamed: 0": "model_number"})
        # print(f"Model coefficients successfully loaded for {style} faulting.")
        return df
    else:
        raise ValueError(MSG.format(style=style))


def main():
    # Command-line argument parser
    parser = argparse.ArgumentParser(
        description="Load model coefficients based on style of faulting."
    )
    parser.add_argument(
        "-s",
        "--style",
        required=True,
        type=str,
        help="Style of faulting (case-sensitive). Valid options are 'strike-slip', 'reverse', or 'normal'.",
    )
    args = parser.parse_args()

    style = args.style

    # Check if the style is valid before calling load_data
    if style not in FILENAMES:
        print(MSG.format(style=style))
        return

    try:
        data = load_data(style)
        print(data)
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
