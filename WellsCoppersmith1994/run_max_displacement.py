"""This file runs the WC94 model to calculate the maximum displacement as a function of magnitude
for a single scenario.
- A single scenario is defined as one magnitude and one style.
- The results are returned in a pandas DataFrame.
- Command-line use is supported; try `python run_max_displacement.py --help`
- Module use is supported; try `from run_max_displacement import run_md`

Reference: https://doi.org/10.1785/BSSA0840040974
"""

# Python imports
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Add path for module
# FIXME: shouldn't need this with a package install (`__init__` should suffice)
MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(MODEL_DIR))

# Module imports
from WellsCoppersmith1994.functions import _calc_distrib_params_mag_md

# Adjust display for readability
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 500)


def run_md(magnitude, percentile=0.5, style="all"):
    """
    Run WC94 model to calculate the maximum displacement as a function of magnitude for a single
    scenario.

    Parameters
    ----------
    magnitude : float
        Earthquake moment magnitude. Only one value allowed.

    percentile : float, optional
        Percentile value. Default is 0.5. Use -1 for mean. Only one value allowed.

    style : str, optional
        Style of faulting (case-insensitive). Default is 'all'. Valid options are 'strike-slip',
        'reverse', 'normal', or 'all'. Only one value allowed.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'style': Style of faulting [from user input].
        - 'percentile': Percentile value [from user input].
        - 'mu': Log10 transform of mean displacement in m.
        - 'sigma': Standard deviation in same units as `mu`.
        - 'max_displ': Maximum displacement in meters.

    Raises
    ------
    ValueError
        If the provided `style` is not one of the supported styles.

    TypeError
        If more than one value is provided for `magnitude`, `percentile`, or `style`.

    Warns
    -----
    UserWarning
        If reverse `style` is provided.

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `python run_max_displacement.py --magnitude 7 --style strike-slip`
        Run `python run_max_displacement.py --help`

    #TODO
    ------
    Raise a UserWarning for magntiudes outside recommended ranges.
    """

    # Check style
    if style.lower() not in ["strike-slip", "reverse", "normal", "all"]:
        raise ValueError(
            f"Unsupported style '{style}'. Supported styles are 'strike-slip', 'reverse', 'normal', and 'all' (case-insensitive)."
        )

    # Warn if reverse; relations not recommended
    if style in ["reverse", "Reverse"]:
        msg = "Regressions for reverse-slip relationships are not significant at 95% probability level (per WC94). Use with caution."
        warnings.warn(msg)

    # Check for only one scenario
    for variable in [magnitude, percentile]:
        if not isinstance(variable, (float, int, np.int32)):
            raise TypeError(
                f"Expected a float or int, got '{variable}', which is a {type(variable).__name__}."
                f"(In other words, only one value is allowed; check you have not entered a list or array.)"
            )

    if not isinstance(style, (str)):
        raise TypeError(
            f"Expected a string, got '{style}', which is a {type(style).__name__}."
            f"(In other words, only one value is allowed; check you have not entered a list or array.)"
        )

    # Calculate distribution parameters
    mu, sigma = _calc_distrib_params_mag_md(magnitude=magnitude, style=style)

    # Calculate log of displacement
    if percentile == -1:
        log10_displ = mu + (np.log(10) / 2 * np.power(sigma, 2))
    else:
        log10_displ = stats.norm.ppf(percentile, loc=mu, scale=sigma)

    # Calculate displacement
    D = np.power(10, log10_displ)

    # Create output dataframe
    dataframe = pd.concat(
        [
            pd.Series(magnitude, name="magnitude"),
            pd.Series(style, name="style"),
            pd.Series(percentile, name="percentile"),
            pd.Series(mu, name="mu"),
            pd.Series(sigma, name="sigma"),
            pd.Series(D, name="max_displ"),
        ],
        axis=1,
    )

    return dataframe


def main():
    description_text = """Run WC94 model to calculate the maximum displacement as a function of
    magnitude for a single scenario.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'style': Style of faulting [from user input].
        - 'percentile': Percentile value [from user input].
        - 'mu': Log10 transform of mean displacement in m.
        - 'sigma': Standard deviation in same units as `mu`.
        - 'max_displ': Maximum displacement in meters.
    """

    parser = argparse.ArgumentParser(
        description=description_text, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--magnitude",
        required=True,
        type=float,
        help="Earthquake moment magnitude. Only one value allowed.",
    )
    parser.add_argument(
        "-p",
        "--percentile",
        default=0.5,
        type=float,
        help=" Percentile value. Default is 0.5. Use -1 for mean. Only one value allowed.",
    )
    parser.add_argument(
        "-s",
        "--style",
        default="all",
        type=str,
        choices=("strike-slip", "reverse", "normal", "all"),
        help="Style of faulting (case-insensitive). Default is 'all'. Only one value allowed.",
    )

    args = parser.parse_args()

    magnitude = args.magnitude
    percentile = args.percentile
    style = args.style

    try:
        results = run_md(magnitude, percentile, style)
        print(results)

        # Prompt to save results to CSV
        save_option = input("Do you want to save the results to a CSV (yes/no)? ").strip().lower()

        if save_option in ["y", "yes"]:
            file_path = input("Enter filepath to save results: ").strip()
            if file_path:
                # Create the directory if it doesn't exist
                path = Path(file_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                results.to_csv(file_path, index=False)
                print(f"Results saved to {file_path}")
            else:
                print("Invalid file path. Results not saved.")
        else:
            print("Results not saved.")

    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
