"""This file runs the KEA23 displacement model to create slip profile for a single scenario.
- A single scenario is defined as one magnitude, one style, and one percentile.
- The mean model (i.e., mean coefficients) is used.
- The results are returned in a pandas dataframe.
- Results for left-peak, right-peak, and folded (symmetrical) profiles are always returned.
- Command-line use is supported; try `python run_displacement_profile.py --help`
- Module use is supported; try `from run_displacement_profile import run_profile`

# NOTE: This script just loops over locations in `run_displacement_model.py`

Reference: https://doi.org/10.1177/ToBeAssigned
"""


# Python imports
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add path for module
# FIXME: shouldn't need this with a package install (`__init__` should suffice)
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

# Module imports
from run_displacement_model import run_model

# Adjust display for readability
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 500)


def run_profile(magnitude, style, percentile, location_step=0.05):
    """
    Run KEA23 displacement model to create slip profile for a single scenario. The mean model
    (i.e., mean coefficients) is used.

    Parameters
    ----------
    magnitude : float
        Earthquake moment magnitude. Only one value allowed.

    style : str
        Style of faulting (case-sensitive). Valid options are 'strike-slip', 'reverse', or
        'normal'. Only one value allowed.

    percentile : float
        Percentile value. Use -1 for mean. Only one value allowed.

    location_step : float, optional
        Profile step interval in percentage. Default 0.05.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [generated from location_step].
        - 'style': Style of faulting [from user input].
        - 'percentile': Percentile value [from user input].
        - 'model_number': Model coefficient row number. Returns -1 for mean model.
        - 'lambda': Box-Cox transformation parameter.
        - 'mu_left': Median transformed displacement for the left-peak profile.
        - 'sigma_left': Standard deviation transformed displacement for the left-peak profile.
        - 'mu_right': Median transformed displacement for the right-peak profile.
        - 'sigma_right': Standard deviation transformed displacement for the right-peak profile.
        - 'Y_left': Transformed displacement for the left-peak profile.
        - 'Y_right': Transformed displacement for the right-peak profile.
        - 'Y_folded': Transformed displacement for the folded (symmetrical) profile.
        - 'displ_left': Displacement in meters for the left-peak profile.
        - 'displ_right': Displacement in meters for the right-peak profile.
        - 'displ_folded': Displacement in meters for the folded (symmetrical) profile.

    Raises
    ------
    ValueError
        If the provided `style` is not one of the supported styles.

    TypeError
        If more than one value is provided for `magnitude`, `style`, or `percentile`.

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `python run_displacement_profile.py --magnitude 7 --style strike-slip --percentile 0.5 -step 0.01`
        Run `python run_displacement_profile.py --help`

    #TODO
    ------
    Raise a ValueError for invalid location_step size.
    Raise a ValueError for invalid percentile.
    Raise a UserWarning for magntiudes outside recommended ranges.
    """

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

    # NOTE: Check for appropriate style is handled in `run_model`

    # Create profile location array
    locations = np.arange(0, 1 + location_step, location_step).tolist()

    # Calculations
    run_results = []
    for location in locations:
        results = run_model(
            magnitude=magnitude,
            location=location,
            style=style,
            percentile=percentile,
            mean_model=True,
        )
        run_results.append(results)

    dataframe = pd.concat(run_results, ignore_index=True)

    return dataframe


def main():
    description_text = """Run KEA23 displacement model to create slip profile for a single scenario.
    The mean model (i.e., mean coefficients) are used.

    Returns
    -------
    pandas.DataFrame
        A DataFpandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [generated from location_step].
        - 'style': Style of faulting [from user input].
        - 'percentile': Percentile value [from user input].
        - 'model_number': Model coefficient row number. Returns -1 for mean model.
        - 'lambda': Box-Cox transformation parameter.
        - 'mu_left': Median transformed displacement for the left-peak profile.
        - 'sigma_left': Standard deviation transformed displacement for the left-peak profile.
        - 'mu_right': Median transformed displacement for the right-peak profile.
        - 'sigma_right': Standard deviation transformed displacement for the right-peak profile.
        - 'Y_left': Transformed displacement for the left-peak profile.
        - 'Y_right': Transformed displacement for the right-peak profile.
        - 'Y_folded': Transformed displacement for the folded (symmetrical) profile.
        - 'displ_left': Displacement in meters for the left-peak profile.
        - 'displ_right': Displacement in meters for the right-peak profile.
        - 'displ_folded': Displacement in meters for the folded (symmetrical) profile.
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
        "-s",
        "--style",
        required=True,
        type=str,
        help="Style of faulting (case-sensitive). Valid options are 'strike-slip', 'reverse', or 'normal'. Only one value allowed.",
    )
    parser.add_argument(
        "-p",
        "--percentile",
        required=True,
        type=float,
        help="Percentile value. Use -1 for mean. Only one value allowed.",
    )
    parser.add_argument(
        "-step",
        "--location_step",
        default=0.05,
        type=float,
        help="Profile step interval in percentage. Default 0.05.",
    )

    args = parser.parse_args()

    magnitude = args.magnitude
    style = args.style
    percentile = args.percentile
    location_step = args.location_step

    try:
        results = run_profile(magnitude, style, percentile, location_step)
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
