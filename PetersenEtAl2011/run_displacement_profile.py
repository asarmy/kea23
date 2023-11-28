"""This file runs the PEA11 principal fault displacement model to create a slip profile for a
single scenario.
- A single scenario is defined as one magnitude, one style, and one percentile.
- The results are returned in a pandas DataFrame.
- Only the principal fault displacement models for direct (i.e., not normalized) predictions are
implemented herein currently.
- Command-line use is supported; try `python run_displacement_profile.py --help`
- Module use is supported; try `from run_displacement_profile import run_profile`

# NOTE: This script just loops over locations in `run_displacement_model.py`

Reference: https://doi.org/10.1785/0120100035

# TODO: There is a potential issue with the bilinear model. Because the standard deviation changes
across l/L', there is a weird step in any profile that is not median. Confirm this is a model
issue and not misunderstanding in implementation.
"""

# Python imports
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add path for module
# FIXME: shouldn't need this with a package install (`__init__` should suffice)
MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(MODEL_DIR))

# Module imports
from PetersenEtAl2011.run_displacement_model import run_model

# Adjust display for readability
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 500)


def run_profile(
    magnitude, percentile, submodel="elliptical", style="strike-slip", location_step=0.05
):
    """
    Run PEA11 principal fault displacement model to create slip profile for a single scenario.

    Parameters
    ----------
    magnitude : float
        Earthquake moment magnitude. Only one value allowed.

    percentile : float
        Percentile value. Use -1 for mean. Only one value allowed.

    submodel : str, optional
        PEA11 shape model name. Default is 'elliptical'. Valid options are 'elliptical',
        'quadratic', or 'bilinear'. Only one value allowed.

    style : str, optional
        Style of faulting (case-sensitive). Default is "strike-slip". Only one value allowed.

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
        - 'model_name': Profile shape model name [from user input].
        - 'mu': Natural log transform of mean displacement in cm.
        - 'sigma': Standard deviation in same units as `mu`.
        - 'displ': Displacement in meters.

    Raises
    ------
    TypeError
        If more than one value is provided for `magnitude`, `submodel`, `style`, or `percentile`.

    Warns
    -----
    UserWarning
        If an unsupported `style` is provided.

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `python run_displacement_profile.py --magnitude 7 --percentile 0.5 -shape bilinear -step 0.01`
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

    if not isinstance(submodel, (str)):
        raise TypeError(
            f"Expected a string, got '{submodel}', which is a {type(submodel).__name__}."
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
            percentile=percentile,
            submodel=submodel,
            style=style,
        )
        run_results.append(results)

    dataframe = pd.concat(run_results, ignore_index=True)

    return dataframe


def main():
    description_text = """Run PEA11 principal fault displacement model to create slip profile for
    a single scenario.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [generated from location_step].
        - 'style': Style of faulting [from user input].
        - 'percentile': Percentile value [from user input].
        - 'model_name': Profile shape model name [from user input].
        - 'mu': Natural log transform of mean displacement in cm.
        - 'sigma': Standard deviation in same units as `mu`.
        - 'displ': Displacement in meters.
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
        required=True,
        type=float,
        help="Percentile value. Use -1 for mean. Only one value allowed.",
    )
    parser.add_argument(
        "-shape",
        "--submodel",
        default="elliptical",
        type=str,
        choices=("elliptical", "quadratic", "bilinear"),
        help="PEA11 shape model name (case-sensitive). Default is 'elliptical'. Only one value allowed.",
    )
    parser.add_argument(
        "-s",
        "--style",
        default="strike-slip",
        type=str,
        help="Style of faulting. Default is 'strike-slip'; other styles not recommended. Only one value allowed.",
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
    percentile = args.percentile
    submodel = args.submodel
    style = args.style
    location_step = args.location_step

    try:
        results = run_profile(magnitude, percentile, submodel, style, location_step)
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
