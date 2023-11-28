"""This file runs the PEA11 principal fault displacement model for a single scenario.
- A single scenario is defined as one magnitude, one u_star location, one style, and one percentile.
- The results are returned in a pandas DataFrame.
- Only the principal fault displacement models for direct (i.e., not normalized) predictions are
implemented herein currently.
- Command-line use is supported; try `python run_displacement_model.py --help`
- Module use is supported; try `from run_displacement_model import run_model`

Reference: https://doi.org/10.1785/0120100035
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
from PetersenEtAl2011.functions import (
    _calc_distrib_params_elliptical,
    _calc_distrib_params_quadratic,
    _calc_distrib_params_bilinear,
)

# Adjust display for readability
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 500)


def run_model(magnitude, location, percentile, submodel="elliptical", style="strike-slip"):
    """
    Run PEA11 principal fault displacement model for a single scenario.

    Parameters
    ----------
    magnitude : float
        Earthquake moment magnitude. Only one value allowed.

    location : float
        Normalized location along rupture length, range [0, 1.0]. Only one value allowed.

    percentile : float
        Percentile value. Use -1 for mean. Only one value allowed.

    submodel : str, optional
        PEA11 shape model name. Default is 'elliptical'. Valid options are 'elliptical',
        'quadratic', or 'bilinear'. Only one value allowed.

    style : str, optional
        Style of faulting (case-sensitive). Default is "strike-slip". Only one value allowed.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [from user input].
        - 'style': Style of faulting [from user input].
        - 'percentile': Percentile value [from user input].
        - 'model_name': Profile shape model name [from user input].
        - 'mu': Natural log transform of mean displacement in cm.
        - 'sigma': Standard deviation in same units as `mu`.
        - 'displ': Displacement in meters.

    Raises
    ------
    TypeError
        If more than one value is provided for `magnitude`, `location`, `submodel`, `style`, or
        `percentile`.

    Warns
    -----
    UserWarning
        If an unsupported `style` is provided.

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `python run_displacement_model.py --magnitude 7 --location 0.5 --percentile 0.5 --submodel quadratic`
        Run `python run_displacement_model.py --help`

    #TODO
    ------
    Raise a ValueError for invalid location
    Raise a ValueError for invalid percentile.
    Raise a UserWarning for magntiudes outside recommended ranges.
    """

    # Check style
    if style not in ("strike-slip", "Strike-Slip"):
        warnings.warn(
            f"This model is only recommended for strike-slip faulting, but '{style}' was entered.",
            category=UserWarning,
        )

    # Check for only one scenario
    for variable in [magnitude, location, percentile]:
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

    # Check for supported submodels
    supported_submodels = {
        "elliptical": _calc_distrib_params_elliptical,
        "quadratic": _calc_distrib_params_quadratic,
        "bilinear": _calc_distrib_params_bilinear,
    }

    func = supported_submodels.get(submodel)

    if func is None:
        raise ValueError(
            f"Invalid submodel name. Valid options are {list(supported_submodels.keys())} (case-sensitive)."
        )

    # Calculate distribution parameters
    mu, sigma = func(magnitude=magnitude, location=location)

    # Calculate natural log of displacement
    if percentile == -1:
        ln_displ = mu + np.power(sigma, 2) / 2
    else:
        ln_displ = stats.norm.ppf(percentile, loc=mu, scale=sigma)

    # NOTE: exp(mu) is in centimeters, must be converted to  meters.
    D = np.exp(ln_displ) / 100

    # Create a DataFrame
    col_vals = (
        magnitude,
        location,
        style,
        percentile,
        submodel,
        mu,
        sigma,
        D,
    )

    cols_dict = {
        "magnitude": float,
        "location": float,
        "style": str,
        "percentile": float,
        "model_name": str,
        "mu": float,
        "sigma": float,
        "displ": float,
    }
    dataframe = pd.DataFrame(np.column_stack(col_vals), columns=cols_dict.keys())
    dataframe = dataframe.astype(cols_dict)

    return dataframe


def main():
    description_text = """Run PEA11 principal fault displacement model for a single scenario.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [from user input].
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
        "-l",
        "--location",
        required=True,
        type=float,
        help="Normalized location along rupture length, range [0, 1.0]. Only one value allowed.",
    )
    parser.add_argument(
        "-p",
        "--percentile",
        required=True,
        type=float,
        help=" Percentile value. Use -1 for mean. Only one value allowed.",
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

    args = parser.parse_args()

    magnitude = args.magnitude
    location = args.location
    percentile = args.percentile
    submodel = args.submodel
    style = args.style

    try:
        results = run_model(magnitude, location, percentile, submodel, style)
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
