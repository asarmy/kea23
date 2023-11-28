"""This file runs the KEA23 displacement model for a single scenario.
- A single scenario is defined as one magnitude, one u_star location, one style, and one percentile.
- The results are returned in a pandas dataframe.
- Results for the location, its complement, and folded location are always returned.
- The mean model (i.e., mean coefficients) is run by default, but results for all coefficients can be computed.
- Command-line use is supported; try `python run_displacement_model.py --help`
- Module use is supported; try `from run_displacement_model import run_model`

# NOTE: Several helper functions are defined herein, but the main function is `run_model()`.

Reference: https://doi.org/10.1177/ToBeAssigned
"""

# Python imports
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Add path for module
# FIXME: shouldn't need this with a package install (`__init__` should suffice)
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

# Module imports
from data import load_data
from functions import func_nm, func_rv, func_ss

# Adjust display for readability
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 500)


def _calculate_mean_coefficients(coefficients):
    """
    Helper function to calculate mean model coefficients.

    Parameters
    ----------
    coefficients : pd.DataFrame
        A pandas DataFrame containing model coefficients.

    Returns
    -------
    coeffs : pd.DataFrame
        A pandas DataFrame containing mean model coefficients.
    """

    coeffs = coefficients.mean(axis=0).to_frame().transpose()
    coeffs.loc[0, "model_number"] = -1  # Define model id as -1 for mean coeffs
    coeffs["model_number"] = coeffs["model_number"].astype(int)

    return coeffs


def _calculate_distribution_parameters(*, magnitude, location, style, coefficients):
    """
    Helper function to calculate predicted mean and standard deviation in transformed units.

    Parameters
    ----------
    coefficients : Union[np.recarray, pd.DataFrame]
        A numpy recarray or a pandas DataFrame containing model coefficients.

    magnitude : float
        Earthquake moment magnitude.

    location : float
        Normalized location along rupture length, range [0, 1.0].

    Returns
    -------
    Tuple[np.array, np.array]
        mu : Mean prediction in transformed units.
        sigma : Total standard deviation in transformed units.
    """

    function_map = {"strike-slip": func_ss, "reverse": func_rv, "normal": func_nm}

    # NOTE: Check for appropriate style is handled in `run_model`
    model = function_map[style]
    mu, sigma = model(coefficients, magnitude, location)

    return mu, sigma


def _calculate_Y(*, mu, sigma, lam, percentile):
    """
    Helper function to calculate predicted displacement in transformed units.

    Parameters
    ----------
    mu : np.array
        Mean prediction in transformed units.

    sigma : np.array
        Total standard deviation in transformed units.

    lam : np.array
        "Lambda" transformation parameter in Box-Cox transformation.

    percentile : float
        Percentile value. Use -1 for mean. Only one value allowed.

    Returns
    -------
    Y : np.array
        Predicted displacement in transformed units.
    """

    if percentile == -1:
        # NOTE: Analytical solution from https://robjhyndman.com/hyndsight/backtransforming/
        D = (np.power(lam * mu + 1, 1 / lam)) * (
            1 + (np.power(sigma, 2) * (1 - lam)) / (2 * np.power(lam * mu + 1, 2))
        )
        # NOTE: Analytical soluion is in meters, so convert back to Y transform for consistency
        Y = (np.power(D, lam) - 1) / lam
    else:
        Y = stats.norm.ppf(percentile, loc=mu, scale=sigma)
    return Y


def _calculate_displacement(*, predicted_Y, lam):
    """
    Helper function to calculate predicted displacement in meters.

    Parameters
    ----------
    predicted_Y : np.array
        Predicted displacement in transformed units.

    lam : np.array
        "Lambda" transformation parameter in Box-Cox transformation.

    Returns
    -------
    D : np.array
        Predicted displacement in meters.
    """

    D = np.power(predicted_Y * lam + 1, 1 / lam)

    # Check for values that are too small to calculate
    D = 0 if np.isnan(D).all() else D
    return D


# Preliminaries before defining `run_model`
# Load coefficients
COEFFS_SS = load_data("strike-slip")
COEFFS_RV = load_data("reverse")
COEFFS_NM = load_data("normal")
COEFFS_DICT = {"strike-slip": COEFFS_SS, "reverse": COEFFS_RV, "normal": COEFFS_NM}

# Get mean coefficients
COEFFS_MEAN_SS = _calculate_mean_coefficients(COEFFS_SS)
COEFFS_MEAN_RV = _calculate_mean_coefficients(COEFFS_RV)
COEFFS_MEAN_NM = _calculate_mean_coefficients(COEFFS_NM)
COEFFS_MEAN_DICT = {
    "strike-slip": COEFFS_MEAN_SS,
    "reverse": COEFFS_MEAN_RV,
    "normal": COEFFS_MEAN_NM,
}


def run_model(magnitude, location, style, percentile, mean_model=True):
    """
    Run KEA23 displacement model for a single scenario.

    Parameters
    ----------
    magnitude : float
        Earthquake moment magnitude. Only one value allowed.

    location : float
        Normalized location along rupture length, range [0, 1.0]. Only one value allowed.

    style : str
        Style of faulting (case-sensitive). Valid options are 'strike-slip', 'reverse', or
        'normal'. Only one value allowed.

    percentile : float
        Percentile value. Use -1 for mean. Only one value allowed.

    mean_model : bool, optional
        If True, use mean coefficients. If False, use full coefficients. Default True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [from user input].
        - 'style': Style of faulting [from user input].
        - 'percentile': Percentile value [from user input].
        - 'model_number': Model coefficient row number. Returns -1 for mean model.
        - 'lambda': Box-Cox transformation parameter.
        - 'mu_site': Median transformed displacement for the location.
        - 'sigma_site': Standard deviation transformed displacement for the location.
        - 'mu_complement': Median transformed displacement for the complementary location.
        - 'sigma_complement': Standard deviation transformed displacement for the complementary location.
        - 'Y_site': Transformed displacement for the location.
        - 'Y_complement': Transformed displacement for the complementary location.
        - 'Y_folded': Transformed displacement for the folded location.
        - 'displ_site': Displacement in meters for the location.
        - 'displ_complement': Displacement in meters for the complementary location.
        - 'displ_folded': Displacement in meters for the folded location.

    Raises
    ------
    ValueError
        If the provided `style` is not one of the supported styles.

    TypeError
        If more than one value is provided for `magnitude`, `location`, `style`, or `percentile`.

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `python run_displacement_model.py --magnitude 7 --location 0.5 --style strike-slip --percentile 0.5`
        Run `python run_displacement_model.py --help`

    #TODO
    ------
    Raise a ValueError for invalid location
    Raise a ValueError for invalid percentile.
    Raise a UserWarning for magntiudes outside recommended ranges.
    """

    # Check style
    if style not in COEFFS_DICT:
        raise ValueError(
            f"Unsupported style '{style}'. Supported styles are 'strike-slip', 'reverse', and 'normal' (case-sensitive)."
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

    # Define coefficients
    if mean_model:
        coeffs = COEFFS_MEAN_DICT.get(style)
    else:
        coeffs = COEFFS_DICT.get(style)

    # NOTE: Coefficients are pandas dataframes; convert here to recarray for faster computations
    coeffs = coeffs.to_records(index=False)

    # Get distribution parameters for site and complement
    mu_site, sigma_site = _calculate_distribution_parameters(
        magnitude=magnitude, location=location, style=style, coefficients=coeffs
    )
    mu_complement, sigma_complement = _calculate_distribution_parameters(
        magnitude=magnitude,
        location=1 - location,
        style=style,
        coefficients=coeffs,
    )

    # Calculate Y (transformed displacement)
    lam = coeffs["lambda"]
    Y_site = _calculate_Y(mu=mu_site, sigma=sigma_site, lam=lam, percentile=percentile)
    Y_complement = _calculate_Y(
        mu=mu_complement, sigma=sigma_complement, lam=lam, percentile=percentile
    )
    Y_folded = np.mean([Y_site, Y_complement], axis=0)

    # Calculate displacement in meters
    displ_site = _calculate_displacement(predicted_Y=Y_site, lam=lam)
    displ_complement = _calculate_displacement(predicted_Y=Y_complement, lam=lam)
    displ_folded = _calculate_displacement(predicted_Y=Y_folded, lam=lam)

    # Create a DataFrame
    n = len(coeffs)

    col_vals = (
        np.repeat(magnitude, n),
        np.repeat(location, n),
        np.repeat(style, n),
        np.repeat(percentile, n),
        coeffs["model_number"],
        lam,
        mu_site,
        sigma_site,
        mu_complement,
        sigma_complement,
        Y_site,
        Y_complement,
        Y_folded,
        displ_site,
        displ_complement,
        displ_folded,
    )

    cols_dict = {
        "magnitude": float,
        "location": float,
        "style": object,
        "percentile": float,
        "model_number": int,
        "lambda": float,
        "mu_site": float,
        "sigma_site": float,
        "mu_complement": float,
        "sigma_complement": float,
        "Y_site": float,
        "Y_complement": float,
        "Y_folded": float,
        "displ_site": float,
        "displ_complement": float,
        "displ_folded": float,
    }
    dataframe = pd.DataFrame(np.column_stack(col_vals), columns=cols_dict.keys())
    dataframe = dataframe.astype(cols_dict)

    return dataframe


def main():
    description_text = """Run KEA23 displacement model for a single scenario.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'location':  Normalized location along rupture length [from user input].
        - 'style': Style of faulting [from user input].
        - 'percentile': Percentile value [from user input].
        - 'model_number': Model coefficient row number. Returns -1 for mean model.
        - 'lambda': Box-Cox transformation parameter.
        - 'mu_site': Median transformed displacement for the location.
        - 'sigma_site': Standard deviation transformed displacement for the location.
        - 'mu_complement': Median transformed displacement for the complementary location.
        - 'sigma_complement': Standard deviation transformed displacement for the complementary location.
        - 'Y_site': Transformed displacement for the location.
        - 'Y_complement': Transformed displacement for the complementary location.
        - 'Y_folded': Transformed displacement for the folded location.
        - 'displ_site': Displacement in meters for the location.
        - 'displ_complement': Displacement in meters for the complementary location.
        - 'displ_folded': Displacement in meters for the folded location.
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
        help=" Percentile value. Use -1 for mean. Only one value allowed.",
    )
    parser.add_argument(
        "--mean_model",
        dest="mean_model",
        action="store_true",
        help="Use mean model coefficients (default).",
    )
    parser.add_argument(
        "--no-mean_model",
        dest="mean_model",
        action="store_false",
        help="Use full model coefficients.",
    )
    parser.set_defaults(mean_model=True)

    args = parser.parse_args()

    magnitude = args.magnitude
    location = args.location
    style = args.style
    percentile = args.percentile
    mean_model = args.mean_model

    try:
        results = run_model(magnitude, location, style, percentile, mean_model)
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
