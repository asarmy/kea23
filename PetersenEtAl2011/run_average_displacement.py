"""This file runs the PEA11 principal fault displacement model to calculate the average
displacement that is implied by the model prediction for a single scenario.
- A single scenario is defined as one magnitude and one style.
- The model-implied Average Displacement is calculated as the area under the mean slip profile.
- The results are returned in a pandas DataFrame.
- Only the principal fault displacement models for direct (i.e., not normalized) predictions are
implemented herein currently.
- Command-line use is supported; try `python run_average_displacement.py --help`
- Module use is supported; try `from run_average_displacement import run_ad`

# NOTE: This script just calls `run_displacement_profile.py` which in turn calls `run_displacement_model.py`

Reference: https://doi.org/10.1785/0120100035

# TODO: There is a potential issue with the bilinear model. Because the standard deviation changes
across l/L', there is a weird step in any profile that is not median. Confirm this is a model
issue and not misunderstanding in implementation. The issue affects the AD calc for bilinear model.
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
from PetersenEtAl2011.run_displacement_profile import run_profile

# Adjust display for readability
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 500)


def run_ad(magnitude, submodel="elliptical", style="strike-slip"):
    """
    Run PEA11 principal fault displacement model to calculate the average displacement that is
    implied by the model prediction for a single scenario.

    Parameters
    ----------
    magnitude : float
        Earthquake moment magnitude. Only one value allowed.

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
        - 'style': Style of faulting [from user input].
        - 'model_name': Profile shape model name [from user input].
        - 'avg_displ': Averaged displacement in meters.

    Raises
    ------
    TypeError
        If more than one value is provided for `magnitude`, `submodel`, or `style`.

    Warns
    -----
    UserWarning
        If an unsupported `style` is provided.

    Notes
    ------
    Command-line interface usage
        Run (e.g.) `python run_average_displacement.py --magnitude 7 --submodel quadratic`
        Run `python run_average_displacement.py --help`

    #TODO
    ------
    Raise a UserWarning for magntiudes outside recommended ranges.
    This runs very slowly when you loop through magnitudes (e.g., creating an M-AD plot). Vectorize somehow (move to numpy? cython?)
    """

    # Check for only one scenario
    if not isinstance(magnitude, (float, int, np.int32)):
        raise TypeError(
            f"Expected a float or int, got '{magnitude}', which is a {type(magnitude).__name__}."
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

    # NOTE: Check for appropriate style is handled upstream

    # Calculate mean slip profile
    # NOTE: `percentile=-1` is used for mean
    # NOTE: `location_step=0.01` is used to create well-descritized profile for intergration
    results = run_profile(
        magnitude=magnitude, percentile=-1, submodel=submodel, style=style, location_step=0.01
    )

    # Calculate area under the mean slip profile; this is the Average Displacement (AD)
    x, y = results["location"], results["displ"]
    area = np.trapz(y, x)

    # Create output dataframe
    model_id = results["model_name"].iloc[0]
    dataframe = pd.concat(
        [
            pd.Series(magnitude, name="magnitude"),
            pd.Series(style, name="style"),
            pd.Series(model_id, name="model_name"),
            pd.Series(area, name="avg_displ"),
        ],
        axis=1,
    )

    return dataframe


def main():
    description_text = """Run PEA11 principal fault displacement model to calculate the average displacement that is
    implied by the model prediction for a single scenario.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the following columns:
        - 'magnitude': Earthquake moment magnitude [from user input].
        - 'style': Style of faulting [from user input].
        - 'model_name': Profile shape model name [from user input].
        - 'avg_displ': Averaged displacement in meters.
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
    submodel = args.submodel
    style = args.style

    try:
        results = run_ad(magnitude, submodel, style)
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
