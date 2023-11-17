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
    """

    if style in FILENAMES:
        fn = FILENAMES[style]
        ffp = DIR_DATA / fn
        df = pd.read_csv(ffp)
        print(f"Model coefficients successfully loaded for {style} faulting.")
        return df
    else:
        raise ValueError(MSG.format(style=style))


def main():
    # Command-line argument parser
    parser = argparse.ArgumentParser(
        description="Load model coefficients based on style of faulting."
    )
    parser.add_argument(
        "--style",
        required=True,
        help="Choose 'strike-slip', 'reverse', or 'normal' (case-sensitive).",
    )
    args = parser.parse_args()

    style = args.style

    # Check if the style is valid before calling load_data
    if style not in FILENAMES:
        print(MSG.format(style=style))
        return

    try:
        load_data(style)
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()