""" This script calculates the average displacement that is implied by the PEA11 model prediction. """

# Python imports
import pandas as pd  # noqa: F401

# Module imports
import proj_setup
from PetersenEtAl2011.run_average_displacement import run_ad

# Calculations
df = run_ad(magnitude=proj_setup.MAGS, submodel="elliptical")

# Save
fout = proj_setup.RESULTS_DIR / "pea11_ss_ad.csv"
df.to_csv(fout, index=False)
