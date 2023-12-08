""" This script calculates the average displacement using the relation in MR11. """

# Python imports
import pandas as pd  # noqa: F401

# Module imports
import proj_setup
from MossRoss2011.run_average_displacement import run_ad

# Calculations
df = run_ad(magnitude=proj_setup.MAGS)

# Save
fout = proj_setup.RESULTS_DIR / "mr11_rv_ad.csv"
df.to_csv(fout, index=False)
