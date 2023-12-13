import pandas as pd

# Console output formatting
pd.set_option("display.max_columns", 800)
pd.set_option("display.width", 800)

# Monte Carlo sample size
# NOTE: N=500,000 was chosen because it is still reasonably fast and produces smooth slip profiles.
N_SAMPLES = 500000

# Numpy seed
NP_SEED = 123
