# Define variables for Python interpreter
PYTHON=python

# Define scripts for calculations
CALCS=\
	calc_kea23_profiles.py \
	calc_mr11_profiles.py \
	calc_p11_profiles.py \
	calc_yea03_profiles.py

# Define scripts for plots
PLOTS=\
	plot_profiles_ss.py \
	plot_profiles_rv.py \
	plot_profiles_nm.py

# Define default target for make to run all scripts
all: $(CALCS) $(PLOTS)
calcs: $(CALCS)
plots: $(PLOTS)

# Define targets for the scripts
$(CALCS) $(PLOTS):
	cd $(shell dirname $(MAKEFILE_LIST)) && $(PYTHON) $@
