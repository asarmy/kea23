# Define variables for Python interpreter
PYTHON=python

# Define scripts for calculations
CALCS=\
	calc_kea23_avg_displ.py \
	calc_mr11_avg_displ.py \
	calc_p11_avg_displ.py \
	calc_wc94_avg_displ.py

# Define scripts for plots
PLOTS=\
	plot_avg_displ_ss.py \
	plot_avg_displ_rv.py \
	plot_avg_displ_nm.py

# Define default target for make to run all scripts
all: $(CALCS) $(PLOTS)
calcs: $(CALCS)
plots: $(PLOTS)

# Define targets for the scripts
$(CALCS) $(PLOTS):
	cd $(shell dirname $(MAKEFILE_LIST)) && $(PYTHON) $@
