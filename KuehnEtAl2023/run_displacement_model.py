import pandas as pd

from data import load_data
from functions import func_ss, func_rv, func_nm



# first load coeffs; then alter for mean if necessary; then that is inout into here
def _calculate_distribution_parameters(*, magnitude, location, style, coefficients):

    function_map = {"strike-slip": func_ss, "reverse": func_rv, "normal": func_nm}
    
    if style in function_map:
        model = function_map[style]
        mu, sigma = model(coefficients, magnitude, location)
    else:
        raise ValueError(f"Unsupported 's' value: {style}")

    return mu, sigma

    # mu_left, sigma_left = func_nm(coeffs, m, l)

    # func_ss(coeffs, mag, loc
    # mu_right, sigma_right = func_nm(coeffs, m, 1 - l)

    # return mu_left, sigma_left, mu_right, sigma_right



# def _calculate_Y(

m, l, s = 7, 0.5, "strike-slip"
coeffs = load_data(s)

ans = _calculate_distribution_parameters(magnitude=m, location=l, style= "strike-slip", coefficients=coeffs)
print(ans)

# Load coefficients
# COEFFS_SS = load_data("strike-slip")
# COEFFS_RV = load_data("reverse")
# COEFFS_NM = load_data("normal")