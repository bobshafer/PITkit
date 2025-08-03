import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar, minimize

# --- Constants and Data Loading ---
COLUMN_NAMES = [
    'Radius', 'V_obs', 'Err_V_obs', 'V_gas', 'V_disk', 'V_bulge', 
    'SB_disk', 'SB_bulge'
]
galaxy_file = 'data/NGC3198_rotmod.dat'
galaxy_data = pd.read_csv(
    galaxy_file, delim_whitespace=True, comment='#', header=None, names=COLUMN_NAMES
)
# We'll treat these as fixed universal constants for now
A0_KMS = 3700.0   # (km/s)^2 / kpc
ALPHA_MAX = 15.2

# --- The Core PIT Model (from previous script) ---
def calculate_pit_velocity(radius_kpc, v_gas, v_disk, upsilon_disk, a_0_kms):
    """Calculates the predicted rotation curve using the PIT/MOND framework."""
    v_baryon_sq = v_gas**2 + upsilon_disk * v_disk**2
    g_newtonian = np.divide(v_baryon_sq, radius_kpc, where=radius_kpc!=0)
    
    def mu(x):
        return x / np.sqrt(1 + x**2)

    total_acceleration = np.zeros_like(g_newtonian)
    for i, g_n_val in enumerate(g_newtonian):
        if g_n_val == 0:
            continue
        def equation_to_solve(a):
            return a * mu(a / a_0_kms) - g_n_val
        sol = root_scalar(equation_to_solve, bracket=[0, 2 * g_n_val + a_0_kms], method='brentq')
        total_acceleration[i] = sol.root

    v_pred_sq = total_acceleration * radius_kpc
    return np.sqrt(v_pred_sq)

# --- NEW: The Chi-Squared Cost Function ---
def chi_squared(params, radius, v_obs, err_v_obs, v_gas, v_disk, a_0):
    """
    Calculates the chi-squared value for a given set of parameters.
    """
    upsilon_disk = params[0] # The parameter we are fitting
    
    # Calculate the model's prediction for this upsilon_disk
    v_predicted = calculate_pit_velocity(radius, v_gas, v_disk, upsilon_disk, a_0)
    
    # Calculate chi-squared
    # χ² = Σ [ (V_obs - V_pred) / Err_V_obs ]^2
    chi_sq = np.sum(((v_obs - v_predicted) / err_v_obs)**2)
    
    return chi_sq

# --- NEW: The Fitting Routine ---

# Initial guess for the parameter we want to fit (upsilon_disk)
initial_guess = [0.5] 

# The data we need to pass to the chi-squared function
# Note: root_scalar and minimize need the data passed in as numpy arrays (.values)
args = (
    galaxy_data['Radius'].values,
    galaxy_data['V_obs'].values,
    galaxy_data['Err_V_obs'].values,
    galaxy_data['V_gas'].values,
    galaxy_data['V_disk'].values,
    A0_KMS
)

# Run the optimizer!
result = minimize(chi_squared, initial_guess, args=args, method='Nelder-Mead')

# Extract the results
best_fit_upsilon = result.x[0]
min_chi_sq = result.fun

print(f"--- Fit Results for NGC 3198 ---")
print(f"Best-fit Upsilon_disk (Υ⋆): {best_fit_upsilon:.3f}")
print(f"Minimum Chi-Squared (χ²): {min_chi_sq:.2f}")

# Calculate the rotation curve with the BEST-FIT parameter
v_predicted_best_fit = calculate_pit_velocity(
    galaxy_data['Radius'].values,
    galaxy_data['V_gas'].values,
    galaxy_data['V_disk'].values,
    best_fit_upsilon, # Use the value we just found
    A0_KMS
)

# --- Plotting the Final Results ---
plt.figure(figsize=(10, 6))
plt.errorbar(
    galaxy_data['Radius'], galaxy_data['V_obs'], yerr=galaxy_data['Err_V_obs'],
    fmt='o', capsize=3, label='Observed Data (NGC 3198)'
)
plt.plot(galaxy_data['Radius'], v_predicted_best_fit, 'r-', linewidth=2, label=f'Best-Fit PIT Model (Υ⋆ = {best_fit_upsilon:.2f})')
plt.xlabel('Radius (kpc)')
plt.ylabel('Velocity (km/s)')
plt.title('Rotation Curve of NGC 3198 - Best-Fit PIT Model')
plt.legend()
plt.grid(True)
plt.show()
