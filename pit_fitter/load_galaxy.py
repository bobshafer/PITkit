import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

# --- Constants and Data Loading (from previous script) ---
COLUMN_NAMES = [
    'Radius', 'V_obs', 'Err_V_obs', 'V_gas', 'V_disk', 'V_bulge', 
    'SB_disk', 'SB_bulge'
]
galaxy_file = 'data/NGC3198_rotmod.dat'
galaxy_data = pd.read_csv(
    galaxy_file, delim_whitespace=True, comment='#', header=None, names=COLUMN_NAMES
)

# --- The Core PIT Model ---

def calculate_pit_velocity(radius_kpc, v_gas, v_disk, upsilon_disk, alpha_max, a_0_kms):
    """
    Calculates the predicted rotation curve using the PIT/MOND framework.
    
    Args:
        radius_kpc (array): Radii in kpc.
        v_gas (array): Gas velocity component in km/s.
        v_disk (array): Stellar disk velocity template in km/s.
        upsilon_disk (float): Stellar mass-to-light ratio (a free parameter).
        alpha_max (float): PIT universal parameter (not used directly here, but part of a_0).
        a_0_kms (float): MOND acceleration constant in km^2/s^2/kpc.

    Returns:
        array: Predicted total velocity in km/s.
    """
    
    # 1. Calculate the total BARYONIC velocity contribution
    # V_baryon^2 = V_gas^2 + (Upsilon * V_disk^2)
    v_baryon_sq = v_gas**2 + upsilon_disk * v_disk**2
    
    # 2. Calculate the NEWTONIAN acceleration from baryons
    # g_N = V_baryon^2 / r
    # Handle the center (r=0) to avoid division by zero
    g_newtonian = np.divide(v_baryon_sq, radius_kpc, where=radius_kpc!=0)
    
    # --- MOND Interpolation Function ---
    def mu(x):
        return x / np.sqrt(1 + x**2)

    # 3. Solve for the 'a' at each radius
    total_acceleration = np.zeros_like(g_newtonian)
    for i, g_n_val in enumerate(g_newtonian):
        if g_n_val == 0:
            continue
        
        # We need to solve the equation: a * mu(a/a_0) - g_n = 0
        def equation_to_solve(a):
            return a * mu(a / a_0_kms) - g_n_val
            
        # Use a numerical root finder to solve for 'a'
        # We search for a root in a reasonable range, e.g., [0, 2*g_n_val]
        sol = root_scalar(equation_to_solve, bracket=[0, 2 * g_n_val + a_0_kms], method='brentq')
        total_acceleration[i] = sol.root

    # 4. Calculate the final predicted velocity
    # V_pred^2 = a * r
    v_pred_sq = total_acceleration * radius_kpc
    
    return np.sqrt(v_pred_sq)


# --- Example Usage ---

# Define some initial test parameters for PIT
# Note: a0 is ~1.2e-10 m/s^2. We need to convert this to galactic units.
# 1.2e-10 m/s^2 * (1 kpc / 3.086e19 m) * ( (1 s)^2 / (1 km/1000m)^2 ) * (3.154e7 s/yr)^2  -> this is messy!
# Let's use the known value in useful units: a0 ≈ 3700 (km/s)^2 / kpc
A0_KMS = 3700.0   # (km/s)^2 / kpc
ALPHA_MAX = 15.2   # From the original paper, for context
UPSILON_DISK = 0.5 # A typical stellar mass-to-light ratio

# Calculate the predicted rotation curve with our new function
v_predicted = calculate_pit_velocity(
    galaxy_data['Radius'].values,
    galaxy_data['V_gas'].values,
    galaxy_data['V_disk'].values,
    UPSILON_DISK,
    ALPHA_MAX,
    A0_KMS
)

# --- Plotting the Results ---

plt.figure(figsize=(10, 6))
# Plot the observed data
plt.errorbar(
    galaxy_data['Radius'], galaxy_data['V_obs'], yerr=galaxy_data['Err_V_obs'],
    fmt='o', capsize=3, label='Observed Data (NGC 3198)'
)
# Plot the predicted curve
plt.plot(galaxy_data['Radius'], v_predicted, 'r-', linewidth=2, label='PIT Model Prediction')
plt.xlabel('Radius (kpc)')
plt.ylabel('Velocity (km/s)')
plt.title('Rotation Curve of NGC 3198 - PIT Model')
plt.legend()
plt.grid(True)
plt.show()
