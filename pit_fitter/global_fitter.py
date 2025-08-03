import os
import pandas as pd
import numpy as np
from scipy.optimize import root_scalar, minimize
import time

# --- Constants and Data Loading ---
COLUMN_NAMES = [
    'Radius', 'V_obs', 'Err_V_obs', 'V_gas', 'V_disk', 'V_bulge', 
    'SB_disk', 'SB_bulge'
]
DATA_DIR = 'data/'

# --- The Core PIT Model (Unchanged) ---
def calculate_pit_velocity(radius_kpc, v_gas, v_disk, upsilon_disk, a_0_kms):
    """Calculates the predicted rotation curve using the PIT/MOND framework."""
    # This function remains the same as in the previous script.
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
        try:
            sol = root_scalar(equation_to_solve, bracket=[0, 2 * g_n_val + a_0_kms], method='brentq')
            total_acceleration[i] = sol.root
        except ValueError:
            # If the root is not in the bracket, it can happen for edge cases.
            # We'll mark this as a failure by returning a large number.
            return np.inf

    v_pred_sq = total_acceleration * radius_kpc
    return np.sqrt(v_pred_sq)

# --- The Local Fitter (for a single galaxy) ---
def fit_single_galaxy(galaxy_data, fixed_a0):
    """
    Finds the best-fit Upsilon_disk for a single galaxy, given a fixed a0.
    Returns the minimum chi-squared found for that galaxy.
    """
    
    # Chi-squared function for the LOCAL fit (fitting for Upsilon only)
    def local_chi_squared(params, radius, v_obs, err_v_obs, v_gas, v_disk, a_0):
        upsilon_disk = params[0]
        v_predicted = calculate_pit_velocity(radius, v_gas, v_disk, upsilon_disk, a_0)
        
        # If the velocity calculation failed, return a massive chi-squared
        if np.isinf(v_predicted).any():
            return 1e99

        chi_sq = np.sum(((v_obs - v_predicted) / err_v_obs)**2)
        return chi_sq

    initial_guess_upsilon = [0.5]
    bounds_upsilon = [(0.01, 4.0)] # Upsilon should be positive and not extreme
    
    args = (
        galaxy_data['Radius'].values,
        galaxy_data['V_obs'].values,
        galaxy_data['Err_V_obs'].values,
        galaxy_data['V_gas'].values,
        galaxy_data['V_disk'].values,
        fixed_a0 # a0 is FIXED for this local fit
    )

    result = minimize(local_chi_squared, initial_guess_upsilon, args=args, method='L-BFGS-B', bounds=bounds_upsilon)
    
    return result.fun # Return the minimum chi-squared found

# --- The Global Cost Function (for the entire sample) ---
def global_chi_squared(global_params, all_galaxy_files):
    """
    Calculates the TOTAL chi-squared for the entire galaxy sample
    for a given universal value of a0.
    """
    universal_a0 = global_params[0]
    total_chi_sq = 0
    galaxies_processed = 0
    
    print(f"Testing universal a₀ = {universal_a0:.2f} ...")

    for filename in all_galaxy_files:
        try:
            filepath = os.path.join(DATA_DIR, filename)
            galaxy_data = pd.read_csv(filepath, delim_whitespace=True, comment='#', header=None, names=COLUMN_NAMES)
            
            # For each galaxy, find its best possible chi-squared using this universal_a0
            min_chi_for_galaxy = fit_single_galaxy(galaxy_data, universal_a0)
            
            total_chi_sq += min_chi_for_galaxy
            galaxies_processed += 1
        except Exception as e:
            print(f"  Skipping {filename} due to an error: {e}")
            
    print(f"  -> Total χ² for this a₀: {total_chi_sq:.2f} across {galaxies_processed} galaxies.")
    return total_chi_sq

# --- Main Execution ---
if __name__ == "__main__":
    start_time = time.time()
    
    # Get a list of all galaxy data files
    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith('_rotmod.dat')]
    
    print("--- Starting the Global Fit for the Universal Acceleration Constant a₀ ---")
    
    # Initial guess and bounds for the GLOBAL parameter, a0
    initial_guess_a0 = [2400.0] 
    bounds_a0 = [(500, 5000)] # Search in a plausible range around the literature value

    # Run the global optimizer!
    global_result = minimize(
        global_chi_squared, 
        initial_guess_a0, 
        args=(all_files,), 
        method='L-BFGS-B', 
        bounds=bounds_a0,
        options={'eps': 100.0} # Use a larger step size for the gradient calculation
    )

    end_time = time.time()
    
    # --- Print the Final Results ---
    best_universal_a0 = global_result.x[0]
    final_total_chi_sq = global_result.fun
    
    print("\n--- GLOBAL FIT COMPLETE ---")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Best-fit UNIVERSAL Acceleration (a₀): {best_universal_a0:.2f} (km/s)²/kpc")
    print(f"Final Total Chi-Squared (χ²) for the entire sample: {final_total_chi_sq:.2f}")
