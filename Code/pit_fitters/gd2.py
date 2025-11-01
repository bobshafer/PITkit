# pit_global_fitter.py
# Implements a global fit for PIT Modal Kernel Model on SPARC galaxy data using JAX.
# Uses Gradient Descent optimizer and float64 precision.

import os
import jax
# --- FIX: Enable float64 precision ---
from jax import config
config.update("jax_enable_x64", True)
# --- END FIX ---
import jax.numpy as jnp
import jax.tree_util as jtu
# Use jax.pure_callback to wrap non-JAX functions
from jax import pure_callback
# Import trapezoid integration function with robust fallbacks
try:
    from jax.scipy.integrate import trapezoid
except ImportError:
    try:
        from jax.numpy import trapz as trapezoid
    except ImportError:
        print("Warning: Using jnp-based fallback for trapezoid integration.")
        @jax.jit
        def trapezoid(y, x=None, axis=-1):
            """A jnp-based trapezoid integrator that supports an axis argument."""
            y = jnp.asarray(y)
            if x is None:
                dx = 1.0
            else:
                x = jnp.asarray(x)
                if x.ndim == 1:
                    dx = jnp.diff(x)
                    shape = [1] * y.ndim
                    shape[axis] = -1
                    dx = dx.reshape(shape)
                else:
                     dx = jnp.diff(x, axis=axis)
            slice1 = [slice(None)] * y.ndim
            slice2 = [slice(None)] * y.ndim
            slice1[axis] = slice(1, None)
            slice2[axis] = slice(None, -1)
            y_sum = y[tuple(slice1)] + y[tuple(slice2)]
            integral = jnp.sum(y_sum / 2.0 * dx, axis=axis)
            return integral

import jaxopt
import scipy.special
import numpy as np
import time

# --- Physical Constants ---
G_const = 4.30091e-6 # (kpc/M_sun) * (km/s)^2
LARGE_FINITE_VAL = 1e20 # Reduced large finite value (float64 compatible)

# --- Model Implementation ---

def scipy_j0(x):
    """Calculates J0 using SciPy's jv, with NaN/inf handling."""
    # SciPy often works best with float64
    x_np = np.array(x, dtype=np.float64)
    with np.errstate(invalid='ignore'):
        j0_np = scipy.special.jv(0, x_np)
    j0_np = np.nan_to_num(j0_np, nan=0.0, posinf=0.0, neginf=0.0)
    # Return as float64 since JAX is now using float64
    return j0_np.astype(np.float64)

def j0_abstract_eval(x_shape_dtype):
    """Abstract evaluation function for j0 for pure_callback."""
    shape = getattr(x_shape_dtype, 'shape', None)
    if shape is None and isinstance(x_shape_dtype, jax.ShapeDtypeStruct):
        shape = x_shape_dtype.shape
    if shape is None:
        raise TypeError(f"Unsupported abstract value type: {type(x_shape_dtype)}")
    # Return abstract shape with float64 dtype
    return jax.core.ShapedArray(shape, np.float64)


def compute_hankel_transform(r_grid, sigma_r, k_grid):
    """Computes forward Hankel transform. Uses pure_callback for J0."""
    # Ensure inputs are float64
    r_grid_f64 = r_grid.astype(jnp.float64)
    k_grid_f64 = k_grid.astype(jnp.float64)
    sigma_r_f64 = sigma_r.astype(jnp.float64)

    kr_product = jnp.outer(k_grid_f64, r_grid_f64)
    abstract_output = jax.ShapeDtypeStruct(kr_product.shape, jnp.float64) # Expect float64
    j0_values = pure_callback(scipy_j0, abstract_output, kr_product, vmap_method='legacy_vectorized')

    j0_values = jnp.nan_to_num(j0_values, nan=0.0, posinf=0.0, neginf=0.0)
    sigma_r_safe = jnp.nan_to_num(sigma_r_f64, nan=0.0, posinf=LARGE_FINITE_VAL, neginf=0.0)
    integrand = r_grid_f64 * sigma_r_safe * j0_values
    integrand = jnp.nan_to_num(integrand, nan=0.0, posinf=LARGE_FINITE_VAL, neginf=-LARGE_FINITE_VAL)

    phi_tilde_k = trapezoid(integrand, r_grid_f64, axis=1)
    phi_tilde_k = jnp.nan_to_num(phi_tilde_k, nan=0.0, posinf=LARGE_FINITE_VAL, neginf=-LARGE_FINITE_VAL)
    return phi_tilde_k.astype(jnp.float64) # Ensure output is float64

def compute_inverse_hankel_transform(k_grid, rho_tilde_k, r_grid):
    """Computes inverse Hankel transform. Uses pure_callback for J0."""
    # Ensure inputs are float64
    r_grid_f64 = r_grid.astype(jnp.float64)
    k_grid_f64 = k_grid.astype(jnp.float64)
    rho_tilde_k_f64 = rho_tilde_k.astype(jnp.float64)

    kr_product = jnp.outer(r_grid_f64, k_grid_f64)
    abstract_output = jax.ShapeDtypeStruct(kr_product.shape, jnp.float64) # Expect float64
    j0_values = pure_callback(scipy_j0, abstract_output, kr_product, vmap_method='legacy_vectorized')

    j0_values = jnp.nan_to_num(j0_values, nan=0.0, posinf=0.0, neginf=0.0)
    rho_tilde_k_safe = jnp.nan_to_num(rho_tilde_k_f64, nan=0.0, posinf=LARGE_FINITE_VAL, neginf=0.0)
    integrand = k_grid_f64 * rho_tilde_k_safe * j0_values
    integrand = jnp.nan_to_num(integrand, nan=0.0, posinf=LARGE_FINITE_VAL, neginf=-LARGE_FINITE_VAL)

    sigma_r = (1.0 / (2.0 * jnp.pi)) * trapezoid(integrand, k_grid_f64, axis=1)
    sigma_r = jnp.nan_to_num(sigma_r, nan=0.0, posinf=LARGE_FINITE_VAL, neginf=0.0)
    return sigma_r.astype(jnp.float64) # Ensure output is float64

@jax.jit
def cumulative_trapezoid_jax(y, x):
    """Computes cumulative trapezoid rule using JAX primitives."""
    y = jnp.asarray(y, dtype=jnp.float64); x = jnp.asarray(x, dtype=jnp.float64) # Use float64
    y_shape = jnp.shape(y); x_shape = jnp.shape(x)
    if len(x_shape) != 1 or len(y_shape) != 1 or x_shape[0] != y_shape[0]:
        return jnp.zeros_like(y, dtype=jnp.float64) # Return float64
    size = x_shape[0]
    dx = jnp.diff(x)
    segment_areas = (y[:-1] + y[1:]) / 2.0 * dx
    cumulative_integral = jnp.where(
        size < 2, jnp.zeros_like(y, dtype=jnp.float64), # Return float64
        jnp.concatenate((jnp.array([0.0], dtype=jnp.float64), jnp.cumsum(segment_areas))) # Use float64
    )
    return cumulative_integral


def predict_galaxy_rotation_curve(params_phys, galaxy_data):
    """Predicts rotation curve with safeguards, using float64."""
    # Ensure params are float64
    params_phys_f64 = jnp.asarray(params_phys, dtype=jnp.float64)
    K0, kc, p, mu, Upsilon_disk, sigma_int = params_phys_f64

    # Ensure data arrays are float64
    r = galaxy_data['r'].astype(jnp.float64)
    v_gas_sq = galaxy_data['v_gas_sq'].astype(jnp.float64)
    v_disk_sq = galaxy_data['v_disk_sq'].astype(jnp.float64)

    K0=jnp.maximum(K0,1e-9); kc=jnp.maximum(kc,1e-9); p=jnp.maximum(p,1e-3)
    mu=jnp.maximum(mu,1e-15); Upsilon_disk=jnp.maximum(Upsilon_disk,1e-3)

    v_baryon_sq = Upsilon_disk * v_disk_sq + v_gas_sq
    v_baryon_sq = jnp.maximum(v_baryon_sq, 1e-9)
    M_b = r * v_baryon_sq / G_const
    M_b = jnp.nan_to_num(M_b, nan=0.0, posinf=LARGE_FINITE_VAL, neginf=0.0)

    r_float = r # Already float64
    epsilon = 1e-12 # Can use smaller epsilon with float64
    r_offset = epsilon * jnp.arange(r_float.shape[0], dtype=jnp.float64)
    r_for_grad = r_float + r_offset

    dMb_dr = jnp.where(jnp.shape(r_float)[0]<2, jnp.zeros_like(M_b), jnp.gradient(M_b, r_for_grad))
    dMb_dr = jnp.nan_to_num(dMb_dr, nan=0.0, posinf=LARGE_FINITE_VAL, neginf=-LARGE_FINITE_VAL)
    r_safe_denom = jnp.maximum(r, 1e-12) # Use smaller floor
    Sigma_b = (1.0 / (2.0 * jnp.pi * r_safe_denom)) * dMb_dr
    Sigma_b = jnp.maximum(Sigma_b, 0.0); Sigma_b = jnp.nan_to_num(Sigma_b, nan=0.0, posinf=LARGE_FINITE_VAL)

    r_max_safe = jnp.maximum(jnp.max(r), 1e-6)
    k_min = 1e-3 / r_max_safe; k_min = jnp.maximum(k_min, 1e-12) # Smaller floor
    r_pos_or_large = jnp.where(r > 1e-12, r, LARGE_FINITE_VAL) # Smaller floor
    min_r_pos = jnp.min(r_pos_or_large)
    min_r_pos_safe = jnp.maximum(jnp.where(min_r_pos >= LARGE_FINITE_VAL, 1e-6, min_r_pos), 1e-12) # Smaller floor
    k_max = 1e3 / min_r_pos_safe; k_max = jnp.maximum(k_max, k_min * 1.1); num_k = 100
    log10_k_min = jnp.log10(k_min); log10_k_max = jnp.log10(k_max)
    log10_k_min = jnp.nan_to_num(log10_k_min, nan=-12.0); log10_k_max = jnp.nan_to_num(log10_k_max, nan=-11.0) # Adjust nan replacements
    log10_k_max = jnp.maximum(log10_k_max, log10_k_min + 1e-6)
    k_grid = jnp.logspace(log10_k_min, log10_k_max, num_k, dtype=jnp.float64) # Use float64
    k_grid = jnp.maximum(k_grid, 1e-12); k_grid = jnp.nan_to_num(k_grid, nan=1e-12) # Smaller floor

    Phi_tilde_k = compute_hankel_transform(r, Sigma_b, k_grid) # Pass float64 r

    k_ratio = k_grid / kc; k_ratio_safe = jnp.maximum(k_ratio, 0.0)
    exponent_val = k_ratio_safe**p; exponent_val_clamped = jnp.minimum(exponent_val, 100.0)
    W_k = K0 * jnp.exp(-exponent_val_clamped); W_k = jnp.nan_to_num(W_k, nan=0.0, posinf=LARGE_FINITE_VAL)

    Phi_tilde_k_abs_sq = jnp.abs(Phi_tilde_k)**2; Phi_tilde_k_abs_sq = jnp.nan_to_num(Phi_tilde_k_abs_sq, nan=0.0, posinf=LARGE_FINITE_VAL)
    W_k_abs_sq = jnp.abs(W_k)**2; W_k_abs_sq = jnp.nan_to_num(W_k_abs_sq, nan=0.0, posinf=LARGE_FINITE_VAL)
    Rho_tilde_info_k = mu * Phi_tilde_k_abs_sq * W_k_abs_sq
    Rho_tilde_info_k = jnp.maximum(Rho_tilde_info_k, 0.0); Rho_tilde_info_k = jnp.nan_to_num(Rho_tilde_info_k, nan=0.0, posinf=LARGE_FINITE_VAL)

    Sigma_info_r = compute_inverse_hankel_transform(k_grid, Rho_tilde_info_k, r) # Pass float64 r
    Sigma_info_r = jnp.maximum(Sigma_info_r, 0.0); Sigma_info_r = jnp.nan_to_num(Sigma_info_r, nan=0.0, posinf=LARGE_FINITE_VAL)

    integrand_mass_info = 2.0 * jnp.pi * r * Sigma_info_r
    integrand_mass_info = jnp.nan_to_num(integrand_mass_info, nan=0.0, posinf=LARGE_FINITE_VAL, neginf=0.0)
    M_info = cumulative_trapezoid_jax(integrand_mass_info, r) # Pass float64 r
    M_info = jnp.maximum(M_info, 0.0); M_info = jnp.nan_to_num(M_info, nan=0.0, posinf=LARGE_FINITE_VAL)

    v_info_sq = G_const * M_info / r_safe_denom
    v_info_sq = jnp.maximum(v_info_sq, 0.0); v_info_sq = jnp.nan_to_num(v_info_sq, nan=0.0, posinf=LARGE_FINITE_VAL)

    v_model_sq = v_baryon_sq + v_info_sq
    v_model_sq = jnp.maximum(v_model_sq, 1e-9)
    v_model_sq = jnp.nan_to_num(v_model_sq, nan=0.0, posinf=LARGE_FINITE_VAL, neginf=0.0)
    v_model_sq = jnp.minimum(v_model_sq, 1e6) # Keep aggressive clamp

    return v_model_sq.astype(jnp.float64) # Ensure output is float64

# --- Loss Function ---

def transform_params(params_unconstrained):
    # Ensure input is float64 for softplus consistency
    params_unconstrained_f64 = jnp.asarray(params_unconstrained, dtype=jnp.float64)
    transformed = jax.nn.softplus(params_unconstrained_f64)
    transformed = jnp.nan_to_num(transformed, nan=1e-12) # Smaller floor
    return transformed.astype(jnp.float64) # Ensure output is float64

@jax.jit
def galaxy_loss_internal(params_unconstrained, galaxy_data, mask):
    # Ensure inputs within JIT are expected dtype (float64)
    params_unconstrained_f64 = jnp.asarray(params_unconstrained, dtype=jnp.float64)
    galaxy_data_f64 = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), galaxy_data)
    mask_bool = jnp.asarray(mask, dtype=bool)

    params_phys = transform_params(params_unconstrained_f64)
    sigma_int = params_phys[5]; sigma_int_safe = jnp.maximum(sigma_int, 1e-12) # Smaller floor
    v_obs_sq = galaxy_data_f64['v_obs_sq']; v_err = galaxy_data_f64['v_err']
    v_model_sq = predict_galaxy_rotation_curve(params_phys, galaxy_data_f64)

    v_err_safe = jnp.maximum(v_err, 1e-12) # Smaller floor
    total_err_sq = v_err_safe**2 + sigma_int_safe**2
    safe_total_err_sq = jnp.maximum(total_err_sq, 1e-24) # Smaller floor
    squared_diff = (v_obs_sq - v_model_sq)**2
    squared_diff = jnp.minimum(squared_diff, LARGE_FINITE_VAL**2)
    chi_sq_terms = squared_diff / safe_total_err_sq
    chi_sq_terms = jnp.nan_to_num(chi_sq_terms, nan=0.0, posinf=LARGE_FINITE_VAL, neginf=0.0)
    masked_chi_sq_terms = jnp.where(mask_bool, chi_sq_terms, 0.0)
    chi_sq = jnp.sum(masked_chi_sq_terms)
    chi_sq = jnp.nan_to_num(chi_sq, nan=LARGE_FINITE_VAL*10, posinf=LARGE_FINITE_VAL*10, neginf=-LARGE_FINITE_VAL*10)
    return chi_sq.astype(jnp.float64) # Ensure output is float64

vmapped_loss = jax.vmap(galaxy_loss_internal, in_axes=(None, 0, 0))

@jax.jit
def global_loss(params_unconstrained, all_galaxy_data, all_masks):
    # Ensure inputs are float64 where appropriate
    params_unconstrained_f64 = jnp.asarray(params_unconstrained, dtype=jnp.float64)
    # Assume all_galaxy_data leaves will be converted inside vmap call if needed by galaxy_loss_internal
    # all_masks is boolean, dtype doesn't need changing
    losses = vmapped_loss(params_unconstrained_f64, all_galaxy_data, all_masks)
    total_loss = jnp.sum(losses)
    total_loss = jnp.nan_to_num(total_loss, nan=LARGE_FINITE_VAL*100, posinf=LARGE_FINITE_VAL*100, neginf=-LARGE_FINITE_VAL*100)
    return total_loss.astype(jnp.float64) # Ensure output is float64

# --- Data Loading and Preprocessing ---
def load_galaxy_data(data_dir):
    all_data = []; max_len = 0; galaxy_names = []
    print(f"Loading data from: {data_dir}")
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory '{data_dir}' not found."); return create_dummy_data()
    files = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
    if not files:
        print(f"Warning: No .dat files found in '{data_dir}'."); return create_dummy_data()
    print(f"Found {len(files)} galaxy files.")
    skipped_count = 0
    for filename in sorted(files):
        filepath = os.path.join(data_dir, filename)
        galaxy_name = filename.replace('.dat', '')
        try:
            # Load as float64 from the start
            raw_data = np.genfromtxt(filepath, comments='#', invalid_raise=False, loose=True, skip_header=0, dtype=np.float64)
            if raw_data is None or raw_data.size == 0: skipped_count += 1; continue
            if raw_data.ndim == 1:
                if raw_data.shape[0] >= 5: raw_data = raw_data.reshape(1, -1)
                else: skipped_count += 1; continue
            if raw_data.shape[1] < 5: skipped_count += 1; continue
            data = raw_data[:, :5]
            # No need to cast again if loaded as float64
            if np.any(~np.isfinite(data)): skipped_count += 1; continue
            if data.shape[0] < 2: skipped_count += 1; continue
            if np.any(np.diff(data[:, 0]) <= 1e-12): # Use smaller tolerance for float64
                skipped_count += 1; continue
            galaxy_dict = {'name': galaxy_name, 'r': data[:, 0], 'v_obs': data[:, 1],
                           'v_err': data[:, 2], 'v_gas': data[:, 3], 'v_disk': data[:, 4]}
            if np.any(galaxy_dict['r'] <= 1e-12) or np.any(galaxy_dict['v_err'] <= 1e-12): # Smaller tolerance
                 skipped_count += 1; continue
            # Calculate squares using float64
            galaxy_dict['v_obs_sq'] = galaxy_dict['v_obs']**2
            galaxy_dict['v_gas_sq'] = galaxy_dict['v_gas']**2
            galaxy_dict['v_disk_sq'] = galaxy_dict['v_disk']**2
            all_data.append(galaxy_dict); max_len = max(max_len, data.shape[0]); galaxy_names.append(galaxy_name)
        except Exception as e: print(f"ERROR loading {filename}: {e}"); skipped_count += 1
    if skipped_count > 0: print(f"Skipped {skipped_count} files (including those with issues).")
    if not all_data: raise ValueError("No valid galaxy data loaded.")
    all_data = [g for g in all_data if len(g['r']) > 0]
    if not all_data: raise ValueError("No valid galaxies remaining after filtering.")
    max_len = max(len(g['r']) for g in all_data)
    galaxy_names = [g['name'] for g in all_data]
    print(f"Loaded {len(all_data)} galaxies after filtering. Max data points per galaxy: {max_len}")
    return all_data, max_len, galaxy_names

def create_dummy_data(num_galaxies=5, num_points=50):
    all_data = []; max_len = num_points; galaxy_names = []
    np.random.seed(42)
    for i in range(num_galaxies):
        r = np.linspace(0.1, 10.0, num_points, dtype=np.float64) # Use float64
        v_true = 100 * (1 - np.exp(-r/2.0))
        v_obs = (v_true + np.random.normal(0, 5, num_points)).astype(np.float64)
        v_err = np.abs(np.random.normal(5, 1, num_points) + 2.0).astype(np.float64)
        v_gas = np.abs(20 * np.exp(-r/5.0) + np.random.normal(0, 2, num_points)).astype(np.float64)
        v_disk = np.abs(80 * (1 - np.exp(-r/1.5)) + np.random.normal(0, 4, num_points)).astype(np.float64)
        v_err = np.maximum(v_err, 1e-9) # Use smaller floor
        galaxy_dict = {'name': f"Dummy_{i+1}", 'r': r, 'v_obs': v_obs, 'v_err': v_err,
                       'v_gas': v_gas, 'v_disk': v_disk, 'v_obs_sq': v_obs**2,
                       'v_gas_sq': v_gas**2, 'v_disk_sq': v_disk**2}
        all_data.append(galaxy_dict); galaxy_names.append(f"Dummy_{i+1}")
    print(f"Created {len(all_data)} dummy galaxies with {max_len} data points each.")
    return all_data, max_len, galaxy_names

def pad_and_stack_data(all_galaxy_data, max_len):
    padded_data = {}; masks = []
    keys_to_pad = ['r', 'v_gas_sq', 'v_disk_sq', 'v_obs_sq', 'v_err']
    for key in keys_to_pad:
        padded_arrays = []
        for i, galaxy in enumerate(all_galaxy_data):
            arr = galaxy.get(key); gal_name = galaxy.get('name', f'index {i}')
            if arr is None: raise KeyError(f"Key '{key}' missing for {gal_name}")
            arr = np.asarray(arr).astype(np.float64); current_len = len(arr) # Use float64
            pad_width = max_len - current_len
            if pad_width < 0: raise ValueError(f"pad_width negative ({pad_width}) for key '{key}' in {gal_name}. max_len={max_len}, arr_len={current_len}")
            padded_arr = np.pad(arr, (0, pad_width), mode='constant', constant_values=0.0)
            padded_arrays.append(padded_arr)
        try:
             stacked_np_array = np.stack(padded_arrays)
             stacked_array = jnp.array(stacked_np_array, dtype=jnp.float64) # Use float64
             padded_data[key] = stacked_array
        except ValueError as e:
             print(f"Error stacking '{key}': {e}")
             for i, p_arr in enumerate(padded_arrays): print(f" Padded shape {i}: {p_arr.shape}")
             raise
    for galaxy in all_galaxy_data:
        original_len = len(galaxy['r']); mask = np.zeros(max_len, dtype=bool)
        mask[:original_len] = True; masks.append(mask)
    masks = jnp.array(np.stack(masks)) # Mask remains boolean
    return padded_data, masks

# --- Optimization ---

def run_optimization(initial_params_unconstrained, all_galaxy_data, all_masks, maxiter=500):
    loss_fn_for_opt = global_loss
    # --- Switch back to LBFGS for potentially faster convergence if stable ---
    optimizer = jaxopt.LBFGS(fun=loss_fn_for_opt, maxiter=maxiter, tol=1e-5,
                             history_size=20, verbose=True, jit=True,
                             implicit_diff=False, stepsize=-1.0)
    # ---
    # optimizer = jaxopt.GradientDescent(fun=loss_fn_for_opt, maxiter=maxiter, tol=1e-5,
    #                                    stepsize=1e-15, verbose=True, jit=True, # May need even smaller stepsize
    #                                    acceleration=False)
    print(f"Starting optimization with {optimizer.__class__.__name__}...")
    start_time = time.time()
    # Ensure data passed is float64
    all_galaxy_data_jax = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), all_galaxy_data)
    all_masks_jax = jnp.asarray(all_masks) # Boolean is fine

    sol, state = optimizer.run(init_params=initial_params_unconstrained.astype(jnp.float64), # Ensure initial params are float64
                               all_galaxy_data=all_galaxy_data_jax,
                               all_masks=all_masks_jax)

    end_time = time.time()
    print(f"Optimization finished in {end_time - start_time:.2f} seconds.")

    # Check state based on optimizer type
    best_params_unconstrained = sol # LBFGS also returns params in sol
    if hasattr(state, 'error') and state.error is not None:
         if np.isnan(state.error): print("Warning: Optimization failed - Optimizer error is NaN.")
         elif state.error > optimizer.tol: print(f"Warning: Opt might not have converged. Final optimizer error: {state.error:.2e} > tol {optimizer.tol:.1e}")
         else: print(f"Optimization converged. Final optimizer error: {state.error:.2e}")
    elif hasattr(state, 'iter_num') and state.iter_num is not None:
         if state.iter_num >= maxiter: print(f"Optimization stopped after reaching max iterations ({state.iter_num}).")
         else: print(f"Optimization stopped after {state.iter_num} iterations.")
    else: print("Optimization finished (final state details unavailable).")

    return best_params_unconstrained, state

# --- Main Execution ---

if __name__ == "__main__":
    data_directory = './sparc_data/'
    max_opt_iterations = 100

    # --- Disable NaN/Inf debugging ---
    # jax.config.update("jax_debug_nans", False)
    # jax.config.update("jax_debug_infs", False)
    # ---

    # Load and preprocess data
    try:
        # Load data with float64 precision
        raw_galaxy_data, max_length, galaxy_names = load_galaxy_data(data_directory)
        for i, gal in enumerate(raw_galaxy_data): gal['name'] = galaxy_names[i]
        stacked_galaxy_data, stacked_masks = pad_and_stack_data(raw_galaxy_data, max_length)
    except Exception as e:
        print(f"Error during data loading/processing: {e}")
        import traceback; traceback.print_exc(); exit(1)

    # --- Use More Moderate Goldilocks-inspired guesses ---
    initial_guesses_phys = jnp.array([1.0, 0.1, 1.0, 0.01, 0.5, 1.0], dtype=jnp.float64) # Use float64
    print(f"Using MODIFIED Goldilocks-inspired initial guess: {initial_guesses_phys}")
    # ---

    offset = 1e-12 # Use smaller offset for float64
    initial_guesses_phys_safe = jnp.maximum(initial_guesses_phys, offset)
    initial_params_unconstrained = jnp.log(jnp.expm1(initial_guesses_phys_safe))
    initial_params_unconstrained = jnp.where(initial_params_unconstrained == -jnp.inf, -40.0, initial_params_unconstrained) # Adjust floor for float64
    initial_params_unconstrained = jnp.nan_to_num(initial_params_unconstrained, nan=0.0)
    initial_params_unconstrained = initial_params_unconstrained.astype(jnp.float64) # Ensure float64

    print("Initial physical parameters guess (target):", initial_guesses_phys)
    print("Initial unconstrained parameters (input to opt):", initial_params_unconstrained)
    print("Mapped back to check:", transform_params(initial_params_unconstrained))

    try:
        # Ensure data is float64 before putting on device
        stacked_galaxy_data_jax = jtu.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), stacked_galaxy_data)
        stacked_masks_jax = jnp.asarray(stacked_masks) # Boolean ok
        initial_params_unconstrained_dev = jax.device_put(initial_params_unconstrained)
        stacked_galaxy_data_dev = jax.device_put(stacked_galaxy_data_jax)
        stacked_masks_dev = jax.device_put(stacked_masks_jax)

        # Calculate initial loss with float64
        init_loss = global_loss(initial_params_unconstrained_dev, stacked_galaxy_data_dev, stacked_masks_dev)
        init_loss.block_until_ready()
        print(f"Initial loss (pre-optimization): {init_loss}")

        if jnp.isnan(init_loss) or jnp.isinf(init_loss) or init_loss >= LARGE_FINITE_VAL*10:
             print(f"ERROR: Initial loss is {init_loss}. Aborting optimization.")
             exit(1)

    except Exception as e:
        print(f"Error computing initial loss diagnostics: {e}")
        import traceback; traceback.print_exc(); exit(1)

    best_params_unconstrained = initial_params_unconstrained_dev # Initialize with initial params
    final_state = None
    try:
        best_params_unconstrained, final_state = run_optimization(
            initial_params_unconstrained_dev,
            stacked_galaxy_data_dev,
            stacked_masks_dev,
            maxiter=max_opt_iterations
        )
    except Exception as e:
        print(f"An error occurred during optimization: {e}")
        import traceback; traceback.print_exc();
        if jnp.any(jnp.isnan(best_params_unconstrained)):
             print("Setting best_params to NaN due to optimization failure.")
             best_params_unconstrained = jnp.full_like(initial_params_unconstrained_dev, jnp.nan)

    # Ensure final params are float64 before transform
    best_params_unconstrained = best_params_unconstrained.astype(jnp.float64)
    best_params_phys = transform_params(best_params_unconstrained)
    final_chi_sq = float('nan')
    if not jnp.any(jnp.isnan(best_params_unconstrained)):
        try:
             final_chi_sq_verify = global_loss(best_params_unconstrained, stacked_galaxy_data_dev, stacked_masks_dev)
             final_chi_sq_verify.block_until_ready()
             if not jnp.isnan(final_chi_sq_verify) and not jnp.isinf(final_chi_sq_verify):
                 final_chi_sq = float(final_chi_sq_verify)
             else: print("Warning: Final loss calculated is NaN or Inf.")
        except Exception as e: print(f"Error calculating final loss: {e}")

    if np.isnan(final_chi_sq) or np.isinf(final_chi_sq):
         print("Attempting fallback to optimizer state for final loss...")
         if final_state is not None:
              state_value = getattr(final_state, 'value', getattr(final_state, 'error', None))
              if state_value is not None:
                  state_value_float = float(state_value)
                  if not np.isnan(state_value_float) and not np.isinf(state_value_float):
                      final_chi_sq = state_value_float
                      print(f"Using final loss from optimizer state: {final_chi_sq:.4f}")
                  else: print("Warning: Optimizer state value is NaN or Inf.")
              else: print("Warning: Could not get valid loss from optimizer state ('value' or 'error').")
         else: print("Warning: No final optimizer state available for fallback.")

    total_data_points = int(jnp.sum(stacked_masks))
    num_parameters = len(initial_params_unconstrained)
    degrees_of_freedom = total_data_points - num_parameters
    reduced_chi_sq = float('nan')
    if degrees_of_freedom > 0 and not np.isnan(final_chi_sq) and not np.isinf(final_chi_sq):
        reduced_chi_sq = final_chi_sq / degrees_of_freedom
    elif degrees_of_freedom <= 0: print(f"Warning: DoF <= 0.")

    print("\n--- Fit Results ---")
    param_names = ['K0', 'kc', 'p', 'mu', 'Upsilon_disk', 'sigma_int']
    print("Best-fit physical parameters:")
    best_params_phys_np = np.array(best_params_phys)
    for name, val in zip(param_names, best_params_phys_np):
        val_display = f"{val:.4e}" if not np.isnan(val) else "NaN"
        print(f"  {name:<15}: {val_display}")

    print("\nBest-fit unconstrained parameters (log-like space):")
    best_params_unconstrained_np = np.array(best_params_unconstrained)
    for name, val in zip(param_names, best_params_unconstrained_np):
        val_display = f"{val:.4f}" if not np.isnan(val) else "NaN"
        print(f"  log_{name:<12}: {val_display}")

    print("\nFit Statistics:")
    chi_sq_display = f"{final_chi_sq:.4f}" if not np.isnan(final_chi_sq) and not np.isinf(final_chi_sq) else "NaN" if np.isnan(final_chi_sq) else "Inf"
    print(f"  Final Total Chi-squared (χ²): {chi_sq_display}")
    print(f"  Total Number of Data Points (N): {total_data_points}")
    print(f"  Number of Parameters (p): {num_parameters}")
    print(f"  Degrees of Freedom (ν = N - p): {degrees_of_freedom}")

    if np.isnan(reduced_chi_sq): red_chi_sq_display = "NaN"
    elif np.isinf(reduced_chi_sq): red_chi_sq_display = "Inf"
    else: red_chi_sq_display = f"{reduced_chi_sq:.4f}"
    print(f"  Reduced Chi-squared (χ²/ν): {red_chi_sq_display}")

    print("------------------\n")


