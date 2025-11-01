# pit_burkert_fitter.jl
#
# This is a new test. The previous k-space model (pit_corrected_fitter)
# was numerically unstable and could not be optimized.
#
# This script tests the SAME PHYSICAL HYPOTHESIS from Math.md:
# v_total^2 = v_baryon^2 + v_dark_halo^2
#
# It replaces the unstable k-space Hankel transform with a stable,
# analytical model for a "cored" dark matter halo (the Burkert profile),
# which represents the same physics (a non-singular "resonant node").
#
# FREE PARAMETERS (4):
#   rho_0 (Core Density), r_0 (Core Radius), Upsilon_disk, sigma_int
#
# --- Dependencies ---
using Printf
using Statistics
using DataFrames
using CSV
using SpecialFunctions
using Optim
using FiniteDifferences

# --- Physical Constants ---
const G_const = 4.30091e-6 # (kpc/M_sun) * (km/s)^2
const LARGE_FINITE_VAL = 1e20

# --- DATA LOADING (Unchanged) ---

struct GalaxyData
    name::String
    r::Vector{Float64}
    v_obs::Vector{Float64}
    v_err::Vector{Float64}
    v_gas::Vector{Float64}
    v_disk::Vector{Float64}
    # Derived quantities
    v_obs_sq::Vector{Float64}
    v_gas_sq::Vector{Float64}
    v_disk_sq::Vector{Float64}
end

function load_galaxy_data(data_dir::String="sparc_data/")
    all_galaxy_data = GalaxyData[]
    max_len = 0
    println("Loading data from: ", data_dir)
    if !isdir(data_dir)
        @warn "Data directory '$data_dir' not found. Cannot load data."
        return all_galaxy_data, max_len # Return empty list
    end

    files = filter(f -> endswith(f, ".dat"), readdir(data_dir))
    if isempty(files)
        @warn "No .dat files found in '$data_dir'. Cannot load data."
        return all_galaxy_data, max_len # Return empty list
    end
    println("Found $(length(files)) potential galaxy files.")

    skipped_count = 0
    processed_count = 0
    for filename in sort(files)
        filepath = joinpath(data_dir, filename)
        galaxy_name = replace(filename, ".dat" => "")
        try
            df = CSV.File(filepath; comment="#", header=false, delim=' ', ignorerepeated=true, types=Float64, silencewarnings=true) |> DataFrame

            if isempty(df) || size(df, 2) < 5
                skipped_count += 1; continue
            end

            data_matrix = Matrix{Float64}(df[:, 1:5])

            if !all(isfinite, data_matrix)
                skipped_count += 1; continue
            end

            if size(data_matrix, 1) < 2
                skipped_count += 1; continue
            end
            radii = data_matrix[:, 1]
            if !all(diff(radii) .> 1e-9)
                 skipped_count += 1; continue
            end
            v_errs = data_matrix[:, 3]
            if any(radii .<= 1e-9) || any(v_errs .<= 1e-9)
                 skipped_count += 1; continue
            end

            v_obs = data_matrix[:, 2]
            v_gas = data_matrix[:, 4]
            v_disk = data_matrix[:, 5]

            galaxy = GalaxyData(
                galaxy_name,
                radii, v_obs, v_errs, v_gas, v_disk,
                v_obs.^2, v_gas.^2, v_disk.^2
            )

            push!(all_galaxy_data, galaxy)
            max_len = max(max_len, length(radii))
            processed_count += 1

        catch e
            println("ERROR loading $filename: $e")
            skipped_count += 1
        end
    end

    if skipped_count > 0
        println("Skipped $skipped_count files due to issues.")
    end
    if processed_count == 0
         error("No valid galaxy data could be loaded.")
    end
    println("Loaded $processed_count galaxies. Max data points per galaxy: $max_len")
    return all_galaxy_data, max_len
end


# --- MODEL IMPLEMENTATION (Burkert Profile) ---

softplus(x::Real) = log1p(exp(x))
inv_softplus(y::Real) = log(max(expm1(y), 1e-15))

function transform_params(params_unconstrained::Vector{Float64})
    # This transforms the 4 free parameters
    phys_params = softplus.(params_unconstrained)
    return map(x -> (!isfinite(x) || x <= 0.0 ? 1e-12 : x), phys_params)
end

# --- Utility functions (trapz_rule, gradients, etc.) are NOT NEEDED ---
# We are using a simple analytical formula

function predict_galaxy_rotation_curve(
    params_phys_3::Vector{Float64}, # rho_0, r_0, Upsilon_disk
    r::Vector{Float64}, 
    v_gas_sq::Vector{Float64}, 
    v_disk_sq::Vector{Float64}
)
    # Get the 3 free parameters for this function
    rho_0 = max(params_phys_3[1], 1e-9)  # Core density
    r_0 = max(params_phys_3[2], 1e-9)    # Core radius
    Upsilon_disk = max(params_phys_3[3], 1e-3)

    # --- Baryon Component (Unchanged) ---
    v_baryon_sq = Upsilon_disk .* v_disk_sq .+ v_gas_sq
    v_baryon_sq = max.(v_baryon_sq, 1e-9)
    
    # --- K-Field (Info Matter) Component (Corrected) ---
    #
    # We now model v_info^2 directly using the Burkert profile.
    # This is numerically stable and represents a "cored" halo.
    # v_info^2(r) = (2 * pi * G * rho_0 * r_0^3 / r) * [ln(1 + (r/r_0)^2) + 2*ln(1 + r/r_0) - 2*atan(r/r_0)]
    
    C = 2.0 * pi * G_const * rho_0 * (r_0^3)
    r_safe = max.(r, 1e-9) # Avoid division by zero at r=0
    x = r ./ r_0 # x is the dimensionless radius
    
    # Calculate each term of the Burkert velocity formula
    # Note: log is natural log (ln) in Julia
    term1 = log.(1.0 .+ x.^2)
    term2 = 2.0 .* log.(1.0 .+ x)
    term3 = -2.0 .* atan.(x)
    
    # Sum the terms in the bracket
    bracket_term = term1 .+ term2 .+ term3
    
    # Calculate v_info^2
    v_info_sq = (C ./ r_safe) .* bracket_term
    v_info_sq = max.(v_info_sq, 0.0)
    v_info_sq = map(x -> !isfinite(x) ? 0.0 : min(x, LARGE_FINITE_VAL), v_info_sq)

    # --- Total Model Velocity ---
    # v_model^2 = v_baryon^2 + v_info^2
    v_model_sq = v_baryon_sq .+ v_info_sq
    v_model_sq = max.(v_model_sq, 1e-9)
    v_model_sq = map(x -> !isfinite(x) ? 1e6 : clamp(x, 0.0, 1e6), v_model_sq) # Clamp final

    return v_model_sq
end


# --- LOSS FUNCTION (Modified for 4 params) ---

const num_points_total_global = Ref(0) # Global counter

function objective_function(params_unconstrained::Vector{Float64}, all_galaxy_data::Vector{GalaxyData})
     
     # --- MODIFICATION: We now optimize 4 parameters ---
     params_phys = transform_params(params_unconstrained)
     if length(params_phys) != 4 # rho_0, r_0, Upsilon_disk, sigma_int
        return LARGE_FINITE_VAL * 1000.0
     end
     
     sigma_int = max(params_phys[4], 1e-9) # 4th param is sigma_int
     params_phys_3 = params_phys[1:3]      # 1-3 are for the predictor
     # --- END MODIFICATION ---

     total_chi_sq = 0.0
     current_num_points = 0

     for galaxy in all_galaxy_data
         r=galaxy.r; v_obs_sq=galaxy.v_obs_sq; v_err=galaxy.v_err; v_gas_sq=galaxy.v_gas_sq; v_disk_sq=galaxy.v_disk_sq
         if length(r) < 2 continue end
         
         # --- MODIFICATION: Pass the 3 free params to the prediction function ---
         v_model_sq = predict_galaxy_rotation_curve(params_phys_3, r, v_gas_sq, v_disk_sq)
         # --- END MODIFICATION ---

         v_err_safe = max.(v_err, 1e-9)
         total_err_sq = v_err_safe.^2 .+ sigma_int^2
         safe_total_err_sq = max.(total_err_sq, 1e-12)
         squared_diff = (v_obs_sq .- v_model_sq).^2
         squared_diff = min.(squared_diff, LARGE_FINITE_VAL^2)
         chi_sq_terms = squared_diff ./ safe_total_err_sq
         chi_sq_terms = map(x -> !isfinite(x) ? LARGE_FINITE_VAL : x, chi_sq_terms)
         total_chi_sq += sum(chi_sq_terms)
         current_num_points += length(r)
     end
     num_points_total_global[] = current_num_points
     return !isfinite(total_chi_sq) ? LARGE_FINITE_VAL * 100.0 : total_chi_sq
end


# --- MAIN EXECUTION (Modified for 4 params) ---

function main()
    data_directory = "sparc_data/"
    # We need more iterations for a real fit
    max_opt_iterations = 1000
    all_galaxy_data, max_len = load_galaxy_data(data_directory)
    if isempty(all_galaxy_data); println("No data loaded."); return; end

    # --- MODIFICATION: Initial guesses for 4 free parameters ---
    # rho_0 (Core Density, M_sun/kpc^3), r_0 (Core Radius, kpc), Upsilon_disk, sigma_int
    # These are standard starting guesses for a Burkert fit
    initial_guesses_phys = [0.01, 10.0, 0.5, 1.0] 
    println("--- RUNNING STABLE FITTER (Burkert Profile) ---")
    println("Testing Corrected Hypothesis (v_total^2 = v_baryon^2 + v_halo^2):")
    println("Optimizing 4 free parameters: rho_0, r_0, Upsilon_disk, sigma_int")
    println("Initial physical guess: ", initial_guesses_phys)
    # --- END MODIFICATION ---

    offset = 1e-12
    initial_guesses_phys_safe = max.(initial_guesses_phys, offset)
    initial_params_unconstrained = map(inv_softplus, initial_guesses_phys_safe)
    initial_params_unconstrained = map(x -> x == -Inf ? -40.0 : x, initial_params_unconstrained)
    initial_params_unconstrained = map(x -> !isfinite(x) ? 0.0 : x, initial_params_unconstrained)
    initial_params_unconstrained = convert(Vector{Float64}, initial_params_unconstrained)
    println("Initial unconstrained parameters (input to opt): ", initial_params_unconstrained)
    println("Mapped back to check: ", transform_params(initial_params_unconstrained))

    obj_func = params -> objective_function(params, all_galaxy_data)
    try
        init_loss = obj_func(initial_params_unconstrained)
        println("Initial loss (pre-optimization): ", init_loss)
        if !isfinite(init_loss) || init_loss >= LARGE_FINITE_VAL * 10; @error "Initial loss issue ($init_loss). Aborting."; return; end
    catch e
        @error "Error computing initial loss:"; showerror(stdout, e); return;
    end

    best_params_unconstrained = copy(initial_params_unconstrained)
    final_loss = NaN; opt_result = nothing
    try
        println("Starting optimization with LBFGS (using Optim's automatic gradient)...")
        start_time = time()

        opt_result = optimize(obj_func, initial_params_unconstrained, LBFGS(),
                              Optim.Options(iterations = max_opt_iterations,
                                            show_trace = true,
                                            f_tol = 1e-4, 
                                            g_tol = 1e-4
                                           ))

        end_time = time()
        println("Optimization finished in $(round(end_time - start_time, digits=2)) seconds.")

        if Optim.converged(opt_result); println("Optimization converged."); else; @warn "Optimization did NOT converge."; println("Reason: ", Optim.summary(opt_result)); end
        best_params_unconstrained = Optim.minimizer(opt_result)
        final_loss = Optim.minimum(opt_result)

    catch e
         @error "\n !!! An error occurred during optimization execution !!!";
         showerror(stdout, e); Base.show_backtrace(stdout, catch_backtrace())
         println("\n Attempting to report last valid state...");
         opt_result_defined = @isdefined opt_result
         if opt_result_defined && !isnothing(opt_result) && !isempty(Optim.minimizer(opt_result))
             best_params_unconstrained = Optim.minimizer(opt_result)
             final_loss = Optim.minimum(opt_result); @warn "Using parameters from failed/incomplete optimization."
         else
            best_params_unconstrained = initial_params_unconstrained
            final_loss = obj_func(initial_params_unconstrained); @warn "Optimization failed severely. Reporting initial parameters."
         end
    end

    # --- RESULTS (Modified) ---
    best_params_phys = transform_params(best_params_unconstrained)
    final_chi_sq = isfinite(final_loss) ? final_loss : NaN
    total_data_points = num_points_total_global[]
    
    # --- MODIFICATION: Number of parameters is now 4 ---
    num_parameters = 4
    # --- END MODIFICATION ---
    
    degrees_of_freedom = total_data_points > num_parameters ? total_data_points - num_parameters : 0
    reduced_chi_sq = NaN; if degrees_of_freedom > 0 && isfinite(final_chi_sq); reduced_chi_sq = final_chi_sq / degrees_of_freedom; end

    println("\n--- Fit Results (STABLE/Burkert FIT) ---");
    # --- MODIFICATION: Update param_names list ---
    param_names = ["rho_0 (free)", "r_0 (free)", "Upsilon_disk (free)", "sigma_int (free)"]
    # --- END MODACTION ---

    println("Best-fit physical parameters:"); if any(!isfinite, best_params_phys); println("  (Parameters contain NaN or Inf)"); end
    for (name, val) in zip(param_names, best_params_phys); if !isfinite(val); @printf "  %-20s: %s\n" name string(val); else; @printf "  %-20s: %.4e\n" name val; end; end
    
    println("\nBest-fit unconstrained parameters (log-like space):"); if any(!isfinite, best_params_unconstrained); println("  (Parameters contain NaN or Inf)"); end
    for (name, val) in zip(param_names, best_params_unconstrained); if !isfinite(val); @printf "  log_%-17s: %s\n" name string(val); else; @printf "  log_%-17s: %.4f\n" name val; end; end
    
    println("\nFit Statistics:")
    chi_sq_display = isfinite(final_chi_sq) ? @sprintf("%.4f", final_chi_sq) : string(final_chi_sq)
    println("  Final Total Chi-squared (χ²): ", chi_sq_display)
    println("  Total Number of Data Points (N): ", total_data_points)
    println("  Number of *Free* Parameters (p_free): ", num_parameters)
    println("  Degrees of Freedom (ν = N - p_free): ", degrees_of_freedom)
    reduced_chi_sq_display = isfinite(reduced_chi_sq) ? @sprintf("%.4f", reduced_chi_sq) : string(reduced_chi_sq)
    println("  Reduced Chi-squared (χ²/ν): ", reduced_chi_sq_display)
    println("----------------------------------\n")
end

# Execute main function
main()
