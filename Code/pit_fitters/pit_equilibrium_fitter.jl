# pit_equilibrium_fitter.jl
#
# This is the "Equilibrium Fitter".
# It directly implements the equilibrium solution derived from the
# pit_simulator.jl and Math.md.
#
# HYPOTHESIS:
# 1. K is the gravitational potential (Psi_info)
# 2. K is found by solving the equilibrium equation:
#    mu*K^2 + (beta-mu)*K - beta*F_Phi = 0
#    where F_Phi is a Gaussian blur of the baryon density (Phi).
# 3. v_info^2 = r * |dK/dr|
# 4. v_model^2 = v_baryon^2 + v_info^2
#
# FREE PARAMETERS (5):
#   mu, beta, kernel_sigma, Upsilon_disk, sigma_int
#
# --- Dependencies ---
# In Julia Pkg mode (press ']' in the REPL):
#   add ImageFiltering
#
using Printf
using Statistics
using DataFrames
using CSV
using SpecialFunctions
using Optim
using FiniteDifferences
using ImageFiltering # For Gaussian convolution (our F[Phi] operator)

# --- Physical Constants ---
const G_const = 4.30091e-6 # (kpc/M_sun) * (km/s)^2
const LARGE_FINITE_VAL = 1e20

# --- DATA LOADING (Corrected) ---

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
            # --- THIS LINE IS NOW CORRECTED ---
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


# --- MODEL IMPLEMENTATION (PIT EQUILIBRIUM) ---

softplus(x::Real) = log1p(exp(x))
inv_softplus(y::Real) = log(max(expm1(y), 1e-15))

function transform_params(params_unconstrained::Vector{Float64})
    # This transforms the 5 free parameters
    phys_params = softplus.(params_unconstrained)
    return map(x -> (!isfinite(x) || x <= 0.0 ? 1e-12 : x), phys_params)
end

function simple_gradient(y::AbstractVector{T}, x::AbstractVector{T}) where T<:Real
    n = length(x)
    if n < 2 || length(y) != n; return zeros(T, n); end
    grad = similar(y)
    dx1 = x[2] - x[1]; dy1 = y[2] - y[1]
    grad[1] = dy1 / (dx1 + eps(T))
    dx_n = x[n] - x[n-1]; dy_n = y[n] - y[n-1]
    grad[n] = dy_n / (dx_n + eps(T))
    if n > 2
        dx_c = @views x[3:n] .- x[1:n-2]
        dy_c = @views y[3:n] .- y[1:n-2]
        grad[2:n-1] .= dy_c ./ (dx_c .+ eps(T))
    end
    return map(val -> !isfinite(val) ? zero(T) : clamp(val, T(-LARGE_FINITE_VAL), T(LARGE_FINITE_VAL)), grad)
end


function predict_galaxy_rotation_curve(
    params_phys_4::Vector{Float64}, # mu, beta, kernel_sigma, Upsilon_disk
    r::Vector{Float64}, 
    v_gas_sq::Vector{Float64}, 
    v_disk_sq::Vector{Float64}
)
    # Get the 4 free parameters for this function
    mu = max(params_phys_4[1], 1e-9)
    beta = max(params_phys_4[2], 1e-9)
    kernel_sigma_kpc = max(params_phys_4[3], 1e-3) # Kernel width in kpc
    Upsilon_disk = max(params_phys_4[4], 1e-3)

    # --- 1. Baryon Component (Phi) ---
    v_baryon_sq = Upsilon_disk .* v_disk_sq .+ v_gas_sq
    v_baryon_sq = max.(v_baryon_sq, 1e-9)
    
    # We need the baryon *surface density* (Phi)
    M_b = r .* v_baryon_sq ./ G_const
    dMb_dr = simple_gradient(M_b, r)
    r_safe = max.(r, 1e-9)
    Phi = (1.0 ./ (2.0 * pi .* r_safe)) .* dMb_dr
    Phi = max.(Phi, 0.0) # Density cannot be negative
    Phi = map(x -> !isfinite(x) ? 0.0 : min(x, LARGE_FINITE_VAL), Phi)

    # --- 2. Blurred Baryon Component (F_Phi) ---
    # We convolve Phi with a Gaussian kernel
    
    # We need to define the kernel in *grid steps*, not kpc
    # Find the average grid spacing
    avg_dr = (r[end] - r[1]) / (length(r) - 1)
    kernel_sigma_steps = kernel_sigma_kpc / max(avg_dr, 1e-6)
    
    # Use a 1D Gaussian kernel
    # Use reflect boundary conditions to handle edges
    kernel = Kernel.gaussian(kernel_sigma_steps)
    F_Phi = imfilter(Phi, kernel, "reflect")
    F_Phi = max.(F_Phi, 0.0) # Blurred density also can't be negative
    
    # --- 3. Solve for K (The Potential) ---
    # mu*K^2 + (beta-mu)*K - beta*F_Phi = 0
    
    a = mu
    b = beta - mu
    
    # We need a new vector c = -beta * F_Phi
    c = -beta .* F_Phi
    
    # Calculate discriminant d = b^2 - 4*a*c
    # d = (beta - mu)^2 - 4*mu*(-beta*F_Phi)
    # d = (beta - mu)^2 + 4*mu*beta*F_Phi
    d = (b^2) .+ (4.0 * a .* beta .* F_Phi) # (b^2 is scalar, rest are vectors)
    d = max.(d, 0.0) # Discriminant can't be negative
    
    # K = (-b + sqrt(d)) / (2a)
    K_potential = (-b .+ sqrt.(d)) ./ (2.0 * a)
    K_potential = map(x -> !isfinite(x) ? 0.0 : x, K_potential)

    # --- 4. Calculate v_info^2 from K ---
    # v_info^2 = r * |dK/dr|
    
    dK_dr = simple_gradient(K_potential, r)
    v_info_sq = r_safe .* abs.(dK_dr)
    v_info_sq = map(x -> !isfinite(x) ? 0.0 : min(x, LARGE_FINITE_VAL), v_info_sq)

    # --- 5. Total Model Velocity ---
    v_model_sq = v_baryon_sq .+ v_info_sq
    v_model_sq = max.(v_model_sq, 1e-9)
    v_model_sq = map(x -> !isfinite(x) ? 1e6 : clamp(x, 0.0, 1e6), v_model_sq) # Clamp final

    return v_model_sq
end


# --- LOSS FUNCTION (Modified for 5 params) ---

const num_points_total_global = Ref(0) # Global counter

function objective_function(params_unconstrained::Vector{Float64}, all_galaxy_data::Vector{GalaxyData})
     
     params_phys = transform_params(params_unconstrained)
     if length(params_phys) != 5 # mu, beta, kernel_sigma, Upsilon_disk, sigma_int
        return LARGE_FINITE_VAL * 1000.0
     end
     
     sigma_int = max(params_phys[5], 1e-9) # 5th param is sigma_int
     params_phys_4 = params_phys[1:4]      # 1-4 are for the predictor

     total_chi_sq = 0.0
     current_num_points = 0

     for galaxy in all_galaxy_data
         r=galaxy.r; v_obs_sq=galaxy.v_obs_sq; v_err=galaxy.v_err; v_gas_sq=galaxy.v_gas_sq; v_disk_sq=galaxy.v_disk_sq
         if length(r) < 2 continue end
         
         v_model_sq = predict_galaxy_rotation_curve(params_phys_4, r, v_gas_sq, v_disk_sq)

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


# --- MAIN EXECUTION (Modified for 5 params) ---

function main()
    data_directory = "sparc_data/"
    max_opt_iterations = 1000
    all_galaxy_data, max_len = load_galaxy_data(data_directory)
    if isempty(all_galaxy_data); println("No data loaded."); return; end

    # --- Define Physical Boundaries ---
    # [mu, beta, kernel_sigma, Upsilon_disk, sigma_int]
    lower_bounds_phys = [1e-6, 1e-6, 0.1,  0.1, 0.1]
    upper_bounds_phys = [1.0,  1.0,  50.0, 2.0, 50.0]
    
    # --- Initial guesses for 5 free parameters ---
    # Use values from the simulator as a starting point
    initial_guesses_phys = [0.005, 0.04, 3.0, 0.5, 1.0] 
    
    println("--- RUNNING PIT EQUILIBRIUM FITTER ---")
    println("Testing Equilibrium Hypothesis (K=Potential, v^2=r*|dK/dr|):")
    println("Optimizing 5 free parameters with BOUNDARIES:")
    println("  mu: [$(lower_bounds_phys[1]), $(upper_bounds_phys[1])]")
    println("  beta: [$(lower_bounds_phys[2]), $(upper_bounds_phys[2])]")
    println("  kernel_sigma (kpc): [$(lower_bounds_phys[3]), $(upper_bounds_phys[3])]")
    println("  Upsilon_disk: [$(lower_bounds_phys[4]), $(upper_bounds_phys[4])]")
    println("  sigma_int: [$(lower_bounds_phys[5]), $(upper_bounds_phys[5])]")
    println("Initial physical guess: ", initial_guesses_phys)

    lower_bounds_unconstrained = map(inv_softplus, lower_bounds_phys)
    upper_bounds_unconstrained = map(inv_softplus, upper_bounds_phys)
    initial_params_unconstrained = map(inv_softplus, initial_guesses_phys)
    initial_params_unconstrained = max.(min.(initial_params_unconstrained, upper_bounds_unconstrained), lower_bounds_unconstrained)
    
    println("Initial unconstrained parameters (input to opt): ", initial_params_unconstrained)

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
        println("Starting optimization with Fminbox(LBFGS()) (Box-Constrained)...")
        start_time = time()

        opt_method = Fminbox(LBFGS())
        opt_result = optimize(obj_func, 
                              lower_bounds_unconstrained, 
                              upper_bounds_unconstrained, 
                              initial_params_unconstrained, 
                              opt_method,
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

    best_params_phys = transform_params(best_params_unconstrained)
    final_chi_sq = isfinite(final_loss) ? final_loss : NaN
    total_data_points = num_points_total_global[]
    
    num_parameters = 5
    
    degrees_of_freedom = total_data_points > num_parameters ? total_data_points - num_parameters : 0
    reduced_chi_sq = NaN; if degrees_of_freedom > 0 && isfinite(final_chi_sq); reduced_chi_sq = final_chi_sq / degrees_of_freedom; end

    println("\n--- Fit Results (PIT EQUILIBRIUM FIT) ---");
    param_names = ["mu (free)", "beta (free)", "kernel_sigma (free)", "Upsilon_disk (free)", "sigma_int (free)"]

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
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
