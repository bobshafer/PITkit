# pit_global_fitter.jl
# Implements a global fit for PIT Modal Kernel Model on SPARC galaxy data using Julia.

# --- Dependencies ---
# Add these packages in Julia's Pkg mode:
# ] add Printf Statistics DataFrames CSV SpecialFunctions Optim FiniteDifferences
# using Pkg; Pkg.add(["Printf", "Statistics", "DataFrames", "CSV", "SpecialFunctions", "Optim", "FiniteDifferences"]) # QuadGK removed

using Printf
using Statistics
using DataFrames
using CSV
using SpecialFunctions # For Bessel functions (besselj0)
# using QuadGK         # Stick to trapezoidal rule
using Optim          # For optimization algorithms (LBFGS)
# using Zygote         # Removing Zygote
using FiniteDifferences # Optim might use this internally if needed

# --- Physical Constants ---
const G_const = 4.30091e-6 # Gravitational constant in (kpc/M_sun) * (km/s)^2
const LARGE_FINITE_VAL = 1e20 # A large number to replace Inf/NaN penalties

# --- Data Loading ---

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

# Adapted data loading function (remains the same)
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
            # Read using CSV.File for robustness, skipping comments
            # Explicitly type columns that should be Float64
            df = CSV.File(filepath; comment="#", header=false, delim=' ', ignorerepeated=true, types=Float64, silencewarnings=true) |> DataFrame

            if isempty(df) || size(df, 2) < 5
                skipped_count += 1; continue
            end

            # Select first 5 columns and convert to Float64 matrix (already done by types=Float64)
            data_matrix = Matrix{Float64}(df[:, 1:5])

            # Check for NaN/Inf
            if !all(isfinite, data_matrix)
                skipped_count += 1; continue
            end

            # Check for sufficient points and strictly increasing radius
            if size(data_matrix, 1) < 2
                skipped_count += 1; continue
            end
            radii = data_matrix[:, 1]
            if !all(diff(radii) .> 1e-9) # Check for strictly increasing radius
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
            # Show stack trace for debugging file errors
            # Base.show_backtrace(stdout, catch_backtrace())
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


# --- Model Implementation ---

softplus(x::Real) = log1p(exp(x))
inv_softplus(y::Real) = log(max(expm1(y), 1e-15))

function transform_params(params_unconstrained::Vector{Float64})
    phys_params = softplus.(params_unconstrained)
    return map(x -> (!isfinite(x) || x <= 0.0 ? 1e-12 : x), phys_params)
end

function trapz_rule(y::AbstractVector{T}, x::AbstractVector{T}) where T<:Real
    len = length(x)
    if len < 2 || length(y) != len; return zero(T); end
    dx = diff(x)
    y_avg = @views (y[1:len-1] .+ y[2:len]) ./ T(2.0)
    integral = sum(y_avg .* dx)
    return isfinite(integral) ? integral : zero(T)
end

function compute_hankel_transform(r_grid::Vector{Float64}, sigma_r::Vector{Float64}, k::Float64)
    len = length(r_grid)
    if len < 2 || len != length(sigma_r); return 0.0; end
    bessel_vals = besselj0.(k .* r_grid)
    bessel_vals = map(x -> !isfinite(x) ? 0.0 : x, bessel_vals)
    integrand = r_grid .* sigma_r .* bessel_vals
    integrand = map(x -> !isfinite(x) ? 0.0 : clamp(x, -LARGE_FINITE_VAL, LARGE_FINITE_VAL), integrand)
    integral_val = trapz_rule(integrand, r_grid)
    return integral_val
end

function compute_hankel_transform_vec(r_grid::Vector{Float64}, sigma_r::Vector{Float64}, k_grid::Vector{Float64})
    return compute_hankel_transform.(Ref(r_grid), Ref(sigma_r), k_grid)
end

function compute_inverse_hankel_transform(k_grid::Vector{Float64}, rho_tilde_k::Vector{Float64}, r::Float64)
     len = length(k_grid)
     if len < 2 || len != length(rho_tilde_k); return 0.0; end
    bessel_vals = besselj0.(r .* k_grid)
    bessel_vals = map(x -> !isfinite(x) ? 0.0 : x, bessel_vals)
    integrand = k_grid .* rho_tilde_k .* bessel_vals
    integrand = map(x -> !isfinite(x) ? 0.0 : clamp(x, -LARGE_FINITE_VAL, LARGE_FINITE_VAL), integrand)
    integral_val = trapz_rule(integrand, k_grid)
    sigma_val = (1.0 / (2.0 * pi)) * integral_val
    return isfinite(sigma_val) ? sigma_val : 0.0
end

function compute_inverse_hankel_transform_vec(k_grid::Vector{Float64}, rho_tilde_k::Vector{Float64}, r_grid::Vector{Float64})
    return compute_inverse_hankel_transform.(Ref(k_grid), Ref(rho_tilde_k), r_grid)
end

function cumulative_trapezoid_rule(y::AbstractVector{T}, x::AbstractVector{T}) where T<:Real
    len = length(x)
    if len < 2 || length(y) != len; return zeros(T, len); end
    dx = diff(x)
    y_avg = (@views (y[1:len-1] .+ y[2:len])) ./ T(2.0)
    segment_areas = y_avg .* dx
    result = vcat(zero(T), cumsum(segment_areas))
    return map(v -> isfinite(v) ? v : zero(T), result)
end

# Simple finite difference gradient (more AD friendly than central_gradient loop)
function simple_gradient(y::AbstractVector{T}, x::AbstractVector{T}) where T<:Real
    n = length(x)
    if n < 2 || length(y) != n; return zeros(T, n); end
    grad = similar(y)
    # Forward difference for first point
    dx1 = x[2] - x[1]; dy1 = y[2] - y[1]
    grad[1] = dy1 / (dx1 + eps(T))
    # Backward difference for last point
    dx_n = x[n] - x[n-1]; dy_n = y[n] - y[n-1]
    grad[n] = dy_n / (dx_n + eps(T))
    # Central difference for interior points (vectorized)
    if n > 2
        dx_c = @views x[3:n] .- x[1:n-2]
        dy_c = @views y[3:n] .- y[1:n-2]
        grad[2:n-1] .= dy_c ./ (dx_c .+ eps(T))
    end
    return map(val -> !isfinite(val) ? zero(T) : clamp(val, T(-LARGE_FINITE_VAL), T(LARGE_FINITE_VAL)), grad)
end


function predict_galaxy_rotation_curve(params_phys_5::Vector{Float64}, r::Vector{Float64}, v_gas_sq::Vector{Float64}, v_disk_sq::Vector{Float64})
    K0 = max(params_phys_5[1], 1e-9); kc = max(params_phys_5[2], 1e-9); p = max(params_phys_5[3], 1e-3)
    mu_param = max(params_phys_5[4], 1e-15); Upsilon_disk = max(params_phys_5[5], 1e-3)

    v_baryon_sq = Upsilon_disk .* v_disk_sq .+ v_gas_sq; v_baryon_sq = max.(v_baryon_sq, 1e-9)
    M_b = r .* v_baryon_sq ./ G_const; M_b = map(x -> !isfinite(x) ? 0.0 : min(x, LARGE_FINITE_VAL), M_b)
    # Use simple_gradient instead of central_gradient
    dMb_dr = simple_gradient(M_b, r)
    r_safe_denom = max.(r, 1e-9)
    Sigma_b = (1.0 ./ (2.0 * pi .* r_safe_denom)) .* dMb_dr; Sigma_b = max.(Sigma_b, 0.0)
    Sigma_b = map(x -> !isfinite(x) ? 0.0 : min(x, LARGE_FINITE_VAL), Sigma_b)

    r_max_safe = max(maximum(r), 1e-6); k_min = max(1e-3 / r_max_safe, 1e-9)
    r_pos_or_large = map(ri -> ri > 1e-9 ? ri : LARGE_FINITE_VAL, r); min_r_pos = minimum(r_pos_or_large)
    min_r_pos_safe = max(min_r_pos >= LARGE_FINITE_VAL ? 1e-6 : min_r_pos, 1e-9)
    k_max = max(1e3 / min_r_pos_safe, k_min * 1.1); num_k = 100
    log10_k_min = log10(k_min); log10_k_max = log10(k_max)
    log10_k_min = !isfinite(log10_k_min) ? -9.0 : log10_k_min; log10_k_max = !isfinite(log10_k_max) ? -8.0 : log10_k_max
    log10_k_max = max(log10_k_max, log10_k_min + 1e-6)
    k_grid = 10.0 .^ range(log10_k_min, stop=log10_k_max, length=num_k); k_grid = max.(k_grid, 1e-9)
    k_grid = map(x -> !isfinite(x) ? 1e-9 : x, k_grid)

    Phi_tilde_k = compute_hankel_transform_vec(r, Sigma_b, k_grid)

    k_ratio = k_grid ./ kc; k_ratio_safe = max.(k_ratio, 0.0)
    exponent_val = map(x -> x^p, k_ratio_safe .+ 1e-15); exponent_val_clamped = min.(exponent_val, 100.0)
    W_k = K0 .* exp.(-exponent_val_clamped); W_k = map(x -> !isfinite(x) ? 0.0 : min(x, LARGE_FINITE_VAL), W_k)

    Phi_tilde_k_abs_sq = abs.(Phi_tilde_k).^2; Phi_tilde_k_abs_sq = map(x -> !isfinite(x) ? 0.0 : min(x, LARGE_FINITE_VAL), Phi_tilde_k_abs_sq)
    W_k_abs_sq = abs.(W_k).^2; W_k_abs_sq = map(x -> !isfinite(x) ? 0.0 : min(x, LARGE_FINITE_VAL), W_k_abs_sq)
    Rho_tilde_info_k = mu_param .* Phi_tilde_k_abs_sq .* W_k_abs_sq; Rho_tilde_info_k = max.(Rho_tilde_info_k, 0.0)
    Rho_tilde_info_k = map(x -> !isfinite(x) ? 0.0 : min(x, LARGE_FINITE_VAL), Rho_tilde_info_k)

    Sigma_info_r = compute_inverse_hankel_transform_vec(k_grid, Rho_tilde_info_k, r); Sigma_info_r = max.(Sigma_info_r, 0.0)
    Sigma_info_r = map(x -> !isfinite(x) ? 0.0 : min(x, LARGE_FINITE_VAL), Sigma_info_r)

    integrand_mass_info = 2.0 * pi .* r .* Sigma_info_r; integrand_mass_info = map(x -> !isfinite(x) ? 0.0 : clamp(x, -LARGE_FINITE_VAL, LARGE_FINITE_VAL), integrand_mass_info)
    M_info = cumulative_trapezoid_rule(integrand_mass_info, r); M_info = max.(M_info, 0.0)
    M_info = map(x -> !isfinite(x) ? 0.0 : min(x, LARGE_FINITE_VAL), M_info)

    v_info_sq = G_const .* M_info ./ r_safe_denom; v_info_sq = max.(v_info_sq, 0.0)
    v_info_sq = map(x -> !isfinite(x) ? 0.0 : min(x, LARGE_FINITE_VAL), v_info_sq)

    v_model_sq = v_baryon_sq .+ v_info_sq; v_model_sq = max.(v_model_sq, 1e-9)
    v_model_sq = map(x -> !isfinite(x) ? 1e6 : clamp(x, 0.0, 1e6), v_model_sq) # Clamp final

    return v_model_sq
end


# --- Loss Function (Objective for Optim.jl) ---

const num_points_total_global = Ref(0) # Global counter using Ref

function objective_function(params_unconstrained::Vector{Float64}, all_galaxy_data::Vector{GalaxyData})
     params_phys = transform_params(params_unconstrained)
     if length(params_phys) != 6; return LARGE_FINITE_VAL * 1000.0; end
     sigma_int = max(params_phys[6], 1e-9)
     total_chi_sq = 0.0; current_num_points = 0
     params_phys_5 = params_phys[1:5]

     for galaxy in all_galaxy_data
         r=galaxy.r; v_obs_sq=galaxy.v_obs_sq; v_err=galaxy.v_err; v_gas_sq=galaxy.v_gas_sq; v_disk_sq=galaxy.v_disk_sq
         if length(r) < 2 continue end
         v_model_sq = predict_galaxy_rotation_curve(params_phys_5, r, v_gas_sq, v_disk_sq)
         v_err_safe = max.(v_err, 1e-9)
         total_err_sq = v_err_safe.^2 .+ sigma_int^2
         safe_total_err_sq = max.(total_err_sq, 1e-12)
         squared_diff = (v_obs_sq .- v_model_sq).^2; squared_diff = min.(squared_diff, LARGE_FINITE_VAL^2)
         chi_sq_terms = squared_diff ./ safe_total_err_sq
         chi_sq_terms = map(x -> !isfinite(x) ? LARGE_FINITE_VAL : x, chi_sq_terms)
         total_chi_sq += sum(chi_sq_terms); current_num_points += length(r)
     end
     num_points_total_global[] = current_num_points # Update Ref correctly
     return !isfinite(total_chi_sq) ? LARGE_FINITE_VAL * 100.0 : total_chi_sq
end


# --- Main Execution ---

function main()
    data_directory = "sparc_data/"; max_opt_iterations = 100
    all_galaxy_data, max_len = load_galaxy_data(data_directory)
    if isempty(all_galaxy_data); println("No data loaded."); return; end

    initial_guesses_phys = [1.0, 0.1, 1.0, 0.01, 0.5, 1.0] # K0, kc, p, mu, Ups, sig
    println("Using initial physical guess: ", initial_guesses_phys)
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

    # --- Use Optim's built-in gradient calculation ---
    # Optim will use ForwardDiff if available, otherwise FiniteDifferences

    best_params_unconstrained = copy(initial_params_unconstrained); final_loss = NaN; opt_result = nothing
    try
        println("Starting optimization with LBFGS (using Optim's automatic gradient)...")
        start_time = time()

        # Call optimize without explicit gradient function `g!`
        opt_result = optimize(obj_func, initial_params_unconstrained, LBFGS(),
                              Optim.Options(iterations = max_opt_iterations,
                                            show_trace = true,
                                            f_tol = 1e-4, # Function tolerance (relative)
                                            g_tol = 1e-4  # Gradient norm tolerance (absolute)
                                           )) # Removed gradient function argument

        end_time = time()
        println("Optimization finished in $(round(end_time - start_time, digits=2)) seconds.")

        if Optim.converged(opt_result); println("Optimization converged."); else; @warn "Optimization did NOT converge."; println("Reason: ", Optim.summary(opt_result)); end
        best_params_unconstrained = Optim.minimizer(opt_result); final_loss = Optim.minimum(opt_result)

    catch e
         @error "\n !!! An error occurred during optimization execution !!!"; showerror(stdout, e); Base.show_backtrace(stdout, catch_backtrace())
         println("\n Attempting to report last valid state..."); opt_result_defined = @isdefined opt_result
         if opt_result_defined && !isnothing(opt_result) && !isempty(Optim.minimizer(opt_result))
             best_params_unconstrained = Optim.minimizer(opt_result); final_loss = Optim.minimum(opt_result); @warn "Using parameters from failed/incomplete optimization."
         else
            best_params_unconstrained = initial_params_unconstrained; final_loss = obj_func(initial_params_unconstrained); @warn "Optimization failed severely. Reporting initial parameters."
         end
    end

    # --- Results ---
    best_params_phys = transform_params(best_params_unconstrained); final_chi_sq = isfinite(final_loss) ? final_loss : NaN
    total_data_points = num_points_total_global[]; num_parameters = length(best_params_unconstrained)
    degrees_of_freedom = total_data_points > num_parameters ? total_data_points - num_parameters : 0
    reduced_chi_sq = NaN; if degrees_of_freedom > 0 && isfinite(final_chi_sq); reduced_chi_sq = final_chi_sq / degrees_of_freedom; end

    println("\n--- Fit Results ---"); param_names = ["K0", "kc", "p", "mu", "Upsilon_disk", "sigma_int"]
    println("Best-fit physical parameters:"); if any(!isfinite, best_params_phys); println("  (Parameters contain NaN or Inf)"); end
    for (name, val) in zip(param_names, best_params_phys); if !isfinite(val); @printf "  %-15s: %s\n" name string(val); else; @printf "  %-15s: %.4e\n" name val; end; end
    println("\nBest-fit unconstrained parameters (log-like space):"); if any(!isfinite, best_params_unconstrained); println("  (Parameters contain NaN or Inf)"); end
    for (name, val) in zip(param_names, best_params_unconstrained); if !isfinite(val); @printf "  log_%-12s: %s\n" name string(val); else; @printf "  log_%-12s: %.4f\n" name val; end; end
    println("\nFit Statistics:")
    chi_sq_display = isfinite(final_chi_sq) ? @sprintf("%.4f", final_chi_sq) : string(final_chi_sq); println("  Final Total Chi-squared (χ²): ", chi_sq_display)
    println("  Total Number of Data Points (N): ", total_data_points); println("  Number of Parameters (p): ", num_parameters); println("  Degrees of Freedom (ν = N - p): ", degrees_of_freedom)
    reduced_chi_sq_display = isfinite(reduced_chi_sq) ? @sprintf("%.4f", reduced_chi_sq) : string(reduced_chi_sq); println("  Reduced Chi-squared (χ²/ν): ", reduced_chi_sq_display)
    println("------------------\n")
end

# Execute main function
main()


