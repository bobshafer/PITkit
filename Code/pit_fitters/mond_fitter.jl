#!/usr/bin/env julia

"""
MOND (Modified Newtonian Dynamics) Galaxy Rotation Curve Fitter
Tests if rotation curves follow emergent gravity modification
rather than additive dark matter
"""

using Optim
using DelimitedFiles
using Printf
using Statistics

# Physical constants
const G = 4.302e-6  # kpc (km/s)^2 / M_sun

# ============================================================================
# MOND INTERPOLATION FUNCTIONS
# ============================================================================

"""
Standard MOND interpolation function
Transitions from Newtonian (high a) to MOND (low a) regime
"""
function mond_mu_standard(x::Float64)
    # Standard interpolation: μ(x) = x / sqrt(1 + x^2)
    return x / sqrt(1 + x^2)
end

"""
Simple MOND interpolation function
"""
function mond_mu_simple(x::Float64)
    # Simple: μ(x) = x / (1 + x)
    return x / (1 + x)
end

"""
Compute MOND acceleration from Newtonian acceleration
a_MOND = μ(a_N/a0) * a_N
"""
function mond_acceleration(a_newton::Float64, a0::Float64, interp_func=mond_mu_standard)
    if a_newton <= 0
        return 0.0
    end
    
    x = a_newton / a0
    mu = interp_func(x)
    return mu * a_newton
end

# ============================================================================
# MOND ROTATION CURVE PREDICTION
# ============================================================================

"""
Predict rotation curve using MOND

Uses the "simple" MOND formula that's always well-behaved:
In deep-MOND: v^4 = G*M*a0 (independent of r for exponential disk!)
In Newtonian: v^2 = G*M/r

This uses the exact algebraic solution.
"""
function predict_mond_rotation(r::Vector{Float64}, 
                               v_baryon_unit::Vector{Float64},
                               params::Vector{Float64})
    a0, Upsilon_disk, sigma_int = params
    
    # Scale baryonic velocity by mass-to-light ratio
    v_baryon = Upsilon_disk .* v_baryon_unit
    
    # Use "simple" MOND formula (always positive):
    # v^4 = v_N^2 * (v_N^2 + sqrt(v_N^4 + 4*(a0*r)^2))
    # where v_N is the Newtonian (baryonic) velocity
    
    v_mond = zeros(length(r))
    
    for i in 1:length(r)
        if r[i] <= 0 || v_baryon[i] <= 0
            v_mond[i] = 0.0
            continue
        end
        
        v_N_sq = v_baryon[i]^2
        v_N_fourth = v_N_sq^2
        a0_r_sq = (a0 * r[i])^2
        
        # Algebraic formula (always gives positive result)
        discriminant = v_N_fourth + 4 * a0_r_sq
        v_fourth = v_N_sq * (v_N_sq + sqrt(discriminant))
        
        if v_fourth > 0
            v_mond[i] = v_fourth^0.25
        else
            v_mond[i] = v_baryon[i]  # Fallback to Newtonian
        end
    end
    
    # Add intrinsic scatter in quadrature to squared velocity
    v_mond_sq = v_mond.^2 .+ sigma_int^2
    
    return sqrt.(v_mond_sq)
end

# ============================================================================
# FITTING FUNCTIONS
# ============================================================================

"""
Chi-squared objective function
"""
function chi_squared_mond(params::Vector{Float64}, 
                         r::Vector{Float64},
                         v_obs::Vector{Float64},
                         v_err::Vector{Float64},
                         v_baryon_unit::Vector{Float64})
    
    # Predict velocities
    v_pred = predict_mond_rotation(r, v_baryon_unit, params)
    
    # Chi-squared
    residuals = (v_obs .- v_pred) ./ v_err
    chi2 = sum(residuals.^2)
    
    # Penalty for unphysical parameters
    a0, Upsilon_disk, sigma_int = params
    
    # a0 should be around 1e-10 to 1e-8 m/s^2 ≈ 3e-7 to 3e-5 (km/s)^2/kpc
    # Upsilon should be 0.1 to 5
    # sigma_int should be 0 to 50
    
    if a0 < 1e-8 || a0 > 1e-4 || Upsilon_disk < 0.1 || sigma_int < 0
        return 1e10
    end
    
    return chi2
end

"""
Fit single galaxy with MOND
"""
function fit_galaxy_mond(r::Vector{Float64},
                        v_obs::Vector{Float64},
                        v_err::Vector{Float64},
                        v_baryon_unit::Vector{Float64};
                        verbose::Bool=false)
    
    # Initial guess
    # a0 ≈ 1.2e-10 m/s^2 ≈ 3.7e-6 (km/s)^2/kpc (Milgrom's constant)
    # Convert: 1.2e-10 m/s^2 = 1.2e-10 * (km/s)^2 / (3.086e19 m) = 3.89e-6 (km/s)^2/kpc
    
    params_init = [4e-6, 0.5, 5.0]  # [a0, Upsilon_disk, sigma_int]
    
    # Bounds
    lower = [1e-7,  0.1, 0.0]   # Minimum physical values
    upper = [1e-4,  5.0, 50.0]  # Maximum reasonable values
    
    # Optimize
    result = optimize(
        p -> chi_squared_mond(p, r, v_obs, v_err, v_baryon_unit),
        lower,
        upper,
        params_init,
        Fminbox(LBFGS()),
        Optim.Options(
            iterations = 1000,
            show_trace = verbose,
            g_tol = 1e-6
        )
    )
    
    # Extract results
    params_best = Optim.minimizer(result)
    chi2_min = Optim.minimum(result)
    
    n_data = length(v_obs)
    n_params = length(params_best)
    dof = n_data - n_params
    chi2_reduced = chi2_min / dof
    
    return params_best, chi2_min, chi2_reduced, dof
end

# ============================================================================
# DATA LOADING (same as memory fitter)
# ============================================================================

"""
Load galaxy data from SPARC format
"""
function load_galaxy_data(filename::String)
    lines = readlines(filename)
    data_lines = filter(l -> !startswith(strip(l), "#") && !isempty(strip(l)), lines)
    
    data = []
    for line in data_lines
        parts = split(line)
        if length(parts) >= 4
            push!(data, parse.(Float64, parts[1:end]))
        end
    end
    
    if isempty(data)
        error("No valid data found in $filename")
    end
    
    data_matrix = hcat(data...)'
    
    r = data_matrix[:, 1]
    v_obs = data_matrix[:, 2]
    v_err = data_matrix[:, 3]
    
    # SPARC format: compute total baryonic velocity
    if size(data_matrix, 2) >= 6
        v_gas = data_matrix[:, 4]
        v_disk = data_matrix[:, 5]
        v_bul = data_matrix[:, 6]
        v_baryon_unit = sqrt.(v_gas.^2 .+ v_disk.^2 .+ v_bul.^2)
    elseif size(data_matrix, 2) >= 4
        v_baryon_unit = data_matrix[:, 4]
    else
        error("Insufficient columns in data file")
    end
    
    return r, v_obs, v_err, v_baryon_unit
end

# ============================================================================
# BATCH FITTING
# ============================================================================

"""
Fit all galaxies in directory with MOND
"""
function fit_all_galaxies_mond(data_dir::String; output_file::String="mond_fit_results.csv")
    
    files = filter(f -> endswith(f, ".dat") || endswith(f, ".csv") || endswith(f, ".txt"), 
                   readdir(data_dir))
    
    if isempty(files)
        println("No data files found in $data_dir")
        return
    end
    
    println("Found $(length(files)) galaxies to fit with MOND")
    println()
    
    results = []
    
    for (i, file) in enumerate(files)
        filepath = joinpath(data_dir, file)
        galaxy_name = replace(file, r"\.(dat|csv|txt)$" => "")
        
        try
            r, v_obs, v_err, v_baryon = load_galaxy_data(filepath)
            
            println("[$i/$(length(files))] Fitting $galaxy_name with MOND...")
            params, chi2, chi2_red, dof = fit_galaxy_mond(r, v_obs, v_err, v_baryon)
            
            a0, Upsilon_disk, sigma_int = params
            
            # Convert a0 to m/s^2 for comparison with Milgrom
            a0_SI = a0 / 3.086e16  # (km/s)^2/kpc to m/s^2
            
            @printf("  a0 = %.3e (km/s)²/kpc  [%.3e m/s²]\n", a0, a0_SI)
            @printf("  Υ_disk = %.3f\n", Upsilon_disk)
            @printf("  σ_int = %.2f km/s\n", sigma_int)
            @printf("  χ²/ν = %.2f (dof=%d)\n", chi2_red, dof)
            println()
            
            push!(results, (galaxy_name, a0, a0_SI, Upsilon_disk, sigma_int, chi2, chi2_red, dof))
            
        catch e
            println("  ERROR: Failed to fit $galaxy_name")
            println("  $e")
            println()
        end
    end
    
    # Save results
    if !isempty(results)
        open(output_file, "w") do io
            println(io, "galaxy,a0_kpc,a0_SI,Upsilon_disk,sigma_int,chi2,chi2_reduced,dof")
            for res in results
                println(io, join(res, ","))
            end
        end
        println("Results saved to: $output_file")
        
        # Summary statistics
        chi2_reds = [r[7] for r in results]
        a0_values = [r[3] for r in results]  # SI units
        sigma_ints = [r[5] for r in results]
        
        println("\n" * "="^70)
        println("MOND FIT SUMMARY")
        println("="^70)
        println("Number of galaxies: $(length(results))")
        @printf("Mean χ²/ν: %.2f\n", mean(chi2_reds))
        @printf("Median χ²/ν: %.2f\n", median(chi2_reds))
        @printf("Std χ²/ν: %.2f\n", std(chi2_reds))
        println()
        @printf("Mean a0: %.3e m/s²\n", mean(a0_values))
        @printf("Median a0: %.3e m/s²\n", median(a0_values))
        @printf("Milgrom's a0: 1.2e-10 m/s²\n")
        println()
        @printf("Mean σ_int: %.2f km/s\n", mean(sigma_ints))
        @printf("Median σ_int: %.2f km/s\n", median(sigma_ints))
        @printf("Fraction with σ_int > 45: %.1f%%\n", 100 * count(sigma_ints .> 45) / length(sigma_ints))
        println("="^70)
    end
end

# ============================================================================
# SINGLE GALAXY TEST
# ============================================================================

"""
Test MOND fit on single galaxy
"""
function test_single_galaxy_mond(filename::String)
    println("Testing MOND model on: $filename")
    println()
    
    r, v_obs, v_err, v_baryon = load_galaxy_data(filename)
    
    println("Data loaded: $(length(r)) data points")
    println("Radius range: $(minimum(r)) - $(maximum(r)) kpc")
    println()
    
    params, chi2, chi2_red, dof = fit_galaxy_mond(r, v_obs, v_err, v_baryon, verbose=true)
    
    a0, Upsilon_disk, sigma_int = params
    a0_SI = a0 / 3.086e16
    
    println("\n" * "="^70)
    println("MOND BEST FIT PARAMETERS")
    println("="^70)
    @printf("a0 (MOND scale)         : %.3e (km/s)²/kpc\n", a0)
    @printf("a0 (SI units)           : %.3e m/s²\n", a0_SI)
    @printf("Milgrom's a0            : 1.2e-10 m/s²\n")
    @printf("Υ_disk (M/L ratio)      : %.3f\n", Upsilon_disk)
    @printf("σ_int (scatter)         : %.2f km/s\n", sigma_int)
    println("-"^70)
    @printf("χ² total                : %.2f\n", chi2)
    @printf("χ²/ν (reduced)          : %.2f\n", chi2_red)
    @printf("Degrees of freedom      : %d\n", dof)
    println("="^70)
    
    if chi2_red < 2.0
        println("✓ EXCELLENT FIT - MOND matches data well")
    elseif chi2_red < 5.0
        println("⚠ ACCEPTABLE FIT - MOND captures main trends")
    else
        println("✗ POOR FIT - MOND does not match data")
    end
    
    # Save comparison
    v_pred = predict_mond_rotation(r, v_baryon, params)
    
    base_name = replace(basename(filename), r"\.(dat|csv|txt)$" => "")
    output_dir = dirname(filename)
    if isempty(output_dir)
        output_dir = "."
    end
    output_file = joinpath(output_dir, base_name * "_mond_fit.csv")
    
    open(output_file, "w") do io
        println(io, "r,v_obs,v_err,v_pred,v_baryon")
        for i in 1:length(r)
            v_bar = Upsilon_disk * v_baryon[i]
            println(io, "$(r[i]),$(v_obs[i]),$(v_err[i]),$(v_pred[i]),$v_bar")
        end
    end
    println("\nFit curve saved to: $output_file")
end

# ============================================================================
# MAIN
# ============================================================================

function main()
    if length(ARGS) < 1
        println("""
        MOND Fitter - Usage:
        
        Single galaxy:
          julia mond_fitter.jl <galaxy_file.dat>
          
        Batch mode:
          julia mond_fitter.jl --batch <data_directory> [output.csv]
          
        MOND predicts: v(r) from v_baryon(r) alone, no dark matter
        Key test: Does a0 cluster around Milgrom's value (1.2e-10 m/s²)?
        """)
        return
    end
    
    if ARGS[1] == "--batch"
        if length(ARGS) < 2
            println("Error: --batch requires data directory")
            return
        end
        data_dir = ARGS[2]
        output_file = length(ARGS) >= 3 ? ARGS[3] : "mond_fit_results.csv"
        fit_all_galaxies_mond(data_dir, output_file=output_file)
    else
        test_single_galaxy_mond(ARGS[1])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
