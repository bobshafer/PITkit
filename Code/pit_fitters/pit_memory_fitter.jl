#!/usr/bin/env julia

"""
PIT Memory Halo Galaxy Rotation Curve Fitter
Models dark matter as accumulated gravitational memory (K field)
that is a smoothed, time-integrated version of baryon distribution
"""

using Optim
using DelimitedFiles
using Printf
using Statistics
using LinearAlgebra

# Physical constants
const G = 4.302e-6  # kpc (km/s)^2 / M_sun
const KPC_TO_KM = 3.086e16  # meters

# ============================================================================
# MEMORY HALO MODEL
# ============================================================================

"""
Compute memory density profile
K accumulates where Φ (baryons) has been, smoothed by Interface Operator
"""
function memory_density(r::Float64, r_disk::Float64, sigma_memory::Float64, K0::Float64)
    # Effective memory kernel - combines disk scale with memory smoothing
    effective_scale = r_disk + sigma_memory
    
    # Memory decays from center, with characteristic scale
    # The (1 + (r/sigma_memory)^2) term prevents over-concentration
    central_term = exp(-r / effective_scale)
    smoothing_term = 1.0 / (1.0 + (r / sigma_memory)^2)
    
    # Total memory density (arbitrary units, will be scaled by K0)
    rho_memory = K0 * central_term * smoothing_term
    
    return rho_memory
end

"""
Compute cumulative mass from density profile
Using trapezoidal integration
"""
function cumulative_mass(r_points::Vector{Float64}, rho::Vector{Float64})
    n = length(r_points)
    M = zeros(n)
    
    for i in 2:n
        # Shell mass: 4π r² ρ(r) dr
        dr = r_points[i] - r_points[i-1]
        r_mid = (r_points[i] + r_points[i-1]) / 2
        rho_mid = (rho[i] + rho[i-1]) / 2
        
        shell_mass = 4π * r_mid^2 * rho_mid * dr
        M[i] = M[i-1] + shell_mass
    end
    
    return M
end

"""
Estimate disk scale length from galaxy data
Simple heuristic: scale where baryon velocity peaks
"""
function estimate_disk_scale(r::Vector{Float64}, v_baryon::Vector{Float64})
    # Find peak of baryon rotation curve
    peak_idx = argmax(v_baryon)
    
    if peak_idx == 1 || peak_idx == length(v_baryon)
        # Fallback: use median radius
        return median(r)
    end
    
    r_disk = r[peak_idx]
    
    # Ensure reasonable bounds (3-15 kpc for typical galaxies)
    r_disk = clamp(r_disk, 3.0, 15.0)
    
    return r_disk
end

# ============================================================================
# ROTATION CURVE PREDICTION
# ============================================================================

"""
Predict rotation curve using PIT Memory Halo model

Parameters:
- K0: Memory field strength (controls halo normalization)
- sigma_memory: Memory coherence scale (kpc) - the σ from simulator
- Upsilon_disk: Stellar mass-to-light ratio
- sigma_int: Intrinsic scatter (km/s)

Galaxy data should contain:
- r: radii (kpc)
- v_baryon: baryonic circular velocity (km/s) at unit Upsilon
"""
function predict_memory_halo_rotation(r::Vector{Float64}, 
                                     v_baryon_unit::Vector{Float64},
                                     params::Vector{Float64})
    K0, sigma_memory, Upsilon_disk, sigma_int = params
    
    # Baryonic contribution (scaled by Upsilon)
    v_baryon_sq = (Upsilon_disk .* v_baryon_unit).^2
    
    # Estimate disk scale from baryonic profile
    v_baryon = sqrt.(v_baryon_sq)
    r_disk = estimate_disk_scale(r, v_baryon)
    
    # Compute memory density at each radius
    rho_memory = [memory_density(ri, r_disk, sigma_memory, K0) for ri in r]
    
    # Integrate to get enclosed memory mass
    M_memory = cumulative_mass(r, rho_memory)
    
    # Circular velocity from memory (v² = GM/r)
    v_memory_sq = G .* M_memory ./ r
    
    # Total velocity (additive: v²_total = v²_baryon + v²_memory)
    v_total_sq = v_baryon_sq .+ v_memory_sq
    
    # Add intrinsic scatter in quadrature
    v_total_sq .+= sigma_int^2
    
    return sqrt.(v_total_sq)
end

# ============================================================================
# FITTING FUNCTIONS
# ============================================================================

"""
Objective function for optimization (chi-squared)
"""
function chi_squared(params::Vector{Float64}, 
                    r::Vector{Float64},
                    v_obs::Vector{Float64},
                    v_err::Vector{Float64},
                    v_baryon_unit::Vector{Float64})
    
    # Predict velocities
    v_pred = predict_memory_halo_rotation(r, v_baryon_unit, params)
    
    # Chi-squared
    residuals = (v_obs .- v_pred) ./ v_err
    chi2 = sum(residuals.^2)
    
    # Penalty for unphysical parameters
    K0, sigma_memory, Upsilon_disk, sigma_int = params
    
    if K0 < 0 || sigma_memory < 0.5 || Upsilon_disk < 0.1 || sigma_int < 0
        return 1e10
    end
    
    return chi2
end

"""
Fit single galaxy
"""
function fit_galaxy(r::Vector{Float64},
                   v_obs::Vector{Float64},
                   v_err::Vector{Float64},
                   v_baryon_unit::Vector{Float64};
                   verbose::Bool=false)
    
    # Initial guess (reasonable values)
    # K0: ~1e8 M_sun/kpc^3 (typical halo density scale)
    # sigma_memory: ~5 kpc (from simulator σ≈3-5 grid units)
    # Upsilon_disk: ~0.5 (typical stellar M/L)
    # sigma_int: ~5 km/s (typical scatter)
    
    params_init = [1e8, 5.0, 0.5, 5.0]
    
    # Bounds
    lower = [1e6,  0.5,  0.1, 0.0]  # Minimum physical values
    upper = [1e11, 50.0, 5.0, 50.0] # Maximum reasonable values
    
    # Optimize
    result = optimize(
        p -> chi_squared(p, r, v_obs, v_err, v_baryon_unit),
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
# DATA LOADING
# ============================================================================

"""
Load galaxy data from file
Handles both SPARC .dat format and custom CSV format
"""
function load_galaxy_data(filename::String)
    # Read file, skipping comment lines starting with #
    lines = readlines(filename)
    data_lines = filter(l -> !startswith(strip(l), "#") && !isempty(strip(l)), lines)
    
    # Parse data
    data = []
    for line in data_lines
        # Split on whitespace or comma
        parts = split(line)
        if length(parts) >= 4
            push!(data, parse.(Float64, parts[1:end]))
        end
    end
    
    if isempty(data)
        error("No valid data found in $filename")
    end
    
    # Convert to matrix
    data_matrix = hcat(data...)'
    
    # Extract columns (SPARC format: Rad Vobs errV Vgas Vdisk Vbul ...)
    r = data_matrix[:, 1]
    v_obs = data_matrix[:, 2]
    v_err = data_matrix[:, 3]
    
    # For SPARC data: v_baryon = sqrt(v_gas^2 + v_disk^2 + v_bulge^2)
    if size(data_matrix, 2) >= 6
        v_gas = data_matrix[:, 4]
        v_disk = data_matrix[:, 5]
        v_bul = data_matrix[:, 6]
        v_baryon_unit = sqrt.(v_gas.^2 .+ v_disk.^2 .+ v_bul.^2)
    elseif size(data_matrix, 2) >= 4
        # Custom format with v_baryon already provided
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
Fit all galaxies in a directory
"""
function fit_all_galaxies(data_dir::String; output_file::String="memory_fit_results.csv")
    
    # Find all data files (.dat, .csv, .txt)
    files = filter(f -> endswith(f, ".dat") || endswith(f, ".csv") || endswith(f, ".txt"), 
                   readdir(data_dir))
    
    if isempty(files)
        println("No data files found in $data_dir")
        println("Looking for: .dat, .csv, .txt files")
        return
    end
    
    println("Found $(length(files)) galaxies to fit")
    println()
    
    # Results storage
    results = []
    
    for (i, file) in enumerate(files)
        filepath = joinpath(data_dir, file)
        galaxy_name = replace(file, r"\.(dat|csv|txt)$" => "")
        
        try
            # Load data
            r, v_obs, v_err, v_baryon = load_galaxy_data(filepath)
            
            # Fit
            println("[$i/$(length(files))] Fitting $galaxy_name...")
            params, chi2, chi2_red, dof = fit_galaxy(r, v_obs, v_err, v_baryon)
            
            K0, sigma_memory, Upsilon_disk, sigma_int = params
            
            # Print results
            @printf("  K0 = %.3e M_sun/kpc^3\n", K0)
            @printf("  σ_memory = %.2f kpc\n", sigma_memory)
            @printf("  Υ_disk = %.3f\n", Upsilon_disk)
            @printf("  σ_int = %.2f km/s\n", sigma_int)
            @printf("  χ²/ν = %.2f (dof=%d)\n", chi2_red, dof)
            println()
            
            push!(results, (galaxy_name, K0, sigma_memory, Upsilon_disk, sigma_int, chi2, chi2_red, dof))
            
        catch e
            println("  ERROR: Failed to fit $galaxy_name")
            println("  $e")
            println()
        end
    end
    
    # Save results
    if !isempty(results)
        open(output_file, "w") do io
            println(io, "galaxy,K0,sigma_memory,Upsilon_disk,sigma_int,chi2,chi2_reduced,dof")
            for res in results
                println(io, join(res, ","))
            end
        end
        println("Results saved to: $output_file")
        
        # Summary statistics
        chi2_reds = [r[7] for r in results]
        println("\n" * "="^70)
        println("SUMMARY STATISTICS")
        println("="^70)
        println("Number of galaxies: $(length(results))")
        @printf("Mean χ²/ν: %.2f\n", mean(chi2_reds))
        @printf("Median χ²/ν: %.2f\n", median(chi2_reds))
        @printf("Std χ²/ν: %.2f\n", std(chi2_reds))
        println("="^70)
    end
end

# ============================================================================
# SINGLE GALAXY TEST
# ============================================================================

"""
Test fit on a single galaxy with visualization
"""
function test_single_galaxy(filename::String)
    println("Testing Memory Halo model on: $filename")
    println()
    
    # Load data
    r, v_obs, v_err, v_baryon = load_galaxy_data(filename)
    
    println("Data loaded: $(length(r)) data points")
    println("Radius range: $(minimum(r)) - $(maximum(r)) kpc")
    println()
    
    # Fit
    params, chi2, chi2_red, dof = fit_galaxy(r, v_obs, v_err, v_baryon, verbose=true)
    
    K0, sigma_memory, Upsilon_disk, sigma_int = params
    
    # Results
    println("\n" * "="^70)
    println("BEST FIT PARAMETERS")
    println("="^70)
    @printf("K0 (memory strength)    : %.3e M_sun/kpc^3\n", K0)
    @printf("σ_memory (coherence)    : %.2f kpc\n", sigma_memory)
    @printf("Υ_disk (M/L ratio)      : %.3f\n", Upsilon_disk)
    @printf("σ_int (scatter)         : %.2f km/s\n", sigma_int)
    println("-"^70)
    @printf("χ² total                : %.2f\n", chi2)
    @printf("χ²/ν (reduced)          : %.2f\n", chi2_red)
    @printf("Degrees of freedom      : %d\n", dof)
    println("="^70)
    
    # Quality assessment
    if chi2_red < 2.0
        println("✓ EXCELLENT FIT - Model matches data well")
    elseif chi2_red < 5.0
        println("⚠ ACCEPTABLE FIT - Model captures main trends")
    else
        println("✗ POOR FIT - Model does not match data")
    end
    
    # Predict curve for output
    v_pred = predict_memory_halo_rotation(r, v_baryon, params)
    
    # Save comparison (ensure we don't overwrite original)
    base_name = replace(basename(filename), r"\.(dat|csv|txt)$" => "")
    output_dir = dirname(filename)
    if isempty(output_dir)
        output_dir = "."
    end
    output_file = joinpath(output_dir, base_name * "_memory_fit.csv")
    
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
# MAIN ENTRY POINT
# ============================================================================

function main()
    if length(ARGS) < 1
        println("""
        PIT Memory Halo Fitter - Usage:
        
        Single galaxy:
          julia pit_memory_fitter.jl <galaxy_file.csv>
          
        Batch mode:
          julia pit_memory_fitter.jl --batch <data_directory>
          
        Data format (CSV):
          r,v_obs,v_err,v_baryon_unit
          
        Where v_baryon_unit is baryonic velocity at Upsilon=1
        """)
        return
    end
    
    if ARGS[1] == "--batch"
        if length(ARGS) < 2
            println("Error: --batch requires data directory")
            return
        end
        data_dir = ARGS[2]
        output_file = length(ARGS) >= 3 ? ARGS[3] : "memory_fit_results.csv"
        fit_all_galaxies(data_dir, output_file=output_file)
    else
        test_single_galaxy(ARGS[1])
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
