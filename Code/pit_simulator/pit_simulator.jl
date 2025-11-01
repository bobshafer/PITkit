#!/usr/bin/env julia

"""
PIT (Participatory Interface Theory) Field Simulator
Detects cosmological phase transitions in memory-novelty dynamics
"""

using FFTW
using Statistics
using LinearAlgebra
using Printf
using Dates
using DelimitedFiles
using Random

# ============================================================================
# CONFIGURATION
# ============================================================================

struct PITParams
    mu::Float64        # Memory reinforcement
    nu::Float64        # Novelty/noise amplitude
    alpha::Float64     # Conformance rate (Phi -> K)
    beta::Float64      # Learning rate (K <- Phi)
    N::Int            # Grid size (N x N)
    max_steps::Int    # Maximum simulation steps
    kernel_size::Int  # Gaussian kernel size
    kernel_sigma::Float64  # Gaussian kernel width
    save_interval::Int     # Save power spectrum every N steps
    seed::Int         # Random seed for reproducibility
end

mutable struct SimulationState
    Phi::Matrix{Float64}
    K::Matrix{Float64}
    tau::Int
end

struct Diagnostics
    tau::Vector{Int}
    coherence::Vector{Float64}
    info_flow::Vector{Float64}
    entropy::Vector{Float64}
    correlation_length::Vector{Float64}
    power_spectrum_slope::Vector{Float64}
    mean_k_value::Vector{Float64}
    mean_phi_value::Vector{Float64}
    transition_detected::Bool
    transition_tau::Int
    power_spectra::Vector{Vector{Float64}}
    power_spectra_tau::Vector{Int}
end

# ============================================================================
# GAUSSIAN KERNEL GENERATION
# ============================================================================

function create_gaussian_kernel_1d(size::Int, sigma::Float64)
    kernel = zeros(size)
    center = div(size, 2) + 1
    
    for i in 1:size
        x = i - center
        kernel[i] = exp(-x^2 / (2 * sigma^2))
    end
    
    # Normalize
    kernel ./= sum(kernel)
    return kernel
end

# ============================================================================
# SEPARABLE 2D CONVOLUTION (Fast Gaussian Blur)
# ============================================================================

function separable_convolve!(output::Matrix{Float64}, 
                            input::Matrix{Float64}, 
                            kernel::Vector{Float64})
    N = size(input, 1)
    temp = zeros(N, N)
    center = div(length(kernel), 2) + 1
    
    # Horizontal pass
    for i in 1:N
        for j in 1:N
            sum_val = 0.0
            for k in 1:length(kernel)
                jj = mod1(j + k - center, N)
                sum_val += input[i, jj] * kernel[k]
            end
            temp[i, j] = sum_val
        end
    end
    
    # Vertical pass
    for i in 1:N
        for j in 1:N
            sum_val = 0.0
            for k in 1:length(kernel)
                ii = mod1(i + k - center, N)
                sum_val += temp[ii, j] * kernel[k]
            end
            output[i, j] = sum_val
        end
    end
end

# ============================================================================
# FIELD EVOLUTION (Core PIT Dynamics)
# ============================================================================

function evolve_fields!(state::SimulationState, 
                       params::PITParams, 
                       kernel::Vector{Float64},
                       F_Phi::Matrix{Float64})
    N = params.N
    
    # Compute F[Phi] - the pattern in Phi
    separable_convolve!(F_Phi, state.Phi, kernel)
    
    # Create new fields
    new_Phi = zeros(N, N)
    new_K = zeros(N, N)
    
    # Evolve each grid point
    for i in 1:N
        for j in 1:N
            # Dissonance: how much K differs from pattern
            dissonance = state.K[i, j] - F_Phi[i, j]
            
            # Noise term
            noise = params.nu * (2 * rand() - 1)
            
            # Evolve Phi: Reality conforms to Law K
            new_Phi[i, j] = (1 - params.alpha) * state.Phi[i, j] + 
                           params.alpha * state.K[i, j] + noise
            
            # Evolve K: Law learns from dissonance, with logistic memory
            k_logistic = state.K[i, j] * (1.0 - state.K[i, j])
            new_K[i, j] = state.K[i, j] - params.beta * dissonance + 
                         params.mu * k_logistic
            
            # Clipping for stability
            new_Phi[i, j] = clamp(new_Phi[i, j], -10.0, 10.0)
            new_K[i, j] = clamp(new_K[i, j], -10.0, 10.0)
        end
    end
    
    # Update state
    state.Phi .= new_Phi
    state.K .= new_K
    state.tau += 1
end

# ============================================================================
# DIAGNOSTICS: Coherence, Info Flow, Entropy
# ============================================================================

function compute_coherence(K::Matrix{Float64}, F_Phi::Matrix{Float64})
    dissonance = sum(abs.(K .- F_Phi))
    return -dissonance / length(K)
end

function compute_info_flow(Phi::Matrix{Float64}, K::Matrix{Float64})
    phi_flat = vec(Phi)
    k_flat = vec(K)
    N = length(phi_flat)
    
    sum_phi = sum(phi_flat)
    sum_k = sum(k_flat)
    sum_phi_sq = sum(phi_flat .^ 2)
    sum_k_sq = sum(k_flat .^ 2)
    sum_phi_k = sum(phi_flat .* k_flat)
    
    num = N * sum_phi_k - sum_phi * sum_k
    den_phi = sqrt(N * sum_phi_sq - sum_phi^2)
    den_k = sqrt(N * sum_k_sq - sum_k^2)
    
    if den_phi > 0 && den_k > 0
        return num / (den_phi * den_k)
    else
        return 0.0
    end
end

function compute_entropy(Phi::Matrix{Float64})
    # Bin the values
    bins = Dict{Int, Int}()
    for val in Phi
        bin = round(Int, val * 100)
        bins[bin] = get(bins, bin, 0) + 1
    end
    
    # Compute Shannon entropy
    n_total = length(Phi)
    entropy = 0.0
    for count in values(bins)
        if count > 0
            prob = count / n_total
            entropy -= prob * log2(prob)
        end
    end
    
    return entropy
end

# ============================================================================
# ADVANCED DIAGNOSTICS: Correlation Length
# ============================================================================

function compute_correlation_length(field::Matrix{Float64})
    N = size(field, 1)
    
    # Compute autocorrelation via FFT
    field_centered = field .- mean(field)
    ft = fft(field_centered)
    power = abs2.(ft)
    autocorr_2d = real(ifft(power))
    
    # Radial average
    center = div(N, 2) + 1
    max_r = div(N, 3)  # Don't go too far due to periodic boundaries
    
    radial_corr = zeros(max_r)
    counts = zeros(Int, max_r)
    
    for i in 1:N
        for j in 1:N
            dx = i - center
            dy = j - center
            r = round(Int, sqrt(dx^2 + dy^2))
            if r > 0 && r <= max_r
                radial_corr[r] += autocorr_2d[i, j]
                counts[r] += 1
            end
        end
    end
    
    # Normalize
    for r in 1:max_r
        if counts[r] > 0
            radial_corr[r] /= counts[r]
        end
    end
    
    # Normalize to C(0) = 1
    if radial_corr[1] > 0
        radial_corr ./= radial_corr[1]
    end
    
    # Find correlation length (where C(r) drops to 1/e)
    threshold = 1.0 / exp(1)
    for r in 1:max_r
        if radial_corr[r] < threshold
            return Float64(r)
        end
    end
    
    return Float64(max_r)  # Correlation extends beyond our measurement
end

# ============================================================================
# ADVANCED DIAGNOSTICS: Power Spectrum
# ============================================================================

function compute_power_spectrum(field::Matrix{Float64})
    N = size(field, 1)
    
    # 2D FFT
    field_centered = field .- mean(field)
    ft = fft(field_centered)
    power_2d = abs2.(ft)
    
    # Radial average
    center = div(N, 2) + 1
    max_k = div(N, 2)
    
    power_1d = zeros(max_k)
    counts = zeros(Int, max_k)
    
    for i in 1:N
        for j in 1:N
            # k-space coordinates
            kx = i <= center ? i - 1 : i - N - 1
            ky = j <= center ? j - 1 : j - N - 1
            k = round(Int, sqrt(kx^2 + ky^2))
            
            if k > 0 && k <= max_k
                power_1d[k] += power_2d[i, j]
                counts[k] += 1
            end
        end
    end
    
    # Normalize
    for k in 1:max_k
        if counts[k] > 0
            power_1d[k] /= counts[k]
        end
    end
    
    return power_1d
end

function compute_power_spectrum_slope(power::Vector{Float64})
    # Fit P(k) ~ k^β in log space, excluding k=0 and very high k
    n = length(power)
    k_min = 2
    k_max = div(n, 2)
    
    if k_max <= k_min
        return 0.0
    end
    
    log_k = log.(Float64(k_min):Float64(k_max))
    log_p = log.(power[k_min:k_max] .+ 1e-10)  # Add small constant to avoid log(0)
    
    # Linear regression in log space
    n_pts = length(log_k)
    sum_x = sum(log_k)
    sum_y = sum(log_p)
    sum_xy = sum(log_k .* log_p)
    sum_xx = sum(log_k .^ 2)
    
    slope = (n_pts * sum_xy - sum_x * sum_y) / (n_pts * sum_xx - sum_x^2)
    
    return slope
end

# ============================================================================
# PHASE TRANSITION DETECTION
# ============================================================================

function detect_phase_transition(diag::Diagnostics)
    # Look for sudden drop in correlation length
    # This indicates transition from large-scale structure to pointillist
    
    if length(diag.correlation_length) < 50
        return false, 0
    end
    
    # Compute derivative of correlation length
    window = 20
    for i in (window+1):(length(diag.correlation_length)-window)
        # Compare average before and after
        before = mean(diag.correlation_length[(i-window):(i-1)])
        after = mean(diag.correlation_length[(i+1):(i+window)])
        
        # If correlation length drops by > 50%
        if before > 0 && (before - after) / before > 0.5
            return true, diag.tau[i]
        end
    end
    
    return false, 0
end

# ============================================================================
# MAIN SIMULATION LOOP
# ============================================================================

function run_simulation(params::PITParams; verbose::Bool=true, galaxy_seed::Bool=false)
    # Set random seed for reproducibility
    Random.seed!(params.seed)
    
    # Initialize state
    state = SimulationState(
        randn(params.N, params.N) * 0.1,  # Phi: small random initial state
        randn(params.N, params.N) * 0.1,  # K: small random initial kernel
        0                                  # tau
    )
    
    # Add galaxy seed if requested
    if galaxy_seed
        cx, cy = div(params.N, 2), div(params.N, 2)
        radius = 10
        amplitude = 2.0
        
        for i in 1:params.N
            for j in 1:params.N
                dx = i - cx
                dy = j - cy
                dist = sqrt(dx^2 + dy^2)
                if dist < radius
                    bump = amplitude * exp(-(dist^2) / (2 * (radius/2)^2))
                    state.Phi[i, j] += bump
                end
            end
        end
    end
    
    # Create Gaussian kernel
    kernel = create_gaussian_kernel_1d(params.kernel_size, params.kernel_sigma)
    
    # Preallocate working arrays
    F_Phi = zeros(params.N, params.N)
    
    # Initialize diagnostics
    diag = Diagnostics(
        Int[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        Float64[],
        false,
        0,
        Vector{Float64}[],
        Int[]
    )
    
    if verbose
        println("PIT Simulation Starting...")
        println("Parameters: μ=$(params.mu), ν=$(params.nu), α=$(params.alpha), β=$(params.beta)")
        println("Grid: $(params.N)×$(params.N), Steps: $(params.max_steps)")
        println("Gaussian kernel: σ=$(params.kernel_sigma), size=$(params.kernel_size)")
        println("Galaxy seed: $galaxy_seed")
        println()
    end
    
    start_time = time()
    
    # Main evolution loop
    for step in 1:params.max_steps
        # Evolve fields
        evolve_fields!(state, params, kernel, F_Phi)
        
        # Compute diagnostics
        coherence = compute_coherence(state.K, F_Phi)
        info_flow = compute_info_flow(state.Phi, state.K)
        entropy = compute_entropy(state.Phi)
        corr_length = compute_correlation_length(state.K)
        
        push!(diag.tau, state.tau)
        push!(diag.coherence, coherence)
        push!(diag.info_flow, info_flow)
        push!(diag.entropy, entropy)
        push!(diag.correlation_length, corr_length)
        push!(diag.mean_k_value, mean(state.K))
        push!(diag.mean_phi_value, mean(state.Phi))
        
        # Compute power spectrum periodically
        if step % params.save_interval == 0
            power = compute_power_spectrum(state.K)
            slope = compute_power_spectrum_slope(power)
            push!(diag.power_spectra, power)
            push!(diag.power_spectra_tau, state.tau)
            push!(diag.power_spectrum_slope, slope)
        else
            push!(diag.power_spectrum_slope, NaN)
        end
        
        # Progress indicator
        if verbose && step % 100 == 0
            elapsed = time() - start_time
            rate = step / elapsed
            eta = (params.max_steps - step) / rate
            @printf("Step %5d/%d | C=%.4f | ξ=%.2f | η=%.3f | %.1f steps/s | ETA: %.1fs\n",
                   step, params.max_steps, coherence, corr_length, entropy, rate, eta)
        end
        
        # Check for instability
        if isnan(coherence) || isnan(info_flow)
            if verbose
                println("\n⚠️  Simulation became unstable at τ=$(state.tau)")
            end
            break
        end
    end
    
    # Detect phase transition
    transition_detected, transition_tau = detect_phase_transition(diag)
    diag = Diagnostics(
        diag.tau,
        diag.coherence,
        diag.info_flow,
        diag.entropy,
        diag.correlation_length,
        diag.power_spectrum_slope,
        diag.mean_k_value,
        diag.mean_phi_value,
        transition_detected,
        transition_tau,
        diag.power_spectra,
        diag.power_spectra_tau
    )
    
    if verbose
        elapsed = time() - start_time
        println("\n" * "="^70)
        println("Simulation Complete!")
        println("Total time: $(round(elapsed, digits=1))s")
        println("Final τ: $(state.tau)")
        println("Final coherence: $(round(diag.coherence[end], digits=4))")
        println("Final correlation length: $(round(diag.correlation_length[end], digits=2))")
        
        if transition_detected
            println("\n🎯 Phase transition detected at τ = $transition_tau")
            idx = findfirst(diag.tau .== transition_tau)
            if idx !== nothing && idx > 10
                before_idx = max(1, idx - 10)
                ξ_before = mean(diag.correlation_length[before_idx:(idx-1)])
                ξ_after = mean(diag.correlation_length[(idx+1):min(end, idx+10)])
                drop_pct = round((ξ_before - ξ_after) / ξ_before * 100, digits=1)
                println("   Correlation length: $(round(ξ_before, digits=2)) → $(round(ξ_after, digits=2)) ($drop_pct% drop)")
            end
        else
            println("\n📊 No clear phase transition detected")
        end
        println("="^70)
    end
    
    return state, diag
end

# ============================================================================
# DATA EXPORT
# ============================================================================

function save_results(diag::Diagnostics, params::PITParams, filename::String)
    # Save main time series
    data = hcat(
        diag.tau,
        diag.coherence,
        diag.info_flow,
        diag.entropy,
        diag.correlation_length,
        diag.mean_k_value,
        diag.mean_phi_value
    )
    
    # Write header as separate row, not as repeated column values
    open(filename, "w") do io
        println(io, "tau,coherence,info_flow,entropy,correlation_length,mean_K,mean_Phi")
        writedlm(io, data, ',')
    end
    
    # Save power spectra
    if !isempty(diag.power_spectra)
        ps_filename = replace(filename, ".csv" => "_power_spectra.csv")
        n_spectra = length(diag.power_spectra)
        max_k = length(diag.power_spectra[1])
        
        ps_data = zeros(max_k, n_spectra + 1)
        ps_data[:, 1] = 1:max_k  # k values
        for i in 1:n_spectra
            ps_data[:, i+1] = diag.power_spectra[i]
        end
        
        # Build proper header with individual column names
        ps_header_parts = ["k"]
        for t in diag.power_spectra_tau
            push!(ps_header_parts, "tau_$t")
        end
        ps_header = join(ps_header_parts, ",")
        
        # Write header properly
        open(ps_filename, "w") do io
            println(io, ps_header)
            writedlm(io, ps_data, ',')
        end
        
        println("Power spectra saved to: $ps_filename")
    end
    
    println("Results saved to: $filename")
end

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

function parse_args()
    # THIS IS THE NEW TEST:
    println(">>> Running v3 FIXED parse_args function (handles key=value)...")
    
    # Default parameters (Goldilocks zone)
    mu_val = 0.005
    nu_val = 0.010
    alpha_val = 0.09
    beta_val = 0.04
    N_val = 64
    max_steps_val = 5000
    kernel_size_val = 11
    kernel_sigma_val = 3.0
    save_interval_val = 100
    seed_val = 42
    
    galaxy = false
    output = "pit_results_" * Dates.format(now(), "yyyymmdd_HHMMSS") * ".csv"
    
    # --- New Robust Parser ---
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        key = ""
        value = ""
        
        try
            if occursin("=", arg)
                # Handle "--key=value"
                parts = split(arg, '=', limit=2)
                key = parts[1]
                value = parts[2]
                i += 1 # Consume this one argument
            elseif startswith(arg, "--") && i < length(ARGS) && !startswith(ARGS[i+1], "--")
                # Handle "--key value"
                key = arg
                value = ARGS[i+1]
                i += 2 # Consume both arguments
            else
                # Handle flag like "--galaxy" or "--help"
                key = arg
                i += 1
            end

            # --- Argument Processing ---
            if key == "--mu"
                mu_val = parse(Float64, value)
            elseif key == "--nu"
                nu_val = parse(Float64, value)
            elseif key == "--alpha"
                alpha_val = parse(Float64, value)
            elseif key == "--beta"
                beta_val = parse(Float64, value)
            elseif key == "--sigma"
                kernel_sigma_val = parse(Float64, value)
                kernel_size_val = 2 * round(Int, 3 * kernel_sigma_val) + 1
                println(">>> PARSER: Set kernel_sigma = $(kernel_sigma_val), kernel_size = $(kernel_size_val)")
            elseif key == "--steps"
                max_steps_val = parse(Int, value)
            elseif key == "--N"
                N_val = parse(Int, value)
            elseif key == "--seed"
                seed_val = parse(Int, value)
            elseif key == "--galaxy"
                galaxy = true
            elseif key == "--output"
                output = String(value)
                println(">>> PARSER: Set output file = $(output)")
            elseif key == "--help"
                println("""
                PIT Simulator - Usage:
                
                julia pit_simulator.jl [OPTIONS]
                
                Options: (Can use --key value OR --key=value)
                  --mu VALUE       Memory parameter (default: 0.005)
                  --nu VALUE       Novelty parameter (default: 0.010)
                  --alpha VALUE    Conformance rate (default: 0.09)
                  --beta VALUE     Learning rate (default: 0.04)
                  --sigma VALUE    Gaussian kernel width (default: 3.0)
                  --N VALUE        Grid size (default: 64)
                  --steps VALUE    Max simulation steps (default: 5000)
                  --seed VALUE     Random seed (default: 42)
                  --galaxy         Add galaxy seed to initial conditions
                  --output FILE    Output filename (default: auto-generated)
                  --help           Show this help
                  
                Example:
                  julia pit_simulator.jl --sigma=4.5 --galaxy --steps=5000 --output=sigma_4p5.csv
                """)
                exit(0)
            else
                println("Warning: Ignoring unknown argument: $(key)")
            end
        catch e
            println("Error parsing argument $(arg): $e")
            # Safely increment to avoid infinite loop
            if !occursin("=", arg) && i <= length(ARGS)
                 i += 1
            end
        end
    end
    
    # Now, construct the *final* params struct
    params = PITParams(
        mu_val,
        nu_val,
        alpha_val,
        beta_val,
        N_val,
        max_steps_val,
        kernel_size_val,
        kernel_sigma_val,
        save_interval_val,
        seed_val
    )
    
    return params, galaxy, output
end

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

function main()
    params, galaxy, output = parse_args()
    
    # Run simulation
    state, diag = run_simulation(params, verbose=true, galaxy_seed=galaxy)
    
    # Save results
    save_results(diag, params, output)
    
    return diag
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
