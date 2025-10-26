#!/usr/bin/env julia

"""
PIT Simulation Results Plotter
Visualizes diagnostics and power spectrum evolution
"""

using Plots
using DelimitedFiles
using Statistics
using Printf

# Set plotting backend and style
gr()
theme(:dark)

# ============================================================================
# LOAD DATA
# ============================================================================

function load_results(filename::String)
    """Load main diagnostics CSV"""
    data, header = readdlm(filename, ',', header=true)
    
    results = Dict(
        "tau" => data[:, 1],
        "coherence" => data[:, 2],
        "info_flow" => data[:, 3],
        "entropy" => data[:, 4],
        "correlation_length" => data[:, 5],
        "mean_K" => data[:, 6],
        "mean_Phi" => data[:, 7]
    )
    
    return results
end

function load_power_spectra(filename::String)
    """Load power spectrum CSV"""
    data, header = readdlm(filename, ',', header=true)
    
    k_values = data[:, 1]
    n_spectra = size(data, 2) - 1
    
    # Parse tau values from header
    tau_values = Int[]
    for i in 2:size(data, 2)
        tau_str = string(header[i])
        tau_match = match(r"tau_(\d+)", tau_str)
        if tau_match !== nothing
            push!(tau_values, parse(Int, tau_match.captures[1]))
        end
    end
    
    # Extract spectra
    spectra = data[:, 2:end]
    
    return k_values, tau_values, spectra
end

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

function plot_diagnostics(results::Dict, output_file::String="diagnostics.png")
    """Create multi-panel diagnostic plot"""
    
    tau = results["tau"]
    
    # Create 4-panel figure
    p1 = plot(tau, results["coherence"],
              xlabel="τ", ylabel="Coherence C",
              label="", lw=2, color=:cyan,
              title="Coherence Evolution")
    
    p2 = plot(tau, results["correlation_length"],
              xlabel="τ", ylabel="ξ (grid units)",
              label="", lw=2, color=:yellow,
              title="Correlation Length",
              yscale=:log10)
    
    p3 = plot(tau, results["entropy"],
              xlabel="τ", ylabel="Shannon Entropy",
              label="", lw=2, color=:orange,
              title="Entropy (Complexity)")
    
    p4 = plot(tau, results["info_flow"],
              xlabel="τ", ylabel="Correlation (Φ,K)",
              label="", lw=2, color=:green,
              title="Info Flow (Coupling)")
    
    layout = @layout [a b; c d]
    p = plot(p1, p2, p3, p4, layout=layout, size=(1200, 800),
             plot_title="PIT Simulation Diagnostics")
    
    savefig(p, output_file)
    println("Diagnostics saved to: $output_file")
    
    return p
end

function plot_power_spectrum_evolution(k_values, tau_values, spectra, 
                                       output_file::String="power_evolution.png";
                                       log_scale::Bool=true)
    """Plot power spectrum at multiple time points"""
    
    n_spectra = length(tau_values)
    
    # Select ~6 evenly spaced time points
    indices = unique(round.(Int, range(1, n_spectra, length=min(6, n_spectra))))
    
    p = plot(xlabel="k (mode number)", 
             ylabel="P(k)",
             title="Power Spectrum Evolution",
             legend=:topright,
             size=(800, 600))
    
    if log_scale
        p = plot!(xscale=:log10, yscale=:log10)
    end
    
    # Color gradient from early (blue) to late (red)
    colors = range(colorant"blue", colorant"red", length=length(indices))
    
    for (i, idx) in enumerate(indices)
        tau = tau_values[idx]
        spectrum = spectra[:, idx]
        
        # Skip if all zeros
        if all(spectrum .== 0)
            continue
        end
        
        plot!(p, k_values, spectrum .+ 1e-10,  # Add small offset for log scale
              label="τ=$(tau)",  # FIX: Was always showing first tau
              lw=2,
              color=colors[i])
    end
    
    savefig(p, output_file)
    println("Power spectrum evolution saved to: $output_file")
    
    return p
end

function plot_power_spectrum_heatmap(k_values, tau_values, spectra,
                                     output_file::String="power_heatmap.png")
    """Create heatmap of power spectrum over time"""
    
    # Find where we have actual non-zero data
    has_power = vec(sum(spectra, dims=1)) .> 0
    last_idx = findlast(has_power)
    
    if last_idx === nothing || last_idx < 2
        println("⚠️  No power spectrum data to plot")
        return nothing
    end
    println("tau_plot last_idx is $last_idx")
    
    # Trim to relevant range
    tau_plot = tau_values[1:last_idx]
    spectra_plot = spectra[:, 1:last_idx]
    
    # Log transform
    log_spectra = log10.(spectra_plot .+ 1e-10)
    
    # Use contourf which is more reliable than heatmap
    p = contourf(tau_plot, k_values, log_spectra,
                 xlabel="τ (process time)",
                 ylabel="k (wavenumber)",
                 title="Power Spectrum Evolution: log₁₀(P(k,τ))",
                 color=:turbo,
                 levels=20,
                 size=(1200, 700),
                 colorbar_title="log₁₀(P)",
                 linewidth=0)
    
    savefig(p, output_file)
    
    min_power = minimum(filter(x -> x > 0, spectra_plot))
    max_power = maximum(spectra_plot)
    
    println("Power spectrum heatmap saved to: $output_file")
    println("  τ range: $(tau_plot[1]) to $(tau_plot[end]), tau_plot length is $(length(tau_plot))")
    println("  Power range: $(min_power) to $(max_power)")
    println("  Log₁₀ range: $(log10(min_power)) to $(log10(max_power))")
    
    return p
end

function plot_spectral_slope(k_values, tau_values, spectra,
                             output_file::String="spectral_slope.png")
    """Compute and plot power spectrum slope over time"""
    
    slopes = Float64[]
    valid_tau = Int[]
    
    for i in 1:length(tau_values)
        spectrum = spectra[:, i]
        
        # Skip if all zeros or negligible
        if sum(spectrum) < 1.0
            continue
        end
        
        # Fit slope in log-log space (k=2 to k=10)
        k_range = 2:min(10, length(k_values))
        log_k = log.(k_values[k_range])
        log_p = log.(spectrum[k_range] .+ 1e-10)
        
        # Skip if not enough data
        if all(isfinite.(log_p))
            # Linear regression
            n = length(log_k)
            sum_x = sum(log_k)
            sum_y = sum(log_p)
            sum_xy = sum(log_k .* log_p)
            sum_xx = sum(log_k .^ 2)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x^2)
            
            push!(slopes, slope)
            push!(valid_tau, tau_values[i])
        end
    end
    
    if isempty(slopes)
        println("⚠️  No valid spectral slopes computed")
        return nothing
    end
    
    p = plot(valid_tau, slopes,
             xlabel="τ",
             ylabel="β (spectral slope)",
             title="Power Spectrum Slope: P(k) ~ k^β",
             label="",
             lw=2,
             color=:purple,
             size=(800, 500))
    
    # Add reference lines
    hline!([0], label="White noise (β=0)", ls=:dash, color=:white, alpha=0.3)
    hline!([-2], label="Scale-free (β=-2)", ls=:dash, color=:red, alpha=0.3)
    
    savefig(p, output_file)
    println("Spectral slope saved to: $output_file")
    
    return p
end

function plot_total_power(tau_values, spectra, output_file::String="total_power.png")
    """Plot total integrated power over time"""
    
    total_power = vec(sum(spectra, dims=1))
    
    p = plot(tau_values, total_power,
             xlabel="τ",
             ylabel="Total Power ∑P(k)",
             title="Integrated Power Spectrum",
             label="",
             lw=2,
             color=:cyan,
             yscale=:log10,
             size=(800, 500))
    
    # Mark when power drops to near-zero
    threshold = maximum(total_power) * 0.01
    collapse_idx = findfirst(total_power .< threshold)
    if collapse_idx !== nothing
        vline!([tau_values[collapse_idx]], 
               label="Structure collapse",
               ls=:dash, color=:red, lw=2)
    end
    
    savefig(p, output_file)
    println("Total power plot saved to: $output_file")
    
    return p
end

function plot_phase_portrait(results::Dict, output_file::String="phase_portrait.png")
    """Plot coherence vs entropy phase portrait"""
    
    coherence = results["coherence"]
    entropy = results["entropy"]
    tau = results["tau"]
    
    # Color by time
    p = scatter(coherence, entropy,
                xlabel="Coherence C",
                ylabel="Entropy η",
                title="Phase Space Trajectory",
                marker_z=tau,
                markersize=3,
                markerstrokewidth=0,
                color=:plasma,
                colorbar_title="τ",
                size=(700, 600),
                label="")
    
    # Add arrow to show direction
    n = length(coherence)
    if n > 1
        annotate!(coherence[1], entropy[1], 
                 text("Start", :white, :right, 8))
        annotate!(coherence[end], entropy[end],
                 text("End", :white, :left, 8))
    end
    
    savefig(p, output_file)
    println("Phase portrait saved to: $output_file")
    
    return p
end

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

function detect_transitions(results::Dict)
    """Detect and report phase transitions"""
    
    tau = results["tau"]
    xi = results["correlation_length"]
    coherence = results["coherence"]
    entropy = results["entropy"]
    
    println("\n" * "="^70)
    println("PHASE TRANSITION ANALYSIS")
    println("="^70)
    
    # Find correlation length drops
    window = 20
    for i in (window+1):(length(xi)-window)
        before = mean(xi[(i-window):(i-1)])
        after = mean(xi[(i+1):(i+window)])
        
        if before > 0 && (before - after) / before > 0.3
            @printf("🎯 Correlation drop at τ = %d: ξ = %.2f → %.2f (%.1f%% drop)\n",
                   tau[i], before, after, 100*(before-after)/before)
        end
    end
    
    # Find entropy peaks
    for i in 2:(length(entropy)-1)
        if entropy[i] > entropy[i-1] && entropy[i] > entropy[i+1] && entropy[i] > 5
            @printf("📊 Entropy peak at τ = %d: η = %.3f\n", tau[i], entropy[i])
        end
    end
    
    # Find coherence spikes (but skip early transient)
    c_std = std(coherence[100:end])  # Use stats from after initial transient
    c_mean = mean(coherence[100:end])
    for i in 100:(length(coherence)-1)  # Skip first 100 steps
        if abs(coherence[i] - c_mean) > 5 * c_std  # Make threshold higher
            @printf("⚠️  Coherence anomaly at τ = %d: C = %.4f\n", tau[i], coherence[i])
        end
    end
    
    println("="^70 * "\n")
end

function print_summary(results::Dict)
    """Print summary statistics"""
    
    println("\n" * "="^70)
    println("SIMULATION SUMMARY")
    println("="^70)
    
    tau_final = results["tau"][end]
    c_final = results["coherence"][end]
    xi_final = results["correlation_length"][end]
    eta_final = results["entropy"][end]
    
    c_mean = mean(results["coherence"])
    c_std = std(results["coherence"])
    
    @printf("Total steps: %d\n", tau_final)
    @printf("\nFinal state:\n")
    @printf("  Coherence: %.4f\n", c_final)
    @printf("  Correlation length: %.2f\n", xi_final)
    @printf("  Entropy: %.3f\n", eta_final)
    
    @printf("\nCoherence statistics:\n")
    @printf("  Mean: %.4f\n", c_mean)
    @printf("  Std: %.4f\n", c_std)
    @printf("  Min: %.4f\n", minimum(results["coherence"]))
    @printf("  Max: %.4f\n", maximum(results["coherence"]))
    
    # Check stability
    if abs(c_final) < 0.1 && c_std < 0.1
        println("\n✅ System appears STABLE (Goldilocks zone)")
    else
        println("\n⚠️  System may be UNSTABLE or near boundary")
    end
    
    println("="^70 * "\n")
end

# ============================================================================
# MAIN SCRIPT
# ============================================================================

function main()
    if length(ARGS) < 1
        println("""
        Usage: julia plot_results.jl <results_file.csv>
        
        This will automatically look for:
          - results_file.csv (main diagnostics)
          - results_file_power_spectra.csv (power spectra)
          
        And generate:
          - diagnostics.png
          - power_evolution.png
          - power_heatmap.png
          - spectral_slope.png
          - phase_portrait.png
        """)
        return
    end
    
    results_file = ARGS[1]
    
    # Derive power spectrum filename
    base_name = replace(results_file, ".csv" => "")
    power_file = base_name * "_power_spectra.csv"
    
    println("Loading results from: $results_file")
    results = load_results(results_file)
    
    # Print summary
    print_summary(results)
    
    # Detect transitions
    detect_transitions(results)
    
    # Generate plots
    println("Generating plots...")
    
    plot_diagnostics(results, base_name * "_diagnostics.png")
    plot_phase_portrait(results, base_name * "_phase_portrait.png")
    
    # Try to load and plot power spectra
    if isfile(power_file)
        println("\nLoading power spectra from: $power_file")
        k_values, tau_values, spectra = load_power_spectra(power_file)
        
        # Add total power plot
        plot_total_power(tau_values, spectra, base_name * "_total_power.png")
        
        plot_power_spectrum_evolution(k_values, tau_values, spectra,
                                      base_name * "_power_evolution.png")
        plot_power_spectrum_heatmap(k_values, tau_values, spectra,
                                   base_name * "_power_heatmap.png")
        plot_spectral_slope(k_values, tau_values, spectra,
                          base_name * "_spectral_slope.png")
    else
        println("\n⚠️  Power spectrum file not found: $power_file")
        println("   Skipping power spectrum plots")
    end
    
    println("\n✅ Analysis complete!")
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
