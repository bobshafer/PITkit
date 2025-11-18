#!/usr/bin/env julia

#= 
PIT-based planetary simulation for HD 110067.
Implements the ProcessBasedPIT spec: each planet is a coherence-seeking process,
step_process() is the only evolution rule, and no Kepler/gravitation constants
appear in the dynamics. Emergent habits are observed via logged metrics.
=#

using LinearAlgebra
using Random
using FFTW
using Statistics
using Printf
using JSON

const NDIM = 64                   # resolution of Φ/K arrays
const PLANET_NAMES = ["b","c","d","e","f","g"]
const PLANET_PERIODS = [9.114, 13.673, 20.519, 30.793, 41.058, 54.770]  # days; used only as phase rates
const PLANET_MASSES = [2.5, 3.2, 5.0, 3.9, 2.6, 4.1]                    # M_earth; scales μ/ν bias
const OBSERVED_HD110067_RATIOS = [1.500, 1.500, 1.500, 1.333, 1.333]

struct ProcessState
    name::String
    phi::Vector{Float64}
    phi_velocity::Vector{Float64}
    K::Vector{Float64}
    mu::Float64
    nu::Float64
    neighbors::Vector{Int}
    memory_scale::Float64
    novelty_scale::Float64
end

struct SimulationParams
    λ::Float64
    memory_gain::Float64
    novelty_gain::Float64
    damping::Float64
    k_smoothing::Float64
end

struct CLIConfig
    steps::Int
    dt::Float64
    log_every::Int
    seed::Int
    save_history::Bool
    history_every::Int
    perturb_at::Int
    perturb_planet::String
    perturb_amplitude::Float64
    output_prefix::String
end

struct HistorySnapshot
    step::Int
    states::Vector{ProcessState}
end

struct SimulationHistory
    snapshots::Vector{HistorySnapshot}
    metrics::Vector{Dict{String,Any}}
end

function parse_cli(args)::CLIConfig
    steps = 8000
    dt = 0.01
    log_every = 200
    seed = 2025
    save_history = false
    history_every = 50
    perturb_at = 0
    perturb_planet = ""
    perturb_amplitude = 0.5
    output_prefix = "hd110067"

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--steps"
            i += 1; i > length(args) && error("--steps requires value")
            steps = parse(Int, args[i])
        elseif arg == "--dt"
            i += 1; i > length(args) && error("--dt requires value")
            dt = parse(Float64, args[i])
        elseif arg == "--log-every"
            i += 1; i > length(args) && error("--log-every requires value")
            log_every = parse(Int, args[i])
        elseif arg == "--seed"
            i += 1; i > length(args) && error("--seed requires value")
            seed = parse(Int, args[i])
        elseif arg == "--save-history"
            save_history = true
        elseif arg == "--history-every"
            i += 1; i > length(args) && error("--history-every requires value")
            history_every = parse(Int, args[i])
        elseif arg == "--perturb-at"
            i += 1; i > length(args) && error("--perturb-at requires value")
            perturb_at = parse(Int, args[i])
        elseif arg == "--perturb-planet"
            i += 1; i > length(args) && error("--perturb-planet requires value")
            perturb_planet = args[i]
        elseif arg == "--perturb-amplitude"
            i += 1; i > length(args) && error("--perturb-amplitude requires value")
            perturb_amplitude = parse(Float64, args[i])
        elseif arg == "--output-prefix"
            i += 1; i > length(args) && error("--output-prefix requires value")
            output_prefix = args[i]
        elseif arg in ("-h","--help")
            println("""
            Usage: julia hd110067_process_sim.jl [OPTIONS]

            Options:
              --steps N                Number of simulation steps (default: 8000)
              --dt DELTA               Time step size (default: 0.01)
              --log-every N            Log interval (default: 200)
              --seed S                 Random seed (default: 2025)
              --save-history           Save full history for period extraction
              --history-every N        Snapshot interval when --save-history (default: 50)
              --perturb-at STEP        Inject perturbation at this step
              --perturb-planet NAME    Planet to perturb (b,c,d,e,f,g)
              --perturb-amplitude A    Perturbation amplitude (default: 0.5)
              --output-prefix PREFIX   Output filename prefix (default: hd110067)
              -h, --help               Show this help

            Examples:
              julia hd110067_process_sim.jl --steps 50000 --log-every 1000 --save-history
              julia hd110067_process_sim.jl --steps 50000 --perturb-at 25000 --perturb-planet f --save-history
            """)
            exit(0)
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end

    history_every = max(1, history_every)
    return CLIConfig(steps, dt, log_every, seed, save_history, history_every,
                     perturb_at, perturb_planet, perturb_amplitude, output_prefix)
end

interface(phi::AbstractVector{<:Real}) = real.(fft(phi)) ./ length(phi)

function local_dissonance(p::ProcessState)
    diff = p.K .- interface(p.phi)
    return sum(abs2, diff) / length(diff)
end

function dissonance_grad_phi(p::ProcessState)
    diff_freq = interface(p.phi) .- p.K
    return real.(ifft(diff_freq))
end

function mutual_coherence(p::ProcessState, q::ProcessState)
    denom = max(norm(p.K) * norm(q.K), 1e-6)
    return dot(p.K, q.K) / denom
end

Gt(x::Real) = tanh(x)

function pit_lagrangian(p::ProcessState, params::SimulationParams)
    kinetic = sum(abs2, p.phi_velocity)
    coupling = -params.λ * local_dissonance(p)
    resonance = dot(p.phi, p.K) / length(p.phi)
    memory = -p.mu * resonance^2
    novelty = -p.nu * resonance * Gt(resonance)
    return kinetic + coupling + memory + novelty
end

function exploration_term(p::ProcessState, rng::AbstractRNG)
    return randn(rng, length(p.phi))
end

function step_process(p::ProcessState, neighbors::Vector{ProcessState}, params::SimulationParams, dt::Float64, rng::AbstractRNG)
    current_dis = local_dissonance(p)
    if current_dis > 1.0
        @warn "Planet $(p.name) high dissonance: $(current_dis)"
    end

    neighbor_drive = isempty(neighbors) ? 0.0 : mean(mutual_coherence(p, n) for n in neighbors)
    resonant_alignment = dot(p.phi, p.K) / length(p.phi)
    dis_grad = dissonance_grad_phi(p)
    novelty_factor = 1.0 - abs(Gt(resonant_alignment))
    noise_vec = exploration_term(p, rng)

        memory_drive = params.memory_gain * p.memory_scale
        novelty_drive = params.novelty_gain * p.novelty_scale
        delta_phi = -params.λ .* dis_grad
        delta_phi .+= memory_drive * p.mu * resonant_alignment .* real.(ifft(p.K))
        delta_phi .-= novelty_drive * p.nu * novelty_factor .* noise_vec
        damp_factor = inv(1.0 + 4.0 * max(0.0, current_dis - 0.1))
        delta_phi .*= damp_factor

    new_phi_velocity = (1 - params.damping) .* p.phi_velocity + dt .* delta_phi
    new_phi = p.phi + dt .* new_phi_velocity

    iface = interface(new_phi)
        coherence_boost = p.memory_scale * p.mu * neighbor_drive .* iface
        novelty_decay = p.novelty_scale * p.nu * novelty_factor .* noise_vec
    new_K = p.K + dt .* (coherence_boost - params.k_smoothing .* (p.K - iface) - novelty_decay)

    new_mu = p.mu
    new_nu = p.nu

    if any(isnan, new_phi) || any(isnan, new_K)
        error("NaN detected in planet $(p.name) at μ=$(p.mu), ν=$(p.nu), dis=$(current_dis)")
    end

    return ProcessState(p.name, new_phi, new_phi_velocity, new_K, new_mu, new_nu, p.neighbors, p.memory_scale, p.novelty_scale)
end

function step_all_processes(states::Vector{ProcessState}, params::SimulationParams, dt::Float64, rng::AbstractRNG)
    new_states = Vector{ProcessState}(undef, length(states))
    for (idx, state) in pairs(states)
        neighbor_states = [states[j] for j in state.neighbors]
        new_states[idx] = step_process(state, neighbor_states, params, dt, rng)
    end
    return new_states
end

function initialize_processes(; seed::Int=2025)
    rng = MersenneTwister(seed)
    phases = range(0, 2π, length=NDIM)
    max_mass = maximum(PLANET_MASSES)
    states = ProcessState[]
    nplanets = length(PLANET_NAMES)

    for (idx, name) in enumerate(PLANET_NAMES)
        period = PLANET_PERIODS[idx]
        base_freq = 2π / period
        mass_scale = PLANET_MASSES[idx] / max_mass
        phase_shift = 2π * (idx - 1) / nplanets

        phi = [sin(base_freq * t + phase_shift) +
               0.2 * mass_scale * cos(2 * base_freq * t + phase_shift)
               for t in phases]
        phi .+= 0.05 .* randn(rng, NDIM)
        phi_velocity = 0.01 .* randn(rng, NDIM)
        K = interface(phi)

        base_mu = 0.55 + 0.08 * mass_scale
        mu = clamp(base_mu + 0.02 * randn(rng), 0.50, 0.65)
        target_ratio = 2.0 + 0.3 * (mass_scale - 0.5)
        nu = mu / target_ratio
        nu = clamp(nu + 0.02 * randn(rng), 0.25, 0.35)
        actual_ratio = mu / nu
        if actual_ratio < 1.5
            nu = mu / 1.5
        elseif actual_ratio > 3.0
            nu = mu / 3.0
        end
        mu = clamp(mu, 0.50, 0.65)
        nu = clamp(nu, 0.25, 0.35)

        left = idx == 1 ? nplanets : idx - 1
        right = idx == nplanets ? 1 : idx + 1
        memory_scale = clamp(0.8 + 0.3 * mass_scale, 0.75, 1.2)
        novelty_scale = clamp(1.05 - 0.25 * mass_scale, 0.75, 1.15)

        push!(states, ProcessState(name, phi, phi_velocity, K, mu, nu, [left, right], memory_scale, novelty_scale))
    end

    println("=== COSMIC PRIORS (FROZEN) ===")
    println("These μ/ν values represent the system at formation (z≈0) and remain fixed during simulation:\n")
    for state in states
        @printf "  Planet %s: μ=%.3f ν=%.3f ratio=%.2f\n" state.name state.mu state.nu (state.mu / state.nu)
    end
    avg_mu = mean(s.mu for s in states)
    avg_nu = mean(s.nu for s in states)
    @printf "\n  System average: μ=%.3f ν=%.3f ratio=%.2f\n" avg_mu avg_nu (avg_mu / avg_nu)
    println("="^50, "\n")

    return states
end

function collect_metrics(states::Vector{ProcessState})
    avg_mu = mean(p.mu for p in states)
    avg_nu = mean(p.nu for p in states)
    dissonances = [local_dissonance(p) for p in states]
    resonance = mean(dot(p.phi, p.K) / length(p.phi) for p in states)
    return Dict(
        "avg_mu" => avg_mu,
        "avg_nu" => avg_nu,
        "avg_dissonance" => mean(dissonances),
        "resonance" => resonance,
        "max_dissonance" => maximum(dissonances),
    )
end

function fftfreq(n::Int, d::Float64=1.0)
    n > 0 || error("n must be positive")
    val = 1.0 / (n * d)
    freqs = Vector{Float64}(undef, n)
    m = n ÷ 2
    for i in 0:(n-1)
        freqs[i+1] = (i <= m) ? i * val : (i - n) * val
    end
    return freqs
end

function extract_phi_oscillations(history::SimulationHistory, planet_idx::Int)
    if isempty(history.snapshots)
        return Float64[], Int[]
    end
    snapshot = history.snapshots[1]
    planet_idx >= 1 && planet_idx <= length(snapshot.states) || error("planet_idx out of bounds")
    steps = [snap.step for snap in history.snapshots]
    phi_means = [mean(snap.states[planet_idx].phi) for snap in history.snapshots]
    return phi_means, steps
end

function find_dominant_period(signal::Vector{Float64}, dt_steps::Int, dt_physical::Float64)
    if length(signal) < 10
        return NaN
    end
    sample_spacing = dt_steps * dt_physical
    signal_centered = signal .- mean(signal)
    spectrum = abs.(fft(signal_centered))
    freqs = fftfreq(length(signal_centered), sample_spacing)
    n_half = length(freqs) ÷ 2
    if n_half <= 1
        return NaN
    end
    spectrum_pos = spectrum[2:n_half]
    freqs_pos = freqs[2:n_half]
    isempty(spectrum_pos) && return NaN
    peak_idx = argmax(spectrum_pos)
    dominant_freq = freqs_pos[peak_idx]
    return abs(dominant_freq) > 1e-10 ? 1.0 / abs(dominant_freq) : NaN
end

function analyze_emergent_resonances(history::SimulationHistory, dt::Float64, history_stride::Int)
    println("\n=== EMERGENT RESONANCE ANALYSIS ===")
    if isempty(history.snapshots)
        println("No history data available. Run with --save-history flag.")
        return Float64[], Float64[]
    end

    n_planets = length(history.snapshots[1].states)
    periods = Vector{Float64}(undef, n_planets)

    println("\nExtracting periods from phi oscillations:")
    for i in 1:n_planets
        phi_series, _ = extract_phi_oscillations(history, i)
        period = find_dominant_period(phi_series, history_stride, dt)
        periods[i] = period
        planet_name = history.snapshots[1].states[i].name
        @printf "  Planet %s: period = %.3f time units\n" planet_name period
    end

    println("\nEmergent period ratios:")
    ratios = Float64[]
    for i in 1:(length(periods)-1)
        ratio = (isfinite(periods[i]) && isfinite(periods[i+1])) ? periods[i+1] / periods[i] : NaN
        push!(ratios, ratio)
        @printf "  %s/%s = %.3f" PLANET_NAMES[i+1] PLANET_NAMES[i] ratio
        if isfinite(ratio)
            if abs(ratio - 1.5) < 0.1
                print("  ≈ 3:2 ✓")
            elseif abs(ratio - (4/3)) < 0.1
                print("  ≈ 4:3 ✓")
            end
        end
        println()
    end

    if length(ratios) == length(OBSERVED_HD110067_RATIOS)
        println("\nComparison to HD 110067:")
        total_error = 0.0
        for i in 1:length(ratios)
            emergent = ratios[i]
            observed = OBSERVED_HD110067_RATIOS[i]
            error_val = isfinite(emergent) ? abs(emergent - observed) : NaN
            if isfinite(error_val)
                total_error += error_val
            end
            @printf "  Position %d: emergent=%.3f observed=%.3f error=%.3f\n" i emergent observed error_val
        end
        if isfinite(total_error)
            @printf "\nTotal ratio error: %.4f\n" total_error
            if total_error < 0.2
                println("✓ EXCELLENT AGREEMENT with observations!")
            elseif total_error < 0.5
                println("✓ Good agreement with observations")
            else
                println("⚠ Significant deviation from observations")
            end
        else
            println("⚠ Unable to compute total error (non-finite ratios)")
        end
    end

    return periods, ratios
end

function save_metrics_jsonl(metrics_log::Vector{Dict{String,Any}}, filename::String)
    open(filename, "w") do io
        for m in metrics_log
            println(io, JSON.json(m))
        end
    end
    println("Metrics saved to $filename")
end

function save_final_states(states::Vector{ProcessState}, params::SimulationParams, filename::String)
    output = Dict(
        "params" => Dict(
            "lambda" => params.λ,
            "memory_gain" => params.memory_gain,
            "novelty_gain" => params.novelty_gain,
            "damping" => params.damping,
            "k_smoothing" => params.k_smoothing,
        ),
        "final_states" => [
            Dict(
                "name" => s.name,
                "mu" => s.mu,
                "nu" => s.nu,
                "dissonance" => local_dissonance(s),
                "mu_nu_ratio" => s.mu / s.nu,
                "phi_mean" => mean(s.phi),
                "phi_std" => std(s.phi),
                "K_norm" => norm(s.K),
                "memory_scale" => s.memory_scale,
                "novelty_scale" => s.novelty_scale,
            ) for s in states
        ],
    )

    open(filename, "w") do io
        JSON.print(io, output, 4)
    end
    println("Final states saved to $filename")
end

function run_simulation(; steps::Int=8000,
                         dt::Float64=0.01,
                         log_every::Int=200,
                         seed::Int=2025,
                         save_history::Bool=false,
                         history_every::Int=50,
                         perturb_at::Int=0,
                         perturb_planet::String="",
                         perturb_amplitude::Float64=0.0)
    params = SimulationParams(0.75, 0.30, 0.55, 0.05, 0.35)
    states = initialize_processes(seed=seed)
    rng = MersenneTwister(seed + 99)
    metrics_log = Dict{String,Any}[]
    history_snapshots = HistorySnapshot[]
    last_logged_step = 0

    for step in 1:steps
        if perturb_at > 0 && step == perturb_at && !isempty(perturb_planet)
            planet_idx = findfirst(s -> s.name == perturb_planet, states)
            if planet_idx !== nothing
                perturbed_state = states[planet_idx]
                new_phi = perturbed_state.phi .+ perturb_amplitude .* randn(rng, NDIM)
                states[planet_idx] = ProcessState(
                    perturbed_state.name,
                    new_phi,
                    perturbed_state.phi_velocity,
                    perturbed_state.K,
                    perturbed_state.mu,
                    perturbed_state.nu,
                    perturbed_state.neighbors,
                    perturbed_state.memory_scale,
                    perturbed_state.novelty_scale,
                )
                println(">>> PERTURBATION INJECTED at step $step: planet $perturb_planet, amplitude=$perturb_amplitude")
            else
                println(">>> WARNING: Requested perturbation planet '$perturb_planet' not found.")
            end
        end

        states = step_all_processes(states, params, dt, rng)

        if step % log_every == 0
            metrics = collect_metrics(states)
            metrics["step"] = step
            push!(metrics_log, metrics)
            last_logged_step = step

            if save_history && (step % history_every == 0)
                snapshot_states = [ProcessState(
                    s.name,
                    copy(s.phi),
                    copy(s.phi_velocity),
                    copy(s.K),
                    s.mu,
                    s.nu,
                    copy(s.neighbors),
                    s.memory_scale,
                    s.novelty_scale,
                ) for s in states]
                push!(history_snapshots, HistorySnapshot(step, snapshot_states))
            end

            @printf "[step %6d] dis=%.4f res=%.4f" step metrics["avg_dissonance"] metrics["resonance"]

            n_high_dis = count(s -> local_dissonance(s) > 0.5, states)
            if n_high_dis > 0
                print(" ⚠ HIGH-DIS($(n_high_dis))")
            end
            max_dis = maximum(local_dissonance(s) for s in states)
            if max_dis < 0.1
                print(" ✓ STABLE")
            end
            println()
        end
    end

    if last_logged_step != steps
        metrics = collect_metrics(states)
        metrics["step"] = steps
        push!(metrics_log, metrics)
        if save_history
            snapshot_states = [ProcessState(
                s.name,
                copy(s.phi),
                copy(s.phi_velocity),
                copy(s.K),
                s.mu,
                s.nu,
                copy(s.neighbors),
                s.memory_scale,
                s.novelty_scale,
            ) for s in states]
            push!(history_snapshots, HistorySnapshot(steps, snapshot_states))
        end
    end

    return SimulationHistory(history_snapshots, metrics_log), states
end

if abspath(PROGRAM_FILE) == @__FILE__
    cli = parse_cli(ARGS)

    println("=== HD 110067 Process-Based Simulation ===")
    println("Configuration:")
    @printf "  Steps: %d\n" cli.steps
    @printf "  dt: %.4f\n" cli.dt
    @printf "  Log interval: %d\n" cli.log_every
    if cli.save_history
        @printf "  History interval: %d\n" cli.history_every
    end
    @printf "  Seed: %d\n" cli.seed
    @printf "  Save history: %s\n" cli.save_history
    if cli.perturb_at > 0 && !isempty(cli.perturb_planet)
        @printf "  Perturbation: planet %s at step %d, amplitude %.2f\n" cli.perturb_planet cli.perturb_at cli.perturb_amplitude
    end
    println()

    history, final_states = run_simulation(
        steps=cli.steps,
        dt=cli.dt,
        log_every=cli.log_every,
        seed=cli.seed,
        save_history=cli.save_history,
        history_every=cli.history_every,
        perturb_at=cli.perturb_at,
        perturb_planet=cli.perturb_planet,
        perturb_amplitude=cli.perturb_amplitude,
    )

    last_metrics = history.metrics[end]
    println("\n=== FINAL SNAPSHOT ===")
    @printf "Steps: %d\n" last_metrics["step"]
    @printf "Average μ: %.3f\n" last_metrics["avg_mu"]
    @printf "Average ν: %.3f\n" last_metrics["avg_nu"]
    @printf "Average dissonance: %.4f\n" last_metrics["avg_dissonance"]
    @printf "Resonance: %.4f\n" last_metrics["resonance"]

    println("\n=== PER-PLANET STATUS ===")
    for state in final_states
        ratio = state.mu / state.nu
        @printf "Planet %s → μ=%.3f ν=%.3f μ/ν=%.2f dis=%.4f" state.name state.mu state.nu ratio local_dissonance(state)
        if 1.5 < ratio < 3.0
            print(" ✓ Goldilocks (FROZEN)")
        else
            print(" (FROZEN)")
        end
        println()
    end

    println("\nNote: μ/ν values remain fixed throughout the simulation (cosmic-timescale parameters).")

    if cli.save_history
        analyze_emergent_resonances(history, cli.dt, cli.history_every)
    else
        println("\nNote: Run with --save-history to analyze emergent period ratios")
    end

    metrics_file = "$(cli.output_prefix)_metrics.jsonl"
    states_file = "$(cli.output_prefix)_final_states.json"
    save_metrics_jsonl(history.metrics, metrics_file)
    params = SimulationParams(0.75, 0.30, 0.55, 0.05, 0.35)
    save_final_states(final_states, params, states_file)

    println("\n✓ Simulation complete")
end
