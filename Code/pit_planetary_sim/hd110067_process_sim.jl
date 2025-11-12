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

const NDIM = 64                   # resolution of Φ/K arrays
const PLANET_NAMES = ["b","c","d","e","f","g"]
const PLANET_PERIODS = [9.114, 13.673, 20.519, 30.793, 41.058, 54.770]  # days; used only as phase rates
const PLANET_MASSES = [2.5, 3.2, 5.0, 3.9, 2.6, 4.1]                    # M_earth; scales μ/ν bias

struct ProcessState
    name::String
    phi::Vector{Float64}
    phi_velocity::Vector{Float64}
    K::Vector{Float64}
    mu::Float64
    nu::Float64
    neighbors::Vector{Int}
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
end

function parse_cli(args)::CLIConfig
    steps = 8000
    dt = 0.01
    log_every = 200
    seed = 2025
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
        elseif arg in ("-h","--help")
            println("Usage: julia hd110067_process_sim.jl [--steps N] [--dt Δt] [--log-every N] [--seed S]")
            exit(0)
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end
    return CLIConfig(steps, dt, log_every, seed)
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

function update_memory_rate(mu::Float64, new_K::Vector{Float64}, neighbor_drive::Float64, dt::Float64)
    target = clamp(neighbor_drive, -1, 1)
    return clamp(mu + dt * 0.1 * (target - (mu - 0.5)), 0.1, 0.95)
end

function update_novelty_rate(nu::Float64, phi::Vector{Float64}, dt::Float64)
    spread = std(phi)
    desired = clamp(0.4 - spread, -0.3, 0.3)
    return clamp(nu + dt * 0.1 * (desired - (nu - 0.3)), 0.05, 0.9)
end

function exploration_term(p::ProcessState, rng::AbstractRNG)
    return randn(rng, length(p.phi))
end

function step_process(p::ProcessState, neighbors::Vector{ProcessState}, params::SimulationParams, dt::Float64, rng::AbstractRNG)
    neighbor_drive = isempty(neighbors) ? 0.0 : mean(mutual_coherence(p, n) for n in neighbors)
    resonant_alignment = dot(p.phi, p.K) / length(p.phi)
    dis_grad = dissonance_grad_phi(p)
    novelty_factor = 1.0 - abs(Gt(resonant_alignment))
    noise_vec = exploration_term(p, rng)

    delta_phi = -params.λ .* dis_grad
    delta_phi .+= params.memory_gain * p.mu * resonant_alignment .* real.(ifft(p.K))
    delta_phi .-= params.novelty_gain * p.nu * novelty_factor .* noise_vec

    new_phi_velocity = (1 - params.damping) .* p.phi_velocity + dt .* delta_phi
    new_phi = p.phi + dt .* new_phi_velocity

    iface = interface(new_phi)
    coherence_boost = p.mu * neighbor_drive .* iface
    novelty_decay = p.nu * novelty_factor .* noise_vec
    new_K = p.K + dt .* (coherence_boost - params.k_smoothing .* (p.K - iface) - novelty_decay)

    new_mu = update_memory_rate(p.mu, new_K, neighbor_drive, dt)
    new_nu = update_novelty_rate(p.nu, new_phi, dt)

    return ProcessState(p.name, new_phi, new_phi_velocity, new_K, new_mu, new_nu, p.neighbors)
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
        phi = [sin(base_freq * t + phase_shift) + 0.2 * mass_scale * cos(2 * base_freq * t + phase_shift) for t in phases]
        phi .+= 0.05 .* randn(rng, NDIM)
        phi_velocity = 0.01 .* randn(rng, NDIM)
        K = interface(phi)
        base_mu = 0.62 + 0.05 * (mass_scale - 0.5)
        mu = clamp(base_mu + 0.02 * randn(rng), 0.2, 0.9)
        nu = clamp(1.0 - mu + 0.02 * randn(rng), 0.1, 0.8)
        left = idx == 1 ? nplanets : idx - 1
        right = idx == nplanets ? 1 : idx + 1
        push!(states, ProcessState(name, phi, phi_velocity, K, mu, nu, [left, right]))
    end
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

function run_simulation(; steps::Int=8000, dt::Float64=0.01, log_every::Int=200, seed::Int=2025)
    params = SimulationParams(0.8, 0.6, 0.4, 0.05, 0.3)
    states = initialize_processes(seed=seed)
    rng = MersenneTwister(seed + 99)
    metrics_log = Dict{String,Any}[]
    for step in 1:steps
        states = step_all_processes(states, params, dt, rng)
        if step % log_every == 0
            metrics = collect_metrics(states)
            metrics["step"] = step
            push!(metrics_log, metrics)
            @printf "[step %6d] μ=%.3f ν=%.3f dis=%.4f res=%.4f\n" step metrics["avg_mu"] metrics["avg_nu"] metrics["avg_dissonance"] metrics["resonance"]
        end
    end
    return states, metrics_log
end

if abspath(PROGRAM_FILE) == @__FILE__
    cli = parse_cli(ARGS)
    final_states, metrics_log = run_simulation(steps=cli.steps, dt=cli.dt, log_every=cli.log_every, seed=cli.seed)
    last_metrics = metrics_log[end]
    println("\nFinal snapshot:")
    @printf "steps=%d avg_mu=%.3f avg_nu=%.3f avg_dissonance=%.4f resonance=%.4f\n" last_metrics["step"] last_metrics["avg_mu"] last_metrics["avg_nu"] last_metrics["avg_dissonance"] last_metrics["resonance"]
    println("\nPer-planet μ/ν:")
    for state in final_states
        @printf "Planet %s → μ=%.3f ν=%.3f dis=%.4f\n" state.name state.mu state.nu local_dissonance(state)
    end
end

