# hd110067_adaptive.jl
#
# PIT HD 110067 Simulation Engine (Adaptive / Process Fractal)
# Based on Loom Spec v10.1 + Addendum v0.1

using LinearAlgebra
using StaticArrays
using Statistics
using DataFrames
using CSV
using Printf
using Random

# ------------------------------------------------------------------------------
# 1. ONTOLOGY (Structures)
# ------------------------------------------------------------------------------

mutable struct Planet
    id::Int
    mass::Float64
    pos::MVector{3,Float64}
    vel::MVector{3,Float64}
    name::String
end

mutable struct KernelField
    modes::Vector{ComplexF64}   # Resonant memories for each rule

    # Goldilocks parameters
    mu::Float64
    nu::Float64
    alpha::Float64

    # Adaptive drift parameters (Addendum v0.1)
    k_alpha::Float64
    k_mu::Float64
    d_target::Float64
end

struct ResonanceRule
    inner_id::Int
    outer_id::Int
    p::Int
    q::Int
end

const G = 2.95912208286e-4   # AU^3 / (Solar Mass * Day^2)

# ------------------------------------------------------------------------------
# 2. SUPPORT FUNCTIONS
# ------------------------------------------------------------------------------

# Mean longitude approximation
get_lambda(p::Planet) = atan(p.pos[2], p.pos[1])

# Interface: measure the phasors z = exp(iϕ)
function measure_phasors(planets, rules)
    ph = Vector{ComplexF64}(undef, length(rules))
    @inbounds for (i, rule) in enumerate(rules)
        λi = get_lambda(planets[rule.inner_id])
        λo = get_lambda(planets[rule.outer_id])
        ϕ = rule.p*λo - rule.q*λi
        ph[i] = exp(im*ϕ)
    end
    return ph
end

# ------------------------------------------------------------------------------
# 3. K-FIELD UPDATE (Memory Evolution)
# ------------------------------------------------------------------------------

function update_kernel!(K::KernelField, z::Vector{ComplexF64}, dt::Float64)
    sat = abs(mean(K.modes))            # saturation (0 → flexible, 1 → rigid)
    η = K.nu * (1 - sat)                # effective learning rate
    total_d = 0.0

    @inbounds for i in 1:length(K.modes)
        diff = z[i] - K.modes[i]
        total_d += abs(diff)
        K.modes[i] += η * diff * dt

        # keep |K| <= 1
        r = abs(K.modes[i])
        if r > 1
            K.modes[i] /= r
        end
    end

    return total_d / length(K.modes), sat
end

# ------------------------------------------------------------------------------
# 4. ADAPTIVE PARAMETER DRIFT (Addendum v0.1)
# ------------------------------------------------------------------------------

function adapt_parameters!(K::KernelField, D::Float64, dt::Float64)

    # α drift: increases when dissonance > target
    K.alpha += K.k_alpha * (D - K.d_target) * dt

    # μ drift: high dissonance → weaken rigidity
    K.mu -= K.k_mu * D * dt

    # clamp to safe bands
    K.alpha = clamp(K.alpha, 1e-6, 0.2)
    K.mu    = clamp(K.mu,    0.01, 1.0)

    return nothing
end

# ------------------------------------------------------------------------------
# 5. MANIFESTATION (Φ UPDATE): Gravity + PIT torque
# ------------------------------------------------------------------------------

function apply_forces!(planets, star_mass, K::KernelField, rules)
    # 1. Gravity
    for p in planets
        r = p.pos
        d = norm(r)
        p.vel += (-G * star_mass * r / d^3)
    end

    # 2. PIT torque
    if K.alpha > 1e-9
        z = measure_phasors(planets, rules)
        @inbounds for (i, rule) in enumerate(rules)
            k = K.modes[i]
            zi = z[i]
            Δ = angle(k) - angle(zi)
            torque = K.alpha * abs(k) * sin(Δ)

            pin = planets[rule.inner_id]
            pout = planets[rule.outer_id]

            tin  = cross(@MVector [0,0,1.0], pin.pos)  / norm(pin.pos)
            tout = cross(@MVector [0,0,1.0], pout.pos) / norm(pout.pos)

            pin.vel  += (torque / pin.mass)  * tin
            pout.vel -= (torque / pout.mass) * tout
        end
    end
end

# position integrator
step_pos!(planets, dt) = for p in planets; p.pos += p.vel * dt; end

# ------------------------------------------------------------------------------
# 6. MAIN SIMULATION
# ------------------------------------------------------------------------------

function main()
    println("---------------------------------------------------")
    println("     PIT ADAPTIVE ENGINE: HD 110067 (v10.1+)       ")
    println("---------------------------------------------------")

    # 6 planets in 3:2 chain
    star_mass = 0.81
    m_scale = 3.003e-6

    planets = [
        Planet(1, 5.69m_scale, MVector(0.0793,0,0), MVector(0,0,0), "b"),
        Planet(2, 5.69m_scale, MVector(0.1039,0,0), MVector(0,0,0), "c"),
        Planet(3, 5.69m_scale, MVector(0.1364,0,0), MVector(0,0,0), "d"),
        Planet(4, 5.69m_scale, MVector(0.1790,0,0), MVector(0,0,0), "e"),
        Planet(5, 5.69m_scale, MVector(0.2166,0,0), MVector(0,0,0), "f"),
        Planet(6, 5.69m_scale, MVector(0.2621,0,0), MVector(0,0,0), "g")
    ]

    # randomize phases
    Random.seed!(123)
    for p in planets
        r = norm(p.pos)
        θ = rand()*2π
        v = sqrt(G*star_mass/r)
        p.pos = MVector(r*cos(θ), r*sin(θ), 0)
        p.vel = MVector(-v*sin(θ), v*cos(θ), 0)
    end

    rules = [
        ResonanceRule(1,2,3,2),
        ResonanceRule(2,3,3,2),
        ResonanceRule(3,4,3,2),
        ResonanceRule(4,5,4,3),
        ResonanceRule(5,6,4,3)
    ]

    # random weak memory
    init_modes = [0.01exp(im*2π*rand()) for _ in rules]

    K = KernelField(
        init_modes,
        0.90,     # mu
        0.05,     # nu
        1e-5,     # alpha
        1e-4,     # k_alpha
        1e-5,     # k_mu
        0.10      # D_target
    )

    dt = 0.1
    steps = 20000

    trace_step = Int[]
    trace_phi1 = Float64[]
    trace_K = Float64[]
    trace_alpha = Float64[]
    trace_mu = Float64[]
    trace_D = Float64[]

    println("Running simulation for $steps steps...")

    for t in 1:steps

        # A: measure current matter state
        z = measure_phasors(planets, rules)

        # B: update memory (K)
        D, sat = update_kernel!(K, z, dt)

        # C: adaptive parameter drift
        adapt_parameters!(K, D, dt)

        # D: apply planetary forces
        apply_forces!(planets, star_mass, K, rules)

        # E: integrate positions
        step_pos!(planets, dt)

        # F: logging
        if t % 20 == 0
            push!(trace_step, t)
            push!(trace_phi1, angle(z[1]))
            push!(trace_K, abs(K.modes[1]))
            push!(trace_alpha, K.alpha)
            push!(trace_mu, K.mu)
            push!(trace_D, D)
        end

        if t % 5000 == 0
            @printf(
                "Step %d | |K|=%.3f | α=%.3e | μ=%.3f | D=%.3f\n",
                t, abs(K.modes[1]), K.alpha, K.mu, D
            )
        end
    end

    df = DataFrame(
        Step = trace_step,
        Phi_Angle = trace_phi1,
        K_Strength = trace_K,
        Alpha = trace_alpha,
        Mu = trace_mu,
        Dissonance = trace_D
    )

    CSV.write("hd110067_adaptive_trace.csv", df)
    println("\nSimulation complete. Saved hd110067_adaptive_trace.csv")
end

main()

