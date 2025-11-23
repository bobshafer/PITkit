# pit_stress_test.jl
# Experiment A: The Ultimate Stress Test
#
# Comparison:
#   1. Fixed PIT ("The Rock"): Constant Alpha/Mu.
#   2. Adaptive PIT ("The Living"): Evolving Alpha/Mu.
#
# Procedure:
#   - Both start in resonance.
#   - Noise (random kicks) ramps up linearly from 0% to 5% of orbital velocity.
#   - We measure when the resonance breaks.

using LinearAlgebra
using StaticArrays
using Statistics
using DataFrames
using CSV
using Printf
using Random

# --- ONTOLOGY ---
mutable struct Planet
    id::Int
    mass::Float64
    pos::MVector{3, Float64}
    vel::MVector{3, Float64}
end

mutable struct KernelField
    modes::Vector{ComplexF64}
    mu::Float64
    nu::Float64
    alpha::Float64
    
    # Adaptive flags
    is_adaptive::Bool
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

const G = 2.95912208286e-4

# --- PHYSICS ---

function get_lambda(p::Planet)
    return atan(p.pos[2], p.pos[1])
end

function measure_phasors(planets::Vector{Planet}, rules::Vector{ResonanceRule})
    phasors = Vector{ComplexF64}(undef, length(rules))
    for (i, rule) in enumerate(rules)
        p_in = planets[rule.inner_id]
        p_out = planets[rule.outer_id]
        phi = rule.p * get_lambda(p_out) - rule.q * get_lambda(p_in)
        phasors[i] = exp(im * phi)
    end
    return phasors
end

function update_kernel!(K::KernelField, current_phasors::Vector{ComplexF64}, dt::Float64)
    # Basic Learning
    learning_rate = K.nu * (1.0 - abs(mean(K.modes)))
    avg_diss = 0.0
    
    for i in 1:length(K.modes)
        diff = current_phasors[i] - K.modes[i]
        avg_diss += abs(diff)
        K.modes[i] += learning_rate * diff * dt
        if abs(K.modes[i]) > 1.0
            K.modes[i] /= abs(K.modes[i])
        end
    end
    avg_diss /= length(K.modes)
    
    # Adaptive Logic (The "Immune System")
    if K.is_adaptive
        # If Dissonance is high, increase Alpha (Fight) and decrease Mu (Plasticity)
        d_alpha = K.k_alpha * (avg_diss - K.d_target)
        K.alpha = clamp(K.alpha + d_alpha * dt, 1.0e-7, 1.0e-4)
        
        d_mu = -K.k_mu * avg_diss
        K.mu = clamp(K.mu + d_mu * dt, 0.1, 0.999)
    end
    
    return avg_diss
end

function compute_accelerations(planets::Vector{Planet}, K::KernelField, rules::Vector{ResonanceRule})
    acc = [MVector{3,Float64}(0.0, 0.0, 0.0) for _ in planets]

    # Gravity (central star)
    for (i, p) in enumerate(planets)
        r = norm(p.pos)
        acc[i] = -G * 0.81 * p.pos / (r^3)
    end

    # PIT Torque
    if K.alpha > 1e-9
        phasors = measure_phasors(planets, rules)
        for (i_rule, rrule) in enumerate(rules)
            k_mem = K.modes[i_rule]
            curr = phasors[i_rule]
            err = angle(k_mem) - angle(curr)
            torque = 0.1 * K.alpha * abs(k_mem) * sin(err)

            p_in = planets[rrule.inner_id]
            p_out = planets[rrule.outer_id]

            t_in = cross([0,0,1.0], p_in.pos); t_in /= norm(t_in)
            t_out = cross([0,0,1.0], p_out.pos); t_out /= norm(t_out)

            acc[rrule.inner_id] += (torque / p_in.mass) * t_in
            acc[rrule.outer_id] -= (torque / p_out.mass) * t_out
        end
    end

    return acc
end

function apply_noise!(planets::Vector{Planet}, noise_level::Float64)
    # Add random kick scaling with orbital velocity
    for p in planets
        v_mag = norm(p.vel)
        kick = randn(3)
        kick -= dot(kick, p.vel/v_mag) * (p.vel/v_mag) # Perpendicular kick mostly
        kick /= norm(kick)
        
        p.vel += kick * (v_mag * noise_level)
    end
end

# --- SIMULATION ---

function run_universe(label::String, adaptive::Bool)
    println("Starting Universe: $label (Adaptive=$adaptive)")
    Random.seed!(42) # Same start for both
    
    m_scale = 3.003e-6
    planets = [
        Planet(1, 5.69*m_scale, MVector(0.0793,0.0,0.0), MVector(0.0,0.0,0.0)),
        Planet(2, 5.69*m_scale, MVector(0.1039,0.0,0.0), MVector(0.0,0.0,0.0)),
        Planet(3, 5.69*m_scale, MVector(0.1364,0.0,0.0), MVector(0.0,0.0,0.0)),
        Planet(4, 5.69*m_scale, MVector(0.1790,0.0,0.0), MVector(0.0,0.0,0.0)),
        Planet(5, 5.69*m_scale, MVector(0.2166,0.0,0.0), MVector(0.0,0.0,0.0)),
        Planet(6, 5.69*m_scale, MVector(0.2621,0.0,0.0), MVector(0.0,0.0,0.0))
    ]
    
    # Init Velocities
    for p in planets
        v = sqrt(G * 0.81 / norm(p.pos))
        p.vel = MVector(0.0, v, 0.0)
        
        # Random phase
        th = rand() * 2pi
        r = norm(p.pos)
        v_mag = norm(p.vel)
        p.pos = MVector(r*cos(th), r*sin(th), 0.0)
        p.vel = MVector(-v_mag*sin(th), v_mag*cos(th), 0.0)
    end
    
    rules = [
        ResonanceRule(1,2,3,2), ResonanceRule(2,3,3,2),
        ResonanceRule(3,4,3,2), ResonanceRule(4,5,4,3),
        ResonanceRule(5,6,4,3)
    ]
    
    # Init Kernel
    # Fixed System gets alpha=1e-4 (tuned "stable" value from before)
    # Adaptive System starts there but can move. Initialize modes to current phasors
    # so initial torque is near zero (starts in resonance).
    init_modes = measure_phasors(planets, rules)
    K = KernelField(init_modes, 0.9, 0.05, 1.0e-6, adaptive, 1.0e-5, 1.0e-5, 0.1)
    
    dt = 0.02
    steps = 5000
    ramp_start = 500
    ramp_end = 5000
    max_noise = 0.02 # 2% kick per step
    
    trace_step = Int[]
    trace_phi = Float64[]
    trace_alpha = Float64[]
    trace_noise = Float64[]
    broke_at = missing
    base_r = maximum(norm(p.pos) for p in planets)
    
    for t in 1:steps
        # Noise Ramp
        curr_noise = 0.0
        if t > ramp_start
            prog = (t - ramp_start) / (ramp_end - ramp_start)
            curr_noise = prog * max_noise
        end
        
        apply_noise!(planets, curr_noise)
        phasors = measure_phasors(planets, rules)
        avg_diss = update_kernel!(K, phasors, dt)

        # Velocity Verlet / leapfrog style
        acc1 = compute_accelerations(planets, K, rules)
        for (i, p) in enumerate(planets)
            p.vel += acc1[i] * (0.5 * dt)
            p.pos += p.vel * dt
        end
        acc2 = compute_accelerations(planets, K, rules)
        for (i, p) in enumerate(planets)
            p.vel += acc2[i] * (0.5 * dt)
        end

        # Break detection: runaway distance or NaN/Inf
        max_r = maximum(norm(p.pos) for p in planets)
        if isnan(max_r) || isinf(max_r) || max_r > 25 * base_r
            broke_at = t
            println("Universe $label broke at step $t (max_r=$max_r)")
            break
        end
        
        if t % 20 == 0
            push!(trace_step, t)
            push!(trace_phi, angle(measure_phasors(planets, rules)[1]))
            push!(trace_alpha, K.alpha)
            push!(trace_noise, curr_noise)
        end
    end
    
    return DataFrame(Step=trace_step, Phi=trace_phi, Alpha=trace_alpha, Noise=trace_noise, Type=label, BrokeAt=broke_at)
end

if abspath(PROGRAM_FILE) == @__FILE__
    df1 = run_universe("Fixed", false)
    df2 = run_universe("Adaptive", true)
    df = vcat(df1, df2)
    CSV.write("stress_test_results.csv", df)
    println("Stress test complete.")
end
