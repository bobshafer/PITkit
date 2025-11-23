# hd110067_adaptive.jl
#
# PIT HD 110067 Simulation Engine (Adaptive / Process Fractal)
# Based on Loom Spec v10.1 + Addendum v0.1
#
# Features:
# - Complex K-Field (Phasors)
# - Adaptive Parameters (alpha, mu drift based on Dissonance)
# - Coherence History Tracking
# - "Frozen" Gravity (Baseline) + "Living" Resonance (PIT)
#
# Dependencies:
#   ] add StaticArrays DataFrames CSV
#

using LinearAlgebra
using StaticArrays
using Statistics
using DataFrames
using CSV
using Printf
using Random

# --- 1. ONTOLOGY (Structures) --------------------------------------------

mutable struct Planet
    id::Int
    mass::Float64       # Solar Masses
    pos::MVector{3, Float64} # AU
    vel::MVector{3, Float64} # AU/Day
    name::String
end

mutable struct KernelField
    # The "Habits" (Resonant Phasors for each pair)
    modes::Vector{ComplexF64} 
    
    # Adaptive Parameters (Now Mutable)
    mu::Float64     # Memory Stiffness (Habit Strength)
    nu::Float64     # Novelty (Learning/Plasticity)
    alpha::Float64  # Coupling Strength (PIT Force)
    
    # Adaptive Drift Config (from Addendum)
    k_alpha::Float64 # Drift rate for alpha
    k_mu::Float64    # Drift rate for mu
    d_target::Float64 # Target dissonance (D*)
end

struct ResonanceRule
    inner_id::Int
    outer_id::Int
    p::Int
    q::Int
end

# Coherence History Buffer
mutable struct HistoryBuffer
    window_size::Int
    coherence_log::Vector{Float64} # Rolling window of coherence values
end

const G = 2.95912208286e-4  # AU^3 / (Solar Mass * Day^2)

# --- 2. PHYSICS ENGINE ---------------------------------------------------

function get_lambda(p::Planet)
    x, y = p.pos[1], p.pos[2]
    return atan(y, x)
end

# A. The Interface (Measure Phi)
function measure_phasors(planets::Vector{Planet}, rules::Vector{ResonanceRule})
    phasors = Vector{ComplexF64}(undef, length(rules))
    for (i, rule) in enumerate(rules)
        p_in = planets[rule.inner_id]
        p_out = planets[rule.outer_id]
        
        lambda_in = get_lambda(p_in)
        lambda_out = get_lambda(p_out)
        
        # Resonance Angle: phi = p*λ_out - q*λ_in
        phi = rule.p * lambda_out - rule.q * lambda_in
        phasors[i] = exp(im * phi)
    end
    return phasors
end

# B. Adaptive Parameter Drift (Addendum v0.1)
function adapt_parameters!(K::KernelField, current_dissonance::Float64, dt::Float64)
    # 1. Alpha Drift: Increases when Dissonance is high (Needs stronger correction)
    #    d_alpha = k_alpha * (D - D_target)
    d_alpha = K.k_alpha * (current_dissonance - K.d_target)
    K.alpha += d_alpha * dt
    
    # 2. Mu Drift: Decreases when Dissonance is high (Habit becomes flexible)
    #    d_mu = -k_mu * D
    d_mu = -K.k_mu * current_dissonance
    K.mu += d_mu * dt
    
    # Safety Bounds
    K.alpha = clamp(K.alpha, 1.0e-7, 1.0e-3)
    K.mu    = clamp(K.mu,    0.1,    0.999)
end

# C. The Memory Update (K-Field Evolution)
function update_kernel!(K::KernelField, current_phasors::Vector{ComplexF64}, dt::Float64)
    # Effective Learning Rate depends on Nu and current Saturation
    saturation = abs(mean(K.modes))
    learning_rate = K.nu * (1.0 - saturation)
    
    total_dissonance = 0.0
    
    for i in 1:length(K.modes)
        # Dissonance Vector = Phi - K
        diff = current_phasors[i] - K.modes[i]
        total_dissonance += abs(diff)
        
        # Update K: Move towards Phi
        K.modes[i] += learning_rate * diff * dt
        
        # Renormalize if habit is too strong (Limit |K| <= 1)
        if abs(K.modes[i]) > 1.0
            K.modes[i] = K.modes[i] / abs(K.modes[i])
        end
    end
    
    return total_dissonance / length(K.modes) # Return Avg Dissonance
end

# D. The Manifestation (Forces)
function apply_forces!(planets::Vector{Planet}, star_mass::Float64, K::KernelField, rules::Vector{ResonanceRule})
    # 1. Frozen Habit (Gravity)
    for p in planets
        r_mag = norm(p.pos)
        acc = -G * star_mass * p.pos / (r_mag^3)
        p.vel += acc
    end
    
    # 2. Active Habit (PIT Resonance Torque)
    # Only applies if alpha is significant
    if K.alpha > 1.0e-9
        current_phasors = measure_phasors(planets, rules)
        
        for (i, rule) in enumerate(rules)
            k_mem = K.modes[i]
            z_curr = current_phasors[i]
            
            # Phase Error
            delta_phi = angle(k_mem) - angle(z_curr)
            
            # Torque = Coupling * Memory_Strength * Sin(Error)
            torque = K.alpha * abs(k_mem) * sin(delta_phi)
            
            # Apply tangential kicks
            p_in = planets[rule.inner_id]
            p_out = planets[rule.outer_id]
            
            t_in = cross([0,0,1.0], p_in.pos) / norm(p_in.pos)
            t_out = cross([0,0,1.0], p_out.pos) / norm(p_out.pos)
            
            p_in.vel += (torque / p_in.mass) * t_in
            p_out.vel -= (torque / p_out.mass) * t_out
        end
    end
end

function step_pos!(planets::Vector{Planet}, dt::Float64)
    for p in planets
        p.pos += p.vel * dt
    end
end

# --- 3. MAIN SIMULATION LOOP ---------------------------------------------

function main()
    println("--- PIT ADAPTIVE ENGINE: HD 110067 ---")
    println("Initializing Process Fractal...")
    
    # 1. Init Planets (HD 110067)
    star_mass = 0.81
    m_scale = 3.003e-6
    
    # Init in approximate resonance positions
    planets = [
        Planet(1, 5.69 * m_scale, MVector(0.0793, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "b"),
        Planet(2, 5.69 * m_scale, MVector(0.1039, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "c"),
        Planet(3, 5.69 * m_scale, MVector(0.1364, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "d"),
        Planet(4, 5.69 * m_scale, MVector(0.1790, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "e"),
        Planet(5, 5.69 * m_scale, MVector(0.2166, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "f"),
        Planet(6, 5.69 * m_scale, MVector(0.2621, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "g")
    ]
    
    # Set initial circular velocities + random phase
    Random.seed!(123)
    for p in planets
        v_circ = sqrt(G * star_mass / norm(p.pos))
        theta = rand() * 2 * pi
        r = norm(p.pos)
        p.pos = MVector(r*cos(theta), r*sin(theta), 0.0)
        p.vel = MVector(-v_circ*sin(theta), v_circ*cos(theta), 0.0)
    end

    # 2. Init Kernel (Memory)
    rules = [
        ResonanceRule(1, 2, 3, 2), ResonanceRule(2, 3, 3, 2),
        ResonanceRule(3, 4, 3, 2), ResonanceRule(4, 5, 4, 3),
        ResonanceRule(5, 6, 4, 3)
    ]
    
    # Start with weak, random memory (System hasn't learned yet)
    init_modes = [0.01 * exp(im * rand() * 2 * pi) for _ in 1:length(rules)]
    
    K = KernelField(
        init_modes,
        0.90,   # Initial Mu (High Memory)
        0.05,   # Nu (Learning Rate)
        1.0e-5, # Initial Alpha (Weak Coupling)
        
        # Adaptive Params (ThePage.loom.md)
        1.0e-4, # k_alpha (Drift speed)
        1.0e-5, # k_mu (Drift speed)
        0.10    # D_target (Target dissonance ~ 10%)
    )
    
    println("Kernel Ready. Initial Alpha=$(K.alpha), Mu=$(K.mu)")

    # 3. Run Loop
    dt = 0.1
    steps = 20000
    
    # Logs
    trace_step = Int[]
    trace_phi1 = Float64[]
    trace_K_str = Float64[]
    trace_alpha = Float64[]
    trace_mu = Float64[]
    trace_diss = Float64[]
    
    for t in 1:steps
        # A. Measure Phi
        current_phasors = measure_phasors(planets, rules)
        
        # B. Update K (and get Dissonance)
        avg_diss = update_kernel!(K, current_phasors, dt)
        
        # C. Adaptive Drift (The Living System)
        adapt_parameters!(K, avg_diss, dt)
        
        # D. Apply Forces
        apply_forces!(planets, star_mass, K, rules)
        step_pos!(planets, dt)
        
        # E. Logging
        if t % 20 == 0
            push!(trace_step, t)
            push!(trace_phi1, angle(current_phasors[1]))
            push!(trace_K_str, abs(K.modes[1]))
            push!(trace_alpha, K.alpha)
            push!(trace_mu, K.mu)
            push!(trace_diss, avg_diss)
        end
        
        if t % 5000 == 0
            @printf("Step %d: K-Str=%.3f, Alpha=%.2e, Mu=%.3f, Diss=%.3f\n", 
                    t, abs(K.modes[1]), K.alpha, K.mu, avg_diss)
        end
    end
    
    # 4. Save Output
    df = DataFrame(
        Step = trace_step,
        Phi_Angle = trace_phi1,
        K_Strength = trace_K_str,
        Alpha = trace_alpha,
        Mu = trace_mu,
        Dissonance = trace_diss
    )
    
    filename = "hd110067_adaptive_trace.csv"
    CSV.write(filename, df)
    println("\nSimulation Complete. Data saved to $filename")
end

main()
