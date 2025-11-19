# main.jl
#
# PIT x HD 110067 Simulation Driver (Process Fractal / Complex K-Field)
#
# Purpose:
#   Simulate the 6-planet resonant chain of HD 110067.
#   Gravity is treated as a "Frozen Habit" (Standard N-Body).
#   Resonance is treated as a "Living Habit" (Complex K-Field).
#   The K-field "learns" the phase angles and "pulls" planets into coherence.
#
#   We use COMPLEX NUMBERS for the K-field to capture Phase and Amplitude.
#
# Usage:
#   julia main.jl
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

# --- 1. THE ONTOLOGY (Structures) ----------------------------------------

# The "Manifest" (Phi-Field)
mutable struct Planet
    id::Int
    mass::Float64       # In Solar Masses
    pos::MVector{3, Float64} # AU
    vel::MVector{3, Float64} # AU/Day (Gaussian)
    name::String
end

# The "Memory" (K-Field)
mutable struct KernelField
    # We track specific resonant habits between pairs.
    # K_modes[i] stores the Complex Phasor (Ae^iθ) for the i-th pair.
    #   Amplitude (A) = Strength of the habit
    #   Phase (θ) = The preferred angle of the resonance
    modes::Vector{ComplexF64} 
    
    # The "Goldilocks" Parameters
    mu::Float64    # Memory Stiffness (How hard K holds onto a pattern)
    nu::Float64    # Novelty/Plasticity (How fast K adapts to new Phi)
    alpha::Float64 # Coupling Strength (How strongly K affects Phi)
end

# The "Law" (Resonance Definitions)
struct ResonanceRule
    inner_id::Int
    outer_id::Int
    p::Int
    q::Int
end

# --- 2. THE PHYSICS (Process Fractal) ------------------------------------

const G = 2.95912208286e-4  # AU^3 / (Solar Mass * Day^2)

# Helper: Mean Longitude (approximate for low eccentricity)
function get_lambda(p::Planet)
    r = norm(p.pos)
    x, y = p.pos[1], p.pos[2]
    return atan(y, x) # Simplified lambda for circular-ish orbits
end

# A. The Interface (Phi -> K)
# Calculate the current state of the resonant angles (The "Voice" of the planets)
function measure_resonance_phasors(planets::Vector{Planet}, rules::Vector{ResonanceRule})
    phasors = Vector{ComplexF64}(undef, length(rules))
    
    for (i, rule) in enumerate(rules)
        p_in = planets[rule.inner_id]
        p_out = planets[rule.outer_id]
        
        # Calculate Mean Longitudes
        lambda_in = get_lambda(p_in)
        lambda_out = get_lambda(p_out)
        
        # Resonant Angle: phi = p*lambda_out - q*lambda_in
        # (Ignoring periapsis varpi for this MVP to keep it stable)
        phi = rule.p * lambda_out - rule.q * lambda_in
        
        # Convert to Complex Phasor: z = e^(i*phi)
        phasors[i] = exp(im * phi)
    end
    return phasors
end

# B. The Memory Update (Learning Loop)
# K_new = K + (Learning_Rate * Dissonance)
function update_kernel!(K::KernelField, current_phasors::Vector{ComplexF64}, dt::Float64)
    # The "Learning Rate" comes from the mu-nu balance.
    # If nu (Novelty) is high, we learn fast (high plasticity).
    # If mu (Memory) is high, we resist change.
    
    # Simple logistic learning rate:
    learning_rate = K.nu * (1.0 - abs(mean(K.modes))) # Slow down as we saturate
    
    for i in 1:length(K.modes)
        # Dissonance = Difference between Reality (Phasor) and Memory (Mode)
        dissonance = current_phasors[i] - K.modes[i]
        
        # Update K
        K.modes[i] += learning_rate * dissonance * dt
        
        # Normalize K (Habit strength maxes at 1.0)
        if abs(K.modes[i]) > 1.0
            K.modes[i] = K.modes[i] / abs(K.modes[i])
        end
    end
end

# C. The Manifestation (K -> Phi)
# Apply forces. 
# 1. Frozen Habit (Gravity)
# 2. Active Habit (PIT Resonance Kick)
function apply_forces!(planets::Vector{Planet}, star_mass::Float64, K::KernelField, rules::Vector{ResonanceRule})
    
    # 1. Apply Standard Gravity (The Frozen Habit)
    for i in 1:length(planets)
        p = planets[i]
        
        # Force from Star
        r_vec = p.pos
        r_mag = norm(r_vec)
        acc_star = -G * star_mass * r_vec / (r_mag^3)
        
        p.vel += acc_star # Apply acceleration (Symplectic-ish Euler)
        
        # (Ignoring planet-planet gravity for this MVP to isolate the PIT effect,
        #  but in a full run, you'd add N-body summation here)
    end
    
    # 2. Apply PIT Resonance Force (The Active Habit)
    # We apply a tangential kick to minimize phase dissonance
    current_phasors = measure_resonance_phasors(planets, rules)
    
    for (i, rule) in enumerate(rules)
        # Get the "Memory" of this resonance
        k_mem = K.modes[i]
        
        # Get "Reality"
        z_curr = current_phasors[i]
        
        # Dissonance Angle (Phase difference)
        # We want to pull z_curr towards k_mem
        delta_phi = angle(k_mem) - angle(z_curr)
        
        # Strength of the kick depends on:
        # - Coupling constant (alpha)
        # - Strength of the memory (abs(k_mem))
        # - The dissonance (sin(delta_phi))
        torque = K.alpha * abs(k_mem) * sin(delta_phi)
        
        # Apply tangential kicks (Conservation of angular momentum implies opposite kicks)
        # Inner planet gets kicked one way, outer the other, scaled by mass
        p_in = planets[rule.inner_id]
        p_out = planets[rule.outer_id]
        
        # Tangential unit vectors (approximate)
        t_in = cross([0,0,1.0], p_in.pos) / norm(p_in.pos)
        t_out = cross([0,0,1.0], p_out.pos) / norm(p_out.pos)
        
        # F = ma -> a = F/m
        # We approximate torque as a tangential force
        p_in.vel += (torque / p_in.mass) * t_in
        p_out.vel -= (torque / p_out.mass) * t_out
    end
end

# D. Symplectic Integrator Step (Drift)
function step_positions!(planets::Vector{Planet}, dt::Float64)
    for p in planets
        p.pos += p.vel * dt
    end
end

# --- 3. SETUP & EXECUTION ------------------------------------------------

function main()
    println("--- INITIALIZING PIT SIMULATION: HD 110067 ---")
    println("Ontology: Process Fractal")
    println("Logic: Complex-Valued Coherence Seeking")
    
    # 1. Setup System (HD 110067 Data)
    star_mass = 0.81 # Solar masses
    
    # Approximate masses (M_earth / M_sun)
    m_scale = 3.003e-6 
    
    # Initialize Planets (Approximate semi-major axes for resonance chain)
    # 3:2, 3:2, 3:2, 4:3, 4:3
    planets = [
        Planet(1, 5.69 * m_scale, MVector(0.0793, 0.0, 0.0), MVector(0.0, 2.0 * pi * 0.0793 / 9.114 * 365.25 / 2/pi * sqrt(G*star_mass/0.0793), 0.0), "b"), # Period ~9d
        Planet(2, 5.69 * m_scale, MVector(0.1039, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "c"), # Period ~13d
        Planet(3, 5.69 * m_scale, MVector(0.1364, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "d"), # Period ~20d
        Planet(4, 5.69 * m_scale, MVector(0.1790, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "e"), # Period ~30d
        Planet(5, 5.69 * m_scale, MVector(0.2166, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "f"), # Period ~41d
        Planet(6, 5.69 * m_scale, MVector(0.2621, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "g")  # Period ~54d
    ]
    
    # Fix velocities to be roughly circular Keplerian
    for p in planets
        v_mag = sqrt(G * star_mass / norm(p.pos))
        p.vel = MVector(0.0, v_mag, 0.0)
        # Add slight randomness to Initial Phase (Phi)
        theta = rand() * 2 * pi
        r = norm(p.pos)
        v = norm(p.vel)
        p.pos = MVector(r*cos(theta), r*sin(theta), 0.0)
        p.vel = MVector(-v*sin(theta), v*cos(theta), 0.0)
    end

    # 2. Setup PIT Kernel (The Memory)
    # Rules: b:c (3:2), c:d (3:2), d:e (3:2), e:f (4:3), f:g (4:3)
    rules = [
        ResonanceRule(1, 2, 3, 2),
        ResonanceRule(2, 3, 3, 2),
        ResonanceRule(3, 4, 3, 2),
        ResonanceRule(4, 5, 4, 3),
        ResonanceRule(5, 6, 4, 3)
    ]
    
    # Initialize K-field with random low-amplitude habits (Low Coherence)
    initial_modes = [0.01 * exp(im * rand() * 2 * pi) for _ in 1:length(rules)]
    
    # PARAMS: Goldilocks Zone
    K = KernelField(
        initial_modes,
        0.95,   # Mu (Memory) - High, wants to hold habits
        0.05,   # Nu (Novelty) - Low, allows slow adaptation
        1.0e-5  # Alpha (Coupling) - Weak PIT force (The perturbation)
    )
    
    println("Kernel Initialized. Mu=$(K.mu), Nu=$(K.nu)")
    
    # 3. The Simulation Loop (Coherence Seeking)
    dt = 0.1 # Days
    steps = 10000
    
    println("Running $steps steps...")
    
    # Data Logging
    results_phi = Float64[] # Track first resonance angle
    results_k_amp = Float64[] # Track strength of first K-habit
    
    for t in 1:steps
        # A. Update Memory (Process Fractal Step 1)
        current_phasors = measure_resonance_phasors(planets, rules)
        update_kernel!(K, current_phasors, dt)
        
        # B. Update Manifestation (Process Fractal Step 2)
        # Apply Frozen Habit (Gravity) + Active Habit (PIT)
        apply_forces!(planets, star_mass, K, rules)
        step_positions!(planets, dt)
        
        # C. Trace
        if t % 10 == 0
            push!(results_phi, angle(current_phasors[1]))
            push!(results_k_amp, abs(K.modes[1]))
        end
    end
    
    println("Simulation Complete.")
    println("Final K-Field Strength (Habit Depth): ", abs(K.modes[1]))
    
    # Save Results
    df = DataFrame(Step = 1:length(results_phi), Phi_Angle = results_phi, K_Strength = results_k_amp)
    CSV.write("pit_hd110067_trace.csv", df)
    println("Trace saved to pit_hd110067_trace.csv")
end

main()
