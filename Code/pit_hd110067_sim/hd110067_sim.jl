# ==============================================================================
# HD 110067: Planetary Resonance Simulation (PIT v10.1 Edition)
# ==============================================================================
# Based on: 
#   1. Math.md (The Laws): Lagrangian Dynamics, Gating, Dissonance
#   2. HD110067.loom.md (The System): 6-planet resonant chain
#   3. ProcessBasedPIT.loom.md (The Engine): Dual-substrate update
#
# "The universe determines itself at every moment..." - History.md
# ==============================================================================

using LinearAlgebra
using Random
using Printf

# ------------------------------------------------------------------------------
# 1.0 CONFIGURATION & CONSTANTS (The Laws of this Universe)
# ------------------------------------------------------------------------------

# Simulation Steps
const DT = 0.01         # Time step
const TOTAL_STEPS = 10000 # Extended run to see the "Exhale"
const SNAPSHOT_FREQ = 100

# Newtonian Constants
const G = 1.0           # Gravitational constant (normalized)
const M_STAR = 1000.0   # Mass of the central star

# PIT Parameters (tuned to Math.md Section 3.1)
# The Balance: Manifestation (Phi) is fast, Memory (K) is slow.
const ALPHA = 0.1       # Coupling strength (Dissonance drive: -λ)
const GAMMA = 0.005     # K-Inertia (How fast the habit track moves)
const MU    = 0.001     # Memory Depth (Resistance to leaving the track)
const NU    = 0.002     # Base Novelty (Exploration factor)

# ------------------------------------------------------------------------------
# 2.0 DATA STRUCTURES
# ------------------------------------------------------------------------------

mutable struct Planet
    id::Int
    mass::Float64
    
    # PHI STATE (Manifestation - "What Is")
    pos::Vector{Float64}
    vel::Vector{Float64}
    
    # K STATE (Memory/Potential - "How It Resonates")
    # In this sim, K represents the "Habitual Orbit" point 
    # that the planet is magnetically attracted to.
    k_pos::Vector{Float64} 
    k_vel::Vector{Float64} # The flow of the habit
end

# ------------------------------------------------------------------------------
# 3.0 THE PIT ENGINE (Math.md Implementation)
# ------------------------------------------------------------------------------

"""
    gating_function(coherence_metric)

Implements G_tau from Math.md Section 3.3.
Prevents runaway by modulating novelty based on current coherence.
"""
function gating_function(dissonance::Float64)
    # High dissonance = Low coherence.
    # We want novelty to be highest at INTERMEDIATE dissonance.
    
    # Using a simplified Gaussian window as per Math.md logic
    sigma = 1.0
    return exp(-(dissonance^2) / (2 * sigma^2))
end

"""
    step_system!(planets)

The main update loop representing one 'moment' of determining.
"""
function step_system!(planets::Vector{Planet})
    
    # A. CALCULATE NEWTONIAN FORCES (Gravity)
    forces = [zeros(2) for _ in planets]
    
    for i in 1:length(planets)
        p = planets[i]
        
        # 1. Star Gravity
        r_vec = -p.pos # Vector from planet to star (at 0,0)
        r_dist = norm(r_vec)
        f_grav = (G * M_STAR * p.mass / (r_dist^3)) * r_vec
        forces[i] += f_grav
        
        # 2. Planet-Planet Interaction (Perturbations)
        for j in 1:length(planets)
            if i != j
                pj = planets[j]
                r_rel = pj.pos - p.pos
                dist = norm(r_rel) + 1e-5 # Softening to avoid singularity
                f_inter = (G * p.mass * pj.mass / (dist^3)) * r_rel
                forces[i] += f_inter
            end
        end
    end

    # B. THE INTERFACE UPDATE (PIT Dynamics)
    for i in 1:length(planets)
        p = planets[i]
        
        # 1. Measure Dissonance (Phi vs K)
        # Math.md: ||K - F[Phi]||^2
        dissonance_vec = p.k_pos - p.pos
        dissonance_mag = norm(dissonance_vec)
        
        # 2. PIT Force (Dissonance Minimization)
        # The planet is pulled towards its "Habit" (K)
        # This acts like the stabilizing force in the resonance chain
        f_pit = ALPHA * dissonance_vec
        
        # 3. Novelty Injection (Gated)
        # Math.md: -nu * G_tau * ...
        g_val = gating_function(dissonance_mag)
        noise = randn(2) * NU * g_val
        
        # 4. Update PHI (Manifestation)
        # F = ma -> a = F/m
        total_force = forces[i] + f_pit
        acc = total_force / p.mass
        
        p.vel += acc * DT + noise
        p.pos += p.vel * DT
        
        # 5. Update K (Habit Formation)
        # Math.md: K-Kinetic + Memory
        # The "Track" (K) slowly drifts towards where the planet actually IS.
        # This is how the universe "learns" the orbit.
        k_diff = p.pos - p.k_pos
        p.k_vel += (GAMMA * k_diff) * DT  # K accelerates toward Phi
        p.k_pos += p.k_vel * DT
        
        # Apply drag to K to prevent it from oscillating wildly (Stability)
        p.k_vel *= 0.99 
    end
end

# ------------------------------------------------------------------------------
# 4.0 INITIALIZATION (HD 110067 Setup)
# ------------------------------------------------------------------------------

function init_hd110067()
    # Approximate resonant radii (normalized for simulation stability)
    # Chain: 3:2 resonances
    # r ~ T^(2/3). If T ratio is 1.5, r ratio is 1.5^(2/3) ≈ 1.31
    
    base_r = 100.0
    r_ratio = 1.31
    
    planets = Planet[]
    
    # Create 6 planets
    for i in 0:5
        radius = base_r * (r_ratio ^ i)
        mass = 1.0 # Assume earth-like/sub-neptune
        
        # Keplerian Velocity for circular orbit: v = sqrt(GM/r)
        vel_mag = sqrt(G * M_STAR / radius)
        
        # Start at random angles but organized
        angle = 2 * pi * (i / 6.0) 
        
        pos = [radius * cos(angle), radius * sin(angle)]
        vel = [-vel_mag * sin(angle), vel_mag * cos(angle)]
        
        # INITIALIZE K-FIELD
        # At t=0, the "Habit" is perfectly aligned with the state.
        # As time progresses, they will drift and interact.
        k_pos = copy(pos)
        k_vel = copy(vel)
        
        push!(planets, Planet(i+1, mass, pos, vel, k_pos, k_vel))
    end
    
    return planets
end

# ------------------------------------------------------------------------------
# 5.0 MAIN EXECUTION
# ------------------------------------------------------------------------------

function run_simulation()
    println("Initializing HD 110067 PIT Simulation...")
    println("Paradigm: Process Fractal (Math.md v10.1)")
    
    planets = init_hd110067()
    
    # Store history for analysis
    history = []
    
    println("Running evolution (10,000 steps)...")
    for t in 1:TOTAL_STEPS
        step_system!(planets)
        
        if t % SNAPSHOT_FREQ == 0
            # Log data: Step, Planet 1 Dissonance, Planet 1 Radius, P1 Pos X/Y
            dissonance = norm(planets[1].k_pos - planets[1].pos)
            radius = norm(planets[1].pos)
            
            # We also save the position to visualize the orbit later if needed
            px = planets[1].pos[1]
            py = planets[1].pos[2]
            kx = planets[1].k_pos[1]
            ky = planets[1].k_pos[2]
            
            push!(history, (t, dissonance, radius, px, py, kx, ky))
        end
    end
    
    println("Simulation Complete.")
    
    # Write CSV for external plotting
    open("hd110067_results.csv", "w") do io
        println(io, "Time,Dissonance,Radius,Phi_X,Phi_Y,K_X,K_Y")
        for h in history
            @printf(io, "%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", 
                    h[1], h[2], h[3], h[4], h[5], h[6], h[7])
        end
    end
    println("Data exported to 'hd110067_results.csv'")
end

# Fire it up
run_simulation()
