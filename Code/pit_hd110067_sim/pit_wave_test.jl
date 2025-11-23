# pit_wave_test.jl
#
# PIT Wave Propagation Test
#
# Purpose:
#   Measure the speed of a "Coherence Wave" (Signal) propagating through
#   a 1D chain of PIT-coupled nodes. This gives us c_sim.
#
# Setup:
#   - Chain of 50 nodes.
#   - Neighbor Coupling: 1:1 Resonance (Phase Locking).
#   - Physics: Process Fractal (Phi -> K -> Force -> Phi).
#
# Dependencies:
#   ] add DataFrames CSV
#

using Statistics
using DataFrames
using CSV
using Printf

# --- 1. ONTOLOGY ---------------------------------------------------------

mutable struct Node
    id::Int
    phi::Float64      # Phase Angle (Position)
    vel::Float64      # Phase Velocity
    mass::Float64     # Inertia
end

mutable struct Kernel
    # Stores the "Habit" of the phase difference between neighbors.
    # For 1:1 resonance, habit is usually 0 (in sync).
    modes::Vector{ComplexF64} 
    
    # Parameters
    alpha::Float64  # Stiffness (Coupling)
    nu::Float64     # Plasticity (Learning Rate)
end

# --- 2. PHYSICS ENGINE ---------------------------------------------------

function update_kernel!(K::Kernel, nodes::Vector{Node}, dt::Float64)
    # The K-field learns the phase relationship between neighbors.
    # There are N-1 links for N nodes.
    
    n_links = length(nodes) - 1
    if length(K.modes) != n_links
        resize!(K.modes, n_links)
        fill!(K.modes, 1.0 + 0.0im) # Init with sync habit
    end
    
    for i in 1:n_links
        # Measure Reality (Phi)
        # Phase Diff: Node[i+1] - Node[i]
        delta_phi = nodes[i+1].phi - nodes[i].phi
        z_meas = exp(im * delta_phi)
        
        # Measure Habit (K)
        z_mem = K.modes[i]
        
        # Dissonance
        dissonance = z_meas - z_mem
        
        # Learning (Simple Relaxation)
        learning_rate = K.nu
        K.modes[i] += learning_rate * dissonance * dt
        
        # Normalize
        if abs(K.modes[i]) > 0.0
            K.modes[i] /= abs(K.modes[i])
        end
    end
end

function apply_forces!(nodes::Vector{Node}, K::Kernel)
    # Forces come from the K-field trying to enforce the habit.
    n_links = length(nodes) - 1
    
    # Reset forces (velocity changes will be applied directly)
    acc = zeros(length(nodes))
    
    for i in 1:n_links
        # Link i connects Node i and Node i+1
        z_mem = K.modes[i]
        
        # Current Reality
        delta_phi = nodes[i+1].phi - nodes[i].phi
        z_curr = exp(im * delta_phi)
        
        # Phase Error (Angle diff between Habit and Reality)
        error = angle(z_mem) - angle(z_curr)
        
        # Force ~ Alpha * Sin(Error)
        # Restoring force tries to minimize error.
        force = K.alpha * sin(error)
        
        # Newton's 3rd Law: Equal & Opposite
        # If error > 0 (Reality is "behind" Memory?), pull forward.
        # Actually: if z_mem is ahead of z_curr, we want to increase z_curr.
        # We treat 'force' as a torque on the bond.
        
        acc[i+1] += force / nodes[i+1].mass
        acc[i]   -= force / nodes[i].mass
    end
    
    return acc
end

# --- 3. SIMULATION -------------------------------------------------------

function main()
    println("--- PIT WAVE TEST ---")
    
    # Constants
    N = 50
    DT = 0.1
    STEPS = 2000
    
    # Parameters (Must match our "Stable" regime from HD110067)
    ALPHA = 0.15    # Stiffness (Coupling) - From Heatmap Green Zone
    NU = 0.01       # Learning Rate
    MASS = 1.0      # Unit Mass
    
    # Init Chain
    nodes = [Node(i, 0.0, 0.0, MASS) for i in 1:N]
    
    # Init Kernel (N-1 links)
    # Initialize perfectly synced (Habit = 0 phase diff)
    K = Kernel(fill(1.0 + 0.0im, N-1), ALPHA, NU)
    
    # Data Log
    # We will track the velocity of every node to visualize the wave
    trace_data = zeros(STEPS, N)
    
    println("Running simulation...")
    
    for t in 1:STEPS
        
        # --- THE PULSE ---
        if t == 50
            # Kick Node 1
            nodes[1].vel += 1.0
            println("  Pulse injected at t=$t")
        end
        
        # 1. Update K (Memory)
        update_kernel!(K, nodes, DT)
        
        # 2. Calculate Forces (Manifestation)
        acc = apply_forces!(nodes, K)
        
        # 3. Update Physics (Symplectic Euler)
        for i in 1:N
            nodes[i].vel += acc[i] * DT
            nodes[i].phi += nodes[i].vel * DT
            
            # Log velocity (kinetic energy proxy)
            trace_data[t, i] = nodes[i].vel
        end
    end
    
    # Save CSV
    # Format: Step, Node1_Vel, Node2_Vel, ...
    df = DataFrame(trace_data, :auto)
    rename!(df, [Symbol("Node_$i") for i in 1:N])
    df[!, :Step] = 1:STEPS
    
    CSV.write("wave_test_results.csv", df)
    println("Wave trace saved to wave_test_results.csv")
end

main()
