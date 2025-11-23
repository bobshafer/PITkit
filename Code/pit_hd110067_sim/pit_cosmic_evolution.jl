# pit_cosmic_evolution.jl
#
# PIT Cosmic History Simulator
#
# Purpose:
#   Simulate the evolution of the Global K-Field (Memory/Novelty balance)
#   from the Big Bang (t=0) to Present Day (t=13.8 Gyr).
#
#   We test if the "Emergent Constants" (Lambda, a0) match observation:
#   1. Lambda (Dark Energy) should emerge/grow as Memory (mu) saturates.
#   2. a0 (MOND scale) should evolve/decay as Novelty (nu) decreases.
#
# Dependencies:
#   ] add DataFrames CSV
#

using DataFrames
using CSV
using Printf

# --- 1. THE UNIVERSE OBJECT ----------------------------------------------

mutable struct Universe
    time_gyr::Float64     # Cosmic Time (Billions of Years)
    redshift_z::Float64   # Redshift (Observable Time)
    
    mu::Float64           # Global Memory (Stiffness)
    nu::Float64           # Global Novelty (Plasticity)
    
    # Emergent Constants
    lambda_eff::Float64   # Effective Dark Energy
    a0_eff::Float64       # Effective Acceleration Scale
end

# --- 2. THE PHYSICS ENGINE (Process Fractal) -----------------------------

function update_universe!(u::Universe, dt::Float64)
    # A. The Learning Rate (The Arrow of Time)
    # The universe learns fastest when it has both:
    #  1. Memory (mu) to build upon
    #  2. Novelty (nu) to explore with
    # This creates a Logistic Growth curve.
    learning_rate = 0.5 * u.mu * u.nu 
    
    # B. Update State
    d_mu = learning_rate * dt
    u.mu += d_mu
    u.mu = clamp(u.mu, 0.0, 1.0)
    u.nu = 1.0 - u.mu
    
    # C. Update Time (Cosmic Expansion proxy)
    u.time_gyr += dt
    
    # Simple Redshift approx: z = (1 / scale_factor) - 1
    # Assuming roughly linear expansion for this toy model (t / t_now)
    # To avoid divide by zero, we clamp min scale
    scale_factor = max(u.time_gyr / 13.8, 0.001)
    u.redshift_z = (1.0 / scale_factor) - 1.0
    
    # D. Calculate Emergent Constants (The Predictions)
    
    # Hypothesis 1: Dark Energy (Lambda) is Vacuum Stiffness (Memory)
    # As the universe ages, it gets "stiffer". 
    u.lambda_eff = u.mu^2  # Quadratic scaling (like the Lagrangian)
    
    # Hypothesis 2: a0 is the Novelty Threshold
    # In a high-Novelty universe, you need a huge acceleration to break habit.
    # In a low-Novelty universe, habits are rigid, a0 drops? 
    # Or does a0 track with H(z)? Let's try direct proportionality to Nu.
    u.a0_eff = u.nu 
end

# --- 3. SIMULATION LOOP --------------------------------------------------

function main()
    println("--- PIT COSMIC EVOLUTION SIMULATION ---")
    println("Simulating 13.8 Billion Years of Habit Formation...")
    
    # Initial State (The Big Bang / First Distinction)
    # High Novelty, Low Memory (but not zero, need a seed)
    u = Universe(0.01, 1000.0, 0.01, 0.99, 0.0, 0.0)
    
    dt = 0.01 # 10 Million Years per step
    steps = Int(13.8 / dt)
    
    # Data Logging
    trace_time = Float64[]
    trace_z = Float64[]
    trace_mu = Float64[]
    trace_lambda = Float64[]
    trace_a0 = Float64[]
    
    for i in 1:steps
        update_universe!(u, dt)
        
        if i % 10 == 0
            push!(trace_time, u.time_gyr)
            push!(trace_z, u.redshift_z)
            push!(trace_mu, u.mu)
            push!(trace_lambda, u.lambda_eff)
            push!(trace_a0, u.a0_eff)
        end
    end
    
    # Output Results
    df = DataFrame(
        Time_Gyr = trace_time,
        Redshift = trace_z,
        Mu_Memory = trace_mu,
        Lambda_DarkEnergy = trace_lambda,
        a0_MOND = trace_a0
    )
    
    filename = "pit_cosmic_history.csv"
    CSV.write(filename, df)
    println("History saved to $filename")
    println("Final State (Today):")
    @printf("  Memory (mu): %.4f\n", u.mu)
    @printf("  Dark Energy (Lambda): %.4f\n", u.lambda_eff)
    @printf("  MOND Scale (a0): %.4f\n", u.a0_eff)
end

main()
