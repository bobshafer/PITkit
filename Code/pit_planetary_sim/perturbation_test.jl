# perturbation_test.jl
#
# PIT vs. Newtonian: The "Kick" Test
#
# Purpose:
#   Run two parallel simulations of the HD 110067 system.
#   1. Universe A (Newtonian): Standard Gravity (alpha = 0).
#   2. Universe B (PIT): Coherence Seeking (alpha = 1.0e-5).
#
#   Both run for 5000 steps to "settle".
#   At step 5000, Planet C receives a violent "Kick" (velocity perturbation).
#   We observe if the "Memory" (K-field) in Universe B helps it recover
#   resonance faster than Universe A.
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

# --- 1. SHARED PHYSICS ENGINE --------------------------------------------
# (Identical to main.jl)

mutable struct Planet
    id::Int
    mass::Float64
    pos::MVector{3, Float64}
    vel::MVector{3, Float64}
    name::String
end

mutable struct KernelField
    modes::Vector{ComplexF64} 
    mu::Float64
    nu::Float64
    alpha::Float64 
end

struct ResonanceRule
    inner_id::Int
    outer_id::Int
    p::Int
    q::Int
end

const G = 2.95912208286e-4 

function get_lambda(p::Planet)
    x, y = p.pos[1], p.pos[2]
    return atan(y, x)
end

function measure_resonance_phasors(planets::Vector{Planet}, rules::Vector{ResonanceRule})
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
    learning_rate = K.nu * (1.0 - abs(mean(K.modes)))
    for i in 1:length(K.modes)
        dissonance = current_phasors[i] - K.modes[i]
        K.modes[i] += learning_rate * dissonance * dt
        if abs(K.modes[i]) > 1.0
            K.modes[i] = K.modes[i] / abs(K.modes[i])
        end
    end
end

function apply_forces!(planets::Vector{Planet}, star_mass::Float64, K::KernelField, rules::Vector{ResonanceRule})
    # 1. Standard Gravity
    for p in planets
        r_vec = p.pos
        r_mag = norm(r_vec)
        acc_star = -G * star_mass * r_vec / (r_mag^3)
        p.vel += acc_star
    end
    
    # 2. PIT Resonance Force (Only if alpha > 0)
    if K.alpha > 0.0
        current_phasors = measure_resonance_phasors(planets, rules)
        for (i, rule) in enumerate(rules)
            k_mem = K.modes[i]
            z_curr = current_phasors[i]
            delta_phi = angle(k_mem) - angle(z_curr)
            torque = K.alpha * abs(k_mem) * sin(delta_phi)
            
            p_in = planets[rule.inner_id]
            p_out = planets[rule.outer_id]
            t_in = cross([0,0,1.0], p_in.pos) / norm(p_in.pos)
            t_out = cross([0,0,1.0], p_out.pos) / norm(p_out.pos)
            
            p_in.vel += (torque / p_in.mass) * t_in
            p_out.vel -= (torque / p_out.mass) * t_out
        end
    end
end

function step_positions!(planets::Vector{Planet}, dt::Float64)
    for p in planets
        p.pos += p.vel * dt
    end
end

# --- 2. SIMULATION RUNNER ------------------------------------------------

function run_universe(label::String, alpha_val::Float64, kick_step::Int, kick_mag::Float64)
    println("Initializing Universe: $label (Alpha=$alpha_val)...")
    
    # Setup System (Identical Initial Conditions for Fairness)
    Random.seed!(42) # Ensure both universes start exactly the same
    
    star_mass = 0.81
    m_scale = 3.003e-6 
    
    planets = [
        Planet(1, 5.69 * m_scale, MVector(0.0793, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "b"),
        Planet(2, 5.69 * m_scale, MVector(0.1039, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "c"),
        Planet(3, 5.69 * m_scale, MVector(0.1364, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "d"),
        Planet(4, 5.69 * m_scale, MVector(0.1790, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "e"),
        Planet(5, 5.69 * m_scale, MVector(0.2166, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "f"),
        Planet(6, 5.69 * m_scale, MVector(0.2621, 0.0, 0.0), MVector(0.0, 0.0, 0.0), "g")
    ]
    
    for p in planets
        v_mag = sqrt(G * star_mass / norm(p.pos))
        theta = rand() * 2 * pi
        r = norm(p.pos)
        p.pos = MVector(r*cos(theta), r*sin(theta), 0.0)
        p.vel = MVector(-v_mag*sin(theta), v_mag*cos(theta), 0.0)
    end

    rules = [
        ResonanceRule(1, 2, 3, 2), ResonanceRule(2, 3, 3, 2),
        ResonanceRule(3, 4, 3, 2), ResonanceRule(4, 5, 4, 3),
        ResonanceRule(5, 6, 4, 3)
    ]
    
    # Pre-load K-field (Simulating an "Old" system with habits)
    # In Newtonian run, alpha=0, so this memory is ignored (latent).
    # In PIT run, alpha>0, so this memory is active.
    initial_modes = [0.01 * exp(im * rand() * 2 * pi) for _ in 1:length(rules)]
    K = KernelField(initial_modes, 0.95, 0.05, alpha_val)
    
    dt = 0.1
    total_steps = 8000
    
    history_phi = Float64[]
    history_step = Int[]
    
    for t in 1:total_steps
        
        # --- THE KICK ---
        if t == kick_step
            println("  [!] KICKING Planet C in Universe $label at step $t")
            # Apply a sudden velocity change to Planet C (index 2)
            # Kick direction: Tangential (to speed it up/slow it down and break sync)
            p = planets[2]
            v_unit = p.vel / norm(p.vel)
            p.vel += v_unit * kick_mag * norm(p.vel) # Percentage kick
        end
        
        # Physics Loop
        current_phasors = measure_resonance_phasors(planets, rules)
        update_kernel!(K, current_phasors, dt)
        apply_forces!(planets, star_mass, K, rules)
        step_positions!(planets, dt)
        
        # Logging (Track the b:c resonance angle)
        if t % 10 == 0
            push!(history_step, t)
            push!(history_phi, angle(current_phasors[1]))
        end
    end
    
    return DataFrame(Step = history_step, Phi = history_phi, Universe = label)
end

# --- 3. MAIN EXECUTION ---------------------------------------------------

function main()
    # 1. Run Newtonian (Control)
    # Alpha = 0.0
    # Kick = 2% velocity boost at step 5000
    df_newton = run_universe("Newtonian", 0.0, 5000, 0.02)
    
    # 2. Run PIT (Experimental)
    # Alpha = 1.0e-5
    # Kick = 2% velocity boost at step 5000
    df_pit = run_universe("PIT", 1.0e-5, 5000, 0.02)
    
    # 3. Save Combined Data
    df_total = vcat(df_newton, df_pit)
    CSV.write("perturbation_results.csv", df_total)
    println("Comparison saved to perturbation_results.csv")
end

main()
