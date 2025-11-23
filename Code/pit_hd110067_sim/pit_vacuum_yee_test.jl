"""
PIT Vacuum Wave Test — 2D Yee-grid FDTD (Julia)

Goals
- Correct Yee staggering (Ex: Nx×(Ny-1), Ey: (Nx-1)×Ny, Bz: (Nx-1)×(Ny-1)).
- Stable run for ≥800 steps (no NaN/Inf) with sponge + mild damping.
- Wave speed near theory (c_theory = sqrt(lambda_vac/gamma_vac)).
- Outputs:
    pit_yee_wavefront_tracking_stable.png
    pit_yee_propagation_heatmap_stable.png

Note: plotting uses PyPlot; ensure it is available in your Julia environment.
"""

using LinearAlgebra
using Statistics
using DelimitedFiles
HAVE_PYPLOT = false
try
    using PyPlot
    global HAVE_PYPLOT = true
catch
    @warn "PyPlot not available; skipping PNG generation."
end

# -----------------------------
# Parameters
# -----------------------------
struct VacuumParams
    lambda_vac::Float64
    gamma_vac::Float64
    dx::Float64
    dt::Float64
    Nx::Int
    Ny::Int
    steps::Int
    sponge_width::Int
    sponge_strength::Float64
    global_damp::Float64
    source_amp::Float64
    source_w_space::Float64
    source_w_time::Float64
    source_t0::Float64
end

function VacuumParams(; lambda_vac=1.0, gamma_vac=1.0, dx=1.0, Nx=201, Ny=201, steps=900,
    sponge_width=25, sponge_strength=0.03, global_damp=0.999,
    source_amp=0.05, source_w_space=4.0, source_w_time=15.0, source_t0=45.0,
    dt=nothing)
    c_th = sqrt(lambda_vac / gamma_vac)
    dt === nothing && (dt = 0.30 * dx / (c_th * sqrt(2))) # conservative CFL
    VacuumParams(lambda_vac, gamma_vac, dx, dt, Nx, Ny, steps, sponge_width,
        sponge_strength, global_damp, source_amp, source_w_space, source_w_time, source_t0)
end

epsilon_eff(p::VacuumParams) = 1 / p.lambda_vac
mu_eff(p::VacuumParams) = p.gamma_vac
c_theory(p::VacuumParams) = sqrt(p.lambda_vac / p.gamma_vac)

# -----------------------------
# Field allocation
# -----------------------------
struct YeeState
    Ex::Matrix{Float64}  # (Nx, Ny-1)
    Ey::Matrix{Float64}  # (Nx-1, Ny)
    Bz::Matrix{Float64}  # (Nx-1, Ny-1)
end

function init_fields(p::VacuumParams)
    Ex = zeros(p.Nx, p.Ny - 1)
    Ey = zeros(p.Nx - 1, p.Ny)
    Bz = zeros(p.Nx - 1, p.Ny - 1)
    YeeState(Ex, Ey, Bz)
end

# -----------------------------
# Sponge / damping masks
# -----------------------------
function make_sponge(p::VacuumParams)
    Xc = repeat(collect(0:p.Nx-2), 1, p.Ny - 1)
    Yc = repeat(collect(0:p.Ny-2)', p.Nx - 1, 1)
    dist = min.(Xc, Yc, (p.Nx - 2) .- Xc, (p.Ny - 2) .- Yc)
    sponge_B = ones(p.Nx - 1, p.Ny - 1)
    edge = dist .< p.sponge_width
    sponge_B[edge] .= exp.(-p.sponge_strength .* (p.sponge_width .- dist[edge]))

    sponge_Ex = ones(p.Nx, p.Ny - 1)
    sponge_Ex[1:end-1, :] .= sponge_B

    sponge_Ey = ones(p.Nx - 1, p.Ny)
    sponge_Ey[:, 1:end-1] .= sponge_B
    return sponge_Ex, sponge_Ey, sponge_B
end

# -----------------------------
# Source
# -----------------------------
function inject_source!(Bz, n, p::VacuumParams, cx, cy)
    g_t = p.source_amp * exp(-((n - p.source_t0) / p.source_w_time)^2)
    x = collect(0:size(Bz, 1)-1) .- cx
    y = collect(0:size(Bz, 2)-1) .- cy
    X = repeat(x, 1, length(y))
    Y = repeat(y', length(x), 1)
    g_xy = exp.(-((X .^ 2 .+ Y .^ 2) ./ (2 * p.source_w_space^2)))
    @. Bz += g_t * g_xy
end

# -----------------------------
# Updates (Yee staggered)
# -----------------------------
function update_E!(state::YeeState, p::VacuumParams)
    dBz_dy = (state.Bz[:, 2:end] .- state.Bz[:, 1:end-1]) ./ p.dx        # (Nx-1, Ny-2)
    state.Ex[1:end-1, 2:end] .+= (p.dt / epsilon_eff(p)) .* dBz_dy

    dBz_dx = (state.Bz[2:end, :] .- state.Bz[1:end-1, :]) ./ p.dx        # (Nx-2, Ny-1)
    state.Ey[2:end, 1:end-1] .-= (p.dt / epsilon_eff(p)) .* dBz_dx
end

function update_B!(state::YeeState, p::VacuumParams)
    dEy_dx = (state.Ey[2:end, 1:end-1] .- state.Ey[1:end-1, 1:end-1]) ./ p.dx  # (Nx-2, Ny-1)
    dEx_dy = (state.Ex[1:end-1, 2:end] .- state.Ex[1:end-1, 1:end-1]) ./ p.dx  # (Nx-1, Ny-2)
    curlE = dEy_dx[:, 1:end-1] .- dEx_dy[1:end-1, :]                            # (Nx-2, Ny-2)
    state.Bz[2:end, 2:end] .-= (p.dt / mu_eff(p)) .* curlE
end

function apply_damping!(state::YeeState, sponge_Ex, sponge_Ey, sponge_B, p::VacuumParams)
    @. state.Ex = state.Ex * sponge_Ex * p.global_damp
    @. state.Ey = state.Ey * sponge_Ey * p.global_damp
    @. state.Bz = state.Bz * sponge_B * p.global_damp
end

# -----------------------------
# Simulation
# -----------------------------
function run_simulation(p::VacuumParams)
    state = init_fields(p)
    sponge_Ex, sponge_Ey, sponge_B = make_sponge(p)
    cx, cy = (p.Nx - 2) ÷ 2, (p.Ny - 2) ÷ 2  # center on Bz grid

    # radial diagnostics on Bz grid
    x = collect(0:p.Nx-2) .- cx
    y = collect(0:p.Ny-2) .- cy
    X = repeat(x, 1, length(y))
    Y = repeat(y', length(x), 1)
    R = sqrt.(X .^ 2 .+ Y .^ 2)
    max_r = floor(Int, maximum(R))
    shells = collect(1:max_r)
    shell_masks = [(R .>= r - 0.5) .& (R .< r + 0.5) for r in shells]

    heatmap = zeros(p.steps, length(shells))
    wavefront = zeros(p.steps)
    radial_arrival = Dict{Int,Int}()
    threshold = 1e-3

    for n in 1:p.steps
        if n <= 140
            inject_source!(state.Bz, n, p, cx, cy)
        end

        update_E!(state, p)
        update_B!(state, p)
        apply_damping!(state, sponge_Ex, sponge_Ey, sponge_B, p)

        shell_vals = similar(shells, Float64)
        @inbounds for (i, mask) in enumerate(shell_masks)
            val = maximum(abs.(state.Bz[mask]))
            heatmap[n, i] = val
            shell_vals[i] = val
            if !haskey(radial_arrival, shells[i]) && val > threshold
                radial_arrival[shells[i]] = n
            end
        end

        if sum(shell_vals) > 0
            wavefront[n] = shells[argmax(shell_vals)]
        end

        if any(isnan, state.Bz) || any(isinf, state.Bz)
            error("Numerical instability at step $n")
        end
    end

    return state, heatmap, wavefront, radial_arrival, shells
end

# -----------------------------
# Wave speed fit and plots
# -----------------------------
function fit_wave_speed_from_series(r_fit::Vector{Float64}, t_fit::Vector{Float64}, c_th::Float64)
    slope_rt, intercept_r = fit_line(t_fit, r_fit) # r = v*t + b
    c_meas = slope_rt
    rel_err = abs(c_meas - c_th) / c_th
    slope_t = 1 / slope_rt
    intercept_t = -intercept_r / slope_rt
    return c_meas, slope_t, intercept_t, rel_err
end

fit_line(x, y) = begin
    X = [ones(length(x)) x]
    coef = X \ y
    intercept, slope = coef[1], coef[2]
    return slope, intercept
end

function plot_wavefront_tracking(r_vals, t_vals, slope_t, intercept_t, c_measured)
    HAVE_PYPLOT || return
    figure(figsize=(7,5))
    scatter(r_vals, t_vals, s=12, label="Wavefront radii")
    plot(r_vals, slope_t .* r_vals .+ intercept_t, lw=2, label="Fit: c=$(round(c_measured, digits=4))")
    title("PIT Vacuum Wavefront Tracking (Yee Grid, stable)")
    xlabel("Radius (node index)")
    ylabel("Time (steps)")
    grid(true)
    legend()
    tight_layout()
    savefig("pit_yee_wavefront_tracking_stable.png", dpi=160)
    close()
end

function plot_heatmap(heatmap, shells)
    HAVE_PYPLOT || return
    figure(figsize=(8,4))
    imshow(permutedims(heatmap), origin="lower", aspect="auto",
           extent=[1, size(heatmap, 1), shells[1], shells[end]])
    colorbar(label="|Bz| amplitude")
    title("PIT Vacuum Wave Propagation Heatmap (Yee Grid, stable)")
    xlabel("Time step")
    ylabel("Radius (nodes)")
    tight_layout()
    savefig("pit_yee_propagation_heatmap_stable.png", dpi=160)
    close()
end

# -----------------------------
# Main
# -----------------------------
function main()
    p = VacuumParams()
    println("\n--- PIT Yee Vacuum Test (Julia) ---")
    println("lambda_vac=$(p.lambda_vac), gamma_vac=$(p.gamma_vac)")
    println("epsilon_eff=1/lambda=$(1/p.lambda_vac), mu_eff=gamma=$(p.gamma_vac)")
    println("c_theory=$(c_theory(p)) nodes/step")
    println("dt=$(p.dt) (CFL guard 0.30/sqrt(2))")
    println("steps=$(p.steps)\n")

    state, heatmap, wavefront, radial_arrival, shells = run_simulation(p)

    r_sorted = sort(collect(keys(radial_arrival)))
    t_sorted = [radial_arrival[r] for r in r_sorted]
    if length(r_sorted) < 10
        error("Wavefront detection failed; insufficient arrival data.")
    end
    slope_t, intercept_t = fit_line(r_sorted, t_sorted)   # t = slope*r + b
    c_measured = 1 / slope_t
    rel_err = abs(c_measured - c_theory(p)) / c_theory(p)
    r_fit = Float64.(r_sorted)
    t_fit = Float64.(t_sorted)

    println("--- Results ---")
    println("c_measured = $(round(c_measured, digits=6)) nodes/step")
    println("relative error = $(round(rel_err*100, digits=2))%")

    for (name, arr) in [("Ex", state.Ex), ("Ey", state.Ey), ("Bz", state.Bz)]
        any(isnan, arr) && error("$name has NaN")
        any(isinf, arr) && error("$name has Inf")
    end

    plot_wavefront_tracking(r_fit, t_fit, slope_t, intercept_t, c_measured)
    plot_heatmap(heatmap, shells)

    # CSV outputs for offline plotting
    writedlm("pit_yee_wavefront_series.csv", [t_fit r_fit], ',')
    writedlm("pit_yee_heatmap.csv", heatmap, ',')
    r_sorted = sort(collect(keys(radial_arrival)))
    t_sorted = [radial_arrival[r] for r in r_sorted]
    writedlm("pit_yee_radial_arrivals.csv", [r_sorted t_sorted], ',')

    println("Wrote:")
    println("  pit_yee_wavefront_series.csv")
    println("  pit_yee_heatmap.csv")
    println("  pit_yee_radial_arrivals.csv")
    if HAVE_PYPLOT
        println("  pit_yee_wavefront_tracking_stable.png")
        println("  pit_yee_propagation_heatmap_stable.png")
    else
        println("Skipped PNGs (PyPlot not available).")
    end
end

main()
