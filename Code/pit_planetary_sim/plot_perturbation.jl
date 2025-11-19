# plot_perturbation.jl
using CSV
using DataFrames
using Plots

df = CSV.read("perturbation_results.csv", DataFrame)

# Split Data
df_newton = filter(row -> row.Universe == "Newtonian", df)
df_pit = filter(row -> row.Universe == "PIT", df)

# Create Plot
p = plot(layout=(2,1), size=(800, 600), link=:x)

# 1. Newtonian Plot
plot!(p[1], df_newton.Step, rad2deg.(df_newton.Phi), 
    label="Newtonian (No Memory)", 
    ylabel="Resonance Angle (Deg)",
    title="Universe A: Standard Gravity",
    color=:blue, lw=1, alpha=0.7)

# Add Kick Marker
vline!(p[1], [5000], label="Kick", color=:black, linestyle=:dash)

# 2. PIT Plot
plot!(p[2], df_pit.Step, rad2deg.(df_pit.Phi), 
    label="PIT (Active Memory)", 
    ylabel="Resonance Angle (Deg)",
    xlabel="Time Step",
    title="Universe B: Participatory Interface",
    color=:red, lw=1, alpha=0.7)

vline!(p[2], [5000], label="Kick", color=:black, linestyle=:dash)

# Save
savefig(p, "perturbation_comparison.png")
println("Comparison plot saved to perturbation_comparison.png")
