# plot_trace.jl
using CSV
using DataFrames
using Plots

# Load the data
df = CSV.read("pit_hd110067_trace.csv", DataFrame)

# 1. Plot the Growth of Memory (K-Field Strength)
p1 = plot(df.Step, df.K_Strength, 
    label="Habit Strength (K)", 
    ylabel="|K| (0 to 1)",
    xlabel="Time Step",
    title="Formation of a Resonant Habit",
    lw=2, color=:blue, legend=:bottomright)

# 2. Plot the Phase Angle (The "Voice" of the Planets)
# We want to see this line go flat (lock) or oscillate stably.
p2 = plot(df.Step, rad2deg.(df.Phi_Angle), 
    label="Resonance Angle (Φ)", 
    ylabel="Degrees",
    xlabel="Time Step",
    title="Phase Locking",
    lw=1, color=:red, legend=:none)

# Combine
l = @layout [a; b]
final_plot = plot(p1, p2, layout=l, size=(800, 800))

# Save
savefig(final_plot, "pit_trace_plot.png")
println("Plot saved to pit_trace_plot.png")
