# plot_hd110067.jl
using CSV
using DataFrames
using Plots
using Statistics

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------
csv_file = "hd110067_results.csv"
output_file = "hd110067_analysis.png"

# ------------------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------------------
println("Reading $csv_file...")
df = CSV.read(csv_file, DataFrame)

# ------------------------------------------------------------------------------
# PLOT 1: THE BREATH OF DISSONANCE (Time Series)
# ------------------------------------------------------------------------------
# This visualizes the "Learning Arc" described in Math.md Section 8.1
# We expect to see a rise (inhale) as K stretches, then stabilization (exhale).

p1 = plot(df.Time, df.Dissonance,
    title = "The Breath of Dissonance (Coherence Seeking)",
    ylabel = "Dissonance ||K - Φ||",
    xlabel = "Time Step",
    label = "Dissonance",
    color = :purple,
    lw = 1.5,
    legend = :topright,
    grid = true,
    gridalpha = 0.3,
    margin = 5Plots.mm
)

# Add a smoothing line to show the trend better (if you have many points)
# This represents the "Habit" forming
if length(df.Dissonance) > 50
    # Simple moving average or just plotting the trend
    plot!(p1, df.Time, fill(mean(df.Dissonance), length(df.Time)), 
          label="Mean Coherence", color=:black, linestyle=:dash)
end

# ------------------------------------------------------------------------------
# PLOT 2: THE SURFACE OF BECOMING (Phase Space / Orbit)
# ------------------------------------------------------------------------------
# This visualizes the "Fuzzy Donut" or the "Track"
# We plot the actual Planet (Phi) vs its Memory (K)

# We'll take the last 2000 steps to see the stable orbit, 
# otherwise the spiral-in (migration) might make it look messy.
tail_len = min(2000, nrow(df))
df_tail = last(df, tail_len)

p2 = plot(
    df_tail.Phi_X, df_tail.Phi_Y,
    title = "Orbit Trace: Matter (Φ) vs Habit (K)",
    xlabel = "Position X",
    ylabel = "Position Y",
    label = "Planet (Φ - Manifestation)",
    seriestype = :scatter,
    markersize = 1.5,
    markeralpha = 0.4,
    color = :blue,
    aspect_ratio = :equal
)

# Overlay the K-Field (The Habit Track)
plot!(p2, 
    df_tail.K_X, df_tail.K_Y,
    label = "Habit Track (K - Memory)",
    linewidth = 2.0,
    color = :red,
    alpha = 0.6
)

# ------------------------------------------------------------------------------
# COMBINE AND SAVE
# ------------------------------------------------------------------------------
# Layout: Dissonance on top, Orbit on bottom
l = @layout [a; b{0.6h}] # Give the orbit map a bit more height

final_plot = plot(p1, p2, layout=l, size=(800, 1000))

println("Saving plot to $output_file...")
savefig(final_plot, output_file)
println("Done!")
