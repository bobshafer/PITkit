# pit_vacuum_wave_test.py
#
# PIT Vacuum Wave Test (Phase 2b)
# Goal: measure c_sim in a pure Φ–K vacuum coherence substrate.
#
# Linearized PIT vacuum limit:
#   ∂_tt K = (λ/γ) ∇^2 K
# so c^2 = λ/γ.
#
# We evolve K on a 2D lattice, inject a delta pulse, and measure wavefront speed.

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# PIT vacuum parameters
# -----------------------------
lam = 25.0      # λ_vac (stiffness of dissonance)
gamma = 1.0    # γ_vac (inertia of memory)
c_theory = np.sqrt(lam / gamma)

# -----------------------------
# Lattice / integration settings
# -----------------------------
N = 101          # grid size (NxN)
dx = 1.0         # spatial lattice spacing
dt = 0.1         # time step (must satisfy CFL: c*dt/dx <= 1/sqrt(2) for 2D)
steps = 800      # how long to run
CFL = c_theory * dt / dx
assert CFL <= 1/np.sqrt(2) + 1e-9, f"CFL too high: {CFL:.3f}. Lower dt or raise dx."

# -----------------------------
# Helpers
# -----------------------------
def laplacian(Z):
    """5-point Laplacian with periodic boundaries."""
    return (
        np.roll(Z,  1, axis=0) +
        np.roll(Z, -1, axis=0) +
        np.roll(Z,  1, axis=1) +
        np.roll(Z, -1, axis=1) -
        4.0 * Z
    ) / dx**2

# -----------------------------
# Initialize K field with a pulse
# -----------------------------
K_prev = np.zeros((N, N), dtype=float)
K_curr = np.zeros((N, N), dtype=float)

cx = cy = N // 2
K_curr[cx, cy] = 1.0   # delta pulse in K

# For diagnostics
centerline_history = []
radial_arrival = {}   # radius -> first arrival time
threshold = 0.05      # wavefront detection threshold

# Precompute radius grid
x = np.arange(N) - cx
y = np.arange(N) - cy
X, Y = np.meshgrid(x, y, indexing="ij")
R = np.sqrt(X**2 + Y**2)

# We will detect arrivals on integer radii shells
max_r = int(R.max())
r_shells = np.arange(1, max_r+1)

# Heatmap storage (sample K along a ray / radius index)
# We'll store |K| vs (time, radius)
heatmap = np.zeros((steps, max_r+1))

# -----------------------------
# Time evolution (2nd order wave update)
# -----------------------------
c2 = c_theory**2

for t in range(steps):
    # Wave equation update:
    # K_next = 2K_curr - K_prev + c^2 dt^2 Laplacian(K_curr)
    K_next = 2.0 * K_curr - K_prev + c2 * (dt**2) * laplacian(K_curr)

    # record radial max amplitude per shell
    for r in r_shells:
        mask = (R >= r-0.5) & (R < r+0.5)
        heatmap[t, r] = np.max(np.abs(K_curr[mask]))

        # detect first arrival
        if r not in radial_arrival and heatmap[t, r] > threshold:
            radial_arrival[r] = t

    # record a centerline snapshot for sanity
    centerline_history.append(K_curr[cx, :].copy())

    # shift
    K_prev, K_curr = K_curr, K_next

# -----------------------------
# Extract measured wave speed
# -----------------------------
r_vals = np.array(sorted(radial_arrival.keys()))
t_vals = np.array([radial_arrival[r] for r in r_vals])

# linear fit t = a r + b  =>  c_sim = 1/a  (in nodes/step)
a, b = np.polyfit(r_vals, t_vals, 1)
c_sim = 1.0 / a

print("\n--- PIT Vacuum Wave Test Results ---")
print(f"λ_vac = {lam}")
print(f"γ_vac = {gamma}")
print(f"c_theory = sqrt(λ/γ) = {c_theory:.6f} nodes/step")
print(f"c_measured = {c_sim:.6f} nodes/step")
print(f"fit slope a = {a:.3f} steps/node, intercept b = {b:.3f}\n")

# -----------------------------
# Plot 1: wavefront tracking
# -----------------------------
plt.figure(figsize=(7,5))
plt.scatter(r_vals, t_vals, s=12, label="Peak arrival times")
plt.plot(r_vals, a*r_vals+b, lw=2, label=f"Fit: c_sim={c_sim:.4f}")
plt.title("PIT Vacuum Wavefront Tracking")
plt.xlabel("Radius (node index)")
plt.ylabel("Arrival time (steps)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pit_vacuum_wavefront_tracking.png", dpi=160)

# -----------------------------
# Plot 2: propagation heatmap
# -----------------------------
plt.figure(figsize=(8,4))
plt.imshow(
    heatmap.T,
    origin="lower",
    aspect="auto",
    extent=[0, steps, 0, max_r],
)
plt.colorbar(label="|K| amplitude")
plt.title("PIT Vacuum Wave Propagation Heatmap")
plt.xlabel("Time step")
plt.ylabel("Radius (nodes)")
plt.tight_layout()
plt.savefig("pit_vacuum_propagation_heatmap.png", dpi=160)

print("Wrote:")
print("  pit_vacuum_wavefront_tracking.png")
print("  pit_vacuum_propagation_heatmap.png")

