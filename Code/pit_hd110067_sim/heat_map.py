import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Experiment A: The Breaking Point (Phase Diagram)
# ==============================================================================

# CONFIGURATION
ALPHA_STEPS = 20
NU_STEPS = 20
GRID_ALPHA = np.linspace(0.0, 0.3, ALPHA_STEPS)
GRID_NU = np.linspace(0.0, 0.015, NU_STEPS)
STEPS_PER_PIXEL = 2000 # Short run to check stability
DT = 0.01
G = 1.0
M_STAR = 1000.0
GAMMA = 0.005

# ... [Planet Class & Engine Logic same as hd110067_adaptive.py but with fixed Alpha] ...

def run_sweep():
    heatmap_data = np.zeros((NU_STEPS, ALPHA_STEPS))
    
    for i, nu in enumerate(GRID_NU):
        for j, alpha in enumerate(GRID_ALPHA):
            # Run simulation for this specific (alpha, nu) pair
            max_diss = run_simulation_cell(alpha, nu)
            heatmap_data[i, j] = max_diss
            
    # Plotting
    plt.imshow(heatmap_data, origin='lower', aspect='auto',
               extent=[GRID_ALPHA.min(), GRID_ALPHA.max(), GRID_NU.min(), GRID_NU.max()],
               cmap='viridis_r', vmin=0, vmax=20)
    plt.xlabel('Coupling Strength (α)')
    plt.ylabel('Noise Level (ν)')
    plt.title('Phase Transition of Coherence')
    plt.colorbar(label='Max Dissonance')
    plt.savefig('breaking_point_heatmap.png')

if __name__ == "__main__":
    run_sweep()
