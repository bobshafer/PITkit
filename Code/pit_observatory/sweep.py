# sweep.py
import numpy as np
from pit_simulator import Simulator

def run_sweep(mu_vals, nu_vals, grid_shape=(64,64), steps=200, seed=12345):
    results = []
    total = len(mu_vals) * len(nu_vals)
    idx = 0
    for mu in mu_vals:
        for nu in nu_vals:
            idx += 1
            sim = Simulator(shape=grid_shape, seed=seed)
            params = {'mu': float(mu), 'nu': float(nu), 'alpha': 0.05, 'beta': 0.01, 'safe_mode': False}
            sim.run(steps=steps, params=params)
            last = max(1, int(0.2 * steps))
            coh_mean = float(np.mean(sim.history['coherence'][-last:]))
            info_mean = float(np.mean(sim.history['info_flow'][-last:]))
            ent_mean = float(np.mean(sim.history['entropy'][-last:]))
            results.append({'mu': float(mu), 'nu': float(nu),
                            'coh_mean': coh_mean, 'info_mean': info_mean, 'ent_mean': ent_mean})
            print(f"Sweep {idx}/{total}: mu={mu:.4f}, nu={nu:.4f} -> coh={coh_mean:.4f}")
    out = {'results': results, 'mu_vals': np.array(mu_vals), 'nu_vals': np.array(nu_vals)}
    np.savez('sweep_results.npz', **out)
    print('Saved sweep_results.npz')
    return out

if __name__ == '__main__':
    mu_vals = np.linspace(0.005, 0.025, 20)
    nu_vals = np.linspace(0.01, 0.013, 20)
    run_sweep(mu_vals, nu_vals, grid_shape=(32,32), steps=120)

