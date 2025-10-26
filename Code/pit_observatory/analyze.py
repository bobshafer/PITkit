# analyze.py
import numpy as np

def top_k_coherence(npzfile='sweep_results.npz', k=5):
    data = np.load(npzfile, allow_pickle=True)
    results = data['results'].tolist()
    results_sorted = sorted(results, key=lambda r: r['coh_mean'], reverse=True)
    for i, r in enumerate(results_sorted[:k]):
        print(f"Top {i+1}: mu={r['mu']:.6f}, nu={r['nu']:.6f}, coh={r['coh_mean']:.6f}, info={r['info_mean']:.6f}, ent={r['ent_mean']:.6f}")

if __name__ == '__main__':
    top_k_coherence()

