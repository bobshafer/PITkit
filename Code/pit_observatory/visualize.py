# visualize.py
import numpy as np
import matplotlib.pyplot as plt

def load_and_plot(npzfile='sweep_results.npz'):
    data = np.load(npzfile, allow_pickle=True)
    results = data['results'].tolist()
    mu_vals = sorted(list({r['mu'] for r in results}))
    nu_vals = sorted(list({r['nu'] for r in results}))
    mu_vals = np.array(mu_vals)
    nu_vals = np.array(nu_vals)
    coh_mat = np.zeros((len(mu_vals), len(nu_vals)))
    info_mat = np.zeros_like(coh_mat)
    ent_mat = np.zeros_like(coh_mat)
    for r in results:
        i = int(np.where(mu_vals == r['mu'])[0][0])
        j = int(np.where(nu_vals == r['nu'])[0][0])
        coh_mat[i,j] = r['coh_mean']
        info_mat[i,j] = r['info_mean']
        ent_mat[i,j] = r['ent_mean']
    fig, axes = plt.subplots(1,3, figsize=(15,4))
    im0 = axes[0].imshow(coh_mat, origin='lower', aspect='auto', cmap='plasma',
                        extent=[nu_vals[0], nu_vals[-1], mu_vals[0], mu_vals[-1]])
    axes[0].set_title('Coherence (mean last 20%)')
    axes[0].set_xlabel('nu'); axes[0].set_ylabel('mu')
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(info_mat, origin='lower', aspect='auto', cmap='viridis',
                        extent=[nu_vals[0], nu_vals[-1], mu_vals[0], mu_vals[-1]])
    axes[1].set_title('Info Flow (corr)'); axes[1].set_xlabel('nu'); axes[1].set_ylabel('mu')
    fig.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(ent_mat, origin='lower', aspect='auto', cmap='magma',
                        extent=[nu_vals[0], nu_vals[-1], mu_vals[0], mu_vals[-1]])
    axes[2].set_title('Entropy (phi)'); axes[2].set_xlabel('nu'); axes[2].set_ylabel('mu')
    fig.colorbar(im2, ax=axes[2])
    plt.tight_layout()
    plt.savefig('sweep_coherence_heatmaps.png', dpi=150)
    print('Saved sweep_coherence_heatmaps.png')

if __name__ == '__main__':
    load_and_plot()

