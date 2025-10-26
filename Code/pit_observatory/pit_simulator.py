# pit_simulator.py
# (copy file content exactly)
import numpy as np

def gaussian_kernel(size, sigma):
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def convolve_fft(image, kernel):
    s1 = np.array(image.shape) + np.array(kernel.shape) - 1
    fshape = [int(2**np.ceil(np.log2(s))) for s in s1]
    fft_image = np.fft.rfftn(image, fshape)
    fft_kernel = np.fft.rfftn(kernel, fshape)
    conv = np.fft.irfftn(fft_image * fft_kernel, fshape)
    ks = np.array(kernel.shape)
    start = (ks - 1) // 2
    end = start + np.array(image.shape)
    sx = slice(start[0], end[0])
    sy = slice(start[1], end[1])
    return conv[sx, sy]

def shannon_entropy(array, bins=64):
    flat = array.ravel()
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return 0.0
    hist, _ = np.histogram(finite, bins=bins, density=True)
    hist = hist + 1e-12
    prob = hist / np.sum(hist)
    return -np.sum(prob * np.log2(prob))

class Simulator:
    def __init__(self, shape=(64,64), kernel_size=11, kernel_sigma=3.0, seed=None):
        self.shape = tuple(shape)
        self.rng = np.random.default_rng(seed)
        self.phi = self.rng.normal(loc=0.0, scale=0.01, size=self.shape)
        self.k   = self.rng.normal(loc=0.0, scale=0.01, size=self.shape)
        self.kernel = gaussian_kernel(kernel_size, kernel_sigma)
        # Params (safer defaults)
        self.alpha = 0.05
        self.beta  = 0.01
        self.mu    = 0.0
        self.nu    = 0.0
        # clipping bounds to maintain numeric stability (tight)
        self.clip_phi = (-1e3, 1e3)
        self.clip_k   = (-1e3, 1e3)
        # storage
        self.history = {'coherence': [], 'info_flow': [], 'entropy': []}

    def F_of_phi(self):
        return convolve_fft(self.phi, self.kernel)

    def step(self):
        Fphi = self.F_of_phi()
        dissonance = self.k - Fphi
        noise = self.rng.normal(scale=self.nu, size=self.shape)
        self.phi = self.phi - self.alpha * dissonance + noise
        self.k = self.k + self.beta * (Fphi - self.k) + self.mu * (self.k * (1.0 - self.k))
        self.phi = np.clip(self.phi, self.clip_phi[0], self.clip_phi[1])
        self.k   = np.clip(self.k, self.clip_k[0], self.clip_k[1])
        coh = -np.mean(np.abs(dissonance))
        phi_flat = self.phi.ravel()
        k_flat = self.k.ravel()
        std_phi = np.std(phi_flat)
        std_k = np.std(k_flat)
        if std_phi < 1e-12 or std_k < 1e-12 or not np.isfinite(std_phi) or not np.isfinite(std_k):
            info_flow = 0.0
        else:
            info_flow = np.corrcoef(phi_flat, k_flat)[0,1]
            if not np.isfinite(info_flow):
                info_flow = 0.0
        ent = shannon_entropy(self.phi, bins=64)
        self.history['coherence'].append(coh)
        self.history['info_flow'].append(info_flow)
        self.history['entropy'].append(ent)
        return coh, info_flow, ent

    def run(self, steps=200, params=None, safe_mode=True):
        if params:
            for k,v in params.items():
                setattr(self, k, v)
        if safe_mode:
            self.alpha = min(max(self.alpha, 0.0), 0.2)
            self.beta  = min(max(self.beta, 0.0), 0.05)
            self.mu    = min(max(self.mu, -0.2), 0.2)
            self.nu    = min(max(self.nu, 0.0), 0.2)
        for i in range(steps):
            self.step()
        return self.history

