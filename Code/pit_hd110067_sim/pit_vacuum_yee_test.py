"""
PIT Vacuum Wave Test — 2D Yee Grid (TEz)

Goals:
- Correct Yee staggering with matching derivative shapes (no broadcasting).
- Stable for >=800 steps (no NaN/inf).
- Wave speed close to theory.
- Outputs:
    pit_yee_wavefront_tracking_stable.png
    pit_yee_propagation_heatmap_stable.png
"""

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class VacuumParams:
    lambda_vac: float = 1.0
    gamma_vac: float = 1.0
    dx: float = 1.0
    dt: float | None = None
    Nx: int = 201
    Ny: int = 201
    steps: int = 900
    sponge_width: float = 30.0
    sponge_strength: float = 0.03
    global_damp: float = 0.9995
    diffusion: float = 0.002
    source_amp: float = 0.05
    source_w_space: float = 4.0
    source_w_time: float = 15.0
    source_t0: float = 45.0

    def __post_init__(self):
        self.epsilon_eff = 1.0 / self.lambda_vac
        self.mu_eff = self.gamma_vac
        self.c_theory = np.sqrt(self.lambda_vac / self.gamma_vac)
        if self.dt is None:
            # Conservative CFL to improve stability.
            self.dt = 0.55 * self.dx / (self.c_theory * np.sqrt(2))


@dataclass
class YeeState:
    Ex: np.ndarray  # (Nx,   Ny-1)
    Ey: np.ndarray  # (Nx-1, Ny)
    Bz: np.ndarray  # (Nx-1, Ny-1)


def initialize_fields(params: VacuumParams) -> YeeState:
    Ex = np.zeros((params.Nx, params.Ny - 1), dtype=float)
    Ey = np.zeros((params.Nx - 1, params.Ny), dtype=float)
    Bz = np.zeros((params.Nx - 1, params.Ny - 1), dtype=float)
    return YeeState(Ex, Ey, Bz)


def make_sponge(params: VacuumParams):
    Xc = np.arange(params.Nx - 1)
    Yc = np.arange(params.Ny - 1)
    XXc, YYc = np.meshgrid(Xc, Yc, indexing="ij")
    dist = np.minimum.reduce([XXc, YYc, (params.Nx - 2 - XXc), (params.Ny - 2 - YYc)]).astype(float)

    sponge_B = np.ones((params.Nx - 1, params.Ny - 1), dtype=float)
    edge = dist < params.sponge_width
    sponge_B[edge] = np.exp(-params.sponge_strength * (params.sponge_width - dist[edge]))

    sponge_Ex = np.ones((params.Nx, params.Ny - 1), dtype=float)
    sponge_Ex[:-1, :] = sponge_B

    sponge_Ey = np.ones((params.Nx - 1, params.Ny), dtype=float)
    sponge_Ey[:, :-1] = sponge_B
    return sponge_Ex, sponge_Ey, sponge_B


def inject_source(Bz: np.ndarray, n: int, params: VacuumParams, cx: int, cy: int):
    g_t = params.source_amp * np.exp(-((n - params.source_t0) / params.source_w_time) ** 2)
    x = np.arange(Bz.shape[0]) - cx
    y = np.arange(Bz.shape[1]) - cy
    X, Y = np.meshgrid(x, y, indexing="ij")
    g_xy = np.exp(-((X**2 + Y**2) / (2 * params.source_w_space**2)))
    Bz += g_t * g_xy


def update_E_fields(state: YeeState, params: VacuumParams):
    # Ex update from dBz/dy -> shape (Nx-1, Ny-2)
    dBz_dy = (state.Bz[:, 1:] - state.Bz[:, :-1]) / params.dx
    state.Ex[:-1, 1:] += (params.dt / params.epsilon_eff) * dBz_dy

    # Ey update from dBz/dx -> shape (Nx-2, Ny-1)
    dBz_dx = (state.Bz[1:, :] - state.Bz[:-1, :]) / params.dx
    state.Ey[1:-1, :-1] -= (params.dt / params.epsilon_eff) * dBz_dx


def update_B_field(state: YeeState, params: VacuumParams):
    # curlE on Bz grid: dEy/dx - dEx/dy
    dEy_dx = (state.Ey[1:, :-1] - state.Ey[:-1, :-1]) / params.dx  # (Nx-2, Ny-1)
    dEx_dy = (state.Ex[:-1, 1:] - state.Ex[:-1, :-1]) / params.dx  # (Nx-1, Ny-2)
    curlE = dEy_dx[:, :-1] - dEx_dy[:-1, :]                         # (Nx-2, Ny-2)
    state.Bz[1:-1, 1:-1] -= (params.dt / params.mu_eff) * curlE


def apply_damping(state: YeeState, sponge_Ex, sponge_Ey, sponge_B, params: VacuumParams):
    state.Ex *= sponge_Ex * params.global_damp
    state.Ey *= sponge_Ey * params.global_damp
    state.Bz *= sponge_B * params.global_damp

    if params.diffusion > 0:
        B = state.Bz
        Bp = np.pad(B, ((1, 1), (1, 1)), mode="edge")
        lap = (Bp[2:, 1:-1] + Bp[:-2, 1:-1] + Bp[1:-1, 2:] + Bp[1:-1, :-2] - 4 * B)
        state.Bz = B + params.diffusion * lap


def run_simulation(params: VacuumParams):
    state = initialize_fields(params)
    sponge_Ex, sponge_Ey, sponge_B = make_sponge(params)
    cx, cy = (params.Nx - 2) // 2, (params.Ny - 2) // 2  # Bz-centered

    x = np.arange(params.Nx - 1) - cx
    y = np.arange(params.Ny - 1) - cy
    X, Y = np.meshgrid(x, y, indexing="ij")
    R = np.sqrt(X**2 + Y**2)
    max_r = int(R.max())
    shells = np.arange(1, max_r + 1)
    shell_masks = [(R >= r - 0.5) & (R < r + 0.5) for r in shells]

    heatmap = np.zeros((params.steps, len(shells)), dtype=float)
    radial_arrival: dict[int, int] = {}
    wavefront_series = np.zeros(params.steps, dtype=float)
    threshold = 2e-3

    for n in range(params.steps):
        if n < 140:
            inject_source(state.Bz, n, params, cx, cy)

        update_E_fields(state, params)
        update_B_field(state, params)
        apply_damping(state, sponge_Ex, sponge_Ey, sponge_B, params)

        shell_vals = []
        for i, (r, mask) in enumerate(zip(shells, shell_masks)):
            val = float(np.max(np.abs(state.Bz[mask])))
            heatmap[n, i] = val
            if r not in radial_arrival and val > threshold:
                radial_arrival[r] = n
            shell_vals.append(val)

        if np.sum(shell_vals) > 0:
            wavefront_series[n] = shells[int(np.argmax(shell_vals))]

        if np.isnan(state.Bz).any() or np.isinf(state.Bz).any():
            raise RuntimeError(f"Numerical instability at step {n}")

    return state, heatmap, radial_arrival, wavefront_series, shells


def fit_wave_speed_from_series(r_fit: np.ndarray, t_fit: np.ndarray, c_theory: float):
    slope_rt, intercept_r = np.polyfit(t_fit, r_fit, 1)  # r = v*t + b
    c_measured = slope_rt
    rel_err = abs(c_measured - c_theory) / c_theory
    slope_t = 1.0 / slope_rt if slope_rt != 0 else np.inf
    intercept_t = -intercept_r / slope_rt if slope_rt != 0 else 0.0
    return c_measured, slope_t, intercept_t, rel_err


def plot_wavefront_tracking(r_vals, t_vals, slope_t, intercept_t, c_measured):
    plt.figure(figsize=(7, 5))
    plt.scatter(r_vals, t_vals, s=12, label="Wavefront radii")
    plt.plot(r_vals, slope_t * r_vals + intercept_t, lw=2, label=f"Fit: c_sim={c_measured:.3f}")
    plt.title("PIT Vacuum Wavefront Tracking (Yee Grid, stable)")
    plt.xlabel("Radius (node index)")
    plt.ylabel("Time (steps)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pit_yee_wavefront_tracking_stable.png", dpi=160)
    plt.close()


def plot_heatmap(heatmap, shells):
    plt.figure(figsize=(8, 4))
    plt.imshow(
        heatmap.T,
        origin="lower",
        aspect="auto",
        extent=[0, heatmap.shape[0], shells[0], shells[-1]],
    )
    plt.colorbar(label="|Bz| amplitude")
    plt.title("PIT Vacuum Wave Propagation Heatmap (Yee Grid, stable)")
    plt.xlabel("Time step")
    plt.ylabel("Radius (nodes)")
    plt.tight_layout()
    plt.savefig("pit_yee_propagation_heatmap_stable.png", dpi=160)
    plt.close()


def main():
    params = VacuumParams()
    print("\n--- PIT Yee Vacuum Test (stable) ---")
    print(f"lambda_vac={params.lambda_vac}, gamma_vac={params.gamma_vac}")
    print(f"epsilon_eff=1/lambda={params.epsilon_eff:.6g}, mu_eff=gamma={params.mu_eff:.6g}")
    print(f"c_theory=sqrt(lambda/gamma)={params.c_theory:.6f} nodes/step")
    print(f"dt = {params.dt:.6f} (CFL guard 0.55/sqrt(2))")
    print(f"steps = {params.steps}\n")

    state, heatmap, radial_arrival, wavefront_series, shells = run_simulation(params)

    valid = np.where(wavefront_series > 0)[0]
    if len(valid) < 10:
        raise RuntimeError("Wavefront detection failed; insufficient data.")
    t_fit = valid
    r_fit = wavefront_series[valid]
    c_measured, slope_t, intercept_t, rel_err = fit_wave_speed_from_series(r_fit, t_fit, params.c_theory)

    print("--- Results ---")
    print(f"c_measured = {c_measured:.6f} nodes/step")
    print(f"slope = {slope_t:.6f} steps/node")
    print(f"relative error = {rel_err*100:.2f}%\n")

    for name, arr in [("Ex", state.Ex), ("Ey", state.Ey), ("Bz", state.Bz)]:
        assert not np.isnan(arr).any(), f"{name} has NaN"
        assert not np.isinf(arr).any(), f"{name} has inf"

    plot_wavefront_tracking(r_fit, t_fit, slope_t, intercept_t, c_measured)
    plot_heatmap(heatmap, shells)

    print("Wrote:")
    print("  pit_yee_wavefront_tracking_stable.png")
    print("  pit_yee_propagation_heatmap_stable.png")


if __name__ == "__main__":
    main()
