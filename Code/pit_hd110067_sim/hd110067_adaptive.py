# hd110067_adaptive.py
# PIT HD 110067 Simulation Engine (Adaptive / Process Fractal) — Python port
#
# Mirrors hd110067_adaptive.jl + Addendum v0.1 refinements
# - Φ planets in AU / days
# - K resonant modes as complex phasors
# - Adaptive alpha(t), mu(t)
# - dt-consistent gravity + PIT torque
#
# Outputs:
#   hd110067_adaptive_trace_py.csv
#   hd110067_adaptive_trace_py.png

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

G = 2.95912208286e-4  # AU^3 / (Solar Mass * Day^2)


# -----------------------------
# Structures
# -----------------------------

class Planet:
    def __init__(self, pid, mass, pos, vel, name):
        self.id = pid
        self.mass = float(mass)
        self.pos = np.array(pos, dtype=float)  # shape (3,)
        self.vel = np.array(vel, dtype=float)  # shape (3,)
        self.name = name


class KernelField:
    def __init__(
        self,
        modes,
        mu=0.9,
        nu=0.05,
        alpha=1e-5,
        k_alpha=1e-4,
        k_mu=1e-5,
        d_target=0.10,
        alpha_bounds=(1e-6, 0.2),
        mu_bounds=(0.01, 1.0),
    ):
        self.modes = np.array(modes, dtype=np.complex128)
        self.mu = float(mu)
        self.nu = float(nu)
        self.alpha = float(alpha)
        self.k_alpha = float(k_alpha)
        self.k_mu = float(k_mu)
        self.d_target = float(d_target)
        self.alpha_bounds = alpha_bounds
        self.mu_bounds = mu_bounds


class ResonanceRule:
    def __init__(self, inner_id, outer_id, p, q):
        self.inner_id = inner_id  # 1-based
        self.outer_id = outer_id  # 1-based
        self.p = int(p)
        self.q = int(q)


# -----------------------------
# Helpers
# -----------------------------

def get_lambda(p: Planet):
    """Mean longitude approximation."""
    return np.arctan2(p.pos[1], p.pos[0])


def measure_phasors(planets, rules):
    """Interface measurement: z_i = exp(i * phi_i)."""
    z = np.empty(len(rules), dtype=np.complex128)
    for i, rule in enumerate(rules):
        li = get_lambda(planets[rule.inner_id - 1])
        lo = get_lambda(planets[rule.outer_id - 1])
        phi = rule.p * lo - rule.q * li
        z[i] = np.exp(1j * phi)
    return z


# -----------------------------
# K-field update (memory evolution)
# -----------------------------

def update_kernel(K: KernelField, z, dt):
    """
    Discrete Euler-Lagrange for K:
      K <- K + eta*(z-K)*dt
    where eta = nu*(1-saturation).
    Returns dissonance D and saturation.
    """
    saturation = np.abs(np.mean(K.modes))
    eta = K.nu * (1.0 - saturation)

    diff = z - K.modes
    D = np.mean(np.abs(diff))

    K.modes = K.modes + eta * diff * dt

    # renormalize each mode to keep |K|<=1
    mags = np.abs(K.modes)
    too_big = mags > 1.0
    if np.any(too_big):
        K.modes[too_big] /= mags[too_big]

    return D, saturation


# -----------------------------
# Adaptive drift (Addendum v0.1)
# -----------------------------

def adapt_parameters(K: KernelField, D, dt):
    """
    Adaptive alpha + mu:
      alpha_dot = k_alpha*(D - D*)
      mu_dot    = -k_mu*D
    """
    K.alpha += K.k_alpha * (D - K.d_target) * dt
    K.mu    -= K.k_mu * D * dt

    K.alpha = float(np.clip(K.alpha, *K.alpha_bounds))
    K.mu    = float(np.clip(K.mu,    *K.mu_bounds))


# -----------------------------
# Φ update: gravity + PIT torque
# -----------------------------

def apply_forces(planets, star_mass, K: KernelField, rules, dt):
    # Gravity
    for p in planets:
        r = p.pos
        d = np.linalg.norm(r)
        acc = -G * star_mass * r / (d**3)
        p.vel += acc * dt

    # PIT torque
    if K.alpha > 1e-9:
        z = measure_phasors(planets, rules)
        zhat = np.array([0.0, 0.0, 1.0])  # axis for tangential direction

        for i, rule in enumerate(rules):
            k = K.modes[i]
            zi = z[i]

            phase_error = np.angle(k) - np.angle(zi)
            torque = K.alpha * np.abs(k) * np.sin(phase_error)

            pin = planets[rule.inner_id - 1]
            pout = planets[rule.outer_id - 1]

            tin3 = np.cross(zhat, pin.pos)
            tout3 = np.cross(zhat, pout.pos)

            tin = tin3 / (np.linalg.norm(tin3) + 1e-12)
            tout = tout3 / (np.linalg.norm(tout3) + 1e-12)

            pin.vel  += (torque / pin.mass)  * tin  * dt
            pout.vel -= (torque / pout.mass) * tout * dt


def step_positions(planets, dt):
    for p in planets:
        p.pos += p.vel * dt


# -----------------------------
# Main simulation
# -----------------------------

def run_sim(seed=123, dt=0.1, steps=20000, log_every=20):
    star_mass = 0.81
    m_scale = 3.003e-6

    radii = [0.0793, 0.1039, 0.1364, 0.1790, 0.2166, 0.2621]
    planets = [
        Planet(i+1, 5.69*m_scale, [radii[i], 0, 0], [0,0,0], chr(ord("b")+i))
        for i in range(6)
    ]

    rng = np.random.default_rng(seed)
    for p in planets:
        r = np.linalg.norm(p.pos)
        theta = rng.random() * 2*np.pi
        v = np.sqrt(G * star_mass / r)
        p.pos = np.array([r*np.cos(theta), r*np.sin(theta), 0.0])
        p.vel = np.array([-v*np.sin(theta), v*np.cos(theta), 0.0])

    rules = [
        ResonanceRule(1,2,3,2),
        ResonanceRule(2,3,3,2),
        ResonanceRule(3,4,3,2),
        ResonanceRule(4,5,4,3),
        ResonanceRule(5,6,4,3),
    ]

    init_modes = 0.01 * np.exp(1j*2*np.pi*rng.random(len(rules)))
    K = KernelField(
        init_modes,
        mu=0.90, nu=0.05, alpha=1e-5,
        k_alpha=1e-4, k_mu=1e-5,
        d_target=0.10
    )

    trace = {
        "Step": [], "Phi_Angle": [], "K_Strength": [],
        "Alpha": [], "Mu": [], "Dissonance": [], "Saturation": []
    }

    for t in range(1, steps+1):
        z = measure_phasors(planets, rules)
        D, sat = update_kernel(K, z, dt)
        adapt_parameters(K, D, dt)

        apply_forces(planets, star_mass, K, rules, dt)
        step_positions(planets, dt)

        if t % log_every == 0:
            trace["Step"].append(t)
            trace["Phi_Angle"].append(np.angle(z[0]))
            trace["K_Strength"].append(np.abs(K.modes[0]))
            trace["Alpha"].append(K.alpha)
            trace["Mu"].append(K.mu)
            trace["Dissonance"].append(D)
            trace["Saturation"].append(sat)

    return pd.DataFrame(trace)


def main():
    df = run_sim()

    csv_path = "hd110067_adaptive_trace_py.csv"
    df.to_csv(csv_path, index=False)

    # Plot like GeminiOut: habit + phase, adaptive params, dissonance
    plt.figure(figsize=(10,8))

    ax1 = plt.subplot(3,1,1)
    ax1.plot(df.Step, df.K_Strength, label="|K| (rule 1)")
    ax1.plot(df.Step, np.abs(df.Phi_Angle), label="|phase error| (proxy)")
    ax1.set_ylabel("Habit / Phase")
    ax1.legend()

    ax2 = plt.subplot(3,1,2)
    ax2.plot(df.Step, df.Alpha, label="alpha(t)")
    ax2.plot(df.Step, df.Mu, label="mu(t)")
    ax2.set_ylabel("Adaptive params")
    ax2.legend()

    ax3 = plt.subplot(3,1,3)
    ax3.plot(df.Step, df.Dissonance, label="Dissonance")
    ax3.set_ylabel("D")
    ax3.set_xlabel("Step")
    ax3.legend()

    plt.tight_layout()
    png_path = "hd110067_adaptive_trace_py.png"
    plt.savefig(png_path, dpi=160)

    print(f"Wrote {csv_path} and {png_path}")


if __name__ == "__main__":
    main()

