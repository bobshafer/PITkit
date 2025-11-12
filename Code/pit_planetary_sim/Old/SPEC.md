# Loom Spec — HD 110067 Case Study (v0.4)

**Status**: Implementation-ready
**Authors**: Bob (facilitator), ChatGPT-5, Gemini (co-authors)

## 1. Contract

### Types (Julia sketches; Codex may refine)

```julia
struct Resonance
    i::Int   # inner planet index
    j::Int   # outer planet index
    p::Int   # numerator (e.g., 3 in 3:2)
    q::Int   # denominator (e.g., 2 in 3:2)
end

struct PlanetState
    name::String
    m::Float64          # mass [M_earth or kg; be consistent]
    period::Float64     # [days]
    l::Float64          # mean anomaly or mean longitude (rad)
    varpi::Float64      # longitude of periapsis (rad)
    r::SVector{3,Float64}
    v::SVector{3,Float64}
end

struct OrbitState
    planets::Vector{PlanetState}
end

struct KernelState
    modes::Float64   # C ∈ [0,1]
    mu::Float64
    nu::Float64
    xi::Float64
    alpha::Float64
    beta::Float64
    sigma::Float64
end

struct Star
    m::Float64   # [M_sun or kg]
    J2::Float64
end
```

### Core functions (signatures and semantics)

1. **Resonant angles**

[
\phi_{ij} = p,\lambda_j - q,\lambda_i - (p-q),\varpi_i \quad \text{for target } p!:!q
]

```julia
resonant_angles(orbits::OrbitState, R::Vector{Resonance})::Vector{Float64}
```

2. **Window weights (Interface F[Φ])**
   Let (n_k = 2\pi / P_k). Normalized detuning for pair (i,j):

[
\Delta_{ij} = \frac{p,n_j - q,n_i}{p,n_j}
\qquad
W_{ij} = \exp!\left(-(\Delta_{ij}/\sigma)^2\right)
]

```julia
interface_weights(orbits::OrbitState, K::KernelState, R::Vector{Resonance})::Vector{Float64}
```

3. **Coherence update (Kernel relax)**

Logistic-damped growth/decay (stable, bounded):

[
\frac{dC}{dt} = \xi\left(\mu,\bar{W},C - \nu,C - \beta,C(1-C)\right),\quad
\bar{W}=\text{mean}(W_{ij})
]

```julia
relax_C(K::KernelState, W_avg::Float64, dt::Float64)::Float64  # returns new C
```

4. **PIT “spring” (tangential kicks)**

Effective stiffness (\kappa = \alpha,C).
PIT Hamiltonian per pair: (-\kappa,W_{ij}\cos\phi_{ij}).
Apply **equal/opposite tangential kicks** proportional to (\kappa,W_{ij}\sin\phi_{ij}).

```julia
pit_tangential_kicks!(orbits::OrbitState, K::KernelState, W::Vector{Float64}, R::Vector{Resonance})::Nothing
```

*Implementation note*: use unit vectors in velocity direction for “tangential” direction; scale by per-body mass; keep magnitudes small (numerically gentle).

5. **Main step order (per time step dt)**

```
W = interface_weights(orbits, K, R)
C′ = relax_C(K, mean(W), dt); K.modes = C′
pit_tangential_kicks!(orbits, K, W, R)   # small corrective kicks
symplectic_step!(orbits, star, dt)       # N-body integrator step
```

## 2. Data & Targets

### HD 110067 (nominal)

* Star: (M_\star \approx 0.81,M_\odot), (J_2 \approx 10^{-7}) (assumed).
* Planets (periods in days, masses from TTV posteriors—placeholders ok initially):

```
b: P≈  9.114,  M≈2.5
c: P≈ 13.673,  M≈3.2
d: P≈ 20.519,  M≈5.0
e: P≈ 30.793,  M≈3.9
f: P≈ 41.058,  M≈2.6
g: P≈ 54.770,  M≈4.1
```

* Resonance chain: 3:2, 3:2, 3:2, 4:3, 4:3 (adjacent pairs, inner→outer).

### Acceptance metrics

* **Libration amplitudes** of all (\phi_{ij}): typically < 20° over long windows.
* **Capture/recapture** frequency under small stochastic kicks.
* **Period-ratio stability** (low jitter) consistent with published TTV behavior.

## 3. Parameter Ranges (Goldilocks priors)

Use as sweep defaults (tune later):

```
mu    ∈ [0.05, 1.0]
nu    ∈ [0.01, 0.8]
xi    ∈ [0.3, 1.2]
alpha ∈ [0.05, 0.8]
beta  ∈ [0.05, 0.8]
sigma ∈ [1e-4, 5e-2]   # detuning window width
dt    ∈ [1e-4, 5e-3]   # days (choose stable step)
```

Heuristic: start near
`mu=0.6, nu=0.3, xi=0.9, alpha=0.4, beta=0.2, sigma=0.01`.

## 4. Integrator choice

Prefer a simple, explicit **kick–drift–kick** leapfrog (custom `symplectic_step!`) for transparency and control. Alternatively, use `DifferentialEquations.jl` with a symplectic method. Keep PIT kicks as a small, explicit correction before the orbital step.

## 5. Logging & Analysis

At configurable cadence, record:

* (t), (C(t)), (\bar{W}(t))
* Resonant angles (\phi_{ij}(t)), their moving-window amplitudes
* Periods (P_i(t)), period ratios (P_{j}/P_{i})
* Energy drift (diagnostic), per-pair kick magnitudes

Derive metrics:

* `libration_amp_deg[i]` (95% range)
* `recapture_time` after injection kick
* `stability_score` (combines jitter, energy drift, libration size)

## 6. Tests (must pass)

* **Unit**:

  * `interface_weights` yields (W≈1) at exact p:q, decays Gaussian with Δ.
  * `relax_C` keeps (C∈[0,1]) and converges to fixed point when Φ frozen.
  * `pit_tangential_kicks!` conserves equal/opposite momentum on pair.

* **Property**:

  * With `xi=0` (PIT off), strong random kicks break resonance and fail to recapture within `τ_max`.
  * With PIT on (defaults), small kicks recapture within `τ_max`.
