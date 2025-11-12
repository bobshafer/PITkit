# Implementation Tasks (for Codex)

**Priority A — Make it run**

* [ ] Create `Project.toml` with listed deps.
* [ ] Implement `pit_physics.jl` per SPEC (functions & types).
* [ ] Implement `symplectic_step!` (kick–drift–kick) in `main.jl` or local module.
* [ ] Initialize star + planets from `config/default.toml`.
* [ ] Build resonances vector: `[(1,2,3,2),(2,3,3,2),(3,4,3,2),(4,5,4,3),(5,6,4,3)]`.
* [ ] Main loop order exactly as in SPEC §1.5.
* [ ] Logging to `out/trajectory.parquet` + `out/metrics.json`.
* [ ] CLI: `--config`, `--steps`, `--dt`, `--override K.xi=0.0` etc. (simple parser ok).

**Priority B — Validate**

* [ ] Compute resonant‐angle libration amplitudes over rolling windows.
* [ ] Derive acceptance metrics; print summary on exit.
* [ ] Plot: period ratios vs time; φ-libration; (C(t)) with (\bar{W}(t)).

**Priority C — Sweeps**

* [ ] Add `scripts/run_sweep.sh` to scan ((\mu,\nu,\alpha,\beta,\sigma)) grids.
* [ ] Save sweep heatmaps of stability score and libration amplitude.

**Priority D — Tests**

* [ ] Implement `tests/runtests.jl` with unit + property tests listed in SPEC §6.

**Coding standards**

* Keep functions pure where possible; avoid hidden globals.
* Use `StaticArrays` for small vectors; avoid heap churn in inner loops.
* Guard divisions; clamp (C) to ([0,1]); keep kicks small (config gain).

---

# File: `projects/loom-hd110067/config/default.toml`

```toml
[star]
mass = 0.81
J2   = 1e-7

[planets]  # periods in days; masses in Earth masses (initially)
names  = ["b","c","d","e","f","g"]
period = [9.114, 13.673, 20.519, 30.793, 41.058, 54.770]
mass   = [2.5, 3.2, 5.0, 3.9, 2.6, 4.1]

[K]
modes = 0.5
mu    = 0.6
nu    = 0.3
xi    = 0.9
alpha = 0.4
beta  = 0.2
sigma = 0.01

[run]
dt        = 0.001
steps     = 200000
log_every = 100
rng_seed  = 42
```

---

## Notes for Codex

* If a dependency is awkward, prefer **small local implementations** (e.g., simple leapfrog; CSV logging) and keep interfaces stable.
* Favor clarity over micro-optimizations; this is a research code.
* Stick to SPEC signatures; add helpers as needed.
* If masses/units are inconsistent, document the choice and convert once at init (e.g., days→seconds, Earth masses→kg).
