## Appendix C: Preliminary Outline of Numerical Methods

This appendix outlines the computational strategy for solving the coupled, non-linear field equations of Participatory Interface Theory (PIT) to fit galactic rotation curves. The goal is to create a robust pipeline for testing the **Coherence-Acceleration Hypothesis** against the full SPARC galaxy sample.

### C.1 The Core Computational Problem

For a given galaxy with a known baryonic mass distribution $\rho_b(r)$, we must find a self-consistent solution to the following system of equations:

1.  **The Coherence Field Equation:**
    $$\nabla^2\theta = V'(\theta) - \alpha(a)R$$
    where the Ricci scalar source is $R \approx 8\pi G \rho_b(r)$.

2.  **The Acceleration Definition:**
    $$a(r) = |\nabla\Phi_{total}(r)|$$
    where $\Phi_{total}$ is the total gravitational potential sourced by both the baryons and the coherence field θ.

3.  **The Coupling Function (Coherence-Acceleration Hypothesis):**
    $$\alpha(a) = \frac{\alpha_{max}}{1 + a/a_0}$$

This system is self-referential: the solution for θ depends on α, which depends on a, which in turn depends on θ.

### C.2 Discretization and Grid Setup

The first step is to discretize the problem. For an axisymmetric galaxy, we can work on a one-dimensional radial grid, r_i, from the galactic center to the outermost data point. All physical quantities ($\rho_b$, θ, a, α) will be represented as arrays on this grid.

### C.3 The Iterative Relaxation Method

A standard and effective way to solve such self-consistent field problems is through an **iterative relaxation method**. This approach starts with an initial guess and iteratively refines the solution until it converges.

**The Algorithm:**

1.  **Initialization:** For a given galaxy, begin with an initial guess for the acceleration profile, $a^{(0)}(r_i)$. A good starting point is the purely Newtonian acceleration from the observed baryons.

2.  **Iteration Loop (Step n):**
    a. **Calculate Coupling:** Using the current acceleration profile $a^{(n)}(r_i)$, compute the coupling profile $\alpha^{(n)}(r_i) = \alpha_{max} / (1 + a^{(n)}(r_i)/a_0)$.
    b. **Solve for Coherence Field:** With the source term now fixed, solve the Coherence Field Equation for $\theta^{(n+1)}(r_i)$. This is a sourced Klein-Gordon equation. On the discrete grid, it becomes a system of linear equations that can be solved efficiently using a finite-difference matrix solver.
    c. **Calculate New Acceleration:** Compute the new total gravitational potential $\Phi_{total}^{(n+1)}$ resulting from the baryons and the new coherence field $\theta^{(n+1)}$. From this, calculate the new acceleration profile $a^{(n+1)}(r_i)$.
    d. **Check for Convergence:** Compare the new acceleration profile to the previous one. If the maximum fractional difference, max(|a^{(n+1)} - a^{(n)}| / a^{(n)}), is below a set tolerance (e.g., $10^{-6}$), the solution has converged.
    e. **Repeat:** If not converged, set $a^{(n)} = a^{(n+1)}$ (or use a weighted average of the two for improved stability) and repeat the loop from step 2a.

### C.4 The Global Fitting Routine

The iterative solver described above finds the rotation curve for a *single* galaxy given a pair of universal constants ($\alpha_{max}$, $a_0$). To test the theory, we must find the best-fit universal constants across the entire SPARC sample.

This requires wrapping the single-galaxy solver inside a **global optimization routine** (e.g., using Python's scipy.optimize.minimize or an MCMC sampler). This routine will:
1.  Propose a pair of universal constants ($\alpha_{max}$, $a_0$).
2.  For these constants, loop through all ~175 galaxies in the SPARC sample, running the iterative solver for each one to find its predicted rotation curve.
3.  Calculate the total chi-squared ($\chi^2_{total}$) by summing the $\chi^2$ values from each individual galaxy fit.
4.  Adjust ($\alpha_{max}$, $a_0$) and repeat until the $\chi^2_{total}$ is minimized.

The final output will be the single best-fit pair of universal constants for the PIT framework.
