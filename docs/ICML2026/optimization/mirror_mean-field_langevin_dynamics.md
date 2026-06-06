---
title: >-
  [Paper Note] Mirror Mean-Field Langevin Dynamics
description: >-
  [ICML2026][Optimization][mean-field Langevin] This work synthesizes Mean-Field Langevin Dynamics (MFLD) and Mirror Langevin Dynamics (MLD) into "Mirror Mean-Field Langevin Dynamics" (MMFLD). It provides the first global…
tags:
  - "ICML2026"
  - "Optimization"
  - "mean-field Langevin"
  - "mirror descent"
  - "constrained sampling"
  - "propagation of chaos"
  - "logarithmic Sobolev inequality"
date: 2026-05-08
content_hash: 491f3472e4c9d35e
---

# Mirror Mean-Field Langevin Dynamics

**Conference**: ICML2026  
**arXiv**: [2505.02621](https://arxiv.org/abs/2505.02621)  
**Code**: Not released  
**Area**: optimization  
**Keywords**: mean-field Langevin, mirror descent, constrained sampling, propagation of chaos, logarithmic Sobolev inequality  

## TL;DR
This work synthesizes Mean-Field Langevin Dynamics (MFLD) and Mirror Langevin Dynamics (MLD) into "Mirror Mean-Field Langevin Dynamics" (MMFLD). It provides the first global convergence algorithm for minimizing entropy-regularized functionals $\mathcal{L}(\mu)=F(\mu)+\lambda\,\mathrm{Ent}(\mu)$ on a convex constrained domain $X\subseteq\mathbb{R}^d$. In continuous time, it proves $e^{-2C_{\mathrm{LSI}}\lambda t}$ linear convergence via a uniform mirror LSI; in discretization, it provides a uniform-in-time propagation of chaos analysis using an $N$-particle system with Euler-Maruyama integration.

## Background & Motivation

**Background**: The distribution optimization objective $\mathcal{L}(\mu)=F(\mu)+\lambda\,\mathrm{Ent}(\mu)$ formulates various machine learning problems (infinite-width two-layer neural networks, tensor decomposition, sparse spike deconvolution, density estimation, and discrepancy minimization) as convex optimization in Wasserstein space. When $X=\mathbb{R}^d$, MFLD (the McKean-Vlasov process $dX_t=-\nabla\frac{\delta F(\mu_t)}{\delta \mu}(X_t)dt+\sqrt{2\lambda}dB_t$) coupled with a uniform LSI already provides linear convergence $L(\mu_t)-L(\mu^\ast)\le e^{-2C_{\mathrm{LSI}}\lambda t}$ and mature propagation of chaos analysis.

**Limitations of Prior Work**: In practice, many domains $X$ are bounded convex sets (trajectory inference requires a probability simplex, Wasserstein barycenters require bounded support, and mean-matching in discrepancy minimization is often constrained to a simplex or spectral shape; norm-constrained neural networks require parameters within a ball). Directly applying projection to MFLD tends to accumulate mass on the boundary $\partial X$. Conversely, single-particle mirror Langevin cannot handle cases where $F$ is a distributional functional with a nonlinear $\frac{\delta F}{\delta\mu}$. This leaves an open question: **Is there a mean-field algorithm with global convergence guarantees for constrained distribution optimization objectives $\mathcal{L}$?**

**Key Challenge**: The diffusion in MFLD is "all-space Gaussian," which inevitably sends mass outside $X$. MLD modifies the geometry via a mirror map to confine diffusion within $X$, but it is designed to sample a fixed $\mu^\ast\propto e^{-f/\lambda}$, failing to handle the mean-field coupling where the target distribution depends on the current $\mu$. These two mechanisms were previously disconnected.

**Goal**: (1) Propose a unified SDE where diffusion remains naturally within $X$ while the drift handles the mean-field term $\frac{\delta F(\mu_t)}{\delta \mu}$; (2) prove global exponential convergence in continuous time using mirror LSI; (3) prove uniform-in-time propagation of chaos for an $N$-particle time-discretized algorithm, with LSI constants decoupled from the number of particles; (4) extend the convergence rates to stochastic gradient settings.

**Key Insight**: The authors observe that the difference between the dual-space SDE of MLD $dY_t=-\nabla f(X_t)dt+\sqrt{2\lambda\nabla^2\phi(X_t)}dB_t$ and MFLD is merely the replacement of $\nabla f$ with $\nabla\frac{\delta F(\mu_t)}{\delta\mu}$. By implementing this substitution, they derive a mean-field version of mirror dynamics and adapt the "configuration space + entropy sandwich" proof technique to mirror geometry.

**Core Idea**: Treat the mirror map $\nabla\phi$ as a tool to "fold" the constrained geometry into the diffusion. All theoretical components of MFLD (Wasserstein gradient flow, entropy sandwich, uniform LSI, propagation of chaos) are upgraded to the Hessian metric $\nabla^2\phi$ to achieve a unified "Mirror MFLD."

## Method

### Overall Architecture
To minimize $\mathcal{L}(\mu)=F(\mu)+\lambda\,\mathrm{Ent}(\mu)$ for $\mu\in\mathcal{P}_2(X)$ where $X\subseteq\mathbb{R}^d$ is convex, a thrice-differentiable, Legendre-type barrier $\phi:X\to\mathbb{R}$ is selected (e.g., $\phi(x)=\sum_i x_i\log x_i$ for the simplex). The explosion of $\phi$ at $\partial X$ ensures diffusion remains in $X$. The continuous-time SDE for MMFLD is $X_t=\nabla\phi^\ast(Y_t)$, $dY_t=-\nabla\tfrac{\delta F(\mu_t)}{\delta\mu}(X_t)\,dt+\sqrt{2\lambda\nabla^2\phi(X_t)}\,dB_t$. Its Fokker-Planck equation is $\partial_t\mu_t=\lambda\nabla\cdot(\mu_t[\nabla^2\phi]^{-1}\nabla\log(\mu_t/\hat\mu_t))$, where $\hat\mu_t\propto\exp(-\tfrac{1}{\lambda}\tfrac{\delta F(\mu_t)}{\delta\mu})$ is the proximal Gibbs distribution. This form maintains mean-field coupling while restricting diffusion via the Hessian metric. The $N$-particle algorithm (Algorithm 1) discretizes this SDE using mirror gradients and Euler-Maruyama: particles $X_k^i$ transition to dual space via the mirror map, follow the $-\eta_k\nabla\frac{\delta F(\mu_k)}{\delta\mu}(X_k^i)$ drift, simulate pure diffusion $dY_t^i=\sqrt{2\lambda[\nabla^2\phi^\ast(Y_t^i)]^{-1}}dB_t$, and return to primal space via $\nabla\phi^\ast$.

### Key Designs

1.  **Continuous Time Convergence: Mirror Entropy Sandwich + Uniform Mirror LSI**:
    *   **Function**: Proves $L(\mu_t)-L(\mu^\ast)\le e^{-2C_{\mathrm{LSI}}\lambda t}(L(\mu_0)-L(\mu^\ast))$ (Theorem 3.2), extending MFLD's exponential convergence to constrained geometries.
    *   **Mechanism**: Employs relative Lipschitz and relative smoothness assumptions (Assumption 5) using the local norm $\|\cdot\|_{[\nabla^2\phi(x)]^{-1}}$ to prove uniqueness of the minimizer satisfying the fixed-point condition $\mu^\ast\propto\exp(-\tfrac{1}{\lambda}\frac{\delta F(\mu^\ast)}{\delta\mu})$. It assumes the proximal Gibbs $\hat\mu$ satisfies a uniform mirror LSI: $\mathrm{KL}(\mu\|\hat\mu)\le \frac{1}{2C_{\mathrm{LSI}}}\mathrm{FI}_\phi(\mu\|\hat\mu)$ for any $\mu\in\mathcal{P}_2(X)$. Lyapunov estimation on $\frac{d}{dt}L(\mu_t)$ via an entropy sandwich yields exponential decay.
    *   **Design Motivation**: Mirror LSI is derived from classical LSI and strong convexity of $\phi$. The entropy sandwich remains valid in constrained cases, allowing the convergence proof to be translated to mirror geometry.

2.  **Discretization + Uniform-in-Time Propagation of Chaos**:
    *   **Function**: Proves $\tfrac{1}{N}L^{(N)}(\mu_k^{(N)})-L(\mu^\ast)\le e^{-C_{\mathrm{LSI}}\lambda\eta k}\cdot(\cdot)+\tfrac{LR^2}{2N}+\tfrac{\delta_\eta}{2C_{\mathrm{LSI}}\lambda}$ (Theorem 4.2) for $N$ particles.
    *   **Mechanism**: Lifts the $N$-particle problem to configuration space. Theorem 4.1 provides an LSI-free particle approximation error $\tfrac{1}{N}L^{(N)}(\mu^{(N)}_\ast)-L(\mu^\ast)\le \tfrac{LR^2}{2N}$. Discretization bias $\delta_\eta$ is controlled using self-concordance (Assumption 7: $|\nabla^3\phi^\ast[u,u,u]|\le 2c_1\langle u,\nabla^2\phi u\rangle^{3/2}$) and uniform-in-$N$ mirror LSA.
    *   **Design Motivation**: A core challenge in propagation of chaos is the coupling of particle error with LSI constants. By utilizing an LSI-free bound, the $1/N$ term depends only on $LR^2$, ensuring the error vanishes uniformly as $N\to\infty$.

3.  **Mirror Geometry Choice and Boundary Handling**:
    *   **Function**: Maps abstract SDEs to three classic constrained domains for executable algorithms.
    *   **Mechanism**: Uses entropy mirror for the unit simplex $\Delta^d$, von Neumann mirror for the spectraplex, and log-barrier for the unit ball. Diffusion steps are performed in the dual space.
    *   **Design Motivation**: Unlike projection methods, the mirror map causes particles to naturally avoid $\partial X$, preventing the mass accumulation typical of projected MFLD.

### Loss & Training
Key hyperparameters include temperature $\lambda$ (regularization strength), learning rate $\eta_k$, and particle count $N$. Constants $c_1, c_2$ from Assumption 7 determine discretization bias $\delta_\eta$, requiring $\phi$ to be self-concordant and $c_2$-strongly convex.

## Key Experimental Results

Experiments provide qualitative sanity checks on low-dimensional synthetic domains.

### Main Results

| Experiment | Domain $X$ / Mirror Map | Objective | MMFLD vs Projected MFLD |
| :--- | :--- | :--- | :--- |
| Simplex mean-matching | $\Delta^3$ / Entropy | $F(\mu)=\|\mathbb{E}_\mu x-q\|^2+\beta\mathbb{E}_\mu \sum\log(1/x_i)$ | MFLD accumulates mass at $\partial\Delta^3$; MMFLD achieves lower loss and uniform distribution. |
| Spectraplex density matching | $\{\Sigma\succeq 0:\mathrm{Tr}\Sigma=1\}$ / von Neumann | $F(\mu)=\tfrac12\|\mathbb{E}_\mu \Sigma-\Sigma^\ast\|_F^2+\tfrac{1}{2\gamma}\mathbb{E}_\mu\|\Sigma\|_F^2$ | Projected MFLD shows zero progress; MMFLD converges near optimal. |
| Norm-constrained Two-layer ReLU | Unit Ball / Log-barrier | XOR classification with noise | MMFLD loss drops rapidly; MFLD stagnates as neurons hit the boundary. |

### Ablation Study

| Configuration | Key Observation |
| :--- | :--- |
| Projected MFLD (baseline) | Mass accumulation at boundaries; zero progress on spectraplex; neurons stick to $\|w\|=1$. |
| Projected MFLD w/ boundary barrier | Particles repelled from boundary, but overall distribution is worse than without barrier. |
| MMFLD with one-step diffusion | No significant difference from multi-step simulation; confirms efficiency of discretization. |
| MMFLD with stochastic gradient | Adds $\sigma^2/c_2$ term; linear convergence remains preserved. |

### Key Findings
*   Projection methods are particularly incompatible with mean-field settings: projection can nullify the Wasserstein progress made in the drift step. Inverting the constraint into the geometry via mirror maps allows continuous progress.
*   Single-step diffusion discretization is sufficient to maintain convergence, with runtime comparable to projected MFLD.
*   In norm-constrained neural networks, MMFLD aligns neurons with decision boundaries, whereas MFLD results in neurons diverging toward the $\|w\|=1$ limit.

## Highlights & Insights
*   Successfully synthesizes MFLD and MLD without ad-hoc projections, porting the LSI-free propagation of chaos framework to mirror geometry with quantifiable constants.
*   The assumptions (self-concordance + uniform-in-$N$ mirror LSI) are verifiable for standard constraints like simplices, spectraplexes, and balls.
*   Conceptualizing the mirror map as "folding the geometry into the diffusion" offers a template for other constrained mean-field optimizations, such as private synthetic trajectory generation or entropic OT.

## Limitations & Future Work
*   Experiments are limited to low-dimensional synthetic tasks ($d \le 10$); scalability to large-scale deep MFNNs remains unverified.
*   Convergence rates rely on the abstract uniform-in-$N$ mirror LSI assumption; quantitative LSI constants for complex constraints like the spectraplex remain an open problem.
*   Discretization follows a specific forward scheme; while Euler-Maruyama works well empirically, the theoretical gap for single-particle discretization bias still requires $\eta \to 0$.
*   Future work involves extending the analysis from mirror LSI to mirror Poincaré inequalities.

## Related Work & Insights
*   **vs. Mirror Langevin (Chewi et al. 2020, etc.)**: Those works handle single-particle sampling for fixed $\mu^\ast$; Ours extends this to mean-field coupling where $\mu^\ast$ depends on $\mu_t$.
*   **vs. MFLD (Nitanda 2024, etc.)**: Ours adopts the entropy sandwich and LSI-free framework but generalizes all metrics to the Hessian metric of the mirror map.
*   **vs. Mirrored Langevin with Stochastic Gradients (Hsieh et al. 2018)**: They focused on single-particle sampling; Theorem 4.3 in this work is the first to introduce SGD to MMFLD.
*   **vs. Application Frameworks (Chizat 2023)**: Provides the missing unified convergence guarantee for applications like entropic Wasserstein barycenters.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ (Clean synthesis of MFLD and MLD with a complete theoretical framework)
*   Experimental Thoroughness: ⭐⭐⭐ (Sufficient for sanity checks, lacks large-scale validation)
*   Writing Quality: ⭐⭐⭐⭐⭐ (Excellent progression and clear theorem structure)
*   Value: ⭐⭐⭐⭐ (Provides a standardized algorithm for constrained mean-field optimization)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Mirror Descent Under Generalized Smoothness](mirror_descent_under_generalized_smoothness.md)
- [\[ICML 2026\] Muon in Associative Memory Learning: Training Dynamics and Scaling Laws](muon_in_associative_memory_learning_training_dynamics_and_scaling_laws.md)
- [\[ICML 2026\] Ubiquity of Emergent Hebbian Dynamics in Regularized Learning](ubiquity_of_emergent_hebbian_dynamics_in_regularized_learning.md)
- [\[ICML 2026\] Bregman meets Lévy: Stochastic Mirror Descent with Heavy-Tailed Noise in Continuous and Discrete Time](bregman_meets_lévy_stochastic_mirror_descent_with_heavy-tailed_noise_in_continuo.md)
- [\[ICML 2026\] Learning Dynamics of Zeroth-Order Optimization: A Kernel Perspective](learning_dynamics_of_zeroth-order_optimization_a_kernel_perspective.md)

</div>

<!-- RELATED:END -->
