---
title: >-
  [Paper Note] REX: A Family of Reversible Exponential Stochastic Runge-Kutta Solvers
description: >-
  [ICML 2026][Physics & Scientific Computing][Reversible solver] This paper proposes Rex—a family of algebraically reversible (stochastic) Runge-Kutta solvers constructed based on Lawson exponential integrators. It automat…
tags:
  - "ICML 2026"
  - "Physics & Scientific Computing"
  - "Reversible solver"
  - "Exponential integrator"
  - "Stochastic Runge-Kutta"
  - "Diffusion model inversion"
  - "Boltzmann sampling"
date: 2026-05-08
content_hash: 4e49f97866b6be6e
---

# REX: A Family of Reversible Exponential Stochastic Runge-Kutta Solvers

**Conference**: ICML 2026  
**arXiv**: [2502.08834](https://arxiv.org/abs/2502.08834)  
**Code**: https://github.com/zblasingame/Rex-solver  
**Area**: Scientific Computing / Numerical Methods / Diffusion Model Sampling  
**Keywords**: Reversible solver, Exponential integrator, Stochastic Runge-Kutta, Diffusion model inversion, Boltzmann sampling

## TL;DR
This paper proposes Rex—a family of algebraically reversible (stochastic) Runge-Kutta solvers constructed based on Lawson exponential integrators. It automatically transforms any explicit (S)RK scheme into a precisely reversible ODE/SDE solver, ensuring arbitrary high-order convergence and non-zero stability regions while achieving near machine-precision inversion for image reconstruction/editing in diffusion models and Boltzmann sampling for flow models.

## Background & Motivation
**Background**: Diffusion models and continuous normalizing flows based on neural differential equations have become the SOTA for generative tasks. Forward integration (noise → data) typically uses high-order (S)RK schemes like DDIM, DPM-Solver, or SEEDS-1. Various key applications—such as fine-tuning via gradient descent through generative models, real image editing, differentiable rewards, and exact likelihood estimation of Boltzmann distributions—require **strictly exact backward integration (data → noise)**.

**Limitations of Prior Work**: Standard explicit solvers accumulate discretization errors $\varepsilon>0$ during forward-backward round-trips, causing the endpoints to deviate from the original trajectory. Existing "exact inversion" methods (e.g., EDICT, BDIA, BELM/O-BELM) suffer from multiple issues: poor stability (BDIA's LPIPS spikes to 0.885 in editing tasks), low order, and **almost exclusive limitation to ODEs**. Reversible solutions for diffusion SDEs remain largely unexplored, except for "pseudo-reversible" methods that cache the entire Brownian motion in memory.

**Key Challenge**: It is difficult to simultaneously satisfy four properties: algebraic reversibility (operator-level equality), high-order accuracy, non-zero linear stability regions, and support for adaptive step sizes/SDEs. The McCallum-Foster (MF) method achieved "reversibility + non-zero stability" for ODEs but did not consider SDEs or the semi-linear structure $f(t)\bm{x} + g(t)\bm{f}_\theta$ common in diffusion models.

**Goal**: (1) Extend MF reversibility to diffusion SDEs; (2) Leverage the semi-linear structure to construct exponential integrators for significantly improved precision; (3) Support adaptive step sizes while retaining arbitrary-order convergence and non-zero stability regions.

**Key Insight**: The drift term of diffusion ODEs/SDEs is naturally semi-linear: $a(t)\bm{X}_t+b(t)\bm{f}_\theta$. By using the Lawson method with an integrating factor $\Xi(t)=\exp\int_0^t a(\tau)d\tau$, the state variable can be transformed to $\bm{Y}=\Xi^{-1}\bm{X}$. This yields an equivalent SDE with **pure drift + unit diffusion**, onto which explicit (S)RK and MF packaging can be applied. Reverting the transformation completes the three-step "recipe" for Rex.

**Core Idea**: Use Lawson exponential integrators to handle the semi-linear component and wrap explicit (S)RK schemes with McCallum-Foster coupling to construct a family of diffusion solvers that are simultaneously reversible, high-order, stable, and SDE-compatible.

## Method
Rex is a **recipe** rather than a single scheme: given an explicit (S)RK scheme $\bm{\Phi}$, Rex is constructed in three steps and denoted as $\bm{\Upsilon}$.

### Overall Architecture
- **Input**: Diffusion reverse SDE $d\bm{X}_t=[f(t)\bm{X}_t-g^2(t)\nabla\log p_t(\bm{X}_t)]dt+g(t)d\bar{\bm{W}}_t$ (or ODE counterpart), noise schedule $(\alpha_t,\sigma_t)$, and the extended Butcher table of the explicit (S)RK base scheme $\bm{\Phi}$.
- **Mechanism**:
  1. **Reparameterize**: Rewrite the SDE as $d\bm{X}_t=[a(t)\bm{X}_t+b(t)\bm{f}_\theta]dt+g(t)d\bar{\bm{W}}_t$. Apply integrating factor $\Xi(t)=\exp\int_0^t a(\tau)d\tau$ and time transformation $\varsigma_t=\int\Xi^{-1}(t)b(t)dt$ to obtain $d\bm{Y}_\varsigma=\bm{f}_\theta(\varsigma,\Xi(\varsigma)\bm{Y}_\varsigma)d\varsigma+d\bm{W}_\varsigma$ (Locally Proposition 3.1).
  2. **Princeps**: Apply the explicit (S)RK $\bm{\Phi}$ to the transformed equation, then transform back to the original $\bm{X}$ to obtain an exponentially weighted solver $\bm{\Psi}_h$ (Eq. 11). Princeps generalizes DDIM, DPM-Solver-1/2/12, DPM-Solver++, SDE-DPM-Solver, SEEDS-1, and gDDIM (Theorem 3.3).
  3. **Rex**: Wrap $\bm{\Psi}_h$ in McCallum-Foster dual-state coupling to obtain the reversible scheme $\bm{\Upsilon}$ (Proposition 3.2).
- **Output**: A family of solvers including Rex (Euler), Rex (Euler-Maruyama), Rex (ShARK), Rex (RK4), and Rex (Dopri5), which serve as "reversible versions" of DDIM, DPM-Solver, etc.

### Key Designs

1. **Princeps: Exponentially Weighted (S)RK Sub-schemes**:
    - **Function**: Merges any explicit (S)RK $\bm{\Phi}$ with the semi-linear structure of the diffusion equation to create an "exponential (S)RK" that absorbs the integrating factor.
    - **Mechanism**: On $d\bm{Y}_\varsigma=\bm{f}_\theta(\varsigma,\Xi(\varsigma)\bm{Y}_\varsigma)d\varsigma+d\bm{W}_\varsigma$, use an $s$-stage SRK: $\bm{f}_\theta^i=\bm{f}_\theta(\varsigma_n+c_ih,\Xi(\varsigma_n+c_ih)\bm{Z}_i)$, where $\bm{Z}_i=\Xi^{-1}(\varsigma_n)\bm{X}_n+h\sum_{j<i}a_{ij}\bm{f}_\theta^{j}+a_i^W\bm{W}_n+a_i^H\bm{H}_n$. The update is $\bm{X}_{n+1}=\frac{\Xi(\varsigma_{n+1})}{\Xi(\varsigma_n)}\bm{X}_n+\Xi(\varsigma_{n+1})\bm{\Psi}$. Here $\bm{H}_n$ denotes the spatio-temporal Lévy area of the Brownian bridge (Foster 2024 system), enabling high strong convergence orders for SRK with additive noise.
    - **Design Motivation**: By combining the precision of exponential weighting with high-order (S)RK, the solver inherits the order of the base scheme (Theorem 3.4) while replicating mainstream samplers to ensure smooth transition for users.

2. **McCallum-Foster Dual-State Coupling for Algebraic Reversibility**:
    - **Function**: Wraps any explicit scheme $\bm{\Psi}$ into an algebraically reversible $\bm{\Upsilon}$ where forward and backward iterations are closed-form inverses regardless of step size.
    - **Mechanism**: Introduces a coupling parameter $\zeta\in(0,1]$ and an auxiliary state $\hat{\bm{X}}_n$. Forward: $\bm{X}_{n+1}=\tfrac{\kappa_{n+1}}{\kappa_n}(\zeta\bm{X}_n+(1-\zeta)\hat{\bm{X}}_n)+\kappa_{n+1}\bm{\Psi}_h(\varsigma_n,\hat{\bm{X}}_n,\bm{W}_n)$, and $\hat{\bm{X}}_{n+1}=\tfrac{\kappa_{n+1}}{\kappa_n}\hat{\bm{X}}_n-\kappa_{n+1}\bm{\Psi}_{-h}(\varsigma_{n+1},\bm{X}_{n+1},\bm{W}_n)$. The backward steps solve for $\hat{\bm{X}}_n,\bm{X}_n$ in closed form. $\kappa_n, \varsigma_t$ vary by ODE/SDE and prediction type.
    - **Design Motivation**: MF is currently the only framework offering "reversibility + non-zero linear stability." Rex inherits this stability, while $\zeta$ acts as a tunable knob for "inversion precision vs. stability."

3. **Replayable Brownian Motion via Splittable PRNG**:
    - **Function**: Ensures the backward SDE iteration uses the **exact same** Brownian path $\bm{W}_n(\omega)$ as the forward pass without storing the full trajectory in memory.
    - **Mechanism**: Implements schemes from Li 2020 / Jelinčič 2024 using splittable PRNGs (Salmon 2011). Brownian increments and Lévy areas $\bm{H}_{s,t}$ are generated recursively via a binary tree from a single seed for any interval $[s, t]$.
    - **Design Motivation**: Previous SDE inversion methods (Nie 2024) require caching $\bm{W}$, which exhausts memory and precludes adaptive step sizes. This design makes Rex the **first diffusion SDE solver to support exact inversion without full trajectory storage**, enabling adaptive schemes like Rex (Dopri5).

### Loss & Training
Rex is an **inference-time solver** and does not require new training losses. It can be directly integrated into pretrained models (DDPM, SD v1.5, DiT). Theoretically, Theorem 3.4 proves Rex is a $k$-th order reversible solver $\|\bm{x}_n-\bm{x}_{t_n}\|\le Ch^k$ if $\bm{\Phi}$ is a $k$-th order RK. Theorem 3.5 proves Princeps fully inherits the strong convergence order $\xi$ of the SRK.

## Key Experimental Results

### Main Results

| Task | Method | Key Metric | Note |
|------|------|---------|------|
| SD v1.5 Reconstruction Error (50 steps, latent MSE) | DDIM | Order ≫ Rex | Non-reversible baseline |
| Same as above | EDICT / BDIA / O-BELM | 1–Several orders higher than Rex | O-BELM error grows with steps (no stability) |
| Same as above | **Rex (Euler)** | **Near machine precision** | Lowest across all step counts (10/20/50) |
| T2I Generation (COCO, SD v1.5) | EDICT / BDIA / O-BELM | Inferior to Rex | Across three metrics |
| Same as above | **Rex (Euler-Maruyama / ShARK)** | **Top Image Reward & PickScore** | SDE variants lead |
| Image Editing (pix2pix, 50+50 steps) | DDIM (non-reversible) | LPIPS = 0.214 | Baseline |
| Same as above | O-BELM | LPIPS = 0.140 | Strongest reversible baseline |
| Same as above | BDIA | LPIPS = 0.885, ImgReward = −2.21 | Catastrophic failure (no stability) |
| Same as above | **Rex (Dopri5)** | **LPIPS = 0.107**, Top ImgReward/PickScore | ~2× gain; first adaptive reversible editor |
| Boltzmann Sampling (tri-alanine) | DiT + Dopri5 (non-reversible) | $\mathcal{E}\text{-}\mathcal{W}_2$=0.737 | Baseline for inversion necessity |
| Same as above | **DiT + Rex (Dopri5)** | **$\mathcal{E}\text{-}\mathcal{W}_2$=0.495** | Best energy distribution; SOTA-level |

### Ablation Study

| Configuration | Key Observation | Explanation |
|------|---------|------|
| Rex (Euler) vs Rex (RK4) in Recon | Euler is slightly better on CelebA-HQ FD | High-order inversion may not excel at low step counts |
| $\zeta=0.999$ (Images) vs $\zeta=0.001$ (Boltzmann) | Former for precision, latter for stability | Rex covers both "precision-type" and "stability-type" scenarios via $\zeta$ |
| BDIA / O-BELM (Stability) | Reconstruction errors diverge over time | Validates Rex’s inheritance of MF’s stability region |
| Hyperparameter Tuning | Rex outperforms tuned baselines without tuning | Demonstrates the robustness of the recipe |

### Key Findings
- **Unique Stable Reversible SDE Solver**: Rex is the first method to achieve precise diffusion SDE inversion without full Brownian caching, enabling SDE editing and Boltzmann sampling scenarios.
- **Criticality of Stability Region**: The failure of BDIA and O-BELM in reconstruction/editing is strongly correlated with their lack of linear stability regions; Rex's inherited stability from MF resolves this.
- **No Trade-off between Inversion and Quality**: Rex not only significantly reduces reconstruction error but also matches or exceeds non-reversible DDIM in sampling quality (e.g., FD).
- **Adaptive Step Size Breakthrough**: Rex (Dopri5) improves image editing LPIPS by approximately 2× over O-BELM, a feat previously impossible for reversible methods.

## Highlights & Insights
- **"Operator Recipe" Perspective**: The authors provide a standardized process to make **any** explicit (S)RK reversible, offering high reuse value for future community-developed RK schemes.
- **Princeps as a Unified Theory**: Theorem 3.3 proves Princeps covers many existing samplers, revealing they are specializations of specific (S)RKs under exponential integrators.
- **Splittable PRNG as Engineering Key**: Replacing trajectory caching with seed replay reduces memory complexity from $O(N)$ to $O(1)$, moving reversible SDE solvers from theory to practical SD/DiT deployment.
- **Tunability of $\zeta$**: Decoupling exact inversion from stability via $\zeta$ allows the solver to adapt to different task requirements (precision vs. stability).
- **Extensible Semi-linear Logic**: The recipe is applicable to any additive noise SDE, including affine probability paths in flow matching, as demonstrated by the Boltzmann sampling results.

## Limitations & Future Work
- **Theoretical Scope**: Stability and convergence proofs are currently focused on Variance Preserving (VP) schedules; Variance Exploding (VE) schedules require further analysis.
- **Noise Structure**: The recipe depends on additive noise SDEs and does not yet cover multiplicative noise or general SDE neural networks.
- **High-order Performance**: High-order Rex variants do not necessarily outperform low-order ones at very low step counts, a common trait in SDE numerical methods.
- **Computational Cost**: While memory-efficient, the calculation of Lévy areas and splittable PRNG overhead for SRKs introduces some additional wall-clock time compared to pure ODE solvers.

## Related Work & Insights
- **vs McCallum-Foster (2024)**: Rex extends the MF ODE-only framework to handle semi-linear structures and SDEs using exponential integrators.
- **vs EDICT / BDIA / O-BELM**: Unlike these ad-hoc diffusion ODE solvers that lack stability regions, Rex uses a general, stable MF wrapper.
- **vs SDE "Caching" Schemes**: Rex achieves $O(1)$ memory complexity and algebraic reversibility instead of relying on memory-intensive trajectoy storage.
- **vs DPM-Solver / DDIM**: Rex provides a unifying framework (Princeps) that effectively grants these samplers "reversible versions" for free.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Quantum Doubly Stochastic Transformers](../../NeurIPS2025/physics/quantum_doubly_stochastic_transformers.md)
- [\[ICML 2026\] Iterative Refinement Neural Operators are Learned Fixed-Point Solvers: A Principled Approach to Spectral Bias Mitigation](iterative_refinement_neural_operators_are_learned_fixed-point_solvers_a_principl.md)
- [\[ICLR 2026\] One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](../../ICLR2026/physics/one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)
- [\[NeurIPS 2025\] Hamiltonian Neural PDE Solvers through Functional Approximation](../../NeurIPS2025/physics/hamiltonian_neural_pde_solvers_through_functional_approximation.md)
- [\[NeurIPS 2025\] INC: An Indirect Neural Corrector for Auto-Regressive Hybrid PDE Solvers](../../NeurIPS2025/physics/inc_an_indirect_neural_corrector_for_auto-regressive_hybrid_pde_solvers.md)

</div>

<!-- RELATED:END -->
