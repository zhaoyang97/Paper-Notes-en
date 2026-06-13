---
title: >-
  [Paper Note] Flow Sampling: Learning to Sample from Unnormalized Densities via Denoising Conditional Processes
description: >-
  [ICML 2026][Computational Biology][Diffusion Sampling] This paper proposes Flow Sampling, which reverses the flow matching/diffusion model paradigm from "data-driven" to "noise-driven." By constructing a denoising condit…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Diffusion Sampling"
  - "Flow Matching"
  - "Amortized Sampling"
  - "Riemannian Manifolds"
  - "Molecular Conformation"
date: 2026-05-08
content_hash: c86a424bee7523f1
---

# Flow Sampling: Learning to Sample from Unnormalized Densities via Denoising Conditional Processes

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.03984](https://arxiv.org/abs/2605.03984)  
**Code**: Not released  
**Area**: Diffusion Models / Sampling / Molecular Conformation Generation  
**Keywords**: Diffusion Sampling, Flow Matching, Amortized Sampling, Riemannian Manifolds, Molecular Conformation

## TL;DR
This paper proposes Flow Sampling, which reverses the flow matching/diffusion model paradigm from "data-driven" to "noise-driven." By constructing a denoising conditional process based on source noise samples and using a detached model to sample $X_1$ on an interpolant as a regression target for the energy gradient, an efficient data-free diffusion sampler is learned and naturally extended to constant curvature Riemannian manifolds.

## Background & Motivation

**Background**: Many scientific computing problems (molecular dynamics, materials, chemical reaction paths) require sampling from an unnormalized density $q(x)=\exp(r(x))/Z$, where $r(x)$ and $\nabla r(x)$ are known but no samples are available. While MCMC/Langevin are asymptotically correct, they suffer from sequential generation and slow mixing. Recent diffusion samplers fall into two categories: (a) methods like iDEM/PIS/DDS that learn sampling dynamics via Monte Carlo correction (importance sampling, resampling); (b) methods following the SOC/Schrödinger bridge route (Adjoint Sampling, ASBS) that learn diffusion dynamics by optimizing path measure divergence.

**Limitations of Prior Work**: Category (a) methods are computationally expensive as they require multiple energy evaluations per step to control variance. Category (b) methods involve complex training pipelines often requiring auxiliary networks to characterize optimal control or bridges. Furthermore, both categories default to Euclidean space, requiring re-design for manifold extensions (spheres, hyperboloids).

**Key Challenge**: The training objective of standard FM/diffusion models is to "construct a noising process conditioned on a given data point $x_1$ and let the model regress to the velocity field of this process." This path is blocked when $x_1$ is unavailable—one must indirectly introduce target information into training via the score $\nabla r$.

**Goal**: To find a "dual perspective"—constructing a denoising process conditioned on a given noise point $x_0$ such that the marginal still matches the target distribution, while designing an efficient training loop that reuses energy gradients to minimize NFE.

**Key Insight**: The authors reverse the conditional direction of FM. Traditional FM is the push-forward of source ($p_{t|1}(x|x_1)=\tfrac{1}{\sigma_t^d}p_0(\tfrac{x-\alpha_t x_1}{\sigma_t})$ with $t=1$ as the condition). In contrast, this paper uses the push-forward of target ($p_{t|0}(x|x_0)=\tfrac{1}{\alpha_t^d}q(\tfrac{x-\sigma_t x_0}{\alpha_t})$ with $t=0$ as the condition). Both have the same marginal $p_t$, but the former requires data samples while the latter requires only noise samples—which exactly fits the data-free setting.

**Core Idea**: Construct a "denoising diffusion process" conditioned on $X_0\sim p_0$. The supervising drift along the interpolant $X_t=\sigma_t X_0+\alpha_t X_1$ can be written in closed form as $u_{t|0}(X_t|X_0)=\dot\alpha_t X_1+\dot\sigma_t X_0+\tfrac{g_t^2}{2\alpha_t}\nabla r(X_1)$. $X_1^{\bar\theta}$ is sampled using the detached current model, and $\nabla r(X_1^{\bar\theta})$ is cached in a replay buffer for repeated use.

## Method

### Overall Architecture
A two-stage alternating cycle (similar to Adjoint Sampling): ① **Exploration**—Use the detached model $u^{\bar\theta}$ to simulate from $X_0\sim\mathcal{N}(0,I)$ to $X_1^{\bar\theta}$ via Euler-Maruyama, evaluate $\nabla r(X_1^{\bar\theta})$, and push the pair $(X_1^{\bar\theta}, \nabla r)$ into a replay buffer. ② **Optimization**—Sample a batch $(X_1, \nabla r)$ from the buffer, sample $X_0\sim p_0$ and $t\sim\text{Unif}[0,1]$, calculate the conditional drift target $u_{t|0}=X_1-X_0+\gamma t\nabla r(X_1)$ according to the interpolant $X_t=(1-t)X_0+tX_1$, and perform MSE regression for the model $u^\theta(X_t)$. Iteration continues until convergence.

### Key Designs

1. **Denoising supervising drift under noise condition**:
    - **Function**: Constructs a stochastic process whose marginal is the target $q$ without requiring data samples, enabling model learning via regression.
    - **Mechanism**: Defines a conditional probability path $p_{t|0}(x|x_0)=\tfrac{1}{\alpha_t^d}q(\tfrac{x-\sigma_t x_0}{\alpha_t})$ (push-forward of target) and identifies two processes generating it: (i) conditional velocity $v_{t|0}(x|x_0)=\tfrac{\dot\alpha_t}{\alpha_t}(x-\sigma_t x_0)+\dot\sigma_t x_0$, solved in closed form by the interpolant $X_t=\sigma_t X_0+\alpha_t X_1$; (ii) conditional drift $u_{t|0}=v_{t|0}+\tfrac{g_t^2}{2}\nabla\log p_{t|0}$, used to train the diffusion sampler. Proposition 3.2 proves that on the interpolant, $u_{t|0}(X_t|x_0)=\dot\alpha_t X_1+\dot\sigma_t x_0+\tfrac{g_t^2}{2\alpha_t}\nabla r(X_1)$, meaning $\nabla r$ only needs to be evaluated **once** for each $X_1$ and can be reused for all $t\in[0,1]$ and any $X_0$.
    - **Design Motivation**: Energy evaluation is the primary cost in scenarios like molecular dynamics. Reusing $\nabla r$ over time and source samples is a critical engineering optimization.

2. **Fixed-point iteration training with detached model**:
    - **Function**: Replaces the unavailable true $X_1\sim q$ with $X_1^{\bar\theta}$ generated by the current model to construct a fixed point.
    - **Mechanism**: Defines $X_1^{\bar\theta}\sim p_1^{\bar\theta}$ as endpoint samples of the detached model ($\theta$ does not compute gradients). Solve the diffusion process via Euler-Maruyama $X_{t+h}^{\bar\theta}=X_t^{\bar\theta}+h u_t^{\bar\theta}(X_t^{\bar\theta})+\sqrt{2\gamma th}Z_t$. Then minimize $\mathcal{L}_{FS}=\mathbb{E}\|u^\theta(X_t)-u_{t|0}(X_t|X_0)\|^2$, where $X_1$ comes from $X_1^{\bar\theta}$ instead of the true $q$.
    - **Design Motivation**: Fixed-point iteration is a common trick in EM-like training. The theoretical optimum satisfies $p_1^\theta=q$, and empirical convergence is stabilized with a replay buffer.

3. **Extension to constant curvature Riemannian manifolds (Hypersphere/Hyperbolic closed-form drift)**:
    - **Function**: Generalizes Flow Sampling to constant curvature $\kappa$ manifolds such as $\mathbb{S}^d$ and hyperbolic space.
    - **Mechanism**: Replaces the affine interpolant with a geodesic interpolant $X_t=\exp_{X_1}[(1-t)\log_{X_1}(x_0)]$ and replaces Euclidean Brownian motion with $P_{X_t}^\perp\circ dB_t$ to ensure diffusion remains in the tangent space. Proposition 4.1 provides a rank-1 closed-form for the geodesic Jacobian $J_t=t T_{X_1\to X_t}P_{\dot X_1}+c_t T_{X_1\to X_t}P_{\dot X_1}^\perp$, where $c_t=\sin(t\omega_1\sqrt\kappa)/\sin(\omega_1\sqrt\kappa)$ for $\kappa>0$ (or the $\sinh$ version for $\kappa<0$). This allows closed-form computation of the conditional score and drift, avoiding numerical back-differentiation.
    - **Design Motivation**: Existing diffusion samplers default to Euclidean space. Applications in spherical/hyperbolic spaces (directional data, robot poses, graph embeddings) previously required rewriting the entire pipeline. The closed-form formulas provided here make generalization nearly cost-free.

### Loss & Training
Linear scheduler $\alpha_t=t,\sigma_t=1-t,g_t^2=2\gamma t$; adaptive $\gamma=c/\sqrt{\mathbb{E}_{x_1\sim\mathcal{B}}[\|\nabla r(x_1)\|^2]+\varepsilon}$ to suppress energy gradient scale. Buffer size is 10k-60k, with 100-300 gradient updates per round and 128-2048 new samples per round.

## Key Experimental Results

### Main Results
Synthetic energy benchmarks (DW-4/LJ-13/LJ-55) + Peptides (Ala2/Ala4) + Large-scale amortized molecular conformation generation (SPICE/GEOM-DRUGS) + Spherical vMF mixtures.

| Dataset | Metric | Flow Sampling | ASBS (Prev. SOTA) | Gain |
|--------|------|------|----------|------|
| DW-4 | $\mathcal{W}_2$ ↓ | **0.36** | 0.43 | -16% |
| DW-4 | $E(\cdot)\mathcal{W}_2$ ↓ | **0.11** | 0.20 | -45% |
| LJ-13 | $E(\cdot)\mathcal{W}_2$ ↓ | **0.97** | 1.99 | -51% |
| LJ-55 | $E(\cdot)\mathcal{W}_2$ ↓ | **21.32** | 28.10 | -24% |
| Ala2 JSD (NFE=1024) | ↓ | **0.018** | 0.242 | -93% |
| Ala2 Energy $\mathcal{W}_2$ (NFE=128) | ↓ | **3.58** | $10^7$ | catastrophic |

### Ablation Study

| NFE (Train) | SPICE Recall Cov | SPICE Recall AMR |
|------|---------|------|
| 256 (Ours) | **91.89** | **0.86** |
| 128 (Ours) | 91.39 | 0.87 |
| 64 (Ours) | 90.13 | 0.87 |
| 256 (AS) | 88.60 | 0.87 |
| 128 (AS) | 77.97 | 0.98 |
| 64 (AS) | 29.13 (Collapse) | 1.43 |
| 512 (ASBS) | 89.66 | 0.86 |

### Key Findings
- Under a low budget of NFE=64, AS collapses (Cov 5.50), while Flow Sampling remains stable (Cov 71.14), indicating that FS's supervising target has significantly lower variance than AS's stochastic adjoint signal.
- Training costs can be reduced by 4-8 times: On SPICE, performance at NFE=64 is nearly identical to NFE=256, meaning the exploration phase cost can be directly reduced by 4x.
- The spherical vMF mixture experiment is the first demo of diffusion sampling on curved manifolds, proving the Riemannian extension is practical.
- Appendix D provides a rigorous proof that FS and Adjoint Sampling have the same conditional expectation on Brownian bridge paths (Theorem D.1), making them essentially control-variate equivalents.

## Highlights & Insights
- **The "conditional direction reversal" is the soul of this paper**: Data-driven FM/DM conditions on $X_1$, while data-free Flow Sampling conditions on $X_0$. This simple symmetry brings massive engineering benefits—FS reuses a single energy gradient across the entire time axis and source sample space, minimizing expensive energy evaluations.
- **The elegant supervising target $\dot\alpha_t X_1+\dot\sigma_t x_0+\tfrac{g_t^2}{2\alpha_t}\nabla r(X_1)$** is reminiscent of score matching: minimalist, interpretable, and analyzable. It serves as the technical hinge merging data-free diffusion with a replay buffer.
- **Rank-1 Jacobian on constant curvature manifolds** (Proposition 4.1) is a beautiful mathematical result that turns manifold sampling from a concept into an industrializable algorithm. The specific closed forms for hyperspheres and hyperboloids open doors for directional data and hyperbolic embeddings.

## Limitations & Future Work
- Fixed-point iteration lacks global convergence guarantees, and replay buffer dynamics may be unstable on pathological energy landscapes (strong multimodality, large gradients).
- Riemannian generalization is currently limited to constant curvature. General curvature manifolds (e.g., non-Euclidean metrics in protein conformation space) require numerical Jacobian inversion, significantly reducing efficiency.
- Compared to SOC/Schrödinger bridge methods, FS lacks optimal control guarantees, potentially resulting in less efficient generation paths for certain tasks.
- Experiments focus on molecules/spheres, lacking coverage of popular RL scenarios like image reward model fine-tuning.

## Related Work & Insights
- **vs iDEM**: iDEM uses MC to estimate target scores along a noising path, requiring multiple energy evaluations per step; FS uses a single evaluation on a denoising path and reuses it, drastically reducing NFE.
- **vs Adjoint Sampling (AS) / ASBS**: AS uses the stochastic adjoint method to backpropagate SOC gradients; ASBS adds a Schrödinger bridge corrector. FS requires no corrector and regresses directly on a closed-form target, simplifying the training logic. Theorem D.1 proves they are essentially equivalent (up to a control variate).
- **vs Tilt Matching (concurrent)**: TM also uses a data-free reward-tilted approach but via annealing paths; FS uses a fixed linear scheduler, offering superior simplicity.
- **Insight**: The "conditional direction reversal" idea can be generalized to any scenario where one type of condition is known but another must be constructed, such as inverse problems or retrosynthesis.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Dual inversion from data-driven to noise-driven + closed-form drift for manifolds.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage (DW, LJ, Ala, SPICE, vMF), but lacks image/text reward fine-tuning.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear visualization of conditional duality; reproducible pseudo-code; concise key formulas.
- Value: ⭐⭐⭐⭐⭐ Reduces training costs by 4-8x in molecular generation + first manifold diffusion implementation; high impact for computational chemistry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Temporal Score Rescaling for Temperature Sampling in Diffusion and Flow Models](temporal_score_rescaling_for_temperature_sampling_in_diffusion_and_flow_models.md)
- [\[ICML 2026\] Transformed Latent Variable Multi-Output Gaussian Processes](transformed_latent_variable_multi-output_gaussian_processes.md)
- [\[ICML 2026\] CoSiNE: Conditional Site-Independent Neural Evolution Model for Antibody Sequences](conditionally_site-independent_neural_evolution_of_antibody_sequences.md)
- [\[ICML 2026\] EvoEGF-Mol: Evolving Exponential Geodesic Flow for Structure-based Drug Design](evoegf-mol_evolving_exponential_geodesic_flow_for_structure-based_drug_design.md)
- [\[ICML 2026\] LineageFlow: Flow Matching for High-Fidelity Family-Aware Protein Sequence Generation](lineageflow_flow_matching_for_high-fidelity_family-aware_protein_sequence_genera.md)

</div>

<!-- RELATED:END -->
