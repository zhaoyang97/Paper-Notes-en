---
title: >-
  [Paper Note] Dimension-Free Multimodal Sampling via Preconditioned Annealed Langevin Dynamics
description: >-
  [ICML 2026][Multimodal VLM][Annealed Langevin Dynamics] This work provides the first **dimension-free** non-asymptotic convergence analysis for Preconditioned Annealed Langevin Dynamics (PALD)—reducing the sampling complexity for multimodal distributions from $\tilde{O}(d/\epsilon^2)$ to $\tilde{O}(1/\epsilon^2)$, liberating diffusion-based sampling algorithms from the "curse of dimensionality" in high-dimensional settings.
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Annealed Langevin Dynamics"
  - "Multimodal Distributions"
  - "Dimension-Free Convergence"
  - "Hessian Preconditioning"
date: 2026-05-08
content_hash: 203f76ec89abdd78
---

# Dimension-Free Multimodal Sampling via Preconditioned Annealed Langevin Dynamics

**Conference**: ICML 2026  
**arXiv**: [2605.30396](https://arxiv.org/abs/2605.30396)  
**Code**: To be confirmed  
**Area**: Optimization / Sampling Algorithms / Theory of Diffusion Models  
**Keywords**: Annealed Langevin Dynamics, Multimodal Distributions, Dimension-Free Convergence, Hessian Preconditioning

## TL;DR
This work provides the first **dimension-free** non-asymptotic convergence analysis for Preconditioned Annealed Langevin Dynamics (PALD)—reducing the sampling complexity for multimodal distributions from $\tilde{O}(d/\epsilon^2)$ to $\tilde{O}(1/\epsilon^2)$, liberating diffusion-based sampling algorithms from the "curse of dimensionality" in high-dimensional settings.

## Background & Motivation

**Background**: Sampling from multimodal distributions is a central challenge in machine learning and statistics. Langevin Dynamics (LD) requires exponential time to cross the "potential barriers" of a distribution. Annealed LD (ALD) gradually lowers the energy landscape through temperature annealing and has proven practical in NCSN and diffusion models.

**Limitations of Prior Work**: Although existing convergence analyses for ALD provide guarantees, the **complexity depends linearly on the dimension $d$** or even worse. This leads to a sample complexity explosion in high-dimensional tasks (e.g., ImageNet where $d \approx 10^6$).

**Key Challenge**: ALD effectively performs sampling in millions of dimensions in practice, yet theoretical analysis fails to explain this phenomenon—resulting in a "dimensionality gap" between theory and practice.

**Goal**: To establish **dimension-free convergence guarantees** for ALD on high-dimensional multimodal distributions, bridging the gap between theory and practice.

**Key Insight**: The dimensionality dependence in existing analyses stems from the assumption of an **equidistant isotropic step-size**. By employing **preconditioning** (local Hessian adaptation), an effective step size can be maintained across high-dimensional directions, thereby achieving dimension-free convergence.

**Core Idea**: The update rule of Langevin Dynamics is replaced with a version preconditioned by the local Hessian—$\theta_{t+1} = \theta_t - \eta H(\theta_t)^{-1} \nabla U(\theta_t) + \sqrt{2\eta H(\theta_t)^{-1}} \xi_t$—to obtain dimension-free convergence while preserving the annealing framework.

## Method

### Overall Architecture
The algorithmic procedure is straightforward: (1) Define the target distribution $\pi(\theta) \propto \exp(-U(\theta))$; (2) Construct a temperature sequence $\beta_1 < \beta_2 < ... < \beta_K = 1$; (3) Execute preconditioned Langevin updates at each temperature; (4) Obtain the preconditioner $H(\theta_t)$ via Hessian adaptation or low-rank approximation; (5) Collect target samples at the final temperature. The modification of PALD relative to standard Annealed Langevin (ALD) is limited to replacing the isotropic step size with Hessian preconditioning. The true contribution of this work lies in **proving that its sampling complexity on multimodal distributions is independent of the dimension $d$**. Consequently, the following three key designs form a progressive logical chain: how preconditioning decouples effective step size from dimension (Design 1), how annealing decouples potential barrier height from dimension (Design 2), and how log-Sobolev inequalities tighten these synergistic effects into a rigorous dimension-free upper bound (Design 3).

### Key Designs

**1. Preconditioned Hessian Adaptation: Compensating step sizes with local curvature to decouple "effective steps" from dimension.**

Standard Langevin Dynamics uses a uniform step size in all directions, causing the overall progress to be bottlenecked by the sharpest direction; this bottleneck worsens as dimensions increase. This method utilizes the local Hessian $H(\theta) = \nabla^2 U(\theta)$ (or a regularized version $H + \lambda I$) as a preconditioner: step sizes are reduced in sharp directions (large eigenvalues) to ensure stability and increased in flat directions (small eigenvalues) to accelerate exploration. As a result, the relative step size $\eta / \lambda_i$ in each direction reaches its own stability threshold. Consequently, the effective number of steps required for convergence no longer expands with dimension, which is the source of dimension independence.

**2. Annealing Schedule + Dimension-Free Barrier Crossing: Decoupling barrier height from dimension.**

Multimodal sampling is difficult due to the "potential barriers"—LD requires infinite time to jump from one mode to another. Annealing bridges global exploration and local refinement via a temperature sequence $\beta_1 < \beta_2 < ... < \beta_K = 1$: at high temperatures (small $\beta_k$), the potential function is flattened and modes are easily crossed. This work utilizes geometric annealing $\beta_k = \beta_0 \cdot r^k$ ($r>1$). Traditional complexity proofs for annealing depend on the maximum barrier height, a quantity roughly $O(d)$. With preconditioning, the "effort" required for crossing is determined by the effective curvature across potential directions rather than dimension, so the barrier height $\Delta$ no longer grows linearly with $d$. The synergy of preconditioning and annealing decouples dimension from the potential barriers.

**3. Theoretical Analysis Framework: Dimension-free upper bound of $\tilde{O}(\log(1/\epsilon)/\epsilon^2)$ via log-Sobolev and transport inequalities.**

To transform these intuitions into rigorous guarantees, the analysis proves that the KL divergence $\text{KL}(p_k \| \pi_{\beta_k})$ decreases monotonically along the temperature sequence. It uses log-Sobolev inequalities and Talagrand transport inequalities to derive the complexity upper bound and explicitly constructs a preconditioning-assisted synchronous coupling to avoid dimensionality explosion. The difficulty lies in the fact that log-Sobolev constants are typically $O(d^{-1})$, which would reintroduce dimensionality dependence; the role of preconditioning is to map the analysis into a "transformed equidistant space" where this $d^{-1}$ factor is absorbed. The final complexity is $\tilde{O}(\log K / \epsilon^2)$, independent of dimension.

## Key Experimental Results

### Convergence Complexity

| Method | Sampling Complexity | Dimension Dependence |
| :--- | :--- | :--- |
| Standard LD | $\tilde{O}(d \beta^* / \epsilon^2)$ | Linear $d$ |
| Standard ALD | $\tilde{O}(d \log K / \epsilon^2)$ | Linear $d$ |
| **PALD (Ours)** | $\tilde{O}(\log K / \epsilon^2)$ | **None** |
| MCMC (HMC) | $\tilde{O}(d^{1/4} / \epsilon^{1/2})$ | $d^{1/4}$ |

### Main Results: Synthetic Multimodal Distributions

| Distribution | Dimension | Modes | LD Crossing Rate | ALD Crossing Rate | **PALD Crossing Rate** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Mixture of 2 Gaussians | 100 | 2 | 12% | 89% | **97%** |
| Mixture of 2 Gaussians | 10000 | 2 | 0% | 23% | **94%** |
| 4-Mixture (Rotated) | 100 | 4 | 8% | 73% | **96%** |
| 4-Mixture (Rotated) | 10000 | 4 | 0% | 12% | **91%** |

PALD maintains high crossing rates in high dimensions, while ALD/LD degrade severely.

### High-Dimensional Specific Benchmarks

| Task | Algorithm | Dimension | Convergence Time (vs ALD) |
| :--- | :--- | :--- | :--- |
| Neural Network Posterior | PALD vs ALD | 50000 | **0.07× Time** |
| High-dim GMM | PALD vs ALD | 100000 | **0.02× Time** |

### Key Findings
- **Experimental Verification of Dimension-Free Property**: PALD convergence time remains relatively stable from 100 to 10,000 dimensions, whereas ALD degrades sharply.
- **Multimodal Preservation**: In 4-mode distributions, PALD accurately captures the relative weights of all modes, whereas ALD tends to bias toward the initial mode in high dimensions.
- **Preconditioner Update Frequency**: Updating every 100 steps is found to be optimal; excessive updates increase computational overhead.

## Highlights & Insights
- **First Dimension-Free Convergence Proof**: Breaks the "curse of dimensionality" in multimodal sampling, providing theoretical support for high-dimensional diffusion models.
- **Elegant Combination of Preconditioning + Annealing**: The synergy between these two independent techniques far exceeds their individual use—preconditioning ensures step-size efficiency, while annealing ensures global exploration.
- **Rigorous Experimental Validation**: Systematically demonstrates dimension independence from low (100) to high ($10^5$) dimensions, highly consistent with theoretical predictions.

## Limitations & Future Work
- Hessian Computation Cost: Each step requires $O(d^2)$ storage or $O(d^3)$ factorization; this remains difficult for ultra-high dimensions ($d > 10^7$).
- Accuracy Loss in Low-Rank Approximations: Theoretical analysis assumes an exact Hessian preconditioner; the low-rank or diagonal approximations commonly used in practice might violate dimension-free conditions.
- Non-Smooth Potentials: Current analysis requires $U$ to be twice differentiable; distributions with non-smooth potentials or those on Stiefel manifolds are not directly applicable.
- Improvements: Explore fast approximations based on efficient preconditioners like K-FAC or Shampoo; extend the analysis to non-smooth or geometrically constrained distributions.

## Related Work & Insights
- **vs Standard ALD (Song-Ermon 2019)**: The primary innovation of this work is the preconditioning mechanism and the theoretical analysis providing dimension-free convergence proofs.
- **vs Hamiltonian Monte Carlo (HMC)**: HMC accelerates mixing by introducing momentum, but theoretical analysis remains dimension-dependent; PALD directly tackles dimensionality through preconditioning.
- **vs Second-Order Preconditioning in Adam/SGD**: This work is the first to apply preconditioning to sampling instead of optimization scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First dimension-free multimodal sampling guarantee, a major theoretical breakthrough)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Synthetic multimodal experiments are complete; verification on real-world high-dimensional tasks is limited)
- Writing Quality: ⭐⭐⭐⭐ (Mathematically rigorous, clear proof steps, theory and experiments support each other)
- Value: ⭐⭐⭐⭐⭐ (Establishes a theoretical foundation for diffusion models and high-dimensional Bayesian inference)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Conditional Diffusion Sampling](conditional_diffusion_sampling.md)
- [\[ICML 2025\] RollingQ: Reviving the Cooperation Dynamics in Multimodal Transformer](../../ICML2025/multimodal_vlm/rollingq_reviving_the_cooperation_dynamics_in_multimodal_transformer.md)
- [\[ICML 2026\] FreeRet: MLLMs as Training-Free Retrievers](freeret_mllms_as_training-free_retrievers.md)
- [\[ICLR 2026\] Asynchronous Matching with Dynamic Sampling for Multimodal Dataset Distillation](../../ICLR2026/multimodal_vlm/asynchronous_matching_with_dynamic_sampling_for_multimodal_dataset_distillation.md)
- [\[ICLR 2026\] Importance Sampling for Multi-Negative Multimodal Direct Preference Optimization](../../ICLR2026/multimodal_vlm/importance_sampling_for_multi-negative_multimodal_direct_preference_optimization.md)

</div>

<!-- RELATED:END -->
