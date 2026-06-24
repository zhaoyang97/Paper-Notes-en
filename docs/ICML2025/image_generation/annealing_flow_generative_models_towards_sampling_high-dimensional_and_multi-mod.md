---
title: >-
  [Paper Note] Annealing Flow Generative Models Towards Sampling High-Dimensional and Multi-Modal Distributions
description: >-
  [ICML 2025][Image Generation][Annealing Flow] Proposes Annealing Flow (AF)—a Continuous Normalizing Flow (CNF) based method for sampling high-dimensional multi-modal distributions. It trains with a dynamic Optimal Transport (OT) objective combined with Wasserstein regularization to guide mode exploration through an annealing process, significantly outperforming existing NF and MCMC methods in high-dimensional multi-modal settings.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Annealing Flow"
  - "Continuous Normalizing Flows"
  - "Optimal Transport"
  - "Multi-modal Distribution"
  - "High-dimensional Sampling"
date: 2026-05-08
content_hash: 4cbe40656759bc86
---

# Annealing Flow Generative Models Towards Sampling High-Dimensional and Multi-Modal Distributions

**Conference**: ICML 2025  
**arXiv**: [2409.20547](https://arxiv.org/abs/2409.20547)  
**Code**: None  
**Area**: Image Generation/Sampling  
**Keywords**: Annealing Flow, Continuous Normalizing Flows, Optimal Transport, Multi-modal Distribution, High-dimensional Sampling

## TL;DR
Proposes Annealing Flow (AF)—a Continuous Normalizing Flow (CNF) based method for sampling high-dimensional multi-modal distributions. It trains with a dynamic Optimal Transport (OT) objective combined with Wasserstein regularization to guide mode exploration through an annealing process, significantly outperforming existing NF and MCMC methods in high-dimensional multi-modal settings.

## Background & Motivation

**Background**: Sampling from high-dimensional multi-modal distributions is a fundamental challenge in statistical Bayesian inference and physical machine learning. MCMC methods mix slowly in high dimensions and are prone to getting trapped in local modes; discrete NFs suffer from mode collapse.

**Limitations of Prior Work**:
   - MCMC (including HMC) requires an exponential number of steps to mix even on distributions with only two modes.
   - Existing NF annealing methods (e.g., Annealed Importance Sampling, Path-guided NFs) still rely on MCMC-assisted sampling or score estimation with a large number of discretization steps.
   - Particle optimization methods (e.g., SVGD) are sensitive to kernel choices and hyperparameters, with computational complexity scaling polynomially with sample size.

**Key Challenge**: The distance between modes in multi-modal distributions can be extremely large (especially in high dimensions), requiring sampling methods to "skip" low-probability regions to reach different modes, but prior methods are limited in this capacity.

**Goal**: Design a training-efficient and stable high-dimensional multi-modal sampling method without MCMC assistance.

**Key Insight**: Design the training objective of continuous normalizing flows as dynamic optimal transport + annealing. This allows the flow to learn how to transition smoothly from a simple distribution to the target distribution, utilizing intermediate annealed distributions to navigate the multi-modal landscape.

**Core Idea**: The combination of annealing and OT—annealing constructs a bridge of intermediate distributions from simple to target, the OT objective ensures smooth and efficient flow paths, and Wasserstein regularization constrains the complexity of the flow.

## Method

### Overall Architecture
The core workflow of Annealing Flow:
1. Construct a sequence of annealed intermediate distributions $\{f_k(x)\}$: $f_k(x) \propto \pi_0(x)^{1-\beta_k} \tilde{q}(x)^{\beta_k}$, where $\beta_0=0$ (initial Gaussian) to $\beta_K=1$ (target distribution).
2. Divide the time interval $[0,1]$ into $K$ segments, each learning an optimal transport mapping $T_k$ from $f_{k-1}$ to $f_k$.
3. Combine all mappings into a complete Continuous Normalizing Flow: $\pi_0 \to f_1 \to \cdots \to f_K = q$.
4. The training does not require the normalizing constant $Z_k$—all calculations rely solely on the unnormalized density $\tilde{f}_k$.

### Key Designs

1. **Dynamic Optimal Transport Objective + Wasserstein Regularization**:

    - Function: Learn the optimal transport mapping $T_k$ for each segment.
    - Mechanism: Objective function = KL divergence (ensuring density matching) + $\gamma \int \|v_k\|^2 dt$ (Wasserstein regularization constraining velocity field smoothness).
    - Mathematical Form: $T_k = \arg\min_T \{KL(T_{\#} f_{k-1} \| f_k) + \gamma \int \mathbb{E}[\|v_k\|^2] dt\}$.
    - Design Motivation: Pure KL objectives can lead to irregular transport paths (severe fluctuations in velocity fields). Wasserstein regularization forces the flow to take "straight lines," greatly improving training stability.
    - Differences from Prior Work: It does not require score estimation and matching during training (unlike Path-guided NFs), nor does it require MCMC assistance.

2. **Annealing Temperature Scheduling**:

    - Function: Construct a bridge of intermediate distributions from simple to complex.
    - Mechanism: Geometric annealing—$\tilde{f}_k(x) = \pi_0(x)^{1-\beta_k} \tilde{q}(x)^{\beta_k}$. At lower temperatures, the distribution is flat (easy to sample), and at higher temperatures, it gradually approaches the target.
    - Design Motivation: Annealing allows the flow to "see" all modes in the early stage (where barriers between modes are low under flat distributions) and then progressively concentrate on each mode.
    - Key Advantage: AF requires far fewer annealing steps than traditional methods because the OT objective makes transport at each step more efficient.

3. **Theoretical Guarantee (Theorem 1)**:

    - Function: Prove the analytical form of the optimal velocity field.
    - Mechanism: The infinitesimal optimal velocity field equals the score difference of adjacent annealed densities: $v^* \propto \nabla \log f_k - \nabla \log f_{k-1}$.
    - Design Motivation: This property is unique to the dynamic OT objective of AF—meaning the flow naturally learns to transport along the direction of the score gradient without explicit score estimation.
    - Significance: Connects the theories of OT and score matching, providing a theoretical foundation for the effectiveness of AF.

### Loss & Training
- Segment loss: KL divergence + Wasserstein regularization ($\gamma$ controls the trade-off).
- The KL term does not require normalizing constants (computed via the change-of-variables formula for log density of flow).
- Velocity field $v_k(x(t), t)$ is parameterized by neural networks.
- Training only requires sampling from the current flow without MCMC.
- The number of annealing steps $K$ is typically only 3-5 (far fewer than the hundreds of steps required by traditional annealing methods).

## Key Experimental Results

### Main Results
Multi-modal synthetic distributions (Gaussian Mixture Models, up to 128 dimensions):

| Method | 2D GMM (W₂↓) | 32D GMM (W₂↓) | 128D GMM (W₂↓) | Annealing Steps |
|------|-------------|---------------|----------------|---------|
| HMC | 0.012 | Failed | Failed | - |
| SVGD | 0.008 | 0.45 | Failed | - |
| Path-guided NF | 0.005 | 0.12 | 0.89 | >50 |
| **Ours (AF)** | **0.003** | **0.05** | **0.21** | **3-5** |

### Boltzmann Distribution Sampling (Physics Application)

| Target Distribution | Ours (AF) (KL↓) | Best Baseline (KL↓) | Gain |
|---------|---------|-------------|------|
| Double-well (2D) | 0.02 | 0.08 | 75% |
| Lennard-Jones (30D) | 0.15 | 0.42 | 64% |
| Adversarial Distribution | 0.08 | 0.31 | 74% |

### Ablation Study

| Configuration | 128D GMM (W₂) | Description |
|------|--------------|------|
| No Annealing (Direct OT) | 1.52 | Mode collapse |
| No Wasserstein Regularization | 0.78 | Unstable training |
| K=1 Annealing Step | 0.89 | Insufficient annealing |
| K=3 Annealing Steps | 0.24 | Near-optimal |
| **K=5 Annealing Steps** | **0.21** | **Optimal** |
| K=20 Annealing Steps | 0.22 | Diminishing marginal returns |

### Key Findings
- AF remains effective on 128D multi-modal distributions, while MCMC and SVGD fail completely.
- Only 3-5 annealing steps are needed to achieve near-optimal results—far fewer than the 50+ steps required by Path-guided NF.
- Wasserstein regularization is crucial for training stability—without it, the loss curve fluctuates wildly.
- The theoretical prediction of Theorem 1 (optimal velocity field $\propto$ score difference) matches highly with the learned velocity field in experiments.
- Performed well even on deliberately adversarial distributions, showing robustness.

## Highlights & Insights
- **The combination of annealing + OT** is extremely natural and powerful—annealing resolves the mode collapse of NFs, and OT ensures transport efficiency, making them complementary rather than redundant.
- Achieving superior results with only 3-5 annealing steps compared to methods requiring 50+ steps indicates that the quality of transport at each step under the dynamic OT objective is far superior to traditional score matching.
- The theoretical insight of Theorem 1 (optimal velocity field = score difference) has independent academic value—connecting two seemingly distinct fields: optimal transport and score matching.
- The ability to sample Boltzmann distributions enables its direct application to practical problems in molecular dynamics and statistical physics.
- No reliance on MCMC assistance is a major practical advantage—MCMC mixing diagnostics are notoriously difficult and unreliable.

## Limitations & Future Work
- Needs access to the unnormalized density $\tilde{q}(x)$ of the target distribution—inapplicable to sample-only scenarios.
- Performance in ultra-high-dimensional (>1000D) scenarios has not been validated.
- The annealing temperature schedule $\{\beta_k\}$ is currently set manually; adaptive scheduling could offer further improvements.
- Parameterization of the velocity field using neural networks shows some sensitivity to architectural choices.
- Inference in Continuous Normalizing Flows requires solving ODEs, which is slightly slower than discrete flows (though far faster than MCMC).

## Related Work & Insights
- **vs MCMC (HMC/PT)**: No mixing required, avoiding exponential mixing times in high dimensions; however, cannot provide asymptotic exactness guarantees like MCMC.
- **vs SVGD**: AF does not rely on kernel computation, avoiding polynomial complexity; SVGD suffers from particle collapse in high dimensions.
- **vs Path-guided NF (Tian et al.)**: Requires 50+ annealing steps and step-by-step score estimation, whereas AF only requires 3-5 steps.
- **vs Score-based Diffusion**: Diffusion models learn scores from data, while AF directly computes scores from annealed densities; they apply to different scenarios (AF for sampling with known densities, Diffusion for data-driven generation).
- **Insights**: The concept of combining annealing + continuous dynamics can be extended to other sampling/optimization problems that require crossing energy barriers.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of dynamic OT + annealing and its theoretical analysis is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Fully validated across synthetic and physical distributions, with rich ablation studies in high dimensions.
- Writing Quality: ⭐⭐⭐⭐ Rigorous theoretical derivations and clear illustrations.
- Value: ⭐⭐⭐⭐⭐ A key breakthrough in high-dimensional multi-modal sampling, with broad applications in statistical physics and Bayesian inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Diffusion Models Are Statistically Optimal for Learning Low-Dimensional Multi-Modal Distributions](../../ICML2026/image_generation/diffusion_models_are_statistically_optimal_for_learning_low-dimensional_multi-mo.md)
- [\[CVPR 2026\] Evaluating Generative Models via One-Dimensional Code Distributions](../../CVPR2026/image_generation/evaluating_generative_models_via_one-dimensional_code_distributions.md)
- [\[NeurIPS 2025\] Progressive Inference-Time Annealing of Diffusion Models for Sampling from Boltzmann Densities](../../NeurIPS2025/image_generation/progressive_inference-time_annealing_of_diffusion_models_for_sampling_from_boltz.md)
- [\[ICML 2025\] ContinualFlow: Learning and Unlearning with Neural Flow Matching](continualflow_learning_and_unlearning_with_neural_flow_matching.md)
- [\[ICML 2025\] Distillation of Discrete Diffusion through Dimensional Correlations (Di4C)](distillation_of_discrete_diffusion_through_dimensional_correlations.md)

</div>

<!-- RELATED:END -->
