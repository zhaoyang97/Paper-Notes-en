---
title: >-
  [Paper Note] Elucidating Flow Matching ODE Dynamics via Data Geometry and Denoisers
description: >-
  [ICML 2025][Image Generation][Flow Matching] This paper analyzes the sampling trajectory dynamics of the Flow Matching (FM) ODE from the perspective of denoisers, revealing three stages of trajectory evolution (initial -> intermediate -> terminal) and establishing a convergence theory for the FM ODE when data is supported on a low-dimensional submanifold.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Flow Matching"
  - "ODE Dynamics"
  - "Data Geometry"
  - "Denoiser"
  - "Trajectory Convergence"
date: 2026-05-08
content_hash: 39211b1a427ec639
---

# Elucidating Flow Matching ODE Dynamics via Data Geometry and Denoisers

**Conference**: ICML 2025  
**arXiv**: [2412.18730](https://arxiv.org/abs/2412.18730)  
**Code**: None  
**Area**: Diffusion Models / Generative Modeling Theory  
**Keywords**: Flow Matching, ODE Dynamics, Data Geometry, Denoiser, Trajectory Convergence

## TL;DR

This paper analyzes the sampling trajectory dynamics of the Flow Matching (FM) ODE from the perspective of denoisers, revealing three stages of trajectory evolution (initial -> intermediate -> terminal) and establishing a convergence theory for the FM ODE when data is supported on a low-dimensional submanifold.

## Background & Motivation

### Background

**Background**: Flow Matching models generalize the ODE sampling of diffusion models into a unified framework, significantly reducing the number of sampling steps by learning vector fields. However, the theoretical understanding of FM models remains insufficient, particularly:

### Limitations of Prior Work

**Limitations of Prior Work**: Relationship between trajectories and data geometry: How do geometric structures of data (such as clusters and manifold structures) affect and guide the sampling trajectories?

### Key Challenge

**Key Challenge**: Terminal convergence: When data lies on a low-dimensional subspace or manifold, does the trajectory guarantee convergence to the data distribution? This is crucial for generation quality.

### Goal

**Goal**: Practical significance: Trajectory convergence is the theoretical foundation of single-step generative models like consistency models; understanding the relationship between data geometry and ODE trajectories can guide the optimization of sampling strategies.

Prior works (such as Biroli et al., Li & Chen) mainly focus on distribution-level analysis of randomized samplers in simplified settings. This work is the first to systematically reveal how data geometry plays a role at the individual ODE trajectory level.

## Method

### Overall Architecture

This work utilizes the **denoiser** (conditional mean of the posterior distribution, $m_t(x) = \mathbb{E}[X | X_t = x]$) as the core analytical tool. The denoiser is the only data-dependent component in the vector field $u_t$, completely determining the dynamical behavior of the FM ODE.

Through reparameterization with the noise-to-signal ratio $\sigma = \beta_t / \alpha_t$, the FM ODE can be uniformly expressed as:
$$\frac{dx_\sigma}{d\sigma} = -\frac{1}{\sigma}(m_\sigma(x_\sigma) - x_\sigma)$$

The intuitive meaning of this ODE is that: the trajectory moves along the denoiser direction.

### Key Designs

**Attracting-Absorbing Framework**

This work discovers that the FM ODE possesses two key properties:

1. **Attracting**: The trajectory is pulled toward a closed set $\Omega$.
2. **Absorbing**: Once entering a neighborhood of $\Omega$, the trajectory remains within it.

These two properties are unified and characterized by the **acute angle condition**:
$$\langle m_\sigma(x) - x, \text{proj}_\Omega(x) - x \rangle > 0$$
When the denoiser direction forms an acute angle with the projection direction, the distance from the trajectory to $\Omega$ decreases.

**Three-Stage Trajectory Analysis**

**Stage 1: Initial Stage** (large $\sigma$)
- When $t=0$, $m_0(x) \equiv \mathbb{E}[X]$, and the trajectory initially moves toward the data mean.
- Proposition 4.2 provides a quantitative convergence rate: $\|x_\sigma - \mathbb{E}[X]\| < R_0 \cdot (\sigma^2 + \delta^2)^{(1-\zeta)/2}$.

**Stage 2: Intermediate Stage** (moderate $\sigma$)
- The trajectory starts to be influenced by the coarse-grained geometry (local clustering) of the data.
- Under the local clustering assumption (cluster separation > 2 times the diameter), the trajectory is attracted to and absorbed into the convex hull of the local clusters.
- This explains why FM models achieve effective feature separation and mode coverage.

**Stage 3: Terminal Stage** ($\sigma \to 0$)
- Core result: the denoiser converges to the projection (Theorem 5.1).
  - $\lim_{\sigma \to 0} m_\sigma(x) = \text{proj}_\Omega(x)$, for almost all $x$.
- The convergence rate depends on the data geometry: $O(\sqrt{m}\sigma)$ on manifolds, and exponential convergence on discrete distributions.
- Main Theorem (Theorem 5.3): Under the conditions of positive reach and local measure lower bounds, the flow map $\Psi_1$ exists and $(\Psi_1)_\# p_{\text{prior}} = p$.

### Loss & Training

As a theoretical work, this paper does not introduce new training algorithms. However, it points out that the vector field learning (Eq. 5) can be replaced by the denoiser loss (Eq. 10), as the denoiser is bounded while the vector field may diverge as $t \to 1$.

## Key Experimental Results

### Main Results

This work is predominantly theoretical, with experiments designed to validate theoretical predictions:

- **Synthetic Data Validation** (Figure 2): Visualizes the three-stage trajectory behavior on 2D synthetic data—initially moving towards the mean, then being attracted by local clusters, and finally converging to data points.
- **High-Dimensional Validation** (Section J): Even when theoretical constants are loose, the qualitative behavior of the initial mean-attraction stage consistently holds in practice.

### Ablation Study

- Even when clusters overlap (violating the strict local clustering assumption), the ODE trajectories still tend to move toward locally dense regions.
- The generalization to smoothed Gaussian distributions (Corollary H.1) demonstrates that the attracting and absorbing properties of dense regions on the flow are still maintained.

### Key Findings

| Data Geometry | Convergence Rate $\|\Psi_1(x) - \Psi_t(x)\|$ |
|---------|--------------------------------------|
| General Distribution (Positive Reach) | $O(\sigma_t^{\zeta/2})$ for any $\zeta < 1$ |
| Compact Submanifold | $O(\sqrt{\sigma_t})$ |
| Discrete Distribution | $O(\sigma_t)$ (Optimal Rate) |
| Gaussian on Subspace | $\Theta(\sigma_t)$ (Exact Rate) |

## Highlights & Insights

1. **Unified Theoretical Framework**: Unifies the full-stage dynamics of FM ODE trajectories into a coherent analytical framework via the attracting-absorbing meta-theorem.
2. **First to Handle Low-Dimensional Manifold Data**: Prior convergence results required the data distribution to have support over the entire space, whereas this work is the first to cover the scenario where data lies on submanifolds.
3. **Equivariance**: Establishes the equivariance of the flow map under similarity transformations (scaling + rotation + translation) (Proposition 5.7), which has practical implications for the stability of data augmentation.
4. **Memorization Phenomenon**: Analysis of discrete measures suggests that terminal-time training plays a key role in resolving the memorization phenomenon.
5. **Practical Implications**: Trajectory movement is very small in the terminal stage ($O(\sigma_t^{\zeta/2})$), implying that fewer sampling steps can be used in the terminal phase.

## Limitations & Future Work

1. **Loose Theoretical Constants**: As worst-case bounds, the theoretical constants can be significantly larger than actual values.
2. **Convergence Rate on Manifolds**: The current $O(\sqrt{\sigma_t})$ may not be optimal, as the exact solution on subspaces suggests possible improvement to $O(\sigma_t)$.
3. **Assumption Constraints**: The positive reach condition excludes manifolds with "sharp turns"; the local clustering assumption requires cluster separation to be much larger than cluster diameter.
4. **Lack of Real Image Generation Experiments**: All experiments are conducted on synthetic data, leaving high-dimensional image generation scenarios unverified.
5. **Vector Field Singularity**: Discovered that the FM ODE vector field has singularities and diverges when the data distribution is not fully supported (Section C.2), which affects numerical solving.

## Related Work & Insights

- **Chen et al. (2024)**: Connects FM sampling with the mean shift algorithm, focusing on algorithmic strategies in high-curvature regions.
- **Permenter & Yuan (2024)**: Proves that the denoiser converges to the projection near data support, but this work generalizes it to almost every point.
- **Gao & Li (2024)**: Analyzes local cluster absorption of discrete measures, but implicitly assumes ODE convergence and requires bounded prior support.
- **Baptista et al. (2025)**: Concurrent work, analyzing the memorization dynamics of diffusion models under empirical measures.
- The attracting-absorbing framework in this work is promising to guide geometry-aware latent space design and sampling strategy optimization.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First full-stage analytical framework for FM ODE trajectories, covering submanifold data for the first time.
- Experimental Thoroughness: ⭐⭐⭐ — Mainly theoretical, with experiments primarily for validation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear structure, rigorous mathematics, and intuitive illustrations.
- Value: ⭐⭐⭐⭐⭐ — Provides a solid theoretical foundation for FM models, with direct guidance for sampling strategy optimization.

## Related Papers

- [\[NeurIPS 2025\] Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data](../../NeurIPS2025/image_generation/composite_flow_matching_for_reinforcement_learning_with_shifted-dynamics_data.md)
- [\[ICML 2025\] Gaussian Mixture Flow Matching Models](gaussian_mixture_flow_matching_models.md)
- [\[ICML 2025\] ContinualFlow: Learning and Unlearning with Neural Flow Matching](continualflow_learning_and_unlearning_with_neural_flow_matching.md)
- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](../../ICCV2025/image_generation/contrastive_flow_matching.md)
- [\[ICML 2025\] Expressive Score-Based Priors for Distribution Matching with Geometry-Preserving Regularization](expressive_score-based_priors_for_distribution_matching_with_geometry-preserving.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data](../../NeurIPS2025/image_generation/composite_flow_matching_for_reinforcement_learning_with_shifted-dynamics_data.md)
- [\[ICML 2025\] Gaussian Mixture Flow Matching Models](gaussian_mixture_flow_matching_models.md)
- [\[ICML 2025\] ContinualFlow: Learning and Unlearning with Neural Flow Matching](continualflow_learning_and_unlearning_with_neural_flow_matching.md)
- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](../../ICCV2025/image_generation/contrastive_flow_matching.md)
- [\[ICML 2025\] Expressive Score-Based Priors for Distribution Matching with Geometry-Preserving Regularization](expressive_score-based_priors_for_distribution_matching_with_geometry-preserving.md)

</div>

<!-- RELATED:END -->
