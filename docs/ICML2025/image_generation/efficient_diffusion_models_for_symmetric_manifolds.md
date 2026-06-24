---
title: >-
  [Paper Note] Efficient Diffusion Models for Symmetric Manifolds
description: >-
  [ICML 2025][Image Generation][Symmetric Manifolds] This paper proposes an efficient diffusion model framework on symmetric manifolds (tori, spheres, SO(n), U(n)), which bypasses heat kernel computation through the projection of Euclidean Brownian motion and Itô's lemma, reducing training complexity from exponential to nearly linear, and providing polynomial-class sampling accuracy guarantees.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Symmetric Manifolds"
  - "Riemannian Diffusion Models"
  - "Heat Kernel Bypassing"
  - "Itô's Lemma"
  - "Sampling Guarantees"
date: 2026-05-08
content_hash: 486c68c5851a17d3
---

# Efficient Diffusion Models for Symmetric Manifolds

**Conference**: ICML 2025  
**arXiv**: [2505.21640](https://arxiv.org/abs/2505.21640)  
**Code**: None  
**Area**: Diffusion Models / Manifold Generative Models  
**Keywords**: Symmetric Manifolds, Riemannian Diffusion Models, Heat Kernel Bypassing, Itô's Lemma, Sampling Guarantees

## TL;DR

This paper proposes an efficient diffusion model framework on symmetric manifolds (tori, spheres, SO(n), U(n)), which bypasses heat kernel computation through the projection of Euclidean Brownian motion and Itô's lemma, reducing training complexity from exponential to nearly linear, and providing polynomial-class sampling accuracy guarantees.

## Background & Motivation

Data in many applications (robotics, drug discovery, quantum physics) are naturally constrained on non-Euclidean manifolds, such as tori $\mathbb{T}_d$, spheres $\mathbb{S}_d$, special orthogonal groups SO(n), and unitary groups U(n) ($d \approx n^2$).

Key bottlenecks faced by existing manifold diffusion models:

| Problem | Euclidean Space | On Manifold |
|------|------------|--------|
| Heat Kernel Computation | Closed-form (Gaussian) | No closed-form, requires $2^d$-level summation |
| Gradient Computation | $O(1)$ times | $O(d)$ times (ISM objective requires Riemannian divergence) |
| Forward Sampling | Direct Gaussian sampling | Requires SDE/ODE solvers |
| Per-iteration Compute | $O(d)$ | Exponential or $O(d)$ gradients |

**Goal**: Narrow the gap between manifold and Euclidean diffusion models in training efficiency and sampling guarantees.

## Method

### Overall Architecture

**Core Idea**: Introduce a diffusion model with spatially-varying covariance, allowing the forward diffusion to be represented as Euclidean Brownian motion projected onto the manifold via a projection map $\varphi$.

**Forward Process**:
- Run an Ornstein-Uhlenbeck process $Z_t$ in $\mathbb{R}^d$
- Project to the manifold: $X_t := \varphi(Z_t)$
- Since the OU process has a closed-form Gaussian transition kernel, forward sampling **does not require an SDE solver**

**Reverse Process**: Project the reverse SDE in $\mathbb{R}^d$ onto the manifold via Itô's lemma:
$$dY_t = f^\star(Y_t, t)dt + g^\star(Y_t, t)dB_t$$

### Key Designs

#### 1. Choice of Projection Maps

| Manifold | Projection Map $\varphi$ | Computational Complexity |
|------|-------------------|----------|
| Torus $\mathbb{T}_d$ | $\varphi(x)[i] = x[i] \mod 2\pi$ | $O(d)$ |
| Sphere $\mathbb{S}_d$ | $\varphi(x) = x/\|x\|$ | $O(d)$ |
| SO(n) / U(n) | Spectral decomposition $U^*\Lambda U$, take $U$ | $O(n^\omega)=O(d^{\omega/2})$ |

where $\omega \approx 2.37$ is the matrix multiplication exponent.

#### 2. Efficient Training Objectives (Bypassing Heat Kernels)

Derive the training objectives using Itô's lemma (Algorithm 1):
- **Drift term** training: Learn $f(x,t)$ to approximate $f^\star(x,t)$, where the objective function only requires $\nabla\varphi$ and the Euclidean heat kernel (closed-form)
- **Covariance term** training: Learn $g(x,t)$, utilizing manifold symmetry to reduce the free parameters of the $d \times d$ covariance matrix to $n^2$ scalars

#### 3. Average-case Lipschitz Condition (Assumption 2.1)

The projection map $\varphi$ is not globally Lipschitz (e.g., there is a singularity when eigenvalues coincide on SO(n)). This paper utilizes random matrix theory to prove Lipschitz continuity in the **average case**:
- U(n): $L_1 = O(d^{1.5}\sqrt{T}\alpha^{-1/3})$, $L_2 = O(d^2 T \alpha^{-2/3})$
- Sphere: $L_1 = L_2 = O(\alpha^{-1/d})$
- Torus: $L_1 = L_2 = 1$ (always holds)

### Loss & Training

Training consists of two independent optimization problems:
1. Minimizing the MSE loss of the drift model $f_\theta$ (involving $\nabla\varphi$, $\nabla^2\varphi$)
2. Minimizing the covariance model $g_\phi$ to approximate $J_\varphi^T J_\varphi$

Optimized using SGD, requiring only **1** model gradient evaluation + $O(d^{\omega/2})$ arithmetic operations per iteration.

## Key Experimental Results

### Main Results

**Training Speed** (Table 3):
- On SO(n) and U(n) ($d>1000$), the training time per iteration is within only 3x of the Euclidean diffusion model
- Achieving exponential speedup compared to RSGM (heat kernel version)

**Sample Quality** (Table 2, Figure 1):
- On wrapped Gaussian mixture models and quantum evolution operator datasets on tori, SO(n), and U(n)
- Both C2ST and likelihood scores outperform prior methods
- The higher the dimension, the larger the improvement margin

### Ablation Study

- Comparison of training times across different dimensions
- Comparison of the effects of Covariance Model vs. Fixed Covariance

### Key Findings

**Per-iteration Complexity Comparison** (Table 1):

| Method | Gradients | SO(n)/U(n) Compute |
|------|---------|---------------|
| RSGM (heat kernel) | 1 | $2^d + \text{poly}(d,1/\delta)$ |
| RSGM (ISM) | $d$ | $\text{poly}(d,1/\delta)$ |
| TDM (ISM) | $d$ | $\text{poly}(d,1/\delta)$ |
| **Ours** | **1** | $d^{\omega/2}\log(1/\delta)$ |

**Sampling Guarantees** (Theorem 2.2 / Corollary 2.3):
- SO(n)/U(n): $\|\nu - \pi\|_{TV} < O(\varepsilon \cdot d^9 \log(d/\varepsilon))$
- Torus/Sphere: $\|\nu - \pi\|_{TV} < O(\varepsilon \cdot d^6 \log(d/\varepsilon))$
- Number of iterations: poly(d) · log(d/ε)

## Highlights & Insights

1. **Bypassing the heat kernel fundamentally**: Instead of approximating the heat kernel, it completely avoids heat kernel computation through a novel definition of diffusion.
2. **Ingenious use of spatially-varying covariance**: Introducing extra degrees of freedom to compensate for manifold curvature, making the projection framework possible.
3. **Exploiting manifold symmetry**: The covariance of SO(n)/U(n) is entirely determined by $n^2$ scalar parameters, achieving sub-linear computation.
4. **First poly(d) sampling guarantee**: Sampling accuracy and runtime guarantees of prior methods have manifold-dependent constants that are not explicitly specified; this work provides explicit polynomial bounds.
5. **Optimal Transport coupling**: Developed a probability coupling method based on optimal transport to replace Girsanov's theorem, applicable to SDEs with spatially-varying covariance.

## Limitations & Future Work

1. **Limited to symmetric manifolds**: Theoretical guarantees require manifold symmetry and average-case Lipschitz conditions; asymmetric manifolds remain an open problem.
2. **Large exponents in poly(d)**: For instance, a $d^9$ accuracy bound and a $d^{5.5}$ iteration complexity; improving the dimension dependence is a future direction.
3. **Experiments only on synthetic data**: Lacks validation on real-world applications (e.g., protein or molecular conformation generation).
4. **Non-convex training objectives**: As with all diffusion models, global convergence of the training process cannot be guaranteed.
5. **Covariance model increases model complexity**: Requires training an additional $g_\phi$, which increases implementation complexity.

## Related Work & Insights

- **RSGM** [De Bortoli et al., 2022]: Riemannian score-based generative models based on the heat kernel
- **RDM** [Huang et al., 2022]: Riemannian diffusion models using the ISM objective
- **TDM** [Yim et al., 2023]: Triangulated momentum diffusion models
- **SCRD** [Urain et al., 2023]: Scaled Riemannian Diffusion
- **Flow matching on manifolds** [Chen & Lipman, 2023]: Flow matching on manifolds
- Appendix D discusses the possibility of extending to non-smooth spaces such as convex polytopes

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — A completely new projection + Itô's lemma framework, fundamentally shifting the computational paradigm of manifold diffusion
- Experimental Thoroughness: ⭐⭐⭐ — Outstanding theoretical contribution, but experiments are limited to synthetic data
- Writing Quality: ⭐⭐⭐⭐⭐ — Rigorous paper structure, with clear Algorithm pseudocode
- Value: ⭐⭐⭐⭐⭐ — Takes a significant step forward in the efficiency and theoretical guarantees of manifold diffusion models

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Zero-Shot Adaptation of Parameter-Efficient Fine-Tuning in Diffusion Models](zero-shot_adaptation_of_parameter-efficient_fine-tuning_in_diffusion_models.md)
- [\[ICML 2026\] Balancing Fidelity and Diversity in Diffusion Models via Symmetric Attention Decomposition: Hopfield Perspective](../../ICML2026/image_generation/balancing_fidelity_and_diversity_in_diffusion_models_via_symmetric_attention_dec.md)
- [\[ICML 2026\] Rao-Blackwellized Score Matching on Manifolds](../../ICML2026/image_generation/rao-blackwellized_score_matching_on_manifolds.md)
- [\[ICML 2026\] Riemannian MeanFlow for One-Step Generation on Manifolds](../../ICML2026/image_generation/riemannian_meanflow_for_one-step_generation_on_manifolds.md)
- [\[CVPR 2025\] Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](../../CVPR2025/image_generation/efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)

</div>

<!-- RELATED:END -->
