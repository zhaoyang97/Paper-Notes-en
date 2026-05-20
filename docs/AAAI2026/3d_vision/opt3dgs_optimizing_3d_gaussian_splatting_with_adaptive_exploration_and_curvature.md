---
title: >-
  [Paper Note] Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] This paper proposes Opt3DGS, a framework that divides 3DGS training into two phases — exploration and exploitation. The exploration phase employs adaptively weighted SGLD to…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Non-convex Optimization"
  - "Stochastic Gradient Langevin Dynamics"
  - "Quasi-Newton Methods"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: e7f7350d8ee6d612
---

# Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation

**Conference**: AAAI 2026
**arXiv**: [2511.13571](https://arxiv.org/abs/2511.13571)  
**Code**: None  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Non-convex Optimization, Stochastic Gradient Langevin Dynamics, Quasi-Newton Methods, Novel View Synthesis

## TL;DR

This paper proposes Opt3DGS, a framework that divides 3DGS training into two phases — exploration and exploitation. The exploration phase employs adaptively weighted SGLD to escape local optima, while the exploitation phase uses a local quasi-Newton Adam optimizer for precise convergence. The method achieves state-of-the-art rendering quality without modifying the Gaussian representation.

## Background & Motivation

3D Gaussian Splatting (3DGS) models scenes with explicit Gaussian primitives and has demonstrated remarkable performance in novel view synthesis. However, optimizing Gaussian primitives for radiance field reconstruction is inherently a **highly non-convex optimization problem**, presenting two core challenges:

### Challenge 1: Local Optima Traps

The original 3DGS relies on heuristic rules (Adaptive Density Control, ADC) for Gaussian cloning, splitting, and pruning, which lack robustness. A subsequent work, 3DGSMCMC, models the optimization as a Stochastic Gradient Langevin Dynamics (SGLD) process, introducing stochastic noise to promote exploration. However, 3DGSMCMC suffers from a **clustering effect**:

- New Gaussian positions are sampled i.i.d. from an opacity-based probability distribution $\pi(x)$
- Dominant structures discovered early accumulate high opacity, causing subsequent sampling to concentrate heavily in these regions
- This leads to over-accumulation of Gaussians in already well-reconstructed areas, while geometrically complex or under-explored regions receive insufficient coverage
- From an MCMC perspective, this bias confines the sampling chain to a single posterior mode

### Challenge 2: Insufficient Convergence Quality

Existing 3DGS methods predominantly use first-order optimizers (Adam), which lack curvature information and struggle to converge precisely to optimal solutions in later training stages. Although Newton or Levenberg–Marquardt methods have been attempted, they are computationally expensive, requiring the Hessian matrix or its approximation.

**Core Idea**: Divide training into an **Exploration** phase and an **Exploitation** phase to address the above challenges separately.

## Method

### Overall Architecture

- **Exploration phase** (first 29,000 iterations): Employs Adaptively Weighted SGLD (AW-SGLD) to enhance global search and escape local optima.
- **Exploitation phase** (final 1,000 iterations): Employs the Local Quasi-Newton Adam optimizer (LQNAdam) for precise, curvature-aware convergence.

Total training: 30,000 iterations; Gaussian primitive growth rate: 5%.

### Key Designs

1. **Adaptively Weighted SGLD (AW-SGLD)**

   **Core Idea**: Inspired by the flat histogram principle, the posterior distribution is flattened to reduce energy barriers between modes, enabling the model to more readily traverse local optima.

   The configuration of Gaussian primitives is treated as a probability distribution:
   $P(g) \propto \exp\left(-\frac{\mathcal{L}_{total}(g)}{\tau}\right)$

   The sample space is partitioned into $m$ sub-regions by energy level: $\mathcal{G}_n = \{g: u_{n-1} < \mathcal{L}_{total}(g) < u_n\}$.

   A flattened distribution $\rho(g)$ is constructed as:
   $\rho(g) \propto \frac{P(g)}{\Psi^\zeta(\Theta, \mathcal{L}_{total}(g))}$
   where $\zeta > 0$ controls the degree of flattening, $\Psi$ is an energy-based piecewise exponential interpolation weighting function, and the weight vector $\Theta$ is updated online via stochastic approximation.

   The flattened distribution introduces an additional **gradient multiplier** $\nu$:
   $\nu = 1 + \zeta\tau \frac{\log\theta(J(g)) - \log(\theta(J(g)-1) \vee 1)}{\Delta u}$

   The gradient multiplier is incorporated into the SGLD update:
   $g_k \leftarrow g_{k-1} - \lambda_{lr} \cdot \nu \cdot \nabla_g \mathbb{E}[\mathcal{L}_{total}(g_{k-1})] + \lambda_{noise} \cdot \epsilon$

   The weight vector $\Theta$ is updated via stochastic approximation:
   $\theta_k(i) = \theta_{k-1}(i) + \lambda_\theta \theta_{k-1}^\zeta(J(g_k)) \cdot (1_{i=J(g_k)} - \theta_{k-1}(i))$

   **Design Motivation**: Directly increasing the noise magnitude $\lambda_{noise}$ is not robust due to varying scene complexity. The adaptive weighting approach automatically adjusts exploration intensity according to the energy distribution, achieving more uniform mode exploration by flattening the posterior. Regions with high energy (i.e., poorly reconstructed regions) receive greater exploration incentives.

2. **Local Quasi-Newton Adam Optimizer (LQNAdam)**

   **Core Idea**: During the exploitation phase, L-BFGS is applied independently to each Gaussian primitive to estimate a quasi-Newton direction, which is then used as a pseudo-gradient input to Adam, yielding curvature-aware update directions.

   Procedure:
   - L-BFGS (history length $K=5$) is applied independently to the position $\mu$ of each Gaussian primitive to estimate the quasi-Newton direction $\mathbb{D}$
   - $\mathbb{D}$ is fed as a pseudo-gradient to Adam to compute the final update direction $\text{Adam}(\mathbb{D})$
   - The update rule under the MCMC framework is:
   $\mu_{t+1} = \mu_t - \lambda_{lr} \cdot \text{Adam}(\mathbb{D}) + \lambda_{noise} \cdot \epsilon_\mu$

   **Key Design Choices**:
   - "Local": each Gaussian primitive is processed independently, enabling parallel execution on CUDA
   - No line search: Adam replaces the line search used in conventional quasi-Newton methods, preserving robustness
   - L-BFGS does not require explicit Hessian computation and is compatible with various loss functions
   - The exploitation phase replaces the L1 loss with L2 loss and disables the gradient multiplier $\nu$

   **Design Motivation**: Based on observations from 3DGS², the positional attributes exert the greatest influence on rendering quality, and Gaussian primitives are weakly coupled with each other, making independent quasi-Newton optimization of positions well-suited to this setting.

3. **Exploration-to-Exploitation Switching Strategy**

    - Switching occurs at iteration 29,000
    - The exploration phase uses AW-SGLD with a 2,500-iteration warm-up to stabilize energy estimation
    - The exploitation phase disables the gradient multiplier, switches to L2 loss, and activates LQNAdam
    - The flattening coefficient $\zeta = 0.75$ is applied universally across all datasets

### Loss & Training

The same loss function as 3DGSMCMC is used:
$$L_{total} = (1-\lambda_{ssim}) \times L_1 + \lambda_{ssim} \times L_{ssim} + \lambda_o \sum_i |o_i|_1 + \lambda_\Sigma \sum_{ij} |\sqrt{\text{eig}_j(\Sigma_i)}|_1$$

The last two terms are opacity sparsity regularization and covariance matrix scale constraints, respectively. L1 loss is replaced by L2 during the exploitation phase.

Energy intervals: $[0.0, 0.2]$ for most scenes and $[0.0, 0.3]$ for special scenes (Train), divided into 200 uniform bins.

## Key Experimental Results

### Main Results

**Standard Setting (SfM Initialization)**:

| Method | MipNeRF360 PSNR/SSIM/LPIPS | T&T PSNR/SSIM/LPIPS | DeepBlending PSNR/SSIM/LPIPS |
|---|---|---|---|
| 3DGS | 28.69/0.870/0.182 | 23.14/0.841/0.183 | 29.41/0.903/0.243 |
| 3DGSMCMC | 29.89/0.900/0.190 | 24.29/0.860/0.190 | 29.67/0.900/0.320 |
| SSS | 29.90/0.893/0.145 | 24.87/0.873/0.138 | 30.07/0.907/0.247 |
| **Opt3DGS** | **29.96**/0.897/**0.143** | 24.80/**0.875**/**0.139** | **30.09**/**0.911**/**0.229** |

Opt3DGS achieves the best performance on 5 of 9 metrics and second-best on the remaining 4. Compared to 3DGSMCMC, LPIPS improves by 26.84% on T&T.

**Random Initialization (No SfM)**:

| Method | MipNeRF360 PSNR/SSIM/LPIPS | T&T PSNR/SSIM/LPIPS | DeepBlending PSNR/SSIM/LPIPS |
|---|---|---|---|
| 3DGS | 27.89/0.840/0.260 | 21.93/0.800/0.270 | 29.55/0.900/0.330 |
| 3DGSMCMC | 29.72/0.890/0.190 | 24.21/0.860/0.190 | 29.71/0.900/0.320 |
| **Opt3DGS** | **29.78/0.893/0.149** | **24.39/0.865/0.151** | **29.90/0.905/0.236** |

Opt3DGS achieves the best results on all 9 metrics, demonstrating that even under poor initialization, the proposed optimization framework can guide the model toward high-quality solutions.

### Ablation Study

| Configuration | Train PSNR | Truck PSNR | Train Time | Truck Time |
|---|---|---|---|---|
| Baseline (3DGSMCMC) | 22.47 | 26.11 | 11 min | 22 min |
| + AW-SGLD | 22.74 (+0.27) | 26.49 (+0.38) | 12 min | 22 min |
| + AW-SGLD + LQNAdam | **23.01** (+0.54) | **26.61** (+0.50) | 12 min | 23 min |

Both components contribute positively; AW-SGLD yields the larger gain, while LQNAdam provides further refinement. The additional computational overhead is less than 1 minute.

**Effect of flattening coefficient $\zeta$**: The optimal range is $\zeta = 0.75$–$0.8$; values that are too small result in insufficient exploration, while excessively large values may cause training instability.

### Key Findings

- Pure optimization improvements (without modifying the Gaussian representation) can match or surpass methods that modify the representation (e.g., SSS)
- Advantages are more pronounced under random initialization, confirming that enhanced exploration is especially valuable under challenging conditions
- Opt3DGS maintains its advantages under high-resolution inputs, which correspond to more complex posterior landscapes
- Opt3DGS performs well with a limited number of Gaussians, indicating that improved optimization efficiency can compensate for reduced representational capacity
- Additional computational overhead is minimal (approximately 1 minute)

## Highlights & Insights

1. **Purity of the optimization perspective**: This work improves 3DGS purely from an optimization standpoint, without modifying the Gaussian representation or introducing auxiliary networks, supporting the view that "optimization matters more than representation."
2. **Transferability of the exploration-exploitation framework**: The two-phase optimization framework is independent of the representation and can serve as a plug-and-play replacement for the optimization component in other 3DGS systems.
3. **Application of the flat histogram principle to 3DGS**: Advanced sampling techniques from statistical physics and MCMC (originally developed for simulating protein folding and similar problems) are introduced into 3D reconstruction, demonstrating valuable cross-domain inspiration.
4. **Clever combination of quasi-Newton direction and Adam**: LQNAdam preserves the robustness of Adam while incorporating curvature information, avoiding the line search overhead of traditional second-order methods.
5. **Greater advantages under challenging conditions (random initialization, high resolution, fewer Gaussians)**: This indicates that the value of enhanced exploration is greatest when the solution space is complex.

## Limitations & Future Work

- The flattening coefficient $\zeta$ and energy interval bounds still require manual specification and may need tuning for different scenes
- The exploitation phase spans only 1,000 iterations, which may be insufficient to fully leverage curvature information
- The L-BFGS history length is fixed at 5; adaptive adjustment has not been explored
- The quasi-Newton direction is applied only to positional attributes and has not been extended to other Gaussian parameters (color, opacity, etc.)
- Opt3DGS and SSS trade off on certain metrics; combining Opt3DGS's optimization strategy with a stronger representation could be a promising direction

## Related Work & Insights

- **3DGSMCMC (2024)**: The pioneering work modeling 3DGS optimization as SGLD/MCMC and the direct baseline of this paper
- **SSS (2025)**: Improves the Gaussian representation (Student's T distribution) combined with SGHMC sampling; complementary to the present work
- **Wang-Landau Algorithm (2001)**: The origin of the flat histogram principle and the theoretical inspiration of this paper
- **L-BFGS**: A classical limited-memory quasi-Newton optimization method, applied here innovatively to the independent optimization of individual Gaussian primitives
- **Insight**: Optimization strategy improvements are orthogonal to representation improvements and the two can be combined; sampling methods from statistical mechanics deserve greater attention in this domain

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Introduces the flat histogram principle and quasi-Newton directions into 3DGS optimization from a unique perspective)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Covers standard/random initialization/high-resolution/few-Gaussian settings comprehensively)
- Writing Quality: ⭐⭐⭐⭐⭐ (Rigorous theoretical derivation, clear Bayesian perspective analysis, intuitive illustrations)
- Value: ⭐⭐⭐⭐⭐ (Highly generalizable framework; can serve as a standard optimization component for 3DGS)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives](../../CVPR2026/3d_vision/prune_wisely_reconstruct_sharply_compact_3d_gaussian_splatting_via_adaptive_prun.md)
- [\[AAAI 2026\] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)
- [\[AAAI 2026\] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction](oceansplat_object-aware_gaussian_splatting_with_trinocular_view_consistency_for_.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)
- [\[AAAI 2026\] Debiasing Diffusion Priors via 3D Attention for Consistent Gaussian Splatting](debiasing_diffusion_priors_via_3d_attention_for_consistent_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
