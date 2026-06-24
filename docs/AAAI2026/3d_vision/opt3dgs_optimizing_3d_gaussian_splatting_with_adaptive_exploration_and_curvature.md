---
title: >-
  [Paper Note] Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation
description: >-
  [AAAI 2026][3D Vision][3D Gaussian Splatting] This paper proposes the Opt3DGS framework, which divides 3DGS training into exploration and exploitation stages. The exploration stage utilizes adaptive weighted SGLD to escape local optima, while the exploitation stage adopts a local quasi-Newton Adam optimizer for precise convergence, achieving SOTA rendering quality without modifying the Gaussian representation.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Non-convex Optimization"
  - "Stochastic Gradient Langevin Dynamics"
  - "Quasi-Newton Method"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: 51308509598fa853
---

# Opt3DGS: Optimizing 3D Gaussian Splatting with Adaptive Exploration and Curvature-Aware Exploitation

**Conference**: AAAI 2026  
**arXiv**: [2511.13571](https://arxiv.org/abs/2511.13571)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, Non-convex Optimization, Stochastic Gradient Langevin Dynamics, Quasi-Newton Method, Novel View Synthesis

## TL;DR

This paper proposes the Opt3DGS framework, which divides 3DGS training into exploration and exploitation stages. The exploration stage utilizes adaptive weighted SGLD to escape local optima, while the exploitation stage adopts a local quasi-Newton Adam optimizer for precise convergence, achieving SOTA rendering quality without modifying the Gaussian representation.

## Background & Motivation

3D Gaussian Splatting (3DGS) has achieved outstanding performance in novel view synthesis by modeling scenes with explicit Gaussian primitives. However, optimizing Gaussian primitives to reconstruct radiance fields is essentially a **highly non-convex optimization problem**, facing two key challenges:

### Challenge 1: Local Optima Trap

The original 3DGS utilizes heuristic rules (Adaptive Density Control, ADC) for cloning, splitting, and pruning Gaussians, which lack robustness. Subsequent work 3DGSMCMC models optimization as a Stochastic Gradient Langevin Dynamics (SGLD) process, introducing random noise to facilitate exploration. However, 3DGSMCMC suffers from a **clustering effect**:

- The positions of newly added Gaussians are sampled i.i.d. from an opacity-based probability distribution $\pi(x)$.
- After dominant structures discovered early on become highly opaque, subsequent sampling is heavily biased toward these regions.
- This leads to an over-accumulation of Gaussians in empty or already well-reconstructed regions, while geometrically complex or under-explored areas remain uncovered.
- From an MCMC perspective, this bias confines the sampling chain within a single posterior mode.

### Challenge 2: Insufficient Convergence Quality

Existing 3DGS methods commonly employ first-order optimizers (Adam), which lack curvature information and struggle to converge precisely to optimal points in the late stages of training. Although some works attempt to use Newton's method or the Levenberg-Marquardt (LM) algorithm, they are computationally expensive (requiring the Hessian matrix or its approximation).

**Core Idea**: The training process is divided into two stages, **Exploration** and **Exploitation**, to address the two aforementioned problems respectively.

## Method

### Overall Architecture

- **Exploration Stage** (first 29,000 iterations): Employs Adaptive Weighted SGLD (AW-SGLD) to enhance global search and escape local optima.
- **Exploitation Stage** (last 1,000 iterations): Employs Local Quasi-Newton Adam (LQNAdam) for precise, curvature-aware convergence.
- Total training consists of 30,000 iterations, with a Gaussian primitive growth rate of 5%.

### Key Designs

1. **Adaptive Weighted SGLD (AW-SGLD)**

   **Core Idea**: Inspired by the "flat histogram" principle, this design flattens the posterior distribution to lower the energy barriers between modes, making it easier for the model to cross local optima.

   Viewing the configuration of Gaussian primitives as a probability distribution:
    $P(g) \propto \exp\left(-\frac{\mathcal{L}_{total}(g)}{\tau}\right)$

   The sample space is partitioned into $m$ sub-regions based on energy levels: $\mathcal{G}_n = \{g: u_{n-1} < \mathcal{L}_{total}(g) < u_n\}$.

   A flattened distribution $\rho(g)$ is constructed:
    $\rho(g) \propto \frac{P(g)}{\Psi^\zeta(\Theta, \mathcal{L}_{total}(g))}$
   where $\zeta > 0$ controls the degree of flattening, $\Psi$ is a piecewise exponential interpolation weighting function based on energy, and the weight vector $\Theta$ is updated online via stochastic approximation.

   The flattened distribution introduces an additional **gradient multiplier** $\nu$:
    $\nu = 1 + \zeta\tau \frac{\log\theta(J(g)) - \log(\theta(J(g)-1) \vee 1)}{\Delta u}$

   Incorporating the gradient multiplier into the SGLD update:
    $g_k \leftarrow g_{k-1} - \lambda_{lr} \cdot \nu \cdot \nabla_g \mathbb{E}[\mathcal{L}_{total}(g_{k-1})] + \lambda_{noise} \cdot \epsilon$

   The weight vector $\Theta$ is updated using stochastic approximation:
    $\theta_k(i) = \theta_{k-1}(i) + \lambda_\theta \theta_{k-1}^\zeta(J(g_k)) \cdot (1_{i=J(g_k)} - \theta_{k-1}(i))$

   **Design Motivation**: Simply increasing the noise $\lambda_{noise}$ is not robust (due to varying scene complexities). The adaptive weighting method automatically adjusts the intensity of exploration based on the energy distribution, yielding more uniform mode exploration by flattening the posterior. High-energy regions (which correspond to poorly reconstructed areas) receive stronger exploration enhancement.

2. **Local Quasi-Newton Optimizer (LQNAdam)**

   **Core Idea**: During the exploitation stage, L-BFGS is applied independently to each Gaussian primitive to estimate the quasi-Newton direction, which serves as the pseudo-gradient input for Adam to obtain a curvature-aware update direction.

   Specific steps:
    - Perform L-BFGS (with history length $K=5$) independently for each Gaussian primitive's position $\mu$ to estimate the quasi-Newton direction $\mathbb{D}$.
    - Feed $\mathbb{D}$ as a pseudo-gradient into Adam to compute the final update direction $\text{Adam}(\mathbb{D})$.
    - The update rule under the MCMC framework is:
    $\mu_{t+1} = \mu_t - \lambda_{lr} \cdot \text{Adam}(\mathbb{D}) + \lambda_{noise} \cdot \epsilon_\mu$

   **Key Design Choices**:
    - "Local": Each Gaussian primitive is processed independently, facilitating parallelization on CUDA.
    - No line search: Adam is used to replace the line search of traditional quasi-Newton methods, maintaining robustness.
    - L-BFGS does not require computing the Hessian matrix and is compatible with various loss functions.
    - The exploitation stage replaces the L1 loss with L2 loss and disables the gradient multiplier $\nu$.

   **Design Motivation**: Based on the observations in 3DGS², position attributes have the greatest impact on rendering quality, and there is weak coupling between different Gaussian primitives, making independent quasi-Newton optimization for positions highly suitable.

3. **Exploration-to-Exploitation Switch Strategy**

    - Switch occurs at the 29,000th iteration.
    - The exploration stage utilizes AW-SGLD, with the first 2,500 iterations serving as a warm-up phase to stabilize energy estimation.
    - The exploitation stage disables the gradient multiplier and switches to the L2 loss and LQNAdam.
    - The flattening coefficient $\zeta = 0.75$ is generalized across all datasets.

### Loss & Training

Same loss function as 3DGSMCMC:
$$L_{total} = (1-\lambda_{ssim}) \times L_1 + \lambda_{ssim} \times L_{ssim} + \lambda_o \sum_i |o_i|_1 + \lambda_\Sigma \sum_{ij} |\sqrt{\text{eig}_j(\Sigma_i)}|_1$$

The latter two terms are opacity sparsity regularization and covariance matrix scale constraints, respectively. The exploitation stage replaces L1 with L2.

Energy intervals: [0.0, 0.2] for most scenes, [0.0, 0.3] for specific scenes (e.g., Train), divided into 200 uniform bins.

## Key Experimental Results

### Main Results

**Standard Setting (SfM Initialization)**:

| Method | MipNeRF360 PSNR/SSIM/LPIPS | T&T PSNR/SSIM/LPIPS | DeepBlending PSNR/SSIM/LPIPS |
|------|---------------------------|---------------------|----------------------------|
| 3DGS | 28.69/0.870/0.182 | 23.14/0.841/0.183 | 29.41/0.903/0.243 |
| 3DGSMCMC | 29.89/0.900/0.190 | 24.29/0.860/0.190 | 29.67/0.900/0.320 |
| SSS | 29.90/0.893/0.145 | 24.87/0.873/0.138 | 30.07/0.907/0.247 |
| **Opt3DGS** | **29.96**/0.897/**0.143** | 24.80/**0.875**/**0.139** | **30.09**/**0.911**/**0.229** |

Achieves the best performance in 5 out of 9 metrics, and the second best in 4 metrics. Compared to 3DGSMCMC, it improves LPIPS by 26.84% on the T&T dataset.

**Random Initialization (Without SfM)**:

| Method | MipNeRF360 PSNR/SSIM/LPIPS | T&T PSNR/SSIM/LPIPS | DeepBlending PSNR/SSIM/LPIPS |
|------|---------------------------|---------------------|----------------------------|
| 3DGS | 27.89/0.840/0.260 | 21.93/0.800/0.270 | 29.55/0.900/0.330 |
| 3DGSMCMC | 29.72/0.890/0.190 | 24.21/0.860/0.190 | 29.71/0.900/0.320 |
| **Opt3DGS** | **29.78/0.893/0.149** | **24.39/0.865/0.151** | **29.90/0.905/0.236** |

Achieves optimal performance across all 9 metrics, demonstrating that even with a poor initial state, the optimization framework can still guide the model to find high-quality solutions.

### Ablation Study

| Configuration | Train PSNR | Truck PSNR | Train Time | Truck Time |
|------|-----------|-----------|----------|----------|
| Baseline (3DGSMCMC) | 22.47 | 26.11 | 11min | 22min |
| + AW-SGLD | 22.74 (+0.27) | 26.49 (+0.38) | 12min | 22min |
| + AW-SGLD + LQNAdam | **23.01** (+0.54) | **26.61** (+0.50) | 12min | 23min |

Both components make contributions; AW-SGLD provides a larger contribution, while LQNAdam brings further refinement. The extra computational overhead is < 1 minute.

**Impact of Flattening Coefficient $\zeta$**: $\zeta = 0.75-0.8$ is the optimal range; a value that is too small leads to insufficient exploration, while a value that is too large may cause training instability.

### Key Findings

- Pure optimization improvements (without modifying the Gaussian representation) can achieve or even exceed methods that modify representations (e.g., SSS).
- The advantage is more pronounced under random initialization, proving that enhancing the exploration capability is particularly crucial in difficult conditions.
- Under high-resolution inputs (which possess more complex posterior topography), the advantages of Opt3DGS persist.
- Opt3DGS still performs exceptionally well with a limited number of Gaussians, indicating that improved optimization efficiency can compensate for insufficient representation capacity.
- The additional computational overhead is extremely minimal (approximately 1 minute).

## Highlights & Insights

1. **Purity of Optimization Perspective**: This work enhances 3DGS entirely from an optimization standpoint, without modifying Gaussian representations or introducing auxiliary networks, proving the viewpoint that "optimization matters more than representation."
2. **Transferability of the Exploration-Exploitation Framework**: This two-stage optimization framework is independent of the representation type and can act as a plug-and-play module to replace optimization components in other 3DGS systems.
3. **Application of the Flat Histogram Principle in 3DGS**: This work introduces advanced sampling techniques from statistical physics/MCMC (originally used for simulating protein folding, etc.) into 3D reconstruction, offering cross-disciplinary inspiration.
4. **Ingenious Combination of Quasi-Newton Directions and Adam**: LQNAdam preserves the robustness of Adam while introducing curvature information, avoiding the line search overhead of traditional second-order methods.
5. **More Pronounced Advantages under Challenging Conditions (Random Initialization, High Resolution, Few Gaussians)**: This demonstrates that enhanced exploration capability holds the greatest value when the solution space is complex.

## Limitations & Future Work

- The flattening coefficient $\zeta$ and the energy intervals still require manual configuration and may require fine-tuning across different scenes.
- Merely 1,000 iterations in the exploitation stage might be insufficient to fully exploit the curvature information.
- The history length of L-BFGS is fixed at 5, and the possibility of adaptive adjustment remains unexplored.
- Quasi-Newton directions are applied only to position attributes, and have not been extended to other Gaussian parameters (e.g., color, opacity).
- Since the method alternates wins with SSS on certain metrics, combining the optimization strategy of Opt3DGS with better representations could be a promising direction.

## Related Work & Insights

- **3DGSMCMC (2024)**: A pioneering work modeling 3DGS optimization as an SGLD/MCMC process, serving as the direct baseline for this paper.
- **SSS (2025)**: Complements this work by improving Gaussian representation (using Student's t-distributions) + SGHMC sampling.
- **Wang-Landau Algorithm (2001)**: The original source of the flat histogram principle, serving as the theoretical inspiration for this paper.
- **L-BFGS**: A classic limited-memory quasi-Newton optimization method, which this paper innovatively applies to the independent optimization of each Gaussian primitive.
- **Insights**: Improvements in optimization strategies are orthogonal to innovations in representation; the two can be used in combination. Sampling methods from statistical mechanics deserve more attention.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Introduces the flat histogram principle and quasi-Newton directions into 3DGS optimization, offering a unique perspective)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Standard/random initialization, high-resolution, few-Gaussians; covers a comprehensive set of conditions)
- Writing Quality: ⭐⭐⭐⭐⭐ (Rigorous theoretical derivation, clear analysis from a Bayesian perspective, intuitive illustrations)
- Value: ⭐⭐⭐⭐⭐ (The framework matches high versatility and can serve as a standard component for 3DGS optimization)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)
- [\[CVPR 2026\] Prune Wisely, Reconstruct Sharply: Compact 3D Gaussian Splatting via Adaptive Pruning and Difference-of-Gaussian Primitives](../../CVPR2026/3d_vision/prune_wisely_reconstruct_sharply_compact_3d_gaussian_splatting_via_adaptive_prun.md)
- [\[AAAI 2026\] OceanSplat: Object-aware Gaussian Splatting with Trinocular View Consistency for Underwater Scene Reconstruction](oceansplat_object-aware_gaussian_splatting_with_trinocular_view_consistency_for_.md)
- [\[AAAI 2026\] Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting](gaussian_blending_rethinking_alpha_blending_in_3d_gaussian_splatting.md)
- [\[ICLR 2026\] Gradient-Direction-Aware Density Control for 3D Gaussian Splatting](../../ICLR2026/3d_vision/gradient-direction-aware_density_control_for_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
