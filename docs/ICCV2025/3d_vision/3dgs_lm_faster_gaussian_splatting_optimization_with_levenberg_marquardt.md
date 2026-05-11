---
title: >-
  [Paper Note] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes 3DGS-LM, which replaces the ADAM optimizer in 3D Gaussian Splatting with a customized second-order Levenberg-Marquardt (LM) optimizer. Combined with an ef…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Levenberg-Marquardt"
  - "Optimization Acceleration"
  - "Novel View Synthesis"
  - "CUDA Parallelism"
date: 2026-05-08
content_hash: c734809a64b6a645
---

# 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt

**Conference**: ICCV 2025
**arXiv**: N/A (CVF Open Access)
**Code**: [https://lukashoel.github.io/3DGS-LM/](https://lukashoel.github.io/3DGS-LM/)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Levenberg-Marquardt, Optimization Acceleration, Novel View Synthesis, CUDA Parallelism

## TL;DR

This paper proposes 3DGS-LM, which replaces the ADAM optimizer in 3D Gaussian Splatting with a customized second-order Levenberg-Marquardt (LM) optimizer. Combined with an efficient GPU parallelization scheme and a gradient caching structure, the method achieves a 20% training speedup while preserving reconstruction quality.

## Background & Motivation

1. **Background**: 3D Gaussian Splatting (3DGS) is one of the most popular novel view synthesis methods, representing scenes as a set of 3D Gaussians via a differentiable rasterizer, enabling real-time rendering and high-quality image synthesis. Existing acceleration methods primarily focus on reducing the number of Gaussians (e.g., improved densification strategies) and accelerating the rasterizer implementation.

2. **Limitations of Prior Work**: Despite numerous improvements to rasterization and densification, 3DGS optimization still requires thousands of gradient descent iterations to converge, potentially taking up to an hour on high-resolution real-world datasets. All existing methods rely on ADAM, a first-order optimizer, to fit Gaussian parameters, which constitutes the primary bottleneck in training time.

3. **Key Challenge**: First-order optimizers such as ADAM are straightforward to implement but yield limited update quality per iteration, requiring a large number of iterations to converge. In traditional 3D reconstruction tasks such as RGB-D fusion, second-order methods like Gauss-Newton / Levenberg-Marquardt converge in orders-of-magnitude fewer iterations. However, directly applying LM to 3DGS poses severe memory and computational challenges, as the Jacobian matrix is too large to store explicitly.

4. **Goal**: Design an efficient LM optimizer to replace ADAM in 3DGS, significantly reducing optimization time while maintaining reconstruction quality.

5. **Key Insight**: The authors observe that LM converges extremely fast (only 5–10 iterations) given a good initialization, and therefore propose a two-stage strategy: first use ADAM for densification and coarse optimization, then apply LM for rapid convergence to the final result.

6. **Core Idea**: Solve the normal equations in a matrix-free manner via the Preconditioned Conjugate Gradient (PCG) algorithm, and design a cache-driven per-pixel-per-splat parallelization scheme to efficiently compute Jacobian-vector products.

## Method

### Overall Architecture

3DGS-LM adopts a two-stage optimization pipeline. The inputs are an SfM point cloud and a set of calibrated images. In the first stage, the standard 3DGS ADAM optimizer runs for 20K iterations, performing Gaussian densification and initial parameter fitting. In the second stage, the optimizer switches to LM, which converges to the final result in only 5 iterations. The output is a high-quality 3D Gaussian scene representation.

### Key Designs

1. **LM Optimizer for 3DGS**:

    - **Function**: Replace ADAM with the Levenberg-Marquardt algorithm based on a second-order approximation to achieve more efficient parameter updates.
    - **Mechanism**: The rendering loss is rewritten as a sum-of-squares form $E(x) = \sum_i r_i^2$, where $r_i^{abs} = \sqrt{\lambda_1}|c_i - C_i|$ and $r_i^{SSIM} = \sqrt{\lambda_2}(1 - SSIM(c_i, C_i))$. At each iteration, the update direction $\Delta$ is obtained by solving the normal equations $(J^TJ + \lambda_{reg}\text{diag}(J^TJ))\Delta = -J^TF(x)$, and the optimal step size $\gamma$ is determined via line search. The PCG algorithm is employed to solve the system in a matrix-free manner, avoiding explicit construction and storage of the large Jacobian matrix.
    - **Design Motivation**: Each LM iteration exploits curvature information, yielding update directions of far superior quality compared to ADAM's first-order gradients. This allows convergence in as few as 5–10 iterations, whereas ADAM requires 10K+.

2. **Cache-Driven CUDA Parallelization Scheme**:

    - **Function**: Efficiently implement Jacobian-vector product computations within the PCG algorithm.
    - **Mechanism**: The conventional 3DGS gradient computation uses per-pixel parallelization (each thread processes one ray). In PCG, ray marching must be repeated multiple times, causing intermediate states (e.g., $T_s$, $\partial c / \partial \alpha_s$) to be recomputed repeatedly. This work proposes per-pixel-per-splat parallelization instead: a gradient cache is constructed once, storing the intermediate gradients $\partial c / \partial s$ for each pixel-splat pair. During PCG iterations, the cache is read to compute $u = Jp$ and $g = J^Tu$ in parallel, decoupling splats along rays for large-scale parallelism.
    - **Design Motivation**: Per-pixel parallelization is highly inefficient in PCG (the same intermediate state is recomputed 18 times). The caching scheme replaces redundant computation with a single cache construction followed by multiple fast reads, at the cost of additional GPU memory.

3. **Image Subsampling Scheme**:

    - **Function**: Control cache memory consumption to scale the method to high-resolution, large-scale datasets.
    - **Mechanism**: All images are divided into $n_b$ batches. The normal equations are solved independently for each batch to obtain update vectors $\Delta_i$, which are then merged by weighted averaging: $\Delta = \sum_i M_i \Delta_i / \sum_k M_k$, where the weights $M_i = \text{diag}(J_i^T J_i)$ reflect each Gaussian parameter's contribution in that batch. In practice, 25–70 images per batch and at most 4 batches are used.
    - **Design Motivation**: The cache for the full dataset is too large to fit in GPU memory. The subsampling scheme reduces memory consumption to a manageable level through batched processing and weighted merging, without significantly affecting convergence.

### Loss & Training

- The objective function is identical to the original 3DGS: $L(x) = \frac{1}{N}\sum_i (\lambda_1|c_i - C_i| + \lambda_2(1 - SSIM(c_i, C_i)))$, with $\lambda_1 = 0.8$ and $\lambda_2 = 0.2$.
- Two-stage strategy: Stage 1 runs ADAM for 20K iterations (with densification); Stage 2 runs LM for 5 iterations (each with 8 PCG inner iterations).
- The regularization strength $\lambda_{reg}$ is adaptively adjusted based on update quality.

## Key Experimental Results

### Main Results

| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ | Time (s) |
|--------|------|-------|-------|--------|---------|
| MipNeRF-360 | 3DGS | 27.40 | 0.813 | 0.218 | 1271 |
| MipNeRF-360 | 3DGS + LM | 27.39 | 0.813 | 0.221 | **972** |
| MipNeRF-360 | DISTWAR | 27.42 | 0.813 | 0.217 | 966 |
| MipNeRF-360 | DISTWAR + LM | 27.42 | 0.814 | 0.221 | **764** |
| Deep Blending | Taming-3DGS | 29.84 | 0.900 | 0.274 | 447 |
| Deep Blending | Taming-3DGS + LM | 29.91 | 0.901 | 0.275 | **347** |

### Ablation Study

| Configuration | PSNR↑ | Time (s) | Notes |
|------|-------|---------|------|
| L1/SSIM + ADAM | 27.23 | 1573 | Original 3DGS |
| L1/SSIM + LM | 27.29 | 1175 | 25% speedup, consistent quality |
| L2 + ADAM | 27.31 | 1528 | Inferior quality with L2 loss |
| Batch=100 | 33.77 | 242 | Highest quality |
| Batch=40 | 33.51 | 212 | Slight quality drop, faster |

### Key Findings

- The LM optimizer can be **seamlessly integrated** into multiple 3DGS baselines (3DGS, DISTWAR, gsplat, Taming-3DGS), achieving an average 20% speedup with virtually no quality degradation.
- LM requires only 5 iterations to match the quality achieved by ADAM in 10K iterations, though each LM iteration is computationally heavier.
- Memory consumption increases substantially (average 53 GB vs. 6–11 GB for baselines), reflecting a speed–memory trade-off.

## Highlights & Insights

- **Successful Application of Second-Order Optimization to 3DGS**: The paper demonstrates the viability of LM optimizers for explicit 3D representation optimization. The per-iteration update quality far exceeds that of first-order methods, with 5 LM iterations equivalent to 10K ADAM iterations.
- **Cache-Driven Parallelization Design**: By caching intermediate ray-marching states and sorting them by Gaussian index, the paper elegantly transforms the Jacobian-vector product computation in PCG from per-pixel to per-pixel-per-splat parallelization. This idea is transferable to other differentiable rendering tasks requiring frequent computation of Jacobian-related quantities.
- **Orthogonality Advantage**: The LM optimizer is orthogonal and complementary to rasterization acceleration and densification improvements, and can be directly combined with existing work.

## Limitations & Future Work

- High memory consumption (53 GB); high-resolution scenes may require CPU offloading.
- The densification stage still relies on ADAM and cannot be unified within the LM framework.
- The method is sensitive to initialization; applying LM directly on SfM initialization yields poor results.

## Related Work & Insights

- **vs. ADAM with Multi-View Batches**: Increasing ADAM's batch size to the same scale still requires more iterations and time to converge to equivalent quality.
- **vs. gsplat / DISTWAR**: These methods accelerate rasterization forward and backward passes, and are orthogonal and complementary to the optimizer replacement proposed in this paper.

## Rating

- Novelty: ⭐⭐⭐⭐ Introduces LM into 3DGS optimization and addresses the engineering challenges of GPU parallelization.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 3 datasets × 4 baselines with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and algorithmic details are complete.
- Value: ⭐⭐⭐⭐ A practical plug-and-play acceleration solution, with memory overhead as the primary limitation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Faster and Better 3D Splatting via Group Training](faster_and_better_3d_splatting_via_group_training.md)
- [\[ICCV 2025\] RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS](robustsplat_decoupling_densification_and_dynamics_for_transient-free_3dgs.md)
- [\[ICCV 2025\] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction](mugs_multi-baseline_generalizable_gaussian_splatting_reconstruction.md)
- [\[ICCV 2025\] EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images](evagaussians_event_stream_assisted_gaussian_splatting_from_blurry_images.md)
- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
