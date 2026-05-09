---
title: >-
  [Paper Note] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes 3DGS-LM, which replaces the Adam optimizer in 3DGS with a customized Levenberg-Marquardt (LM) optimizer. By introducing a GPU cache-driven parallelization scheme for efficient Jacobian-vector product computation, the method achieves a 20% speedup in 3DGS optimization while maintaining equivalent reconstruction quality.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - Levenberg-Marquardt
  - Optimization Acceleration
  - Novel View Synthesis
  - CUDA Parallelism
date: 2026-05-08
content_hash: 956ae89af0530d2c
---

# 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt

**Conference**: ICCV 2025
**arXiv**: [2409.12892](https://arxiv.org/abs/2409.12892)
**Code**: [https://lukashoel.github.io/3DGS-LM/](https://lukashoel.github.io/3DGS-LM/)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Levenberg-Marquardt, Optimization Acceleration, Novel View Synthesis, CUDA Parallelism

## TL;DR
This paper proposes 3DGS-LM, which replaces the Adam optimizer in 3DGS with a customized Levenberg-Marquardt (LM) optimizer. By introducing a GPU cache-driven parallelization scheme for efficient Jacobian-vector product computation, the method achieves a 20% speedup in 3DGS optimization while maintaining equivalent reconstruction quality.

## Background & Motivation
3D Gaussian Splatting (3DGS) has emerged as one of the most prevalent 3D scene representation methods, achieving high-quality real-time rendering by projecting 3D Gaussians into 2D splats via a differentiable rasterizer. Nevertheless, the optimization process still requires tens of thousands of iterations, with convergence taking up to an hour.

Existing acceleration approaches fall into two main categories: improving the efficiency of the differentiable rasterizer (e.g., warp reduction in DISTWAR) and reducing the number of Gaussians through improved densification strategies (e.g., Taming-3DGS). However, these methods continue to rely on Adam, a first-order optimizer that requires a large number of iterations to converge.

The root cause lies in the fundamental limitation of first-order optimizers: Adam computes update directions based solely on gradients, which constrains convergence speed. Second-order optimizers such as LM derive higher-quality update directions by solving the normal equations, substantially reducing the required number of iterations. Yet efficient Jacobian-vector product computation under the large parameter counts and high-resolution images characteristic of 3DGS remains highly challenging.

The paper's starting point is to design a cache-based parallelization scheme that makes the LM optimizer practically viable within 3DGS. **Core Idea**: cache intermediate gradients from α-blending and shift the parallelization paradigm from per-pixel to per-pixel-per-splat, substantially accelerating Jacobian-vector product computation within the PCG algorithm.

## Method

### Overall Architecture
The pipeline adopts a two-stage optimization strategy. In the first stage, the standard Adam optimizer runs for approximately 20K iterations to complete Gaussian densification and initial optimization. In the second stage, the optimizer is switched to LM, which requires only 5 iterations to achieve convergence. This two-stage design exploits the complementary strengths of gradient descent for rapid early-stage progress and LM for efficient fine-grained optimization.

### Key Designs
1. **Adapting the LM Optimizer to 3DGS**:

    - **Function**: Reformulates the 3DGS rendering loss as a sum-of-squares objective compatible with the LM algorithm.
    - **Mechanism**: The L1 and SSIM losses are rewritten in residual form by taking square roots: $r_i^{abs} = \sqrt{\lambda_1|c_i - C_i|}$ and $r_i^{SSIM} = \sqrt{\lambda_2(1 - \text{SSIM}(c_i, C_i))}$. The objective value is preserved while satisfying LM's sum-of-squares requirement. Each iteration solves the normal equations $(\mathbf{J}^T\mathbf{J} + \lambda_{reg}\text{diag}(\mathbf{J}^T\mathbf{J}))\Delta = -\mathbf{J}^T\mathbf{F}(\mathbf{x})$ to obtain the update direction.
    - **Design Motivation**: LM's approximate second-order updates yield substantially higher per-iteration quality than Adam, requiring only 5 iterations compared to 10K.

2. **Cache-Driven Parallelization Scheme**:

    - **Function**: Accelerates the repeated Jacobian-vector product computations required by the PCG algorithm.
    - **Mechanism**: Gradient computation is decomposed into three independent stages: $\frac{\partial r}{\partial x_i} = \frac{\partial r}{\partial c} \frac{\partial c}{\partial s} \frac{\partial s}{\partial x_i}$. The key observation is that intermediate quantities $T_s$ and $\frac{\partial c}{\partial \alpha_s}$ are recomputed up to 18 times during PCG iterations. The proposed scheme caches these intermediate gradients once (buildCache), then shifts the parallelization granularity from per-pixel to per-pixel-per-splat, with each thread handling a single splat along a given ray.
    - **Design Motivation**: The original per-pixel parallelism in 3DGS requires each thread to traverse all splats along a ray, resulting in redundant recomputation of intermediate quantities. Caching decouples splats and enables finer-grained parallelism.

3. **Image Subsampling Scheme**:

    - **Function**: Controls cache size to fit within GPU memory constraints.
    - **Mechanism**: Images are partitioned into $n_b$ batches; the normal equations are solved independently per batch, and the resulting update directions are merged via weighted averaging: $\Delta = \sum_{i=1}^{n_b} \frac{\mathbf{M}_i \Delta_i}{\sum_k \mathbf{M}_k}$, where the weight $\mathbf{M}_i = \text{diag}(\mathbf{J}_i^T\mathbf{J}_i)$ reflects the contribution of each Gaussian parameter to rendering.
    - **Design Motivation**: Cache requirements for high-resolution, dense-capture scenes may exceed GPU memory. Weighted averaging is more principled than a simple mean.

### Loss & Training
- Identical L1 + SSIM loss as the original 3DGS.
- Stage 1: Adam optimizer for 20K iterations, with densification performed during the first 15K iterations.
- Stage 2: LM optimizer for 5 iterations, each with 8 inner PCG rounds.
- After each LM iteration, line search is performed on a 30% image subset to determine the optimal step size $\gamma$, and the regularization strength $\lambda_{reg}$ is adaptively adjusted based on the update quality $\rho$.

## Key Experimental Results

### Main Results

| Dataset | Baseline | PSNR | Time (s) | +Ours PSNR | +Ours Time (s) | Speedup |
|--------|----------|------|---------|------------|-------------|--------|
| MipNeRF360 | 3DGS | 27.40 | 1271 | 27.39 | 972 | 23.5% |
| MipNeRF360 | gsplat | 27.42 | 1064 | 27.42 | 818 | 23.1% |
| Deep Blending | 3DGS | 29.51 | 1222 | 29.72 | 951 | 22.2% |
| Deep Blending | Taming-3DGS | 29.84 | 447 | 29.91 | 347 | 22.4% |
| Tanks&Temples | gsplat | 23.50 | 646 | 23.68 | 414 | 35.9% |

### Ablation Study

| Configuration | PSNR | Time (s) | Notes |
|------|------|---------|------|
| 3DGS + Ours (L1/SSIM) | 27.29 | 1175 | Full model |
| 3DGS + Ours (L2 only) | 27.48 | 1131 | L2 loss only |
| 3DGS Adam (L1/SSIM) | 27.23 | 1573 | Baseline |
| Batch=100 | 33.77 | 242 | GPU 32.5 GB |
| Batch=60 | 33.69 | 223 | GPU 22.6 GB |
| Batch=40 | 33.51 | 212 | GPU 15.4 GB |

### Key Findings
- LM requires only 5 iterations to accomplish the work of 10K Adam iterations, with substantially higher per-update quality.
- Even when Adam is given the same multi-view batch size (75 images), 130 iterations are still needed to match the quality of 5 LM iterations.
- Image subsampling has a negligible effect on LM convergence; as few as 40 images yield results close to the full-batch setting.
- The L1/SSIM loss outperforms pure L2 loss under both LM and Adam.

## Highlights & Insights
- The successful application of classical second-order optimization to large-scale 3DGS optimization is enabled by the cache-driven parallelization design.
- The method is orthogonal to and composable with other 3DGS acceleration techniques — it can be directly substituted as the optimizer in any 3DGS variant.
- The two-stage strategy judiciously exploits the complementary strengths of first- and second-order optimizers: Adam excels at rapid convergence from poor initializations, while LM is superior for fine-grained refinement.
- The negligible impact of image subsampling on LM convergence quality suggests that 3DGS optimization exhibits favorable locality properties.
- Only 5 LM iterations versus 10K Adam iterations are needed to achieve equivalent quality, representing a more than 1000× improvement in per-update efficiency.

## Limitations & Future Work
- High memory overhead: the cache requires approximately 53 GB of GPU memory, compared to only 6–11 GB for the baseline, limiting applicability on consumer-grade GPUs.
- The LM optimizer is currently not supported during the densification stage; Adam must first complete densification before LM can be applied.
- SSIM gradient computation employs a simplified approximation that ignores contributions from neighboring pixels, which may affect convergence accuracy.
- The 20% speedup, while consistent, is modest; the benefit is limited for scenarios requiring short optimization times.
- Hyperparameters such as batch size and number of batches must be set manually, with different datasets requiring different configurations.

## Related Work & Insights
- **vs. Original 3DGS (Adam)**: Equivalent quality with a 20% speedup, demonstrating the advantage of second-order optimization in the fine-tuning stage.
- **vs. Taming-3DGS**: Taming-3DGS accelerates training by reducing the number of Gaussians, while 3DGS-LM does so by improving the optimizer; the two approaches are orthogonal and composable (combined: 447s → 347s).
- **vs. GN/LM Methods in NeRF**: LM has been widely adopted in RGB-D fusion, but the explicit Gaussian representation in 3DGS provides a unique opportunity for efficient Jacobian-vector product computation.
- **vs. DISTWAR**: DISTWAR accelerates the backward pass of gradient descent via warp reduction, whereas this work replaces the optimizer entirely; the two strategies are complementary.
- **Insight**: The idea of caching intermediate gradients to amortize repeated computation is generalizable to other optimization problems that require multiple Jacobian evaluations.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First successful application of the LM optimizer to 3DGS, with an elegantly designed cache-driven parallelization scheme.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 3 datasets across 13 scenes with 4 baselines, complemented by detailed runtime analysis and ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear exposition, complete algorithmic pseudocode, and intuitive illustrations of the parallelization strategy.
- **Value**: ⭐⭐⭐⭐ The lossless 20% speedup has practical significance, and orthogonality with other acceleration methods enhances real-world utility.

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
