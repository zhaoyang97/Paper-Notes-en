---
title: >-
  [Paper Note] 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper replaces the ADAM optimizer in 3D Gaussian Splatting with a custom Levenberg-Marquardt (LM) second-order optimizer. By leveraging an efficient CUDA-parallelized PC…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Levenberg-Marquardt"
  - "optimization acceleration"
  - "CUDA parallelism"
  - "second-order optimization"
date: 2026-05-08
content_hash: c904a420547afa03
---

# 3DGS-LM: Faster Gaussian-Splatting Optimization with Levenberg-Marquardt

**Conference**: ICCV 2025
**arXiv**: [2409.12892](https://arxiv.org/abs/2409.12892)
**Code**: [GitHub](https://github.com/lukasHoel/3DGS-LM)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Levenberg-Marquardt, optimization acceleration, CUDA parallelism, second-order optimization

## TL;DR
This paper replaces the ADAM optimizer in 3D Gaussian Splatting with a custom Levenberg-Marquardt (LM) second-order optimizer. By leveraging an efficient CUDA-parallelized PCG algorithm and a gradient cache structure to accelerate Jacobian-vector products, the method reduces optimization time by approximately 20% while maintaining equivalent reconstruction quality.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has become the dominant method for novel view synthesis, achieving real-time rendering and high-quality image synthesis through differentiable rasterization of 3D Gaussian primitives. Current optimization typically employs the ADAM optimizer, requiring 30K iterations and up to one hour of training time.

**Limitations of Prior Work**: Existing acceleration approaches follow two main directions—accelerating rasterizer implementations (e.g., DISTWAR's warp reduction, gsplat's parallelism patterns) or reducing the number of Gaussians (e.g., GS-MCMC, Taming-3DGS's densification strategies)—yet none address the underlying optimizer itself, still relying on ADAM's gradual first-order convergence.

**Key Challenge**: As a first-order method, ADAM utilizes only gradient direction information per step and requires thousands of iterations to reach a local optimum. Second-order methods such as LM approximate second-order updates by solving the normal equations and theoretically converge in far fewer iterations. However, in the 3DGS setting, the Jacobian matrix spanning millions of Gaussian parameters and high-resolution images is far too large to store explicitly.

**Goal**: To efficiently apply the LM optimizer to 3DGS and realize scalable Jacobian-vector product computation on the GPU.

**Key Insight**: The method exploits the sparsity of 3DGS Gaussian primitives—each pixel is influenced by only a small number of Gaussians, making the Jacobian extremely sparse—and designs a cache-friendly per-pixel-per-splat parallel strategy that caches intermediate gradients once and reuses them across PCG iterations.

**Core Idea**: By combining gradient caching with per-pixel-per-splat CUDA parallelization to enable matrix-free PCG solving, the method grafts LM onto the second phase of 3DGS optimization, replacing 10K ADAM iterations with only 5 LM iterations.

## Method

### Overall Architecture
The method proceeds in two phases. **Phase 1** runs the original 3DGS ADAM optimizer for the first 20K iterations, completing densification and producing an unconverged Gaussian initialization. **Phase 2** switches to the custom LM optimizer, which converges to quality on par with 30K ADAM iterations in only 5–10 LM iterations. The input consists of posed images and an SfM point cloud; the output is an optimized 3D Gaussian scene.

### Key Designs

1. **Adapting LM to 3DGS**

   - **Function**: Reformulates the 3DGS rendering loss as a sum-of-squares energy function compatible with the LM framework.
   - **Mechanism**: The L1 and SSIM loss terms are each converted to residuals by taking square roots: $r_i^{\text{abs}} = \sqrt{\lambda_1|c_i - C_i|}$ and $r_i^{\text{SSIM}} = \sqrt{\lambda_2(1-\text{SSIM}(c_i, C_i))}$, such that the objective $E(\mathbf{x}) = \sum r_i^2$ takes the standard least-squares form. Each step solves the normal equations $(\mathbf{J}^T\mathbf{J} + \lambda_{\text{reg}}\text{diag}(\mathbf{J}^T\mathbf{J}))\Delta = -\mathbf{J}^T\mathbf{F}$ for the update direction, followed by a line search to find the optimal step size $\gamma$.
   - **Design Motivation**: Preserves the original L1+SSIM objective (experimentally verified to yield better quality than pure L2) while enabling LM to exploit curvature information for higher-quality update steps.

2. **Gradient Caching and Per-Pixel-Per-Splat Parallelization**

   - **Function**: Accelerates the Jacobian-vector products $\mathbf{J}\mathbf{p}$ and $\mathbf{J}^T\mathbf{u}$ repeatedly needed in PCG by caching intermediate gradients.
   - **Mechanism**: In the original 3DGS per-pixel parallelization, each thread processes all splats along a ray, causing the intermediate $\alpha$-blending states ($T_s$, $\partial c/\partial \alpha_s$, $\partial c/\partial c_s$) to be recomputed up to 18 times during PCG. This work instead introduces a buildCache stage that computes and stores all intermediate gradients $\partial c/\partial s$ once; subsequent PCG steps adopt per-pixel-per-splat parallelization (one thread per splat per ray) and directly read from the cache. The cache is first stored in pixel order, then reordered via sortCacheByGaussians to enable coalesced memory access.
   - **Design Motivation**: Eliminates redundant computation in PCG iterations, decomposes the computational granularity from per-pixel to per-pixel-per-splat, and substantially improves GPU occupancy and parallelism.

3. **Image Subsampling Scheme**

   - **Function**: Controls cache memory usage to make the method scalable to high-resolution, densely captured scenes.
   - **Mechanism**: Images are divided into $n_b$ batches; the normal equations are solved independently per batch to obtain update vectors $\Delta_i$, which are then merged via a weighted average: $\Delta = \sum_i \frac{\mathbf{M}_i \Delta_i}{\sum_k \mathbf{M}_k}$, where the weights $\mathbf{M}_i = \text{diag}(\mathbf{J}_i^T\mathbf{J}_i)$ are the diagonal entries of the Jacobi preconditioner for each batch. In practice, 25–70 images per batch and at most 4 batches are used.
   - **Design Motivation**: In high-resolution scenes, caching all images simultaneously would exceed GPU memory (~53 GB for the full scheme). Batched processing reduces memory consumption to a manageable level, while the weighted merging ensures consistency of the update direction across batches.

### Loss & Training
- The objective function is identical to the original 3DGS (L1 + SSIM); only the optimizer is changed.
- The LM regularization strength $\lambda_{\text{reg}}$ is adaptively adjusted based on the update quality metric $\rho$: when $\rho > 10^{-5}$, regularization is decreased ($\lambda_{\text{reg}} \mathrel{*}= 1-(2\rho-1)^3$); otherwise the update is rejected and $\lambda_{\text{reg}}$ is doubled.
- Phase 1 runs for 20K iterations (including densification); Phase 2 runs for only 5 LM iterations (each with 8 PCG inner iterations).
- Line search is performed on 30% of the image subset to reduce rendering overhead.

## Key Experimental Results

### Main Results

| Method | Dataset | SSIM↑ | PSNR↑ | Time (s) | Speedup |
|--------|---------|-------|-------|----------|---------|
| 3DGS | MipNeRF360 | 0.813 | 27.40 | 1271 | — |
| 3DGS + Ours | MipNeRF360 | 0.813 | 27.39 | 972 | 23.5% |
| gsplat | MipNeRF360 | 0.814 | 27.42 | 1064 | — |
| gsplat + Ours | MipNeRF360 | 0.814 | 27.42 | 818 | 23.1% |
| DISTWAR | Deep Blending | 0.899 | 29.47 | 841 | — |
| DISTWAR + Ours | Deep Blending | 0.902 | 29.60 | 672 | 20.1% |
| Taming-3DGS | Tanks&Temples | 0.833 | 23.76 | 366 | — |
| Taming-3DGS + Ours | Tanks&Temples | 0.832 | 23.72 | 310 | 15.3% |

### Ablation Study

| Configuration | PSNR↑ | Time (s) | Note |
|---------------|-------|----------|------|
| ADAM L1/SSIM (30K) | 27.23 | 1573 | Original 3DGS baseline |
| LM L1/SSIM | 27.29 | 1175 | Proposed method: 25% faster, marginally better quality |
| ADAM L2 only | 27.31 | 1528 | Lower quality but higher PSNR |
| LM L2 only | 27.48 | 1131 | LM+L2 also accelerates |
| Batch=100 images | 33.77 | 242 | Best quality with all images |
| Batch=60 images | 33.69 | 223 | Slight quality drop, faster |
| Batch=40 images | 33.51 | 212 | 15 GB memory, acceptable quality |
| Multi-view ADAM (50 iters) | 29.54 | 962 | LM outperforms at equal iteration count |
| LM (5 iters) | 29.72 | 951 | Better quality with only 5 iterations |

### Key Findings
- Replacing ADAM with LM yields approximately 20% speedup across all baselines, demonstrating that optimizer-level improvements are orthogonal to rasterizer and densification improvements.
- Only 5–10 LM iterations suffice to replace 10K ADAM iterations, with each LM step providing substantially higher update quality than first-order steps.
- Image subsampling has minimal impact on quality (dropping from 100 to 40 images reduces PSNR by only 0.26), while significantly reducing GPU memory consumption.
- Initialization quality is critical for LM: applying LM directly from SfM initialization is actually slower; ADAM must first produce a good initialization for the second-order method to be effective.

## Highlights & Insights
- The **cache-driven parallelization scheme** is the core technical contribution: by caching intermediate $\alpha$-blending gradients once, up to 18 redundant computations per PCG iteration are eliminated and a finer per-pixel-per-splat parallelism is enabled. This approach generalizes to any optimization problem that repeatedly requires Jacobians with respect to differentiable rendering.
- The **two-phase strategy** precisely leverages the strengths of both first- and second-order optimizers: ADAM excels at making rapid progress from poor initializations while completing densification; LM excels at fast convergence near a good initialization. This coarse-to-fine optimization paradigm has broad applicability.
- The **weighted batch merging** (Equation 8) cleverly uses $\text{diag}(\mathbf{J}^T\mathbf{J})$ as weights, effectively assigning larger update weights to Gaussian parameters that contribute more to the current batch of images, ensuring physical consistency of the update direction across batches.

## Limitations & Future Work
- **High GPU memory consumption**: The full scheme requires approximately 53 GB of GPU memory (vs. 6–11 GB for the baseline), limiting applicability on consumer-grade GPUs.
- **Dependence on initialization**: 20K ADAM steps must be run first; training with LM from scratch is actually slower, and the choice of phase-switching point is heuristic.
- **Densification phase not optimized**: LM currently optimizes only the parameters of a fixed number of Gaussians; how to incorporate second-order information during densification remains an open problem.
- Potential directions include exploring lower-memory implicit Jacobian computation (e.g., randomized Hutchinson trace estimation) and designing adaptive phase-switching strategies.

## Related Work & Insights
- **vs. ADAM-based 3DGS**: This work demonstrates that first-order methods are not the only option for 3DGS optimization and that second-order methods converge faster given a good initialization.
- **vs. Taming-3DGS**: Taming-3DGS accelerates by reducing the number of Gaussians; this work accelerates by replacing the optimizer. The two approaches are orthogonal and complementary (combined use achieves 310 s on T&T vs. 736 s for the original 3DGS).
- **vs. GN/LM in RGB-D fusion**: RGB-D reconstruction has long employed Gauss-Newton; this work is the first to successfully introduce it into 3DGS and address the associated scalability challenges.

## Rating
- Novelty: ⭐⭐⭐⭐ — Applying LM to 3DGS is not an entirely novel concept, but the cache-driven GPU parallelization constitutes a significant engineering contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three standard datasets, four baselines, and detailed ablations covering objective function, batch size, initialization, and multi-view ADAM comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Method description is clear; algorithmic pseudocode and diagrams aid comprehension.
- Value: ⭐⭐⭐⭐ — A practical 20% speedup that is orthogonal to and composable with existing methods, though high memory consumption limits real-world deployment.

---

This paper proposes 3DGS-LM, which replaces the ADAM optimizer with a custom Levenberg-Marquardt (LM) optimizer to accelerate 3DGS reconstruction. Through a gradient cache data structure and custom CUDA kernels, it achieves efficient GPU-parallel PCG solving, improving optimization speed by approximately 20% while preserving reconstruction quality. The method is orthogonal to and composable with other 3DGS acceleration approaches.

## Background & Motivation

- 3D Gaussian Splatting (3DGS) projects 3D Gaussians onto 2D splats via a differentiable rasterizer and composites pixel colors through $\alpha$-blending, enabling real-time rendering and high-quality novel view synthesis.
- Existing 3DGS acceleration methods focus on two directions: (1) improving densification strategies to reduce the number of Gaussians (GS-MCMC, Taming-3DGS, Mini-Splatting); (2) improving differentiable rasterizer implementations (DISTWAR's warp reduction, gsplat's splat-parallel scheme).
- However, **all methods still use the ADAM optimizer**, requiring thousands of gradient descent iterations to converge—training on high-resolution, densely captured scenes can take up to one hour.
- In contrast, 3D reconstruction pipelines in the RGB-D fusion domain have long used Gauss-Newton/LM algorithms, converging in an order of magnitude fewer iterations than SGD—providing direct motivation for accelerating 3DGS with a second-order optimizer.

## Core Problem

How can the Levenberg-Marquardt (second-order) optimizer be efficiently adapted to the 3DGS optimization pipeline? Key challenges include: (1) the Jacobian matrix is enormous (millions of Gaussians × hundreds of high-resolution images) and cannot be stored explicitly; (2) Jacobian-vector products must be computed efficiently in parallel on the GPU; (3) LM is more sensitive to initialization and performs poorly when initialized directly from SfM point clouds.

## Method

### Overall Architecture

A **two-phase optimization** strategy is adopted:
1. **Phase 1 (ADAM)**: Runs standard 3DGS optimization for 20K iterations (including densification up to 15K iterations), producing a Gaussian initialization that is unconverged but has a fixed number of primitives.
2. **Phase 2 (LM)**: Only 5 LM iterations (each with 8 PCG inner iterations) are needed to converge.

The rationale for the two-phase design is that gradient descent makes rapid progress initially but slows near convergence, whereas LM requires a good initialization to be effective (applying LM directly from SfM is slower). This design also avoids handling densification within LM's 5–10 iterations (3DGS requires 140 densification operations).

### Key Designs

**1. LM Objective Function Reformulation**

The 3DGS L1+SSIM loss is rewritten as a sum-of-squared residuals for compatibility with LM:

$$E(\mathbf{x}) = \sum_{i=1}^{N} \lambda_1 |c_i - C_i|^2 + \lambda_2 (1 - \text{SSIM}(c_i, C_i))^2$$

Two classes of residuals, $r_i^{\text{abs}}$ and $r_i^{\text{SSIM}}$, are obtained by taking square roots of the original loss terms, making the objective fully equivalent to the original 3DGS formulation while satisfying LM requirements.

**2. Normal Equations and PCG Solving**

Each LM iteration solves the normal equations for the update direction $\Delta$:

$$(J^T J + \lambda_{\text{reg}} \cdot \text{diag}(J^T J)) \Delta = -J^T F(\mathbf{x})$$

Since the Jacobian $J$ cannot be stored explicitly, a **preconditioned conjugate gradient (PCG)** method is used in a matrix-free fashion, with at most 8 PCG iterations per LM step.

**3. Gradient Caching and Per-Pixel-Per-Splat Parallelization**

This is the central technical contribution of the paper:

- **Problem**: The original 3DGS per-pixel parallelization is inefficient in PCG, because computing $\mathbf{u}=J\mathbf{p}$ and $\mathbf{g}=J^T\mathbf{u}$ per PCG iteration requires the same intermediate $\alpha$-blending states ($T_s$, $\partial c/\partial \alpha_s$, $\partial c/\partial c_s$) to be recomputed up to 18 times.
- **Solution**: Prior to PCG, a buildCache pass computes and stores these intermediate gradients once. Computation then shifts to **per-pixel-per-splat parallelization**—each thread handles all residuals for one splat along one ray.
- The chain rule decomposes the Jacobian into three independently parallelizable factors: $\frac{\partial r}{\partial x_i} = \frac{\partial r}{\partial c} \cdot \frac{\partial c}{\partial s} \cdot \frac{\partial s}{\partial x_i}$.
- After sorting the cache by Gaussian index, **coalesced memory access** is achieved; the applyJT kernel uses **segmented warp reduction** for efficient aggregation.

**4. Image Subsampling Scheme**

In high-resolution, densely captured scenes the cache may be too large, so the following approach is adopted:
- Images are divided into $n_b$ batches (25–70 images each); the normal equations are solved independently per batch.
- Update vectors are merged via a weighted mean: $\Delta = \frac{\sum_i M_i \Delta_i}{\sum_k M_k}$, where the weights $M_i = \text{diag}(J_i^T J_i)$ are the PCG preconditioner inverses.
- The weighting derivation is based on decomposing the full normal-equation solution into a weighted combination of per-subset solutions under a diagonal approximation.

**5. Line Search and $\lambda$ Adjustment**

Once $\Delta$ is obtained, a line search over 30% of the image subset finds the optimal step size $\gamma$. The regularization strength is then adjusted based on the quality metric $\rho$: when $\rho > 10^{-5}$, the update is accepted and $\lambda$ is decreased; otherwise the update is rejected and $\lambda$ is doubled.

### Loss & Training

- **Loss**: Identical to the original 3DGS L1+SSIM ($\lambda_1=0.8$, $\lambda_2=0.2$), adapted to LM's sum-of-squared-residuals form via a square-root transformation.
- **SSIM gradient handling**: SSIM's local neighborhood convolution gradient is back-propagated only to the center pixel (ignoring contributions to neighboring pixels) to preserve ray independence and support parallelization.
- **Implementation details**: Phase 1 uses default hyperparameters for 20K iterations; Phase 2 runs 5 LM iterations with 8 PCG inner iterations each. Batch size is adjusted per dataset resolution (MipNeRF360: 25 images × 4 batches; Deep Blending: 25 × 3; Tanks&Temples: 70 × 3).

## Key Experimental Results

| Dataset | Metric | 3DGS | 3DGS+Ours | Speedup (s) |
|---------|--------|------|-----------|-------------|
| MipNeRF360 | SSIM/PSNR/Time | 0.813/27.40/1271s | 0.813/27.39/972s | **23.5%** |
| Deep Blending | SSIM/PSNR/Time | 0.900/29.51/1222s | 0.903/29.72/951s | **22.2%** |
| Tanks&Temples | SSIM/PSNR/Time | 0.844/23.68/736s | 0.845/23.73/663s | **9.9%** |

**Composability with other acceleration methods** (Deep Blending):

| Method | Orig. Time | +Ours Time | Speedup |
|--------|------------|------------|---------|
| DISTWAR | 841s | 672s | **20.1%** |
| gsplat | 919s | 716s | **22.1%** |
| Taming-3DGS | 447s | 347s | **22.4%** |

Adding the LM optimizer to all baselines yields approximately 20% speedup with quality metrics essentially unchanged. Average GPU memory consumption is 53 GB (vs. 6–11 GB for baselines), reflecting the time–memory trade-off.

### Ablation Study Highlights

1. **L1/SSIM vs. L2 objective**: Using an L2 loss degrades quality for both 3DGS and 3DGS+Ours (SSIM 0.862→0.854), confirming the importance of the L1/SSIM loss; LM achieves speedup under both objectives.
2. **Batch size**: Reducing from 100 to 40 images causes only a minor quality drop (PSNR 33.77→33.51) while reducing runtime from 242s to 212s and GPU memory from 32.5 GB to 15.4 GB—image subsampling does not impair LM convergence.
3. **LM vs. multi-view ADAM**: Under the same multi-view constraint (75 images/step), LM reaches with just 5 iterations the quality that ADAM requires 10K iterations to achieve. Even increasing the ADAM batch size to 50 or 130 images requires more iterations to reach comparable quality, confirming that LM's second-order update direction is substantially higher quality than a first-order gradient.
4. **Number of initialization iterations**: LM converges faster when initialized with $K=6000$ or $K=8000$ ADAM iterations; with too few iterations LM is actually slower, validating the necessity of the two-phase strategy.

## Highlights & Insights

- **Orthogonal design**: Targeting acceleration at the optimizer level is entirely orthogonal to modifications of the rasterizer or densification strategy, and can be directly combined with them for compounding speedups.
- **Cache-driven parallelization**: By caching intermediate gradients once and resorting, the per-pixel parallelism is transformed into per-pixel-per-splat parallelism; the applyJT kernel runs 4.8× faster than buildCache.
- **Engineering efficiency**: The entire Phase 2 converges in just 5 LM × 8 PCG = 40 kernel calls, whereas ADAM requires an additional 10K iterations.
- **Clever SSIM gradient treatment**: Back-propagating only to the center pixel preserves ray independence, enabling seamless integration of SSIM into the parallel PCG framework.
- **Precomputed residual weights**: Extracting $(\partial r/\partial c)^2$ outside the kernel avoids additional global memory reads, making the computation overhead of L1/SSIM residuals nearly identical to that of L2.

## Limitations & Future Work

1. **High GPU memory consumption**: The gradient cache increases memory from 6–11 GB to 53 GB; large-scale, high-resolution scenes may require CPU offloading.
2. **Continued reliance on ADAM for densification**: The two-phase design means Phase 1 still requires considerable time; further acceleration would be possible if densification could be integrated into LM (difficult given only 5–10 iterations vs. 140 densification operations).
3. **Variable speedup across datasets**: Speedup on Tanks&Temples is only ~10%, far below the 23.5% on MipNeRF360—possibly because lower-resolution images make LM's fixed overhead relatively larger.
4. **Cache sorting overhead**: Although sortCacheByGaussians is faster than PCG itself, it constitutes a fixed cost per LM iteration.

## Related Work & Insights

| Method | Acceleration Strategy | Relationship to 3DGS-LM |
|--------|----------------------|------------------------|
| DISTWAR | Warp reduction to accelerate backward pass | Orthogonal, composable (DISTWAR+Ours: 764s vs. DISTWAR: 966s) |
| gsplat | Splat-parallel rasterizer improvement | Orthogonal, composable |
| Taming-3DGS | Improved densification to reduce Gaussian count | Orthogonal, composable (fastest baseline+Ours: 347s) |
| GS-MCMC | MCMC-based densification replacing heuristics | Complementary: modifies densification strategy rather than optimizer |
| Rasmuson et al. | Gauss-Newton for low-resolution NeRF voxel optimization | Inspiration; 3DGS-LM achieves more efficient parallel JVP via explicit Gaussian primitives |
| 3DGS² (Lan et al. 2025) | Near-second-order convergence for 3DGS | Same direction (second-order optimization for 3DGS); warrants attention |

**Broader implications**:
- The optimizer-replacement approach is not limited to 3DGS; any reconstruction pipeline based on differentiable rendering and ADAM (e.g., NeRF variants, differentiable mesh optimization) may benefit from a similar LM substitution.
- The cache-then-reorder GPU optimization pattern generalizes to other scenarios requiring repeated computation of the same intermediate quantities.
- The image subsampling scheme—decomposing large-scale least squares into independent per-batch solves followed by weighted merging—has affinities with distributed optimization and federated learning.
- The coarse-to-fine training philosophy—first-order methods for rapid initialization followed by second-order methods for fine convergence—is effective across a range of optimization problems.

## Rating

- Novelty: ⭐⭐⭐⭐ — Accelerating 3DGS from the optimizer perspective offers a concise and effective new angle, though LM itself is not a novel algorithm; the core contribution lies in the engineering implementation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive experiments across 3 datasets, 13 scenes, and 4 baselines; ablations cover objective function, batch size, initialization, and multi-view comparisons; per-scene results and profiling analyses are thorough.
- Writing Quality: ⭐⭐⭐⭐⭐ — Logical progression from motivation to method to experiments; the two-phase design motivation via Fig. 4 is well presented; algorithmic pseudocode and cache diagrams are intuitive.
- Value: ⭐⭐⭐⭐ — High practical value due to 20% speedup with no quality loss and orthogonal composability with existing methods; the 53 GB memory requirement limits real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RegGS: Unposed Sparse Views Gaussian Splatting with 3DGS Registration](reggs_unposed_sparse_views_gaussian_splatting_with_3dgs_registration.md)
- [\[ICCV 2025\] Faster and Better 3D Splatting via Group Training](faster_and_better_3d_splatting_via_group_training.md)
- [\[ICCV 2025\] RobustSplat: Decoupling Densification and Dynamics for Transient-Free 3DGS](robustsplat_decoupling_densification_and_dynamics_for_transient-free_3dgs.md)
- [\[ICCV 2025\] Gaussian Splatting with Discretized SDF for Relightable Assets](gaussian_splatting_with_discretized_sdf_for_relightable_assets.md)
- [\[ICCV 2025\] MuGS: Multi-Baseline Generalizable Gaussian Splatting Reconstruction](mugs_multi-baseline_generalizable_gaussian_splatting_reconstruction.md)

</div>

<!-- RELATED:END -->
