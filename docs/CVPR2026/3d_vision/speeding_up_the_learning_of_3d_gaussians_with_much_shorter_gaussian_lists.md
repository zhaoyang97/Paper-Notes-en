---
title: >-
  [Paper Note] Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] By periodically resetting Gaussian scales (Scale Reset) and imposing an entropy constraint on alpha blending weights…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Training Acceleration"
  - "Scale Reset"
  - "Entropy Constraint"
  - "Gaussian List Shortening"
date: 2026-05-08
content_hash: a3bc9d3924eab415
---

# Speeding Up the Learning of 3D Gaussians with Much Shorter Gaussian Lists

**Conference**: CVPR 2026
**arXiv**: [2603.09277](https://arxiv.org/abs/2603.09277)
**Code**: [MachinePerceptionLab/ShorterSplatting](https://github.com/MachinePerceptionLab/ShorterSplatting)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Training Acceleration, Scale Reset, Entropy Constraint, Gaussian List Shortening

## TL;DR

By periodically resetting Gaussian scales (Scale Reset) and imposing an entropy constraint on alpha blending weights, this paper shortens the per-pixel Gaussian list length to achieve **5–12× training acceleration** in 3DGS while maintaining comparable rendering quality.

## Background & Motivation

3D Gaussian Splatting (3DGS) offers significant advantages over NeRF in rendering efficiency and quality, yet its training process remains slow, limiting time-sensitive applications. Existing acceleration methods primarily approach the problem from the following angles:

- **Reducing training iterations**: Lowering iteration counts from 30K to 5K–8K through better initialization or densification strategies
- **Optimizer and CUDA implementation**: Accelerating convergence with second-order optimizers and optimizing CUDA kernels
- **Training strategies**: Freezing converged Gaussians, pruning redundant Gaussians, and progressive resolution training

However, these methods either rely on reducing the total number of Gaussians (unsuitable for large-scale complex scenes) or provide only marginal speedups (e.g., more accurate coverage estimation yields only ~10% acceleration). This paper proposes a novel perspective: **shortening the per-pixel Gaussian list length**, thereby reducing the number of Gaussians involved in each splatting operation and directly lowering the cost of forward rendering and backward gradient computation.

## Method

### Overall Architecture

The core idea of this paper is to encourage each Gaussian to concentrate its influence on a local image region rather than spreading it across a large number of pixels. The method comprises three modules:

1. **Scale Reset**: Periodically shrinks the scale of all Gaussians to reduce the pixel coverage of individual Gaussians
2. **Entropy Constraint**: Applies entropy regularization to alpha blending weights to sharpen the weight distribution
3. **Resolution Scheduler**: Progressive resolution scheduling, training from low resolution to full resolution

### Key Design 1: Scale Reset

Larger Gaussians cover more pixels, leading to longer Gaussian lists. Intuitively, smaller scales should be encouraged, but naively adding a volume penalty to the loss function is difficult to tune—an excessively large weight causes Gaussians to become too small, while too small a weight renders it ineffective.

Scale Reset adopts a more direct strategy: at fixed intervals, the scale of all Gaussians is multiplied by a shrinkage factor $\zeta < 1$:

$$s_i \leftarrow \zeta \cdot s_i, \quad \forall i$$

The default setting applies this every 20 epochs with $\zeta = 0.2$ (i.e., shrinking to 20% of the original scale).

**Why Scale Reset outperforms volume regularization:**

- **Immediate effect**: Scale Reset instantly shortens the Gaussian list for all pixels upon execution, so all subsequent iterations benefit immediately; volume regularization requires gradual optimization through gradients and is slow to take effect
- **Adaptive recovery**: After resetting, Gaussians have sufficient iterations to adjust other attributes (color, opacity, position, etc.) to approximate the radiance field, preserving rendering quality
- **Beneficial side effects**: Scale Reset encourages higher opacity, making Gaussians more compact and effective

### Key Design 2: Entropy Constraint

In alpha blending, the weight of the $i$-th Gaussian at pixel $p$ is:

$$w_i = T_i \cdot \alpha_i, \quad T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$$

For a Gaussian list of length $N$, these weights together with the background contribution form a valid probability distribution:

$$\sum_{i=1}^{N+1} w_i = 1, \quad w_{N+1} = T_{N+1}$$

This normalization property is key—it enables entropy computation without additional normalization steps, avoiding the overhead of maintaining global statistics and accessing them during backpropagation.

The entropy at pixel $j$ is defined as:

$$H_j = -\sum_{i=1}^{N+1} w_{i,j} \log w_{i,j}$$

The entropy loss is averaged over all $M$ pixels:

$$\mathcal{L}_E = \frac{1}{M} \sum_{j=1}^{M} H_j$$

**Effect of entropy minimization**: Drives dominant weights larger and minor weights smaller, causing each Gaussian to concentrate on its dominant pixel region and reducing its influence on neighboring pixels, thereby shortening the Gaussian list.

### Gradient Computation

The gradient of the total loss with respect to Gaussian attribute $x_i$ is:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}_{\text{base}}}{\partial x_i} + \frac{\gamma}{M} \sum_{j=1}^{M} \frac{\partial H_j}{\partial \alpha_{i,j}} \frac{\partial \alpha_{i,j}}{\partial x_i}$$

The gradient of entropy with respect to $\alpha_{i,j}$ is computed efficiently by introducing an intermediate variable $R_{i,j}$:

$$R_{i,j} = \sum_{k=i}^{N+1} (\log w_{k,j} + 1) w_{k,j}$$

$$\frac{\partial H_j}{\partial \alpha_{i,j}} = (-\log w_{i,j} - 1) T_{i,j} + \frac{R_{i+1,j}}{1-\alpha_{i,j}}$$

The paper designs an $O(N)$ backward scan algorithm that accumulates $R_j$ from $i=N$ to $1$, avoiding redundant computation.

### Loss & Training

The overall loss combines the base reconstruction loss and entropy regularization:

$$\mathcal{L} = \mathcal{L}_{\text{base}} + \gamma \mathcal{L}_E$$

$$\mathcal{L}_{\text{base}} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$$

where $\gamma = 0.015$ is the entropy loss weight and $\lambda$ is the D-SSIM weight (following 3DGS defaults).

### Resolution Scheduler

A coarse-to-fine resolution strategy from DashGaussian is adopted, progressively training from low resolution (downsampling factor $r > 1$) to full resolution ($r=1$). However, excessively large $r$ can reduce efficiency—at low resolution, each tile covers a larger scene region, causing excessive Gaussian overlap. The empirical threshold caps the number of Gaussians per tile at 150, with $r_{\max} \leq 4$.

When combined with the resolution scheduler, a stage-adaptive strategy is employed: weaker regularization is used in the early low-resolution stage to preserve scene structure, while $\zeta$ and $\gamma$ are increased in the later full-resolution stage.

## Key Experimental Results

### Main Results

Evaluated on three benchmarks—Mip-NeRF 360, Deep Blending, and Tanks & Temples—using an RTX 5090 D GPU.

| Method | Iterations | # Gaussians | PSNR↑ | SSIM↑ | LPIPS↓ | Training Time (s)↓ |
|--------|-----------|-------------|-------|-------|--------|-------------------|
| 3DGS | 30K | 3.3M | 27.55 | 0.819 | 0.209 | 919.51 |
| AdR-Gaussian | 30K | 1.3M | 26.92 | 0.792 | 0.257 | 504.86 |
| Taming-3DGS | 30K | 3.3M | 27.85 | 0.823 | 0.208 | 402.54 |
| EDGS | 5K | 3.5M | 26.46 | 0.817 | 0.205 | 318.06 |
| Mini-Splatting2 | 18K | 3.6M | 27.56 | 0.827 | 0.184 | 220.22 |
| DashGaussian | 30K | 3.3M | 27.84 | 0.824 | 0.203 | 218.85 |
| LiteGS | 30K | 3.3M | 27.75 | 0.822 | 0.208 | 191.17 |
| **Ours** | **30K** | **3.3M** | **27.28** | **0.810** | **0.224** | **99.58** |

*Results on the Mip-NeRF 360 dataset.*

**Speedup**: Achieves **9.2×** (Mip-NeRF 360), **11.9×** (Deep Blending), and **5.3×** (Tanks & Temples) speedup over 3DGS; approximately **50%** faster than the LiteGS baseline.

### Ablation Study

| Module Combination | PSNR↑ | SSIM↑ | LPIPS↓ | Time (s)↓ |
|-------------------|-------|-------|--------|----------|
| L (LiteGS) | 27.75 | 0.822 | 0.208 | 191.17 |
| L+R (Scale Reset) | 27.33 | 0.815 | 0.212 | 147.33 |
| L+E (Entropy) | 27.35 | 0.815 | 0.218 | 162.53 |
| L+R+E | 27.14 | 0.812 | 0.215 | 141.28 |
| L+D (DashGaussian) | 27.85 | 0.822 | 0.213 | 134.99 |
| L+D+R | 27.52 | 0.815 | 0.219 | 108.62 |
| L+D+E | 27.38 | 0.813 | 0.223 | 112.13 |
| L+D+R+E (Full Method) | 27.28 | 0.810 | 0.224 | 99.58 |

### Key Findings

1. **Scale Reset vs. volume regularization**: Scale Reset outperforms volume regularization in both quality and speed (PSNR 27.28 vs. 27.17; training time 99.58s vs. 107.91s)
2. **Entropy vs. opacity constraint**: The entropy constraint outperforms opacity regularization, as entropy acts on blending weights that depend on multiple attributes (opacity, scale, position, rotation), whereas opacity regularization constrains only a single attribute and is overly conservative
3. **Complementary modules**: Scale Reset provides immediate geometric regularization, while the Entropy Constraint continuously adjusts the contribution distribution throughout optimization; joint use yields the best results
4. **Tile size independence**: 3DGS uses 16×16 tiles vs. LiteGS's 8×8 tiles with comparable quality and training time, since smaller tiles yield shorter lists but more tiles overall
5. **Moderate quality trade-off**: PSNR loss on Mip-NeRF 360 is only 0.27 dB (27.28 vs. 27.55), while training time is reduced from 919s to 100s

## Highlights & Insights

- **Novel perspective**: Rather than reducing the total number of Gaussians, acceleration is achieved by shortening the per-pixel Gaussian list length, making the approach applicable to large-scale complex scenes
- **Simple yet effective**: Scale Reset requires only a single line of code (element-wise multiplication); the Entropy Constraint has an efficient $O(N)$ gradient algorithm
- **No data priors required**: The method does not rely on pretrained models or geometric foundation models; it is a purely training-strategy-level improvement
- **Orthogonal to existing methods**: Compatible with and stackable upon CUDA optimizations (LiteGS) and resolution scheduling methods (DashGaussian)

## Limitations & Future Work

- **Modest quality degradation**: PSNR decreases by approximately 0.3–0.5 dB, which may be unacceptable in quality-critical applications
- **Parameter sensitivity**: The selection of $\zeta$ and $\gamma$ requires balancing speed and quality, and different scenes may require different configurations
- **Dependence on LiteGS backbone**: Performance degradation in constrained settings (fewer iterations/Gaussians) stems from limitations of the LiteGS backbone
- **Combination with higher-order optimizers unexplored**: Second-order optimizers such as Levenberg–Marquardt may further complement the proposed method
- **Evaluation limited to static scenes**: Applicability to dynamic scene reconstruction and other tasks has not been verified

## Rating

- Novelty: ⭐⭐⭐⭐ — Accelerating training by targeting Gaussian list length is a novel perspective; Scale Reset is simple but effective
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three standard benchmarks, detailed ablations, comparisons with multiple alternatives, and timing breakdown analysis
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich figures, and complete mathematical derivations
- Value: ⭐⭐⭐⭐ — A 9× speedup is practically significant, and the method is straightforward to integrate into existing 3DGS pipelines

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CrowdGaussian: Reconstructing High-Fidelity 3D Gaussians for Human Crowd from a Single Image](crowdgaussian_reconstructing_high-fidelity_3d_gaussians_for_human_crowd_from_a_s.md)
- [\[ICLR 2026\] Learning Unified Representation of 3D Gaussian Splatting](../../ICLR2026/3d_vision/learning_unified_representation_of_3d_gaussian_splatting.md)
- [\[CVPR 2026\] MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping](magician_efficient_long-term_planning_with_imagined_gaussians_for_active_mapping.md)
- [\[ICLR 2026\] MEGS2: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning](../../ICLR2026/3d_vision/megs2_memory-efficient_gaussian_splatting_via_spherical_gaussians_and_unified_pr.md)
- [\[ICCV 2025\] AAA-Gaussians: Anti-Aliased and Artifact-Free 3D Gaussian Rendering](../../ICCV2025/3d_vision/aaa-gaussians_anti-aliased_and_artifact-free_3d_gaussian_rendering.md)

</div>

<!-- RELATED:END -->
