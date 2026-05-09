---
title: >-
  [Paper Note] Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] Momentum-GS proposes a momentum-based self-distillation mechanism to address cross-block consistency issues in block-parallel training of large-scale 3D Gaussian Splatting. By introducing a momentum teacher Gaussian decoder for global guidance and decoupling the number of blocks from the number of GPUs, the method achieves state-of-the-art performance on multiple large-scale scene datasets, improving LPIPS by 18.7% over CityGaussian.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D Gaussian Splatting
  - Large-Scale Scene Reconstruction
  - Momentum Self-Distillation
  - Hybrid Representation
  - Block-Parallel Training
date: 2026-05-08
content_hash: db4942f5d0e80000
---

# Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction

**Conference**: ICCV 2025
**arXiv**: [2412.04887](https://arxiv.org/abs/2412.04887)
**Code**: [https://jixuan-fan.github.io/Momentum-GS_Page/](https://jixuan-fan.github.io/Momentum-GS_Page/)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, Large-Scale Scene Reconstruction, Momentum Self-Distillation, Hybrid Representation, Block-Parallel Training

## TL;DR

Momentum-GS proposes a momentum-based self-distillation mechanism to address cross-block consistency issues in block-parallel training of large-scale 3D Gaussian Splatting. By introducing a momentum teacher Gaussian decoder for global guidance and decoupling the number of blocks from the number of GPUs, the method achieves state-of-the-art performance on multiple large-scale scene datasets, improving LPIPS by 18.7% over CityGaussian.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has demonstrated strong performance in scene reconstruction. Large-scale scene reconstruction commonly adopts a divide-and-conquer strategy, partitioning the scene into independent blocks for parallel training.

**Limitations of Prior Work**:
   - **Memory and Storage**: The explicit representation of millions of Gaussians incurs substantial memory and storage overhead. Hybrid representations (implicit + explicit), such as Scaffold-GS, can alleviate this but introduce new challenges.
   - **Cross-Block Inconsistency**: Independent per-block training neglects inter-block relationships, resulting in visible illumination discontinuities and unnatural transitions at block boundaries (e.g., abrupt lighting changes in CityGaussian).
   - **GPU Count Constraint**: Hybrid representations employ a shared Gaussian decoder (MLP); during parallel training, the number of blocks is thus limited by the number of available GPUs.

**Key Challenge**: Independent block training → insufficient data diversity → poor decoder quality → degraded reconstruction accuracy; shared decoder with parallel training → block count constrained by GPU count → poor scalability.

**Goal**: How to ensure cross-block consistency in hybrid-representation block-wise training of large-scale scenes without being constrained by the number of available GPUs?

**Key Insight**: Drawing inspiration from momentum teacher models in self-supervised learning, the paper uses a momentum-updated teacher decoder to provide stable, globally consistent guidance across all blocks.

**Core Idea**: A momentum-updated teacher Gaussian decoder performs self-distillation on the student decoder, decoupling the block count from the GPU count constraint, while reconstruction-quality-guided block weighting enables globally consistent optimization.

## Method

### Overall Architecture

Input: Multi-view images of a large-scale scene + SfM sparse point cloud. Output: High-quality 3D Gaussian scene representation.

Pipeline: The large scene is partitioned into $n$ blocks; at each step, $k$ blocks are sampled and assigned to $k$ GPUs for parallel training. All blocks share a single student Gaussian decoder $D_s$, while a momentum-updated teacher decoder $D_t$ provides consistency guidance.

### Key Designs

1. **Scene Momentum Self-Distillation**

    - **Function**: Decouple the block count from the GPU count constraint while enhancing cross-block consistency.
    - **Mechanism**: A teacher Gaussian decoder $D_t$ (parameters $\theta_t$) and a student Gaussian decoder $D_s$ (parameters $\theta_s$) are maintained. Teacher parameters are updated via momentum: $\theta_t \leftarrow m \cdot \theta_t + (1-m) \cdot \theta_s$, with momentum coefficient $m=0.9$. The teacher decoder does not participate in gradient back-propagation; instead, it provides a stable global reference and guides the student decoder via a consistency loss: $\mathcal{L}_{\text{consistency}} = \|D_m(f_b, v_b; \theta_t) - D_o(f_b, v_b; \theta_s)\|_2$
    - **Design Motivation**: A sequential training strategy allows each GPU to cycle through different blocks, breaking the dependency of block count on GPU count. The momentum teacher evolves smoothly, providing consistent global guidance across interleaved blocks and preventing decoder oscillation caused by inter-block data heterogeneity.
    - **Novelty**: Methods such as CityGaussian either train each block independently (precluding decoder merging) or share the decoder with strictly parallel training (block count equals GPU count). This paper achieves decoupling via momentum distillation.

2. **Reconstruction-Guided Block Weighting**

    - **Function**: Dynamically adjust training weights per block, prioritizing blocks with poorer reconstruction quality.
    - **Mechanism**: Momentum-smoothed PSNR and SSIM are used to track per-block reconstruction quality. The deviation of each block from the best-performing block, $\delta_p$ and $\delta_s$, is computed, and a Gaussian-like distribution assigns block weights: $w_i = 2 - \exp\left(\frac{\delta_p^2 + \lambda \cdot \delta_s^2}{-2\sigma^2}\right)$
    - **Design Motivation**: Due to non-uniform scene partitioning, some blocks are inherently more challenging to reconstruct. Adaptive weighting directs the decoder's attention toward weaker blocks, preventing convergence to local optima and improving global consistency.

### Loss & Training

Total loss function: $\mathcal{L} = \mathcal{L}_1 + \lambda_{\text{SSIM}}\mathcal{L}_{\text{SSIM}} + \lambda_{\text{consistency}}\mathcal{L}_{\text{consistency}}$

At each forward pass, each block randomly selects one viewpoint; the shared decoder predicts Gaussian parameters and renders an image, which is compared against the ground truth to compute the reconstruction loss. Gradients from all blocks are accumulated into the shared decoder.

## Key Experimental Results

### Main Results (5 Large-Scale Scenes)

| Method | Building PSNR↑ | Rubble PSNR↑ | Residence PSNR↑ | Sci-Art PSNR↑ | Rubble LPIPS↓ |
|------|----------------|--------------|-----------------|---------------|---------------|
| 3D-GS | 22.53 | 25.51 | 22.36 | 24.13 | 0.316 |
| VastGaussian | 21.80 | 25.20 | 21.01 | 22.64 | 0.264 |
| CityGaussian | 22.70 | 26.45 | 23.35 | 24.49 | 0.232 |
| DOGS | 22.73 | 25.78 | 21.94 | 24.42 | 0.257 |
| **Momentum-GS** | **23.65** | **26.66** | **23.37** | **25.06** | **0.200** |

MatrixCity results: Momentum-GS achieves PSNR 29.11 / SSIM 0.881 / LPIPS 0.180, surpassing all baselines across all metrics.

Rendering efficiency: **59.91 FPS** with only **4.62 GB** memory (best among all methods), significantly outperforming CityGaussian (26.10 FPS, 14.68 GB).

### Ablation Study

| Training Strategy | #Block | PSNR↑ | SSIM↑ | LPIPS↓ |
|----------|--------|-------|-------|--------|
| (a) Baseline single block | 1 | 22.25 | 0.742 | 0.272 |
| (b) Parallel training | 4 | 23.10 | 0.790 | 0.221 |
| (c) Independent training | 4 | 22.85 | 0.781 | 0.229 |
| (d) Independent training | 8 | 23.23 | 0.796 | 0.211 |
| (e) + Momentum self-distillation | 8 | 23.56 | 0.806 | 0.205 |
| (f) Full (+ block weighting) | 8 | **23.65** | **0.813** | **0.194** |

### Key Findings

- Parallel training (b) outperforms independent training (c) at the same block count: data diversity from the shared decoder is the key factor.
- Momentum self-distillation (e) significantly outperforms independent training with 8 blocks (d): LPIPS improves from 0.211 to 0.205.
- Block weighting (f) further improves consistency: LPIPS improves from 0.205 to 0.194.
- Combining PSNR + SSIM for block weighting yields the best results compared to using either metric alone.

## Highlights & Insights

- **Momentum Distillation Decouples Hardware Constraints**: Decoupling block count from GPU count is a highly practical contribution, removing the hardware bottleneck from large-scale scene reconstruction. This paradigm is transferable to any task requiring distributed block training with global consistency.
- **Reconstruction-Guided Adaptive Weighting**: Dynamically weighting blocks via a Gaussian-like function based on reconstruction deviation is both simple and effective—a trick directly applicable to other block-wise training frameworks.
- **Dual Gains in Efficiency and Quality**: Momentum-GS achieves the best reconstruction quality while simultaneously attaining the highest rendering FPS (59.91) and lowest memory usage (4.62 GB), demonstrating that hybrid representation combined with self-distillation offers efficiency advantages as well.

## Limitations & Future Work

- The momentum coefficient $m=0.9$ is fixed and may require adaptive scheduling based on scene complexity.
- Block weighting currently relies solely on PSNR/SSIM as quality indicators; perceptual metrics such as LPIPS may be more appropriate.
- The block partitioning strategy is inherited from CityGaussian; superior partitioning schemes are not explored.
- Validation on indoor or dynamic scenes has not been conducted.

## Related Work & Insights

- **vs. CityGaussian**: CityGaussian adopts a divide-and-conquer strategy but trains each block independently, lacking inter-block interaction and resulting in boundary inconsistencies. Momentum-GS substantially improves consistency via momentum distillation and block weighting, achieving an 18.7% LPIPS gain.
- **vs. VastGaussian**: VastGaussian also employs a divide-and-conquer strategy but similarly lacks a cross-block consistency mechanism.
- **vs. DOGS**: DOGS applies ADMM for distributed training acceleration but does not focus on improving the Gaussian representation itself. Momentum-GS simultaneously addresses efficiency and quality through hybrid representation and self-distillation.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of applying momentum distillation to block-wise 3DGS training is novel and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five large-scale scenes, multiple baselines, and detailed ablations—very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear; the comparative figure (Figure 2) is particularly illustrative.
- Value: ⭐⭐⭐⭐ Strong practical applicability to large-scale scene reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction](s3r-gs_streamlining_the_pipeline_for_large-scale_street_scene_reconstruction.md)
- [\[ICCV 2025\] OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering](occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)
- [\[ICCV 2025\] SegmentDreamer: Towards High-Fidelity Text-to-3D Synthesis with Segmented Consistency Trajectory Distillation](segmentdreamer_towards_high-fidelity_text-to-3d_synthesis_with_segmented_consist.md)
- [\[ICCV 2025\] RayZer: A Self-supervised Large View Synthesis Model](rayzer_a_self-supervised_large_view_synthesis_model.md)
- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
