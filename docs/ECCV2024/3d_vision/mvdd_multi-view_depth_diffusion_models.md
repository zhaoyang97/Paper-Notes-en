---
title: >-
  [Paper Note] MVDD: Multi-View Depth Diffusion Models
description: >-
  [ECCV 2024][3D Vision][Diffusion Models] MVDD is proposed, a diffusion model based on multi-view depth map representations. By incorporating epipolar "line segment" attention and denoising depth fusion, it achieves 3D-consistent, high-quality shape generation, enabling the synthesis of dense point clouds with over 20K points.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Diffusion Models"
  - "Multi-View Depth"
  - "3D Shape Generation"
  - "Epipolar Attention"
  - "Depth Completion"
date: 2026-05-08
content_hash: 20dbf3ad3ca4c1f8
---

# MVDD: Multi-View Depth Diffusion Models

**Conference**: ECCV 2024  
**arXiv**: [2312.04875](https://arxiv.org/abs/2312.04875)  
**Code**: [Project Page](https://mvdepth.github.io/)  
**Area**: Image Generation  
**Keywords**: Diffusion Models, Multi-View Depth, 3D Shape Generation, Epipolar Attention, Depth Completion

## TL;DR

MVDD is proposed, a diffusion model based on multi-view depth map representations. By incorporating epipolar "line segment" attention and denoising depth fusion, it achieves 3D-consistent, high-quality shape generation, enabling the synthesis of dense point clouds with over 20K points.

## Background & Motivation

**Background**: 3D shape generation is an important direction in AIGC. Existing generative models include methods based on implicit functions (AutoSDF, 3D-LDM), voxels (Vox-Diff), and point clouds (DPM, PVD, LION). While diffusion models have achieved immense success in 2D image generation, replicating this success in 3D generation remains challenging.

**Limitations of Prior Work**: (a) Implicit methods suffer from cubic computational growth with resolution or generate over-smoothed shapes; (b) point cloud diffusion models train extremely slowly on unstructured data ($>10000$ epochs) and can only generate around 2048 points, failing to capture fine details; (c) multi-view RGB diffusion suffers from the Janus problem and 3D inconsistency.

**Key Challenge**: High-quality 3D generation requires high resolution and fine details, but existing point cloud, voxel, or implicit representations are either limited in resolution or excessively costly to train. A representation is needed that fits the diffusion framework well and efficiently represents complex 3D shapes.

**Goal**: Design a diffusion model capable of generating 3D-consistent multi-view depth maps for high-quality point cloud and mesh generation.

**Key Insight**: Multi-view depth maps are a representation that "registers" 3D surfaces onto 2D grids, naturally fitting 2D diffusion architectures while generating higher-resolution outputs than voxel or point cloud methods.

**Core Idea**: Replace point clouds or implicit functions with multi-view depth maps as the generation target of the 3D diffusion model, and leverage epipolar line segment attention to resolve cross-view consistency.

## Method

### Overall Architecture

MVDD represents a 3D shape $\mathcal{X}$ as $N$ multi-view depth maps $\mathbf{x} \in \mathbb{R}^{N \times H \times W}$. The forward process adds noise to each depth map independently for $T$ steps, while the reverse process denoises them using a U-Net. The key is introducing cross-view conditioning during the denoising process, so that the denoising step for each view is conditioned on its neighboring views:

$$p_\theta(\mathbf{x}_{t-1}^v | \mathbf{x}_t^v, \mathbf{x}_t^{r_1:r_R}) := \mathcal{N}(\mathbf{x}_{t-1}^v; \mu_\theta(\mathbf{x}_t^v, \mathbf{x}_t^{r_1:r_R}, t), \beta_t \mathbf{I})$$

Finally, the multi-view depth maps are back-projected and fused to obtain a dense point cloud (20K+ points). Optionally, SAP can be used for high-quality mesh reconstruction.

### Key Designs

1. **Epipolar Line Segment Attention**: Unlike the full attention of MVDream or the epipolar attention of SyncDreamer, MVDD leverages the current step's depth estimation to narrow down the attention range. For a pixel $v_{ij}$ on the source view $v$, its depth value $\mathbf{x}_t^{v_{ij}}$ is first back-projected into the 3D space to obtain a point $\rho^{v_{ij}}$:

$$\rho^{v_{ij}} = \mathbf{x}_t^{v_{ij}} A^{-1} v_{ij}$$

Then, $k-1$ equally spaced points along the ray direction of this 3D point are selected and projected onto the neighboring view $r$ to form a "line segment." Features are sampled only at these locations to serve as K and V. Design Motivation: Leveraging the current depth estimate avoids searching the entire epipolar line, narrowing the search to the vicinity of the estimated position, which balances efficiency and effectiveness.

2. **Visibility Threshold Filtering**: Visibility is determined by checking the difference between the back-projected depth and the predicted depth of the neighboring view:

$$M(r_{mn}) = \|z(\pi_{v \to r} \rho^{v_{ij}}) - \mathbf{x}_t^{r_{mn}}\| < \tau$$

Attention weights at locations where this condition is not met are set to a very small value.

3. **Depth Concatenation**: The depth values of the sampled points $\{z(\rho_1^{v_{ij}}), \ldots, z(\rho_k^{v_{ij}})\}$ are concatenated to the feature dimension of V, enabling the model to perceive the spatial positions of these points. Intuition: If the geometric feature of $v_{ij}$ matches highly with that of a sampled point $\rho_1^{v_{ij}}$, the denoised depth should shift closer to the depth value of that point.

4. **Denoising Depth Fusion**: Even with epipolar attention ensuring semantic consistency, back-projected 3D points might still not align perfectly, causing "double-layer" artifacts. Borrowing from MVS methods, depth averaging is performed after the U-Net output during denoising steps: the back-projected depths of each pixel from other visible views are averaged, re-noised, and passed to the next step. Visibility check uses dual thresholds:

$$\|v_{ij} - v_{\tilde{i}\tilde{j}}\| < \psi_{\max}, \quad \frac{|\mathbf{x}^{v_{ij}} - z(\rho^{v_{\tilde{i}\tilde{j}}})|}{|\mathbf{x}^{v_{ij}}|} < \epsilon_\theta$$

This fusion is only applied in the final 20 steps, with an extra depth filtering in the final step to remove invisible points.

### Loss & Training

Standard DDPM objective function to predict noise:

$$L_t = \mathbb{E}_{t, \mathbf{x}_0, \epsilon_t} \left[\|\epsilon_t - \epsilon_\theta(\sqrt{\bar{\alpha}_t} \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t} \epsilon_t, t)\|^2\right]$$

Training setup: $T=1000$ steps, cosine schedule, depth map resolution of $128 \times 128$, 8 views, trained on 8×A100 for approximately 3000 epochs.

## Key Experimental Results

### Main Results

**ShapeNet Unconditional Generation (1-NNA EMD↓)**:

| Category | DPM | PVD | LION | 3D-LDM | IM-GAN | **MVDD** |
|------|-----|-----|------|--------|--------|----------|
| Airplane | 73.47 | 64.89 | 63.49 | 80.10 | 64.04 | **62.50** |
| Car | 80.33 | 71.29 | 65.70 | - | 57.04 | **56.80** |
| Chair | 65.73 | 56.14 | 57.31 | 65.30 | 55.54 | **54.51** |

**Depth Completion (EMD×10²↓)**:

| Category | PointFlow | PVD | DPF-Net | **MVDD** |
|------|-----------|-----|---------|----------|
| Airplane | 1.180 | 1.030 | 1.105 | **0.900** |
| Chair | 3.649 | 2.939 | 3.320 | **2.400** |
| Car | 2.851 | 2.146 | 2.318 | **1.460** |

### Ablation Study

**Ablation on Epipolar Attention Design (Chair, 1-NNA CD↓)**:

| Component | Full Model | w/o Line Segment Attention | w/o Depth Concatenation | w/o Threshold Filtering |
|------|----------|-------------|-----------|-----------|
| 1-NNA↓ | Best | Significant degradation | Quality degradation | Slight degradation |

The effectiveness of denoising depth fusion is intuitively shown in Fig. 4: noticeable "double-layer" artifacts appear when fusion is not utilized.

### Key Findings

- The point cloud density generated by MVDD is **10 times** that of existing point cloud diffusion models (20K+ vs 2048), enabling it to capture fine structures like chair slats and thin airplane wings.
- As point cloud density increases, the performance of sparse methods like LION drops sharply, whereas MVDD remains stable.
- MVDD comprehensively surpasses all baselines in the depth completion task, demonstrating that the model has learned realistic 3D shape priors.
- It can serve as a 3D prior for downstream tasks such as GAN inversion to prevent geometric collapse.

## Highlights & Insights

- **Insights on representation choice**: Multi-view depth maps "reduce the dimensionality" of 3D generation to 2D generation, perfectly fitting mature 2D diffusion architectures, which is more efficient than denoising on unstructured point clouds.
- **Utilizing intermediate results**: Epipolar "line segment" attention cleverly utilizes the depth estimates from diffusion intermediate steps to narrow down the search range, which was unexploited in previous multi-view diffusion works.
- **Versatility**: The same unconditional generative model can be directly applied to depth completion and used as a 3D prior, demonstrating exceptional flexibility.

## Limitations & Future Work

- The $128 \times 128$ depth map resolution still has room for improvement; higher resolutions could provide more geometric details.
- The fixed camera configuration of 8 views may not be suitable for all scenes (e.g., objects with thin structures might require more views).
- Training data is limited to single classes from ShapeNet, and cross-category generalization has not been demonstrated.
- Integration with current mainstream text-to-3D methods (SDS-based) has not yet been explored.
- Inference requires 1000 denoising steps, leaving room for speed optimization.

## Related Work & Insights

- **DDPM/DDIM**: Foundation diffusion frameworks.
- **MVDream**: Multi-view RGB diffusion using 3D self-attention. MVDD borrows the concept of cross-view interaction but designs a more efficient attention mechanism specifically for depth maps.
- **PVD**: Point cloud diffusion baseline, which denoises directly on point coordinates, leading to slow training and limited numbers of points.
- **SAP**: A post-processing method for reconstructing meshes from point clouds.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to introduce multi-view depth representations to 3D diffusion generation, with a cleverly designed epipolar line segment attention.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers three tasks (generation, completion, and GAN prior) with comprehensive quantitative and qualitative results.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation with explicit physical intuition behind each module design.
- **Value**: ⭐⭐⭐⭐ — The potential of multi-view depth as a 3D representation warrants further research follow-up.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Diffusion Models for Monocular Depth Estimation: Overcoming Challenging Conditions](diffusion_models_for_monocular_depth_estimation_overcoming_challenging_condition.md)
- [\[ECCV 2024\] SEDiff: Structure Extraction for Domain Adaptive Depth Estimation via Denoising Diffusion Models](sediff_structure_extraction_for_domain_adaptive_depth_estimation_via_denoising_d.md)
- [\[ECCV 2024\] Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [\[ECCV 2024\] DiffusionDepth: Diffusion Denoising Approach for Monocular Depth Estimation](diffusiondepth_diffusion_denoising_approach_for_monocular_depth_estimation.md)
- [\[ICCV 2025\] SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models](../../ICCV2025/3d_vision/spinmeround_consistent_multi-view_identity_generation_using_diffusion_models.md)

</div>

<!-- RELATED:END -->
