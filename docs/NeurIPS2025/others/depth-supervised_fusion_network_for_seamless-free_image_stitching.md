---
title: >-
  [Paper Note] Depth-Supervised Fusion Network for Seamless-Free Image Stitching
description: >-
  [NeurIPS 2025][image stitching] DSFN proposes a seamless image stitching method with depth consistency constraints: a depth-aware two-stage transformation estimation addresses large-parallax alignment…
tags:
  - "NeurIPS 2025"
  - "image stitching"
  - "depth supervision"
  - "large-parallax alignment"
  - "soft-seam fusion"
  - "re-parameterization"
date: 2026-05-08
content_hash: 5eb20a92780b69d4
---

# Depth-Supervised Fusion Network for Seamless-Free Image Stitching

**Conference**: NeurIPS 2025
**arXiv**: [2510.21396](https://arxiv.org/abs/2510.21396)
**Code**: [GitHub](https://github.com/DLUT-YRH/DSFN)
**Area**: Others (Computer Vision / Image Stitching)
**Keywords**: image stitching, depth supervision, large-parallax alignment, soft-seam fusion, re-parameterization

## TL;DR
DSFN proposes a seamless image stitching method with depth consistency constraints: a depth-aware two-stage transformation estimation addresses large-parallax alignment, soft-seam region diffusion enables natural blending, and a re-parameterization strategy improves efficiency. The method comprehensively surpasses the state of the art on the UDIS-D and IVSD datasets.

## Background & Motivation

**Background**: Image stitching synthesizes multi-view images into wide field-of-view panoramas, with broad applications in panoramic photography, remote sensing, medical imaging, and VR.

**Limitations of Prior Work**:
   - Feature-based methods (SIFT + RANSAC + homography) assume a planar scene model, producing ghosting and misalignment in scenes with multiple depth layers.
   - Mesh-based warping (e.g., APAP) improves local alignment but introduces artifacts at depth discontinuities.
   - Seam optimization is computationally expensive.

**Limitations of Deep Learning Methods**: Reliance on synthetic training data leads to weak cross-domain generalization; structural consistency under large-parallax conditions remains challenging.

**Core Idea**: Leverage depth information provided by monocular depth estimation (Depth Anything) as a geometric prior to supervise multi-view alignment learning; replace hard seam cutting with graph-based soft-seam diffusion.

## Method

### Overall Architecture

Input target image $I_t$ and reference image $I_r$ → ResNet50 feature encoding → depth-aware transformation estimation → alignment → soft-seam fusion → output wide field-of-view stitched result $I_s$.

### Depth-Aware Transformation Estimation

**Two-stage progressive strategy**:

**Stage 1: Coarse Alignment (1/16 scale)**
- Feature Correlation Aggregation (FCA) computes cross-view correspondences: $C_{i,j} = FCA(F_r^{1/16}, F_t^{1/16})$
- Regresses quadrilateral vertex offsets $\Delta p \in \mathbb{R}^{4 \times 2}$
- DLT solves the coarse homography: $H_C = \arg\min_H \sum_{k=1}^4 \|p_k' - H \cdot p_k\|_2^2$

**Stage 2: Fine Alignment (1/8 scale)**
- Transforms target features into the reference space using $H_C$
- Estimates grid-level offsets; RBF interpolation generates a continuous deformation field:

$$\Delta(x,y) = \sum_{m=1}^M w_m \phi(\|(x,y) - (x_m, y_m)\|)$$

where $\phi(r) = -e^{-(\epsilon r)^2}$ is a Gaussian basis function. The final dense deformation field is:

$$\mathcal{W}(p) = H_C \cdot p + \Delta(p)$$

**Depth Supervision**: Depth maps $I_{dr}, I_{dt}$ are obtained via Depth Anything; after normalization over the overlapping region, they are incorporated into the alignment loss:

$$\mathcal{L}_{depth} = f_{alignment}(I_{dr}, I_{dt}, \lambda', \gamma', \eta')$$

**Total Transformation Loss**:

$$\mathcal{L}^t = \mathcal{L}_{alignment} + \mu \mathcal{L}_{edge} + \zeta \mathcal{L}_{angle} + \xi \mathcal{L}_{depth}$$

where $\mathcal{L}_{edge}$ penalizes mesh stretching and $\mathcal{L}_{angle}$ enforces parallelism of adjacent edges in non-overlapping regions.

### Soft-Seam Fusion

**Core Idea**: Relaxing the traditional hard-seam definition, any region within the overlapping area that requires blending is treated as a potential seam zone.

1. **SSE Module**: Based on a UNet architecture (standard convolutions replaced with dilated convolutions, dilation rates 1–5), takes the aligned image mask as input and outputs a soft-seam mask $M_s$.
2. **Adaptive Weights**: $M_s$ and the original mask are passed through sigmoid to generate pixel-wise adaptive blending weights $M_{sr}$ and $M_{st}$.

**Fusion Loss**:

$$\mathcal{L}^f = \rho \mathcal{L}_{terminal} + \tau \mathcal{L}_{cost} + \iota \mathcal{L}_{smooth} + \sigma \mathcal{L}_{reg}$$

- $\mathcal{L}_{cost}$: A cost map based on squared pixel differences, penalizing high-cost regions at mask transitions.
- $\mathcal{L}_{smooth}$: Smoothness constraint on neighboring pixels.
- $\mathcal{L}_{reg}$: Depth consistency regularization — enforcing local consistency of the aligned depth maps in the stitching region.

### Re-Parameterization-Based Regression (RBA)

RepBlock (parallel 1×1 and 3×3 convolutions) is introduced into shift regression. During training, the contribution of each branch is evaluated:

$$c_1 = \frac{\frac{1}{C_{out}}\sum \mathbf{w_1}}{\frac{1}{C_{out}}\sum \mathbf{w_1} + \frac{1}{C_{out}}\sum \mathbf{w_3}}$$

If $c_1 < \hat{c}$ (threshold), the 1×1 branch is merged into the 3×3 branch:

$$\mathbf{W}_3^{new} = \mathbf{w_3} \cdot \mathbf{W_3} + \mathbf{w_1} \cdot pad(\mathbf{W_1})$$

Experiments identify $\hat{c} = 0.25$ as the optimal threshold.

## Key Experimental Results

### Quantitative Comparison on UDIS-D

| Method | PSNR↑ | SSIM↑ | SIQE↑ | LPIPS↓ |
|--------|-------|-------|-------|--------|
| APAP | 23.792 | 0.794 | 41.707 | 0.472 |
| ELA | 24.012 | 0.808 | 41.781 | 0.470 |
| UDIS | 21.171 | 0.648 | 42.186 | 0.475 |
| UDIS++ | 25.426 | 0.837 | 43.184 | 0.469 |
| SRS | 24.828 | 0.811 | 41.857 | 0.473 |
| **DSFN (Ours)** | **25.467** | **0.839** | **43.732** | **0.462** |

### Generalization on IVSD

| Method | PSNR↑ | SSIM↑ | SIQE↑ | LPIPS↓ |
|--------|-------|-------|-------|--------|
| UDIS++ | 26.649 | 0.819 | 46.383 | 0.439 |
| SRS | 24.234 | 0.796 | 35.641 | 0.445 |
| **DSFN (Ours)** | **26.778** | **0.820** | **46.568** | **0.436** |

Performance consistently leads across datasets.

### Runtime Efficiency (512×512 images)

| Method | Time (ms) |
|--------|----------|
| APAP | 6683 |
| ELA | 8348 |
| UDIS | 194 |
| UDIS++ | 80 |
| SRS | 83 |
| **DSFN** | **67** |

DSFN is the fastest method, despite incorporating depth estimation in the inference pipeline.

### Ablation Study

| Configuration | PSNR | SSIM | SIQE | LPIPS |
|---------------|------|------|------|-------|
| w/o $\mathcal{L}_{smooth}$ | 25.431 | 0.833 | 43.156 | 0.466 |
| w/o $\mathcal{L}_{cost}$ | 25.438 | 0.836 | 43.186 | 0.463 |
| w/o $\mathcal{L}_{depth}$ | 25.434 | 0.838 | 43.703 | 0.463 |
| w/o $\mathcal{L}_{mesh}$ | 25.473 | 0.840 | 43.701 | 0.463 |
| **Full** | **25.470** | **0.839** | **43.732** | **0.462** |

Removing the mesh constraint yields marginally higher metrics (due to relaxed deformation constraints), but introduces visible distortion in qualitative results.

### User Study
50 participants (30 with a computer vision background) rated methods on a 1–5 scale; DSFN consistently received the highest scores.

## Highlights & Insights
- **Depth information as an alignment prior** is a natural and effective approach for large-parallax image stitching — obtained at zero additional cost via the off-the-shelf Depth Anything model.
- **Soft-seam fusion** replacing hard seam cutting is a key innovation: pixel-wise adaptive blending via mask diffusion is smoother than graph-cut and supports end-to-end training.
- **RBA re-parameterization** maintains multi-branch diversity during training while merging into a single branch at inference — balancing efficiency and performance.
- The fastest runtime (67 ms) demonstrates a compact and efficient overall architecture.

## Limitations & Future Work
- Depth supervision relies on the quality of Depth Anything — unreliable monocular depth estimation in specific scenes may propagate errors.
- Validation is limited to the UDIS-D and IVSD datasets; large-scale real-world panoramic datasets (e.g., Google Street View) are not evaluated.
- Occlusion from dynamic objects (moving pedestrians/vehicles) is not addressed.
- Dilation rates in the soft-seam module's dilated convolutions are manually set without automated search.
- The two-stage training (transformation and fusion trained separately) leaves room for improvement through end-to-end joint training.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of depth supervision and soft-seam fusion is novel; the RBA strategy demonstrates engineering innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers quantitative, qualitative, ablation, user study, and efficiency comparisons — relatively comprehensive.
- **Writing Quality**: ⭐⭐⭐ Formulas and loss function definitions are clear, but some symbols are overloaded (e.g., $\sigma$ denotes both an activation function and a loss weight).
- **Value**: ⭐⭐⭐⭐ Directly applicable to real-world large-parallax image stitching tasks, with the fastest reported inference speed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Revisiting Image Fusion for Multi-Illuminant White-Balance Correction](../../ICCV2025/others/revisiting_image_fusion_for_multi-illuminant_white-balance_correction.md)
- [\[NeurIPS 2025\] Depth-Bounds for Neural Networks via the Braid Arrangement](depth-bounds_for_neural_networks_via_the_braid_arrangement.md)
- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)
- [\[NeurIPS 2025\] Prediction-Powered Semi-Supervised Learning with Online Power Tuning](prediction-powered_semi-supervised_learning_with_online_power_tuning.md)
- [\[ICCV 2025\] Omni-DC: Highly Robust Depth Completion with Multiresolution Depth Integration](../../ICCV2025/others/omni-dc_highly_robust_depth_completion_with_multiresolution_depth_integration.md)

</div>

<!-- RELATED:END -->
