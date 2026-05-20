---
title: >-
  [Paper Note] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper proposes DeGauss, a self-supervised framework based on decoupled dynamic-static Gaussian Splatting. By combining foreground dynamic Gaussians and background static…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "dynamic scene reconstruction"
  - "dynamic-static decomposition"
  - "distractor-free reconstruction"
  - "self-supervised"
date: 2026-05-08
content_hash: e5d07d569a70b816
---

# DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction

**Conference**: ICCV 2025
**arXiv**: [2503.13176](https://arxiv.org/abs/2503.13176)  
**Code**: [GitHub](https://batfacewayne.github.io/DeGauss.io/)  
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, dynamic scene reconstruction, dynamic-static decomposition, distractor-free reconstruction, self-supervised

## TL;DR

This paper proposes DeGauss, a self-supervised framework based on decoupled dynamic-static Gaussian Splatting. By combining foreground dynamic Gaussians and background static Gaussians via a probabilistic composition mask, it achieves distractor-free 3D reconstruction across a broad range of scenarios, from casually captured image collections to highly dynamic egocentric videos.

## Background & Motivation

Reconstructing clean static 3D scenes from real-world videos and images is a core challenge in computer vision. Existing methods face the following difficulties:

**Limitations of the static-scene assumption**: NeRF and 3DGS perform well in controlled static settings, but in everyday captures containing dynamic elements (pedestrians, moving objects), dynamic content is modeled as view-dependent artifacts ("floaters"), severely degrading reconstruction quality.

**Unique challenges of egocentric video**: Videos recorded with head-mounted cameras involve severe camera motion, motion blur, frequent human–object interactions, and the operator's own body, making dynamic-static separation extremely difficult.

**Shortcomings of existing distractor-removal methods**:
   - **Residual-based methods** (NeRF-on-the-go, WildGaussians, SpotlessSplats) use reconstruction-loss residuals and semantic features to identify dynamic regions, but are sensitive to initialization and generalize poorly under intra-class dynamic-static ambiguity and scene deformation.
   - **NeRF-based decomposition methods** (Nerf-W, NeuralDiff, D2NeRF) model dynamics through a dedicated branch, but train slowly; their dynamic branches may under-segment in sparse-input settings and over-segment in highly dynamic videos.

**Key insight**: Dynamic Gaussians excel at temporal modeling but tend to overfit training viewpoints and generalize poorly to novel views; static Gaussians cannot handle motion but provide more stable cross-view representations. DeGauss leverages the complementary strengths of both via a decoupled foreground–background design that enables coordinated optimization.

## Method

### Overall Architecture

DeGauss employs two sets of Gaussians: foreground dynamic Gaussians $\mathcal{G}_f$ with a spatiotemporal deformation module to handle dynamic elements, and background static Gaussians $\mathcal{G}_b$ for static content. A probabilistic composition mask derived by rasterizing the foreground Gaussians governs their combination, allowing the two branches to be optimized independently while remaining mutually complementary.

### Key Designs

1. **Foreground Deformable Gaussian**: In addition to standard Gaussian attributes, three extra features $\{m_f, m_b, b\}$ are introduced, corresponding to foreground probability, background probability, and brightness control, respectively. A spatiotemporal deformation module uses a HexPlane encoder $\mathcal{H}$ to extract spatiotemporal features $\mathbf{f_d} = \mathcal{H}(\mathcal{G}_f, t)$, which are then decoded by a multi-head MLP decoder $\mathcal{D}$ to predict deformation deltas for all attributes:
    $\mathcal{G}'_f = \Delta\mathcal{G}_f + \mathcal{G}_f$
   Each attribute (position, rotation, scale, opacity, color, mask, brightness) is predicted by a dedicated MLP.

2. **Probabilistic Composition Mask**: Differentiable rendering applied to the $m_f', m_b'$ attributes of the foreground Gaussians yields raw foreground probability $\mathbf{M}_f$ and background probability $\mathbf{M}_b$, which are then normalized into probability masks:
    $\mathbf{P}_f = \frac{\mathbf{M}_f}{\mathbf{M}_f + \mathbf{M}_b + \epsilon}, \quad \mathbf{P}_b = \frac{\mathbf{M}_b}{\mathbf{M}_f + \mathbf{M}_b + \epsilon}$
   This probabilistic formulation naturally suppresses intermediate values (near 0.5), driving predictions toward binary extremes and producing clear dynamic-static decompositions.

3. **Background Brightness Control**: In casual captures, illumination varies significantly, and the high expressive capacity of foreground Gaussians tends to over-segment regions with lighting changes as dynamic. A brightness control mask $\mathbf{B}$ is introduced to enhance the background branch's ability to model non-Lambertian effects. A piecewise linear activation maps brightness values to the range $[0.5, 1.25+]$, preventing dark dynamic objects from being misinterpreted as brightness variations while supporting overexposure modeling:
    $\hat{\mathbf{C}}_b = \hat{\mathbf{B}} * \mathbf{C}_b$

4. **Unsupervised Scene Decomposition**: The final composite rendering is $\hat{\mathbf{C}} = \mathbf{P}_f * \mathbf{C}_f + \mathbf{P}_b * \hat{\mathbf{B}} * \mathbf{C}_b$. Unlike NeRF-based methods that blend densities along rays during rendering, DeGauss composites the foreground and background independently after rendering, avoiding local optima caused by premature ray termination and the loss of static scene detail.

5. **Partial Opacity Reset**: Periodically resetting opacity completely—as done in SpotlessSplats—causes training instability. Benefiting from the additional stability provided by the foreground–background decoupling, DeGauss performs periodic partial opacity resets at a 50% rate, effectively controlling Gaussian density, eliminating floaters, and avoiding local optima.

### Loss & Training

Losses are divided into two groups, with gradient separation used to govern the adaptive Gaussian densification process:
- **Main loss $\mathcal{L}_\text{main}$** (used for densification) $= \mathcal{L}_1 + \mathcal{L}_\text{diversity} + \mathcal{L}_\text{reg} + \mathcal{L}_\text{depth} + \mathcal{L}_f + \mathcal{L}_b$
- **Utility loss $\mathcal{L}_\text{uti}$** (not used for densification) $= \mathcal{L}_\text{SSIM} + \mathcal{L}_\text{entropy} + \mathcal{L}_\text{brightness} + \mathcal{L}_\text{scale}$

Training proceeds in two stages: a coarse stage (1K iterations, deformation module disabled) followed by a fine stage (20K–120K iterations, end-to-end joint optimization).

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | SpotlessSplats | Gain |
|---------|--------|------|----------------|------|
| NeRF-on-the-go (Mean) | PSNR↑ | 23.91 | 23.42 | +0.49 |
| NeRF-on-the-go (Mean) | SSIM↑ | 0.819 | 0.813 | +0.006 |
| NeRF-on-the-go (Mean) | LPIPS↓ | 0.113 | 0.145 | -0.032 |
| Neu3D (Mean) | LPIPS↓ | 0.047 | - | SOTA |
| Neu3D (Mean) | PSNR↑ | 31.52 | - | Comparable to 4DGS |

DeGauss outperforms SpotlessSplats in LPIPS on all 6 scenes of NeRF-on-the-go. On the Neu3D dataset, it achieves substantially better LPIPS than all baselines (0.047 vs. 0.058 for 4DGS).

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Note |
|---------------|-------|-------|--------|------|
| Full model | 23.91 | 0.819 | 0.113 | All components |
| w/o brightness control | 23.54 | 0.814 | 0.118 | Illumination ambiguity causes over-segmentation |
| w/o partial reset | 23.56 | 0.814 | 0.117 | Floater residuals remain |
| w/o depth loss | 23.68 | 0.816 | 0.113 | Geometric constraints absent |
| w/o background mask element | 23.83 | 0.817 | 0.115 | Decomposition less clean |

### Key Findings

- The decoupled post-rendering composition strategy is more robust than NeRF's along-ray density blending (in-rendering composition), with more complete gradient flow.
- The brightness control mask is critical for resolving dynamic-static ambiguity caused by illumination variation.
- The method operates stably across settings ranging from casual image collections to highly dynamic egocentric long videos (2,800–5,000 frames).
- The dynamic foreground modeling capability simultaneously endows the method with high-quality dynamic scene representation.

## Highlights & Insights

- The design is remarkably clean and elegant: decoupled foreground–background Gaussians, probabilistic composition mask, and brightness control — three simple components combined to achieve strong generalization.
- The probability mask naturally tends toward binary values through normalization, requiring no additional regularization.
- "Utility Gaussians" within the foreground branch contribute to mask and brightness computation without contributing to color, increasing the model's expressiveness.
- Comprehensive evaluation spanning image collections to long videos demonstrates the generality of the approach.

## Limitations & Future Work

- Quantitative evaluation is not performed on the HyperNeRF dataset due to inaccurate camera poses.
- Training on long videos (thousands of frames) requires 120K iterations, incurring substantial computational cost.
- The method's ability to handle extreme motion blur (e.g., fast hand movements) has not been thoroughly validated.
- The potential to leverage semantic priors (e.g., SAM/DINO) to further improve decomposition quality remains unexplored.

## Related Work & Insights

- The diffusion-feature clustering approach of SpotlessSplats is limited by initialization sensitivity, motivating the need for explicit decomposition methods.
- The HexPlane encoder from 4DGS is successfully repurposed as a spatiotemporal encoder for foreground deformation.
- The decoupled design paradigm is potentially extensible to downstream tasks such as multi-body dynamic reconstruction and semantic segmentation.

## Rating

- Novelty: ⭐⭐⭐⭐ The probabilistic composition design with decoupled foreground–background Gaussians is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers diverse data sources including image collections, egocentric videos, and fixed multi-view setups.
- Writing Quality: ⭐⭐⭐⭐ Method is clearly described with rich experimental support.
- Value: ⭐⭐⭐⭐ Establishes a highly generalizable baseline for distractor-free reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction](event-boosted_deformable_3d_gaussians_for_dynamic_scene_reconstruction.md)
- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [\[ICCV 2025\] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)
- [\[ICCV 2025\] Proactive Scene Decomposition and Reconstruction](proactive_scene_decomposition_and_reconstruction.md)
- [\[ICCV 2025\] PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations](pcr-gs_colmap-free_3d_gaussian_splatting_via_pose_co-regularizations.md)

</div>

<!-- RELATED:END -->
