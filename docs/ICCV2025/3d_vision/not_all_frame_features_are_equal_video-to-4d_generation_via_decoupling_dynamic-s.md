---
title: >-
  [Paper Note] Not All Frame Features Are Equal: Video-to-4D Generation via Decoupling Dynamic-Static Features
description: >-
  [ICCV 2025][3D Vision][4D generation] DS4D is the first method to decouple dynamic and static features along both the temporal and spatial axes in video-to-4D generation. It introduces a Dynamic-Static Feature Decoupling module (DSFD) to extract dynamic representations, and a Spatiotemporal Similarity Fusion module (TSSF) to adaptively aggregate dynamic information across viewpoints, achieving state-of-the-art performance on the Consistent4D and Objaverse datasets.
tags:
  - ICCV 2025
  - 3D Vision
  - 4D generation
  - dynamic 3D Gaussians
  - dynamic-static feature decoupling
  - video-to-4D
  - spatiotemporal similarity fusion
date: 2026-05-08
content_hash: 877fbb237622def7
---

# Not All Frame Features Are Equal: Video-to-4D Generation via Decoupling Dynamic-Static Features

**Conference**: ICCV 2025
**arXiv**: [2502.08377](https://arxiv.org/abs/2502.08377)
**Code**: Coming soon
**Area**: 3D Vision
**Keywords**: 4D generation, dynamic 3D Gaussians, dynamic-static feature decoupling, video-to-4D, spatiotemporal similarity fusion

## TL;DR

DS4D is the first method to decouple dynamic and static features along both the temporal and spatial axes in video-to-4D generation. It introduces a Dynamic-Static Feature Decoupling module (DSFD) to extract dynamic representations, and a Spatiotemporal Similarity Fusion module (TSSF) to adaptively aggregate dynamic information across viewpoints, achieving state-of-the-art performance on the Consistent4D and Objaverse datasets.

## Background & Motivation

Generating dynamic 3D content from video (i.e., 4D generation) is an important problem in computer vision and graphics, with broad applications in virtual reality, gaming, and film production. However, accurately predicting motion trajectories from sparse viewpoints while maintaining high-quality generation remains a significant challenge.

**Core Limitation**: Existing methods—whether optimization-based or inference-based—directly use all information from full frames to model spatiotemporal correlations, entirely ignoring the distinction between dynamic and static regions within a frame. When static regions dominate (which is extremely common in practice, e.g., a person moving against a fixed background), the model is strongly biased toward fitting static regions, causing the texture details of dynamic regions to become blurry and motion information to be neglected.

**Illustrative Example**: Consider a video of a walking person: the human body (dynamic region) may occupy only 20–30% of the frame, while the background and ground (static regions) occupy 70–80%. In traditional methods, the dominant gradients during Gaussian optimization originate from static regions, causing dynamic regions (e.g., clothing wrinkles, arm swings) to lose texture fidelity.

**Core Idea**: If the dynamic and static information within frame features can be explicitly separated and the dynamic representation strengthened, the "overwhelming" effect of static regions on dynamic regions can be avoided. DS4D is built upon this intuition, proposing to decouple dynamic-static features along the temporal axis and fuse dynamic information across viewpoints along the spatial axis.

## Method

### Overall Architecture

The DS4D pipeline proceeds as follows: (1) Given a monocular video, pseudo-multi-view image sequences are generated using Zero123++; (2) Frame features are extracted with DINOv2; (3) A large reconstruction model initializes a 3D Gaussian point cloud from the middle frame; (4) The DSFD module decouples dynamic and static features along the temporal axis; (5) The TSSF module fuses dynamic information along the spatial axis; (6) A Deformation MLP produces the final 4D content.

### Key Designs

1. **Dynamic-Static Feature Decoupling Module (DSFD)**:

    - Function: Decomposes frame features into dynamic and static components along the temporal axis.
    - Mechanism: The middle-frame feature $f^{(t/2,j)}$ (representing semantic reference) and the mean feature across all frames $\bar{f}^{(\bar{t},j)}$ (representing average motion variation) are combined to form the reference frame feature $r^j$. The current frame feature is then projected onto the reference feature direction to obtain the static component; the orthogonal residual constitutes the dynamic feature:
    - Static feature: $f_{static}^{(i,j)} = \frac{f^{(i,j)} \cdot r^j}{\|r^j\|_2} \cdot \frac{r^j}{\|r^j\|_2}$
    - Dynamic feature: $f_{dynamic}^{(i,j)} = f^{(i,j)} - f_{static}^{(i,j)}$
    - The dynamic feature is concatenated with the current frame feature to yield the decoupled feature $f_d^{(i,j)}$.
    - Design Motivation: The geometric relationship of vector projection is exploited: the component projected onto the reference direction represents the "invariant part" (static), while the orthogonal component represents the "varying part" (dynamic). This formulation is both concise and physically intuitive.

2. **Spatiotemporal Similarity Fusion Module (TSSF)**:

    - Function: Adaptively selects and fuses similar dynamic information from decoupled features across different viewpoints.
    - Core Problem: Due to spatial occlusion, dynamic features from a single viewpoint cannot fully represent dynamic information in 4D space.
    - Design: Decoupled features are first mapped to Gaussian points via viewpoint projection to obtain point features $f_p^{(i,j)}$, which are then aggregated along the spatial axis.
    - **Global-Aware Fusion (GA)**: A fully connected layer followed by Softmax generates a score map $\bm{W}$ for each viewpoint; point features from all viewpoints are then aggregated via weighted summation: $f_a^i = \sum_{j=0}^{v} w^{(i,j)} f_p^{(i,j)}$.
    - **Distance-Aware Fusion (DA)**: The front view (i.e., the real input viewpoint) contains the most accurate motion region information. L1 distances between the point features of other viewpoints and the front view are computed; other viewpoints are first fused (reducing the weight of heavily occluded viewpoints), then combined with the front-view features.
    - Design Motivation: Texture and motion are similar across viewpoints for the same spatial region; this similarity is exploited to complement occluded regions from a single viewpoint using information from other viewpoints.

3. **Dynamic Gaussian Features and HexPlane**:

    - Function: Combines fused point features with HexPlane-derived dynamic Gaussian features.
    - Mechanism: HexPlane regularizes Gaussian attributes such as position, scale, and rotation spatiotemporally. HexPlane features $f_{hg}^i$ and fused point features $f_a^i$ are mapped to the final fused Gaussian features via a learnable linear transformation.
    - Design Motivation: HexPlane ensures field smoothness, while fused point features provide rich dynamic information; the two are complementary.

### Loss & Training

The training loss comprises: an SDS loss (leveraging priors from a pretrained multi-view diffusion model), a photometric loss (between rendered views and ground-truth images), and an LPIPS loss (perceptual similarity between pseudo-multi-view images and rendered views).

Initialization strategy: A large reconstruction model generates a point cloud from the middle frame to initialize Gaussian points, providing geometric priors and ensuring topological stability (ablation studies confirm this outperforms random initialization).

## Key Experimental Results

### Main Results

| Method | Dataset | CLIP↑ | LPIPS↓ | FVD↓ | FID-VID↓ |
|--------|---------|-------|--------|------|----------|
| STAG4D | Consistent4D | 0.9078 | 0.1354 | 986.83 | 26.37 |
| SC4D | Consistent4D | 0.9117 | 0.1370 | 852.98 | 26.48 |
| **DS4D-DA** | **Consistent4D** | **0.9225** | **0.1309** | **784.02** | **24.05** |
| STAG4D | Objaverse | 0.8790 | 0.1811 | 1061.36 | 30.14 |
| SC4D | Objaverse | 0.8490 | 0.1852 | 1067.76 | 40.51 |
| **DS4D-DA** | **Objaverse** | **0.8881** | **0.1759** | **870.95** | **25.38** |

### Ablation Study

| Configuration | CLIP↑ | LPIPS↓ | FVD↓ | FID-VID↓ | Notes |
|---------------|-------|--------|------|----------|-------|
| A. Baseline (no feature enhancement) | 0.9133 | 0.1341 | 953.63 | 27.37 | No point init., no DSFD |
| B. + Point cloud initialization | 0.9151 | 0.1313 | 913.37 | 27.14 | Geometric prior improves stability |
| D. + Frame features (no decoupling) | 0.9174 | 0.1350 | 888.66 | 26.85 | Improvement but prone to static overfitting |
| E. + DSFD (decoupled) | 0.9186 | 0.1333 | 861.61 | 26.54 | Explicit decoupling brings further gains |
| F. + TSSF (average pooling) | 0.9194 | 0.1313 | 839.66 | 26.51 | Simple averaging is suboptimal |
| G. + TSSF-GA | 0.9206 | 0.1311 | 799.94 | 26.18 | Adaptive selection is more effective |
| H. + TSSF-DA | **0.9225** | **0.1309** | **784.02** | **24.05** | Distance-aware fusion is optimal |

### Key Findings

- FVD improves from 953.63 (baseline) to 784.02, indicating a substantial reduction in temporal artifacts.
- DSFD with vs. without decoupling (E vs. D): FVD drops from 888.66 to 861.61, demonstrating the necessity of explicit decoupling.
- TSSF-DA outperforms TSSF-GA: distance-aware fusion better preserves true motion information from the front view by down-weighting heavily occluded novel viewpoints.
- Effectiveness is also demonstrated on the real-world dataset Neu3D (PSNR 32.40 vs. 32.16 for 4D-GS).

## Highlights & Insights

- **Precise Problem Formulation**: The paper is the first to explicitly identify the "static overfitting" problem caused by imbalanced proportions of dynamic and static regions in 4D generation.
- **Clear Methodological Intuition**: The vector-projection-based decoupling of dynamic and static features is concise and elegant, with unambiguous geometric interpretation.
- **Compelling Visualization**: Heatmaps clearly demonstrate that dynamic features successfully capture motion regions (e.g., elephant trunk, Triceratops legs).
- **Plug-and-Play Design**: DSFD and TSSF can be directly integrated into existing methods such as 4D-GS.

## Limitations & Future Work

- The method relies on the quality of pseudo-multi-view images generated by Zero123++; inaccurate multi-view generation can degrade the decoupling effectiveness.
- The current reference frame selection strategy (middle frame + average frame) is relatively simple; a more adaptive strategy may yield further improvements.
- Explicit motion cues such as optical flow are not utilized to assist decoupling; integrating optical flow and depth features from 3D-aware foundation models is a promising direction.
- Validation is limited to object-level 4D generation; performance on large-scale dynamic scenes remains to be explored.

## Related Work & Insights

- **STAG4D**: A representative optimization-based 4D method that generates multi-view videos via temporal anchors, but does not distinguish between dynamic and static regions.
- **DreamGaussian4D**: Significantly reduces optimization time but lacks detail quality.
- **HexPlane**: The spatiotemporal regularization tool adopted by DS4D to ensure smoothness of the 4D field.
- Insight: The dynamic-static decoupling paradigm is transferable to other 3D tasks that involve motion.

## Rating

- Novelty: ⭐⭐⭐⭐ (Dynamic-static decoupling is a novel contribution, though the method itself is relatively straightforward)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multi-dataset validation + comprehensive ablations + visual analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, high-quality figures and tables)
- Value: ⭐⭐⭐⭐ (Valuable problem formulation, transferable methodology)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Gaussian Variation Field Diffusion for High-fidelity Video-to-4D Synthesis](gaussian_variation_field_diffusion_for_high-fidelity_video-to-4d_synthesis.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)
- [\[ICCV 2025\] Shape of Motion: 4D Reconstruction from a Single Video](shape_of_motion_4d_reconstruction_from_a_single_video.md)
- [\[ICCV 2025\] LocalDyGS: Multi-view Global Dynamic Scene Modeling via Adaptive Local Implicit Feature Decoupling](localdygs_multi-view_global_dynamic_scene_modeling_via_adaptive_local_implicit_f.md)

<!-- RELATED:END -->
