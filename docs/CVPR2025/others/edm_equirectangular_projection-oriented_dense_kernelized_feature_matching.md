---
title: >-
  [Paper Note] EDM: Equirectangular Projection-Oriented Dense Kernelized Feature Matching
description: >-
  [CVPR 2025][panoramic images] EDM is proposed as the first learning-based dense feature matching method for Equirectangular Projection (ERP) panoramic images. It addresses the polar distortion of ERP through a Spherical Space Alignment Module (SSAM, utilizing spherical positional encoding with 3D Cartesian coordinates + Gaussian Process regression) and geodesic flow refinement. On Matterport3D, it outperforms DKM by 26.72% in AUC@5°, and on Stanford2D3D by 42.62%.
tags:
  - "CVPR 2025"
  - "panoramic images"
  - "equirectangular projection"
  - "dense matching"
  - "spherical alignment"
  - "geodesic refinement"
date: 2026-05-08
content_hash: 97edc40b8ac210e3
---

# EDM: Equirectangular Projection-Oriented Dense Kernelized Feature Matching

**Conference**: CVPR 2025  
**arXiv**: [2502.20685](https://arxiv.org/abs/2502.20685)  
**Code**: [https://jdk9405.github.io/EDM](https://jdk9405.github.io/EDM)  
**Area**: Other  
**Keywords**: panoramic images, equirectangular projection, dense matching, spherical alignment, geodesic refinement

## TL;DR
EDM is proposed as the first learning-based dense feature matching method for Equirectangular Projection (ERP) panoramic images. It addresses the polar distortion of ERP through a Spherical Space Alignment Module (SSAM, utilizing spherical positional encoding with 3D Cartesian coordinates + Gaussian Process regression) and geodesic flow refinement. On Matterport3D, it outperforms DKM by 26.72% in AUC@5°, and on Stanford2D3D by 42.62%.

## Background & Motivation

**Background**: Dense feature matching is widely applied in 3D reconstruction and visual localization. Existing dense matching methods (e.g., DKM, RoMa) are designed for perspective projection images, but 360° panoramic ERP images outputted by panorama cameras exhibit severe geometric distortion, especially in the polar regions.

**Limitations of Prior Work**: (1) Directly applying perspective image matching methods to ERP images leads to significant performance degradation, as the feature extractors and positional encodings do not account for spherical geometry. (2) Alternative solutions such as cubemap projection partially resolve distortion but lose global information and require six inference runs. (3) Currently, there are no learning-based dense matching methods specifically designed for panoramic images.

**Key Challenge**: The geometric distortion of ERP images (where the same real-world distance corresponds to a much larger pixel distance in polar regions compared to the equator) completely invalidates positional encodings and matching strategies that are based on Euclidean space.

**Goal**: To design a dense feature matching method that natively supports the spherical geometry of ERP.

**Key Insight**: To lift the matching process from the 2D image plane to the 3D unit sphere—using spherical coordinates for positional encoding, replacing Euclidean distance with geodesic distance, and replacing pixel distance loss with angular difference loss.

**Core Idea**: To perform dense matching in spherical space instead of the image plane, using spherical positional encoding and Gaussian Process spherical regression to achieve coarse matching, followed by iterative refinement through geodesic flow for fine matching.

## Method

### Overall Architecture
Given two input ERP images, features are first extracted using a CNN: (1) The SSAM module executes coarse matching on the unit sphere using spherical positional encoding and Gaussian Process regression; (2) The geodesic flow refinement module performs bidirectional transformation between ERP and spherical coordinates to iteratively optimize displacement along the spherical surface; (3) The final outputs are dense pixel correspondences and confidence maps.

### Key Designs

1. **Spherical Space Alignment Module (SSAM)**:

    - **Function**: Achieves distortion-aware coarse matching at the global level.
    - **Mechanism**: Converts ERP pixel coordinates to 3D Cartesian coordinates $(x,y,z)$ on the unit sphere via an inverse projection function $\pi^{-1}$, using these 3D coordinates as positional encodings instead of traditional 2D sinusoidal positional encodings. It then establishes correspondences between the two images using Gaussian Process (GP) regression in the spherical space, where the GP kernel function is naturally suited for handling the non-uniformity of spherical geometry.
    - **Design Motivation**: 2D positional encodings cannot reflect the geometric distortion of ERP, where the actual spherical distance between two adjacent pixels in polar regions is much smaller than at the equator. 3D spherical coordinates eliminate this distortion.

2. **Geodesic Flow Refinement**:

    - **Function**: Iteratively optimizes matching displacement along the spherical surface at a fine scale.
    - **Mechanism**: Establishes bidirectional transformations $\pi$ and $\pi^{-1}$ between ERP coordinates and spherical coordinates. In each iteration, the current matching points are first transformed to the sphere to update the displacement (along the geodesic), and then projected back to the ERP space to verify the correspondence. The displacement on the sphere follows the shortest path of the great circle.
    - **Design Motivation**: Directly refining displacements on the ERP plane would be warped by polar distortion; refining displacements on the sphere guarantees geometric correctness.

3. **ERP Data Augmentation**:

    - **Function**: Increases training data diversity while maintaining geometric consistency.
    - **Mechanism**: Performs random rotation $\theta_{aug} \in [0, 2\pi]$ along the azimuth direction, which corresponds to horizontal translation of the ERP image. Due to the circular nature of ERP, it remains a valid ERP image after horizontal translation, and the ground truth correspondence can be accurately calculated via simple coordinate transformation.
    - **Design Motivation**: Azimuthal rotation of ERP images is the most natural way of data augmentation, which can significantly increase the number of training pairs.

### Loss & Training
An angular difference loss (cosine similarity) is used to replace the Euclidean distance loss: $L = 1 - \cos(\angle(\hat{p}, p_{gt}))$, where $\hat{p}$ and $p_{gt}$ are the direction vectors of the matched points on the sphere. Training takes about 2 days (300K steps) on a single RTX 3090.

## Key Experimental Results

### Main Results

| Dataset | Method | AUC@5° | AUC@10° | AUC@20° |
|--------|------|--------|---------|---------|
| Matterport3D | SphereGlue (Sparse) | 11.29| 19.95 | 31.10 |
| Matterport3D | DKM (Dense) | 18.43 | 28.50 | 38.44 |
| Matterport3D | **EDM** | **+26.72↑** | - | - |
| Stanford2D3D | DKM | - | - | - |
| Stanford2D3D | **EDM** | **+42.62↑**| - | - |

EDM outperforms the strongest perspective method, DKM, by 26.72 AUC@5° points on Matterport3D, and by 42.62 points on Stanford2D3D, achieving an order-of-magnitude improvement.

### Key Findings
- Dense matching methods for perspective images deteriorate severely on ERP, verifying the necessity of spherical geometry handling.
- Although sparse methods (e.g., SphereGlue) consider the spherical domain, their density is insufficient, and the dense method EDM comprehensively outperforms them.
- Angular difference loss is more suitable for spherical matching than Euclidean distance loss.
- Robust generalization capabilities are also demonstrated on different ERP datasets such as EgoNeRF and OmniPhotos.

## Highlights & Insights
- **Filling the gap in ERP dense matching for the first time**: Extending dense matching from perspective projection to equirectangular projection opens up a brand-new research direction.
- **Naturalness of spherical positional encoding**: Replacing 2D coordinates with 3D Cartesian coordinates for positional encoding essentially "informs" the network of the ERP image's spherical geometry, which is simple yet highly effective.
- **Substantial performance improvement**: An improvement of 26-42 points in AUC@5° indicates that ERP matching is indeed a severely neglected yet unique problem.

## Limitations & Future Work
- Evaluated only on indoor datasets (Matterport3D, Stanford2D3D); outdoor panoramic scenes have not been tested.
- The computational overhead of Gaussian Process regression is relatively high in large-scale matching.
- Currently only handles horizontal ERP images; tilted ERP and fisheye projections are not addressed.
- The volume of training data is limited (44,700 pairs); larger-scale data may yield further improvements.

## Related Work & Insights
- **vs DKM/RoMa**: The poor performance of these powerful perspective dense matching methods on ERP demonstrates that projection geometry issues cannot be automatically resolved by "brute-force features."
- **vs SphereGlue**: Sparse matching takes the spherical domain into account but lacks density; EDM combines the dual advantages of spherical awareness and dense matching.
- **vs Cubemap Projection Solutions**: Cubemaps require six inference iterations and lose global information; EDM operates directly on ERP, obtaining global dense matches in a single inference run.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first ERP dense matching method, with highly original problem definition and solution.
- Experimental Thoroughness: ⭐⭐⭐ Benchmark datasets are limited, and outdoor scenes as well as ablation analyses are lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation of spherical geometry.
- Value: ⭐⭐⭐⭐ Highly valuable for panoramic vision and indoor 3D reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Feature Selection for Latent Factor Models](feature_selection_for_latent_factor_models.md)
- [\[CVPR 2025\] Improving Transferable Targeted Attacks with Feature Tuning Mixup](improving_transferable_targeted_attacks_with_feature_tuning_mixup.md)
- [\[NeurIPS 2025\] Dense Associative Memory with Epanechnikov Energy](../../NeurIPS2025/others/dense_associative_memory_with_epanechnikov_energy.md)
- [\[CVPR 2026\] HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition](../../CVPR2026/others/hypevpr_exploring_hyperbolic_space_for_perspective_to_equirectangular_visual_pla.md)
- [\[ICML 2025\] Score Matching with Missing Data](../../ICML2025/others/score_matching_with_missing_data.md)

</div>

<!-- RELATED:END -->
