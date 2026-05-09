---
title: >-
  [Paper Note] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes
description: >-
  [ICCV 2025][3D Vision][point cloud registration] BUFFER-X is proposed as a zero-shot point cloud registration method that requires no manual parameter tuning. Through adaptive voxel size/search radius estimation, FPS as a replacement for learned keypoint detectors, and patch-level coordinate normalization, it achieves out-of-the-box cross-domain generalization across 11 datasets.
tags:
  - ICCV 2025
  - 3D Vision
  - point cloud registration
  - zero-shot generalization
  - multi-scale descriptors
  - geometric bootstrapping
  - cross-domain generalization
date: 2026-05-08
content_hash: cd4ab82ccf2cd5d9
---

# BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes

**Conference**: ICCV 2025
**arXiv**: [2503.07940](https://arxiv.org/abs/2503.07940)
**Code**: [https://github.com/MIT-SPARK/BUFFER-X](https://github.com/MIT-SPARK/BUFFER-X)
**Area**: 3D Vision
**Keywords**: point cloud registration, zero-shot generalization, multi-scale descriptors, geometric bootstrapping, cross-domain generalization

## TL;DR

BUFFER-X is proposed as a zero-shot point cloud registration method that requires no manual parameter tuning. Through adaptive voxel size/search radius estimation, FPS as a replacement for learned keypoint detectors, and patch-level coordinate normalization, it achieves out-of-the-box cross-domain generalization across 11 datasets.

## Background & Motivation

Although deep learning-based point cloud registration methods perform well within their training domain, they suffer severe degradation in cross-domain scenarios. The authors identify three core bottlenecks:

**Voxel size and search radius are scene-dependent**: Optimal voxel size and search radius vary dramatically across datasets (indoor 0.025 m vs. outdoor 0.3 m), making manual tuning impractical.

**Raw coordinate input induces scale dependency**: Directly using xyz coordinates causes the model to overfit to the training distribution, leading to catastrophic failure across scales (e.g., 3DMatch max range 3.5 m vs. KITTI 80 m).

**Learned keypoint detectors generalize poorly**: Keypoint detection failure on out-of-distribution data triggers cascading failures.

Even methods with the patch normalization advantage of BUFFER still require users to manually specify optimal parameters (referred to as oracle tuning), preventing truly zero-shot inference.

## Method

### Overall Architecture

BUFFER-X consists of three core steps: (1) geometric bootstrapping to determine voxel size and multi-scale search radii; (2) a multi-scale patch embedder to generate descriptors; (3) hierarchical inlier search for cross-scale consistent matching.

### Key Designs

1. **Geometric Bootstrapping**:

    - **Sphericity-adaptive voxelization**: PCA is applied to sampled points to obtain eigenvalues $\lambda_1 \geq \lambda_2 \geq \lambda_3$, and sphericity $\lambda_3/\lambda_1$ is computed. LiDAR point clouds are disk-like (low sphericity) and require larger voxels; RGB-D clouds are more isotropic (high sphericity) and use smaller voxels. The formula is $v = \kappa \sqrt{s}$, where $s$ is the point distribution span along the minimum eigenvector direction and $\kappa$ is selected based on a sphericity threshold $\tau_v$.
    - **Density-aware radius estimation**: At local/middle/global scales, the radius $r_\xi$ is found iteratively such that the average number of neighborhood points satisfies a preset threshold $\tau_\xi$, avoiding misfit from fixed radii under varying point densities.

2. **Multi-scale Patch Embedder**:

    - **FPS (Farthest Point Sampling) replaces learned keypoint detectors**, independently sampling distinct keypoint sets at each scale (rather than extracting multi-scale features for the same point).
    - Mini-SpinNet generates descriptors; the key operation is normalizing patch-internal coordinates to $[-1, 1]$ (by dividing by radius $r_\xi$), eliminating scale dependency.
    - PCA defines local reference axes ($z$-axis taken as the eigenvector corresponding to the smallest eigenvalue), removing dataset-specific inductive bias.

3. **Hierarchical Inlier Search**:

    - **Intra-scale matching**: Mutual nearest-neighbor matching within each scale produces initial correspondences $\mathcal{A}_\xi$.
    - **Pairwise transformation estimation**: Leveraging the SO(2) equivariance of cylindrical coordinate features, relative rotation is estimated via circular cross-correlation (4D matching cost volume + 3DCCN), followed by full 3D rotation recovery via the Rodrigues formula.
    - **Cross-scale consensus maximization**: Candidate transformations and correspondence pairs from all scales are pooled, and a globally consistent inlier set $\mathcal{I}$ is selected via consensus maximization (Eq. 9), which is then passed to RANSAC for pose estimation.

### Loss & Training

- Two-stage training (simpler than BUFFER's four stages): Mini-SpinNet descriptor discriminability is first trained with contrastive learning, followed by Huber loss training for rotation offset $d$ estimation.
- Huber loss balances robustness to outliers and sensitivity to small errors: $\mathcal{L}_d = \frac{1}{N_d}\sum \rho_{\text{Huber}}(d_\gamma - d^*_\gamma)$
- **Patch distribution augmentation**: During training, radii are uniformly sampled from $[2r/3, 4r/3]$ to increase diversity of intra-patch point distributions.
- Only **single-scale training** is required (leveraging normalization properties); separate multi-scale training is unnecessary.

## Key Experimental Results

### Main Results

Zero-shot registration recall (%) across 11 datasets, trained on 3DMatch only:

| Method | 3DMatch | 3DLoMatch | ScanNet++F | TIERS | KITTI | WOD | KAIST | MIT | ETH | Oxford | Avg. Rank |
|------|---------|-----------|------------|-------|-------|-----|-------|-----|-----|--------|---------|
| BUFFER (oracle) | 92.90 | 71.80 | 94.69 | 88.96 | 99.46 | 100.0 | 97.24 | 95.65 | 99.30 | 99.00 | 3.82 |
| GeoTransformer (oracle+scale) | 92.00 | 75.00 | 97.02 | 92.99 | 92.43 | 89.23 | 91.86 | 95.65 | 71.53 | 97.01 | 6.27 |
| **BUFFER-X (Ours)** | **95.58** | **74.18** | **99.90** | **93.45** | **99.82** | **100.0** | **99.15** | **97.39** | **99.72** | **99.67** | **1.55** |

BUFFER-X achieves the best average rank without any oracle tuning or scale alignment.

### Ablation Study

Effect of multi-scale combinations on 3DMatch performance:

| Local | Middle | Global | RTE (cm)↓ | RRE (°)↓ | Recall (%)↑ | Speed (Hz)↑ |
|-------|--------|--------|-----------|----------|-----------|----------|
| ✓ | | | 6.57 | 2.15 | 84.06 | 5.61 |
| | ✓ | | 5.87 | 1.85 | 93.38 | 5.47 |
| ✓ | ✓ | ✓ | 5.78 | 1.79 | **95.58** | 1.81 |

FPS vs. learned detectors: FPS is not only more robust out-of-domain but also performs comparably or better within the training domain (3DMatch/3DLoMatch).

### Key Findings

- Most deep learning methods suffer a drastic drop in cross-domain recall without oracle parameters (e.g., FCGF on KITTI drops from 98.92% to 0%).
- Patch-level coordinate normalization is a critical factor for generalization (Predator improves significantly with scale alignment).
- Multi-scale complementarity further boosts recall, but at a precision–speed trade-off (three scales: 1.81 Hz vs. single scale: 5.47 Hz).

## Highlights & Insights

- **Thorough problem analysis**: Three factors limiting zero-shot registration are clearly identified, each supported by theoretical and experimental evidence.
- **Elegant adaptive mechanisms**: PCA-based sphericity distinguishes LiDAR from RGB-D; density-aware radius estimation requires no human intervention.
- **Significant benchmark contribution**: A generalization benchmark spanning 11 datasets is constructed, covering diverse scene scales, sensor types, and geographic/cultural variety.
- Replacing complex learned detectors with simple FPS yields better cross-domain performance — demonstrating that **simplicity and reliability outweigh in-domain optimality** under domain shift.

## Limitations & Future Work

- Performance is relatively weaker on 3DLoMatch (only 10–30% overlap), as consensus maximization tends to select the largest-cardinality inlier set, which may not be the true optimum under low overlap (global optimality ambiguity).
- Three-scale inference runs at only 1.81 Hz, insufficient for real-time applications.
- Training is limited to 3DMatch or KITTI; the impact of broader training data remains unexplored.
- Solid-state LiDARs with extremely sparse point clouds (e.g., Livox Horizon/Avia) remain challenging.

## Related Work & Insights

- The patch normalization concept from BUFFER (CVPR 2023) is a key foundation; this work eliminates its manual tuning requirement.
- SpinNet's local coordinate normalization to $[-1,1]$ is central to achieving data-agnostic representations.
- Traditional methods such as KISS-Matcher remain competitive under oracle settings, affirming the continued value of feature-based approaches.
- The global optimality ambiguity analysis for low-overlap scenarios warrants attention in future research.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Systematically addresses three key factors with cleverly designed adaptive mechanisms.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 11 datasets with detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Problem analysis is clear and figures are informative, though some formulations are verbose.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a genuine practical pain point (zero-shot generalization) with a lasting benchmark contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TurboReg: TurboClique for Robust and Efficient Point Cloud Registration](turboreg_turboclique_for_robust_and_efficient_point_cloud_registration.md)
- [\[ICCV 2025\] RobuSTereo: Robust Zero-Shot Stereo Matching under Adverse Weather](robustereo_robust_zero-shot_stereo_matching_under_adverse_weather.md)
- [\[ICCV 2025\] Towards More Diverse and Challenging Pre-training for Point Cloud Learning: Self-Supervised Cross Reconstruction with Decoupled Views](towards_more_diverse_and_challenging_pre-training_for_point_cloud_learning_self-.md)
- [\[ICCV 2025\] ZeroStereo: Zero-shot Stereo Matching from Single Images](zerostereo_zero-shot_stereo_matching_from_single_images.md)
- [\[ICCV 2025\] Zero-Shot Inexact CAD Model Alignment from a Single Image](zero-shot_inexact_cad_model_alignment_from_a_single_image.md)

</div>

<!-- RELATED:END -->
