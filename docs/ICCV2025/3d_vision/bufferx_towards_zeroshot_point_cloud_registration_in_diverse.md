---
title: >-
  [Paper Note] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes
description: >-
  [ICCV 2025 (Highlight)][3D Vision][point cloud registration] BUFFER-X is a registration pipeline that determines voxel size and search radii via geometry-adaptive bootstrapping, replaces learned keypoint detectors with FPS, and applies patch-level coordinate normalization. Without any manual parameter tuning, it achieves zero-shot point cloud registration across 11 cross-domain datasets, attaining the best average-rank success rate across indoor/outdoor, multi-sensor, and multi-scene settings.
tags:
  - ICCV 2025 (Highlight)
  - 3D Vision
  - point cloud registration
  - zero-shot generalization
  - multi-scale descriptors
  - adaptive parameters
  - cross-domain robustness
date: 2026-05-08
content_hash: f15ce4000ed59656
---

# BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes

**Conference**: ICCV 2025 (Highlight)
**arXiv**: [2503.07940](https://arxiv.org/abs/2503.07940)
**Code**: [https://github.com/MIT-SPARK/BUFFER-X](https://github.com/MIT-SPARK/BUFFER-X)
**Area**: 3D Vision / Point Cloud Registration
**Keywords**: point cloud registration, zero-shot generalization, multi-scale descriptors, adaptive parameters, cross-domain robustness

## TL;DR
BUFFER-X is a registration pipeline that determines voxel size and search radii via geometry-adaptive bootstrapping, replaces learned keypoint detectors with FPS, and applies patch-level coordinate normalization. Without any manual parameter tuning, it achieves zero-shot point cloud registration across 11 cross-domain datasets, attaining the best average-rank success rate across indoor/outdoor, multi-sensor, and multi-scene settings.

## Background & Motivation
Deep learning-based point cloud registration has achieved strong performance within the same domain, yet faces a critical practical challenge: **poor generalization**. Existing methods (e.g., FCGF, Predator, GeoTransformer, BUFFER) typically require users to manually tune core hyperparameters such as **voxel size** and **search radius** when transferring from the training domain to unseen environments — a practice known as "oracle tuning." For instance, a model trained on indoor RGB-D data (3DMatch, range ~3.5 m) applied directly to outdoor LiDAR data (KITTI, range ~80 m) may crash due to GPU memory overflow from excessive point counts, or fail entirely due to parameter mismatch.

Furthermore, existing benchmarks typically evaluate cross-domain generalization only between 3DMatch (indoor) and KITTI (outdoor), failing to capture the real-world diversity in sensor types (RGB-D vs. various LiDARs), geographic/cultural variation (Europe/Asia/America), and acquisition modes (handheld/vehicle-mounted/robot).

## Core Problem
The authors identify three key factors limiting zero-shot generalization:
1. **Dependence on scene-specific voxel size and search radius**: Optimal parameters vary drastically across datasets (indoor ~0.025 m vs. outdoor ~0.3 m); improper settings cause OOM errors or severe performance degradation.
2. **Out-of-domain fragility of learned keypoint detectors**: On data outside the training domain, learned detectors select unreliable keypoints, causing cascading failures.
3. **Direct use of raw coordinates**: After fitting to the coordinate scale distribution of training data, networks fail catastrophically on test data with vastly different scales (3DMatch max range ~3.5 m vs. KITTI ~80 m).

## Method

### Overall Architecture
The BUFFER-X pipeline consists of three stages:
- **Input**: source point cloud $\mathcal{P}$ and target point cloud $\mathcal{Q}$
- **Stage 1 — Geometric Bootstrapping**: Adaptively determines voxel size $v$ and search radii $r_l, r_m, r_g$ at three scales.
- **Stage 2 — Multi-scale Patch Embedder**: Independently samples keypoints via FPS at each scale and generates Mini-SpinNet descriptors.
- **Stage 3 — Hierarchical Inlier Search**: Performs within-scale matching, then selects global inliers by maximizing cross-scale consistency, followed by RANSAC pose estimation.
- **Output**: rotation matrix $\hat{\boldsymbol{R}}$ and translation vector $\hat{\boldsymbol{t}}$

### Key Designs

1. **Geometric Bootstrapping (Adaptive Parameter Determination)**:

    - **Sphericity-based voxelization**: PCA is applied to a subsampled point cloud, and sphericity $\lambda_3/\lambda_1$ is computed. High sphericity (e.g., RGB-D point clouds with relatively uniform distribution) uses a small coefficient $\kappa_{\text{spheric}}=0.10$ multiplied by the extent $s$ along the minor eigenvector direction; low sphericity (e.g., LiDAR point clouds with disc-like distributions) uses a larger coefficient $\kappa_{\text{disc}}=0.15$. This allows voxel size to automatically adapt to sensor type and environmental scale.
    - **Density-aware radius estimation**: Rather than using fixed radii, the method searches for a radius at which the average fraction of neighboring points relative to the total point count approaches a target threshold $\tau_\xi$. Three scale-specific thresholds are defined: local ($\tau_l=0.005$), middle ($\tau_m=0.02$), and global ($\tau_g=0.05$), with a maximum cutoff radius of $r_{\max}=5.0\text{m}$.

2. **Multi-scale Patch Embedder (Detector-free Design)**:

    - **FPS as a replacement for learned detectors**: Farthest point sampling independently selects 1500 keypoints per scale, entirely avoiding cascading failures caused by learned detectors on out-of-domain data. Experiments show that FPS is competitive with or superior to learned detectors even within the training domain.
    - **Patch-level coordinate normalization**: Neighboring points within radius $r_\xi$ around each keypoint form a patch; coordinates are divided by $r_\xi$ to normalize to $[-1,1]$, eliminating scale dependence. PCA is used within each patch to define a local reference frame (with the $z$-axis set to the eigenvector corresponding to the smallest eigenvalue), avoiding dataset-specific inductive biases.
    - **Mini-SpinNet descriptor**: Based on the lightweight SpinNet variant from BUFFER, producing a $D$-dimensional feature vector $\mathcal{F}$ and a cylindrical coordinate feature map $\mathcal{C}$ of size $H \times W \times D = 7 \times 20 \times 32$. Thanks to coordinate normalization, the entire network requires training at only a **single scale** to generalize to multiple scales at inference.

3. **Hierarchical Inlier Search (Cross-scale Consistency)**:

    - **Within-scale matching**: Mutual nearest-neighbor matching is performed independently at each scale to obtain correspondences $\mathcal{A}_\xi$.
    - **Pairwise transformation estimation**: Exploiting the SO(2)-equivariance of cylindrical coordinate features, a 4D matching cost volume and a 3D cylindrical convolutional network (3DCCN) estimate the yaw rotation $\boldsymbol{R}_{\text{yaw}}$; the full 3D rotation is recovered by combining with the PCA reference axes.
    - **Cross-scale consensus maximization**: Candidate transformations and point pairs from all scales are aggregated, and the transformation maximizing the inlier set cardinality is selected. This is formulated as a consensus maximization problem: $\max |\mathcal{I}|$ s.t. $\|\boldsymbol{R}\boldsymbol{p}_n + \boldsymbol{t} - \boldsymbol{q}_n\|_2 < \epsilon$

### Loss & Training
- **Two-stage training** (simpler than BUFFER's four-stage procedure): The Mini-SpinNet feature discriminability is first trained via contrastive learning; the 3DCCN yaw offset $d$ prediction accuracy is then trained with a Huber loss.
- **Huber Loss**: $\mathcal{L}_d = \frac{1}{N_d}\sum_{\gamma}\rho_{\text{Huber}}(d_\gamma - d_\gamma^*)$, with truncation threshold $\delta=1.0$ for robustness to outliers.
- **Patch distribution augmentation**: During training, the search radius is uniformly sampled from $[\frac{2}{3}r, \frac{4}{3}r]$, exposing the network to more diverse patch patterns.
- Training is conducted exclusively on **3DMatch**, enabling zero-shot generalization to all 11 test datasets.

## Key Experimental Results

**Zero-shot generalization (trained on 3DMatch, success rate %) — Core results from Table 1:**

| Dataset | BUFFER-X | BUFFER+oracle | GeoT+oracle+scale | Predator+oracle+scale | FCGF+oracle+scale |
|--------|----------|---------------|--------------------|-----------------------|-------------------|
| 3DMatch | **95.58** | 92.90 | 92.00 | 90.60 | 88.18 |
| 3DLoMatch | 74.18 | 71.80 | 75.00 | 62.40 | 40.09 |
| ScanNet++i | **94.99** | 93.01 | 92.72 | 75.94 | 85.87 |
| ScanNet++F | **99.90** | 94.69 | 97.02 | 86.01 | 88.69 |
| TIERS | **93.45** | 88.96 | 92.99 | 75.74 | 80.11 |
| KITTI | **99.82** | 99.46 | 92.43 | 77.29 | 94.41 |
| WOD | **100.00** | 100.00 | 89.23 | 86.92 | 97.69 |
| KAIST | **99.15** | 97.24 | 91.86 | 87.09 | 93.55 |
| MIT | **97.39** | 95.65 | 95.65 | 79.56 | 93.04 |
| ETH | **99.72** | 99.30 | 71.53 | 54.42 | 55.53 |
| Oxford | **99.67** | 99.00 | 97.01 | 93.68 | 95.68 |
| **Avg. Rank** | **1.55** | 3.82 | 6.27 | 11.82 | 9.55 |

Key takeaway: BUFFER-X **requires no oracle tuning whatsoever**, yet outperforms all baselines under their best oracle-tuned + scale-aligned configurations.

**In-domain performance (KITTI, Table 2)**: RTE 7.74 cm, RRE 0.27°, success rate 99.82%, on par with state of the art.

**Multi-scale ablation (3DMatch, Table 3)**:

| Scale combination (L/M/G) | RTE (cm) | RRE (°) | SR (%) | FMR (%) |
|----------------------------|----------|---------|--------|---------|
| Local only | 6.57 | 2.15 | 84.06 | 5.61 |
| Middle only | 5.87 | 1.85 | 93.38 | 5.47 |
| Global only | 6.06 | 1.91 | 93.57 | 5.49 |
| L+M+G (all) | **5.78** | **1.79** | **95.58** | **1.81** |

### Ablation Study
- **Geometric bootstrapping is critical**: Fig. 6 shows that voxel size has a large impact on BUFFER performance (with large variation in optimal values across datasets), whereas BUFFER-X's adaptive mechanism automatically identifies reasonable values.
- **FPS ≥ learned detectors**: Fig. 7 shows that FPS is competitive with or superior to BUFFER's learned keypoint detector both in-domain (3DMatch/3DLoMatch) and out-of-domain.
- **Multi-scale complementarity**: Correspondences from the three scales are mutually complementary; the full multi-scale combination outperforms any single-scale or two-scale configuration.
- **Computational trade-off**: Multi-scale processing improves accuracy at the cost of increased computation (Fig. 8); users may select the number of scales according to their requirements.

## Highlights & Insights
- **Genuine zero-shot generalization**: For the first time, a model trained solely on 3DMatch achieves state-of-the-art registration performance across 11 datasets spanning indoor/outdoor environments, diverse sensors, and multiple geographic regions — without any manual parameter tuning.
- **Rigorous problem analysis**: The three key factors limiting generalization (voxel size dependency, fragility of learned detectors, coordinate scale mismatch) are clearly identified and individually validated with thorough experimental support.
- **Minimalist yet effective design philosophy**: Replacing the learned detector with FPS yields equal or better performance; training at a single scale generalizes to multi-scale inference — a compelling "less is more" result.
- **Benchmark contribution**: A comprehensive generalization benchmark covering 11 datasets is established, encompassing RGB-D and multiple LiDAR modalities, handheld/vehicle/robot acquisition, and diverse geographic regions across Europe, Asia, and America, filling a notable gap in the community.
- **Sphericity-adaptive voxelization** represents an elegant use of geometric priors, leveraging PCA eigenvalues to distinguish LiDAR (disc-like) from RGB-D (sphere-like) point distributions.

## Limitations & Future Work
- **Suboptimal performance in low-overlap scenarios**: The 74.18% success rate on 3DLoMatch (10–30% overlap) is notably below GeoTransformer's 75%. Consensus maximization tends to select the correspondence set with the largest cardinality, but in partial-overlap scenarios, the largest cardinality set may not correspond to the true global optimum — a problem the authors term "global optimum ambiguity."
- **Inference speed**: Multi-scale processing (especially descriptor generation across three scales) introduces significant computational overhead; the authors note plans to accelerate inference in future work.
- **Single training dataset**: Training is limited to 3DMatch or KITTI; joint multi-dataset training has not been explored.
- **Rigid registration only**: Non-rigid deformation scenarios are not addressed.
- **Global optimum ambiguity** currently has no mathematical solution and requires prior knowledge — itself a valuable direction for future research.

## Related Work & Insights
- **vs. BUFFER**: BUFFER-X is a direct extension of BUFFER. Although BUFFER's patch-level normalization offers reasonable generalization potential, it still requires manual tuning of voxel size and search radius (oracle tuning). BUFFER-X automates this via geometric bootstrapping, replaces BUFFER's learned keypoint detector with FPS, and simplifies training from four stages to two.
- **vs. GeoTransformer**: GeoTransformer performs well in-domain (3DMatch 92%), but struggles in many cross-domain settings even with oracle tuning and scale alignment (e.g., only 71.53% on ETH). The root cause is scale dependence from direct use of raw coordinates.
- **vs. Predator**: Predator is similarly constrained by coordinate scale issues (scale alignment provides noticeable but insufficient improvement) and suffers from high memory consumption, causing OOM errors on some datasets.
- **vs. KISS-Matcher and traditional methods**: Handcrafted descriptors (e.g., FPFH) combined with traditional pipelines offer a degree of generalization, but overall accuracy is lower than learned methods.

- **General methodology for adaptive parameter design**: The PCA-based sphericity discrimination in geometric bootstrapping is transferable to other 3D tasks for adaptive hyperparameter setting (e.g., anchor sizes in 3D detection, ellipsoid initialization in 3DGS).
- **Insight from removing learned modules**: In domain generalization settings, learned modules can become a liability — a broadly applicable lesson for zero-shot and cross-domain method design across 3D vision tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Individual components are not novel in isolation (PCA, FPS, and SpinNet are established techniques), but the problem analysis is insightful, the combination is elegant, and the systematic achievement of zero-shot registration is unprecedented.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The 11-dataset benchmark is exceptionally comprehensive; ablation studies cover every design decision, supplemented by cross-validation with KITTI-trained models and detailed trade-off analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain is clear (problem analysis → key observations → corresponding solutions); figures and tables are carefully designed; the appendix is well-developed (dataset selection rationale, global optimum ambiguity analysis, etc.).
- Value: ⭐⭐⭐⭐⭐ The Highlight designation is well deserved — zero-shot cross-domain registration has strong practical utility, and the benchmark and open-sourced code represent a significant contribution to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TurboReg: TurboClique for Robust and Efficient Point Cloud Registration](turboreg_turboclique_for_robust_and_efficient_point_cloud_registration.md)
- [\[ICCV 2025\] ZeroStereo: Zero-shot Stereo Matching from Single Images](zerostereo_zero-shot_stereo_matching_from_single_images.md)
- [\[ICCV 2025\] RobuSTereo: Robust Zero-Shot Stereo Matching under Adverse Weather](robustereo_robust_zero-shot_stereo_matching_under_adverse_weather.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[ICCV 2025\] Blended Point Cloud Diffusion for Localized Text-guided Shape Editing](blended_point_cloud_diffusion_for_localized_textguided_shape.md)

</div>

<!-- RELATED:END -->
