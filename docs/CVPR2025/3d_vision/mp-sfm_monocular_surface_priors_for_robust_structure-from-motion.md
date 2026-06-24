---
title: >-
  [Paper Note] MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion
description: >-
  [CVPR 2025][3D Vision][structure-from-motion] Integrates monocular depth and normal priors tightly into classical incremental SfM. Through uncertainty propagation and alternating optimization, it breaks the fundamental limitation of three-view tracks, achieving reliable 3D reconstruction from only two-view tracks for the first time, and significantly outperforms all existing methods in extremely low-overlap and low-parallax scenarios.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "structure-from-motion"
  - "monocular depth"
  - "surface normals"
  - "COLMAP"
  - "low overlap"
  - "symmetry rejection"
date: 2026-05-08
content_hash: 86f7a81ad23a5674
---

# MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion

**Conference**: CVPR 2025  
**arXiv**: [2504.20040](https://arxiv.org/abs/2504.20040)  
**Code**: [github.com/cvg/mpsfm](https://github.com/cvg/mpsfm)  
**Area**: 3D Vision  
**Keywords**: structure-from-motion, monocular depth, surface normals, COLMAP, low overlap, symmetry rejection

## TL;DR

Integrates monocular depth and normal priors tightly into classical incremental SfM. Through uncertainty propagation and alternating optimization, it breaks the fundamental limitation of three-view tracks, achieving reliable 3D reconstruction from only two-view tracks for the first time, and significantly outperforms all existing methods in extremely low-overlap and low-parallax scenarios.

## Background & Motivation

**Background**: After decades of development in SfM, systems like COLMAP perform excellently under normal conditions. However, they are still prone to failure under extreme view changes (low overlap, low parallax, and highly symmetric scenes).

**Limitations of Prior Work**: (1) All state-of-the-art systems such as COLMAP/GLOMAP fundamentally require three-view overlap and tracks to ensure cross-view consistent 3D reconstruction scale; (2) non-professional users find it difficult to guarantee sufficient overlap and viewpoint variation during capture; (3) symmetric structures in architectural indoor scenes easily lead to false matches and catastrophic reconstruction failures.

**Key Challenge**: The requirement of three-view tracks in classical SfM is a fundamental mathematical constraint (used to determine cross-view consistent scale); simply improving matching cannot resolve the limitations of subsequent reconstruction algorithms.

**Goal**: To enable reliable SfM work under extremely low overlap (even zero three-view overlap), allowing reconstructions from casual captures by non-professional users.

**Key Insight**: Leveraging recent advances in monocular depth estimation to provide scale priors, lifting two-view matched points to 3D via depth to constrain the scale, thereby breaking the rigid constraint of three-view tracks.

## Method

### Overall Architecture

Deep modifications are made based on the COLMAP incremental SfM framework:
1. Input: Image set + intrinsics + monocular depth/normals/confidence maps + sparse/dense feature matches
2. Two-view initialization: Prioritizes relative pose; uses depth-lifting + PnP to estimate absolute pose under low parallax
3. Incremental registration: Leverages depth-lifted single-view 3D points (without requiring three-view tracks)
4. Alternating optimization: Normal integration + depth constraints ↔ BA + depth regularization
5. Depth consistency check: Rejects incorrect registrations caused by symmetry/etc.

### Key Designs

#### 1. Two-view initialization and single-view point lifting

- **Low parallax handling**: When a pair with sufficient parallax cannot be found, monocular depth is used to lift feature points to 3D, and the absolute pose is estimated via PnP, completely bypassing the limitation of low parallax.
- **Initial 3D points**: Low-parallax inliers are reconstructed via depth lifting, and others via triangulation, complementing each other.
- **Depth alignment**: The monocular depth is aligned to the multi-view scale using median statistics to compute a scale factor $D_i^* = D_i \cdot \text{median}(\hat{D}_i(X_k) / D_i(x_j))$.

#### 2. Joint refinement via alternating optimization

**Global objective function**: $\arg\min_{\mathcal{P}, \mathcal{X}, \mathcal{D}^*} C_{BA} + C_{reg} + C_{int}$

- **$C_{BA}$**: Standard BA reprojection error (Smooth-L1 loss) using Mahalanobis distance.
- **$C_{reg}$**: Depth regularization, which penalizes deviation of 3D point depth from refined depth maps (Cauchy robust loss).
- **$C_{int}$**: Depth integration, combining monocular depth priors and **bilateral normal integration**, weighted by uncertainty.

**Alternating strategy**: (1) Fix 3D points, optimize refined depth for each image (GPU); (2) Fix depth maps, jointly optimize poses and 3D points (CPU, Ceres). Multiple alternating rounds maintain a linear amortized runtime.

#### 3. Depth consistency check for symmetry rejection

- Projects the depth map of the newly registered image to overlapping images, checking the ratio of forward-backward inconsistent pixels.
- Rejects the registration if it exceeds threshold $\hat{\beta}$ — effectively identifying incorrect poses caused by symmetry.
- Performed during both the registration phase and final stage.

### Loss & Training

Three optimization objectives:

$$C_{BA} = \sum_{i,j,k} \rho_{BA}(\|\pi(K_i, P_i, X_k) - x_j\|^2_{\Sigma_{x_j}})$$

$$C_{reg} = \sum_{i,j,k} \rho_{reg}(\|\hat{D}_i(X_k) - D_i^*(x_j)\|^2)$$

$$C_{int} = \sum_{i,u,v} [\rho_{prior}(\|D_i^* - D_i\|^2_{\Sigma_{D_i}}) + \rho_{int}(\|N_i - \Delta D_i^*\|^2_{\Sigma_{N_i}})]$$

All terms use uncertainty-weighted and robust loss functions, ensuring robustness against noisy priors.

## Key Experimental Results

### Main Results: ETH3D Low-Overlap SfM (Table 1, Pose AUC@1/5/20°)

| Matching | Method | Zero Overlap | <5% Overlap | <10% Overlap | All Images |
|---------|------|--------|---------|----------|---------|
| SP+LG | COLMAP | 7.9/12.7/14.6 | 12.0/19.8/23.2 | 44.9/60.3/65.0 | 67.2/80.6/83.9 |
| SP+LG | GLOMAP | 8.4/15.8/22.5 | 12.1/25.3/35.7 | 50.1/66.7/71.8 | 67.5/78.5/82.3 |
| SP+LG | **Ours** | **27.3/55.9/71.8** | **30.0/56.4/70.1** | **57.0/79.1/86.0** | **74.3/88.3/92.0** |
| MASt3R | M-SfM | 20.1/39.7/52.2 | 19.8/37.3/48.1 | 31.4/50.4/59.2 | 50.5/67.9/74.1 |
| MASt3R | **Ours** | **34.9/67.2/81.7** | **37.7/67.8/80.6** | **55.5/79.3/86.6** | **70.3/88.2/93.6** |

### SMERF Dataset (Table 2)

| Matching | Method | Lowest Overlap | High Overlap |
|------|------|---------|--------|
| SP+LG | COLMAP | 2.2/4.2/4.9 | 42.9/55.4/59.5 |
| SP+LG | **Ours** | **9.2/41.0/69.8** | **47.3/79.3/90.6** |

### Ablation Study (Table 5)

| Configuration | ETH3D Lowest Overlap | ETH3D All | SMERF Lowest | SMERF High |
|------|-------------|-----------|-----------|---------|
| Full | 27.3/55.9/71.8 | 74.3/88.3/92.0 | 9.2/41.0/69.8 | 47.3/79.3/90.6 |
| No depth refine | 26.8/55.2/69.9 | 71.9/87.6/91.7 | 8.4/37.6/66.7 | 32.2/63.6/82.0 |
| No depth reg. | 23.6/49.6/65.9 | 75.1/88.7/92.2 | 5.7/21.0/45.9 | 45.5/64.1/73.0 |
| No lifting | 10.6/16.1/18.7 | 74.1/87.2/90.6 | 1.0/1.7/2.1 | 51.9/69.6/75.2 |

### Key Findings

1. **Qualitative Leap in Low-Overlap Scenes**: Under zero three-view overlap, Ours achieves AUC@20°=71.8% vs. COLMAP's 14.6% — a 5-fold improvement.
2. **Maintains Performance in High-Overlap Scenes**: On all images, AUC@20°=92.0% vs. COLMAP's 83.9% — showing significant improvement even under standard conditions.
3. **Point Lifting is Core**: Eliminating single-view point lifting (no lifting) leads to a performance collapse in low overlap scenarios (71.8 → 18.7), proving it is key to breaking the three-view track limit.
4. **Uncertainty is Crucial**: With depth uncertainty vs. without uncertainty, SMERF lowest overlap AUC@20° drops from 69.8 to 65.8.
5. **Robustness to Depth Models**: Even with a weaker depth model (DepthAnything-v2), respectable results are achieved (ETH3D All AUC@20°=89.9%), though Metric3D-v2 with uncertainty yields the best performance.

## Highlights & Insights

1. **Fundamental Breakthrough**: For the first time, it breaks the rigid requirement of classical SfM on three-view tracks — one of the most fundamental limitations in the SfM field for over a decade.
2. **Elegant Fusion of Classical & Learning Approaches**: Instead of replacing the classical SfM pipeline, it injects monocular priors into each key step, retaining the generality and scalability of incremental SfM.
3. **Rigorous Uncertainty Propagation**: Full-link uncertainty propagation from monocular priors to depth integration and BA ensures robustness to noisy priors.
4. **Symmetry Rejection**: Dense depth consistency checks provide an effective defense mechanism for symmetry misjudgments in SfM for the first time.
5. **Practical Value**: Enables reliable reconstruction from casual captures by non-professional users — significantly lowering the barrier to entry for SfM.

## Limitations & Future Work

1. It relies on high-quality monocular depth estimation (Metric3D-v2); the quality of the depth model directly affects performance.
2. Alternating optimization increases computation time, making it slower than pure COLMAP.
3. In object-centric scenes (e.g., Tanks & Temples), AUC@1° is weaker than MASt3R-SfM due to a lack of foreground matching.
4. It still assumes known camera intrinsics.

## Related Work & Insights

- **COLMAP**: The gold standard of classical incremental SfM — this work makes the minimal necessary modifications to it to achieve maximal benefits.
- **MASt3R-SfM**: End-to-end learned SfM — excels in specific scenes but lacks the generalizability of classical methods.
- **Metric3D-v2**: Key monocular depth prior — its uncertainty estimation is critical to this method.
- **Insights**: The tight coupling of monocular priors and classical optimization is currently the most effective paradigm for enhancing SfM; future progress in depth models will directly benefit this method.

## Rating

⭐⭐⭐⭐⭐ (9/10)

- **Novelty**: ⭐⭐⭐⭐⭐ — Solves one of the most fundamental limitations in the SfM field, representing a prominent academic contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Detailed ablations, multiple datasets, and diverse matchers yield highly convincing results.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clearly defined problem and mathematically rigorous derivations.
- **Value**: ⭐⭐⭐⭐⭐ — Open-source code directly lowers the entry barrier for SfM, holding significant value for fields like gaming, AR, and robotics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Light3R-SfM: Towards Feed-forward Structure-from-Motion](light3r-sfm_towards_feed-forward_structure-from-motion.md)
- [\[CVPR 2025\] Dense-SfM: Structure from Motion with Dense Consistent Matching](dense-sfm_structure_from_motion_with_dense_consistent_matching.md)
- [\[CVPR 2025\] MegaSaM: Accurate, Fast and Robust Structure and Motion from Casual Dynamic Videos](megasam_accurate_fast_and_robust_structure_and_motion_from_casual_dynamic_videos.md)
- [\[CVPR 2025\] SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video](splinegs_robust_motion-adaptive_spline_for_real-time_dynamic_3d_gaussians_from_m.md)
- [\[CVPR 2025\] ColabSfM: Collaborative Structure-from-Motion by Point Cloud Registration](colabsfm_collaborative_structure-from-motion_by_point_cloud_registration.md)

</div>

<!-- RELATED:END -->
