---
title: >-
  [Paper Note] Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty
description: >-
  [CVPR2026][3D Vision][3D Gaussian Splatting] This paper proposes UGS-Loc, a framework that jointly models pose prior uncertainty and geometric uncertainty via Monte Carlo pose sampling and Fisher information-guided PnP optimization, achieving significantly improved robustness in camera pose refinement within 3DGS scenes without requiring retraining.
tags:
  - CVPR2026
  - 3D Vision
  - 3D Gaussian Splatting
  - visual localization
  - pose refinement
  - Monte Carlo sampling
  - Fisher information
  - uncertainty modeling
date: 2026-05-08
content_hash: 3a55c1d931c4de4a
---

# Rethinking Pose Refinement in 3D Gaussian Splatting under Pose Prior and Geometric Uncertainty

**Conference**: CVPR2026
**arXiv**: [2603.16538](https://arxiv.org/abs/2603.16538)
**Code**: [Project Page](https://arxiv.org/abs/2603.16538) (code available)
**Area**: 3D Vision
**Keywords**: 3D Gaussian Splatting, visual localization, pose refinement, Monte Carlo sampling, Fisher information, uncertainty modeling

## TL;DR

This paper proposes UGS-Loc, a framework that jointly models pose prior uncertainty and geometric uncertainty via Monte Carlo pose sampling and Fisher information-guided PnP optimization, achieving significantly improved robustness in camera pose refinement within 3DGS scenes without requiring retraining.

## Background & Motivation

- **Background**: 3D Gaussian Splatting has become a powerful scene representation for visual localization, with render-and-compare pose refinement methods achieving state-of-the-art accuracy.
- **Limitations of Prior Work**: Existing methods rely on a single deterministic pose estimate (from APR/SCR); when the initial pose has large errors or severe occlusion, the alignment quality between rendered views and query images degrades sharply. Additionally, the ellipsoidal primitives in 3DGS only approximate geometry, and depth rendered from sparse training viewpoints or regions contaminated by dynamic objects is unreliable — yet existing methods treat all depth values equally.
- **Key Challenge**: Unreliable depth is used to lift 2D–2D correspondences to 2D–3D correspondences, injecting erroneous geometric information directly into the PnP solver and causing unstable pose estimation. Methods such as GS-CPR are highly sensitive to the initial pose, as illustrated in Figure 2.
- **Goal**: To develop a training-free, plug-and-play, uncertainty-aware refinement scheme suitable for AR/VR, autonomous driving, and robotics applications that demand robust pose estimation.

## Method

### Overall Architecture (UGS-Loc)

UGS-Loc comprises two core modules: (1) **Monte Carlo Refinement** to address pose prior uncertainty, and (2) **Fisher information-guided PnP optimization** to address geometric uncertainty. The entire framework requires no retraining or additional supervision and completes refinement in only 2 iterations with 8 particles.

### Monte Carlo Pose Refinement

- The pose prior is represented as a weighted particle set $\mathcal{P}=\{(\mathbf{T}^{(m)}, w^{(m)})\}_{m=1}^{M}$, where each particle $\mathbf{T}^{(m)} \in SE(3)$.
- Local optimization replaces the random perturbation prediction step of conventional MCL, guiding each particle toward nearby modes of the likelihood distribution and substantially reducing the required number of particles.
- Importance weights incorporate two metrics: matching confidence $S_m$ and geometric uncertainty $U_m$:

$$w^{(m)} = \frac{\sum_i S_m(r_i) \cdot (1 - U_m(r_i))}{\sum_j \sum_i S_j(r_i) \cdot (1 - U_j(r_i))}$$

- The final pose is obtained by selecting the highest-weight particle via importance resampling, or by weighted averaging.

### Fisher Information-Guided Geometric Uncertainty

- Fisher information is extended to anchor-based GS (Scaffold-GS), parameterized by anchor features and local Gaussian offsets.
- The diagonal Hessian under a Laplace approximation enables efficient computation:

$$\mathrm{H}'' \simeq \mathrm{diag}((\nabla_\theta f)^\top (\nabla_\theta f)) + \lambda I$$

- Fisher information is aggregated across all training views into a global matrix $\mathrm{G}$, then projected into a per-pixel 2D uncertainty map via the 3DGS rendering equation.
- RANSAC sampling weights are defined as $s_i = e^{-\beta \bar{U}(r_i)} + \epsilon$, biasing sampling toward geometrically reliable regions. Consensus scoring also employs this weighting, and the final pose is selected by maximizing the weighted consensus $\sum s_i$.

### Loss & Training

- No explicit loss function is used for training — the framework operates entirely as an inference-time refinement pipeline.
- PnP is solved via EPnP with uncertainty-weighted RANSAC.
- First iteration: uniform sampling with translation perturbation of 10 cm and rotation perturbation of 0.01°; subsequent iterations: 1 cm / 0.01°.

## Key Experimental Results

### Indoor Benchmark (7Scenes)

| Method | Chess | Fire | Heads | Office | Pumpkin | RedKitchen | Stairs | Avg (cm/°) |
|--------|-------|------|-------|--------|---------|------------|--------|------------|
| ACE + GS-CPR | 0.5/0.15 | 0.6/0.25 | 0.4/0.28 | 0.9/0.26 | 1.0/0.23 | 0.7/0.17 | 1.4/0.42 | 0.8/0.25 |
| ACE + UGS-Loc | **0.37/0.12** | **0.47/0.20** | **0.36/0.25** | **0.77/0.22** | **0.79/0.18** | **0.58/0.15** | **1.11/0.33** | **0.64/0.21** |

- ACE + UGS-Loc reduces average error by approximately 20% compared to ACE + GS-CPR.
- Accuracy reaches 95.6% under the strict threshold [2 cm, 2°] (vs. 93.1% for GS-CPR and its iterative variant GS-CPR²).

### Outdoor Benchmark (Cambridge Landmarks)

| Method | Kings | Hospital | Shop | Church | Avg (cm/°) |
|--------|-------|----------|------|--------|------------|
| DFNet + GS-CPR | 23/0.32 | 42/0.74 | 10/0.36 | 27/0.62 | 26/0.51 |
| DFNet + UGS-Loc | **18.7/0.19** | **14.5/0.29** | **3.9/0.15** | **5.5/0.17** | **10.7/0.20** |
| ACE + GS-CPR | 20/0.29 | 21/0.40 | 5/0.24 | 13/0.40 | 15/0.33 |
| ACE + UGS-Loc | **17.8/0.18** | **13.8/0.30** | **4.2/0.16** | **6.3/0.20** | **10.5/0.21** |

- UGS-Loc reduces the median translation error of GS-CPR by approximately 30% on Cambridge Landmarks.
- With the weaker DFNet prior, UGS-Loc even surpasses the ACE + GS-CPR combination after refinement.

### Ablation Study

- **Number of particles**: Increasing from 2 to 16 particles monotonically reduces the Cambridge average error from 11.8/0.24 to 10.3/0.20 (DFNet prior).
- **Iterative refinement**: Simple iteration of GS-CPR saturates quickly after the first pass, whereas UGS-Loc continues to converge to lower error within 2 iterations.
- **Matching module**: MASt3r slightly outperforms SuperPoint+LightGlue (11/0.22 vs. 13/0.26), but uncertainty-aware refinement enables even lightweight matchers to approach high accuracy.
- **Runtime**: The standard configuration (m=8) achieves end-to-end inference at 1.1 s/iteration, substantially faster than MCLoc at 2.4 s/query.

## Highlights & Insights

- **Dual uncertainty modeling**: UGS-Loc is the first work to jointly address both pose prior uncertainty and geometric uncertainty in 3DGS-based pose refinement.
- **Training-free**: The entire framework operates at inference time and is plug-and-play compatible with different pose estimators and matching modules.
- **Efficient Monte Carlo**: By substituting local optimization for the random prediction step of conventional MCL, only 8 particles and 2 iterations are needed to reach state-of-the-art performance.
- **Elegant integration of Fisher information and PnP**: Geometric uncertainty is naturally incorporated into RANSAC via sampling weights without modifying the PnP solver itself.
- **Cross-prior robustness**: A weak prior (DFNet) refined by UGS-Loc can approach the results of a strong prior (ACE).

## Limitations & Future Work

- Inference time scales linearly with the number of particles (16 particles ≈ 2× latency), limiting real-time applicability.
- Fisher information must be precomputed and aggregated from all training views; scene updates require recomputation.
- Geometric uncertainty is validated only on Scaffold-GS and has not been extended to other 3DGS variants (vanilla 3DGS, 2DGS, etc.).
- Improvements in rotation error on the outdoor Cambridge scenes are less pronounced than those in translation error.
- Uncertainty modeling under dynamic scenes or extreme illumination changes is not explored.
- The perturbation range for Monte Carlo sampling is a manually set hyperparameter.

## Related Work & Insights

- **vs. GS-CPR**: GS-CPR is a deterministic refinement method; UGS-Loc introduces a probabilistic framework, achieving approximately 20–30% improvement.
- **vs. MCLoc**: Both adopt Monte Carlo strategies, but MCLoc is NeRF-based and requires 80 iterations at 2.4 s; UGS-Loc converges in 2 iterations at 1.1 s with higher accuracy.
- **vs. STDLoc**: STDLoc achieves results close to UGS-Loc on 7Scenes (0.76 vs. 0.64 cm), but UGS-Loc generalizes more robustly across different priors.
- **vs. Bayes' Rays / FisherRF**: These methods quantify uncertainty for reconstruction quality rather than localization; UGS-Loc is the first to apply Fisher information to pose refinement.
- **vs. HR-APR / NeFeS**: These methods require additional training and achieve 35/0.78 on Cambridge, whereas UGS-Loc reaches 10.5/0.21 without any training.

## Rating

- Novelty: ⭐⭐⭐⭐ — The dual uncertainty modeling concept is clearly motivated and systematically introduced to 3DGS-based localization for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three benchmarks, multiple priors, and detailed ablations; testing on more 3DGS variants would further strengthen the work.
- Writing Quality: ⭐⭐⭐⭐ — Figures are clear, motivation is well-articulated, and mathematical derivations are complete.
- Value: ⭐⭐⭐⭐ — A plug-and-play inference-time solution with strong practicality and direct impact on the 3DGS localization community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[CVPR 2026\] E2EGS: Event-to-Edge Gaussian Splatting for Pose-Free 3D Reconstruction](e2egs_event-to-edge_gaussian_splatting_for_pose-free_3d_reconstruction.md)
- [\[CVPR 2026\] SR3R: Rethinking Super-Resolution 3D Reconstruction With Feed-Forward Gaussian Splatting](sr3r_rethinking_super-resolution_3d_reconstruction_with_feed-forward_gaussian_sp.md)
- [\[ICCV 2025\] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](../../ICCV2025/3d_vision/no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)
- [\[CVPR 2026\] GAP: Action-Geometry Prediction with 3D Geometric Prior for Bimanual Manipulation](action-geometry_prediction_with_3d_geometric_prior_for_bimanual_manipulation.md)

</div>

<!-- RELATED:END -->
