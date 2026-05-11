---
title: >-
  [Paper Note] Rectified Point Flow: Generic Point Cloud Pose Estimation
description: >-
  [NeurIPS 2025][3D Vision][point cloud pose estimation] This paper proposes Rectified Point Flow, a unified generative framework that reformulates pairwise point cloud registration and multi-part shape assembly as a condi…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "point cloud pose estimation"
  - "rectified flow"
  - "shape assembly"
  - "registration"
  - "symmetry handling"
date: 2026-05-08
content_hash: 03052f82ed5ec8ef
---

# Rectified Point Flow: Generic Point Cloud Pose Estimation

**Conference**: NeurIPS 2025
**arXiv**: [2506.05282](https://arxiv.org/abs/2506.05282)
**Code**: [Project Page](https://rectified-pointflow.github.io/)
**Area**: 3D Vision / Point Cloud Registration
**Keywords**: point cloud pose estimation, rectified flow, shape assembly, registration, symmetry handling

## TL;DR

This paper proposes Rectified Point Flow, a unified generative framework that reformulates pairwise point cloud registration and multi-part shape assembly as a conditional generation problem, estimating part poses by learning a continuous point-wise velocity field.

## Background & Motivation

Estimating the relative poses of rigid parts from 3D point clouds is a core task in computer vision and robotics, encompassing pairwise registration and multi-part shape assembly. Existing methods suffer from several issues:

**Task Fragmentation**: Object pose estimation, part registration, and shape assembly rely on different assumptions and architectures, making cross-task generalization difficult.

**Difficulty Handling Symmetry**: Traditional per-part pose regression methods require manual treatment of symmetry and part interchangeability.

**Assembly Ambiguity**: Parts may be symmetric, interchangeable, or geometrically ambiguous, leading to multiple locally valid but globally inconsistent configurations.

The core idea is to reframe pose estimation as a generative problem of learning a continuous point-wise flow field over input geometry, implicitly encoding part transformations.

## Method

### Overall Architecture

The pipeline consists of two stages:
1. **Overlap-aware Point Encoding**: A pretrained encoder identifies overlapping regions between parts.
2. **Conditional Rectified Point Flow**: A conditional generative model predicts the assembled point cloud positions, from which poses are recovered via SVD.

### Key Designs

1. **Overlap-aware Encoder Pretraining**: PointTransformerV3 (PTv3) is used as the backbone. The pretraining task is binary classification: for each point, predicting whether it overlaps with another part (distance $< \varepsilon$). Random rigid transformations are applied for data augmentation. Compared to GARF, which relies on physical mesh simulation to generate fracture supervision signals, this approach is more lightweight and scalable and does not require watertight meshes. Pretraining data is drawn from diverse sources: part segmentation, shape assembly, registration datasets, and Objaverse.

2. **Rectified Point Flow Generative Model**: Built on the Rectified Flow framework, operating directly on point cloud coordinates in 3D Euclidean space. The forward process is defined as $X_i(t) = (1-t)X_i(0) + tX_i(1)$, where $t=0$ corresponds to the assembled point cloud and $t=1$ to Gaussian noise. The velocity field $dX_i(t)/dt = X_i(1) - X_i(0)$ is learned. A Diffusion Transformer (DiT) serves as the flow model, incorporating two-stage self-attention: intra-part attention and global attention.

3. **Shape-to-Pose Recovery**: After predicting the assembled point cloud $\hat{X}_i(0)$, the rigid transformation for each non-anchor part is recovered via the Procrustes problem (SVD): $\hat{T}_i = \arg\min \|\hat{T}_i X_i - \hat{X}_i(0)\|_F$. The velocity of the anchor part (the part with the largest volume) is set to zero.

### Loss & Training

- Conditional Flow Matching (CFM) loss: $\mathcal{L}_{CFM}(V) = \mathbb{E}_{t,X}[\|V(t, X_i(t) | X) - \nabla_t X(t)\|^2]$
- Timesteps sampled from a U-shaped distribution
- 8× NVIDIA A100 80GB GPUs, 400k iterations, effective batch size 256
- AdamW optimizer, initial learning rate $5 \times 10^{-4}$, halved every 25k steps after 275k iterations
- Encoder weights frozen after pretraining

## Key Experimental Results

### Main Results

| Dataset | Method | RE↓ (deg) | TE↓ (cm) | Part Acc↑ (%) |
|--------|------|-----------|----------|---------------|
| BreakingBad | GARF | 9.9 | 2.0 | 93.0 |
| BreakingBad | **Ours (Joint)** | **7.4** | 2.0 | 91.1 |
| TwoByTwo | GARF | 22.1 | 7.1 | - |
| TwoByTwo | **Ours (Joint)** | **13.2** | **3.0** | - |
| PartNet-Assembly | GARF | 66.9 | 21.9 | 25.7 |
| PartNet-Assembly | **Ours (Joint)** | **21.8** | **14.8** | **53.9** |

The proposed method also comprehensively outperforms GeoTransformer and Diff-RPMNet on registration benchmarks (TUD-L and ModelNet-40).

### Ablation Study

| Configuration | Metric | Notes |
|------|------|------|
| Single vs. Joint training | RE/TE | Joint training outperforms on most datasets |
| w/ vs. w/o encoder pretraining | Accuracy | Pretraining significantly improves performance across all tasks |
| Anchor-free evaluation | Part Acc | A fairer evaluation protocol is proposed |

### Key Findings

- Part Acc on PartNet-Assembly more than doubles from 25.7% to 53.9% compared to GARF.
- Joint training learns shared geometric priors across datasets, improving single-task performance.
- This is the first method to address furniture assembly on PartNet-Assembly and IKEA-Manual datasets.
- GARF fails to transfer to registration tasks, while the proposed method achieves state-of-the-art performance on both assembly and registration.

## Highlights & Insights

- The **unified parameterization** philosophy is elegant: operating on dense point clouds in Euclidean space simultaneously encodes both shape and pose.
- **Intrinsic symmetry handling** is a key contribution: Theorem 1 proves that the learning objective is invariant under the assembly symmetry group $\mathcal{G}$, requiring neither symmetry labels nor manual augmentation.
- **Joint training** across datasets with different part definitions enables transfer of geometric knowledge, a unique advantage of generative approaches.
- The lightweight overlap prediction pretraining is more scalable than the physics simulation-based pretraining used in GARF.

## Limitations & Future Work

- SVD-based pose recovery is a post-processing step that may introduce errors.
- The choice of anchor part (largest volume) may be impractical in real-world applications.
- Inference requires multi-step ODE solving, making it slower than direct regression methods.
- Only rigid transformations are considered; non-rigid deformations are not addressed.
- The impact of the point sampling count $M_i$ on performance is not thoroughly discussed.

## Related Work & Insights

- Relation to DUSt3R: both directly regress point coordinates and extract poses, but DUSt3R targets camera pose while this work targets part pose.
- Comparison with GARF: GARF performs 6-DoF regression with fracture pretraining; this work performs dense point flow with overlap pretraining, offering greater generality.
- The application of flow matching to 3D geometry warrants continued attention.
- The proposed anchor-free evaluation protocol is a positive contribution to the community.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ A generative framework unifying registration and assembly with elegant symmetry handling.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six datasets, multiple tasks, comprehensive comparisons and ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with solid theoretical grounding.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a new unified paradigm for 3D pose estimation with significant potential impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PCSTracker: Long-Term Scene Flow Estimation for Point Cloud Sequences](../../CVPR2026/3d_vision/pcstracker_long-term_scene_flow_estimation_for_point_cloud_sequences.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](../../ICCV2025/3d_vision/efficient_spiking_point_mamba_for_point_cloud_analysis.md)
- [\[NeurIPS 2025\] U-CAN: Unsupervised Point Cloud Denoising with Consistency-Aware Noise2Noise Matching](u-can_unsupervised_point_cloud_denoising_with_consistency-aware_noise2noise_matc.md)
- [\[NeurIPS 2025\] SingRef6D: Monocular Novel Object Pose Estimation with a Single RGB Reference](singref6d_monocular_novel_object_pose_estimation_with_a_single_rgb_reference.md)
- [\[NeurIPS 2025\] PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion](pointmac_meta-learned_adaptation_for_robust_test-time_point_cloud_completion.md)

</div>

<!-- RELATED:END -->
