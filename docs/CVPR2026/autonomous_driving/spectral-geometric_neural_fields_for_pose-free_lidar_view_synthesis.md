---
title: >-
  [Paper Note] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis
description: >-
  [CVPR 2026][Autonomous Driving][LiDAR view synthesis] This paper proposes SG-NLF, a framework that achieves pose-free LiDAR novel view synthesis via a hybrid spectral-geometric representation, combined with a confidence-aware pose graph and adversarial learning strategy. It significantly outperforms state-of-the-art methods on KITTI-360 and nuScenes (Chamfer Distance reduced by 35.8%, ATE reduced by 68.8%).
tags:
  - CVPR 2026
  - Autonomous Driving
  - LiDAR view synthesis
  - NeRF
  - pose-free
  - spectral embedding
  - pose graph optimization
date: 2026-05-08
content_hash: 0f4151aed6c3f807
---

# Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis

**Conference**: CVPR 2026
**arXiv**: [2603.12903](https://arxiv.org/abs/2603.12903)
**Code**: Unavailable
**Area**: Autonomous Driving
**Keywords**: LiDAR view synthesis, NeRF, pose-free, spectral embedding, pose graph optimization

## TL;DR

This paper proposes SG-NLF, a framework that achieves pose-free LiDAR novel view synthesis via a hybrid spectral-geometric representation, combined with a confidence-aware pose graph and adversarial learning strategy. It significantly outperforms state-of-the-art methods on KITTI-360 and nuScenes (Chamfer Distance reduced by 35.8%, ATE reduced by 68.8%).

## Background & Motivation

**State of the Field**: NeRF has been successfully extended to LiDAR novel view synthesis (NVS), with methods such as LiDAR-NeRF and LiDAR4D implicitly reconstructing scenes via volume rendering. However, two critical bottlenecks remain: reliance on accurate poses and geometric discontinuities caused by LiDAR data sparsity.

**Limitations of Prior Work**: (a) Nearly all existing methods (LiDAR-NeRF, NFL, LiDAR4D, STGC) require accurate camera pose inputs, which are difficult to obtain in practice. (b) Geometry interpolation based on multi-resolution hash encoding tends to produce geometric holes and discontinuous surfaces in textureless regions (as illustrated in Fig. 2).

**Root Cause**: The only existing pose-free method, GeoNLF, employs pairwise alignment constraints that cannot guarantee global trajectory accuracy. Furthermore, purely geometric interpolation representations fail to reconstruct continuous surfaces in sparse LiDAR regions.

**Paper Goals**: To simultaneously achieve high-quality LiDAR view synthesis and accurate pose estimation without requiring precise pose inputs.

**Starting Point**: Spectral embeddings are introduced to provide global structural priors for geometric representation, and a confidence-aware pose graph based on feature compatibility is constructed for global pose optimization.

**Core Idea**: The isometry-invariant property of spectral embeddings is naturally suited to filling geometric holes in sparse LiDAR data. Combined with geometric encoding and a global pose graph, this approach simultaneously addresses the dual challenges of scene representation and pose estimation.

## Method

### Overall Architecture

Given a multi-view LiDAR sequence $\{S_i\}$, SG-NLF proceeds as follows: (1) projects LiDAR point clouds into range images; (2) encodes 3D point features with a hybrid spectral-geometric representation; (3) optimizes global poses via a confidence-aware pose graph; (4) feeds optimized poses and hybrid features into a NeRF to render novel views; (5) applies an adversarial learning strategy to enhance cross-frame consistency.

### Key Designs

1. **Hybrid Spectral-Geometric Representation**:

   - Geometric encoding $f_\text{geo}(x)$: based on multi-resolution hash grid encoding, capturing local structure and high-frequency details.
   - Spectral embedding $f_\text{spe}(x)$: approximates the first $K$ eigenfunctions of the Laplace-Beltrami Operator (LBO) via an MLP, possessing intrinsic isometry invariance.
   - Core optimization objective: minimization of the discrete Rayleigh quotient + orthogonality constraint + normalization constraint.
   - The two representations are progressively fused into a hybrid representation $f_\text{hyb}(x)$, balancing low-frequency smooth geometry and high-frequency details.

2. **Confidence-Aware Global Pose Optimization**:

   - A pose graph $G = (V, E)$ is constructed, where nodes represent LiDAR frames and edges include temporally adjacent edges and non-adjacent high-compatibility edges.
   - Point correspondences are established via coarse-to-fine mutual nearest neighbor (MNN) matching on hybrid features.
   - Edge compatibility scores are computed as the mean cosine similarity of feature pairs; edges are included only when the score exceeds an adaptive threshold.
   - Edge weights are based on the spatial consistency score (distance preservation) of correspondences.
   - Pose graph loss: weighted Chamfer Distance.

3. **Cross-Frame Consistency**:

   - An adversarial learning strategy is introduced: reconstructed point clouds are transformed into adjacent frame coordinate systems using estimated relative poses and rendered as depth maps.
   - Fake samples (synthesized depth + real depth) and real samples (real transformed depth + real depth) are constructed.
   - A multi-scale PatchGAN discriminator evaluates geometric alignment quality.
   - Hinge loss is used to stabilize training.

### Loss & Training

- The overall training objective comprises: range image supervision loss (depth / intensity / raydrop) + pose-graph-weighted Chamfer Distance loss + spectral loss (Rayleigh quotient + orthogonality + normalization) + adversarial consistency loss.
- Training runs for 60,000 iterations with a batch size of 4,096 rays, an initial learning rate of 0.01, and linear decay.
- Poses are parameterized as 6D Lie algebra vectors and optimized as increments in $\mathfrak{se}(3)$ space.
- Training is feasible on a single RTX 4090 GPU.

## Key Experimental Results

### Main Results: KITTI-360 Low-Frequency Setting

| Method | Pose | CD↓ | F-score↑ | Depth RMSE↓ | Depth PSNR↑ | Intensity PSNR↑ |
|:---|:---|:---|:---|:---|:---|:---|
| LiDARsim | GT | 11.04 | 0.598 | 10.20 | 17.94 | 13.61 |
| PCGen | GT | 1.036 | 0.786 | 7.57 | 20.62 | 13.42 |
| LiDAR4D | GT | 0.276 | 0.884 | 4.73 | 24.73 | 16.95 |
| GeoNLF | Pose-free | 0.236 | 0.918 | 4.03 | 25.28 | 16.58 |
| **SG-NLF** | **Pose-free** | **0.170** | **0.919** | **2.95** | **28.71** | **19.27** |

### Ablation Study: Component Contributions on nuScenes

| Method | HR | GP | CFC | CD↓ | Depth PSNR↑ | Intensity PSNR↑ | ATE (m)↓ |
|:---|:---|:---|:---|:---|:---|:---|:---|
| Baseline | ✗ | ✗ | ✗ | 0.618 | 21.32 | 25.86 | 1.328 |
| w/o HR | ✗ | ✓ | ✓ | 0.217 | 25.10 | 28.43 | 0.204 |
| w/o GP | ✓ | ✗ | ✓ | 0.463 | 23.94 | 27.55 | 0.798 |
| w/o CFC | ✓ | ✓ | ✗ | 0.182 | 26.60 | 29.30 | 0.076 |
| **Full SG-NLF** | **✓** | **✓** | **✓** | **0.155** | **28.41** | **30.50** | **0.071** |

### Key Findings

- Under the nuScenes low-frequency setting, SG-NLF reduces CD by 35.8% and ATE by 68.8% compared to GeoNLF.
- Even compared to LiDAR4D which uses GT poses, the pose-free SG-NLF achieves improvements of 38.5%, 37.5%, and 25.4% in CD, depth RMSE, and intensity RMSE, respectively.
- SG-NLF also achieves state-of-the-art performance on standard-frequency KITTI-360, demonstrating generalization ability.
- The contribution of spectral embedding: it produces smoother and more complete geometry (see Fig. 6), though using it alone leads to missing high-frequency details; fusion with geometric encoding achieves the best of both.

## Highlights & Insights

- **First introduction of spectral methods into LiDAR NeRF**: The isometry-invariant eigenfunctions of the LBO naturally compensate for LiDAR sparsity and texturelessness, making them a well-motivated design choice.
- **Global pose graph vs. pairwise alignment**: By leveraging feature compatibility to discover constraints between non-adjacent frames, trajectory accuracy is substantially improved.
- **Clever application of adversarial learning**: Cross-frame depth map pairs serve as real/fake samples, allowing the discriminator to jointly evaluate reconstruction quality and pose accuracy.
- **Strong performance in low-frequency scenarios**: The advantage is especially pronounced in low-frequency sequences characterized by large inter-frame motion and limited overlap.

## Limitations & Future Work

- The paper acknowledges that only one effective instantiation of SG-NLF is presented; future work may explore additional technical combinations.
- Spectral embedding requires sampling on implicit surfaces and solving the LBO, which increases computational complexity.
- Dynamic scenes are not handled (a capability already present in LiDAR4D and STGC).
- Edge filtering in the pose graph relies on adaptive thresholds, and the robustness of the thresholding strategy has not been thoroughly analyzed.

## Related Work & Insights

- Compared to GeoNLF (pairwise alignment), SG-NLF achieves global optimization via the confidence-aware pose graph, yielding ATE reductions of 56%–69%.
- Successful practices from spectral methods (SNS, Neural Geometry Processing) in 3D geometric processing are introduced into LiDAR NeRF for the first time.
- The application of adversarial learning for cross-frame consistency can inspire other multi-frame reconstruction tasks.
- The framework is extensible to joint LiDAR-camera representation learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Introducing spectral embeddings into LiDAR NeRF is a highly creative design; the global pose graph and adversarial consistency also constitute substantial contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two datasets × two frequency settings, with comprehensive comparisons against both pose-supervised and pose-free methods, and clear ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with complete mathematical derivations, though some sections are notation-heavy.
- **Value**: ⭐⭐⭐⭐⭐ A significant advance in pose-free LiDAR view synthesis, with particularly strong gains over state-of-the-art in low-frequency scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](sgnlf_spectralgeometric_neural_fields_for_posefre.md)
- [\[CVPR 2026\] Neural Distribution Prior for LiDAR Out-of-Distribution Detection](neural_distribution_prior_for_lidar_ood_detection.md)
- [\[CVPR 2026\] Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization](learning_geometric_and_photometric_features_from_panoramic_lidar_scans_for_outdo.md)
- [\[CVPR 2026\] LiREC-Net: A Target-Free and Learning-Based Network for LiDAR, RGB, and Event Calibration](lirec-net_a_target-free_and_learning-based_network_for_lidar_rgb_and_event_calib.md)
- [\[CVPR 2026\] EMDUL: Expanding mmWave Datasets for Human Pose Estimation with Unlabeled Data and LiDAR Datasets](expanding_mmwave_datasets_for_human_pose_estimation_with_unlabeled_data_and_lida.md)

<!-- RELATED:END -->
