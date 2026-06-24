---
title: >-
  [Paper Note] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis
description: >-
  [CVPR2025][Autonomous Driving][LiDAR view synthesis] SG-NLF proposes a pose-free LiDAR NeRF framework. By reconstructing smooth geometry with a hybrid spectral-geometric representation, achieving global alignment via a confidence-aware pose graph, and enhancing cross-frame consistency with adversarial learning, it outperforms the state-of-the-art by 35.8% in reconstruction quality and 68.8% in pose accuracy under low-frequency LiDAR scenarios.
tags:
  - "CVPR2025"
  - "Autonomous Driving"
  - "LiDAR view synthesis"
  - "NeRF"
  - "pose estimation"
  - "spectral embedding"
  - "point cloud reconstruction"
date: 2026-05-08
content_hash: 2a9b0f1f9ee173bf
---

# Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis

**Conference**: CVPR2025  
**arXiv**: [2603.12903](https://arxiv.org/abs/2603.12903)  
**Code**: TBD  
**Area**: Autonomous Driving  
**Keywords**: LiDAR view synthesis, NeRF, pose estimation, spectral embedding, point cloud reconstruction

## TL;DR
SG-NLF proposes a pose-free LiDAR NeRF framework. By reconstructing smooth geometry with a hybrid spectral-geometric representation, achieving global alignment via a confidence-aware pose graph, and enhancing cross-frame consistency with adversarial learning, it outperforms the state-of-the-art by 35.8% in reconstruction quality and 68.8% in pose accuracy under low-frequency LiDAR scenarios.

## Background & Motivation
1. LiDAR novel view synthesis (NVS) is crucial for expanding the perception range and enhancing the robustness of autonomous driving systems.
2. Traditional LiDAR simulation (ray casting) struggles to accurately model the intensity and ray-drop characteristics of real-world LiDAR.
3. While NeRF has been successfully extended to LiDAR NVS, most methods rely heavily on accurate poses, which are difficult to acquire in practice.
4. LiDAR data is sparse and lacks texture information; interpolation encoding (such as multi-resolution hash encoding) struggles to reconstruct continuous surfaces, leading to geometric cavities and discontinuities.
5. The existing pose-free method, GeoNLF, relies on pairwise alignment constraints, making it difficult to guarantee global pose accuracy.
6. Low-frequency LiDAR sequences (characterized by large inter-frame motion and low overlap rates) further exacerbate the challenges of multi-view consistency.

## Method

### Overall Architecture
SG-NLF consists of three core components: (1) a hybrid spectral-geometric representation for smooth and consistent scene reconstruction; (2) a confidence-aware pose graph for global pose optimization; and (3) an adversarial learning strategy to enhance cross-frame consistency. Given a sequence of multi-frame LiDAR point clouds as input, it jointly recovers global poses and reconstructs a continuous implicit scene representation.

### Key Designs

**1. Hybrid Spectral-Geometric Representation**
- **Geometric Encoding**: Extracts local geometric features $\boldsymbol{f}_{\text{geo}}(\mathbf{x})$ based on a multi-resolution hash grid.
- **Spectral Embedding**: Learns the first $K$ eigenfunctions $\Psi_k(\mathbf{x})$ of the Laplace-Beltrami operator, which possess intrinsic isometric invariance.
    - Differentiably approximates the eigenfunctions via an MLP while minimizing the Rayleigh quotient.
    - Employs rejection sampling to uniformly sample points on the implicit surface to compute the area elements of the First Fundamental Form.
    - Orthogonality loss $\mathcal{L}_{\text{ortho}}$ + Normalization loss $\mathcal{L}_{\text{norm}}$.
- **Progressive Fusion**: Gradually fuses the spectral and geometric features into $\boldsymbol{f}_{\text{hyb}}(\mathbf{x})$ during training.

**2. Confidence-Aware Pose Graph**
- Constructs a pose graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ containing sequential edges as well as non-adjacent, highly-compatible edges.
- Establishes point correspondences using a coarse-to-fine Mutual Nearest Neighbor (MNN) matching based on hybrid features.
- **Edge Compatibility Score**: Measures the average cosine similarity $E^{ij}$ of corresponding feature pairs, with adaptive thresholding controlling edge selection.
- **Spatial Consistency Weighting**: Computes a distance-preservation score $P_{mn}$ between corresponding pairs to serve as the edge weight $\alpha^{ij}$.
- **Pose Graph Loss**: Weighted Chamfer Distance $\mathcal{L}_{\text{graph}} = \sum_{(i,j) \in \mathcal{E}} \alpha^{ij} \cdot \mathcal{L}_{\text{cd}}^{ij}$.
- **Pose Parameterization**: Employs 6D Lie algebra + exponential mapping, omitting the Jacobian to stabilize convergence.

**3. Adversarial Learning**
- For adjacent frames $(i,j)$, transforms the reconstructed point cloud $\hat{\mathcal{S}}_i$ into the coordinate system of frame $j$ using the estimated relative pose to render a depth map.
- Constructs real pairs $[\hat{D}_{ij}, D_j]$ and fake pairs $[D_{ij}, D_j]$.
- Multi-scale PatchGAN discriminator + hinge loss.
- The discriminator simultaneously evaluates the quality of frame-by-frame reconstruction and the accuracy of cross-frame geometric alignment.

### Loss & Training
- Range image supervision (depth + intensity + ray-drop).
- Spectral loss $\mathcal{L}_{\text{spe}}$ (Rayleigh quotient + orthogonality + normalization).
- Pose graph loss $\mathcal{L}_{\text{graph}}$.
- Adversarial consistency loss $\mathcal{L}_{\text{con}}$.

## Key Experimental Results

### Main Results (KITTI-360 Low-Frequency Configuration)

| Method | CD↓ | F-score↑ | Depth RMSE↓ | Depth PSNR↑ | Intensity PSNR↑ |
|------|-----|----------|-------------|-------------|-----------------|
| LiDAR4D (GT pose) | 0.2760 | 0.8843 | 4.7303 | 24.73 | 16.95 |
| GeoNLF | 0.2363 | 0.9178 | 4.0293 | 25.28 | 16.58 |
| **SG-NLF** | **0.1695** | **0.9191** | **2.9514** | **28.71** | **19.27** |

### nuScenes Low-Frequency Configuration

| Method | CD↓ | Depth RMSE↓ | Intensity RMSE↓ |
|------|-----|-------------|-----------------|
| GeoNLF | 0.2408 | 5.8208 | 0.0378 |
| **SG-NLF** | **0.1545** | **3.0706** | **0.0299** |

CD decreases by 35.8%, and ATE decreases by 68.8% (nuScenes).

### KITTI-360 Standard-Frequency Configuration

| Method | CD↓ | Depth PSNR↑ | Intensity PSNR↑ |
|------|-----|-------------|-----------------|
| LiDAR-NeRF | 0.0923 | 26.77 | 16.17 |
| LiDAR4D | 0.0894 | 27.88 | 17.45 |
| GeoNLF | 0.1855 | 29.39 | 16.57 |
| **SG-NLF** | **0.0867** | **32.72** | **19.55** |

### Key Findings
- Even though LiDAR4D uses GT poses, the pose-free SG-NLF comprehensively outperforms it in terms of CD and RMSE.
- Spectral embedding significantly reduces geometric holes, rendering reconstructed surfaces more continuous and smooth.
- Non-adjacent edges in the pose graph effectively improve global trajectory accuracy.
- Adversarial learning yields a pronounced improvement in cross-frame consistency.

## Highlights & Insights
1. **Innovative Application of Spectral Embedding**: First to introduce LBO eigenfunctions into LiDAR NeRF, leveraging intrinsic isometric invariance to reconstruct smooth geometry.
2. **Global vs. Pairwise Alignment**: The confidence-aware pose graph discovers non-adjacent loop closure constraints through feature compatibility, breaking the limitations of pairwise alignment.
3. **Outperforming GT-Pose Methods**: Without requiring pre-computed poses, SG-NLF outperforms methods utilizing ground-truth poses in reconstruction quality.
4. **Adversarial Learning for Geometric Consistency**: The PatchGAN discriminator simultaneously evaluates reconstruction quality and pose accuracy.

## Limitations & Future Work
1. The MLP optimization for spectral embedding increases training time, and inference efficiency is not reported in detail.
2. Validated only on two autonomous driving datasets (KITTI-360 and nuScenes), with no testing on indoor or unstructured scenarios.
3. The sensitivity of performance to the adaptive threshold selection in pose graph construction is not fully analyzed.
4. The stability of adversarial training may be affected under extreme scenarios.

## Related Work & Insights
- Compared to the pairwise alignment in GeoNLF, the global optimization using a pose graph combined with feature compatibility is a key breakthrough.
- The concept of spectral embedding can be generalized to other sparse 3D reconstruction tasks (e.g., RGB-D, event cameras).
- Adversarial learning-based cross-frame consistency supervision can be applied to other scene reconstruction frameworks.
- Provides a high-quality view synthesis tool for LiDAR data augmentation and simulation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Triple innovation of spectral embedding + global pose graph + adversarial learning)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multiple datasets and configurations, comprehensive ablation studies)
- Writing Quality: ⭐⭐⭐⭐ (Clear description of methodology, complete mathematical derivation)
- Value: ⭐⭐⭐⭐⭐ (Significantly advances the pose-free LiDAR NVS state-of-the-art)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](../../CVPR2026/autonomous_driving/sgnlf_spectralgeometric_neural_fields_for_posefre.md)
- [\[CVPR 2025\] VIRD: View-Invariant Representation through Dual-Axis Transformation for Cross-View Pose Estimation](vird_view-invariant_representation_through_dual-axis_transformation_for_cross-vi.md)
- [\[CVPR 2025\] Closed-Loop Supervised Fine-Tuning of Tokenized Traffic Models](closed-loop_supervised_fine-tuning_of_tokenized_traffic_models.md)
- [\[CVPR 2025\] EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis](evolsplat_efficient_volume-based_gaussian_splatting_for_urban_view_synthesis.md)
- [\[CVPR 2025\] PIDLoc: Cross-View Pose Optimization Network Inspired by PID Controllers](pidloc_cross-view_pose_optimization_network_inspired_by_pid_controllers.md)

</div>

<!-- RELATED:END -->
