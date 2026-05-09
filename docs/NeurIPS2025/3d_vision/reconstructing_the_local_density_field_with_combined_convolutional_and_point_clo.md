---
title: >-
  [Paper Note] Reconstructing the Local Density Field with Combined Convolutional and Point Cloud Architecture
description: >-
  [NeurIPS 2025 (ML4PS Workshop)][3D Vision][Dark matter density field reconstruction] This paper proposes a hybrid neural network architecture combining convolutional networks (U-Net) and point cloud networks (DeepSets) to reconstruct the local dark matter density field from line-of-sight peculiar velocities of dark matter halos, achieving significant improvements over purely convolutional and linear reconstruction methods at small scales.
tags:
  - NeurIPS 2025 (ML4PS Workshop)
  - 3D Vision
  - Dark matter density field reconstruction
  - U-Net
  - DeepSets
  - point cloud
  - peculiar velocity
date: 2026-05-08
content_hash: 4d984fc586649b9d
---

# Reconstructing the Local Density Field with Combined Convolutional and Point Cloud Architecture

**Conference**: NeurIPS 2025 (ML4PS Workshop)
**arXiv**: [2510.08573](https://arxiv.org/abs/2510.08573)
**Code**: None
**Area**: 3D Vision / Cosmology
**Keywords**: Dark matter density field reconstruction, U-Net, DeepSets, point cloud, peculiar velocity

## TL;DR
This paper proposes a hybrid neural network architecture combining convolutional networks (U-Net) and point cloud networks (DeepSets) to reconstruct the local dark matter density field from line-of-sight peculiar velocities of dark matter halos, achieving significant improvements over purely convolutional and linear reconstruction methods at small scales.

## Background & Motivation

### Root Cause

**Key Challenge**: **Background**: One of the central goals of observational cosmology is to infer the late-time matter density field in the local universe. Dark matter is invisible and must be probed indirectly, e.g., via galaxy peculiar velocities. Traditional linear reconstruction methods (e.g., Wiener filtering) are effective at large scales but fail to fully exploit the dense information available in modern datasets at nonlinear scales. While pure convolutional networks offer improvements, voxelizing radial velocities before passing them to a CNN discards small-scale information. This paper proposes using a DeepSets point cloud network to directly process local collections of dark matter halos in high-density regions, thereby recovering small-scale features.

## Method

### Overall Architecture
The architecture consists of three components: (1) a confidence U-Net that predicts error distributions and selects voxels for DeepSets evaluation; (2) a DeepSets module that processes local point cloud information at selected high-uncertainty voxels; and (3) a main U-Net that integrates the linear reconstruction, DeepSets output, and confidence network output for final prediction.

### Key Designs
- **Input Preprocessing**: Rather than directly using voxelized radial velocities (whose translational-invariance inductive bias is mismatched with convolutional networks), the method first obtains $\delta_L(\mathbf{k})$ via direct inversion using linear theory. Three smoothing scales (1, 2, 4 $h^{-1}$Mpc) are used as independent input channels, allowing the network to learn optimal weighting.
- **DeepSets Point Cloud Module**: For each voxel, dark matter halos within a 10 $h^{-1}$Mpc radius are collected; relative positions and line-of-sight velocities serve as point-level features, while global position and mean velocity serve as global features.
- **Confidence Network for Computational Efficiency**: Evaluating DeepSets across all $128^3$ voxels is computationally prohibitive. A dual-head U-Net is therefore trained to predict both the mean $\bar{\delta}$ and uncertainty $\sigma$, with DeepSets evaluated only for the $5 \times 10^4$ voxels with the highest $\sigma$ (approximately 2.4% of the total volume).
- **Output Transform**: $\hat{\delta} = \text{ReLU}(\sinh(y)) - 1$, which better accommodates the dynamic range of the density field.

### Loss & Training
- The confidence U-Net is trained with $\beta$-NLL loss ($\beta=0.5$).
- The main model is optimized with field-level mean squared error (MSE).
- Two-stage training: the confidence U-Net is trained first, followed by the full model including DeepSets.
- AdamW optimizer with a one-cycle learning rate schedule; training takes approximately 30 hours on 24 NVIDIA A100 GPUs.

## Key Experimental Results

### Main Results

| Model | MSE |
|-------|-----|
| Direct inversion (1 $h^{-1}$Mpc smoothing) | 4.94 |
| Direct inversion (4 $h^{-1}$Mpc smoothing) | 4.48 |
| 3D Wiener filtering | 3.80 |
| Vanilla U-Net | 3.20 |
| Confidence U-Net ($\mu$ prediction) | 3.26 |
| **U-Net + DeepSets (50k voxels)** | **2.99** |

### Ablation Study
- The improvement from DeepSets is primarily concentrated at small scales, i.e., $k \sim 0.1$–$0.2~h\text{Mpc}^{-1}$ and above.
- The correlation coefficient between the $\sigma$ predicted by the confidence network and the underlying $\delta$ exceeds 0.8 for $k = 1~h\text{Mpc}^{-1}$ and above.
- The confidence network preferentially selects high-density regions for DeepSets evaluation, which is physically well-motivated.

### Key Findings
- Machine learning reconstruction yields dramatic improvements over linear Wiener filtering.
- Even at moderate tracer densities, the additional small-scale information recovered by the point cloud network is substantial.
- U-Net + DeepSets outperforms the pure U-Net in both cross-correlation coefficient and transfer function.
- Some density peaks are entirely missed by the confidence network, indicating room for further improvement.

## Highlights & Insights
1. **Innovative Architecture Design**: The method cleverly combines the large-scale efficiency of CNNs with the fine-grained small-scale information capture of point cloud networks.
2. **Computational Efficiency Trick**: The confidence network restricts DeepSets evaluation to only 2.4% of voxels, making the computation tractable.
3. **Physics-Driven Input Design**: Using the linear reconstruction $\delta_L$ rather than the raw velocity field as input better matches the inductive bias of CNNs.

## Limitations & Future Work
- Validation is performed solely on simulation data; observational systematics are not addressed.
- The method does not incorporate stochasticity (i.e., no generative modeling); only the posterior mean is regressed.
- The influence of the tracer density field on predictions has not been disentangled.
- Known and fixed background cosmological parameters are assumed.

## Related Work & Insights
- The combination of U-Net and confidence-network-controlled DeepSets is generalizable to analogous problems in cosmology and other domains involving spatial field reconstruction.
- Future work should account for observational errors and incorporate simulations with varying cosmological parameters.

## Rating
- Novelty: ⭐⭐⭐⭐ (hybrid architecture design is original)
- Technical Depth: ⭐⭐⭐⭐ (computational efficiency optimization is impressive)
- Experimental Thoroughness: ⭐⭐⭐ (workshop paper; experiments are appropriately scoped)
- Practical Value: ⭐⭐⭐ (promising applications but requires further validation)
- Overall: ⭐⭐⭐⭐ (innovative architecture addressing a concrete scientific problem)

## Supplementary Analysis

Data are drawn from 100 simulation boxes from the Quijote high-resolution suite (box side length 1 $h^{-1}$Gpc). For each box, a dark matter halo is randomly selected as the observer, and approximately 30,000 dark matter halos within a 200 $h^{-1}$Mpc radius are collected. The total dataset size is 25,600, split 80/10/10 across simulation boxes for training/validation/testing to prevent information leakage. A regular $128^3$ grid (voxel spacing approximately 3.23 $h^{-1}$Mpc) is used.

Key findings from the two-point statistics analysis (Fig. 3): the Wiener filtering baseline uses the full 3D velocity information (whereas real observations provide only the line-of-sight component), making it an optimistic upper bound for linear reconstruction. The improvement from DeepSets is most pronounced at nonlinear scales above $k \sim 0.1$–$0.2~h\text{Mpc}^{-1}$. The correlation coefficient between the confidence network output and the density field exceeds 0.8 for $k > 1~h\text{Mpc}^{-1}$, confirming that intelligently allocating computational resources to high-density regions is physically justified.

The output transform $\hat{\delta} = \text{ReLU}(\sinh(y)) - 1$ is designed to accommodate the characteristic distribution of density fields: a large fraction of values near $-1$ and a small number of large values in high-density regions; the $\sinh$ transform stretches the dynamic range at the high-density end. The proposed architecture is generalizable to spatial field reconstruction problems beyond cosmology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Graph Smoothing for Enhanced Local Geometry Learning in Point Cloud Analysis](../../AAAI2026/3d_vision/graph_smoothing_for_enhanced_local_geometry_learning_in_point_cloud_analysis.md)
- [\[CVPR 2026\] ECKConv: Learning Coordinate-based Convolutional Kernels for Continuous SE(3) Equivariant Point Cloud Analysis](../../CVPR2026/3d_vision/learning_coordinate-based_convolutional_kernels_for_continuous_se3_equivariant_a.md)
- [\[AAAI 2026\] DANCE: Density-Agnostic and Class-Aware Network for Point Cloud Completion](../../AAAI2026/3d_vision/dance_density-agnostic_and_class-aware_network_for_point_cloud_completion.md)
- [\[NeurIPS 2025\] Rectified Point Flow: Generic Point Cloud Pose Estimation](rectified_point_flow_generic_point_cloud_pose_estimation.md)
- [\[ICCV 2025\] Efficient Spiking Point Mamba for Point Cloud Analysis](../../ICCV2025/3d_vision/efficient_spiking_point_mamba_for_point_cloud_analysis.md)

</div>

<!-- RELATED:END -->
