---
title: >-
  [Paper Note] POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality
description: >-
  [CVPR 2025][3D Vision][3D Gaussian Splatting] Introduces the P-Optimality theory from classical optimal experimental design into 3D-GS, deriving a general covariance matrix based on the Hessian matrix. Two approximation schemes, diagonal and block-diagonal, are proposed, significantly outperforming the information gain quantification of FisherRF under D-Optimality and T-Optimality criteria.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "uncertainty quantification"
  - "optimal experimental design"
  - "next best view"
  - "Fisher information"
date: 2026-05-08
content_hash: 2c44bc34283fd9b3
---

# POp-GS: Next Best View in 3D-Gaussian Splatting with P-Optimality

**Conference**: CVPR 2025  
**arXiv**: [2503.07819](https://arxiv.org/abs/2503.07819)  
**Code**: None  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, uncertainty quantification, optimal experimental design, next best view, Fisher information

## TL;DR

Introduces the P-Optimality theory from classical optimal experimental design into 3D-GS, deriving a general covariance matrix based on the Hessian matrix. Two approximation schemes, diagonal and block-diagonal, are proposed, significantly outperforming the information gain quantification of FisherRF under D-Optimality and T-Optimality criteria.

## Background & Motivation

Although 3D-GS achieves high rendering quality, it lacks inherent uncertainty quantification capabilities, limiting its application in SLAM, active perception, and other fields:

1. **Limitations of FisherRF**: Though it quantifies information gain via diagonal approximation of Fisher information, it neglects correlation between parameters and fails to utilize the rich literature of optimal experimental design.
2. **Large Covariance Matrix**: 3D-GS may contain millions of parameters, making the full covariance matrix computationally and memory-wise intractable.
3. **Lack of Unified Framework**: Existing methods independently design information metrics, lacking a systematic theoretical framework.

This work derives the covariance matrix of 3D-GS from the perspective of maximum likelihood estimation and applies P-Optimality theory to provide a family of information metric solutions.

## Method

### Overall Architecture

From the perspective of maximum likelihood, the covariance matrix of 3D-GS parameters $\theta$ is $\Sigma = \sigma_e^2 (J^TJ)^{-1}$, where $J$ is the Jacobian of the rendering function with respect to the parameters. After adding the candidate image $i$, the new Hessian is $H_i = H_- + J_i^T J_i$. Information metrics are defined through different $p$ values of P-Optimality.

### Key Designs

**1. P-Optimality-based Information Quantification Framework**

- **Function**: Provides a unified family of information gain metrics, where different $p$ values correspond to different geometric meanings.
- **Mechanism**: $U_p(\Sigma_i) = (\frac{1}{l} \text{trace}(\Sigma_i^p))^{1/p}$. T-Optimality ($p=1$) represents average variance (trace), A-Optimality ($p=-1$) represents harmonic mean variance, D-Optimality ($p \to 0$) represents the volume of the covariance hyper-ellipsoid (determinant), and E-Optimality ($p \to \pm\infty$) represents extreme eigenvalues.
- **Design Motivation**: D-Optimality possesses monotonicity guarantees in active mapping (uncertainty decreases monotonically with exploration) and corresponds to the differential entropy of multivariate Gaussians from an information-theoretic perspective.

**2. Block-Diagonal Covariance Approximation**

- **Function**: Captures the correlations between parameters of the same ellipsoid on top of the diagonal approximation.
- **Mechanism**: Approximates the full Hessian matrix as a block-diagonal matrix, where each block contains all parameters of a single 3D ellipsoid (position, rotation, scale, opacity, color). Channel-wise pixel gradients are calculated to avoid singularity issues, and the block matrices can be processed in parallel on the GPU.
- **Design Motivation**: Parameters within the same ellipsoid are most likely to be correlated (e.g., changes in position affect the color contribution), and diagonal approximation completely ignores these correlations.

**3. Batch Selection Algorithm**

- **Function**: Iteratively selects a subset of views with the highest information gains from a set of candidate views.
- **Mechanism**: Greedy strategy—iteratively selects the candidate image that maximizes the improvement of the P-Optimality metric, updates the Hessian, and repeats. No extra training is required, and redundancy between views is captured via incremental Hessian updates.
- **Design Motivation**: Single-view selection neglects redundancy between views, whereas batch selection naturally handles redundancy by updating the covariance through the Hessian.

### Loss & Training

No extra training loss is involved. The information quantification is calculated based on the Hessian of the pre-trained 3D-GS model:

$$H_i = H_- + J_i^T J_i, \quad J = \frac{\partial h}{\partial \theta}\bigg|_{\theta_*, p_i}$$

## Key Experimental Results

### Blender Dataset (Select 10 views from 100 candidates)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Uniform | 25.82 | 0.944 | 0.051 |
| FisherRF | 27.14 | 0.956 | 0.039 |
| Diag T-Opt (Ours) | 27.89 | 0.960 | 0.035 |
| **Block D-Opt (Ours)** | **28.31** | **0.963** | **0.032** |

### Mip-NeRF360 Dataset

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Uniform | 22.15 | 0.698 | 0.271 |
| FisherRF | 23.42 | 0.732 | 0.243 |
| **Block D-Opt (Ours)** | **24.18** | **0.756** | **0.221** |

### Comparison of Different $p$ Values in P-Optimality

| Metric Criterion | Approximation Method | Blender PSNR↑ |
|---------|---------|-------------|
| T-Optimality (p=1) | Diagonal | 27.89 |
| D-Optimality (p→0) | Diagonal | 27.95 |
| T-Optimality (p=1) | Block | 28.12 |
| **D-Optimality (p→0)** | **Block** | **28.31** |

### Key Findings

- Block D-Optimality outperforms FisherRF by ~1.2 PSNR on Blender and ~0.8 PSNR on Mip-NeRF360.
- D-Optimality consistently outperforms T/A/E-Optimality, aligning with theoretical expectations (monotonicity guarantee + information-theoretic significance).
- Block-diagonal approximation improves over simple diagonal approximation by ~0.4 PSNR, validating the importance of parameter correlations.
- It does not require candidate image content; only camera poses are needed to evaluate the info gain.

## Highlights & Insights

1. **Elegant Theoretical Framework**: Unifies 3D-GS information quantification into classical optimal experimental design, providing a class of solutions with theoretical guarantees.
2. **Practical Block-Diagonal Approximation**: Effectively captures parameter correlations with acceptable overhead increase.
3. **No Extra Training Required**: Purely based on gradient information of the pre-trained model, plug-and-play.

## Limitations & Future Work

- The computational cost of block-diagonal approximation is still high, cubic in the dimension of ellipsoid parameters.
- Changes in parameter counts during 3D-GS densification/pruning are not considered.
- Greedy batch selection is not globally optimal; more efficient combinatorial optimization methods can be explored.
- Extension to active perception in dynamic scenes can be explored in the future.

## Related Work & Insights

- **FisherRF**: A pioneer in 3D-GS information quantification, but only utilizes diagonal Fisher information.
- **Classical SLAM Literature**: P-Optimality is widely applied in keyframe selection and loop closure detection.
- **3D-GS Pruning**: Block-diagonal approximation is also applied to identify redundant ellipsoids.

## Rating

⭐⭐⭐⭐ — Represents a solid theoretical contribution, successfully introducing classical optimal design theory to 3D-GS. Experiments consistently outperform baselines across multiple datasets, and the block-diagonal approximation is a practical innovation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting](pup_3d-gs_principled_uncertainty_pruning_for_3d_gaussian_splatting.md)
- [\[CVPR 2025\] Mani-GS: Gaussian Splatting Manipulation with Triangular Mesh](mani-gs_gaussian_splatting_manipulation_with_triangular_mesh.md)
- [\[CVPR 2025\] Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)
- [\[CVPR 2025\] Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes](horizon-gs_unified_3d_gaussian_splatting_for_large-scale_aerial-to-ground_scenes.md)
- [\[CVPR 2025\] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting](s2gaussian_sparse-view_super-resolution_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
