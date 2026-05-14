---
title: >-
  [Paper Note] DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] This paper natively integrates the Kannala-Brandt fisheye projection model into the 3DGS pipeline and proposes a cross-view joint optimization strategy based on feature overl…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "fisheye camera"
  - "cross-view joint optimization"
  - "novel view synthesis"
  - "Kannala-Brandt model"
date: 2026-05-08
content_hash: fa7e4f4de1c5cb34
---

# DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization

**Conference**: CVPR 2026
**arXiv**: [2604.00648](https://arxiv.org/abs/2604.00648)
**Code**: None
**Area**: 3D Vision / Novel View Synthesis
**Keywords**: 3D Gaussian Splatting, fisheye camera, cross-view joint optimization, novel view synthesis, Kannala-Brandt model

## TL;DR

This paper natively integrates the Kannala-Brandt fisheye projection model into the 3DGS pipeline and proposes a cross-view joint optimization strategy based on feature overlap, eliminating the information loss caused by pre-undistortion and achieving state-of-the-art performance on multiple public benchmarks.

## Background & Motivation

3D Gaussian Splatting (3DGS) has achieved breakthrough progress in novel view synthesis; however, its core rasterization renderer relies on the pinhole camera model and cannot directly handle the nonlinear distortion images produced by fisheye cameras. Fisheye cameras are widely adopted in autonomous driving, robotics, and VR/AR applications due to their ultra-wide field of view (>90°, commonly 120°–180°).

Two major limitations of existing approaches:

**Information loss from pre-undistortion**: Converting fisheye images to pinhole images crops or stretches peripheral regions through interpolation, diluting high-frequency details. 3DGS tends to overfit these low-frequency regions, producing blurring and floating artifacts.

**Geometric inconsistency from single-view optimization**: Even with a correct fisheye projection model, the original single-view random sampling strategy in 3DGS ignores inter-view correlations of the same Gaussian. Regions with the most severe edge distortion are especially prone to excessively large or elongated Gaussians, degrading reconstruction quality.

Limitations of existing fisheye 3DGS methods:
- **Fisheye-GS**: Still requires preprocessing into equidistant projection; cannot directly use raw fisheye images.
- **3DGUT**: Approximates projection via the Unscented Transform with only 7 sigma points, yielding insufficient accuracy in strongly distorted regions; also disrupts the fully explicit architecture of 3DGS.
- **Self-Cali-GS**: Learns a deformation field via a neural network, converging slowly and losing high-frequency details.

## Method

### Overall Architecture

DirectFisheye-GS comprises two core innovations: (1) deep coupling of the Kannala-Brandt fisheye projection model into both the forward rendering and backward gradient computation of 3DGS; and (2) a cross-view joint optimization (CVO) strategy based on feature overlap and viewpoint diversity.

### Key Designs

1. **Kannala-Brandt Fisheye Projection Integration**: Natively supports fisheye distortion via a polynomial-expansion-based general model, avoiding the information loss introduced by pre-undistortion.

   - Given a 3D point $\mu_{cam} = (x_c, y_c, z_c)^T$ in camera coordinates, the incident angle $\theta$ is computed.
   - The effective incident angle is approximated by a four-parameter polynomial: $\theta_d = \theta + k_1\theta^3 + k_2\theta^5 + k_3\theta^7 + k_4\theta^9$
   - After obtaining the projected pixel coordinates, the complete Jacobian matrix $\mathbf{J}_\theta \in \mathbb{R}^{3\times3}$ is derived.
   - The Jacobian consists of two components: a radial variation term (related to $\theta_d'$) and a tangential variation term (related to $\theta_d/d$).
   - **Design Motivation**: Preserves the fully rasterized architecture of 3DGS, ensuring complete compatibility with existing 3DGS viewers and commercial tools.

2. **Cross-View Joint Optimization (CVO)**: Addresses geometric inconsistency from single-view optimization via adaptive view grouping based on feature overlap and viewpoint diversity, enabling each Gaussian to receive consistent constraints across multiple views.

   - **Step 1 – Camera association graph**: Leverages COLMAP SfM feature matching results to rank neighboring cameras by the number of shared SIFT feature points.
   - **Step 2 – Angular difference ranking**: Computes the pose angular difference for each camera pair in the association graph and sorts them in descending order.
   - **During training**: Each iteration samples one primary view and the top-(batchsize-1) associated cameras, accumulates multi-view losses, and performs a unified backward pass.
   - **Design Motivation**: Maximizes Gaussian overlap across training views while ensuring sufficient viewpoint diversity, promoting cross-view consistency in both shape and appearance.

### Loss & Training

- The loss function follows the standard 3DGS formulation: $L_k = L_1(I_k, \hat{I}_k) + \text{SSIM}(I_k, \hat{I}_k)$
- Multi-view losses are accumulated: $L_{total} = \sum_{k} L_k$, followed by a unified backward pass to update Gaussian parameters.
- Default batch size is set to 2 (one primary view + one associated view).
- All experiments are conducted on a single NVIDIA A100 80 GB GPU.

## Key Experimental Results

### Main Results

**FisheyeNeRF dataset** (6 object-level small scenes, training views):

| Method | SSIM ↑ | PSNR ↑ | LPIPS ↓ |
|--------|--------|--------|---------|
| 3DGS (direct fisheye input) | 0.6124 | 18.85 | 0.5228 |
| 3DGS* (undistorted) | 0.8240 | 25.54 | 0.2431 |
| Fisheye-GS | 0.8183 | 25.18 | 0.2658 |
| 3DGUT | 0.8020 | 25.28 | 0.3290 |
| Self-Cali-GS | 0.7460 | 24.01 | 0.4507 |
| **Ours** | **0.8284** | **26.25** | **0.2295** |

**Scannet++ dataset** (6 medium-scale indoor scenes, test views): The proposed method ranks first or second among all baselines on this dataset as well.

**Den-SOFT dataset** (large-scale outdoor scenes): The proposed method significantly outperforms other approaches on outdoor scenes such as Ruziniu, attributed to large illumination variation and rich fine-grained details.

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| Vanilla 3DGS + fisheye | Complete failure; SSIM only 0.61 |
| 3DGS + undistortion | Acceptable but loses edge information |
| Fisheye projection model alone | Effective but floating artifacts remain at edges |
| Fisheye projection + CVO | Edge artifacts significantly reduced; global illumination consistency improved |

### Key Findings

- Even with a correct fisheye camera model, single-view optimization still produces floating Gaussians at image boundaries — **cross-view constraints are necessary**.
- The CVO strategy is not limited to fisheye cameras; it also improves reconstruction quality in conventional pinhole camera pipelines.
- 3DGUT achieves comparable performance on indoor scenes but falls noticeably behind on unbounded outdoor scenes, where its strong illumination modeling introduces discontinuous artifacts.
- Fisheye-GS suffers sharp quality degradation at image boundaries due to the overly simplified equidistant projection model.
- While undistort-then-train is a viable alternative, the proposed method matches or surpasses it on all metrics, validating the advantage of direct native input.

## Highlights & Insights

1. **High engineering completeness**: The paper derives the full Jacobian for fisheye projection (complete $3\times3$ matrix in Eq. 7) while maintaining full compatibility with the original 3DGS viewer — a significant practical advantage for real-world deployment.
2. **Generality of CVO**: The cross-view joint optimization strategy reuses feature matching information already available from SfM at no additional computational cost, and is applicable to any camera model.
3. **Precise problem diagnosis**: The paper clearly identifies that "even with a correct camera model, insufficient optimization leads to extreme Gaussian shapes," and addresses this from both geometric and photometric consistency perspectives.
4. **Efficiency**: A batch size of 2 is sufficient to achieve the effect, incurring minimal additional training cost.

## Limitations & Future Work

1. The batch size in CVO is fixed at 2; whether larger batches yield further improvements remains unexplored.
2. The method relies on COLMAP for feature matching to construct the association graph — if SfM quality is poor, the effectiveness of CVO may be limited.
3. Validation is performed solely on the Kannala-Brandt model; extensibility to other nonlinear camera models (e.g., omnidirectional cameras) has not been verified.
4. The paper does not report a comparison of training time against vanilla 3DGS, despite the increased rendering volume introduced by batch size 2.

## Related Work & Insights

- **MVGS**: An early exploration of multi-view training, but relies on random sampling of view subsets without geometric association.
- **3DGUT**: Handles nonlinear projection via the Unscented Transform, representing an alternative paradigm (sampling-based approximation vs. analytical derivation).
- **Scaffold-GS / Compact3DGS**: Works focused on optimizing 3DGS data structures and memory, orthogonal to the proposed method.
- Insight: The single-view optimization paradigm of 3DGS is inherently limited; cross-view consistency constraints should become a standard component of future pipelines.

## Rating

- Novelty: ⭐⭐⭐⭐ — Fisheye projection integration is an engineering innovation; the CVO strategy constitutes a methodological contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three datasets, comprehensive comparisons, and analysis across training/test views.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and precise problem formulation.
- Value: ⭐⭐⭐⭐ — High direct applicability to VR/AR and autonomous driving scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DropAnSH-GS: Dropping Anchor and Spherical Harmonics for Sparse-view Gaussian Splatting](dropping_anchor_and_spherical_harmonics_for_sparse-view_gaussian_splatting.md)
- [\[CVPR 2026\] NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation](ng_gs_nerf_guided_3d_gaussian_splatting_segmentation.md)
- [\[CVPR 2026\] Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment](cross-instance_gaussian_splatting_registration_via_geometry-aware_feature-guided.md)
- [\[CVPR 2026\] Learning Multi-View Spatial Reasoning from Cross-View Relations](learning_multi-view_spatial_reasoning_from_cross-view_relations.md)
- [\[CVPR 2026\] PixARMesh: Autoregressive Mesh-Native Single-View Scene Reconstruction](pixarmesh_autoregressive_mesh-native_single-view_scene_reconstruction.md)

</div>

<!-- RELATED:END -->
