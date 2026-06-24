---
title: >-
  [Paper Note] Relative Pose Estimation through Affine Corrections of Monocular Depth Priors
description: >-
  [CVPR 2025][3D Vision][Relative pose estimation] This paper proposes three new relative pose solvers that leverage monocular depth priors by explicitly modeling the affine (scale + shift) ambiguity of depth predictions. It also designs a hybrid estimation framework that combines depth-aware solvers with classical point solvers, significantly improving pose estimation accuracy under both calibrated and uncalibrated settings.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Relative pose estimation"
  - "Monocular depth priors"
  - "Affine correction"
  - "RANSAC"
  - "Geometric solvers"
date: 2026-05-08
content_hash: f856848daec0814d
---

# Relative Pose Estimation through Affine Corrections of Monocular Depth Priors

**Conference**: CVPR 2025  
**arXiv**: [2501.05446](https://arxiv.org/abs/2501.05446)  
**Code**: [https://github.com/MarkYu98/madpose](https://github.com/MarkYu98/madpose)  
**Area**: 3D Vision  
**Keywords**: Relative pose estimation, Monocular depth priors, Affine correction, RANSAC, Geometric solvers

## TL;DR
This paper proposes three new relative pose solvers that leverage monocular depth priors by explicitly modeling the affine (scale + shift) ambiguity of depth predictions. It also designs a hybrid estimation framework that combines depth-aware solvers with classical point solvers, significantly improving pose estimation accuracy under both calibrated and uncalibrated settings.

## Background & Motivation

**Background**: Monocular depth estimation (MDE) models have achieved significant progress in recent years, capable of providing reasonable depth predictions on open-domain images (e.g., from Marigold to Depth Anything v2 to MoGe). However, how to effectively leverage these depth priors to improve classical geometric vision tasks (specifically, relative pose estimation) remains under-explored.

**Limitations of Prior Work**: Existing methods assume that the depth maps of two images differ by only a global scale factor (scale-only). However, state-of-the-art depth models are usually affine-invariant (with both scale and shift discrepancies relative to ground truth). Even models claiming to predict "metric depth" lack perfect consistency across views, displaying shift differences. Neglecting the shift modeling leads to incorrect depth alignment and inaccurate pose estimation.

**Key Challenge**: Depth priors provide rich cross-view constraints, but their inherent noise and affine ambiguity make it difficult to simply integrate them to improve classical keypoint-based methods. A robust pipeline is required to leverage depth priors without being misled by their inaccuracies.

**Goal**: To develop relative pose solvers that can explicitly handle independent affine (scale + shift) ambiguities in both calibrated and uncalibrated scenarios, and to design a hybrid estimation framework compatible with both depth-aware and classical methods.

**Key Insight**: Treat the depth map affine correction parameters ($\alpha, \beta_1, \beta_2$) as unknowns to be solved jointly with the relative pose. By leveraging the distance invariance of rigid body transformations to eliminate rotation and translation, the affine parameters and focal lengths can be solved first, followed by the recovery of $R, t$.

**Core Idea**: Design three new solvers for relative pose estimation corresponding to calibrated (3-point), shared focal length (4-point), and independent focal length (4-point) scenarios, and integrate them with classical solvers within a hybrid RANSAC framework to utilize both depth and epipolar constraints.

## Method

### Overall Architecture
The inputs are two images, where feature matching (e.g., SuperPoint+LightGlue or RoMa) and monocular depth estimation are executed to obtain pixel correspondences and depth maps $D_1, D_2$. The depth maps are transformed via affine corrections $\hat{D}_1 = D_1 + \beta_1$ and $\hat{D}_2 = \alpha(D_2 + \beta_2)$ to constrain the pose. In a hybrid LO-MSAC framework, depth-aware solvers and classical point solvers are used alternately. Joint scoring is performed using both depth reprojection error and Sampson error, followed by joint optimization of all parameters in local optimization.

### Key Designs

1. **Affine-Corrected Solvers**:

    - **Function**: Jointly estimate the relative pose and the affine parameters (scale ratio $\alpha$ and shifts $\beta_1, \beta_2$) of the two depth maps, as well as the unknown focal lengths if applicable.
    - **Mechanism**: Utilize the distance conservation of rigid body transformations $\|\delta_{jk}^{(1)}\|^2 = \|\delta_{jk}^{(2)}\|^2$ to eliminate rotation and translation, yielding a system of polynomial equations in only the affine parameters. (a) Calibrated 3-point solver: 3 correspondences yield 3 quartic equations (reparameterizing $\gamma = \alpha^2$ reduces them to cubic), solved using Gröbner bases, resulting in at most 4 solutions via a 12x12 elimination matrix and a 4x4 eigenvalue problem. (b) Shared focal length 4-point solver: 4 correspondences yield 4 equations with an additional unknown $\omega = 1/f^2$, resulting in at most 8 solutions via a 36x36 template. (c) Dual focal length 4-point solver: 5 equations solve 5 unknowns, unexpectedly yielding only 4 solutions with a 40x40 elimination matrix. After solving the affine parameters, $R, t$ are recovered by aligning the 3D points of both views via SVD.
    - **Design Motivation**: Existing methods (e.g., 2pt+D) either assume scale-only or suffer from rank-deficient degeneracies. Explicitly modeling shifts is crucial because state-of-the-art MDE models (including metric depth models) still suffer from cross-view shift inconsistencies.

2. **Hybrid Estimation Pipeline**:

    - **Function**: Utilize both depth-aware constraints and classical epipolar constraints in a RANSAC framework to complement and enhance robustness.
    - **Mechanism**: In each RANSAC iteration, either a depth-aware solver or a classical solver (5-point/6-point/7-point) is randomly selected, initially with equal probability, and later adjusted based on the inlier ratio of each data type. Each correspondence is categorized into three data types: $(p_1, p_2, d_1)$, $(p_1, p_2, d_2)$, and $(p_1, p_2)$, evaluated using depth reprojection errors $E_{r(1\to2)}$, $E_{r(2\to1)}$, and Sampson error $E_s$, respectively. Affine parameters are also estimated for the results of classical solvers (via triangulation and least squares fitting). The joint MSAC score is formulated as $\bar{E} = \bar{E}_r + 2\lambda_s \frac{\tau_r}{\tau_s}\bar{E}_s$.
    - **Design Motivation**: Relying solely on depth priors is prone to failures when depth is unreliable, while purely point-based methods neglect depth information. The hybrid approach allows the system to adaptively balance between the two signal sources. The bidirectional depth reprojection error design improves robustness against cross-view depth inconsistencies.

3. **Hybrid Local Optimization**:

    - **Function**: Perform joint optimization of all parameters $\Theta = (R, t, \alpha, \beta_1, \beta_2)$ within the RANSAC loop.
    - **Mechanism**: Minimize $E(\Theta) = \sum_{I_1} E_{r(1\to2)} + \sum_{I_2} E_{r(2\to1)} + 2\lambda_s \frac{\tau_r}{\tau_s} \sum_{I_3} E_s$ over the inlier sets of the three data types, implemented using the Ceres solver's automatic differentiation.
    - **Design Motivation**: The local optimization step in RANSAC is critical for the final accuracy. The hybrid objective function ensures that the optimization considers both depth and epipolar constraints.

### Loss & Training
This method is a traditional geometric approach (non-learning-based) and does not require training. The solvers are implemented in C++ and exported via pybind11 as a Python interface. It can be combined with any feature matcher and MDE model.

## Key Experimental Results

### Main Results
Evaluated under the calibrated setting on ScanNet-1500 (indoor) and MegaDepth-1500 (outdoor):

| Matcher | Method | Depth Model | AUC@5° | AUC@10° | AUC@20° |
|--------|------|---------|--------|---------|---------|
| SP+LG | PoseLib-5pt | - | 21.55 | 39.11 | 55.60 |
| SP+LG | PoseLib-PnP | DA-metric | 15.05 | 34.16 | 53.97 |
| SP+LG | **Ours** | **DA-metric** | **Significant Gain** | **Significant Gain** | **Significant Gain** |
| RoMa | PoseLib-5pt | - | Baseline | Baseline | Baseline |
| RoMa | **Ours** | **MoGe** | **Significant Gain** | **Significant Gain** | **Significant Gain** |

### Ablation Study

| Configuration | AUC@5°↑ | Description |
|------|---------|------|
| Scale-only correction | Moderate gain | No shift modeled |
| Scale + shift correction | **Substantial gain** | Full affine modeling |
| Depth solver only | Improved but unstable | Lacks epipolar constraint fallback |
| Classical solver only | Baseline | Does not leverage depth |
| **Hybrid estimation** | **Highest** | Complementary constraints |
| Apply shift to "metric depth" | Still brings gain | Proves metric depth is not perfect |

### Key Findings
- Shift modeling is the most critical contribution—even for models claiming to predict "metric depth" (e.g., Depth Anything v2 metric), incorporating shift correction further improves pose accuracy, demonstrating that all MDE models exhibit cross-view shift inconsistency.
- Hybrid estimation significantly outperforms using either solver alone, proving the complementarity of depth and epipolar constraints under different scenarios.
- The method consistently improves performance across different feature matchers (SP+LG, RoMa) and MDE models (Marigold, Depth Anything, MoGe), offering plug-and-play capability.
- The improvement is orthogonal to MASt3R—further performance gains can be achieved by utilizing pixel correspondences from MASt3R.

## Highlights & Insights
- **Generality of Affine Correction**: Showing that even metric depth models require affine correction is a counter-intuitive yet vital finding, providing guiding insights for the community leveraging MDE priors for geometric tasks.
- **Elegant Design of Hybrid Estimation**: Three data types + bidirectional reprojection errors + adaptive solver selection naturally integrate depth and geometric constraints within the RANSAC framework.
- **High Practicality**: C++ implementation with Python bindings enjoys compatibility with arbitrary matchers and MDE models, facilitating direct integration into existing SfM/SLAM pipelines.

## Limitations & Future Work
- The solvers are based on algebraic geometry (Gröbner bases), and the template sizes for the shared focal length and dual focal length solvers are relatively large (36×36 and 40×40), which may affect numerical stability under extreme configurations.
- The method only handles relative poses for pinhole cameras, without extension to rolling shutter or fisheye camera models.
- Compared to end-to-end approaches like DUSt3R/MASt3R, it still requires running matching and depth estimation separately, leading to a longer pipeline.
- Performance may degrade in scenarios where depth priors are entirely incorrect (e.g., specular surfaces, transparent objects).

## Related Work & Insights
- **vs PoseLib (5pt/7pt)**: Classical pure geometric methods do not leverage depth priors. The proposed method yields a consistent improvement of approximately 3-8% AUC@5°.
- **vs PnP-RANSAC**: Directly performing PnP on MDE depths without modeling shifts can perform worse than pure 5pt, highlighting the necessity of shift modeling.
- **vs Ding et al. (3p3d/4p4d)**: Although they also use depth priors, they only model scale without scale-shift (affine) adaptation. The proposed affine modeling achieves superior performance under all settings.
- **vs DUSt3R/MASt3R**: End-to-end learning methods requiring large-scale training. This method is training-free and its improvements are orthogonal.

## Rating
- Novelty: ⭐⭐⭐⭐ The formulation of affine correction and the hybrid estimation framework are solid contributions; the finding that "metric depth also needs shifts" is highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple datasets, matchers, and MDE models, with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem formulation and rigorous derivation from solver formulation to hybrid framework design.
- Value: ⭐⭐⭐⭐⭐ A highly practical plug-and-play solution providing valuable reference for the broader geometric vision community utilizing depth priors.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] RePoseD: Efficient Relative Pose Estimation with Known Depth Information](../../ICCV2025/3d_vision/reposed_efficient_relative_pose_estimation_with_known_depth_information.md)
- [\[CVPR 2025\] Scalable Autoregressive Monocular Depth Estimation](scalable_autoregressive_monocular_depth_estimation.md)
- [\[CVPR 2025\] Vision-Language Embodiment for Monocular Depth Estimation](vision-language_embodiment_for_monocular_depth_estimation.md)
- [\[ICCV 2025\] Single-Scanline Relative Pose Estimation for Rolling Shutter Cameras](../../ICCV2025/3d_vision/single-scanline_relative_pose_estimation_for_rolling_shutter_cameras.md)
- [\[CVPR 2025\] BLADE: Single-view Body Mesh Learning through Accurate Depth Estimation](blade_single-view_body_mesh_estimation_through_accurate_depth_estimation.md)

</div>

<!-- RELATED:END -->
