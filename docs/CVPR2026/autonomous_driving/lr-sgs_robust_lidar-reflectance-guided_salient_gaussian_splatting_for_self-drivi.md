---
title: >-
  [Paper Note] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction
description: >-
  [CVPR 2026][Autonomous Driving][3D Gaussian Splatting] LR-SGS proposes a structure-aware Salient Gaussian representation guided by LiDAR reflectance. By calibrating LiDAR intensity into an illumination-invariant reflectance channel appended to each Gaussian, initializing structured Salient Gaussians from geometric and reflectance feature points, and enforcing RGB–reflectance cross-modal gradient consistency, the method surpasses OmniRe by 1.18 dB PSNR on complex-lighting scenes of the Waymo dataset while using fewer Gaussians and shorter training time.
tags:
  - CVPR 2026
  - Autonomous Driving
  - 3D Gaussian Splatting
  - LiDAR Reflectance
  - Self-Driving Scene Reconstruction
  - Novel View Synthesis
  - Multimodal Fusion
date: 2026-05-08
content_hash: 4ba3bb266c79db6d
---

# LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2603.12647](https://arxiv.org/abs/2603.12647)
**Code**: To be confirmed
**Area**: Autonomous Driving
**Keywords**: 3D Gaussian Splatting, LiDAR Reflectance, Self-Driving Scene Reconstruction, Novel View Synthesis, Multimodal Fusion

## TL;DR

LR-SGS proposes a structure-aware Salient Gaussian representation guided by LiDAR reflectance. By calibrating LiDAR intensity into an illumination-invariant reflectance channel appended to each Gaussian, initializing structured Salient Gaussians from geometric and reflectance feature points, and enforcing RGB–reflectance cross-modal gradient consistency, the method surpasses OmniRe by 1.18 dB PSNR on complex-lighting scenes of the Waymo dataset while using fewer Gaussians and shorter training time.

## Background & Motivation

**Background**: 3DGS has demonstrated fast, high-fidelity rendering capabilities for autonomous driving scene reconstruction and novel view synthesis. Existing methods such as StreetGS and OmniRe have constructed dynamic scene graphs to handle temporally dynamic objects.

**Limitations of Prior Work**: Camera-only approaches are prone to texture inconsistency and optimization instability under complex illumination (nighttime, backlight) and large ego-vehicle motion. Existing LiDAR-augmented methods (e.g., PVG, OmniRe) use point clouds only for Gaussian initialization or depth supervision, leaving the reflectance information and geometric structural information embedded in LiDAR point clouds largely unexploited.

**Key Challenge**: RGB signals are sensitive to external factors such as illumination and exposure, and cannot provide reliable constraints in weakly textured regions or at material boundaries. Although LiDAR provides accurate depth and is insensitive to illumination, the material properties (reflectance) and geometric structural features encoded in its return intensity have not been effectively utilized.

**Goal**: (1) How to integrate LiDAR's illumination-invariant material information into the Gaussian representation? (2) How to model edges and planar surfaces more precisely using structure-aware Gaussians? (3) How to align RGB and reflectance boundaries across modalities?

**Key Insight**: The raw LiDAR return signal contains intensity, which can be corrected by range $R$ and incidence angle $\alpha$ to yield an approximately illumination-invariant reflectance $\rho$. Key structural elements in the scene—edges, planes, etc.—can be extracted from point cloud smoothness and reflectance gradients, serving as initializations for a Salient Gaussian representation that uses fewer parameters while accurately capturing scene structure.

**Core Idea**: Append LiDAR-calibrated reflectance as an illumination-invariant channel to each Gaussian, initialize structure-aware Salient Gaussians from geometric and reflectance feature points, and enforce RGB–reflectance gradient alignment to enhance material boundary consistency.

## Method

### Overall Architecture

The inputs are RGB image sequences and LiDAR point cloud sequences. The method constructs a 3DGS scene graph (background, dynamic objects, and sky nodes), where the initial scene Gaussians consist of two components: Salient Gaussians initialized from LiDAR feature points and Non-Salient Gaussians initialized from SfM points. Rendering produces color images, depth maps, and reflectance maps; all Gaussian attributes—position, opacity, scale, rotation, appearance, and reflectance—are jointly optimized using a Color Loss, a LiDAR Loss, and a Joint Loss.

### Key Designs

#### 1. LiDAR Intensity Calibration

- **Function**: Calibrates raw LiDAR intensity into an illumination-invariant reflectance $\rho$.
- **Mechanism**: Based on the radar equation $I = \eta_{all} \frac{\rho \cos\alpha}{R^2}$, intensity is normalized by range $R$ and incidence angle $\alpha$. The incidence angle is computed by constructing a local surface normal $\mathbf{n}$ from point $\mathbf{p}$ and its neighboring points $\mathbf{p}_1, \mathbf{p}_2$: $\cos\alpha = \frac{\mathbf{p}^T \cdot \mathbf{n}}{\|\mathbf{p}\|}$. The calibrated reflectance is normalized to $[0,1]$ and projected onto the camera plane to yield a sparse reflectance map $F_{gt}$.
- **Design Motivation**: Raw intensity varies with range and incidence angle and cannot directly represent material properties. The calibrated reflectance approximately captures surface material characteristics, is invariant to illumination changes, and thus serves as a stable cross-frame constraint signal.

#### 2. Structure-Aware Salient Gaussian Representation

- **Function**: Designs a parameter-efficient structured Gaussian, categorized into Edge type (elongated along edges) and Planar type (flattened along surfaces).
- **Mechanism**: Each Salient Gaussian has a dominant direction $d_{spec}$ and a corresponding dominant scale $\sigma_\|$, while the remaining two axes share a single scale $\sigma_\perp$. The covariance matrices are simplified as: Edge type $\Sigma_{edge} = \mathbf{R} \operatorname{diag}(\sigma_\|^2, \sigma_\perp^2, \sigma_\perp^2) \mathbf{R}^T$, Planar type $\Sigma_{plane} = \mathbf{R} \operatorname{diag}(\sigma_\perp^2, \sigma_\perp^2, \sigma_\|^2) \mathbf{R}^T$.
- **Design Motivation**: Standard 3DGS uses three independent scale parameters per Gaussian, whereas the abundant edge and planar structures in driving environments only require stretching or flattening along specific directions. This reduces the parameter count while more accurately fitting environmental features.

#### 3. LiDAR Feature Point Initialization

- **Function**: Extracts three categories of feature points from LiDAR point clouds to initialize Salient Gaussians.
- **Mechanism**: (a) The smoothness $c_j = \frac{1}{|K| \cdot \|\mathbf{p}_j\|} \|\sum_{\mathbf{p}_i \in \mathcal{P}_j}(\mathbf{p}_i - \mathbf{p}_j)\|$ of each point is computed and thresholded to classify geometric edge points and geometric planar points; (b) The reflectance gradient $G_j$ (mean difference between left and right neighbors along the same scan ring) is computed to extract reflectance edge points. The three categories of feature points are used to initialize Edge and Planar Salient Gaussians respectively.
- **Design Motivation**: Unlike using all LiDAR points for initialization, feature points concentrate on structurally significant locations (object contours, road boundaries, material transitions), providing stable structural scaffolding for training and accelerating convergence.

#### 4. Improved Density Control and Salient Transform

- **Function**: Adapts the splitting strategy for Salient Gaussians and enables adaptive state transitions between Salient and Non-Salient modes.
- **Mechanism**: During splitting, Edge Salient Gaussians split along the dominant direction, while Planar Gaussians split within the orthogonal plane. Linearity $L(g) = (s_1 - s_2)/s_1$ and planarity $P(g) = (s_2 - s_3)/s_1$ are defined. A Non-Salient Gaussian is promoted to Salient if $\max\{L, P\} > \tau_{max}=0.5$ for two consecutive iterations; a Salient Gaussian is demoted to Non-Salient if $\max\{L, P\} < \tau_{min}=0.1$ for two consecutive iterations.
- **Design Motivation**: Scene structure evolves dynamically during training. Salient Gaussians must be able to grow naturally in areas not covered by LiDAR, and redundant Salient Gaussians that no longer exhibit clear directionality should be removed.

#### 5. RGB–Reflectance Cross-Modal Consistency (Joint Loss)

- **Function**: Aligns the gradient direction and magnitude between the rendered reflectance and the grayscale RGB image.
- **Mechanism**: The rendered RGB is first converted to grayscale $C^g$; both $C^g$ and the rendered reflectance $F$ undergo Gaussian smoothing followed by Scharr gradient computation. The Joint Loss consists of two terms: directional consistency $\mathcal{L}_{dir} = 1 - \hat{\nabla}F \cdot \hat{\nabla}C^g$ (dot product of normalized gradient vectors); and magnitude consistency $\mathcal{L}_{val} = \|g_F / F - g_{C^g} / C^g\|_1$ (L1 difference of normalized magnitudes, eliminating cross-modal scale discrepancy).
- **Design Motivation**: Material boundaries should manifest as salient gradients in both RGB and reflectance, but their absolute scales differ. By aligning normalized gradient directions and magnitudes, the method sharpens material boundaries without introducing scale bias, reducing blurring artifacts.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{rgb} + \mathcal{L}_{lidar} + \mathcal{L}_{joint}$, where:

- $\mathcal{L}_{rgb} = (1-\lambda_c)\mathcal{L}_1 + \lambda_c \mathcal{L}_{D\text{-}SSIM}$ (photometric consistency)
- $\mathcal{L}_{lidar} = \lambda_{depth}\mathcal{L}_{depth} + \lambda_{fle}\mathcal{L}_{fle} + \lambda'_{fle}\mathcal{L}'_{fle}$ (depth + reflectance + reflectance gradient)
- $\mathcal{L}_{joint} = \lambda_{dir}\mathcal{L}_{dir} + \lambda_{val}\mathcal{L}_{val}$ (cross-modal gradient alignment)

Loss weights: $\lambda_c = \lambda_{val} = 0.2$, $\lambda_{depth} = \lambda_{fle} = \lambda_{dir} = 0.1$, $\lambda'_{fle} = 0.05$. Training runs for 30k iterations with Salient transform thresholds $\tau_{max}=0.5$ and $\tau_{min}=0.1$.

## Key Experimental Results

### Main Results

Novel view synthesis results (PSNR/SSIM/LPIPS) across four scene categories on the Waymo Open Dataset:

| Scene Type | Metric | LR-SGS (Ours) | OmniRe | StreetGS | Gain |
|---------|------|:---:|:---:|:---:|:---:|
| Dense Traffic | PSNR↑ | **28.89** | 28.44 | 27.01 | +0.45 |
| Dense Traffic | PSNR*↑ | **24.02** | 23.72 | 21.73 | +0.30 |
| High-Speed | PSNR↑ | **28.77** | 28.12 | 28.06 | +0.65 |
| Complex Lighting | PSNR↑ | **30.51** | 29.33 | 29.16 | **+1.18** |
| Static | PSNR↑ | **28.73** | 28.23 | 28.15 | +0.50 |

The proposed method achieves comprehensive improvements in PSNR/SSIM/LPIPS across all scene categories. The gain is most pronounced in Complex Lighting scenes, reaching +1.18 dB PSNR.

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:---:|:---:|:---:|
| Full model (Ours) | **29.22** | **0.850** | **0.139** |
| w/o Salient Gaussian | 28.74 | 0.830 | 0.152 |
| w/o LiDAR Feature Init | 28.94 | 0.839 | 0.144 |
| w/o Reflectance | 28.87 | 0.831 | 0.147 |
| w/o Joint Loss | 28.96 | 0.835 | 0.144 |

### Efficiency Analysis

| Method | PSNR↑ | # Gaussians↓ | Training Time↓ | FPS↑ |
|------|:---:|:---:|:---:|:---:|
| StreetGS | 28.20 | 2,929,851 | 64m30s | 33.61 |
| OmniRe | 28.62 | 2,744,275 | 67m11s | 30.55 |
| **LR-SGS** | **28.95** | **2,510,883** | **59m25s** | **36.95** |

### Key Findings

1. **Salient Gaussian contributes the most**: Removing it reduces PSNR by 0.48 dB and increases LPIPS by 0.013, demonstrating that the structure-aware representation is critical for both quality and efficiency.
2. **Reflectance channel is particularly effective under complex illumination**: In nighttime scenes, LiDAR reflectance provides stable constraints unavailable from RGB, effectively suppressing illumination-induced artifacts.
3. **Clear efficiency advantages**: Compared to OmniRe, the method uses approximately 230k fewer Gaussians, trains 8 minutes faster, and achieves 6.4 higher FPS, attributed to the two-parameter Salient Gaussian reducing redundancy and accelerating convergence.
4. **Salient Transform extends structural coverage**: Even in areas not covered by LiDAR, Non-Salient Gaussians can be promoted to Salient through the transform mechanism, ensuring structural modeling across the full scene.
5. **Joint Loss improves both RGB quality and reflectance rendering**: Reconstruction sharpness in high-frequency regions such as license plates and lamp assemblies is notably improved.

## Highlights & Insights

1. **Simple yet effective calibration from LiDAR intensity to reflectance**: Physical correction using only range and incidence angle yields an illumination-invariant material representation without requiring complex inverse rendering or material estimation networks. This idea is transferable to any reconstruction task involving LiDAR.
2. **Two-parameter Salient Gaussian**: Replacing three independent scale parameters with a "dominant direction + shared non-dominant scale" formulation achieves a dual benefit of fewer parameters and better structural modeling, breaking the common accuracy–efficiency trade-off.
3. **Cross-modal gradient alignment rather than pixel alignment**: Directly aligning RGB and reflectance pixel values lacks physical meaning due to differing units; aligning normalized gradient directions and magnitudes elegantly circumvents the scale discrepancy and focuses on boundary consistency.
4. **Bidirectional Salient Transform mechanism**: Analogous to a promotion/demotion mechanism, it allows Salient Gaussians to emerge and dissolve naturally during training, avoiding information loss caused by rigid classification.

## Limitations & Future Work

1. **Evaluation limited to the Waymo dataset**: Generalizability to other driving datasets such as nuScenes and KITTI has not been verified; differences in intensity characteristics across LiDAR models may affect reflectance calibration accuracy.
2. **Reflectance calibration relies on a simplified physical model**: Real-world material BRDFs are far more complex than Lambertian reflection, and calibration accuracy may be insufficient for specular or translucent materials.
3. **Dynamic object modeling depends on external masks**: Object masks are obtained from pretrained models, and the accuracy of this step directly constrains the quality of dynamic reconstruction.
4. **Salient Transform thresholds are manually set**: The values $\tau_{max}=0.5$ and $\tau_{min}=0.1$ may not generalize across all scenes; adaptive threshold strategies warrant further exploration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling](f3dgs_federated_3d_gaussian_splatting_for_decentralized_multi-agent_world_modeli.md)
- [\[CVPR 2026\] Learning Geometric and Photometric Features from Panoramic LiDAR Scans for Outdoor Place Categorization](learning_geometric_and_photometric_features_from_p.md)
- [\[CVPR 2026\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction.md)
- [\[CVPR 2026\] x2-Fusion: Cross-Modality and Cross-Dimension Flow Estimation in Event Edge Space](x2-fusion_cross-modality_and_cross-dimension_flow_estimation_in_event_edge_space.md)
- [\[CVPR 2026\] BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird's-Eye View Images](bev-sld_self-supervised_scene_landmark_detection_for_global_localization_with_li.md)

</div>

<!-- RELATED:END -->
