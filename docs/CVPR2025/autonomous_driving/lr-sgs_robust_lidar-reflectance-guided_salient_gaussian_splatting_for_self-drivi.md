---
title: >-
  [Paper Note] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction
description: >-
  [CVPR 2025][Autonomous Driving][3D Gaussian Splatting] LR-SGS proposes a robust LiDAR-reflectance-guided salient Gaussian splatting method, introducing structure-aware salient Gaussian representations (initialized by LiDAR geometry and reflectance feature points) and an illumination-invariant reflectance channel as extra constraints. On the challenging scenes (complex lighting) of the Waymo dataset, its PSNR outperforms OmniRe by 1.18 dB.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "3D Gaussian Splatting"
  - "LiDAR Reflectance"
  - "Salient Gaussian"
  - "Autonomous Driving Scene Reconstruction"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: 6ed58386c136108b
---

# LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction

**Conference**: CVPR 2025  
**arXiv**: [2603.12647](https://arxiv.org/abs/2603.12647)  
**Code**: None  
**Area**: Autonomous Driving / 3D Reconstruction  
**Keywords**: 3D Gaussian Splatting, LiDAR Reflectance, Salient Gaussian, Autonomous Driving Scene Reconstruction, Novel View Synthesis

## TL;DR
LR-SGS proposes a robust LiDAR-reflectance-guided salient Gaussian splatting method, introducing structure-aware salient Gaussian representations (initialized by LiDAR geometry and reflectance feature points) and an illumination-invariant reflectance channel as extra constraints. On the challenging scenes (complex lighting) of the Waymo dataset, its PSNR outperforms OmniRe by 1.18 dB.

## Background & Motivation
**Background**: 3DGS performs exceptionally in autonomous driving scene reconstruction and novel view synthesis (fast and high fidelity), but mainly relies on RGB images, with LiDAR only used for initialization or depth supervision.

**Limitations of Prior Work**: (1) RGB signals are easily affected by external factors such as lighting and exposure, making it difficult to maintain geometric and appearance consistency in complex lighting and high-speed motion scenarios; (2) Existing methods do not fully utilize the reflectance and geometric structural information in LiDAR point clouds.

**Key Challenge**: Photometric consistency based solely on RGB has insufficient constraint in low-texture areas and varying illumination scenes; while LiDAR depth is accurate, its resolution is low—requiring better mining of LiDAR geometric and material information as complementary constraints.

**Goal**: How to fully utilize LiDAR geometric features and reflectance information to enhance the reconstruction quality of 3DGS in challenging autonomous driving scenes?

**Key Insight**: (a) LiDAR reflectance, after distance and incidence angle calibration, is approximately illumination-invariant—acting as an extra material constraint channel; (b) Edge and planar points in LiDAR point clouds correspond to key structural features of the scene—which can be used to initialize directional "salient Gaussians."

**Core Idea**: Salient Gaussians (directional Gaussians initialized by LiDAR geometry and reflectance feature points) + illumination-invariant reflectance channel + joint RGB-reflectance gradient consistency loss.

## Method

### Overall Architecture
Input RGB + LiDAR sequences $\to$ LiDAR intensity calibration to reflectance $\to$ Feature point extraction to initialize salient Gaussians $\to$ 3DGS scene graph (background + dynamic objects + sky) $\to$ Joint color/depth/reflectance rendering $\to$ Optimization using three types of loss (color + LiDAR + joint).

### Key Designs

1. **LiDAR Intensity Calibration to Reflectance**

    - **Function**: Calibrates original LiDAR intensity $I$ to illumination-independent material reflectance $\rho$
    - **Mechanism**: $I = \eta_{all} \frac{\rho \cos\alpha}{R^2}$, corrected using range $R$ and angle of incidence $\alpha$ (calculated via local normals)
    - **Design Motivation**: Original intensity is affected by distance and incidence angle. Calibrated reflectance represents the intrinsic property of the material and is illumination-invariant.

2. **Salient Gaussian Representation**

    - **Function**: Defines Gaussians with dominant directions—edge salient Gaussians (stretched along the edge) and planar salient Gaussians (flattened along the normal)
    - **Mechanism**: Simplifies the covariance matrix to $\Sigma_\text{edge} = \mathbf{R}\text{diag}(\sigma_\|^2, \sigma_\perp^2, \sigma_\perp^2)\mathbf{R}^T$, optimizing only one dominant scale $\sigma_\|$ and a shared non-dominant scale $\sigma_\perp$
    - **Design Motivation**: Reduces optimization parameters (one less scale parameter per Gaussian) while better matching linear and planar structures in the environment.

3. **LiDAR Feature Point Initialization**

    - **Function**: Extracts geometric edge points, geometric planar points, and reflectance edge points from the LiDAR point cloud to initialize salient Gaussians
    - **Mechanism**: Computes the smoothness $c_j$ of each point (neighborhood deviation) to classify edges/planes; computes the reflectance gradient $G_j$ (reflectance difference between left and right neighborhoods) to extract reflectance edges
    - **Design Motivation**: Geometric feature points distributed at edges and planes provide structural constraints; reflectance feature points emphasize material differences, compensating for the lack of RGB information in low-texture areas.

4. **Salient Transform**

    - **Function**: Implements adaptive conversion between salient and non-salient Gaussians based on linearity and planarity
    - **Mechanism**: $L(g) = (s_1 - s_2)/s_1$, $P(g) = (s_2 - s_3)/s_1$. When $\max\{L,P\} > \tau_\text{max}$, non-salient $\to$ salient; when $\max\{L,P\} < \tau_\text{min}$, salient $\to$ non-salient
    - **Design Motivation**: Allows salient Gaussians to grow into regions not covered by LiDAR, ensuring sufficient coverage of key structures.

5. **Joint RGB-Reflectance Loss**

    - **Function**: Aligns reflectance and RGB boundaries through gradient direction and magnitude consistency
    - **Mechanism**: $\mathcal{L}_\text{dir} = 1 - \hat{\nabla}F \cdot \hat{\nabla}C^g$ (gradient direction consistency); $\mathcal{L}_\text{val} = \|g_F/F - g_{C^g}/C^g\|_1$ (normalized magnitude consistency)
    - **Design Motivation**: Material boundaries should be consistent in both reflectance and grayscale RGB images—joint constraints reduce boundary blurriness.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{rgb} + \mathcal{L}_{lidar} + \mathcal{L}_{joint}$. $\mathcal{L}_{rgb}$: L1 + D-SSIM photometric loss; $\mathcal{L}_{lidar}$: depth + reflectance + reflectance gradient L1 loss (with mask); $\mathcal{L}_{joint}$: direction consistency + magnitude consistency loss. 30k iterations training.

## Key Experimental Results

### Main Results (Waymo Open Dataset)

| Scene Type | Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|---------|------|-------|-------|--------|
| Dense Traffic | OmniRe | 28.44 | 0.847 | 0.085 |
| | **LR-SGS** | **28.89** | **0.869** | **0.081** |
| High-speed | OmniRe | 28.12 | 0.871 | 0.135 |
| | **LR-SGS** | **28.77** | **0.896** | **0.122** |
| Complex Lighting | OmniRe | 29.33 | 0.727 | 0.278 |
| | **LR-SGS** | **30.51** | **0.755** | **0.236** |

PSNR in complex lighting scenes outperforms OmniRe by **1.18 dB**

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| w/o SG (Salient Gaussian) | 28.74 | 0.830 | 0.152 |
| w/o LF-Init (Feature Point Initialization) | 28.94 | 0.839 | 0.144 |
| w/o Reflectance | 28.87 | 0.831 | 0.147 |
| w/o Joint Loss | 28.96 | 0.835 | 0.144 |
| **Full** | **29.22** | **0.850** | **0.139** |

### Key Findings
- Salient Gaussians achieve 0.9 dB higher PSNR than the baseline at 7k iterations, indicating that feature point initialization accelerates convergence.
- Salient Gaussians reduce the number of Gaussians and shorten training time (61m vs 70m)—due to fewer parameters and stronger structural priors.
- Complex lighting scenes benefit the most (+1.18 dB), as the illumination-invariant property of the reflectance channel provides critical constraints in these scenes.
- Joint loss significantly improves the reconstruction quality of dynamic object edges and material boundaries.

## Highlights & Insights
- **LiDAR Reflectance as an Illumination-Invariant Channel**: Calibrating often-ignored LiDAR intensity information into material properties—a conceptually simple yet effective approach that could be transferred to other multimodal 3D reconstruction tasks.
- **Directional Gaussians + Adaptive Transform**: Salient Gaussians not only reduce parameters but also better fit the scene's topological structures (edges and planes); the salient transform allows natural propagation to non-LiDAR-covered regions.
- **RGB-Reflectance Gradient Consistency**: Instead of simply adding loss terms, cross-modal consistency is designed along direction and magnitude—constraining material boundaries more effectively than simple photometric/depth losses.

## Limitations & Future Work
- LiDAR reflectance calibration assumes a Lambertian reflection model, which is inapplicable to specular reflections (e.g., wet roads, glass).
- Verified only on the Waymo dataset, without evaluating other datasets like nuScenes.
- The thresholds $\tau_\text{max}=0.5, \tau_\text{min}=0.1$ for the salient transform are manually set hyperparameters.
- Robustness of the reflectance channel under varying weather conditions (e.g., rain, fog) has not been analyzed.

## Related Work & Insights
- **vs OmniRe**: OmniRe is a representative method for multi-type Gaussian scene graphs but does not utilize LiDAR reflectance; LR-SGS incorporates the reflectance channel and salient Gaussians into a similar framework.
- **vs StreetGS**: StreetGS uses LiDAR to initialize Gaussians but does not extract feature points, nor does it utilize reflectance information.
- **vs TCLC-GS**: TCLC-GS tightly couples LiDAR and camera but relies on colored meshes and octrees, leading to higher computational costs.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined design of the reflectance channel, salient Gaussians, and joint loss is novel, with each component clearly conceptualized.
- Experimental Thoroughness: ⭐⭐⭐⭐ 24 sequences across 4 kinds of scenarios + detailed ablation + scene editing demos.
- Writing Quality: ⭐⭐⭐⭐ Complete technical details, rich figures, and a clearly structured paper.
- Value: ⭐⭐⭐⭐ Significantly improves reconstruction quality in challenging driving scenarios (complex lighting, high speed).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ReconDreamer: Crafting World Models for Driving Scene Reconstruction via Online Restoration](recondreamer_crafting_world_models_for_driving_scene_reconstruction_via_online_r.md)
- [\[CVPR 2025\] Generative Gaussian Splatting for Unbounded 3D City Generation](generative_gaussian_splatting_for_unbounded_3d_city_generation.md)
- [\[CVPR 2025\] EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis](evolsplat_efficient_volume-based_gaussian_splatting_for_urban_view_synthesis.md)
- [\[ICCV 2025\] GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](../../ICCV2025/autonomous_driving/gs-livm_real-time_photo-realistic_lidar-inertial-visual_mapping_with_gaussian_sp.md)
- [\[AAAI 2026\] LiDAR-GS++: Improving LiDAR Gaussian Reconstruction via Diffusion Priors](../../AAAI2026/autonomous_driving/lidar-gsimproving_lidar_gaussian_reconstruction_via_diffusion_priors.md)

</div>

<!-- RELATED:END -->
