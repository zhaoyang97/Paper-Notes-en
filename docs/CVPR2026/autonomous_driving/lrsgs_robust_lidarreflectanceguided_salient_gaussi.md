---
title: >-
  [Paper Note] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction
description: >-
  [CVPR 2026][Autonomous Driving][3D Gaussian Splatting] This paper proposes LR-SGS, which calibrates LiDAR intensity into an illumination-invariant reflectance channel attached to 3D Gaussians, and introduces a structure-aware Salient Gaussian representation (initialized from LiDAR geometry and reflectance feature points) with improved densification control and a salient transform strategy. LR-SGS achieves higher-fidelity reconstruction than OmniRe on complex Waymo autonomous driving scenes while using fewer Gaussians and requiring less training time.
tags:
  - CVPR 2026
  - Autonomous Driving
  - 3D Gaussian Splatting
  - LiDAR reflectance
  - salient Gaussian
  - self-driving reconstruction
  - multi-modal fusion
date: 2026-05-08
content_hash: 301aec467fe43026
---

# LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2603.12647](https://arxiv.org/abs/2603.12647)
**Code**: None (not provided)
**Area**: 3D Vision / Autonomous Driving / 3DGS / Scene Reconstruction
**Keywords**: 3D Gaussian Splatting, LiDAR reflectance, salient Gaussian, self-driving reconstruction, multi-modal fusion

## TL;DR
This paper proposes LR-SGS, which calibrates LiDAR intensity into an illumination-invariant reflectance channel attached to 3D Gaussians, and introduces a structure-aware Salient Gaussian representation (initialized from LiDAR geometry and reflectance feature points) with improved densification control and a salient transform strategy. LR-SGS achieves higher-fidelity reconstruction than OmniRe on complex Waymo autonomous driving scenes while using fewer Gaussians and requiring less training time.

## Background & Motivation
3DGS-based methods have demonstrated promising results for autonomous driving scene reconstruction and novel view synthesis, but face two major challenges: (1) camera-only approaches are prone to texture inconsistency and optimization instability under complex illumination and high-speed ego-motion; (2) existing LiDAR+3DGS methods (e.g., StreetGS, OmniRe, PVG) use LiDAR only for initialization or depth supervision, without fully exploiting the reflectance information embedded in point clouds or the complementarity between LiDAR and RGB. In particular, the absence of stable constraints in low-texture regions and at material boundaries degrades reconstruction quality. LiDAR reflectance is correlated with object material properties and is approximately illumination-invariant, making it a natural complementary signal to RGB.

## Core Problem
How to fully leverage LiDAR geometric and reflectance information to enhance 3DGS-based autonomous driving scene reconstruction. This encompasses three sub-problems: (1) how to guide Gaussian distribution using LiDAR structural information (edge points, planar points, reflectance gradient edge points); (2) how to exploit illumination-invariant reflectance signals to stabilize reconstruction under complex lighting; (3) how to enforce consistency between RGB and reflectance at material boundaries.

## Method

### Overall Architecture
The input consists of RGB image sequences and LiDAR point cloud sequences. The scene is represented as a 3DGS scene graph (background node + dynamic object nodes + sky node). The initial Gaussians comprise two components: Salient Gaussians extracted from LiDAR feature points and Non-Salient Gaussians initialized from SfM points. Each Gaussian carries an additional reflectance channel alongside standard attributes. Alpha-blending rendering produces three maps — Color, Depth, and Reflectance — which are jointly optimized via Color Loss + LiDAR Loss + Joint Loss.

### Key Designs
1. **LiDAR Intensity Calibration to Reflectance**: Raw LiDAR intensity is affected by range and incidence angle ($I = \eta \cdot \rho \cdot \cos\alpha / R^2$). Dividing by $R^2$ and $\cos\alpha$ (estimated from the angle between the surface normal and laser direction) yields an approximately illumination-invariant reflectance $\rho$. The reflectance is projected onto the camera plane to obtain a sparse reflectance image $F_{gt}$, from which a reflectance gradient map $F'_{gt}$ is computed (reflectance difference between adjacent pixels divided by 3D distance).

2. **Salient Gaussian Representation**: Designed to capture the abundant edge and planar structures in autonomous driving scenes. Each Salient Gaussian has a principal direction $d_{spec}$; the remaining two directions share a single scale parameter — edge-type Gaussians extend along the maximum-scale direction ($\Sigma = R \cdot \text{diag}(\sigma^2_\parallel, \sigma^2_\perp, \sigma^2_\perp) \cdot R^T$), while planar-type Gaussians are flattened along the minimum-scale direction. This reduces the number of optimizable scale parameters (from three to two) while preserving expressiveness for edges and planes.

3. **LiDAR Feature Point Initialization**: Three categories of feature points are extracted from LiDAR: (a) geometric edge points — high-curvature points selected by computing neighborhood smoothness $c_j$; (b) geometric planar points — low-curvature points; (c) reflectance edge points — selected by computing reflectance gradients $G_j$ over left and right neighborhoods within the same scan ring. Geometric edge + reflectance edge points initialize Edge Salient Gaussians; geometric planar points initialize Planar Salient Gaussians; SfM points initialize Non-Salient Gaussians.

4. **Improved Densification and Salient Transform**: During splitting, Edge Salient Gaussians split along their principal direction, while Planar Salient Gaussians split along the orthogonal plane. A salient transform strategy is introduced: linearity $L(g)=(s_1-s_2)/s_1$ and planarity $P(g)=(s_2-s_3)/s_1$ are defined. A Non-Salient Gaussian is promoted to Salient if $\max\{L,P\} > \tau_{max}$ holds for two consecutive evaluations; a Salient Gaussian is demoted to Non-Salient if $\max\{L,P\} < \tau_{min}$ for two consecutive evaluations. $\tau_{max}=0.5$, $\tau_{min}=0.1$.

5. **RGB–LiDAR Reflectance Joint Loss**: After converting RGB to grayscale, both the grayscale image and the reflectance map are processed with Gaussian smoothing followed by the Scharr operator to compute gradients. Directional consistency loss: $\mathcal{L}_{dir} = 1 - \hat{\nabla}F \cdot \hat{\nabla}C_g$ (dot product of gradient directions); magnitude consistency loss: $\mathcal{L}_{val} = \|g_F/F - g_{C_g}/C_g\|_1$ (normalized magnitude difference). This enforces gradient alignment between RGB and reflectance at material boundaries, sharpening boundary reconstruction.

### Loss & Training
Total loss: $\mathcal{L} = \mathcal{L}_{rgb} + \mathcal{L}_{lidar} + \mathcal{L}_{joint}$
- $\mathcal{L}_{rgb} = (1-\lambda_c)\mathcal{L}_1 + \lambda_c \mathcal{L}_{D-SSIM}$, $\lambda_c=0.2$
- $\mathcal{L}_{lidar} = \lambda_{depth}\mathcal{L}_{depth} + \lambda_{fle}\mathcal{L}_{fle} + \lambda_{fle'}\mathcal{L}_{fle'}$ (depth L1 + reflectance L1 + reflectance gradient L1)
- $\mathcal{L}_{joint} = \lambda_{dir}\mathcal{L}_{dir} + \lambda_{val}\mathcal{L}_{val}$
- Weights: $\lambda_{depth}=\lambda_{fle}=\lambda_{dir}=0.1$, $\lambda_{val}=0.2$, $\lambda_{fle'}=0.05$
- Training: 30k iterations, input resolution 1066×1600, results averaged over 5 runs

## Key Experimental Results

| Dataset/Scene | Metric | LR-SGS (Ours) | OmniRe | StreetGS | Gain |
|---|---|---|---|---|---|
| Dense Traffic | PSNR↑ | 28.89 | 28.44 | 27.01 | +0.45 vs OmniRe |
| Dense Traffic | PSNR* (dynamic) | 24.02 | 23.72 | 21.73 | +0.30 vs OmniRe |
| High-Speed | PSNR↑ | 28.77 | 28.12 | 28.06 | +0.65 vs OmniRe |
| High-Speed | SSIM↑ | 0.896 | 0.871 | 0.878 | +0.025 vs OmniRe |
| Complex Lighting | PSNR↑ | 30.51 | 29.33 | 29.16 | **+1.18 vs OmniRe** |
| Static | PSNR↑ | 28.73 | 28.23 | 28.15 | +0.50 vs OmniRe |
| Efficiency | # Gaussians | 2,510,883 | 2,744,275 | 2,929,851 | −8.5% vs OmniRe |
| Efficiency | Training Time | 59m25s | 67m11s | 64m30s | −11.5% vs OmniRe |
| Efficiency | FPS | 36.95 | 30.55 | 33.61 | +21% vs OmniRe |

### Ablation Study
- **Salient Gaussian**: Removing SG decreases PSNR from 29.22 to 28.74 (−0.48); SG also reduces training time (61m57s vs. 70m21s at 30k iters).
- **LiDAR Feature Point Initialization**: Removing LF-Init decreases PSNR from 29.22 to 28.94 (−0.28); the gap is more pronounced at 7k iterations (26.41 vs. 25.52).
- **Reflectance Channel**: Removing reflectance decreases PSNR from 29.22 to 28.87 and SSIM from 0.850 to 0.831, with the most significant degradation under complex lighting conditions.
- **Joint Loss**: Removing it decreases PSNR from 30.39 to 30.08, while reflectance RMSE increases from 0.0854 to 0.1063; vehicle contours and crosswalk markings become blurred.
- All components contribute positively; Salient Gaussian yields the largest individual gain (+0.48 PSNR).

## Highlights & Insights
- **Reflectance as an illumination-invariant channel**: LiDAR intensity is elegantly calibrated into a material-dependent reflectance signal, providing stable supervision for 3DGS beyond RGB — particularly effective in nighttime and complex lighting scenarios.
- **Structure-aware design of Salient Gaussians**: Reducing the number of scale parameters (two instead of three) paradoxically improves the ability to fit edges and planes while reducing redundant Gaussians.
- **Salient transform strategy**: Bidirectional promotion and demotion between Non-Salient and Salient states enables Salient Gaussians to emerge from SfM points even in LiDAR-uncovered regions.
- **RGB–Reflectance Joint Loss**: Cross-modal consistency enforced via gradient direction and magnitude alignment achieves more precise boundary sharpening than simple pixel-level constraints.

## Limitations & Future Work
- Validation is limited to the Waymo dataset; generalization to other autonomous driving datasets (e.g., nuScenes) remains unverified.
- Reflectance calibration relies on normal estimation from neighboring points, which may be inaccurate in sparse point cloud regions.
- Scene editing (object replacement/removal) is demonstrated only qualitatively, without quantitative evaluation.
- $\tau_{max}$ and $\tau_{min}$ are manually set hyperparameters; adaptive thresholds may be preferable.
- Real-time performance on embedded platforms is not discussed despite a reasonably good FPS of 36.95.
- The effect of adverse weather conditions (e.g., rain, snow, fog) on LiDAR reflectance is not considered.

## Related Work & Insights
- **vs. OmniRe**: OmniRe is a multi-type Gaussian scene graph method that uses LiDAR for initialization and depth supervision but does not exploit reflectance. It underperforms LR-SGS by 1.18 dB PSNR in the Complex Lighting scenario. LR-SGS achieves superior results with fewer Gaussians and shorter training time.
- **vs. StreetGS**: StreetGS uses LiDAR to initialize 3DGS for both background and dynamic objects but lacks structure-aware Gaussian design. LR-SGS outperforms it across all scene types, especially for dynamic object reconstruction in Dense Traffic (PSNR* 24.02 vs. 21.73).
- **vs. TCLC-GS**: TCLC-GS relies on LiDAR mesh and octree feature initialization, resulting in a more complex pipeline.

**Broader Implications**:
- The Salient Gaussian concept could be combined with occupancy prediction, using structure-aware Gaussians as geometric priors for 3D occupancy.
- The reflectance channel could be extended to multi-spectral modalities (e.g., infrared) to further enrich autonomous driving scene representations.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of reflectance channel, Salient Gaussians, and salient transform is proposed for the first time in autonomous driving 3DGS.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four scene categories (Dense Traffic / High-Speed / Complex Lighting / Static), complete ablation study, and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, complete mathematical derivations, and rich figures (intuitive qualitative comparisons).
- Value: ⭐⭐⭐⭐ Contributes a novel multi-modal fusion paradigm to the autonomous driving 3DGS field with practical value for simulation data generation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling](f3dgs_federated_3d_gaussian_splatting_for_decentralized_multi-agent_world_modeli.md)
- [\[CVPR 2026\] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](towards_balanced_multimodal_learning_in_3d_human_p.md)
- [\[CVPR 2026\] BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird's-Eye View Images](bev-sld_self-supervised_scene_landmark_detection_for_global_localization_with_li.md)
- [\[CVPR 2026\] Efficient Equivariant Transformer for Self-Driving Agent Modeling](efficient_equivariant_transformer_for_self-driving_agent_modeling.md)
- [\[CVPR 2026\] TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR](terraseg_self-supervised_ground_segmentation_for_any_lidar.md)

<!-- RELATED:END -->
