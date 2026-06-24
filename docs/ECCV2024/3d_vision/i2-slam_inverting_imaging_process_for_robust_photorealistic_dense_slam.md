---
title: >-
  [Paper Note] I²-SLAM: Inverting Imaging Process for Robust Photorealistic Dense SLAM
description: >-
  [ECCV 2024][3D Vision][SLAM] Proposed I²-SLAM, which integrates the physical imaging process (motion blur modeling + tone mapping) into a visual SLAM system. Through the joint optimization of an HDR radiance field map, multi-virtual-camera motion blur simulation, and differentiable tone mapping, it reconstructs sharp HDR 3D maps and more accurate camera trajectories from degraded hand-held casual videos.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "SLAM"
  - "Motion Blur"
  - "HDR Reconstruction"
  - "3D Gaussian Splatting"
  - "Neural Radiance Fields"
date: 2026-05-08
content_hash: 0e70dc6d0ce03d08
---

# I²-SLAM: Inverting Imaging Process for Robust Photorealistic Dense SLAM

**Conference**: ECCV 2024  
**arXiv**: [2407.11347](https://arxiv.org/abs/2407.11347)  
**Code**: [https://3d.snu.ac.kr/publications/I2SLAM](https://3d.snu.ac.kr/publications/I2SLAM) (Project Page)  
**Area**: 3D Vision  
**Keywords**: SLAM, Motion Blur, HDR Reconstruction, 3D Gaussian Splatting, Neural Radiance Fields

## TL;DR
Proposed I²-SLAM, which integrates the physical imaging process (motion blur modeling + tone mapping) into a visual SLAM system. Through the joint optimization of an HDR radiance field map, multi-virtual-camera motion blur simulation, and differentiable tone mapping, it reconstructs sharp HDR 3D maps and more accurate camera trajectories from degraded hand-held casual videos.

## Background & Motivation

**Background**: Visual SLAM maps environments to 3D representations and is widely used in VR/AR, robotic navigation, and collision handling. Recently, dense visual SLAM methods based on NeRF and 3D Gaussian Splatting (3DGS) have achieved photorealistic image synthesis. Representative methods include iMAP, NICE-SLAM, Co-SLAM, Point-SLAM, and SplaTAM. These methods achieve good results under ideal inputs.

**Limitations of Prior Work**: Despite many SLAM methods for building visual representations, the vast majority fail to maintain performance in real-world scenes. Casual hand-held videos—the most common inputs for SLAM systems—face two core challenges: (1) motion blur caused by camera movement degrades the images; (2) automatic exposure and white balance lead to appearance inconsistency across frames. These degradations not only reduce map quality but also cumulatively damage pose estimation accuracy, acting as a critical bottleneck for the entire SLAM system.

**Key Challenge**: Existing SLAM systems assume that input images are "ideal" instantaneous exposures at a single pose, whereas real-world images are the result of integration along the camera trajectory over an exposure time period, and undergo a non-linear ISP pipeline. This gap between construction assumptions and physical reality is the root cause of performance degradation. Although 2D image deblurring (such as NAFNet) can preprocess inputs, the deblurred images lack 3D consistency across different views, which may instead introduce new artifacts.

**Goal**: To directly model the physical imaging process within the SLAM framework, enabling the system to recover sharp HDR maps and accurate trajectories from degraded inputs. This is specifically decomposed into: (1) How to model the motion blur process to recover sharp representations? (2) How to handle variation in cross-frame exposure and white balance? (3) How to seamlessly integrate these modules into existing SLAM pipelines?

**Key Insight**: Unlike "restoration-then-mapping", the authors choose to "model degradation during mapping"—embedding the inverse process of image formation into the analysis-of-synthesis optimization loop of SLAM. Motion blurred images are viewed as the integration of images from multiple poses along the trajectory within the exposure time, and appearance changes are decomposed into three differentiable explicit variables: white balance, exposure time, and camera response function.

**Core Idea**: By inverting the physical imaging process (motion blur integration + tone mapping), the HDR linear radiance field is leveraged as the core representation of SLAM, jointly optimizing the scene, poses, and imaging parameters.

## Method

### Overall Architecture
I²-SLAM is a general module that can be integrated into any image-input-based dense visual SLAM pipeline. The input is an RGB/RGB-D video stream that may contain motion blur and appearance changes. The map is represented as an HDR radiance field (linear color space), supporting both NeRF and 3DGS representations. For each frame, the system maintains: start and end poses $\mathbf{T}(t_s), \mathbf{T}(t_e)$ (which together form the motion blur kernel), exposure time $\Delta t$, white balance parameters WB, and camera response function CRF. All variables are jointly optimized by minimizing the discrepancy between the synthesized degraded images and the actual inputs.

### Key Designs

1. **HDR Radiance Field + Motion Blur Simulation**:

    - **Function**: Represent the map as linear HDR radiance values and simulate motion blur via multi-virtual-camera integration.
    - **Mechanism**: The map's color output $\mathbf{c}(\mathbf{T}, \mathbf{p})$ is a linear HDR value. A motion-blurred image is modeled as the integration along the camera trajectory during the exposure time $[t_s, t_e]$: $C_{\text{HDR}}(\mathbf{p}) = \Delta t \cdot \frac{1}{N_{\text{cam}}} \sum_{j=1}^{N_{\text{cam}}} \mathbf{c}(\mathbf{T}(t_j), \mathbf{p})$, where $N_{\text{cam}}=5$ virtual camera poses are linearly interpolated between $t_s$ and $t_e$ (linear interpolation for translation, spherical interpolation Slerp for rotation). The optimized variables include start pose, end pose, and exposure time.
    - **Design Motivation**: The HDR linear color space makes the physical "light intensity superposition" of motion blur physically correct, whereas weighted averaging in the non-linear sRGB space is physically incorrect. Meanwhile, explicitly modeling the blur kernel as pose interpolation rather than a traditional convolution kernel naturally aligns with SLAM's pose optimization framework.

2. **Differentiable Tone Mapping Module**:

    - **Function**: Model the complete mapping process from the HDR radiance field to LDR camera pixel values, handling appearance changes across frames.
    - **Mechanism**: The tone mapping $\Psi_i$ consists of three learnable components: channel-wise white balance $\text{WB}_i$ (channel-wise multiplication with 3 parameters), camera response function $\text{CRF}_i$ (a monotonically increasing function parameterized by a 256-dimensional uniformly sampled grid for each color channel), and dynamic range clipping. Accurate gradient backpropagation through saturated regions is ensured by appending a leaky clipping function ($\alpha=0.01$) to the CRF. Finally, $C_{\text{LDR}} = \text{CRF}(\text{WB}(\Delta t \cdot \mathbf{c}))$. All parameters are learned frame-by-frame.
    - **Design Motivation**: In real scenes, the ISP automatically adjusts exposure, white balance, and tone curves, leading to color inconsistency across frames. For instance, the 3DGS color in SplaTAM is fixed and cannot adapt to brightness changes in different frames, causing severe stitching artifacts. By explicitly modeling the ISP pipeline, the map representation maintains physically consistent HDR values.

3. **Trajectory Regularization and Initialization Strategy**:

    - **Function**: Utilize existing global trajectory information from SLAM to constrain the direction and magnitude of motion blur, stabilizing optimization.
    - **Mechanism**: A trajectory loss $\mathcal{L}_{\text{traj}}$ is designed with two constraints: (1) camera motion direction during exposure should align with the global trajectory—constraining $\mathbf{t}(t_e^{i-1})$ to the linear interpolation of the previous frame's and current frame's center poses; (2) the motion magnitude should be proportional to the exposure time and temporal velocity. A global scale parameter $a$ (related to video frame rate) is introduced and jointly optimized. During initialization, start and end poses of a new frame are extrapolated using the poses of the previous two frames, with a small initial separation distance.
    - **Design Motivation**: Relying solely on image rendering loss to optimize start/end poses easily falls into local optima. SLAM itself maintains global trajectory estimation; using this free prior to regularize the blur motion direction is a unique advantage of SLAM—something static 3D reconstruction methods lack.

### Loss & Training
The total loss is a weighted sum of three parts: $\mathcal{L} = \lambda_{\text{img}} \mathcal{L}_{\text{img}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}} + \lambda_{\text{traj}} \mathcal{L}_{\text{traj}}$. The image loss uses the L1 distance to compare the synthesized LDR image with the actual input. The depth loss renders the depth error using the pose that minimizes depth error during the exposure period (assuming the depth sensor captures at an instantaneous moment). RGB-SLAM uses the poses from DROID-SLAM as initial values; RGBD-SLAM is deployed on SplaTAM.

## Key Experimental Results

### Main Results (RGB-SLAM, based on NeRF-SLAM†)

| Dataset | Metric | I²-SLAM | NeRF-SLAM† | Gain |
|--------|------|---------|------------|------|
| Synthetic Dataset (Avg) | PSNR | 28.89 | 26.70 | +2.19dB |
| Synthetic Dataset (Avg) | SSIM | 0.887 | 0.842 | +0.045 |
| TUM-RGBD fr3/office | ATE-RMSE | 1.95cm | 7.13cm | 3.6x |
| Synthetic Dataset SP | ATE-RMSE | 1.50cm | 3.97cm | 2.6x |

### Ablation Study (ScanNet RGBD)

| Configuration | ATE-RMSE | PSNR | SSIM | LPIPS |
|------|----------|------|------|-------|
| Full (Motion Blur + HDR + Trajectory Reg.) | **2.56** | **25.62** | **0.801** | **0.195** |
| w/o Motion Blur Modeling | 2.66 | 24.80 | 0.769 | 0.203 |
| w/o HDR Map | 2.60 | 22.39 | 0.756 | 0.226 |
| w/o HDR + w/o Motion Blur | 2.63 | 22.36 | 0.755 | 0.228 |
| All removed (baseline) | 2.71 | 23.05 | 0.793 | 0.235 |

### Key Findings
- HDR map brings the largest improvement in rendering quality—contributing about 3dB out of the PSNR increase from 22.36 to 25.62.
- Motion blur modeling primarily improves tracking accuracy—lowering ATE-RMSE from 2.66 to 2.56, while also enhancing rendering quality.
- Trajectory regularization is crucial for stabilizing multi-camera optimization, affecting both tracking and mapping.
- The advantages of I²-SLAM are most significant in scenes with drastic appearance changes (ScanNet 0785-00, synthetic dataset SP).
- I²-SLAM-S running with only 20% iterations still outperforms the full-iteration SplaTAM with similar running time.
- 2D deblurring preprocessing (NAFNet + NeRF-SLAM) shows limited effectiveness due to the lack of multi-view consistency.

## Highlights & Insights
- **Physically Correct Inverse Imaging Process**: Rather than "patching" degradations retrospectively, it directly models the root cause of degradation in the optimization objective. This makes motion blur a constraint rather than noise—blurred images actually provide trajectory information. This "analysis-by-synthesis" philosophy is highly elegant.
- **Crucial Role of HDR Linear Color Space**: Linear HDR not only makes the additive superposition of motion blur physically correct but also simplifies the modeling of appearance variations—all non-linear factors are concentrated into the differentiable tone mapping module. This design decomposes complex problems very cleanly.
- **General Modular Design**: As a pluggability module, I²-SLAM has been validated on both NeRF-based and 3DGS-based SLAM, exhibiting compatibility with different scene representations. This design can be transferred to future new SLAM methods.

## Limitations & Future Work
- Multi-virtual-camera simulation of motion blur introduces significant computational overhead—rendering time and optimization time increase by approximately 4-5x, respectively.
- The discrete approximation using 5 virtual cameras may be insufficient for extremely blurred scenes. The authors' ablation study indicates monotonic improvements with more cameras, but with diminishing marginal returns.
- Only linear motion blur is handled (constant velocity assumption), which may have limited effectiveness for blur dominated by rotation or accelerated motion.
- Currently, CRF is modeled as a monotonically increasing function, which may be less flexible for automatic white balance corrections of extreme purple or red color casts.
- The motion blur of depth sensors is not explicitly modeled (assuming acquisition at an instantaneous moment), but ToF depth sensors also suffer from errors during rapid motion.

## Related Work & Insights
- **vs. BAD-NeRF**: Also models blur using multi-pose integration, but BAD-NeRF is a static reconstruction method not embedded in a SLAM framework. I²-SLAM utilizes the global trajectory of SLAM for regularization, which is an advantage BAD-NeRF lacks.
- **vs. SplaTAM**: An RGB-D SLAM based on 3DGS, where the color representation is fixed and cannot handle brightness inconsistency artifacts caused by appearance changes. I²-SLAM resolves this issue via its tone mapping module.
- **vs. NeRF-SLAM**: Employs DROID-SLAM for tracking and NeRF for mapping, but lacks modeling of degradations. I²-SLAM introduces the inverse imaging process on top of it, significantly improving performance in both aspects.
- **vs. HDR-NeRF/HDR-Plenoxels**: Models HDR in static reconstruction, but does not address motion blur, nor is it within a SLAM framework.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically integrates the inverse imaging process into SLAM, with every component design physically motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Synthetic + real datasets, RGB + RGBD modes, comparisons with multiple baselines, and highly detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematical derivations, with smooth transitions connecting physical modeling and SLAM integration.
- Value: ⭐⭐⭐⭐ Solves key pain points in physical deployment, and the modular design ensures ease of integration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SGS-SLAM: Semantic Gaussian Splatting for Neural Dense SLAM](sgs-slam_semantic_gaussian_splatting_for_neural_dense_slam.md)
- [\[ECCV 2024\] CG-SLAM: Efficient Dense RGB-D SLAM in a Consistent Uncertainty-Aware 3D Gaussian Field](cg-slam_efficient_dense_rgb-d_slam_in_a_consistent_uncertainty-aware_3d_gaussian.md)
- [\[CVPR 2026\] AERGS-SLAM: Auto-Exposure-Robust Stereo 3D Gaussian Splatting SLAM](../../CVPR2026/3d_vision/aergs-slam_auto-exposure-robust_stereo_3d_gaussian_splatting_slam.md)
- [\[CVPR 2026\] ODGS-SLAM: Omnidirectional Gaussian Splatting SLAM](../../CVPR2026/3d_vision/odgs-slam_omnidirectional_gaussian_splatting_slam.md)
- [\[ECCV 2024\] Deep Patch Visual SLAM](deep_patch_visual_slam.md)

</div>

<!-- RELATED:END -->
