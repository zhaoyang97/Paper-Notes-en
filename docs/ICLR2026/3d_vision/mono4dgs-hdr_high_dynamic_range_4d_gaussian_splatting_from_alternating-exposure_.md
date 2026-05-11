---
title: >-
  [Paper Note] Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos
description: >-
  [ICLR 2026][3D Vision][4D Gaussian Splatting] This work is the first to address the problem of reconstructing renderable 4D HDR scenes from pose-free alternating-exposure monocular videos. Through a two-stage optimization pipeline (orthographic video space → world space), a Video-to-World Gaussian transformation strategy, and temporal luminance regularization, the method achieves 37.64 dB HDR PSNR and 161 FPS on synthetic data, comprehensively outperforming existing approaches.
tags:
  - ICLR 2026
  - 3D Vision
  - 4D Gaussian Splatting
  - HDR
  - monocular video
  - alternating exposure
  - dynamic scene
date: 2026-05-08
content_hash: aa16a920fd62b3f9
---

# Mono4DGS-HDR: High Dynamic Range 4D Gaussian Splatting from Alternating-exposure Monocular Videos

**Conference**: ICLR 2026
**arXiv**: [2510.18489](https://arxiv.org/abs/2510.18489)
**Code**: [https://liujf1226.github.io/Mono4DGS-HDR](https://liujf1226.github.io/Mono4DGS-HDR)
**Area**: 3D Vision / HDR Reconstruction
**Keywords**: 4D Gaussian Splatting, HDR, monocular video, alternating exposure, dynamic scene

## TL;DR
This work is the first to address the problem of reconstructing renderable 4D HDR scenes from pose-free alternating-exposure monocular videos. Through a two-stage optimization pipeline (orthographic video space → world space), a Video-to-World Gaussian transformation strategy, and temporal luminance regularization, the method achieves 37.64 dB HDR PSNR and 161 FPS on synthetic data, comprehensively outperforming existing approaches.

## Background & Motivation

**Background**: 4D dynamic scene reconstruction (particularly 3DGS-based) has seen significant progress, and HDR reconstruction methods (GaussHDR, HDR-HexPlane) have also been proposed. However, the combination of the two—performing 4D HDR reconstruction from monocular alternating-exposure video—remains unaddressed.

**Limitations of Prior Work**: (a) Alternating exposure causes adjacent frames to differ in brightness, causing standard photometric reprojection to fail and making camera pose estimation infeasible; (b) 2D priors (tracking, depth, optical flow) are noisy across frames with varying brightness; (c) existing dynamic methods (SplineGS, MoSca) are designed for constant-exposure video and perform poorly when naively extended with an HDR head; (d) HDR methods (GaussHDR) require known poses and multi-view inputs.

**Key Challenge**: Alternating exposure → unreliable pose estimation → geometric instability → inconsistent HDR appearance. A vicious cycle.

**Key Insight**: The proposed approach first trains "video Gaussians" in orthographic camera coordinate space (bypassing pose estimation) to recover HDR training frames, then uses the recovered HDR frames for photometric reprojection to estimate poses, and finally transforms to world space for joint optimization.

**Core Idea**: Two-stage decoupling—first solve HDR (in orthographic space), then solve for pose and 3D geometry (in world space), bridging the two stages via a Video-to-World transformation.

## Method

### Overall Architecture
Alternating-exposure monocular video → Preprocessing (DepthCrafter depth + SpatialTracker trajectories + RAFT optical flow) → **Stage 1**: Train video Gaussians in orthographic space (4K iterations) to recover HDR training frames → **Video-to-World Transformation**: dynamic/static decomposition + position/rotation/scale transformation → **Stage 2**: Joint optimization of Gaussians and poses in world space (11K iterations) → Output renderable 4D HDR scene.

### Key Designs

1. **Orthographic-space Video Gaussians (Stage 1)**:

    - Function: Train fully dynamic Gaussians under an orthographic projection model, where $(x^v, y^v) \in [-1,1]^2$ are normalized pixel coordinates and $z^v$ is depth.
    - Mechanism: Orthographic projection avoids the need for camera intrinsics and extrinsics required by perspective projection, eliminating the dependency on prior pose knowledge.
    - Design Motivation: Pose estimation is unreliable under alternating exposure; performing HDR recovery in a "pose-free" space is therefore more robust.

2. **Video-to-World Gaussian Transformation**:

    - Function: Transfers video-space Gaussians to world space, using a dynamic mask and occlusion detection to separate dynamic and static components.
    - Novelty: **2D covariance-invariant scale refitting** — solves for the world-space scale $S^w$ such that the projected 2D Gaussian shape is consistent with that in video space: $\min_{S^w} \sum_t \|\Sigma'^v_t - \Sigma'^w_t\|_2$
    - Design Motivation: Directly inheriting video-space scales to world space distorts the Gaussian shapes.

3. **Temporal Luminance Regularization (TLR)**:

    - Function: Ensures temporal consistency of HDR appearance.
    - Mechanism: Adjacent-frame HDR images are aligned via rendered optical flow warping, and inconsistencies are penalized using a normalized difference: $\mathcal{L}_{tlr} = |V \odot \frac{\tilde{H}_{t-1 \to t} - \tilde{H}_t}{\tilde{H}_{t-1 \to t} + \tilde{H}_t}|_1$
    - Design Motivation: Direct supervision cannot guarantee temporal consistency in the HDR domain given the brightness discrepancy between alternating-exposure frames; normalization removes the influence of absolute HDR irradiance scale.
    - Effect: TLR primarily affects TAE (temporal consistency) rather than PSNR — without TLR, TAE degrades from 0.057 to 0.071.

4. **HDR Photometric Reprojection Loss**:

    - Function: In Stage 2, uses Stage 1's recovered HDR frames for photometric reprojection to jointly optimize poses and world-space Gaussians.
    - Design Motivation: Standard photometric reprojection fails under alternating exposure, but consistent luminance in the HDR domain makes it feasible.

### Loss & Training
$$\mathcal{L} = \lambda_{rgb}\mathcal{L}_{rgb} + \lambda_{ue}\mathcal{L}_{ue} + \lambda_{dep}\mathcal{L}_{dep} + \lambda_{track}\mathcal{L}_{track} + \lambda_{arap}\mathcal{L}_{arap} + \lambda_{vel}\mathcal{L}_{vel} + \lambda_{acc}\mathcal{L}_{acc} + \lambda_{tlr}\mathcal{L}_{tlr} + \lambda_{pr}\mathcal{L}_{pr}$$

Dynamic representation: cubic Hermite splines (position) + cubic polynomials (rotation). Total training time is approximately 1.5 hours.

## Key Experimental Results

### Main Results

| Method | Syn-Exp-3 HDR PSNR↑ | Syn-Exp-3 TAE↓ | FPS↑ |
|------|-------------------|---------------|------|
| GaussHDR† | 31.25 | 0.089 | 51 |
| HDR-HexPlane† | 29.60 | 0.155 | 1 |
| MoSca-HDR‡ | 36.89 | 0.059 | 82 |
| **Mono4DGS-HDR** | **37.64** | **0.057** | **161** |

| Method | Real-Exp-2 PSNR↑ | Real-Exp-2 TAE↓ | Real-Exp-3 PSNR↑ | Real-Exp-3 TAE↓ |
|------|-----------------|----------------|-----------------|----------------|
| MoSca-HDR‡ | 30.28 | 0.054 | 27.23 | 0.076 |
| **Mono4DGS-HDR** | **31.82** | **0.046** | **27.65** | **0.067** |

### Ablation Study

| Configuration | Syn-Exp-3 HDR PSNR | Syn-Exp-3 TAE |
|------|-------------------|---------------|
| w/o Video Gaussian Init | 36.07 (-1.57) | 0.057 |
| w/o Occlusion Handling | 37.22 (-0.42) | 0.059 |
| w/o 2D Covariance Invariance | 37.25 (-0.39) | 0.057 |
| w/o TLR | 37.58 (-0.06) | **0.071** (+0.014) |
| **Full Model** | **37.64** | **0.057** |

### Key Findings
- **Video Gaussian Init is the most critical component**: Removing it causes HDR PSNR to drop by 1.57 dB, confirming that the two-stage strategy is the cornerstone of the method.
- **TLR is crucial for temporal consistency**: Its impact on PSNR is minimal (-0.06), but TAE degrades by 24.6% without it.
- **Superior rendering speed**: 161 FPS, approximately 2× faster than MoSca-HDR (82 FPS).
- Constant-exposure methods such as SplineGS and GFlow perform extremely poorly when naively extended with an HDR head (PSNR 17.59 / failure).

## Highlights & Insights
- **The wisdom of two-stage decoupling**: HDR recovery and 3D reconstruction are decoupled—HDR is first resolved in a simpler space (orthographic/pose-free), and the resulting consistent luminance is then leveraged to facilitate 3D reconstruction (pose estimation). This mutual-assistance strategy breaks the vicious cycle introduced by alternating exposure.
- **2D covariance invariance**: During the Video-to-World Gaussian transformation, enforcing invariance of the projected 2D shape is a concise yet essential constraint for preventing geometric distortion.
- **Normalized difference for temporal regularization**: $\frac{H_1 - H_2}{H_1 + H_2}$ eliminates the influence of absolute HDR scale and focuses solely on relative changes, making it applicable across arbitrary dynamic ranges.

## Limitations & Future Work
- The method relies on multiple preprocessing models (DepthCrafter + SpatialTracker + RAFT), resulting in a complex pipeline that may introduce cascading errors.
- Only 2–3 alternating exposure patterns are supported; more complex strategies such as random exposure scheduling remain unexplored.
- A training time of 1.5 hours is still substantial, limiting applicability in real-time scenarios.
- HDR ground truth is available for evaluation on synthetic data, but real-world scenes lack HDR ground truth, leaving only LDR metrics for assessment.

## Related Work & Insights
- **vs. GaussHDR**: GaussHDR requires known poses and multi-view inputs and addresses only static HDR reconstruction; Mono4DGS-HDR handles dynamic scenes with monocular, pose-free input.
- **vs. MoSca-HDR**: MoSca is designed for constant-exposure video with an HDR head appended, which is less effective than the purpose-built two-stage strategy proposed in this work.
- **vs. HDR-HexPlane**: An NeRF-based dynamic HDR method rendering at only 1 FPS; this work achieves 161 FPS.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pioneering work that addresses the novel problem of 4D HDR reconstruction from alternating-exposure monocular video.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 25 scenes (synthetic + real), multiple exposure patterns, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The method is complex but clearly described, with rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ Addresses a core practical challenge in HDR video reconstruction with real-time rendering at 161 FPS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dynamic Novel View Synthesis in High Dynamic Range](dynamic_novel_view_synthesis_in_high_dynamic_range.md)
- [\[ICLR 2026\] HDR-NSFF: High Dynamic Range Neural Scene Flow Fields](hdr-nsff_high_dynamic_range_neural_scene_flow_fields.md)
- [\[ICLR 2026\] Uncertainty Matters in Dynamic Gaussian Splatting for Monocular 4D Reconstruction](uncertainty_matters_in_dynamic_gaussian_splatting_for_monocular_4d_reconstructio.md)
- [\[CVPR 2026\] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](../../CVPR2026/3d_vision/learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)
- [\[CVPR 2026\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](../../CVPR2026/3d_vision/instanthdr_singleforward_gaussian_splatting_for_hi.md)

</div>

<!-- RELATED:END -->
