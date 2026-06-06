---
title: >-
  [Paper Note] VoluMe: Authentic 3D Video Calls from Live Gaussian Splat Prediction
description: >-
  [ICCV 2025][3D Vision][3D video calling] Microsoft proposes the first method for real-time prediction of 3D Gaussian Splatting reconstructions from a monocular 2D camera…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D video calling"
  - "Gaussian Splatting"
  - "monocular reconstruction"
  - "real-time 3D reconstruction"
  - "digital humans"
  - "video conferencing"
date: 2026-05-08
content_hash: 4e5b9fbef0e86173
---

# VoluMe: Authentic 3D Video Calls from Live Gaussian Splat Prediction

**Conference**: ICCV 2025
**arXiv**: [2507.21311](https://arxiv.org/abs/2507.21311)  
**Area**: 3D Vision
**Keywords**: 3D video calling, Gaussian Splatting, monocular reconstruction, real-time 3D reconstruction, digital humans, video conferencing

## TL;DR

Microsoft proposes the first method for real-time prediction of 3D Gaussian Splatting reconstructions from a monocular 2D camera, simultaneously satisfying four requirements: authenticity, realism, liveness, and temporal stability. This enables anyone to conduct volumetric 3D video calls using only a standard laptop camera.

## Background & Motivation

### State of the Field

**Background**: 3D virtual meetings hold promise for enhancing the sense of co-presence and engagement in remote communication, yet existing 3D representation approaches each suffer from fundamental drawbacks:

**Complex hardware solutions** (e.g., Google Project Starline): require multi-camera arrays, dedicated sensors, and substantial compute, making them prohibitively expensive.

**Avatar-based solutions** (e.g., Meta Codec Avatars): require prior enrollment to model the user's appearance and cannot adapt in real time to current clothing, hairstyle, glasses, or other changes.

**Generative model solutions** (e.g., Live 3D Portrait / TriPlaneNet): rely on GAN inversion via EG3D and are constrained by the model's training distribution, yielding insufficient texture and geometric fidelity.

**Four core requirements for video calling**:

### Limitations of Prior Work

**Limitations of Prior Work**: **Authenticity**: Rendering from the original camera viewpoint must faithfully reproduce the input video, including every detail such as glasses, accessories, and hairstyle.

### Root Cause

**Key Challenge**: **Realism**: Generating plausible and photorealistic 3D reconstructions under novel viewpoints.

### Starting Point

**Key Insight**: **Liveness**: Running in real time on consumer-grade devices.

### Remarks

**Remarks**: **Stability**: No flickering across temporal sequences or viewpoint changes.

No existing method satisfies all four requirements simultaneously. Authenticity is particularly challenging—avatar-based methods rely on a fixed appearance captured during a past enrollment session and cannot reflect the user's state at the present moment.

## Method

### Overall Architecture

Building on the Splatter Image architecture, the method takes a single 2D image as input and regresses 3D Gaussian parameters for each pixel via a U-Net, directly outputting a complete 3D Gaussian Splatting scene representation. Compared to the original Splatter Image, four key improvements are introduced.

### Key Design 1: Planar Homography Correction

In video calls, faces frequently appear near image borders, introducing severe perspective distortion. The method applies a planar homography transform to align the face to a frontal, centered position before feeding it into the network, eliminating the impact of perspective distortion on reconstruction quality.

### Key Design 2: Distance-Adaptive Scaling

The distance between the user and the camera varies considerably. A scaling step is incorporated during training that estimates distance from the size of the face detection bounding box and adjusts the depth and scale of the predicted Gaussians accordingly, ensuring correct reconstruction at both near and far distances.

### Key Design 3: Dual Gaussian Prediction

Each pixel outputs two 3D Gaussian kernels instead of one, significantly improving reconstruction quality. The first Gaussian captures the primary surface, while the second captures sub-surface details such as semi-transparency in hair or reflections from lens surfaces.

### Key Design 4: Stability Loss

Frame-by-frame independent prediction leads to temporal flickering. A stability loss is introduced: the 3D Gaussian reconstructions from two consecutive frames are rendered from multiple viewpoints, and inter-frame differences are computed and penalized to suppress unnecessary temporal variation. This enables the network to produce temporally consistent reconstructions while preserving authenticity.

### Loss & Training

- **Reconstruction loss**: Gaussians are rendered from multiple training viewpoints and compared against ground-truth images using $\ell_1$ + SSIM + LPIPS losses.
- **Stability loss**: Minimization of rendering differences between consecutive frames.
- **Training on synthetic data only**: The model is trained exclusively on synthetically rendered face data but generalizes well to real images through the skip connections of the feed-forward network.

### Network Architecture

A lightweight U-Net: a smaller backbone than the original SongUNet, with only two convolutional layers per resolution level, five resolution levels to maintain a large receptive field, and self-attention layers removed. Real-time inference at 30 FPS is preserved.

## Key Experimental Results

### Visual Quality

The paper reports state-of-the-art PSNR and LPIPS metrics on the Ava256 and Cafca datasets:
- Outperforms all existing methods on both PSNR and LPIPS.
- Achieves the best temporal jitter metric on Ava256.
- Substantially surpasses avatar-based methods in authenticity when rendering from the original camera viewpoint.

### Authenticity Comparison

In qualitative comparisons with Live 3D Portrait and TriPlaneNet (both EG3D-based), the proposed method faithfully reproduces all details present in the input frame—glasses, hats, diverse hairstyles, etc.—whereas generative model-based methods, constrained by their training distribution, frequently exhibit eye-tracking artifacts and texture distortions.

### Real-Time System Demonstration

A complete one-to-one 3D video calling system is demonstrated:
- **Input**: Standard 2D RGB camera
- **Output**: 3D display with motion parallax
- **Frame rate**: 30 FPS
- **Device**: Consumer-grade PC
- **Viewpoint range**: ±40° from frontal input view

### Key Findings

- A model trained purely on synthetic data generalizes well to real images—skip connections in the feed-forward network are the critical factor.
- Dual Gaussians substantially improve quality over a single Gaussian per pixel (independently validated by concurrent work).
- The stability loss is essential for video sequences, markedly reducing temporal flickering.
- Full 360° reconstruction is unnecessary—the video calling scenario requires only ±55° around the frontal view.

## Highlights & Insights

1. **Formalization of "authenticity"**: The paper formally establishes authenticity as a core requirement for 3D video calling, drawing a sharp contrast with generative approaches—users expect to see *themselves as they are now*, not a pre-modeled avatar.
2. **Extreme pragmatism**: Every design decision serves real-world deployment (lightweight U-Net, removal of self-attention, synthetic data training to avoid privacy concerns).
3. **Successful generalization from synthetic data**: Through skip connections in the feed-forward architecture and direct sampling, input information flows directly into the output representation, circumventing the reconstruction bias inherent in generative models.
4. **Precise problem scoping**: Rather than pursuing 360° reconstruction, the paper accurately defines the practical angular range needed for video calling scenarios.

## Limitations & Future Work

- Supports only ±40° viewpoints (sufficient for video calls, but unsuitable for wider-range 3D presentation).
- Reconstruction quality in back/side regions relies on geometry priors and may lack fidelity.
- Cached paper is incomplete; some quantitative comparison data were not retrieved.
- Generalization of synthetic-data training may degrade under extreme lighting or occlusion conditions.
- Multi-party video calling scenarios are not discussed.

## Related Work & Insights

- **Splatter Image**: The foundational architecture for this work; a U-Net that directly predicts Gaussian Splatting. The original version is not face-specific and does not address video temporal stability.
- **Project Starline (Google)**: A high-end multi-camera 3D communication system with a 55° viewing range, requiring dedicated hardware.
- **Codec Avatars (Meta)**: Photorealistic avatars requiring prior enrollment, with a fixed appearance representation.
- **Live 3D Portrait**: Real-time 3D facial reconstruction based on EG3D, constrained by the expressive capacity of the generative model.
- **TriPlaneNet**: Maps a single image to an EG3D latent code; reconstruction is limited by the EG3D training distribution.
- **GS-LRM / Xu et al. / Zou et al.**: Heavier transformer architectures or multi-view diffusion models that do not support real-time inference.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Key contributions (homography correction, dual Gaussians, stability loss) are practical but incremental.
- **Technical Depth**: ⭐⭐⭐⭐ — Comprehensive system design, though individual modules are not particularly deep.
- **Experimental Thoroughness**: ⭐⭐⭐☆ — Cached paper is incomplete; visible content suggests reasonable dataset and metric coverage, but user studies are absent.
- **Practical Value**: ⭐⭐⭐⭐⭐ — A directly deployable 3D video calling system with clear commercial prospects.
- **Overall Recommendation**: ⭐⭐⭐⭐ — A solid, engineering-oriented contribution; the authenticity perspective offers meaningful insights for the 3D communication field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction](boosting_multiview_indoor_3d_object_detection_via_adaptive_3.md)
- [\[ICCV 2025\] Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs](compression_of_3d_gaussian_splatting_with_optimized_feature_planes_and_standard_.md)
- [\[NeurIPS 2025\] Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting](../../NeurIPS2025/3d_vision/segment_then_splat_unified_3d_open-vocabulary_segmentation_via_gaussian_splattin.md)
- [\[ICCV 2025\] AR-1-to-3: Single Image to Consistent 3D Object Generation via Next-View Prediction](ar1to3_single_image_to_consistent_3d_object_via_nextview_pre.md)
- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)

</div>

<!-- RELATED:END -->
