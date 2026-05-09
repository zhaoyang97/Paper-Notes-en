---
title: >-
  [Paper Note] FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning
description: >-
  [CVPR 2026][3D Vision][Camera control] This paper proposes FaceCam, a system that addresses camera control in monocular portrait videos by using facial landmarks as a scale-aware camera representation, thereby avoiding the scale ambiguity inherent in conventional extrinsic camera representations. Two data augmentation strategies—synthetic camera motion and multi-clip stitching—are further designed to support continuous camera trajectory inference.
tags:
  - CVPR 2026
  - 3D Vision
  - Camera control
  - portrait video generation
  - scale-awareness
  - facial landmarks
  - video diffusion models
date: 2026-05-08
content_hash: 31140fc4f60779e7
---

# FaceCam: Portrait Video Camera Control via Scale-Aware Conditioning

**Conference**: CVPR 2026
**arXiv**: [2603.05506](https://arxiv.org/abs/2603.05506)
**Code**: [Available](https://weijielyu.github.io/FaceCam)
**Area**: 3D Vision
**Keywords**: Camera control, portrait video generation, scale-awareness, facial landmarks, video diffusion models

## TL;DR

This paper proposes FaceCam, a system that addresses camera control in monocular portrait videos by using facial landmarks as a scale-aware camera representation, thereby avoiding the scale ambiguity inherent in conventional extrinsic camera representations. Two data augmentation strategies—synthetic camera motion and multi-clip stitching—are further designed to support continuous camera trajectory inference.

## Background & Motivation

Camera motion control in controllable video generation is a core research problem, and is particularly critical for portrait videos in applications such as social media, post-production, and AR/VR. Existing approaches face two fundamental challenges:

**Challenge 1: Scale Ambiguity**
- **Scene-agnostic parameter representations** (e.g., Plücker rays, extrinsic matrices): the same parameter variation produces drastically different visual effects at different scene scales.
- Monocular video cannot determine absolute depth; the scene is recoverable only up to a global similarity transformation (unknown scale and translation).
- Mathematically: for any $\alpha > 0$, letting $\mathbf{x}' = \alpha\mathbf{x}$, $\mathbf{t}' = \alpha\mathbf{t}$, perspective projection remains invariant.

**Challenge 2: Training Data**
- Acquiring paired videos of the same dynamic portrait scene under different camera trajectories is extremely difficult.
- Synthetic 3D dynamic portrait data rarely achieves photorealism.

**Core Motivation**: A camera representation is needed that does not expose the unobservable global scale, while generalizing from limited static multi-view data to continuous camera trajectories.

## Method

### Overall Architecture

**Training Phase**:
1. Facial landmarks are extracted from the anchor frame of the target video as the camera condition.
2. Source video, target video, and camera conditions are each encoded into latents via a VAE.
3. These are fed into a diffusion Transformer to predict the target latent, optimized with a flow-matching loss.

**Inference Phase**:
1. FaceLift is used to generate a generic 3D Gaussian head model (identity-agnostic; shared across all experiments).
2. The proxy head is rendered along the target camera trajectory.
3. MediaPipe detects per-frame facial landmarks as the camera condition.
4. The diffusion Transformer generates the controlled video.

### Key Designs

#### 1. Scale-Aware Camera Representation via Facial Landmarks

**Theoretical Basis**: Classical multi-view geometry establishes that point correspondences in image space are sufficient to characterize relative camera motion. Given 7+ 2D correspondences, the fundamental matrix $F$ can be estimated, from which the essential matrix $E = \mathbf{K}^\top F \mathbf{K}$ and relative pose $[\mathbf{R}|\mathbf{t}]$ are recoverable up to a global scale.

**Implementation**:
- $m$ facial landmarks are detected in the anchor frame with 3D positions $\mathbf{X} = \{\mathbf{x}_k\}_{k=1}^m$.
- These are projected to 2D pixel coordinates $\mathbf{U} = \{\mathbf{u}_k\}_{k=1}^m$ according to the target camera pose.
- Landmarks are rasterized into a pixel-space image to serve as the conditioning signal.

**Scale Invariance Proof**: Under simultaneous scaling of 3D landmarks and translation ($\mathbf{x}_k' = s\mathbf{x}_k$, $\mathbf{t}' = s\mathbf{t}$), the 2D projections remain invariant:

$$\mathbf{u}_k' = \mathcal{N}(\mathbf{K}(\mathbf{R}\mathbf{x}_k' + \mathbf{t}')) = \mathcal{N}(\mathbf{K}s(\mathbf{R}\mathbf{x}_k + \mathbf{t})) = \mathbf{u}_k$$

**Sufficiency**: Given 3D landmarks and their 2D projections, a PnP solver recovers camera rotation and translation up to a global scale.

**User-Friendliness**: The rasterized landmark map provides a direct preview of the target viewpoint, making camera control intuitively interpretable.

#### 2. Training Data Generation Strategies

The system is built on the NeRSemble dataset (425 identities, 16 synchronized views, ~9.4K videos) supplemented by ~800 in-the-wild monocular videos.

| Strategy | Method | Problem Addressed |
|----------|--------|-------------------|
| Scale + color augmentation | Random scaling [0.75, 1.25], foreground segmentation + random background color | Increases data diversity |
| Synthetic camera motion | Simulated zoom (scale interpolation) and pan (crop offset interpolation) | Introduces dynamic cameras, but limited to parallel motion |
| Multi-clip stitching | Random concatenation of 1–4 clips from different camera positions | Introduces camera rotation (discrete pose changes) |
| In-the-wild data supplement | Synthetic camera motion applied to monocular videos | Mitigates overfitting to studio lighting |

**Key Finding**: Although training includes only discrete camera pose changes (via multi-clip stitching), the model **generalizes to continuous camera trajectory** inference.

### Loss & Training

- Built on the Wan open-source video foundation model with a flow-matching loss.
- Source video latents are concatenated with noisy latents via frame conditioning.
- Camera condition latents are injected via channel conditioning.
- Only 3D attention layers and projection layers are fine-tuned (following ReCamMaster).
- Training: 24 NVIDIA A100 GPUs, 3K steps, learning rate 5e-5, batch size 24.
- Total training data: ~9.1K videos (8.9K NeRSemble + ~200 in-the-wild).

## Key Experimental Results

### Main Results

**Table 1: Static Camera Evaluation on the Ava-256 Dataset**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | ArcFace↑ |
|--------|-------|-------|--------|----------|
| ReCamMaster | 9.73 | 0.557 | 0.581 | 0.701 |
| TrajectoryCrafter | 10.32 | 0.546 | 0.567 | 0.522 |
| FaceCam* (generic head) | 9.83 | 0.582 | 0.549 | 0.807 |
| **FaceCam** | **15.85** | **0.721** | **0.252** | **0.857** |

**Table 2: Dynamic Camera Evaluation on In-the-Wild Videos (100 videos, 10 motion types)**

| Method | Camera Accuracy | ArcFace | Quality | Aesthetics | Subject Consistency | Background Consistency |
|--------|----------------|---------|---------|------------|--------------------|-----------------------|
| ReCamMaster | 83% | 78.92 | 69.05 | 55.85 | 93.26 | 93.02 |
| TrajectoryCrafter | 99% | 49.79 | 71.37 | 55.76 | 92.23 | 92.25 |
| FaceCam (w/o wild) | **100%** | 77.73 | 70.71 | 55.73 | **94.52** | **95.16** |
| **FaceCam** | 97% | **83.94** | **73.49** | **59.91** | 94.77 | 94.98 |

### Ablation Study

**Training Data Ablation** (last two rows of Table 2):
- NeRSemble only: camera control is near-perfect (100%), but identity preservation (77.73) and image quality (70.71) are lower.
- Adding in-the-wild videos: identity preservation improves substantially (83.94), image quality improves (73.49), with a marginal drop in camera accuracy (97%).

### Key Findings

1. FaceCam substantially outperforms baselines in PSNR (15.85 vs. 10.32), demonstrating that scale-aware representation is critical for precise camera control.
2. ReCamMaster fails under large viewpoint changes, with scale ambiguity causing the head to move out of frame.
3. TrajectoryCrafter suffers from geometric errors in dynamic point cloud estimation, causing facial distortion (ArcFace: 0.522).
4. Facial landmark conditioning encodes not merely facial position, but **camera pose and scale decoupled from head motion**.
5. Training on discrete camera changes generalizes to continuous trajectories—an unexpected and practically valuable finding.
6. The generic 3D head model (FaceCam*) underperforms full FaceCam (which uses ground-truth landmarks), yet still surpasses baselines in identity preservation.

## Highlights & Insights

1. **Elegant theoretical foundation**: The camera representation is derived from first principles of multi-view geometry, with mathematically guaranteed scale invariance.
2. **Practical inference pipeline**: Target trajectories are rendered using a generic 3D head model followed by landmark detection, requiring no input-specific 3D reconstruction.
3. **Data efficiency**: State-of-the-art results are achieved with only ~9.1K videos and 3K training steps, far fewer than typically required.
4. **Unexpected generalization from multi-clip stitching**: Generalization from discrete pose changes to continuous trajectories is a significant empirical finding.

## Limitations & Future Work

1. Robustness depends on facial landmark detection; extreme profile views or heavy occlusions may degrade performance.
2. The generic proxy head model disregards variation in actual head shape across input videos.
3. Validation is limited to single-person scenes; camera control for multi-person portraits is not addressed.
4. The approach is restricted to portrait videos and does not generalize to camera control in arbitrary scenes.
5. Only ~200 in-the-wild videos are used for training; incorporating more may further improve generalization.

## Related Work & Insights

- **vs. ReCamMaster**: ReCamMaster uses scene-agnostic extrinsic conditioning and is susceptible to scale ambiguity; FaceCam uses a scene-aware landmark representation.
- **vs. TrajectoryCrafter**: The latter relies on 3D point cloud reconstruction and inpainting, where geometric errors are amplified into facial distortions.
- **Similarity to ControlNet**: Camera conditions are likewise injected via image channels, though here landmark maps encode geometric transformations rather than structural appearance.
- **Complementary to NeRF/3DGS methods**: No per-instance optimization is required; generation is achieved in a single forward pass.
- **Broader inspiration**: Using facial landmarks as proxies for geometric correspondences is a paradigm extensible to hands, bodies, and other anatomical structures with stable keypoints.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Scale-aware camera representation is theoretically elegant; the discrete-to-continuous generalization finding is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Dual evaluation on studio and in-the-wild data, though additional ablations (e.g., number of landmarks, base model selection) are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear, problem formulation is precise, and figures are intuitive.
- Value: ⭐⭐⭐⭐⭐ — A practical and training-efficient solution to portrait video camera control with strong empirical performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] VerseCrafter: Dynamic Realistic Video World Model with 4D Geometric Control](versecrafter_dynamic_realistic_video_world_model_with_4d_geometric_control.md)
- [\[CVPR 2026\] Ego-1K: A Large-Scale Multiview Video Dataset for Egocentric Vision](ego-1k_--_a_large-scale_multiview_video_dataset_for_egocentric_vision.md)
- [\[CVPR 2026\] SceneScribe-1M: A Large-Scale Video Dataset with Comprehensive Geometric and Semantic Annotations](scenescribe-1m_a_large-scale_video_dataset_with_comprehensive_geometric_and_sema.md)
- [\[CVPR 2026\] SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation](seethrough3d_occlusion_aware_3d_control_in_text-to-image_generation.md)
- [\[CVPR 2026\] 4C4D: 4 Camera 4D Gaussian Splatting](4c4d_4_camera_4d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
