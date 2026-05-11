---
title: >-
  [Paper Note] MikuDance: Animating Character Art with Mixed Motion Dynamics
description: >-
  [ICCV 2025][Video Understanding][Character Animation] This paper proposes MikuDance, a diffusion-based character art animation system that achieves high-dynamic animation of complex character artwork through two core con…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Character Animation"
  - "diffusion model"
  - "Camera Control"
  - "Mixed Motion"
  - "Image-to-Video"
date: 2026-05-08
content_hash: cf8bfc1caf55e1a8
---

# MikuDance: Animating Character Art with Mixed Motion Dynamics

**Conference**: ICCV 2025
**arXiv**: [2411.08656](https://arxiv.org/abs/2411.08656)
**Code**: [https://kebii.github.io/MikuDance](https://kebii.github.io/MikuDance)
**Area**: Video Understanding
**Keywords**: Character Animation, diffusion model, Camera Control, Mixed Motion, Image-to-Video

## TL;DR

This paper proposes MikuDance, a diffusion-based character art animation system that achieves high-dynamic animation of complex character artwork through two core contributions: Mixed Motion Modeling, which unifies character motion and 3D camera motion into a pixel-space representation, and Mixed-Control Diffusion, which implicitly aligns character shape/scale with motion guidance within the Reference UNet.

## Background & Motivation

Animating static character artwork has substantial demand in film, gaming, and digital design. Traditional tools (MMD, Live2D) require professional expertise, while existing image-to-video methods (Animate Anyone, DISCO) primarily target real humans and cannot be directly applied to character artwork. Two major challenges arise:

**High-dynamic motion guidance**: Animating character artwork requires simultaneously handling complex foreground character motion and large-scale camera motion in the background. Existing methods only support static backgrounds with human body motion and cannot model full-scene dynamics.

**Reference–guidance misalignment**: Anime characters exhibit unique head-to-body ratios, exaggerated poses, and diverse artistic styles, creating severe scale and body-shape discrepancies with motion guidance (typically derived from real-person videos). Explicit alignment is infeasible given the variability of character forms, necessitating implicit alignment.

## Method

### Overall Architecture

MikuDance is built on SD-1.5 and consists of:
- **Mixed Motion Modeling**: Extracts character poses (Xpose) and camera poses (DROID-SLAM) from a driving video, then converts camera motion into a pixel-level scene motion representation via Scene Motion Tracking.
- **Mixed-Control Diffusion**: Concatenates the reference image, reference pose, and all driving poses (VAE-encoded) along the channel dimension as input to the Reference UNet, and injects scene motion via Motion-Adaptive Normalization.
- **Two-stage mixed-source training**: First trains the Reference UNet on stylized frame pairs; then trains MAN and temporal modules jointly.

### Key Designs

1. **Scene Motion Tracking (SMT)**:

    - Constructs a scene point cloud $\phi^l \in \mathbb{R}^{N \times 3}$ from the depth map of the reference image.
    - Projects the point cloud from frame $l$ to frame $l+1$ using camera-to-world transformation matrices $\mathcal{T}^l$ and $\mathcal{Y}^{l+1}$.
    - Computes pixel correspondences between the two projected images to obtain scene motion $\mathbf{m}^s \in \mathbb{R}^{N \times 2}$.
    - Formulation: $(z^l - z^{l+1})[\mathbf{m}^s; \mathbf{1}] = \mathcal{K}^l[\phi^l; \mathbf{1}] - \mathcal{K}^{l+1}\mathcal{Y}^{l+1}\mathcal{T}^l[\phi^l; \mathbf{1}]$

   Key distinctions from optical flow: (1) SMT is independent of driving video content, whereas optical flow is content-dependent; (2) SMT tracks 3D point clouds while optical flow tracks 2D pixels. SMT thus provides disentangled camera dynamic information.

   **Design Motivation**: Converts 3D camera motion into a 2D pixel representation that is in the same domain as character poses, enabling unified motion guidance.

2. **Mixed-Control Diffusion**:

    - **No separate encoders**: Unlike Animate Anyone and similar methods, no independent pose encoder or ControlNet is used.
    - The reference image, reference pose, and all driving poses are VAE-encoded and concatenated along the channel dimension as input to the Reference UNet.
    - The input convolutional layer of the Reference UNet is expanded accordingly; newly added parameters are initialized with zero convolution.
    - The reference image is also encoded via a CLIP image encoder and used as cross-attention keys for both UNets.

   **Design Motivation**: Implicit alignment through mixed inputs outperforms the combination of explicit alignment and separate encoders. Ablation experiments confirm this simpler architecture achieves the best results.

3. **Motion-Adaptive Normalization (MAN)**:

    - Inspired by SPADE, applies instance normalization to features at each downsampling block of the Reference UNet, then uses the scene motion $\mathbf{m}^s$ to generate spatially adaptive scale $\gamma^i$ and shift $\beta^i$ parameters.
    - $f^{i'} = \gamma^i_{C,H,W}(\mathbf{m}^s) \frac{f^i_{C,H,W} - \mu^i_C}{\sigma^i_C} + \beta^i_{C,H,W}(\mathbf{m}^s)$
    - $\gamma^i$ and $\beta^i$ carry spatial dimensions, enabling pixel-level scene motion guidance.

   **Design Motivation**: Scene motion exerts a global influence on animated frames; adaptive normalization effectively injects global motion while preserving local consistency.

### Loss & Training

**Two-stage mixed-source training**:

- **Stage 1**: Paired video frame training without MAN or temporal modules.
    - Randomly mixes stylized frame pairs: the initial frame is spatially concatenated and style-transferred using SDXL-Neta.
    - The reference frame is randomly sampled from frames outside the target sequence (simulating the inference scenario where reference and driving are unrelated).
    - Resolution 768×768, batch size 128, trained for 120k steps.

- **Stage 2**: MAN and temporal modules added (all other parameters frozen).
    - Mixes MMD videos and camera-motion-only videos without characters.
    - 24-frame sequences, batch size 16, trained for 60k steps.
    - 16× A800 GPUs.

Both stages randomly drop out pose and scene motion guidance (rate 0.2) to improve robustness. Inference uses DDIM with 20 steps.

Standard diffusion training loss: $\mathcal{L}_{simple} = \mathbb{E}_{\epsilon, t, c}[\|\epsilon - \epsilon_\theta(x_t, t, c)\|_2^2]$

## Key Experimental Results

### Main Results (Table)

**Quantitative comparison (100 MMD test videos):**

| Method | FID ↓ | SSIM ↑ | PSNR ↑ | LPIPS ↓ | L1 ↓ | FID-VID ↓ | FVD ↓ |
|--------|-------|--------|--------|---------|------|-----------|-------|
| AniAny | 43.945 | 0.488 | 12.530 | 0.548 | 7.31E-5 | 38.179 | 846.414 |
| AniAny* | 28.833 | 0.526 | 13.610 | 0.517 | 6.23E-5 | 26.764 | 575.304 |
| DISCO | 59.221 | 0.313 | 10.732 | 0.615 | 9.25E-5 | 46.852 | 923.921 |
| MagicPose | 44.258 | 0.424 | 12.357 | 0.554 | 7.77E-5 | 41.347 | 886.691 |
| UniAnimate | 47.328 | 0.417 | 12.074 | 0.571 | 7.93E-5 | 40.924 | 882.245 |
| **MikuDance** | **24.597** | **0.576** | **14.592** | **0.493** | **5.73E-5** | **22.868** | **502.380** |

### Ablation Study (Table)

**Ablation of key designs:**

| Setting | FID ↓ | SSIM ↑ | PSNR ↑ | FID-VID ↓ | FVD ↓ |
|---------|-------|--------|--------|-----------|-------|
| w/o MIX (separate Ref UNet + 2 ControlNets) | 27.315 | 0.523 | 14.004 | 24.124 | 541.453 |
| w/o MAN (scene motion via direct concatenation) | 24.985 | 0.542 | 14.501 | 23.366 | 509.342 |
| w/o SMT (no scene motion) | 25.472 | 0.534 | 14.312 | 23.362 | 517.673 |
| w/ Plücker (Plücker coordinates as substitute) | 25.918 | 0.538 | 14.261 | 23.471 | 521.853 |
| w/ Flow (optical flow as substitute) | 26.141 | 0.516 | 14.088 | 23.079 | 505.533 |
| **MikuDance (full)** | **24.597** | **0.576** | **14.592** | **22.868** | **502.380** |

**User study**: 50 volunteers ranked the anonymous results of 4 methods on 20 videos. MikuDance substantially outperforms all baselines across three dimensions—overall quality, frame quality, and temporal quality—with >97% of users preferring MikuDance.

### Key Findings

1. **Mixed control outperforms separate encoders**: w/o MIX (2 ControlNets) achieves FID 27.3 vs. 24.6 for the full model; separate processing fails to resolve character–guidance misalignment.
2. **MAN outperforms direct concatenation**: Spatially adaptive normalization injects global motion more effectively, yielding FVD 502 vs. 509.
3. **SMT outperforms Plücker coordinates and optical flow**: Pixel-level scene motion representation shares the same domain as character poses; Plücker coordinates introduce a domain gap, and optical flow is content-dependent and non-generalizable.
4. **Even domain-finetuned baselines fall far short**: AniAny* (fine-tuned on anime domain) achieves FVD 575 vs. MikuDance's 502.
5. **Unique capability for high-dynamic scenes**: MikuDance maintains high-fidelity animation under the combination of large dance motions and rapid camera movements.

## Highlights & Insights

- **Elegant unification into pixel-space motion representation**: Heterogeneous character poses (keypoints) and camera motion (3D poses) are unified into a 2D pixel motion space.
- **"Simplicity wins" mixed-control design**: Removing complex architectures such as ControlNet and directly mixing all guidance signals within the Reference UNet yields superior results.
- **Practical two-stage training strategy**: Stylized frame pairs simulate diverse character styles, while camera-motion videos teach background dynamics.
- **MMD dataset of 3,600 animations**: A dataset of 120k clips and 10.2 million frames is constructed from MMD animations.
- **Overwhelming user study victory**: >97% preference rate.

## Limitations & Future Work

1. Some generated animations exhibit background distortion and artifacts—3D ambiguity in image animation is an inherent challenge.
2. SMT assumes a static scene; in practice, background objects may also move.
3. Depth map quality affects scene point cloud accuracy and consequently SMT quality.
4. The method relies on DROID-SLAM for camera pose extraction, which may be inaccurate under extreme motion.
5. Long video generation depends on temporal aggregation methods for stitching, potentially introducing transition artifacts.
6. Training cost is substantial (16× A800, 180k steps total).

## Related Work & Insights

- **Animate Anyone series**: Establishes the foundational Reference UNet + Denoising UNet architecture; this work addresses the additional challenges specific to character artwork on top of this framework.
- **Camera-controlled video generation**: Methods such as Human4DiT and HumanVid use Plücker coordinates; this work demonstrates the superiority of pixel-level representations.
- **SPADE → MAN**: The concept of spatially adaptive normalization in semantic space is transferred to motion-adaptive normalization.
- The method holds significant commercial value for anime and game character animation applications.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The SMT strategy and mixed-control design are original; the unification of heterogeneous motion signals into pixel space is an elegant idea.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive quantitative comparisons, thorough ablations, and convincing user study; character animation–specific benchmarks beyond MMD are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with rich illustrations; the two challenges and their corresponding solutions are explicitly paired.
- **Value**: ⭐⭐⭐⭐ The first work to introduce high-dynamic camera motion control for character artwork animation, with direct applicability to the anime and gaming industries.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Simultaneous Motion And Noise Estimation with Event Cameras](simultaneous_motion_and_noise_estimation_with_event_cameras.md)
- [\[ICCV 2025\] EMoTive: Event-Guided Trajectory Modeling for 3D Motion Estimation](emotive_event-guided_trajectory_modeling_for_3d_motion_estimation.md)
- [\[NeurIPS 2025\] Token Bottleneck: One Token to Remember Dynamics](../../NeurIPS2025/video_understanding/token_bottleneck_one_token_to_remember_dynamics.md)
- [\[ICCV 2025\] FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases](flowseek_optical_flow_made_easier_with_depth_foundation_models_and_motion_bases.md)
- [\[ICCV 2025\] Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow](flow4agent_long-form_video_understanding_via_motion_prior_from_optical_flow.md)

</div>

<!-- RELATED:END -->
