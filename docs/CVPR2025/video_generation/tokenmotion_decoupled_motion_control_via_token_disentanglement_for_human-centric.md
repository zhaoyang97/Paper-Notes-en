---
title: >-
  [Paper Note] TokenMotion: Decoupled Motion Control via Token Disentanglement for Human-centric Video Generation
description: >-
  [CVPR 2025][Video Generation][Video Diffusion Models] TokenMotion proposes the first DiT-based video diffusion framework that represents camera trajectories and human poses as spatiotemporal tokens. By leveraging a "decouple-and-fuse" strategy in conjunction with a human-aware dynamic mask, it achieves fine-grained and joint control over both camera and human motion, outperforming existing state-of-the-art (SOTA) methods in both text-to-video (T2V) and image-to-video (I2V) pa…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Video Diffusion Models"
  - "Motion Control"
  - "Token Disentanglement"
  - "Camera Motion"
  - "Human Pose"
date: 2026-05-08
content_hash: 1dad15807d664972
---

# TokenMotion: Decoupled Motion Control via Token Disentanglement for Human-centric Video Generation

**Conference**: CVPR 2025  
**arXiv**: [2504.08181](https://arxiv.org/abs/2504.08181)  
**Code**: None  
**Area**: Diffusion Models / Video Generation  
**Keywords**: Video Diffusion Models, Motion Control, Token Disentanglement, Camera Motion, Human Pose

## TL;DR
TokenMotion proposes the first DiT-based video diffusion framework that represents camera trajectories and human poses as spatiotemporal tokens. By leveraging a "decouple-and-fuse" strategy in conjunction with a human-aware dynamic mask, it achieves fine-grained and joint control over both camera and human motion, outperforming existing state-of-the-art (SOTA) methods in both text-to-video (T2V) and image-to-video (I2V) paradigms.

## Background & Motivation

**Background**: Human-centric video generation is one of the core application scenarios of current video diffusion models. Especially in creative production (such as the Grammy Glambot slow-motion effect), there is a critical need to simultaneously and precisely control both camera motion (pan, tilt, zoom, dolly) and human actions (pose sequences).

**Limitations of Prior Work**: Existing video diffusion methods for motion control suffer from two major problems: (1) Limited motion representation capabilities—most methods rely on global conditions (such as text descriptions or a single motion vector) to guide motion, which fails to achieve fine-grained, frame-by-frame, and region-by-region control; (2) Insufficient integration of camera and human motion—existing methods typically manage only one type of motion control and struggle to handle the spatiotemporal coupling between the two.

**Key Challenge**: Camera motion and human motion are intrinsically coupled in spatiotemporal dimensions—camera movement changes the position and scale of the human subject in the frame, while human motion itself is independent of the camera. Directly blending these two control signals leads to conflicts and unnatural results. The core of the challenge lies in how to maintain the independent controllability of both motions within a unified framework while correctly modeling their interaction.

**Goal**: To build a unified video diffusion framework capable of independently controlling camera motion, human motion, and their joint interaction, while supporting both T2V and I2V generation paradigms.

**Key Insight**: The authors observe that the Diffusion Transformer (DiT) architecture naturally processes data using tokens as the basic unit. Thus, motion signals can also be represented as tokens—unlike global condition injection, tokenized motion representations allow control forces to be applied "at the right time and at the right place."

**Core Idea**: To represent camera trajectories and human poses using spatiotemporal tokens respectively, and handle the spatiotemporal overlap and separation of the two motion signals through a "decouple first, fuse later" strategy in conjunction with a human-aware dynamic mask.

## Method

### Overall Architecture
TokenMotion is built upon the DiT architecture. The input consists of text prompts (or reference images), along with camera trajectory sequences and/or human pose sequences. The entire pipeline is divided into three stages: (1) Motion tokenization—encoding camera trajectories and human skeletons into spatiotemporal tokens respectively; (2) Decoupled control injection—injecting the two motion tokens into the DiT denoising process through independent control branches; (3) Dynamic mask fusion—utilizing a human-aware dynamic mask to distinguish between "human regions" and "background regions" to properly fuse the influence of both motion signals. The final output is high-quality controlled human videos.

### Key Designs

1. **Spatiotemporal Tokenization of Motion Signals**:

    - **Function**: Convert continuous camera trajectories and human pose sequences into spatiotemporal token representations aligned with video latent tokens.
    - **Mechanism**: For camera motion, the camera extrinsic parameters (rotation matrix + translation vector) of each frame are encoded into pixel-wise ray maps via Plücker coordinates. These maps are then transformed into a token sequence aligned with the DiT latent space using a patchify operation. For human motion, keypoint sequences extracted by DWPose are rendered into frame-by-frame skeleton heatmaps, which are similarly patchified to obtain human motion tokens. Both types of tokens preserve spatiotemporal position information, enabling control to act precisely on specific regions and frames of the video.
    - **Design Motivation**: Compared to global conditional encoding (such as concatenating camera parameters into a single vector), tokenized representation preserves local spatial information. This achieves the effect of "applying influence exactly where control is needed," which is particularly critical for handling scenarios where the human subject continuously moves within the frame.

2. **Decouple-and-Fuse Control Strategy**:

    - **Function**: Independently inject camera and human motion control signals within a unified framework, followed by dynamic fusion.
    - **Mechanism**: The framework employs two parallel ControlNet-style branches to separately process camera motion tokens and human motion tokens. Each branch contains independent DiT blocks to extract the corresponding motion features. The key lies in the fusion stage—instead of simple addition or concatenation, a Human-Aware Dynamic Mask is utilized to determine whether each spatiotemporal position should be more influenced by the camera control or the human control. Specifically, the mask assigns higher weights to the human control branch in human regions, higher weights to the camera control branch in background regions, and smooth transitions in boundary areas.
    - **Design Motivation**: Directly mixing two motion signals leads to conflicts (e.g., distorting the human pose when the camera translates). The decoupling strategy ensures the independence of each control, while the dynamic mask resolves the issue of how to reconcile both signals at the same spatial position.

3. **Human-Aware Dynamic Mask**:

    - **Function**: Generate spatiotemporally varying attention masks to guide the fusion weights of the two motion signals.
    - **Mechanism**: Utilize human skeleton sequences to generate frame-by-frame human region masks, expanding the boundary regions through Gaussian blurring. For each denoising step $t$, the mask value $M_t(x,y)$ is close to 1 in human regions (favoring human control) and close to 0 in background regions (favoring camera control), with smooth transitions between 0 and 1 in boundary regions. This mask varies across frames, adaptively handling human movement within the frame and avoiding the limitations of static area partitioning.
    - **Design Motivation**: Since the location and scale of the human subject in the video change dynamically (especially when the camera is also moving), static masks cannot properly handle such spatiotemporal variations. The dynamic mask ensures that "human-motion-dominant regions" and "camera-motion-dominant regions" are properly distinguished at any given moment.

### Loss & Training
TokenMotion adopts the standard diffusion denoising loss (in the $v$-prediction formulation) and is jointly optimized under three training modes: (1) camera control only; (2) human control only; (3) joint control. During training, one of the control signals is randomly dropped to enhance the model's single-control capability. The human-aware mask is generated using ground-truth skeletons during training and is automatically obtained via the input skeleton sequences during inference.

## Key Experimental Results

### Main Results

| Task | Metric | TokenMotion | CameraCtrl | MotionCtrl | Direct-a-Video |
|------|------|-------------|------------|------------|----------------|
| Camera Control (T2V) | RotErr ↓ | **0.87** | 1.34 | 2.01 | 1.56 |
| Camera Control (T2V) | TransErr ↓ | **0.42** | 0.71 | 1.15 | 0.83 |
| Human Control (T2V) | PCK@0.2 ↑ | **78.3** | - | 61.5 | - |
| Joint Control (T2V) | FVD ↓ | **198** | 287 | 312 | 265 |
| Joint Control (I2V) | FVD ↓ | **172** | 241 | 278 | 233 |
| Visual Quality | FID ↓ | **14.2** | 18.7 | 22.3 | 17.5 |

### Ablation Study

| Config | RotErr ↓ | PCK@0.2 ↑ | FVD ↓ | Description |
|------|----------|-----------|-------|------|
| Full TokenMotion | **0.87** | **78.3** | **198** | Full model |
| w/o Dynamic Mask | 1.12 | 71.6 | 234 | Joint control quality drops significantly without the mask |
| w/o Decoupled Branches | 1.25 | 68.2 | 251 | A single branch mixing both motions yields the worst performance |
| Global Conditioning instead of Tokens | 1.08 | 73.1 | 225 | Global injection performs worse than tokenization, validating the necessity of local control |
| w/o Random Drop Training | 0.95 | 74.8 | 213 | Single-control capability weakens without random drop training |

### Key Findings
- The decoupled branch is the most critical design; removing it degrades the joint control FVD from 198 to 251 (+27%).
- The dynamic mask contributes significantly in joint control scenarios (FVD 198 vs 234) but has minimal impact on single-control scenarios.
- Tokenized representations improve all metrics compared to global conditioning, proving the value of fine-grained local control.
- In the I2V paradigm, the advantages of TokenMotion are even more pronounced. Since reference images provide appearance priors, the precision of motion control becomes the primary differentiator.

## Highlights & Insights
- **Spatiotemporal tokenized motion representation**: Aligning motion signals with the video latent in the same token space is a highly natural and efficient design. This idea can be transferred to other conditional video generation tasks (such as object trajectory control or scene flow control).
- The **Human-Aware Dynamic Mask** elegantly resolves the spatial conflict of multi-signal fusion. It is essentially a semantic region-based adaptive condition weight allocation mechanism, which can be generalized to any multi-conditional fusion scenario.
- The "decouple first, fuse later" design paradigm holds general value in multi-conditional control, avoiding mutual interference when directly mixing multiple signals.

## Limitations & Future Work
- Currently, only single-person scenarios are supported; motion control and mask generation in multi-person scenarios remain unresolved.
- The capability to handle extreme camera movements (e.g., $360^\circ$ orbits) is limited, which may cause geometric distortions.
- Using human skeletons as motion representations lacks hand details and facial expressions, limiting fine-grained performance control.
- Inference speed is constrained by the dual-branch architecture, making real-time generation still out of reach.

## Related Work & Insights
- **vs CameraCtrl**: CameraCtrl only supports camera control and uses global condition injection. TokenMotion extends control to joint motion and adopts tokenized local representations, also improving camera control accuracy.
- **vs MotionCtrl**: MotionCtrl supports both camera and object motion but uses simple condition concatenation and lacks a decoupling mechanism, leading to poor joint control performance.
- **vs AnimateAnyone**: Human animation methods typically ignore camera motion. TokenMotion is the first to address their joint control within a unified framework.

## Rating
- Novelty: ⭐⭐⭐⭐ First camera + human joint motion control under the DiT framework; creative decouple-and-fuse strategy.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on both T2V and I2V paradigms with comprehensive ablations, though lacking quantitative details from user studies.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, detailed method descriptions, and highly illustrative schemas.
- Value: ⭐⭐⭐⭐ Direct application value for creative video production; the decoupling strategy is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ConMo: Controllable Motion Disentanglement and Recomposition for Zero-Shot Motion Transfer](conmo_controllable_motion_disentanglement_and_recomposition_for_zero-shot_motion.md)
- [\[CVPR 2025\] MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation](motionstone_decoupled_motion_intensity_modulation_with_diffusion_transformer_for.md)
- [\[CVPR 2025\] VidTwin: Video VAE with Decoupled Structure and Dynamics](vidtwin_video_vae_with_decoupled_structure_and_dynamics.md)
- [\[CVPR 2026\] 3D-Aware Implicit Motion Control for View-Adaptive Human Video Generation](../../CVPR2026/video_generation/3d-aware_implicit_motion_control_for_view-adaptive_human_video_generation.md)
- [\[CVPR 2025\] MotionPro: A Precise Motion Controller for Image-to-Video Generation](motionpro_a_precise_motion_controller_for_image-to-video_generation.md)

</div>

<!-- RELATED:END -->
