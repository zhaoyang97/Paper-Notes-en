---
title: >-
  [Paper Note] EPiC: Efficient Video Camera Control Learning with Precise Anchor-Video Guidance
description: >-
  [ICML 2026][Video Generation][anchor video] EPiC utilizes a "first-frame visibility mask" approach to directly construct pixel-aligned anchor videos from arbitrary in-the-wild videos. Combined with a lightweight Anchor-C…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "anchor video"
  - "visibility mask"
  - "Anchor-ControlNet"
  - "I2V/V2V camera control"
  - "lightweight adaptation"
date: 2026-05-08
content_hash: 00527ec862edac32
---

# EPiC: Efficient Video Camera Control Learning with Precise Anchor-Video Guidance

**Conference**: ICML 2026  
**arXiv**: [2505.21876](https://arxiv.org/abs/2505.21876)  
**Code**: https://zunwang1.github.io/Epic (Project Page)  
**Area**: Video Generation  
**Keywords**: anchor video, visibility mask, Anchor-ControlNet, I2V/V2V camera control, lightweight adaptation

## TL;DR
EPiC utilizes a "first-frame visibility mask" approach to directly construct pixel-aligned anchor videos from arbitrary in-the-wild videos. Combined with a lightweight Anchor-ControlNet (26M parameters, <1% of the backbone) that operates exclusively on visible regions, it achieves SOTA I2V camera control accuracy while freezing the CogVideoX-5B-I2V backbone. It requires only 5K videos and 500 training steps, generalizing zero-shot to V2V tasks.

## Background & Motivation
**Background**: Mainstream camera control for video generation follows two routes: one feeds camera parameters (Plücker embedding, extrinsic matrices) directly into the VDM as conditions (CameraCtrl, AC3D), while the other lifts a single image to a point cloud and re-renders an "anchor video" based on the target trajectory to serve as a structured prior for the diffusion model (ViewCrafter, Gen3C, TrajectoryCrafter, Uni3C). The latter typically yields better precision due to explicit geometric guidance.

**Limitations of Prior Work**: The anchor video route faces two major pain points. First, anchors are rendered from estimated point clouds (DAv2, MoGe) and estimated trajectories (COLMAP), where estimation errors lead to pixel-level misalignment between the anchor and the source video (measured at only 16 dB PSNR). The model must simultaneously fix misalignments and inpaint invisible regions, confounding the learning objective. Second, to align these errors, existing methods often require full-backbone fine-tuning or heavy modules and are restricted to datasets like RealEstate10K with precise annotations, resulting in poor generalization to dynamic in-the-wild videos.

**Key Challenge**: There is a trade-off between the "3D information richness" of an anchor video and its "pixel alignment with the source video." Attempting to carry full re-rendered point cloud information increases misalignment. Existing methods prioritize the former, shifting the burden of error correction to the model.

**Goal**: (1) Enable training anchors to achieve pixel-level alignment with source videos in visible regions without relying on camera/point cloud estimation. (2) Inject anchor signals into a frozen VDM with minimal learnable parameters. (3) Retain precise trajectory control via 3D point clouds during inference.

**Key Insight**: The authors observe that the "geometric characteristics" of an anchor video only require information regarding which pixels remain visible relative to the first frame and which are occluded or moved out of view. This does not necessitate actual 3D reconstruction. By using dense optical flow to trace pixels back to the first frame—retaining those that can be traced and masking those that cannot—one can forge an anchor that is geometrically equivalent to point cloud rendering but perfectly aligned with the source video.

**Core Idea**: Shift anchor construction from "hard-to-align 3D re-rendering" to "easy-to-align visibility masking." The ControlNet is restricted to copying visible regions, while invisible regions are handled entirely by the frozen base model, effectively reducing the ControlNet's task from "repairing + inpainting" to just "copying."

## Method

### Overall Architecture
EPiC is based on CogVideoX-5B-I2V (DiT-based, 3D full self-attention). The training pipeline consists of two steps: (1) Synthesizing training anchors from in-the-wild videos using visibility masks (no camera/point cloud required). (2) Encoding the anchor via a 3D-VAE, concatenating it with noisy latents, and feeding it into a 26M Anchor-ControlNet. The output is spatially gated by the visibility mask $M$ before being added to the corresponding layers of the frozen base DiT. During inference, anchors are rendered using actual point clouds along user-defined trajectories. The visibility gate isolates 3D reconstruction misalignments and "flying pixels," and switching between "static camera control" and "dynamic foreground" modes is achieved via point cloud masking.

### Key Designs

1.  **Visibility-Based Masking**:
    - **Function**: Synthesizes training anchors from source videos that are pixel-aligned, without needing camera poses or point clouds.
    - **Mechanism**: RAFT is used to estimate dense optical flow, tracing each pixel in frame $t$ back to frame 1. Only pixels that can be stably traced are preserved; others are masked as black to produce a binary visibility mask $M_t$. The resulting anchor is pixel-consistent with the source in visible regions (PSNR improves from 16.01 dB to 40.12 dB), while the black regions maintain geometric equivalence to point cloud rendering. To bridge the training-inference gap, "flying pixel forgery" is used by drawing faint lines in visible regions using colors sampled from the first frame.
    - **Design Motivation**: Removes the misalignment burden from the learning process and decouples "misalignment repair" from "occlusion inpainting." It also bypasses the need for camera annotations, allowing the use of dynamic wild video datasets like Panda-70M.

2.  **Anchor-ControlNet (Lightweight DiT-based Adapter)**:
    - **Function**: Injects anchor video guidance into the frozen VDM using <1% of the backbone parameters.
    - **Mechanism**: A single DiT block with hidden dimensions reduced from 3072 to 256 (approx. 8%), connected only to the first 25% of the backbone layers (total 26M parameters). The anchor video $\mathbf{A}$ is encoded by the 3D-VAE as $\mathbf{z}_{\text{anchor}}$, concatenated with $\mathbf{z}_t$, patchified, and processed by the DiT-ctrl. The output is zero-initialized and projected back to 3072 dimensions: $\tilde{\mathbf{z}} = \text{Proj}(\text{DiT}_{\text{ctrl}}([\mathbf{z}_t, \mathbf{z}_{\text{anchor}}]))$. 
    - **Design Motivation**: Minimizing learnable parameters preserves the base model's generative capacity. This "light, shallow, and narrow" configuration is feasible because the aligned anchors simplify the ControlNet's task.

3.  **Visibility-Aware Output Masking**:
    - **Function**: Restricts the ControlNet's influence to visible regions, leaving invisible regions to the frozen base model.
    - **Mechanism**: The visibility mask $M \in \{0,1\}^{T'\times h\times w}$ is downsampled to latent resolution for hard-gating integration: $\hat{\mathbf{z}} = \text{DiT}_{\text{base}}(\mathbf{z}_t) + M \odot \tilde{\mathbf{z}}$.
    - **Design Motivation**: Extreme "separation of concerns." It prevents re-rendering artifacts from polluting the output, speeds up convergence, and enables "camera movement + foreground action" by masking specific foreground objects during inference.

### Loss & Training
Standard latent diffusion denoising loss:
$$\mathcal{L}_{\text{denoise}} = \mathbb{E}_{\mathbf{z}_0, t, \boldsymbol{\epsilon}, c}[\|\boldsymbol{\epsilon}_\theta(\mathbf{z}_t, t, c) - \boldsymbol{\epsilon}\|_2^2]$$
where condition $c$ includes text and the anchor. Only the 26M Anchor-ControlNet is updated. 5,000 videos from Panda-70M are used (batch size 16, 8×A100-40G, 500 steps, ~15 GPU hours).

## Key Experimental Results

### Main Results
On the RealCam-Vid test set (RE10K and MiraData), EPiC outperforms benchmarks across camera metrics and quality scores.

| Dataset | Method | RotErr ↓ | TransErr ↓ | CamMC ↓ | Total Quality ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| RE10K | AC3D† | 0.86±0.37 | 1.50±0.82 | 1.97±0.86 | 82.63 |
| RE10K | Gen3C | 0.45±0.13 | 0.99±0.22 | 1.35±0.30 | 82.27 |
| RE10K | **EPiC** | **0.40±0.11** | **0.86±0.18** | **1.17±0.23** | **82.63** |
| MIRA | Gen3C | 0.81±0.24 | 2.05±0.77 | 2.75±0.72 | 80.50 |
| MIRA | **EPiC** | **0.66±0.22** | **1.78±0.67** | **2.10±0.60** | **82.89** |

EPiC ranks first in all evaluated camera and quality metrics with the smallest standard deviation. Zero-shot V2V performance on Kubric-4D matches specialized models (GCD, TrajCrafter) while using significantly less training data and fewer parameters.

### Ablation Study

| Configuration | Anchor PSNR ↑ | RotErr ↓ | TransErr ↓ | CamMC ↓ |
| :--- | :--- | :--- | :--- | :--- |
| Point Cloud anchor | 16.01 | 0.60±0.20 | 1.07±0.39 | 1.45±0.62 |
| **Masking anchor (Ours)** | **40.12** | **0.40±0.11** | **0.86±0.18** | **1.17±0.23** |

### Key Findings
- Anchor-source PSNR is highly correlated with downstream camera precision. Aligning the anchor is more critical than the amount of 3D information.
- Removing visibility gating results in tearing artifacts and blurred content in invisible regions.
- Inference-side masking allows for decoupled camera and foreground motion without additional training.

## Highlights & Insights
- **Task Redefinition**: Decoupling camera control into "copying visible areas" and "inpainting invisible areas" simplifies the learning objective structurally rather than just architecturally.
- **Data-Architecture Synergy**: Better data alignment allows for a more lightweight model. This "upstream alignment for downstream efficiency" trade-off is applicable to other conditioning-heavy tasks.
- **Optical Flow as Geometry**: Using flow traceability as a proxy for visibility provides a cheap alternative to full 3D reconstruction when the task only requires knowing *where* is visible.

## Limitations & Future Work
- Reliance on first-frame visibility: For large rotations, the visible region shrinks rapidly, shifting the burden to the base model and potentially reducing long-range consistency.
- Dependence on RAFT: In cases of extreme occlusion or textureless regions, flow errors can degrade the visibility mask during training.
- V2V interpretation: Trajectories are interpreted as relative transforms, limiting global coordinate-based positioning for complex storytelling.

## Related Work & Insights
- **vs ViewCrafter/Gen3C**: These require full-backbone tuning due to misaligned anchors; EPiC remains efficient by eliminating misalignment at the source.
- **vs CameraCtrl/AC3D**: These lack explicit 3D guidance; EPiC maintains precision in OOD scenes by utilizing point cloud re-rendering during inference paired with visibility gating.
- **vs ReCamMaster/SynCamMaster**: These use large-scale 4D simulated data for V2V; EPiC achieves comparable V2V results zero-shot via strong inductive bias.

## Rating
- Novelty: ⭐⭐⭐⭐ (Structural reduction of task difficulty via visibility masks).
- Experimental Thoroughness: ⭐⭐⭐⭐ (SOTA across 6 indicators; clean ablations).
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation and standard notation).
- Value: ⭐⭐⭐⭐⭐ (Highly efficient; 15 GPU hours for SOTA control with a frozen backbone).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Rays as Pixels: Learning A Joint Distribution of Videos and Camera Trajectories](rays_as_pixels_learning_a_joint_distribution_of_videos_and_camera_trajectories.md)
- [\[ICLR 2026\] Frame Guidance: Training-Free Guidance for Frame-Level Control in Video Diffusion Models](../../ICLR2026/video_generation/frame_guidance_training-free_guidance_for_frame-level_control_in_video_diffusion.md)
- [\[ICLR 2026\] PreciseCache: Precise Feature Caching for Efficient and High-fidelity Video Generation](../../ICLR2026/video_generation/precisecache_precise_feature_caching_for_efficient_and_high-fidelity_video_gener.md)
- [\[ICML 2026\] iTryOn: Mastering Interactive Video Virtual Try-On with Spatial-Semantic Guidance](itryon_mastering_interactive_video_virtual_try-on_with_spatial-semantic_guidance.md)
- [\[ICLR 2026\] Learning Video Generation for Robotic Manipulation with Collaborative Trajectory Control](../../ICLR2026/video_generation/learning_video_generation_for_robotic_manipulation_with_collaborative_trajectory.md)

</div>

<!-- RELATED:END -->
