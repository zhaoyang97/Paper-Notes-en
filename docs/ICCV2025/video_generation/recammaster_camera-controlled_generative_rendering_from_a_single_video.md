---
title: >-
  [Paper Note] ReCamMaster: Camera-Controlled Generative Rendering from A Single Video
description: >-
  [ICCV 2025][Video Generation][Camera Control] This paper proposes ReCamMaster, which achieves camera-trajectory-controlled video re-generation from a single input video via a frame-dimension conditioning mechanism and a…
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Camera Control"
  - "Video Re-rendering"
  - "Diffusion Models"
  - "Multi-Camera Dataset"
date: 2026-05-08
content_hash: bb51b3e839734cbc
---

# ReCamMaster: Camera-Controlled Generative Rendering from A Single Video

**Conference**: ICCV 2025
**arXiv**: [2503.11647](https://arxiv.org/abs/2503.11647)  
**Code**: [https://github.com/KwaiVGI/ReCamMaster](https://github.com/KwaiVGI/ReCamMaster)  
**Area**: Video Generation
**Keywords**: Camera Control, Video Re-rendering, Video Generation, Diffusion Models, Multi-Camera Dataset

## TL;DR

This paper proposes ReCamMaster, which achieves camera-trajectory-controlled video re-generation from a single input video via a frame-dimension conditioning mechanism and a multi-camera synchronized dataset synthesized in UE5, significantly outperforming existing methods.

## Background & Motivation

Camera motion is a fundamental element in film and television production, profoundly affecting audience experience and narrative intent. Amateur videographers, however, are often constrained by hardware and technical limitations, making it difficult to achieve professional-grade camera movements. The goal is to enable post-hoc modification of camera trajectories in videos to present dynamic scenes from more desirable viewpoints.

Limitations of prior work:

**GCD** pioneered camera-controlled video-to-video generation but suffers from a domain gap with Kubric synthetic data, limiting its effectiveness on real-world videos.

**ReCapture** requires per-video optimization (LoRA fine-tuning), restricting practical applicability.

3. Reconstruction-based 4D methods are bottlenecked by the quality of single-video reconstruction.

Core motivation: To leverage the generative capacity of pretrained text-to-video (T2V) models and realize open-domain camera trajectory modification through a simple yet powerful video conditioning mechanism.

## Method

### Overall Architecture

ReCamMaster is built upon a pretrained T2V diffusion model (DiT architecture + 3D VAE + Rectified Flow). Given a source video $V_s$, a target camera trajectory $\text{cam}_t$, and a text prompt $p_t$, the model generates a target video $V_t$. The framework consists of three key designs: frame-dimension conditioning, camera pose encoding, and a multi-task training strategy.

### Key Designs

1. **Frame Dimension Conditioning**: Source video tokens and target video tokens are concatenated along the frame dimension, $x_i = [x_s, x_t]_{\text{frame-dim}} \in \mathbb{R}^{b \times 2f \times s \times d}$. No additional attention layers are required; source and target videos interact naturally within the 3D spatiotemporal attention. Compared to channel-dimension concatenation (e.g., GCD) and view-dimension concatenation, frame-dimension concatenation enables full spatiotemporal interaction between source and target across all Transformer blocks, preserving content consistency and dynamic synchronization.

2. **Camera Pose Encoding**: Only the extrinsic parameters of the target camera $\text{cam}_t \in \mathbb{R}^{f \times 3 \times 4}$ (rotation + translation matrices) are used as conditioning, without source camera parameters (which are difficult to estimate accurately at inference time). A learnable camera encoder $\mathcal{E}_c$ (a fully connected layer, $12 \to d$) injects the camera parameters into the spatial attention output of each Transformer block: $F_i = F_o + \mathcal{E}_c(\text{cam}_t)$.

3. **Multi-Camera Synchronized Dataset (Multi-Cam Video)**: Constructed using Unreal Engine 5, comprising 136K videos, 13.6K dynamic scenes, 40 high-quality 3D environments, and 122K distinct camera trajectories. Real-world cinematographic characteristics are carefully simulated to reduce the synthetic-to-real domain gap.

### Loss & Training

- Base loss: velocity regression loss $\mathcal{L}_{LCM}$ from Conditional Flow Matching.
- **Enhancing generalization**: Only the camera encoder and 3D attention layers are fine-tuned; all other parameters are frozen. Moderate noise (200–500 steps) is added to the conditioning video latent to mitigate the synthetic domain gap.
- **Unified multi-task training**: With 20% probability, all frames are replaced with noise (T2V task); with 20% probability, $f-1$ frames are replaced (I2V task), enhancing content generation capability.

## Key Experimental Results

### Main Results

Evaluated on 1,000 WebVid videos with 10 camera trajectories:

| Method | FID↓ | FVD↓ | CLIP-T↑ | RotErr↓ | TransErr↓ | Mat.Pix.(K)↑ | FVD-V↓ | CLIP-V↑ |
|------|------|------|---------|---------|-----------|--------------|--------|---------|
| GCD | 72.83 | 367.32 | 32.86 | 2.27 | 5.51 | 639.39 | 365.75 | 85.92 |
| Traj-Attn | 69.21 | 276.06 | 33.43 | 2.18 | 5.32 | 619.13 | 256.30 | 88.65 |
| DaS | 63.25 | 159.60 | 33.05 | 1.45 | 5.59 | 633.53 | 154.25 | 87.33 |
| **ReCamMaster** | **57.10** | **122.74** | **34.53** | **1.22** | **4.85** | **906.03** | **90.38** | **90.36** |

VBench quality evaluation:

| Method | Aesthetic↑ | Imaging Quality↑ | Temporal Flickering↑ | Subject Consistency↑ | Background Consistency↑ |
|------|-----------|------------------|---------------------|---------------------|------------------------|
| GCD | 38.21 | 41.56 | 95.81 | 88.94 | 92.00 |
| **ReCamMaster** | **42.70** | **53.97** | **97.36** | **92.05** | **93.83** |

### Ablation Study

| Conditioning Method | FID↓ | FVD↓ | Mat.Pix.(K)↑ | FVD-V↓ | CLIP-V↑ |
|------------|------|------|-------------|--------|---------|
| Channel-dim concatenation | 74.09 | 187.94 | 521.10 | 148.51 | 84.62 |
| View-dim concatenation | 80.51 | 194.47 | 573.92 | 177.68 | 83.40 |
| **Frame-dim concatenation** | **57.10** | **122.74** | **906.03** | **90.38** | **90.36** |

Training strategy ablation:

| Strategy | FID↓ | FVD↓ | Aesthetic↑ | Imaging Quality↑ |
|------|------|------|-----------|------------------|
| Baseline | 66.67 | 171.80 | 40.02 | 51.93 |
| + Noise injection | 65.17 | 164.04 | 40.36 | 52.22 |
| + 3D-Attn fine-tuning | 59.47 | 132.58 | 43.08 | 52.80 |
| + All strategies | **57.10** | **122.74** | **42.70** | **53.97** |

### Key Findings

- Frame-dimension concatenation significantly outperforms other conditioning methods, improving matched pixels by 73% (906K vs. 521K).
- The model can also be extended to applications such as video stabilization, super-resolution, and video outpainting.

## Highlights & Insights

- The frame-dimension concatenation mechanism is elegant in its simplicity, leveraging the full spatiotemporal attention of the pretrained model for conditional generation without additional modules.
- The realistic cinematographic simulation design of the UE5 dataset is critical for generalization to real-world videos.
- Unified multi-task training (T2V/I2V/V2V) is an effective strategy for enhancing content generation capability.

## Limitations & Future Work

- Frame-dimension concatenation doubles the number of input tokens, increasing computational overhead.
- The model inherits limitations of the pretrained T2V model (e.g., poor hand generation quality).
- Only extrinsic parameters are used for conditioning; intrinsic camera parameters are not exploited.

## Related Work & Insights

- The choice of conditioning injection strategy has a substantial impact on conditional generation quality; frame-dimension concatenation could serve as a general-purpose solution applicable to other video-conditioned generation tasks.
- High-quality synthetic data combined with domain-adaptive training strategies can effectively compensate for the scarcity of real-world data.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Unified Camera Positional Encoding for Controlled Video Generation](../../CVPR2026/video_generation/unified_camera_positional_encoding_for_controlled_video_generation.md)
- [\[ICCV 2025\] RealCam-I2V: Real-World Image-to-Video Generation with Interactive Complex Camera Control](realcam-i2v_real-world_image-to-video_generation_with_interactive_complex_camera.md)
- [\[ICCV 2025\] SteerX: Creating Any Camera-Free 3D and 4D Scenes with Geometric Steering](steerx_creating_any_camera-free_3d_and_4d_scenes_with_geometric_steering.md)
- [\[ICCV 2025\] Free-Form Motion Control: Controlling the 6D Poses of Camera and Objects in Video Generation](free-form_motion_control_controlling_the_6d_poses_of_camera_and_objects_in_video.md)
- [\[ICCV 2025\] Reangle-A-Video: 4D Video Generation as Video-to-Video Translation](reangle-a-video_4d_video_generation_as_video-to-video_translation.md)

</div>

<!-- RELATED:END -->
