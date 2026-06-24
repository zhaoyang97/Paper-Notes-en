---
title: >-
  [Paper Note] Tora: Trajectory-Oriented Diffusion Transformer for Video Generation
description: >-
  [CVPR 2025][Video Generation][Motion Control] Proposes Tora, the first trajectory-oriented Diffusion Transformer (DiT) framework for video generation. By employing a trajectory extractor (3D VAE encoding motion trajectories into spatiotemporal patches) and a motion-guidance fuser (adaptive normalization injecting into DiT blocks), it achieves scalable trajectory-controlled video generation supporting multiple resolutions, durations, and aspect ratios. In 128-frame tests…
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Motion Control"
  - "Diffusion Transformer"
  - "Trajectory Guidance"
  - "Optical Flow Encoding"
date: 2026-05-08
content_hash: 29679fb07fe019a2
---

# Tora: Trajectory-Oriented Diffusion Transformer for Video Generation

**Conference**: CVPR 2025  
**arXiv**: [2407.21705](https://arxiv.org/abs/2407.21705)  
**Code**: [https://github.com/alibaba/Tora](https://github.com/alibaba/Tora)  
**Area**: Image Generation / Video Generation  
**Keywords**: Video Generation, Motion Control, Diffusion Transformer, Trajectory Guidance, Optical Flow Encoding

## TL;DR
Proposes Tora, the first trajectory-oriented Diffusion Transformer (DiT) framework for video generation. By employing a trajectory extractor (3D VAE encoding motion trajectories into spatiotemporal patches) and a motion-guidance fuser (adaptive normalization injecting into DiT blocks), it achieves scalable trajectory-controlled video generation supporting multiple resolutions, durations, and aspect ratios. In 128-frame tests, it achieves trajectory control accuracy 3 to 5 times higher than UNet-based methods.

## Background & Motivation

1. **Background**: Diffusion models have made significant progress in video generation. The Diffusion Transformer (DiT) architecture, represented by Sora, has demonstrated the capability to generate 10-60 second high-quality videos, vastly outperforming UNet-based methods. One of the key challenges in video generation is motion control—enabling objects in the generated video to follow specific user-defined trajectories.

2. **Limitations of Prior Work**: (a) Previous motion control methods (e.g., DragNUWA, MotionCtrl) are based on the UNet architecture and are limited to 16 frames/fixed resolutions, suffering from motion blur, object distortion, and unnatural translation drift on longer sequences; (b) while the DiT architecture is highly scalable, no prior work has adapted trajectory control to the DiT framework; (c) DiT converts videos into patch sequences using a Video Autoencoder and patchification, making the frame-by-frame displacement motion representation incompatible with the patch space.

3. **Key Challenge**: How to seamlessly align user-specified trajectory conditions with the scalable architecture of DiT? The injection strategies of UNet-based methods (such as simple concatenation or linear projection) are incompatible with DiT's patchified latent space and alternating spatial-temporal attention mechanisms.

4. **Goal**: Design a trajectory encoding and fusion mechanism compatible with DiT's scalability to achieve precise control over video content motion while supporting variable durations, resolutions, and aspect ratios.

5. **Key Insight**: Convert trajectory displacement into optical flow visualization in the RGB domain $\rightarrow$ Compress it into the same latent space as video patches using a 3D VAE $\rightarrow$ Extract multi-level motion conditions through stacked convolutional layers $\rightarrow$ Inject them into DiT blocks using adaptive normalization layers.

6. **Core Idea**: Use a 3D Motion VAE to encode trajectories into a motion representation in the same latent space as video patches, and then inject multi-level motion conditions layer-by-layer into DiT using adaptive normalization, preserving DiT's scalability while achieving precise trajectory tracking.

## Method

### Overall Architecture
Tora is built upon OpenSora (an open-source Sora implementation) and consists of three core components: (1) Spatial-Temporal DiT (ST-DiT)—a foundation generative model employing alternating spatial and temporal self-attention; (2) Trajectory Extractor (TE)—encoding user trajectories into hierarchical spatiotemporal motion patches; (3) Motion-guidance Fuser (MGF)—injecting the motion patches into DiT blocks. The inputs are a text description + optional image + trajectory coordinate sequence, and the output is a generated video following the specified trajectory (up to 204 frames, up to 720p).

### Key Designs

1. **Trajectory Extractor (TE)**:
    - **Function**: Encodes arbitrary user trajectories into hierarchical motion conditions within the same latent space as video patches.
    - **Mechanism**: Processes in three steps: (a) converts trajectory coordinates into frame-to-frame displacement $(u, v)$ to generate a trajectory map $g \in \mathbb{R}^{L \times H \times W \times 2}$, applying Gaussian filtering to mitigate sparsity; (b) converts the displacement map into the RGB color space to obtain $g_{vis}$, then encodes it into a motion latent representation $g_m$ using a self-trained 3D VAE (simplified based on the MAGVIT-v2 architecture, with 8x spatial and 4x temporal compression); (c) applies patchification to $g_m$, followed by extracting multi-level motion features $f_i$ through stacked residual convolutional layers, corresponding to each DiT block.
    - **Design Motivation**: In DiT, videos are converted into patch sequences via an autoencoder and patchification, where each patch spans multiple frames. Directly using frame-to-frame displacement does not match the patch space. The 3D VAE compresses trajectories into the same latent space as video patches, perfectly aligning information density and dimensions.

2. **Motion-guidance Fuser (MGF)**:
    - **Function**: Injecting multi-level motion conditions into their corresponding DiT blocks.
    - **Mechanism**: The authors explore three fusion architectures—additional channel concatenation (MLP), cross-attention, and Adaptive Normalization (Adaptive Norm). They ultimately choose Adaptive Norm: motion features $f_i$ are passed through two zero-initialized convolutional layers to generate scale $\gamma_i$ and shift $\beta_i$, respectively, and a linear modulation is applied to the hidden states of the DiT block: $h_i = \gamma_i \cdot h_{i-1} + \beta_i + h_{i-1}$.
    - **Design Motivation**: Adaptive Norm does not require strict alignment (which is a challenge for cross-attention), dynamically adapts to different conditions, and its zero-initialization ensures that it does not disrupt the generative capability of the base model in the initial stages of training. Empirical results demonstrate its global superiority over channel concatenation and cross-attention across FVD, CLIPSIM, and TrajError metrics.

3. **Two-Stage Training Strategy**:
    - **Function**: Gradually transitioning from dense optical flow to sparse trajectories to enhance motion learning efficacy.
    - **Mechanism**: The first stage uses dense optical flow (extracted by RAFT) as trajectories to train for 2 epochs, providing rich motion information to help the model comprehend motion patterns. The second stage switches to sparse trajectories (randomly selecting 1-N object trajectories + Gaussian smoothing) and fine-tunes for 1 epoch to adapt the model to user-friendly interaction styles.
    - **Design Motivation**: Training directly with sparse trajectories provides insufficient information, making it difficult for the model to learn; conversely, training solely with dense optical flow fails to adapt to sparse inputs during inference. The two-stage strategy balances both information richness and operational flexibility.

### Loss & Training
- Uses the standard diffusion model noise prediction loss $\ell_\epsilon = \|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2$.
- Adopts an Adapter-like strategy: only trains temporal blocks + TE + MGF, freezing spatial blocks to retain the generative knowledge of the base model.
- The 3D Motion VAE is pre-trained on an optical flow dataset and then frozen.
- Data processing: Scene detection splitting $\rightarrow$ Invalid video removal $\rightarrow$ Aesthetic/optical flow score filtering $\rightarrow$ Camera motion filtering $\rightarrow$ PLLaVA annotation generation.

## Key Experimental Results

### Main Results

Comparison of video generation at different frame counts:

| Method | FVD↓ (128 frames) | CLIPSIM↑ (128 frames) | TrajError↓ (128 frames) |
|------|-------------|-------------------|---------------------|
| VideoComposer | 856 | 0.2236 | 58.76 |
| DragNUWA | 784 | 0.2305 | 41.25 |
| MotionCtrl | 731 | 0.2331 | 38.39 |
| OpenSora (w/o trajectory) | 533 | 0.2411 | 373.17 |
| OpenSora-DragNUWA* | 565 | 0.2393 | 21.75 |
| **Tora** | **494** | **0.2418** | **11.72** |

The trajectory error of Tora is only 1/3 to 1/5 of UNet-based methods, and its FVD is 30% to 40% better than UNet-based approaches. Simultaneously, Tora outperforms the base OpenSora in visual quality (FVD 494 vs. 533), indicating that trajectory control, in turn, improves generation stability.

### Ablation Study

Trajectory Compression Method:

| Configuration | FVD↓ | TrajError↓ |
|------|------|------------|
| Keyframe Sampling | 581 | 27.61 |
| Average Pooling | 558 | 20.97 |
| **3D VAE (Ours)** | **513** | **14.25** |

Fusion Architecture:

| Configuration | FVD↓ | TrajError↓ |
|------|------|------------|
| Extra Channel Concatenation | 542 | 21.07 |
| Cross Attention | 526 | 18.36 |
| **Adaptive Normalization** | **513** | **14.25** |

Training Strategy:

| Configuration | FVD↓ | TrajError↓ |
|------|------|------------|
| Dense Optical Flow Only | 601 | 39.34 |
| Sparse Trajectory Only | 556 | 24.73 |
| **Two-stage Hybrid** | **513** | **14.25** |

### Key Findings
- Compressing trajectories into the video latent space via 3D VAE is crucial, outperforming simple keyframe sampling and average pooling by 48% and 32% in TrajError, respectively.
- Adaptive Normalization completely outperforms cross-attention and channel concatenation in both efficacy and efficiency.
- Placing the MGF in Temporal DiT blocks yields significantly better results than placing it in Spatial blocks (TrajError drops from 23.39 to 14.25).
- The two-stage training strategy is indispensable; using dense or sparse trajectories in isolation exhibits far worse performance compared to the hybrid strategy.
- Tora seamlessly transfers to CogVideoX (2B/5B), validating the module's generalizability and scaling capabilities.
- Introducing trajectory control not only improves motion precision but also enhances the visual quality of the base model (suppressing temporal artifacts).

## Highlights & Insights
- **3D Motion VAE Latent Space Alignment**: Encoding trajectories into the same latent space as video patches is the most core innovation of this work. This paradigm of "condition signals sharing the same latent space as the primary signal" can be extended to other conditional control tasks (such as audio-to-video, etc.).
- **Zero-Initialization Trick in Adaptive Norm**: Zero-initialization ensures that the injection of motion conditions starts at zero in the early stages of training, preserving the pre-trained basic generative capacity before progressively learning motion guidance. This "progressive activation" strategy is highly elegant.
- **Motion Control Enhances Visual Quality**: Unexpectedly, incorporating trajectory control leads to a better FVD compared to the uncontrolled OpenSora. This indicates that reasonable motion priors can assist the model in mitigating temporal inconsistency issues in long videos.
- **User Study Competitiveness**: Performs on par with the closed-source Vidu, lagging only slightly behind Kling.

## Limitations & Future Work
- Trajectory control error increases proportionally with video duration (longer videos exhibit larger errors); hence, long-range consistency still has room for improvement.
- Currently, only object trajectory control is supported; camera motion control requires dedicated design.
- The training data processing pipeline is highly complex (scene detection $\rightarrow$ aesthetic filtering $\rightarrow$ motion segmentation $\rightarrow$ camera detection), incurring non-trivial data engineering costs.
- For fast-moving or occluded scenes, errors inherent in optical flow estimation will propagate into the trajectory encoding.
- Future work could integrate 3D perception (depth, scene structure) to achieve physically more plausible motion control.

## Related Work & Insights
- **vs DragNUWA**: DragNUWA is the pioneering work that introduced trajectory control in VDMs, but being based on UNet, it only supports 16-frame low-resolution output. Tora extends trajectory control to the DiT architecture, supporting up to 204 frames at 720p, achieving a 3-to-5-fold improvement in trajectory accuracy.
- **vs MotionCtrl**: MotionCtrl separately controls camera and object motion with excellent flexibility but is similarly bottlenecked by UNet's scalability. Tora registers a TrajError of only 11.72 in 128-frame tests compared to MotionCtrl's 38.39.
- **vs OpenSora**: OpenSora delivers high visual quality but lacks motion control. Tora introduces precise trajectory control while maintaining or even enhancing visual quality.
- **vs CogVideoX**: Tora's motion module can be seamlessly transferred to CogVideoX, demonstrating the strong generalizability of the proposed approach.

## Rating
- Novelty: ⭐⭐⭐⭐ The first work to achieve trajectory control on the DiT architecture, featuring an ingenious 3D VAE latent space alignment design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparisons across 16/64/128-frame settings, three types of ablation studies (compression/fusion/training), multi-model scaling, and user studies.
- Writing Quality: ⭐⭐⭐⭐ Well-structured paper with intuitive methodology diagrams and solid experimental design.
- Value: ⭐⭐⭐⭐⭐ Pioneers trajectory control on DiT frameworks, backed by open-source code from Alibaba, substantially advancing the video generation field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] LeviTor: 3D Trajectory Oriented Image-to-Video Synthesis](levitor_3d_trajectory_oriented_image-to-video_synthesis.md)
- [\[CVPR 2025\] PoseTraj: Pose-Aware Trajectory Control in Video Diffusion](posetraj_pose-aware_trajectory_control_in_video_diffusion.md)
- [\[CVPR 2025\] MotionStone: Decoupled Motion Intensity Modulation with Diffusion Transformer for Image-to-Video Generation](motionstone_decoupled_motion_intensity_modulation_with_diffusion_transformer_for.md)
- [\[CVPR 2025\] FlashMotion: Few-Step Controllable Video Generation with Trajectory Guidance](flashmotion_few-step_controllable_video_generation_with_trajectory_guidance.md)
- [\[CVPR 2025\] ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models](shotadapter_text-to-multi-shot_video_generation_with_diffusion_models.md)

</div>

<!-- RELATED:END -->
