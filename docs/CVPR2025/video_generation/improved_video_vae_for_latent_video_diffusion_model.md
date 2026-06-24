---
title: >-
  [Paper Note] Improved Video VAE for Latent Video Diffusion Model
description: >-
  [CVPR 2025][Video Generation][Video VAE] This paper proposes IV-VAE to resolve the issues in existing video VAEs where image weight initialization suppresses the learning of temporal compression, and causal convolution leads to unbalanced inter-frame performance. By introducing a Keyframe Temporal Compression (KTC) architecture and Grouped Causal Convolution (GCConv), it achieves SOTA video reconstruction and generation quality on multiple benchmarks.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Video VAE"
  - "Temporal Compression"
  - "Causal Convolution"
  - "Keyframe"
  - "Latent Space Diffusion"
date: 2026-05-08
content_hash: 055c28badb327849
---

# Improved Video VAE for Latent Video Diffusion Model

**Conference**: CVPR 2025  
**arXiv**: [2411.06449](https://arxiv.org/abs/2411.06449)  
**Code**: [https://wpy1999.github.io/IV-VAE](https://wpy1999.github.io/IV-VAE)  
**Area**: Diffusion Model  
**Keywords**: Video VAE, Temporal Compression, Causal Convolution, Keyframe, Latent Space Diffusion

## TL;DR

This paper proposes IV-VAE to resolve the issues in existing video VAEs where image weight initialization suppresses the learning of temporal compression, and causal convolution leads to unbalanced inter-frame performance. By introducing a Keyframe Temporal Compression (KTC) architecture and Grouped Causal Convolution (GCConv), it achieves SOTA video reconstruction and generation quality on multiple benchmarks.

## Background & Motivation

**Background**: Latent video diffusion models (e.g., Sora, SVD) rely on VAEs to compress high-dimensional pixel space into a low-dimensional latent space. Existing video VAEs (e.g., Open-Sora, OD-VAE, CogVideoX) commonly adopt the strategy of inflating pretrained 2D image VAEs into 3D causal structures to achieve simultaneous spatiotemporal compression.

**Limitations of Prior Work**: The authors identify two key problems: (1) Initializing from an image VAE with the same latent channel dimension suppresses the subsequent learning of temporal compression, because the spatial compression gains of high-dimensional image VAEs diminish, and high-dimensional initialization leads to a steep decline in spatial compression performance and slow convergence of temporal compression; (2) The unidirectional information flow of 3D causal convolution results in unequal information interaction among frames within the same frame group where preceding frames cannot receive information from subsequent frames, causing reconstructed video quality to fluctuate across frames (poor for preceding frames, good for subsequent ones) and manifesting as distinct flickering artifacts.

**Key Challenge**: There is a conflict between spatial and temporal compression: image VAE initialization occupies all latent channels for spatial compression, leaving insufficient learning capacity for temporal compression; causal convolution ensures the continuity of long video encoding but sacrifices the equality of information interaction within frame groups.

**Goal**: (1) How to provide a more balanced spatiotemporal compression initialization to accelerate convergence? (2) How to achieve balanced information interaction within frame groups while maintaining causal logic?

**Key Insight**: The authors observe that the spatial compression capability of low-dimensional (Z/2) image VAEs is already sufficient to support spatiotemporal compression video VAEs (dimension Z). Inheriting low-dimensional weights and using the remaining channels for initial temporal compression actually leads to faster and better convergence. They also observe that the root cause of the inter-frame imbalance within a frame group is the unidirectionality of causal convolution, which suggests that standard convolutions should be used within frame groups instead.

**Core Idea**: A twin-branch keyframe architecture is proposed to decouple spatiotemporal compression initialization, and grouped causal convolution is used to replace causal convolution for balanced interaction between frames.

## Method

### Overall Architecture

IV-VAE is based on the UNet architecture, where the input is a sequence of video frames and the output is the latent space representation and the reconstructed video. Both the encoder and decoder adopt the proposed KTC Unit as the fundamental block. The overall pipeline incorporates three core improvements: Grouped Causal Convolution (GCConv) replacing standard causal convolution, the KTC twin-branch architecture decoupling spatiotemporal compression, and Temporal Motion Perception Enhancement (TMPE) expanding the receptive field. The temporal compression rate is 4×, and the spatial compression rate is 8×8.

### Key Designs

1. **Grouped Causal Convolution (GCConv)**:

    - **Function**: Achieves equal information interaction among frames within a frame group while maintaining causal logic across frame groups.
    - **Mechanism**: Continuous frames are grouped according to the temporal compression rate $t_c=4$, where every 4 frames form a frame group (with the first frame grouped separately). Causal padding is applied between frame groups (each group is padded with the last frame of the previous group at the front, and with zeros at the end). Standard convolution is used within each frame group to allow all frames to share equivalent interactive information. After padding, each group is processed separately with the same standard convolution. Note that the number of frames within a frame group changes accordingly after each temporal downsampling/upsampling.
    - **Design Motivation**: Standard causal convolution prevents preceding frames in a frame group from perceiving information from subsequent frames, leading to poor reconstruction in preceding frames and good reconstruction in subsequent ones, which causes flickering. Grouped causal convolution retains causal dependencies between frame groups (suitable for segmented long video encoding/decoding) while employing standard convolution to eliminate imbalance within frame groups.

2. **Keyframe Temporal Compression (KTC)**:

    - **Function**: Provides a more balanced spatiotemporal compression initialization to accelerate video VAE convergence.
    - **Mechanism**: Each fundamental block is divided into a 2D branch and a 3D branch, each outputting $C_{out}/2$ channels. The 2D branch extracts spatial information of keyframes using 2D convolution, while the 3D branch extracts overall temporal motion information using GCConv. Both branches are normalized via RMS-Norm and then concatenated along the channel dimension. During initialization, both branches are initialized with pretrained image VAE weights of latent channel dimension Z/2 (the 3D branch is inflated via center initialization). During output, the 2D branch is responsible for keyframe reconstruction, while the 3D branch accounts for reconstructing the remaining frames.
    - **Design Motivation**: Directly initializing with a Z-dimensional image VAE causes all channels to be fully occupied by spatial compression, leaving no room for learning temporal compression. Utilizing a Z/2-dimensional initialization and reserving the remaining channels for temporal compression is equivalent to acquiring 2× temporal compression capability from the very beginning, leading to faster convergence.

3. **Temporal Motion Perception Enhancement (TMPE)**:

    - **Function**: Expands the receptive field to enhance temporal motion perception capability under high resolution.
    - **Mechanism**: Introduces multi-scale Parallel Atrous Convolution (PAC, drawing inspiration from ASPP) in the last layer of the encoder, where features from different dilation rates are concatenated and passed through a 1×1 convolution to adjust channel dimensions. Concurrently, attention blocks are expanded from 2 to 7, and all attention blocks are executed after full spatiotemporal compression to reduce computation.
    - **Design Motivation**: In high-resolution videos, the same motion corresponds to a larger pixel span, making local receptive fields insufficient to capture motion patterns. Combining atrous convolution and attention blocks expands the receptive field, substantially improving temporal motion capture, especially under high resolution.

### Loss & Training

Training is conducted in stages: first, the image VAE is trained (256×256, 200k steps), then inflated to 3D and trained on 256×256 videos for 500k steps, expanded to 512×512 for 200k steps, and finally trained on different resolutions and frame counts for 100k steps (incorporating 3D GAN discriminator loss in this phase). Loss functions include KL divergence, MAE, and LPIPS. Structurally, GroupNorm is replaced by RMSNorm to maintain temporal causality, and the channel reduction position during spatial upsampling is optimized to reduce peak VRAM usage by 29%.

## Key Experimental Results

### Main Results

| Dataset | Metric | IV-VAE (Z=4) | OD-VAE | Gain |
|--------|------|-------------|--------|------|
| Kinetics-600 | FVD↓ | 8.01 | 10.69 | -2.68 |
| Kinetics-600 | PSNR↑ | 34.29 | 33.88 | +0.41 |
| ActivityNet | FVD↓ | 6.08 | 8.10 | -2.02 |
| Kinetics-600 (Z=16) | FVD↓ | 2.97 | 3.28 (Causal) | -0.31 |
| Kinetics-600 (Z=16) | PSNR↑ | 39.02 | 38.38 (CogX) | +0.64 |

IV-VAE outperforms corresponding baselines across all latent channel dimensions (Z=4/8/16), while using less than half the parameters of OD-VAE (107M vs 239M) and reducing parameters by 73% compared to OS-VAE.

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | Description |
|------|-------|-------|--------|------|
| (A) Baseline Causal VAE | 31.29 | 0.9042 | 0.05233 | Base |
| (B) + GCConv | 31.64 | 0.9082 | 0.05028 | Effective inter-frame balanced interaction |
| (C) + KTC | 31.86 | 0.9116 | 0.04865 | Effective twin-branch spatiotemporal decoupling |
| (D) + GCConv + KTC | 32.12 | 0.9145 | 0.04744 | Complementary to each other |
| (E) Full IV-VAE | 32.24 | 0.9158 | 0.04725 | TMPE further improves performance |

### Key Findings

- KTC is the most contributing module (+0.57 PSNR), followed by GCConv (+0.35 PSNR); TMPE has a smaller contribution but only adds 3M parameters.
- The advantage of KTC is more significant at higher latent channel dimensions (Z=8, 16), as the spatial compression gain of high-dimensional image VAEs diminishes.
- In high-resolution scenarios (1080P MotionHD), the relative advantage of IV-VAE is more pronounced (its PSNR margin over CogX-VAE expands from 0.28 at 480P to 1.65 at 1080P).
- Compared with the Overlap mechanism, the Cache mechanism yields completely equivalent single-step reconstruction results, while being more time-efficient and memory-saving.

## Highlights & Insights

- **Low-dimensional initialization outperforms high-dimensional initialization**: It is patchily counterintuitive to discover that initializing with an image VAE with fewer latent channels yields better results, revealing the tension between spatial and temporal compression. This observation can be transferred to any VAE design requiring multi-dimensional compression.
- **Elegant design of Grouped Causal Convolution**: Maintaining causal logic between frame groups for long video encoding, and employing standard convolution within frame groups to ensure balance, achieves the best of both worlds. This "intra-group bidirectionality, inter-group unidirectionality" approach can be extended to other sequence modeling tasks.
- **MotionHD Dataset**: A 1080P evaluation dataset uniformly sampled based on motion distribution is proposed, compensating for the limitations of existing datasets which either lack resolution or feature overly slow motion.

## Limitations & Future Work

- The overall architecture is still based on UNet, lacking a global receptive field. The authors point out that DiT or Mamba architectures could be considered in the future.
- Aligning the spatial downsampling frequency with the compression rate limits the receptive field, leaving room for improvement in high-resolution, fast-motion scenarios.
- Expanding the attention blocks to 7 in TMPE increases computation; more efficient global modeling schemes are required for larger resolutions or longer videos.

## Related Work & Insights

- **vs CogVideoX VAE**: CogX utilizes 16 channels + causal convolution, whereas IV-VAE achieves higher PSNR under the same channel count (39.02 vs 38.38) with only half the parameter size.
- **vs Open-Sora VAE**: OS-VAE utilizes a stacked two-stage compression with a large parameter count (393M). IV-VAE achieves a lower FVD (8.01 vs 19.05) with 107M parameters.
- **vs CV-VAE**: CV-VAE applies latent space regularization to avoid distribution shifts but suffers from poorer reconstruction quality. IV-VAE addresses the spatiotemporal compression conflict from an architectural level.

## Rating

- Novelty: ⭐⭐⭐⭐ The designs of the keyframe twin-branch structure and grouped causal convolution are novel and supported by theoretical observations, though the overall framework is still based on UNet.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluations across five benchmarks, three channel dimensions, multiple resolutions, thorough ablation studies, and the proposal of a new dataset.
- Writing Quality: ⭐⭐⭐⭐ The motivation analysis is clear and concise, and the information preservation curves in Fig.1 are highly persuasive.
- Value: ⭐⭐⭐⭐ Valuable insights and improvements are proposed for video VAE design, making it suitable as a foundational component for video generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] VidTwin: Video VAE with Decoupled Structure and Dynamics](vidtwin_video_vae_with_decoupled_structure_and_dynamics.md)
- [\[ICCV 2025\] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](../../ICCV2025/video_generation/leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)
- [\[CVPR 2025\] FADE: Frequency-Aware Diffusion Model Factorization for Video Editing](fade_frequency-aware_diffusion_model_factorization_for_video_editing.md)
- [\[CVPR 2025\] Timestep Embedding Tells: It's Time to Cache for Video Diffusion Model](timestep_embedding_tells_its_time_to_cache_for_video_diffusion_model.md)
- [\[CVPR 2025\] VideoScene: Distilling Video Diffusion Model to Generate 3D Scenes in One Step](videoscene_distilling_video_diffusion_model_to_generate_3d_scenes_in_one_step.md)

</div>

<!-- RELATED:END -->
