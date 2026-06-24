---
title: >-
  [Paper Note] RGBAvatar: Reduced Gaussian Blendshapes for Online Modeling of Head Avatars
description: >-
  [CVPR 2025][Human Understanding][Head Avatar] RGBAvatar proposes a "Reduced Gaussian Blendshapes" representation that efficiently represents animatable head avatars using only 20 learnable bases. Combined with batch-parallel rendering and a color initialization strategy, it achieves online real-time (reconstructing while capturing) head avatar reconstruction for the first time.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Head Avatar"
  - "Gaussian Blendshapes"
  - "Real-time Reconstruction"
  - "Online Modeling"
  - "Facial Animation"
date: 2026-05-08
content_hash: b8eb3bd0131305f4
---

# RGBAvatar: Reduced Gaussian Blendshapes for Online Modeling of Head Avatars

**Conference**: CVPR 2025  
**arXiv**: [2503.12886](https://arxiv.org/abs/2503.12886)  
**Code**: [github.com/gapszju/RGBAvatar](https://github.com/gapszju/RGBAvatar)  
**Area**: Human Understanding / 3D Vision  
**Keywords**: Head Avatar, Gaussian Blendshapes, Real-time Reconstruction, Online Modeling, Facial Animation

## TL;DR

RGBAvatar proposes a "Reduced Gaussian Blendshapes" representation that efficiently represents animatable head avatars using only 20 learnable bases. Combined with batch-parallel rendering and a color initialization strategy, it achieves online real-time (reconstructing while capturing) head avatar reconstruction for the first time.

## Background & Motivation

3DGS has significantly advanced the reconstruction quality of animatable head avatars. Existing methods (such as GaussianBlendshapes) linearly blend Gaussian attributes using pre-defined blendshape bases from 3DMMs, yielding impressive results. However, two limitations persist:

- **Parameter volume scales linearly with the number of bases**: FLAME defines 50+ blendshape bases, each corresponding to a full set of Gaussian attributes, resulting in slow training and high memory consumption.
- **Pre-defined bases lack personalization**: Generic 3DMM bases are suboptimal for capturing the facial details of a specific individual.

Core innovation: Implicitly learn a set of **reduced and personalized** blendshape bases (only 20 bases) using an MLP, rather than binding them to the fixed bases of a 3DMM. Simultaneously, GPU-optimizations achieve unprecedented training speeds (630 frames/second) and rendering speeds (400 FPS).

## Method

### Overall Architecture

Given a monocular video, FLAME tracking yields the parameters $\theta$. An MLP $\mathcal{F}$ maps $\theta$ to reduced weights $\psi \in \mathbb{R}^K$ ($K=20$), which linearly blend the base model $G^\psi = G_0 + \sum_{k=1}^K \psi_k \Delta G_k$. Finally, the Gaussians are deformed to the target space according to FLAME mesh deformation for rendering.

### Key Designs

**1. Reduced Gaussian Blendshapes**

- **Function**: Efficiently represent arbitrary facial expressions using a minimal set of learnable bases ($K=20$).
- **Mechanism**: Instead of using fixed blendshape bases from 3DMM, let an MLP $\mathcal{F}: \mathbb{R}^H \rightarrow \mathbb{R}^K$ learn the mapping from FLAME parameters to reduced weights. The bases $\{\Delta G_k\}$ and the MLP are optimized jointly, allowing the model to adaptively discover compact base combinations.
- **Design Motivation**: Generic 3DMM bases are designed for the general population and are redundant for specific individuals. Reducing the bases to 20 not only reduces parameter size but also yields better reconstruction quality than 50 3DMM bases through end-to-end optimization.

**2. Color Initialization Estimation + Batch-Parallel Gaussian Rasterization**

- **Function**: Scale training throughput from ~100 frames/second to 630 frames/second, completing reconstruction in 80 seconds.
- **Mechanism**: Color initialization treats Gaussians projected to 2D as Gaussian kernels and directly estimates the initial color via weighted convolution: $\mathbf{c}^{\text{init}} = \frac{\sum w_{ij} \mathbf{I}_{ij}}{\sum w_{ij}}$. Batch-parallel rendering decouples the preprocessing and rasterization of multiple samples into two stages, requiring only a single GPU-CPU synchronization and utilizing CUDA Streams to achieve 100% Stream Processor utilization.
- **Design Motivation**: Head avatars typically contain <100k Gaussians, where traditional single-sample training results in <60% GPU utilization due to the bottleneck being GPU-CPU synchronization rather than computation.

**3. Local-Global Sampling Online Reconstruction Strategy**

- **Function**: Enable an online "capture-and-reconstruct" mode with quality close to offline reconstruction.
- **Mechanism**: Maintain a local sample pool $\mathcal{M}_l$ (size 150, FIFO for new frames) and a global sample pool $\mathcal{M}_g$ (size 1000, reservoir sampling for historical frames). Each batch samples from both pools with a ratio of $\eta=0.7$ to balance rapid adaptation to new data and mitigation of forgetting.
- **Design Motivation**: The Key Challenge in online data streams is fast convergence on new frames versus prevention of forgetting old frames. The local pool ensures rapid adaptation, while the global pool with reservoir sampling guarantees an equal retention probability for each frame.

### Loss & Training

L1 color reconstruction loss + random background color regularization (constraining Gaussians to remain within the head area).

## Key Experimental Results

### Main Results: INSTA & GaussianBlendShapes datasets

| Method | INSTA Mean PSNR↑ | GBS Mean PSNR↑ | Training Time | Rendering FPS |
|------|--------------|-------------|---------|--------|
| GaussianAvatars | 29.6 | 32.8 | Slow | ~200 |
| FlashAvatar | 28.1 | 31.2 | Medium | ~300 |
| GaussianBlendShapes | 30.4 | 33.4 | Slow | ~300 |
| **RGBAvatar (K=20)** | **31.2** | **34.0** | **80s** | **~400** |

### Ablation Study: Effect of the Number of Bases

| Number of Bases K | PSNR↑ | Training Time↓ |
|--------|-------|---------|
| 10 | 30.5 | ~60s |
| 20 | 31.2 | ~80s |
| 50 | 31.3 | ~150s |
| GBS-50 Bases | 30.4 | ~300s |

### Key Findings

- **Only 20 reduced bases outperform the 50 3DMM bases** in reconstruction quality (PSNR gain of ~0.8dB), while training is 3-4 times faster.
- Batch-parallel rendering boosts GPU utilization from 60% to 100%, improving training throughput by over 6x.
- The online reconstruction quality (processing frame-by-frame) is close to offline reconstruction (using the full dataset), proving the effectiveness of the local-global sampling strategy.
- Color initialization is performed only once (when the splatting weight exceeds a threshold for the first time) but significantly accelerates initial convergence.
- The method is compatible with various 3DMM trackers and does not rely on a specific tracking framework.

## Highlights & Insights

1. **Replacing generic bases with personalized bases** is a simple yet powerful insight: the facial variation space of a specific individual is far smaller than that of the entire population, and a 20-dimensional space is sufficient for expression.
2. **System-level GPU optimizations** (batch rendering via CUDA Streams with single synchronization) offer valuable insights for all 3DGS training scenarios.
3. For the first time, **online real-time** head avatar reconstruction is achieved, enabling instantaneous avatar generation in real-time video calls.

## Limitations & Future Work

- Color initialization depends on Gaussian projection visibility, which may be unstable under extreme viewing angles.
- The reservoir sampling strategy in the online mode is basic and does not consider variations in sample importance.
- Non-FLAME modeled regions, such as hair and accessories, are not fully detailed.
- Future work could explore more adaptive selections for the number of bases and incremental base learning on streaming data.

## Related Work & Insights

- **Relationship to GaussianBlendshapes**: A direct improvement—replacing fixed 3DMM bases with reduced learnable bases.
- **Relationship to FlashAvatar**: Both bind Gaussians to the FLAME mesh, but RGBAvatar uses linear blending instead of MLP offsets.
- **Insight**: In parametric human body modeling, "fewer but better" learnable bases are more efficient than "more but generic" broad bases.

## Rating

⭐⭐⭐⭐

The core innovation of reduced bases is simple and effective. Coupled with system-level GPU optimizations, it achieves impressive speeds of 80-second reconstruction and 400 FPS rendering. Online reconstruction is a novel and practically valuable capability. The overall technical execution is solid, and the code is open-source.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Avat3r: Large Animatable Gaussian Reconstruction Model for High-fidelity 3D Head Avatars](../../ICCV2025/human_understanding/avat3r_large_animatable_gaussian_reconstruction_model_for_hi.md)
- [\[NeurIPS 2025\] VASA-3D: Lifelike Audio-Driven Gaussian Head Avatars from a Single Image](../../NeurIPS2025/human_understanding/vasa-3d_lifelike_audio-driven_gaussian_head_avatars_from_a_single_image.md)
- [\[CVPR 2025\] FATE: Full-head Gaussian Avatar with Textural Editing from Monocular Video](fate_full-head_gaussian_avatar_with_textural_editing_from_monocular_video.md)
- [\[CVPR 2025\] WildAvatar: Learning In-the-Wild 3D Avatars from the Web](wildavatar_learning_in-the-wild_3d_avatars_from_the_web.md)
- [\[ICCV 2025\] GGTalker: Talking Head Synthesis with Generalizable Gaussian Priors and Identity-Specific Adaptation](../../ICCV2025/human_understanding/ggtalker_talking_head_systhesis_with_generalizable_gaussian_priors_and_identity-.md)

</div>

<!-- RELATED:END -->
