---
title: >-
  [Paper Note] Wonderland: Navigating 3D Scenes from a Single Image
description: >-
  [CVPR 2025][3D Vision][3D Scene Generation] Wonderland proposes a pipeline for generating high-quality, wide-range 3D scenes from a single image: it first generates 3D-aware video latent variables using a video diffusion Transformer with dual-branch camera control, and then directly regresses the 3D Gaussian Splatting representation in the latent space using a Latent Large Reconstruction Model (LaLRM). This represents the first demonstration that 3D reconstruction models can…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Scene Generation"
  - "Video Diffusion Models"
  - "Single-Image Reconstruction"
  - "3D Gaussian Splatting"
  - "Feed-Forward Reconstruction"
date: 2026-05-08
content_hash: 080a5cc1ac1a8f67
---

# Wonderland: Navigating 3D Scenes from a Single Image

**Conference**: CVPR 2025  
**arXiv**: [2412.12091](https://arxiv.org/abs/2412.12091)  
**Code**: [https://snap-research.github.io/wonderland/](https://snap-research.github.io/wonderland/)  
**Area**: 3D Vision / Scene Reconstruction  
**Keywords**: 3D Scene Generation, Video Diffusion Models, Single-Image Reconstruction, 3D Gaussian Splatting, Feed-Forward Reconstruction

## TL;DR

Wonderland proposes a pipeline for generating high-quality, wide-range 3D scenes from a single image: it first generates 3D-aware video latent variables using a video diffusion Transformer with dual-branch camera control, and then directly regresses the 3D Gaussian Splatting representation in the latent space using a Latent Large Reconstruction Model (LaLRM). This represents the first demonstration that 3D reconstruction models can be efficiently built on the latent space of video diffusion models.

## Background & Motivation

**Background**: 3D scene generation typically adopts a two-stage pipeline: first generating novel views using a diffusion model, and then optimizing the 3D representation (NeRF/3DGS) on a per-scene basis. Recently, feed-forward Large Reconstruction Models (LRM series) can directly regress 3D representations from sparse views, but they are limited to object-level or narrow-view scenes.

**Limitations of Prior Work**: (1) Image diffusion models lack spatiotemporal modeling mechanisms, leading to 3D inconsistencies in the generated multi-views (e.g., deformation in occluded areas, blurred backgrounds); (2) Per-scene optimization is time-consuming; (3) Feed-forward LRMs require a large number of high-resolution input views when processing complex scenes, leading to immense memory demands (e.g., 260K+ tokens), making it difficult to scale to wide-range 3D scenes.

**Key Challenge**: Wide-range scenes require a large number of views for coverage, but the computational and memory overhead of processing these views directly in the image space is prohibitive.

**Goal**: To design a feed-forward method that takes a single image as input and outputs a wide-range, high-quality 3D scene.

**Key Insight**: Video diffusion models naturally possess multi-view 3D-aware capabilities, and their latent space provides $256\times$ spatiotemporal compression. If one could directly regress 3DGS in the latent space, more views and wider ranges could be processed under the same memory constraints.

**Core Idea**: Build a 3D reconstruction model on the latent space of a video diffusion model, leveraging the high compression rate and inherent 3D consistency of the latent space, while achieving precise camera trajectory control over the video diffusion model through a dual-branch camera control module.

## Method

### Overall Architecture

Given a single image, a video latent variable $z \in \mathbb{R}^{t \times h \times w \times c}$ along a specified camera trajectory is first generated via a video diffusion Transformer conditioned on dual-branch camera parameters. Then, LaLRM concatenates the video latents and camera pose tokens as input to a Transformer to perform feed-forward regression of pixel-aligned 3DGS parameters (position, color, rotation, scaling, opacity), ultimately yielding the 3D scene.

### Key Designs

1. **Dual-Branch Camera Guidance**:

    - **Function**: Allows the pretrained video diffusion model to precisely follow specified camera trajectories without compromising the quality of the pretrained visual generation.
    - **Mechanism**: The pixel-wise Plücker coordinates $p \in \mathbb{R}^{T \times H \times W \times 6}$ are used as camera embeddings, from which $o_{\text{ctrl}}$ and $o_{\text{lora}}$ are generated via two lightweight encoders. The ControlNet branch creates trainable copies of the first $N$ Transformer blocks, adds $o_{\text{ctrl}}$ to the video tokens, processes them through the trainable blocks, and routes the outputs to the corresponding blocks of the frozen backbone via zero-linear layers. The LoRA branch concatenates $o_{\text{lora}}$ with the video tokens, passes them through physical linear layers, and trains camera-LoRA adapters within the frozen backbone.
    - **Design Motivation**: A single conditioning method struggles to achieve precise camera control while maintaining visual quality; ControlNet provides deep conditioning integration, while LoRA offers efficient fine-tuning and adaptation to static scenes.

2. **Latent Large Reconstruction Model (LaLRM)**:

    - **Function**: Direct feed-forward regression of 3DGS parameters within the video latent space.
    - **Mechanism**: The video latent variable $z \in \mathbb{R}^{t \times h \times w \times c}$ and Plücker pose embeddings are respectively patchified into token sequences of the same length, concatenated, and fed into the Transformer blocks. The output is decoded via a 3D deconvolutional layer into a high-resolution Gaussian feature map $G \in \mathbb{R}^{(T \times H \times W) \times 12}$ (RGB + scaling + rotation + opacity + ray distance), establishing pixel-level correspondences with the source video. Compared to operating in image space (which requires 260K+ tokens), the latent space requires only about 1K tokens ($256\times$ compression).
    - **Design Motivation**: The video latent space retains perceptually equivalent information (since the VAE is trained with perceptual loss), while the $256\times$ compression makes wide-range scene reconstruction feasible under memory constraints.

3. **Progressive Training Strategy**:

    - **Function**: Solves the challenging training issue of bridging the large domain gap from video latent space to 3DGS.
    - **Mechanism**: LaLRM is trained in three stages: (1) training on ground-truth (GT) video latents, supervised only by the input views; (2) introducing supervision from more unseen views to ensure 3D consistency; (3) mixing GT latents with latents generated by the diffusion model to adapt the model to the distribution discrepancy during inference.
    - **Design Motivation**: Training directly on generated latents is susceptible to generation noise; the progressive strategy allows the model to gradually adapt from paired data to generated data.

### Loss & Training

The reconstruction loss is formulated as $\mathcal{L}_{\text{recon}} = \lambda_1 \mathcal{L}_{\text{mse}} + \lambda_2 \mathcal{L}_{\text{perc}}$ (MSE + VGG-19 perceptual loss), computed over $V$ randomly selected supervised views. The video diffusion model is trained with the standard denoising objective. Training datasets include large-scale 3D scene datasets such as RealEstate10K and DL3DV.

## Key Experimental Results

### Main Results — Zero-Shot Novel View Synthesis

| Method | RealEstate10K PSNR ↑ | DL3DV PSNR ↑ | Inference Time ↓ | Per-Scene Optimization Required |
|------|---------|---------|---------|---------|
| ZeroNVS | 16.8 | 15.2 | ~10 min | Yes |
| ReconFusion | 18.3 | 16.9 | ~5 min | Yes |
| MotionCtrl+opt | 19.1 | 17.5 | ~8 min | Yes |
| Wonderland | **21.4** | **19.8** | **~30 sec** | No |

### Camera Control Accuracy

| Method | Rotation Error (°) ↓ | Translation Error ↓ |
|------|-----------|---------|
| MotionCtrl | 8.2 | 0.31 |
| CameraCtrl | 5.7 | 0.22 |
| VD3D | 4.9 | 0.18 |
| Wonderland | **3.1** | **0.12** |

### Key Findings

- Wonderland achieves a PSNR of **21.4 dB** on RealEstate10K, which is 2.3 dB higher than the strongest baseline, MotionCtrl+opt, while reducing the inference time from ~8 min to **~30 sec** (including both video generation and 3D reconstruction).
- The rotation error of the dual-branch camera control is only **3.1°**, representing a 62% reduction compared to MotionCtrl (8.2°).
- Generalization performance on out-of-distribution (OOD) images (e.g., artistic paintings, concept art) is significantly superior to other methods, thanks to the prior knowledge of the pretrained video diffusion model.
- This work is the first to demonstrate that building a 3D reconstruction model directly on the video latent space is feasible and highly efficient.

## Highlights & Insights

- **"3D reconstruction in the latent space"** represents a significant methodological innovation, bridging the gap between the latent space of generative models and 3D representations.
- The complementary design of **dual-branch camera control** (ControlNet + LoRA) is elegant, where ControlNet provides deep conditioning integration and LoRA offers lightweight adaptation.
- The finding that the **$256\times$ compressed latent space still retains sufficient 3D information** is highly impressive.

## Limitations & Future Work

- The upper bound of scene quality is limited by the generation capability of the video diffusion model (e.g., hallucinated textures, repetitive structures).
- Single-image input still suffers from uncertainty in highly occluded scenes.
- The current LaLRM is constrained by the number of video frames; very long trajectories require chunkwise processing and merging.
- The progressive training strategy increases training complexity.

## Related Work & Insights

- **vs ZeroNVS/ReconFusion**: These are based on image diffusion models and lack 3D consistency; Wonderland's video diffusion model inherently possesses spatiotemporal consistency.
- **vs LRM/InstantMesh**: These methods operate in the image space, where the number of tokens limits the scene scale; LaLRM operates in the $256\times$ compressed latent space, allowing it to handle wider ranges.
- **vs MotionCtrl/CameraCtrl**: These methods only perform video generation but do not construct 3D representations; Wonderland unifies video generation and 3D reconstruction within the latent space.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Building a 3D reconstruction model on the video latent space is a pioneering and highly impactful concept.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple datasets and compared with various baselines; thoroughly tested on out-of-distribution generalization.
- Writing Quality: ⭐⭐⭐⭐⭐ High-quality architecture diagrams; motivations for each module are clearly explained.
- Value: ⭐⭐⭐⭐⭐ Opens up a new paradigm for "video diffusion model-driven 3D generation."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation](midi_multi-instance_diffusion_for_single_image_to_3d_scene_generation.md)
- [\[CVPR 2025\] Disco4D: Disentangled 4D Human Generation and Animation from a Single Image](disco4d_disentangled_4d_human_generation_and_animation_from_a_single_image.md)
- [\[CVPR 2025\] WonderWorld: Interactive 3D Scene Generation from a Single Image](wonderworld_interactive_3d_scene_generation_from_a_single_image.md)
- [\[CVPR 2025\] Symmetry Strikes Back: From Single-Image Symmetry Detection to 3D Generation](symmetry_strikes_back_from_single-image_symmetry_detection_to_3d_generation.md)
- [\[CVPR 2025\] SpectroMotion: Dynamic 3D Reconstruction of Specular Scenes](spectromotion_dynamic_3d_reconstruction_of_specular_scenes.md)

</div>

<!-- RELATED:END -->
