---
title: >-
  [Paper Note] SIR-DIFF: Sparse Image Sets Restoration with Multi-View Diffusion Model
description: >-
  [CVPR 2025][Image Generation][Multi-view image restoration] SIR-Diff is proposed, a multi-view diffusion model that achieves cross-view consistent image restoration by jointly denoising multiple degraded images of the same scene. It integrates multi-view complementary information using a Spatial-3D ResNet and a 3D Self-Attention Transformer, outperforming single-view and video restoration methods in deblurring and super-resolution tasks.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Multi-view image restoration"
  - "Diffusion model"
  - "3D consistency"
  - "Deblurring"
  - "Super-resolution"
date: 2026-05-08
content_hash: 0f308dc226b6a490
---

# SIR-DIFF: Sparse Image Sets Restoration with Multi-View Diffusion Model

**Conference**: CVPR 2025  
**arXiv**: [2503.14463](https://arxiv.org/abs/2503.14463)  
**Code**: [Project Page](https://myc634.github.io/sirdiff/)  
**Area**: 3D Vision  
**Keywords**: Multi-view image restoration, Diffusion model, 3D consistency, Deblurring, Super-resolution

## TL;DR

SIR-Diff is proposed, a multi-view diffusion model that achieves cross-view consistent image restoration by jointly denoising multiple degraded images of the same scene. It integrates multi-view complementary information using a Spatial-3D ResNet and a 3D Self-Attention Transformer, outperforming single-view and video restoration methods in deblurring and super-resolution tasks.

## Background & Motivation

Traditional image restoration methods process each degraded image independently, whereas in practice, multiple photos of the same scene are often captured. The core hypothesis is that multiple degraded photos of the same scene contain complementary information, and joint processing can better constrain the restoration problem. For instance, a region blurred in one photo might be sharp in another view.

This multi-view restoration is particularly crucial for 3D computer vision tasks, such as SLAM and novel view synthesis, which rely on the geometric consistency of multi-view images. Single-view restoration methods inevitably introduce cross-view inconsistencies, violating the underlying 3D scene assumptions. Traditional multi-image fusion (e.g., sub-pixel displacement fusion in classical super-resolution) is limited to simple scenes.

SIR-Diff reformulates image restoration as a multi-view collaborative restoration task, implicitly fusing overlapping information in general multi-view images through neural attention mechanisms.

## Method

### Overall Architecture

SIR-Diff is based on the latent diffusion model architecture of SD 2.1. The inputs are a set of degraded images $\{I^c\}_{i=1}^N$ (blurred/low-resolution), and the outputs are restored 3D-consistent images. The core modification is extending the UNet of SD into a multi-view joint denoising model, which consists of two key components: Spatial-3D ResNet and 3D Self-Attention Transformer. The degraded images are encoded by VAE and then channel-concatenated with the noisy latent representation as conditional inputs.

### Key Designs

#### Key Design 1: Spatial-3D ResNet

**Function**: To simultaneously capture 2D spatial and cross-view 3D relationships within convolutional layers.

**Mechanism**: Parallel 3D convolutional layers are added alongside the standard 2D convolutions of SD. 2D convolutions are initialized with SD 2.1 weights to process spatial information, while 3D convolutions are initialized with Stable Video Diffusion (SVD) weights to handle cross-view information. The outputs of both are blended via learnable weights: $O_{\text{ResNet}} = \sigma(\alpha) \times O_{2D} + \sigma(1-\alpha) \times O_{3D}$.

**Design Motivation**: Pure 2D convolutions cannot capture cross-view geometric relationships. Training 3D convolutions from scratch is unstable and time-consuming. Since temporal similarity (neighboring frames in video) shares commonalities with spatial similarity (neighboring views in multi-view set), the temporal convolution weights of SVD can be used to initialize the spatial 3D convolutions.

#### Key Design 2: 3D Self-Attention Transformer

**Function**: To enable each token to attend to all spatial positions across all views, achieving global cross-view information fusion.

**Mechanism**: The latent features of $N$ views are patched into $N \times p$ tokens, where each token computes attention scores against all $N \times p$ tokens. Inspired by CAT3D, this module is only injected into low-resolution layers to control computational overhead, and the cross-attention module is removed to accelerate convergence.

**Design Motivation**: Self-attention allows each pixel in every view to access information from all other views, maximizing information complementarity. During inference, it can adapt to different numbers of input frames, offering flexibility.

#### Key Design 3: Unified Degradation Input Encoder

**Function**: To handle different types of degraded inputs (blurred, low-resolution, clean RGB).

**Mechanism**: The VAE encoder of SD 2.1 is used to encode degraded images, and a zero-initialized convolutional layer is employed to adapt to the change in channel dimension (arising from channel concatenation of the degradation condition and the noisy latent representation).

**Design Motivation**: A unified encoding scheme enables the model to simultaneously handle multiple degradation types, enhancing system versatility.

### Loss & Training

The standard diffusion training loss is employed: $\mathcal{L} = \mathbb{E}\|\epsilon - \hat{\epsilon}\|_2^2$, supervising the UNet to predict noise consistent with that added during the forward diffusion process. The model is trained on synthetic datasets (Hypersim + TartanAir).

## Key Experimental Results

### Multi-view Deblurring (Zero-shot Evaluation)

| Method | Scannet++ FID ↓ | Scannet++ LPIPS ↓ | Scannet++ VConsis ↓ |
|------|--------|---------|----------|
| PromptIR | 81.28 | 0.248 | 7.81 |
| Restormer | 49.72 | 0.232 | 6.52 |
| VRT (Video) | 134.5 | 0.371 | 7.67 |
| SIR-Diff (Single Frame) | 81.58 | 0.247 | 6.45 |
| **SIR-Diff** | **40.09** | **0.160** | **5.75** |

### Key Findings

- SIR-Diff outperforms single-view and video restoration methods in FID, LPIPS, and visual consistency metrics across all datasets.
- The multi-view version significantly outperforms the single-frame version (FID 40.09 vs 81.58), validating the value of multi-view complementary information.
- The video restoration method (VRT) performs worst in sparse multi-view scenarios because it assumes temporal continuity rather than sparsely sampled distinct views.
- Trained on synthetic data, it demonstrates strong zero-shot transfer capabilities to real-world scenes (Scannet++, ETH3D, CO3D).
- Restored images can be directly utilized for 3DGS reconstruction, significantly boosting the quality of novel view synthesis and feature matching success rates.
- Inference supports an arbitrary number of input frames, while training uses only a small number of images.

## Highlights & Insights

1. **Paradigm Shift**: Shifting image restoration from "independent single-view processing" to "collaborative multi-view restoration", fully leveraging the complementary information from multi-capture scenes.
2. **SVD Initialization Trick**: The weight transfer from temporal domain $\rightarrow$ spatial domain cleverly utilizes the geometric understanding capabilities of existing video diffusion models.
3. **VConsis Metric**: A new metric is proposed to evaluate the internal consistency of generated view sets, offering valuable reference for the multi-view generation field.

## Limitations & Future Work

- Camera poses must be known to compute overlapping areas between views (during training), which limits certain application scenarios.
- The computational overhead of 3D self-attention scales quadratically with the number of views $N$, posing a potential bottleneck when handling large numbers of views.
- Currently, only deblurring and super-resolution degradations are addressed; other types (e.g., dehazing, deraining) remain untested.
- Training on synthetic data may lead to domain gaps, especially regarding complex real-world degradations.

## Related Work & Insights

- **MVDream / CAT3D**: Multi-view generative diffusion models, providing design references for the 3D self-attention layer.
- **Stable Video Diffusion**: Its 3D convolution weights are reused to initialize the 3D convolutions for multi-view understanding.
- **Restormer / PromptIR**: Strong single-view restoration baselines; this work demonstrates that multi-view approaches can outperform them.

## Rating

⭐⭐⭐⭐ — The problem definition is clear and practical (multi-view restoration), the SVD weight transfer trick is clever, and the zero-shot generalization is impressive. It shows prominent downstream value for 3D reconstruction pipelines (e.g., 3DGS).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Pippo: High-Resolution Multi-View Humans from a Single Image](pippo_high-resolution_multi-view_humans_from_a_single_image.md)
- [\[CVPR 2026\] InstructMix2Mix: Consistent Sparse-View Editing Through Multi-View Model Personalization](../../CVPR2026/image_generation/instructmix2mix_consistent_sparse-view_editing_through_multi-view_model_personal.md)
- [\[CVPR 2025\] OSDFace: One-Step Diffusion Model for Face Restoration](osdface_one-step_diffusion_model_for_face_restoration.md)
- [\[NeurIPS 2025\] A Data-Driven Prism: Multi-View Source Separation with Diffusion Model Priors](../../NeurIPS2025/image_generation/a_data-driven_prism_multi-view_source_separation_with_diffusion_model_priors.md)
- [\[CVPR 2025\] ZoomLDM: Latent Diffusion Model for Multi-Scale Image Generation](zoomldm_latent_diffusion_model_for_multi-scale_image_generation.md)

</div>

<!-- RELATED:END -->
