---
title: >-
  [Paper Note] VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models
description: >-
  [ECCV 2024][Video Generation] This work proposes utilizing a pre-trained video diffusion model (EMU Video) as a multi-view data engine. By fine-tuning it to generate 3-D consistent multi-view videos, the authors construct approximately 3 million synthetic data points to train a feedforward 3D generative model, VFusion3D. This enables generating 3D assets from a single image in seconds, achieving a user preference rate of over 90%.
tags:
  - "ECCV 2024"
  - "Video Generation"
date: 2026-05-08
content_hash: da5655222cf2d838
---

# VFusion3D: Learning Scalable 3D Generative Models from Video Diffusion Models

**Conference**: ECCV 2024  
**arXiv**: [2403.12034](https://arxiv.org/abs/2403.12034)  
**Area**: Video Generation

## TL;DR

This work proposes utilizing a pre-trained video diffusion model (EMU Video) as a multi-view data engine. By fine-tuning it to generate 3-D consistent multi-view videos, the authors construct approximately 3 million synthetic data points to train a feedforward 3D generative model, VFusion3D. This enables generating 3D assets from a single image in seconds, achieving a user preference rate of over 90%.

## Background & Motivation

### Background

**Background**: The core obstacle to building foundational 3D generative models is the **extreme scarcity of 3D data**—the largest public datasets are only in the tens of millions and suffer from inconsistent quality, falling far short of the scale of text-image and video data.

### Limitations of Prior Work

**Limitations of Prior Work**: The powerful capabilities of foundational models (GPT, diffusion models) stem from joint scaling of data and models. The 3D field lacks the data foundation required to satisfy this condition.

### Key Challenge

**Key Challenge**: Existing approaches either rely on time-consuming Score Distillation Sampling (SDS) or are limited by the quality of scarce 3D training data.

### Proposed Solution

**Key Insight**: Video diffusion models, trained on massive text, image, and video data, contain an implicit understanding of the 3D world (camera motion, multi-view consistency) that can be unlocked as a 3D data generator.

## Method

### Overall Architecture

Three-stage pipeline:
1. **Fine-tuning the Video Diffusion Model**: Fine-tune EMU Video using multi-view videos rendered from 100k 3D data to turn it into a multi-view data engine.
2. **Large-scale Synthetic Data Generation**: Collect 4 million text prompts, generating and filtering to obtain 2.7 million high-quality synthetic multi-view data points.
3. **Training VFusion3D**: Train a feedforward 3D generative model using the LRM architecture on the synthetic data, followed by fine-tuning with 3D data.

### Key Designs

**EMU Video Fine-tuning**:
- Freeze all parameters and only fine-tune the temporal convolutional and attention layers to preserve generation quality.
- Render 16 views for each 3D asset (uniformly spaced $360^\circ$ azimuth angles, random elevation angles from $0 \sim \pi/4$).
- No camera parameters are required as input; the model infers the trajectory from the implicit view perspective in the first frame.

**Data Filtering and Annotation**:
- Manually label 2,000 videos to train an SVM classifier for filtering low-quality data (based on DINO features), retaining 2.7 million samples.
- Train an elevation estimator (MLP on DINO features) to annotate the elevation angle for each multi-view video.

**VFusion3D Training Strategy** (Improvements for Synthetic Data):
- **Multi-stage Training**: Gradually increase rendering resolution from 128 to 384 to stabilize training.
- **Image-level instead of Pixel-level Supervision**: Replace $L_1/L_2$ loss with LPIPS to tolerate minor inconsistencies in synthetic data.
- **Opacity Loss**: Use a saliency detection model to obtain foreground masks and suppress background noise.
- **Camera Noise Injection**: Add random offsets to the intrinsic and extrinsic camera matrices to enhance robustness.

### Loss & Training

- Training Phase: LPIPS loss + opacity loss (foreground mask supervision).
- Fine-tuning Phase: Further fine-tune on 100k 3D data with multi-view ground truths, combining with synthetic data to achieve optimal results.

## Key Experimental Results

### Main Results

Quantitative comparison of single-image 3D reconstruction:

| Method | CLIP Text Similarity ↑ | CLIP Image Similarity ↑ |
|------|------------------------|-------------------------|
| OpenLRM | 0.234 | 0.793 |
| LGM | 0.241 | 0.796 |
| **VFusion3D** | **0.253** | **0.851** |

Text-to-3D generation comparison:

| Method | CLIP Text Sim ↑ | CLIP Image Sim ↑ |
|------|-----------------|-------------------|
| DreamFusion | 0.261 | 0.640 |
| Magic3D | 0.293 | 0.687 |
| ProlificDreamer | 0.293 | 0.699 |
| MVDream | 0.284 | 0.688 |
| OpenLRM | 0.255 | 0.826 |
| LGM | 0.270 | 0.832 |
| **VFusion3D** | 0.272 | **0.899** |

User Study: VFusion3D outperforms other methods in both generation quality and image fidelity with a preference rate exceeding 90%.

### Ablation Study

Step-by-step ablation of training strategies:

| Component | SSIM ↑ | LPIPS ↓ | CLIP Text Sim ↑ | CLIP Image Sim ↑ |
|------|--------|---------|-----------------|-------------------|
| Baseline | 0.826 | 0.206 | 0.223 | 0.712 |
| + Multi-stage training | 0.829 | 0.168 | 0.249 | 0.801 |
| + No pixel-level loss | 0.831 | 0.167 | 0.257 | 0.798 |
| + Opacity supervision | 0.831 | 0.167 | 0.256 | 0.802 |
| + Camera noise | 0.830 | 0.169 | 0.252 | 0.800 |

3D Data vs. Synthetic Multi-view Data:

| Data Type | SSIM ↑ | LPIPS ↓ | CLIP Text Sim ↑ | CLIP Image Sim ↑ |
|----------|--------|---------|-----------------|-------------------|
| 3D Data Only | 0.839 | 0.161 | 0.205 | 0.631 |
| Synthetic Data Only | 0.832 | 0.160 | 0.261 | 0.839 |
| **Combination of Both** | **0.842** | **0.143** | **0.266** | **0.836** |

### Key Findings

1. Multi-stage training contributes the most (with CLIP Image Sim jumping from 0.712 to 0.801), serving as the key to stabilizing learning from synthetic data.
2. 3D data is more efficient at reconstructing common objects (100k samples can match 2.7 million synthetic data points), but synthetic data shows stronger generalization to rare objects.
3. The two data types are complementary: synthetic pre-training combined with 3D fine-tuning yields the best performance.
4. Generation quality continues to scale with synthetic data size, validating the scalability of the proposed method.

## Highlights & Insights

- **Paradigm Innovation**: Repositioning the video diffusion model as a 3D data engine, replacing SDS with explicit knowledge distillation to elegantly solve the scarcity of 3D data.
- **Scalability**: Theoretically capable of generating infinite synthetic data, with performance continuing to grow along with data volume.
- **Engineering Utility**: Feedforward inference is completed in seconds without requiring per-shape optimization, vastly outperforming distillation-based methods.
- **Simple Design Philosophy**: Focuses on data and training strategies without modifying the LRM architecture, demonstrating that data is more critical than architecture.

## Limitations & Future Work

- The fine-tuned video diffusion model struggles with specific categories (e.g., vehicles, text-related objects), inheriting the limitations of the pre-trained model.
- Despite the large volume, synthetic data still exhibits 3D inconsistencies, requiring complex filtering and training strategies to compensate.
- The 3D representation employs tri-plane + NeRF, where resolution and detail details are limited by volume rendering.
- Training requires 128 A100 GPUs for approximately 6 days, presenting a high resource barrier.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MOFA-Video: Controllable Image Animation via Generative Motion Field Adaptions in Frozen Image-to-Video Diffusion Model](mofa-video_controllable_image_animation_via_generative_motion_field_adaptions_in.md)
- [\[ECCV 2024\] Videoshop: Localized Semantic Video Editing with Noise-Extrapolated Diffusion Inversion](videoshop_localized_semantic_video_editing_with_noise-extrapolated_diffusion_inv.md)
- [\[ECCV 2024\] MagDiff: Multi-Alignment Diffusion for High-Fidelity Video Generation and Editing](magdiff_multi-alignment_diffusion_for_high-fidelity_video_generation_and_editing.md)
- [\[ICCV 2025\] LeanVAE: An Ultra-Efficient Reconstruction VAE for Video Diffusion Models](../../ICCV2025/video_generation/leanvae_an_ultra-efficient_reconstruction_vae_for_video_diffusion_models.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](../../ICCV2025/video_generation/vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)

</div>

<!-- RELATED:END -->
