---
title: >-
  [Paper Note] 2S-ODIS: Two-Stage Omni-Directional Image Synthesis by Geometric Distortion Correction
description: >-
  [ECCV 2024][Image Generation][Panoramic Image Generation] 2S-ODIS utilizes a pre-trained VQGAN (without fine-tuning) to synthesize panoramic images via a two-stage architecture: the first stage generates a low-resolution coarse ERP image, and the second stage corrects geometric distortions by generating and fusing 26 NFoV perspective images. This reduces training time from 14 days to 4 days while achieving superior image quality.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Panoramic Image Generation"
  - "VQGAN"
  - "MaskGIT"
  - "Equirectangular Projection"
  - "Geometric Distortion Correction"
date: 2026-05-08
content_hash: fdf2593eb8c8eceb
---

# 2S-ODIS: Two-Stage Omni-Directional Image Synthesis by Geometric Distortion Correction

**Conference**: ECCV 2024  
**arXiv**: [2409.09969](https://arxiv.org/abs/2409.09969)  
**Code**: No public repository (core network architecture source code is provided in the appendix of the paper)  
**Area**: Image Generation  
**Keywords**: Panoramic Image Generation, VQGAN, MaskGIT, Equirectangular Projection, Geometric Distortion Correction  

## TL;DR
2S-ODIS utilizes a pre-trained VQGAN (without fine-tuning) to synthesize panoramic images via a two-stage architecture: the first stage generates a low-resolution coarse ERP image, and the second stage corrects geometric distortions by generating and fusing 26 NFoV perspective images. This reduces training time from 14 days to 4 days while achieving superior image quality.

## Background & Motivation
Panoramic images (360-degree images) are widely used in VR and social media, but data remains scarce due to the high cost of dedicated cameras. Existing generation methods face two major issues: (1) GAN-based methods suffer from training instability; (2) VQGAN-based methods (such as OmniDreamer) require retraining VQGAN on panoramic datasets to learn geographic distortions under Equirectangular Projection (ERP), which alone takes a week. **Key Challenge**: Pre-trained VQGANs perform excellently on NFoV (Normal Field-of-View) images but fail to handle the severe distortion in the polar regions of ERP images (e.g., straight lines becoming curved).

## Core Problem
How to leverage VQGAN pre-trained on large-scale NFoV datasets (like ImageNet) to generate high-quality panoramic images, avoiding the need for retraining VQGAN on panoramic data?

## Method

### Overall Architecture
A two-stage pipeline where both stages can be trained in parallel:
- **Stage 1 (Low-Resolution Model)**: Takes conditional images (NFoV embedded in ERP) as input, utilizing MaskGIT sampling to generate a 256×512 coarse ERP panoramic image.
- **Stage 2 (High-Resolution Model)**: Conditioned on the Stage 1 output, it generates 26 NFoV images of size 256×256 along the 26 facial direction vectors of a rhombicuboctahedron, which are then weighted and fused into a final 1024×2048 ERP panoramic image.

### Key Designs
1. **Key Insight for Avoiding VQGAN Fine-Tuning**: Pre-trained VQGANs cannot correctly reconstruct polar regions in the ERP format (causing texture distortions and edge discontinuities) but can perfectly reconstruct them in extracted NFoV perspectives. Therefore, Stage 2 operates in the NFoV space, completely bypassing ERP distortion issues.

2. **MaxViT replacing Transformer**: Both stages utilize an 8-layer MaxViT instead of the original Transformer. The low-resolution model employs circular padding to ensure ERP continuity in the horizontal direction. For the high-resolution model, block attention is computed within each local NFoV image, while grid attention is calculated across corresponding positions of different NFoV images.

3. **26-Directional NFoV Image Fusion**: The 26 directions correspond to the face normal vectors of a rhombicuboctahedron, with FOV = 60° resulting in overlapping regions. Overlapping areas are fused via weight-to-distance mapping: regions closer to the center of each NFoV image receive higher weights, ensuring smooth transitions.

4. **MaskGIT Parallel Sampling**: Instead of autoregressive token prediction for VQGAN codes, parallel prediction is performed across all positions in $T=16$ steps. At each step, predictions with the highest confidence are retained, and the remaining ones are re-masked and re-sampled. This accelerates inference speed from 39.33 seconds to 1.54 seconds.

### Loss & Training
- Both stages utilize cross-entropy loss to train the model to predict the masked VQGAN codes.
- During training, randomly masked panoramic images are used as conditional inputs (simulating inference-time input) to enhance diversity.
- Both stages can be trained in parallel (2 days each), totaling 4 days of training.
- Optimizer: AdamW, learning rate = 0.001, scheduled with ExponentialLR (0.95 multiplier per 5,000 iterations).

## Key Experimental Results

| Method | IS ↑ | FID ↓ | LPIPS ↑ |
|------|------|-------|---------|
| **2S-ODIS (4 days)** | **5.969** | **18.263** | **0.662** |
| 2S-ODIS (2 days) | 5.857 | 18.656 | 0.668 |
| OmniDreamer (14 days) | 4.458 | 23.101 | 0.655 |
| CNN-based cGAN | 4.684 | 40.049 | 0.633 |
| MLPMixer-based cGAN | 4.402 | 47.690 | 0.634 |
| LAMA | 5.784 | 69.485 | 0.478 |

Dataset: SUN360 (47,938 train + 5,000 test, outdoor images)  
Inference Speed: 1.54 seconds vs OmniDreamer's 39.33 seconds

### Ablation Study
- The two-stage structure is indispensable: Stage 1 only $\rightarrow$ FID 28.3; Stage 2 only $\rightarrow$ FID 52.5; both stages $\rightarrow$ FID 18.3.
- Stage 2 performs better when using low-resolution images as conditions rather than VQGAN codes (FID 18.3 vs. 21.8).
- MaxViT outperforms other architectures (Transformer, MultiAxisTransformer, NeighborAttention) in both stages.
- Even with only 2 days of training, the model outperforms OmniDreamer (14 days), demonstrating exceptional training efficiency.

## Highlights & Insights
- **The concept of "not fine-tuning pre-trained models"** is highly practical: instead of modifying the pre-trained model, the problem is mapped to the domain where the pre-trained model already excels (NFoV).
- **Decomposing panoramic images into multiple NFoV perspectives** is a general strategy to address ERP distortion, compatible with migration to other tasks like panoramic semantic segmentation and object detection.
- **MaskGIT parallel sampling** is 25 times faster than autoregressive generation, substantially enhancing inference efficiency without compromising quality.
- The coarse-to-fine two-stage framework is a classic paradigm in image generation; this work flexibly adapts each stage to a distinct image space.

## Limitations & Future Work
- Potential discontinuities might emerge across the boundaries of the 26 NFoV images (although guided by the global coarse image, explicit consistency constraints are lacking).
- Transcoding the panoramic dataset through the VQGAN encoder requires 1-2 days (which can be accelerated via model distillation).
- The configuration is fixed to 26 directions with 60° FOV, and optimal settings remain unexplored.
- It only supports conditional image inputs, neglecting text-to-image extensions.
- Comparisons with recent diffusion-based state-of-the-art models are lacking.

## Related Work & Insights
- **OmniDreamer**: Requires a week of VQGAN training and has an autoregressive inference time of 39s; 2S-ODIS bypasses VQGAN training and achieves parallel inference in 1.5s.
- **AOGNet**: Stable Diffusion-based autoregressive outpainting, exhibiting extremely slow inference; 2S-ODIS gains higher efficiency via parallel NFoV generation and fusion.
- **LAMA**: General-purpose inpainting method, whose FID (69.5) is far worse than specialized panoramic methods.

## Related Work & Insights
- The strategy of "transforming a difficult domain (ERP) into a domain where pre-trained models excel (NFoV)" possesses wide generalizability.
- The idea of 26-direction decomposition and fusion can inspire research on multi-view consistent generation.

## Rating
- Novelty: ⭐⭐⭐⭐ The concepts of two-stage coarse-to-fine and NFoV decomposition are not entirely new, but the integration design and the insight of "no VQGAN fine-tuning" are highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid ablation studies, but evaluated on a single dataset (SUN360 only) with a lack of comparisons against diffusion models.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear methodological motivation with convincing visualizations.
- Value: ⭐⭐⭐ Panoramic image generation is not my primary focus, but the domain transformation strategy is highly referable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Diffusion-based Image-to-Image Translation by Noise Correction via Prompt Interpolation](diffusion-based_image-to-image_translation_by_noise_correction_via_prompt_interp.md)
- [\[ECCV 2024\] FineMatch: Aspect-based Fine-grained Image and Text Mismatch Detection and Correction](finematch_aspect-based_fine-grained_image_and_text_mismatch_detection_and_correc.md)
- [\[ECCV 2024\] Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)
- [\[ECCV 2024\] Editable Image Elements for Controllable Synthesis](editable_image_elements_for_controllable_synthesis.md)
- [\[ICML 2026\] Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](../../ICML2026/image_generation/stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)

</div>

<!-- RELATED:END -->
