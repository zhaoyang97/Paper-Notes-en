---
title: >-
  [Paper Note] LivePhoto: Real Image Animation with Text-guided Motion Control
description: >-
  [ECCV 2024][Image Generation] The LivePhoto image animation framework is proposed to address the ambiguity of text-to-motion mapping through a motion intensity estimation module and a text reweighting module. It achieves high-quality video generation based on real images and text descriptions, allowing users additional control over motion intensity.
tags:
  - "ECCV 2024"
  - "Image Generation"
date: 2026-05-08
content_hash: 077c2700feec127e
---

# LivePhoto: Real Image Animation with Text-guided Motion Control

**Conference**: ECCV 2024  
**arXiv**: [2312.02928](https://arxiv.org/abs/2312.02928)  
**Area**: Image Generation

## TL;DR

The LivePhoto image animation framework is proposed to address the ambiguity of text-to-motion mapping through a motion intensity estimation module and a text reweighting module. It achieves high-quality video generation based on real images and text descriptions, allowing users additional control over motion intensity.

## Background & Motivation

Although text-guided video generation has made progress, existing methods suffer from an overlooked problem: **text can only control the spatial content of the video but struggles to control temporal motion**. When users input an image and a motion description (e.g., "shaking head", "camera zoom in"), the generated videos are often either nearly static or excessively violent.

This work analyzes two core causes:
1. Text cannot fully describe the speed and magnitude of motion (e.g., "shaking head" lacks velocity information), which leads to multiple motion intensities for the same text description, causing optimization ambiguity.
2. Text contains both "content descriptions" and "motion descriptions." When the content description conflicts with the reference image, the influence of the entire text is suppressed, consequently weakening the motion description as well.

## Method

### Overall Architecture

LivePhoto is built on top of a frozen Stable Diffusion v1.5. The inputs include a reference image, text, and motion intensity. Core components:
- The reference image latent representation is concatenated with noise as input to the UNet (containing frame embeddings and intensity embeddings, totaling 10 channels).
- A content encoder (DINOv2) extracts image patch tokens, which are injected via cross-attention.
- A trainable Motion Module captures temporal relationships across frames.
- A text reweighting module adjusts the weights of text embeddings.

### Key Designs

**Motion Intensity Estimation**: SSIM is used to measure the structural similarity between adjacent frames to parameterize motion intensity:

$$\mathbf{I}(\mathbf{X}^n) = \frac{1}{n} \sum_{i=0}^{n-2} \text{SSIM}(\mathbf{x}^i, \mathbf{x}^{i+1})$$

The motion intensity of training data is divided into 10 levels from 1 to 10 and converted into a 1-channel embedding map concatenated with the UNet input. During inference, level 5 is used by default, and users can adjust it freely.

**Text Reweighting**: A 3-layer Transformer encoder and a linear projection layer are added after the CLIP text encoder to predict a weight (ranging from 0 to 1) for each token, which is then multiplied by the corresponding text embedding. The module automatically learns to emphasize motion-related words (e.g., "waving") and suppress content description words that might conflict with the reference image.

**Prior Inversion**: During inference, the DDIM inversion noise of the reference image is added to the initial noise to provide an appearance prior:

$$\tilde{\mathbf{z}}_T^n = \alpha^n \cdot \text{Inv}(\mathbf{r}_0) + (1-\alpha^n) \cdot \mathbf{z}_T^n$$

### Loss & Training

A simple MSE noise prediction loss is employed, with text randomly dropped with a probability of 0.5 during training to achieve classifier-free guidance.

## Key Experimental Results

### Main Results

User study scores (on a 5-point scale), comparing with VideoComposer, Pikalabs, and GEN-2:

| Method | Image Consistency ↑ | Text Consistency ↑ | Content Quality ↑ | Motion Quality ↑ |
|------|:-:|:-:|:-:|:-:|
| VideoComposer | 2.8 | 3.5 | 3.6 | 3.6 |
| Pikalabs | 3.9 | 2.7 | **4.6** | 3.1 |
| GEN-2 | 3.7 | 2.5 | **4.8** | 3.3 |
| **LivePhoto** | 3.6 | **4.7** | 3.7 | **3.9** |

LivePhoto significantly leads in text consistency and motion quality, validating its motion control capabilities.

### Ablation Study

Step-wise ablation of the image content guidance modules (on the WebVID validation set):

| Method | DINO Score ↑ | CLIP Score ↑ |
|------|:-:|:-:|
| Reference Latent Only | 82.3 | 91.7 |
| + Content Encoder | 85.9 | 93.2 |
| + Prior Inversion | **90.8** | **95.2** |

Ablation of new modules:

| Method | DINO Score ↑ | CLIP Score ↑ |
|------|:-:|:-:|
| LivePhoto (Full) | **90.8** | **95.2** |
| w/o Motion Intensity Guidance | 90.3 | 94.8 |
| w/o Text Reweighting | 90.1 | 93.9 |

### Key Findings

- Motion intensity guidance solves the binary dilemma of "either static or overly moving," providing continuous controllability.
- Text reweighting effectively distinguishes motion descriptions from content descriptions, preventing content conflicts from undermining motion control.
- The model generalizes across domains: it is effective for animals, humans, cartoons, and natural landscapes.
- It can generate "out-of-thin-air" contents (e.g., pouring water into an empty cup, simulating lightning).

## Highlights & Insights

- **Precise Problem Analysis**: Thorough analysis of the root causes of the text's inability to control motion (ambiguity and content conflicts).
- **Parameterized Motion Intensity**: The design of using SSIM to quantify motion intensity and discretizing it into 10 levels is simple yet practical.
- **Interpretable Text Reweighting**: The module automatically learns to emphasize verbs and motion-related words.
- **Academic Method vs. Commercial Products**: Significantly outperforms GEN-2 and Pikalabs in text consistency.

## Limitations & Future Work

- Implemented based on SD-1.5, with an output resolution of only 256×256.
- Trained on the WebVID dataset; generalizing to specific domains may require additional adaptation.
- Extremely high motion intensity (level 10) may introduce motion blur.

## Rating

⭐⭐⭐⭐ In-depth problem analysis; the design of motion intensity and text reweighting is ingenious, and the model significantly leads in the dimension of text-guided motion control.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] ReNoise: Real Image Inversion Through Iterative Noising](renoise_real_image_inversion_through_iterative_noising.md)
- [\[ECCV 2024\] Text2Place: Affordance-aware Text Guided Human Placement](text2place_affordance-aware_text_guided_human_placement.md)

</div>

<!-- RELATED:END -->
