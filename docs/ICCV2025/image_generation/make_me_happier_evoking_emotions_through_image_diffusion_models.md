---
title: >-
  [Paper Note] Make Me Happier: Evoking Emotions Through Image Diffusion Models
description: >-
  [ICCV 2025][Image Generation][Affective Image Editing] EmoEditor presents the first systematic **emotion-driven image generation** framework…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Affective Image Editing"
  - "Diffusion Models"
  - "Sentiment Analysis"
  - "Dual-Branch Architecture"
  - "Iterative Emotion Inference"
date: 2026-05-08
content_hash: 55e5c41b32552858
---

# Make Me Happier: Evoking Emotions Through Image Diffusion Models

**Conference**: ICCV 2025

**arXiv**: [2403.08255](https://arxiv.org/abs/2403.08255)

**Area**: Image Generation / Emotion Editing

**Keywords**: Affective Image Editing, Diffusion Models, Sentiment Analysis, Dual-Branch Architecture, Iterative Emotion Inference

## TL;DR

EmoEditor presents the first systematic **emotion-driven image generation** framework, employing a dual-branch diffusion model (global emotion conditioning + local semantic features) to generate target-emotion images from only a source image and a target emotion label — without manual text instructions or reference images. The work also introduces the EmoPair dataset containing 340K emotion-annotated image pairs.

## Background & Motivation

- **Gap in Emotion Editing**: Despite rapid advances in image generation and editing (InstructPix2Pix, SDEdit, ControlNet), **emotion-conditioned image editing** remains largely unexplored, despite its relevance to psychotherapy (autism/schizophrenia), commercial advertising, and artistic design.
- **Limitations of Global Methods**: Traditional approaches such as color transfer and style transfer perform only global attribute adjustment (brightness, hue) and cannot identify or edit the **local regions** responsible for specific emotions. For instance, simply brightening a volcanic hillside fails to convey "awe" — one may need to remove flames and add colorful clouds instead.
- **Locality of Emotion**: Emotions conveyed in images are jointly determined by global factors (overall color tone) and local factors (facial expressions, specific objects such as graveyards evoking sadness or balloons evoking joy). Humans are more sensitive to negative stimuli than positive ones, making simple global adjustments insufficient for altering strong negative affect.
- **Absence of Datasets**: No dataset exists with source–target emotion annotations and editing instructions over image pairs.

## Method

### Overall Architecture

EmoEditor consists of three components: EmoPair dataset construction, a dual-branch latent diffusion model, and Iterative Emotion Inference (IEI). The core pipeline is: source image + target emotion → emotion editing direction computation → global emotion conditioning + local source feature interaction → reverse diffusion generation → emotion predictor iterative verification.

### Key Designs

**1. EmoPair Dataset Construction**

The Mikels 8-category emotion model is adopted: positive (amusement, awe, contentment, excitement) and negative (anger, disgust, fear, sadness).

- **EPAS (annotated subset, 331,595 pairs)**: Image pairs from InstructPix2Pix are labeled with source/target emotion categories using a pretrained emotion predictor trained on EmoSet.
- **EPGS (generated subset, 6,949 pairs)**: Emotionally salient images from EmoSet are selected → GPT-3 generates 50 generic editing instructions per emotion category → human annotators rank and retain the top 10 → Ip2p edits EmoSet source images accordingly → quality filtering is applied.

**2. Dual-Branch Diffusion Model Architecture**

- **Global Branch (Emotion Conditioning)**:
    - The target emotion is encoded as a one-hot vector.
    - A pretrained emotion predictor estimates the source image's emotion probability distribution (discrete labels are not used, as real images may carry multiple emotions simultaneously).
    - The emotion editing direction is computed as: target one-hot − source emotion distribution.
    - A trainable emotion encoder maps this direction into the latent space.

- **Local Branch (Source Image Conditioning)**:
    - A pretrained VAE encoder extracts source image features $z$.
    - Forward diffusion adds noise to $z$, producing $z_t$.
    - During the reverse process, both emotion and source image conditions are injected into the denoising network via cross-attention.

**3. Alignment Loss (Neuro-Symbolic Alignment)**

A key contribution: emotion embeddings are aligned with CLIP embeddings of text editing instructions from EmoPair, enabling the model to implicitly learn the mapping from emotions to concrete editing operations. The loss is defined as the complement of cosine similarity. This approach does not enforce exact text prediction; rather, it performs soft alignment in a neuro-symbolic space, preserving the model's generative diversity.

**4. Iterative Emotion Inference (IEI)**

An emotion predictor is introduced as a critic during inference:

- After generation, two conditions are evaluated: (1) SSIM falls between 0.5 and 0.8; (2) the predicted target emotion confidence exceeds 0.6.
- If unmet: the generated image is used as the new input → emotion state is re-estimated → editing direction is updated → generation repeats.
- Up to 30 iterations are performed; if no result meets the criteria, the output with the highest confidence is selected.

### Loss & Training

Total loss = latent diffusion noise prediction loss + $0.5 \times$ emotion–text alignment loss.

## Key Experimental Results

### Main Results (Cross-Valence Emotion Editing)

| Method | ESMI (CAM) | ESMI (HA) |
|------|-----------|----------|
| Color Transfer | 28.33 | 26.68 |
| Neural Style Transfer | 47.48 | 46.00 |
| CLIP-Styler | 34.33 | 32.77 |
| ControlNet | 32.94 | 31.53 |
| SDEdit | 35.91 | 34.27 |
| InstructPix2Pix | 27.37 | 25.72 |
| AIF | 29.71 | 28.31 |
| EmoEdit | 32.94 | 37.53 |
| LMS (large model cascade) | 35.40 | 33.76 |
| **EmoEditor (Ours)** | **51.56** | **49.99** |

- Human psychophysical experiment (136 participants × 180 trials = 24,480 trials): EmoEditor exceeded 50% preference in pairwise comparisons against all baselines.

### Ablation Study

| Variant | Input | IEI | $\mathcal{L}_{emb}$ | ESMI |
|------|------|-----|-------|-------|
| Text condition only | text | ✗ | ✗ | 5.85 |
| Text + IEI | text | ✓ | ✗ | 50.24 |
| One-hot + $\mathcal{L}_{emb}$ | $e_{oh}$ | ✗ | ✓ | 7.36 |
| Edit direction + $\mathcal{L}_{emb}$ | $e_{dir}$ | ✗ | ✓ | 8.51 |
| Edit direction + IEI | $e_{dir}$ | ✓ | ✗ | 47.36 |
| Edit direction + IEI + $\mathcal{L}_{emb}$ | $e_{dir}$ | ✓ | ✓ | **51.56** |

### Key Findings

- IEI is the **largest single contributor** to performance improvement (8.51 → 47.36).
- The emotion editing direction outperforms a fixed one-hot encoding by accounting for the source image's emotional state.
- The alignment loss provides additional gains (47.36 → 51.56).
- CAM-based and HA-based ESMI scores are highly consistent, validating the reliability of the evaluation metric.
- LMS (a cascade of GPT-4o, GPT-o4, and Ip2p) still fails to surpass end-to-end trained EmoEditor.

## Highlights & Insights

1. **Novelty of Problem Formulation**: The first complete formalization of "emotion-driven image editing" — given a source image and a target emotion, generate an image that preserves scene structure while evoking the target emotion.
2. **Elegant Design of Emotion Editing Direction**: Editing direction = target − source, enabling the model to adaptively scale its edits — making minor changes when the source already partially reflects the target emotion, and larger modifications when emotions are diametrically opposed.
3. **Neuro-Symbolic Alignment**: Rather than enforcing exact text prediction, soft alignment in the embedding space both constrains the model to learn emotion-to-edit correspondences and preserves generative diversity.
4. **Practicality of IEI**: The closed-loop design of iterative generation and automatic evaluation requires no human intervention at inference time.
5. **Rigorous Human Evaluation**: A large-scale psychophysical experiment with 136 participants and MTurk quality control meets high standards in both HCI and psychology.
6. **Interesting Finding**: The same emotion trigger (flames evoking anger) calls for different optimal edits depending on context (indoors: replaced with cute lamps; outdoors: replaced with a tranquil meadow), demonstrating that the model has learned context-aware emotion editing.

## Limitations & Future Work

- Input images are fixed at $224\times224$ resolution, limiting practical applicability.
- IEI performs up to 30 iterations, incurring non-trivial inference cost.
- The quality of IEI and editing direction computation depends directly on the accuracy of the emotion predictor.
- The 8-category Mikels emotion model is coarse-grained and cannot capture more nuanced affective states.
- Emotion annotations in the EPAS subset of EmoPair (derived from Ip2p) are automatic rather than human-labeled.
- Cross-cultural differences in emotion perception are not considered.

## Related Work & Insights

- **InstructPix2Pix** [Brooks et al., 2023]: Text-instruction-driven image editing; serves as the technical baseline for EmoEditor.
- **EmoEdit** [Yang et al., 2024]: A pioneer in emotion editing, but reliance on a fixed query dictionary limits editing diversity.
- **AIF** [Li et al., 2023]: Reflects textual emotion onto images, but achieves only global filter-level effects.
- **EmoSet** [Yang et al., 2023]: The largest visual sentiment analysis dataset (120K images with emotion labels).
- **SDEdit** [Meng et al., 2022]: SDE-based image editing baseline.

## Rating

| Dimension | Score |
|------|------|
| Novelty | 5/5 |
| Effectiveness | 4/5 |
| Practicality | 4/5 |
| Clarity | 4/5 |
| Overall | 4/5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DiffDoctor: Diagnosing Image Diffusion Models Before Treating](diffdoctor_diagnosing_image_diffusion_models_before_treating.md)
- [\[NeurIPS 2025\] Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models](../../NeurIPS2025/image_generation/shallow_diffuse_robust_and_invisible_watermarking_through_low-dimensional_subspa.md)
- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)
- [\[NeurIPS 2025\] Enhancing Diffusion Model Guidance through Calibration and Regularization](../../NeurIPS2025/image_generation/enhancing_diffusion_model_guidance_through_calibration_and_regularization.md)
- [\[ICCV 2025\] Golden Noise for Diffusion Models: A Learning Framework](golden_noise_for_diffusion_models_a_learning_framework.md)

</div>

<!-- RELATED:END -->
