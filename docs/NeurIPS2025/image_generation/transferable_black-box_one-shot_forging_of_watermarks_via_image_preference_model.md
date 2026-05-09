---
title: >-
  [Paper Note] Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models
description: >-
  [NeurIPS 2025][Image Generation][watermark forging] This paper proposes a black-box watermark forging method based on image preference models. Given only a single watermarked image, the method extracts the watermark via backpropagation and transfers it to arbitrary new images, effectively forging multiple post-hoc watermarking schemes without access to the underlying watermarking algorithm.
tags:
  - NeurIPS 2025
  - Image Generation
  - watermark forging
  - preference model
  - one-shot attack
  - black-box attack
  - post-hoc watermarking
date: 2026-05-08
content_hash: 1006cf57781b40df
---

# Transferable Black-Box One-Shot Forging of Watermarks via Image Preference Models

**Conference**: NeurIPS 2025  
**arXiv**: [2510.20468](https://arxiv.org/abs/2510.20468)  
**Code**: [https://github.com/facebookresearch/videoseal/tree/main/wmforger](https://github.com/facebookresearch/videoseal/tree/main/wmforger)  
**Area**: Image Generation  
**Keywords**: watermark forging, preference model, one-shot attack, black-box attack, post-hoc watermarking

## TL;DR

This paper proposes a black-box watermark forging method based on image preference models. Given only a single watermarked image, the method extracts the watermark via backpropagation and transfers it to arbitrary new images, effectively forging multiple post-hoc watermarking schemes without access to the underlying watermarking algorithm.

## Background & Motivation

- **Background**: Digital watermarking is a critical technology for content authenticity and provenance, especially in the generative AI era. Post-hoc watermarking has been widely adopted due to its modularity and ease of deployment (e.g., Google DeepMind's SynthID, Meta's Video Seal). The EU AI Act and U.S. AI Executive Order explicitly mandate watermarking of AI-generated content.
- **Limitations of Prior Work**: Existing watermark security research focuses primarily on **watermark removal**, with little attention to **watermark forging**—where an attacker steals a legitimate watermark and embeds it into malicious content. Forging poses a distinct threat: it can make fabricated content appear to originate from a legitimate source, or flood detection systems with false positives. Prior forging methods face severe practical constraints: Wang et al. (2021) requires paired original/watermarked images; Warfare (Li et al., 2023) and Dong et al. (2025) require thousands of watermarked images carrying the same hidden message—conditions that are nearly impossible to satisfy in real black-box scenarios.
- **Goal**: To investigate whether watermark forging is feasible given only a single watermarked image and no knowledge of the watermarking algorithm.

## Method

### Overall Architecture

A two-stage pipeline: (1) train an image preference model $R$ to distinguish "artifact-containing" images from "clean" images; (2) use the preference model as a surrogate loss, and extract the watermark from a watermarked image via gradient ascent, then paste it onto new images.

### Key Designs

1. **Preference Model Training**:

    - Architecture: ConvNeXt V2-Tiny, taking RGB images as input and outputting a scalar score $R(\mathbf{x}) \in \mathbb{R}$.
    - Loss function: Bradley-Terry ranking loss $-\mathbb{E}[\log \sigma(R(\mathbf{x}^+) - R(\mathbf{x}^-))]$, where $\mathbf{x}^+$ denotes the original (preferred) image and $\mathbf{x}^-$ denotes the image with synthetic artifacts (dispreferred).
    - **Core Idea — Synthetic Artifact Training Data**: No real watermarks are used at any point. Three types of artifacts are randomly generated in the Fourier domain: (a) wave artifacts — nonzero amplitude concentrated at a few random polar-coordinate points; (b) noise — amplitude sampled randomly with Gaussian decay; (c) line artifacts — nonzero amplitude along random horizontal/vertical lines. Artifacts are scaled to the range $[-0.05, 0.05]$, randomly applied in RGB or grayscale, and multiplied by a JND map with 50% probability.
    - **Adversarial Training**: Every other batch, negative samples are replaced with adversarially perturbed examples $\tilde{\mathbf{x}}^- = \mathbf{x}^- + \epsilon \cdot \nabla R(\mathbf{x}^-)$, ensuring the model produces semantically interpretable gradients. Without adversarial training, backpropagation yields checkerboard artifacts.

2. **Watermark Extraction and Forging**:

    - Given a watermarked image $\mathbf{x}_w$, the watermark is estimated by maximizing the preference score: $\hat{w} = \arg\max_\delta R(\mathbf{x}_w - \delta)$.
    - Optimization uses SGD with a fixed learning rate of 0.05 for 50–500 steps of gradient ascent.
    - The extracted watermark $\hat{w} = \mathbf{x}_w - \hat{\mathbf{x}}$ can be directly added to any new image: $\mathbf{y}_{\hat{w}} = \mathbf{y} + \hat{w}$.
    - For images of different resolutions, the watermarked image is first downscaled to the smaller resolution for extraction, then the watermark is upsampled to the target resolution.

3. **Watermark Removal**: The same procedure yields a dewatermarked image via $\hat{\mathbf{x}} = \mathbf{x}_w - \text{resize}(\hat{w}')$.

### Loss & Training

- Ranking loss (Equation 2); binary cross-entropy and hinge losses are not used (ablations confirm ranking loss is optimal).
- Training data: SA-1b dataset, images resized to 768×768 followed by random 256×256 crops.
- Trained from scratch for 120k steps on 8 GPUs with batch size 16 per GPU; AdamW optimizer with learning rate $1 \times 10^{-5}$.
- Total training time: approximately 60 hours on V100 GPUs.

## Key Experimental Results

### Main Results (Watermark Forging)

| Method | CIN Bit acc.↑ | MBRS Bit acc.↑ | TrustMark Bit acc.↑ | Video Seal Bit acc.↑ | PSNR↑ |
|------|-------------|---------------|-------------------|-------------------|-------|
| Gray image blending* | 1.00 | 0.80 | 0.54 | 0.83 | 52.9 |
| Warfare (n=1000) | 0.93 | 0.50 | 0.53 | 0.74 | 39.6 |
| DiffPure (FLUX) | 1.00 | 0.83 | 0.59 | 0.75 | 26.6 |
| Image averaging (n=100) | 1.00 | 0.91 | 0.61 | 0.59 | 26.2 |
| **Ours (n=1)** | **1.00** | **0.83** | **0.61** | **0.83** | **31.3** |

*Gray image blending requires access to the watermarking API and is therefore not applicable in realistic attack scenarios.

### Ablation Study

| Configuration | CIN | MBRS | TrustMark | Video Seal | PSNR |
|------|-----|------|-----------|-----------|------|
| Binary cross-entropy loss | 0.60 | 0.53 | 0.52 | 0.47 | 39.9 |
| Hinge loss | 0.62 | 0.55 | 0.52 | 0.47 | 44.1 |
| No adversarial perturbation | 0.97 | 0.65 | 0.52 | 0.49 | 34.7 |
| Trained with real watermarks | 1.00 | 0.67 | 0.58 | 0.77 | 36.9 |
| **Full method** | **1.00** | **0.83** | **0.61** | **0.83** | **31.3** |

### Key Findings

- **Only a single watermarked image is required**, yet the proposed method matches or exceeds approaches that need 100–1000 images.
- For **content-aware** watermarking methods such as Video Seal, image averaging fails entirely (0.59), whereas the proposed method still achieves 0.83.
- For content-agnostic methods such as CIN and MBRS, simple image averaging is already effective, since their watermark patterns are largely fixed.
- TrustMark is the most difficult to forge, as both its encoder and decoder are highly conditioned on the input image content.
- Ranking loss substantially outperforms classification losses, because positive and negative samples are extremely similar and no global classification boundary exists.
- **Counterintuitive finding**: The model trained on programmatically synthesized artifacts outperforms the model trained on real watermarks (row 4 vs. row 5), due to insufficient diversity in real watermarks causing overfitting.
- Adversarial training is essential for producing interpretable gradients; without it, gradient directions are entirely ineffective.

## Highlights & Insights

- **Minimal-resource threat model**: The black-box, one-shot setting represents the most realistic attack scenario to date.
- **Creative application of preference models**: Drawing on the RLHF paradigm from LLMs, a ranking-loss-trained image quality discriminator is used as a surrogate for watermark detection.
- **Synthetic artifacts as a substitute for real watermarks**: The training pipeline requires no watermarking model whatsoever, and the resulting model outperforms one trained on real watermarks—a counterintuitive finding with broad implications.
- **Adversarial training for gradient interpretability**: The connection to the adversarial robustness literature (Santurkar et al., 2019) is elegant and theoretically grounded.
- The method supports both watermark removal and forging simultaneously, achieving competitive performance on the removal task as well.

## Limitations & Future Work

- The method targets only post-hoc watermarking; it cannot forge semantic watermarks (e.g., Tree-Ring, RingID) that embed information via structural changes to the generated content.
- Excessive optimization steps may blur high-frequency texture regions (e.g., water surfaces, foliage).
- The quality of the extracted watermark from a single image is inherently constrained by the characteristics of that specific image.
- Defensive recommendation: decoders should be designed to be genuinely content-aware—for example, trained to reject watermarks transferred from a different source image.

## Related Work & Insights

- Warfare and image averaging represent traditional forging methods that require large collections of watermarked images.
- DiffPure and CtrlRegen are diffusion-based removal methods, but tend to introduce hallucinated details.
- UnMarker operates in the frequency domain to remove watermarks.
- This paper serves as a warning to the watermark security community: even content-aware post-hoc watermarking schemes may be broken by low-resource attacks, necessitating a fundamental rethinking of decoder design.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The combination of preference models and synthetic artifacts is highly original)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 watermarking methods, multiple baselines, comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, well-defined threat model)
- Value: ⭐⭐⭐⭐⭐ (Reveals fundamental security vulnerabilities in current watermarking schemes)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] WMCopier: Forging Invisible Image Watermarks on Arbitrary Images](wmcopier_forging_invisible_image_watermarks_on_arbitrary_images.md)
- [\[NeurIPS 2025\] Amortized Sampling with Transferable Normalizing Flows](amortized_sampling_with_transferable_normalizing_flows.md)
- [\[CVPR 2026\] BlackMirror: Black-Box Backdoor Detection for Text-to-Image Models via Instruction-Response Deviation](../../CVPR2026/image_generation/blackmirror_black-box_backdoor_detection_for_text-to-image_models_via_instructio.md)
- [\[NeurIPS 2025\] Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation](distilled_decoding_2_onestep_sampling_of_image_autoregressiv.md)
- [\[NeurIPS 2025\] Semantic Surgery: Zero-Shot Concept Erasure in Diffusion Models](semantic_surgery_zero-shot_concept_erasure_in_diffusion_models.md)

<!-- RELATED:END -->
