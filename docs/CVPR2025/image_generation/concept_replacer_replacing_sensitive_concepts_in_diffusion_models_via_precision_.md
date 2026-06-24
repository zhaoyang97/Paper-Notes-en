---
title: >-
  [Paper Note] Concept Replacer: Replacing Sensitive Concepts in Diffusion Models via Precision Localization
description: >-
  [CVPR 2025][Image Generation][Concept Erasure] Proposes Concept Replacer, which precisely identifies sensitive concept regions during the denoising process through a few-shot-trained concept localizer, and then replaces the localized region with safe content using training-free Dual-Prompt Cross-Attention (DPCA). This achieves precise local concept replacement instead of global image distortion.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Concept Erasure"
  - "Precise Localization"
  - "Few-Shot Segmentation"
  - "Dual-Prompt Cross-Attention"
  - "Content Safety"
date: 2026-05-08
content_hash: 9f71b90930328d1a
---

# Concept Replacer: Replacing Sensitive Concepts in Diffusion Models via Precision Localization

**Conference**: CVPR 2025  
**arXiv**: [2412.01244](https://arxiv.org/abs/2412.01244)  
**Code**: [https://github.com/zhang-lingyun/ConceptReplacer](https://github.com/zhang-lingyun/ConceptReplacer)  
**Area**: Diffusion Models / AI Safety  
**Keywords**: Concept Erasure, Precise Localization, Few-Shot Segmentation, Dual-Prompt Cross-Attention, Content Safety

## TL;DR
Proposes Concept Replacer, which precisely identifies sensitive concept regions during the denoising process through a few-shot-trained concept localizer, and then replaces the localized region with safe content using training-free Dual-Prompt Cross-Attention (DPCA). This achieves precise local concept replacement instead of global image distortion.

## Background & Motivation

### Background

**Background**: Diffusion models may generate unsafe content (nudity, violence, etc.). Existing concept erasure methods (such as SLD, ESD) suppress sensitive concepts by globally modifying the guidance direction or model weights.

### Limitations of Prior Work

**Limitations of Prior Work**: Global methods affect non-target areas—SLD degrades overall image quality, and ESD may interfere with normal generation after weight modification. These methods cannot precisely "replace only" the problematic regions while keeping other parts unchanged.

### Key Challenge

**Key Challenge**: The need to accurately locate the spatial positions of sensitive concepts during the denoising process without adding excessive inference overhead.

### Goal

**Goal**: Achieve spatially precise concept replacement—modifying only the regions containing sensitive concepts while keeping the rest of the generated image unchanged.

### Key Insight

**Key Insight**: Detect concept locations using a few-shot fine-tuned localizer (reusing the U-Net structure, only tuning the attention $W_k/W_v$), and use dual-prompt cross-attention to generate within the localized region using a replacement prompt.

### Core Idea

**Core Idea**: The few-shot concept localizer detects the sensitive region mask during the first 2-3 denoising steps, and the DPCA module performs cross-attention using the replacement prompt inside the mask and the original prompt outside the mask.

## Method

### Overall Architecture
Consists of two modules. **Concept Localizer**: Reusing the U-Net structure, it fine-tunes only $W_k/W_v$ (few-shot with 1-10 annotated images), and fuses self-attention and cross-attention scores to output a concept mask. It is only activated during the first 2-3 denoising steps. **DPCA Module**: Training-free; during each denoising step, it performs cross-attention using the replacement prompt (e.g., "clothes") inside the masked region, and the original prompt outside the mask, achieving localized replacement.

### Key Designs

1. **Few-Shot Concept Localizer**: Shares the U-Net encoder, tuning only $W_k/W_v$ (extremely low overhead), and fuses attention maps from self-attention (spatial coherence) and cross-attention (concept recognition) to generate masks. Achieves 78.1% mIoU on CelebA with 10-shot learning.

2. **Dual-Prompt Cross-Attention (DPCA)**: Employs $Q \cdot K_{replace}^T$ inside the mask and $Q \cdot K_{original}^T$ outside the mask; training-free. Ensures that non-target areas remain completely unaffected.

3. **Sparse Activation**: The localizer runs only during the first 2-3 steps, as the layout information from high-noise steps is sufficient to determine concept locations, after which the mask is kept static and reused.

## Key Experimental Results

### Main Results

| Method | CelebA mIoU (10-shot)↑ | Pascal-Car mIoU (10-shot)↑ |
|------|----------------------|--------------------------|
| SegDDPM | 78.0% | 62.5% |
| SLiMe | 75.7% | 68.7% |
| **Concept Replacer** | **78.1%** | **69.3%** |

Nudity Removal: Achieves the highest rate of unsafe content reduction on the I2P prompt dataset while maintaining the best consistency in non-target regions.

### Key Findings
- Localization accuracy is comparable to dedicated segmentation models (78.1% vs. SegDDPM 78.0%), using only 10 annotated images.
- Global methods (SLD, ESD) distort the entire image, whereas Ours modifies only the target region.
- Achieves 70.2% mIoU even with 1-shot, showing extremely low annotation requirements.
- Achieves the highest rate of unsafe content reduction in nudity removal experiments on the I2P prompt dataset, while maintaining the best consistency in non-target regions.
- The localizer runs only during the first 2-3 denoising steps, incurring extremely low computational overhead and not affecting overall inference speed.

## Highlights & Insights
- The paradigm of **precise localization -> local replacement** is more reasonable than global erasure—acting as a "scalpel" rather than a "sledgehammer".
- The training-free design of **DPCA** makes the method easy to deploy.
- Few-shot fine-tuning only modifies $W_k/W_v$, resulting in a minimal number of parameter changes.

## Limitations & Future Work
- The localizer needs to be retrained for each new concept.
- Applicable only to spatially localizable concepts; global style-level concepts (such as "violent style") cannot be handled.
- Fixed-threshold mask binarization may not be suitable for all concepts.
- When sensitive and non-sensitive concepts highly overlap spatially (such as a person holding a sensitive object), the difficulty of precise segmentation and replacement increases significantly.
- The replaced images may suffer from subtle unnaturalness in semantic coherence, especially regarding lighting and shadow matching.
- Robustness against adversarial prompts (prompt engineering designed to bypass concept detection) has not been fully evaluated.
- In scenarios with simultaneous multi-concept replacement, overlapping masks from different concepts may lead to conflicts.

## Rating
- Novelty: ⭐⭐⭐⭐ The decomposition paradigm of localization + replacement is innovative, and the DPCA design is simple.
- Experimental Thoroughness: ⭐⭐⭐⭐ Segmentation + safe generation + multi-concept verification.
- Writing Quality: ⭐⭐⭐⭐ The methodology is described clearly.
- Value: ⭐⭐⭐⭐ Directly valuable for AI safety content filtering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] When Are Concepts Erased From Diffusion Models?](../../NeurIPS2025/image_generation/when_are_concepts_erased_from_diffusion_models.md)
- [\[CVPR 2025\] Memories of Forgotten Concepts](memories_of_forgotten_concepts.md)
- [\[CVPR 2026\] Erasing Thousands of Concepts: Towards Scalable and Practical Concept Erasure for Text-to-Image Diffusion Models](../../CVPR2026/image_generation/erasing_thousands_of_concepts_towards_scalable_and_practical_concept_erasure_for.md)
- [\[NeurIPS 2025\] Emergence and Evolution of Interpretable Concepts in Diffusion Models](../../NeurIPS2025/image_generation/emergence_and_evolution_of_interpretable_concepts_in_diffusi.md)
- [\[CVPR 2025\] Efficient Fine-Tuning and Concept Suppression for Pruned Diffusion Models](efficient_fine-tuning_and_concept_suppression_for_pruned_diffusion_models.md)

</div>

<!-- RELATED:END -->
