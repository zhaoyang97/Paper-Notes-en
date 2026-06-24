---
title: >-
  [Paper Note] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception
description: >-
  [CVPR 2025][Image Generation][Image Aesthetic Enhancement] This paper proposes DIAE, which translates vague aesthetic instructions into multimodal control signals (HSV/contour maps + text) via a Multimodal Aesthetic Perception (MAP) module. It constructs an "imperfectly-paired" dataset, IIAEData, and uses a dual-branch supervision strategy to achieve weakly-supervised aesthetic enhancement, achieving SOTA performance on LAION and MLLM aesthetic scores.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image Aesthetic Enhancement"
  - "Diffusion Models"
  - "Multimodal Perception"
  - "Weakly Supervised Learning"
  - "ControlNet"
date: 2026-05-08
content_hash: 94b50b63d9e17136
---

# Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception

**Conference**: CVPR 2025  
**arXiv**: [2603.11556](https://arxiv.org/abs/2603.11556)  
**Code**: To be confirmed  
**Area**: Image Editing / Aesthetic Enhancement  
**Keywords**: Image Aesthetic Enhancement, Diffusion Models, Multimodal Perception, Weakly Supervised Learning, ControlNet

## TL;DR

This paper proposes DIAE, which translates vague aesthetic instructions into multimodal control signals (HSV/contour maps + text) via a Multimodal Aesthetic Perception (MAP) module. It constructs an "imperfectly-paired" dataset, IIAEData, and uses a dual-branch supervision strategy to achieve weakly-supervised aesthetic enhancement, achieving SOTA performance on LAION and MLLM aesthetic scores.

## Background & Motivation

**Background**: Image Aesthetic Enhancement (IAE) requires models to possess aesthetic perception and creative editing capabilities—identifying defects in color, composition, and lighting, and improving them.

**Limitations of Prior Work**: (1) Text encoders in diffusion models struggle to comprehend abstract aesthetic instructions; (2) There is a lack of "perfectly-paired" training data (characterized by identical content but different aesthetics).

**Key Challenge**: Aesthetics represent high-level human perception, which is difficult to convey to generative models purely through text.

**Goal**: To enable diffusion models to understand and execute aesthetic enhancement, addressing the lack of training data.

**Key Insight**: Utilizing visual modalities to assist text in conveying aesthetics, combined with weakly-supervised learning using "imperfectly-paired" images.

**Core Idea**: Aesthetic perception = text + visual representation; weak supervision = dual-branch training.

## Method

### Overall Architecture

Three parts: IIAEData dataset (LLaVA matching + UNIAA evaluation) -> MAP Multimodal Aesthetic Perception (HSV + contour + text control signals) -> Dual-branch supervision (semantic preservation for content + aesthetic learning for quality).

### Key Designs

1. **IIAEData Dataset**

    - **Function**: To construct a weakly-paired training set with identical semantic content but different aesthetics.
    - **Mechanism**: Filtering high/low MOS from AVA/TAD66K/KonIQ/FLICKR -> generating captions with LLaVA -> semantic matching -> UNIAA aesthetic evaluation.
    - Scale: 47.5K (45K for training + 1.5K for testing).

2. **Multimodal Aesthetic Perception (MAP)**

    - **Function**: To translate vague aesthetic text instructions into comprehensible multimodal control signals.
    - **Mechanism**: Utilizing HSV maps + text for color, and contour maps + text for structure. CNNs extract visual features, CLIP encodes text, and ControlNet injects these into the UNet.
    - **Design Motivation**: HSV maps align closely with human color perception, while contour maps emphasize spatial arrangement.

3. **Dual-Branch Supervision**

    - **Function**: To resolve the issue where fully-supervised learning cannot be directly applied to imperfectly-paired data.
    - **Mechanism**: The parameter t_s divides the denoising process into two phases. In the early phase, input image supervision is used to preserve semantics; throughout the entire process, reference image supervision is applied to learn aesthetics.
    - Total Loss: L = L_ref + lambda * L_inp, with t_s defaulting to 900.

### Loss & Training

Based on SD v1.5, AdamW lr=1e-5, 4xA800, 100K iterations.

## Key Experimental Results

### Main Results

| Method | LAION(256) | LAION(512) | MLLM(256) | MLLM(512) | CLIP-I(256) | CLIP-I(512) |
|------|----------|----------|---------|---------|-----------|-----------|
| Original | 4.962 | 5.123 | 3.243 | 3.300 | 1.000 | 1.000 |
| InstructPix2Pix | 4.991 | 5.396 | 3.264 | 3.325 | 0.764 | 0.690 |
| DOODL | 5.102 | 5.140 | 3.255 | 3.297 | 0.775 | 0.703 |
| **DIAE** | **5.324** | **6.012** | **3.339** | **3.662** | 0.772 | **0.784** |

### Ablation Study

| Configuration | LAION | MLLM | CLIP-I |
|------|-------|------|--------|
| DIAE (w/o visual) | 5.250 | 3.343 | 0.623 |
| DIAE (w/o text) | 5.428 | 3.410 | 0.792 |
| DIAE (Full) | **5.668** | **3.501** | 0.778 |

### Key Findings

- At 512 resolution, LAION increases by +17.4%, and MLLM increases by +11.0%.
- Content consistency drops significantly when the visual modality is removed.
- Performance improvement is more pronounced for images with low MOS.
- DIAE does not arbitrarily add or delete content.

## Highlights & Insights

- Aesthetics are decomposed into two dimensions, color and structure, represented through a multimodal approach combining vision and text.
- The "imperfectly-paired" + weakly-supervised paradigm cleverly bypasses high dataset construction costs.
- The dual-branch architecture, controlled by t_s, precisely balances semantic preservation versus aesthetic enhancement.
- It can be integrated with MLLMs to achieve end-to-end processing.

## Limitations & Future Work

- Portrait scenarios are not addressed.
- The approach is based on SD v1.5.
- The parameter t_s requires tuning.

## Related Work & Insights

- The "perfectly-paired" data construction concept of InstructPix2Pix and the "imperfectly-paired" approach of this paper complement each other, representing two extremes of supervision signals.
- Using aesthetic scores for classifier guidance, as in DOODL/RAHF, is another alternative path, but it does not modify model behaviors.
- The architectural concept of adding structure in ControlNet is innovatively extended by this work into multimodal aesthetic control (HSV + contours + text).
- The content-style decoupling framework of StyleDiffusion inspired the design of the dual-branch supervision in this work.
- Advancements in aesthetic MLLMs like Q-ALIGN enable the automatic generation of aesthetic evaluation text, serving as inputs for the MAP module in DIAE.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Creative problem formulation and multimodal aesthetic perception.
- **Experimental Thoroughness**: ⭐⭐⭐ Relatively few baseline methods compared.
- **Writing Quality**: ⭐⭐⭐ Overall clear and structured.
- **Value**: ⭐⭐⭐⭐ High practical demand for image aesthetic enhancement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)
- [\[CVPR 2025\] Not All Parameters Matter: Masking Diffusion Models for Enhancing Generation Ability](not_all_parameters_matter_masking_diffusion_models_for_enhancing_generation_abil.md)
- [\[CVPR 2025\] Enhancing Creative Generation on Stable Diffusion-based Models](enhancing_creative_generation_on_stable_diffusion-based_models.md)
- [\[CVPR 2025\] Enhancing Vision-Language Compositional Understanding with Multimodal Synthetic Data (SPARCL)](enhancing_vision-language_compositional_understanding_with_multimodal_synthetic_.md)
- [\[CVPR 2025\] Enhancing Facial Privacy Protection via Weakening Diffusion Purification](enhancing_facial_privacy_protection_via_weakening_diffusion_purification.md)

</div>

<!-- RELATED:END -->
