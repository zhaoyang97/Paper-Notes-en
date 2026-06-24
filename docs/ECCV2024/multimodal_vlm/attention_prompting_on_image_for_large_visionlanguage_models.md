---
title: >-
  [Paper Note] Attention Prompting on Image for Large Vision-Language Models
description: >-
  [ECCV 2024][Multimodal VLM][visual prompting] This paper proposes Attention Prompting on Image (API), which utilizes an auxiliary VLM (CLIP or LLaVA) to generate attention attribution maps based on text queries. These maps are overlaid as heatmaps onto the original image to guide the LVLM to focus on relevant regions. API improves LLaVA-1.5 by up to 3.8% on MM-Vet and is widely effective across various LVLMs, including GPT-4V.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "visual prompting"
  - "attention heatmap"
  - "LVLM"
  - "self-reflection"
  - "model ensemble"
date: 2026-05-08
content_hash: 04624f80ddd0d35a
---

# Attention Prompting on Image for Large Vision-Language Models

**Conference**: ECCV 2024  
**arXiv**: [2409.17143](https://arxiv.org/abs/2409.17143)  
**Code**: [GitHub](https://github.com/yu-rp/apiprompting)  
**Area**: Large Vision-Language Models / Visual Prompting  
**Keywords**: visual prompting, attention heatmap, LVLM, self-reflection, model ensemble

## TL;DR

This paper proposes Attention Prompting on Image (API), which utilizes an auxiliary VLM (CLIP or LLaVA) to generate attention attribution maps based on text queries. These maps are overlaid as heatmaps onto the original image to guide the LVLM to focus on relevant regions. API improves LLaVA-1.5 by up to 3.8% on MM-Vet and is widely effective across various LVLMs, including GPT-4V.

## Background & Motivation

**Background**: Visual prompting guides LVLMs to focus on specific regions by adding annotations such as circles, arrows, and masks to the image. Existing methods (e.g., FGVP, SoM) rely on segmentation models to generate annotations, which are training-free and intuitively effective.

**Limitations of Prior Work**: (1) Existing visual prompting techniques only process the image itself without considering the text query—regardless of the question asked, the visual prompting result remains the same for the same image; (2) This leads to a mismatch between the prompted regions and the areas that actually need attention for the given question; (3) Segmentation-based methods (FGVP/SoM) are essentially instance-level proposals, which are not suitable for general VQA tasks.

**Key Challenge**: How to make visual prompting dynamically vary with the text query, enabling the model to focus on different regions of the image based on different questions?

**Key Insight**: Utilizing the intrinsic vision-language alignment capability of the LVLM to generate query-aware attention attribution maps, which are then overlaid onto the image as visual prompts.

## Method

### Overall Architecture

Given an image $I$ and a text query $T^i$, API operates in two steps: (1) **Attribution map generation**: An auxiliary VLM $g$ (which can be CLIP or LLaVA itself) computes the importance scores of each image patch with respect to the text query, generating the attribution map $\Psi \in \mathbb{R}^{P \times P}$; (2) **Heatmap overlay**: The attribution map is upsampled to the pixel space, smoothed using mean filtering, and then blended with the original image as an alpha channel to produce the annotated image $I^a$, which is fed into the inference VLM $f$. If $g=f$, this corresponds to self-reflection; if $g \neq f$, it corresponds to a model ensemble.

### Key Designs

1. **Attribution Map Decomposition via CLIP's cls Token**

    - Exploits the residual connections in ViT to decompose CLIP's image-level similarity into the contributions of individual patches.
    - Computes the similarity between the deep MSA output of each patch and the text embedding $\hat{T}$ to obtain $\Psi^{cls}$, directly localizing patches related to the query entities.
    - Complementary attribution map $\Psi^{comp}$: Negates the similarity between non-cls tokens in the final layer and $\hat{T}$—low-information "register" tokens have high similarity, while patches with actual content have low similarity.
    - Final attribution map $\Psi = \Psi^{cls} + \Psi^{comp} - \Psi^{cls} \cdot \Psi^{comp}$ (a soft OR operation), balancing explicit entity localization and implicit relevant region preservation.

2. **Attribution Map based on LLaVA's Attention Weights**

    - Direct extraction of the cross-attention weights from LLaVA's deep decoder (attention values of output tokens on image tokens).
    - Averages across all generated tokens and all attention heads to obtain the average attention level of each image patch.
    - Simpler than the CLIP scheme, but requires an initial inference run to generate the output sequence.

### Loss & Training

API is a **training-free** inference-time technique that does not involve any loss function or training process. Core hyperparameters include the starting layer $L'$ used for the attribution map and the mean filter kernel size $k$.

## Key Experimental Results

### Main Results

| Model | Prompting Method | MM-Vet | LLaVA-Bench | MMMU |
|------|---------|--------|------------|------|
| LLaVA-1.5 | No Prompt | 32.8 | 71.9 | 35.2 |
| LLaVA-1.5 | FGVP (Mask) | 31.0 (-1.8) | 57.4 (-14.5) | 36.1 (+1.0) |
| LLaVA-1.5 | SoM | 26.4 (-6.4) | 56.1 (-15.8) | 35.6 (+0.4) |
| LLaVA-1.5 | **API (CLIP)** | **35.3 (+2.5)** | **74.1 (+2.2)** | **37.5 (+2.4)** |
| LLaVA-1.5 | **API (LLaVA)** | **36.6 (+3.8)** | **74.8 (+2.9)** | **37.0 (+1.8)** |
| GPT-4V | No Prompt | 67.0 | 102.0 | 50.6 |
| GPT-4V | **API (CLIP)** | **67.7 (+0.7)** | **103.3 (+1.3)** | **51.0 (+0.4)** |

### Ablation Study

| Ablation Item | MM-Vet | Description |
|--------|--------|------|
| Only $\Psi^{cls}$ | 34.1 | Lacks implicit relevant patches |
| Only $\Psi^{comp}$ | 33.8 | Lacks explicit entity localization |
| $\Psi^{cls} + \Psi^{comp}$ (Soft OR) | 35.3 | Complementary combination is optimal |
| No Mean Filtering | 33.5 | Rectangular mask mismatches object shape |
| Different Starting Layers $L'$ | $L'=L-2$ is optimal | Shallow layer information is less discriminative |

### Key Findings

- API performs significantly better than FGVP and SoM on LLaVA-1.5—the key lies in being query-aware (the latter two do not consider the question).
- FGVP and SoM actually degrade performance in most model-dataset combinations—query-agnostic visual prompts can cause mismatches.
- API is equally effective on closed-source models like GPT-4V and Gemini (+0.7% to +11.6%), proving its generalizability.
- The performance is best when the auxiliary model $g$ is the same as the inference model $f$ (self-reflection) (API-LLaVA on LLaVA: +3.8%).

## Highlights & Insights

- Upgrading visual prompting from "query-agnostic" to "query-aware" is a crucial paradigm shift—for the same image, different questions should highlight different regions.
- The cls token decomposition is an ingenious attribution method—utilizing the additivity of residual connections to decompose global similarity to the patch level.
- Discoveries show that the high similarity of non-cls tokens actually serves a "register" function—consistent with recent studies on register tokens.

## Limitations & Future Work

- Requires an extra forward pass of the auxiliary model, doubling the inference cost.
- Heatmap overlay relies on pixel-level multiplication, which might lose information in darkened regions—this can be detrimental to tasks requiring global understanding.
- CLIP's attribution maps may have limited effectiveness for non-entity queries (e.g., "What is the style of this painting?").
- The effectiveness has not been tested in complex scenarios such as Video VQA or multi-image comparison.

## Related Work & Insights

- **vs FGVP/SoM**: FGVP/SoM use segmentation models to generate fixed annotations independent of the text query, whereas API dynamically generates attribution heatmaps based on the query.
- **vs Self-Reflection**: Traditional self-reflection iterates in the text space (repeatedly answering and revising), while API performs self-reflection in the pixel space—which is more direct.
- **vs GradCAM**: GradCAM requires gradient backpropagation, whereas API only requires a forward pass; the key innovation is using cls token decomposition to replace gradient-based methods.
- **Insights**: The attention weights of LVLMs are valuable "free" signals—they can be utilized in more scenarios such as attention distillation, token pruning, and hallucination detection.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Query-aware visual prompting is an important paradigm upgrade; the cls decomposition method is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 5 LVLMs × 6 benchmarks + detailed ablation + closed-source model validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear methodological motivation, comprehensive comparison between the two attribution map schemes.
- **Value**: ⭐⭐⭐⭐ Training-free, plug-and-play method with high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Vary: Scaling up the Vision Vocabulary for Large Vision-Language Models](vary_scaling_up_the_vision_vocabulary_for_large_visionlanguag.md)
- [\[ECCV 2024\] AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting](adashield_safeguarding_multimodal_large_language_models_from_structure-based_att.md)
- [\[ICML 2026\] Large Vision-Language Models Get Lost in Attention](../../ICML2026/multimodal_vlm/large_vision-language_models_get_lost_in_attention.md)
- [\[ECCV 2024\] BLINK: Multimodal Large Language Models Can See but Not Perceive](blink_multimodal_large_language_models_can_see_but_not_perceive.md)
- [\[ECCV 2024\] AddressCLIP: Empowering Vision-Language Models for City-wide Image Address Localization](addressclip_empowering_vision-language_models_for_city-wide_image_address_locali.md)

</div>

<!-- RELATED:END -->
