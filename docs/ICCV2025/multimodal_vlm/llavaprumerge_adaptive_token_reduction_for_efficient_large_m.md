---
title: >-
  [Paper Note] LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models
description: >-
  [ICCV 2025][Multimodal VLM][token reduction] By exploiting the sparsity of attention scores between the [CLS] token and visual tokens in CLIP-ViT, PruMerge adaptively selects important visual tokens via IQR-based outlier detection, then merges pruned tokens back into retained tokens through k-nearest-neighbor clustering, achieving up to 14× visual token compression with negligible performance degradation.
tags:
  - ICCV 2025
  - Multimodal VLM
  - token reduction
  - visual token pruning
  - token merging
  - LMM efficiency
  - CLS attention sparsity
date: 2026-05-08
content_hash: 2ac057c7b56736d5
---

# LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models

**Conference**: ICCV 2025
**arXiv**: [2403.15388](https://arxiv.org/abs/2403.15388)
**Code**: [https://llava-prumerge.github.io/](https://llava-prumerge.github.io/)
**Area**: Multimodal VLM / Model Acceleration
**Keywords**: token reduction, visual token pruning, token merging, LMM efficiency, CLS attention sparsity

## TL;DR
By exploiting the sparsity of attention scores between the [CLS] token and visual tokens in CLIP-ViT, PruMerge adaptively selects important visual tokens via IQR-based outlier detection, then merges pruned tokens back into retained tokens through k-nearest-neighbor clustering, achieving up to 14× visual token compression with negligible performance degradation.

## Background & Motivation

### Root Cause

**State of the Field**: Large multimodal models (LMMs) such as LLaVA-1.5 feed hundreds of visual tokens (e.g., 576) as a prefix into the LLM, whose attention complexity scales quadratically with input length. With the adoption of high-resolution images and video, the number of visual tokens grows further (Video-LLaVA uses 2,048 tokens). Existing acceleration methods primarily compress the LLM itself (via smaller LLMs or quantization), yet overlook a critical fact: a large fraction of visual tokens is redundant.

### Limitations of Prior Work

**Paper Goals**: Can the number of visual tokens passed from the visual encoder to the LLM be substantially reduced without sacrificing LMM reasoning capability? The central challenge lies in determining which tokens are important and how to preserve useful information from pruned tokens.

## Method

### Overall Architecture
PruMerge is a plug-and-play module inserted after the visual encoder output and before the MLP projection layer. It operates in three steps: (1) adaptively selecting important tokens via IQR-based outlier detection on CLS attention scores; (2) clustering pruned tokens to their nearest retained tokens via k-nearest neighbors; (3) merging each cluster by weighted averaging according to attention scores. PruMerge+ additionally supplements the selected tokens with uniformly sampled spatial tokens.

### Key Designs
1. **Adaptive Important Token Selection (AITS) via CLS Attention Sparsity**: The attention scores between the [CLS] token and visual tokens in the penultimate layer of CLIP-ViT exhibit a highly sparse distribution — the vast majority of tokens receive near-zero attention, while only a small subset attains significantly elevated scores. An IQR (interquartile range) outlier detection scheme is applied: tokens exceeding $Q3 + 1.5 \times \text{IQR}$ are retained as important. This renders token selection adaptive — complex images (e.g., those containing dense text) retain more tokens, while simpler images retain fewer.

2. **Key-Similarity-Based Token Merging (TS)**: Pruned tokens, though insufficiently "important," may carry complementary information (e.g., large background regions). Token similarity is computed using the Key vectors from the last ViT layer: $\text{Sim}(y_i, y_j) = k_i \cdot k_j^T$. For each retained token, the $k$ nearest pruned tokens are identified and merged via CLS-attention-weighted averaging. This approach achieves $O(n)$ complexity, substantially more efficient than graph-matching methods such as CrossGet's $O(n^2)$.

3. **Spatial Sampling Supplement in PruMerge+**: Purely attention-based selection may miss information from certain spatial regions. PruMerge+ augments the IQR-selected tokens with uniformly sampled spatial tokens to ensure full-image coverage. The compression ratio decreases from 14× to 4×, but performance more closely matches the original model.

### Loss & Training
PruMerge itself requires no training and can be applied in a completely training-free manner. However, the authors find that LoRA fine-tuning for one epoch further improves performance by allowing the LLM to adapt to the new token structure. On Video-LLaVA, the training-free variant alone yields performance gains.

## Key Experimental Results

| Method | LLM | # Tokens | VQAv2 | SQAI | TextVQA | POPE | MME | MMB |
|--------|-----|----------|-------|------|---------|------|-----|-----|
| LLaVA-1.5 | 7B | 576 | 78.5 | 66.8 | 58.2 | 85.9 | 1510.7 | 64.3 |
| +PruMerge | 7B | ~32 | 72.0 | **68.5** | 56.0 | 76.3 | 1350.3 | 60.9 |
| +PruMerge+ | 7B | ~144 | 76.8 | **68.3** | 57.1 | 84.0 | 1462.4 | **64.9** |

- PruMerge retains most performance using on average 32 tokens (5.5%); on ScienceQA it even surpasses the original model.
- PruMerge+ closely matches the original model using 25% of tokens, with a slight improvement on MMB.
- FLOPs are reduced by approximately 10× (9.3 TB → 0.91 TB); prefill latency drops from 88.6 ms to 15.3 ms.
- On Video-LLaVA, the training-free variant directly improves ActivityNet-QA by 3 points.
- PruMerge+ substantially outperforms single-modal methods such as ToMe/ATS/EViT (POPE: 84.0 vs. 51.0/57.4/60.1).

### Ablation Study
- The AITS selection module contributes the most, representing the core design; the TS merging module provides an additional ~2-point improvement.
- PruMerge demonstrates strong adaptivity: average token counts are ~40 for TextVQA/MME, ~35 for POPE, and only ~16 for SQA.
- The IQR method outperforms all alternative token sampling strategies (sequential and uniform spatial sampling).
- LoRA fine-tuning improves over the training-free variant by 2–3 points.

## Highlights & Insights
- **Elegant adaptive mechanism**: Applying IQR outlier detection enables token count to automatically adjust with image complexity — a clever use of classical statistics within deep learning.
- **Insightful observation**: CLS attention sparsity implies that most visual tokens have weak associations with global semantics and can be safely pruned.
- **Highly practical**: The module is plug-and-play, training-free, orthogonal to quantization, and applicable to both image and video LMMs.
- **Prune-then-Merge paradigm**: Important tokens are retained via pruning, while pruned tokens are recovered via merging — ensuring no information is wasted.

## Limitations & Future Work
- PruMerge (14× compression) shows noticeable performance drops on tasks requiring fine-grained understanding (VQAv2: −6.5%, POPE: −9.6%).
- The IQR method assumes a sparse CLS attention distribution; its applicability to non-CLIP encoders has not been validated.
- Token importance is determined solely from the visual side, without considering the relationship between text queries and visual tokens — different questions may require attention to different visual regions.
- Evaluation is limited to LLaVA-1.5; newer models (e.g., LLaVA-NeXT, Qwen-VL) have not been tested.

## Related Work & Insights
- **vs. Feather the Throttle**: Feather prunes within the LLM (FastV strategy), while PruMerge prunes at the encoder output. Feather identifies RoPE positional bias; PruMerge exploits CLS sparsity. The two approaches are motivated differently yet are complementary.
- **vs. FastV**: FastV prunes in the shallow layers of the LLM, whereas PruMerge prunes before the LLM — a more aggressive front-loaded reduction. FastV uses attention ranking; PruMerge uses adaptive IQR thresholding.
- **vs. ToMe/EViT/ATS**: These single-modal ViT acceleration methods perform poorly in LMM settings because their final output is the [CLS] token rather than all tokens.

## Related Work & Insights
- The discovery of CLS attention sparsity complements the RoPE positional bias finding in Feather the Throttle, motivating deeper investigation into visual token attention patterns.
- The adaptive token count paradigm can be extended to broader settings — allocating more tokens to hard samples and fewer to easy ones.
- In conjunction with research on scaling language-free visual representations, the token compression properties of SSL encoders merit further exploration.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The reasoning chain from CLS sparsity → IQR outlier detection → adaptive selection is coherent and insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Six benchmarks, video extension, multi-method comparison, and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear; visualizations (Figure 1/3) are intuitive and persuasive.
- **Value**: ⭐⭐⭐⭐ High practical value as a plug-and-play solution, though more competitive methods have emerged by ICCV 2025.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models](meteor_multi-encoder_collaborative_token_pruning_for_efficient_vision_language_m.md)
- [\[ICCV 2025\] Growing a Twig to Accelerate Large Vision-Language Models](growing_a_twig_to_accelerate_large_vision-language_models.md)
- [\[ICCV 2025\] FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance](folder_accelerating_multi-modal_large_language_models_with_enhanced_performance.md)
- [\[ICCV 2025\] Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](feather_the_throttle_revisiting_visual_token_pruning_for_vision-language_model_a.md)
- [\[ICCV 2025\] ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers](shortv_efficient_multimodal_large_language_models_by_freezing_visual_tokens_in_i.md)

<!-- RELATED:END -->
