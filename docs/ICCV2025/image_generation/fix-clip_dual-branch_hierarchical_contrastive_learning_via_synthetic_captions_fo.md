---
title: >-
  [Paper Note] Fix-CLIP: Dual-Branch Hierarchical Contrastive Learning via Synthetic Captions for Better Understanding of Long Text
description: >-
  [ICCV 2025][Image Generation][CLIP] Fix-CLIP enhances CLIP's long-text understanding capability through three key innovations: (1) a dual-branch training pipeline that aligns short texts with masked images and long texts with original images; (2) learnable Regional Prompts with unidirectional attention masks for local visual feature extraction; and (3) a hierarchical feature alignment module that aligns multi-scale features across intermediate encoder layers. After incremental training on 30M synthetic long-text data, Fix-CLIP substantially outperforms state-of-the-art methods on both long- and short-text retrieval benchmarks. Its text encoder can be directly plugged into diffusion models to improve long-text generation quality.
tags:
  - ICCV 2025
  - Image Generation
  - CLIP
  - long-text understanding
  - dual-branch training
  - regional prompts
  - hierarchical feature alignment
  - synthetic data
date: 2026-05-08
content_hash: 88d26233ce44f585
---

# Fix-CLIP: Dual-Branch Hierarchical Contrastive Learning via Synthetic Captions for Better Understanding of Long Text

**Conference**: ICCV 2025
**arXiv**: [2507.10095](https://arxiv.org/abs/2507.10095)
**Code**: [GitHub](https://github.com/bcwang-sjtu/Fix-CLIP)
**Area**: Image Generation
**Keywords**: CLIP, long-text understanding, dual-branch training, regional prompts, hierarchical feature alignment, synthetic data

## TL;DR
Fix-CLIP enhances CLIP's long-text understanding capability through three key innovations: (1) a dual-branch training pipeline that aligns short texts with masked images and long texts with original images; (2) learnable Regional Prompts with unidirectional attention masks for local visual feature extraction; and (3) a hierarchical feature alignment module that aligns multi-scale features across intermediate encoder layers. After incremental training on 30M synthetic long-text data, Fix-CLIP substantially outperforms state-of-the-art methods on both long- and short-text retrieval benchmarks. Its text encoder can be directly plugged into diffusion models to improve long-text generation quality.

## Background & Motivation

### Core Problem

CLIP excels at short-text tasks such as image classification and retrieval, but its performance degrades sharply on long-text inputs due to the 77-token input length constraint of its text encoder. In practice, a single image often requires dozens of sentences for a thorough description, which severely limits CLIP's applicability in MLLMs and text-to-image generation models.

### Limitations of Prior Work

**Long-text gains at the cost of short-text performance**: Long-CLIP fine-tunes CLIP on ShareGPT4V to improve long-text understanding, but degrades performance on short-text tasks.

**High training cost from scratch**: DreamLIP, LoTLIP, and FLAIR require training on large-scale synthetic datasets from scratch, consuming substantial computational resources.

**Coarse global alignment**: Standard contrastive learning aligns only the [CLS] token's global representation, lacking local alignment capability and thus performing poorly on fine-grained description tasks.

**High cost of explicit region matching**: Generating corresponding descriptions for a large number of image regions requires significant data scale and computational overhead.

**Implicit local consistency methods harm generalization**: Such methods compromise the generalization ability of pre-trained models, leading to degradation on short-text tasks.

### Paper Goals

Fix-CLIP adopts an **incremental training** strategy, fine-tuning a pre-trained CLIP model rather than training from scratch, while employing a carefully designed dual-branch pipeline that **preserves and even enhances short-text capability** alongside improved long-text understanding.

## Method

### Overall Architecture

Fix-CLIP comprises three core modules:
1. **Dual-branch training pipeline**: short texts aligned with masked images; long texts aligned with original images.
2. **Regional Prompts with unidirectional attention masks**: learnable regional prompts injected into the image encoder to extract local features.
3. **Hierarchical feature alignment**: multi-scale visual-language feature alignment at intermediate encoder layers.

### Key Design 1: Dual-Branch Training Pipeline

**Positional encoding extension**: Building on CLIP's original 77 positional encodings $PE$, the first 20 encodings are frozen (covering CLIP's effective text length), while positions 21–77 are expanded to 248 tokens via 4× linear interpolation:

$$PE_l = \text{Concat}(PE[:20], \text{Intpol}(PE[20:], 4))$$

where $\text{Intpol}(PE, q)[i] = (1-\lambda) \cdot PE[\lfloor \frac{i}{q} \rfloor] + \lambda \cdot PE[\lfloor \frac{i}{q} \rfloor + 1]$. Only the expanded positional encoding parameters are learnable.

**Differentiated encoding strategies**:
- **Short-text branch**: 75% of image patch embeddings are randomly masked (replaced with learnable zero-initialized parameters) and paired with short texts for contrastive learning. The intuition follows MAE — sufficient semantic information is retained after 75% masking.
- **Long-text branch**: Full, unmasked image patches are paired with long texts for contrastive learning. The rich detail captured in long texts requires complete image content for effective alignment.

### Key Design 2: Regional Prompts with Unidirectional Masks

**Design Motivation**: The [CLS] token aggregates global features through attention, but local region identification remains insufficient.

**Design**: $M$ learnable regional prompts $R_1^l, \ldots, R_M^l$ are inserted into the $l$-th Transformer layer of the image encoder, with each layer receiving fresh learnable parameters (eliminating cross-depth information interference).

**Unidirectional attention mask**:

$$\text{Attn} = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} \odot \mathbf{Mask}\right) V$$

Mask design rules (core innovation):
- **[CLS]** attends to all regional prompts and patch embeddings.
- **Patch embeddings** attend only to non-prompt tokens (shielded from prompt interference).
- **Regional prompt $R_j$** attends only to itself and the patches within its corresponding region (enabling local feature extraction).

$$\mathbf{Mask}[R_j] = \mathbbm{1}(j, b_j, \ldots, b_j + \lfloor N/M \rfloor - 1)$$

This ensures regional prompts focus on local feature extraction without corrupting the integrity of original patch embeddings.

### Key Design 3: Hierarchical Feature Alignment

**Design Motivation**: The feature space for long texts is more complex; aligning only the final layer is insufficient. Intermediate layer features should also maintain visual-language consistency.

**Group Token Aggregation (GTA)**: The [CLS] tokens from $L$ Transformer layers are divided into $G$ groups, each aggregated with Gaussian-weighted averaging:

$$\text{GTA}(\mathbf{T}_g) = \sum_{j=1}^S \mathbf{Gaussian}(j; S, 1) \cdot \mathbf{T}_g[j]$$

A linear projection followed by LayerNorm then produces the Group Middle Feature (GMF).

**Loss Function**: An InfoNCE loss $L_{m_i}$ is computed for each group's GMF between visual and language representations. Only groups $K$ through $G$ are aligned (shallow-layer features are found to be too dissimilar). The final loss is:

$$L = \sum_{i=K}^G \omega_i L_{m_i} + L_{\text{short}} + L_{\text{long}}$$

### Data Synthesis

- Long-text descriptions are synthesized using Llama3-LLaVA-NeXT-8b with 20 diverse prompts, averaging ~120 tokens per caption.
- Three dataset scales are constructed: 5M, 15M, and 30M.
- Low-quality descriptions (repetitive words, meaningless sentences, overly short outputs) are filtered out.

## Key Experimental Results

### Long-Text Retrieval (R@1)

| Method | Data | DCI I2T | DCI T2I | IIW I2T | IIW T2I | ShareGPT4V I2T | Urban I2T | Avg |
|--------|------|---------|---------|---------|---------|----------------|-----------|-----|
| CLIP (B/16) | 400M | 37.3 | 34.5 | 75.2 | 76.4 | 78.2 | 68.1 | 62.8 |
| Long-CLIP | 1M | 51.1 | 57.0 | 89.2 | 86.9 | 94.6 | 78.9 | 76.8 |
| LoTLIP | 100M | 62.1 | 61.0 | 93.9 | 92.5 | 96.5 | 77.8 | 81.9 |
| **Fix-CLIP** | **1M** | **59.7** | **63.0** | **93.8** | **95.6** | **95.5** | **80.9** | **82.6** |
| **Fix-CLIP** | **30M** | **70.7** | **70.7** | **97.4** | **97.4** | **98.6** | **90.8** | **89.8** |

Key findings:
- With only 1M training data, Fix-CLIP surpasses LoTLIP trained on 100M data (B/16 average: 82.6 vs. 81.9).
- The 30M variant substantially outperforms all baselines across all datasets, with an average improvement of ~8%.
- The L/14 model achieves an average R@1 of 91.2% on 30M data.

### Short-Text Retrieval (R@1)

| Method | COCO I2T | COCO T2I | Flickr I2T | Flickr T2I |
|--------|----------|----------|------------|------------|
| Long-CLIP (B/16) | 57.6 | 40.4 | 87.9 | 72.3 |
| **Fix-CLIP (1M, B/16)** | **60.9** | **44.8** | **88.8** | **77.4** |

Key findings:
- Fix-CLIP **enhances short-text performance** while improving long-text capability (COCO T2I +4.4%, Flickr T2I +5.1%).
- This is attributed to the masked image–short text branch in the dual-branch pipeline, which preserves the original feature space.

### Plug-and-Play for Diffusion Models

Fix-CLIP's text encoder can directly replace the CLIP text encoder in diffusion models (e.g., Stable Diffusion), significantly improving generation quality under long-text inputs. It supports inputs up to 248 tokens (vs. CLIP's original 77), enabling more faithful expression of complex descriptions.

## Highlights & Insights

1. **Incremental training paradigm**: Compared to training from scratch (LoTLIP at 100M), incremental training on just 1M samples already surpasses prior SOTA, demonstrating exceptional data efficiency.
2. **Elegance of the dual-branch design**: Pairing masked images with short texts leverages the MAE insight (75% masking preserves semantics) while avoiding feature space conflicts between long and short texts.
3. **Regional prompts with unidirectional masks**: Local alignment is achieved implicitly through the attention mechanism without requiring any additional region-level annotation.
4. **Hierarchical alignment**: Gaussian-weighted aggregation of intermediate features enables layer-by-layer alignment from shallow to deep, addressing the complexity of the long-text feature space.
5. **Strong data scalability**: Performance improves consistently and stably from 5M to 30M with no signs of saturation.

## Limitations & Future Work

1. Incremental training relies on CLIP's pre-trained weights and cannot fundamentally alter CLIP's visual encoding capacity.
2. Hyperparameters such as the number of regional prompts $M$ and the number of hierarchical groups $G$ require manual tuning.
3. The 248-token upper limit covers most practical scenarios but remains insufficient for document-level ultra-long descriptions.
4. Synthetic long texts may contain MLLM hallucinations; filtering mitigates but cannot entirely eliminate this issue.

## Related Work & Insights

- **Long-text extensions of CLIP**: Long-CLIP (positional encoding interpolation) → LoTLIP/FLAIR (training from scratch) → this work (incremental training + dual-branch pipeline).
- **Region-level alignment**: FILIP (patch-text correspondence) → MaskCLIP (mask-based augmentation) → this work (regional prompts + unidirectional masks).
- **Data augmentation**: LaCLIP (LLM rewriting) → VeCLIP/CAPSFUSION (MLLM-enriched captions) → this work (diverse prompt-based synthesis of 30M data).

## Rating

- Novelty: ⭐⭐⭐⭐ — Multiple modules combined innovatively, though each individual module has traceable precedents.
- Technical Depth: ⭐⭐⭐⭐ — Complete design integrating dual-branch, regional prompts, and hierarchical alignment, with thorough theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers long/short-text retrieval, classification, diffusion model application, multi-scale data experiments, and comprehensive ablation studies.
- Value: ⭐⭐⭐⭐⭐ — The plug-and-play text encoder offers direct practical value for diffusion models; code is open-sourced.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] CURE: Cultural Gaps in the Long Tail of Text-to-Image Systems](cure_cultural_gaps_in_the_long_tail_of_text-to-image_systems.md)
- [\[NeurIPS 2025\] Robustness in Both Domains: CLIP Needs a Robust Text Encoder](../../NeurIPS2025/image_generation/robustness_in_both_domains_clip_needs_a_robust_text_encoder.md)
- [\[ICCV 2025\] Generating Multi-Image Synthetic Data for Text-to-Image Customization](generating_multi-image_synthetic_data_for_text-to-image_customization.md)
- [\[ICCV 2025\] Contrastive Flow Matching (ΔFM)](contrastive_flow_matching.md)
- [\[ICCV 2025\] CoMPaSS: Enhancing Spatial Understanding in Text-to-Image Diffusion Models](compass_enhancing_spatial_understanding_in_text-to-image_diffusion_models.md)

<!-- RELATED:END -->
