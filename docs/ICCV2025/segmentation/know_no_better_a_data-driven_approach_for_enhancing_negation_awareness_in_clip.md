---
title: >-
  [Paper Note] Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP
description: >-
  [ICCV 2025][Segmentation][CLIP] By analyzing the scarcity and misalignment of negation expressions in CLIP's pre-training data, this work designs two LLM/MLLM-based negation data generation pipelines to fine-tune the CLIP text encoder, producing NegationCLIP — a model that enhances negation understanding while preserving general performance. A new benchmark, NegRefCOCOg, is proposed for comprehensive negation evaluation.
tags:
  - ICCV 2025
  - Segmentation
  - CLIP
  - Negation Understanding
  - Data Generation
  - Vision-Language Models
  - Referring Image Segmentation
date: 2026-05-08
content_hash: c4f8aef974c5b682
---

# Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP

**Conference**: ICCV 2025
**arXiv**: [2501.10913](https://arxiv.org/abs/2501.10913)
**Code**: [GitHub](https://github.com/parkquasar/NegationCLIP)
**Area**: Image Segmentation
**Keywords**: CLIP, Negation Understanding, Data Generation, Vision-Language Models, Referring Image Segmentation

## TL;DR

By analyzing the scarcity and misalignment of negation expressions in CLIP's pre-training data, this work designs two LLM/MLLM-based negation data generation pipelines to fine-tune the CLIP text encoder, producing NegationCLIP — a model that enhances negation understanding while preserving general performance. A new benchmark, NegRefCOCOg, is proposed for comprehensive negation evaluation.

## Background & Motivation

CLIP, as a cornerstone of vision-language models, is widely adopted in downstream tasks such as text-to-image generation and referring image segmentation. However, CLIP suffers from a severe yet overlooked deficiency — **its inability to correctly understand negation**. For instance, "parking" and "no parking" are nearly indistinguishable to CLIP.

This paper systematically identifies the root cause of this problem through controlled experiments:

**CelebA Experiments Expose the Issue**: Positive/negative prompt pairs are designed for 40 binary attributes (e.g., "wearing glasses" vs. "not wearing glasses"). CLIP ViT-L/14 achieves an average balanced accuracy of only 60.8% — barely above the 50% random baseline for binary classification, in stark contrast to its 73.4% accuracy on the 1000-class ImageNet benchmark.

**Root Cause Analysis — Missing Training Data**: Analysis of the LAION-400M dataset reveals that only 0.70% of captions contain negation words, which account for merely 0.08% of total vocabulary. More critically, even when negation appears in a caption (e.g., "no smoking"), it is frequently irrelevant or misaligned with the image content (which may depict someone actively smoking). This is inherent to the nature of image captioning, which describes what is present rather than what is absent.

**Limitations of Prior Work**: Methods such as CoN-CLIP generate negation captions and then retrieve visually similar images, but this "text-first, image-second" strategy results in poor semantic alignment between the negation captions and the retrieved images.

## Method

### Overall Architecture

The training pipeline of NegationCLIP consists of two stages: (1) constructing high-quality negation data via two generation pipelines, and (2) fine-tuning only the text encoder while keeping the visual encoder frozen. This design preserves the original image embedding space, allowing NegationCLIP to serve as a plug-and-play replacement for the CLIP text encoder in existing models.

### Key Designs

1. **Pipeline 1 — Object Absence-Based Negation Generation**: This pipeline leverages existing image-caption datasets (COCO) through a three-step process: (a) a LLM (Llama-3-8B) identifies *plausible objects* — entities not mentioned in the caption but reasonably likely to appear in such a scene; (b) a MLLM (LLaVA-1.6) verifies that these objects are indeed **absent** from the image; (c) the LLM naturally integrates the negation of the absent object into the original caption. The core design motivation is an "image-first" strategy — starting from the image and then generating negation descriptions that are **visually grounded**, rather than the reverse. Selecting contextually plausible objects for negation is more effective than random selection, as only contextually relevant negations provide meaningful learning signal.

2. **Pipeline 2 — Negation Diversity Expansion**: Pipeline 1 covers only object-existence negation. To broaden the range of negation types, Pipeline 2 leverages VQA data (VQAv2) containing question–answer pairs where the answer is "no." These pairs span diverse negation categories including actions ("is the person running?" → "no") and attributes ("is the car red?" → "no"). The process: (a) filter triplets with negative answers; (b) prompt the LLM to incorporate the question and negative answer into the original caption. In total, 229K negation samples are generated (P1: 147K, P2: 82K).

3. **NegRefCOCOg Benchmark**: Existing negation evaluation benchmarks exhibit severe bias — CREPE Negate and CC-Neg assume that all captions containing negation words are incorrect matches, meaning a model that simply detects negation words can "cheat" its way to high accuracy. NegRefCOCOg is constructed from RefCOCOg by: (a) filtering prompts containing negation words; (b) identifying positive samples $P^+$ (target regions matching the prompt) and hard negative samples $P^-$ (regions of the same object category that do not satisfy the negation description); (c) evaluating whether the model correctly associates the negation prompt with $P^+$ over $P^-$. The benchmark supports multiple negation words ("no", "not", "without") and varied negation targets (objects, actions, attributes).

### Loss & Training

The standard InfoNCE loss is used to fine-tune the text encoder, with a learning rate of $1e{-6}$ and AdamW optimizer. The visual encoder is fully frozen, and only the text encoder parameters are updated. This design ensures: (1) the image embedding space remains unchanged, so the fine-tuned text encoder can directly replace the original in downstream tasks; (2) training is computationally efficient, requiring no updates to the visual side.

## Key Experimental Results

### Main Results

**Negation Understanding vs. General Performance (Multi-Architecture Comparison)**

| Model | Architecture | VALSE ↑ | NegRefCOCOg ↑ | ImageNet ↑ | COCO ↑ |
|-------|-------------|---------|--------------|-----------|--------|
| CLIP | ViT-B/32 | 70.97 | 57.73 | 62.02 | 54.78 |
| CLIP-bnl | ViT-B/32 | 76.78 | 62.05 | 53.33 | 55.47 |
| CoN-CLIP | ViT-B/32 | 71.72 | 55.45 | 63.08 | 55.66 |
| **NegationCLIP** | ViT-B/32 | **80.15** | **64.09** | 60.97 | **68.00** |
| CLIP | ViT-L/14 | 66.85 | 57.27 | 73.44 | 59.99 |
| **NegationCLIP** | ViT-L/14 | **79.59** | **62.95** | 73.91 | **72.77** |

NegationCLIP outperforms all baselines on negation benchmarks by 9–13 points, while maintaining or exceeding the original CLIP's performance on ImageNet and COCO.

### Ablation Study

**Impact of Data Configuration on Negation Understanding (ViT-B/32)**

| Data Configuration | VALSE ↑ | NegRefCOCOg ↑ | Notes |
|-------------------|---------|--------------|-------|
| Original CLIP | 70.97 | 57.73 | Baseline |
| + Rand-P1 (random objects) | 73.78 | 62.05 | Random negation offers limited benefit |
| + P1 (plausible objects) | **80.15** | 63.18 | Context-aware negation is more effective |
| + P2 (VQA negation) | 76.78 | 64.32 | Diversifies negation types |
| + **P1 + P2** | **80.15** | **64.09** | Complementary pipelines, best overall |

Rand-P1 consistently underperforms P1, validating the importance of plausible object selection. P1 + P2 achieves the best results on NegRefCOCOg, suggesting the benchmark more effectively captures negation diversity than VALSE.

**Downstream Task Validation**

| Model | PhraseCut mIoU | RefCOCOg(Neg) mIoU |
|-------|--------------|-------------------|
| CLIPSeg | 0.562 | 0.267 |
| CoN-CLIPSeg | 0.539 | 0.123 |
| **NegationCLIPSeg** | 0.561 | **0.288** |

| Model | T2I TIFA ↑ | T2I Neg Score ↑ |
|-------|-----------|----------------|
| SD-1.4 | 0.786 | 0.295 |
| SD-1.4 + CoN-CLIP | 0.783 | 0.243 |
| **SD-1.4 + NegationCLIP** | **0.790** | **0.449** |

### Key Findings

- **Data Quality over Quantity**: 229K high-quality negation samples suffice to significantly improve CLIP's negation understanding.
- **Image-First Strategy**: Generating aligned negation captions from images outperforms CoN-CLIP's "caption-first, image-retrieval-second" approach.
- **Plug-and-Play Enhancement**: Replacing only the text encoder improves negation performance in T2I generation (Neg Score: 0.295 → 0.449) and referring image segmentation.
- **Benchmark Bias in Prior Work**: Existing benchmarks (CREPE, CC-Neg) are negation-biased and susceptible to blind-model cheating; NegRefCOCOg addresses this through hard negative sample design.
- **CoN-CLIP Can Be Harmful**: CoN-CLIP underperforms the original CLIP on certain tasks, attributed to misalignment between its negation captions and retrieved images.

## Highlights & Insights

- **Precise Problem Formulation**: Through CelebA experiments and LAION-400M data analysis, the paper clearly identifies the root cause of CLIP's negation understanding deficiency.
- **High Practical Utility**: NegationCLIP functions as a plug-and-play text encoder that improves negation handling without modifying any downstream model.
- **Rigorous Benchmark Design**: NegRefCOCOg avoids the biases of existing benchmarks and supports diverse negation types.
- **Cross-Task Validation**: Effectiveness is demonstrated across three distinct tasks: image-text matching, text-to-image generation, and referring image segmentation.

## Limitations & Future Work

- Only the text encoder is fine-tuned; the visual encoder remains unadapted, which may limit deeper negation semantic understanding.
- The negation data generation relies on LLM/MLLM capabilities, introducing potential noise and computational cost.
- A slight drop in TIFA is observed in the SDXL dual-text-encoder setting, suggesting that the replacement strategy requires further optimization for complex architectures.
- Other negation types (e.g., double negation, implicit negation) remain unexplored.
- The generated dataset (229K) is small relative to LAION-400M; whether larger-scale negation data yields further gains is an open question.
- The proposed approach could be extended to video VLMs and multilingual settings.

## Related Work & Insights

- CLIP-bnl and CoN-CLIP are the most directly related prior works, but are limited by their respective data generation strategies (text-first vs. image-first).
- Hard negative mining is widely used in NLP negation research (e.g., BERT-based studies); this paper transfers that paradigm to the multimodal domain.
- Negation understanding is critical for AI safety (e.g., distinguishing "no weapons" from "weapons"), and the proposed approach carries practical safety implications.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The data-driven negation augmentation approach is clear and effective; the NegRefCOCOg benchmark design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive validation across 4 architectures and 3 tasks, with detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ In-depth problem analysis with a complete logical chain from root cause to methodology to evaluation.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a critical blind spot in CLIP; the plug-and-play design enables broad practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Enhancing Transformers Through Conditioned Embedded Tokens](enhancing_transformers_through_conditioned_embedded_tokens.md)
- [\[ICCV 2025\] A Plug-and-Play Physical Motion Restoration Approach for In-the-Wild High-Difficulty Motions](a_plugandplay_physical_motion_restoration_approach_for_inthe.md)
- [\[ICCV 2025\] Know Your Attention Maps: Class-specific Token Masking for Weakly Supervised Semantic Segmentation](know_your_attention_maps_class-specific_token_masking_for_weakly_supervised_sema.md)
- [\[ICCV 2025\] DDB: Diffusion Driven Balancing to Address Spurious Correlations](ddb_diffusion_driven_balancing_to_address_spurious_correlations.md)
- [\[CVPR 2026\] DeBias-CLIP: CLIP Is Shortsighted — Paying Attention Beyond the First Sentence](../../CVPR2026/segmentation/clip_shortsighted_beyond_first_sentence.md)

</div>

<!-- RELATED:END -->
