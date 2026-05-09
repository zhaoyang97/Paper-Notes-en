---
title: >-
  [Paper Note] DeBias-CLIP: CLIP Is Shortsighted — Paying Attention Beyond the First Sentence
description: >-
  [CVPR 2026][Segmentation][CLIP] The paper shows that CLIP and Long-CLIP suffer from a serious early-token bias and a first-sentence summary shortcut. DeBias-CLIP uses three simple augmentations — removing the summary sentence, sentence sub-sampling, and prefix-token padding — that introduce no extra parameters and reach SOTA on multiple long-text retrieval benchmarks.
tags:
  - CVPR 2026
  - Segmentation
  - CLIP
  - long-text retrieval
  - attention bias
  - data augmentation
  - positional encoding stretching
date: 2026-05-08
content_hash: 416a1d32fc6fa661
---

# DeBias-CLIP: CLIP Is Shortsighted — Paying Attention Beyond the First Sentence

**Conference**: CVPR 2026  
**arXiv**: [2602.22419](https://arxiv.org/abs/2602.22419)  
**Code**: [https://github.com/TRAILab/DeBias-CLIP.git](https://github.com/TRAILab/DeBias-CLIP.git)  
**Area**: Segmentation  
**Keywords**: CLIP, long-text retrieval, attention bias, data augmentation, positional encoding stretching

## TL;DR

The paper shows that CLIP and Long-CLIP suffer from a serious early-token bias and a first-sentence summary shortcut. DeBias-CLIP uses three simple augmentations — removing the summary sentence, sentence sub-sampling, and prefix-token padding — that introduce no extra parameters and reach SOTA on multiple long-text retrieval benchmarks.

## Background & Motivation

**Background**: CLIP builds a multimodal joint representation space via image–text contrastive learning and is widely used for zero-shot classification, multimodal retrieval, and text-to-image generation. Long-CLIP and similar works extend CLIP's text-understanding length by stretching positional encodings and fine-tuning on long-caption data.

**Limitations of Prior Work**: CLIP's pre-training data is dominated by short captions, leading to a strong bias toward early tokens. More importantly, existing long-caption datasets (e.g. ShareGPT4V) almost universally follow a "summary first sentence + detailed description" format, where the first sentence carries the bulk of the caption's information and looks very similar to a short caption.

**Key Challenge**: When training Long-CLIP on such captions, the model can minimize the contrastive loss by attending only to the first summary sentence — forming a shortcut that lets it produce a low training loss without truly extending its effective context window. After deleting or moving the first sentence, retrieval performance drops sharply (-17.1% and -9.7%).

**Goal**: (1) How can the early-token bias of CLIP's text encoder be quantitatively analyzed? (2) How can the first-sentence summary shortcut in the Long-CLIP training framework be eliminated? (3) How can long-text retrieval be improved without adding any extra parameters?

**Key Insight**: The authors start from data augmentation, observing that the summary first sentence is the training-time shortcut — since it is the source of the bias, simply remove it during training, and use sentence sampling and prefix padding to spread attention.

**Core Idea**: Break CLIP's early-token bias shortcut by removing the summary first sentence from training captions, randomly sub-sampling the remaining sentences, and adding prefix padding.

## Method

### Overall Architecture

DeBias-CLIP follows Long-CLIP's dual-caption training framework: for each image, two caption versions are prepared — a long caption $C^{\ell}$ (full text) and a short caption $C^{s}$ (the augmented subset). Contrastive losses are computed for each and combined as a weighted sum. The key difference is in how the short caption is constructed.

### Key Designs

1. **Replacing the Summary Sentence**:

    - Function: define the short caption as the long caption with the first sentence removed, $C^{\mathrm{no\_sum}} = [s_2, \ldots, s_k]$.
    - Mechanism: Long-CLIP uses the first sentence as the short caption to preserve short-text performance, but this is exactly what enables the early-token bias shortcut. DeBias-CLIP does the opposite — drop the first sentence to force the model to attend to fine-grained descriptions deeper in the caption.
    - Design Motivation: experiments verify that on DOCCI, Long-CLIP's positive-sample similarity using only the first sentence (0.320) is actually higher than using the full long caption (0.308), confirming that the model relies on the first sentence and ignores the rest.

2. **Sentence Sampling**:

    - Function: randomly sample several sentences from the post-removal caption to form a new short caption $C^{\mathrm{samp}} = [s_4, s_2]$.
    - Mechanism: the number of sampled sentences $n_{\mathrm{sampled}} \sim \mathcal{U}\{1, 2, \ldots, n_{\mathrm{sents}}-1\}$ is drawn from a uniform distribution; the original sentence order is not preserved, introducing diversity in length and content.
    - Design Motivation: enlarging the difference between the short caption and the long caption forces the model to remain sensitive to details at every text position.

3. **Token Padding (Prefix Padding)**:

    - Function: move some of the trailing padding tokens to the front of the caption as prefix padding $T^s_{\mathrm{ours}} = [\mathtt{SOT}, \mathtt{PAD}_{\mathrm{pre}}, s_4, s_2, \mathtt{EOT}, \mathtt{PAD}_{\mathrm{post}}]$.
    - Mechanism: $n_{\mathrm{pre}} \sim \mathcal{U}\{0, 1, \ldots, n_{\mathrm{post}}\}$ tokens are randomly drawn from the existing post-padding and prepended; no text token is truncated.
    - Design Motivation: it solves two problems — (1) uneven training of positional encodings (long captions over-train the early positions) and (2) the sampled short caption being shorter than the first sentence, which hurts the pre-trained model's short-text performance.

### Loss & Training

The final loss is a weighted dual-caption contrastive loss: $\mathcal{L} = \lambda^s \mathcal{L}^s + (1-\lambda^s) \mathcal{L}^{\ell}$, where $\mathcal{L}^s$ uses the PCA-approximated image features against the augmented short caption. Long-CLIP's positional-encoding stretching scheme is reused (first 20 positions frozen, stretching factor 4); the model is trained on ShareGPT4V for 3 epochs with batch size 256 on 4× A100.

## Key Experimental Results

### Main Results

| Dataset | Metric | DeBias-CLIP (B/16) | SmartCLIP | Long-CLIP | CLIP |
|---------|--------|---------------------|-----------|-----------|------|
| Urban1k | T2I Top-1 | **93.0** | 87.4 | 79.5 | 53.4 |
| DCI | T2I Top-1 | **67.6** | 64.0 | 57.1 | 42.9 |
| Long-DCI | T2I Top-1 | **57.4** | 52.8 | 47.0 | 32.7 |
| DOCCI | T2I Top-1 | **80.0** | 78.0 | 71.4 | 57.1 |
| COCO | T2I Top-1 | **43.0** | 42.4 | 40.4 | 32.7 |
| Flickr30k | I2T Top-1 | **57.0** | 55.6 | 46.8 | 44.1 |

### Ablation Study

| Configuration | Urban1k T2I | DOCCI T2I | COCO T2I | Note |
|---------------|-------------|-----------|----------|------|
| Long-CLIP baseline | 79.5 | 71.4 | 40.4 | original method |
| + Remove summary | 88.4 | 77.2 | 41.6 | remove the first sentence; biggest gain |
| + Sentence sampling | 89.8 | 77.5 | 41.2 | add sentence sampling |
| + Token padding (full) | **93.0** | **80.0** | **43.0** | full model |

### Key Findings

- Removing the summary sentence is the single most important improvement, contributing +8.9% on Urban1k by itself.
- After moving the first sentence, DeBias-CLIP only loses 3.5%, while Long-CLIP loses 9.7% — robustness improves substantially.
- The method generalizes across pre-trained variants such as SigLIP and SigLIP2, with consistent improvements.
- Attention-weight analysis shows that DeBias-CLIP's attention is more uniformly distributed across text token positions.

## Highlights & Insights

- **Extremely simple method design**: no extra trainable parameters — just a training-time text augmentation strategy that reaches SOTA. It is a drop-in replacement for Long-CLIP.
- **Discovery of the first-sentence bias**: it reveals a structural problem in long-caption datasets, with implications for dataset construction.
- **Transferable augmentation strategies**: prefix padding and sentence sampling can be applied to any contrastive learning model trained on long text.

## Limitations & Future Work

- SigLIP and SigLIP2 still show large position sensitivity (more than 6% drop on the Move setting); the pre-training-induced bias is more deeply rooted.
- The impact on downstream VLM or diffusion-model tasks has not been explored.
- A trade-off between short-text and long-text performance still remains.

## Related Work & Insights

- **vs Long-CLIP**: also based on positional-encoding stretching and dual-caption training, but Long-CLIP uses the first sentence as the short caption and reinforces the bias, whereas DeBias-CLIP does the opposite to remove it.
- **vs SmartCLIP**: SmartCLIP adds a text-conditioned masking network; DeBias-CLIP surpasses it without adding any parameters.
- **vs FineLIP**: FineLIP adds a cross-modal feature refinement module and requires knowing the positive pair at inference, limiting its practicality.

## Rating

- Novelty: ⭐⭐⭐⭐ — the observation is fresh and deep; the method is intuitive data augmentation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — multiple datasets, multiple models, rich ablations and analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ — clear logic, experiment-driven storytelling.
- Value: ⭐⭐⭐⭐ — concrete improvement for CLIP long-text understanding.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] CLIP Is Shortsighted: Paying Attention Beyond the First Sentence](clip_is_shortsighted_paying_attention_beyond_the_first_sentence.md)
- [\[CVPR 2026\] Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation](looking_beyond_the_window_global-local_aligned_clip_for_training-free_open-vocab.md)
- [\[CVPR 2026\] PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)
- [\[CVPR 2026\] Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)
- [\[CVPR 2026\] DPAD: Discriminative Perception via Anchored Description for Reasoning Segmentation](discriminative_perception_via_anchored_description_for_reasoning_segmentation.md)

<!-- RELATED:END -->
