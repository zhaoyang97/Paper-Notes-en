---
title: >-
  [Paper Note] CaptionSmiths: Flexibly Controlling Language Pattern in Image Captioning
description: >-
  [ICCV 2025][Multimodal VLM][image captioning] CaptionSmiths is a framework that enables slider-style flexible control over three caption attributes — length, descriptiveness…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "image captioning"
  - "controllable generation"
  - "continuous conditioning"
  - "vision-language models"
  - "language pattern control"
date: 2026-05-08
content_hash: fc96916157c487bd
---

# CaptionSmiths: Flexibly Controlling Language Pattern in Image Captioning

**Conference**: ICCV 2025
**arXiv**: [2507.01409](https://arxiv.org/abs/2507.01409)
**Code**: [https://github.com/omron-sinicx/captionsmiths](https://github.com/omron-sinicx/captionsmiths)
**Area**: Multimodal / VLM
**Keywords**: image captioning, controllable generation, continuous conditioning, vision-language models, language pattern control

## TL;DR
CaptionSmiths is a framework that enables slider-style flexible control over three caption attributes — length, descriptiveness, and lexical uniqueness — via continuous scalar interpolation rather than discrete clustering. Trained jointly on multiple datasets, it achieves more precise attribute control and higher lexical alignment quality than baselines.

## Background & Motivation
Image captioning is a fundamental computer vision task with broad applications, including assistance for visually impaired users. While vision-language foundation models (e.g., LLaVA, CLIP+LLM) have significantly advanced caption quality, they offer little flexible control over linguistic patterns such as length, information density, and lexical granularity. Existing controllable captioning methods suffer from three key limitations:
1. Validation is limited to a single dataset (COCO), leaving generalizability unclear.
2. Control is restricted to a single attribute — length.
3. Discrete cluster indices are used for conditioning, forcing the model to jump between cluster centers without representing intermediate states.

The root cause lies in how discrete conditioning artificially partitions a continuous attribute space into bins, erasing within-bin diversity and introducing the cluster count as an additional hyperparameter. This paper addresses the problem by quantifying each attribute as a continuous scalar in $[0,1]$ and performing smooth slider-style transitions via linear interpolation between two endpoint vectors. The core idea is: **continuous interpolation conditioning is mathematically equivalent to a single linear layer, yielding high parameter efficiency and far better utilization of training samples compared to discrete binning.**

## Method

### Overall Architecture
Built on the LLaVA architecture (CLIP ViT-L visual encoder + LLaMA-2 7B decoder), the framework computes a three-dimensional condition scalar (length $L$, descriptiveness $D$, uniqueness $U$) for each training caption. Each scalar is encoded as a token embedding and prepended to the caption sequence for conditioned autoregressive training. At inference time, users control output by specifying scalar values or providing a reference sentence.

### Key Designs

1. **Condition Calculator**:

    - Function: Automatically quantifies three attributes as scalars for each caption.
    - Mechanism:
        - **Length ($L$)**: Directly uses the LLaMA tokenizer token count $L_c$.
        - **Descriptiveness ($D$)**: Ratio of adjectives and nouns to total tokens: $D_c = \frac{1}{T_c}\sum_{t=1}^{T_c}\mathbb{I}[w_t \in \text{ADJ} \cup \text{NOUN} \setminus \mathcal{V}_{excl}]$, excluding non-descriptive nouns such as "image."
        - **Uniqueness ($U$)**: Mean inverse word frequency across all tokens: $U_c = \frac{1}{T_c}\sum_{t=1}^{T_c}\frac{1}{F(w_t)}$, assigning higher scores to captions with rarer vocabulary.
    - Design Motivation: No manual annotation is required; all values are computed automatically from corpus statistics. The three attributes cover orthogonal dimensions: length control, information density, and lexical granularity.

2. **Decorrelation + Normalization**:

    - Function: Decorrelates the three attribute values and normalizes them to $[0,1]$.
    - Mechanism: Length is treated as the primary attribute and left unchanged. Uniqueness is residualized against length via linear regression; descriptiveness is then residualized against both preceding attributes. All values are finally normalized by their maximum.
    - Design Motivation: Although empirical correlations are small (maximum $-0.11$), decorrelation ensures that adjusting one attribute does not inadvertently affect the others.

3. **Condition Encoding**:

    - Function: Encodes a $[0,1]$ scalar into a token embedding for the language model.
    - Mechanism: Two endpoint vectors $E_0$ and $E_1$ (each of dimension $d$) are learned per attribute. The condition embedding is produced by linear interpolation: $E_c^L = \bar{L}_c \cdot E_1^L + (1-\bar{L}_c) \cdot E_0^L$.
    - Design Motivation:
        - Mathematically equivalent to a single linear layer ($w \cdot x + b$) but more interpretable.
        - Adds only $2 \times d$ parameters per attribute (vs. $k \times d$ for discrete methods).
        - Both endpoint vectors are trained on virtually all samples (vs. discrete methods where each bin uses only its own subset), greatly improving training efficiency.

### Loss & Training
Standard autoregressive cross-entropy loss is used, with condition tokens prepended:
$$\mathcal{L}_c = -\sum_{t=1}^{T} \log p(w_{t+1}|w_{<t}, I, E_c^L, E_c^D, E_c^U)$$

Training data comprises 1.3 million image-caption pairs from six datasets (LN COCO, Detail23K, Docci, Laion-COCO, COCO, Monkey), providing rich diversity in length and style.

## Key Experimental Results

### Main Results

| Dataset / Model | BLEU@4 | METEOR | CIDEr | Rouge-L |
|---|---|---|---|---|
| **COCO (Short)** | | | | |
| Blip-3 | 8.2 | 28.5 | 57.5 | 36.4 |
| Qwen2-VL-7B | 9.9 | 34.4 | 84.3 | 39.7 |
| Concap | 11.5 | 36.7 | 95.9 | 39.9 |
| **CaptionSmiths** | **11.4** | **38.8** | **104.8** | **39.8** |
| **LN COCO (Middle)** | | | | |
| Concap | 9.6 | 33.5 | 23.5 | 32.3 |
| **CaptionSmiths** | **9.6** | **36.9** | **37.4** | **32.8** |
| **Docci (Long)** | | | | |
| Concap | 7.2 | 26.8 | 8.3 | 26.0 |
| **CaptionSmiths** | **9.1** | **32.2** | **29.7** | **26.9** |

CaptionSmiths improves CIDEr by 9.3%, 59.1%, and 257.8% respectively across the three datasets, with particularly pronounced gains on long-caption benchmarks.

### Ablation Study

| Configuration | CIDEr↑ | BLEU@4↑ | Length Error↓ | Note |
|------|--------|---------|-----------|------|
| Discrete (5 bins) | 110 | 12.5 | 13.8 | Coarse control |
| Discrete (20 bins) | 105 | 11.9 | 10.9 | Improved control but reduced alignment |
| Discrete (100 bins) | 103 | 11.6 | 9.7 | Sample efficiency issues emerge |
| **CaptionSmiths** | **112** | **12.6** | **1.6** | Best control precision and alignment |

Continuous conditioning reduces length control error from 9.7 to 1.6 — a 506% improvement over discrete (100 bins) — while simultaneously surpassing it on CIDEr.

### Key Findings
- Increasing the descriptiveness value improves CLIPScore (better image-caption alignment), even surpassing ground-truth captions.
- Increasing the uniqueness value significantly improves fine-grained category recall (CUB birds, Stanford Dogs, Stanford Cars).
- Self-retrieval evaluation: CaptionSmiths R@1 = 36.9 vs. Concap 32.5 vs. GT 32.6, indicating generated captions are more image-specific than GT.
- Lexical diversity far exceeds baselines (significantly higher unique word ratio).
- The three conditions are largely decoupled: changing one attribute affects others by only 1–2 tokens.

## Highlights & Insights
- The simplicity of continuous interpolation conditioning is compelling: only six $d$-dimensional vectors are added as parameters, yet best-in-class control precision is achieved.
- The conditioning mechanism can be understood through the lens of model soup / model merging: endpoint vectors correspond to model weights specialized for different "tasks," and interpolation achieves task blending.
- Self-retrieval performance exceeding GT captions demonstrates that conditioning teaches the model more discriminative description strategies.
- The decoupling of the three attribute controls validates the effectiveness of the decorrelation procedure.

## Limitations & Future Work
- Long captions suffer from hallucination, an inherent limitation of LLMs.
- Only three attributes are controlled; other important dimensions (e.g., sentiment, domain expertise) are not addressed.
- Intuitively specifying uniqueness values is non-trivial due to its dependence on dataset statistics.
- Human preferences for "good captions" are subjective; human preference feedback has not been incorporated.

## Related Work & Insights
- The essential distinction from length-control methods such as FlexCap lies in supporting multiple attributes with continuous conditioning.
- The continuous conditioning paradigm is generalizable to other conditional generation tasks, including controllable text generation and style transfer.
- The application of model soup ideas in parameter space offers a new paradigm for controllable generation.

## Rating
- Novelty: ⭐⭐⭐⭐ — The application of continuous interpolation conditioning to controllable captioning is novel and elegant, though the core technique is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Joint multi-dataset training and evaluation, per-attribute validation, self-retrieval assessment, ablation studies, and fine-grained classification verification.
- Writing Quality: ⭐⭐⭐⭐ — Logic is clear, figures and tables are intuitive, and formal derivations are paired with accessible intuitive explanations.
- Value: ⭐⭐⭐⭐ — Provides a practical and efficient solution for controllable image captioning; slider-style control has clear real-world application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Controlling Multimodal LLMs via Reward-guided Decoding](controlling_multimodal_llms_via_rewardguided_decoding.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)
- [\[CVPR 2026\] Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction](../../CVPR2026/multimodal_vlm/text-only_training_for_image_captioning_with_retrieval_augmentation_and_modality.md)
- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)
- [\[AAAI 2026\] DisCode: Distribution-Aware Score Decoder for Robust Automatic Evaluation of Image Captioning](../../AAAI2026/multimodal_vlm/discode_distribution-aware_score_decoder_for_robust_automatic_evaluation_of_imag.md)

</div>

<!-- RELATED:END -->
