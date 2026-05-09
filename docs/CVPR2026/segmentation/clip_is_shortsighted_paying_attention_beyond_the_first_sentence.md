---
title: >-
  [Paper Note] CLIP Is Shortsighted: Paying Attention Beyond the First Sentence
description: >-
  [CVPR2026][Segmentation][CLIP] This paper reveals a systematic bias in CLIP-family models toward the summary sentence and early tokens in long-form text, and proposes DeBias-CLIP, which eliminates this bias via three text augmentation strategies — summary removal, sentence sub-sampling, and token padding — achieving state-of-the-art performance on both long- and short-text retrieval benchmarks without introducing any additional parameters.
tags:
  - CVPR2026
  - Segmentation
  - CLIP
  - long-text retrieval
  - text-image alignment
  - contrastive learning
  - positional bias
  - data augmentation
date: 2026-05-08
content_hash: e81b696dd173b4f2
---

# CLIP Is Shortsighted: Paying Attention Beyond the First Sentence

**Conference**: CVPR2026
**arXiv**: [2602.22419](https://arxiv.org/abs/2602.22419)
**Code**: [TRAILab/DeBias-CLIP](https://github.com/TRAILab/DeBias-CLIP)
**Area**: Semantic Segmentation / Vision-Language Alignment
**Keywords**: CLIP, long-text retrieval, text-image alignment, contrastive learning, positional bias, data augmentation

## TL;DR

This paper reveals a systematic bias in CLIP-family models toward the summary sentence and early tokens in long-form text, and proposes DeBias-CLIP, which eliminates this bias via three text augmentation strategies — summary removal, sentence sub-sampling, and token padding — achieving state-of-the-art performance on both long- and short-text retrieval benchmarks without introducing any additional parameters.

## Background & Motivation

**Short-text training bias in CLIP**: CLIP models are trained on internet-scale image-text pairs dominated by short captions (averaging 13–16 tokens), causing the model to primarily encode simple descriptions of salient objects, with insufficient capacity for aligning complex scenes and dense descriptions.

**Context window limitation**: The original CLIP text encoder has a token limit of 77 (approximately 3–4 sentences), which precludes complete encoding of paragraph-level text and constrains fine-grained scene understanding.

**Limitations of long-text fine-tuning**: Methods such as Long-CLIP extend the context window by stretching positional encodings and fine-tuning on long-caption data, but do not address the fundamental bias problem.

**First-sentence summary shortcut**: Whether human-annotated or LLM-generated, long captions typically begin with a single summarizing sentence. This structure allows the model to minimize contrastive loss by relying solely on the first sentence, resulting in shortcut learning.

**Empirical evidence for early-token bias**: Experiments demonstrate that inserting meaningless padding sentences before informative tokens significantly degrades retrieval performance, confirming that CLIP systematically favors information appearing early in the sequence.

**Sensitivity to sentence permutation**: Swapping the first and fourth sentences in long captions causes a 9.7% drop in T2I retrieval on DOCCI for Long-CLIP; removing the first sentence causes a further drop of 17.1%, indicating that existing models rely heavily on both the presence and position of the summary sentence.

## Method

### Overall Architecture

DeBias-CLIP follows the dual-encoder architecture of Long-CLIP, extending the text positional encoding from 77 to 248 tokens (the first 20 positions are frozen; the remainder are linearly interpolated by a factor of 4). Training employs two contrastive losses to align long and short captions with image features separately. The key innovation lies in the construction strategy for short captions.

### Key Designs

**(1) Summary Removal**
The short caption is defined as the long caption with its first summary sentence removed: $C^{\text{no\_sum}} = [s_2, \dots, s_k]$. This compels the model to attend to fine-grained details deeper in the caption rather than relying on the informationally dense first sentence as a shortcut.

**(2) Sentence Sub-sampling**
From the sentence pool after summary removal, $n_{\text{sampled}} \sim \mathcal{U}\{1, \dots, n_{\text{sents}}-1\}$ sentences are sampled randomly without replacement, without preserving the original sentence order. This introduces training variation at negligible cost, increases the diversity between long and short captions, and encourages the model to remain sensitive to details throughout the text.

**(3) Token Padding**
A random portion of the padding tokens following the caption is relocated to precede it (after the SOT token), i.e., $n_{\text{pre}} \sim \mathcal{U}\{0, \dots, n_{\text{post}}\}$. This causes the effective text to appear at varying positional encoding positions, promoting more uniform positional embedding learning while preserving short-text retrieval performance.

The final tokenized short caption takes the form: $T^s_{\text{ours}} = [\text{SOT}, \text{PAD}_{\text{pre}}, s_4, s_2, \text{EOT}, \text{PAD}_{\text{post}}]$.

### Loss & Training

A weighted sum of two contrastive losses is employed:

$$\mathcal{L} = \lambda^s \mathcal{L}^s + (1 - \lambda^s) \mathcal{L}^\ell$$

where $\mathcal{L}^s$ is the alignment loss for short captions (after sub-sampling and padding) and $\mathcal{L}^\ell$ is the alignment loss for full long captions. Short captions are aligned with PCA-compressed image features. The optimal weight $\lambda^s = 0.1$ yields the best trade-off between short- and long-text retrieval.

## Key Experimental Results

### Basic Setup

- Training data: ShareGPT4V (1.2M image-text pairs)
- Pre-trained weights: OpenAI CLIP (default)
- Training: 3 epochs, batch size 256, 4×A100 GPUs
- Evaluation: Urban1k, DCI, Long-DCI, DOCCI (long-text); COCO, Flickr30k (short-text)

### Main Results — Long-Text Retrieval

| Method | Urban1k T2I / I2T | DCI T2I / I2T | DOCCI T2I / I2T |
|--------|-------------------|---------------|-----------------|
| CLIP (ViT-B/16) | 53.4 / 67.5 | 42.9 / 44.1 | 57.1 / 60.6 |
| Long-CLIP | 79.5 / 78.9 | 57.1 / 51.6 | 71.4 / 63.1 |
| SmartCLIP | 87.4 / 90.0 | 64.0 / 64.9 | 78.0 / 77.4 |
| **DeBias-CLIP** | **93.0 / 93.1** | **67.6 / 68.5** | **80.0 / 79.7** |
| CLIP (ViT-L/14) | 56.1 / 68.5 | 43.8 / 44.8 | 63.0 / 65.8 |
| Long-CLIP | 86.0 / 82.5 | 63.9 / 57.0 | 78.6 / 66.5 |
| SmartCLIP | 90.1 / 93.3 | 69.8 / 68.2 | 82.5 / 81.6 |
| **DeBias-CLIP** | **95.2 / 95.2** | **73.5 / 72.8** | **85.6 / 85.2** |

### Main Results — Short-Text Retrieval

| Method (ViT-B/16) | COCO T2I / I2T | Flickr30k T2I / I2T |
|--------------------|----------------|----------------------|
| Long-CLIP | 40.4 / 57.6 | 34.1 / 46.8 |
| SmartCLIP | 42.4 / 61.9 | 36.3 / 55.6 |
| **DeBias-CLIP** | **43.0** / 61.3 | **36.9 / 57.0** |

### Ablation Study

| Configuration | COCO T2I | Flickr T2I | Urban1k T2I | DOCCI T2I |
|---------------|----------|------------|-------------|-----------|
| Long-CLIP | 40.4 | 34.1 | 79.5 | 71.4 |
| + 3 epochs | 39.6 | 33.1 | 81.0 | 74.1 |
| + Summary removal | 42.2 | 36.0 | **92.6** | **80.9** |
| + Sentence sub-sampling | 41.9 | 36.1 | 92.5 | 80.8 |
| + Token padding (full) | **43.0** | **36.9** | 93.0 | 79.7 |

### Key Findings

- **Summary removal is the core contribution**: Removing only the first summary sentence improves Urban1k T2I from 81.0 to 92.6 (+11.6%) and DOCCI from 74.1 to 80.9 (+6.8%).
- **Sub-sampling and padding are complementary**: Sentence sub-sampling and token padding primarily improve short-text retrieval (COCO +1.1%, Flickr +0.9%), with minor fluctuations on long-text benchmarks that are maintained overall.
- **More uniform attention distribution**: Long-CLIP exhibits attention weight spikes at early tokens that decay rapidly; DeBias-CLIP maintains a flatter attention distribution across all token positions.
- **Strong generalizability**: The method is effective across diverse pre-trained models including OpenAI CLIP, OpenCLIP (LAION-2B), SigLIP, and SigLIP2; the performance drop under sentence permutation is reduced from −9.7% (Long-CLIP) to −3.5%.

## Highlights & Insights

- **Insightful analysis**: The paper systematically identifies early-token bias and the summary-sentence shortcut in CLIP models, supported by thorough empirical validation.
- **Extreme simplicity**: Zero additional parameters and no additional training stages are required; significant performance gains are achieved solely through text sampling strategies during training, making it a plug-and-play replacement for Long-CLIP.
- **Comprehensive evaluation**: Experiments span 4 long-text and 2 short-text benchmarks, across ViT-B/16 and ViT-L/14 scales and 4 pre-trained models, with thorough ablation and robustness analyses.
- **High practical value**: The approach improves retrieval robustness to sentence ordering, making it well-suited for real-world applications such as document passage retrieval in RAG pipelines.

## Limitations & Future Work

- Long-text evaluation benchmarks (Urban1k, DCI, DOCCI) all adopt the same "summary-then-detail" structure, which may obscure the model's true performance on long-text formats with different organizational patterns.
- The method treats sentences as independent semantic units during sub-sampling, neglecting cross-sentence contextual dependencies and potentially discarding information that requires multi-sentence joint understanding.
- For stronger pre-trained models such as SigLIP/SigLIP2, performance drops under sentence permutation remain at −6.1%/−6.5%, indicating that positional bias is not fully eliminated.
- Fine-tuning is conducted solely on ShareGPT4V (1.2M); larger-scale datasets or training-from-scratch scenarios are not explored.

## Related Work & Insights

- **Long-CLIP** (ECCV'24): Extends CLIP's context window via positional encoding interpolation and dual-loss training, but retains summary-sentence bias.
- **SmartCLIP** (CVPR'25): Learns text-conditioned channel masking networks, introducing additional parameters and sampling sentences starting from the summary.
- **FineLIP** (CVPR'25): Adds cross-modal feature refinement modules requiring additional computation at inference time.
- **TULIP** (ICLR'25): Replaces interpolated positional encodings with rotary positional embeddings but requires two-stage training.
- **Fix-CLIP** (ICCV'25): Employs token masking and local aggregation tokens to preserve short-text performance.

## Rating

- Novelty: ⭐⭐⭐⭐ — The analytical perspective on CLIP bias is novel, though the method itself falls within the scope of training-time data augmentation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multi-dataset, multi-model, multi-scale evaluation with comprehensive ablation and robustness analyses
- Writing Quality: ⭐⭐⭐⭐⭐ — Logically clear, with a natural progression from problem analysis to method design
- Value: ⭐⭐⭐⭐ — The zero-extra-parameter plug-and-play solution offers high practical utility and provides meaningful insights into CLIP biases for the research community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DeBias-CLIP: CLIP Is Shortsighted — Paying Attention Beyond the First Sentence](clip_shortsighted_beyond_first_sentence.md)
- [\[CVPR 2026\] Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation](looking_beyond_the_window_global-local_aligned_clip_for_training-free_open-vocab.md)
- [\[CVPR 2026\] ConceptPrism: Concept Disentanglement in Personalized Diffusion Models via Residual Token Optimization](conceptprism_concept_disentanglement_in_personalized_diffusion_models_via_residu.md)
- [\[CVPR 2026\] A Mixed Diet Makes DINO An Omnivorous Vision Encoder](a_mixed_diet_makes_dino_an_omnivorous_vision_encoder.md)
- [\[CVPR 2026\] Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation](seeing_beyond_extrapolative_domain_adaptive_panoramic_segmentation.md)

</div>

<!-- RELATED:END -->
