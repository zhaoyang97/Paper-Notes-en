---
title: >-
  [Paper Note] LaMI: Augmenting Large Language Models via Late Multi-Image Fusion
description: >-
  [ACL 2026][Multimodal VLM][Late fusion] This paper proposes LaMI, a late-fusion architecture that integrates visual features with LLM outputs at the final prediction stage, and at inference time generates multiple images from input text for confidence-weighted aggregation. LaMI significantly enhances the visual commonsense reasoning capability of LLMs without compromising their text reasoning performance.
tags:
  - ACL 2026
  - Multimodal VLM
  - Late fusion
  - multi-image generation
  - visual commonsense reasoning
  - vision-augmented LLM
  - inference-time visual injection
date: 2026-05-08
content_hash: 394e2968d5416d5f
---

# LaMI: Augmenting Large Language Models via Late Multi-Image Fusion

**Conference**: ACL 2026
**arXiv**: [2406.13621](https://arxiv.org/abs/2406.13621)
**Code**: [Project Page](https://guyyariv.github.io/LaMI/)
**Area**: Multimodal VLM
**Keywords**: Late fusion, multi-image generation, visual commonsense reasoning, vision-augmented LLM, inference-time visual injection

## TL;DR

This paper proposes LaMI, a late-fusion architecture that integrates visual features with LLM outputs at the final prediction stage, and at inference time generates multiple images from input text for confidence-weighted aggregation. LaMI significantly enhances the visual commonsense reasoning capability of LLMs without compromising their text reasoning performance.

## Background & Motivation

**Background**: LLMs excel at purely textual tasks but lack visual commonsense knowledge (e.g., "What color is a penguin's belly?"). While vision-language models (VLMs) can handle visual tasks, they often sacrifice text reasoning performance, and multimodal training is costly.

**Limitations of Prior Work**: Existing vision-augmented language model (VaLM) approaches suffer from two core issues: (1) most adopt **early fusion**, where visual signals injected too early into the LLM interfere with its language behavior; (2) reliance on a **single image** introduces noise and bias.

**Key Challenge**: How to efficiently endow a text-only LLM with visual knowledge without degrading its text reasoning ability and without expensive multimodal retraining.

**Goal**: Design a lightweight, plug-and-play visual augmentation framework that simultaneously improves visual commonsense reasoning and preserves text performance.

**Key Insight**: Deferring the fusion of visual features to the final prediction stage (late fusion) avoids interference with intermediate LLM representations; generating multiple images at inference time provides diverse visual evidence.

**Core Idea**: Late Fusion + Multi-Image = visual augmentation without sacrificing language capability. The LLM remains frozen; only lightweight projection and fusion layers are trained.

## Method

### Overall Architecture

LaMI consists of four components: a frozen pretrained LLM, a frozen pretrained visual encoder, a trainable Visual Token Projector (VTP), and a trainable Late Fusion Attention Layer (LFAL). During training, image–text pairs are used; at inference time, multiple images are generated from the input text and aggregated for prediction.

### Key Designs

1. **Visual Token Projector (VTP)**: Maps patch features $z^v \in \mathbb{R}^{n_v \times d_v}$ extracted by the visual encoder to pseudo-text embeddings $u^v = W_1 \sigma(W_2 z^v) \in \mathbb{R}^{n_v \times d_x}$ via a two-layer MLP. **Design Motivation**: Aligns visual features to the LLM's text embedding space so that the subsequent fusion layer can effectively integrate cross-modal information.

2. **Late Fusion Attention Layer (LFAL)**: An attention layer inserted after the LLM's final representation and before the prediction head, with $K=V=[u^v; z^x_{(<t)}]$ and $Q=z^x_{(<t)}$, enabling text tokens to attend to visual tokens in a single step. **Design Motivation**: Deferring visual injection to the final stage allows the LLM to focus entirely on language processing throughout computation and access visual information only when needed, minimizing interference with language capability.

3. **Multi-Image Inference with Confidence Weighting**: At inference time, $k$ images are generated; each yields a distribution $p_i$, alongside a text-only distribution $p_0$. CLIP scores are used for confidence weighting: $p_{\text{final}} = \sum_i f(\bar{x}_i, v_i) p_i + (1 - f(\bar{x}_i, v_i)) p_0$. High-alignment images receive greater weight; when alignment is low, the method automatically falls back to the text-only LLM. **Design Motivation**: A single generated image may be noisy or biased; multiple images provide redundant visual evidence, and confidence weighting ensures that unreliable images do not degrade prediction.

### Loss & Training

Standard language modeling objective $\max_\theta \log P_\theta(x_{(t)} | x_{(<t)}, v)$ is used for training. Training data includes real image–text pairs and text paired with synthetically generated images. Only the VTP and LFAL parameters are trainable; the LLM and visual encoder remain frozen. At inference time, a distilled text-to-image generator is used for batched parallel sampling to minimize overhead.

## Key Experimental Results

### Main Results

| Model | Base | Visual Commonsense (VC) | Commonsense Reasoning (CR) | Reading Comprehension (RC) | Avg. |
|-------|------|------------------------|---------------------------|---------------------------|------|
| Llama3-8B | — | 52.0 | 72.0 | 57.9 | 60.6 |
| LaMI (Llama3-8B) | Llama3-8B | **55.0** | **72.9** | **58.0** | **62.0** |
| Llama3-8B-Instruct | — | 53.0 | 71.6 | 59.2 | 61.2 |
| Llava-Next (Llama3-8B-Inst.) | Llama3-8B-Inst. | 56.5 | 70.8 | 54.8 | 60.7 |
| LaMI (Llama3-8B-Inst.) | Llama3-8B-Inst. | 55.6 | **71.7** | **60.9** | **62.7** |

### Ablation Study

| Method | Memory Color | Color Terms | Object Shape | Relative Size |
|--------|-------------|-------------|-------------|--------------|
| GPT-2 (Base) | 32.4 | 34.6 | 44.5 | 43.1 |
| Early Fusion | 49.1 | 45.3 | 40.3 | 70.1 |
| Early Fusion + Multi | 55.5 | 52.1 | 41.2 | 75.5 |
| Intermediate Fusion + Multi | 69.7 | 67.8 | 63.0 | 81.1 |
| **Late Fusion + Multi (Ours)** | **72.5** | **69.2** | **66.8** | **85.5** |

### Key Findings

- **Late fusion consistently outperforms early and intermediate fusion**: Late fusion achieves best performance across all tasks, with particularly notable advantages on shape-related tasks.
- **Multi-image generation yields gains across all fusion strategies**: Improvements are especially significant for color and relative size reasoning.
- **LaMI improves visual commonsense without degrading—and even improves—text tasks**: This stands in sharp contrast to VLMs such as InstructBLIP and Llava-Next.
- Inference-time compute control experiment: Best-of-N sampling improves commonsense reasoning but fails to close the visual commonsense gap (VC: 47.8 vs. LaMI 50.1), confirming that LaMI's gains stem from visual evidence rather than additional computation.
- Performance saturates at approximately $k \approx 6$ images; $k=3$ already yields substantial gains.

## Highlights & Insights

- The design philosophy of **late fusion preserving language capability** is highly practical—the LLM remains frozen and visual information only "lightly touches" the model at the final stage, representing a minimally invasive paradigm for multimodal augmentation.
- The **automatic fallback via CLIP confidence weighting** is an elegant mechanism: when generated images are unreliable, the method automatically reverts to the text-only path, preventing visual noise from harming prediction.
- The method can be applied plug-and-play to any newly released LLM without costly multimodal retraining.

## Limitations & Future Work

- The approach depends on the quality of the text-to-image generator; generated images may introduce out-of-distribution noise.
- Inference requires generating multiple images, increasing latency and computational overhead (though parallelizable).
- Visual commonsense performance remains slightly below fully trained VLMs (e.g., Llava-Next VC: 56.5 vs. LaMI 55.6), though LaMI leads on overall average.
- Evaluation is limited to discriminative tasks (multiple-choice); effectiveness on open-ended generation tasks remains unexplored.
- Future work could explore more efficient means of obtaining visual evidence (e.g., retrieval rather than generation).

## Related Work & Insights

- **VaLM series** (VaLM, Z-LaVI, LiVE): LaMI's late-fusion strategy uniformly addresses the language capability degradation observed in early-fusion approaches.
- **CLIP as a cross-modal bridge**: CLIP scores are used to automatically assess image–text alignment and perform weighting without additional training.
- Insight: Deep fusion is not a prerequisite for multimodal augmentation; shallow late fusion has a natural advantage in preserving unimodal capabilities.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of late fusion and multi-image inference is not entirely novel, but the paper systematically validates its advantages; the CLIP-weighted fallback mechanism is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation from small to large models and from BERT to LLaMA3, with complete ablation studies; validation on larger-scale models is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with clearly articulated motivation, though notation is occasionally inconsistent.
- **Value**: ⭐⭐⭐⭐ Provides a practical and lightweight solution for rapidly adapting new LLMs to multimodal settings.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)
- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](../../CVPR2026/multimodal_vlm/multi-modal_image_fusion_via_intervention-stable_feature_learning.md)
- [\[ACL 2026\] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)
- [\[ACL 2026\] Leave My Images Alone: Preventing Multi-Modal Large Language Models from Analyzing Unauthorized Images](leave_my_images_alone_preventing_multi-modal_large_language_models_from_analyzin.md)
- [\[ACL 2026\] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)

<!-- RELATED:END -->
