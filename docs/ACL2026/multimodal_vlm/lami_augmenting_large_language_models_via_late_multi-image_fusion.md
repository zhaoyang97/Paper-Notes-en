---
title: >-
  [Paper Note] LaMI: Augmenting Large Language Models via Late Multi-Image Fusion
description: >-
  [ACL 2026][Multimodal VLM][Late fusion] Ours proposes LaMI, which fuses visual features with LLM outputs at the final stage of prediction through a late fusion architecture and aggregates confidence-based predictions fro…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Late fusion"
  - "multi-image generation"
  - "visual commonsense reasoning"
  - "vision-augmented LLM"
  - "inference-time visual injection"
date: 2026-05-08
content_hash: 106100848289554a
---

# LaMI: Augmenting Large Language Models via Late Multi-Image Fusion

**Conference**: ACL 2026  
**arXiv**: [2406.13621](https://arxiv.org/abs/2406.13621)  
**Code**: [Project Page](https://guyyariv.github.io/LaMI/)  
**Area**: Multimodal VLM  
**Keywords**: Late fusion, multi-image generation, visual commonsense reasoning, vision-augmented LLM, inference-time visual injection  

## TL;DR

Ours proposes LaMI, which fuses visual features with LLM outputs at the final stage of prediction through a late fusion architecture and aggregates confidence-based predictions from multiple images generated at inference time. It significantly enhances the visual commonsense reasoning of LLMs without compromising their textual reasoning capabilities.

## Background & Motivation

**Background**: LLMs perform exceptionally well on text-only tasks but lack visual commonsense (e.g., "what color is an emperor penguin's belly"). While Vision-Language Models (VLMs) can handle visual tasks, they often sacrifice text reasoning performance, and multimodal training is prohibitively expensive.

**Limitations of Prior Work**: Existing Vision-augmented Language Model (VaLM) solutions face two core issues: (1) most adopt **early fusion**, where visual signals injected too early into the LLM interfere with its linguistic behavior; (2) they rely solely on a **single image**, which is prone to introducing noise and bias.

**Key Challenge**: How to efficiently augment text-only LLMs with visual knowledge without affecting their text reasoning capabilities and without requiring expensive multimodal retraining.

**Goal**: Design a lightweight, plug-and-play vision augmentation scheme that balances visual commonsense improvement with the preservation of textual performance.

**Key Insight**: Postpone the fusion of visual features to the final stage of prediction (late fusion) to avoid interfering with the LLM's intermediate representations; generate multiple images during inference to provide diverse visual evidence.

**Core Idea**: Late Fusion + Multi-Image = visual enhancement without loss of linguistic capability. The LLM remains frozen, and only lightweight projection and fusion layers are trained.

## Method

### Overall Architecture

LaMI consists of four components: a frozen pre-trained LLM, a frozen pre-trained vision encoder, a trainable Visual Token Projector (VTP), and a trainable Late Fusion Attention Layer (LFAL). The model is trained using image-text pairs, and at inference time, it generates multiple images from the input text for aggregated prediction.

### Key Designs

1.  **Visual Token Projector (VTP)**: Maps patch features $z^v \in \mathbb{R}^{n_v \times d_v}$ extracted by the vision encoder to pseudo-text embeddings $u^v = W_1 \sigma(W_2 z^v) \in \mathbb{R}^{n_v \times d_x}$ via a two-layer MLP. The **Design Motivation** is to align visual features with the LLM's text embedding space, enabling the subsequent fusion layer to effectively integrate cross-modal information.

2.  **Late Fusion Attention Layer (LFAL)**: An attention layer is inserted after the LLM outputs the final representation but before the prediction head, with $K=V=[u^v; z^x_{(<t)}]$ and $Q=z^x_{(<t)}$, allowing text tokens to attend to visual tokens in one go. The **Design Motivation** is to delay the injection of visual information until the final stage, allowing the LLM to focus on language processing throughout its computation and only access visual information when needed, thereby minimizing interference with linguistic capabilities.

3.  **Multi-image Reasoning and Confidence Weighting**: At inference time, $k$ images are generated, each producing a distribution $p_i$, alongside a text-only distribution $p_0$. Confidence weighting is performed using CLIP scores: $p_{\text{final}} = \sum_i f(\bar{x}_i, v_i) p_i + (1 - f(\bar{x}_i, v_i)) p_0$. Images with high alignment receive higher weights, while the model automatically falls back to the text-only LLM when alignment is low. The **Design Motivation** is that a single generated image may be noisy or biased; multiple images provide redundant visual evidence, and confidence weighting ensures that unreliable images do not harm the prediction.

### Loss & Training

The model is trained using the standard language modeling objective $\max_\theta \log P_\theta(x_{(t)} | x_{(<t)}, v)$. Training data includes real image-text pairs and text plus synthetically generated image pairs. Only the parameters of the VTP and LFAL are trainable; the LLM and vision encoder remain frozen. During inference, a distilled text-to-image generator is used for batch parallel sampling to minimize overhead.

## Key Experimental Results

### Main Results

| Model | Base | Visual Commonsense (VC) | Commonsense Reasoning (CR) | Reading Comp. (RC) | Average |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Llama3-8B | - | 52.0 | 72.0 | 57.9 | 60.6 |
| LaMI (Llama3-8B) | Llama3-8B | **55.0** | **72.9** | **58.0** | **62.0** |
| Llama3-8B-Instruct | - | 53.0 | 71.6 | 59.2 | 61.2 |
| Llava-Next (Llama3-8B-Inst.) | Llama3-8B-Inst. | 56.5 | 70.8 | 54.8 | 60.7 |
| LaMI (Llama3-8B-Inst.) | Llama3-8B-Inst. | 55.6 | **71.7** | **60.9** | **62.7** |

### Ablation Study

| Method | Memory Color | Color Terms | Object Shape | Relative Size |
| :--- | :--- | :--- | :--- | :--- |
| GPT-2 (Base) | 32.4 | 34.6 | 44.5 | 43.1 |
| Early Fusion | 49.1 | 45.3 | 40.3 | 70.1 |
| Early Fusion + Multi | 55.5 | 52.1 | 41.2 | 75.5 |
| Intermediate Fusion + Multi | 69.7 | 67.8 | 63.0 | 81.1 |
| **Late Fusion + Multi (Ours)** | **72.5** | **69.2** | **66.8** | **85.5** |

### Key Findings

*   **Late fusion consistently outperforms early and intermediate fusion**: Late Fusion achieved optimal performance across all tasks, with particularly significant advantages in shape-related tasks.
*   **Multi-image generation provides gains across all fusion strategies**: Improvements were especially notable for reasoning about color and relative size.
*   **LaMI enhances visual commonsense without harming, and even improving, text tasks**: This stands in sharp contrast to VLMs like InstructBLIP and Llava-Next.
*   Inference-time computation control experiments: While Best-of-N sampling improves commonsense reasoning, it cannot bridge the visual commonsense gap (VC: 47.8 vs. LaMI 50.1), confirming that LaMI's improvements stem from visual evidence rather than extra computation.
*   Performance saturates at image count $k \approx 6$, with significant gains achievable at $k=3$.

## Highlights & Insights

*   The design philosophy of **late fusion protecting language capability** is highly practical—the LLM remains frozen and vision only "lightly touches" at the final stage. This represents a minimally invasive multimodal enhancement paradigm for LLMs.
*   The **automatic degradation mechanism with CLIP confidence weighting** is cleverly designed: it automatically retreats to the text-only path when images are unreliable, preventing visual noise from damaging predictions.
*   The method is plug-and-play and can be applied to any newly released LLM without the need for expensive multimodal retraining.

## Limitations & Future Work

*   Reliance on the quality of the text-to-image generator; generated images may introduce out-of-distribution noise.
*   Generating multiple images at inference time increases latency and computational overhead (though it can be parallelized).
*   Visual commonsense performance is still slightly lower than fully trained VLMs (e.g., Llava-Next VC: 56.5 vs. LaMI 55.6), though it leads in overall average.
*   Validation was only performed on discriminative tasks (multiple-choice); the effect on open-ended generation tasks remains unknown.
*   Future work could explore more efficient ways to acquire visual evidence, such as retrieval instead of generation.

## Related Work & Insights

*   **VaLM Series** (VaLM, Z-LaVI, LiVE): LaMI's late fusion strategy uniformly addresses the language capability degradation found in early fusion schemes.
*   **CLIP as a Cross-modal Bridge**: CLIP scores are used to automatically evaluate image-text alignment and weight them without additional training.
*   Insight: Multimodal enhancement does not necessarily require deep fusion; shallow late fusion has a natural advantage in preserving unimodal capabilities.

## Rating

*   **Novelty**: ⭐⭐⭐⭐ The combination of late fusion and multi-image reasoning is not a brand-new concept, but its advantages are systematically verified, and the CLIP-weighted fallback mechanism is creative.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation ranging from small to large models and from BERT to LLaMA3, with complete ablation studies, though verification on even larger-scale models is missing.
*   **Writing Quality**: ⭐⭐⭐⭐ The structure is clear, and the motivation is well-articulated, although some notation usage is not entirely consistent.
*   **Value**: ⭐⭐⭐⭐ Provides a practical and lightweight solution for quickly adapting new LLMs to multimodal scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)
- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](../../CVPR2026/multimodal_vlm/multi-modal_image_fusion_via_intervention-stable_feature_learning.md)
- [\[ACL 2026\] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)
- [\[ACL 2026\] Leave My Images Alone: Preventing Multi-Modal Large Language Models from Analyzing Unauthorized Images](leave_my_images_alone_preventing_multi-modal_large_language_models_from_analyzin.md)
- [\[ACL 2026\] Jailbreaking Multimodal Large Language Models using Multi-Clip Video](jailbreaking_multimodal_large_language_models_using_multi-clip_video.md)

</div>

<!-- RELATED:END -->
