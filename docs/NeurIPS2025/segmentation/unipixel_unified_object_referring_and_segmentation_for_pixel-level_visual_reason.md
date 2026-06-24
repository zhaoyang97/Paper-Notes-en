---
title: >-
  [Paper Note] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning
description: >-
  [NeurIPS 2025][Segmentation][Multimodal Large Models] UniPixel proposes the first end-to-end large multimodal model that unifies object referring and segmentation, leveraging a novel Object Memory Bank design to transform sparse visual prompts into dense object mask features and inject them into the reasoning process. The model achieves state-of-the-art performance on 10 benchmarks and introduces PixelQA, a new task requiring simultaneous referring, segmentation…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Multimodal Large Models"
  - "Pixel-Level Reasoning"
  - "Unified Object Referring and Segmentation"
  - "Object Memory Bank"
  - "Video Understanding"
date: 2026-05-08
content_hash: 4ed6c103eba0eacb
---

# UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2509.18094](https://arxiv.org/abs/2509.18094)  
**Code**: [Project Page](https://polyu-chenlab.github.io/unipixel/)  
**Area**: Image Segmentation
**Keywords**: Multimodal Large Models, Pixel-Level Reasoning, Unified Object Referring and Segmentation, Object Memory Bank, Video Understanding

## TL;DR

UniPixel proposes the first end-to-end large multimodal model that unifies object referring and segmentation, leveraging a novel Object Memory Bank design to transform sparse visual prompts into dense object mask features and inject them into the reasoning process. The model achieves state-of-the-art performance on 10 benchmarks and introduces PixelQA, a new task requiring simultaneous referring, segmentation, and question answering.

## Background & Motivation

Large multimodal models (LMMs) excel at holistic image/video understanding but face two fundamental limitations in fine-grained pixel-level understanding:

**Limited interaction modality**: Users can only interact via text, lacking more intuitive communication means (e.g., clicking points or drawing boxes as references, or using masks to ground model responses).

**Coarse reasoning granularity**: Internal reasoning operates at the holistic level, directly perceiving all content rather than reasoning about specific objects or regions, making it difficult to capture fine-grained details.

Existing methods (e.g., LISA, VISA) have explored LMM-driven segmentation but suffer from a fundamental limitation: they can only perform referring or segmentation independently, relying on rigid input/output templates (e.g., LISA's "It's \<SEG\>."), and cannot flexibly understand user-referenced concepts while simultaneously generating mask-grounded responses. More critically, these methods fail to integrate fine-grained perceptual capabilities with general multimodal reasoning, leading to performance degradation on general understanding benchmarks.

UniPixel's core innovation lies in unifying the internal representations of referring and segmentation through the Object Memory Bank, enabling the model to dynamically segment key objects during reasoning, encode their features, and conduct subsequent reasoning grounded in object-level information.

## Method

### Overall Architecture

UniPixel is built upon the Qwen2.5-VL framework, incorporating an LLM backbone and a ViT visual encoder supporting dynamic resolution. Three key components are introduced: a Prompt Encoder (supporting point/box/mask visual prompts), an Object Memory Bank (storing and injecting object information), and a Mask Decoder (generating spatiotemporal masks based on SAM 2.1). The LLM vocabulary is extended with three special tokens: `<REF>`, `<MEM>`, and `<SEG>`.

### Key Designs

1. **Prompt Encoder**: Encodes each type of visual prompt into a single token fed into the LLM. For sparse prompts (points and boxes), 2D Fourier embeddings encode spatial coordinates combined with learnable type embeddings; 1D Fourier temporal encoding is innovatively added to represent frame indices, followed by a GELU→Linear projection into the LLM embedding space. For dense prompts (masks), masked pooling is applied directly on visual encoder outputs and mapped via an M→L projector. Design motivation: inspired by SAM but with two key differences — temporal information is incorporated and negative points are omitted.

2. **Object Memory Bank**: This is the core innovation — a hashmap keyed by object ID and valued by spatiotemporal masks, initialized empty per dialogue session and updated dynamically on demand. It comprises two operations: (a) **Memory Pre-filling**: triggered when a `<REF>` token is detected in the input; the model generates an object ID and a `<SEG>` token to predict spatiotemporal masks, which are then stored in the memory bank; (b) **Memory Injection**: stored object masks are downsampled via masked pooling, compressed per-frame into a single feature token, projected through a projector, and used to replace `<MEM>` tokens, injecting object-level information into subsequent reasoning. Design motivation: directly appending `<SEG>` after `<REF>` suffers from two problems — causal self-attention prevents the mask from accessing full context, degrading quality, and referring and segmentation cannot be decoupled during training.

3. **Mask Decoder**: Adopts SAM 2.1 to decouple discrete language modeling from continuous mask prediction. For each `<SEG>` token, the last-layer hidden state is extracted, projected via an L→M projector, downsampled, and reshaped into two tokens (to better preserve information during high-to-low dimensional channel-space downsampling). These tokens prompt SAM 2.1 to predict a mask on the first frame, which is then propagated to subsequent frames.

### Loss & Training

The total loss is a linear combination of language modeling loss and mask decoding loss:
- Language modeling: standard cross-entropy, weight 1
- Mask prediction: focal loss (weight 100) + dice loss (weight 5) + IoU prediction MAE (weight 5) + objectness cross-entropy (weight 5)

Three-stage progressive training:
1. **Stage 1**: Pre-train the sparse prompt encoder on 851K region description data.
2. **Stage 2**: Train the L→M projector on 87K referring segmentation data to align the LLM and mask decoder.
3. **Stage 3**: Unfreeze the M→L projector and mask decoder, apply LoRA to the visual encoder and LLM, and jointly train on ~2M multi-task samples.

## Key Experimental Results

### Main Results — Referring Video Object Segmentation (ReVOS)

| Method | Model Size | Overall $\mathcal{J}\&\mathcal{F}$ | Prev. SOTA | Gain |
|--------|-----------|--------------------------------------|------------|------|
| VISA | 13B | 50.9 | — | — |
| ViLLa | 6B | 57.0 | — | — |
| UniPixel | 3B | **62.1** | 57.0 | +5.1 |
| UniPixel | 7B | **63.7** | 57.0 | +6.7 |

### Ablation Study

| Configuration | $\mathcal{J}\&\mathcal{F}$ | Acc | Note |
|---------------|---------------------------|-----|------|
| Referring only | — | 64.6 | No segmentation |
| Segmentation only | 47.5 | — | No referring |
| Refer + Segment (w/o Memory) | 48.2 | 67.4 | Unified but no memory |
| Refer + Segment + Memory | **49.0** | **68.5** | Full UniPixel |
| ① Single-token referring | 46.8 | 64.5 | Minimal referring |
| ② \<REF\>\<SEG\> | 47.8 | 64.9 | With auxiliary segmentation |
| ③ + Pooling | 47.5 | 66.3 | With pooled features |
| ④ Object Memory Bank | **49.0** | **68.5** | Decoupled design is optimal |

### Key Findings

- **Mutual enhancement between referring and segmentation**: Joint training of referring and segmentation improves both tasks (segmentation: 47.5→48.2; referring QA: 64.6→67.4).
- **3B model surpasses 7–13B counterparts**: On ReVOS, UniPixel-3B outperforms all 7B–13B competitors, demonstrating that a unified design is more effective than simply scaling model size.
- **Temporal encoding is critical**: Removing temporal encoding from the prompt encoder drops $\mathcal{J}\&\mathcal{F}$ from 49.0 to 44.3.
- **Large margin on Ref-SAV**: On the challenging long-video dataset Ref-SAV, UniPixel-3B achieves 67.2 $\mathcal{J}\&\mathcal{F}$, far exceeding Sa2VA-8B's 41.3 (without fine-tuning).

## Highlights & Insights

- **First end-to-end unified referring + segmentation framework**: The elegant Object Memory Bank design eliminates the need for external frame samplers, mask generators, or object trackers.
- **New PixelQA task**: Introduces a new paradigm requiring simultaneous referring, segmentation, and QA, bridging the gap between pixel-level perception and language reasoning.
- **Object-level test-time scaling**: The approach can be interpreted as an object-centric test-time scaling strategy — segmenting key objects first, then encoding them to assist reasoning.
- **Decoupled memory bank design**: Fundamentally resolves the problem that `<SEG>` tokens cannot access full context under causal self-attention constraints.

## Limitations & Future Work

- The 7B version yields lower segmentation quality than the 3B on PixelQA ($\mathcal{J}\&\mathcal{F}$: 44.6 vs. 60.9), suggesting potential degradation of segmentation capability at larger scale.
- The reasoning segmentation dataset (ReasonSeg) contains only 239 samples, making it susceptible to being dominated by large-scale data during joint training.
- Mask propagation relies on SAM 2.1, and robustness to extreme motion or occlusion is bounded by this external module.

## Related Work & Insights

- Compared to LISA (which pioneered the LMM-driven segmentation paradigm), UniPixel unifies referring and segmentation, avoiding rigid templates.
- Compared to Sa2VA (which also uses a SAM2 decoder), UniPixel's Object Memory Bank provides stronger object-aware capabilities.
- The pre-filling–injection mechanism of the Object Memory Bank is generalizable to other multimodal tasks requiring object-level reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The Object Memory Bank design is highly original; the PixelQA task definition is forward-looking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across 10 benchmarks and 9 tasks, with detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with a problem-driven narrative style.
- **Value**: ⭐⭐⭐⭐⭐ The unified framework has broad applicability; PixelQA opens a new research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Pixel-Level Reasoning Segmentation via Multi-turn Conversations](../../ACL2025/segmentation/pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](../../ICCV2025/segmentation/towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)
- [\[CVPR 2025\] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](../../CVPR2025/segmentation/dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)
- [\[NeurIPS 2025\] Robust Egocentric Referring Video Object Segmentation via Dual-Modal Causal Intervention](robust_egocentric_referring_video_object_segmentation_via_dual-modal_causal_inte.md)
- [\[ICCV 2025\] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations](../../ICCV2025/segmentation/referdino_referring_video_object_segmentation_with_visual_grounding_foundations.md)

</div>

<!-- RELATED:END -->
