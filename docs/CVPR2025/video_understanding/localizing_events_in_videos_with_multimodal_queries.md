---
title: >-
  [Paper Note] Localizing Events in Videos with Multimodal Queries
description: >-
  [CVPR 2025][Video Understanding][Multimodal Queries] This work proposes the ICQ benchmark and the ICQ-Highlight dataset, representing the first systematic study of replacing text-only queries with multimodal queries (image + text) for video event localization. It also designs three query adaptation methods and the SUIT proxy fine-tuning strategy.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Multimodal Queries"
  - "Video Event Localization"
  - "Benchmarking"
  - "Query Adaptation"
  - "Spatio-Temporal Video Grounding"
date: 2026-05-08
content_hash: 37bb0ed5288f2bbb
---

# Localizing Events in Videos with Multimodal Queries

**Conference**: CVPR 2025  
**arXiv**: [2406.10079](https://arxiv.org/abs/2406.10079)  
**Code**: [https://icq-benchmark.github.io/](https://icq-benchmark.github.io/)  
**Area**: Video Understanding  
**Keywords**: Multimodal Queries, Video Event Localization, Benchmarking, Query Adaptation, Spatio-Temporal Video Grounding

## TL;DR

This work proposes the ICQ benchmark and the ICQ-Highlight dataset, representing the first systematic study of replacing text-only queries with multimodal queries (image + text) for video event localization. It also designs three query adaptation methods and the SUIT proxy fine-tuning strategy.

## Background & Motivation

Video event localization (including moment retrieval, highlight detection, and temporal grounding) has long relied on natural language queries (NLQ), which exhibit significant limitations in practical applications:

1. **Ambiguity of Text Queries**: Users tend to write short queries like "swimming," which can refer to various modes such as freestyle or butterfly, making NLQ difficult to be descriptive enough.
2. **Difficulty in Expressing Non-Verbal Concepts**: Unfamiliar objects or abstract aesthetic concepts (e.g., geometric styles) are challenging to describe accurately using text.
3. **Language Barriers**: For illiterate users or cross-lingual scenarios, image queries are more intuitive.
4. **Inability of Existing Methods to Handle Multimodal Queries Directly**: The input encoders of all NLQ-based models only accept text.

Therefore, **Multimodal Query (MQ) = Reference Image + Modifying Text** presents a more flexible and general paradigm, but faces two major challenges: visual queries may introduce irrelevant details, and reference images show a distribution shift compared to the target videos.

## Method

### Overall Architecture

ICQ comprises three major contributions:

- **ICQ-Highlight Dataset**: Built upon the QVHighlights validation set, it constructs multimodal queries (4 reference image styles $\times$ modifying text) with human annotations for each original text query.
- **Three Multimodal Query Adaptation (MQA) Methods**: Converts MQ into inputs compatible with existing NLQ models.
- **SUIT Proxy Fine-Tuning Strategy**: Fine-tunes MLLMs using pseudo MQs to improve adaptation quality.

### Key Designs

1. **Multimodal Query Definition and Data Construction**:
    - Reference image $v_{ref}$: Generated in 4 styles via DALL-E-2 and Stable Diffusion—scribble, cartoon, cinematic, and realistic.
    - Modifying text $t_{ref}$: Divided into 5 categories—objects, actions, relations, attributes, and environments—to provide complementary or corrective information.
    - Human annotation: Each query is annotated and verified by different annotators to ensure consistency.
    - Task definition: Given $q_m = (v_{ref}, t_{ref})$, predict all relevant segments $[\tau_{start}, \tau_{end}]$ in the video.

2. **Three MQA Adaptation Methods**:
    - **MQ-Cap (Language-Space)**: Generates a description for the reference image using an MLLM (LLaVA), then integrates the modifying text using an LLM (GPT-3.5) to produce the NLQ input. This decoupled two-step pipeline is more controllable.
    - **MQ-Sum (Language-Space)**: Merges the reference image and modifying text into a text summary in a single step using an MLLM. It is more concise but less controllable and sensitive to prompt design.
    - **VQ-Enc (Embedding-Space)**: Directly encodes the reference image into a query embedding $e_q$ using the CLIP visual encoder, utilizing the shared embedding space of CLIP. It does not utilize the modifying text.

3. **SUIT Proxy Fine-Tuning Strategy**: Addresses the lack of training data for MQ:
    - **Pseudo MQ Generation**: Starting from image-text pairs in Flickr30K and COCO, GPT-3.5 is used to decompose the caption into a "tampered caption" and "modifying text." The original image combined with the modifying text forms a pseudo MQ.
    - **Proxy Fine-Tuning**: Fine-tunes LLaVA on the task of mapping pseudo MQ to the tampered caption (using LoRA with rank 32 and alpha 64).
    - **Transfer**: The fine-tuned MLLM is directly applied to the ICQ-Highlight evaluation.
    - Formulates 89,420 training instances with a learning rate of $2 \times 10^{-4}$.

### Loss & Training

- SUIT utilizes next-token prediction loss with LoRA parameter-efficient fine-tuning (PEFT).
- After adaptation, pre-trained checkpoints of various backbones are used directly without modifying the backbone models.
- Twelve backbones are evaluated: nine specialized models (e.g., Moment-DETR, QD-DETR) and three LLM-based models (SeViLA, TimeChat, VTimeLLM).

## Key Experimental Results

### Main Results

| Method | Model | R1@0.5 (realistic) | R1@0.7 (realistic) | Description |
|------|------|---------------------|---------------------|------|
| VQ-Enc | CG-DETR | 24.74 | 14.23 | Reference image only |
| MQ-Cap | TR-DETR | **56.94** | **41.99** | Best training-free method |
| MQ-Cap | CG-DETR | 56.72 | 41.79 | Runner-up |
| MQ-Sum | TR-DETR | 52.87 | 36.77 | Inferior to MQ-Cap |
| MQ-Sum+SUIT | TR-DETR | **57.39** | **42.64** | Overall best |
| MQ-Sum+SUIT | CG-DETR | 55.47 | 40.17 | |
| MQ-Cap | SeViLA | 26.83 | 16.83 | Poor performance of LLM model |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| No modifying text vs with | Decrease of 2.8%-14% | Modifying text helps precise localization |
| scribble vs realistic | Difference <3% | Even highly simplified scribbles are effective |
| Synthetic vs retrieved real images | Similar performance | Generation artifacts do not affect conclusions |
| MQ-Cap vs MQ-Sum | MQ-Cap +3.6% avg | Captioning is more stable |
| MQ-Sum vs MQ-Sum+SUIT | SUIT +4.3%-9.7% | Fine-tuning significantly boosts performance and stability |
| t-SNE visualization | SUIT output distribution closer to NLQ | Explains why SUIT is effective |

### Key Findings

- **MQ Can Effectively Localize Video Events**: Consistent performance across different styles for various adaptation methods validates the feasibility of MQ.
- **MQ-Cap > MQ-Sum > VQ-Enc**: Decoupled captioning + modification is more controllable than single-step summarization, while pure visual encoding performs the worst.
- **SUIT is the Best Strategy**: It brings non-marginal gains (4.3%-9.7%) with more stable performance (smaller standard deviation).
- **Scribble Images Are Also Effective**: The scribble style performs only slightly lower than realistic/cinematic styles, demonstrating the potential of minimal visual queries.
- **Specialized Models >> LLM-based Models**: SeViLA, TimeChat, and VTimeLLM are significantly weaker than TR-DETR, CG-DETR, and UVCOM across all adaptation methods.
- **Consistent Ranking of Different Backbones across Adaptation Methods**: Indicates that the capacity of the backbone is the decisive factor.
- The performance gap between MQ and NLQ remains significant, indicating semantic loss during cross-modal translation in multimodal queries.

## Highlights & Insights

- **Pioneering definition of "Spatio-Temporal Video Event Localization with Multimodal Queries"**, filling the gap in NLQ-only research.
- **Design of four reference image styles** (ranging from scribble to realistic) covers real-world scenarios from the simplest to the highly detailed.
- **SUIT's pseudo MQ generation pipeline** cleverly exploits existing image-text data, bypassing expensive manual MQ annotations.
- **t-SNE visualization** intuitively illustrates how SUIT alleviates distribution shift.
- **Large-scale systematic benchmark** (12 models $\times$ 4 adaptation methods $\times$ 4 styles) provides a comprehensive reference for future research.

## Limitations & Future Work

- ICQ-Highlight is based on the QVHighlights validation set, which has limited scale, and reference images are synthesized rather than from actual users.
- The distribution of modifying text categories may be imbalanced, with fewer samples for certain categories (e.g., "relations").
- All adaptation methods are pipeline-based; end-to-end MQ model architectures remain unexplored.
- The poor performance of LLM-based models might stem from their inherently sub-optimal performance on NLQ benchmarks rather than constraints of MQ.
- Real-time user interaction scenarios (e.g., searching while watching) are not considered.

## Related Work & Insights

- Shares similarities with the Composed Image Retrieval (CIR) task, but CIR focuses on instance-level matching while ICQ requires dense temporal processing, posing higher complexity.
- The proxy fine-tuning concept of SUIT resembles knowledge distillation and can be extended to other adaptation scenarios deficient in training data.
- The decoupled step-by-step strategy of MQ-Cap (captioning followed by modification) offers reference value for other VLM adaptation tasks.
- The finding that scribbles are effective inspires possibilities for zero-shot or few-shot video search.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Brand-new task definition + systematic benchmark + innovative adaptation strategies
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 models $\times$ 4 methods $\times$ 4 styles, thorough ablation
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich charts, slightly wordy
- Value: ⭐⭐⭐⭐⭐ Establishes a new paradigm with a broad target audience, offering long-term dataset value

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] BlinkTrack: Feature Tracking over 80 FPS via Events and Images](../../ICCV2025/video_understanding/blinktrack_feature_tracking_over_80_fps_via_events_and_images.md)
- [\[CVPR 2025\] VideoGEM: Training-Free Action Grounding in Videos](videogem_training-free_action_grounding_in_videos.md)
- [\[CVPR 2025\] DPU: Dynamic Prototype Updating for Multimodal Out-of-Distribution Detection](dpu_dynamic_prototype_updating_for_multimodal_out-of-distribution_detection.md)
- [\[CVPR 2025\] DivPrune: Diversity-Based Visual Token Pruning for Large Multimodal Models](divprune_diversity-based_visual_token_pruning_for_large_multimodal_models.md)
- [\[CVPR 2025\] ReWind: Understanding Long Videos with Instructed Learnable Memory](rewind_understanding_long_videos_with_instructed_learnable_memory.md)

</div>

<!-- RELATED:END -->
