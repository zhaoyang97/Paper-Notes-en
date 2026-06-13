---
title: >-
  [Paper Note] MSTAR: Box-Free Multi-Query Scene Text Retrieval with Attention Recycling
description: >-
  [NeurIPS 2025][Object Detection][Scene text retrieval] This paper presents MSTAR, the first multi-query scene text retrieval method that requires no bounding box annotations. Through Progressive Vision Embedding (PVE)…
tags:
  - "NeurIPS 2025"
  - "Object Detection"
  - "Scene text retrieval"
  - "box-free annotation"
  - "multi-query retrieval"
  - "attention recycling"
  - "vision-language model"
date: 2026-05-08
content_hash: 423706639252a9e3
---

# MSTAR: Box-Free Multi-Query Scene Text Retrieval with Attention Recycling

**Conference**: NeurIPS 2025
**arXiv**: [2506.10609](https://arxiv.org/abs/2506.10609)  
**Code**: [GitHub](https://github.com/yingift/MSTAR)  
**Area**: Object Detection / Scene Text Retrieval
**Keywords**: Scene text retrieval, box-free annotation, multi-query retrieval, attention recycling, vision-language model

## TL;DR

This paper presents MSTAR, the first multi-query scene text retrieval method that requires no bounding box annotations. Through Progressive Vision Embedding (PVE), MSTAR progressively shifts attention from salient to non-salient regions. Combined with style-aware instructions and a Multi-Instance Matching (MIM) module, it achieves unified retrieval across four query types—word, phrase, combined, and semantic—and introduces MQTR, the first multi-query text retrieval benchmark.

## Background & Motivation

**Background**: Scene Text Retrieval aims to search for images containing relevant text from an image collection given a query, with broad applications in signature retrieval, keyframe extraction, and related tasks. Significant progress has been made in recent years with the aid of accurate text localization.

**Limitations of Prior Work**: (1) Existing methods typically require expensive bounding box annotations (at word level, text-line level, etc.) for training; (2) most methods adopt customized retrieval strategies that struggle to handle multiple query types (word, phrase, combined, and semantic) in a unified manner.

**Key Challenge**: Vision-language models (VLMs) demonstrate strong performance under large-scale box-free pretraining, yet tend to focus on salient visual concepts while neglecting fine-grained scene text instances (e.g., small text within images).

**Goal**: To achieve scene text retrieval without bounding box supervision while unifying multiple query types.

**Key Insight**: Leveraging the attention mechanism of VLMs, the model progressively transfers focus from high-attention regions to overlooked regions via Attention Recycling.

**Core Idea**: Progressively masking high-attention regions forces the model to attend to non-salient text, while style instructions unify multiple query types.

## Method

### Overall Architecture

MSTAR is built upon BLIP-2 and consists of four core components: a visual encoder $\phi$ (SigLIP ViT-Base-512), Progressive Vision Embedding (PVE), a multimodal encoder $\psi$ (BLIP-2), and a Multi-Instance Matching module (MIM). Training employs joint optimization with contrastive learning and image-text matching losses. At inference, candidate images are first ranked by cosine similarity, and the top-$K$ images are subsequently re-ranked.

### Key Designs

1. **Progressive Vision Embedding (PVE)**:

    - **Function**: Progressively extracts visual embeddings, shifting attention from salient regions to non-salient fine-grained text regions.
    - **Design Motivation**: VLMs tend to focus on salient visual elements (e.g., red circles) while overlooking small scene text, leading to high miss rates in small-text retrieval.
    - **Mechanism**:
        - The visual encoder extracts initial image features $f_0$; the multimodal encoder generates initial visual embeddings $E_V^0$.
        - The Salient Attention Shift (SAS) module computes an attention map $C_{t-1}$ from cross-attention weights, which is binarized and inverted to obtain a mask $M_{t-1} = 1 - \sigma(C_{t-1})$.
        - The masked attention layer forces self-attention to reduce weights on already-attended regions, redirecting focus to neglected areas.
        - After $T$ iterations, all embeddings are concatenated: $E_V \in \mathbb{R}^{(T+1)Q \times d}$.
    - **Novelty**: Unlike masking methods that require ground-truth supervision, SAS derives masks entirely from the model's own cross-attention, without any external annotation.

2. **Style-Aware Instruction**:

    - **Function**: Guides the multimodal encoder to distinguish different query styles via short textual instructions.
    - **Design Motivation**: When training over multiple query types (word/phrase/combined/semantic) jointly, differences in format and semantics cause inconsistent representations.
    - **Mechanism**: $E_T = \psi(\text{Concat}[T_i, T_Q])$, where $T_i$ is the style instruction and $T_Q$ is the text query. To accelerate training, all queries for the same image are encoded together.
    - **Novelty**: Eliminates the need for separate models or branches for each query type.

3. **Multi-Instance Matching (MIM)**:

    - **Function**: Explicitly establishes one-to-one correspondence between visual and text embeddings.
    - **Design Motivation**: Conventional embedding aggregation and late interaction strategies require extensive training to achieve vision-language alignment.
    - **Mechanism**: Two parallel branches:
        - Word branch: employs the Hungarian matching algorithm to establish one-to-one correspondence between $E_w$ and $E_V$.
        - Multi-word branch: aggregates visual features under text constraints via a lightweight cross-attention layer: $E_{vt} = \mathcal{F}(\mathcal{C}(\mathcal{Q}=E_w, \mathcal{K,V}=E_V))$.

### Loss & Training

- Contrastive loss: $\mathcal{L}_c = \alpha \mathcal{L}_{t2v} + \mathcal{L}_{v2t}$, with $\alpha=1.5$
- Image-text matching loss: $\mathcal{L}_m$ (cross-entropy)
- Total loss: $\mathcal{L} = \mathcal{L}_c + \mathcal{L}_m$
- Multi-stage training with progressively increasing resolutions: $512 \to 640 \to 800$
- Re-ranking applied to the top 2% of retrieved images

## Key Experimental Results

### Main Results

**MQTR Multi-Query Retrieval MAP%:**

| Method | Type | AVG | Word | Phrase | Combined | Semantic |
|--------|------|-----|------|--------|----------|----------|
| TDSL | Box-Based | 58.25 | 69.11 | 40.83 | 72.71 | 50.36 |
| TG-Bridge | Box-Based | 54.09 | 69.89 | 30.21 | 75.53 | 40.73 |
| BLIP-2 (FT) | Box-Free | 58.11 | 58.09 | 42.23 | 60.84 | 71.24 |
| **MSTAR** | **Box-Free** | **66.78** | **73.27** | **44.22** | **74.48** | **75.14** |

MSTAR surpasses the previous best method by 8.53% in average MAP.

**Word-Level Retrieval MAP% on 6 Public Datasets:**

| Method | SVT | STR | CTR | Total-Text | CTW | IC15 | Avg |
|--------|-----|-----|-----|------------|-----|------|-----|
| TDSL | 89.38 | 77.09 | 66.45 | 74.75 | 59.34 | 77.67 | 74.16 |
| FDP-RN50×16 | 89.63 | **89.46** | - | 79.18 | - | - | - |
| MSTAR | **91.31** | 86.25 | 60.13 | **85.55** | **90.87** | **81.21** | **82.56** |
| MSTAR (+re-rank) | 91.11 | 86.14 | **65.25** | **86.96** | **92.95** | **82.69** | **84.18** |

Without bounding box annotations, MSTAR outperforms TDSL by 8.40% in average MAP and surpasses FDP by 6.37% on Total-Text.

### Ablation Study

| Ins | MIM | PVE | CTR | SVT | STR | Total-Text | CTW | IC15 | MQTR |
|-----|-----|-----|-----|-----|-----|------------|-----|------|------|
| ✗ | ✗ | ✗ | 52.87 | 90.07 | 81.57 | 82.32 | 87.28 | 76.71 | 65.79 |
| ✓ | ✗ | ✗ | 54.65 | 90.70 | 82.81 | 83.19 | 88.96 | 77.15 | 66.15 |
| ✓ | ✓ | ✗ | 55.77 | 91.02 | 85.00 | 84.01 | 90.31 | 79.23 | 65.69 |
| ✓ | ✓ | ✓ | **60.13** | **91.31** | **86.25** | **85.55** | **90.87** | **81.21** | **66.78** |

- PVE yields the most significant gains on small-text scenarios: +4.36% on CTR, +1.98% on IC15.
- MIM substantially improves word-level retrieval: +2.19% on STR, +1.35% on CTW.

### Key Findings

- The box-free MSTAR achieves competitive performance against fully supervised text localization methods (TG-Bridge) while running at twice the inference speed (14.2 FPS vs. 6.7 FPS).
- The attention recycling mechanism in PVE effectively addresses the tendency of VLMs to ignore small text.
- MSTAR achieves 95.71% MAP on the phrase retrieval dataset.

## Highlights & Insights

- **Paradigm Innovation**: The first box-free scene text retrieval method, demonstrating that expensive bounding box annotations are unnecessary to match or surpass box-based approaches.
- **Unified Multi-Query Handling**: Style-aware instructions enable unified processing of four query types without training separate models for each.
- **Elegant Attention Recycling Design**: Leverages the model's own attention maps to guide attention shifting without requiring additional supervision.
- **MQTR Benchmark Contribution**: The first multi-query scene text retrieval benchmark, filling an important evaluation gap in the community.
- Practical inference advantage: no text detection module required.

## Limitations & Future Work

- MSTAR still underperforms box-based method TDSL on the CTR dataset (extremely small text), an inherent limitation of box-free approaches.
- Hungarian matching introduces a slight performance drop on MQTR for complex combined queries.
- Support and evaluation for Chinese scene text remain insufficient.
- The number of PVE iteration steps $T$ is a hyperparameter that may require tuning across different scenarios.

## Related Work & Insights

- **BLIP-2 (li2023blip)**: Foundational architecture providing the pretrained multimodal encoder.
- **SigLIP (zhai2023sigmoid)**: Visual encoder initialization.
- **FDP (zeng2024focus)**: Previous SOTA, leveraging CLIP with box supervision for text region localization.
- **TDSL (wang2021scene)**: Classic end-to-end scene text retrieval method.
- Insight: The attention bias of VLMs is a systematic issue; attention recycling represents a generalizable solution that could be extended to other fine-grained retrieval tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ First box-free multi-query text retrieval; PVE attention recycling is an elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on 7 public datasets plus the self-constructed MQTR benchmark with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with compelling motivation.
- Value: ⭐⭐⭐⭐ Significantly reduces annotation cost; the MQTR benchmark is a valuable community contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)
- [\[AAAI 2026\] REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion](../../AAAI2026/object_detection/rexo_indoor_multi-view_radar_object_detection_via_3d_bounding_box_diffusion.md)
- [\[NeurIPS 2025\] Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](video-rag_visually-aligned_retrieval-augmented_long_video_comprehension.md)
- [\[CVPR 2026\] MRD: Multi-resolution Retrieval-Detection Fusion for High-Resolution Image Understanding](../../CVPR2026/object_detection/mrd_multi-resolution_retrieval-detection_fusion_for_high-resolution_image_unders.md)
- [\[ICCV 2025\] YOLO-Count: Differentiable Object Counting for Text-to-Image Generation](../../ICCV2025/object_detection/yolo-count_differentiable_object_counting_for_text-to-image_generation.md)

</div>

<!-- RELATED:END -->
