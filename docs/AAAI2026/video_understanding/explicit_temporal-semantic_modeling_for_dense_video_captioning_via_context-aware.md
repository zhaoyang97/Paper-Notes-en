---
title: >-
  [Paper Note] Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction
description: >-
  [AAAI 2026][Video Understanding][Dense Video Captioning] This paper proposes the CACMI framework, which addresses two fundamental limitations in dense video captioning (insufficient temporal modeling and modality gap) th…
tags:
  - "AAAI 2026"
  - "Video Understanding"
  - "Dense Video Captioning"
  - "Cross-modal Retrieval"
  - "Temporal Clustering"
  - "Feature Enhancement"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: 61f3d093c75f2c03
---

# Explicit Temporal-Semantic Modeling for Dense Video Captioning via Context-Aware Cross-Modal Interaction

**Conference**: AAAI 2026
**arXiv**: [2511.10134](https://arxiv.org/abs/2511.10134)  
**Code**: None  
**Area**: Video Understanding
**Keywords**: Dense Video Captioning, Cross-modal Retrieval, Temporal Clustering, Feature Enhancement, Retrieval-Augmented Generation

## TL;DR
This paper proposes the CACMI framework, which addresses two fundamental limitations in dense video captioning (insufficient temporal modeling and modality gap) through explicit temporal-semantic modeling. It employs Cross-modal Frame Aggregation (CFA) to extract temporally coherent event semantics, and Context-aware Feature Enhancement (CFE) to bridge the visual-textual modality gap, achieving state-of-the-art performance on ActivityNet Captions and YouCook2.

## Background & Motivation
Dense Video Captioning (DVC) requires simultaneously localizing and describing all salient events with precise temporal boundaries in untrimmed videos. Recent retrieval-augmented generation (RAG)-based methods (e.g., CM2) have begun incorporating external semantic knowledge to enhance understanding and generation capabilities.

**Limitations of Prior Work**: Existing memory-based methods rely on implicit RAG frameworks that use manually designed fixed windows for cross-modal retrieval, leading to two fundamental limitations:

**Insufficient Temporal Modeling**: Fixed-window visual features focus only on local segments, resulting in discontinuous semantic retrieval that fails to capture temporal coherence across event sequences.

**Modality Gap**: Retrieved semantic features are fused with visual representations via simple operations (concatenation or basic attention), which is insufficient to bridge the inherent gap between visual and textual modalities.

**Key Challenge**: Effective retrieval-augmented DVC requires exploiting the temporal structure and rich semantic information inherent in video data, yet current methods naively concatenate frame-level or fragmented textual information, neglecting temporal coherence.

**Key Insight**: Adjacent frames share similar visual and temporal contexts and typically represent the same semantic event. Based on this observation, the paper introduces explicit temporal-semantic modeling via pseudo-events, endowing retrieved textual semantics with temporal properties.

## Method

### Overall Architecture
CACMI follows a RAG paradigm: a CLIP image encoder extracts frame-level features → the CFA module aggregates temporally coherent frames and retrieves event-aligned text → the CFE module fuses visual and textual features → a Deformable Transformer with multi-task heads produces event localization and captioning outputs.

### Key Designs

1. **Cross-modal Frame Aggregation (CFA)**:

    - **Event Context Clustering**:
        - Function: Aggregates frame-level visual features into temporally coherent pseudo-event representations.
        - Mechanism: Applies agglomerative clustering (Euclidean distance + Ward linkage) to CLIP frame features, augmented with a temporal aggregation constraint (the temporal gap between any two frames within the same cluster must not exceed $t_{\max}$), ensuring that semantically similar and temporally contiguous frames are grouped together.
        - Design Motivation: Agglomerative clustering assumes no fixed cluster shape, making it well-suited for discovering flexible patterns in feature space; the temporal constraint enforces event-level temporal coherence.
        - Output: $c$ cluster-level feature vectors $F^c$, each representing a pseudo-event, computed via boundary-enhanced weighted averaging (inverted bell-shaped weights that assign higher weights to boundary frames).
    - **Event Semantic Retrieval**:
        - Function: Retrieves the most relevant textual descriptions from a sentence pool for each pseudo-event.
        - Mechanism: A CLIP text encoder preprocesses the sentence pool; cosine similarity matrices between pseudo-event features and all text features are computed; top-$k$ candidates per pseudo-event are retrieved and average-pooled.
        - Design Motivation: The core innovation lies in performing retrieval at the event granularity rather than at the frame or fixed-window level, preserving the integrity of temporal structure.

2. **Context-aware Feature Enhancement (CFE)**:

    - Function: Fine-grained cross-modal fusion that uses textual queries to guide visual feature enhancement.
    - Mechanism: Computes a similarity matrix $M$ between frame-level visual features $F^v$ and event-level textual queries $F^q$; applies dual attention (column-wise and row-wise softmax) to obtain cross-attention features $F^{v'}$ and $F^{q'}$; concatenates these with the original features and projects them; finally incorporates a global text vector via 1D convolution for fusion.
    - Design Motivation: CM2 uses shared self-attention weights for feature enhancement, and this parameter-sharing scheme is insufficient to bridge the semantic gap between modalities. Query-guided fusion selectively suppresses irrelevant visual elements and enhances semantically aligned regions.

3. **Multi-task Prediction Heads**:

    - **Localization Head**: An MLP that regresses event centers and temporal spans.
    - **Captioning Head**: An LSTM with deformable soft attention for word-by-word caption generation.
    - **Event Counter**: Max-pooling + FC layers to predict the number of events in the video.
    - Hungarian algorithm is used for matching predictions to ground truth.

### Loss & Training
- Matching loss: $L_{\text{match}} = L_{\text{cls}} + \alpha \cdot L_{\text{loc}}$ (focal classification loss + generalized IoU loss)
- Total loss: $L = \alpha_{\text{cls}} \cdot L_{\text{cls}} + \alpha_{\text{loc}} \cdot L_{\text{loc}} + \alpha_{\text{count}} \cdot L_{\text{count}} + \alpha_{\text{cap}} \cdot L_{\text{cap}}$
- Frame sampling: 1 FPS; 100 frames fixed for ActivityNet, 200 for YouCook2.
- Event queries: 10 for ActivityNet, 100 for YouCook2.
- Cluster count: 10 for ActivityNet, 20 for YouCook2.
- Retrieval top-$k$ = 40.

## Key Experimental Results

### Main Results (Captioning Performance)
**ActivityNet Captions (comparison with non-pretrained methods)**:

| Method | BLEU4↑ | METEOR↑ | CIDEr↑ | SODA_c↑ |
|--------|--------|---------|--------|---------|
| PDVC (ICCV'21) | 2.21 | 8.06 | 29.97 | 5.92 |
| CM2 (CVPR'24) | 2.38 | 8.55 | 33.01 | 6.18 |
| E2DVC (CVPR'25) | 2.43 | 8.57 | 33.63 | 6.13 |
| **CACMI (Ours)** | **2.44** | **8.68** | **33.80** | **6.39** |

**YouCook2**:

| Method | BLEU4↑ | METEOR↑ | CIDEr↑ | SODA_c↑ |
|--------|--------|---------|--------|---------|
| PDVC | 1.40 | 5.56 | 29.69 | 4.92 |
| CM2 | 1.63 | 6.08 | 31.66 | 5.34 |
| **CACMI (Ours)** | **1.70** | **6.21** | **34.83** | **5.57** |

### Event Localization Performance

| Method | ActivityNet F1↑ | YouCook2 F1↑ |
|--------|----------------|--------------|
| PDVC | 54.78 | 26.81 |
| CM2 | 55.21 | 28.43 |
| E2DVC | 56.42 | 28.87 |
| **CACMI (Ours)** | **57.10** | **29.34** |

### Ablation Study

| CFA | CFE | CIDEr | SODA_c | F1 |
|-----|-----|-------|--------|----|
| ✗ | ✗ | 33.01 | 6.18 | 55.21 |
| ✓ | ✗ | 33.62 | 6.26 | 56.07 |
| ✗ | ✓ | 33.48 | 6.31 | 56.95 |
| ✓ | ✓ | **33.80** | **6.39** | **57.10** |

| Cluster Count | CIDEr | F1 | Note |
|---------------|-------|-----|------|
| 3 | 32.84 | 54.91 | Too coarse |
| 10 | **33.80** | **57.10** | Optimal |
| 15 | 32.98 | 55.15 | Over-segmented |

| Top-$k$ | CIDEr | F1 | Note |
|---------|-------|-----|------|
| 10 | 32.20 | 55.95 | Insufficient semantic diversity |
| 40 | **33.80** | **57.10** | Optimal balance |
| 80 | 32.57 | 56.15 | Redundant information dilution |

### Key Findings
- **CFE contributes more to localization**: CFE alone improves F1 from 55.21 to 56.95 (+1.74), while CFA alone yields +0.86, indicating that cross-modal fusion is critical for temporal boundary prediction.
- **CFA contributes more to captioning quality**: CFA improves CIDEr by 0.61 vs. 0.47 for CFE alone, demonstrating that event-level semantic retrieval enriches caption content.
- **Complementary effect of both modules**: The combination achieves the best performance across all metrics.
- **Most pronounced gains on SODA_c**: The proposed method surpasses all baselines most notably on SODA_c, which evaluates narrative coherence, confirming that explicit temporal modeling effectively captures inter-event temporal dependencies.

## Highlights & Insights
- **Explicit vs. implicit temporal modeling**: Discovering natural event boundaries via clustering, rather than manually designing fixed windows, better reflects the intrinsic structure of videos.
- **Boundary-weighted event representation**: The inverted bell-shaped weights assign higher importance to boundary frames, facilitating precise temporal localization.
- **Effectiveness of query-guided fusion**: More effectively bridges the modality gap compared to CM2's shared self-attention mechanism.
- **No large-scale pretraining required**: Surpasses certain pretrained methods without additional video pretraining data.

## Limitations & Future Work
- The number of clusters is a hyperparameter that may require adaptive tuning for videos of varying lengths and complexity.
- The construction and quality of the sentence pool directly affect retrieval performance, yet this aspect is not thoroughly discussed in the paper.
- A performance gap relative to the pretrained Vid2Seq remains on YouCook2, likely due to limited domain coverage of the training videos.
- The captioning head still employs an LSTM; stronger generative models (e.g., LLM decoders) have not been explored.
- Euclidean distance used for clustering may not be the optimal metric in the high-dimensional CLIP feature space.
- Dynamic top-$k$ or adaptive retrieval strategies have not been investigated.

## Related Work & Insights
- **CM2 as a pioneering work**: CM2 first introduced a memory retrieval mechanism into DVC; the present work improves upon it in retrieval granularity and fusion design.
- **Adapting the RAG paradigm to video**: Extending text-retrieval-augmented generation from NLP to video understanding requires preserving temporal structure as a key consideration.
- **Influence of PDVC's design**: The parallel decoding structure allows localization and captioning subtasks to share intermediate representations.
- **Insight**: In any RAG system involving temporal data, performing retrieval at the granularity of semantically coherent segments—rather than fixed windows—may be a superior strategy.

## Rating
- Novelty: ⭐⭐⭐⭐ (Clear motivation for explicit temporal-semantic modeling; theoretically grounded CFA+CFE design)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive evaluation on two datasets, thorough ablation studies, convincing visualizations)
- Writing Quality: ⭐⭐⭐⭐ (Well-structured, sufficiently motivated, mathematically rigorous)
- Value: ⭐⭐⭐⭐ (Provides new state-of-the-art results and a meaningful methodological contribution to DVC)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](../../CVPR2026/video_understanding/cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [\[CVPR 2026\] SAIL: Similarity-Aware Guidance and Inter-Caption Augmentation-based Learning for Weakly-Supervised Dense Video Captioning](../../CVPR2026/video_understanding/sail_similarity-aware_guidance_and_inter-caption_augmentation-based_learning_for.md)
- [\[ACL 2026\] Response-G1: Explicit Scene Graph Modeling for Proactive Streaming Video Understanding](../../ACL2026/video_understanding/response-g1_explicit_scene_graph_modeling_for_proactive_streaming_video_understa.md)
- [\[CVPR 2026\] Stay in your Lane: Role Specific Queries with Overlap Suppression Loss for Dense Video Captioning](../../CVPR2026/video_understanding/stay_in_your_lane_role_specific_queries_with_overlap_suppression_loss_for_dense_.md)
- [\[CVPR 2026\] Understanding Temporal Logic Consistency in Video-Language Models through Cross-Modal Attention Discriminability](../../CVPR2026/video_understanding/understanding_temporal_logic_consistency_in_video-language_models_through_cross-.md)

</div>

<!-- RELATED:END -->
