---
title: >-
  [Paper Note] Sim-DETR: Unlock DETR for Temporal Sentence Grounding
description: >-
  [ICCV 2025][Object Detection][Temporal sentence grounding] This paper systematically analyzes the root causes of anomalous behavior in DETR-based temporal sentence grounding (TSG) — inter-query conflict and intra-query g…
tags:
  - "ICCV 2025"
  - "Object Detection"
  - "Temporal sentence grounding"
  - "DETR"
  - "query conflict"
  - "self-attention adjustment"
  - "global-local bridging"
date: 2026-05-08
content_hash: cc1b7e1db3d45d64
---

# Sim-DETR: Unlock DETR for Temporal Sentence Grounding

**Conference**: ICCV 2025
**arXiv**: [2509.23867](https://arxiv.org/abs/2509.23867)  
**Code**: [github.com/SooLab/Sim-DETR](https://github.com/SooLab/Sim-DETR)  
**Area**: Object Detection / Temporal Sentence Grounding
**Keywords**: Temporal sentence grounding, DETR, query conflict, self-attention adjustment, global-local bridging

## TL;DR

This paper systematically analyzes the root causes of anomalous behavior in DETR-based temporal sentence grounding (TSG) — inter-query conflict and intra-query global-local contradiction — and proposes two simple decoder modifications (Query Grouping & Ranking + Global-Local Bridging) to form Sim-DETR, unlocking the full potential of DETR for TSG.

## Background & Motivation

TSG requires localizing the temporal segment in an untrimmed video corresponding to a natural language query. Dominant methods adopt the DETR framework, using learnable queries to predict temporal segments in the decoder.

**Anomalous observations**: In object detection, increasing the number of queries and decoder layers typically improves DETR performance. However, in TSG:
- Increasing queries from 10 to 20: performance drops by more than 2%
- Increasing decoder layers: performance drops by 1–2.5%

**Root cause analysis** (one of the paper's core contributions):

**Inter-query conflict**: In TSG, multiple target segments share the same linguistic semantics (e.g., multiple events corresponding to the same sentence), causing their associated queries to be highly similar. Under one-to-one matching, the same query may be matched to different target segments across decoder layers (a "random matching" phenomenon), resulting in very low cross-layer matching consistency.

**Intra-query conflict**: Each query must simultaneously serve two roles — (a) encoding global segment semantics for matching, and (b) decoding local boundaries for precise localization. These two objectives are inherently contradictory: should the query attend to global semantics or local boundaries? Experiments show that a high global matching score does not guarantee accurate local localization.

## Method

### Overall Architecture

Sim-DETR introduces two "small but critical" decoder modifications to a standard DETR-based TSG architecture:
- Feature extraction: CLIP [CLS] + SlowFast concatenation
- Multimodal encoder: video-language cross-attention fusion
- Decoder: augmented with QGR and GLB modules

### Key Designs

1. **Query Grouping and Ranking (QGR)**:

    - **Query grouping**: Soft grouping based on L2 distance between predicted temporal segments; queries with close predictions are treated as likely corresponding to the same target segment:
    $\mathcal{S}^{intra}_{i,j} = \|b_i - b_j\|_2$
    L2 is used over L1 distance because when two segments are close (normalized distance ≤ 1), L2 decays faster and imposes smaller penalties for minor differences.

    - **Query ranking**: An IoU prediction head is introduced, and queries are ranked by combining classification confidence and predicted IoU:
    $R_{rank}(q_i, q_j) = \begin{cases} +1 & \mathcal{P}^{cls}_i \circ \mathcal{P}^{IoU}_i \geq \mathcal{P}^{cls}_j \circ \mathcal{P}^{IoU}_j \\ -1 & \text{otherwise} \end{cases}$

    - **Self-attention adjustment**: Grouping and ranking information is incorporated into self-attention weights:
    $\mathcal{S}^{attn} = \text{sigmoid}(\text{MLP}(\mathcal{S}^{intra} \circ \mathcal{R}_{rank}))$
    High values encourage high-quality queries to aggregate information from similar queries within the same group.
    - **Design motivation**: Enable indistinguishable queries to attend to different contexts, reduce inter-query similarity, and allow the most suitable query to draw information from related queries.

2. **Global-Local Bridging (GLB)**:

    - Introduces a query-to-frame matching loss to strengthen the alignment between each query and every frame within the predicted segment.
    - Computes semantic similarity between the query and all frames:
    $z = \text{sigmoid}(\tau \cdot \cos(q_i, \hat{\mathcal{T}}))$
    - The loss maximizes similarity to intra-segment frames and minimizes similarity to extra-segment frames:
    $\mathcal{L}_{bridge} = \lambda_{bridge} \frac{-\sum_j z_j \mathbb{I}[b^{gt}_i]_j}{\sum_j z_j(1-\mathbb{I}[b^{gt}_i]_j) + \sum_j \mathbb{I}[b^{gt}_i]_j}$
    where $\tau$ is a learnable scaling factor.
    - **Design motivation**: The complete frame sequence within a segment (from start to end) serves as a bridge connecting global semantics and local boundaries.

3. **IoU Prediction Head**:

    - Serves as an auxiliary signal for evaluating localization precision, used jointly with classification scores for query ranking.
    - Addresses the issue that "high confidence ≠ precise localization."
    - **Design motivation**: Ranking queries by classification score alone is ineffective in TSG (unlike object detection), necessitating an explicit local localization signal.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{MD} + \lambda_{bridge}\mathcal{L}_{bridge} + \lambda_{iou}\mathcal{L}_{iou}$$

- $\mathcal{L}_{MD}$: Standard Moment DETR loss (L1 + gIoU + classification + saliency loss)
- $\mathcal{L}_{bridge}$: Global-local bridging loss
- $\mathcal{L}_{iou}$: IoU prediction head loss
- Trained for 200 epochs on a single A40 GPU, AdamW with lr=1e-4, 6 decoder layers.

## Key Experimental Results

### Main Results

QVHighlights test set:

| Method | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg |
|--------|--------|--------|---------|----------|---------|
| M-DETR | 52.89 | 33.02 | 54.82 | 29.40 | 30.73 |
| QD-DETR | 62.40 | 44.98 | 62.52 | 39.88 | 39.86 |
| TR-DETR | 64.66 | 48.96 | 63.98 | 43.73 | 42.62 |
| BAM-DETR | 62.71 | 48.64 | 64.57 | 46.33 | 45.36 |
| CG-DETR | 65.43 | 48.38 | 64.51 | 42.77 | 42.86 |
| **Sim-DETR** | **67.64** | **50.91** | **67.81** | **47.59** | **46.93** |

Charades-STA / TACoS:

| Dataset | Method | R1@0.5 | R1@0.7 | mIoU |
|---------|--------|--------|--------|------|
| TACoS | CG-DETR | 39.61 | 22.23 | 36.48 |
| TACoS | **Sim-DETR** | **42.79** | **26.82** | **39.44** |
| Charades | SpikeMba | 59.65 | 36.12 | 51.74 |
| Charades | **Sim-DETR** | **61.34** | **39.62** | **52.56** |

### Ablation Study

Component ablation (QVHighlights val):

| Configuration | R1@0.5 | R1@0.7 | mAP Avg |
|---------------|--------|--------|---------|
| Baseline (TR-DETR) | 65.48 | 50.84 | 44.97 |
| + $\mathcal{L}_{iou}$ | 66.58 | 51.94 | 45.22 |
| + QGR | 68.77 | 52.26 | 47.03 |
| + GLB | 67.16 | 52.77 | 48.17 |
| **+ QGR + GLB** | **69.48** | **54.06** | **49.50** |

Conflict metric ablation:

| Inner Relevance | Outer Global | Outer Local | mAP Avg |
|----------------|--------------|-------------|---------|
| span border dist | confidence | IoU pred | **49.50** |
| center dist | confidence | IoU pred | 48.59 |
| span IoU | confidence | IoU pred | 48.94 |
| span border dist | w/o | IoU pred | 48.93 |
| span border dist | confidence | w/o | 48.66 |

### Key Findings

- **QGR effectively differentiates queries**: Analysis shows that Sim-DETR successfully separates the similarity distributions of intra-segment and inter-segment queries, reducing query "oscillation" across different target segments.
- **Cross-layer matching consistency significantly improves**: QGR substantially increases query-segment matching consistency between consecutive decoder layers.
- **GLB aligns global semantics with local localization**: After introducing GLB, queries concentrate attention on intra-segment frames rather than dispersing across multiple target segments.
- **Anomalous behavior eliminated**: Increasing the number of queries and decoder layers no longer degrades performance, and yields slight improvements instead.
- **Accelerated convergence**: Eliminating the anomalous behavior significantly speeds up training convergence.
- **L2 distance outperforms L1/IoU**: Boundary distance proves superior to center distance and span IoU as a query grouping criterion.

## Highlights & Insights

- **Diagnosis-driven method design**: Solutions are designed through systematic observation of phenomena and root cause analysis (three dedicated investigation sections), followed by targeted remedies.
- The paper reveals a fundamental distinction between TSG and object detection: multiple target segments share linguistic semantics.
- **Minimal modifications, significant gains**: Only two decoder modifications are sufficient to comprehensively surpass all state-of-the-art methods.
- **Joint classification + IoU ranking**: Specifically addresses the problem that "high confidence ≠ precise localization."
- **Thorough analysis of effects**: Beyond reporting performance gains, the paper also validates anomaly elimination and convergence acceleration.

## Limitations & Future Work

- Query grouping relies on predicted span distances, which may be inaccurate during early training and thus affect grouping quality.
- Frame-level alignment in GLB uses simple cosine similarity; more sophisticated alignment strategies may yield further improvements.
- Validation is limited to video temporal grounding; generalization to other DETR-based tasks has not been explored.
- The IoU prediction head introduces additional parameters and computation, which may be a concern in resource-constrained settings.

## Related Work & Insights

- The query ranking strategy from EASE-DETR inspired QGR, but directly using predicted scores for ranking is ineffective in TSG.
- The baseline loss design from Moment DETR provides a standard framework for TSG.
- The analysis of "semantically similar but spatially distinct target segments" in TSG may generalize to analogous multi-instance detection scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The diagnostic analysis is exceptionally thorough, uncovering fundamental issues in applying DETR to TSG.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensively outperforms prior work on three benchmarks, with multiple backbones, detailed ablations, and visual analyses.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The "diagnose-then-treat" narrative structure is very clear, with analysis and validation tightly integrated.
- **Value**: ⭐⭐⭐⭐⭐ Provides a simple yet effective strong baseline for DETR-based TSG with broad methodological implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EvRT-DETR: Latent Space Adaptation of Image Detectors for Event-based Vision](evrt-detr_latent_space_adaptation_of_image_detectors_for_event-based_vision.md)
- [\[CVPR 2026\] PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection](../../CVPR2026/object_detection/paq-detr_learning_pattern_and_quality-aware_dynamic_queries_for_object_detection.md)
- [\[CVPR 2026\] EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer](../../CVPR2026/object_detection/ew-detr_evolving_world_object_detection_via_incremental_low-rank_detection_trans.md)
- [\[AAAI 2026\] TubeRMC: Tube-conditioned Reconstruction with Mutual Constraints for Weakly-supervised Spatio-Temporal Video Grounding](../../AAAI2026/object_detection/tubermc_tube-conditioned_reconstruction_with_mutual_constraints_for_weakly-super.md)
- [\[ICCV 2025\] The Devil is in the Spurious Correlations: Boosting Moment Retrieval with Dynamic Learning](the_devil_is_in_the_spurious_correlations_boosting_moment_retrieval_with_dynamic.md)

</div>

<!-- RELATED:END -->
