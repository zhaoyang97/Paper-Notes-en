---
title: >-
  [Paper Note] Beyond Caption-Based Queries for Video Moment Retrieval
description: >-
  [CVPR 2026][Object Detection][Video moment retrieval] This paper identifies a substantial gap between caption-based queries and real-world search queries in VMR, introduces three search-query benchmarks…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "Video moment retrieval"
  - "search query generalization"
  - "DETR decoder query collapse"
  - "multi-moment retrieval"
  - "query under-specification"
date: 2026-05-08
content_hash: 9496eb590671f126
---

# Beyond Caption-Based Queries for Video Moment Retrieval

**Conference**: CVPR 2026
**arXiv**: [2603.02363](https://arxiv.org/abs/2603.02363)  
**Code**: Available (code, models, and data provided on the project page)  
**Area**: Object Detection
**Keywords**: Video moment retrieval, search query generalization, DETR decoder query collapse, multi-moment retrieval, query under-specification

## TL;DR

This paper identifies a substantial gap between caption-based queries and real-world search queries in VMR, introduces three search-query benchmarks, and mitigates active decoder-query collapse in DETR via two architectural modifications—self-attention removal and query dropout—achieving gains of up to 21.83% mAPm on multi-moment search queries.

## Background & Motivation

### 1. State of the Field
Video Moment Retrieval (VMR) aims to localize temporal segments in a video given a text query. Dominant methods adopt DETR-based architectures with $K$ learnable decoder queries, each mapped to a candidate moment and its confidence score. Existing benchmarks (HD-EPIC, YouCook2, ActivityNet-Captions, etc.) use descriptive text written by annotators **after watching the video** as queries.

### 2. Limitations of Prior Work
Text queries in existing datasets are **caption-based**—annotators compose fine-grained descriptions after viewing the video. This introduces a "visual bias": queries are overly detailed and highly aligned with visual content. For instance, an annotator might write "a man in a yellow jersey intercepts a loose pass…," whereas a real user might simply search "when are goals being scored?" These two query types differ fundamentally in linguistic granularity and semantic coverage.

### 3. Root Cause
- **At training time**: each caption-based query corresponds to a **single GT moment** with highly specific language.
- **At inference time**: real search queries tend to be more abstract and under-specified, potentially corresponding to **multiple moments** in the video.
- This mismatch causes drastic performance degradation in real-world search scenarios (up to 77.4% drop in Rm@0.3).

### 4. Paper Goals
(1) Quantify the performance gap between caption-based and search queries; (2) identify two root causes of degradation—**language gap** and **multi-moment gap**; (3) alleviate decoder-query collapse induced by the multi-moment gap.

### 5. Starting Point
The approach operates purely at the **model architecture** level, without modifying training data or the training paradigm. Only structural modifications are introduced to enable models trained on single-moment data to generalize to multi-moment search scenarios.

### 6. Core Idea
DETR models exhibit **active decoder-query collapse**—only a small subset of queries participates in prediction while the rest remain silent. This is attributed to two structural causes: (i) **coordination collapse** induced by self-attention, where queries coordinate to let only a few activate; and (ii) **index collapse**, where a fixed small set of query indices monopolizes activation. Removing self-attention (-SA) and introducing query dropout (+QD) address both issues simultaneously.

## Method

### Overall Architecture

The paper consists of two major components:

1. **Benchmark construction**: an LLM-based search-query generation pipeline that converts existing caption-based datasets into search-query benchmarks.
2. **Architectural improvement**: two modifications to DETR-based VMR models (-SA + QD) to mitigate active decoder-query collapse.

### Key Designs

#### Design 1: Search-Query Generation Pipeline

**Function**: Converts fine-grained captions into under-specified search queries and automatically establishes multi-moment correspondences.

**Mechanism**: A two-stage pipeline—
- **Per-query under-specification stage**: A rewriter–validator dual-agent system built on Gemma-12B. The rewriter paraphrases detailed captions into vague versions (e.g., "a man tying his running shoes before starting a marathon" → "a person getting ready to exercise"), and the validator detects inconsistencies for human correction.
- **Query-grouping stage**: Pairwise sentence-embedding similarities are computed across all under-specified queries; highly similar queries are merged into groups (corresponding to multiple moments), and an LLM aggregator generates a representative search query for each group.

**Design Motivation**: Real search queries cannot be collected through simple annotation (since decoupling textual labeling from video viewing is inherently difficult). The pipeline therefore repurposes existing densely annotated datasets and simulates the distributional shift of search queries through controlled under-specification.

#### Design 2: Self-Attention Removal (-SA)

**Function**: Directly removes the self-attention module among decoder queries in each DETR decoder layer.

**Mechanism**: The standard decoder layer follows $\hat{Q}^{l+1} = \text{FFN}(\text{CA}(\text{SA}(\hat{Q}^l), M))$; after modification it becomes $Q^{l+1} = \text{FFN}(\text{CA}(Q^l, M))$. NMS is applied as post-processing to suppress redundant predictions.

**Design Motivation**: Self-attention encourages decoder queries to repel each other so as to reduce redundancy. However, under single-moment training, this coordination mechanism causes queries to collectively agree to let only a few handle GT moments while the others shut down—termed **coordination collapse**. Removing self-attention allows each query to operate independently, breaking this coordination shortcut.

#### Design 3: Query Dropout (+QD)

**Function**: Randomly zeros out $k$% of the learnable decoder queries during training.

**Mechanism**: $\hat{Q} = Q \odot M, \quad M \sim \mathbb{B}(1-k)$, where $\mathbb{B}$ denotes the Bernoulli distribution; $k=0.25$ yields the best performance.

**Design Motivation**: Even after removing self-attention, **index collapse** persists—a fixed small set of query indices (e.g., indices 1–4) repeatedly attains high confidence while the rest remain permanently silent. QD forces the model to distribute supervisory signals across more queries by randomly masking a subset during training, preventing over-reliance on a fixed subset.

### Loss & Training

- Loss functions are kept identical to the baselines (CG-DETR, LD-DETR), using standard **one-to-one Hungarian matching**.
- A key finding is that retaining 1-to-1 matching is critical—it introduces competition among queries, ensuring that queries additionally activated by -SA+QD remain diverse rather than generating redundant predictions.
- Query dropout is applied only during training; all queries are activated at inference.
- An NMS post-processing step is added to replace the redundancy-suppression function formerly provided by self-attention.

## Key Experimental Results

### Main Results

**Table 1: Results on HD-EPIC-S{1,2,3} benchmarks (CG-DETR & LD-DETR)**

| Model | Input | Method | Rm@0.1 | Rm@0.3 | Rm@0.5 | mAPm@0.1 | mAPm@0.3 | mAPm@0.5 |
|------|------|------|--------|--------|--------|----------|----------|----------|
| CG-DETR | S1 | base | 28.61 | 17.95 | 8.99 | 36.21 | 22.84 | 11.59 |
| CG-DETR | S1 | -SA+QD | 29.87 | 19.69 | 10.86 | 39.74 | 26.49 | 14.87 |
| CG-DETR | S2 | base | 24.71 | 15.52 | 7.89 | 32.15 | 20.10 | 10.29 |
| CG-DETR | S2 | -SA+QD | 26.17 | 17.00 | 9.40 | 35.38 | 23.39 | 13.04 |
| CG-DETR | S3 | base | 9.50 | 4.61 | 2.08 | 16.20 | 8.01 | 3.58 |
| CG-DETR | S3 | -SA+QD | 10.57 | 6.52 | 3.45 | 17.27 | 10.65 | 5.54 |
| LD-DETR | S2 | base | 25.23 | 16.38 | 8.46 | 32.42 | 21.11 | 10.93 |
| LD-DETR | S2 | -SA+QD | 26.36 | 16.98 | 8.87 | 36.37 | 23.75 | 12.54 |

**Table 2: Results on YC2-S and ANC-S benchmarks**

| Model | Dataset | Method | Rm@0.3 | mAPm@0.1 | mAPm@0.3 | mAPm@0.5 |
|------|--------|------|--------|----------|----------|----------|
| CG-DETR | YC2-S | base | 19.87 | 38.83 | 26.96 | 15.21 |
| CG-DETR | YC2-S | -SA+QD | 20.32 | 41.00 | 29.40 | 17.21 |
| LD-DETR | YC2-S | base | 23.48 | 41.69 | 30.04 | 15.58 |
| LD-DETR | YC2-S | -SA+QD | 24.76 | 45.66 | 33.09 | 18.74 |
| CG-DETR | ANC-S | base | 40.89 | 72.12 | 54.92 | 36.42 |
| CG-DETR | ANC-S | -SA+QD | 43.12 | 74.00 | 56.42 | 38.20 |

### Ablation Study

**Component ablation (HD-EPIC-S2, CG-DETR)**

| -SA | +QD | Rm (avg) | mAPm (avg) | #active queries |
|-----|-----|----------|------------|-----------------|
| ✗ | ✗ | 16.04 | 20.84 | 3.64±1.18 |
| ✓ | ✗ | 15.31 | 21.02 | 3.72±1.16 |
| ✗ | ✓ | 16.50 | 21.43 | 3.77±1.28 |
| ✓ | ✓ | **17.52** | **23.93** | **6.43±2.16** |

**Comparison with alternative query activation methods**

| Method | Rm | mAPm | #active | %match GT |
|------|-----|------|---------|-----------|
| base | 16.04 | 20.84 | 3.64 | 0.36 |
| +1-to-5 matching | 14.66 | 16.30 | 9.56 | 0.21 |
| +1-to-k matching | 10.78 | 11.01 | 20.00 | 0.07 |
| +group matching | 15.34 | 17.97 | 8.69 | 0.27 |
| -SA+QD (ours) | **17.52** | **23.93** | 6.43 | **0.42** |

### Key Findings

1. **Both modifications are necessary**: applying -SA or +QD alone yields only marginal gains (mAPm from 20.84 to ~21); combining them doubles active queries from 3.64 to 6.43 and improves mAPm by 3.09.
2. **Simply increasing active queries is ineffective**: 1-to-k matching raises active queries to 20 but mAPm collapses to 11.01—the activated queries generate redundant predictions (%match GT drops from 0.36 to 0.07).
3. **The critical role of 1-to-1 matching**: retaining Hungarian 1-to-1 matching ensures that newly activated queries compete rather than replicate each other.
4. **Sensitivity to QD rate**: $k=0.25$ is optimal; $k=0.50$ causes performance collapse (mAPm drops from 23.93 to 3.84).
5. **Multi-moment queries benefit most**: -SA+QD yields gains of up to 34.3% mAPm@0.3 on multi-moment instances, with moderate improvements on single-moment instances as well.
6. **The method recovers approximately 70% of the oracle gap** (where the oracle denotes a model trained directly on search queries).

## Highlights & Insights

- **Insightful problem formulation**: the paper identifies a fundamental issue long overlooked in the VMR community—the distributional shift between training captions and real user search queries—which is critical for practical deployment yet has been neglected by the field at large.
- **Novel multi-moment evaluation metrics**: Rm and mAPm address the unfairness of conventional R1/mAP metrics when evaluating multi-moment retrieval.
- **Precise diagnosis of decoder-query collapse**: the analysis via the orthogonal dimensions of coordination collapse and index collapse is well-motivated, and the proposed solution is concise and effective.
- **Architecture-only approach**: modifying only the model structure without altering data or training is highly practical, as it avoids costly re-annotation.

## Limitations & Future Work

1. **Language gap remains unaddressed**: the paper resolves only the multi-moment gap; the language gap is left as future work, with the authors suggesting the use of stronger vision-language models to handle cross-granularity semantic reasoning.
2. **Search queries are LLM-generated rather than from real users**: despite validation, the generated queries may still deviate from actual user search behavior.
3. **High sensitivity to the QD rate**: increasing $k$ from 0.25 to 0.50 causes performance to collapse from 23.93 to 3.84 mAPm, indicating limited robustness.
4. **Benchmarks cover only cooking and sports domains**: generalization to open-domain or long-video settings remains unexplored.
5. **Reliance on NMS post-processing**: removing self-attention necessitates NMS for redundancy suppression, introducing additional hyperparameters and computational overhead.

## Related Work & Insights

- **Connection to DETR query collapse literature**: query collapse has been reported in object detection ([53,28,21]), temporal action detection ([17]), and 3D detection ([44,52]), though the underlying causes differ—those cases stem from sparse one-to-one matching, whereas in VMR the cause is the single-moment prior.
- **Implications for search and retrieval**: Liang et al. [24] study the effect of ambiguous queries on ranked retrieval; this paper addresses multi-moment retrieval within a single video, and the two lines of work are complementary.
- **Transferability of the method**: the -SA+QD design can be applied to any DETR variant that suffers from decoder-query collapse.

## Rating

⭐⭐⭐⭐ The paper offers a novel problem formulation, rigorous analysis, and a concise yet effective solution, representing an important step toward deploying VMR in real-world settings. Its main shortcomings are the unresolved language gap and the high sensitivity of the QD rate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond Semantic Search: Towards Referential Anchoring in Composed Image Retrieval](beyond_semantic_search_towards_referential_anchoring_in_composed_image_retrieval.md)
- [\[ICCV 2025\] Large-scale Pre-training for Grounded Video Caption Generation](../../ICCV2025/object_detection/large-scale_pre-training_for_grounded_video_caption_generation.md)
- [\[NeurIPS 2025\] Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension](../../NeurIPS2025/object_detection/video-rag_visually-aligned_retrieval-augmented_long_video_comprehension.md)
- [\[CVPR 2026\] PaQ-DETR: Learning Pattern and Quality-Aware Dynamic Queries for Object Detection](paq-detr_learning_pattern_and_quality-aware_dynamic_queries_for_object_detection.md)
- [\[ICCV 2025\] Augmenting Moment Retrieval: Zero-Dependency Two-Stage Learning](../../ICCV2025/object_detection/augmenting_moment_retrieval_zero-dependency_two-stage_learning.md)

</div>

<!-- RELATED:END -->
