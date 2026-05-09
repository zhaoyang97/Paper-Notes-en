---
title: >-
  [Paper Note] PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation
description: >-
  [CVPR 2026][Segmentation][Open-vocabulary segmentation] PCA-Seg proposes a Parallel Cost Aggregation (PCA) paradigm to replace the conventional serial spatial-categorical aggregation architecture. It efficiently integrates semantic and spatial context streams via an Expert-driven Perception Learning (EPL) module, and eliminates redundancy between the two knowledge streams through a Feature Orthogonal Decoupling (FOD) strategy. Each parallel block adds only 0.35M parameters while achieving state-of-the-art performance across 8 open-vocabulary semantic and part segmentation benchmarks.
tags:
  - CVPR 2026
  - Segmentation
  - Open-vocabulary segmentation
  - cost aggregation
  - parallel architecture
  - expert-driven learning
  - feature orthogonal decoupling
date: 2026-05-08
content_hash: ffa583960c234ffb
---

# PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.17520](https://arxiv.org/abs/2603.17520)
**Code**: [https://github.com/NUST-Machine-Intelligence-Laboratory/PCA-Seg](https://github.com/NUST-Machine-Intelligence-Laboratory/PCA-Seg)
**Area**: Semantic Segmentation / Open-Vocabulary Segmentation
**Keywords**: Open-vocabulary segmentation, cost aggregation, parallel architecture, expert-driven learning, feature orthogonal decoupling

## TL;DR

PCA-Seg proposes a Parallel Cost Aggregation (PCA) paradigm to replace the conventional serial spatial-categorical aggregation architecture. It efficiently integrates semantic and spatial context streams via an Expert-driven Perception Learning (EPL) module, and eliminates redundancy between the two knowledge streams through a Feature Orthogonal Decoupling (FOD) strategy. Each parallel block adds only 0.35M parameters while achieving state-of-the-art performance across 8 open-vocabulary semantic and part segmentation benchmarks.

## Background & Motivation

1. **Background**: Open-vocabulary semantic and part segmentation (OSPS) leverages the powerful vision-language alignment of models such as CLIP to enable segmentation of arbitrary categories. Mainstream methods (e.g., CAT-Seg, DeCLIP, PartCATSeg) extract image-text alignment cues from cost volumes.
2. **Limitations of Prior Work**: Existing methods adopt a serial architecture—performing spatial aggregation before categorical aggregation (or vice versa)—which causes knowledge interference between category-level semantics and spatial context. For instance, spatial aggregation may distort the semantics of a "truck" category, and subsequent categorical aggregation further amplifies the deviation, leading to misclassification.
3. **Key Challenge**: The cascaded behavior in serial architectures means that aggregating one type of information triggers a chain reaction in the other, making mutual contamination of the two knowledge streams inevitable.
4. **Goal**: Design a parallel architecture that allows the two aggregation operations to function independently, while addressing the challenge of efficiently integrating the resulting separate knowledge streams.
5. **Key Insight**: Empirical observation shows that a naive parallel implementation (a single convolution capturing both types of information simultaneously) actually degrades performance by 0.2%, indicating that a carefully designed integration mechanism is necessary.
6. **Core Idea**: Parallel aggregation + multi-expert parser for multi-perspective fusion + orthogonal decoupling to eliminate redundancy.

## Method

### Overall Architecture

Image and text features are extracted via CLIP encoders, and the Hadamard product is computed to construct the cost volume $\mathcal{S}$. The cost volume is simultaneously passed through two parallel branches—spatial aggregation and categorical aggregation—producing spatial context features $\mathcal{B}_n$ and categorical semantic features $\mathcal{E}_n$, respectively. The EPL module extracts complementary knowledge from both streams and fuses them, while the FOD strategy enforces orthogonality between the two streams at the knowledge source level.

### Key Designs

1. **Expert-driven Perception Learning (EPL)**:

    - **Function**: Efficiently integrates the two knowledge streams produced by categorical and spatial aggregation.
    - **Mechanism**: Comprises two components—(a) *Multi-Expert Parser (ME-Parser)*: employs multiple sets of weights to extract complementary features from both streams, with each expert focusing on a different perspective; (b) *Coefficient Mapper (Co-Mapper)*: performs dimensionality reduction on the semantic and spatial features to learn pixel-wise adaptive weight coefficients, which highlight key regions in the expert parsing results, ultimately producing a unified and robust feature embedding.
    - **Design Motivation**: A single-pass fusion cannot fully exploit the complementarity of two independent knowledge streams; multi-perspective parsing and adaptive weighting are therefore required.

2. **Feature Orthogonal Decoupling (FOD)**:

    - **Function**: Reduces redundancy between categorical semantic and spatial context features.
    - **Mechanism**: An orthogonal decoupling loss is designed to constrain the cosine similarity between the representations of the two streams to zero, forcing the two knowledge streams to be orthogonal. This orthogonalization ensures that the two streams provide maximally complementary information at the knowledge source level.
    - **Design Motivation**: Category semantics and spatial structure inherently constitute two independent dimensions of knowledge; orthogonalization ensures that EPL can extract more diverse and complementary knowledge from them.

3. **Parallel Cost Aggregation Architecture**:

    - **Function**: Eliminates knowledge interference present in serial architectures.
    - **Mechanism**: Replaces the conventional serial formulation $\mathcal{V}_{n+1} = \Gamma_n(\Phi_n(\mathcal{V}_n))$ with parallel operations $\mathcal{B}_n = \Phi_n(\mathcal{V}_n)$ and $\mathcal{E}_n = \Gamma_n(\mathcal{V}_n)$, where both branches operate independently and are subsequently fused by EPL. Each parallel block adds only 0.35M parameters and 0.96G GPU memory overhead.
    - **Design Motivation**: Eliminates the distortion of categorical semantics by spatial aggregation and the disruption of spatial structure by categorical aggregation.

### Loss & Training

Standard segmentation cross-entropy loss combined with the FOD orthogonalization loss. Training follows the protocols of CAT-Seg and PartCATSeg.

## Key Experimental Results

### Main Results

| Dataset | Metric (mIoU↑) | Prev. SOTA | Ours | Gain |
|--------|-------------|----------|---------|------|
| A-150 (Semantic) | mIoU | 14.9 (DeCLIP) | 15.6 | +0.7 |
| PAS-20b (Semantic) | mIoU | 81.3 (H-CLIP) | 82.4 | +1.1 |
| ADE-Part-234 (O) | mIoU | 24.1 (PartCATSeg) | 25.3 | +1.2 |
| Pascal-Part-116 (H) | hIoU | 43.8 (PartCATSeg) | 45.1 | +1.3 |

State-of-the-art results are achieved across all 8 benchmarks covering both semantic and part segmentation.

### Ablation Study

| Configuration | mIoU (A-150) | Note |
|------|-------------|------|
| Serial baseline (CAT-Seg) | 14.9 | Original serial architecture |
| Parallel baseline (single conv) | 14.7 | Naive parallel degrades performance |
| +EPL | 15.3 | Multi-expert fusion improves results |
| +FOD | 15.6 | Orthogonalization provides further +0.9% gain |

### Key Findings

- Naive parallelism underperforms the serial baseline (−0.2%); EPL is essential for the parallel design to realize its advantage.
- FOD yields a 0.9% gain on A-150, demonstrating that reducing redundancy is critical for learning diverse and complementary knowledge.
- The method is highly parameter-efficient: each parallel block adds only 0.35M parameters (vs. 0.33M for a serial block).
- Gains are larger on part segmentation, likely because part-level understanding requires finer-grained spatial-semantic decoupling.

## Highlights & Insights

- **Identifying Knowledge Interference**: The paper clearly identifies the cascaded interference between spatial and categorical aggregation in serial architectures, supported by compelling visualizations.
- **Effective Use of Orthogonalization**: Orthogonal constraints are employed to enforce the independence of the two information streams—a simple yet effective design choice.
- **Negligible Parameter Overhead**: Each block adds only 0.35M parameters, achieving performance gains at virtually no cost.

## Limitations & Future Work

- The approach still relies on fine-tuning the CLIP ViT attention layers and is thus bounded by CLIP's visual representation capacity.
- Orthogonalization imposes a hard constraint; in certain scenarios, category and spatial information may genuinely need to interact.
- Validation on 3D or video segmentation has not been conducted.
- Future work could explore more flexible interaction mechanisms between the two knowledge streams.

## Related Work & Insights

- **vs. CAT-Seg/DeCLIP**: These methods rely on serial aggregation; PCA-Seg adopts parallel aggregation to eliminate interference.
- **vs. PartCATSeg**: PCA-Seg's parallel design shows a more pronounced advantage on part segmentation.
- **vs. H-CLIP**: H-CLIP operates in hyperbolic space; PCA-Seg achieves analogous representational decoupling in Euclidean space via orthogonalization.

## Rating

- Novelty: ⭐⭐⭐⭐ Replacing serial aggregation with parallel aggregation is an insightful architectural improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 8 benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Motivation is well-analyzed and visualizations are clear.
- Value: ⭐⭐⭐⭐ Provides meaningful guidance for the cost aggregation paradigm in open-vocabulary segmentation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SPAR: Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation](spar_single-pass_any-resolution_vit_for_open-vocabulary_segmentation.md)
- [\[CVPR 2026\] GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)
- [\[CVPR 2026\] PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)
- [\[CVPR 2026\] Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)
- [\[CVPR 2026\] Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation](looking_beyond_the_window_global-local_aligned_clip_for_training-free_open-vocab.md)

<!-- RELATED:END -->
