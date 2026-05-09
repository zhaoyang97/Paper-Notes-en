---
title: >-
  [Paper Note] PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation
description: >-
  [CVPR 2026][Segmentation][Open-vocabulary segmentation] PCA-Seg revisits the cost aggregation mechanism in open-vocabulary semantic and part segmentation, proposing a parallel cost aggregation paradigm to replace existing serial architectures. It efficiently integrates semantic and contextual streams via an Expert-driven Perception Learning (EPL) module and reduces redundancy between the two knowledge streams through a Feature Orthogonal Decoupling (FOD) strategy. With only 0.35M additional parameters per parallel block, PCA-Seg achieves state-of-the-art performance across 8 benchmarks.
tags:
  - CVPR 2026
  - Segmentation
  - Open-vocabulary segmentation
  - cost aggregation
  - parallel architecture
  - feature orthogonal decoupling
  - part segmentation
date: 2026-05-08
content_hash: bf79b89e25b72dbb
---

# PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.17520](https://arxiv.org/abs/2603.17520)
**Code**: [https://github.com/NUST-Machine-Intelligence-Laboratory/PCA-Seg](https://github.com/NUST-Machine-Intelligence-Laboratory/PCA-Seg)
**Area**: Semantic Segmentation
**Keywords**: Open-vocabulary segmentation, cost aggregation, parallel architecture, feature orthogonal decoupling, part segmentation

## TL;DR

PCA-Seg revisits the cost aggregation mechanism in open-vocabulary semantic and part segmentation, proposing a parallel cost aggregation paradigm to replace existing serial architectures. It efficiently integrates semantic and contextual streams via an Expert-driven Perception Learning (EPL) module and reduces redundancy between the two knowledge streams through a Feature Orthogonal Decoupling (FOD) strategy. With only 0.35M additional parameters per parallel block, PCA-Seg achieves state-of-the-art performance across 8 benchmarks.

## Background & Motivation

1. **State of the Field**: Open-vocabulary semantic and part segmentation (OSPS) methods (e.g., CAT-Seg, DeCLIP, PartCATSeg) typically construct cost volumes from CLIP visual-textual features and extract aligned information through serial spatial aggregation followed by category aggregation.
2. **Limitations of Prior Work**: The cascaded behavior of spatial and category aggregation in serial architectures causes "knowledge interference" — performing spatial aggregation first distorts categorical semantics, and subsequent category aggregation further amplifies this bias, leading to misclassification (e.g., confusing a truck with a runway).
3. **Root Cause**: Category-level semantics and spatial structural information should be represented along two independent dimensions, but serial processing causes the aggregation of one type of information to trigger a chain reaction in the other.
4. **Paper Goals**: Design a parallel architecture in which the two aggregation operations run independently, eliminating knowledge interference while efficiently fusing the two knowledge streams.
5. **Starting Point**: A naive parallelization baseline is found to slightly decrease performance by 0.2%, indicating that the key challenge lies in how to effectively integrate the two independently produced knowledge streams.
6. **Core Idea**: Execute spatial and category aggregation in parallel → fuse via EPL multi-perspective integration → apply FOD orthogonality constraints to ensure knowledge diversity.

## Method

### Overall Architecture

PCA-Seg builds on CLIP's visual and text encoders and constructs a cost volume $\mathcal{S}$ via Hadamard product. The cost volume is simultaneously processed by spatial aggregation and category aggregation to produce spatial contextual features $\mathcal{B}_n$ and categorical semantic features $\mathcal{E}_n$. The EPL module extracts multi-perspective complementary knowledge from both streams, while the FOD strategy enforces orthogonality between the two streams to promote diversity in learning.

### Key Designs

1. **Expert-driven Perception Learning Module (EPL)**:
    - **Function**: Extracts and fuses complementary knowledge from the parallel semantic and contextual streams.
    - **Mechanism**: Comprises two components — a Multi-Expert Parser (ME-Parser) that synthesizes the outputs of both aggregation branches from multiple perspectives to extract complementary features; and a Coefficient Mapper (Co-Mapper) that performs dimensionality-reduction learning on the semantic and spatial features to generate pixel-wise weighting coefficients for adaptively weighting the expert-parsed features. The multi-expert design enables the model to interpret the same object from different angles.
    - **Design Motivation**: Simple parallelization is insufficient; a dedicated fusion mechanism is required to mine complementary information from two independent knowledge streams.

2. **Feature Orthogonal Decoupling Strategy (FOD)**:
    - **Function**: Reduces redundancy between the semantic and contextual streams.
    - **Mechanism**: An orthogonal decoupling loss is formulated to constrain the cosine similarity between features produced by the two streams to zero: $\mathcal{L}_{FOD} = |\cos(\mathcal{B}_n, \mathcal{E}_n)|$. By enforcing orthogonality between the two representations, the EPL module can learn more diverse knowledge from a broader feature space.
    - **Design Motivation**: Without the orthogonality constraint, the two feature streams remain correlated; enforcing orthogonality ensures that EPL extracts more diverse knowledge at the source level.

3. **Parallel Cost Aggregation Paradigm**:
    - **Function**: Replaces the serial architecture to eliminate knowledge interference.
    - **Mechanism**: The serial structure $\mathcal{V}_{n+1} = \Gamma_n(\Phi_n(\mathcal{V}_n))$ is reformulated as a parallel one in which spatial aggregation and category aggregation each independently process the cost volume, with their outputs fed separately into the EPL for fusion. Each parallel block introduces only 0.35M additional parameters and 0.96G GPU memory.
    - **Design Motivation**: The cascading effect of the serial structure is the root cause of knowledge interference; decoupling the two branches into a parallel arrangement fundamentally resolves the problem.

### Loss & Training

- Segmentation cross-entropy loss + FOD orthogonal decoupling loss
- End-to-end fine-tuning of CLIP attention layers
- Supports both open-vocabulary semantic segmentation (OVSS) and open-vocabulary part segmentation (OVPS)

## Key Experimental Results

### Main Results

| Benchmark | PCA-Seg | Prev. SOTA | Gain |
|-----------|---------|------------|------|
| A-150 (semantic) | SOTA | DeCLIP/H-CLIP | Significant |
| PC-459 (semantic) | SOTA | DeCLIP | Significant |
| Pascal-Part-116 (part) | SOTA | PartCATSeg | Significant |
| ADE20K-Part-234 (part) | SOTA | PartCATSeg | Significant |
| 8 benchmarks overall | All SOTA | — | Comprehensive lead |

### Ablation Study

| Configuration | mIoU (A-150) | Note |
|---------------|-------------|------|
| Serial baseline | Baseline | CAT-Seg serial architecture |
| Naive parallel | −0.2% | Direct parallelization slightly degrades performance |
| + EPL | +improvement | Expert-based fusion is effective |
| + FOD | +0.9% | Orthogonality constraint further promotes diversity |
| Full PCA-Seg | SOTA | All components work synergistically |

### Key Findings

- The marginal degradation from naive parallelization confirms the necessity of the EPL fusion mechanism.
- The FOD orthogonality constraint yields a 0.9% mIoU gain on A-150.
- Each parallel block introduces only 0.35M additional parameters, demonstrating exceptional parameter efficiency.
- Visualizations show that PCA-Seg activates finer-grained information compared to baselines.

## Highlights & Insights

- **The analysis of knowledge interference in cost aggregation** is highly convincing: visualized cases where cascaded serial processing amplifies semantic bias are intuitive and compelling.
- **Orthogonal decoupling as a guarantee of knowledge diversity** is a simple yet effective design that can be transferred to any dual-stream architecture.
- **The negligible parameter overhead (0.35M per block)** makes the method practically cost-free in real-world deployment.

## Limitations & Future Work

- The number of experts in EPL requires manual configuration.
- The hard orthogonality constraint in FOD may be overly strict; in certain scenarios, a moderate degree of coupling between semantic and spatial information may be beneficial.
- The effectiveness on larger-scale vision-language models remains unexplored.

## Related Work & Insights

- **vs. CAT-Seg/DeCLIP**: A direct improvement over serial aggregation architectures; parallelization eliminates knowledge interference.
- **vs. PartCATSeg**: The parallel aggregation paradigm generalizes equally well to part segmentation, validating the universality of the proposed method.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The parallel aggregation + orthogonal decoupling idea is concise and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensively validated across 8 benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation analysis is clear; visualizations are persuasive.
- **Value**: ⭐⭐⭐⭐ Provides a superior cost aggregation paradigm for open-vocabulary segmentation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SPAR: Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation](spar_single-pass_any-resolution_vit_for_open-vocabulary_segmentation.md)
- [\[CVPR 2026\] GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)
- [\[CVPR 2026\] PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)
- [\[CVPR 2026\] Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation](direct_segmentation_without_logits_optimization_for_training-free_open-vocabular.md)
- [\[CVPR 2026\] Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation](looking_beyond_the_window_global-local_aligned_clip_for_training-free_open-vocab.md)

<!-- RELATED:END -->
