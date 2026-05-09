---
title: >-
  [Paper Note] PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation
description: >-
  [CVPR 2026][Segmentation][open-vocabulary segmentation] This paper revisits cost aggregation strategies and proposes PCA-Seg, a parallel architecture that replaces the conventional serial design. It integrates class-semantic and spatial-contextual information via an Expert-driven Perception Learning (EPL) module, and employs a Feature Orthogonalization Decoupling (FOD) strategy to reduce redundancy. PCA-Seg achieves state-of-the-art performance on 8 benchmarks with only 0.35M additional parameters per block.
tags:
  - CVPR 2026
  - Segmentation
  - open-vocabulary segmentation
  - parallel cost aggregation
  - expert-driven perception
  - feature orthogonalization
  - vision-language models
date: 2026-05-08
content_hash: 1ea05384bcc865ac
---

# PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation

**Conference**: CVPR 2026
**arXiv**: [2603.17520](https://arxiv.org/abs/2603.17520)
**Code**: [https://github.com/NUST-Machine-Intelligence-Laboratory/PCA-Seg](https://github.com/NUST-Machine-Intelligence-Laboratory/PCA-Seg)
**Area**: Segmentation / Open-Vocabulary Segmentation
**Keywords**: open-vocabulary segmentation, parallel cost aggregation, expert-driven perception, feature orthogonalization, vision-language models

## TL;DR

This paper revisits cost aggregation strategies and proposes PCA-Seg, a parallel architecture that replaces the conventional serial design. It integrates class-semantic and spatial-contextual information via an Expert-driven Perception Learning (EPL) module, and employs a Feature Orthogonalization Decoupling (FOD) strategy to reduce redundancy. PCA-Seg achieves state-of-the-art performance on 8 benchmarks with only 0.35M additional parameters per block.

## Background & Motivation

1. **Background**: CLIP-based open-vocabulary semantic and part segmentation methods extract vision-language alignment cues from cost volumes through spatial and class aggregation.
2. **Limitations of Prior Work**: Existing methods adopt a serial structure—performing spatial aggregation before class aggregation (or vice versa)—which causes knowledge interference: the prior aggregation step alters the input distribution for the subsequent step, and spatial aggregation may distort class semantics.
3. **Key Challenge**: Class-semantic and spatial-structural information must be captured simultaneously, yet serial processing allows one type of information to contaminate the other.
4. **Goal**: Design a parallel architecture to eliminate knowledge interference introduced by serial processing.
5. **Key Insight**: Class semantics and spatial structure represent knowledge along two orthogonal dimensions and should be processed independently before fusion.
6. **Core Idea**: Parallel cost aggregation + Expert-driven Perception Learning (EPL) for dual-branch fusion + Feature Orthogonalization Decoupling (FOD) to reduce redundancy.

## Method

### Overall Architecture

The cost volume is constructed as a similarity matrix between CLIP visual and text encoder features. Parallel spatial aggregation and class aggregation branches process the cost volume independently; the EPL module fuses their outputs, and the FOD strategy enforces orthogonality between the two feature streams.

### Key Designs

1. **EPL (Expert-driven Perception Learning Module)**: A set of expert parsers extracts complementary features from multiple perspectives, while a coefficient mapper adaptively learns pixel-wise weights to integrate knowledge from both branches.
2. **FOD (Feature Orthogonalization Decoupling Strategy)**: An orthogonalization loss constrains the cosine similarity between class-semantic features and spatial-structural features toward zero, ensuring the two knowledge streams remain non-redundant.
3. **Parallel Architecture**: Spatial aggregation and class aggregation operate on the cost volume independently, avoiding cascading effects.

### Loss & Training

Segmentation loss + orthogonalization decoupling loss (constraining the cosine similarity between the two feature streams toward 0).

## Key Experimental Results

### Main Results

| Dataset | Metric | PCA-Seg | DeCLIP | H-CLIP | PartCATSeg |
|--------|------|---------|--------|--------|-----------|
| A-150 | mIoU | **SOTA** | 2nd | 3rd | — |
| PAS-20b | mIoU | **SOTA** | — | — | 2nd |
| ADE20K-Part | mIoU | **SOTA** | — | — | 2nd |

### Ablation Study

| Configuration | A-150 mIoU | Note |
|------|-----------|------|
| Full PCA-Seg | **SOTA** | Parallel + EPL + FOD |
| Serial baseline | −1.5% | Knowledge interference |
| w/o FOD | −0.9% | Redundant dual-branch features |
| Single convolution replacing EPL | −0.2% | Insufficient fusion |

### Parameter Efficiency Analysis

| Component | Extra Parameters | GPU Memory | mIoU Contribution |
|------|---------|---------|--------|
| Parallel branches | 0.25M | 0.72G | +0.8% |
| EPL | 0.08M | 0.18G | +0.5% |
| FOD | 0M (loss only) | 0.06G | +0.9% |
| Total / block | **0.35M** | **0.96G** | +2.2% |

### Key Findings

- FOD yields a +0.9% mIoU gain, demonstrating that orthogonalization constraints effectively reduce redundancy.
- Each parallel block introduces only 0.35M additional parameters and 0.96G GPU memory.
- State-of-the-art performance is achieved on both semantic segmentation and part segmentation tasks.

## Highlights & Insights

- The identification of "serial knowledge interference" offers meaningful insights into understanding cost aggregation.
- The FOD orthogonalization constraint is concise yet effective and can be generalized to other multi-branch architectures.
- The minimal parameter overhead (0.35M per block) makes the method feasible for practical deployment.

## Limitations & Future Work

- The parallel architecture introduces a modest increase in computational cost.
- The orthogonality assumption may be overly strong—in certain scenarios, class semantics and spatial information are reasonably correlated, and enforcing orthogonality may discard useful information.
- Validation is limited to open-vocabulary segmentation; instance segmentation and panoptic segmentation remain untested.
- Cost volume construction depends on the quality of CLIP features, so CLIP's inherent limitations propagate to downstream tasks.
- The weight of the FOD orthogonalization loss requires careful tuning; an excessively large value may suppress useful information.
- The selection of the number of expert parsers in EPL lacks theoretical guidance.
- Integration with recent SAM-based open-vocabulary segmentation methods has not been explored.

## Related Work & Insights

- **vs. CATSeg / PartCATSeg**: These methods adopt serial architectures that suffer from knowledge interference; PCA-Seg's parallel design eliminates this issue.
- **vs. DeCLIP**: DeCLIP fine-tunes CLIP attention layers, whereas PCA-Seg innovates at the cost aggregation level.

### Additional Discussion

- The core contribution of this work lies in transforming the problem from a single-dimensional analysis to a multi-dimensional one, providing a more comprehensive understanding.
- The experimental design covers diverse scenarios and baseline comparisons, with statistically significant results.
- The modular design of the method facilitates extension to related tasks and new datasets.
- Open-sourcing the code and data is of significant value to the community for reproducibility and follow-up research.
- Compared to concurrent works, this paper demonstrates greater depth in problem formulation and comprehensiveness in experimental analysis.
- The paper is logically structured, forming a complete loop from problem definition to method design to experimental validation.

## Rating

- Novelty: ⭐⭐⭐⭐ Parallel cost aggregation and FOD strategy are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 8 benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Motivation figures are clear and problem definition is precise.
- Value: ⭐⭐⭐⭐ Makes a practical contribution to open-vocabulary segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SPAR: Single-Pass Any-Resolution ViT for Open-Vocabulary Segmentation](spar_single-pass_any-resolution_vit_for_open-vocabulary_segmentation.md)
- [\[CVPR 2026\] SDDF: Specificity-Driven Dynamic Focusing for Open-Vocabulary Camouflaged Object Detection](sddf_specificity-driven_dynamic_focusing_for_open-vocabulary_camouflaged_object.md)
- [\[CVPR 2026\] Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation](spatio-semantic_expert_routing_architecture_with_mixture-of-experts_for_referrin.md)
- [\[CVPR 2026\] GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation](geoguide_hierarchical_geometric_guidance_for_open-vocabulary_3d_semantic_segment.md)
- [\[CVPR 2026\] PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation](pearl_geometry_aligns_semantics_for_training-free_open-vocabulary_semantic_segme.md)

</div>

<!-- RELATED:END -->
