---
title: >-
  [Paper Note] CORE-3D: Context-aware Open-vocabulary Retrieval by Embeddings in 3D
description: >-
  [ICLR 2026][3D Vision][Open-vocabulary 3D semantic segmentation] This paper proposes CORE-3D, a training-free open-vocabulary 3D semantic segmentation and natural language object retrieval pipeline that achieves state-of-the-art performance on Replica and ScanNet through progressive multi-granularity mask generation, context-aware CLIP encoding, and multi-view 3D fusion.
tags:
  - ICLR 2026
  - 3D Vision
  - Open-vocabulary 3D semantic segmentation
  - scene graph
  - CLIP embeddings
  - language retrieval
  - SemanticSAM
date: 2026-05-08
content_hash: be7b167b0f19fb08
---

# CORE-3D: Context-aware Open-vocabulary Retrieval by Embeddings in 3D

**Conference**: ICLR 2026
**arXiv**: [2509.24528](https://arxiv.org/abs/2509.24528)
**Code**: To be confirmed
**Area**: 3D Vision
**Keywords**: Open-vocabulary 3D semantic segmentation, scene graph, CLIP embeddings, language retrieval, SemanticSAM

## TL;DR
This paper proposes CORE-3D, a training-free open-vocabulary 3D semantic segmentation and natural language object retrieval pipeline that achieves state-of-the-art performance on Replica and ScanNet through progressive multi-granularity mask generation, context-aware CLIP encoding, and multi-view 3D fusion.

## Background & Motivation

**Background**: 3D scene understanding is a fundamental requirement for robotics and embodied AI. A dominant paradigm has emerged that combines vision-language models (VLMs) with 2D segmentation models, achieving zero-shot open-vocabulary 3D semantic mapping via back-projection into 3D space.

**Limitations of Prior Work**:
- 2D segmentation backbones such as SAM produce fragmented or incomplete masks in cluttered indoor scenes, leading to severe over-segmentation.
- Applying CLIP encoding directly to individual cropped mask regions provides very limited semantic context, resulting in poor embedding quality.
- When aggregating across multiple frames, the same object receives different contextual embeddings due to viewpoint variation, causing inconsistency.

**Key Challenge**: Existing foundation model pipelines, while training-free, suffer from insufficient segmentation quality and semantic embedding quality, making it difficult to construct coherent and reliable 3D semantic maps.

**Goal**: To simultaneously improve 2D segmentation quality, semantic embedding richness, and multi-view consistency without any training.

**Key Insight**: Leveraging the adjustable granularity of SemanticSAM for progressive refinement, combined with multiple contextual crop strategies to enhance CLIP encoding.

**Core Idea**: Construct high-quality zero-shot open-vocabulary 3D semantic maps via progressive granularity segmentation, multi-crop context-aware CLIP encoding, and 3D voxel merging.

## Method

### Overall Architecture
Given an RGB-D sequence and camera poses, the pipeline consists of four stages: (1) progressive multi-granularity mask generation; (2) context-aware CLIP embedding computation; (3) 3D mask merging and refinement; and (4) natural language object retrieval. The final output is a semantically annotated 3D point cloud supporting open-vocabulary segmentation and language-query-based retrieval.

### Key Designs

1. **Progressive Mask Generation**

   - **Function**: Replaces vanilla SAM to generate more accurate and complete 2D instance masks.
   - **Mechanism**: Exploits the granularity parameter $g$ of SemanticSAM, generating masks over an increasing granularity sequence $\{g_1, g_2, \ldots, g_K\}$. At each level, only masks with confidence exceeding threshold $\tau_{cer}$ are retained, and a new mask is added only if its overlap with existing masks satisfies $\frac{|m \cap m'|}{|m|} < \tau_k$. Coarser granularities capture large objects, while finer granularities progressively recover small objects and fine-grained details.
   - DBSCAN clustering in 3D projected space is further applied to separate objects that are adjacent in 2D but spatially separated in 3D (e.g., a vase overlapping a sofa in the image plane).
   - **Design Motivation**: Addresses SAM's fragmentation problem in cluttered scenes while avoiding redundant masks.

2. **Context-Aware CLIP Embedding**

   - **Function**: Generates semantically rich embedding vectors for each mask.
   - **Mechanism**: For each mask, five complementary crops are extracted: mask crop (background zeroed out), bbox crop, large crop (2.5× expansion), huge crop (4× expansion), and surroundings crop (3× expansion with the object itself masked out). Each crop is encoded by a CLIP image encoder and fused with learned weights: $\mathbf{e}(m) = w_{mask}\mathbf{e}^{mask} + w_{bbox}\mathbf{e}^{bbox} + w_{large}\mathbf{e}^{large} + w_{huge}\mathbf{e}^{huge} - w_{sur}\mathbf{e}^{sur}$
   - Crucially, the surroundings embedding is subtracted with a **negative weight**, producing a contrastive effect that penalizes features dominated by the surrounding context rather than the object itself.
   - **Design Motivation**: Isolated mask crops provide insufficient context for accurate CLIP matching.

3. **3D Mask Merging and Refinement**

   - **Function**: Merges multi-view 2D masks in 3D space into a unified object representation.
   - **Mechanism**: Computes the volumetric Intersection over Volume (IoV) between candidate mask pairs. Two masks are merged if both directional IoV values exceed threshold $\gamma$ and their difference is smaller than $\delta$: $\text{IoV}(m_a, m_b) > \gamma$ and $|\text{IoV}(m_a, m_b) - \text{IoV}(m_b, m_a)| < \delta$.
   - The symmetric balance criterion prevents degenerate merges (e.g., a small cushion being absorbed by a large sofa). Merged embeddings are averaged.

4. **Object Retrieval**

   - **Function**: Localizes target objects in a 3D scene given natural language queries.
   - **Mechanism**: A four-stage pipeline — an LLM parses the query into a structured form $\Pi(q) = (m, \mathcal{R}, \Omega)$ (target, reference objects, orientation constraints) → CLIP similarity-based Top-K candidate mining → VLM visual verification (querying with bounding boxes from the best viewpoint) → orientation inference (discretized yaw angle grid with VLM selection) → LLM final reasoning and output.

### Loss & Training
The method is entirely training-free and operates as a zero-shot inference pipeline, relying on pretrained SemanticSAM, CLIP (Eva02-L), and VLM/LLM components.

## Key Experimental Results

### Main Results

| Dataset | Metric | CORE-3D | BBQ-CLIP (Prev. SOTA) | Gain |
|--------|------|---------|-------------------|------|
| Replica | mIoU | 0.29 | 0.27 | +0.02 |
| Replica | fmIoU | 0.56 | 0.48 | +0.08 |
| ScanNet | mIoU | 0.36 | 0.34 | +0.02 |
| ScanNet | fmIoU | 0.46 | 0.36 | +0.10 |
| ScanNet | mAcc | 0.61 | 0.56 | +0.05 |

CORE-3D achieves even larger improvements on the Sr3D+ object retrieval task:

| Metric | CORE-3D | BBQ (Prev. SOTA) | Gain |
|------|---------|-------------|------|
| Overall A@0.1 | 41.8 | 34.2 | +7.6 |
| Overall A@0.25 | 35.6 | 22.7 | +12.9 |

### Ablation Study
- Progressive multi-granularity segmentation substantially outperforms vanilla SAM and single-granularity SemanticSAM.
- Context-aware CLIP encoding, particularly the surroundings negative-weight subtraction, yields notable segmentation quality gains.
- DBSCAN 3D clustering effectively resolves objects that are spatially separated in 3D but overlapping in 2D.
- The VLM verification step improves retrieval precision.

## Highlights & Insights
- A fully training-free zero-shot pipeline with strong practical utility.
- Progressive granularity refinement is a simple yet effective mask generation strategy.
- The surroundings negative-weight subtraction in context-aware CLIP encoding is an intuitively well-motivated design choice.
- The multi-stage LLM+VLM reasoning pipeline for retrieval is well-structured and principled.

## Limitations & Future Work
- The method relies on SemanticSAM's granularity parameters and multiple thresholds ($\tau_{cer}$, $\tau_k$, $\gamma$, $\delta$), requiring non-trivial hyperparameter tuning.
- The five crop weights in CLIP encoding require empirical calibration and may need adjustment across different scene types.
- The retrieval pipeline depends on external LLM and VLM API calls, incurring notable latency and cost.
- Validation is limited to indoor scenes (Replica/ScanNet); generalization to large outdoor environments remains unexplored.
- Despite meaningful fmIoU improvements, absolute values remain modest, leaving a gap before practical deployment.

## Related Work & Insights
- **vs. ConceptFusion/ConceptGraphs**: CORE-3D surpasses these methods through improved segmentation and embedding quality, demonstrating substantial headroom for improvement in the segmentation and encoding stages of foundation model pipelines.
- **vs. BBQ**: BBQ uses 3D scene graphs with LLM-based reasoning for retrieval and performs competitively; CORE-3D achieves clearly better segmentation and substantially larger retrieval gains (A@0.25: 22.7 → 35.6).
- **vs. HOV-SG**: CORE-3D outperforms this hierarchical scene graph approach on Replica in terms of IoU.
- **vs. training-based methods (LERF/LangSplat/OpenGaussian)**: CORE-3D's zero-shot approach surpasses per-scene training methods on multiple metrics.

**Broader Insights**:
- The context-aware encoding strategy can be generalized to other CLIP-dependent applications such as image retrieval and open-vocabulary detection.
- The contrastive surroundings subtraction design is a transferable technique for embedding disambiguation.
- Progressive granularity segmentation is a promising strategy extendable to video segmentation settings.

## Rating
- **Novelty**: ⭐⭐⭐ (Individual components are not novel in isolation, but their combination is well-motivated and effective.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Multi-dataset evaluation, ablation studies, and qualitative results.)
- **Writing Quality**: ⭐⭐⭐⭐
- **Value**: ⭐⭐⭐⭐ (High practical value as a training-free pipeline.)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation](geopurify_a_data-efficient_geometric_distillation_framework_for_open-vocabulary_.md)
- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](../../ICCV2025/3d_vision/open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[AAAI 2026\] Open-World 3D Scene Graph Generation for Retrieval-Augmented Reasoning](../../AAAI2026/3d_vision/open-world_3d_scene_graph_generation_for_retrieval-augmented_reasoning.md)
- [\[AAAI 2026\] OpenScan: A Benchmark for Generalized Open-Vocabulary 3D Scene Understanding](../../AAAI2026/3d_vision/openscan_a_benchmark_for_generalized_open-vocabulary_3d_scene_understanding.md)
- [\[CVPR 2026\] Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation](../../CVPR2026/3d_vision/context-nav_context-driven_exploration_and_viewpoint-aware_3d_spatial_reasoning_.md)

<!-- RELATED:END -->
