---
title: >-
  [Paper Note] Bi-directional Contextual Attention for 3D Dense Captioning
description: >-
  [ECCV2024 (Oral)][3D Vision][3D Dense Captioning] This paper proposes BiCA, which decouples and parallelly decodes instance queries and context queries via a bi-directional contextual attention mechanism. This solves the objective conflict between localization and caption generation in 3D dense captioning, achieving state-of-the-art (SOTA) performance on both the ScanRefer and Nr3D benchmarks.
tags:
  - "ECCV2024 (Oral)"
  - "3D Vision"
  - "3D Dense Captioning"
  - "Transformer"
  - "Contextual Attention"
  - "Point Cloud"
  - "Scene Understanding"
date: 2026-05-08
content_hash: a35ad3707ded85b8
---

# Bi-directional Contextual Attention for 3D Dense Captioning

**Conference**: ECCV2024 (Oral)  
**arXiv**: [2408.06662](https://arxiv.org/abs/2408.06662)  
**Authors**: Minjung Kim, Hyung Suk Lim, Soonyoung Lee, Bumsoo Kim, Gunhee Kim  
**Institutions**: Princeton University, LG AI Research, Seoul National University  
**Code**: To be confirmed  
**Area**: 3D Vision  
**Keywords**: 3D Dense Captioning, Transformer, Contextual Attention, Point Cloud, Scene Understanding

## TL;DR

This paper proposes BiCA, which decouples and parallelly decodes instance queries and context queries via a bi-directional contextual attention mechanism. This solves the objective conflict between localization and caption generation in 3D dense captioning, achieving state-of-the-art (SOTA) performance on both the ScanRefer and Nr3D benchmarks.

## Background & Motivation

3D dense captioning requires localizing all objects in a 3D scene and generating natural language descriptions for each object. Existing methods have two key limitations in constructing contextual information:

1. **Limited contextual scope**: Existing methods obtain context only by modeling object-pair relationships or aggregating nearest-neighbor features. However, spatial relationships between objects span the entire global scene, rather than being restricted to the vicinity of the objects themselves.
2. **Objective conflict**: The localization task requires compact local features to accurately bound object boundaries, whereas caption generation (especially for descriptions involving global spatial relationships) requires contextual features of the global scene. Using a single query set to serve both tasks simultaneously leads to mutual interference.

For example, describing "the chair located at the northernmost-west corner of the room" requires global spatial understanding, but localizing the chair precisely requires compact local features—both of which are difficult to reconcile in a single representation.

## Core Problem

How to design an architecture that effectively aggregates relevant contextual features of the global scene without compromising localization performance, thereby simultaneously improving both localization and caption generation performance in 3D dense captioning?

## Method

### Overall Architecture

BiCA adopts a Transformer encoder-decoder architecture. The core idea is to decouple instance queries (objects) and context queries (non-object context) into two parallel streams, and then perform information interaction through bi-directional attention.

### 1. Encoder

The scene encoder from 3DETR is adopted. The input point cloud is tokenized by set-abstraction layers of PointNet++, and then fed into a masked transformer encoder with set-abstraction, followed by two additional encoding layers, outputting scene tokens $p_{enc} \in \mathbb{R}^{1024 \times 3}$ and $f_{enc} \in \mathbb{R}^{1024 \times 256}$.

### 2. Dual-path Query Generator

- **Instance Query Generator**: Learns voting offsets via an FFN to shift encoded points toward object centers, and then extracts 256 instance queries $(p_o, f_o)$ using set-abstraction with a radius of 0.3. Unlike Vote2Cap-DETR, this method extracts features from the voted candidate coordinates, which prevents multiple queries from focusing on the same object.
- **Context Query Generator**: Uses Farthest Point Sampling (FPS) on the encoded tokens to select 512 seed points, and extracts context queries $(p_c, f_c)$ using set-abstraction with a radius of 1.2. The large radius design allows each context query to capture fine-grained geometry and structural information over a larger range, encoding the spatial relations between objects and between objects and the scene.

### 3. Parallel Decoder

The Instance Decoder and Context Decoder each contain 8 transformer decoder layers, encoding XYZ coordinates using Fourier positional encodings. The two decoders operate independently: the Instance Decoder focuses on object detection and attribute features, while the Context Decoder captures structural context from non-object regions.

### 4. Bi-directional Contextual Attention

This is the core contribution of this work, divided into two stages:

**O4C (Objects for Context)**: Constructs Object-aware Context $V_{ac}$ for each object. Attention weights are computed between each instance query and all context queries to produce a weighted sum of context features. Intuitively, this identifies geometric region information relevant to the current object in the global context, scaled by a learnable parameter $\gamma$.

**C4O (Contexts for Object)**: Constructs Context-aware Object $V_{ao}$. Attention is computed between object-aware context features and instance queries to generate a weighted sum of instance features. This step concretizes the ambiguous relationship of "next to" into specific contexts like "the red chair next to it," regulated by a learnable parameter $\lambda$.

Finally, $(V_o, V_{ac}, V_{ao})$ are concatenated to form $V_a$, which is fed into the caption generation head.

### 5. Localization and Caption Generation

- **Localization**: Regresses the object center offset and bounding box size using the decoded instance query $V_o$ via 5 MLP heads.
- **Caption Generation**: Based on a GPT-2 transformer decoder caption head with 2 decoder blocks. It uses $V_a$ as the prefix instead of the standard SOS token, employing beam search (beam size=5) during inference.

### 6. Loss & Training

Three-stage training:
1. Pre-train the detector (without caption head) on ScanNet for 1080 epochs.
2. Jointly train on ScanRefer/Nr3D using cross-entropy loss (MLE) for 720 epochs.
3. Fine-tune the caption head using Self-Critical Sequence Training (SCST) for 180 epochs.

The loss function is $\mathcal{L} = \beta_1 \mathcal{L}_o + \beta_2 \sum \mathcal{L}_{det}^i + \beta_3 \mathcal{L}_{cap}$, where $\beta_1=10, \beta_2=1, \beta_3=5$.

## Key Experimental Results

### Main Results (SCST, without extra 2D data)

| Method | C@0.25 | C@0.5 | B-4@0.5 | M@0.5 | R@0.5 |
|------|--------|-------|---------|-------|-------|
| Vote2Cap-DETR | 84.15 | 73.77 | 38.21 | 26.64 | 54.71 |
| Vote2Cap-DETR++ | 88.28 | 78.16 | 39.72 | 26.94 | 55.52 |
| **BiCA** | **89.72** | **80.14** | **40.16** | **27.76** | **56.10** |

BiCA outperforms Vote2Cap-DETR++ by **+1.98** and Vote2Cap-DETR by **+6.37** on CIDEr@0.5.

### Nr3D Main Results (SCST, IoU=0.5)

| Method | C | B-4 | M | R |
|------|---|-----|---|---|
| Vote2Cap-DETR++ | 47.62 | 28.41 | 25.63 | 54.77 |
| **BiCA** | **49.81** | **28.83** | **25.85** | **56.46** |

### Ablation Study (ScanRefer, SCST, IoU=0.5)

| Configuration | CIDEr | mAP | AR |
|------|-------|-----|-----|
| Vote2Cap-DETR | 73.77 | 45.56 | 67.77 |
| BiCA ($V_o$ only) | 74.90 | 50.12 | 69.49 |
| BiCA ($V_o$ + KNN($V_c$)) | 79.03 | 55.95 | 69.62 |
| BiCA ($V_o$ + $V_{ac}$) | 81.22 | 56.91 | 70.38 |
| **BiCA ($V_o$ + $V_{ac}$ + $V_{ao}$)** | **85.14** | **57.58** | **72.68** |

Each component brings positive gains. The complete O4C+C4O outperforms using only KNN context by **+6.11** CIDEr.

### Model Efficiency

- Number of parameters: 16.9M
- Inference time: 1.8ms/scene (on a single Titan RTX)

## Highlights

1. **Elegant decoupled design**: Separating instance queries and context queries fundamentally solves the objective conflict between localization and caption generation—localization relies on instance queries to maintain accuracy, while captioning improves quality by incorporating contextual features.
2. **Effective bi-directional attention**: The two-stage design of O4C and C4O not only captures global geometric context but also associates the context with specific objects, transforming vague descriptions like "next to" into specific ones like "the red chair next to it."
3. **Improved Instance Query Generator**: Extracting features from voted candidate coordinates (rather than after FPS and then voting) increases the number of matching candidates from 1498 to 1540, directly improving detection performance.
4. **Context Query design**: Using set-abstraction with a large radius (1.2 vs. 0.3 for instances) to extract structural information from non-object regions, effectively encoding the global spatial relationships of the scene.
5. **Oral paper**, achieving SOTA on all metrics across both benchmarks.

## Limitations & Future Work

1. **Limited to indoor scenes**: Evaluated and trained on the ScanNet dataset; the capability to generalize to large-scale outdoor 3D scenes remains unknown.
2. **Dependency on point cloud quality**: Sparse or noisy point clouds might degrade the quality of context queries.
3. **Fixed number of context queries**: 512 context queries and a radius of 1.2 were empirically determined; different scene scales might require adaptive adjustments.
4. **Simplistic caption head**: Utilizes only a 2-layer GPT-2 decoder; integrating stronger language models or multimodal pre-training could further enhance caption quality.
5. **Lack of exploration with LLMs**: Current caption generation is based on relatively small language models, and integration with large language models warrants further investigation.

## Related Work

| Method | Context Scope | Query Design | Decoupling of Loc/Cap | ECCV/CVPR |
|------|-----------|-----------|------------|-----------|
| Scan2Cap | Object-pair relations | None (Two-stage) | ✗ | CVPR 2021 |
| 3DJCG | Object-pair + Graph Attention | Unified query | ✗ | CVPR 2022 |
| Vote2Cap-DETR | Nearest neighbors | FPS + Voting | ✗ | CVPR 2023 |
| Vote2Cap-DETR++ | Nearest neighbors | Decoupled loc/cap query | Partial | TPAMI 2024 |
| **BiCA** | **Global scene** | **Instance + Context dual-path** | **✓** | **ECCV 2024** |

Although Vote2Cap-DETR++ decouples localization and caption queries, its decoupled queries are still projections of object-centric queries, which are limited by the object-centric design. BiCA structurally achieves true separation between object features and non-object contexts.

## Related Work & Insights

1. **Transferability of Query Decoupling**: Parsing query sets into target detection and context understanding paths for parallel decoding is a concept that can be adapted to other tasks requiring simultaneous localization and understanding (e.g., 3D Visual Grounding, Open-vocabulary 3D Detection).
2. **Bi-directional Attention Paradigm**: The two-stage O4C → C4O information flow can be generalized to other scenarios requiring local-to-global feature interaction.
3. **Importance of Non-object Regions**: Explicitly modeling the spatial structures of non-object regions aids in understanding scene relationships, which offers valuable references for other 3D scene understanding tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ — The bi-directional contextual attention design is novel with clear motivation.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two benchmarks, multiple configurations, and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured and intuitive illustrations.
- Value: ⭐⭐⭐⭐ — Oral paper, SOTA, with highly transferable decoupling insights.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Curvature-Aware Captioning: Leveraging Geodesic Attention for 3D Scene Understanding](../../CVPR2026/3d_vision/curvature-aware_captioning_leveraging_geodesic_attention_for_3d_scene_understand.md)
- [\[ECCV 2024\] View Selection for 3D Captioning via Diffusion Ranking](view_selection_for_3d_captioning_via_diffusion_ranking.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] ScatterFormer: Efficient Voxel Transformer with Scattered Linear Attention](scatterformer_efficient_voxel_transformer_with_scattered_linear_attention.md)
- [\[ICCV 2025\] From One to More: Contextual Part Latents for 3D Generation](../../ICCV2025/3d_vision/from_one_to_more_contextual_part_latents_for_3d_generation.md)

</div>

<!-- RELATED:END -->
