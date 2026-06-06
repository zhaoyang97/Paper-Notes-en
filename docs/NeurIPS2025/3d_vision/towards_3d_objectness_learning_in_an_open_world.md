---
title: >-
  [Paper Note] Towards 3D Objectness Learning in an Open World
description: >-
  [NeurIPS 2025][3D Vision][3D objectness] This paper proposes OP3Det, a class-agnostic open-world 3D detector that requires no text prompts. It leverages 2D foundation models for 3D object discovery and introduces a cross…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "3D objectness"
  - "open-world detection"
  - "class-agnostic"
  - "cross-modal MoE"
  - "SAM"
date: 2026-05-08
content_hash: 2045f5ee642b6bdf
---

# Towards 3D Objectness Learning in an Open World

**Conference**: NeurIPS 2025
**arXiv**: [2510.17686](https://arxiv.org/abs/2510.17686)  
**Code**: [https://github.com/op3det](https://github.com/op3det)  
**Area**: 3D Vision / Open-World Detection
**Keywords**: 3D objectness, open-world detection, class-agnostic, cross-modal MoE, SAM

## TL;DR

This paper proposes OP3Det, a class-agnostic open-world 3D detector that requires no text prompts. It leverages 2D foundation models for 3D object discovery and introduces a cross-modal Mixture-of-Experts (MoE) module to dynamically fuse point cloud and image features, substantially improving recall on novel object categories.

## Background & Motivation

3D perception systems for autonomous driving, robotics, and related applications face a fundamental challenge: object categories in the real world are constantly evolving, and systems must be capable of localizing *all* objects rather than only those seen during training.

**Limitations of Prior Work**:

**Closed-set 3D detectors**: These can only recognize categories predefined at training time and fail entirely when confronted with novel categories.

**Open-vocabulary 3D detectors**: These rely on manually crafted text prompts; when the vocabulary is incomplete or mismatched to the scene, novel-category recall remains poor.

**Scarcity of 3D data**: 3D point cloud datasets are far more limited in scale and category coverage compared to their 2D counterparts.

**Key Challenge**: How can a detector learn generalizable *3D objectness* under severely limited 3D annotation categories, enabling it to discover objects of arbitrary classes?

**Key Insight**: Given that large-scale pretrained 2D foundation models (e.g., SAM) exhibit strong zero-shot generalization, their capabilities can be transferred to the 3D domain to learn open-world 3D objectness. The core ideas are: (1) use SAM for class-agnostic 3D object discovery to augment training data, and (2) design a cross-modal MoE to dynamically fuse multi-modal features for learning generalizable 3D objectness.

## Method

### Overall Architecture

OP3Det follows a two-stage design:
1. **3D Object Discovery** (pre-training): Apply SAM to RGB images to extract class-agnostic masks → multi-scale point sampling and denoising → class-agnostic 2D detector post-processing → project into 3D space to obtain new 3D bounding boxes.
2. **Cross-Modal MoE Training** (training phase): Voxelized point cloud features $F_P$ + image features $F_I'$ + multimodal concatenated features $F_M$ → self-attention encoding → multimodal router assigns weights → modal experts perform weighted fusion → detection head.

### Key Designs

1. **Multi-scale Point Sampling**:

    - **Function**: Addresses the fragmented mask outputs produced by SAM.
    - **Mechanism**: SAM generates masks using a 64×64 uniform grid of point prompts, but the outputs often correspond to object fragments or partial regions. The method first selects source points $(x_s, y_s)$ most likely belonging to objects based on IoU scores and self-supervised model attention values, then filters out neighboring points whose 3D distance exceeds a threshold $\delta$ to ensure local geometric consistency.
    - **Multi-scale Fusion**: Four scales $\delta = (0.2, 0.5, 1, 2)$ are used for separate sampling; results are merged via NMS and further filtered by a class-agnostic 2D detector to suppress noise.
    - **Design Motivation**: A single scale is either insufficiently selective (small $\delta$) or excludes useful objects (large $\delta$); combining multiple scales compensates for the weaknesses of each.

2. **Cross-Modal Mixture-of-Experts (CM-MoE)**:

    - **Function**: Addresses multi-modal fusion degradation in the open-world setting, where naive fusion (concatenation or addition) hurts performance.
    - **Mechanism**: Self-attention independently encodes the three feature streams: $\mathcal{F}_P = \text{SelfAttn}(F_P)$, $\mathcal{F}_I = \text{SelfAttn}(F_I')$, $\mathcal{F}_M = \text{SelfAttn}(F_M)$. A multimodal router $\mathcal{R}$ then computes routing probabilities $(p_P, p_I, p_M) = \mathcal{R}(\mathcal{F}_M)$, and three modal experts perform weighted fusion: $\mathcal{F} = \sum_{i \in (P,I,M)} p_i \cdot \mathcal{E}_i(\mathcal{F}_i)$.
    - **Design Motivation**: In class-agnostic binary classification under open-world conditions, the relative importance of geometric cues (point clouds) and semantic cues (images) varies across scenes. The router enables the model to adaptively determine which modality to rely on, avoiding cross-modal noise interference.

3. **2D→3D Projection for Object Discovery**:

    - **Function**: Maps 2D bounding boxes to 3D space.
    - **Mechanism**: 3D points are projected into 2D using camera intrinsics $K$ and extrinsics $R_t$; points falling within the 2D box are identified and then clustered to produce a 3D bounding box.
    - **Post-processing**: The SAM IoU prediction score is multiplied by the class-agnostic 2D detector's objectness score, and a threshold of 0.6 is applied to filter low-quality discoveries.

### Loss & Training

- The classification loss is a class-agnostic binary loss (foreground vs. background).
- All other losses follow the design of OV-Uni3DETR.
- ResNet50 + FPN serves as the image feature extractor; Sparse 3D ResNet is used as the voxel feature extractor.
- Training uses the AdamW optimizer.
- At inference time, neither SAM nor any additional modules are required; the model runs directly on point cloud–image pairs.

## Key Experimental Results

### Main Results

**Cross-Category Generalization (SUN RGB-D & ScanNet)**:

| Method | Dataset | AR_novel | AR_all | AR_base | AP_all |
|--------|---------|----------|--------|---------|--------|
| FCAF3D (closed) | SUN RGB-D | 65.3 | 86.5 | 92.7 | 62.0 |
| OV-Uni3DETR (open-vocab) | SUN RGB-D | 62.8 | 82.5 | 88.8 | 57.4 |
| **OP3Det (Ours)** | **SUN RGB-D** | **78.8** | **89.7** | **93.1** | **65.4** |
| FCAF3D (closed) | ScanNet | 61.7 | 71.3 | 83.2 | 24.7 |
| OV-Uni3DETR (open-vocab) | ScanNet | 67.6 | 71.6 | 76.5 | 25.9 |
| **OP3Det (Ours)** | **ScanNet** | **79.9** | **83.2** | **87.3** | **28.6** |

OP3Det achieves gains of 13.5% (vs. FCAF3D) and 16.0% (vs. OV-Uni3DETR) on novel categories, respectively.

**Cross-Dataset Generalization**:

| Setting | Method | AR25 | AP25 |
|---------|--------|------|------|
| ScanNet→SUN RGB-D | FCAF3D | 59.3 | 17.9 |
| ScanNet→SUN RGB-D | **OP3Det** | **73.1** | **22.3** |
| SUN RGB-D→ScanNet | FCAF3D | 47.7 | 12.9 |
| SUN RGB-D→ScanNet | **OP3Det** | **77.9** | **21.2** |

In the cross-dataset setting, improvements reach up to 30% in AR25.

### Ablation Study

| SAM | Multi-scale Sampling | CM-MoE | AR_novel | AR_all |
|-----|---------------------|--------|----------|--------|
| ✗ | ✗ | ✗ | 54.2 | 84.0 |
| ✓ | ✗ | ✗ | 50.0 | 74.1 |
| ✓ | ✓ | ✗ | 69.2 | 87.9 |
| ✓ | ✓ | ✓ | **78.8** | **89.7** |

- Adding SAM alone degrades performance due to fragmented masks introducing noise.
- Multi-scale sampling raises AR_novel from 50.0 to 69.2 (+19.2%).
- CM-MoE further improves AR_novel to 78.8 (+9.6%).

| Fusion Strategy | AR_novel | AR_all |
|----------------|----------|--------|
| Point cloud only | 69.2 | 87.9 |
| Feature addition | 65.4 | 85.6 |
| Feature concatenation | 66.0 | 85.8 |
| **CM-MoE** | **78.8** | **89.7** |

Naive fusion strategies underperform the single-modality baseline; only CM-MoE effectively exploits complementary multi-modal information.

### Key Findings

- In the class-agnostic setting, naively concatenating or adding multi-modal features causes RGB features to interfere with 3D geometric cues.
- SAM's fragmented outputs require carefully designed post-processing pipelines before they can be reliably used in 3D scenes.
- The proposed method generalizes directly to outdoor scenes (KITTI) and category-specific detection, demonstrating broad applicability.

## Highlights & Insights

1. **Novel Problem Formulation**: This work is the first to formally define and address class-agnostic open-world 3D object detection.
2. **2D→3D Transfer Strategy**: The approach cleverly leverages the zero-shot capabilities of 2D foundation models to compensate for the scarcity of 3D data.
3. **Multi-scale Point Sampling**: This technique effectively resolves SAM's fragmented output problem and constitutes a key technical contribution for deploying SAM in 3D scenes.
4. **Dynamically Routed MoE**: The CM-MoE addresses the degradation problem of multi-modal fusion in open-world settings, representing a methodological innovation.

## Limitations & Future Work

- SAM incurs high inference costs, making the pre-training object discovery stage time-consuming.
- In outdoor scenes (e.g., KITTI), sparse foreground and large background clutter limit the magnitude of improvement.
- Under the class-agnostic setting, AP metrics cannot be computed per category, limiting evaluation granularity.
- The potential of larger 2D foundation models (e.g., SAM 2) or additional modalities (e.g., depth estimation) remains unexplored.

## Related Work & Insights

- **SAM in 3D**: Methods such as SAM3D and OpenMask3D apply SAM to 3D segmentation, but direct application introduces noise. The multi-scale point sampling strategy proposed here offers important reference for SAM-to-3D transfer.
- **Multi-modal Fusion**: Methods such as BEVFusion and SparseFusion focus on comprehensive fusion; this work highlights that in open-world settings, **dynamic modal selection** is preferable to indiscriminate fusion.
- **Takeaway**: In data-scarce 3D tasks, leveraging 2D foundation models for automatic annotation is a promising direction.

## Rating

- Novelty: ⭐⭐⭐⭐ First to define the class-agnostic open-world 3D detection problem; the framework design is original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers cross-category, cross-dataset, cross-scene, and ablation settings; highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear and method description is thorough.
- Value: ⭐⭐⭐⭐ Makes an important contribution to open-world 3D perception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] EA3D: Online Open-World 3D Object Extraction from Streaming Videos](ea3d_online_open-world_3d_object_extraction_from_streaming_videos.md)
- [\[AAAI 2026\] Open-World 3D Scene Graph Generation for Retrieval-Augmented Reasoning](../../AAAI2026/3d_vision/open-world_3d_scene_graph_generation_for_retrieval-augmented_reasoning.md)
- [\[NeurIPS 2025\] OpenLex3D: A Tiered Evaluation Benchmark for Open-Vocabulary 3D Scene Representations](openlex3d_a_tiered_evaluation_benchmark_for_open-vocabulary_3d_scene_representat.md)
- [\[NeurIPS 2025\] Segment then Splat: Unified 3D Open-Vocabulary Segmentation via Gaussian Splatting](segment_then_splat_unified_3d_open-vocabulary_segmentation_via_gaussian_splattin.md)
- [\[ICML 2026\] SVL: Spike-based Vision-Language Pretraining for Efficient 3D Open-World Understanding](../../ICML2026/3d_vision/svl_spike-based_vision-language_pretraining_for_efficient_3d_open-world_understa.md)

</div>

<!-- RELATED:END -->
