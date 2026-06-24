---
title: >-
  [Paper Note] Mosaic3D: Foundation Dataset and Model for Open-Vocabulary 3D Segmentation
description: >-
  [CVPR 2025][3D Vision][open-vocabulary 3D segmentation] This paper proposes an automated data generation pipeline to construct a large-scale 3D mask-text dataset named Mosaic3D-5.6M (containing 5.6M pairs and 30K scenes). By training a language-aligned 3D encoder and a mask decoder, it achieves the first single-stage open-vocabulary 3D instance segmentation.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "open-vocabulary 3D segmentation"
  - "foundation model"
  - "contrastive learning"
  - "data engine"
  - "mask decoder"
date: 2026-05-08
content_hash: e77c971cf6ecaa9c
---

# Mosaic3D: Foundation Dataset and Model for Open-Vocabulary 3D Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2502.02548](https://arxiv.org/abs/2502.02548)  
**Code**: [https://nvlabs.github.io/Mosaic3D/](https://nvlabs.github.io/Mosaic3D/)  
**Area**: 3D Vision  
**Keywords**: open-vocabulary 3D segmentation, foundation model, contrastive learning, data engine, mask decoder

## TL;DR

This paper proposes an automated data generation pipeline to construct a large-scale 3D mask-text dataset named Mosaic3D-5.6M (containing 5.6M pairs and 30K scenes). By training a language-aligned 3D encoder and a mask decoder, it achieves the first single-stage open-vocabulary 3D instance segmentation.

## Background & Motivation

**Background**: Open-vocabulary 3D scene understanding is a fundamental challenge in computer vision and is essential for robotics, AR/VR, and autonomous driving. Although 2D VLMs have achieved powerful open-vocabulary capabilities via web-scale data, the 3D field is severely constrained by the lack of high-quality training data due to high annotation costs.

**Limitations of Prior Work**:
1. **Imprecise regional boundaries**: Methods such as RegionPLC rely on coarse 2D bounding box detectors, resulting in poor mask boundary quality (Entropy=81.0).
2. **Insufficient text descriptions**: OV3D only generates simple attribute labels (Unique Nouns are limited to 2.5K), lacking fine-grained visual descriptions.
3. **Data scale bottleneck**: Existing datasets only contain a few thousand scenes, which is far below the coverage of 2D datasets.
4. **Instance segmentation relies on closed vocabularies**: Existing open-vocabulary 3D instance segmentation methods rely on closed-vocabulary proposal networks like Mask3D, preventing them from detecting novel categories.

**Core Motivation**: To break through the data bottleneck of open-vocabulary 3D scene understanding, three key requirements must be satisfied simultaneously: precise 3D regional segmentation, rich text descriptions, and sufficient data scale.

## Method

### Overall Architecture

The system consists of two major components: (1) Mosaic3D-5.6M Data Engine, which automatically generates high-quality 3D mask-text pairs from multi-view RGB-D frames using 2D vision foundation models; (2) Mosaic3D Model Training, which adopts a two-stage strategy to first train a language-aligned 3D encoder via contrastive learning, and then train a lightweight mask decoder for instance segmentation.

### Key Designs

#### Module 1: Enhanced Segmentation + Region Description Data Engine

The data generation pipeline comprises three stages:
- **Enhanced Segmentation**: It combines Grounded-SAM (for precise foreground object boundaries) and SEEM (for open-vocabulary panoptic segmentation to handle background stuff like walls and floors) while utilizing RAM++ to automatically detect object categories as text prompts for Grounding-DINO.
- **Enhanced Region Description**: It employs a region-aware VLM (Osprey) to generate detailed captions for each segmentation mask. The captions describe visual attributes and spatial contexts instead of using simple category labels.
- **2D-3D Association**: It projects 2D masks into 3D point clouds based on camera parameters, and performs an inclusion test using depth thresholds to obtain 3D mask-text pairs.

Applying this pipeline to five datasets (ScanNet, ARKitScenes, ScanNet++, Matterport3D, and Structured3D) yields 30K scenes and 5.6M mask-text pairs.

#### Module 2: Contrastive Learning for Language-Aligned 3D Encoder

A SparseUNet34C is adopted as the 3D backbone. The features of each 3D point are aligned with the corresponding caption text embedding via a point-level contrastive loss. The text encoder employs Recap-CLIP, which supports long text descriptions. The loss function weights each 3D region mask to ensure that regions of different sizes are optimized with equal weight. Point Prompt Training (PPT) is further integrated to enhance joint training across multiple datasets.

#### Module 3: Caption Merging + Mask Decoder

- **Caption Merging**: Multi-view mask-caption data and class-agnostic 3D masks from Segment3D are merged using IoU matching, associating multiple captions with each Segment3D mask.
- **Mask Decoder**: Mask3D (a transformer-based architecture) is used as the mask decoder. It takes position-encoded sampled queries and language-aligned features from the backbone as inputs to output mask embeddings.
- **Key Innovation**: This is the first single-shot open-vocabulary 3D instance segmentation framework that operates without relying on closed-vocabulary proposal networks or requiring ground truth labels.

### Loss & Training

**Stage 1 - Language Alignment**:
$$\mathcal{L}_{point} = -\frac{1}{K}\sum_{k=1}^{K}\sum_{i=1}^{N}(s_k)_i \log\frac{\exp(z_i^{3D}\cdot z_k^{text}/\tau)}{\sum_{j=1}^{K}\exp(z_i^{3D}\cdot z_j^{text})}$$

**Stage 2 - Mask Decoder**:
$$\mathcal{L}_{mask} = \lambda_{obj}\mathcal{L}_{obj} + \lambda_{dice}\mathcal{L}_{dice} + \lambda_{bce}\mathcal{L}_{bce} + \lambda_{cap}\mathcal{L}_{cap}$$
where $\mathcal{L}_{cap}$ denotes the contrastive loss between mask embeddings and caption embeddings, with $\lambda_{obj}=2, \lambda_{dice}=5, \lambda_{bce}=2, \lambda_{cap}=1$.

## Key Experimental Results

### Main Results

**Open-Vocabulary 3D Semantic Segmentation** (f-mIoU / f-mAcc):

| Method | ScanNet20 | ScanNet200 | ScanNet++ | Matterport3D |
|------|-----------|------------|-----------|--------------|
| RegionPLC | 59.6/77.5 | 9.1/17.3 | - | - |
| OV3D | 64.0/76.3 | 8.7/- | - | - |
| **Mosaic3D (SN only)** | **65.0/82.5** | **13.0/24.5** | **16.2/27.1** | **8.6/17.8** |
| **Mosaic3D (full)** | **68.1/84.4** | **15.7/28.3** | **18.0/29.0** | **13.1/27.7** |

**Open-Vocabulary 3D Instance Segmentation** (mAP on ScanNet200):
- Using Mask3D proposals: Mosaic3D achieves 11.8 mAP, outperforming OpenIns3D (8.8) by +3.0p.
- Using Segment3D proposals: 2.7 mAP.
- **Mosaic3D w/ Decoder (First single-stage)**: 3.9 mAP, with a latency of only 1.2s.

### Ablation Study

Data engine component ablation (ScanNet only, f-mIoU/f-mAcc):

| Segmentation Scheme | Description Scheme | ScanNet20 | ScanNet200 |
|----------|----------|-----------|------------|
| Detic + Kosmos-2 | General caption | 52.3/73.2 | 7.4/14.2 |
| RAM++ + G-SAM + Ferret | region-aware | 59.6/79.2 | 9.0/17.8 |
| **RAM++ + G-SAM + SEEM + Osprey** | **Final choice** | **65.0/82.5** | **13.0/24.5** |

**Data scale ablation**: Incorporating datasets like ARKitScenes and ScanNet++ consistently improves the ScanNet200 f-mIoU from approximately 10.5 to 15.7, validating the vital role of scale.

### Key Findings

1. Both data quality (precise masks + descriptive captions) and data scale are critical to performance improvements.
2. The joint SEEM+Grounded-SAM segmentation outperforms using either model individually by successfully complementing foreground objects and background stuff.
3. Captions generated by region-aware VLMs are significantly superior to those from generic image captioning models.
4. The single-stage mask decoder bypasses high latencies of multi-view CLIP inference, reducing runtime from 47.3s to 1.2s.

## Highlights & Insights

1. **Valuable Data Engine Paradigm**: The pipeline of utilizing 2D baseline models to automatically annotate 3D data provides a clear path for other 3D applications (e.g., 3D grounding, navigation).
2. **Scale Matters**: Training on 5.6M pairs vs. tens of thousands of pairs brings qualitative changes. High-quality captions per scene are more effective than simply increasing the number of scenes (outperforming SceneVerse which uses more scenes).
3. **First Single-Stage Open-Vocabulary 3D Instance Segmentation**: It eliminates dependencies on closed-vocabulary proposal networks, substantially streamlining the workflow.
4. **Practical Efficiency**: Inference using only 3D representation takes only 1.2s per scene, compared to 33–285s required by 2D+3D methods.

## Limitations & Future Work

1. The data generation quality remains bounded by the performance limits of the selected 2D foundation models (SAM, Osprey, etc.).
2. The performance of the single-stage mask decoder (3.9 mAP) still trails significantly behind methods that employ 2D CLIP (23.7 mAP).
3. The training features only indoor scene datasets, leaving generalization to outdoor or large-scale scenes unverified.
4. The IoU matching threshold used in caption merging is globally fixed and may not fit all scene layouts.

## Related Work & Insights

- **OpenScene**: The pioneering zero-shot 3D semantic segmentation work (distilling CLIP into 3D). Mosaic3D surpasses it by utilizing contrastive learning on large-scale datasets.
- **RegionPLC / OV3D**: Earlier automatic annotation studies, which however suffer from limited segmentation precision and lack detailed captions.
- **Segment3D**: Offers class-agnostic 3D mask proposals, which Mosaic3D leverages for training the instance segmentation task.
- **Insights**: Future endeavors could scale the data engine towards outdoor scenes (e.g., ScanNet-style to UrbanScene3D) or explore video-based 3D annotations to improve mask consistency.

## Rating

⭐⭐⭐⭐ — Highly systematic engineering. The data engine and model framework are well-rounded and report state-of-the-art results across several benchmarks. The single-stage instance segmentation is a major contribution, though the core conceptual novelty is rooted in complex system integration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Masked Point-Entity Contrast for Open-Vocabulary 3D Scene Understanding](masked_point-entity_contrast_for_open-vocabulary_3d_scene_understanding.md)
- [\[CVPR 2025\] JOPP-3D: Joint Open Vocabulary Semantic Segmentation on Point Clouds and Panoramas](jopp-3d_joint_open_vocabulary_semantic_segmentation_on_point_clouds_and_panorama.md)
- [\[CVPR 2025\] Open-Vocabulary Functional 3D Scene Graphs for Real-World Indoor Spaces](open-vocabulary_functional_3d_scene_graphs_for_real-world_indoor_spaces.md)
- [\[CVPR 2025\] Reconstructing In-the-Wild Open-Vocabulary Human-Object Interactions](reconstructing_in-the-wild_open-vocabulary_human-object_interactions.md)
- [\[CVPR 2025\] SeeGround: See and Ground for Zero-Shot Open-Vocabulary 3D Visual Grounding](seeground_see_and_ground_for_zero-shot_open-vocabulary_3d_visual_grounding.md)

</div>

<!-- RELATED:END -->
