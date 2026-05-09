---
title: >-
  [Paper Note] Language Driven Occupancy Prediction (LOcc)
description: >-
  [ICCV 2025][Autonomous Driving][open-vocabulary occupancy] This paper proposes LOcc, an effective and generalizable open-vocabulary occupancy (OVO) prediction framework. Its core contribution is a semantic transitive labeling pipeline (LVLM + OV-Seg → LiDAR → voxel) that generates dense, fine-grained 3D language occupancy pseudo-GT, replacing the noisy and sparse intermediate feature distillation used in prior work. LOcc comprehensively surpasses state-of-the-art methods on Occ3D-nuScenes.
tags:
  - ICCV 2025
  - Autonomous Driving
  - open-vocabulary occupancy
  - language-driven
  - semantic transitive labeling
  - 3D perception
  - occupancy prediction
  - CLIP
date: 2026-05-08
content_hash: 609c8985a9aa09a1
---

# Language Driven Occupancy Prediction (LOcc)

**Conference**: ICCV 2025
**arXiv**: [2411.16072](https://arxiv.org/abs/2411.16072)
**Code**: [https://github.com/pkqbajng/LOcc](https://github.com/pkqbajng/LOcc)
**Institution**: Zhejiang University, CaiNiao (Alibaba)
**Area**: Autonomous Driving / Occupancy Prediction / Open-Vocabulary
**Keywords**: open-vocabulary occupancy, language-driven, semantic transitive labeling, 3D perception, occupancy prediction, CLIP

## TL;DR
This paper proposes LOcc, an effective and generalizable open-vocabulary occupancy (OVO) prediction framework. Its core contribution is a semantic transitive labeling pipeline (LVLM + OV-Seg → LiDAR → voxel) that generates dense, fine-grained 3D language occupancy pseudo-GT, replacing the noisy and sparse intermediate feature distillation used in prior work. LOcc comprehensively surpasses state-of-the-art methods on Occ3D-nuScenes.

## Background & Motivation
Vision-based occupancy prediction is a core task in 3D perception for autonomous driving, requiring estimation of scene geometry and semantics from image inputs. Conventional supervised methods are constrained to a fixed set of semantic categories, and constructing dense ground truth requires per-frame annotation of LiDAR point clouds, entailing prohibitively high labor costs. Open-vocabulary occupancy (OVO) aims to predict occupancy states for arbitrary vocabulary sets using only unannotated data during training.

Existing OVO methods suffer from two critical limitations:
1. **Use of image features as an intermediate medium**: Features of the same object category vary across different images (encoding both semantics and appearance), leading to inconsistent semantic representations and high noise.
2. **Model-view projection based on voxels**: Projecting voxels directly onto the image plane to obtain labels ignores occlusion relationships and relies on single-frame images, resulting in sparse and coarse voxel-to-text correspondences.

## Core Problem
How to generate dense and fine-grained 3D language occupancy pseudo-GT?

## Method

### Overall Architecture
LOcc consists of two main components: a **semantic transitive labeling pipeline** (pseudo-GT generation) and **OVO model training**.

### Part 1: Semantic Transitive Labeling Pipeline (Core Contribution)

#### Step 1: Vocabulary Extraction (LVLM)
- An LVLM such as Qwen-VL is applied to each frame of surround-view images via chain-of-thought dialogue.
- The model is first prompted to describe the scene, then asked to enumerate all object category names.
- Results from multi-frame surround-view images are merged to produce the complete vocabulary set for that frame.

#### Step 2: Pixel-Text Association (OV-Seg)
- An open-vocabulary segmentation model (e.g., FC-CLIP / SAN / CAT-Seg) is employed.
- Each pixel is matched to the highest-scoring text label via cosine similarity.
- Output: each pixel carries a consistent text label rather than a feature vector.

#### Step 3: LiDAR Point Cloud Label Transfer
- Unannotated LiDAR points are projected onto the image plane to acquire the text labels of corresponding pixels.
- **Key improvement**: Occlusion relationships are explicitly modeled to prevent occluded points from receiving incorrect labels.

#### Step 4: Scene Reconstruction and Voxelization
- Multi-frame LiDAR point clouds are aggregated for temporally dense scene reconstruction.
- Each voxel is assigned a label via **majority voting**, selecting the most frequently occurring label.
- This reduces the influence of single-frame segmentation noise.

### Part 2: OVO Model Architecture

#### Language Autoencoder (Dimensionality Reduction)
- CLIP embeddings have high dimensionality (512/768); a text autoencoder is designed to compress them into a low-dimensional latent space.

#### Occupancy Prediction Model Adaptation
- Built upon existing supervised occupancy models (BEVFormer / BEVDet / BEVDet4D).
- The original classification head is replaced with a geometry head (binary occupancy state) and a language head (low-dimensional language features).
- At inference, predicted features are matched against arbitrary text embeddings via cosine similarity.

### Loss & Training
- Geometry loss: Binary cross-entropy for occupancy state prediction.
- Language loss: Cosine similarity loss between predicted language features and pseudo-GT language labels.

## Key Experimental Results

### OVO Performance on Occ3D-nuScenes

| Method | Backbone | Input Resolution | mIoU |
|--------|----------|-----------------|------|
| POP-3D | R101 | 900×1600 | 11.70 |
| VEON | ViT-B | 900×1600 | 16.78 |
| VEON (temporal) | ViT-B | 900×1600 | 17.51 |
| **LOcc-BEVDet** | R50 | 256×704 | **20.29** |
| **LOcc-BEVDet4D** | R50 | 256×704 | **21.07** |
| **LOcc-BEVFormer** | R101 | 900×1600 | **23.15** |

- LOcc-BEVDet surpasses all prior state-of-the-art methods using only R50 and 256×704 resolution.
- LOcc-BEVFormer outperforms VEON by +6.37 mIoU.

### Ablation Study
- LVLM vocabulary extraction vs. fixed category set: LVLM yields more comprehensive coverage.
- Occlusion modeling: +1.8 mIoU.
- Multi-frame fusion + majority voting: significantly reduces single-frame noise.
- Language autoencoder dimensionality reduction (512→64): mIoU drops by only 0.3 with substantially reduced computational cost.

## Highlights & Insights
- **The semantic transitive labeling pipeline is the core contribution**: Transferring text labels—rather than distilling visual features—fundamentally resolves the semantic inconsistency problem.
- **LVLM-based scene vocabulary discovery**: Eliminates the constraint of predefined category sets.
- **Occlusion-aware label transfer**: Significantly improves annotation accuracy.
- **Strong framework generalizability**: Compatible with multiple mainstream models including BEVFormer, BEVDet, and BEVDet4D.
- **Pseudo-GT quality approaching human annotation**: Holds promise for substantially reducing 3D annotation costs.

## Limitations & Future Work
- The pipeline depends on the accuracy of the underlying LVLM and OV-Seg models.
- The vocabulary extraction stage requires per-frame LVLM inference, incurring significant offline computational overhead.
- Evaluation is conducted only on nuScenes; generalization to datasets such as Waymo remains unverified.
- A gap of approximately 5 mIoU still exists between pseudo-GT and human annotations.

## Related Work & Insights
- **vs. POP-3D**: Relies on sparse LiDAR + LSeg feature distillation, leading to high semantic noise; LOcc employs text label transfer and dense reconstruction.
- **vs. VEON**: Uses CLIP features with direct voxel projection, ignoring occlusion; LOcc applies OV-Seg text labels with occlusion modeling and multi-frame fusion.

The principle that "text label transfer outperforms feature distillation" is transferable to other 3D language understanding tasks. The bottleneck of OVO lies not in model architecture but in pseudo-GT quality—data quality is paramount.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The semantic transitive labeling pipeline is conceptually novel and empirically impactful.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-backbone validation, pseudo-GT quality comparison, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear figures and precise problem formulation.
- Value: ⭐⭐⭐⭐⭐ High-quality annotation-free 3D language GT generation has far-reaching implications for the 3D perception community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Semantic Causality-Aware Vision-Based 3D Occupancy Prediction](semantic_causality-aware_vision-based_3d_occupancy_prediction.md)
- [\[ICCV 2025\] AGO: Adaptive Grounding for Open World 3D Occupancy Prediction](ago_adaptive_grounding_for_open_world_3d_occupancy_predictio.md)
- [\[ICCV 2025\] UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving](uniocc_a_unified_benchmark_for_occupancy_forecasting_and_prediction_in_autonomou.md)
- [\[ICCV 2025\] SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World](sa-occ_satellite-assisted_3d_occupancy_prediction_in_real_world.md)
- [\[ICCV 2025\] EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding](embodiedocc_embodied_3d_occupancy_prediction_for_vision-based_online_scene_under.md)

</div>

<!-- RELATED:END -->
