---
title: >-
  [Paper Note] PseudoMapTrainer: Learning Online Mapping without HD Maps
description: >-
  [ICCV2025][3D Vision][Online Mapping] This paper proposes PseudoMapTrainer, the first framework to train online mapping models **entirely without GT HD Maps**: it reconstructs road surfaces from multi-camera images via 2D Gaussian Splatting (RoGS) and combines a pretrained semantic segmentation model (Mask2Former) to generate vectorized pseudo labels. A mask-aware matching algorithm and loss function are further designed to handle partially occluded pseudo labels, supporting both single-trip and multi-trip (crowdsourced) modes.
tags:
  - ICCV2025
  - 3D Vision
  - Online Mapping
  - Pseudo Labels
  - Gaussian Splatting
  - HD Map
  - Semi-Supervised Learning
  - Vectorized Map
date: 2026-05-08
content_hash: bd7290719184f04d
---

# PseudoMapTrainer: Learning Online Mapping without HD Maps

**Conference**: ICCV2025
**arXiv**: [2508.18788](https://arxiv.org/abs/2508.18788)
**Code**: [github.com/boschresearch/PseudoMapTrainer](https://github.com/boschresearch/PseudoMapTrainer)
**Area**: 3D Vision
**Keywords**: Online Mapping, Pseudo Labels, Gaussian Splatting, HD Map, Semi-Supervised Learning, Vectorized Map

## TL;DR

This paper proposes PseudoMapTrainer, the first framework to train online mapping models **entirely without GT HD Maps**: it reconstructs road surfaces from multi-camera images via 2D Gaussian Splatting (RoGS) and combines a pretrained semantic segmentation model (Mask2Former) to generate vectorized pseudo labels. A mask-aware matching algorithm and loss function are further designed to handle partially occluded pseudo labels, supporting both single-trip and multi-trip (crowdsourced) modes.

## Background & Motivation

- **Key Challenge**: Current online mapping models (e.g., MapTR, MapTRv2, MapVR) require only multi-camera images at inference time to predict vectorized maps, yet **training still relies on costly HD Maps as ground truth**. HD Map annotation is extremely expensive and geographically limited, hindering generalization to unannotated regions.
- **Limitations of Prior Work**:
    - HD Map datasets (e.g., nuScenes) cover only a handful of cities with limited scene diversity.
    - Massive crowdsourced driving data cannot be exploited due to the absence of corresponding GT maps.
    - No prior work has explored training online mapping models without GT HD Maps.
- **Core Problem**: Can pseudo labels be automatically generated from unannotated sensor data (cameras/LiDAR) to replace GT HD Maps, enabling HD-Map-free online mapping training?

## Method

### Overall Architecture

PseudoMapTrainer consists of two stages:

1. **Pseudo Label Generation**: Automatically constructs vectorized map pseudo labels from unannotated driving data.
2. **Online Model Training**: Trains online mapping models using pseudo labels in place of GT HD Maps.

### Stage 1: Pseudo Label Generation

Pseudo label generation involves two sub-steps:

#### 1.1 Perspective-View Semantic Segmentation (Mask2Former)

- A Mask2Former model (Swin-Large backbone) is trained on **Mapillary Vistas V2** for semantic segmentation.
- Six road-related semantic categories are selected (lane markings, sidewalks, road boundaries, etc.).
- Inference is performed on nuScenes multi-camera images to obtain per-frame, per-view pixel-level semantic labels.
- **Key point**: Mapillary Vistas is an open dataset containing no HD Map information, so this step introduces no GT map dependency.

#### 1.2 Road Surface Reconstruction and Vectorization (RoGS)

- **RoGS (Road Gaussian Splatting)** is used to reconstruct a 3D representation of the road surface from multi-camera images.
- RoGS employs **2D Gaussian Splatting**:
    - Models the road surface as a set of 2D Gaussian elliptical disks (Gaussian splats).
    - Jointly optimizes RGB appearance and semantic label channels.
    - Fuses multi-frame, multi-view observations into a unified 3D space using vehicle pose information.
- BEV (Bird's-Eye View) semantic maps are extracted from the reconstructed Gaussian representation.
- The BEV semantic maps are vectorized to obtain map elements (lane dividers, road boundaries, crosswalks) represented as polylines.
- **Two modes**:
    - **Single-trip**: Generates pseudo labels from a single traversal; coverage is limited but requires no data alignment.
    - **Multi-trip**: Aggregates data from multiple traversals of the same location to improve pseudo label quality by increasing observation density, simulating a crowdsourcing scenario.

### Stage 2: Online Model Training

#### Mask-Aware Matching and Loss

The core challenge of pseudo labels lies in **partial observability**: due to occlusions and limited field of view, pseudo labels cover only portions of the scene, while the remaining regions are unknown (rather than "map-element-free").

- **Problem with standard approaches**: Conventional Hungarian matching performs global optimal assignment between predictions and pseudo labels, but "missing" elements in unobserved regions are incorrectly treated as negatives, penalizing correct predictions.
- **Mask-Aware Matching Algorithm**:
    - A **visibility mask** is generated for each pseudo label sample, marking regions with valid observations.
    - During Hungarian matching, assignment costs are computed only for prediction–GT pairs within the masked regions.
    - Predictions outside the mask do not participate in matching and incur no loss.
- **Mask-Aware Loss Function**:
    - Classification loss: focal loss is computed only for predictions within the mask.
    - Regression loss: L1 distance and orientation losses are computed only for matched pairs within the mask.
    - Predictions outside the mask receive neither reward nor penalty, avoiding noisy gradients.

#### Semi-Supervised Pretraining Strategy

- The online mapping model is first pretrained on large amounts of unannotated data using pseudo labels.
- It is then fine-tuned on a small amount of data with GT HD Maps.
- This strategy enables pretraining on massive crowdsourced driving data, substantially improving model performance.

### Online Mapping Backbone

- **MapVR / MapTRv2** is adopted as the online mapping model.
- A Transformer decoder architecture takes multi-camera features as input and outputs vectorized map elements.
- Geographic splitting (Geo-split) replaces conventional random splitting to ensure spatial non-overlap between training and validation sets.

## Key Experimental Results

### Datasets

- **nuScenes**: 6-camera autonomous driving dataset with 1,000 scenes.
- **Mapillary Vistas V2**: Used to train the semantic segmentation model.

### Pseudo Label Quality Evaluation

| Mode | mAP (Observable Regions) | mAP (Full Range) |
|------|--------------------------|------------------|
| Single-trip | Relatively high | Limited by coverage |
| Multi-trip | Higher | Significantly improved |

- Multi-trip mode substantially improves pseudo label coverage and quality by aggregating multiple traversals.
- Optional LiDAR depth priors can further constrain Gaussian reconstruction accuracy.

### Online Model Performance

- **Core finding**: Online mapping models trained with pseudo labels achieve **performance comparable to models trained with GT HD Maps**.
- The semi-supervised setting (pseudo label pretraining + GT fine-tuning) **surpasses the pure GT training baseline**.
- Mask-aware matching and loss are critical for pseudo label training; removing them leads to significant performance degradation.
- Multi-trip pseudo labels yield better-trained models than single-trip pseudo labels.

### Ablation Study

- **Mask-aware vs. standard matching**: The mask-aware scheme significantly outperforms standard matching that ignores visibility.
- **Single-trip vs. Multi-trip**: Multi-trip provides more complete pseudo labels, leading to better training outcomes.
- **LiDAR assistance**: Optional use of LiDAR point clouds to constrain the z-axis (height) of Gaussian reconstruction further improves quality.
- **Semi-supervised pretraining**: Pseudo label pretraining + GT fine-tuning > pure GT training, demonstrating the value of large-scale unannotated data.

## Highlights & Insights

1. **Pioneering HD-Map-free training**: This work is the first to demonstrate that online mapping models can be trained entirely without GT HD Maps, breaking the field's rigid dependency on expensive annotated data.
2. **Elegant application of Gaussian Splatting**: 2D Gaussian Splatting is transferred from novel-view synthesis to road surface reconstruction, jointly optimizing RGB and semantic channels to generate high-quality BEV pseudo labels.
3. **Necessity of mask-aware design**: Partial observability of pseudo labels is the central technical challenge; the mask-aware matching and loss design enables effective learning under incomplete supervision.
4. **Unlocking crowdsourced data potential**: The multi-trip mode and semi-supervised pretraining strategy pave the way for exploiting massive crowdsourced driving data, offering substantial practical value for the industry.
5. **Modular design**: Pseudo label generation (Mask2Former + RoGS) and online model training (MapVR) are fully decoupled, facilitating easy replacement of individual components.

## Limitations & Future Work

- **Pseudo label generation cost**: Although HD Maps are not required, Gaussian Splatting reconstruction still demands considerable computational resources, with independent optimization per scene.
- **Limited semantic categories**: Only 6 road-related categories are currently handled; more complex map elements such as traffic lights and road signs are not covered.
- **Dependence on vehicle pose**: RoGS reconstruction requires accurate vehicle pose (typically from GNSS/IMU); pose noise degrades pseudo label quality.
- **Validation only on nuScenes**: Although the method is theoretically generalizable, validation on other datasets such as Argoverse 2 and Waymo is absent.
- **Single-trip quality degradation**: Single-trip pseudo labels may suffer significantly in heavily occluded scenarios (e.g., parking areas, congested road segments).
- **No temporal consistency**: Current pseudo label generation is performed independently per scene, without exploiting inter-scene temporal correlations.

## Related Work & Insights

- **MapTR / MapTRv2**: Baseline models for end-to-end online mapping; the online model component of PseudoMapTrainer builds on MapTRv2.
- **MapVR**: A vector-representation-based online mapping method providing the training framework.
- **RoGS**: A road surface Gaussian reconstruction method (Feng et al. 2024) serving as the core of PseudoMapTrainer's pseudo label generation.
- **Mask2Former**: A general-purpose semantic segmentation model used for perspective-view semantic label generation.
- **3D Gaussian Splatting**: The foundational method for novel-view synthesis; RoGS is its domain-specific variant for road scenes.
- **ScalableMap / StreamMapNet**: Alternative online mapping methods that can serve as backbone replacements in this framework.
- **Insights**: This work demonstrates that 3D reconstruction techniques (e.g., Gaussian Splatting, NeRF) can serve as a bridge between unannotated data and supervised training — a paradigm extendable to other autonomous driving perception tasks such as 3D detection and semantic map construction.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Learning 3D Scene Analogies with Neural Contextual Scene Maps](learning_3d_scene_analogies_with_neural_contextual_scene_maps.md)
- [\[CVPR 2026\] OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting](../../CVPR2026/3d_vision/onlinepg_online_open-vocabulary_panoptic_mapping_with_3d_gaussian_splatting.md)
- [\[ICCV 2025\] Easi3R: Estimating Disentangled Motion from DUSt3R Without Training](easi3r_estimating_disentangled_motion_from_dust3r_without_training.md)
- [\[ICCV 2025\] Online Language Splatting](online_language_splatting.md)
- [\[ICCV 2025\] IM360: Large-scale Indoor Mapping with 360 Cameras](im360_large-scale_indoor_mapping_with_360_cameras.md)

<!-- RELATED:END -->
