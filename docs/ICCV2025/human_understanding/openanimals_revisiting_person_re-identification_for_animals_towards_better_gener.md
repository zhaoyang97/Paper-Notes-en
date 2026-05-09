---
title: >-
  [Paper Note] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization
description: >-
  [Human Understanding] > This paper develops the OpenAnimals open-source framework, systematically revisiting the transferability of person re-identification methods to animal re-identification. It proposes ARBase, an animal-oriented strong baseline that substantially outperforms existing person ReID methods across multiple benchmarks.
tags:
  - Human Understanding
date: 2026-05-08
content_hash: b2ee774468a6a8bd
---

# OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization

- **Conference**: ICCV 2025
- **arXiv**: 2410.00204
- **Code**: Submitted with the paper (OpenAnimals codebase)
- **Area**: Human Understanding
- **Keywords**: Animal Re-Identification, Person ReID Transfer, Open-Source Framework, Baseline Model, Cross-Species Generalization

## TL;DR

> This paper develops the OpenAnimals open-source framework, systematically revisiting the transferability of person re-identification methods to animal re-identification. It proposes ARBase, an animal-oriented strong baseline that substantially outperforms existing person ReID methods across multiple benchmarks.

## Background & Motivation

Animal Re-Identification (Animal Re-ID) aims to identify individual animals within a given species, which is critical for wildlife conservation, population monitoring, and behavioral research. Although conceptually similar to Person Re-Identification (Person Re-ID), the two tasks differ fundamentally:

**Species Diversity**: Different species (hyenas, leopards, sea turtles, whale sharks) exhibit dramatically different visual appearances and behaviors.

**Environmental Variability**: Habitats range from savannas to oceans, introducing far greater variation than the relatively controlled urban settings of person ReID.

**Pose Differences**: Quadrupedal locomotion (hyenas/leopards) and aquatic movement (sea turtles/whale sharks) differ fundamentally from human bipedal walking.

**Data Scarcity**: Data collection and annotation in wild environments are difficult, resulting in far less available data than person datasets.

The central question is: **Can the extensive techniques and methodologies accumulated in person re-identification be effectively transferred to animal re-identification?** Existing research lacks a systematic analysis of this question.

## Method

### Overall Architecture

The work is organized into three parts:

1. **OpenAnimals Framework**: A unified animal ReID platform built upon FastReID and WildLifeDatasets.
2. **Systematic Revisiting Experiments**: Ablating key designs from person ReID methods (BoT, AGW, SBS, MGN) one by one on animal benchmarks.
3. **ARBase Model**: A strong animal-oriented baseline constructed from insights gained in the revisiting experiments.

### OpenAnimals Framework Design

Two core principles are followed:

- **Person ReID Compatibility**: Inherits the core layers of FastReID, enabling seamless integration of state-of-the-art person ReID methods.
- **Multi-Species Support**: Integrates the dataset organization strategy of WildLifeDatasets, supporting 30+ species within a unified framework.

The modular design encompasses five stages: Data, Backbone, Head, Loss, and Training & Testing.

### ARBase Model Design

Drawing from key findings in the revisiting experiments, ARBase makes targeted, animal-oriented design choices across five modules:

**Data Module**:
- **Key Modification — Input Resolution**: Person ReID uniformly uses portrait-aspect resolutions (e.g., $[256,128]$) since humans are typically upright. Animals exhibit diverse poses; ARBase adopts a square resolution of $[384,384]$, a simple change with significant effect.
- Only random horizontal flipping ($p=0.5$) is used; Random Erasing and AutoAug are removed.

**Backbone Module**:
- ResNet-50 (ImageNet pre-trained) with last stride set to 1 for fine-grained features.
- Instance-Batch Normalization (IBN) replaces standard BN: IN learns appearance-invariant features (adapting to diverse environments), while BN retains content information.
- Multi-branch architecture: global branch + 2-part branch + 3-part branch (inspired by insights from MGN).

**Head Module**: Global Average Pooling + Linear + BNNeck (decoupling the feature spaces for triplet and cross-entropy losses).

**Loss Module**:
- Triplet loss computed on features before BNNeck: $L_{tp} = \frac{1}{N_b}\sum_{i=1}^{N_b}\text{max}(0, m + d_{pos}^i - d_{neg}^i)$
- Cross-entropy loss with label smoothing computed on features after BNNeck.

**Training & Testing**: Adam optimizer + Cosine Annealing learning rate schedule.

## Experiments

### Main Results: ARBase vs. Person ReID Methods

| Method | HyenaID R1/mAP | LeopardID R1/mAP | SeaTurtleID R1/mAP | WhaleSharkID R1/mAP |
|--------|----------------|-------------------|---------------------|----------------------|
| BoT | 58.64/34.96 | 54.92/27.65 | 84.01/41.92 | 52.54/20.86 |
| AGW | 56.36/32.72 | 54.10/28.67 | 85.17/46.18 | 50.76/21.11 |
| SBS | 51.82/30.56 | 51.23/26.54 | 84.01/44.63 | 47.46/18.84 |
| MGN | 55.91/31.08 | 53.69/28.21 | 86.05/46.67 | 50.25/21.47 |
| **ARBase** | **73.18/44.87** | **64.34/37.08** | **86.92/55.99** | **62.44/29.45** |

ARBase achieves Rank-1 improvements of **14.54%** on HyenaID, **9.90%** on WhaleSharkID, and **9.42%** on LeopardID.

### Key Findings from Revisiting Experiments

| Technique | Effective for Persons | Generalization to Animals |
|-----------|----------------------|---------------------------|
| Random Erasing | ✓ | ✗ (negative effect on 3/4 datasets; disrupts subtle individual details) |
| Label Smoothing | ✓ | ✓ (beneficial on ≥3 datasets) |
| Last Stride=1 | ✓ | ✓ (consistently beneficial) |
| BNNeck | ✓ | ✓ (consistently beneficial) |
| Non-local Attention | ✓ | ✗ (inconsistent effects) |
| Gen-mean Pooling | ✓ | ✗ (inconsistent effects) |
| Weighted Triplet | ✓ | ✗ (inconsistent effects) |
| Freeze Training | ✓ | ✗ (removal improves performance) |
| AutoAug | ✓ | ✗ (removal improves performance) |
| Cosine Annealing | ✓ | ✓ (consistently beneficial) |
| Multi-Branch (MGN) | ✓ | ✓ (multi-granularity features also effective for animals) |

### Ablation Study (Data & Backbone)

| Configuration | HyenaID R1/mAP | WhaleSharkID R1/mAP |
|---------------|----------------|----------------------|
| BoT [256,128] | 58.64/34.96 | 52.54/20.86 |
| BoT [384,384] | 60.45/36.43 | 58.12/24.39 |
| ARBase w/o IBN | 69.09/43.58 | 61.93/29.28 |
| ARBase w/o MB | 71.36/42.87 | 61.42/27.78 |
| **ARBase (Full)** | **73.18/44.87** | **62.44/29.45** |

Simply adjusting the resolution to square raises BoT's Rank-1 on WhaleSharkID from 52.54% to 58.12% (+5.58%).

### Ablation Study (Head, Loss, Training)

| Configuration | HyenaID R1/mAP | WhaleSharkID R1/mAP |
|---------------|----------------|----------------------|
| w/o BNNeck | 64.55/39.23 | 44.42/22.29 |
| w/o Label Smoothing | 68.18/44.72 | 61.42/27.88 |
| w/o Cosine Annealing | 71.82/43.40 | 62.44/28.69 |
| **ARBase (Full)** | **73.18/44.87** | **62.44/29.45** |

BNNeck has a dramatic impact on WhaleSharkID (Rank-1 drops from 62.44% to 44.42%), confirming that decoupling the feature space is equally critical for animal ReID.

## Highlights & Insights

- The systematic revisiting experiments reveal that many techniques widely accepted as effective in person ReID—such as Random Erasing and AutoAug—do not transfer to animal ReID, as inter-individual differences in animals are more subtle and random erasing may destroy discriminative details.
- **The input resolution insight** carries broad implications: the portrait-aspect resolutions long used in person ReID are entirely unsuitable for the diverse poses of animals, and switching to square resolution alone yields substantial gains.
- The introduction of IBN elegantly addresses the high environmental variability inherent in animal ReID.
- ARBase's design philosophy—simple yet targeted—achieves significant improvements without introducing complex modules.

## Limitations & Future Work

- Only four animal species are evaluated; generalization to a broader range of species (e.g., birds, insects) remains unexplored.
- A fixed ResNet-50 backbone is used; stronger recent pre-trained models (e.g., DINOv2, CLIP) are not investigated.
- The horizontal partitioning assumption of the multi-branch architecture may be inappropriate for certain animals (e.g., snakes, fish).
- Spatiotemporal information from video sequences is not considered.
- Dataset scales are relatively small, and in-depth analysis of overfitting and generalization is insufficient.

## Related Work & Insights

- **Person ReID**: BoT (bag of tricks), AGW (non-local attention + GeM pooling), SBS (AutoAug + cosine annealing), MGN (multi-granularity network).
- **Animal ReID**: HotSpotter (hand-crafted features), MegaDescriptor (multi-species pre-training), CLIP/DINOv2-based methods.
- **Open-Source Frameworks**: FastReID (person), WildLifeDatasets (animal dataset management).

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐ |
| Effectiveness | ⭐⭐⭐⭐⭐ |
| Clarity | ⭐⭐⭐⭐⭐ |
| Practical Value | ⭐⭐⭐⭐⭐ |
| Overall | 8.0/10 |

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Sequential Keypoint Density Estimator: An Overlooked Baseline of Skeleton-Based Video Anomaly Detection](sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_v.md)
- [\[ICCV 2025\] What's Making That Sound Right Now? Video-centric Audio-Visual Localization](whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)
- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[ICCV 2025\] SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning](semges_semantics-aware_co-speech_gesture_generation_using_semantic_coherence_and.md)
- [\[CVPR 2026\] TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](../../CVPR2026/human_understanding/tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)

<!-- RELATED:END -->
