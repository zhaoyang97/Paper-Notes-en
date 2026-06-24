---
title: >-
  [Paper Note] Solving Instance Detection from an Open-World Perspective
description: >-
  [CVPR 2025][Robotics][Instance Detection] From an open-world perspective, this work introduces three strategies—metric learning for adapting foundation model features, distractor sampling, and NeRF-based novel-view synthesis—to significantly enhance instance-level feature matching performance in instance detection, substantially outperforming prior arts in both CID and NID setups.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "Instance Detection"
  - "Open World"
  - "Foundation Model Adaptation"
  - "Metric Learning"
  - "NeRF Data Augmentation"
date: 2026-05-08
content_hash: 72f9f96fa7b580bb
---

# Solving Instance Detection from an Open-World Perspective

**Conference**: CVPR 2025  
**arXiv**: [2503.00359](https://arxiv.org/abs/2503.00359)  
**Code**: [Project Page](https://shenqq377.github.io/IDOW)  
**Area**: Robotics  
**Keywords**: Instance Detection, Open World, Foundation Model Adaptation, Metric Learning, NeRF Data Augmentation

## TL;DR

From an open-world perspective, this work introduces three strategies—metric learning for adapting foundation model features, distractor sampling, and NeRF-based novel-view synthesis—to significantly enhance instance-level feature matching performance in instance detection, substantially outperforming prior arts in both CID and NID setups.

## Background & Motivation

Instance Detection (InsDet) aims to locate specific object instances in novel scenes given a visual reference image. Its open-world nature poses core challenges: (1) target scenarios are completely unseen during training; (2) a significant domain gap exists between the visual references and the detection proposals (e.g., due to occlusion, illumination changes).

Existing methods partially exploit open-world information in various dimensions: CPL samples random background images to synthesize training data, VoxDet leverages external 3D datasets to learn voxel representations, and OTS-FM directly utilizes off-the-shelf foundation models (FMs) like SAM and DINOv2. However, a key finding is that while pre-trained open-world detectors can achieve high recall, foundation models (such as DINOv2) are not optimized for instance-level feature matching—making the direct application of FMs to InsDet suboptimal.

The core motivation of this work is to comprehensively leverage available data and foundation models in the open world, adapting FMs via metric learning to obtain more discriminative instance-level feature representations.

## Method

### Overall Architecture

The IDOW framework consists of two stages: (1) utilizing a pre-trained open-world detector (e.g., SAM or GroundingDINO) for proposal extraction to achieve high recall; (2) adapting the foundation model DINOv2 via metric learning to generate features tailored for instance-level matching, while enhancing FM adaptation using two data augmentation techniques: distractor sampling and NeRF novel-view synthesis. Finally, proposals are paired with visual references using a stable matching algorithm.

### Key Design 1: Foundation Model Adaptation based on Metric Learning

- **Function**: Adapting the feature space of general-purpose visual foundation models into a metric space suitable for instance-level matching.
- **Mechanism**: Construct triplet $(I_a, I_p, I_n)$ training data and fine-tune DINOv2 using the triplet loss $\ell = [d(f_\theta(I_a), f_\theta(I_p)) - d(f_\theta(I_a), f_\theta(I_n)) + \alpha]_+$ to pull features of the same instance closer and push different instances further apart. A batch-level hard negative mining strategy is employed.
- **Design Motivation**: Although DINOv2 provides excellent general visual features, it is not optimized for precise instance-level matching. Adaptation via metric learning can significantly boost matching accuracy (+5-7 AP).

### Key Design 2: Distractor Sampling

- **Function**: Sample universal negative examples from open-world images to define open-space boundaries.
- **Mechanism**: Run SAM on random background images to generate object candidate segmentations, which are incorporated as universal negative data into the triplet training of metric learning. These distractors assist the features in better distinguishing meaningful object instances from background clutter.
- **Design Motivation**: Utilizing only known instances as negatives is insufficient to cover the diverse distractor objects encountered during testing. Introducing open-world sampled distractors enhances feature robustness and discriminative capability.

### Key Design 3: NeRF Novel-View Synthesis

- **Function**: Generate synthetic reference images from more camera views for each object instance, increasing the diversity of visual references.
- **Mechanism**: Train a Zip-NeRF for each object instance, rendering synthetic images from novel viewpoints using camera poses estimated via COLMAP. These synthetic images are not only used as data augmentation in training but are also stored as extra visual references during testing.
- **Design Motivation**: In the CID setup, the number of visual reference images is limited. NeRF synthesis can greatly increase viewpoint diversity, helping the FM learn more robust view-invariant features. A particularly innovative aspect is that these synthetic images are also used during the testing phase.

### Loss & Training

Standard Triplet Loss is utilized with inverse cosine similarity as the distance metric, combined with batch-level hard negative mining. The final matching employs a stable matching algorithm, filtering low-confidence matches with a similarity threshold of 0.4.

## Key Experimental Results

### Main Results: CID Setup on the HR-InsDet Dataset

| Method | AP | AP50 | AP75 |
|------|-----|------|------|
| CPL_DINO | 27.99 | 39.62 | 32.19 |
| OTS-FM_SAM | 41.61 | 49.10 | 45.95 |
| OTS-FM_GroundingDINO | 51.68 | 62.50 | 56.78 |
| **IDOW_SAM** | **48.75** | **57.59** | **54.06** |
| **IDOW_GroundingDINO** | **57.01** | **69.33** | **62.84** |

### Ablation Study: Contribution of Each Component

| Configuration | AP |
|------|-----|
| OTS-FM (baseline) | 41.61 |
| + Metric Learning Adaptation | ~46 |
| + Distractor Sampling | ~47 |
| + NeRF Novel-View Synthesis | 48.75 |

### Key Findings

- FM adaptation brings a significant boost of 5-7 AP, demonstrating that general FM features have substantial room for optimization in instance-level matching.
- Using a stronger open-world detector (GroundingDINO vs. SAM) consistently yields a gain of 8+ AP.
- NeRF-synthesized images are also beneficial during testing—including synthetic references in the matching process further improves performance.
- IDOW achieves state-of-the-art results under both CID and NID setups, outperforming prior methods by 10+ AP.

## Highlights & Insights

1. **Unified Open-World Perspective**: The paper unifies background sampling, external data utilization, and FM usage under a coherent "open-world information utilization" framework, providing a clear methodological perspective.
2. **FM Adaptation vs. Zero-Shot FM**: Experiments demonstrate that even the strongest FMs require adaptation for specific tasks, and direct usage is suboptimal.
3. **NeRF for Both Training Augmentation and Testing**: Storing and using NeRF-synthesized reference images during the testing phase is a practical and innovative design.

## Limitations & Future Work

- Training a NeRF model individually for each object incurs a relatively high computational overhead.
- It relies on COLMAP for camera pose estimation from reference images, which can be unstable with few references.
- There is still room for improvement in small object detection (with AP_small at only 35.25).

## Related Work & Insights

- The distinction between instance detection and open-vocabulary detection lies in the requirement for instance-level rather than category-level matching.
- The concept of adapting FMs via metric learning can be extended to other tasks requiring fine-grained matching (e.g., ReID, fine-grained retrieval).

## Rating

⭐⭐⭐⭐ — The method is simple yet effective, and the open-world perspective provides a unified framework for understanding. The performance gain of 10+ AP is convincing. Utilizing NeRF during testing is an ingenious design. However, the novelty of the method lies primarily at the integration level.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MineAnyBuild: Benchmarking Spatial Planning for Open-world AI Agents](../../NeurIPS2025/robotics/mineanybuild_benchmarking_spatial_planning_for_openworld_ai.md)
- [\[NeurIPS 2025\] C-NAV: Towards Self-Evolving Continual Object Navigation in Open World](../../NeurIPS2025/robotics/c-nav_towards_self-evolving_continual_object_navigation_in_open_world.md)
- [\[ICLR 2026\] Virtual Community: An Open World for Humans, Robots, and Society](../../ICLR2026/robotics/virtual_community_an_open_world_for_humans_robots_and_society.md)
- [\[ICML 2025\] FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](../../ICML2025/robotics/founder_grounding_foundation_models_in_world_models_for_open-ended_embodied_deci.md)
- [\[CVPR 2026\] Rethinking Visual Rearrangement from A Diffusion Perspective](../../CVPR2026/robotics/rethinking_visual_rearrangement_from_a_diffusion_perspective.md)

</div>

<!-- RELATED:END -->
