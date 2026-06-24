---
title: >-
  [Paper Note] DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers
description: >-
  [CVPR 2025][3D Vision][Multi-Teacher Distillation] Proposes DUNE, which pioneers the study of heterogeneous teacher distillation (co-distillation). It distills a ViT-Base universal encoder from teacher models with highly distinct task objectives and training data (DINOv2 + MASt3R + Multi-HMR). The student achieves teacher-level performance across 2D vision, 3D scene understanding, and 3D human perception tasks.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Multi-Teacher Distillation"
  - "Heterogeneous Teachers"
  - "Universal Encoder"
  - "2D-3D Unification"
  - "DINOv2"
date: 2026-05-08
content_hash: 75e055492e4293d4
---

# DUNE: Distilling a Universal Encoder from Heterogeneous 2D and 3D Teachers

**Conference**: CVPR 2025  
**arXiv**: [2503.14405](https://arxiv.org/abs/2503.14405)  
**Code**: [https://europe.naverlabs.com/dune](https://europe.naverlabs.com/dune)  
**Area**: 3D Vision  
**Keywords**: Multi-Teacher Distillation, Heterogeneous Teachers, Universal Encoder, 2D-3D Unification, DINOv2

## TL;DR

Proposes DUNE, which pioneers the study of heterogeneous teacher distillation (co-distillation). It distills a ViT-Base universal encoder from teacher models with highly distinct task objectives and training data (DINOv2 + MASt3R + Multi-HMR). The student achieves teacher-level performance across 2D vision, 3D scene understanding, and 3D human perception tasks.

## Background & Motivation

**Background**: Methods like AM-RADIO and UNIC have successfully distilled multiple foundation models into a single encoder. However, these teachers are homogeneous, trained on similar web-scraped data.

**Limitations of Prior Work**: No prior work has investigated distillation from highly heterogeneous teachers whose tasks and training data differ significantly (e.g., dedicated 3D reconstruction models + human perception models + general-purpose vision foundation models).

**Key Challenge**: The training data of heterogeneous teachers varies extremely (general web images vs. synthetic 3D data vs. human-centric images), and their feature spaces represent completely different information.

**Core Idea**: Investigate data sharing strategies and teacher-specific projector designs to achieve effective distillation from heterogeneous teachers.

## Method

### Overall Architecture

A ViT-Base student encoder is aligned with DINOv2 (general 2D), MASt3R (3D scene reconstruction), and Multi-HMR (3D human perception) via teacher-specific projectors. Key questions: what data is used for distillation, and how are the projectors designed?

### Key Designs

1. **Data Sharing Strategy**:

    - Function: Selecting appropriate distillation data for heterogeneous teachers.
    - Mechanism: Rather than solely relying on general data (ImageNet), the training domains of all teachers must be included. Training data from different teachers are mixed, and each teacher only computes distillation loss on its relevant data.
    - Design Motivation: The knowledge of specialized teachers (e.g., MASt3R) can only be effectively transferred on data resembling their training domains.

2. **Teacher-Specific Projectors**:

    - Function: Capture specialized details from each teacher.
    - Mechanism: Independent projectors (Transformer layers) are allocated to each teacher to project the shared encoder's output into individual teacher feature spaces. The impact of projector depth on performance is explored.
    - Design Motivation: The feature spaces of heterogeneous teachers differ significantly, requiring projectors with sufficient capacity to bridge the gap.

3. **Balancing Task-Agnostic vs. Task-Specific Teachers**:

    - Function: Retaining generalizability while acquiring specialized skills.
    - Mechanism: DINOv2 acts as a task-agnostic teacher to provide generalization capability, while MASt3R and Multi-HMR serve as task-specific teachers providing specialized capabilities. The distillation loss is weighted by teacher type.
    - Design Motivation: Prevent the distillation of specialized teachers from degrading the quality of general representations.

### Loss & Training

Standard multi-teacher distillation loss: L2 distance is computed between the student features mapped through projectors and the teacher features. Each teacher is activated only on its relevant data.

## Key Experimental Results

### Main Results

DUNE (ViT-Base) performance:
- 2D tasks (classification/segmentation/depth): Close to the DINOv2 ViT-Large teacher.
- 3D reconstruction: Surpasses MASt3R (which uses a larger encoder) in the Map-free Visual Relocalization challenge.
- 3D human perception: Near the performance of the Multi-HMR teacher.

### Key Findings

- Heterogeneous data is critical for distilling specialized teachers.
- Projector depth is positively correlated with teacher complexity.
- Distilling into a smaller student encoder can sometimes outperform larger teachers.

## Highlights & Insights

- Formulates and defines the heterogeneous teacher distillation problem for the first time.
- Outperforms MASt3R in relocalization using a ViT-Base encoder, demonstrating the strong potential of knowledge compression.
- PCA feature visualization intuitively demonstrates how DUNE merges features from three heterogeneous teachers.

## Limitations & Future Work

- Currently only validates the combination of three teachers; scaling to more heterogeneous teachers remains to be studied.
- Projectors introduce extra parameters and computational overhead.
- Downstream tasks still require teacher-specific decoder heads.

## Rating

- Novelty: 8/10 — First to define and study heterogeneous distillation.
- Technical Depth: 7/10 — Deep experimental analysis.
- Experimental Thoroughness: 8/10 — Extensively validated across multiple tasks.
- Writing Quality: 8/10 — Clear problem formulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Symmetry Strikes Back: From Single-Image Symmetry Detection to 3D Generation](symmetry_strikes_back_from_single-image_symmetry_detection_to_3d_generation.md)
- [\[CVPR 2025\] Multi-View Pose-Agnostic Change Localization with Zero Labels](mv_3dcd_multiview_change_detection.md)
- [\[CVPR 2025\] UniK3D: Universal Camera Monocular 3D Estimation](unik3d_universal_camera_monocular_3d_estimation.md)
- [\[CVPR 2025\] MAGiC-SLAM: Multi-Agent Gaussian Globally Consistent SLAM](magic-slam_multi-agent_gaussian_globally_consistent_slam.md)
- [\[CVPR 2025\] Olympus: A Universal Task Router for Computer Vision Tasks](olympus_a_universal_task_router_for_computer_vision_tasks.md)

</div>

<!-- RELATED:END -->
