---
title: >-
  [Paper Note] CARI4D: Category Agnostic 4D Reconstruction of Human-Object Interaction
description: >-
  [CVPR 2026][3D Vision][human-object interaction reconstruction] CARI4D is proposed as the first category-agnostic method for reconstructing metric-scale 4D human-object interactions from monocular RGB video—encompassing…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "human-object interaction reconstruction"
  - "category-agnostic"
  - "monocular video"
  - "4D tracking"
  - "contact reasoning"
date: 2026-05-08
content_hash: 20a6e2ffd9c18ffd
---

# CARI4D: Category Agnostic 4D Reconstruction of Human-Object Interaction

**Conference**: CVPR 2026
**arXiv**: [2512.11988](https://arxiv.org/abs/2512.11988)  
**Code**: [Project Page](https://nvlabs.github.io/CARI4D/)  
**Area**: 3D Vision / Human Understanding
**Keywords**: human-object interaction reconstruction, category-agnostic, monocular video, 4D tracking, contact reasoning

## TL;DR

CARI4D is proposed as the first category-agnostic method for reconstructing metric-scale 4D human-object interactions from monocular RGB video—encompassing object shape reconstruction, pose tracking, hand contact reasoning, and physics-constrained optimization—with zero-shot generalization to unseen categories.

## Background & Motivation

Capturing human-object interactions from monocular video is critical for gaming, robot learning, and human understanding, yet poses three major challenges: large variations in the shape and pose of both humans and objects; the absence of depth information making scale recovery difficult; and the need to reason about shape, scale, pose, and dynamics under severe occlusion.

Limitations of prior work:
- VisTracker requires known object templates.
- InterTrack is restricted to training categories.
- PICO (an image-based method) is temporally inconsistent across video frames, and its contact retrieval is limited to annotated categories.

Foundation models for shape reconstruction, pose estimation, and depth estimation have each advanced considerably, yet their predictions reside in different coordinate systems, are susceptible to noise, and do not account for fine-grained contact. The paper's core mechanism is to carefully align the predictions of foundation models to obtain robust initialization, then train an interaction-specific model to reason about contact and perform further optimization.

## Method

### Overall Architecture

First-frame image → Hunyuan3D object reconstruction + UniDepth coarse-to-fine scale search → FoundationPose + pose hypothesis selection for object tracking + NLF human estimation + depth alignment → CoCoNet render-and-compare refinement + contact prediction → contact-aware joint optimization → Output: metric-scale 4D human-object interaction.

### Key Designs

1. **Dynamic Pose Hypothesis Selection Algorithm**
    - **Function**: Robustly track object pose under occlusion and noisy depth.
    - **Mechanism**: FoundationPose internally generates $K$ candidate poses per frame. Rather than selecting the top-1 hypothesis, the algorithm dynamically filters candidates based on two criteria—mask IoU (with the human occlusion region subtracted) and temporal smoothness (geodesic distance between rotations). When all candidates are filtered out, the tracker jumps forward to an available frame and tracks backward.
    - **Design Motivation**: Directly using FoundationPose's top-1 prediction frequently fails in interaction scenarios, yet the correct pose is typically present among the $K$ candidates.

2. **CoCoNet (Category-Agnostic Contact Reasoning Network)**
    - **Function**: Refine the human-object relative pose and predict hand contacts.
    - **Mechanism**: A render-and-compare paradigm — the current estimates of the human (SMPL with per-vertex color texture) and the object are rendered as RGB/depth/mask images, which are compared against the input observations; spatio-temporal attention then predicts delta pose updates and binary hand contact labels.
    - **Design Motivation**: Foundation models predict independently without accounting for interaction, resulting in floating or interpenetrating objects. A category-agnostic approach is needed to reason about contact.

3. **Depth Alignment Strategy During Training**
    - **Function**: Eliminate the absolute error between estimated and ground-truth depth in training data, enabling the network to focus on relative pose.
    - **Mechanism**: A scale $s$ and offset $t$ are computed (based on medians) to align estimated depth to ground-truth depth before pose initialization; no alignment is applied at test time.
    - **Design Motivation**: Depth estimation errors vary across datasets; without alignment, the network overfits to dataset-specific error patterns rather than learning interaction reasoning.

### Loss & Training

- **CoCoNet training**: L1 pose loss + BCE contact loss + symmetry loss for symmetric objects.
- **Joint optimization**: contact distance loss + 2D joint reprojection loss + occlusion-aware mask loss + interpenetration loss + acceleration smoothness loss.

## Key Experimental Results

### Main Results (BEHAVE Test Set)

| Method | CD-h (cm)↓ | CD-o (cm)↓ | CD-c (cm)↓ | Acc-o↓ |
|--------|-----------|-----------|-----------|--------|
| InterTrack | 25.71 | 47.66 | 30.20 | 5.64 |
| VisTracker (requires template) | 13.52 | 18.29 | 14.22 | 0.77 |
| CARI4D (Ours) | 7.74 | 12.05 | 9.23 | 0.35 |

### Zero-Shot Generalization (InterCap, Unseen Dataset)

| Method | CD-h↓ | CD-o↓ | CD-c↓ |
|--------|-------|-------|-------|
| VisTracker | 16.12 | 27.41 | 20.17 |
| CARI4D (Ours) | 11.06 | 15.69 | 12.88 |

### Ablation Study

| Configuration | CD-c↓ | Notes |
|--------------|-------|-------|
| Raw NLF + FP tracking | 405.13 | Direct use of FP tracking completely fails |
| Proposed initialization | 10.79 | Hypothesis selection substantially improves results |
| + CoCoNet (w/o alignment) | 9.95 | Refinement is effective but affected by depth error |
| + CoCoNet (w/ alignment) | 8.62 | Alignment eliminates depth error |
| + Joint optimization | 9.35 | Improves smoothness and contact consistency |

### Key Findings

- The pose hypothesis selection algorithm reduces object CD from 1565.42 to 16.85, making it the single most critical component of the pipeline.
- The depth alignment training strategy is essential for CoCoNet; without alignment, human CD actually degrades.
- Joint optimization primarily improves motion smoothness (Acc-o from 3.78 to 0.38) and contact consistency.
- Using ground-truth object meshes or depth yields only marginal gains, indicating the method is close to its upper bound.

## Highlights & Insights

- CARI4D is the first category-agnostic full-body 4D human-object interaction reconstruction method, generalizing zero-shot to in-the-wild videos.
- The method elegantly integrates predictions from multiple foundation models (Hunyuan3D, FoundationPose, UniDepth, NLF).
- CoCoNet's render-and-compare paradigm and SMPL per-vertex color texture design are notable contributions worth adopting in related work.
- This NVIDIA work demonstrates the considerable potential of composing foundation models for complex 3D understanding tasks.

## Limitations & Future Work

- The method assumes the object is largely visible in the first frame, restricting applicable scenarios.
- Object meshes reconstructed by Hunyuan3D are imperfect, particularly for complex shapes.
- CD increases slightly after joint optimization (8.62 → 9.35), possibly due to overly strong regularization.
- Multi-object interactions and non-rigid object deformation are not addressed.

## Related Work & Insights

- **vs. VisTracker**: Requires known object templates and cannot generalize to new categories; CARI4D reconstructs the object from a single image.
- **vs. InterTrack**: Restricted to training categories and outputs point clouds without surfaces; CARI4D is category-agnostic and produces complete meshes.
- **vs. PICO**: An image-based method that is temporally inconsistent and relies on a contact retrieval library; CARI4D is a video-based method that is temporally consistent and category-agnostic.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First category-agnostic 4D human-object interaction reconstruction; pose hypothesis selection and CoCoNet design are distinctive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Full coverage of in-distribution, zero-shot, and in-the-wild settings; detailed ablations; rich visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ — Pipeline is clearly presented; design motivation for each module is well articulated.
- **Value**: ⭐⭐⭐⭐⭐ — Direct applicability to robot learning and AR/VR; establishes a compelling paradigm for composing foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Human Interaction-Aware 3D Reconstruction from a Single Image](human_interaction-aware_3d_reconstruction_from_a_single_image.md)
- [\[AAAI 2026\] AnchorHOI: Zero-shot Generation of 4D Human-Object Interaction via Anchor-based Prior Distillation](../../AAAI2026/3d_vision/anchorhoi_zero-shot_generation_of_4d_human-object_interactio.md)
- [\[CVPR 2026\] TeHOR: Text-Guided 3D Human and Object Reconstruction with Textures](tehor_text-guided_3d_human_and_object_reconstruction_with_textures.md)
- [\[CVPR 2026\] Glove2Hand: Synthesizing Natural Hand-Object Interaction from Multi-Modal Sensing Gloves](glove2hand_synthesizing_natural_hand-object_interaction_from_multi-modal_sensing.md)
- [\[CVPR 2026\] ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](arthoi_taming_foundation_models_for_monocular_4d_reconstruction_of_hand-articula.md)

</div>

<!-- RELATED:END -->
