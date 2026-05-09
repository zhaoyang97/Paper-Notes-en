---
title: >-
  [Paper Note] EAGLE: Episodic Appearance- and Geometry-Aware Memory for Unified 2D-3D Visual Query Localization
description: >-
  [AAAI 2026][Segmentation][Egocentric vision] This paper proposes the EAGLE framework, inspired by avian memory consolidation mechanisms. A segmentation branch guided by an Appearance-aware Meta-learning Memory (AMM) and a tracking branch driven by a Geometry-aware Localization Memory (GLM) operate collaboratively. Combined with VGGT, the framework achieves efficient unified 2D-3D visual query localization, attaining state-of-the-art performance on the Ego4D-VQ benchmark.
tags:
  - AAAI 2026
  - Segmentation
  - Egocentric vision
  - visual query localization
  - episodic memory
  - meta-learning segmentation
  - unified 2D-3D
date: 2026-05-08
content_hash: dc1d90cff31ce629
---

# EAGLE: Episodic Appearance- and Geometry-Aware Memory for Unified 2D-3D Visual Query Localization

**Conference**: AAAI 2026
**arXiv**: [2511.08007](https://arxiv.org/abs/2511.08007)
**Code**: N/A
**Area**: Segmentation
**Keywords**: Egocentric vision, visual query localization, episodic memory, meta-learning segmentation, unified 2D-3D

## TL;DR

This paper proposes the EAGLE framework, inspired by avian memory consolidation mechanisms. A segmentation branch guided by an Appearance-aware Meta-learning Memory (AMM) and a tracking branch driven by a Geometry-aware Localization Memory (GLM) operate collaboratively. Combined with VGGT, the framework achieves efficient unified 2D-3D visual query localization, attaining state-of-the-art performance on the Ego4D-VQ benchmark.

## Background & Motivation

Visual Query Localization (VQL) is a core task in egocentric episodic memory: given a visual crop of a target object, the system must spatiotemporally localize the last occurrence of that object in a video. This capability is critical for VR/AR and embodied AI applications.

Three key limitations of existing methods:

**Deficiencies of the detect-then-track paradigm**: Detectors relying on bounding boxes introduce excessive background pixels, corrupting target appearance representations; trackers fail to robustly handle extreme viewpoint changes, drastic scale variations, and motion blur.

**Inadequacy of static queries**: Relying solely on a single low-visibility query frame cannot capture the temporal appearance variation of the target, whereas humans integrate visual cues from multiple temporal snapshots.

**Lack of 2D-3D unification**: Although the two tasks are closely related in the real world, existing methods have yet to achieve a natural unification.

**Biological inspiration**: Eagles possess remarkable episodic memory and spatial localization abilities—they rapidly form short-term memory "imprints" of salient features, actively disambiguate through continuous observation, and ultimately consolidate stable long-term memories. An ideal VQL system should go beyond passively storing static visual cues, and instead actively filter and encode target information that is intrinsically stable and critical for long-term recognition.

## Method

### Overall Architecture

EAGLE consists of two parallel branches plus a 3D localization module:

1. **Segmentation branch (identifier)**: Guided by AMM, generates pixel-level masks and provides fine-grained semantic cues.
2. **Tracking branch (navigator)**: Driven by GLM, generates discriminative score maps and robustly handles viewpoint changes.
3. **3D branch**: Leverages VGGT to jointly process 2D results, camera poses, and depth to predict 3D positions.

Both branches share a backbone network while maintaining independent online episodic memory banks; their outputs are fused via a decoder.

### Key Designs

#### AMM: Appearance-aware Meta-learning Memory

**Initialization**: SAM is applied to the visual query $\mathcal{Q}_0$ to generate an initial segmentation mask $\mathcal{M}_0$.

**Pseudo-label modulator $\mathcal{P}_\theta$**: A lightweight convolutional network that converts binary masks into multi-channel pseudo-labels encoding rich semantic information such as boundaries and centers.

**Target re-weighting network $\mathcal{W}_\theta$**: Guides the loss function to focus on key regions of the target.

**Query segmentation model $\mathcal{A}_\sigma$**: $\mathcal{A}_\sigma(x) = x * \sigma$, where $\sigma$ denotes the weights of a convolutional layer.

**Meta-learning optimization**: The objective is a convex quadratic form, solved iteratively via steepest descent:
$$\sigma_{i+1} = \sigma_i - \alpha^i g^i$$
where the step size $\alpha^i$ minimizes the loss along the gradient direction and admits a closed-form solution.

**Memory bank update strategy**: Initially contains only the query sample; when the confidence score $s_{conf} \geq 0.6$ for a retrieved frame, the segmentation result is added to $\mathcal{O}_{AMM}$, enabling continuous meta-learning of target appearance.

#### GLM: Geometry-aware Localization Memory

A Discriminative Correlation Filter (DCF) is employed to correct potential instance misassignments from the segmentation branch.

**Memory bank composition**:
- **Static snapshot**: Features and Gaussian labels of the initial query (never replaced; serves as an identity anchor).
- **Dynamic snapshots**: Features of 2D retrieved target regions (updated via FIFO strategy).

**Optimization**: Steepest descent is also applied here, with adaptive step sizes computed efficiently via the Jacobian matrix of the Gauss-Newton method. A Hinge loss variant handles data imbalance: regions inside the mask fit the target score precisely, while regions outside only suppress positive responses.

**Online update strategy**: If the score map fails to consistently produce high responses (over 60% of historical frames), the DCF is updated with the initial static snapshot; otherwise, the most recent dynamic snapshot is used.

#### Dual-Branch Fusion

The score map $\mathcal{H}_\mathcal{J}$ from the tracking branch is encoded via conv-bn-relu to align dimensions with the segmentation mask encoding, fused via element-wise addition, and fed into the decoder to produce the final segmentation score map.

#### VGGT-Driven 3D Localization

1. VGGT infers camera parameters, depth maps, and depth uncertainty for each frame in a single forward pass.
2. Results are aligned to a reference coordinate frame via a Sim(3) transformation.
3. **Multi-view aggregation**: Fusion weights are defined as $\mathcal{FW}_i = s_{conf}^i \cdot g_{conf}^i$, where:
   - Semantic confidence $s_{conf}$: combines three sub-metrics—mean probability, peak probability, and high-threshold probability.
   - Geometric confidence $g_{conf} = \exp(-\zeta \tau_i)$: derived from VGGT depth uncertainty.
4. The final 3D position is obtained by the weighted average of multi-view 3D coordinates.

### Loss & Training

Total loss: $\mathcal{L}_{total} = \mathcal{L}_{seg} + \rho \mathcal{L}_{tck}$
- $\mathcal{L}_{seg}$: Lovász loss (segmentation branch)
- $\mathcal{L}_{tck}$: Hinge loss (tracking branch)
- Training data: Ego4D + EgoTracks + VISOR
- Backbone: pretrained ViT (DINOv2), fine-tuned on EgoTracks then frozen
- AdamW optimizer for 25K iterations with a peak learning rate of 0.0025

## Key Experimental Results

### Main Results

Ego4D-VQ2D Test Server Leaderboard:

| Method | tAP25 | stAP25 | Rec.(%) | Succ.(%) |
|--------|-------|--------|---------|----------|
| RELOCATE (CVPR'25) | 0.43 | 0.35 | 50.60 | 60.10 |
| PRVQL (ICCV'25) | 0.37 | 0.28 | 45.70 | 59.43 |
| **EAGLE (Ours)** | **0.46** | **0.40** | **53.51** | **62.70** |

Improvements over RELOCATE: tAP25 +6.9%, stAP25 +14.3%, Rec +5.8%, Succ +4.3%.

Ego4D-VQ3D Validation Set:

| Method | Succ.(%) | L2 ↓ | Angle ↓ | QwP(%) |
|--------|----------|------|---------|--------|
| EgoLoc-v1 (CVPR'24) | 81.13 | 1.45 | 0.55 | 84.73 |
| **EAGLE (Ours)** | **84.77** | **1.18** | **0.42** | **85.68** |

L2 error reduced by 18.6%; angular error reduced by 23.6%.

### Ablation Study

Component ablation of AMM (VQ2D Validation Set):

| Ablation | tAP25 | stAP25 | Rec.(%) | Succ.(%) |
|----------|-------|--------|---------|----------|
| Full model | 0.47 | 0.42 | 52.09 | 61.29 |
| w/o $\mathcal{O}_{AMM}$ | 0.42 | 0.30 | 42.22 | 54.62 |
| STA → SAM | 0.44 | 0.30 | 45.19 | 57.28 |
| w/o $\mathcal{P}_\theta$ | 0.46 | 0.33 | 51.08 | 60.22 |

GLM ablation:

| Ablation | tAP25 | stAP25 | Rec.(%) | Succ.(%) |
|----------|-------|--------|---------|----------|
| Full model | 0.47 | 0.42 | 52.09 | 61.29 |
| w/o $\mathcal{O}_{GLM}$ | 0.42 | 0.28 | 42.36 | 53.09 |

### Key Findings

- **Memory banks are critical**: Removing the AMM/GLM memory banks leads to stAP25 drops of 40%/29.8%, respectively, validating the essential role of online episodic memory.
- **Opposite update strategies for the two branches**: The AMM (segmentation) requires synchronous updates to the initial query to maintain appearance consistency, whereas the GLM (tracking) retains the initial query as a fixed identity anchor (region-level features are more tolerant of appearance drift).
- **SAM-generated pseudo-labels outperform STA**: They encode richer discriminative information, facilitating more precise segmentation.

## Highlights & Insights

1. **Biologically inspired design**: The eagle's memory consolidation mechanism maps naturally onto an online short-term to long-term memory update strategy, yielding both a compelling metaphor and effective results.
2. **Complementary segmentation and tracking branches**: The segmentation branch provides pixel-level appearance precision, while the tracking branch supplies region-level geometric constraints; the two mutually correct each other's errors.
3. **Efficient 2D-to-3D unification**: Replacing traditional SfM (COLMAP) with VGGT reduces inference time from minutes to seconds while achieving higher accuracy.
4. **Dual semantic–geometric confidence fusion**: Multi-view 3D aggregation jointly considers segmentation quality and depth reliability.

## Limitations & Future Work

- The pipeline relies on SAM to generate initial pseudo-masks; failures in SAM propagate to the entire system.
- The memory bank size is fixed at 50, which may require more intelligent memory management for long videos.
- The 1B-parameter version of VGGT incurs substantial inference overhead; lightweight alternatives remain unaddressed.
- Validation is limited to the Ego4D benchmark; generalization to other egocentric video datasets has not been tested.

## Related Work & Insights

- **RELOCATE** (CVPR 2025): Previous state of the art; a training-free framework that the proposed method substantially surpasses.
- **D3S, DiMP** and other meta-learning trackers: Focus on *how to update*; this work focuses on *what to learn*.
- **VGGT**: A feed-forward visual geometry foundation model serving as a key replacement for COLMAP.
- **SAM**: Initial mask generator providing pixel-level initialization more precise than bounding boxes.

## Rating

- Novelty: ⭐⭐⭐⭐ — Novel dual-branch memory consolidation mechanism and unified 2D-3D pipeline design.
- Technical Depth: ⭐⭐⭐⭐⭐ — Sophisticated combination of meta-learning optimization, DCF steepest descent, and multi-view confidence fusion.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablations across both VQ2D and VQ3D subtasks.
- Writing Quality: ⭐⭐⭐⭐ — Engaging biological-inspiration narrative with clear framework exposition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval](../../ICLR2026/segmentation/efficient-sam2_accelerating_sam2_with_object-aware_visual_encoding_and_memory_re.md)
- [\[CVPR 2026\] AFRO: Bootstrap Dynamic-Aware 3D Visual Representation for Scalable Robot Learning](../../CVPR2026/segmentation/bootstrap_dynamic-aware_3d_visual_representation_for_scalable_robot_learning.md)
- [\[CVPR 2026\] From 2D Alignment to 3D Plausibility: Unifying Heterogeneous 2D Priors and Penetration-Free Diffusion for Occlusion-Robust Two-Hand Reconstruction](../../CVPR2026/segmentation/from_2d_alignment_to_3d_plausibility_unifying_heterogeneous_2d_priors_and_penetr.md)
- [\[ICCV 2025\] WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images](../../ICCV2025/segmentation/wildseg3d_segment_any_3d_objects_in_the_wild_from_2d_images.md)
- [\[CVPR 2026\] SouPLe: Enhancing Audio-Visual Localization and Segmentation with Learnable Prompt Contexts](../../CVPR2026/segmentation/souple_enhancing_audio-visual_localization_and_segmentation_with_learnable_promp.md)

</div>

<!-- RELATED:END -->
