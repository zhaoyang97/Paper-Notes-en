---
title: >-
  [Paper Note] Proactive Scene Decomposition and Reconstruction
description: >-
  [ICCV 2025][3D Vision][Dynamic SLAM] This paper proposes an online scene decomposition and reconstruction task grounded in proactive human-object interaction…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Dynamic SLAM"
  - "Scene Decomposition"
  - "Human-Object Interaction"
  - "Gaussian Splatting"
  - "Online Reconstruction"
date: 2026-05-08
content_hash: ec74a0aba00aa17a
---

# Proactive Scene Decomposition and Reconstruction

**Conference**: ICCV 2025
**arXiv**: [2510.16272](https://arxiv.org/abs/2510.16272)
**Code**: N/A
**Area**: 3D Vision
**Keywords**: Dynamic SLAM, Scene Decomposition, Human-Object Interaction, Gaussian Splatting, Online Reconstruction

## TL;DR

This paper proposes an online scene decomposition and reconstruction task grounded in proactive human-object interaction, where interaction behavior observed from an egocentric viewpoint defines the decomposition granularity, enabling progressive object decoupling and high-quality global reconstruction.

## Background & Motivation

Traditional object-level scene reconstruction methods face a fundamental dilemma: **ambiguity in decomposition granularity**. For instance, should a drawer be separated from its cabinet? Do items inside a cabinet count as part of it? Such questions are inherently ill-posed in static scenes.

Limitations of prior work:

**Object-level NeRF/3DGS methods** (e.g., Gaussian Grouping) rely on predefined segmentation granularity and cannot adapt to context.

**4D reconstruction/dynamic SLAM** methods aim to handle arbitrary dynamics but fail to exploit interaction cues effectively.

**Static decomposition methods** suffer from incomplete observations at occlusion boundaries and require inpainting for completion.

The core insight of this paper: **human behavior itself defines the most natural decomposition granularity** — the part grasped and moved by a hand constitutes an independent unit. By observing human-object interactions, decomposition ambiguity can be progressively resolved while complete observations are obtained.

## Method

### Overall Architecture

The system consists of four modules: prompted segmentation, camera/object pose estimation, mask refinement, and decomposed scene reconstruction.

### Gaussian Primitive Parameterization

Isotropic Gaussian primitives are used, parameterized by color $\mathbf{c} \in \mathbb{R}^3$, position $\mu \in \mathbb{R}^3$, isotropic variance $r \in \mathbb{R}$, and opacity $o \in \mathbb{R}$.

### Joint Optimization Objective

$$L = \lambda_p L_p + \lambda_d L_d + \lambda_{ID} L_{ID}$$

where $L_p, L_d, L_{ID}$ denote the L1 losses for color, depth, and instance segmentation, respectively.

### Prompted Segmentation

Dynamic regions are detected by measuring inconsistencies between rendered outputs from the 3D scene map and current observations:

$$\frac{\sum_{(u,v) \in S_{grid}} \mathbb{1}(\hat{D}[u,v] - D[u,v] > t_d)}{|S_{grid}|} > t_p$$

Upon detecting inconsistencies, centroids are extracted as prompts for SAM2, and YOLO-based hand detection is employed to verify genuine hand-object interactions.

### Mask Refinement

Repair strategies are designed for three common failure modes of SAM2:
1. **Incomplete observations** — A flexible-length memory bank is designed to retain complete observations.
2. **Inter-frame inconsistency** — Positive/negative prompts are added based on comparison between rendered and predicted masks.
3. **Object out of view** — Object state is inferred from 3D position, and zero masks are assigned directly.

### Progressive Decomposition

$$\frac{\sum_{f \in \mathcal{F}_{valid}} \mathbb{1}(P(\tilde{g}) \in M_r)}{|\mathcal{F}_{valid}|} > t_{3d}$$

Gaussians that frequently appear within the mask across multiple keyframes are identified as part of the object and decoupled into an independent set.

## Key Experimental Results

### Mask Quality Comparison (HOI4D Dataset)

| Method | Seq 1 | Seq 2 | Seq 3 | Seq 4 |
|--------|-------|-------|-------|-------|
| SAM2 | 0.913 | 0.884 | 0.318 | 0.941 |
| **Ours** | **0.925** | **0.920** | **0.835** | **0.947** |

The most significant improvement is observed on Seq 3 (complex scissors), where mIoU increases from 0.318 to 0.835.

### SLAM Performance Comparison

| Method | HOI4D ATE | PSNR (Static) | PSNR (Dynamic) | MHOI ATE |
|--------|-----------|---------------|----------------|----------|
| Co-SLAM | 0.172 | 17.35 | – | 0.221 |
| SplaTaM | 0.156 | 18.61 | – | 0.293 |
| NeuDySLAM | 0.094 | 25.15 | – | 0.189 |
| **Ours** | **0.076** | **29.12** | **27.58** | **0.093** |

### Key Findings

1. The proposed method reconstructs not only the static background but also interacted objects in motion, enabling more comprehensive scene understanding.
2. Mask refinement leveraging interaction information substantially outperforms results relying solely on SAM2.
3. Progressive decomposition avoids the granularity ambiguity inherent in static decomposition approaches.

## Highlights & Insights

1. **Task Definition Novelty** — "Proactive scene decomposition" is an original problem formulation that treats intentional human behavior as the natural driver of decomposition.
2. **Effective Use of Interaction Priors** — Hand-object interaction provides a stable and controllable definition of decomposition granularity.
3. **Online System Design** — The system supports real-time feedback, laying the groundwork for incremental map updates.
4. **Unified Framework** — Scene radiance, camera motion, object poses, and instance segmentation are jointly optimized within a single SLAM system.

## Limitations & Future Work

- All interacted objects are assumed to undergo approximately rigid-body motion.
- The method requires RGB-D input; purely RGB settings are not addressed.
- Only dynamics arising from human-object interaction are handled; arbitrary scene dynamics are not supported.

## Related Work & Insights

- **Object decomposition radiance fields**: ObjectSDF, Gaussian Grouping, Panoptic Lifting, EgoGaussian
- **Agent-in-the-loop scene understanding**: Roboexp, Autoscanning, iLabel
- **Dynamic SLAM**: DynaSLAM, DRG-SLAM, CFP-SLAM

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Original problem formulation; interaction as the driver of decomposition)
- Technical Depth: ⭐⭐⭐⭐ (Complex yet coherent system integration)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multi-faceted validation; new MHOI dataset introduced)
- Practical Value: ⭐⭐⭐⭐ (Direct applicability to downstream tasks such as robotic manipulation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SuperDec: 3D Scene Decomposition with Superquadric Primitives](superdec_3d_scene_decomposition_with_superquadrics_primitives.md)
- [\[ICCV 2025\] InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](instascene_towards_complete_3d_instance_decomposition_and_reconstruction_from_cl.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [\[ICCV 2025\] Scene Coordinate Reconstruction Priors](scene_coordinate_reconstruction_priors.md)
- [\[ICCV 2025\] Back on Track: Bundle Adjustment for Dynamic Scene Reconstruction](back_on_track_bundle_adjustment_for_dynamic_scene_reconstruction.md)

</div>

<!-- RELATED:END -->
