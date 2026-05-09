---
title: >-
  [Paper Note] Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling
description: >-
  [ICCV 2025][3D Vision][Zero-shot scene modeling] This paper presents Diorama, the first zero-shot open-world system for complete 3D indoor scene modeling from a single RGB image. It employs a modular pipeline consisting of open-world perception and CAD-based scene assembly to produce full scenes including architectural structures and object placement, requiring neither end-to-end training nor manual annotations.
tags:
  - ICCV 2025
  - 3D Vision
  - Zero-shot scene modeling
  - single-view 3D reconstruction
  - CAD retrieval
  - foundation models
  - indoor scenes
date: 2026-05-08
content_hash: 06f29ddba1a6a2b6
---

# Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling

**Conference**: ICCV 2025
**arXiv**: [2411.19492](https://arxiv.org/abs/2411.19492)
**Code**: [Available](https://3dlg-hcvc.github.io/diorama/)
**Area**: 3D Vision
**Keywords**: Zero-shot scene modeling, single-view 3D reconstruction, CAD retrieval, foundation models, indoor scenes

## TL;DR

This paper presents Diorama, the first zero-shot open-world system for complete 3D indoor scene modeling from a single RGB image. It employs a modular pipeline consisting of open-world perception and CAD-based scene assembly to produce full scenes including architectural structures and object placement, requiring neither end-to-end training nor manual annotations.

## Background & Motivation

Reconstructing structured 3D scenes from a single image is a fundamental task in computer vision. Existing approaches exhibit significant limitations:

**Limitations of reconstruction methods**: NeRF/3DGS-based methods produce incomplete surface meshes that are incompatible with modern graphics pipelines and lack compositionality and interactivity.

**Limitations of CAD alignment methods**: Methods such as Mask2CAD, ROCA, and DiffCAD require large amounts of annotated data for end-to-end training and do not model architectural structures (walls, floors, etc.).

**Weaknesses of generative methods**: LLM-driven scene generation methods lack fine-grained 3D spatial knowledge and cannot accurately localize objects.

**Opportunity from foundation models**: Recent studies demonstrate that foundation models possess 3D perceptual capabilities, yet these have not been fully leveraged for holistic scene modeling.

Diorama addresses the following question: "Can foundation models alone enable holistic 3D scene modeling from a single real image?"

## Method

### Overall Architecture

The system comprises two major components:
1. **Open-world perception**: Interprets the scene from the input image (object recognition, depth/normal estimation, architectural reconstruction, scene graph generation).
2. **CAD-based scene modeling**: Assembles a compact 3D scene (CAD retrieval, 9-DoF pose estimation, semantics-aware layout optimization).

### Key Designs

**1. Open-world Perception**

- **Object detection and segmentation**: Combines OWLv2 (open-vocabulary detector) and SAM (Segment Anything) to localize object instances.
- **Depth/normal estimation**: Uses Metric3DV2 to estimate metric depth and normal maps, then back-projects them into point clouds.
- **Scene graph generation**: Leverages GPT-4o's visual understanding via Set-of-Marks (SoM) prompting to generate a support hierarchy graph $G = \langle V, E \rangle$.

**2. PlainRecon Architectural Reconstruction**

A simple yet effective architectural reconstruction pipeline:
- Segment and remove objects → inpaint to obtain an "empty room" image.
- Depth/normal estimation to extract point clouds of architectural elements.
- Normal-based clustering to fit 3D planes.

**3. Multi-modal CAD Retrieval**

Uses DuoDuoCLIP to jointly encode text, image, and 3D shapes, with a hierarchical retrieval strategy: text-based retrieval to ensure categorical correctness → image-based query for appearance re-ranking.

**4. Zero-shot 9-DoF Pose Estimation**

- Extracts semantically rich patch features via DINOv2.
- Computes 2D correspondences between the query image and multi-view renderings of CAD models.
- Lifts 2D correspondences to 3D via depth → solves rigid transformation with the Umeyama algorithm + RANSAC.
- Incorporates GigaPose's scale prediction network for improved robustness.

**5. Staged Layout Optimization**

Resolves physical implausibilities (interpenetration, floating objects, etc.) arising from coarse pose estimates:
- **Stage 1 – Orientation**: Aligns contact-surface normals with support-surface normals ($e_{align}$) and preserves semantic object orientation ($e_{sem}$).
- **Stage 2 – Placement**: Ensures objects rest on their support surfaces ($e_{place}$) and maintains relative distances ($e_{rel}$).
- **Stage 3 – Spatial containment**: Handles containment relationships (e.g., books inside a bookshelf) and penalizes protrusion ($e_{vol}$).
- **Stage 4 – Refinement**: Re-runs the placement stage with an additional collision penalty ($e_{col}$, computed via the SAT algorithm).

### Loss & Training

No training is required. All modules operate via zero-shot inference on pretrained foundation models:
- Scene understanding: OWLv2 + SAM + Metric3DV2 + GPT-4o.
- Shape retrieval: DuoDuoCLIP (pretrained joint embedding).
- Pose estimation: DINOv2 feature matching + geometric solving.
- Layout optimization is a differentiable gradient descent process.

## Key Experimental Results

### Main Results

**System-level comparison on the SSDB dataset:**

| Method | rAcc↑ | tAcc↑ | sAcc↑ | Acc↑ | CD↓ | User Pref.↑ | API Cost ($)↓ | Time (min)↓ |
|--------|-------|-------|-------|------|-----|-------------|--------------|------------|
| ACDC | 0.20 | 0.56 | 0.36 | 0.04 | 14.0 | 14.9% | 1.44 | 23.2 |
| **Diorama** | **0.23** | **0.68** | **0.49** | **0.08** | **9.5** | **85.1%** | **0.12** | **3.7** |

**Architectural reconstruction comparison:**

| Method | Depth | IoU↑ | PE↓ | EE↓ | CDb↓ | Time (s)↓ |
|--------|-------|------|-----|-----|------|----------|
| RaC | DAv2 | 40.3 | 39.8 | 23.2 | 0.645 | 43 |
| ACDC | DAv2 | 46.8 | 17.0 | 31.0 | 0.563 | 32 |
| **PlainRecon** | M3Dv2 | **58.6** | **9.6** | **18.9** | **0.447** | 29 |

### Ablation Study

**Pose estimation method comparison (SSDB):**

| Method | rAcc↑ | tAcc↑ | sAcc↑ | Acc↑ | Collision↓ | Relation Acc.↑ |
|--------|-------|-------|-------|------|-----------|---------------|
| BM Baseline | 0.34 | 0.93 | 0.52 | 0.19 | 11.43 | 0.58 |
| ZSP | 0.36 | 0.92 | 0.59 | 0.25 | 8.76 | 0.60 |
| GigaPose | 0.36 | 0.95 | 0.71 | 0.27 | 7.91 | 0.61 |
| **Diorama** | **0.47** | **0.95** | **0.70** | **0.37** | **6.42** | **0.62** |

**3D shape retrieval (CD↓):**

| Model | SS-Household | OOD-Household | SS-Furniture | OOD-Furniture |
|-------|-------------|---------------|-------------|---------------|
| CLIP-H | 8.5 | 11.2 | 5.1 | 8.8 |
| DD-V | 6.4 | 12.0 | 8.9 | 12.1 |
| **DD-H (Ours)** | **5.5** | **9.9** | **3.2** | **7.6** |

### Key Findings

- Diorama achieves 85.1% user preference with an API cost of only $0.12 per scene.
- PlainRecon comprehensively outperforms all baselines in architectural reconstruction, improving IoU by more than 12 points.
- Hierarchical text + image retrieval (DD-H) achieves the best performance across all categories and distribution settings.
- Staged optimization substantially reduces collisions and improves support-relation accuracy.
- The system generalizes to real-world internet images and text-to-scene tasks.

## Highlights & Insights

1. **Advantages of modular design**: Each sub-task employs the most suitable foundation model, enabling flexible module replacement without the need to collect 3D annotated data.
2. **PlainRecon pipeline**: The three-step strategy of inpainting → depth estimation → normal-based clustering achieves a favorable balance between simplicity and effectiveness.
3. **Scene-graph-driven optimization**: The support hierarchy provides semantic constraints for layout optimization, giving the process well-defined objectives.
4. **Multi-hypothesis output**: Multiple semantically similar but visually distinct scene configurations can be generated by sampling different retrieval results.

## Limitations & Future Work

1. Depth estimation errors propagate cascadingly to downstream modules, with more pronounced effects in large-scale scenes.
2. Geometric and appearance mismatches between retrieved CAD models and actual objects limit faithful scene reconstruction.
3. Object recognition and pose estimation for heavily occluded objects remain limited.
4. Current evaluation focuses on indoor scenes; outdoor scenarios have not been validated.
5. Scene graph generation relies on the GPT-4o API, increasing system complexity and cost.

## Related Work & Insights

- Compared to classical methods such as IM2CAD, Diorama represents a qualitative leap in open-world capability.
- Unlike training-based methods such as DiffCAD, the zero-shot approach avoids the real-to-synthetic domain gap.
- The work provides compelling evidence for the powerful potential of composing foundation models on 3D tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First zero-shot open-world single-view scene modeling system)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive module-level and system-level evaluation, though dataset scale is limited)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; the complex pipeline is described appropriately)
- Value: ⭐⭐⭐⭐⭐ (Pioneering work demonstrating the immense potential of composing foundation models)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Zero-Shot Inexact CAD Model Alignment from a Single Image](zero-shot_inexact_cad_model_alignment_from_a_single_image.md)
- [\[ICCV 2025\] CATSplat: Context-Aware Transformer with Spatial Guidance for Generalizable 3D Gaussian Splatting from A Single-View Image](catsplat_contextaware_transformer_with_spatial_guidance_for.md)
- [\[ICCV 2025\] MonoMobility: Zero-Shot 3D Mobility Analysis from Monocular Videos](monomobility_zero-shot_3d_mobility_analysis_from_monocular_videos.md)
- [\[ICCV 2025\] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes](buffer-x_towards_zero-shot_point_cloud_registration_in_diverse_scenes.md)
- [\[ICCV 2025\] Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians](can3tok_canonical_3d_tokenization_and_latent_modeling_of_scene-level_3d_gaussian.md)

</div>

<!-- RELATED:END -->
