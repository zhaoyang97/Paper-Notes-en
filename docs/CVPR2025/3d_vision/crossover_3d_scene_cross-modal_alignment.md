---
title: >-
  [Paper Note] CrossOver: 3D Scene Cross-Modal Alignment
description: >-
  [CVPR 2025][3D Vision][Cross-Modal Alignment] The CrossOver framework is proposed to learn a unified scene-level cross-modal embedding space for RGB images, point clouds, CAD models, floor plans, and textual descriptions without requiring complete modal pairing. It utilizes dimensionality-specific encoders and a three-stage training pipeline to support flexible cross-modal retrieval and localization.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Cross-Modal Alignment"
  - "3D Scene Understanding"
  - "Multi-Modal Embedding"
  - "Scene Retrieval"
  - "Missing Modalities"
date: 2026-05-08
content_hash: 92cfec222407a32a
---

# CrossOver: 3D Scene Cross-Modal Alignment

**Conference**: CVPR 2025  
**arXiv**: [2502.15011](https://arxiv.org/abs/2502.15011)  
**Code**: [sayands.github.io/crossover](https://sayands.github.io/crossover)  
**Area**: 3D Vision / Multi-Modal Scene Understanding  
**Keywords**: Cross-Modal Alignment, 3D Scene Understanding, Multi-Modal Embedding, Scene Retrieval, Missing Modalities

## TL;DR

The CrossOver framework is proposed to learn a unified scene-level cross-modal embedding space for RGB images, point clouds, CAD models, floor plans, and textual descriptions without requiring complete modal pairing. It utilizes dimensionality-specific encoders and a three-stage training pipeline to support flexible cross-modal retrieval and localization.

## Background & Motivation

- Existing multi-modal 3D understanding methods (e.g., ULIP, PointBind) focus on **object-level** alignment, lacking scene-level context.
- These methods assume that all modal data are complete and strictly aligned—a condition rarely met in real-world scenarios (e.g., CAD models do not perfectly match actual scanned objects).
- Cross-modally consistent instance segmentation is extremely difficult to obtain in practice.
- Three problems need to be solved: (1) scene-level rather than object-level alignment, (2) not requiring all modalities to be simultaneously present, and (3) not relying on semantic priors at inference time.

## Method

### Overall Architecture

CrossOver aligns five modalities (RGB images $\mathcal{I}$, point clouds $\mathcal{P}$, CAD models $\mathcal{M}$, floor plans $\mathcal{F}$, and text $\mathcal{R}$) into a unified modality-agnostic embedding space. It adopts a three-stage progressive training pipeline: instance-level multi-modal interaction $\rightarrow$ scene-level multi-modal interaction $\rightarrow$ unified dimensionality-specific encoders.

### Key Designs

1. **Dimensionality-Specific Encoders**:
    - Function: Custom-designed encoders based on the dimensionality characteristics of each modality, eliminating the need for semantic labels.
    - Mechanism: 1D encoder (BLIP text encoder to process object referrals), 2D encoder (DinoV2 to process images and floor plans, sharing weights), and 3D encoder (Minkowski sparse convolution to process point clouds/CAD meshes). During inference, raw data is used directly as input without requiring semantic segmentation.
    - Design Motivation: Different dimensions of data require different optimal representations, and it is crucial to eliminate the dependency on semantic instance labels.

2. **Three-Stage Training Pipeline**:
    - Function: Progressively construct a modality-agnostic embedding space.
    - Mechanism:
        - Stage 1 (Instance-level): Pre-trained encoders extract instance features of each modality and align them using the image modality as an anchor: $\mathcal{L}_{\mathcal{O}_i} = \mathcal{L}_{f^I, f^{\mathcal{P}}} + \mathcal{L}_{f^I, f^{\mathcal{M}}} + \mathcal{L}_{f^I, f^{\mathcal{R}}}$
        - Stage 2 (Scene-level): Weight-fuse instance features into a scene embedding $\mathbf{F}_\mathcal{S}$.
        - Stage 3 (Unified Encoders): Train dimensionality-specific encoders to align with the scene embedding: $\mathcal{L}_s = \alpha\mathcal{L}_{\mathbf{F}_\mathcal{S}, \mathbf{F}_{1D}} + \beta\mathcal{L}_{\mathbf{F}_\mathcal{S}, \mathbf{F}_{2D}} + \gamma\mathcal{L}_{\mathbf{F}_\mathcal{S}, \mathbf{F}_{3D}}$
    - Design Motivation: Direct scene-level training is difficult, and progressively distilling instance-to-scene knowledge yields better results.

3. **Emergent Cross-Modal Behavior**:
    - Function: Establishes correspondences between unaligned modality pairs even if all pairings were not seen during training.
    - Mechanism: All modalities are aligned indirectly by using the image modality $\mathcal{I}$ as an anchor. Transitivity naturally induces cross-modal relations (e.g., although point cloud $\rightarrow$ text is not directly trained, emergent alignment is achieved via image bridging).
    - Design Motivation: Requiring paired training data for all modalities is impractical; a single-anchor alignment allows for flexible combinations.

### Loss & Training

- Contrastive loss (InfoNCE style): $\mathcal{L}_{q,k} = -\log \frac{\exp(q_i^T k_i / \tau)}{\exp(q_i^T k_i / \tau) + \sum_{j \neq i} \exp(q_i^T k_j / \tau)}$
- Symmetric loss: $\mathcal{L}_{q,k} + \mathcal{L}_{k,q}$
- Learnable temperature parameter $\tau$
- Mask corresponding loss terms when modalities are missing
- Encoders are frozen (DinoV2, BLIP, I2PMAE), training only projection layers and fusion modules

## Key Experimental Results

### Main Results (Cross-Modal Scene Retrieval on ScanNet)

| Method | Modality Pair | R@1 ↑ | R@5 ↑ | R@10 ↑ |
|------|--------|-------|-------|--------|
| ULIP-2 | $\mathcal{I} \to \mathcal{P}$ | Low | Low | Low |
| PointBind | $\mathcal{I} \to \mathcal{P}$ | Low | Low | Low |
| Instance Baseline | $\mathcal{I} \to \mathcal{P}$ | Mid | Mid | Mid |
| **CrossOver** | $\mathcal{I} \to \mathcal{P}$ | **High** | **High** | **High** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Instance-level encoders only (no scene-level) | Poor scene retrieval | Lacks scene context |
| Training with all modality pairs | Suboptimal | Aligning only to the image anchor yields better results |
| Single modality input at inference | Still usable | Unified encoders eliminate multi-modal dependency |
| No floor plan modality | Slight decrease | Floor plans provide complementary layout information |

### Key Findings

- CrossOver achieves 23.40% on scene-level R@75% in instance retrieval for $\mathcal{I} \to \mathcal{P}$ (compared to only 0.24% for ULIP-2 and 0.32% for PointBind).
- Emergent behavior is effective: $\mathcal{P} \to \mathcal{R}$ achieved strong performance despite not being directly trained.
- Same-modality temporal instance matching surpasses the dedicated LivingScenes method.
- Top-1 scene category retrieval reaches 64.74%, significantly outperforming ULIP-2 (7.37%) and PointBind (13.78%).
- Aligning to a single reference modality performs better than training on all pairs (avoiding conflicting gradients).

## Highlights & Insights

- For the first time, 5 modalities of 3D scenes (RGB, point clouds, CAD, floor plans, text) are unified into a single embedding space.
- The experimental validation of "emergent cross-modal behavior" is impressive—untrained modality pairs can also achieve effective retrieval.
- Clear distillation logic spanning three stages: from instance-level, to scene-level, and finally to semantic-free encoders.
- High practical value: supports missing modalities, eliminates the need for semantic segmentation during inference, and can be used for AR/VR scene retrieval.

## Limitations & Future Work

- Relies on 3D instance segmentation during training (though not required during inference).
- Text modality (object referrals) requires a predefined description format.
- Training was validated only on indoor datasets (ScanNet, 3RScan); outdoor generalization is unknown.
- Fixed token sizes (e.g., 10 referrals, 10 views) may limit complexity in larger scenes.
- Future directions could explore additional modalities (e.g., audio, haptic) or extend the framework to large-scale outdoor scenes.

## Related Work & Insights

- Extends the ideas of CLIP and ImageBind to the 3D scene level.
- Unlike scene-graph-based methods like SGAligner, CrossOver does not require scene graphs.
- LivingScenes processes temporally changing scenes but is single-modal, whereas CrossOver is effective across both modalities and temporal changes.
- Insight: Using images as a "universal anchor" to align other modalities is a highly efficient and robust strategy.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of a scene-level unified embedding space for five modalities is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough evaluations on multiple retrieval tasks and ablation studies, though datasets are limited.
- Writing Quality: ⭐⭐⭐⭐ Clear architectural diagram and the three-stage structure is easy to understand.
- Value: ⭐⭐⭐⭐ Strong potential for practical applications in fields such as AR/VR and architectural design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GEAL: Generalizable 3D Affordance Learning with Cross-Modal Consistency](geal_generalizable_3d_affordance_learning_with_cross-modal_consistency.md)
- [\[CVPR 2026\] Geometry-Aware Cross-Modal Graph Alignment for Referring Segmentation in 3D Gaussian Splatting](../../CVPR2026/3d_vision/geometry-aware_cross-modal_graph_alignment_for_referring_segmentation_in_3d_gaus.md)
- [\[ECCV 2024\] SceneGraphLoc: Cross-Modal Coarse Visual Localization on 3D Scene Graphs](../../ECCV2024/3d_vision/scenegraphloc_cross-modal_coarse_visual_localization_on_3d_scene_graphs.md)
- [\[CVPR 2025\] SP3D: Boosting Sparsely-Supervised 3D Object Detection via Accurate Cross-Modal Semantic Prompts](sp3d_boosting_sparsely-supervised_3d_object_detection_via_accurate_cross-modal_s.md)
- [\[CVPR 2025\] UniPre3D: Unified Pre-training of 3D Point Cloud Models with Cross-Modal Gaussian Splatting](unipre3d_unified_pre-training_of_3d_point_cloud_models_with_cross-modal_gaussian.md)

</div>

<!-- RELATED:END -->
