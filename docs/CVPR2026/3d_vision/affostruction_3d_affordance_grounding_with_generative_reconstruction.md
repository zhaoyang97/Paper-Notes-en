---
title: >-
  [Paper Note] Affostruction: 3D Affordance Grounding with Generative Reconstruction
description: >-
  [CVPR 2026][3D Vision][3D Affordance Grounding] This paper proposes Affostruction, which completes object geometry (including unobserved regions) via sparse voxel fusion-based generative reconstruction, models the multimodal distribution of affordances using Flow Matching, and performs affordance region localization on complete 3D shapes — achieving a 54.8% improvement in reconstruction IoU and a 40.4% improvement in affordance aIoU.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D Affordance Grounding
  - Generative Reconstruction
  - Sparse Voxel Fusion
  - Flow Matching
  - Active View Selection
date: 2026-05-08
content_hash: c50bba7a3b337feb
---

# Affostruction: 3D Affordance Grounding with Generative Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2601.09211](https://arxiv.org/abs/2601.09211)
**Code**: [Project Page](https://chrockey.github.io/Affostruction/)
**Area**: 3D Vision / Robot Perception
**Keywords**: 3D Affordance Grounding, Generative Reconstruction, Sparse Voxel Fusion, Flow Matching, Active View Selection

## TL;DR

This paper proposes Affostruction, which completes object geometry (including unobserved regions) via sparse voxel fusion-based generative reconstruction, models the multimodal distribution of affordances using Flow Matching, and performs affordance region localization on complete 3D shapes — achieving a 54.8% improvement in reconstruction IoU and a 40.4% improvement in affordance aIoU.

## Background & Motivation

Robotic manipulation requires understanding object affordances — "where to grasp." In practice, robots can only observe objects from limited viewpoints via RGBD cameras, resulting in substantial occlusion. Existing methods predict affordances only on visible surfaces, whereas robots need to reason about functional properties in unobserved regions (e.g., the handle on the back of a cup). This demands simultaneous geometric completion and affordance prediction.

**Core Insight:** Generative 3D models such as TRELLIS possess strong geometric priors but do not support depth input or affordance prediction; affordance methods operate only on complete point clouds or visible surfaces. Affostruction extends TRELLIS with sparse voxel fusion to support multi-view RGBD input and introduces a Flow-based affordance module.

## Method

### Overall Architecture

Multi-view RGBD → DINOv2 feature extraction + depth projection to 3D → Sparse voxel fusion → Flow Transformer generative reconstruction of complete geometry → Sparse Flow Transformer generating affordance heatmaps (conditioned on CLIP text) → Affordance-guided active view selection → Output: complete 3D mesh with affordance annotations.

### Key Designs

1. **Sparse Voxel Fusion Conditioning**:
   - Function: Aggregates multi-view RGBD features into a constant-complexity 3D conditioning signal.
   - Mechanism: DINOv2 features from each view are projected into 3D world coordinates via depth and camera parameters; overlapping voxels are averaged and non-overlapping voxels are unioned, followed by 3D sinusoidal positional encoding.
   - Design Motivation: Maintains $O(1)$ token count (independent of the number of views), enabling the Flow Transformer to generalize across 1–8 input views.

2. **Flow-based Affordance Grounding**:
   - Function: Generates affordance heatmaps on the reconstructed geometry.
   - Mechanism: A sparse Flow Transformer is trained to denoise affordance logits conditioned on CLIP text embeddings, using a BCE + Dice mask loss in place of MSE.
   - Design Motivation: Affordances are inherently multimodal — a single query may correspond to multiple valid interaction regions (e.g., multiple grasp locations). Generative modeling captures this distribution.

3. **Affordance-Driven Active View Selection**:
   - Function: Prioritizes coverage of functional regions under a limited view budget.
   - Mechanism: Affordance heatmaps are rendered onto 2D images from candidate viewpoints; the viewpoint with the highest aggregate heatmap value is selected as the next observation.
   - Design Motivation: A single additional view yields twice the improvement of sequential sampling.

### Loss & Training

- Reconstruction stage: Conditional Flow Matching (CFM) loss with rectified flow.
- Affordance stage: BCE + Dice mask loss replacing MSE (mask loss is better suited for binary affordance).
- Random multi-view training (randomly sampling 1–8 views per iteration) to adapt the model to variable inputs.

## Key Experimental Results

### 3D Reconstruction (Toky4K)

| Method | IoU↑ | CD↓ | Uses Depth |
|--------|------|-----|------------|
| TRELLIS | 19.49 | 0.3694 | ✗ |
| MCC | 21.11 | 0.3299 | ✓ |
| Affostruction | 32.67 | 0.2427 | ✓ |

### Partial-Observation Affordance Grounding

| Method | aIoU↑ | aCD↓ |
|--------|-------|------|
| MCC + Espresso-3D | 4.74 | 0.1354 |
| Affostruction | 9.26 | 0.1044 |

### Active View Selection

| Strategy | aIoU (1 extra view) | aIoU (4 extra views) |
|----------|---------------------|----------------------|
| Sequential | 4.7 | 9.1 |
| Random | 6.2 | 11.0 |
| Affordance-driven | 9.2 | 12.4 |

### Key Findings

- Random multi-view training is critical: models trained on single views show almost no improvement when given multi-view inputs.
- BCE + Dice mask loss outperforms MSE for affordance prediction.
- The generative approach substantially outperforms discriminative methods on aIoU (19.1 vs. 13.6), even without fine-tuning the encoder.

## Highlights & Insights

- First framework to unify 3D generative reconstruction with affordance prediction.
- Sparse voxel fusion achieves $O(1)$-complexity multi-view aggregation.
- Modeling the multimodal distribution of affordances via Flow Matching is an elegant design choice.
- Active view selection closes the loop: perception → reconstruction → grounding → selection.

## Limitations & Future Work

- Severe occlusion may introduce reconstruction errors that propagate to affordance prediction.
- Incorrect initial affordance estimates can mislead active view selection.
- Currently limited to single-object scenes; multi-object settings require integration with SAM3D.
- Manipulation feasibility has not been validated on a physical robot.

## Related Work & Insights

- **vs. OpenAD / PointRefer / Espresso-3D**: These methods predict affordances only on visible surfaces; Affostruction predicts on fully reconstructed geometry.
- **vs. TRELLIS**: Operates on single RGB input without depth or affordance support; Affostruction extends to multi-view RGBD with affordance prediction.
- **vs. MCC**: Discriminative reconstruction recovers only observed surfaces; Affostruction generatively extrapolates to unseen regions.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First unified framework combining generative reconstruction, affordance grounding, and active view selection.
- Experimental Thoroughness: ⭐⭐⭐⭐ Quantitative evaluation across reconstruction, affordance, and active view selection with complete ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation is clear, methodology is modular, and failure case analysis is candid.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to affordance understanding for robotic manipulation.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Scene Grounding In the Wild](scene_grounding_in_the_wild.md)
- [\[CVPR 2026\] AffordMatcher: Affordance Learning in 3D Scenes from Visual Signifiers](affordmatcher_affordance_learning_in_3d_scenes_from_visual_signifiers.md)
- [\[CVPR 2026\] AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis](affordgrasp_cross-modal_diffusion_for_affordance-aware_grasp_synthesis.md)
- [\[CVPR 2026\] FluidGaussian: Propagating Simulation-Based Uncertainty Toward Functionally-Intelligent 3D Reconstruction](fluidgaussian_propagating_simulation-based_uncertainty_toward_functionally-intel.md)
- [\[CVPR 2026\] From Orbit to Ground: Generative City Photogrammetry from Extreme Off-Nadir Satellite Images](from_orbit_to_ground_generative_city_photogrammetry_from_extreme_off-nadir_satel.md)

<!-- RELATED:END -->
