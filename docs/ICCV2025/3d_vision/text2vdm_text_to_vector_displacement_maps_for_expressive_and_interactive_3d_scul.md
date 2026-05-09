---
title: >-
  [Paper Note] Text2VDM: Text to Vector Displacement Maps for Expressive and Interactive 3D Sculpting
description: >-
  [ICCV 2025][3D Vision][VDM brush generation] Text2VDM is proposed as the first framework for generating VDM sculpting brushes from text. It addresses the semantic entanglement problem in sub-object structure generation via Sobolev-preconditioned mesh deformation and a semantically enhanced SDS loss.
tags:
  - ICCV 2025
  - 3D Vision
  - VDM brush generation
  - Score Distillation
  - semantic enhancement
  - mesh deformation
  - 3D modeling
date: 2026-05-08
content_hash: 81291bc95b68b90d
---

# Text2VDM: Text to Vector Displacement Maps for Expressive and Interactive 3D Sculpting

**Conference**: ICCV 2025
**arXiv**: [2502.20045](https://arxiv.org/abs/2502.20045)
**Code**: N/A
**Area**: 3D Vision
**Keywords**: VDM brush generation, Score Distillation, semantic enhancement, mesh deformation, 3D modeling

## TL;DR

Text2VDM is proposed as the first framework for generating VDM sculpting brushes from text. It addresses the semantic entanglement problem in sub-object structure generation via Sobolev-preconditioned mesh deformation and a semantically enhanced SDS loss.

## Background & Motivation

Professional 3D asset creation relies on diverse sculpting brushes to add surface details and geometric structures. **Vector Displacement Maps (VDMs)** are the standard brush representation in modeling software, storing a 3D displacement vector per pixel to create complex surface details (e.g., cracks, wood grain) or geometric structures (e.g., ears, antlers).

Existing methods, however, cannot generate VDM brushes:

**T2I models** — VDMs are not natural images and cannot be generated directly.

**3D generation methods** — These generate complete objects and cannot control sub-object structures.

**Semantic entanglement in SDS loss** — Generating antlers yields an entire deer head; generating a turtle shell includes the turtle's tail.

The root cause: **SDS training data consists mostly of complete object images, so the target distribution under text conditioning always biases toward full-object semantics**.

## Method

### Brush Initialization

VDMs are represented as 512×512 three-channel images, with three initialization options:
- **Zero VDM** — flat mesh, default setting
- **Spike VDM** — suitable for convex geometric structures
- **User-specified VDM** — customizable volume and orientation

### Sobolev-Preconditioned Mesh Deformation

Intrinsic smoothness is achieved via mesh Laplacian reparameterization:
$$v \leftarrow v - \eta(I + \lambda L)^{-1}\frac{\partial \mathcal{L}_{SE}}{\partial v}$$

where $L$ is the mesh Laplacian and $\lambda=15$ controls the gradient diffusion range. Compared to directly adding Laplacian regularization, the preconditioning framework maintains correct topology under large deformations and reduces triangle flipping.

### Semantically Enhanced SDS Loss

Standard SDS: $\nabla_\theta\mathcal{L}_{SDS} = \mathbb{E}[\omega(t)(\epsilon_\phi(\mathbf{x}_t; y, t) - \epsilon)\frac{\partial \mathbf{x}}{\partial \theta}]$

**Semantic suppression via CSD is infeasible** — negative prompt semantics are equally entangled, and subtraction yields a meaningless distribution.

**The proposed semantic enhancement approach**: token-level weighted blending of prompt embeddings using Compel:
$$e_w = e_0 + s \cdot (e - e_0)$$

The semantically focused text embedding $\epsilon_\phi^*$ replaces the original $\epsilon_\phi$:
$$\nabla_\theta\mathcal{L}_{SE} = \mathbb{E}[\omega(t)(\epsilon_\phi^*(\mathbf{x}_t; y, t) - \epsilon)\frac{\partial \mathbf{x}}{\partial \theta}]$$

Key advantage: the semantically focused embeddings produced by Compel are **time-step invariant**, making them more stable than Attend-and-Excite.

## Key Experimental Results

### Quantitative Evaluation (40 text prompts)

| Method | Geometry CLIP Score↑ | Mesh Self-Intersection Rate↓ |
|--------|---------------------|------------------------------|
| Paint-it | 0.2375 | 19.42% |
| Text2Mesh | 0.2497 | 7.18% |
| TextDeformer | 0.2477 | 0.04% |
| **Text2VDM** | **0.2556** | **0.77%** |

### User Study (32 participants)

| Method | Geometry Quality↑ | Text Consistency↑ |
|--------|------------------|------------------|
| Paint-it | 3.1% | 1.7% |
| Text2Mesh | 18.3% | 27.3% |
| TextDeformer | 3.3% | 3.4% |
| **Text2VDM** | **75.3%** | **67.6%** |

### Key Findings

1. Text2VDM substantially outperforms baselines on both CLIP Score and user preference.
2. The semantically enhanced SDS effectively resolves semantic entanglement: antlers no longer carry ears and mouths; turtle shells no longer include heads and tails.
3. Sobolev preconditioning preserves mesh topology well, achieving a self-intersection rate of only 0.77%.
4. Generated VDMs are directly usable in Blender/ZBrush.

## Highlights & Insights

1. **Novel task definition** — Text-to-VDM brush generation is pioneered here and is highly compatible with professional artistic workflows.
2. **In-depth analysis of semantic entanglement** — The paper accurately identifies SDS training data bias as the root cause and demonstrates that semantic suppression is not a viable solution.
3. **Elegant Compel-based weighting** — Simple and effective; time-step invariant and more stable than Attend-and-Excite.
4. **Interactive usability** — Generated brushes support real-time interactive modeling without requiring re-optimization for each edit.

## Limitations & Future Work

- Multi-view inconsistency may arise, a common limitation of SDS-based methods.
- The semantic enhancement weight $s=1.1^2$ requires per-prompt tuning.
- The optimization process still takes several minutes.

## Related Work & Insights

- **Local 3D generation**: 3D Highlighter, 3D Paintbrush, FocalDreamer, MagicClay
- **SDS improvements**: CSD, VSD, Attend-and-Excite
- **Geometric brushes**: The VDM concept is established but lacks automatic generation methods

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (novel task + semantically enhanced SDS)
- Technical Depth: ⭐⭐⭐⭐ (preconditioned optimization + thorough semantic analysis)
- Experimental Thoroughness: ⭐⭐⭐⭐ (quantitative evaluation + user study + ablation)
- Value: ⭐⭐⭐⭐⭐ (directly compatible with mainstream modeling software)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] WonderTurbo: Generating Interactive 3D World in 0.72 Seconds](wonderturbo_generating_interactive_3d_world_in_072_seconds.md)
- [\[ICCV 2025\] ExCap3D: Expressive 3D Scene Understanding via Object Captioning with Varying Detail](excap3d_expressive_3d_scene_understanding_via_object_captioning_with_varying_det.md)
- [\[ICCV 2025\] Dynamic Point Maps: A Versatile Representation for Dynamic 3D Reconstruction](dynamic_point_maps_a_versatile_representation_for_dynamic_3d_reconstruction.md)
- [\[ICCV 2025\] Learning 3D Scene Analogies with Neural Contextual Scene Maps](learning_3d_scene_analogies_with_neural_contextual_scene_maps.md)
- [\[ICCV 2025\] Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation](easy3d_a_simple_yet_effective_method_for_3d_interactive_segmentation.md)

<!-- RELATED:END -->
