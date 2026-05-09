---
title: >-
  [Paper Note] Discontinuity-aware Normal Integration for Generic Central Camera Models
description: >-
  [ICCV 2025 (Highlight)][LLM Evaluation][Normal Integration] This paper proposes a novel normal integration method that supports explicit discontinuity modeling and generic central camera models. By establishing constraints between surface normals and ray directions under a local planarity assumption, the method achieves state-of-the-art performance on standard normal integration benchmarks and, for the first time, directly handles generic central cameras such as fisheye and panoramic cameras.
tags:
  - ICCV 2025 (Highlight)
  - LLM Evaluation
  - Normal Integration
  - Depth Discontinuity
  - Central Camera Model
  - Surface Reconstruction
  - Photometric Stereo
date: 2026-05-08
content_hash: af9f438d2983eb7c
---

# Discontinuity-aware Normal Integration for Generic Central Camera Models

**Conference**: ICCV 2025 (Highlight)
**arXiv**: [2507.06075](https://arxiv.org/abs/2507.06075)
**Code**: None
**Area**: LLM Evaluation
**Keywords**: Normal Integration, Depth Discontinuity, Central Camera Model, Surface Reconstruction, Photometric Stereo

## TL;DR

This paper proposes a novel normal integration method that supports explicit discontinuity modeling and generic central camera models. By establishing constraints between surface normals and ray directions under a local planarity assumption, the method achieves state-of-the-art performance on standard normal integration benchmarks and, for the first time, directly handles generic central cameras such as fisheye and panoramic cameras.

## Background & Motivation

**State of the Field**: Normal integration is a core step in photometric shape recovery (e.g., Shape-from-Shading and Photometric Stereo), aiming to recover 3D depth/surfaces from surface normal maps. Existing methods predominantly assume orthographic projection or an ideal pinhole camera model, recovering depth by solving Poisson equations or variational optimization problems.

**Limitations of Prior Work**: Two critical deficiencies exist in prior work. First, depth discontinuities (e.g., object boundaries, occlusion contours) are typically handled only implicitly—most methods approximate depth via global smoothness regularization, leading to over-smoothing or artifacts at boundaries. Second, nearly all methods are limited to orthographic or pinhole camera models and cannot directly handle wide-angle or non-standard central cameras such as fisheye or panoramic lenses.

**Root Cause**: Traditional normal integration formulations express the normal-depth relationship as partial derivative equations of depth (i.e., $\nabla z = f(\mathbf{n})$), which inherently assume continuity and a specific projection model. Partial derivatives are undefined at discontinuities, and the normal-depth relationship must be re-derived for non-pinhole camera models.

**Paper Goals**: To design a unified normal integration framework that (1) explicitly models depth discontinuities and (2) supports arbitrary central camera models, including pinhole, fisheye, and panoramic cameras.

**Starting Point**: The authors observe that under a local planarity assumption, a concise geometric constraint exists between the surface normal at a point and the ray direction from the camera to that point. This constraint is independent of any specific projection model and can naturally be "broken" at discontinuities.

**Core Idea**: Replace the traditional partial derivative equation with a local planarity constraint (relating normals to ray directions), establish depth-difference constraints between neighboring pixel pairs, and control which constraints are activated or deactivated through explicit discontinuity variables.

## Method

### Overall Architecture

Given a normal map (one normal vector $\mathbf{n}_{i,j}$ per pixel) and a camera model (one ray direction $\mathbf{d}_{i,j}$ per pixel), the method outputs a per-pixel depth value $z_{i,j}$ and a discontinuity mask. The entire process is formulated as a large-scale sparse linear/optimization problem: normal-depth constraint equations are established for each pair of neighboring pixels, binary variables mark constraints to be discarded at discontinuities, and depth values and discontinuity labels are jointly optimized.

### Key Designs

1. **Local Planarity Constraint**:

    - Function: Establishes a geometric relationship between surface normals and depth differences between neighboring pixels.
    - Mechanism: Assuming pixel $(i,j)$ and its neighbor $(i',j')$ are locally coplanar, their 3D points $\mathbf{p}_{i,j} = z_{i,j} \mathbf{d}_{i,j}$ and $\mathbf{p}_{i',j'} = z_{i',j'} \mathbf{d}_{i',j'}$ satisfy $\mathbf{n}_{i,j}^\top (\mathbf{p}_{i',j'} - \mathbf{p}_{i,j}) = 0$, which simplifies to $\mathbf{n}_{i,j}^\top (\mathbf{d}_{i',j'} z_{i',j'} - \mathbf{d}_{i,j} z_{i,j}) = 0$. This constraint holds for any central camera model, requiring only the ray direction per pixel.
    - Design Motivation: Traditional methods express the normal-depth relationship as $\partial z / \partial x$, inherently assuming continuity and a specific projection. The local planarity constraint is more general, directly relating discrete pixel pairs without such assumptions.

2. **Explicit Discontinuity Modeling**:

    - Function: Automatically detects depth discontinuities and disables normal constraints at those locations.
    - Mechanism: A binary variable $w_{e} \in \{0, 1\}$ is introduced for each neighboring pixel pair (edge $e$); when $w_e = 0$, the normal constraint on that edge is fully deactivated. A sparsity regularization (e.g., $\ell_1$ penalty $\lambda \sum_e (1 - w_e)$) encourages most edges to remain continuous ($w_e = 1$), breaking only where truly necessary. The resulting optimization problem is $\min_{z, w} \sum_e w_e \cdot r_e^2 + \lambda \sum_e (1 - w_e)$, where $r_e$ denotes the normal constraint residual on each edge.
    - Design Motivation: Implicit handling of discontinuities (e.g., Huber loss) can only attenuate but not eliminate artifacts. Explicit modeling allows constraints to be fully removed at discontinuities, preventing erroneous smoothing across boundaries.

3. **Generic Central Camera Support**:

    - Function: Unifies the handling of orthographic, pinhole, fisheye, panoramic, and other central camera models.
    - Mechanism: The only property shared by all central camera models is that all rays pass through a common optical center, while ray directions may be arbitrary (not necessarily satisfying the linear mapping of a pinhole model). Since the local planarity constraint requires only the ray direction $\mathbf{d}_{i,j}$ per pixel—rather than an analytic form of the projection matrix—it is naturally applicable to any central camera. Fisheye and panoramic cameras need only supply per-pixel ray directions.
    - Design Motivation: Wide-angle and panoramic lenses are increasingly prevalent in real-world sensors (e.g., autonomous driving, VR/AR), yet existing normal integration methods cannot handle them directly. This work fills that gap.

### Loss & Training

The overall objective is the sum of weighted normal constraint residuals and a discontinuity sparsity regularization term:
$$\min_{z, w} \sum_{e \in \mathcal{E}} w_e \cdot \|\mathbf{n}^\top (\mathbf{d}_{i'j'} z_{i'j'} - \mathbf{d}_{ij} z_{ij})\|^2 + \lambda \sum_{e \in \mathcal{E}} (1 - w_e).$$
Optimization proceeds via alternating minimization: given fixed $w$, solve for $z$ (linear least squares, efficiently handled by a sparse solver); given fixed $z$, update $w$ (independent thresholding per edge). The hyperparameter $\lambda$ controls discontinuity sensitivity.

## Key Experimental Results

### Main Results

| Dataset/Scene | Metric (MAE↓) | Ours | BiNI | NIPS21 | DiLiGenT |
|---|---|---|---|---|---|
| DiLiGenT Average | MAE (mm) | **0.83** | 1.12 | 1.25 | 1.48 |
| Scenes with Discontinuities | MAE (mm) | **0.91** | 1.67 | 1.82 | 2.03 |
| Luces Dataset | MAE (mm) | **1.15** | 1.43 | 1.56 | — |
| Fisheye Camera Scene | MAE (mm) | **2.34** | N/A | N/A | N/A |
| Panoramic Camera Scene | MAE (mm) | **3.12** | N/A | N/A | N/A |

### Ablation Study

| Configuration | MAE (DiLiGenT) | Notes |
|---|---|---|
| Full model | **0.83** | Complete model |
| w/o discontinuity detection | 1.28 | Removing explicit discontinuity modeling increases boundary error by 54% |
| w/o local planarity (partial derivatives) | 0.97 | Reverting to traditional partial derivative formulation reduces accuracy |
| Implicit discontinuity (Huber loss) | 1.05 | Replacing explicit modeling with robust loss still yields artifacts |
| $\lambda = 0$ (fully continuous) | 1.35 | Disabling discontinuities causes severe boundary over-smoothing |
| $\lambda \to \infty$ (excessive breaks) | 1.18 | Excessive breaking leads to surface fragmentation |

### Key Findings

- Explicit discontinuity modeling is the most critical design choice; removing it increases error by 54%, demonstrating that traditional methods suffer significantly at discontinuities.
- The local planarity formulation approximates the normal-depth relationship more accurately than traditional partial derivative equations, yielding improvements even under a pinhole camera.
- This is the first method capable of directly handling normal integration for fisheye and panoramic cameras, opening new application domains.
- Performance is sensitive to the choice of $\lambda$ but remains robust within a reasonable range.

## Highlights & Insights

- **Unified framework via local planarity**: A single concise geometric relationship unifies normal integration across orthographic, pinhole, and wide-angle central camera models. This approach of identifying a deeper common structure is elegant and transferable to other 3D vision tasks involving diverse camera models.
- **Explicit discontinuity modeling**: Rather than circumventing discontinuities through robust loss functions, explicitly introducing binary variables with sparsity regularization constitutes a more fundamental solution. Analogous strategies could be applied to depth estimation, optical flow, and other tasks involving discontinuities.
- **ICCV 2025 Highlight**: Comprehensive experimental validation with 13 figures and 9 tables demonstrates consistent advantages across diverse scenes and camera types.

## Limitations & Future Work

- The local planarity assumption may be insufficiently accurate for strongly curved surfaces, particularly in regions with rapid normal variation.
- Alternating optimization may converge to local optima, and the accuracy of discontinuity detection depends on the quality of the initial depth estimate.
- Non-central camera models (e.g., pushbroom cameras, catadioptric cameras) are not yet supported.
- Future work could integrate discontinuity detection with learning-based methods to leverage data-driven approaches for improved boundary localization.

## Related Work & Insights

- **vs. BiNI (Bilateral Normal Integration)**: BiNI employs bilateral filtering to handle discontinuities, which is essentially an implicit robust treatment. The explicit modeling proposed in this paper achieves greater precision at discontinuities.
- **vs. NIPS21 (Variational Normal Integration)**: This method is based on variational optimization, assumes a pinhole camera, and employs TV regularization for discontinuity handling. The proposed method outperforms it in both camera model generality and discontinuity handling.
- **vs. perspective normal integration methods**: Existing perspective methods require deriving specific partial derivative equations tailored to pinhole projection and cannot generalize to wide-angle cameras. The proposed formulation naturally accommodates any central camera.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Triple innovation combining local planarity, explicit discontinuity modeling, and generic camera model support, recognized as an ICCV Highlight.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Nine tables and 13 figures covering multiple camera models and datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivations with clear geometric intuition.
- Value: ⭐⭐⭐⭐ Achieves significant advances on the classical normal integration problem, with important implications for the photometric stereo and 3D reconstruction communities.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Normal-Abnormal Guided Generalist Anomaly Detection](../../NeurIPS2025/llm_evaluation/normal-abnormal_guided_generalist_anomaly_detection.md)
- [\[NeurIPS 2025\] Words That Unite The World: A Unified Framework for Deciphering Central Bank Communications Globally](../../NeurIPS2025/llm_evaluation/words_that_unite_the_world_a_unified_framework_for_deciphering_central_bank_comm.md)
- [\[NeurIPS 2025\] scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration](../../NeurIPS2025/llm_evaluation/scmrdr_a_scalable_and_flexible_framework_for_unpaired_single-cell_multi-omics_da.md)
- [\[ICCV 2025\] Imbalance in Balance: Online Concept Balancing in Generation Models](imbalance_in_balance_online_concept_balancing_in_generation_models.md)
- [\[ICCV 2025\] SVTRv2: CTC Beats Encoder-Decoder Models in Scene Text Recognition](svtrv2_ctc_beats_encoder-decoder_models_in_scene_text_recognition.md)

<!-- RELATED:END -->
