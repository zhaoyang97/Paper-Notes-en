---
title: >-
  [Paper Note] GSAlign: Geometric and Semantic Alignment Network for Aerial-Ground Person Re-Identification
description: >-
  [NeurIPS 2025][Autonomous Driving][Aerial-Ground Person Re-Identification] This paper proposes GSAlign, a framework that addresses geometric distortion and semantic misalignment in aerial-ground person re-identification (AG-ReID) via a Learnable Thin Plate Spline (LTPS) module and a Dynamic Alignment Module (DAM), achieving +18.8% mAP and +16.8% Rank-1 improvements on the CARGO dataset under the aerial-ground protocol.
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "Aerial-Ground Person Re-Identification"
  - "Thin Plate Spline"
  - "Geometric Alignment"
  - "Semantic Masking"
  - "Vision Transformer"
date: 2026-05-08
content_hash: 0b25303089dc0416
---

# GSAlign: Geometric and Semantic Alignment Network for Aerial-Ground Person Re-Identification

**Conference**: NeurIPS 2025
**arXiv**: [2510.22268](https://arxiv.org/abs/2510.22268)  
**Code**: To be confirmed  
**Area**: Autonomous Driving
**Keywords**: Aerial-Ground Person Re-Identification, Thin Plate Spline, Geometric Alignment, Semantic Masking, Vision Transformer

## TL;DR

This paper proposes GSAlign, a framework that addresses geometric distortion and semantic misalignment in aerial-ground person re-identification (AG-ReID) via a Learnable Thin Plate Spline (LTPS) module and a Dynamic Alignment Module (DAM), achieving +18.8% mAP and +16.8% Rank-1 improvements on the CARGO dataset under the aerial-ground protocol.

## Background & Motivation

Aerial-Ground Person Re-Identification (AG-ReID) aims to match pedestrian images captured by UAVs from a bird's-eye view with those from ground-level surveillance cameras. Unlike conventional ground-based ReID, this task presents three core challenges:

1. **Extreme viewpoint discrepancy**: UAV imagery exhibits pronounced top-down perspectives, while ground cameras typically capture frontal or lateral views, resulting in drastically different appearances of the same person across views.
2. **Geometric distortion**: Viewpoint variation induces severe deformation in body pose and spatial structure, which models trained on ground-view data struggle to handle.
3. **Semantic misalignment**: Different views expose different body regions, and frequent occlusions combined with imbalanced part visibility further complicate cross-view matching.

Existing methods primarily rely on global alignment or implicit feature learning to bridge the viewpoint gap, without explicitly modeling geometric deformation or visibility discrepancy, leading to suboptimal performance under large viewpoint differences.

## Core Problem

How to simultaneously address two critical bottlenecks in AG-ReID within a unified ViT framework:

- **Geometric level**: How to compensate for spatial deformation caused by extreme viewpoint changes?
- **Semantic level**: How to dynamically adjust feature representations based on input visibility, suppressing occluded and noisy regions?

## Method

GSAlign builds upon a ViT-Base backbone and the VDT (View-Decoupled Transformer) architecture, incorporating an additional view token to model viewpoint information. Two core modules are introduced:

### 1. Learnable Thin Plate Spline Module (LTPS)

**Design Motivation**: Performs non-rigid deformation based on control-point interpolation, adaptively transforming feature maps to compensate for geometry distortion introduced by viewpoint change.

**Mechanism**:

- A set of source control points $\mathbf{P}_s \in \mathbb{R}^{K \times 2}$ is initialized as learnable parameters, uniformly distributed in normalized coordinate space.
- Target control points $\mathbf{P}_t$ are fixed at the same grid positions, representing the canonical target shape.
- **Rotation prediction**: A rotation prediction network $f_\theta(\cdot)$ predicts a rotation angle $\theta$ from the input features, and the resulting rotation matrix is applied to the source control points.
- A TPS transformation function $\mathbf{T}(\cdot)$ is constructed from the rotated source points and fixed target points to apply a nonlinear spatial transformation to the feature maps.
- **Residual fusion**: The deformed feature $\mathbf{F}_{\text{ltps}}$ is fused with the original feature via a coefficient $\eta$: $\mathbf{F}_{\text{final}} = \mathbf{F} + \eta \cdot \mathbf{F}_{\text{ltps}}$

**Progressive integration**: The LTPS module is embedded into every Transformer layer, where shallow layers focus on local deformation details and deeper layers model global pose variation, achieving layer-wise progressive geometric alignment.

### 2. Dynamic Alignment Module (DAM)

**Design Motivation**: Channel-level semantic masks are generated from input image features to dynamically modulate the feature representations of other samples, suppressing occluded and noisy dimensions while emphasizing discriminative subspaces.

**Mechanism**:

- Within each batch, prototype features $\mathbf{p}_c$ are constructed by averaging features of same-class samples and applying $\ell_2$ normalization.
- A two-layer MLP followed by Sigmoid produces channel-level masks $\mathbf{m}_i$ with values in $[0, 1]$.
- The mask is applied element-wise to the prototype feature: $\mathbf{p}_c^{\text{masked}} = \mathbf{p}_c \odot \mathbf{m}_c$
- **Training-only**: During inference, extracted features are compared directly against unmasked prototypes, requiring no labels or mask generation.

### 3. Loss & Training

The total loss consists of four components:

$$\mathcal{L}_{\text{total}} = (\mathcal{L}_{\text{id}} + \mathcal{L}_{\text{tri}}) + \alpha \mathcal{L}_{\text{deform}} + \beta \mathcal{L}_{\text{mask}}$$

- $\mathcal{L}_{\text{deform}}$: Regularizes the rotation angles of LTPS, penalizing excessively large rotations to ensure geometric plausibility.
- $\mathcal{L}_{\text{mask}} = \mathcal{L}_{\text{align}} + \lambda \mathcal{L}_{\text{entropy}}$: The alignment loss encourages masked sample features to remain consistent with prototypes; entropy regularization drives masks toward binary distributions, preventing degenerate full-activation states.

## Key Experimental Results

### Main Results (CARGO Dataset)

| Method | ALL Rank-1 | ALL mAP | A↔G Rank-1 | A↔G mAP | A↔G mINP |
|--------|-----------|---------|------------|---------|----------|
| VDT (baseline) | 64.10 | 55.20 | 48.12 | 42.76 | 29.95 |
| **GSAlign** | **65.06** | **57.95** | **64.89** | **61.55** | **52.81** |

On the most challenging aerial-ground protocol (A↔G), GSAlign outperforms VDT by: Rank-1 +16.77%, mAP +18.79%, mINP +22.86%.

### Real-World Dataset Validation

- **AG-ReID dataset**: A↔G protocol Rank-1 83.75%, mAP 75.01%, both state-of-the-art.
- **AG-ReID v2 dataset**: Achieves best mAP across all four protocols, validating generalization in real-world scenarios.

### Ablation Study

| Configuration | A↔G Rank-1 | A↔G mAP | A↔G mINP |
|---------------|-----------|---------|----------|
| Baseline | 48.12 | 42.76 | 29.95 |
| + LTPS | 64.89 | 61.08 | 50.54 |
| + LTPS + DAM | 64.89 | 61.55 | 52.81 |

- LTPS is the primary contributor, yielding a substantial mAP gain of +18.3% on A↔G.
- DAM further improves mINP by approximately +2.3% on top of LTPS, indicating that semantic-level alignment is effective for hard samples.
- Four control points represent the optimal configuration; additional control points may introduce local distortion due to excessive flexibility.
- Inserting LTPS into all Transformer layers yields the best performance; insertion into shallow layers only provides limited improvement.

## Highlights & Insights

1. **Explicit geometric alignment**: LTPS achieves end-to-end differentiable non-rigid spatial transformation via learnable control points and rotation prediction, rather than relying on implicit alignment.
2. **Progressive integration strategy**: Embedding LTPS in every Transformer layer allows shallow layers to correct local deformation and deeper layers to rectify global pose, avoiding compounded errors from one-shot correction.
3. **Visibility-aware semantic masking**: DAM dynamically generates channel-level masks conditioned on the input image to modulate prototype features, effectively suppressing noise from occluded regions.
4. **Zero inference overhead**: DAM is used exclusively during training, introducing no additional computational cost at inference.
5. **Dominant performance on A↔G**: GSAlign achieves overwhelming advantages on the most challenging cross-view aerial-ground setting.

## Limitations & Future Work

1. **Large gains only on synthetic data**: CARGO is a simulation-based synthetic dataset; improvements on real-world datasets (AG-ReID, AG-ReID v2) are comparatively modest (+0.5~2%), and generalization capability warrants further investigation.
2. **Limited contribution of DAM**: Ablation results show that DAM primarily contributes to mINP (+2.3%), with smaller gains in Rank-1 and mAP, suggesting room for further optimization.
3. **Single-dataset protocol dependency**: The main experiments rely heavily on CARGO's protocol definitions, lacking evaluation on a broader range of large-scale real-world datasets.
4. **Control point number and placement**: Experiments indicate that four control points are optimal, but fixed uniform grid initialization may not be the best choice; adaptive control point positioning warrants exploration.
5. **Insufficient comparison with diffusion-based methods**: Comparison with SD-ReID is limited to partial datasets, without full evaluation under CARGO's complete protocols.

## Related Work & Insights

| Method | Core Idea | Difference from GSAlign |
|--------|-----------|------------------------|
| VDT | View-Decoupled Transformer separating view-specific and view-invariant features | Only implicit alignment; does not address geometric deformation |
| SD-ReID | Synthesizes view-specific features using diffusion models | Generative approach with high computational cost |
| LATex | Prompt learning based on text attributes | Leverages vision-language models; different direction |
| Explain | Attribute-guided dual-stream network | Relies on attribute annotations; alignment is implicit |
| AG-ReID v2 | Three-stream model with modality-aware supervision | Primarily adversarial/metric learning; no spatial transformation |

The core advantage of GSAlign lies in **explicitly** handling both geometric and semantic alignment, rather than relying on global feature learning.

## Highlights & Insights (Extended)

1. **Inspiration from TPS**: The learnable thin plate spline transformation can be generalized to other cross-domain matching tasks involving viewpoint or deformation discrepancies (e.g., vehicle re-identification, remote sensing image registration).
2. **Progressive alignment**: Iteratively correcting deformation layer by layer outperforms one-step correction—a principle that is also applicable to tasks requiring progressive refinement, such as long-tail multi-class learning.
3. **Asymmetric train-inference design**: The strategy of using DAM only during training provides a useful reference for designing other regularization or auxiliary modules.
4. **Comparison with STN**: Compared to classical Spatial Transformer Networks, LTPS achieves non-rigid transformation via TPS with the addition of rotation prediction, yielding greater representational capacity.

## Rating

- Novelty: 7/10 — Combining TPS with rotation prediction and progressively integrating it into ViT is novel; the channel masking idea in DAM is relatively general.
- Experimental Thoroughness: 7/10 — CARGO results are convincing; gains on real-world datasets are modest; additional benchmarks are lacking.
- Writing Quality: 7/10 — Overall clear with complete formula derivations, though some descriptions are redundant.
- Value: 7/10 — Proposes an effective solution for the emerging AG-ReID direction, though the field remains limited in scale and available datasets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] When Person Re-Identification Meets Event Camera: A Benchmark Dataset and An Attribute-guided Re-Identification Framework](../../AAAI2026/autonomous_driving/when_person_re-identification_meets_event_camera_a_benchmark_dataset_and_an_attr.md)
- [\[AAAI 2026\] Hierarchical Prompt Learning for Image- and Text-Based Person Re-Identification](../../AAAI2026/autonomous_driving/hierarchical_prompt_learning_for_image-_and_text-based_person_re-identification.md)
- [\[AAAI 2026\] Debiased Dual-Invariant Defense for Adversarially Robust Person Re-Identification](../../AAAI2026/autonomous_driving/debiased_dual-invariant_defense_for_adversarially_robust_person_re-identificatio.md)
- [\[ICCV 2025\] SkyDiffusion: Leveraging BEV Paradigm for Ground-to-Aerial Image Synthesis](../../ICCV2025/autonomous_driving/leveraging_bev_paradigm_for_ground-to-aerial_image_synthesis.md)
- [\[NeurIPS 2025\] 3EED: Ground Everything Everywhere in 3D](3eed_ground_everything_everywhere_in_3d.md)

</div>

<!-- RELATED:END -->
