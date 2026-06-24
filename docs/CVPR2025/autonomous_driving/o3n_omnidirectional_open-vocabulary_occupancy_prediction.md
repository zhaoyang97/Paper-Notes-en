---
title: >-
  [Paper Note] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction
description: >-
  [CVPR 2025][Autonomous Driving][Omnidirectional Vision] O3N is the first to propose a purely vision-based end-to-end omnidirectional open-vocabulary occupancy prediction framework. It models omnidirectional spatial continuity via Polar-spiral Mamba (PsM), unifies geometric and semantic supervision via Occupancy Cost Aggregation (OCA), and bridges the pixel-voxel-text modality gap via gradient-free Natural Modality Alignment (NMA), achieving SOTA performance on QuadOcc and Hum…
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Omnidirectional Vision"
  - "Open-Vocabulary"
  - "Occupancy Prediction"
  - "Mamba"
  - "Polar-spiral"
  - "Modality Alignment"
date: 2026-05-08
content_hash: 2e6982fea87dbfa8
---

# O3N: Omnidirectional Open-Vocabulary Occupancy Prediction

**Conference**: CVPR 2025  
**arXiv**: [2603.12144](https://arxiv.org/abs/2603.12144)  
**Code**: Coming soon  
**Area**: Autonomous Driving / 3D Occupancy Prediction  
**Keywords**: Omnidirectional Vision, Open-Vocabulary, Occupancy Prediction, Mamba, Polar-spiral, Modality Alignment

## TL;DR
O3N is the first to propose a purely vision-based end-to-end omnidirectional open-vocabulary occupancy prediction framework. It models omnidirectional spatial continuity via Polar-spiral Mamba (PsM), unifies geometric and semantic supervision via Occupancy Cost Aggregation (OCA), and bridges the pixel-voxel-text modality gap via gradient-free Natural Modality Alignment (NMA), achieving SOTA performance on QuadOcc and Human360Occ.

## Background & Motivation
**Background**: 3D semantic occupancy prediction infers dense voxel semantics from 2D visual evidence, expanding from multi-view camera solutions (SurroundOcc, TPVFormer) to omnidirectional images (OneOcc). However, existing methods are limited to closed-set settings with predefined categories.

**Limitations of Prior Work**: (1) Closed-set models cannot recognize unknown objects outside the training distribution, limiting their applicability in open-world exploration; (2) Omnidirectional images (Equirectangular Projection, ERP) suffer from severe geometric distortion and non-uniform sampling, where distant areas occupy negligible pixel proportions.

**Key Challenge**: The non-uniformity of ERP projection exacerbates the risk of overfitting in "pixel-voxel-text" tripartite alignment, where partially visible semantics under uneven data distributions lead to misaligned novel semantics in the joint embedding space.

**Goal**: How to achieve open-vocabulary 3D occupancy prediction that accurately predicts unseen classes during training, starting from a single omnidirectional image?

**Key Insight**: (a) The polar nature of omnidirectional images requires matching spatial scanning strategies (polar-spiral rather than square grids); (b) Constructing an "occupancy cost volume" to replace direct voxel-text alignment alleviates overfitting; (c) Utilizing gradient-free Random Walk to align text embeddings and semantic prototypes.

**Core Idea**: Polar-spiral Mamba + Occupancy Cost Aggregation + Gradient-free Modality Alignment = Omnidirectional Open-Vocabulary 3D Occupancy.

## Method

### Overall Architecture
Equirectangular omnidirectional image $\rightarrow$ Vision-language encoder extracts image features and text embeddings $\rightarrow$ 2D-to-3D view transformation generates cubic and cylindrical voxels $\rightarrow$ PsM-enhanced 3D decoder $\rightarrow$ OCA constructs voxel-text cost volumes and NMA alignment $\rightarrow$ Occupancy prediction head. End-to-end training.

### Key Designs

1. **Polar-spiral Mamba (PsM) Module**

    - Function: Effectively models the spatial structure of omnidirectional images with a dual-branch architecture.
    - Mechanism: Cylindrical voxels are compressed into BEV $\mathbf{B}_p \in \mathbb{R}^{C \times R \times P}$ $\rightarrow$ P-SMamba scans spirally outward along polar coordinates from the pole (near to far, matching the decreasing information density characteristics of omnidirectional imaging) $\rightarrow$ Each layer resamples the polar voxels back into Cartesian space to aggregate with cubic voxels.
    - Design Motivation: Standard 3D convolutions cannot handle the discontinuity of cylindrical voxel data near the poles; the spiral scanning path progressively captures geometric and semantic details from near to far, conforming to the "dense information nearby, sparse information far away" property of omnidirectional imaging.

2. **Occupancy Cost Aggregation (OCA)**

    - Function: Constructs voxel-text cost volumes for spatial and category aggregation, replacing naive feature alignment.
    - Mechanism: Occupancy cost $C(i,l) = \frac{V_i \cdot T_l}{\|V_i\| \|T_l\|}$ $\rightarrow$ initial 3D convolution processing $\rightarrow$ ASPP multi-scale spatial aggregation $\rightarrow$ linear Transformer inter-class aggregation $\rightarrow$ residual connections followed by the prediction head.
    - Design Motivation: Direct voxel-text feature alignment easily overfits under non-uniform data distributions (partially visible semantics $\rightarrow$ skewed joint embedding space); fine-grained spatial and category aggregation of the cost volume is more robust.

3. **Natural Modality Alignment (NMA)**

    - Function: Reduces the modality gap between text embeddings and semantic prototypes without gradients.
    - Mechanism: EMA updates base-class semantic prototypes $\mathbf{P}_t^b$ $\rightarrow$ Compute text-prototype affinity $\mathcal{S}$ $\rightarrow$ Random Walk iterative aggregation (closed-form Neumann series solution $\mathbf{T}_t^\infty$) $\rightarrow$ Optimized text embeddings are used for OCA.
    - Design Motivation: CLIP's image-text embeddings still suffer from a modality gap; active learned alignment overfits to base classes and harms novel-class generalization; gradient-free Random Walk convergence avoids this issue.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{occ} + \mathcal{L}_{vox\text{-}pix} + \mathcal{L}_{oca}$. $\mathcal{L}_{occ}$ contains cross-entropy + scene relation affinity loss + focal loss; $\mathcal{L}_{vox\text{-}pix}$ aligns voxel and pixel features (from OVO); $\mathcal{L}_{oca}$ uses scene relation affinity loss to constrain cost aggregation. Unseen class voxels are treated uniformly as "unknown" during training.

## Key Experimental Results

### Main Results (QuadOcc)

| Method | Input | mIoU | Novel mIoU | Base mIoU |
|------|------|------|-----------|----------|
| OVO (MonoScene) | C | 14.33 | 18.15 | 10.52 |
| **O3N (MonoScene)** | C | **16.54** | **21.16** | **11.92** |
| Gain | — | +2.21 | +3.01 | +1.40 |

### Human360Occ

| Method | mIoU | Novel mIoU | Base mIoU |
|------|------|-----------|----------|
| OVO | baseline | baseline | baseline |
| **O3N** | **+0.86** | **+1.54** | +Gain |

### Ablation Study

| Configuration | Key Metrics | Note |
|------|---------|------|
| w/o PsM | mIoU decreases | Loses polar spatial modeling capability |
| w/o OCA | Novel mIoU decreases significantly | Naive alignment causes overfitting to novel classes |
| w/o NMA | Novel mIoU decreases | Modality gap affects novel class generalization |
| Full O3N | Optimal | Three components work synergistically |

### Key Findings
- O3N not only achieves the best open-vocabulary performance, but even outperforms some fully supervised methods in certain settings—demonstrating the regularization effect of open-vocabulary training.
- The performance gain in Novel mIoU (+3.01) is significantly higher than that of Base mIoU (+1.40), validating the positive effect of OCA and NMA on enhancing novel-class semantics.
- Novel classes (vehicle, road, building) occupy ~68% of the QuadOcc data, indicating that the method performs well on dominant classes.
- Cross-model validation (MonoScene and SGN) demonstrates the generalizability of O3N.

## Highlights & Insights
- **First Omnidirectional Open-Vocabulary Occupancy Prediction Framework**: Fills the gap in the intersection of omnidirectional vision, open-vocabulary learning, and 3D occupancy.
- **Intuitive Polar-Spiral Scanning Design**: The scanning path naturally matches the information density distribution of omnidirectional imaging—a simple yet effective geometry-aware design.
- **Theoretical Elegance of Gradient-Free NMA**: The closed-form solution based on the Neumann series avoids gradient propagation issues in iterative alignment, theoretically guaranteeing convergence.
- **Occupancy Cost Volume to Replace Direct Alignment**: Extends the concept of cost aggregation from 2D open-vocabulary segmentation to 3D, presenting a transferable solution for other 3D open-vocabulary tasks.

## Limitations & Future Work
- The base model MonoScene is relatively old, which limits the performance ceiling—there might be more room for improvement on stronger fully supervised backbones.
- The reconstruction quality near the polar regions of omnidirectional images remains low (due to the inherent limitations of ERP projection).
- The CLIP text encoder might lack sufficient discriminative power for fine-grained driving semantics (e.g., "dashed lane marking" vs "solid lane marking").
- Validation is only conducted on indoor/campus (QuadOcc) and simulation (Human360Occ) datasets, lacking large-scale outdoor scene validation.

## Related Work & Insights
- **vs OVO**: OVO pioneered open-vocabulary occupancy but targeting multi-view cameras and leveraging 2D segmenter distillation; O3N targets omnidirectional cameras and proposes three novel components: PsM, OCA, and NMA.
- **vs OneOcc**: OneOcc is the SOTA for omnidirectional fully-supervised occupancy; O3N defines a new paradigm under the open-vocabulary setting.
- **vs CAT-Seg**: CAT-Seg proposes image-text cost aggregation in 2D open-vocabulary segmentation; O3N extends this concept to 3D voxel-text cost volumes.
- **vs POP-3D**: POP-3D utilizes VLMs to achieve open-vocabulary 3D occupancy but operates as a multi-stage method; O3N is more efficient with end-to-end training.
- **Inspiration for Embodied AI**: Omnidirectional vision + open-vocabulary occupancy is a core capability requirement for embodied agents exploring the open world, and O3N provides a viable technical path.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Pioneered the intersection of omnidirectional vision, open-vocabulary learning, and occupancy prediction, with novel designs in PsM and NMA.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validation on two datasets with multiple models and ablations, but lacks large-scale physical outdoor datasets.
- Writing Quality: ⭐⭐⭐⭐ Rigorous mathematical derivations, clear architecture diagrams, and a complete overall structure.
- Value: ⭐⭐⭐⭐⭐ Provides an important technical path for open-world 3D scene understanding in embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)](../../CVPR2026/autonomous_driving/monocular_open_vocabulary_occupancy_prediction_for_indoor_scenes.md)
- [\[CVPR 2025\] OccMamba: Semantic Occupancy Prediction with State Space Models](occmamba_semantic_occupancy_prediction_with_state_space_models.md)
- [\[CVPR 2025\] 3D-AVS: LiDAR-based 3D Auto-Vocabulary Segmentation](3d-avs_lidar-based_3d_auto-vocabulary_segmentation.md)
- [\[CVPR 2025\] ProtoOcc: 3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation](3d_occupancy_prediction_with_low-resolution_queries_via_prototype-aware_view_tra.md)
- [\[ICCV 2025\] AGO: Adaptive Grounding for Open World 3D Occupancy Prediction](../../ICCV2025/autonomous_driving/ago_adaptive_grounding_for_open_world_3d_occupancy_predictio.md)

</div>

<!-- RELATED:END -->
