---
title: >-
  [Paper Note] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction
description: >-
  [CVPR 2026][Autonomous Driving][Omnidirectional occupancy prediction] O3N is the first purely vision-based, end-to-end omnidirectional open-vocabulary occupancy prediction framework. Through three core modules—Polar Spiral Mamba (PsM), Occupancy Cost Aggregation (OCA), and Natural Modality Alignment (NMA)—it achieves open-vocabulary 3D occupancy prediction under 360° panoramic image input that surpasses closed-set supervised methods.
tags:
  - CVPR 2026
  - Autonomous Driving
  - Omnidirectional occupancy prediction
  - open-vocabulary
  - Mamba
  - contrastive learning
  - panoramic perception
date: 2026-05-08
content_hash: 90c30161e9f101bb
---

# O3N: Omnidirectional Open-Vocabulary Occupancy Prediction

**Conference**: CVPR 2026
**arXiv**: [2603.12144](https://arxiv.org/abs/2603.12144)
**Code**: Coming soon
**Area**: Autonomous Driving / 3D Scene Understanding
**Keywords**: Omnidirectional occupancy prediction, open-vocabulary, Mamba, contrastive learning, panoramic perception

## TL;DR

O3N is the first purely vision-based, end-to-end omnidirectional open-vocabulary occupancy prediction framework. Through three core modules—Polar Spiral Mamba (PsM), Occupancy Cost Aggregation (OCA), and Natural Modality Alignment (NMA)—it achieves open-vocabulary 3D occupancy prediction under 360° panoramic image input that surpasses closed-set supervised methods.

## Background & Motivation

**Background**: 3D semantic occupancy prediction has become a core perception task in autonomous driving and embodied intelligence. Existing methods such as MonoScene, VoxFormer, and SGN have made significant progress under closed-set settings. Meanwhile, panoramic/omnidirectional images are increasingly adopted for scene understanding in embodied agents due to their single-frame 360° coverage.

**Limitations of Prior Work**: (1) Existing 3D occupancy prediction methods are limited to narrow field-of-view inputs and predefined training category distributions, making them difficult to apply in open-world scenarios requiring comprehensive safety perception. (2) Severe geometric distortions and non-uniform sampling introduced by Equirectangular Projection (ERP) cause distant regions to occupy only a minimal portion of the image. (3) Ternary feature alignment among pixels, voxels, and text tends to overfit under imbalanced data distributions, causing novel-class semantic alignment to fail.

**Key Challenge**: A fundamental conflict exists between the geometric distortion characteristics of omnidirectional images and the precision requirements of open-vocabulary semantic alignment—ERP projection yields sparse pixels for distant regions, exacerbating the overfitting risk in cross-modal feature alignment.

**Goal**: Under omnidirectional visual input, the paper aims to simultaneously address three challenges: 360° spatial continuity modeling, open-vocabulary semantic generalization, and cross-modal feature alignment.

**Key Insight**: (1) Adapt to panoramic geometry via polar spiral scanning; (2) construct voxel-text cost volumes to replace direct feature alignment; (3) bridge the modality gap through gradient-free random walks.

**Core Idea**: Integrate the geometric properties of omnidirectional perception into the full pipeline of voxel representation, cost aggregation, and modality alignment, realizing the first omnidirectional open-vocabulary occupancy prediction framework.

## Method

### Overall Architecture

O3N takes panoramic ERP images as input and proceeds through four core stages: (1) a visual feature extractor for omnidirectional image feature extraction; (2) a 2D-to-3D view transformation that generates both Cartesian and cylindrical voxel representations; (3) a 3D decoder augmented with the PsM module to learn fine-grained spatial geometry and semantics; (4) an occupancy prediction head to produce final outputs. For the open-vocabulary branch, the OCA and NMA modules ensure ternary semantic consistency among pixels, voxels, and text.

### Key Designs

1. **Polar Spiral Mamba (PsM) Module**:
    - Function: Captures long-range dependencies in the intrinsic spatial structure of omnidirectional images and resolves data discontinuities near the poles in cylindrical voxels.
    - Mechanism: Adopts a dual-branch architecture that compresses cylindrical voxels $\mathbf{V}_p \in \mathbb{R}^{C \times R \times P \times Z}$ into BEV features $\mathbf{B}_p \in \mathbb{R}^{C \times R \times P}$, then scans outward from the pole along a spiral path (P-SMamba), naturally aligning with the information density variation from near to far in panoramic imaging.
    - Coordinate Fusion: At each layer, cylindrical voxels are resampled into Cartesian space, and the complementary advantages of both coordinate systems are aggregated via $\mathbf{V}_f^i = \mathbf{V}_c^i + \Phi_{\rho(c)}(\mathbf{V}_p^i)$.
    - Design Motivation: Standard 3D convolutions cannot handle the polar discontinuities of cylindrical coordinates; Transformers incur prohibitive computational costs; Mamba's linear complexity and long-sequence modeling capacity make it well-suited for this requirement.

2. **Occupancy Cost Aggregation (OCA) Module**:
    - Function: Constructs a voxel-text cost volume and aggregates it along spatial and category dimensions, avoiding overfitting caused by direct discrete feature alignment.
    - Mechanism: Computes cosine similarity between voxel embeddings $\mathbf{V}$ and text embeddings $\mathbf{T}$ as the occupancy cost $C(i,l) = \frac{V_i \cdot T_l}{\|V_i\| \|T_l\|}$ to generate coarse 3D semantic masks; spatial aggregation is then performed via ASPP, and category aggregation via a linear Transformer.
    - Supervision Strategy: A scene affinity loss $\mathcal{L}_{oca}$ is used to capture inter-voxel semantic correlations, comprising Precision, Recall, and Specificity terms; during training, the loss is computed only over base-class voxels.
    - Design Motivation: Direct cross-entropy supervision leads to isolated voxel-semantic mappings that weaken generalization; cost aggregation preserves the continuity of semantic relationships.

3. **Natural Modality Alignment (NMA)**:
    - Function: Gradient-free bridging of the modality gap between text embeddings and semantic prototypes, preventing over-reliance on seen semantics.
    - Mechanism: Base-class prototypes are updated via EMA as $\mathbf{P}_t^b = \alpha \cdot \mathbf{P}_{t-1}^b + (1-\alpha) \cdot \frac{1}{|\Omega_b|}\sum_{i \in \Omega_b} \mathbf{f}_{seg}(i)$; text embeddings and prototypes are then iteratively aggregated via random walks.
    - Convergent Form: A closed-form solution is derived via the Neumann series: $\mathbf{T}_t^\infty = (1-\beta)(\mathbf{I} - \beta^2 \mathcal{A})^{-1}(\beta \mathcal{S} \mathbf{P}_t^0 + \mathbf{T}_t^0)$.
    - Design Motivation: Learning-based alignment strategies overfit to seen semantic distributions; a gradient-free approach preserves the ability to understand unlimited novel semantics.

### Loss & Training

The total loss consists of three terms: $\mathcal{L} = \mathcal{L}_{occ} + \mathcal{L}_{vox-pix} + \mathcal{L}_{oca}$

- $\mathcal{L}_{occ}$: Semantic occupancy supervision from MonoScene, including cross-entropy loss, semantic/geometric scene affinity loss, and frustum proportion loss.
- $\mathcal{L}_{vox-pix}$: Voxel-pixel feature alignment loss from OVO.
- $\mathcal{L}_{oca}$: Scene affinity loss from occupancy cost aggregation (computed only over base-class voxels).

During inference, base classes are predicted directly via the occupancy head; novel classes are predicted by combining the similarity between voxel embeddings and text embeddings with the OCA-predicted probabilities.

## Key Experimental Results

### Main Results (QuadOcc Validation Set)

| Method | Input | mIoU | Novel mIoU | Base mIoU |
|--------|-------|------|------------|-----------|
| MonoScene (Fully Supervised) | C | 19.19 | 25.56 | 12.82 |
| OneOcc (Fully Supervised) | C | 20.56 | 27.53 | 13.59 |
| OVO (Open-Vocabulary) | C | 14.33 | 18.15 | 10.52 |
| **O3N (Ours)** | C | **16.54** | **21.16** | **11.92** |
| O3N Gain vs. OVO | - | +2.21 | +3.01 | +1.40 |

### Ablation Study (QuadOcc)

| Configuration | mIoU | Novel mIoU | Base mIoU |
|---------------|------|------------|-----------|
| Baseline (OVO) | 14.33 | 18.15 | 10.52 |
| + PsM | 15.21 | 19.43 | 11.00 |
| + PsM + OCA | 15.89 | 20.31 | 11.48 |
| + PsM + OCA + NMA (Full) | **16.54** | **21.16** | **11.92** |

### Key Findings

- O3N under the open-vocabulary setting (mIoU 16.54) even surpasses certain fully supervised methods (e.g., SSCNet 14.60, LMSCNet 18.44).
- Novel-class mIoU improves from 18.15 to 21.16 (+3.01), validating a significant enhancement in open-vocabulary capability.
- The framework is generalizable: it also achieves improvement from 13.81 to 15.52 on the SGN-S backbone.
- State-of-the-art results are also achieved on the Human360Occ dataset, validating cross-scene generalization.

## Highlights & Insights

- **Originality**: The paper is the first to define and address the omnidirectional open-vocabulary occupancy prediction task, unifying panoramic perception with open semantic prediction.
- **Geometry-Aware Design**: The spiral scanning path of PsM naturally aligns with the information density distribution of panoramic imaging, constituting an elegant inductive bias.
- **Theoretical Elegance**: NMA derives a closed-form solution via random walks, simultaneously avoiding the overfitting risk of gradient-based optimization and guaranteeing convergence.
- **Modular Generality**: O3N can be plug-and-play integrated into various occupancy prediction architectures such as MonoScene and SGN.

## Limitations & Future Work

- The absolute performance on novel classes remains low (e.g., vehicle at only 0.52 mIoU), and recognition of extremely rare classes remains challenging.
- Base-class performance degrades under the open-vocabulary setting compared to fully supervised methods, indicating a persistent trade-off between open-vocabulary capability and closed-set accuracy.
- Validation is limited to panoramic datasets; extension to multi-camera inputs has not been explored.
- Computational overhead analysis is insufficient, and real-time feasibility for practical deployment requires further evaluation.

## Related Work & Insights

- **OVO** (Tan et al., 2023): A pioneer in open-vocabulary occupancy prediction; this paper builds upon it by introducing omnidirectional perception and cost aggregation improvements.
- **CAT-Seg** (Cho et al., 2024): The cost aggregation idea from 2D open-vocabulary segmentation is extended in this paper to 3D voxel space.
- **OneOcc** (Shi et al., 2025): The state-of-the-art in omnidirectional fully supervised occupancy prediction, providing baselines and datasets for this paper.
- **Insight**: The paradigm of replacing direct feature alignment with cost aggregation merits broader adoption in other cross-modal tasks.

## Rating (⭐)

| Dimension | Rating |
|-----------|--------|
| Novelty | ⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐ |
| **Overall** | **⭐⭐⭐⭐** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)](monocular_open_vocabulary_occupancy_prediction_for_indoor_scenes.md)
- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[CVPR 2026\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction.md)
- [\[CVPR 2026\] ProOOD: Prototype-Guided Out-of-Distribution 3D Occupancy Prediction](proood_prototype-guided_out-of-distribution_3d_occupancy_prediction.md)
- [\[CVPR 2026\] An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving](an_instance-centric_panoptic_occupancy_prediction_benchmark_for_autonomous_drivi.md)

</div>

<!-- RELATED:END -->
