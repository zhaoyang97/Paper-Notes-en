---
title: >-
  [Paper Note] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction
description: >-
  [CVPR 2026][Autonomous Driving][Omnidirectional Perception] O3N is the first work to introduce the omnidirectional open-vocabulary occupancy prediction task and proposes a purely vision-based end-to-end framework. Polar-…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "Omnidirectional Perception"
  - "Open-Vocabulary"
  - "Occupancy Prediction"
  - "Panoramic Images"
  - "Mamba"
date: 2026-05-08
content_hash: 3110d58c29a6201c
---

# O3N: Omnidirectional Open-Vocabulary Occupancy Prediction

**Conference**: CVPR 2026
**arXiv**: [2603.12144](https://arxiv.org/abs/2603.12144)
**Code**: [GitHub](https://github.com/) (coming soon)
**Area**: Autonomous Driving
**Keywords**: Omnidirectional Perception, Open-Vocabulary, Occupancy Prediction, Panoramic Images, Mamba

## TL;DR

O3N is the first work to introduce the omnidirectional open-vocabulary occupancy prediction task and proposes a purely vision-based end-to-end framework. Polar-spiral Mamba (PsM) models panoramic geometric continuity via spiral scanning in polar coordinate space; Occupancy Cost Aggregation (OCA) constructs a voxel-text matching cost volume to avoid overfitting from direct feature alignment; Natural Modality Alignment (NMA) aligns pixel-voxel-text tri-modal embeddings through gradient-free random walk. The method achieves 16.54 mIoU / 21.16 Novel mIoU on QuadOcc (SOTA), substantially outperforming the OVO baseline.

## Background & Motivation

**Background**: Omnidirectional images (360° panoramas) are indispensable in autonomous driving and embodied intelligence, providing complete spatial coverage and semantic continuity. 3D semantic occupancy prediction lifts 2D visual perception into 3D space and serves as the foundation for precise spatial reasoning.

**Dual Limitations of Existing Methods**:
- **Viewpoint Limitation**: Most existing occupancy prediction methods rely on multi-view surround cameras (e.g., 6 cameras in nuScenes) and are ill-suited for robots and embodied agents that use a single panoramic camera.
- **Closed Vocabulary**: Existing methods can only recognize semantic categories predefined during training and fail to generalize to unknown objects in the open world (e.g., misclassifying a box as road, or a dog as a bicycle).

**Unique Challenges of Panoramic Images**: Equirectangular projection (ERP) introduces severe geometric distortion—regions far from the viewpoint occupy progressively less area in the image (latitude distortion + extension distortion), leading to: (a) non-uniform pixel-to-voxel mapping; (b) naive tri-modal feature alignment strategies that easily overfit to visible semantics and misalign novel class semantics.

**Contributions**: This paper is the first to define the **omnidirectional open-vocabulary occupancy prediction** task—taking a single panoramic RGB image and arbitrary class-name text as input and outputting 3D semantic occupancy including unseen categories—and proposes O3N, the first purely vision-based end-to-end framework for this task.

## Method

### Overall Architecture

Input: equirectangular panoramic image → CLIP visual encoder extracts image features + CLIP text encoder extracts class-name embeddings → 2D-to-3D view transformation (generating dual voxel representations in cuboid and cylindrical forms) → 3D decoder (with integrated PsM) → OCA + NMA modules → occupancy prediction head outputs voxel-level semantic labels.

Key innovation: three modules respectively address panoramic geometry modeling (PsM), open-vocabulary semantic learning (OCA), and cross-modal alignment (NMA).

### Key Designs

1. **Polar-spiral Mamba (PsM) Module**:

    - **Function**: Efficiently models omnidirectional 3D voxel features in polar coordinate space while preserving spatial continuity.
    - **Core Problem**: Cylindrical voxels exhibit data discontinuity at angular boundaries in polar coordinates (especially near the poles); standard 3D convolutions cannot adapt, and Transformers incur prohibitive computational cost.
    - **Design**: **Dual-branch architecture**—
        - *Polar branch*: Cylindrical voxels $\mathbf{V}_p \in \mathbb{R}^{C \times R \times P \times Z}$ are compressed into BEV features $\mathbf{B}_p \in \mathbb{R}^{C \times R \times P}$, then processed by **P-SMamba spiral scanning**—a spiral path emanating from the pole with increasing radius, naturally matching the information density distribution of panoramic imaging (dense near, sparse far).
        - *Cartesian branch*: Cuboid voxels $\mathbf{V}_c \in \mathbb{R}^{C \times H \times W \times D}$.
        - *Cross-coordinate aggregation*: Resamples and fuses using precomputed polar-Cartesian projection mappings: $\mathbf{V}_f^i = \mathbf{V}_c^i + \Phi_{\rho(c)}(\mathbf{V}_p^i)$.
    - **Advantage**: Spatial-Mamba provides Transformer-level long-range modeling with only linear complexity; the spiral scan path guarantees spatial continuity in polar regions.

2. **Occupancy Cost Aggregation (OCA)**:

    - **Function**: Constructs a voxel-text matching cost volume as a substitute for direct feature alignment, alleviating overfitting in the open-vocabulary setting.
    - **Mechanism**: Analogous to image-text matching costs in 2D open-vocabulary segmentation, defines the **occupancy cost** $C(i,l) = \frac{V_i \cdot T_l}{\|V_i\| \|T_l\|}$ (cosine similarity between voxel and text embeddings) → 3D convolution extracts initial cost embeddings → ASPP spatial aggregation (multi-scale receptive fields) → Linear Transformer for inter-class aggregation → residual prediction.
    - **Scene Affinity Loss** $\mathcal{L}_{oca}$: Instead of simple cross-entropy (which leads to isolated voxel-semantic mappings), jointly measures intra-class and inter-class voxel relationships using Precision + Recall + Specificity, improving generalization.
    - $\mathcal{L}_{oca}$ is computed only on base-class voxels during training.

3. **Natural Modality Alignment (NMA)**:

    - **Function**: Aligns text embeddings and semantic prototypes in a gradient-free manner to close the inherent CLIP image-text domain gap.
    - **Core Problem**: Despite large-scale pretraining, CLIP still exhibits a modality gap between image and text embeddings; panoramic projection errors further exacerbate this. Learnable alignment strategies tend to overfit to base-class distributions.
    - **Design**: **Gradient-free Random Walk iterative alignment**—
        - EMA update of base-class semantic prototypes: $\mathbf{P}_t^b = \alpha \cdot \mathbf{P}_{t-1}^b + (1-\alpha) \cdot \bar{\mathbf{f}}_{seg}$
        - Compute text-prototype affinity $\mathcal{S} = \lambda \frac{\mathbf{T}_t^0 \cdot \mathbf{P}_t^0}{\|\mathbf{T}_t^0\| \|\mathbf{P}_t^0\|}$
        - Alternately update prototypes and text embeddings via Random Walk until convergence: $\mathbf{T}_t^\infty = (1-\beta)(\mathbf{I} - \beta^2 \mathcal{A})^{-1}(\beta \mathcal{S} \mathbf{P}_t^0 + \mathbf{T}_t^0)$
    - **Key Detail**: Learnable prototypes for novel classes are also introduced (implicitly capturing unseen semantics); the entire process involves no gradient backpropagation, preventing overfitting to the training distribution.

### Loss & Training

- **Total Loss**: $\mathcal{L} = \mathcal{L}_{occ} + \mathcal{L}_{vox-pix} + \mathcal{L}_{oca}$
    - $\mathcal{L}_{occ}$: Cross-entropy + geometric/semantic scene-class affinity loss + focal point loss (standard MonoScene losses).
    - $\mathcal{L}_{vox-pix}$: Voxel-pixel feature alignment loss (from OVO).
    - $\mathcal{L}_{oca}$: Scene affinity loss (base-class voxels only).
- **Inference Strategy**: Base classes are predicted directly via the occupancy head; novel classes are predicted by combining the cosine similarity between voxel embeddings $\mathbf{V}$ from the distillation module and novel class text embeddings with OCA prediction probabilities.
- **Training Configuration**: MonoScene as the backbone network, 25 epochs, 4×RTX3090, batch size = 4.

## Key Experimental Results

### Main Results (QuadOcc Validation Set)

| Method | Type | mIoU | Novel mIoU | Base mIoU |
|--------|------|------|-----------|-----------|
| MonoScene (fully supervised) | Camera | 19.19 | 25.56 | 12.82 |
| OneOcc (fully supervised) | Camera | 20.56 | 27.53 | 13.59 |
| OVO (open-vocabulary) | Camera | 14.33 | 18.15 | 10.52 |
| **O3N (open-vocabulary)** | Camera | **16.54** | **21.16** | **11.92** |

- O3N surpasses OVO by +2.21 mIoU / +3.01 Novel mIoU.
- O3N's Novel mIoU (21.16) exceeds multiple fully supervised methods (SSCNet 20.13, OccFormer 20.04, VoxFormer-S 14.54).
- Consistent gains are also observed on the SGN-S backbone (13.81→15.52 mIoU), demonstrating framework generality.

### Ablation Study

| Configuration | Novel mIoU | Base mIoU | mIoU | FPS | Memory (GB) |
|---------------|-----------|-----------|------|-----|------------|
| Baseline (w/o three modules) | 18.06 | 10.90 | 14.48 | 10.67 | 4.28 |
| + PsM | 18.59 (+0.53) | 11.05 | 14.82 | 9.98 | 4.31 |
| + PsM + OCA | 19.78 (+1.72) | 11.02 | 15.40 | 9.71 | 4.86 |
| + PsM + OCA + NMA | **21.16 (+3.10)** | **11.92** | **16.54** | 9.41 | 4.97 |

### Key Findings

- **PsM**: Polar spiral scanning contributes +0.53 Novel mIoU with negligible memory overhead (+0.03 GB) and linear complexity.
- **OCA**: Cost volume aggregation is the primary performance driver, contributing +1.72 Novel mIoU and significantly reducing overfitting in the open-vocabulary setting.
- **NMA**: Gradient-free alignment further releases +1.38 Novel mIoU, underscoring the importance of closing the modality gap.
- **Efficiency**: The full O3N model maintains 9.41 FPS / 4.97 GB memory, supporting near-real-time inference.
- **H3O Dataset**: Consistent improvements are also achieved on the human-perspective simulation dataset (23.39→24.25 mIoU).

## Highlights & Insights

- **Pioneering Task Definition**: The paper is the first to define omnidirectional open-vocabulary occupancy prediction, opening a new research direction for embodied intelligence and robotic perception.
- **Geometric Insight of Polar Spiral Scanning**: The spiral path design of P-SMamba precisely matches the information density distribution of panoramic imaging—dense near the center and sparse at the periphery—offering an elegant solution to ERP distortion.
- **Gradient-Free Alignment Prevents Overfitting**: NMA aligns modal embeddings via Random Walk combined with a closed-form Neumann series solution, providing both theoretical guarantees (convergence) and immunity to base-class overfitting during training.
- **Modularity and Generality**: O3N can be plugged into different occupancy networks such as MonoScene and SGN without depending on a specific architecture.

## Limitations & Future Work

- **Limited Scene Scale**: QuadOcc contains only 6 semantic classes and H3O only 10; the true challenge of open-vocabulary recognition (dozens to hundreds of classes) remains untested.
- **High Novel Class Ratio**: In QuadOcc, vehicle/road/building account for ~68% of voxels; in H3O, novel classes cover ~75% of the scene—novel classes actually dominate most of the scene, making generalization relatively tractable.
- **Weak Panoramic Baselines**: The compared methods (MonoScene, SGN, etc.) are relatively early architectures; comparisons with stronger occupancy methods (e.g., SurroundOcc, GaussianFormer) are absent.
- **Single-Frame Input Only**: Temporal information is not exploited; multi-frame panoramic input could substantially improve performance.
- **Future Directions**: (a) Scale to larger semantic vocabularies and real outdoor scenes; (b) incorporate temporal modeling; (c) integrate with LLMs for interactive scene understanding.

## Related Work & Insights

- **vs. OVO**: OVO pioneered open-vocabulary occupancy prediction using a frozen 2D segmentor with CLIP distillation; O3N builds on this by adding OCA (cost volume) and NMA (gradient-free alignment) to address overfitting and the modality gap, respectively.
- **vs. OneOcc**: OneOcc achieves purely vision-based panoramic occupancy prediction but with a closed vocabulary; O3N extends this to the open-vocabulary setting.
- **vs. CAT-Seg (2D)**: The cost aggregation idea in OCA is inspired by image-text matching costs in 2D open-vocabulary segmentation; O3N extends this paradigm to 3D voxel space.
- **Insights**: The polar coordinate representation combined with spiral scanning can be generalized to other panoramic tasks (e.g., panoramic depth estimation, panoramic detection); the gradient-free alignment strategy is also applicable to other open-vocabulary tasks with domain gaps.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to define omnidirectional open-vocabulary occupancy prediction; each of the three modules has a clearly motivated design insight.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Dual datasets + dual backbones + thorough ablations, though benchmark scale is limited.
- Writing Quality: ⭐⭐⭐⭐ — Derivations are clear (Neumann series for NMA); method diagrams are detailed.
- Value: ⭐⭐⭐⭐ — Opens a new task and methodology for embodied intelligence and panoramic perception; the direction is well-motivated and practically meaningful.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)](monocular_open_vocabulary_occupancy_prediction_for_indoor_scenes.md)
- [\[CVPR 2026\] Generalizing Visual Geometry Priors to Sparse Gaussian Occupancy Prediction](generalizing_visual_geometry_priors_to_sparse_gaussian_occupancy_prediction.md)
- [\[CVPR 2026\] Open-Vocabulary Domain Generalization in Urban-Scene Segmentation](open-vocabulary_domain_generalization_in_urban-scene_segmentation.md)
- [\[CVPR 2026\] Dr.Occ: Depth- and Region-Guided 3D Occupancy from Surround-View Cameras for Autonomous Driving](drocc_depth_region_guided_3d_occupancy.md)
- [\[CVPR 2026\] Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](panoramic_multimodal_semantic_occupancy_prediction_for_quadruped_robots.md)

</div>

<!-- RELATED:END -->
