---
title: >-
  [Paper Note] REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion
description: >-
  [AAAI 2026][Object Detection][Radar Perception] This paper extends the 2D bounding box diffusion paradigm of DiffusionDet to 3D radar space, proposing the REXO framework. It enables explicit cross-view radar feature association guided by noisy 3D bounding box projections, and introduces a ground-level constraint to reduce the diffusion parameter space. REXO surpasses the state of the art by +4.22 AP and +11.02 AP on the HIBER and MMVR indoor radar datasets, respectively.
tags:
  - AAAI 2026
  - Object Detection
  - Radar Perception
  - Multi-View Fusion
  - Diffusion Model
  - 3D Object Detection
  - Indoor Human Detection
  - Cross-View Feature Association
date: 2026-05-08
content_hash: fdc356f2412674bd
---

# REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion

**Conference**: AAAI 2026
**arXiv**: [2511.17806](https://arxiv.org/abs/2511.17806)
**Code**: [https://github.com/merlresearch/radar-bbox-diffusion](https://github.com/merlresearch/radar-bbox-diffusion)
**Area**: Object Detection
**Keywords**: Radar Perception, Multi-View Fusion, Diffusion Model, 3D Object Detection, Indoor Human Detection, Cross-View Feature Association

## TL;DR

This paper extends the 2D bounding box diffusion paradigm of DiffusionDet to 3D radar space, proposing the REXO framework. It enables explicit cross-view radar feature association guided by noisy 3D bounding box projections, and introduces a ground-level constraint to reduce the diffusion parameter space. REXO surpasses the state of the art by +4.22 AP and +11.02 AP on the HIBER and MMVR indoor radar datasets, respectively.

## Background & Motivation

Indoor radar perception has attracted increasing attention due to its low cost and minimal privacy intrusion. However, cross-view feature fusion between horizontal and vertical radar heatmaps remains a core challenge:

**RFMask**: Generates proposals on the horizontal view using Faster R-CNN, then pairs them with fixed-height vertical windows → implicit and constrained.

**RETR**: Uses DETR-style decoder cross-attention to implicitly associate queries with both views → feature matching is ambiguous.

**Direct adaptation of DiffusionDet to radar**: Performs 2D bounding box diffusion on horizontal radar heatmaps, still requiring fixed-height vertical pairing.

The common limitation of these approaches is that **cross-view association is implicit**, which leads to ambiguous matching in complex indoor scenes and degraded detection performance. This problem is particularly acute when multiple detected persons are at similar depths, causing severe aliasing of vertical-view reflected signals that implicit methods struggle to resolve.

## Method

### Overall Architecture

REXO lifts the diffusion state from 2D image space to 3D radar space: $\boldsymbol{x}_t = \{c_x^t, c_y^t, c_z^t, w^t, h^t, d^t\}^\top \in \mathbb{R}^6$. During training, ground-truth 3D bounding boxes are corrupted with noise; during inference, the model iteratively denoises randomly initialized 3D bounding boxes. Feature maps from both radar views are extracted by a shared-weight ResNet backbone and processed through an FPN to produce multi-scale features.

### Key Design 1: Explicit Cross-View Radar Feature Association

At each diffusion timestep, the noisy 3D bounding box $\boldsymbol{x}_t$ is projected onto the horizontal and vertical 2D views to obtain corresponding 2D bounding boxes. RoIAlign is then applied to crop $C \times r \times r$ features from each view's feature map, which are concatenated to form $\boldsymbol{Z}_{\text{radar}}^{\text{crop}} \in \mathbb{R}^{C \times r \times 2r}$.

**Core advantage**: This bounding-box-guided association scales **linearly** with the number of views (whereas proposal- or query-based schemes scale **quadratically**), and is semantically more precise since the 3D bounding box already encodes spatial position.

### Key Design 2: Radar-Conditioned Denoising Detector (DenoisingDet)

The concatenated cross-view radar features $\boldsymbol{Z}_{\text{radar}}^{\text{crop}}$ are fed into a time-dependent Predictor consisting of self-attention, dynamic convolution, and time embeddings. A BBox Head then predicts 3D bounding box offsets, and a Class Head predicts object categories. The denoising step is **naturally conditioned on cross-view radar features**, requiring no additional fusion module design.

### Key Design 3: Ground-Level Constraint

Leveraging the prior knowledge that persons stand on the ground in indoor environments, the vertical center $c_y^t$ is directly set to $h^t/2$, reducing the diffusion parameter space from 6 to 5 dimensions. This constraint not only reduces the search space but also enables joint gradient flow between 3D and 2D representations, leading to faster convergence and improved generalization.

### Loss & Training

A geometry-aware dual-domain loss is employed:

$$\mathcal{L}_{\text{box}}^{\text{GA}} = \lambda_{3D} \mathcal{L}_{\text{box}}^{3D}(\boldsymbol{x}_{\text{radar}}, \hat{\boldsymbol{x}}_{\text{radar}}) + \lambda_{2D} \mathcal{L}_{\text{box}}^{2D}(\boldsymbol{b}_{\text{image}}, \hat{\boldsymbol{b}}_{\text{image}})$$

where each bounding box loss is a weighted combination of GIoU loss and L1 loss. 3D bounding boxes are mapped to camera coordinates via calibrated rotation matrices and translation vectors, then projected to the image plane to obtain 2D bounding boxes; a learnable Refinement module corrects over-projection artifacts. Hungarian matching is used for optimal assignment.

## Key Experimental Results

### Main Results: MMVR Dataset (4 Data Splits)

| Method | P1S1 AP | P1S2 AP | P2S1 AP | P2S2 AP |
|--------|---------|---------|---------|---------|
| RFMask | 25.53 | 24.46 | 31.37 | 6.03 |
| RFMask3D | 34.84 | 30.75 | 39.89 | 12.26 |
| DETR | 35.64 | 28.51 | 29.53 | 9.29 |
| RETR | 39.62 | 30.16 | 46.75 | 12.45 |
| **REXO** | **39.23** | **36.48** | **48.35** | **23.47** |

On the most challenging split P2S2 (completely unseen environments), REXO improves AP from 12.45 to 23.47, a **gain of +11.02**, demonstrating strong generalization ability.

### Ablation Study: P2S2 on MMVR

| Ablation Dimension | Configuration | AP |
|--------------------|---------------|----|
| Ground-level constraint | ✗ | 22.67 |
| Ground-level constraint | ✓ | **23.47** |
| $\lambda_{3D}=0$, $\lambda_{2D}=1$ | 2D supervision only | 0.98 |
| $\lambda_{3D}=1$, $\lambda_{2D}=0.1$ | Weak 2D | 15.55 |
| $\lambda_{3D}=1$, $\lambda_{2D}=1$ | Balanced supervision | **23.47** |
| Inference steps $S=1$ | — | 23.48 |
| Inference steps $S=10$ | — | **24.27** |
| DiffusionDet (horizontal only) | Baseline | 20.75 |
| REXO (dual-view) | — | **23.47** |

### Key Findings

1. **3D supervision is critical**: Without the 3D loss ($\lambda_{3D}=0$), AP drops to near zero, indicating that 3D accuracy in radar coordinate space is essential for 2D prediction in the image plane.
2. **Robustness to number of inference boxes**: REXO exhibits minimal performance degradation as the number of inference bounding boxes increases from 2 to 80 (23.48→21.70), whereas RETR degrades sharply (12.45→2.16).
3. **Depth proximity challenge**: When the depth difference between two persons is less than 20 cm, AP drops sharply from 23.47 to 9.93, revealing the fundamental difficulty of signal aliasing in the vertical view.
4. **HIBER dataset**: REXO achieves 25.33 AP, surpassing RETR's 22.09, a **gain of +3.24 AP**.

## Highlights & Insights

- **Conceptual elegance**: Lifting 2D diffusion directly to 3D space is a natural and elegant idea that resolves cross-view association in a unified manner.
- **Computational advantage of explicit association**: Linear complexity versus quadratic complexity, enabling scalability to more views.
- **Effective use of the ground-level constraint**: Incorporating physical priors into the diffusion process reduces the parameter space and accelerates training convergence.
- **Outstanding generalization**: The decisive performance advantage in completely unseen environments (P2S2) demonstrates the method's robustness.
- **Open-source code**.

## Limitations & Future Work

1. Performance degrades significantly when two persons are at similar depths, as the angular resolution of the vertical view is insufficient to disambiguate overlapping reflections.
2. The ground-level constraint assumes persons are standing on the floor; it becomes inaccurate when subjects are jumping or standing on elevated objects.
3. Validation is limited to indoor radar scenarios and has not been extended to outdoor autonomous driving radar.
4. Increasing the number of inference steps incurs significant latency (1 step: 17 FPS → 10 steps: 2 FPS).

## Related Work & Insights

- **DiffusionDet** (Chen et al., 2023): The diffusion model paradigm for 2D image detection; the direct foundation of REXO.
- **RETR** (Yataka et al., 2024): A DETR-based Transformer for multi-view radar detection; the primary baseline for REXO.
- **RFMask** (Wu et al., 2023): A cross-view scheme based on Faster R-CNN with fixed-height vertical windows.
- **Diffusion-SS3D** (Ho et al., 2023): Application of diffusion models to semi-supervised 3D object detection.

## Rating

- Novelty: ⭐⭐⭐⭐ (Lifting 2D→3D diffusion and the cross-view association insight are highly original)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Two datasets, extensive ablation studies, and visualization analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear diagrams and complete technical description)
- Value: ⭐⭐⭐⭐ (Practical advancement in indoor radar perception; open-source code enhances reproducibility)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction](../../ICCV2025/object_detection/boosting_multiview_indoor_3d_object_detection_via_adaptive_3.md)
- [\[ICCV 2025\] SGCDet: Boosting Multi-View Indoor 3D Object Detection via Adaptive 3D Volume Construction](../../ICCV2025/object_detection/boosting_multi-view_indoor_3d_object_detection_via_adaptive_3d_volume_constructi.md)
- [\[CVPR 2026\] Few-Shot Incremental 3D Object Detection in Dynamic Indoor Environments](../../CVPR2026/object_detection/few-shot_incremental_3d_object_detection_in_dynamic_indoor_environments.md)
- [\[AAAI 2026\] Real-Time 3D Object Detection with Inference-Aligned Learning](real-time_3d_object_detection_with_inference-aligned_learning.md)
- [\[AAAI 2026\] MonoCLUE: Object-Aware Clustering Enhances Monocular 3D Object Detection](monoclue_object-aware_clustering_enhances_monocular_3d_object_detection.md)

</div>

<!-- RELATED:END -->
