---
title: >-
  [Paper Note] EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding
description: >-
  [ICCV 2025][Autonomous Driving][3D occupancy prediction] This paper proposes EmbodiedOcc, a framework that leverages 3D semantic Gaussians as a global memory to enable online indoor 3D occupancy prediction from monocular visual input through progressive exploration and local updating.
tags:
  - ICCV 2025
  - Autonomous Driving
  - 3D occupancy prediction
  - embodied perception
  - 3D Gaussian
  - online scene understanding
  - indoor scene
date: 2026-05-08
content_hash: a6a3f41f8204f70b
---

# EmbodiedOcc: Embodied 3D Occupancy Prediction for Vision-based Online Scene Understanding

**Conference**: ICCV 2025
**arXiv**: [2412.04380](https://arxiv.org/abs/2412.04380)
**Code**: [https://github.com/YkiWu/EmbodiedOcc](https://github.com/YkiWu/EmbodiedOcc)
**Area**: Autonomous Driving
**Keywords**: 3D occupancy prediction, embodied perception, 3D Gaussian, online scene understanding, indoor scene

## TL;DR

This paper proposes EmbodiedOcc, a framework that leverages 3D semantic Gaussians as a global memory to enable online indoor 3D occupancy prediction from monocular visual input through progressive exploration and local updating.

## Background & Motivation

3D occupancy prediction provides agents with a comprehensive understanding of their surroundings and has become a core task in 3D perception. Existing methods focus primarily on offline perception or multi-view 3D occupancy prediction, making them inapplicable to embodied agents that require progressive scene exploration. In indoor environments in particular, multiple traversals are typically needed to achieve room-level global understanding, rather than a single prediction within a local frustum.

The authors observe that: (1) indoor 3D occupancy prediction requires online global scene understanding updated from streaming monocular RGB input; (2) humans perceive new environments by incrementally accumulating knowledge through embodied exploration; and (3) existing outdoor methods (e.g., TPVFormer, SurroundOcc) transfer poorly to indoor scenes, as they focus on coarse layout rather than fine-grained structure.

## Method

### Overall Architecture

EmbodiedOcc adopts a two-stage training pipeline: a local refinement module is first trained for single-frame frustum occupancy prediction, after which the trained local module is used to train the global online framework with Gaussian memory. The framework initializes the scene with uniformly distributed 3D semantic Gaussians in world coordinates, updates the Gaussians within the current frustum at each timestep based on new observations, and obtains the global 3D occupancy via Gaussian-to-voxel splatting.

### Key Designs

1. **Local Refinement Module**: Represents the current frustum using 16,200 3D semantic Gaussians. Each Gaussian is parameterized by mean $\mathbf{m}$, scale $\mathbf{s}$, rotation quaternion $\mathbf{r}$, opacity $\mathbf{o}$, and semantic logits $\mathbf{c}$. An embedding layer lifts Gaussian vectors into high-dimensional features; 3D sparse convolution then enables inter-Gaussian interaction, and deformable attention integrates image features for refinement. **Design Motivation**: Compared to voxel-based representations, Gaussians offer greater flexibility and are better suited for local-global interaction.

2. **Depth-Aware Branch**: Employs DepthAnything-V2 to predict a metric depth map $D_{metric}$. Each Gaussian's mean is projected into image coordinates to retrieve a sampled depth value $d$, which is fed together with the Gaussian's z-component in camera coordinates into a 3-layer MLP to produce a depth-aware feature $\mathbf{Q}_{depth}$ added to the original feature. The formulation is: $\mathbf{Q}_{depth} = \mathcal{M}_{depthaware}(D_{metric}(u,v), z)$, $\hat{\mathbf{Q}}_i = \mathbf{Q}_i + \mathbf{Q}_i^{depth}$. **Design Motivation**: Depth information not only affects Gaussian means but also facilitates the updating of other attributes such as semantics and opacity.

3. **Gaussian Memory with Confidence Refinement**: An explicit global Gaussian memory is maintained in world coordinates. Each Gaussian carries a label $\gamma \in \{0,1\}$ indicating whether it has been updated. For previously updated Gaussians ($\gamma=1$), a confidence score $\theta$ balances memory information against current input: $\Delta\mathbf{G}_{online} = (1-\theta)\Delta\mathbf{G}$. During the first two refinement layers $\theta=0$ (frozen), and $\theta=0.5$ in the final layer. **Design Motivation**: This mimics human fine-tuning behavior when revisiting known scenes, preventing the corruption of prior high-quality predictions.

### Loss & Training

Training employs a weighted combination of four loss functions:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{focal} + \mathcal{L}_{lov} + \mathcal{L}_{scal}^{geo} + \mathcal{L}_{scal}^{sem}$$

- Focal loss ($\mathcal{L}_{focal}$): handles class imbalance
- Lovász-softmax loss ($\mathcal{L}_{lov}$): directly optimizes IoU
- Scene-class affinity loss (geometric and semantic variants)

The local module is first trained for 10 epochs on Occ-ScanNet, followed by 5 epochs of EmbodiedOcc training on EmbodiedOcc-ScanNet. The AdamW optimizer is used with a cosine decay schedule and a peak learning rate of 2e-4 after warm-up.

## Key Experimental Results

### Main Results

**Local Occupancy Prediction (Occ-ScanNet)**:

| Method | Input | IoU | mIoU |
|--------|-------|-----|------|
| TPVFormer | RGB | 33.39 | 24.94 |
| GaussianFormer | RGB | 40.91 | 29.93 |
| MonoScene | RGB | 41.60 | 24.62 |
| SurroundOcc | RGB | 42.52 | 30.83 |
| **EmbodiedOcc** | **RGB** | **53.55** | **45.15** |

**Embodied Occupancy Prediction (EmbodiedOcc-ScanNet)**:

| Method | IoU | mIoU |
|--------|-----|------|
| TPVFormer | 35.88 | 25.70 |
| GaussianFormer | 38.02 | 27.36 |
| SplicingOcc (local stitching) | 49.01 | 40.74 |
| **EmbodiedOcc** | **51.52** | **42.53** |

### Ablation Study

**Model Design Analysis**:

| Method | Gaussian | Structure-Aware | Memory | Local IoU/mIoU | Embodied IoU/mIoU |
|--------|----------|----------------|--------|----------------|-------------------|
| Voxel variant | ✗ | ✓ | ✓ | 47.50/38.12 | 37.53/26.99 |
| w/o memory | ✓ | ✓ | ✗ | 53.55/45.15 | 49.01/40.74 |
| **Full model** | **✓** | **✓** | **✓** | **53.55/45.15** | **51.52/42.53** |

**Depth Branch Ablation**:

| Branch Type | Local IoU/mIoU | Embodied IoU/mIoU |
|-------------|----------------|-------------------|
| w/o depth branch | 48.15/40.07 | 37.52/30.73 |
| Naive depth branch | 50.32/42.73 | - |
| **Depth-aware branch (DAv2)** | **53.93/46.20** | **50.78/41.45** |

### Key Findings

- Look-Back evaluation validates the effectiveness of continuous updating: revisiting explored regions improves mIoU from 40.03 to 40.98 at K=5.
- The voxel variant performs adequately in local prediction but collapses globally, confirming the advantage of Gaussian representations for local-global interaction.
- A moderate confidence update of $\theta=0.5$ yields the best performance.
- Runtime analysis: approximately 114ms per frame, with 61ms for the image backbone and 35ms for depth estimation.

## Highlights & Insights

1. **Task Formalization**: The paper is the first to formally define the "embodied 3D occupancy prediction" task, bridging the gap between offline local prediction and online global perception.
2. **Gaussian Memory Mechanism**: Explicit 3D Gaussians maintain global memory, preserving structural flexibility while enabling efficient local-global information fusion.
3. **Alignment with Human Cognition**: The confidence mechanism mirrors human cognitive behavior when revisiting scenes—fine-tuning knowledge of familiar regions while reconstructing novel ones.
4. **Benchmark Construction**: The EmbodiedOcc-ScanNet benchmark is reorganized, comprising 537/137 training/validation scenes.

## Limitations & Future Work

1. Validation is limited to indoor scenes; extension to large-scale outdoor environments has not been explored.
2. The stopping mechanism is simplistic (threshold based on the proportion of updated Gaussians), lacking learned exploration strategies.
3. The method relies on a pretrained depth estimation model (DepthAnything-V2), whose errors propagate through the pipeline.
4. Per-frame runtime of approximately 114ms remains insufficient for real-time applications.
5. Combining the explicit Gaussian memory with language models to support semantic scene querying is a promising future direction.

## Related Work & Insights

- **GaussianFormer**: The first to apply 3D Gaussians to outdoor 3D occupancy prediction; this work extends the paradigm to indoor embodied scenarios.
- **Online3D**: A pioneer in online 3D perception, but relies on RGB-D input and targets point segmentation and detection rather than occupancy.
- **MonoScene/ISO**: Representative works in monocular 3D occupancy prediction; this paper builds upon them by introducing a global online framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to define the embodied 3D occupancy prediction task; the Gaussian memory design is natural and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive multi-dimensional ablation studies; the Look-Back evaluation is a clever design choice.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, detailed method description, and carefully designed figures and tables.
- **Value**: ⭐⭐⭐⭐ Provides a practical solution for scene understanding in embodied intelligence with open-source code.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Semantic Causality-Aware Vision-Based 3D Occupancy Prediction](semantic_causality-aware_vision-based_3d_occupancy_prediction.md)
- [\[ICCV 2025\] SA-Occ: Satellite-Assisted 3D Occupancy Prediction in Real World](sa-occ_satellite-assisted_3d_occupancy_prediction_in_real_world.md)
- [\[ICCV 2025\] AGO: Adaptive Grounding for Open World 3D Occupancy Prediction](ago_adaptive_grounding_for_open_world_3d_occupancy_predictio.md)
- [\[ICCV 2025\] Hermes: A Unified Self-Driving World Model for Simultaneous 3D Scene Understanding and Generation](hermes_a_unified_self-driving_world_model_for_simultaneous_3d_scene_understandin.md)
- [\[ICCV 2025\] GaussRender: Learning 3D Occupancy with Gaussian Rendering](gaussrender_learning_3d_occupancy_with_gaussian_rendering.md)

<!-- RELATED:END -->
