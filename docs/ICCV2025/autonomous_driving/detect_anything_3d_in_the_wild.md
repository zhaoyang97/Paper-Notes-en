---
title: >-
  [Paper Note] Detect Anything 3D in the Wild
description: >-
  [ICCV 2025][Autonomous Driving][3D detection foundation model] DetAny3D is a promptable 3D detection foundation model that transfers prior knowledge from two 2D foundation models—SAM and depth-pretrained DINO—via a propo…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "3D detection foundation model"
  - "zero-shot generalization"
  - "monocular 3D detection"
  - "2D-to-3D knowledge transfer"
  - "open-world detection"
date: 2026-05-08
content_hash: 2eb6b239852f598f
---

# Detect Anything 3D in the Wild

**Conference**: ICCV 2025
**arXiv**: [2504.07958](https://arxiv.org/abs/2504.07958)
**Code**: [https://github.com/OpenDriveLab/DetAny3D](https://github.com/OpenDriveLab/DetAny3D)
**Area**: Autonomous Driving / 3D Object Detection
**Keywords**: 3D detection foundation model, zero-shot generalization, monocular 3D detection, 2D-to-3D knowledge transfer, open-world detection

## TL;DR

DetAny3D is a promptable 3D detection foundation model that transfers prior knowledge from two 2D foundation models—SAM and depth-pretrained DINO—via a proposed 2D Aggregator and Zero-Embedding Mapping (ZEM) mechanism, enabling stable 2D-to-3D knowledge transfer. Using only monocular images, it achieves zero-shot 3D object detection across arbitrary scenes and camera configurations, surpassing baselines by up to 21% AP3D on novel categories.

## Background & Motivation

3D object detection is a core technology for autonomous driving, robotics, and augmented reality. An ideal general-purpose 3D detector should detect arbitrary objects from monocular image input without relying on specific sensor parameters. However, existing methods suffer from the following limitations:

- **Closed-set assumption**: Existing detectors (e.g., Cube R-CNN, Omni3D) support multi-dataset training but remain confined to predefined category spaces, failing to detect unseen objects.
- **Camera configuration sensitivity**: Cross-dataset deployment suffers from severe domain gaps due to differences in camera parameters.
- **Scarcity of 3D annotation data**: 3D annotations number only in the millions, 3–4 orders of magnitude fewer than 2D image annotations (billions), making training a 3D foundation model from scratch nearly infeasible.

The root cause is that **3D data is insufficient to support foundation model training, yet 2D foundation models (SAM, DINOv2) possess rich shape and geometric priors**. The paper's starting point is to leverage pretrained 2D foundation model knowledge to compensate for the scarcity of 3D data, achieving effective 2D-to-3D knowledge transfer through carefully designed architecture.

## Method

### Overall Architecture

DetAny3D takes a monocular RGB image and prompts (box/point/text/intrinsic) as input. The image is encoded in parallel by two foundation models: SAM (providing pixel-level shape information as a promptable backbone) and depth-pretrained DINO (providing geometric depth priors via UniDepth pretraining). Their features are fused by a 2D Aggregator, then processed by Depth/Camera Modules to extract geometric embeddings, and finally decoded by a 3D Interpreter into 3D bounding box predictions.

### Key Designs

1. **2D Aggregator**:

    - **Function**: Fuses heterogeneous features from SAM and DINO while resolving representational conflicts.
    - **Mechanism**: Employs a hierarchical cross-attention mechanism with 4 cascaded alignment units. Each unit adaptively fuses features from both models via learnable gating weights $\alpha_i$ (initialized to 0.5): $\mathbf{F}_{\text{fused}}^{i}=\alpha_{i}\cdot\mathbf{F}_{s}^{i}+(1-\alpha_{i})\cdot\mathbf{F}_{d}^{i}$, followed by cross-attention using the fused features as K/V and the query features as Q.
    - **Design Motivation**: SAM excels at fine-grained spatial information while DINO excels at depth-geometric information. They are complementary but reside in different feature spaces, necessitating adaptive alignment and fusion.

2. **3D Interpreter and Zero-Embedding Mapping (ZEM)**:

    - **Function**: Progressively injects 3D geometric information into 2D features while ensuring stable 2D-to-3D knowledge transfer.
    - **Mechanism**: Consists of a Two-Way Transformer (inheriting SAM's decoder structure) and a Geometric Transformer. ZEM uses zero-initialized $1\times1$ convolutional layers to progressively inject geometric embeddings $\mathbf{G}$ into features: $\mathbf{G}'=\text{GeoTrans}(\mathbf{Q}, \text{ZEM}(\mathbf{G})+\mathbf{F}_s, \text{ZEM}(\mathbf{G})+\mathbf{F}_s)$
    - **Design Motivation**: Directly injecting 3D geometric features disrupts pretrained 2D features and causes catastrophic forgetting. ZEM's zero initialization ensures that the original 2D features remain unchanged at the start of training, allowing gradual learning of geometric injection and stabilizing cross-dataset training.

3. **Multi-modal Prompt Interaction**:

    - **Function**: Supports four prompt modalities: box, point, text, and intrinsic.
    - **Mechanism**: Box/point prompts follow SAM's positional encoding scheme; text prompts are converted via Grounding DINO to obtain 2D boxes; intrinsic prompts provide optional camera intrinsics, which the model predicts independently when not supplied.
    - **Design Motivation**: Inspired by SAM's promptable design philosophy to enable flexible user interaction and open-world detection.

### Loss & Training

The total loss is the sum of three components:
- **Depth loss** $\mathcal{L}_{\text{depth}}$: SILog loss supervising depth prediction.
- **Camera intrinsic loss** $\mathcal{L}_{\text{cam}}$: SILog loss based on dense camera ray representation.
- **Detection loss** $\mathcal{L}_{\text{det}}$: Smooth L1 loss (3D box parameter regression) + Chamfer loss (rotation matrix) + MSE loss (3D IoU score).

Training details: The SAM encoder is frozen; ViT-L DINOv2 and ViT-H SAM are used for initialization. Training is conducted on 8×8 A100 GPUs with batch size 64 for 80 epochs (~2 weeks). The DA3D dataset aggregates 16 datasets, 0.4M frames, and 20 camera configurations.

## Key Experimental Results

### Main Results

**Zero-shot novel category detection (GT prompt):**

| Dataset | Metric | DetAny3D | OVMono3D | Gain |
|---------|--------|----------|----------|------|
| KITTI | AP3D | 28.96 | 8.44 | +20.52 (3.4×) |
| SUNRGBD | AP3D | 39.09 | 17.16 | +21.93 (2.3×) |
| ARKitScenes | AP3D | 57.72 | 14.12 | +43.60 (4.1×) |

**Zero-shot novel camera configuration detection (Grounding DINO prompt, target-aware metric):**

| Dataset | Metric | DetAny3D | OVMono3D | Gain |
|---------|--------|----------|----------|------|
| Cityscapes3D | AP3D | 15.71 | 10.98 | +4.73 |
| Waymo | AP3D | 15.95 | 10.27 | +5.68 |
| 3RScan | AP3D | 9.58 | 8.48 | +1.10 |

**In-domain Omni3D detection (GT prompt): AP3D = 34.38 vs. OVMono3D 25.32 (+9.06)**

### Ablation Study

| Configuration | AP3D | Notes |
|---------------|------|-------|
| SAM baseline (no additional components) | 5.81 | SAM + 3D head only |
| + Depth & Camera modules | 10.10 | +4.29 from depth and camera modules |
| + Incorporating DINO | 20.20 | Large contribution from DINO geometric priors (+10.10) |
| + 2D Aggregator | 23.21 | Better than naive addition fusion (+3.01) |
| + ZEM | 25.80 | Stable transfer yields +2.59 |

### Key Findings
- Incorporating depth-pretrained DINO is the largest source of gain, demonstrating the critical role of geometric priors in monocular 3D detection.
- The ZEM mechanism significantly improves training stability across datasets, mitigating conflicts between different data distributions.
- 2D prompt quality is a performance bottleneck—AP3D is substantially higher with GT 2D boxes than with Cube R-CNN detections.
- DetAny3D's 3D detection results can be used in downstream tasks such as 3D box-guided video generation with Sora.

## Highlights & Insights
- **Core innovation**: The first genuinely promptable 3D detection foundation model with zero-shot generalization capability far exceeding existing methods.
- **Elegant knowledge transfer**: ZEM's zero-initialization strategy is simple yet highly effective, preventing catastrophic forgetting of 2D pretrained weights.
- **Engineering value**: Aggregating 16 datasets to construct the unified DA3D benchmark establishes a standard for systematic evaluation of 3D detection foundation models.
- Demonstrates an effective knowledge transfer pathway from 2D foundation models to 3D tasks, offering insights for other 3D tasks.

## Limitations & Future Work
- Performance depends on 2D prompt quality; the capability of current 2D detectors (e.g., Cube R-CNN) constitutes a bottleneck.
- The SAM encoder is frozen during training, potentially limiting adaptation to 3D tasks.
- The pinhole camera ray model assumption may be insufficiently accurate in geometrically complex scenes.
- The possibility of multi-modal inputs such as point clouds or depth sensors remains unexplored.
- Performance on datasets with severe label ambiguity, such as 3RScan, remains limited.

## Related Work & Insights
- **Omni3D / Cube R-CNN**: Pioneers in unified multi-dataset training, but limited to closed-set detection.
- **OVMono3D**: An attempt at open-vocabulary 3D detection, but does not fully exploit 2D foundation model priors.
- **SAM / DINOv2**: 2D foundation models whose priors are successfully transferred to 3D tasks in this work.
- **UniDepth**: Provides the framework for depth-pretrained DINO and joint camera-depth estimation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First promptable 3D detection foundation model; ZEM mechanism is novel and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 16 datasets with comprehensive zero-shot, in-domain, and ablation evaluations; results are significant.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, detailed method descriptions, and rich figures and tables.
- Value: ⭐⭐⭐⭐⭐ Opens a foundation model direction for 3D detection with high practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SAM4D: Segment Anything in Camera and LiDAR Streams](sam4d_segment_anything_in_camera_and_lidar_streams.md)
- [\[NeurIPS 2025\] LabelAny3D: Label Any Object 3D in the Wild](../../NeurIPS2025/autonomous_driving/labelany3d_label_any_object_3d_in_the_wild.md)
- [\[ICCV 2025\] 3DRealCar: An In-the-wild RGB-D Car Dataset with 360-degree Views](3drealcar_an_in-the-wild_rgb-d_car_dataset_with_360-degree_views.md)
- [\[ICCV 2025\] Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](adaptive_dual_uncertainty_optimization_boosting_monocular_3d_object_detection_un.md)
- [\[ICCV 2025\] TrackAny3D: Transferring Pretrained 3D Models for Category-unified 3D Point Cloud Tracking](trackany3d_transferring_pretrained_3d_models_for_category-unified_3d_point_cloud.md)

</div>

<!-- RELATED:END -->
