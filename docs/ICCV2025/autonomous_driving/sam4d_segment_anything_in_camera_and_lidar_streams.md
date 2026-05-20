---
title: >-
  [Paper Note] SAM4D: Segment Anything in Camera and LiDAR Streams
description: >-
  [ICCV 2025][Autonomous Driving][Multimodal Segmentation] This paper presents SAM4D, the first promptable multimodal segmentation foundation model for camera and LiDAR streams. It introduces Unified Multimodal Positional…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "Multimodal Segmentation"
  - "Foundation Model"
  - "Camera-LiDAR Fusion"
  - "Temporal Segmentation"
  - "SAM"
date: 2026-05-08
content_hash: 75f8927648df110f
---

# SAM4D: Segment Anything in Camera and LiDAR Streams

**Conference**: ICCV 2025
**arXiv**: [2506.21547](https://arxiv.org/abs/2506.21547)  
**Code**: [SAM4D-Project.github.io](https://SAM4D-Project.github.io)  
**Area**: Autonomous Driving
**Keywords**: Multimodal Segmentation, Foundation Model, Camera-LiDAR Fusion, Temporal Segmentation, SAM

## TL;DR

This paper presents SAM4D, the first promptable multimodal segmentation foundation model for camera and LiDAR streams. It introduces Unified Multimodal Positional Encoding (UMPE) to enable cross-modal prompting and interaction, Motion-aware Cross-Modal Attention (MCMA) for temporal consistency, and constructs the Waymo-4DSeg dataset containing 300K+ masklets, demonstrating strong capabilities in cross-modal segmentation and data annotation.

## Background & Motivation

### Problem Definition

In autonomous driving, cameras and LiDAR complement each other's limitations (e.g., low-light conditions, depth accuracy), making robust multimodal perception essential. Existing segmentation models are restricted to a single modality (image or point cloud) and typically operate on individual frames, failing to exploit cross-modal spatial consistency and temporal continuity.

### Limitations of Prior Work

**SAM/SAM2**: Designed solely for image/video segmentation; no support for LiDAR or other sensor modalities.

**LiDAR segmentation methods** (SAL, PointSAM): Build SAM-like models directly on point clouds but are **limited to a single modality**.

**Projection-based methods** (CLIP2Scene, etc.): Project 2D segmentation results into 3D, but are constrained by sensor viewpoint discrepancies and synchronization issues.

**Multimodal perception methods** (BEVFusion, etc.): Produce 3D predictions only, lacking cross-modal interaction and unified 2D-3D segmentation.

**Frame-level LiDAR segmentation**: Does not leverage LiDAR's precise depth for temporal feature association.

### Core Problem

A **unified multimodal temporal segmentation framework** is needed that can:
- Simultaneously generate segmentation masks in both camera and LiDAR modalities
- Support cross-modal prompting (e.g., using image clicks to guide LiDAR segmentation)
- Maintain temporal consistency over long sequences
- Substantially reduce annotation costs for multimodal data

## Method

### Overall Architecture

SAM4D extends SAM2 to the multimodal domain, consisting of four main components:
1. **Multimodal Feature Extraction**: Image encoder (Hiera) + LiDAR encoder (MinkUNet)
2. **Unified Multimodal Positional Encoding (UMPE)**: Aligns image and LiDAR features in a shared 3D space
3. **Motion-aware Cross-Modal Attention (MCMA)**: Cross-modal fusion with ego-motion compensated temporal attention
4. **Mask Decoder**: Simultaneously outputs 2D and 3D segmentation masks

### Key Designs

#### 1. **Unified Multimodal Positional Encoding (UMPE)**

- **Function**: Aligns the positional representations of image patch tokens and LiDAR voxel tokens in a shared 3D space.
- **Mechanism**:

  UMPE consists of two complementary components: (i) modality-specific positional priors; (ii) shared 3D spatial representations.

  **Image Positional Encoding**:
  - 2D sinusoidal positional encoding preserves image plane structure: $\mathcal{P}_{\text{img\_sin}} = \text{SinPE2D}(u, v)$
  - Pixels are lifted to 3D space via depth estimation (analogous to Lift-Splat-Shoot):
  $$\mathbf{x}_{\text{img}} = T_c^l K^{-1} [u \cdot D(u,v), v \cdot D(u,v), D(u,v), 1]^T$$
  - An MLP encodes the 3D position: $\mathcal{P}_{\text{img\_mlp}} = \text{MLP}(\mathbf{x}_{\text{img}})$

  **LiDAR Positional Encoding**:
  - 3D sinusoidal positional encoding: $\mathcal{P}_{\text{LiDAR\_sin}} = \text{SinPE3D}(x, y, z)$
  - Shared MLP encodes the 3D position: $\mathcal{P}_{\text{LiDAR\_mlp}} = \text{MLP}(\mathbf{x}_{\text{LiDAR}})$

  The final positional encodings $\mathcal{P}_{\text{img}}$ and $\mathcal{P}_{\text{LiDAR}}$ each consist of both components, ensuring cross-modal positional alignment.

- **Design Motivation**:
    - The two-stage encoding captures both modality-specific characteristics (2D structure for images, 3D structure for LiDAR) and cross-modal alignment (shared 3D MLP).
    - The shared 3D MLP renders image and LiDAR features comparable in the same space, enabling cross-modal prompting.
    - Sparse prompts (points, bounding boxes) use the same two-stage encoding, ensuring prompt-feature spatial consistency.

#### 2. **Motion-aware Cross-Modal Attention (MCMA)**

- **Function**: Integrates cross-modal feature fusion with ego-motion-compensated temporal memory attention.
- **Mechanism**:

  Three-stage attention pipeline:

  **Step 1 — Intra-modal Self-Attention**:
  $$\mathcal{F}'_{\text{img}} = \text{SelfAttn}(\mathcal{F}_{\text{img}} + \mathcal{P}_{\text{img}})$$
  $$\mathcal{F}'_{\text{LiDAR}} = \text{SelfAttn}(\mathcal{F}_{\text{LiDAR}} + \mathcal{P}_{\text{LiDAR}})$$

  **Step 2 — Cross-modal Cross-Attention**:
  $$\mathcal{F}''_{\text{img}} = \text{CrossAttn}(\mathcal{F}'_{\text{img}}, \mathcal{F}'_{\text{LiDAR}} + \mathcal{P}_{\text{LiDAR}})$$
  $$\mathcal{F}''_{\text{LiDAR}} = \text{CrossAttn}(\mathcal{F}'_{\text{LiDAR}}, \mathcal{F}'_{\text{img}} + \mathcal{P}_{\text{img}})$$

  **Step 3 — Ego-motion-compensated Temporal Memory Attention**:
  Historical frame features and positions are transformed via ego-motion to align with the current frame coordinate system:
  $$\mathcal{M}_{\text{img}}^{t \leftarrow t'} = \mathcal{M}_{\text{img}}^{t'} + \Phi_{\text{img}}(T_{t \leftarrow t'}(\mathbf{x}_{\text{img}}))$$
  $$\mathcal{M}_{\text{LiDAR}}^{t \leftarrow t'} = \mathcal{M}_{\text{LiDAR}}^{t'} + \Phi_{\text{LiDAR}}(T_{t \leftarrow t'}(\mathbf{x}_{\text{LiDAR}}))$$

  Current frame features are then fused with the aligned memory features via cross-attention:
  $$\mathcal{F}_{\text{img}}^{\text{final}} = \text{CrossAttn}(\mathcal{F}''_{\text{img}}, (\mathcal{M}_{\text{img}}^{t \leftarrow t'}, \mathcal{O}_{\text{img}}^{t'}))$$

  where $T_{t \leftarrow t'} \in SE(3)$ is derived from vehicle odometry.

- **Design Motivation**:
    - Unlike SAM2, which only accounts for short-range motion, MCMA explicitly incorporates ego-motion compensation to handle large-scale scene changes in autonomous driving.
    - The ego-motion transformation spatially aligns historical frame features with the current frame, preventing feature misalignment caused by vehicle movement.
    - The memory bank uses a FIFO queue storing $N$ non-prompt frames and $M$ prompt frames separately, ensuring critical frames are retained.

#### 3. **Multimodal Automatic Data Engine**

- **Function**: Automatically generates high-quality camera-LiDAR aligned pseudo-labels to construct the Waymo-4DSeg dataset.
- **Mechanism**:

  Three-step pipeline:
  1. **VFM-driven video masklet generation**: Grounding-DINO detects objects in keyframes + SAM performs segmentation → SAM2 propagates to intermediate frames.
  2. **4D voxel reconstruction**: LiDAR frames and 3D bounding boxes are used to build a 4D voxel representation, establishing a pixel-to-voxel mapping table.
  3. **Cross-modal label fusion**: Video masklets are projected onto voxels via the mapping table; DBSCAN filters noise; overlapping masklets across views are merged; labels are finally transferred to LiDAR frames.

  Result: cross-modal IoU of 0.56.

- **Design Motivation**:
    - No existing dataset simultaneously supports 2D and 3D segmentation while guaranteeing temporal instance consistency.
    - The strong zero-shot capabilities of VFMs (SAM, Grounding-DINO) enable automatic generation of high-quality labels.
    - 4D reconstruction serves as an intermediate bridge connecting 2D image labels and 3D point cloud labels.

### Loss & Training

- Identical loss functions are applied to image and LiDAR predictions to enforce cross-modal consistency.
- Training simulates an interactive prompting process (analogous to SAM2's strategy).
- The model is trained for 36 epochs on 16 A100 GPUs, processing at most 6 objects per iteration.
- The image encoder uses Hiera-S with SA-V pretraining; the LiDAR encoder uses Mink-34.
- Image resolution is 768×768; LiDAR voxel size is 0.15m.

## Key Experimental Results

### Main Results

Cross-modal frame-level segmentation (Image-Prioritized Prompting):

| Prompt Type | Image mIoU↑ | LiDAR mIoU↑ |
|-------------|------------|-------------|
| 1-click | 68.0% | 42.3% |
| 3-click | 73.6% | **53.1%** |
| Bounding Box | **74.7%** | 47.0% |

Semi-supervised streaming segmentation (first-frame prompt → sequence propagation):

| Prompt Type | Image mIoU↑ | J&F↑ | NMP↓ | LiDAR mIoU↑ | NMP↓ |
|-------------|------------|------|------|-------------|------|
| 1-click | 61.4% | 72.2 | 398 | 50.1% | 784 |
| 3-click | 65.6% | 76.3 | 327 | 52.8% | 711 |
| 5-click | 67.1% | 77.7 | 315 | 52.6% | 702 |
| GT mask | **69.8%** | **80.1** | **280** | **55.7%** | **582** |

Cross-dataset generalization (nuScenes, semi-supervised streaming segmentation):

| Setting | Image mIoU↑ | J&F↑ | LiDAR mIoU↑ |
|---------|------------|------|-------------|
| Zero-shot | 58.4% | 65.8 | 25.9% |
| Fine-tuned | **67.5%** | **75.4** | **44.8%** |

### Ablation Study

Input modality ablation:

| Configuration | Image mIoU↑ | J&F↑ | NMP↓ | LiDAR mIoU↑ | NMP↓ |
|---------------|------------|------|------|-------------|------|
| SAM2 + Projection | 68.2% | 79.7 | 383 | 32.0% | - |
| SAM4D-Camera Only | 68.6% | 80.4 | 301 | - | - |
| SAM4D-LiDAR Only | - | - | - | 47.0% | 799 |
| **SAM4D (Full)** | **69.8%** | 80.1 | **280** | **55.7%** | **582** |

Ego-motion compensation ablation:

| Configuration | Image mIoU↑ | LiDAR mIoU↑ | LiDAR NMP↓ |
|---------------|------------|-------------|-----------|
| Without Ego-motion | 69.7% | 52.2% | 746 |
| **With Ego-motion** | **69.8%** | **55.7%** | **582** |

### Key Findings

1. **Cross-modal prompting is effective**: Prompting on images yields 53.1% LiDAR mIoU, demonstrating that UMPE successfully achieves cross-modal alignment.
2. **Multimodal fusion substantially improves LiDAR**: LiDAR mIoU improves from 47.0% (single-modal) to 55.7% (multimodal) (+8.7%), indicating that image semantic information provides significant benefit to point cloud segmentation.
3. **SAM2 + Projection achieves only 32.0% LiDAR mIoU**: Confirming that simple projection cannot address cross-modal segmentation; deep fusion is required.
4. **Ego-motion compensation primarily benefits LiDAR**: LiDAR NMP decreases from 746 to 582 (−22%), suggesting that LiDAR's sparsity makes it more sensitive to spatial alignment.
5. **Reasonable zero-shot performance on nuScenes**: Image mIoU of 58.4% demonstrates a degree of generalization capability.
6. **Data engine efficiency**: An average of 300 masklets are generated per clip with a cross-modal IoU of 0.56, far exceeding the throughput of manual annotation.

## Highlights & Insights

1. **First unified promptable segmentation model for camera + LiDAR**: Fills the gap in multimodal segmentation foundation models.
2. **Innovation in cross-modal prompting**: Prompts from one modality guide segmentation in the other, substantially improving annotation efficiency.
3. **Elegant design of UMPE**: Lifts image features into 3D space via depth estimation, enabling both modalities to interact in a shared space.
4. **Engineering value of the data engine**: Combining VFMs, 4D reconstruction, and cross-modal fusion, it constructs large-scale high-quality pseudo-labels.
5. **Waymo-4DSeg dataset**: 300K+ masklets covering vehicles, pedestrians, buildings, and other categories, with object sizes ranging from 10 voxels to 200K voxels.

## Limitations & Future Work

1. **Dependence on depth estimation quality**: The image-to-3D lifting in UMPE relies on depth estimation; depth errors degrade cross-modal alignment accuracy.
2. **Remaining gap in LiDAR performance**: LiDAR mIoU of 55.7% is substantially lower than image mIoU of 69.8%, indicating that stronger feature extraction for point clouds is still needed.
3. **Cross-modal IoU of only 0.56**: Label quality from the data engine still has room for improvement.
4. **Training limited to Waymo**: Although nuScenes generalization is evaluated, training data diversity is limited.
5. **Downstream task impact not evaluated**: The transferability of SAM4D annotations to tasks such as 3D detection or trajectory prediction has not been verified.

## Related Work & Insights

- **vs. SAM2**: SAM2 supports video segmentation only; SAM4D extends to camera + LiDAR multimodal streams.
- **vs. PointSAM/SAL**: These methods build promptable segmentation solely on point clouds; SAM4D unifies 2D and 3D.
- **vs. BEVFusion et al.**: Multimodal perception methods produce 3D predictions; SAM4D simultaneously outputs 2D and 3D segmentation masks.
- The **Lift-Splat-Shoot** paradigm is adopted within UMPE to lift 2D features into 3D space.
- The data engine design (VFM → 4D reconstruction → cross-modal fusion) offers a new paradigm for autonomous driving data bootstrapping.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The first promptable segmentation foundation model unifying camera and LiDAR streams; both the task formulation and model design represent significant advances.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple evaluation settings with thorough ablations, though LiDAR segmentation baseline comparisons are limited.
- **Writing Quality**: ⭐⭐⭐⭐ — The three contributions (task, model, data) are presented with a clear and well-organized structure.
- **Value**: ⭐⭐⭐⭐⭐ — Substantial potential impact on multimodal data annotation efficiency; represents an important extension of the SAM family to autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Detect Anything 3D in the Wild](detect_anything_3d_in_the_wild.md)
- [\[ICLR 2026\] SEAL: Segment Any Events with Language](../../ICLR2026/autonomous_driving/segment_any_events_with_language.md)
- [\[ICCV 2025\] MGSfM: Multi-Camera Geometry Driven Global Structure-from-Motion](mgsfm_multi-camera_geometry_driven_global_structure-from-motion.md)
- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[ICCV 2025\] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)

</div>

<!-- RELATED:END -->
