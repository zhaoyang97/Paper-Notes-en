---
title: >-
  [Paper Note] Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors
description: >-
  [CVPR 2025][3D Vision][3D Reconstruction] This paper proposes Pow3R, a general-purpose 3D vision regression model built upon DUSt3R, which can flexibly incorporate any combination of auxiliary information such as camera intrinsics, relative pose, and sparse/dense depth. It achieves SOTA on multiple 3D vision tasks and unlocks new capabilities like native-resolution inference.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Reconstruction"
  - "DUSt3R"
  - "Multimodal Priors"
  - "Depth Completion"
  - "Pose Estimation"
date: 2026-05-08
content_hash: afbda0d6b3aa00b5
---

# Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors

**Conference**: CVPR 2025  
**arXiv**: [2503.17316](https://arxiv.org/abs/2503.17316)  
**Code**: [Project Page](https://europe.naverlabs.com/pow3r)  
**Area**: 3D Vision  
**Keywords**: 3D Reconstruction, DUSt3R, Multimodal Priors, Depth Completion, Pose Estimation

## TL;DR

This paper proposes Pow3R, a general-purpose 3D vision regression model built upon DUSt3R, which can flexibly incorporate any combination of auxiliary information such as camera intrinsics, relative pose, and sparse/dense depth. It achieves SOTA on multiple 3D vision tasks and unlocks new capabilities like native-resolution inference.

## Background & Motivation

DUSt3R is a milestone in 3D vision regression models, capable of regressing 3D pointmaps from uncalibrated, unlocalized image pairs. However, it suffers from a fundamental limitation—it only accepts RGB image inputs, failing to utilize additional information frequently available in practical applications:

- **Wasted camera calibration information**: Many application scenarios provide calibrated camera intrinsics, which DUSt3R cannot exploit.
- **Unused depth sensor data**: Sparse or dense depth provided by RGB-D cameras or LiDAR cannot be integrated into the model.
- **Unutilized relative pose constraints**: Pose estimates provided by IMUs or other sensors cannot be leveraged as additional constraints.
- **Limited resolution**: DUSt3R is trained on a fixed resolution and cannot generalize to higher resolutions.
- **Requirement for bidirectional inference**: Extracting depth for both images requires two forward passes (one forward and one backward).

## Method

### Overall Architecture

Based on the ViT encoder-decoder architecture of DUSt3R, Pow3R introduces a lightweight conditioning injection module to ingest auxiliary information. The encoder receives the images, intrinsics, and depth, while the decoder receives the relative pose. During training, subsets of auxiliary information are randomly sampled, allowing the model to adapt to various conditioning scenarios. Additionally, it predicts three pointmaps $X^{1,1}$, $X^{2,1}$, $X^{2,2}$ (compared to DUSt3R's two), enabling the extraction of all information in a single inference pass.

### Key Designs 1: Flexible Multimodal Auxiliary Information Injection — Lightweight Conditioning

**Function**: Inject camera intrinsics, depth maps, and relative poses into the ViT without altering the core architecture.

**Mechanism**: Tailored injection methods are designed for different modalities:
- **Intrinsics** $K$: The camera rays $K^{-1}[i,j,1]$ are computed, patched, and injected into the encoder via a dedicated MLP.
- **Depth** $D$: Normalized depth is concatenated with a validity mask $M$ to form $[D', M] \in \mathbb{R}^{W \times H \times 2}$, then patched and injected into the encoder.
- **Relative Pose** $P_{12}$: Encoded through an embedding layer and an MLP, then added to the decoder's CLS token.

An "inject-1" strategy is adopted: the MLP for each modality injects features only into the first Transformer block, merging via token-wise addition.

**Design Motivation**: A subset of auxiliary information is randomly dropped during training ("dropout" conditioning), allowing a single model to adapt to all configurations from zero priors to full priors. Experiments show that "inject-1" is as effective as deep injection but far more efficient.

### Key Designs 2: Tri-Pointmap Prediction — Single-Pass Inference for Complete Information

**Function**: Predict an additional $X^{2,2}$ (the pointmap of the second image in its own coordinate frame) to enable single-pass extraction of depth, focal lengths, and relative poses for both images.

**Mechanism**: The decoder branch Head$^2$ concurrently outputs $X^{2,1}$, $X^{2,2}$, and their corresponding confidence scores. The relative pose is directly obtained by Procrustes alignment of $X^{2,2}$ and $X^{2,1}$:

$$R^*, t^* = \arg\min_{\sigma,R,t} \sum_{i,j} \sqrt{C^{2,2}_{i,j} C^{2,1}_{i,j}} \|\sigma(RX^{2,2}_{i,j} + t) - X^{2,1}_{i,j}\|^2$$

**Design Motivation**: DUSt3R requires two forward passes $(I_1, I_2)$ and $(I_2, I_1)$ to obtain the depth maps and poses of both images. The tri-pointmap design doubles the inference efficiency. Although Procrustes alignment can be sensitive to noise, it achieves comparable accuracy to RANSAC+PnP in this setup while being an order of magnitude faster.

### Key Designs 3: Native-Resolution Inference — New Capabilities from Intrinsics Conditioning

**Function**: Leverage intrinsics to handle arbitrary crops, enabling sliding-window high-resolution inference.

**Mechanism**: Because intrinsics are encoded as camera rays, they inherently contain crop position information (i.e., focal lengths and the principal point). Consequently, high-resolution images can be split into multiple cropped patches, and each patch is processed independently alongside its corresponding crop-area intrinsics. Finally, they are assembled using median-scale alignment over overlapping regions and confidence-weighted blending.

**Design Motivation**: DUSt3R is trained on a fixed resolution and assumes centered principal points, limiting its ability to handle non-centered crops or high-resolution images. Intrinsics conditioning naturally lifts this limitation without requiring supplementary architectural changes.

### Loss & Training

$$\mathcal{L} = \mathcal{L}^{\text{conf}}(1,1) + \mathcal{L}^{\text{conf}}(2,1) + \beta \mathcal{L}^{\text{conf}}(2,2)$$

where the confidence-weighted regression loss is defined as $\mathcal{L}^{\text{conf}}(n,m) = \sum C^{n,m}_{i,j} \mathcal{L}^{\text{regr}}_{i,j}(n,m) - \alpha \log C^{n,m}_{i,j}$, and a scale-invariant 3D regression loss is used.

## Key Experimental Results

### Multi-view Depth Prediction (ScanNet++)

| Method | AbsRel↓ | δ<1.25↑ |
|------|---------|---------|
| DUSt3R | 0.068 | 0.967 |
| MASt3R | 0.061 | 0.974 |
| **Pow3R (No Prior)** | 0.063 | 0.972 |
| **Pow3R (w/ K + Depth)** | **0.041** | **0.988** |

### Relative Pose Estimation (Map-free benchmark)

| Method | AUC@5°↑ | AUC@10°↑ | Inference Time |
|------|---------|----------|---------|
| DUSt3R + PnP | 52.3 | 64.1 | ~100ms |
| MASt3R + PnP | 60.1 | 71.5 | ~100ms |
| **Pow3R + Procrustes** | **62.8** | **73.2** | **~10ms** |

### Depth Completion (NYUv2)

| Method | RMSE↓ (500 pts) | RMSE↓ (200 pts) |
|------|-------------|-------------|
| CompletionFormer | 0.089 | 0.105 |
| **Pow3R** | **0.082** | **0.094** |

### Key Findings

- Without priors, Pow3R performs on par with DUSt3R, and shows significant improvement when priors are available (a **40%** AbsRel improvement under the intrinsics+depth configuration).
- Procrustes pose estimation achieves comparable accuracy to RANSAC+PnP while being **approximately 10 times faster**.
- Native-resolution inference yields sharper, more detailed 3D reconstructions without requiring retraining.
- The model outperforms specialized methods on the depth completion task, demonstrating its versatility.
- The training strategy of randomly dropping auxiliary information is crucial for adapting to different configurations.

## Highlights & Insights

1. **Flexible auxiliary information injection is a practical innovation**: The availability of various prior information varies significantly in physical applications; thus, a single model adapting to all scenarios is highly desirable.
2. **The tri-pointmap design kills two birds with one stone**: It doubles inference efficiency while enabling Procrustes pose estimation.
3. **Intrinsics conditioning unlocking high-resolution inference** is an unexpected but highly valuable byproduct.
4. **Compatibility with the DUSt3R ecosystem**: Subsequent works such as MASt3R and MonST3R can directly benefit from these findings.

## Limitations & Future Work

- High-resolution sliding-window inference still requires manual configuration of cropping strategies and overlap regions.
- Procrustes alignment may lack robustness in heavily occluded scenarios.
- The model currently only handles image pairs, requiring global alignment post-processing for multi-view scaling.
- Noise tolerance of auxiliary information warrants more in-depth exploration.
- Future work can explore actively selecting the most valuable auxiliary information for queries.

## Related Work & Insights

- **DUSt3R / MASt3R**: Baseline 3D regression frameworks; Pow3R directly expands on them.
- **UniDepth**: A pioneer in optional intrinsics conditioning for monocular depth estimation.
- **CompletionFormer**: A CNN+ViT depth completion method.
- **RayDiffusion**: Pose estimation based on ray diffusion, serving as inspiration for intrinsic ray encoding.

## Rating

⭐⭐⭐⭐⭐ — High engineering and academic value. Represents a correct and crucial enhancement direction built upon the powerful DUSt3R framework. Flexible prior injection allows a single model to adapt to a wide range of scenarios; the tri-pointmap design doubles efficiency; and native-resolution inference manifests as an exciting new capability. Experiments cover a variety of downstream tasks, achieving SOTA performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Scene Coordinate Reconstruction Priors](../../ICCV2025/3d_vision/scene_coordinate_reconstruction_priors.md)
- [\[CVPR 2025\] Empowering Large Language Models with 3D Situation Awareness](empowering_large_language_models_with_3d_situation_awareness.md)
- [\[CVPR 2025\] MUSt3R: Multi-view Network for Stereo 3D Reconstruction](must3r_multi-view_network_for_stereo_3d_reconstruction.md)
- [\[CVPR 2025\] MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors](mast3r-slam_real-time_dense_slam_with_3d_reconstruction_priors.md)
- [\[CVPR 2025\] Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass](fast3r_towards_3d_reconstruction_of_1000_images_in_one_forward_pass.md)

</div>

<!-- RELATED:END -->
