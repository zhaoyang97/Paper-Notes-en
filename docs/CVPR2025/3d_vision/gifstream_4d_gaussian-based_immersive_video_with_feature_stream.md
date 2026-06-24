---
title: >-
  [Paper Note] GIFStream: 4D Gaussian-based Immersive Video with Feature Stream
description: >-
  [CVPR 2025][3D Vision][Immersive Video] Proposes GIFStream, a 4D Gaussian representation based on a canonical space + deformation field. By attaching a time-dependent feature stream to each anchor point, it enhances the capability of modeling complex motions. Meanwhile, it leverages a time-aligned structure and end-to-end compression to achieve high-quality 1080p immersive video at 30 Mbps.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Immersive Video"
  - "4D Gaussian Splatting"
  - "Dynamic Scene Compression"
  - "Feature Stream"
  - "End-to-end Compression"
date: 2026-05-08
content_hash: cac4d805cd431815
---

# GIFStream: 4D Gaussian-based Immersive Video with Feature Stream

**Conference**: CVPR 2025  
**arXiv**: [2505.07539](https://arxiv.org/abs/2505.07539)  
**Code**: [https://xdimlab.github.io/GIFStream](https://xdimlab.github.io/GIFStream)  
**Area**: 3D Vision  
**Keywords**: Immersive Video, 4D Gaussian Splatting, Dynamic Scene Compression, Feature Stream, End-to-end Compression

## TL;DR

Proposes GIFStream, a 4D Gaussian representation based on a canonical space + deformation field. By attaching a time-dependent feature stream to each anchor point, it enhances the capability of modeling complex motions. Meanwhile, it leverages a time-aligned structure and end-to-end compression to achieve high-quality 1080p immersive video at 30 Mbps.

## Background & Motivation

**Background**: Immersive video allows users to explore dynamic scenes with 6 degrees of freedom (6-DoF). 3D Gaussian Splatting (3DGS) and its 4D extensions have attracted significant attention due to high-quality reconstruction and real-time rendering. Existing methods fall into two paradigms: (1) Deformation-based methods (canonical space + deformation field), which require small storage but struggle to capture rapid motion; (2) 4D Gaussian methods (each primitive covers a local space-time region), which achieve high quality but require massive storage and lack temporal correspondence.

**Limitations of Prior Work**: The deformation fields in deformation-based methods lack sufficient capacity to capture rapid motion details. On the other hand, 4D Gaussian methods distribute primitives discretely in 4D space, missing temporal correspondence among primitives, which hinders effective temporal redundancy elimination and leads to low compression efficiency.

**Key Challenge**: The trade-off between rendering quality and storage efficiency—methods capable of capturing rapid motion require large storage and are hard to compress, whereas methods with small storage cannot model complex dynamics.

**Goal**: Design a 4D representation that can both capture highly dynamic content and be efficiently compressed, achieving an optimal balance between quality and storage.

**Key Insight**: Introduce adaptive, sparse, time-varying feature streams on top of deformation-based methods. These feature streams enhance dynamic modeling capabilities. Graced by the time-aligned structure based on the canonical space, they can be efficiently compressed along the temporal dimension using video codecs.

**Core Idea**: Add time-varying feature streams to each anchor in the deformation-based 3D Gaussian representation (with automatic pruning for static regions). Combined with an end-to-end compression network, this unifies high-quality dynamic scene representation and efficient coding.

## Method

### Overall Architecture

Given multi-view video inputs, GIFStream projects and maintains a set of anchor points in the canonical space. Each anchor contains time-invariant features $\mathbf{f}$ and a set of time-varying feature streams $\{\mathbf{f}_t\}$. At each timestamp $t$, these two types of features are decoded into Gaussian attributes (opacity, scale, rotation, color) and motion (rotation + translation) via MLPs to generate $K$ Gaussian primitives for rendering. After training, the parameters are reorganized into two video sequences (time-invariant + time-dependent) and compressed via end-to-end learned entropy coding or traditional video codecs.

### Key Designs

1. **Motion-Adaptive Feature Stream**:

    - **Function**: Provides time-varying information for each anchor, enhancing the deformation field's capability to model rapid motions.
    - **Mechanism**: Each anchor possesses a time-invariant feature $\mathbf{f} \in \mathbb{R}^C$ and a time-varying feature $\mathbf{f}_t \in \mathbb{R}^P$. The time-varying feature is modulated by a learnable scaling parameter $M_{de}$: $\hat{\mathbf{f}}_t = M_{de} \cdot \mathbf{f}_t$. Regularization encourages $M_{de}$ to approach zero, allowing feature streams in static regions to be automatically pruned. Experiments show that around 30% of anchors retain feature streams in complex scenes, whereas only 0.3% do so in simple scenes.
    - **Design Motivation**: Directly increasing the capacity of the deformation field would dramatically increase storage. The feature stream equips each anchor with extra information at required timesteps while automatically decaying to zero when redundant, achieving an adaptive balance between capacity and storage.

2. **Motion Prediction via KNN Neighborhood Aggregation**:

    - **Function**: Predicts the SE(3) motion of anchors by utilizing a local spatial smoothness prior of motion.
    - **Mechanism**: Before predicting motion, neighbor anchor features are aggregated via KNN: $\tilde{\mathbf{f}}_t = (1-M_{knn})\sum_{k \in \mathbb{N}}\hat{\mathbf{f}}_{k,t} + M_{knn}\hat{\mathbf{f}}_t$. A learnable parameter $M_{knn}$ controls the blend between smooth and fine-detail motions. Motion is represented as rotation $\mathbf{R}_t$ and translation $\mathbf{T}_t$ in the anchor's local coordinate system, controlled by a dynamic scaling factor $M_{dy}$—where $M_{dy}$ of static anchors is regularized to zero.
    - **Design Motivation**: In most scenes, motion exhibits local smoothness. KNN aggregation leverages this prior to reduce the complexity and parameter requirements of motion prediction. Meanwhile, $M_{knn}$ allows individual information to be preserved when non-smooth motion is necessary.

3. **Sorting + End-to-End Video Compression**:

    - **Function**: Efficiently compresses the 3D representation into low-bitrate bitstreams.
    - **Mechanism**: Anchors are mapped to a 2D grid after sorting based on canonical positions and feature PCA. Parameters are stacked into two videos: $\mathbf{V}_{TI}$ (time-invariant, including position/scale/offset/time-invariant features) and $\mathbf{V}_{GF}$ (time-dependent feature stream). For $\mathbf{V}_{GF}$, an autoregressive CNN is utilized to predict the distribution of the next frame $\{\boldsymbol{\mu}_t, \boldsymbol{\sigma}_t\}$, combined with joint training using Quantization-Aware Training (STE) and entropy regularization $\mathcal{L}_{entropy}$. During encoding, rANS is used. The resolution of the feature stream scales down significantly after pruning.
    - **Design Motivation**: Temporal alignment based on canonical space is key to compression. Due to temporal correspondence, an autoregressive approach can efficiently predict the distribution of the next frame, which achieves much higher compression efficiency than discrete distribution methods like 4DGS.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{photo} + \lambda_e \mathcal{L}_{entropy} + \lambda_r(\mathcal{L}_s + \mathcal{L}_{ss} + \mathcal{L}_m)$:
- Photometric loss: L1 + SSIM
- Entropy regularization: Autoregressive probability estimation
- Temporal smoothness loss $\mathcal{L}_s$: L1 penalty on attributes of adjacent timesteps
- Spatial smoothness loss $\mathcal{L}_{ss}$: MSE between reorganized 2D frames and their blurred versions
- Mask regularization $\mathcal{L}_m = |M|$: Encourages sparsity in $M_{de}, M_{dy}, M_{knn}, M_p$

Training Strategy: The first 5% of training steps only optimize the canonical space; 5% to 20% involve joint training without compression; thereafter, quantization-aware training and entropy constraints are introduced. Densification and pruning are performed every 500 steps. In gradient accumulation, a hybrid of temporal maximum and temporal average gradient is used: $\bar{\mathbf{g}} = \alpha\max_t(\mathbf{g}_t) + (1-\alpha)\frac{1}{L}\sum_t \mathbf{g}_t$ to ensure fast-moving regions are not neglected.

## Key Experimental Results

### Main Results

| Dataset | Method | PSNR↑ | SSIM↑ | Storage(MB)↓ | FPS↑ |
|--------|------|-------|-------|-------------|------|
| Panoptic Sport | 4DGS | 28.68 | 0.911 | 973.8 | 200 |
| Panoptic Sport | STG | 25.09 | 0.900 | 180.9 | 270 |
| Panoptic Sport | CSTG+PP | 26.13 | 0.902 | 23.4 | 360 |
| Panoptic Sport | **GIFStream** | **29.50** | **0.931** | **12.6** | 100 |
| MPEG | 4DGS | 30.50 | 0.888 | 114 | 80 |
| MPEG | CSTG+PP | 29.48 | 0.885 | 15 | 115 |
| MPEG | **GIFStream** | **30.72** | **0.892** | **7** | 70 |

GIFStream achieves the smallest storage across all datasets while maintaining or exceeding SOTA rendering quality.

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | Storage(MB)↓ |
|------|-------|-------|-------------|
| Full model | 31.94 | 0.879 | 5.3 |
| Per-frame Scaffold-GS | 31.96 | 0.881 | 1283 |
| w/o compression | 32.13 | 0.884 | 46.1 |
| w/o feature stream $\mathbf{f}_t$ | 30.59 | 0.867 | 4.4 |
| w/o sparse mask $M_{de}$ | 31.93 | 0.879 | 6.5 |

### Key Findings

- The feature stream makes the most substantial contribution: removing the feature stream drops PSNR by 1.35dB, indicating that time-varying features are critical for modeling dynamic scenes.
- The sparse mask $M_{de}$ is effective: removing it increases storage by 1.2MB (23%) with almost no change in quality, verifying that the feature streams of most anchors can indeed be sparsified.
- End-to-end compression reduces the storage from 46.1MB to 5.3MB (an 8.7x compression ratio) while only incurring a 0.2dB drop in PSNR.
- In fast-moving scenes (e.g., basketball in Panoptic Sport), GIFStream correctly reconstructs details like motion blur, whereas 4DGaussian and CSTG yield blur or artifacts.
- Decode speed is acceptable: Feature distribution prediction runs at 100 FPS, and rANS entropy decoding runs at 200 FPS (for the feature stream).

## Highlights & Insights

- **Co-Design of Representation and Compression**: Instead of designing the representation first and then applying compression, this work considers compression-friendliness during the representation design phase. The canonical space provides temporal alignment, the sparsity of the feature stream reduces data volume, and mature video coding technologies can be utilized after sorting and mapping to 2D. This co-design philosophy is highly instructive for all future works in dynamic scene representation.
- **Motion-Adaptive Sparsity**: The model automatically determines which anchors require time-varying information via $M_{de}$, avoiding manual segmentation of static/dynamic regions. Retaining feature streams for only 30% of anchors in complex scenes and 0.3% in simple scenes proves that this data-driven sparsity is highly efficient.
- **Modified Gradient Accumulation**: To solve the problem where gradients of fast-moving objects are diluted by temporal averaging in 4D scenes, the method combines the temporal maximum and mean to guide densification. This small modification is remarkably practical.

## Limitations & Future Work

- Although the rendering FPS (70–100) exceeds the 60 FPS threshold, it is lower than 4DGS (200 FPS) because it requires inference through the deformation MLP.
- Initialization relies on the COLMAP sparse point cloud of the first frame, making it sensitive to the reconstruction quality of the initial frame.
- Joint training of the GOP implies that the entire video sequence must be available, which does not support real-time/online scenarios.
- It is suitable for multi-view videos of moderate complexity but may be limited when facing extreme occlusions or ultra-large scale scenes.
- Future research could explore combinations with NeRF-based dynamic methods or introduce more advanced motion models.

## Related Work & Insights

- **vs 4DGS/STG**: These 4D Gaussian methods deliver high quality but suffer from massive storage (180–970MB) and lack temporal correspondence, making them difficult to compress. GIFStream achieves superior quality with much smaller storage (7–13MB) thanks to its canonical + deformation + feature stream design.
- **vs CSTG**: CSTG performs post-processing compression on top of STG. It achieves storage comparable to GIFStream (15–23MB) but yields lower quality because STG is inherently less capable of handling rapid motion.
- **vs V3/Mega**: These methods also attempt to compress using temporal correspondence, but V3 is trained frame-by-frame and struggles to represent new content, while Mega compresses 4DGS using deformation. GIFStream's feature stream design offers greater flexibility.

## Rating

- Novelty: ⭐⭐⭐⭐ The combined design of feature streams + end-to-end compression is novel, although individual components are inspired by prior works.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive with evaluations on three datasets, RD curve comparisons, detailed ablations, and decoding speed analysis.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with detailed method descriptions.
- Value: ⭐⭐⭐⭐ Significantly advances the practical application of immersive video, bringing the bitrate down to 30Mbps, comparable to 4K 2D video.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] 4DGC: Rate-Aware 4D Gaussian Compression for Efficient Streamable Free-Viewpoint Video](4dgc_rate-aware_4d_gaussian_compression_for_efficient_streamable_free-viewpoint_.md)
- [\[CVPR 2025\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [\[CVPR 2025\] DiET-GS: Diffusion Prior and Event Stream-Assisted Motion Deblurring 3D Gaussian Splatting](diet-gs_diffusion_prior_and_event_stream-assisted_motion_deblurring_3d_gaussian_.md)
- [\[ICCV 2025\] Compression of 3D Gaussian Splatting with Optimized Feature Planes and Standard Video Codecs](../../ICCV2025/3d_vision/compression_of_3d_gaussian_splatting_with_optimized_feature_planes_and_standard_.md)
- [\[CVPR 2025\] Gaussian Splatting Feature Fields for Privacy-Preserving Visual Localization](gaussian_splatting_feature_fields_for_privacy-preserving_visual_localization.md)

</div>

<!-- RELATED:END -->
