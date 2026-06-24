---
title: >-
  [Paper Note] SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos
description: >-
  [CVPR 2025][3D Vision][Real-Time Dense Reconstruction] SLAM3R proposes a two-level feed-forward neural network system. It directly regresses local 3D point maps from video segments using an Image-to-Points (I2P) network, and then progressively aligns them to a global coordinate system using a Local-to-World (L2W) network. This achieves SOTA dense reconstruction accuracy and completeness at 20+ FPS, entirely without explicitly solving for camera parameters.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Real-Time Dense Reconstruction"
  - "End-to-End 3D Reconstruction"
  - "Feed-Forward Neural Network"
  - "Parameter-Free SLAM"
  - "Video Reconstruction"
date: 2026-05-08
content_hash: d7c9c9f951e14cda
---

# SLAM3R: Real-Time Dense Scene Reconstruction from Monocular RGB Videos

**Conference**: CVPR 2025  
**arXiv**: [2412.09401](https://arxiv.org/abs/2412.09401)  
**Code**: [GitHub](https://github.com/PKU-VCL-3DV/SLAM3R)  
**Area**: 3D Vision / Dense Reconstruction  
**Keywords**: Real-Time Dense Reconstruction, End-to-End 3D Reconstruction, Feed-Forward Neural Network, Parameter-Free SLAM, Video Reconstruction

## TL;DR

SLAM3R proposes a two-level feed-forward neural network system. It directly regresses local 3D point maps from video segments using an Image-to-Points (I2P) network, and then progressively aligns them to a global coordinate system using a Local-to-World (L2W) network. This achieves SOTA dense reconstruction accuracy and completeness at 20+ FPS, entirely without explicitly solving for camera parameters.

## Background & Motivation

Dense 3D reconstruction has long struggled to balance accuracy, completeness, and efficiency. Traditional methods rely on multi-stage SfM+MVS pipelines, which offer high reconstruction quality but require offline processing. Existing monocular dense SLAM systems improve quality using neural implicit or 3DGS representations, but run far below real-time speeds (e.g., NICER-SLAM operates at less than 1 FPS).

DUSt3R pioneered the direction of end-to-end dense reconstruction, but multi-view scaling requires exhaustive image pair matching and global optimization, which is highly inefficient. Spann3R accelerates this via an incremental pipeline but suffers from severe cumulative drift. The core idea of SLAM3R is to minimize drift by using multi-frame inputs at each level, while referencing similar frames from long-term history as global references via a self-contained retrieval module.

## Method

### Overall Architecture

SLAM3R uses a sliding window to convert input video into overlapping segments. The I2P network processes frames within each window, selects a keyframe to define a local coordinate system, and regresses dense 3D point maps for all frames. The L2W network progressively fuses local reconstructions into the global coordinate system. Both modules share a similar ViT architecture, and the entire process does not explicitly solve for any camera parameters.

### Key Design 1: Image-to-Points Multi-view Extension

- **Function**: Extends DUSt3R from two-view to multi-view, directly predicting dense 3D point maps from video windows.
- **Mechanism**: Uses a shared encoder $E_{img}$ to encode each frame independently. The keyframe decoder $D_{key}$ introduces **multi-view cross-attention**—performing cross-attention between the keyframe query and the keys/values of each support frame independently, and then aggregating multi-view information via max-pooling. The support frame decoder $D_{sup}$ follows the DUSt3R architecture to interact only with the keyframe. By default, the middle frame is selected as the keyframe (having the maximum overlap with other frames).
- **Design Motivation**: DUSt3R's original two-view design requires exhaustive pairwise matching in multi-view scenarios, compromising efficiency. Multi-view cross-attention allows simultaneous processing of an arbitrary number of support frames. This design of independent cross-attention combined with max-pooling is simple and efficient.

### Key Design 2: Local-to-World Progressive Global Registration

- **Function**: Aligns local reconstructions to the global coordinate system to eliminate cumulative drift.
- **Mechanism**: Maintains a buffer set of limited capacity $B$ storing registered frames, using a reservoir sampling strategy. When registering a new frame, a retrieval module (sharing the first $r$ decoder blocks of I2P + linear projection + average pooling) is used to select the top-$K$ most relevant scene frames from the buffer set. The 3D point maps are encoded as geometric tokens via patch embedding, added to visual tokens, and fed into the registration decoder $D_{reg}$ and scene decoder $D_{sce}$.
- **Design Motivation**: The frame-by-frame increment in Spann3R leads to severe drift. SLAM3R provides a more global reference through multi-frame reference and long-term history retrieval, effectively reducing drift.

### Key Design 3: Self-contained Retrieval and Scene Initialization

- **Function**: Efficiently selects the best reference frames to ensure accurate scene initialization.
- **Mechanism**: The retrieval module measures visual similarity and baseline suitability in the feature space to select the top-$K$ scene frames. For scene initialization, $L$ passes of I2P are executed on the first window (iterating through each frame as the keyframe) to select the result with the highest total confidence.
- **Design Motivation**: The retrieval module reuses the decoder blocks of I2P, introducing zero extra parameters. The accuracy of initialization is critical to global reconstruction.

### Loss & Training

I2P loss: $\mathcal{L}_{I2P} = \sum_{i=1}^{L} M_i \cdot (\hat{C}_i \cdot \text{L1}(\frac{1}{\hat{z}}\hat{X}_i, \frac{1}{z}X_i) - \alpha \log \hat{C}_i)$, using confidence-weighted L1 distance and normalized scale. The L2W loss is similar but without normalization (as the output needs to align with the scale of the scene frames).

## Key Experimental Results

### Main Results: 7-Scenes Dataset Reconstruction Quality (Acc./Comp. cm)

| Method | Avg Acc↓ | Avg Comp↓ | FPS |
|------|---------|----------|-----|
| DUSt3R | 2.19 | 3.24 | ≪1 |
| MASt3R | 3.04 | 3.90 | ≪1 |
| Spann3R | 3.42 | 2.41 | >50 |
| **SLAM3R** | **1.63** | **1.31** | **~25** |

### Ablation Study: Effect of Confidence Filtering

| Configuration | Avg Acc↓ | Avg Comp↓ |
|------|---------|----------|
| SLAM3R-NoConf | 2.40 | 2.24 |
| **SLAM3R** | **1.63** | **1.31** |

### Key Findings

- Both accuracy (Acc) and completeness (Comp) significantly outperform DUSt3R and Spann3R.
- Real-time performance is achieved at 25 FPS, which is several orders of magnitude faster than DUSt3R's global optimization.
- Cumulative drift is less than half of that in Spann3R.
- Confidence filtering successfully removes unreliable points, reducing Acc from 2.40 to 1.63.

## Highlights & Insights

1. **Camera-Parameter-Free Dense SLAM**: Completely bypasses camera parameter estimation and directly predicts 3D point maps in a unified coordinate system. This execution-level simplification delivers dual improvements in both efficiency and quality.
2. **Two-Level Shared Architecture**: I2P and L2W utilize a similar multi-view cross-attention architecture, exhibiting high design consistency.
3. **Retrieval-based Long-Term Memory**: Reservoir sampling + feature retrieval enables scalable processing of arbitrarily long videos.

## Limitations & Future Work

- Requires training on large-scale datasets; the model's generalization ability to out-of-distribution scenes remains to be verified.
- Currently supports static scenes only.
- The sliding window strategy limits the manageable magnitude of inter-frame motion.
- Future work could incorporate dynamic scene reconstruction and larger-scale training data.

## Related Work & Insights

- **DUSt3R**: Pioneered end-to-end dense 3D reconstruction; SLAM3R extends it to multi-view and incremental settings.
- **Spann3R**: Concurrent work extending DUSt3R to videos via spatial memory, but suffers from severe cumulative drift.
- **DROID-SLAM**: Iteratively updates poses and depth, but has inferior reconstruction quality compared to SLAM3R.
- **Insight**: Feed-forward 3D prediction + progressive fusion suggests a promising paradigm for future dense reconstruction.

## Rating

⭐⭐⭐⭐⭐ — Simultaneously achieves optimality across three core metrics of dense 3D reconstruction (accuracy, completeness, and efficiency). Its real-time performance at 20+ FPS holds significant practical value. The two-level framework is clearly designed, decoupling local reconstruction and global registration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ODHSR: Online Dense 3D Reconstruction of Humans and Scenes from Monocular Videos](odhsr_online_dense_3d_reconstruction_of_humans_and_scenes_from_monocular_videos.md)
- [\[CVPR 2025\] MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors](mast3r-slam_real-time_dense_slam_with_3d_reconstruction_priors.md)
- [\[CVPR 2025\] SplineGS: Robust Motion-Adaptive Spline for Real-Time Dynamic 3D Gaussians from Monocular Video](splinegs_robust_motion-adaptive_spline_for_real-time_dynamic_3d_gaussians_from_m.md)
- [\[ICCV 2025\] FROSS: Faster-than-Real-Time Online 3D Semantic Scene Graph Generation from RGB-D Images](../../ICCV2025/3d_vision/fross_faster-than-real-time_online_3d_semantic_scene_graph_generation_from_rgb-d.md)
- [\[CVPR 2025\] Mobile-GS: Real-time Gaussian Splatting for Mobile Devices](mobile-gs_real-time_gaussian_splatting_for_mobile_devices.md)

</div>

<!-- RELATED:END -->
