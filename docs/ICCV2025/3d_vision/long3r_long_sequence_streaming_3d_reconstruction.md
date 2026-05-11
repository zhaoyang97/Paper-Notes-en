---
title: >-
  [Paper Note] LONG3R: Long Sequence Streaming 3D Reconstruction
description: >-
  [ICCV 2025][3D Vision][Streaming 3D reconstruction] This paper proposes LONG3R, a streaming multi-view 3D reconstruction model based on a recurrent memory mechanism. Through three key innovations — memory gating…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Streaming 3D reconstruction"
  - "pointmap regression"
  - "spatio-temporal memory"
  - "long-sequence reconstruction"
  - "real-time inference"
date: 2026-05-08
content_hash: 25654ffc5bbf1c27
---

# LONG3R: Long Sequence Streaming 3D Reconstruction

**Conference**: ICCV 2025
**arXiv**: [2507.18255](https://arxiv.org/abs/2507.18255)
**Code**: [Project Page](https://zgchen33.github.io/LONG3R/)
**Area**: 3D Vision
**Keywords**: Streaming 3D reconstruction, pointmap regression, spatio-temporal memory, long-sequence reconstruction, real-time inference

## TL;DR

This paper proposes LONG3R, a streaming multi-view 3D reconstruction model based on a recurrent memory mechanism. Through three key innovations — memory gating, a dual-source refined decoder, and 3D spatio-temporal memory — LONG3R significantly improves long-sequence reconstruction quality while maintaining real-time inference speed.

## Background & Motivation

Recovering dense geometry from image sequences is a fundamental task in 3D computer vision, with broad applications in robotics, autonomous driving, and indoor/outdoor scene reconstruction. Traditional methods (SfM, SLAM, MVS) rely on hand-crafted pipelines that require substantial engineering effort and exhibit limited generalization. Recent end-to-end neural approaches, typified by DUSt3R/MASt3R, have achieved breakthroughs by directly regressing pointmaps from image pairs, and Spann3R further extended this paradigm to streaming inputs for real-time processing.

However, Spann3R exhibits three critical bottlenecks on long sequences:

**Insufficient memory utilization**: Memory is attended to only once per iteration, precluding effective reuse.

**Spatial redundancy**: As frames accumulate, the memory bank accumulates large numbers of spatially overlapping tokens.

**Limited training strategy**: The absence of long-sequence adaptation during training leads to severe performance degradation on long sequences.

LONG3R addresses these three issues with memory gating, 3D spatio-temporal memory, and a two-stage curriculum training strategy, respectively.

## Method

### Overall Architecture

LONG3R adopts a recurrent network architecture for processing streaming image sequences. For each new observation frame, the model executes the following pipeline:

1. **Feature Encoding**: A ViT-Large encoder partitions the input image into patches and projects them into visual feature tokens.
2. **Coarse Decoding**: A coarse decoder uses PairwiseBlocks to interact with features from the previous frame, generating a coarse 3D structure.
3. **Memory Gating**: Relevant memory tokens are selected from the spatio-temporal memory bank based on the current observation.
4. **Refined Decoding**: A dual-source refined decoder combines the retrieved memory and next-frame context to produce accurate pointmaps.
5. **Memory Update**: The refined features of the current frame are added to the memory bank.

### Key Design 1: Attention-based Memory Gating

The memory gating mechanism serves two core functions: aggregating information from the entire memory bank and filtering irrelevant memory tokens to reduce the computational load on subsequent decoder layers.

Concretely, the coarse decoder output $F_t^c$ serves as the query for cross-attention over memory keys $F_{mem}^K$ and values $F_{mem}^V$. An attention threshold $\tau = 5 \times 10^{-4}$ is applied: any memory token whose maximum attention weight falls below this threshold is discarded. Experiments show that this mechanism filters approximately 27% of redundant memory on average, improving inference speed from 18 FPS to 21.4 FPS (a 20% gain) with negligible impact on reconstruction accuracy.

### Key Design 2: Dual-Source Refined Decoder

Unlike the coarse decoder, which consists solely of PairwiseBlocks, the refined decoder alternates between two types of modules:

- **Odd layers — PairwiseBlock**: The current frame's refined features interact with the next frame's coarse features, maintaining temporal alignment.
- **Even layers — MemoryBlock**: The current frame's refined features interact with the retrieved memory tokens, enhancing long-range spatio-temporal dependencies.

This interleaved design avoids the feature-space misalignment that arises when memory features and next-frame features are directly concatenated. Ablation studies show that the interleaved architecture substantially outperforms the concatenated architecture: on Replica200, Accuracy improves from 29.52 cm to 13.34 cm (mean), and Completeness from 8.88 cm to 3.15 cm.

### Key Design 3: 3D Spatio-Temporal Memory

The memory system comprises two components:

- **Short-term temporal memory**: Stores all tokens (keys/values) from the most recent $K$ frames to capture local temporal information.
- **Long-term 3D spatial memory**: Manages tokens from earlier frames, with voxelization-based pruning to enforce a maximum capacity of 3,000 tokens.

**Adaptive voxel size**: Since scene scales vary widely across datasets, a predefined voxel size is impractical. The model computes the mean 3D Euclidean distance $d_i$ between each token and its 8 spatial neighbors, and takes the minimum value as the per-frame image voxel size $v_{img}$; the scene voxel size $v_{scene}$ is then computed as the mean over all historical frames and updated online throughout inference.

**Spatial memory pruning**: Tokens that are spatially close in 3D are assigned to the same voxel, and only the token with the highest accumulated attention weight is retained per voxel. This mechanism effectively balances memory size against the completeness of spatial scene representation.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{conf} + \mathcal{L}_{scale}$$

- $\mathcal{L}_{conf}$: Confidence-aware 3D regression loss (inherited from DUSt3R/Spann3R).
- $\mathcal{L}_{scale}$: Scale loss that encourages the mean distance of the predicted point cloud to be smaller than that of the ground truth.

**Two-stage curriculum learning**:

- **Stage 1**: Five frames are randomly sampled per video for training, establishing basic 3D understanding. AdamW optimizer, learning rate $1.12 \times 10^{-4}$, batch size 10 per GPU, 120 epochs, trained on 16× A100 GPUs for 28 hours.
- **Stage 2**: The ViT encoder is frozen and the remaining modules are fine-tuned. Sequence length is progressively increased, first to 10 frames and then to 32 frames. Learning rate is reduced to $1 \times 10^{-5}$, with 12 epochs per stage, totaling approximately 20 hours.

## Key Experimental Results

### Main Results: 3D Reconstruction (224×224 Input)

**7Scenes / NRGBD Datasets** (Tab. 1):

| Method | 7Scenes Acc↓ | 7Scenes Comp↓ | NRGBD Acc↓ | NRGBD Comp↓ | FPS |
|--------|-------------|---------------|------------|-------------|-----|
| DUSt3R | 3.01 / 1.47 | 5.11 / 2.79 | 3.94 / 2.48 | 5.31 / 3.58 | ≤3 |
| Spann3R | 3.42 / 1.48 | 2.41 / 0.85 | 6.91 / 3.15 | 2.91 / 1.10 | ~22 |
| CUT3R | 7.73 / 3.57 | 7.75 / 1.83 | 12.48 / 5.57 | 6.34 / 2.35 | ~23 |
| **LONG3R** | **2.57 / 1.14** | **2.08 / 0.73** | **6.66 / 2.54** | 3.11 / 1.21 | ~22 |

**Replica Long-Sequence Dataset** (Tab. 2, 100 frames / 200 frames):

| Method | Rep100 Acc↓ | Rep100 Comp↓ | Rep200 Acc↓ | Rep200 Comp↓ | FPS |
|--------|------------|-------------|------------|-------------|-----|
| Spann3R | 14.08 / 8.88 | 4.67 / 1.61 | 16.29 / 10.17 | 4.02 / 1.16 | ~21 |
| CUT3R | 20.44 / 14.64 | 5.67 / 2.32 | 28.30 / 20.68 | 6.61 / 1.88 | ~23 |
| **LONG3R** | **11.46 / 7.55** | **3.68 / 1.24** | **11.93 / 7.42** | **2.73 / 0.87** | ~21 |

When the sequence length increases from 100 to 200 frames, Spann3R's mean Accuracy degrades from 14.08 to 16.29 and CUT3R's from 20.44 to 28.30, whereas LONG3R increases only marginally from 11.46 to 11.93, demonstrating markedly superior robustness on long sequences.

### Camera Pose Estimation (Tab. 3)

| Method | 7Scenes ATE↓ | ScanNet ATE↓ | TUM ATE↓ |
|--------|-------------|-------------|---------|
| Spann3R | 12.64 | 9.83 | 5.66 |
| CUT3R | 12.40 | 14.27 | 6.25 |
| **LONG3R** | **8.72** | **6.44** | **5.40** |

LONG3R achieves substantial improvements on static scenes (7Scenes, ScanNet) and remains competitive on the TUM dataset, which contains dynamic motion.

### Ablation Study

**Memory gating** (Tab. 4): Removing gating causes negligible change in accuracy but reduces FPS from 21.4 to 18.0, confirming that the primary contribution of gating is efficiency.

**Dual-source decoder architecture** (Tab. 5, Replica200):

| Design | Acc Mean↓ | Comp Mean↓ |
|--------|----------|-----------|
| Concatenated | 29.52 | 8.88 |
| Interleaved | **13.34** | **3.15** |

**3D spatio-temporal memory** (Tab. 6, Replica200):

| Design | Acc Mean↓ | Comp Mean↓ |
|--------|----------|-----------|
| Temporal memory only | 65.75 | 13.24 |
| Spann3R memory | 12.41 | 3.07 |
| LONG3R memory | **11.93** | **2.74** |

Removing 3D spatial memory leads to severe performance collapse (Accuracy surges from 11.93 to 65.75), demonstrating that long-term spatial memory is indispensable for long-sequence reconstruction.

## Highlights & Insights

1. **Simple yet effective memory gating**: A single attention threshold suffices to filter approximately 27% of redundant memory, achieving a favorable accuracy–speed trade-off in an elegant and engineering-friendly manner.
2. **Interleaved attention outperforms feature concatenation**: In multi-source information fusion settings, alternating cross-attention avoids feature-space misalignment, a design paradigm with broad transferability.
3. **Adaptive voxel pruning**: Computing the scene voxel size online enables prior-free memory management that gracefully handles varying scene scales.
4. **Curriculum training strategy**: Progressively increasing sequence length is a simple but effective technique that enables the model to learn long-sequence handling under limited computational resources.
5. **Strong long-sequence robustness**: The performance degradation from 100 to 200 frames is far smaller than that of all baselines, directly validating the core design objectives of the method.

## Limitations & Future Work

1. **First-frame dependency**: All predictions are defined relative to the first frame, which may yield ambiguous results when viewpoints deviate substantially from the initial frame.
2. **Limited capability in dynamic scenes**: Due to the absence of dynamic training data, the model struggles with highly dynamic scenes containing large object motions.
3. **No global optimization or loop closure**: As a purely feed-forward streaming method, accumulated drift is mitigated but not fundamentally resolved.
4. **Resolution constraints**: All experiments use 224×224 inputs; performance and efficiency at higher resolutions remain to be validated.
5. **Training cost**: The two-stage training requires approximately 48 hours on 16× A100 GPUs, representing a considerable resource demand.

## Related Work & Insights

- **DUSt3R / MASt3R**: Pioneering works on end-to-end pointmap regression; LONG3R's encoder is directly initialized from DUSt3R weights.
- **Spann3R**: The first streaming pointmap reconstruction method and the direct predecessor of LONG3R; LONG3R comprehensively improves upon its memory mechanism and decoding strategy.
- **CUT3R**: A recurrent reconstruction method using persistent state tokens, but susceptible to severe drift on long sequences.
- **MV-DUSt3R**: An offline multi-view method with high accuracy but very low FPS (≤7), unsuitable for real-time scenarios.
- **Insights**: The spatio-temporal memory design (short-term temporal + long-term spatial + adaptive pruning) is transferable to other tasks requiring long-range memory management, such as video understanding and SLAM.

## Rating

| Dimension | Score (1–10) | Notes |
|-----------|-------------|-------|
| Novelty | 7 | Each of the three contributions is individually incremental, but their combination yields significant gains. |
| Technical Depth | 8 | Memory gating, adaptive voxelization, and the interleaved decoder are carefully designed. |
| Experimental Thoroughness | 8 | Multiple datasets, metrics, and complete ablations with per-component validation. |
| Writing Quality | 7 | Clear structure with rich figures and tables; some sections are formula-dense but overall readable. |
| Practical Value | 8 | Real-time streaming reconstruction has direct applications in robotics and AR/VR. |
| Overall | **7.5** | A solid systematic contribution that effectively addresses the core bottlenecks of streaming long-sequence 3D reconstruction. |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LongStream: Long-Sequence Streaming Autoregressive Visual Geometry](../../CVPR2026/3d_vision/longstream_long-sequence_streaming_autoregressive_visual_geometry.md)
- [\[ICCV 2025\] FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution](flashdepth_real-time_streaming_video_depth_estimation_at_2k_resolution.md)
- [\[ICCV 2025\] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Videos](longsplat_robust_unposed_3d_gaussian_splatting_for_casual_long_videos.md)
- [\[ICCV 2025\] Boost 3D Reconstruction using Diffusion-based Monocular Camera Calibration](boost_3d_reconstruction_using_diffusion-based_monocular_camera_calibration.md)
- [\[ICCV 2025\] Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images](amodal3r_amodal_3d_reconstruction_from_occluded_2d_images.md)

</div>

<!-- RELATED:END -->
