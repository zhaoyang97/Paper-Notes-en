---
title: >-
  [Paper Note] Self-Supervised Sparse Sensor Fusion for Long Range Perception
description: >-
  [ICCV 2025][Autonomous Driving][long-range perception] LRS4Fusion proposes a long-range LiDAR-camera fusion framework based on sparse voxel representations, combined with a self-supervised pretraining strategy via sparse occupancy and velocity field reconstruction, achieving state-of-the-art performance within a 250-meter perception range: a 26.6% improvement in object detection mAP and a 30.5% reduction in LiDAR prediction Chamfer Distance.
tags:
  - ICCV 2025
  - Autonomous Driving
  - long-range perception
  - sparse voxel fusion
  - self-supervised pretraining
  - LiDAR-camera fusion
  - depth estimation
date: 2026-05-08
content_hash: 332f648cef0b239d
---

# Self-Supervised Sparse Sensor Fusion for Long Range Perception

**Conference**: ICCV 2025
**arXiv**: [2508.13995](https://arxiv.org/abs/2508.13995)
**Code**: [https://light.princeton.edu/LRS4Fusion](https://light.princeton.edu/LRS4Fusion)
**Area**: Autonomous Driving
**Keywords**: long-range perception, sparse voxel fusion, self-supervised pretraining, LiDAR-camera fusion, depth estimation

## TL;DR

LRS4Fusion proposes a long-range LiDAR-camera fusion framework based on sparse voxel representations, combined with a self-supervised pretraining strategy via sparse occupancy and velocity field reconstruction, achieving state-of-the-art performance within a 250-meter perception range: a 26.6% improvement in object detection mAP and a 30.5% reduction in LiDAR prediction Chamfer Distance.

## Background & Motivation

**Background**: Current autonomous driving perception methods are primarily built on Bird's Eye View (BEV) representations and perform well in urban driving scenarios spanning 50–100 meters. Dominant approaches such as BEVFusion and BEVFormer employ dense BEV feature maps for 3D object detection, semantic occupancy prediction, tracking, and planning.

**Limitations of Prior Work**: (1) The memory and computational overhead of BEV representations scales quadratically with range, making extension beyond 250 meters extremely challenging. (2) Highway driving requires at least 250 meters of perception range (dictated by braking distances at 100 km/h), and heavy trucks (40 tons) demand even longer planning horizons—far exceeding the 50–100 meter range of existing methods. (3) Objects at long range are extremely sparse (instance counts decrease with distance), and annotation costs are prohibitively high.

**Key Challenge**: The quadratic complexity of dense BEV representations prevents scaling to long ranges. Self-supervised pretraining methods (e.g., ViDAR, UnO) support only a single modality (camera or LiDAR), failing to leverage multi-modal information. Additionally, existing datasets provide LiDAR coverage only up to 80 meters.

**Goal**: (1) How to efficiently fuse multi-modal data within a 250-meter range? (2) How to perform self-supervised learning when long-range objects are extremely sparse? (3) How to preserve temporal information within a sparse representation?

**Key Insight**: Long-range scenes are naturally sparse (most of the space is empty), making sparse voxel representations a natural fit that avoids the quadratic growth of dense representations. The work also leverages the capabilities of FMCW LiDAR, which can measure radial velocity and covers ranges up to 400 meters.

**Core Idea**: Replace dense BEV with a fully sparse voxel representation. Camera features are lifted to 3D via accurate depth estimation and fused with LiDAR features in a shared sparse voxel space. Temporal information is integrated via sparse window attention, and long-range multi-modal representations are learned through occupancy-velocity self-supervised pretraining.

## Method

### Overall Architecture

LRS4Fusion consists of: (1) a camera encoder with depth estimation that lifts 2D features to 3D sparse voxels; (2) a LiDAR encoder producing sparse voxel features; (3) a sparse fusion module merging the two modalities; (4) a sparse post-encoder (completion + context aggregation) handling multi-scale features; (5) sparse window attention for temporal fusion; and (6) self-supervised pretraining using occupancy and velocity decoders. Final outputs include depth, occupancy, velocity, future LiDAR predictions, and object detections.

### Key Designs

1. **Accurate Depth Estimation Module**:

    - **Function**: Estimates a dense depth map from RGBD input (RGB + sparse LiDAR depth projection).
    - **Mechanism**: A multi-scale recurrent architecture uses a Vim backbone to extract multi-scale features $F_1^i, F_2^i, F_3^i, F_4^i = f_{img}(I_i^{RGBD})$. A Minimal Gated Unit (MGU) replaces GRU to iteratively refine the depth map; at each iteration, a depth gradient $\nabla d_t = F_g(h_t)$ is estimated from context features, and the depth is updated by fusing sparse LiDAR depth and image gradients: $d_{t+1} = d_t - \Delta d$, where $\Delta d = f_{update}(\nabla d_t - g, (d_t - s_d) \odot M, C_{dg}, C_{inp})$.
    - **Design Motivation**: Accurate depth estimation is the key bottleneck for long-range perception. The MGU has only one forget gate, reducing parameters and computation by one-third compared to GRU. Inference requires only 64 ms and 1.3 GB of memory, significantly faster than CompletionFormer (188 ms, 2.1 GB) and OGNI-DC (364 ms, 2.4 GB).

2. **Sparse Multi-Modal Fusion**:

    - **Function**: Fuses camera and LiDAR features in a unified sparse voxel space.
    - **Mechanism**: (1) Camera features are projected to 3D using predicted depth $D_i$ and camera matrix $K$: $\mathbf{X}_C = D_i(u,v) K^{-1}(u,v,1)$, then converted to sparse voxels $F_C^i = [\mathbf{F}_C, \mathbf{X}_C]$. (2) LiDAR features are obtained via voxelized PointNet + sparse U-Net: $F_L = [\mathbf{F}_L, \mathbf{X}_L]$. (3) In the sparse fusion module, features from both modalities are concatenated (with zero-padding for voxels occupied by only one modality), then fused via batch normalization and sparse convolution: $F_{LC} = [\mathbf{F}_{LC}, \mathbf{X}_{LC}]$, with total voxel count $Q = M + N - O$ where $O$ is the number of overlapping voxels.
    - **Design Motivation**: Fully sparse representations make computational complexity linear in the number of occupied voxels rather than quadratic in spatial extent, enabling scaling to 250 meters.

3. **Sparse Window Attention for Temporal Fusion**:

    - **Function**: Integrates historical frame information while preserving sparsity.
    - **Mechanism**: Previous-frame voxels are aligned to the current frame via rigid-body transformation and velocity correction: $\mathbf{X}_q^{t_0'} = (\mathbf{X}_q^{t_{-1}} + \mathbf{v}_q^{t_{-1}} dt) \mathbf{R|T}^{t_{-1} \to t_0}$. Each occupied voxel in the current frame then queries neighboring aligned voxels from the previous frame via 3D window attention: $V_* = \sum_{V^{t_0'} \in J_s} \text{softmax}(\frac{V^{t_0} (V^{t_0'})^T}{\sqrt{d}}) V^{t_0'}$.
    - **Design Motivation**: Simply concatenating past and current voxels causes the voxel count to grow over time, eliminating the sparsity advantage. By using current-frame occupied voxels as queries and past-frame voxels as keys/values in window attention, the output voxel count remains bounded.

4. **Self-Supervised Pretraining**:

    - **Function**: Learns strong multi-modal spatiotemporal representations from unlabeled data.
    - **Mechanism**: Sparse occupancy and velocity decoders accept 4D query points $(x,y,z,t)$, interpolate in voxel space, and predict occupancy $\hat{o}$ and velocity $\hat{v}$. For future/past timestamps, a lightweight network $f_{pose}$ predicts new query positions, and predictions are made from interpolated features at both locations. Ground truth is derived from LiDAR scans: occupancy is determined by point existence, free space by LiDAR ray casting, and velocity is directly measured by FMCW LiDAR.
    - **Design Motivation**: Long-range objects are extremely sparse, making annotation prohibitively expensive. Self-supervised pretraining on 60,000 unlabeled frames learns spatiotemporal representations and substantially reduces annotation requirements.

### Loss & Training

Three-stage training: (1) Training the image feature encoder and depth prediction (image reconstruction + depth supervision + feature distillation losses); (2) Full model self-supervised training (occupancy and velocity reconstruction for past, current, and future frames); (3) Fine-tuning a detection head (CenterPoint) on top of the pretrained backbone.

## Key Experimental Results

### Main Results — Long-Range Object Detection

| Method | Modality | mAP↑ | NDS↑ |
|--------|----------|------|------|
| PointPillars | L | 39.31 | 41.52 |
| BEVFormer | C | 23.67 | 37.99 |
| BEVFormer (w/ ViDAR pretraining) | C | 24.51 | 38.93 |
| BEVFusion | L+C | 40.10 | 48.43 |
| SAMFusion | L+C | 41.55 | 52.44 |
| LRS4Fusion (w/o pretraining) | L+C | 49.58 | 59.12 |
| **LRS4Fusion** | **L+C** | **52.61** | **58.06** |

A 26.6% improvement (+11.06 mAP) over the second-best method, SAMFusion.

### Ablation Study — NuScenes LiDAR Prediction

| Method | Modality | 1s→1s CD↓ | 1s→3s CD↓ |
|--------|----------|-----------|-----------|
| 4DOcc | L | 1.88 | - |
| ViDAR | C | 1.25 | 1.97 |
| **LRS4Fusion** | **L+C** | **0.48** | **1.25** |

CD improves by 61.6% on the NuScenes 1s→1s task and by 36.5% on the 1s→3s task.

### Depth Estimation Comparison

| Method | MAE↓ | RMSE↓ | Latency (ms)↓ | Memory (GB)↓ |
|--------|------|-------|---------------|--------------|
| CompletionFormer | 4.98 | 12.36 | 188 | 2.1 |
| OGNI-DC | 4.76 | 13.16 | 364 | 2.4 |
| **LRS4Fusion** | **3.46** | **9.21** | **64** | **1.3** |

### Key Findings

- **Self-supervised pretraining contributes significantly**: Detection mAP improves from 49.58 to 52.61 (+6.11%), demonstrating the value of learning spatiotemporal representations from unlabeled data.
- **Camera modality alone is insufficient at long range**: BEVFormer achieves only 23.67 mAP due to the lack of depth cues at long distances; however, accurate depth estimation combined with LiDAR fusion effectively compensates for this.
- **BEV fusion methods fail at long range**: BEVFusion outperforms the LiDAR-only PointPillars by only 2.01%, as LSS-based depth estimation is inaccurate at long distances.
- **Sparse voxels remain sparse across all scales**, further reducing memory usage and enabling finer-grained discretization.
- In high-speed scenarios, 4DOcc performance degrades due to large inter-frame displacements (1s history CD: 16.87 vs. 3s history CD: 23.58), highlighting the importance of motion-corrected temporal fusion.

## Highlights & Insights

- **Fully sparse representation is the key to long-range perception**: Dense BEV is infeasible at 250 meters; sparse voxels scale linearly with the number of occupied voxels, naturally suiting the sparse nature of long-range scenes. This design principle generalizes to any application requiring large-scale perception.
- **MGU-based depth estimation as an efficient alternative to GRU**: Reduces parameters by one-third with faster inference, achieving 27% lower MAE and 3× faster inference in long-range scenarios—a strong baseline for efficient depth completion.
- **Self-supervised LiDAR prediction as pretraining**: Using future LiDAR reconstruction as a self-supervised objective learns powerful 3D spatiotemporal representations without any annotations, and is more general than ViDAR (camera-only) or UnO (LiDAR-only).

## Limitations & Future Work

- The method requires FMCW LiDAR for velocity measurements; standard LiDAR cannot directly provide voxel-level velocity, limiting generalizability.
- The proprietary dataset is not publicly released, constraining reproducibility; comparisons are made against a limited number of baselines.
- Temporal fusion uses only one historical frame ($t_0$ and $t_{-1}$); the effect of longer temporal windows is unexplored.
- Object detection still requires annotated data for fine-tuning (Stage 3), so the pipeline is not fully unsupervised.
- The window size in sparse window attention is fixed and not adaptively adjusted.

## Related Work & Insights

- **vs. BEVFusion/BEVFormer**: Dense BEV methods perform well at short range but cannot scale to 250 meters. LRS4Fusion's sparse representation significantly outperforms these methods at long range.
- **vs. ViDAR**: This camera-only self-supervised pretraining method achieves a CD of 1.25 on NuScenes (vs. 0.48 for LRS4Fusion), with the gap attributable to the absence of depth cues and multi-modal information.
- **vs. 4DOcc/UnO**: LiDAR self-supervised methods exhibit significant performance degradation in high-speed scenarios (large inter-frame displacements). LRS4Fusion's velocity correction and multi-modal fusion effectively address this issue.
- **vs. SAMFusion**: This depth-based 3D projection method achieves an mAP of 41.55 at long range vs. 52.61 for LRS4Fusion, demonstrating the combined advantage of accurate depth estimation and sparse representation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of sparse voxel fusion and multi-modal self-supervised pretraining is a genuine contribution, though individual components build on prior work.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Thoroughly validated on a proprietary long-range dataset and NuScenes across multiple tasks: depth estimation, detection, and prediction.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are detailed and problem motivation is clearly articulated.
- **Value**: ⭐⭐⭐⭐⭐ Extending the perception range from 50–100 meters to 250 meters has significant practical implications for autonomous heavy-truck driving.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] GaussianFlowOcc: Sparse and Weakly Supervised Occupancy Estimation using Gaussian Splatting and Temporal Flow](gaussianflowocc_sparse_and_weakly_supervised_occupancy_estimation_using_gaussian.md)
- [\[NeurIPS 2025\] Availability-aware Sensor Fusion via Unified Canonical Space](../../NeurIPS2025/autonomous_driving/availability-aware_sensor_fusion_via_unified_canonical_space.md)
- [\[ICCV 2025\] AD-GS: Object-Aware B-Spline Gaussian Splatting for Self-Supervised Autonomous Driving](ad-gs_object-aware_b-spline_gaussian_splatting_for_self-supervised_autonomous_dr.md)
- [\[NeurIPS 2025\] Layer-wise Modality Decomposition for Interpretable Multimodal Sensor Fusion](../../NeurIPS2025/autonomous_driving/layer-wise_modality_decomposition_for_interpretable_multimodal_sensor_fusion.md)
- [\[AAAI 2026\] Walking Further: Semantic-aware Multimodal Gait Recognition Under Long-Range Conditions](../../AAAI2026/autonomous_driving/walking_further_semantic-aware_multimodal_gait_recognition_under_long-range_cond.md)

<!-- RELATED:END -->
