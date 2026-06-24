---
title: >-
  [Paper Note] RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds
description: >-
  [CVPR 2025][Autonomous Driving][Point cloud compression] RENO proposes Sparse Occupancy Codes and a one-time inference strategy, achieving the first real-time neural compression of 3D LiDAR point clouds (10fps@14-bit). With a model size of only 1MB, it outperforms the G-PCC standard by 12.25% in bitrate savings.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Point cloud compression"
  - "Real-time encoding/decoding"
  - "Sparse occupancy codes"
  - "LiDAR"
  - "Neural encoder"
date: 2026-05-08
content_hash: ed111cc96b77b365
---

# RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds

**Conference**: CVPR 2025  
**arXiv**: [2503.12382](https://arxiv.org/abs/2503.12382)  
**Code**: [github.com/NJUVISION/RENO](https://github.com/NJUVISION/RENO)  
**Area**: Autonomous Driving / 3D Vision  
**Keywords**: Point cloud compression, Real-time encoding/decoding, Sparse occupancy codes, LiDAR, Neural encoder

## TL;DR

RENO proposes Sparse Occupancy Codes and a one-time inference strategy, achieving the first real-time neural compression of 3D LiDAR point clouds (10fps@14-bit). With a model size of only 1MB, it outperforms the G-PCC standard by 12.25% in bitrate savings.

## Background & Motivation

LiDAR point clouds are widely used in autonomous driving, robotics, and 3D mapping. Real-time compression (10Hz matching the LiDAR acquisition frequency) is a critical demand for industrial applications. Existing methods face a dilemma:

- Traditional methods like G-PCC offer good rate-distortion performance but are not real-time (encoding takes about 1 second per frame); Draco is fast but has low compression efficiency.
- Learning-based methods (such as Unicorn) show excellent rate-distortion performance but still require about 2 seconds to encode one frame on an RTX 3090.

The bottleneck of existing neural encoders lies in two stages: (1) The pre-processing stage requires constructing an octree structure, which is significantly time-consuming; (2) The neural inference stage adopts multi-stage processing, requiring inference on $8 \times N_d$ upsampled voxels, which incurs enormous computational overhead.

Core Problem: **How to design a neural LiDAR compressor that is both real-time and efficient?**

## Method

### Overall Architecture

RENO is based on multi-scale sparse tensor representation, skipping the time-consuming octree construction. It formalizes point cloud geometry compression as scale-by-scale compression of a sparse occupancy code sequence $\mathcal{O} = (O^1, O^2, \ldots, O^{D-1})$. Fast Occupancy Generator (FOG) and Fast Coordinate Generator (FCG) are used to realize encoding/decoding mapping, and Target Occupancy Predictor (TOP) models cross-scale context for entropy coding.

### Key Designs

**1. Sparse Occupancy Codes**

- **Function**: Encode point cloud geometry into a sequence of discrete occupancy codes, transforming the compression problem into sequence compression.
- **Mechanism**: Use fixed-weight sparse convolution (kernel size 2, stride 2, weights $[1,2,4,8,16,32,64,128]$) to directly generate occupancy codes $o_i^{d-1} \in [1, 255]$ in sparse space, eliminating the need to construct an octree. The entire point cloud can be losslessly reconstructed from the initial state $(C^0, O^0)$ and the occupancy code sequence $\mathcal{O}$.
- **Design Motivation**: Octree occupancy symbols and occupancy codes in sparse tensors carry the same categorical values (1-255). However, sparse occupancy codes are unordered rather than tree-ordered, allowing them to be efficiently generated via parallel sparse convolutions, thereby avoiding the complexity of tree structure construction.

**2. Target Occupancy Predictor (TOP) + Target Embedding**

- **Function**: Leverage low-scale prior information to estimate the probability distribution of occupancy codes at the current scale for entropy coding.
- **Mechanism**: $P_\theta(O^d) = \text{TOP}(C^{d-1}, O^{d-1}, C^d)$. First, extract low-scale features $F^{d-1}$ via Embedding+ResNet, then duplicate features from the positions of $C^{d-1}$ to $C^d$ using Target Embedding (Feature Replication + Octant Position Infusion), and finally predict 255-dimensional probabilities using MLP+SoftMax.
- **Design Motivation**: Cross-scale correlation is a crucial prior for point cloud compression. Directly replicating features and infusing relative position information (octant) achieves one-stage inference from low to high scales, bypassing the computational bottleneck of multi-stage upsampling.

**3. Bitwise Two-stage Probability Prediction**

- **Function**: Split the 8-bit occupancy code into two 4-bit sub-codes for step-by-step prediction, simultaneously improving compression performance and computational efficiency.
- **Mechanism**: $P_\theta(O^d) = P_\theta(S_2^d | S_1^d) P_\theta(S_1^d)$, where the higher 4 bits are predicted first, followed by the conditional prediction of the lower 4 bits.
- **Design Motivation**: Predicting 4-bit symbols (16 classes) is much easier than 8-bit symbols (255 classes). More importantly, the bandwidth for GPU-to-CPU transfer of the probability tables is reduced by approximately 8 times ($2 \times N \times 16$ vs $N \times 255$), substantially lowering entropy coding latency.

### Loss & Training

Cross-entropy loss: $\mathcal{L} = \sum_{d=1}^{D-1} \mathbb{E}_{O^d \sim P(O^d)} [-\log P_\theta(O^d)]$, which directly optimizes the lossless compression efficiency of the occupancy code sequence.

## Key Experimental Results

### Main Results: BD-BR Gain and Speed Comparison (KITTI Dataset)

| Method | BD-BR D1(%) | 14-bit Encoding Time (s) | 14-bit Decoding Time (s) |
|------|------------|------------------|------------------|
| Draco | baseline(+48.34) | 0.075 | 0.032 |
| G-PCCv23 | baseline(+12.26) | 0.973 | 0.343 |
| RENO | **-12.26 vs G-PCC** | **0.095** | **0.090** |
| Unicorn | SOTA compression | ~2.0 | ~2.0 |

RENO runs in real-time at 10fps, with both encoding and decoding times of approximately 0.1 seconds, which is 10 times faster than G-PCC.

### Ablation Study: Contributions of Components

| Component Variant | BD-BR Change |
|---------|----------|
| Without cross-scale context | +8.5% |
| Without target embedding (input only $C^d$) | +4.2% |
| 8-bit direct prediction vs. 4+4-bit step-by-step | Step-by-step is better and faster |

### Key Findings

- RENO is the **first real-time** neural LiDAR point cloud compressor that simultaneously outperforms the G-PCC standard.
- The model size is only **1MB**, which possesses great practical deployment value.
- It demonstrates a 12.5% BD-BR saving on the Ford dataset as well, showing good generalization.
- In downstream 3D object detection tasks, the point clouds compressed by RENO maintain a detection accuracy close to that of the original data.

## Highlights & Insights

1. **Core Insight of "Skipping Octree"**: Octree occupancy symbols and sparse tensor occupancy codes essentially carry the same information, but the latter can be generated in parallel through fixed-weight convolutions, completely eliminating the bottleneck of tree structure construction.
2. **One-Time Inference Replacing Multi-Stage Processing**: High-scale target positions are directly mapped from low-scale features using Target Embedding, avoiding step-by-step inference on $8 \times N$ upsampled voxels.
3. The bitwise two-stage scheme simultaneously optimizes compression efficiency and the GPU-CPU communication bottleneck, reflecting a deep understanding of system-level optimization.

## Limitations & Future Work

- Current focus is restricted to geometry compression (point positions), leaving attribute compression (such as color and intensity) unaddressed.
- The generalization ability of the model across different LiDAR sensors can be further verified.
- Temporal redundancy (inter-frame compression) is not considered; it can be extended to video point cloud compression in the future.
- Although the design of fixed-weight convolutions is efficient, it lacks adaptability, which may limit the upper bound of compression in certain scenarios.

## Related Work & Insights

- **Relationship with Unicorn**: Both are based on multi-scale sparse tensors, but Unicorn requires multi-stage inference; RENO achieves a 10x speedup through occupancy codes and one-time inference.
- **Relationship with G-PCC/Draco**: RENO is the first neural method to simultaneously surpass the speed and/or compression efficiency of these two traditional standards.
- **Insight**: When designing a real-time system, the bottleneck lies not only in network inference but also in pre-processing and data transmission — requiring global system-level optimization.

## Rating

⭐⭐⭐⭐

Achieving the first real-time neural compression of LiDAR point clouds is a significant engineering milestone. Its 1MB model size and 10fps speed render it highly practical for physical deployment. The core innovations (sparse occupancy codes and one-time inference) are simple yet effective. The slight drawback is that the compression gain still lags behind state-of-the-art learning-based methods, and the application scenarios are limited to geometry compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PSA-SSL: Pose and Size-aware Self-Supervised Learning on LiDAR Point Clouds](psa-ssl_pose_and_size-aware_self-supervised_learning_on_lidar_point_clouds.md)
- [\[CVPR 2025\] WeatherGen: A Unified Diverse Weather Generator for LiDAR Point Clouds via Spider Mamba Diffusion](weathergen_a_unified_diverse_weather_generator_for_lidar_point_clouds_via_spider.md)
- [\[CVPR 2025\] Neural Inverse Rendering from Propagating Light](neural_inverse_rendering_from_propagating_light.md)
- [\[ICLR 2026\] Low-Latency Neural LiDAR Compression with 2D Context Models](../../ICLR2026/autonomous_driving/low-latency_neural_lidar_compression_with_2d_context_models.md)
- [\[CVPR 2025\] Unlocking Generalization Power in LiDAR Point Cloud Registration](unlocking_generalization_power_in_lidar_point_cloud_registration.md)

</div>

<!-- RELATED:END -->
