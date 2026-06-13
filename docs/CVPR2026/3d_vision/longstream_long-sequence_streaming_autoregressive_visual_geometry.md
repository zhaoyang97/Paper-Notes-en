---
title: >-
  [Paper Note] LongStream: Long-Sequence Streaming Autoregressive Visual Geometry
description: >-
  [CVPR 2026][3D Vision][Streaming 3D reconstruction] LongStream is a gauge-decoupled streaming visual geometry model that achieves stable metric-scale scene reconstruction at 18 FPS over thousand-frame sequences…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "Streaming 3D reconstruction"
  - "autoregressive model"
  - "pose estimation"
  - "KV cache"
  - "long sequence"
date: 2026-05-08
content_hash: c665c553e3e010d8
---

# LongStream: Long-Sequence Streaming Autoregressive Visual Geometry

**Conference**: CVPR 2026
**arXiv**: [2602.13172](https://arxiv.org/abs/2602.13172)  
**Code**: [Project Page](https://3dagentworld.github.io/longstream/)  
**Area**: 3D Vision
**Keywords**: Streaming 3D reconstruction, autoregressive model, pose estimation, KV cache, long sequence

## TL;DR

LongStream is a gauge-decoupled streaming visual geometry model that achieves stable metric-scale scene reconstruction at 18 FPS over thousand-frame sequences, via keyframe-relative pose prediction, orthogonal scale learning, and cache-consistent training.

## Background & Motivation

Long-sequence streaming 3D reconstruction remains a major open challenge in visual geometry. Existing autoregressive streaming models (e.g., Stream3R, StreamVGGT) degrade severely on long sequences:

- **Root Cause — gauge-coupled design**: Existing models anchor poses to the first-frame coordinate system and regress absolute poses. This turns long-sequence prediction into an increasingly difficult extrapolation problem — models are trained on short index ranges but must predict large indices at inference, creating a "train-short, test-long" domain gap.
- **Attention decay**: As sequences grow longer, the attention mechanism becomes overly dependent on first-frame tokens (attention sink), leading to keyframe jumps and accumulated pose errors.
- **Scale drift**: Geometry and metric scale estimation are entangled, causing progressive Sim(3) scale drift.
- **KV cache contamination**: Stale or degraded information accumulated in long-term KV caches further exacerbates geometric drift.

Existing streaming methods collapse within tens of meters, whereas applications such as autonomous driving require kilometer-scale stable reconstruction.

## Method

### Overall Architecture

LongStream adopts a ViT encoder + causal Transformer aggregator + task head architecture. Given streaming input frames, it predicts per frame: keyframe-relative pose $\mathbf{T}_{i \leftarrow k}$, depth map, point map, and global scale factor. Training and inference share identical KV cache layouts to ensure consistency.

### Key Designs

1. **SE(3) Gauge Decoupling — Keyframe-Relative Pose**: Rather than anchoring to a fixed first frame, the model predicts each frame's pose relative to the most recent keyframe:

$$\mathbf{T}_{i \leftarrow k} = \mathbf{T}_i \circ \mathbf{T}_k^{-1}$$

This formulation is invariant under any world-coordinate reparameterization $S \in SE(3)$. The key effect is transforming the long-range extrapolation problem (large index range) into a constant-difficulty local estimation task (bounded index interval $i - k$), while eliminating the inherent bias toward the first frame. The pose head fuses current-frame and keyframe features and performs RAFT-style iterative updates $\mathbf{p}^{(t+1)} = \mathbf{p}^{(t)} + \Delta\mathbf{p}^{(t)}$.

2. **Sim(3) Gauge Decoupling — Orthogonal Scale Learning**: Inspired by scale-invariant (SI-Log) approaches, geometry learning and metric scale estimation are decoupled at both the architectural and objective levels:

    - **Geometry branch**: Optimized in normalized space with loss $\mathcal{L}_{geom} = \|\tilde{X}_{pred} - \tilde{X}_{gt}\|_1$, where $\tilde{X} = X / \text{Norm}(X)$, ensuring $\partial\mathcal{L}/\partial s = 0$.
    - **Scale head**: Independently predicts a global scale factor $s = \exp(\mathbf{w}^\top \mathbf{h}_{scale})$, trained only on metrically calibrated data.
    - Scale affects only translation, depth, and point clouds; rotation and field of view are unaffected, achieving complete decoupling.

3. **Cache-Consistent Training (CCT)**: Addresses attention sink dependency and KV cache contamination:

    - Constant sink tokens are removed during training; pure causal masking with a sliding window is used instead.
    - KV caches are explicitly propagated and pruned between training chunks, ensuring that cache visibility during training exactly matches inference.
    - For very long sequences, **periodic cache refresh** is introduced: sink frames and KV caches are hard-reset every $N$ keyframes (analogous to state marginalization in SLAM). Because the entire model operates in keyframe-relative coordinates, this reset does not break consistency.

### Loss & Training

The joint probabilistic framework maximizes the posterior:

$$\mathcal{L} = \mathcal{L}_{geom} + \mathcal{L}_{depth} + \mathcal{L}_{pose} + \mathcal{L}_{scale}$$

- $\mathcal{L}_{pose}$: L1 loss on rotation (quaternion), translation (normalized space), and focal length offset across iterative updates, with decay weight $\gamma^{t-1}$.
- $\mathcal{L}_{geom}$: Normalized point cloud L1 (scale-invariant).
- $\mathcal{L}_{depth}$: Depth supervision.
- $\mathcal{L}_{scale}$: Log-space metric scale L1 $\|\log\hat{s} - \log s_{gt}\|_1$ (applied only on calibrated data).

## Key Experimental Results

### Main Results

| Dataset | Metric | LongStream | Prev. Best Streaming | Gain |
|--------|------|-----------|-----------------|------|
| KITTI (11 sequences) | Mean ATE↓ | **51.90** | 177.73 (TTT3R) | −70.8% |
| KITTI seq00 (3.7 km) | ATE | **92.55** | 190.93 (TTT3R) | −51.5% |
| KITTI seq04 (0.4 km) | ATE | **1.95** | 11.62 (TTT3R) | −83.2% |
| TUM-RGBD | ATE | Best | Stream3R/StreamVGGT collapse | — |
| Waymo | ATE | Best | Severe drift in prior methods | — |

Note: 18 FPS real-time inference; memory and latency remain stable on long sequences (unlike VGGT et al., which run out of memory).

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| No CCT + causal inference | Strong attention sink, poor accuracy | Train-inference inconsistency causes degradation |
| No CCT + sliding window inference (with sink) | Amplified sink, reduced accuracy | Sliding window amplifies sink bias |
| CCT + causal inference | Sink strongly suppressed, best accuracy | CCT eliminates sink dependency |
| CCT + sliding window inference | Sink similarly suppressed, strong accuracy | CCT effective across all inference modes |

### Key Findings

- Existing streaming methods collapse within tens of meters, while LongStream remains stable on kilometer-scale sequences.
- Attention sink is not a useful feature — it is a byproduct of train-inference inconsistency; its elimination via CCT yields substantial performance gains.
- Keyframe-relative pose formulation converts extrapolation into interpolation, providing the theoretical foundation for long-sequence stability.
- Periodic cache refresh is naturally compatible with keyframe-relative coordinates, requiring no additional alignment.

## Highlights & Insights

- **Gauge decoupling**: The two fundamental degrees of freedom responsible for long-sequence degradation — SE(3) coordinate freedom and Sim(3) scale freedom — are identified at the theoretical level and addressed with elegant, targeted solutions.
- **Reinterpreting attention sink**: The sink is shown not to be an intrinsic requirement of streaming Transformers, but rather a symptom of the train-inference gap. CCT directly eliminates this gap.
- **Periodic cache refresh**: Inspired by state marginalization in SLAM, this mechanism pairs seamlessly with keyframe-relative coordinates.
- **Practical significance**: 18 FPS throughput, kilometer-scale stability, and metric scale make LongStream genuinely deployable in autonomous driving and related applications.

## Limitations & Future Work

- Keyframe selection strategy is not discussed in detail; adaptive keyframe scheduling may be needed for fast motion or texture-poor scenes.
- The scale head is trained only on metrically calibrated data; scale quality on uncalibrated data depends on the training data distribution.
- The value of $N$ for periodic cache refresh may require scene-specific tuning.
- Advantages over baselines are less pronounced on small-scale indoor scenes (e.g., TUM-RGBD) than outdoors.
- Dynamic object handling is not discussed; additional processing may be required for scenes with numerous moving objects.

## Related Work & Insights

- **VGGT/StreamVGGT**: Direct baselines for LongStream, demonstrating that absolute pose regression inevitably fails on long sequences.
- **DUSt3R/MASt3R**: Offline reconstruction methods whose ideas LongStream extends to the streaming setting.
- **CUT3R**: RNN-based streaming reconstruction; LongStream replaces the RNN with a Transformer + KV cache for improved long-range dependency modeling.
- The keyframe-relative pose idea generalizes to other vision tasks requiring long-sequence processing.
- The train-inference consistency principle underlying CCT has reference value for all streaming models using KV caches.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Triple innovations — gauge decoupling, CCT, and periodic cache refresh — with clear theoretical motivation and independent contributions per component.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple indoor and outdoor datasets with comprehensive visualizations, including attention map analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem formulation is precise, theoretical derivations are clear, and figures are highly informative.
- **Value**: ⭐⭐⭐⭐⭐ First system to achieve kilometer-scale real-time streaming reconstruction, with significant implications for practical deployment in autonomous driving and robotics.
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] LONG3R: Long Sequence Streaming 3D Reconstruction](../../ICCV2025/3d_vision/long3r_long_sequence_streaming_3d_reconstruction.md)
- [\[CVPR 2026\] tttLRM: Test-Time Training for Long Context and Autoregressive 3D Reconstruction](tttlrm_test-time_training_for_long_context_and_autoregressive_3d_reconstruction.md)
- [\[CVPR 2026\] STAC: Plug-and-Play Spatio-Temporal Aware Cache Compression for Streaming 3D Reconstruction](stac_plug-and-play_spatio-temporal_aware_cache_compression_for_streaming_3d_reco.md)
- [\[CVPR 2026\] FlashVGGT: Efficient and Scalable Visual Geometry Transformers with Compressed Descriptor Attention](flashvggt_efficient_and_scalable_visual_geometry_transformers_with_compressed_descr.md)
- [\[CVPR 2026\] Long-SCOPE: Fully Sparse Long-Range Cooperative 3D Perception](long_scope_fully_sparse_long_range_cooperative_3d_perception.md)

</div>

<!-- RELATED:END -->
