---
title: >-
  [Paper Note] Online Dense Point Tracking with Streaming Memory
description: >-
  [ICCV 2025][Video Understanding][Dense point tracking] This paper proposes SPOT, a framework for online dense long-range point tracking via a customized memory readout module, sensory memory…
tags:
  - "ICCV 2025"
  - "Video Understanding"
  - "Dense point tracking"
  - "streaming memory"
  - "optical flow"
  - "online processing"
  - "visibility estimation"
date: 2026-05-08
content_hash: 5d01e5b7d322f259
---

# Online Dense Point Tracking with Streaming Memory

**Conference**: ICCV 2025
**arXiv**: [2503.06471](https://arxiv.org/abs/2503.06471)
**Code**: [Project Page](https://dqiaole.github.io/SPOT/)
**Area**: Video Understanding
**Keywords**: Dense point tracking, streaming memory, optical flow, online processing, visibility estimation

## TL;DR

This paper proposes SPOT, a framework for online dense long-range point tracking via a customized memory readout module, sensory memory, and visibility-guided splatting. SPOT achieves state-of-the-art performance on the CVO benchmark with 10× fewer parameters and 2× faster speed, while matching or surpassing offline methods on multiple sparse tracking benchmarks.

## Background & Motivation

Dense point tracking requires continuously tracking the position of every point in the initial frame throughout an entire video, even under occlusion. This is fundamentally equivalent to long-range optical flow estimation. Existing methods face three key challenges:

**Appearance drift in optical flow methods**: Conventional methods directly regress long-range optical flow using models trained on adjacent frames, without considering temporal consistency, making them prone to appearance drift. Chaining approaches have inherent limitations: forward accumulation handles occlusion poorly, while backward accumulation leads to processing time that grows linearly.

**Inefficiency of sliding-window methods**: Recent point tracking methods such as CoTracker and SpatialTracker rely on sliding windows for indirect information propagation—transmitting information from the first frame to the current frame step by step through windows—which is both slow and insufficiently effective for long-range tracking. They also require offline processing (relying on future frames to improve accuracy).

**Computational bottleneck in dense tracking**: Although Online TAPIR supports online tracking, per-pixel tracking is extremely time-consuming. DOT combines optical flow and sparse tracking but inherits the drawbacks of both and has a large parameter count.

**Core Problem**: Can dense point tracking achieve both high accuracy and high efficiency using only past observations?

**Mechanism**: The challenging task of long-range information propagation is decomposed into two simpler sub-steps — ① splat first-frame features to recent-frame positions via splatting (using already-predicted accurate long-range optical flow), and ② retrieve relevant information from recent-frame features via attention to augment the current frame (appearance drift is small across recent frames, making similarity reliable).

## Method

### Overall Architecture

SPOT pipeline:
1. Extract 4× downsampled features of the current frame
2. Augment current-frame features using the memory bank via a memory readout module
3. Predict long-range optical flow and visibility masks using a standard optical flow decoder (RAFT)
4. Update the memory bank by splatting first-frame features to current-frame coordinates via visibility-guided splatting

### Key Designs

1. **Feature Augmentation via Memory Readout**

   The memory bank stores key-value pairs, where keys are query features of recent frames and values are first-frame features forwarded via splatting. The current-frame features serve as queries to retrieve from memory via attention:

    $\mathbf{M}_t = \text{Softmax}(\frac{1}{\sqrt{D_k}} \times q \times k^T) \times v$

   **Key design — fusion layer**: Directly using the readout features $\mathbf{M}_t$ for optical flow regression fails because the splatted values contain abundant **hole artifacts** (caused by disocclusion). A simple fusion layer — a single convolution — is introduced to "fill in" the holes using original features:

    $\mathbf{E}_t = \mathbf{F}_t + \text{Conv}(\mathbf{F}_t \oplus \mathbf{M}_t)$

   where $\oplus$ denotes concatenation. This residual design ensures that even when memory features contain artifacts, the original features serve as a reliable fallback.

2. **Optical Flow Decoding and Sensory Memory**

   The RAFT decoder architecture is adopted to compute a 4D correlation volume between augmented features $\mathbf{E}_t$ and reference features $\mathbf{F}_1$. A GRU unit iteratively updates optical flow and visibility:

    $\Delta f_{1\to t}^i, \Delta v_{1\to t}^i, h_t^i = \text{GRU}(h_t^{i-1}, f_c, f_m^i, s_{t-1})$

   **Sensory memory $s_{t-1}$** captures short-term motion dynamics and is updated via an additional GRU:

    $s_t = \text{GRU}_{sensory}(s_{t-1}, f_m^N)$

   **Design motivation**: The correlation volume of the optical flow decoder only captures static spatial similarity; short-term motion trends require additional modeling. Sensory memory enables the model to perceive "in which direction an object is moving," thereby assisting long-range optical flow prediction.

3. **Visibility-Guided Splatting**

   Memory readout relies on feature similarity, which is only reliable within recent frames (due to appearance drift). Therefore, it is necessary to **continuously transfer discriminative first-frame features to the coordinates of the latest frame**.

   Forward warping (splatting) is used for efficient long-range information propagation:

    $F_t^{\Sigma}[(x_t, y_t)] = \sum_{(x_1, y_1)} b(\Delta) \cdot \mathbf{F}_1[(x_1, y_1)]$

   **Visibility guidance**: Splatting in occluded regions produces inconsistent artifacts; weighted normalization using the predicted visibility mask is applied:

    $F_t^{1\to t} = \frac{\sum^{\to}(\mathbf{v}_{1\to t} \cdot \mathbf{F}_1, \mathbf{f}_{1\to t})}{\sum^{\to}(\mathbf{v}_{1\to t}, \mathbf{f}_{1\to t})}$

   The memory bank maintains two FIFO queues (length $L=3$), caching splatting results of recent frames as values and query features of the current frame as keys.

### Loss & Training

- Optical flow prediction: L1 loss
- Visibility prediction: binary cross-entropy loss

**Warm-Start Strategy**: Information from the previous frame is used to initialize the estimate for the current frame. The GRU hidden state is set as $h_t^0 = h_{t-1}^N$, and the optical flow is initialized via one-step extrapolation: $\mathbf{f}_{1\to t}^0 = \mathbf{f}_{1\to t-1}^0 + 2 \times (\mathbf{f}_{1\to t-1}^N - \mathbf{f}_{1\to t-1}^0)$.

Training proceeds in two stages:
- Pre-training on Kubric-CVO for 500K steps (384×384)
- Fine-tuning on Kubric-MOVi-F for 100K steps (24 frames, 384×384)

At inference, the GRU iterates $N=16$ times.

## Key Experimental Results

### Main Results

CVO long-range optical flow (EPE↓, lower is better):

| Method | Mode | Clean | Final | Extended |
|--------|------|-------|-------|----------|
| RAFT | Online | 2.82 | 2.88 | 28.6 |
| MFT | Online | 2.91 | 3.16 | 21.4 |
| DOT† | Online | 1.92 | 1.98 | 12.1 |
| DOT | Offline | 1.34 | 1.37 | 5.12 |
| CoTracker2 | Offline | 1.50 | 1.47 | 5.45 |
| **SPOT** | **Online** | **1.11** | **1.23** | **4.77** |

SPOT achieves the lowest EPE across all splits. On the Extended split, EPE decreases by 60.5% compared to online DOT† (12.08→4.77), **even surpassing all offline methods**.

TAP-Vid sparse point tracking (AJ↑):

| Method | Mode | DAVIS(First) | RGB-S.(First) | Kinetics(First) |
|--------|------|--------------|---------------|-----------------|
| Online TAPIR | Online | 56.2 | 65.9 | 49.6 |
| DOT† | Online | 53.3 | 61.3 | 45.3 |
| CoTracker2 | Offline | 60.8 | 60.5 | 48.4 |
| **SPOT** | **Online** | **61.5** | **73.3** | **50.2** |

### Ablation Study

Module ablation (CVO Extended, trained on 10 frames):

| Configuration | EPE↓ (all/vis/occ) | OA↑ | Notes |
|---------------|---------------------|-----|-------|
| Full | 6.42/3.86/9.98 | 88.5 | Complete model |
| - Feature fusion | NaN | NaN | Training collapse due to hole artifacts |
| - Memory bank | 38.98/28.34/57.15 | 78.8 | Degrades to plain optical flow model |
| - Sensory memory | 8.64/4.55/14.60 | 88.2 | Short-term dynamics modeling matters |
| - Query projector | 6.48/3.88/9.82 | 88.3 | Minor impact |

Splatting type comparison:

| Splatting | EPE↓ | Notes |
|-----------|------|-------|
| **Linear** | **6.42** | Best |
| Softmax | 7.04 | Second best |
| Summation | 7.17 | — |
| Average | 7.34 | Worst |

Warm-start ablation:

| Configuration | EPE↓ | Notes |
|---------------|------|-------|
| Full (warm-start) | 6.42 | — |
| - Hidden state | 7.94 | Hidden state inheritance yields 23.7% improvement |
| - Flow init | 6.49 | Flow initialization provides marginal benefit |

Training video length: 7 frames → 10 frames → 24 frames, EPE improves from 7.14 → 6.42 → 4.77; longer training videos yield consistent gains.

### Key Findings

- **Memory-driven design is central**: Removing the memory bank causes EPE to surge from 6.42 to 38.98, demonstrating that memory is critical for long-range tracking.
- **Fusion layer is indispensable**: Without the fusion layer, the model immediately produces NaN — hole artifacts from splatting are a severe issue.
- **Extremely parameter-efficient**: Only 8.7M parameters, 3–6× smaller than Online TAPIR (29.3M) and DOT (56.5M).
- **Speed advantage**: 12.4 FPS on 512×512 video (H100), faster than all sparse tracking methods and faster than DOT.
- **Causal processing**: Entirely based on past frames, with no dependence on future frames, suitable for real-time online deployment.

## Highlights & Insights

- **Decomposing long-range propagation into two steps**: "Accurate splatting to recent frames + attention from recent to current" is far more efficient and accurate than "direct propagation to the current frame."
- **Unified architecture**: Using the standard RAFT decoder, optical flow and point tracking share the same architecture, enabling future optical flow architecture improvements to be directly leveraged.
- **Elegant design of sensory memory**: Short-term motion trends are modeled via an additional GRU rather than by placing all frames into a window, substantially reducing computational overhead.
- **Dual role of visibility masks**: Used both for evaluating tracking quality and for occlusion handling in splatting.

## Limitations & Future Work

- The memory bank length is fixed at $L=3$; longer or adaptive memory may further improve performance on long videos.
- Only supports "track all points from the first frame"; mid-video query points are not supported.
- Self-supervised fine-tuning on real videos (e.g., BootsTAP) has not been explored; incorporating it may yield further gains.
- The method relies on the RAFT architecture; more advanced optical flow decoders (e.g., SKFlow) may bring additional improvements.

## Related Work & Insights

- MemFlow's short-term memory design inspired this work, though MemFlow only models motion between adjacent frames; this paper extends the idea to long-range tracking.
- Memory mechanisms from VOS (Video Object Segmentation) are informative for pixel-level tracking, though VOS operates at the object level.
- The visibility-guided splatting idea is generalizable to other forward warping scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ Streaming memory design is novel; the two-step propagation idea is elegant
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Full coverage of CVO + TAP-Vid + RoboTAP with detailed ablations
- Writing Quality: ⭐⭐⭐⭐ Problem statement is clear; comparisons are comprehensive
- Value: ⭐⭐⭐⭐⭐ 10× smaller parameter count + 2× faster speed + SOTA accuracy; significant both in engineering and academic terms

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](alltracker_efficient_dense_point_tracking_at_high_resolution.md)
- [\[ICCV 2025\] VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges](videollamb_long_streaming_video_understanding_with_recurrent_memory_bridges.md)
- [\[ICCV 2025\] Hierarchical Event Memory for Accurate and Low-latency Online Video Temporal Grounding](hierarchical_event_memory_for_accurate_and_low-latency_online_video_temporal_gro.md)
- [\[NeurIPS 2025\] Fixed-Point RNNs: Interpolating from Diagonal to Dense](../../NeurIPS2025/video_understanding/fixed-point_rnns_interpolating_from_diagonal_to_dense.md)
- [\[NeurIPS 2025\] LiveStar: Live Streaming Assistant for Real-World Online Video Understanding](../../NeurIPS2025/video_understanding/livestar_live_streaming_assistant_for_real-world_online_video_understanding.md)

</div>

<!-- RELATED:END -->
