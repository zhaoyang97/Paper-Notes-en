---
title: >-
  [Paper Note] LiVOS: Light Video Object Segmentation with Gated Linear Matching
description: >-
  [CVPR 2025][Segmentation][video object segmentation] Proposed LiVOS—the first lightweight VOS network to replace softmax attention with gated linear attention for memory matching. It compresses the spatio-temporal attention matrix into a constant-sized 2D state matrix, achieving constant memory consumption for videos of arbitrary length and supporting 4096p inference on a 32G consumer GPU.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "video object segmentation"
  - "linear attention"
  - "gated linear matching"
  - "memory network"
  - "4096p inference"
date: 2026-05-08
content_hash: d159f12a44b79789
---

# LiVOS: Light Video Object Segmentation with Gated Linear Matching

**Conference**: CVPR 2025  
**arXiv**: [2411.02818](https://arxiv.org/abs/2411.02818)  
**Code**: [uncbiag/LiVOS](https://github.com/uncbiag/LiVOS)  
**Area**: Image Segmentation  
**Keywords**: video object segmentation, linear attention, gated linear matching, memory network, 4096p inference

## TL;DR

Proposed LiVOS—the first lightweight VOS network to replace softmax attention with gated linear attention for memory matching. It compresses the spatio-temporal attention matrix into a constant-sized 2D state matrix, achieving constant memory consumption for videos of arbitrary length and supporting 4096p inference on a 32G consumer GPU.

## Background & Motivation

**Background**: Semi-supervised VOS is primarily driven by Space-Time Memory (STM) networks, which perform pixel-level matching between the query frame and all memory frames via softmax attention. Representative methods include XMem and Cutie.

**Limitations of Prior Work**: Softmax matching requires storing an $\mathcal{O}(HW \times THW)$ attention matrix, with spatial complexity growing linearly with video length and quadratically with resolution. As videos become longer or resolutions increase, this leads to slow computation or out-of-memory (OOM) errors.

**Key Challenge**: Fixed-size memory banks fail under occlusions or rapid motion, while downscaling resolutions leads to a loss of fine-grained mask details; both are inherent limitations of softmax matching.

**Key Insight**: This work identifies softmax matching as the core bottleneck and fundamentally replaces the matching mechanism instead of applying superficial patches.

**Core Idea**: Reformulating softmax attention into a recurrent form of linear attention, where the attention matrix degenerates into a constant-sized 2D state $\mathbf{S}_t \in \mathbb{R}^{C_k \times C_v}$, and introducing a data-dependent gating matrix to enhance selectivity.

## Method

### Overall Architecture

1. Image encoder (ResNet-50) extracts the key of the query frame.
2. Mask encoder (ResNet-18) extracts the value of the memory frame.
3. **Gated Linear Matching (Core)**: Replaces full softmax matching with recurrent updates of a constant-sized state matrix.
4. Integrates sensory memory (low-level information) and object memory (high-level semantics) to enhance the readout.
5. Lightweight mask decoder outputs the segmentation results.

### Key Designs

**1. Linear Matching: From Softmax to Recurrent States**
- **Function**: Reformulate the softmax matching $\mathbf{V}_{t+1} = \text{Softmax}(\mathbf{K}_{t+1}\mathbf{K}_{1:t}^T)\mathbf{V}_{1:t}$ into a kernel function approximation $\phi(\mathbf{K}_{t+1})\mathbf{S}_t$.
- **Mechanism**: Utilizing the associative property of matrix multiplication, $\sum_i \phi(\mathbf{K}_{t+1})\phi(\mathbf{K}_i)^T\mathbf{V}_i$ is regrouped into $\phi(\mathbf{K}_{t+1}) \cdot \sum_i \phi(\mathbf{K}_i)^T\mathbf{V}_i$. The state is defined as $\mathbf{S}_t = \mathbf{S}_{t-1} + \phi(\mathbf{K}_i)^T\mathbf{V}_i$, where $\mathbf{S}_t \in \mathbb{R}^{C_k \times C_v}$ is of constant size. The kernel function $\phi$ uses row-wise softmax.
- **Design Motivation**: The state $\mathbf{S}_t$ is a spatio-temporally independent 2D matrix, whose size depends solely on the feature dimensions ($64 \times 256$), making it independent of video length and resolution.

**2. Gated Linear Matching**
- **Function**: Introduce a data-dependent forget gate $\mathbf{G}_t$ in the state update to selectively retain or discard historical information.
- **Mechanism**: $\mathbf{S}_t = \mathbf{G}_t \odot \mathbf{S}_{t-1} + \phi(\mathbf{K}_i)^T\mathbf{V}_i$. The gate $\mathbf{G}_t = \alpha_t \mathbf{1}^T$ is implemented via low-rank parameterization, where $\alpha_t \in (0,1)^{C_k}$ is extracted from image encoder features using depthwise convolution + spatial pooling + Sigmoid.
- **Design Motivation**: Pure linear matching lacks a selection mechanism, leading to performance degradation in long sequences. Gating provides a forgetting ability similar to GRU/LSTM, enabling active discard of outdated information in scenarios like scene transitions and occlusions.

**3. External Memory Fusion**
- **Function**: Reuse Cutie's sensory memory (element-wise addition fusion of low-level temporal features) and object memory (cross-attention fusion of high-level object semantics).
- **Mechanism**: The readout output from linear matching sequentially interacts with sensory memory and the object transformer to compensate for the information lost through constant-sized state compression.
- **Design Motivation**: Since the constant-sized state compresses spatio-temporal information, external memories provide complementary high-frequency and semantic information.

### Loss & Training

- Cross entropy + soft dice loss combined with equal weights.
- AdamW optimizer, initial learning rate $10^{-4}$, batch size 16, weight decay 0.001.
- 8 frames per batch, cropped to 480x480, trained for 125K iterations.
- Image encoder learning rate scaled by 0.1 to mitigate overfitting, gradient clipping $\tau=3$.
- Point-based supervision (12544 points), following the training strategy of Cutie.

## Key Experimental Results

### Main Results

| Method | STM? | MOSE J&F↑ | DAVIS-17 val J&F↑ | DAVIS-17 test J&F↑ | YouTube-VOS 𝒢↑ |
|---|---|---|---|---|---|
| RDE | ✗ | 46.8 | 84.2 | 77.4 | 81.9 |
| Cutie-small† (1 frame)| ✗ | 49.3 | 76.4 | 71.6 | 79.0 |
| **LiVOS (Ours)** | ✗ | **64.8** | **85.1** | - | - |
| Cutie-small | ✓ | 62.2 | 87.2 | 84.1 | 86.2 |
| Cutie-base | ✓ | 64.0 | 88.8 | 84.2 | 86.1 |
| XMem | ✓ | 56.3 | 86.2 | 81.0 | 85.5 |

### Efficiency Comparison

| Metric | LiVOS vs STM Methods |
|---|---|
| GPU Memory Savings | **53%** |
| Memory Growth on Long Videos | Constant (vs linear growth for softmax) |
| Memory Growth with Resolution | Linear (vs quadratic for softmax) |
| Max Inferable Resolution | **4096p** (32G GPU) |
| CPU Latency vs Frame Count | Constant (vs linear growth for softmax) |

### Key Findings

1. **LiVOS outperforms all non-STM methods and bridges the gap with STM methods**: MOSE 64.8 vs Cutie-small 62.2, DAVIS 85.1 vs 87.2.
2. **Matches STM methods' performance in long video and high-resolution scenarios** while saving 53% GPU memory.
3. **4096p inference becomes feasible**: STM methods suffer from OOM at high resolutions due to softmax attention, whereas the constant state of LiVOS enables processing on consumer-grade GPUs.
4. **The gating mechanism significantly improves long-sequence performance**: In challenging scenarios like scene changes and occlusions, the gated state effectively forgets outdated information.

## Highlights & Insights

- Extending the softmax-to-linear attention reformulation from text/image classification to VOS—a video memory-intensive task—serves as a strong template.
- The insight of the constant-sized state matrix is elegant: $C_k \times C_v = 64 \times 256$ is sufficient to compress all spatio-temporal information of an arbitrarily long video.
- The low-rank parameterized design of the gated linear matching is simple and efficient.
- It paves the way for the development of foundation models for long-duration, high-resolution videos.

## Limitations & Future Work

- The constant state incurs information compression loss, resulting in a performance gap on standard short videos.
- The gating parameterization adopts the simplest low-rank form, and richer parameterizations can be explored.
- It is not optimized for training on high-resolution videos (trained only on 480p); 4096p is zero-shot generalization at test time.
- In multi-object scenarios, a separate state is maintained for each object, still incurring overhead when the number of objects is very large.
- Integration with hardware optimizations such as Flash Attention is not explored.

## Related Work & Insights

- Cutie enhances XMem's readout quality via the object transformer; LiVOS builds on this by replacing the core matching mechanism.
- Gated Linear Attention (GLA) has proven effective in language modeling, and this work extends it to visual memory matching.
- Insight: The idea of constant-state compression can be applied to other video tasks requiring temporal memory (e.g., video QA, video generation).

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to apply linear attention to VOS memory matching, with a rationally designed gating mechanism.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers MOSE/DAVIS/YouTube-VOS/LVOS, including efficiency and high-resolution experiments.
- **Writing Quality**: ⭐⭐⭐⭐ The derivation process from softmax to linear to gated linear is clear and smooth.
- **Value**: ⭐⭐⭐⭐⭐ Resolves the core scalability bottleneck in VOS, opening a new paradigm for high-resolution long video processing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] M3-VOS: Multi-Phase, Multi-Transition, and Multi-Scenery Video Object Segmentation](m3-vos_multi-phase_multi-transition_and_multi-scenery_video_object_segmentation.md)
- [\[CVPR 2026\] ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark](../../CVPR2026/segmentation/elvis_enhance_low-light_for_video_instance_segmentation_in_the_dark.md)
- [\[ICCV 2025\] O-MaMa: Learning Object Mask Matching between Egocentric and Exocentric Views](../../ICCV2025/segmentation/o-mama_learning_object_mask_matching_between_egocentric_and_exocentric_views.md)
- [\[ICCV 2025\] MOVE: Motion-Guided Few-Shot Video Object Segmentation](../../ICCV2025/segmentation/move_motion-guided_few-shot_video_object_segmentation.md)
- [\[ICLR 2026\] Deforming Videos to Masks: Flow Matching for Referring Video Segmentation](../../ICLR2026/segmentation/deforming_videos_to_masks_flow_matching_for_referring_video_segmentation.md)

</div>

<!-- RELATED:END -->
