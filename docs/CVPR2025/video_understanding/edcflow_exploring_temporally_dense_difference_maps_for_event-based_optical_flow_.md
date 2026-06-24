---
title: >-
  [Paper Note] EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation
description: >-
  [CVPR 2025][Video Understanding][Event Camera] EDCFlow is proposed to exploit the complementarity between temporally dense feature difference maps and low-resolution cost volumes across adjacent event frames, achieving high-quality and lightweight event-based optical flow estimation at 1/4 resolution.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Event Camera"
  - "Optical Flow Estimation"
  - "Feature Difference"
  - "Cost Volume"
  - "Efficient Inference"
date: 2026-05-08
content_hash: f7d40f35007236a1
---

# EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation

**Conference**: CVPR 2025  
**arXiv**: [2506.03512](https://arxiv.org/abs/2506.03512)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Event Camera, Optical Flow Estimation, Feature Difference, Cost Volume, Efficient Inference

## TL;DR

EDCFlow is proposed to exploit the complementarity between temporally dense feature difference maps and low-resolution cost volumes across adjacent event frames, achieving high-quality and lightweight event-based optical flow estimation at 1/4 resolution.

## Background & Motivation

Event cameras generate asynchronous event streams by detecting brightness changes, offering advantages such as fine-grained temporal resolution, high dynamic range, and the absence of motion blur, making them highly suitable for motion capture. Existing RAFT-based event optical flow methods (e.g., TMA, MultiFlow) encode intermediate motion by constructing temporally dense multi-cost volumes. However, the computational complexity of cost volumes is $\mathcal{O}(TN^2C)$, leading to redundant computations and making scaling to higher resolutions difficult.

The authors observe a natural complementarity between cost volumes and feature differences:
- **Cost Volumes**: Reflect global pixel matching similarity and are robust but computationally expensive and prone to matching ambiguities.
- **Feature Differences**: Capture local motion details and sharp boundaries efficiently ($\mathcal{O}(TNC)$) but are sensitive to noise.

This complementarity inspires the integration of high-resolution (1/4) difference motion features with low-resolution (1/8) correlation motion features, significantly reducing computational overhead while preserving accuracy.

## Method

### Overall Architecture

EDCFlow adopts a RAFT-like iterative framework consisting of three core modules: (1) a feature extraction module to extract dual-resolution features; (2) a motion encoding module to calculate difference motion features at high resolution and retrieve correlation motion features at low resolution, followed by adaptive fusion; (3) a GRU to progressively update residual optical flow. The input event stream is split into $g+1$ temporal windows, each represented as a voxel grid $\mathcal{V}(b,x,y)$.

### Key Designs

1. **Dual-Resolution Feature Extraction**:
    - Function: Provide features at different resolutions for difference and correlation motion.
    - Mechanism: A weight-sharing encoder simultaneously extracts 1/4 resolution features $F_i \in \mathbb{R}^{d \times H/4 \times W/4}$ and 1/8 resolution features $\bar{F_i}$. A single 4D cost volume $C = \bar{F_0}\bar{F_g}/\sqrt{\bar{d}}$ is constructed using the first and last frames at 1/8 resolution.
    - Design Motivation: Estimating optical flow at 1/4 resolution yields more accurate results, but calculating high-resolution cost volumes is prohibitively expensive. Precision and efficiency are balanced by utilizing difference features at high resolution and cost volumes at low resolution.

2. **Multi-scale Temporal Difference Layer**:
    - Function: Capture temporally dense intermediate motion features at high resolution.
    - Mechanism: Linear motion is assumed to warp the target frame features $F_i$ to the reference frame using the current optical flow $\mathbf{f}^{k-1}_{0 \to i} = \frac{i}{g}\mathbf{f}^{k-1}$, followed by calculating multi-scale differences $D_j^s = \tilde{F}^l_{(j+1)*s} - \tilde{F}^f_{j*s}$ with sampling step size $s$. Different step sizes $s=[1,2,5]$ capture fast and slow motion features respectively. Spatiotemporal features are aggregated using Depthwise-Separable 3D Convolution (DW-3DConv) and adaptively fused through an attention module.
    - Design Motivation: Fast-moving objects exhibit large displacements over short intervals (captured by small steps), whereas slow-moving objects show significant variations only over longer periods (captured by large steps). Simple summation loses details, and GRU/concatenation introduces large computational overheads, leading to the adoption of lightweight DW-3DConv + attention fusion.

3. **Attention-based Motion Fusion**:
    - Function: Adaptively fuse difference motion features and correlation motion features.
    - Mechanism: $F_M = \text{Attention}(\text{Concat}(F_D, F_C))$ fuses features via channel attention (SE block), dynamically adjusting the weights of both features under different scenarios.
    - Design Motivation: Difference features excel at capturing local details but are noise-sensitive, while correlation features provide robust long-range matching information. Their relative importance varies across scenarios, necessitating adaptive trade-offs.

### Loss & Training

A standard RAFT-like multi-iteration $L_1$ loss is adopted, applying exponentially increasing weights to predictions across $K$ iterations:

$$\mathcal{L} = \sum_{k=1}^{K} 0.8^{K-k} \|\mathbf{f}^{gt} - \mathbf{f}^k\|_1$$

Training employs the AdamW optimizer with a one-cycle learning rate schedule (maximum learning rate of 0.0002). The model is trained for 100 epochs on DSEC and 10 epochs on MVSEC, with a batch size of 3.

## Key Experimental Results

### Main Results

| Method | EPE↓ | AE↓ | 1PE↓ | Param(M) | MACs(G) | Runtime (ms) |
|------|------|-----|------|----------|---------|-------------|
| E-RAFT | 0.79 | 2.85 | 12.7 | 5.3 | 256 | 102 |
| TMA | 0.74 | 2.68 | 10.9 | 6.9 | 344 | 58 |
| IDNet-4 | 0.72 | 2.72 | 10.1 | 2.5 | 1200 | 120 |
| **EDCFlow** | **0.72** | **2.65** | **10.0** | **2.5** | **247** | **39** |

### Ablation Study

| Configuration | EPE↓ | AE↓ | 1PE↓ | Description |
|------|------|-----|------|------|
| W/o Diff | 0.82 | 2.88 | 13.6 | Remove difference features, EPE drops by 14% |
| W/o Corr | 0.83 | 3.17 | 14.0 | Remove correlation features, EPE drops by 15% |
| W/o MSAttn | 0.74 | 2.68 | 10.6 | Remove multi-scale attention |
| W/o SE | 0.74 | 2.69 | 10.4 | Remove channel attention fusion |
| Full model | 0.72 | 2.65 | 10.0 | All components |

### Key Findings

- The contributions of difference motion features and correlation motion features are almost equal (performance drops by 14% and 15% respectively when removed), validating the necessity of the complementary design.
- In the multi-scale strategy, any single scale ($s=1$ or $s=5$) underperforms compared to the multi-scale combination $s=[1,2,5]$, as motion at different velocities requires different temporal intervals to be captured.
- EDCFlow can serve as a plug-and-play module cascaded after existing RAFT-like methods for high-resolution refinement, delivering a 5.6% EPE improvement when integrated with TMA.
- In cross-dataset generalization experiments (Blinkflow $\to$ DSEC), the proposed method exhibits the smallest accuracy degradation (-0.53 EPE), verifying superior generalization ability.

## Highlights & Insights

- **Clear Logic**: Starting from the complementarity between cost volumes and feature differences, low-cost differences are used to replace expensive multi-cost volumes to encode intermediate motion, presenting a self-consistent logical flow.
- **Significant Efficiency Advantage**: Compared with IDNet-4 which achieves comparable accuracy, the computational cost is only 20% of its size, and the runtime is 68% faster.
- **Plug-and-Play Capability**: As a refinement module, it can directly improve the quality of motion boundaries for existing methods, showing great practical value.
- **Ingenious Multi-Scale Temporal Sampling**: Spans different motion velocities using simple variations in step size.

## Limitations & Future Work

- The linear motion assumption ($\mathbf{f}_{0 \to i} = \frac{i}{g}\mathbf{f}$) may not hold in non-linear motion scenarios.
- Although noise sensitivity of feature differences is mitigated through fusion, it might still act as a bottleneck under extreme noise conditions.
- Validated only on event camera data; extension to optical flow estimation for conventional frame cameras has not yet been explored.

## Related Work & Insights

- The RAFT framework provides a powerful iterative optimization paradigm for optical flow estimation, but its computational bottleneck lies in the cost volume.
- The concept of feature differences relates to classical frame-differencing methods in traditional image processing, but this work elevates it to the deep-learning feature space.
- The multi-scale temporal sampling strategy can be extended to other tasks requiring multi-scale temporal modeling, such as video prediction and action recognition.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of complementary fusion of difference and correlation features is novel but not entirely disruptive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluations on both DSEC and MVSEC datasets, including generalization experiments and rich ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, complete structure, and good alignment between text and figures.
- Value: ⭐⭐⭐⭐ Achieves a better accuracy-efficiency trade-off in event-based optical flow and can serve as a universal refinement module.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DPFlow: Adaptive Optical Flow Estimation with a Dual-Pyramid Framework](dpflow_adaptive_optical_flow_estimation_with_a_dual-pyramid_framework.md)
- [\[CVPR 2026\] From Contrast to Consistency: Rethinking Event-based Continuous-Time Optical Flow Estimation](../../CVPR2026/video_understanding/from_contrast_to_consistency_rethinking_event-based_continuous-time_optical_flow.md)
- [\[ICCV 2025\] Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](../../ICCV2025/video_understanding/unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)
- [\[CVPR 2026\] U2Flow: Uncertainty-Aware Unsupervised Optical Flow Estimation](../../CVPR2026/video_understanding/u2flow_uncertainty_aware_unsupervised_optical_flow_estimation.md)
- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](../../ICCV2025/video_understanding/memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)

</div>

<!-- RELATED:END -->
