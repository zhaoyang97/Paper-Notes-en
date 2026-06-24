---
title: >-
  [Paper Note] DPFlow: Adaptive Optical Flow Estimation with a Dual-Pyramid Framework
description: >-
  [CVPR 2025][Video Understanding][Optical Flow Estimation] This paper proposes DPFlow, a dual-pyramid recurrent encoder combining an **image pyramid** and a **feature pyramid** with a fully-convolutional **Cross-Gated Unit (CGU)**. Trained only on standard resolutions, DPFlow adaptively generalizes to 8K resolution inputs, achieving state-of-the-art (SOTA) performance on Sintel, KITTI, and Spring benchmarks. Furthermore, the paper introduces **Kubric-NK**…
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Optical Flow Estimation"
  - "High-Resolution Generalization"
  - "Dual-Pyramid Encoder"
  - "Recurrent Network"
  - "Cross-Gated Unit"
date: 2026-05-08
content_hash: e6a08d85aeb1ef7f
---

# DPFlow: Adaptive Optical Flow Estimation with a Dual-Pyramid Framework

**Conference**: CVPR 2025  
**arXiv**: [2503.14880](https://arxiv.org/abs/2503.14880)  
**Code**: [https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/dpflow](https://github.com/hmorimitsu/ptlflow/tree/main/ptlflow/models/dpflow)  
**Area**: Video Understanding  
**Keywords**: Optical Flow Estimation, High-Resolution Generalization, Dual-Pyramid Encoder, Recurrent Network, Cross-Gated Unit

## TL;DR

This paper proposes DPFlow, a dual-pyramid recurrent encoder combining an **image pyramid** and a **feature pyramid** with a fully-convolutional **Cross-Gated Unit (CGU)**. Trained only on standard resolutions, DPFlow adaptively generalizes to 8K resolution inputs, achieving state-of-the-art (SOTA) performance on Sintel, KITTI, and Spring benchmarks. Furthermore, the paper introduces **Kubric-NK**, a multi-resolution optical flow evaluation dataset, supporting quantitative high-resolution evaluation for the first time.

## Background & Motivation

Optical flow estimation provides pixel-level motion information for video processing, serving as the foundation for tasks such as video restoration, action recognition, and video compression. As video standards have reached 8K resolution, optical flow methods face severe challenges in resolution generalization.

**Limitations of Prior Work**:
- **Global matching- or attention-based methods** (e.g., FlowFormer, GMFlow): The quadratic complexity of attention mechanisms limits the input size. These methods rely on **input tiling** to process high resolutions, which leads to **tiling artifacts** and losses in global context.
- **RAFT-like methods**: The architectures are relatively rigid, showing insufficient generalization ability when resolution changes, with errors increasing significantly at high resolutions.
- **High-resolution training**: Although it alleviates the issue, it is impractical for extremely high resolutions (4K/8K).
- **Lack of high-resolution evaluation benchmarks**: Existing datasets scale up only to 2K. Previous high-resolution experiments only showcase qualitative results (hand-picked examples), preventing reliable comparisons among different methods.

**Key Challenge**:
An adaptive architecture is required that can be trained on standard resolutions and perform inference on arbitrary high resolutions, alongside a benchmark to quantitatively evaluate high-resolution generalization capabilities.

## Method

### Overall Architecture

DPFlow adopts a recurrent encoder-decoder structure. The encoder is a newly proposed **dual-pyramid recurrent encoder**, which extracts multi-scale features by combining an image pyramid and a feature pyramid. The decoder employs a CGU-based GRU for iterative optical flow refinement. The training uses a multi-scale Mixture-of-Laplace loss. During inference, the number of pyramid levels automatically adapts to the input resolution.

### Key Designs

1. **Dual-Pyramid Recurrent Encoder**
    - **Function**: Merges the advantages of image pyramids and feature pyramids to adaptively extract multi-scale features.
    - **Mechanism**:
     - **Feature pyramid path** (forward recurrence): A shared ConvGRU propagates multi-scale information along the scale direction, with the output of each layer processed by a CGU Block.
     - **Image pyramid path**: Each layer extracts features directly from the downsampled raw image to maintain direct access to input information.
     - **Bidirectional recurrence**: A backward ConvGRU is introduced to allow shallow layers to access deep information.
     - The final feature is obtained by concatenating the forward, backward, and image paths: $X_s = \phi^{out}(\text{concat}(X_s^f, X_s^b, X_s^i))$
    - **Design Motivation**: A single feature pyramid may dilute input information at deep levels, whereas a single image pyramid cannot propagate cross-scale information. The dual pyramid combines the advantages of both. Moreover, the shared parameters and recurrent structure of ConvGRU enable the network to adapt to different numbers of levels (i.e., different resolutions).

2. **Cross-Gated Unit (CGU)**
    - **Function**: Replaces the attention mechanism, utilizing a fully-convolutional design to extract discriminative matching features.
    - **Mechanism**: Designed based on Gated-CNN, it includes a self-gate (self-gating for single inputs) and a cross-gate (cross-gating for dual inputs), utilizing $1 \times 1$ convolutions, depthwise separable convolutions, GELU activation, and Layer Scale.
    - **Design Motivation**: The high computational cost of attention mechanisms and their sensitivity to changes in input size (due to the need for positional encodings) are major obstacles for high-resolution generalization. The local operations of CNNs are naturally independent of the input size, thereby circumventing these issues.

3. **Kubric-NK Dataset**
    - **Function**: The first optical flow evaluation benchmark supporting four resolutions from 1K to 8K with dense annotations.
    - **Mechanism**: Based on the Kubric rendering engine, 600 samples across 30 sequences are generated. Each sample is rendered at four resolutions: 1K (960×540), 2K, 4K, and 8K, offering annotations for optical flow, depth, surface normals, object coordinates, etc.
    - **Design Motivation**: To provide a unified quantitative evaluation platform, rendering the same scene at multiple resolutions allows for a precise analysis of how resolution changes affect prediction accuracy.
    - The optical flow magnitude distribution is similar to commonly used datasets like Sintel (at 1K), while the higher-resolution versions provide larger and more challenging motions.

## Key Experimental Results

### Main Results

| Dataset | Metric | DPFlow | Runner-up Method | Gain |
|--------|------|--------|---------|------|
| Sintel (clean+final) | Overall Rank | **1st** | FlowFormer++ | Overall Best |
| KITTI 2015 | Overall Rank | **1st** | MemFlow | Overall Best |
| Spring (test) | 1px↓ | **4.53** | SEA-RAFT (4.85) | ↓6.5% |
| Spring (0-shot) | 1px↓ | **6.79** | RPKNet | ↓10.8% |

### Kubric-NK High-Resolution Evaluation (EPE↓)

| Method | 1K | 2K | 4K | 8K |
|------|-----|-----|-----|-----|
| RAFT | 0.68 | 1.46 | 12.3 | 82.7 |
| FlowFormer++ (tiling) | 0.46 | 1.11 | 5.39 | 24.4 |
| RAPIDFlow | 0.36 | 0.82 | 2.14 | 5.83 |
| **DPFlow** | **0.34** | **0.62** | **1.71** | **4.07** |

- On 8K resolution, DPFlow reduces the error by **30%** compared to RAPIDFlow and by **83%** compared to FlowFormer++.
- Non-adaptive methods such as RAFT show an error explosion of over 20x at 8K, while attention-based methods like GMA even suffer from Out-Of-Memory (OOM) errors.

### Ablation Study

- Dual pyramid vs. feature-only pyramid: Sintel final 1.70 $\to$ 1.84 (+8.2% error)
- Dual pyramid vs. image-only pyramid: Sintel final 1.70 $\to$ 1.79 (+5.3% error)
- Removing bidirectionality (forward-only): Kubric-8K 4.07 $\to$ 5.12 (+25.8% error)
- CGU vs. attention: Lower computational cost and significantly better resolution generalization.

### Key Findings

- The choice of training checkpoint has a massive impact: stage 3 (mixed dataset) is suitable for most high-resolution evaluations.
- The magnitude of optical flow vectors at high resolutions is linearly related to the resolution, which is highly challenging for existing methods to adapt to.
- While input tiling can handle high resolutions, the introduced tiling artifacts become increasingly pronounced as the resolution scales up.

## Highlights & Insights

- **Elegant Design of the Dual Pyramid**: The image pyramid ensures direct access to information, while the feature pyramid propagates multi-scale semantics. Recursively shared parameters enable resolution adaptation, presenting a concise and effective architectural innovation.
- **Community Contribution of Kubric-NK**: Fills the gap in high-resolution optical flow evaluation. The design of rendering the same scene at multiple resolutions makes resolution generalization analysis a first-class citizen.
- **Return of the Pure CNN Route**: At a time when attention mechanisms dominate, CGU demonstrates that a carefully designed CNN still holds unique advantages for tasks requiring resolution generalization.
- The number of pyramid levels is automatically calculated based on the input diagonal size: $N=\text{round}(\log_2(\max(1,\sqrt{W^2+H^2}/\sqrt{960^2+540^2})))+3$

## Limitations & Future Work

- The way the recurrent encoder increases levels (hierarchical parameter sharing) might lead to feature degradation at extremely high resolutions.
- High-resolution evaluation relies solely on synthetic data (Kubric-NK); the quality of real-world 8K optical flow has yet to be verified.
- While the fully-convolutional design favors generalization, it may underperform compared to attention-based methods in scenarios requiring global context (such as large-range occluded regions).

## Related Work & Insights

- **RAFT** pioneered the iterative optical flow estimation paradigm, but its single-scale design limits resolution generalization.
- The recurrent encoders in **RAPIDFlow** and **RPKNet** provided the foundation for adaptive architectures. DPFlow builds upon this by introducing the dual pyramid and bidirectional recurrence.
- The Mixture-of-Laplace loss from **SEA-RAFT** is adopted by DPFlow and extended to multi-scale training.
- Insight: In other tasks requiring resolution adaptation (such as video super-resolution and stereo matching), the dual-pyramid and recurrent design principles are highly worth exploring.

## Rating

⭐⭐⭐⭐ — Architecturally, the recurrent encoder with the dual image-feature pyramid is an elegant design contribution, achieving SOTA on four major benchmarks. The release of the Kubric-NK dataset holds significant value for the community. The 30% error reduction at 8K resolution convincingly demonstrates the generalization capability of the architecture.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation](edcflow_exploring_temporally_dense_difference_maps_for_event-based_optical_flow_.md)
- [\[CVPR 2026\] U2Flow: Uncertainty-Aware Unsupervised Optical Flow Estimation](../../CVPR2026/video_understanding/u2flow_uncertainty_aware_unsupervised_optical_flow_estimation.md)
- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](../../ICCV2025/video_understanding/memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [\[CVPR 2026\] FlowFM: Advancing Dark Optical Flow Estimation with Flow Matching](../../CVPR2026/video_understanding/flowfm_advancing_dark_optical_flow_estimation_with_flow_matching.md)
- [\[CVPR 2026\] Efficient All-Pairs Correlation Volume Sampling for Optical Flow Estimation](../../CVPR2026/video_understanding/efficient_all-pairs_correlation_volume_sampling_for_optical_flow_estimation.md)

</div>

<!-- RELATED:END -->
