---
title: >-
  [Paper Note] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation
description: >-
  [ICCV 2025][Video Understanding][Optical Flow Estimation] MEMFOF is the first memory-efficient multi-frame optical flow method. By reducing the correlation volume resolution and introducing a high-resolution training strategy, it achieves state-of-the-art accuracy on Spring, Sintel, and KITTI benchmarks while requiring only 2.09 GB of GPU memory for 1080p inference.
tags:
  - ICCV 2025
  - Video Understanding
  - Optical Flow Estimation
  - Memory Efficiency
  - Multi-Frame Estimation
  - High-Resolution Training
  - RAFT
date: 2026-05-08
content_hash: 43df191e424b744e
---

# MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation

**Conference**: ICCV 2025
**arXiv**: [2506.23151](https://arxiv.org/abs/2506.23151)
**Code**: [https://github.com/msu-video-group/memfof](https://github.com/msu-video-group/memfof)
**Area**: Video Understanding
**Keywords**: Optical Flow Estimation, Memory Efficiency, Multi-Frame Estimation, High-Resolution Training, RAFT

## TL;DR

MEMFOF is the first memory-efficient multi-frame optical flow method. By reducing the correlation volume resolution and introducing a high-resolution training strategy, it achieves state-of-the-art accuracy on Spring, Sintel, and KITTI benchmarks while requiring only 2.09 GB of GPU memory for 1080p inference.

## Background & Motivation

Optical flow estimation is a fundamental low-level vision task with broad applications in video action recognition, object detection, video inpainting, and synthesis. Since RAFT, the paradigm of iterative GRU refinement over all-pairs correlation volumes has become dominant. However, GPU memory consumption scales quadratically with image resolution: RAFT requires approximately 8 GB at FullHD (1920×1080) and exceeds 25 GB at WQHD, severely limiting deployment on consumer-grade GPUs.

Existing work follows two improvement directions: (1) **Memory-efficient methods**—Flow1D decomposes motion into 1D components; SCV employs sparse candidate matching; HCV uses hybrid volumes—but these often sacrifice accuracy. (2) **Multi-frame methods**—VideoFlow, MemFlow, and StreamFlow exploit temporal consistency to handle occlusions, yet none addresses the memory bottleneck at high resolution. For instance, StreamFlow requires 18.97 GB at 1080p, and VideoFlow-MOF runs out of memory entirely.

The root cause is a fundamental tension: **multi-frame methods improve accuracy by leveraging temporal information, but more frames entail more correlation volumes and greater memory overhead**, making the two goals difficult to reconcile at high resolution.

The paper's starting point is to compress the correlation volume resolution at the architecture level (from 1/8 to 1/16) rather than resorting to inference-time downsampling or tiling. This reduces the combined memory of two correlation volumes in a three-frame scheme from 10.4 GB to 0.65 GB. A high-resolution training strategy is then introduced to recover accuracy, achieving a principled balance among memory, accuracy, and speed.

## Method

### Overall Architecture

MEMFOF extends the SEA-RAFT architecture to three-frame input. Given three consecutive frames $I_{t-1}, I_t, I_{t+1}$, the method simultaneously estimates bidirectional optical flows $f_{t \to t-1}$ and $f_{t \to t+1}$. The core pipeline is as follows:

1. **Feature Extraction**: A shared ResNet34 backbone extracts features $F_t, F_{t-1}, F_{t+1}$ from all three frames.
2. **Context Network**: All three frames are passed through a ContextNetwork to produce the initial flow $f^0$, hidden state $h^0$, and context features $g$.
3. **Dual Correlation Volumes**: $C_{t,t-1}$ and $C_{t,t+1}$ are computed.
4. **Iterative Refinement**: Bidirectional flow predictions are progressively refined over $N$ GRU update steps.
5. **Convex Upsampling**: The final flow prediction is upsampled to the input resolution.

During video sequence processing, feature maps and correlation volumes can be reused across frames, yielding additional computational savings.

### Key Designs

1. **Reduced Correlation Volume Resolution (1/8 → 1/16)**:

    - **Function**: Reduces the working resolution from the standard 1/8 to 1/16, shrinking the correlation volume to 1/16 of its original size.
    - **Mechanism**: An additional strided convolution is appended to the ResNet34 backbone of SEA-RAFT to downsample the 1/8 feature maps to 1/16. Concurrently, the feature dimension $D_f$ is increased from 256 to 1024 and the update module dimension $D_c$ from 128 to 512 to compensate for the information loss from spatial downsampling.
    - **Key Impact**: The combined memory of two correlation volumes in the three-frame scheme is reduced from 10.4 GB to 0.65 GB; total GPU memory drops from 8.19 GB (SEA-RAFT) to 2.09 GB at 1080p inference.
    - **Design Motivation**: The correlation volume is the central memory bottleneck in RAFT-based methods, with complexity $\mathcal{O}((HW)^2)$. Reducing resolution is the most direct remedy, but requires increased channel capacity to compensate for the loss of spatial information.

2. **High-Resolution Training Strategy (FullHD-centric)**:

    - **Function**: Upsamples standard datasets (e.g., Things, Sintel, KITTI) by 2× during training and uses near-FullHD crop sizes (e.g., 864×1920).
    - **Mechanism**: Standard optical flow datasets have relatively low resolution (e.g., Sintel at approximately 436×1024). Training at this scale and inferring at FullHD introduces an underfitting effect, particularly for large-motion regions. Training on 2× upsampled data with large crops enables the model to better learn large-motion patterns at high resolution.
    - **Training Schedule (Multi-stage)**: TartanAir (2×, crop 480×960) → Things (2×, crop 864×1920) → TSKH mixed (2×, crop 864×1920) → benchmark-specific fine-tuning (Spring uses original 1080p).
    - **Design Motivation**: Ablation experiments (Table 4) show that 2× full-image upsampling training reduces the 1px error in large-motion regions (s40+) from 33.9% to 28.5% and EPE from 0.430 to 0.341 compared to training at the original scale.

3. **GMA Global Motion Attention and Adaptive Scaling**:

    - **Function**: Reintroduces the GMA module to enhance motion consistency and modifies the attention scaling factor to accommodate varying resolutions.
    - **Mechanism**: The attention scaling factor is changed from $1/\sqrt{D_c}$ to $\log_3(HW)/\sqrt{D_c}$, enabling more stable behavior across different resolutions.
    - **Design Motivation**: Inspired by MemFlow, this modification allows the attention mechanism to adapt dynamically to resolution changes.

### Loss & Training

- **Mixture-of-Laplace (MoL) Loss**: Inherited from SEA-RAFT, replacing L1 loss. A weighted sum is computed over $T$ flow frame predictions and $N$ iterative refinement steps: $\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \sum_{k=0}^{N} \gamma^{N-k} \mathcal{L}_{MoL}^{t,k}$, where $\gamma=0.85$.
- FlyingChairs (two-frame only) is skipped; TartanAir serves as the pretraining starting point.
- Training uses 32 A100 GPUs with mixed-precision, taking 3–4 days in total.
- During the Spring fine-tuning stage, the original 1× resolution (1080×1920) is used, consuming 28.5 GB per GPU.

## Key Experimental Results

### Main Results

**Spring Benchmark (1080p High Resolution)**:

| Method | #Frames | Memory (GB) | 1px↓ | EPE↓ | WAUC↑ |
|--------|---------|-------------|------|------|-------|
| RAFT | 2 | 7.97 | 6.790 | 1.476 | 90.92 |
| SEA-RAFT(M) ft | 2 | 8.19 | 3.686 | 0.363 | 94.53 |
| StreamFlow ft | 4 | 18.97 | 4.152 | 0.467 | 94.40 |
| MemFlow ft | 3 | 8.08 | 4.482 | 0.471 | 93.86 |
| **MEMFOF ft** | **3** | **2.09** | **3.289** | **0.355** | **95.19** |
| MEMFOF (zero-shot) | 3 | 2.09 | 3.600 | 0.432 | 94.48 |

**Sintel & KITTI**:

| Method | Sintel Clean EPE↓ | Sintel Final EPE↓ | KITTI Fl-all↓ |
|--------|-------------------|-------------------|---------------|
| VideoFlow-MOF | 0.991 | 1.649 | 3.65 |
| StreamFlow | 1.041 | 1.874 | 4.24 |
| **MEMFOF** | **0.963** | 1.907 | **2.94** |

### Ablation Study

| Configuration | EPE↓ | 1px (s40+)↓ | WAUC↑ | Notes |
|---------------|------|------------|-------|-------|
| Bi, 1x, half-res inference | 0.402 | 35.4% | 93.84 | Baseline: original-scale training + half-resolution inference |
| Bi, 1x, full-res inference | 0.430 | 33.9% | 94.23 | Original-scale training + full-resolution inference |
| Bi, 2x, crop, full-res inference | 0.378 | 31.9% | 94.19 | 2× upsampling + crop training |
| **Bi, 2x, full, full-res inference** | **0.341** | **28.5%** | **94.52** | 2× full-image upsampling training (best) |
| Uni, 2x, full, full-res inference | 0.423 | 34.8% | 93.62 | Unidirectional flow, significantly worse than bidirectional |

### Key Findings

- MEMFOF's zero-shot result (3.600 1px) outperforms the fine-tuned SEA-RAFT(M) result (3.686 1px), demonstrating the generalization capability of high-resolution training.
- At only 2.09 GB, MEMFOF requires roughly 1/9 the memory of StreamFlow (18.97 GB), making FullHD optical flow estimation feasible on consumer-grade GPUs.
- Bidirectional flow improves EPE by approximately 20% over unidirectional flow, validating the value of multi-frame temporal information.
- High-resolution training yields the most pronounced improvements in large-motion regions (s40+): 1px error drops from 33.9% to 28.5%.

## Highlights & Insights

1. **Replacing "high spatial resolution + narrow channels" with "coarser spatial resolution + wider channels"** is a highly practical trade-off under memory constraints. Although 1/16 resolution appears aggressive, the combination of 4× channel compensation and high-resolution training results in improved rather than degraded accuracy.
2. The high-resolution training strategy addresses a long-overlooked domain gap: the mismatch between low-resolution dataset training and high-resolution inference, particularly the underfitting of large-motion regions.
3. Feature map and correlation volume reuse during video sequence processing further improves efficiency in practical batch-processing scenarios.

## Limitations & Future Work

- The three-frame design limits the depth of temporal information utilization; longer-range temporal memory mechanisms warrant future exploration.
- The 1/16 resolution may impair the model's ability to resolve small objects or subtle motions; this failure mode is not sufficiently discussed in the paper.
- The memory optimization primarily targets the correlation volume; as resolution decreases, the relative memory footprint of the backbone and context network becomes more significant and requires further optimization.

## Related Work & Insights

- Three key techniques from SEA-RAFT (MoL loss, direct initial flow regression, rigid-flow pretraining) are inherited and extended to the three-frame setting.
- The resolution reduction approach of Flow1D and MemFlow is systematically adopted and generalized in this work.
- For other vision tasks requiring high-resolution inference (e.g., depth estimation, video super-resolution), the proposed strategy of "upsampled training + lower-resolution intermediate representations" offers a valuable reference.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] AllTracker: Efficient Dense Point Tracking at High Resolution](alltracker_efficient_dense_point_tracking_at_high_resolution.md)
- [\[ICCV 2025\] Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs](q-frame_query-aware_frame_selection_and_multi-resolution_adaptation_for_video-ll.md)
- [\[ICCV 2025\] Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)
- [\[ICCV 2025\] FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases](flowseek_optical_flow_made_easier_with_depth_foundation_models_and_motion_bases.md)
- [\[ICCV 2025\] PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View](prior-flow_enhancing_primitive_panoramic_optical_flow_with_orthogonal_view.md)

<!-- RELATED:END -->
