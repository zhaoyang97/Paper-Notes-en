---
title: >-
  [Paper Note] FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution
description: >-
  [ICCV 2025][3D Vision][video depth estimation] FlashDepth augments Depth Anything v2 with a Mamba recurrent module for cross-frame scale consistency…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "video depth estimation"
  - "real-time streaming"
  - "2K resolution"
  - "Mamba recurrent network"
  - "hybrid model"
date: 2026-05-08
content_hash: a711f4c87507ce94
---

# FlashDepth: Real-time Streaming Video Depth Estimation at 2K Resolution

**Conference**: ICCV 2025
**arXiv**: [2504.07093](https://arxiv.org/abs/2504.07093)  
**Code**: [GitHub](https://github.com/Eyeline-Research/FlashDepth)  
**Area**: 3D Vision / Video Depth Estimation
**Keywords**: video depth estimation, real-time streaming, 2K resolution, Mamba recurrent network, hybrid model

## TL;DR

FlashDepth augments Depth Anything v2 with a Mamba recurrent module for cross-frame scale consistency, and introduces a Small-Large hybrid architecture that achieves real-time streaming video depth estimation at 2K resolution and 24 FPS with significantly sharper boundaries than existing methods.

## Background & Motivation

- **Three core requirements for video depth estimation**:
  1. Cross-frame consistency and accuracy
  2. High resolution and sharp boundaries
  3. Real-time streaming capability
- **No existing method satisfies all three simultaneously**:
    - Diffusion-based video depth models (DepthCrafter): high accuracy but slow (2.1 FPS), requires 110-frame batches, not real-time
    - Single-frame models (DAv2): high accuracy but inter-frame flickering
    - Early online methods (LSTM / global point clouds): cannot scale to high resolution, low accuracy
    - Video Depth Anything: requires 32-frame batches, limited resolution
    - CUT3R: supports streaming but produces blurry depth maps
- **Key observations**:
    - Depth accuracy (low frequency) is insensitive to resolution — boundaries occupy only ~1% of pixels
    - Boundary sharpness (high frequency), however, is highly resolution-dependent
    - Cross-frame consistency is essentially a scale alignment problem, solvable with a lightweight recurrent network

## Method

### Overall Architecture

Built upon Depth Anything v2 (DAv2) with three key designs:
1. **Mamba recurrent module**: inserted after the DPT decoder and before the final convolutional head to achieve cross-frame scale alignment
2. **Small-Large hybrid model**: a high-resolution lightweight stream + a low-resolution accurate stream + cross-attention fusion
3. **Two-stage training strategy**: consistency learning first (low resolution), then high-resolution fine-tuning

### Key Design 1: Mamba Recurrent Module

**Design considerations**:
- Mamba (rather than LSTM/GRU) is adopted for its ability to selectively retain relevant information over long sequences
- DPT decoder output features are downsampled before processing to reduce computational cost
- Accounts for approximately 1% of total model parameters

**Implementation**:
$$f, H_t = \text{Mamba}(\text{Flatten}(\text{Down}(F)), H_{t-1})$$
$$F_{\text{align}} = F + \text{Up}(\text{UnFlatten}(f))$$

where $F$ is the DPT decoder feature output and $H_t$ is the Mamba hidden state. A residual connection with zero-initialized Mamba output layer ensures the first iteration preserves the original DAv2 depth output.

**Mamba module structure**: 4 blocks, each consisting of LayerNorm + Mamba layer + MLP. Vanilla (unidirectional) Mamba is used rather than bidirectional scanning.

### Key Design 2: Small-Large Hybrid Model

**Motivation**: DAv2-Large runs at only 6 FPS at 2K resolution; DAv2-Small is real-time but suffers a significant accuracy drop.

**Solution**:
- **FlashDepth-S** (primary stream): fine-tuned from DAv2-S, processes full 2K resolution (2044×1148)
- **FlashDepth-L** (auxiliary stream): fine-tuned from DAv2-L, processes low resolution (924×518)

Intermediate features from both streams are fused via cross-attention:
$$F_{\text{fused}_i} = \text{CrossAttn}(Q=F_{S_i}, KV=F_{L_i})$$

The cross-attention module contains 2 transformer blocks with zero initialization to preserve the original capabilities at the start of training. Fusion is applied only to the output features of the first DPT layer to prevent low-resolution artifacts from propagating into the high-resolution stream.

**Inference optimization**: both streams run in parallel via CUDA Graph, achieving an overall throughput of 24 FPS.

### Two-Stage Training

**Stage 1** (low-resolution consistency training):
- Datasets: MVS-Synth, Spring, TartanAir, PointOdyssey, Dynamic Replica — ~500K image–depth pairs in total
- Resolution: 518×518 random crops
- Loss: simple L1 loss (synthetic metric depth data)
- Temporal modules of FlashDepth-L and FlashDepth-S are trained separately
- Long-video generalization is enhanced by increasing inter-frame step size

**Stage 2** (high-resolution hybrid training):
- Datasets: MVS-Synth + Spring — ~16,000 2K depth annotations
- FlashDepth-L is frozen; only FlashDepth-S and the cross-attention module are fine-tuned
- Learning rates: 1e-4 for Mamba/CrossAttn, 1e-6 for remaining parameters
- Duration: half a day per stage (8×A100)

## Key Experimental Results

### Main Results: Boundary Sharpness and Speed

| Method | Unreal4K F1↑ | UrbanSyn F1↑ | FPS↑ | Resolution | Streaming |
|--------|-------------|-------------|------|------------|-----------|
| DAv2 | 0.058 | 0.118 | 30 | 924×518 | Per-frame |
| DepthCrafter | 0.021 | 0.044 | 2.1 | 1024×576 | ✗ (110 frames) |
| VidDepthAny | 0.049 | 0.097 | 24 | 924×518 | ✗ (32 frames) |
| CUT3R | 0.007 | 0.019 | 14 | 512×288 | ✓ |
| FlashDepth-L | 0.048 | 0.136 | 30 | 924×518 | ✓ |
| FlashDepth-L(high) | 0.143 | 0.271 | 6.0 | 2044×1148 | ✓ |
| **FlashDepth(Full)** | **0.109** | **0.185** | **24** | **2044×1148** | **✓** |

**Key Findings**: FlashDepth at 2K resolution and 24 FPS achieves boundary F1 scores substantially higher than all baselines, confirming the critical role of high-resolution processing for boundary sharpness.

### Depth Accuracy Comparison

| Method | ETH3D δ1↑ | Sintel δ1↑ | Waymo δ1↑ | Unreal4K δ1↑ | UrbanSyn δ1↑ | Avg. δ1 Rank |
|--------|----------|----------|----------|-------------|-------------|-------------|
| DAv2 | 0.633 | 0.561 | 0.897 | 0.379 | 0.622 | 5.4 |
| DepthCrafter | 0.745 | 0.697 | 0.790 | 0.399 | 0.556 | 4.6 |
| VidDepthAny | 0.864 | 0.660 | 0.944 | 0.606 | 0.892 | 1.6 |
| CUT3R | 0.836 | 0.509 | 0.902 | 0.851 | 0.878 | 3.6 |
| FlashDepth-L | 0.875 | 0.642 | 0.924 | 0.566 | 0.882 | 2.2 |
| **FlashDepth(Full)** | **0.848** | **0.642** | **0.916** | **0.545** | **0.862** | **3.4** |

FlashDepth accuracy ranks second only to VidDepthAny (which processes 32-frame batches) and surpasses DepthCrafter (which processes 110-frame batches).

### Ablation Study: Super-Resolution Method Comparison

| Method | Unreal4K δ1↑ | Unreal4K F1↑ | UrbanSyn δ1↑ | UrbanSyn F1↑ | FPS |
|--------|-------------|-------------|-------------|-------------|-----|
| FlashDepth-L (baseline) | 0.566 | 0.048 | 0.882 | 0.136 | 30 |
| + UNet | 0.490 | 0.034 | 0.811 | 0.107 | 24 |
| + DAGF | 0.377 | 0.055 | 0.703 | 0.133 | 7.7 |
| **FlashDepth(Full)** | **0.545** | **0.109** | **0.862** | **0.185** | **24** |

**Key Findings**: Depth super-resolution networks (UNet/DAGF) degrade accuracy or generalization, whereas the hybrid model approach maintains both accuracy and boundary sharpness.

### Ablation: Model Size vs. Hybrid Architecture

| Method | ETH3D δ1↑ | Waymo δ1↑ | UrbanSyn δ1↑ |
|--------|----------|----------|-------------|
| FlashDepth-S (~21M params) | 0.786 | 0.898 | 0.784 |
| FlashDepth-L (~300M params) | 0.875 | 0.924 | 0.882 |
| FlashDepth(Full) | 0.848 | 0.916 | 0.862 |

Directly using the smaller model (S vs. L) incurs a 3–10% accuracy drop; the hybrid model sacrifices only 1–3% while matching the speed of the small model.

## Highlights & Insights

1. **Minimalist yet effective temporal consistency**: the Mamba module accounts for only 1% of parameters and achieves consistency over sequences of up to 1,000 frames without optical flow supervision or temporal losses.
2. **Depth–resolution decoupling insight**: accuracy (low frequency) is insensitive to resolution while boundary sharpness (high frequency) is strongly dependent on it — this observation directly motivates the hybrid architecture.
3. **Circumventing data scarcity**: fine-tuning from pretrained DAv2 requires only modest amounts of synthetic video data, without the need for large-scale high-resolution video depth annotations.
4. **High engineering maturity**: open-sourced code and weights enabling practical 2K 24 FPS inference on an A100, with immediate applicability.

## Limitations & Future Work

- Residual flickering persists, particularly in scenes with rapid motion
- The hybrid architecture requires loading two models simultaneously (DAv2-S + DAv2-L), incurring non-trivial GPU memory overhead
- Stage 2 training uses only ~16,000 high-resolution depth annotations, which may limit generalization
- The method produces relative depth only; metric depth estimation is not supported

## Related Work & Insights

- **vs. Video Depth Anything**: both fine-tune from DAv2 but employ different temporal modules; VidDepthAny requires 32-frame batch processing while FlashDepth supports true streaming inference
- **vs. DepthCrafter**: the video diffusion model achieves superior consistency but at prohibitive computational cost
- **Takeaway**: for video depth estimation, augmenting a high-quality single-frame model with a lightweight temporal module is a more practical path than training video diffusion models from scratch

## Rating ⭐⭐⭐⭐

A highly practical contribution. The method strikes an excellent balance in the accuracy–resolution–speed triangle. The design motivation for the hybrid architecture is clear and well-validated experimentally. The simplicity and effectiveness of the Mamba temporal module are impressive. The industrial background of Netflix Eyeline Studios further attests to the engineering quality of the work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] StreamSTGS: Streaming Spatial and Temporal Gaussian Grids for Real-Time Free-Viewpoint Video](../../AAAI2026/3d_vision/streamstgs_streaming_spatial_and_temporal_gaussian_grids_for_real-time_free-view.md)
- [\[ICCV 2025\] Seeing and Seeing Through the Glass: Real and Synthetic Data for Multi-Layer Depth Estimation](seeing_and_seeing_through_the_glass_real_and_synthetic_data_for_multi-layer_dept.md)
- [\[ICCV 2025\] Amodal Depth Anything: Amodal Depth Estimation in the Wild](amodal_depth_anything_amodal_depth_estimation_in_the_wild.md)
- [\[ICCV 2025\] Radiant Foam: Real-Time Differentiable Ray Tracing](radiant_foam_real-time_differentiable_ray_tracing.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)

</div>

<!-- RELATED:END -->
