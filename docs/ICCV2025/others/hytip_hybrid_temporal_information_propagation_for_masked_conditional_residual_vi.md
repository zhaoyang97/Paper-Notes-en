---
title: >-
  [Paper Note] HyTIP: Hybrid Temporal Information Propagation for Masked Conditional Residual Video Coding
description: >-
  [ICCV 2025][Video Compression] This paper proposes HyTIP, a framework that unifies output-recurrence (explicit buffering of decoded frames) and hidden-to-hidden propagation (implicit buffering of latent features) within…
tags:
  - "ICCV 2025"
  - "Video Compression"
  - "Temporal Information Propagation"
  - "Hybrid Buffer Strategy"
  - "RNN"
  - "Conditional Residual Coding"
date: 2026-05-08
content_hash: 56315ae7af34c7b8
---

# HyTIP: Hybrid Temporal Information Propagation for Masked Conditional Residual Video Coding

**Conference**: ICCV 2025
**arXiv**: [2508.02072](https://arxiv.org/abs/2508.02072)  
**Code**: [https://github.com/NYCU-MAPL/HyTIP](https://github.com/NYCU-MAPL/HyTIP)  
**Area**: Video Coding / Learned Video Compression
**Keywords**: Video Compression, Temporal Information Propagation, Hybrid Buffer Strategy, RNN, Conditional Residual Coding

## TL;DR

This paper proposes HyTIP, a framework that unifies output-recurrence (explicit buffering of decoded frames) and hidden-to-hidden propagation (implicit buffering of latent features) within a single learned video coding framework, achieving comparable coding performance to state-of-the-art methods using only 14% of their buffer size.

## Background & Motivation

Learned video codecs can be interpreted as recurrent neural networks (RNNs) operating along the temporal dimension, with two dominant paradigms for temporal information propagation:

**Output-recurrence**: Explicitly buffers the previously decoded frame as a reference. Conceptually straightforward, but the decoded frame must simultaneously satisfy two objectives — approximating the input frame (reconstruction quality) and serving as a sufficient summary of past information (temporal propagation). This dual constraint limits rate-distortion performance.

**Hidden-to-hidden connections**: Implicitly buffers latent feature representations. More flexible, but typically requires storing full-resolution feature maps with 48+ channels, incurring substantial buffer overhead.

Drawing on RNN theory, the authors argue that RNNs employing both connection types simultaneously exhibit the strongest expressive power, motivating the hybrid approach HyTIP.

## Method

### Overall Architecture

HyTIP is a frame-level temporal prediction coding framework operating as follows:
1. **Motion estimation**: Computes optical flow $f_t$ between the input frame $x_t$ and the reference frame $\hat{x}_{t-1}$
2. **Motion coding**: Encodes and decodes the optical flow to obtain $\hat{f}_t$
3. **Inter-frame coding**: Uses the decoded optical flow and temporal information from the hybrid buffer to generate a pixel-domain predictor $x_c$ and multi-scale feature-domain predictors $\{C_1, C_2, C_3\}$ for conditional residual coding
4. **Temporal buffer update**: Stores current information into the hybrid buffer for use in the next frame

The inter-frame codec employs Masked Conditional Residual Coding: the input signal is $x_t - m \odot x_c$, where $m$ is a pixel-wise soft mask that adaptively switches between conditional coding and conditional residual coding.

### Key Designs

1. **Hybrid buffer strategy (inter-frame coding)**: The hybrid buffer simultaneously stores explicit information (decoded frame $\hat{x}_{t-1}$, 3 channels) and implicit information (a small number of latent features $F_{t-1}$, only 2 channels). The decoded frame serves as the primary temporal information source, while the implicit features provide supplementary information. By exploiting the prior that decoded frames are highly correlated with the current frame, the method substantially reduces reliance on implicit features, requiring only a minimal implicit buffer. Compared to the 48+ channels of purely implicit schemes, the hybrid approach requires only 5 channels (3+2), reducing buffer size by approximately 90%.

2. **Hybrid buffer strategy (motion coding)**: A hybrid strategy is also applied to optical flow coding. Since optical flow signals are typically spatially smooth and temporally slow-varying, explicitly buffered decoded flow $\hat{f}_{t-1}$ (2 channels) supplemented by a small implicit feature $F^f_{t-1}$ (equivalent to 0.125 channels) enables direct temporal prediction without motion compensation.

3. **Masked conditional residual coding**: The input signal $x_t - m \odot x_c = (1-m) \odot x_t + m \odot (x_t - x_c)$ switches pixel-wise between conditional coding and residual coding via the soft mask $m$, addressing the bottleneck of purely conditional coding and the degradation of purely residual coding in occluded regions.

### Loss & Training

- Rate-distortion optimization objective: $R + \lambda D$, where $\lambda$ controls the rate-distortion trade-off
- PSNR-optimized models: $\lambda$ sampled from $[227, 2032]$
- MS-SSIM-optimized models: $\lambda$ sampled from $[7, 46]$
- Two-stage training: initial 5-frame training on Vimeo-90K, followed by 10-frame fine-tuning on BVI-DVC
- Variable-rate model supporting a single model across different bitrates

## Key Experimental Results

### Main Results

| Method | UVG | MCL-JCV | HEVC-B | HEVC-C | HEVC-D | HEVC-E | HEVC-RGB | Avg. BD-rate (%) |
|--------|-----|---------|--------|--------|--------|--------|----------|-----------------|
| VTM 17.0 (anchor) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| HM 16.25 | +26.3 | +36.8 | +31.8 | +29.8 | +29.6 | +32.5 | +34.0 | +31.5 |
| DCVC-TCM | +16.1 | +27.4 | +29.3 | +59.8 | +18.8 | +61.3 | +24.3 | +33.9 |
| DCVC-HEM | -19.2 | -8.5 | -4.4 | +14.9 | -15.0 | +2.5 | -11.0 | -5.8 |
| DCVC-DC | -29.9 | -21.4 | -16.5 | -9.4 | -30.3 | -28.1 | -29.7 | -23.6 |
| DCVC-FM | -23.9 | -13.4 | -10.9 | -5.4 | -26.9 | -29.2 | -19.7 | -18.5 |
| **HyTIP (Ours)** | **-34.7** | **-25.0** | **-23.7** | **-16.4** | **-35.6** | **-21.2** | **-29.0** | **-26.5** |

HyTIP achieves an average BD-rate saving of 26.5% (PSNR-RGB), surpassing VTM 17.0 and comparable to DCVC-DC (−23.6%), while requiring only 7.875 buffer channels versus 55.75 for DCVC-DC.

### Ablation Study

| Motion Buffer | Inter-frame Buffer | # Feature Maps (Explicit+Implicit) | Avg. BD-rate (%) |
|--------------|-------------------|-------------------------------------|-----------------|
| Explicit (2+0) | Explicit (3+0) | 5 | 0 (anchor) |
| Implicit (0+4) | Explicit (3+0) | 7 | -12.2 |
| Hybrid (2+4) | Explicit (3+0) | 9 | -16.0 |
| Hybrid (2+0.125) | Explicit (3+0) | 5.125 | -14.7 |
| Hybrid (2+0.125) | Implicit (0+51) | 53.125 | -19.6 |
| Hybrid (2+0.125) | Implicit (0+5) | 7.125 | -15.0 |
| Hybrid (2+0.125) | Hybrid (3+48) | 53.125 | -21.9 |
| Hybrid (2+0.125) | **Hybrid (3+2)** | **7.125** | **-21.5** |

Key finding: Hybrid(3+2) achieves nearly identical performance to Hybrid(3+48) (−21.5% vs. −21.9%) while reducing buffer size by 87%.

### Key Findings

1. **Implicit buffers are sensitive to buffer size**: Reducing purely implicit schemes from 51 to 5 channels causes a significant performance drop from −19.6% to −15.0%
2. **Hybrid buffers are robust to buffer size**: Reducing the implicit component from 48 to 2 channels in the hybrid scheme causes only a marginal drop from −21.9% to −21.5%
3. **Benefit of longer sequence training**: Training with 10 frames versus 5 frames yields an additional 5.2% BD-rate gain for the hybrid scheme
4. **Controlled complexity**: The hybrid scheme incurs negligible additional model size and computational overhead compared to alternative strategies

## Highlights & Insights

- The paper provides a theoretically grounded and conceptually clear unification of temporal propagation mechanisms in learned video coding through the lens of RNN theory
- The hybrid explicit-implicit design is simple yet effective, leveraging the prior knowledge that decoded frames are highly correlated with the current frame to dramatically reduce implicit feature requirements
- Achieving SOTA-comparable performance with only 14% of the buffer size is highly favorable for hardware deployment, significantly reducing off-chip memory bandwidth demands
- The hybrid strategy is readily transferable to other learned video codecs

## Limitations & Future Work

- Validation is currently limited to RGB-domain coding; extension to YUV coding (the standard operating mode of conventional video codecs) remains unexplored
- Performance on HEVC-E (videoconferencing sequences) is slightly below DCVC-DC, suggesting the hybrid strategy offers limited advantage for scenes with static backgrounds
- Training protocols differ from those of DCVC-DC/DCVC-FM, precluding fully fair comparisons; a unified training setup may yield further improvements
- B-frame structures (bidirectional prediction) are not explored; only IPPP structures are supported

## Related Work & Insights

- **DCVC series (DCVC-TCM/HEM/DC/FM)**: Prev. SOTA methods relying on purely implicit buffers with high buffer overhead
- **MaskCRT**: Purely explicit buffering with masked conditional residual coding; this work extends it by incorporating implicit buffering
- **Traditional codecs (HM/VTM)**: Typically buffer 4 reference frames; the hybrid scheme's buffer footprint is comparable
- **RNN theory (Hammer, 2000)**: RNNs with both output-recurrence and hidden-to-hidden connections possess the strongest expressive capacity

## Rating

- **Novelty**: ⭐⭐⭐⭐ Analyzing temporal propagation in video coding through an RNN perspective is a novel starting point; the hybrid strategy is conceptually simple yet empirically effective
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 7 benchmark datasets with comprehensive ablations across buffer strategies, buffer sizes, sequence lengths, and computational complexity
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured presentation with apt RNN analogies and informative figures
- **Value**: ⭐⭐⭐⭐ Provides important guidance for buffer design in learned video coding and substantially reduces memory bandwidth requirements for practical deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sheaf Cohomology of Linear Predictive Coding Networks](../../NeurIPS2025/others/sheaf_cohomology_of_linear_predictive_coding_networks.md)
- [\[NeurIPS 2025\] InFlux: A Benchmark for Self-Calibration of Dynamic Intrinsics of Video Cameras](../../NeurIPS2025/others/influx_a_benchmark_for_self-calibration_of_dynamic_intrinsics_of_video_cameras.md)
- [\[AAAI 2026\] Expressive Temporal Specifications for Reward Monitoring](../../AAAI2026/others/expressive_temporal_specifications_for_reward_monitoring.md)
- [\[NeurIPS 2025\] Hybrid-Balance GFlowNet for Solving Vehicle Routing Problems](../../NeurIPS2025/others/hybrid-balance_gflownet_for_solving_vehicle_routing_problems.md)
- [\[AAAI 2026\] HybriDLA: Hybrid Generation for Document Layout Analysis](../../AAAI2026/others/hybridla_hybrid_generation_for_document_layout_analysis.md)

</div>

<!-- RELATED:END -->
