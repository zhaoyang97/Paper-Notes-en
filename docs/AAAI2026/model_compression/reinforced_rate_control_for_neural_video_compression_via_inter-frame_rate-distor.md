---
title: >-
  [Paper Note] Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness
description: >-
  [AAAI 2026][Model Compression][Neural video compression] This paper proposes the first reinforcement learning rate control framework based on Constrained Markov Decision Processes (CMDP), which jointly captures intra-frame content features and inter-frame rate-distortion coupling dependencies via spatiotemporal state modeling, and directly maps these to per-frame coding parameters. The approach reduces the average bitrate error to 1.20% and achieves BD-Rate savings of up to 13.98% across multiple neural video codecs.
tags:
  - AAAI 2026
  - Model Compression
  - Neural video compression
  - rate control
  - reinforcement learning
  - inter-frame dependency
  - Actor-Critic
  - rate-distortion optimization
date: 2026-05-08
content_hash: 886ebd51f014103b
---

# Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness

**Conference**: AAAI 2026
**arXiv**: [2601.19293](https://arxiv.org/abs/2601.19293)
**Authors**: Wuyang Cong, Junqi Shi, Lizhong Wang, Weijing Shi, Ming Lu, Hao Chen, Zhan Ma (Nanjing University)
**Code**: To be released
**Area**: Model Compression
**Keywords**: Neural video compression, rate control, reinforcement learning, inter-frame dependency, Actor-Critic, rate-distortion optimization

## TL;DR

This paper proposes the first reinforcement learning rate control framework based on Constrained Markov Decision Processes (CMDP), which jointly captures intra-frame content features and inter-frame rate-distortion coupling dependencies via spatiotemporal state modeling, and directly maps these to per-frame coding parameters. The approach reduces the average bitrate error to 1.20% and achieves BD-Rate savings of up to 13.98% across multiple neural video codecs.

## Background & Motivation

### State of the Field
Neural video compression (NVC) leverages the nonlinear modeling capacity and end-to-end optimization of deep neural networks, and has surpassed traditional video coding standards (e.g., VVC/H.266) in compression efficiency. However, rate control—maximizing reconstruction quality subject to a target bitrate constraint—remains a critically underexplored practical problem in NVC. Rate control fundamentally requires learning a policy that allocates target bitrates per frame and maps them to optimal coding parameters (e.g., Lagrange multiplier $\lambda$ and resolution scaling factor $m$).

### Limitations of Prior Work
Existing NVC rate control methods follow the paradigm of traditional codecs by adopting window-based (e.g., GOP-level) schemes: the target bitrate is first uniformly distributed across the window, and frame-level bitrate allocation is then handled by rule-based or heuristic strategies. These methods **model only inter-frame distortion dependencies**, assuming inter-frame bitrate dependencies are negligible. However, NVC exhibits complex and tightly coupled rate-distortion dependencies due to its jointly optimized pixel-level, feature-level, and contextual reference information. Under rate control, even minor changes in reference frame coding parameters can cause significant shifts in the rate-distortion behavior of subsequent frames, rendering methods based on static R-D assumptions suboptimal in both bitrate allocation and coding parameter selection.

### Root Cause
Inter-frame dependencies in NVC involve not only distortion propagation but also **bitrate dependency**—the coding parameters of reference frames directly affect temporal context modeling, altering the probability distribution estimates for subsequent frames and thus their actual bitrates. Ignoring this coupling leads to cascading suboptimal parameter decisions. A dynamic policy that jointly models current frame content and reference frame information is needed to optimize per-frame coding decisions from a global perspective.

## Core Problem

Given a video sequence $\mathcal{X} = \{x_1, x_2, \ldots, x_N\}$ and a target bitrate $R_{tar}$, rate control seeks the optimal coding parameter set $\Pi^{(\mathcal{X})} = \{\boldsymbol{a}_1, \ldots, \boldsymbol{a}_N\}$:

$$\Pi^{(x_t)} = \arg\min_{\Pi^{(\mathcal{X})}} \sum_{t=1}^{N} D_t, \quad \text{s.t.} \quad \frac{1}{N}\sum_{t=1}^{N} R_t \leq R_{tar}$$

After introducing a global Lagrange multiplier $\Lambda$, the necessary condition becomes:

$$\sum_{i=t}^{N} \left( \frac{\Lambda}{N} \frac{\partial R_i}{\partial \mathbf{a}_t} - \frac{\partial D_i}{\partial \mathbf{a}_t} \right) = \sum_{i=t}^{N} \left( \left(\frac{\Lambda}{N} - \lambda_i\right) \frac{\partial R_i}{\partial \mathbf{a}_t} \right) = 0$$

This indicates that the optimal coding parameter $\mathbf{a}_t$ must reflect not only the current frame's R-D behavior but also its rate-distortion propagation effects on future frames—an NP-hard global optimization problem.

## Method

### Overall Architecture: Augmented Actor-Critic

Rate control is formulated as a CMDP, and an Actor-Critic framework with three core components is designed:

### 1. Spatiotemporal State Modeling

The state representation must simultaneously encode historical coding reference information and current frame features. Specifically:
- The current frame $x_t$ and reference frame $x_{t-1}$ are concatenated and fed into a cascaded residual network to extract spatiotemporal features
- Intermediate features of $x_{t-1}$ extracted by the codec at multiple resolutions are incorporated to enhance temporal context
- Auxiliary information (target bitrate, historical coding parameters) is normalized, expanded, and embedded via fully connected layers
- Visual embeddings and auxiliary embeddings are combined to form the complete learnable state representation

### 2. Action Decision

Actions are defined as continuous coding parameter pairs $\{\lambda_t, m_t\}$:
- $\lambda_t \in [\lambda_{min}, \lambda_{max}]$: Lagrange multiplier controlling R-D behavior
- $m_t \in [0.5, 1.0]$: downsampling factor adjusting the spatial resolution of the current and reference frames

The policy $\pi_\phi$ is modeled as a Gaussian distribution with mean and variance predicted by the Actor network. Policy entropy regularization is introduced to encourage exploration, and the Actor gradient is:

$$J_\pi(\phi) = \mathbb{E}_{s_t \sim \mathcal{S}, a_t \sim \pi_\phi} \left[ \epsilon \log \pi_\phi(a_t | s_t) - Q_\theta(s_t, a_t) \right]$$

At inference, a greedy strategy selects the highest-likelihood action. When $m_t < 1.0$, the reference frame is resampled to the current resolution to maintain inter-frame prediction consistency, and the output is bicubically upsampled back to the original resolution.

### 3. Reward Shaping

Meaningful metrics in rate control (total distortion, bitrate deviation) are only available after encoding the complete sequence, resulting in sparse rewards. A weighted inner-product reward is designed as:

$$r_t = -\mathbf{w}_t^\top \mathbf{f}_t, \quad \mathbf{f}_t = \begin{pmatrix} D_t \\ \frac{|R_{\text{rem}}|}{R_{\text{tar}}} \end{pmatrix}, \quad \mathbf{w}_t = \begin{pmatrix} \delta_t \\ \eta_t \end{pmatrix}$$

where $R_{\text{rem}}$ denotes the remaining bitrate budget, and $\mathbf{w}_t = (\delta_t, \eta_t)^\top$ balances distortion and bitrate accuracy, updated adaptively every $\mathcal{K}$ steps based on validation feedback. A larger $\eta_t$ is applied to the last frame to enforce strict rate control. A Twin-Critic architecture estimates two independent Q-values and takes their minimum to mitigate overestimation bias, while modeling the full return distribution to enhance robustness.

## Key Experimental Results

### Experimental Setup
- **Codecs**: DVC, DCVC, DCVC-DC, DCVC-RT
- **Datasets**: UVG, MCL-JCV, HEVC Class B/C/D/E
- **GOP sizes**: 32 and 100
- **Baselines**: Zhang et al. (2023), Chen et al. (2023)

### Main Results 1: Rate Control Performance Comparison (GOP=32, $\Delta R$ ↓ / BD-Rate(%) ↓)

| Codec | Method | UVG | HEVC B | HEVC E | Avg. |
|-------|--------|-----|--------|--------|------|
| DCVC | Chen et al. | 1.85 / -14.28 | 1.96 / -12.23 | 1.18 / -15.25 | 1.76 / -12.18 |
| DCVC | **Ours** | **1.80 / -18.24** | **1.15 / -14.83** | **0.99 / -18.76** | **1.48 / -16.49** |
| DCVC-DC | Chen et al. | 1.66 / -10.33 | 1.71 / -11.02 | 1.28 / -13.00 | 1.61 / -10.63 |
| DCVC-DC | **Ours** | **1.45 / -13.84** | **0.98 / -14.82** | **0.85 / -16.70** | **1.13 / -13.98** |
| DCVC-RT | Chen et al. | 1.49 / -5.12 | 1.44 / -5.26 | 1.50 / -4.98 | 1.45 / -4.81 |
| DCVC-RT | **Ours** | **1.18 / -5.84** | **1.16 / -6.00** | **0.96 / -6.17** | **1.15 / -5.50** |

### Main Results 2: Long-GOP Performance Comparison (GOP=100)

| Codec | Method | Avg. $\Delta R$ ↓ | Avg. BD-Rate ↓ |
|-------|--------|-------------------|----------------|
| DCVC-DC | Zhang et al. | 1.85% | -5.41% |
| DCVC-DC | Chen et al. | 1.62% | -10.42% |
| DCVC-DC | **Ours** | **1.09%** | **-13.93%** |
| DCVC-RT | Chen et al. | 1.32% | -5.39% |
| DCVC-RT | **Ours** | **0.98%** | **-6.03%** |

As the underlying codec improves (DVC→DCVC-RT), the BD-Rate gain of Zhang et al. drops sharply to -5.41%, whereas the proposed method maintains a stable gain of -13.93%.

### Main Results 3: Computational Complexity Comparison (based on DCVC-RT)

| Method | Params | KMACs/pxl | Memory (GB) | Enc. FPS | Dec. FPS |
|--------|--------|-----------|-------------|----------|----------|
| Baseline | 66.33M | 421.31 | 2.27 | 102 | 95 |
| Zhang et al. | +2.12M | +6.40 | +1.22 | 68 | — |
| Chen et al. | — | — | — | 54 | 108 |
| **Ours** | **+0.57M** | **+1.60** | **+0.33** | **111** | **109** |

The proposed method incurs minimal overhead (only +0.57M parameters) and actually **improves** encoding/decoding throughput via the downsampling operation.

### Ablation Study: Number of Training Frames

| Training frames | 4 | 8 | 16 | 32 | 64 |
|----------------|---|---|----|----|-----|
| BD-Rate(%) | -8.84 | -11.15 | -15.03 | -16.49 | -16.90 |
| $\Delta R$(%) | 2.48 | 1.82 | 1.67 | 1.48 | 1.43 |

Increasing the number of training frames consistently improves R-D performance and bitrate accuracy, validating the method's effective modeling of inter-frame dependencies; training complexity grows **linearly** with sequence length.

## Highlights & Insights

- **First CMDP formulation for NVC rate control**: Rate control is formalized as a CMDP, with an RL framework directly optimizing per-frame coding parameters, avoiding the suboptimal two-step approach (bitrate allocation followed by parameter mapping) of traditional methods.
- **In-depth analysis of inter-frame rate-distortion coupling**: Through theoretical derivation and experimental validation, the paper reveals the R-D behavioral shifts caused by reference frame coding parameter changes in NVC—a core issue overlooked by prior methods.
- **Extremely low computational overhead**: Only +0.57M parameters and +0.33 GB memory, while even improving encoding/decoding throughput via downsampling, making the method highly practical.
- **Strong generalization**: On unseen content such as 360° video, the method maintains a low bitrate deviation of 3.9%, significantly outperforming Zhang et al.'s 7.6%.
- **Cross-codec consistency**: Consistent performance gains are achieved across four architectures: DVC, DCVC, DCVC-DC, and DCVC-RT.

## Limitations & Future Work

- **Network transmission conditions not considered**: The current approach only optimizes encoder-side rate-distortion performance without integrating network transmission factors such as packet loss and congestion.
- **Information loss from resolution scaling**: When $m_t < 1.0$, bicubic upsampling is used to restore resolution, which may introduce artifacts such as blurring.
- **RL training cost**: Training requires 50 epochs (4 frames) + 250 epochs (32 frames) of pretraining, resulting in a lengthy training process.
- **GOP structure dependency**: Although the method claims independence from GOP structure, experiments are only conducted under the LDP configuration, leaving scenarios such as random access (RA) unverified.
- **Limited action space**: Only $\lambda_t$ and $m_t$ are included; additional coding parameters such as quantization step size and filter strength have not been explored.

## Related Work & Insights

- **Zhang et al. (2023)**: Uses neural networks to predict bitrate allocation and R-λ mapping, but ignores inter-frame bitrate dependencies, leading to a sharp drop in BD-Rate gains on advanced codecs.
- **Chen et al. (2023)**: Models R-λ-m and D-λ-m relationships with hyperbolic functions and iteratively updates them, but requires pre-encoding of equidistant frames for initialization, increasing encoding time.
- **Li et al. (2022)**: The first NVC rate control method, employing a fixed R-D-λ model with limited flexibility.
- **RL-based methods for traditional codecs** (Zhou et al. 2021; Ho et al. 2021): Apply RL to explore improved rules over traditional toolchains, or rely on heuristic searches with multiple pre-encodings; neither is suitable for NVC.
- **DCVC-RT built-in rate control** (Jia et al. 2025): Implicitly allocates bitrates via hierarchical quality training, but lacks explicit rate control capability.

The application of RL to rate control in video coding can be generalized to other sequential decision-making scenarios, such as adaptive streaming. The discovered inter-frame rate-distortion coupling may also apply to other end-to-end video processing tasks (e.g., video super-resolution, video enhancement). The spatiotemporal state modeling approach is transferable to other RL tasks requiring joint consideration of current inputs and historical context.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first systematic analysis of inter-frame rate-distortion coupling in NVC with CMDP formulation; a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four codecs, six datasets, two GOP settings, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Rigorous problem formulation with clear progression from theoretical derivation to experimental validation.
- Value: ⭐⭐⭐⭐ — Provides a practical rate control solution for NVC with minimal computational overhead suitable for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](../../CVPR2026/model_compression/rdvq_differentiable_vq_image_compression.md)
- [\[ICLR 2026\] Cross-Domain Lossy Compression via Rate- and Classification-Constrained Optimal Transport](../../ICLR2026/model_compression/cross_domain_lossy_compression_optimal_transport.md)
- [\[AAAI 2026\] HCF: Hierarchical Cascade Framework for Distributed Multi-Stage Image Compression](hcf_hierarchical_cascade_framework_for_distributed_multi-stage_image_compression.md)
- [\[AAAI 2026\] Failures to Surface Harmful Contents in Video Large Language Models](failures_to_surface_harmful_contents_in_video_large_language_models.md)
- [\[AAAI 2026\] BD-Net: Has Depth-Wise Convolution Ever Been Applied in Binary Neural Networks?](bd-net_has_depth-wise_convolution_ever_been_applied_in_binary_neural_networks.md)

</div>

<!-- RELATED:END -->
