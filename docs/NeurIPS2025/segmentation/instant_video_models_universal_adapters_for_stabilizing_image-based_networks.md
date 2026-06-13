---
title: >-
  [Paper Note] Instant Video Models: Universal Adapters for Stabilizing Image-Based Networks
description: >-
  [NeurIPS 2025][Segmentation][Temporal Stability] This paper proposes a class of universal Stabilization Adapters that can be inserted into nearly any image model architecture. By freezing the base network and training on…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Temporal Stability"
  - "Stabilization Adapters"
  - "Video Consistency"
  - "Corruption Robustness"
  - "EMA"
date: 2026-05-08
content_hash: 996535bcfe00a9ab
---

# Instant Video Models: Universal Adapters for Stabilizing Image-Based Networks

**Conference**: NeurIPS 2025
**arXiv**: [2512.03014](https://arxiv.org/abs/2512.03014)  
**Code**: None (but method is reproducible)  
**Area**: 3D Vision / Video Processing / Temporal Consistency
**Keywords**: Temporal Stability, Stabilization Adapters, Video Consistency, Corruption Robustness, EMA

## TL;DR

This paper proposes a class of universal Stabilization Adapters that can be inserted into nearly any image model architecture. By freezing the base network and training only the adapter parameters, combined with a unified accuracy–stability–robustness loss function, the method endows frame-level models with video temporal consistency and corruption robustness.

## Background & Motivation

**Background**: Video is commonly processed frame by frame — image datasets are more abundant, image models are cheaper to train, and per-frame performance gains often transfer to video tasks.

**Limitations of Prior Work**:
- Frame-by-frame processing introduces temporal inconsistencies (flickering, abrupt changes), degrading perceptual quality and downstream system reliability.
- Real-world deployment faces transient corruptions (sensor noise, compression artifacts, adverse weather), which simultaneously exacerbate instability and reduce accuracy.
- Existing video models are typically designed for specific tasks and lack generality.

**Key Challenge**: Enhancing temporal stability may lead to over-smoothing, reducing accuracy; a balance between precision and stability is required.

**Goal**: Design a universal, lightweight, and modular approach that enables pretrained image models to achieve temporal stability and corruption robustness during video inference without sacrificing accuracy.

**Key Insight**: Model stability and robustness jointly within a single loss function; theoretically analyze the properties of this loss to avoid "over-smoothing reality"; and design learnable adapters for adaptive stabilization.

**Core Idea**: A controller network predicts element-wise EMA decay rates, allowing the degree of stabilization to adapt to the rate of scene change. It is theoretically shown that when $\lambda < 1/2$, the ground truth is always the global minimum of the loss.

## Method

### Overall Architecture

1. Insert Stabilization Adapters at selected layers and outputs of the pretrained image model.
2. Freeze original model parameters; train only adapter parameters.
3. Train using a unified accuracy–stability–robustness loss.

### Key Designs

1. **Unified Loss Function**: Combines corruption robustness $\mathcal{R}_c$ and corruption stability $\mathcal{S}_c$:
    $\mathcal{U}_c = -(\mathcal{R}_c + \lambda \mathcal{S}_c) = \mathbb{E}\left[\sum_t \delta(f(\varepsilon_t(\bm{x}_t)), \bm{y}_t) + \lambda \sum_{t} \delta(f(\varepsilon_t(\bm{x}_t)), f(\varepsilon_{t+1}(\bm{x}_{t+1})))\right]$
   where $\lambda$ controls the trade-off between stability and accuracy.

2. **Oracle Bound ($\lambda < 1/2$)**: When the distance $\delta$ can be expressed as a norm, as long as $\lambda < 1/2$, the ground truth is always the global minimum of the loss in the prediction space — a perfect model is never incentivized to deviate from correct predictions in exchange for higher stability.

3. **Collapse Bound ($\lambda > \tau - 1$)**: When $\lambda$ exceeds the sequence length minus one, the global minimum of the loss degenerates to "repeating the initial prediction" (prediction collapse). Since $\tau - 1 > 0.5$, the oracle bound and the collapse bound are mutually exclusive.

4. **EMA Stabilizer**: The basic unit is an exponential moving average:
    $\tilde{z}_t = \beta z_t + (1-\beta) \tilde{z}_{t-1}$
   $\beta \in [0,1]$ controls the degree of stabilization ($\beta=1$ means no stabilization).

5. **Stabilization Controller**: Composed of a shared backbone network $g$ and per-layer heads $h_i$, the controller predicts pixel-wise decay rates $\bm{\beta}$ from current/previous frame inputs and features:
    $\tilde{\bm{z}}_{i,t} = \bm{\beta}_{i,t} \odot \bm{z}_{i,t} + (1-\bm{\beta}_{i,t}) \odot \tilde{\bm{z}}_{i,t-1}$
    $\bm{\beta}_{i,t} = \sigma(h_i(g(\bm{x}_t, \bm{x}_{t-1}), \bm{z}_{i,t}, \tilde{\bm{z}}_{i,t-1}, \bm{z}_{i,t-1}))$

6. **Spatial Fusion Extension**: The controller predicts spatial decay kernels $\bm{\eta}$ (instead of a scalar $\beta$), performing weighted aggregation over spatial neighborhoods to allow motion compensation; the maximum trackable motion is determined by the spatial extent of the kernel.

### Design Principles

- **Causality**: Stabilized outputs depend only on current and past inputs (no future frames), making the method suitable for streaming video.
- **Feature-Domain + Output-Domain**: Both intermediate features and final outputs are stabilized, supporting high-level semantic consistency.
- **Non-invasive**: Adapters are modular layers with independent parameters; the original model is not modified.

## Key Experimental Results

### Image Enhancement (HDRNet)

The controller with spatial fusion simultaneously **improves** both PSNR and stability (approximately +2 dB PSNR and 35% reduction in instability), whereas naive EMA can only improve stability at the cost of quality.

### Denoising (NAFNet)

| Noise Level | Method | PSNR | Instability |
|-------------|--------|------|-------------|
| σ=0.1 | Base | Lower | Higher |
| σ=0.1 | **Controlled+Spatial** | **Higher** | **Lower** |

Key finding: Fixed EMA **simultaneously degrades** both PSNR and stability (because the denoising model predicts noise residuals that are entirely uncorrelated across frames — smoothing residuals suppresses denoising); the learned controller automatically avoids destabilizing features.

### Corruption Robustness

| Corruption Type | Model | Enh. PSNR | Enh. Instab. | Den. PSNR | Den. Instab. | Depth AbsRel↓ | Depth Instab. |
|-----------------|-------|-----------|--------------|-----------|--------------|---------------|---------------|
| Patch drop | Base | 17.43 | 164.6 | 18.93 | 151.4 | 0.070 | 9.89 |
| Patch drop | **Ours** | **31.39** | **30.36** | **35.46** | **20.42** | **0.070** | **4.73** |
| JPEG artifacts | Base | 24.85 | 42.06 | 29.01 | 39.71 | 0.057 | 7.32 |
| JPEG artifacts | **Ours** | **26.46** | **23.58** | **32.19** | **20.49** | **0.065** | **4.92** |

Across nearly all corruption types and tasks, the stabilizer substantially reduces instability (typically by 50–80%) while maintaining or improving per-frame accuracy.

### Adverse Weather Robustness

| Stabilizer? | Unfreeze Base? | Rain PSNR | Rain Instab. | Snow PSNR | Snow Instab. |
|-------------|----------------|-----------|--------------|-----------|--------------|
| × | × | 21.43 | 151.76 | 18.62 | 262.48 |
| ✓ | × | 28.63 | 57.88 | 31.34 | 59.31 |
| × | ✓ | 32.19 | 70.84 | 34.33 | 66.57 |
| ✓ | ✓ | **32.61** | **58.30** | **35.20** | **58.98** |

### Ablation Study

| Stabilizer Variant | Characteristics | Effect |
|--------------------|-----------------|--------|
| Output Fixed | Fixed EMA at output layer only | Stability ↑, Accuracy ↓ |
| Simple Fixed | Fixed EMA at all layers | Both accuracy and stability degrade in denoising |
| Simple Learned | Learned per-channel β | Slightly better than fixed |
| **Controlled** | Controller predicts pixel-wise β | Stability ↑, Accuracy ↑ |
| **Spatial** | Controller + spatial fusion | Best performance in most cases |

Prediction collapse is confirmed at $\lambda=8 > \tau-1$ (instability $< 10^{-3}$), empirically validating the theoretical collapse bound.

### Key Findings

- Feature-domain stabilization is critical for high-level tasks (depth estimation, semantic segmentation).
- The core value of the controller lies in adaptively regulating the degree of stabilization based on scene dynamics.
- Stabilization adapters naturally enhance corruption robustness without explicitly modeling corruption types.

## Highlights & Insights

- **Closed Loop Between Theory and Practice**: The oracle bound and collapse bound provide theoretical guidance for choosing $\lambda$, and experiments perfectly validate both.
- **Strong Generality**: The same framework applies to HDRNet (enhancement), NAFNet (denoising), Depth Anything v2 (depth estimation), and DeepLabv3+ (segmentation).
- **Modular Design**: Original model parameters are not modified; adapters are plug-and-play.
- **Causality**: Only current and historical frames are used, making the method suitable for real-time streaming video.
- The approach is conceptually related to Mamba's selective state-space models (Eq. 7 can be viewed as an input-conditioned linear dynamical system).

## Limitations & Future Work

- The theoretical bounds require $\delta$ to be expressible as a norm, which excludes many complex loss functions.
- Depth estimation encounters difficulties in sim-to-real transfer (real videos contain subtle corruptions absent in simulation).
- Spatial fusion exhibits performance degradation on long sequences under extreme noise, partially mitigable by increasing training $\tau$.
- The current formulation uses simple Euclidean norm as $\delta$; exploring more complex metrics such as Wasserstein distance may yield further improvements.
- Computational overhead is not analyzed in detail, particularly the additional cost introduced by the controller backbone.

## Related Work & Insights

- Blind video temporal consistency methods (Bonneel et al. 2015; Lai et al. 2018) operate only in the output space; this work extends stabilization to the feature space.
- Clockwork ConvNets observed that semantic content changes more slowly than pixel values — this paper embodies the same intuition through feature-domain stabilization.
- The approach has direct applicability to any "frame-level model + video deployment" scenario: autonomous driving perception, video editing, AR/VR.
- The adapter training paradigm can be extended to other temporal tasks (e.g., stabilizing frame-level models in audio processing).

## Rating

- Novelty: ⭐⭐⭐⭐ The theoretical analysis of the unified loss is novel and the stabilization controller design is practical, though the EMA concept itself is conventional.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers four tasks (denoising, enhancement, depth, segmentation), multiple corruption types, adverse weather, and thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Theory is clearly presented, experiments are well-organized, and figures are informative.
- Value: ⭐⭐⭐⭐⭐ Addresses a core bottleneck in deploying image models to video; excellent generality and practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unveiling the Spatial-Temporal Effective Receptive Fields of Spiking Neural Networks](unveiling_the_spatial-temporal_effective_receptive_fields_of_spiking_neural_netw.md)
- [\[NeurIPS 2025\] Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2](vanish_into_thin_air_cross-prompt_universal_adversarial_attacks_for_sam2.md)
- [\[ICLR 2026\] VIRTUE: Visual-Interactive Text-Image Universal Embedder](../../ICLR2026/segmentation/virtue_visual-interactive_text-image_universal_embedder.md)
- [\[NeurIPS 2025\] Mechanistic Interpretability of RNNs Emulating Hidden Markov Models](mechanistic_interpretability_of_rnns_emulating_hidden_markov_models.md)
- [\[NeurIPS 2025\] RoMA: Scaling up Mamba-based Foundation Models for Remote Sensing](roma_scaling_up_mamba-based_foundation_models_for_remote_sensing.md)

</div>

<!-- RELATED:END -->
