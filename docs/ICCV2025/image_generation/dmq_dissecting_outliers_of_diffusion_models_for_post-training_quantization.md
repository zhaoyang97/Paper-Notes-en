---
title: >-
  [Paper Note] DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization
description: >-
  [ICCV 2025][Image Generation][Post-training quantization] DMQ is a framework that combines Learned Equivalent Scaling (LES) and channel-wise Power-of-Two Scaling (PTS) to address outlier problems in diffusion model quantization, achieving, for the first time, stable high-quality image generation under the W4A6 low-bit setting.
tags:
  - ICCV 2025
  - Image Generation
  - Post-training quantization
  - diffusion model quantization
  - outlier handling
  - equivalent scaling
  - Power-of-Two scaling
date: 2026-05-08
content_hash: 7da2558abc87e765
---

# DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization

**Conference**: ICCV 2025
**arXiv**: [2507.12933](https://arxiv.org/abs/2507.12933)
**Code**: [GitHub](https://github.com/LeeDongYeun/dmq)
**Area**: Diffusion Models / Image Generation
**Keywords**: Post-training quantization, diffusion model quantization, outlier handling, equivalent scaling, Power-of-Two scaling

## TL;DR

DMQ is a framework that combines Learned Equivalent Scaling (LES) and channel-wise Power-of-Two Scaling (PTS) to address outlier problems in diffusion model quantization, achieving, for the first time, stable high-quality image generation under the W4A6 low-bit setting.

## Background & Motivation

Despite remarkable achievements in image generation, diffusion models face significant deployment challenges in resource-constrained environments due to the heavy computational overhead of iterative denoising. Quantization is a key technique for reducing computation and memory requirements, yet quantizing diffusion models poses unique challenges:

**Dynamically changing activation distributions**: Since diffusion models share parameters across timesteps, activation distributions vary drastically over time, making it difficult to represent them accurately with fixed quantization parameters.

**Accumulated quantization error**: Quantization errors introduced at each denoising step accumulate and amplify in subsequent steps, with errors at early timesteps having the greatest impact.

**Channel-wise outlier problem**: Certain channels—particularly in skip connection layers—exhibit extreme outliers that stretch the quantization range, severely degrading the quantization precision of non-outlier channels.

Existing PTQ methods (e.g., Q-Diffusion, TFMQ-DM) focus primarily on calibration data composition and timestep adaptation, but **neglect the outlier problem**. Directly applying SmoothQuant, which succeeds in LLM quantization, is also infeasible—because activations in diffusion models are far larger than weights, the large scaling factors produced by SmoothQuant severely amplify weight quantization errors (as shown in Table 1, FID rises from 36 to 454).

## Method

### Overall Architecture

DMQ unifies two key techniques: (1) LES, which fine-grainedly adjusts outlier distributions across all layers to balance quantization difficulty; and (2) PTS, which directly eliminates extreme outliers in specific layers (e.g., skip connections) using power-of-two scaling factors. The two components work in concert to enable accurate quantization under low-bit constraints.

### Key Designs

1. **Learned Equivalent Scaling (LES)**:

    - Function: Learns channel-wise scaling factors $\tau \in \mathbb{R}^{C_{in}}$ to redistribute quantization difficulty between weights and activations.
    - Mechanism: Introduces a scaling transform into matrix multiplication:
    $Y = (X/\tau)(\tau^T \odot W) = \hat{X}\hat{W}$
      and optimizes $\tau$ to minimize quantization error:
    $\mathcal{L}_i = \|X_i W - Q(\hat{X}_i)Q(\hat{W})\|^2$
    - Design Motivation: Unlike SmoothQuant, which relies on heuristic rules, LES learns optimal scaling factors via gradient optimization, avoiding the excessively large scaling factors that arise when activations greatly exceed weights in diffusion models. $\tau$ is subsequently absorbed into the quantization scale, incurring no inference overhead.

2. **Adaptive Timestep Weighting**:

    - Function: Applies adaptive weighting to the loss function based on timestep, prioritizing optimization at critical timesteps.
    - Mechanism: The weighting factor is defined as:
    $\lambda_{t_i} = \left(1 - \frac{\Lambda_{t_i}}{\sum_{t' \in T}\Lambda_{t'}}\right)^\alpha$
      where $\Lambda_t$ is a moving average of cumulative loss (momentum $\xi=0.95$).
    - Design Motivation: Analysis reveals a key tension—quantization errors are larger at later timesteps, yet small errors at early timesteps have a greater impact on final quality due to error accumulation. Simple monotonic weighting underperforms uniform weighting, since error trends across layers vary across timesteps. The adaptive strategy, inspired by Focal Loss, dynamically balances error magnitude and error impact.

3. **Channel-wise Power-of-Two Scaling (PTS)**:

    - Function: Applies $2^\delta$ scaling factors to activation channels with high inter-layer variance, directly eliminating extreme outliers.
    - Mechanism: The activation quantization formula is modified to:
    $\tilde{X} = \text{clamp}\left(\lfloor \frac{X}{2^\delta \odot s^{(X)}} \rceil, l, u\right)$
      Multiplying by the PTS factor is equivalent to a bit-shift operation on the weights: $\tilde{W}_{kj} \ll \delta_k$, enabling efficient hardware implementation.
    - Design Motivation: While LES alleviates quantization difficulty by redistributing outliers, it cannot eliminate them. PTS directly scales outlier channels using power-of-two factors, implemented via bit shifts at minimal computational cost. A Voting Algorithm robustly selects scaling factors from small calibration sets to prevent overfitting.

### Loss & Training

- Scaling factors are optimized layer-by-layer, following the layer-wise optimization strategy of AdaRound.
- After learning the scaling factors, BRECQ is applied for weight quantization.
- PTS is selectively applied only to layers with high inter-layer variance (e.g., skip connections), avoiding global overhead.
- The consensus threshold $\kappa$ in the Voting Algorithm controls the conservatism of the scaling.

## Key Experimental Results

### Main Results

Unconditional image generation (FFHQ 256×256, LDM-4):

| Method | Bits (W/A) | FID↓ | sFID↓ |
|--------|-----------|------|-------|
| Full Precision | 32/32 | 31.34 | 25.88 |
| Q-Diffusion | 4/8 | 36.17 | 28.75 |
| TFMQ-DM | 4/8 | 36.08 | 33.06 |
| **DMQ (Ours)** | **4/8** | **30.37** | **22.72** |
| Q-Diffusion | 4/6 | 71.16 | 75.70 |
| TFMQ-DM | 4/6 | 29.76 | 27.07 |
| **DMQ (Ours)** | **4/6** | **26.38** | **20.01** |

Class-conditional generation (ImageNet 256×256, LDM-4):

| Method | Bits (W/A) | IS↑ | FID↓ | sFID↓ | LPIPS↓ |
|--------|-----------|-----|------|-------|--------|
| Full Precision | 32/32 | 366.8 | 11.34 | 7.81 | — |
| TFMQ-DM | 4/8 | 342.1 | 9.51 | 8.10 | 0.181 |
| **DMQ (Ours)** | **4/8** | **350.8** | **9.68** | **7.19** | **0.124** |
| TFMQ-DM | 4/6 | 225.6 | 9.61 | 10.19 | 0.336 |
| **DMQ (Ours)** | **4/6** | **320.6** | **7.81** | **7.26** | **0.194** |

### Ablation Study

Incremental component contribution (FFHQ W4A8):

| Method | FID↓ | sFID↓ | Note |
|--------|------|-------|------|
| Full Precision | 31.34 | 25.88 | Upper bound |
| Baseline | 36.08 | 33.06 | MinMax quantization |
| +LES | 33.46 | 26.29 | Equivalent scaling redistributes outliers |
| +Timestep Weighting | 31.83 | 24.39 | Adaptive timestep weighting |
| **+PTS** | **30.37** | **22.72** | Power-of-Two scaling |

Voting Algorithm vs. MSE-based selection (FFHQ W4A8):

| PTS Factor Selection | Applied To | FID↓ | sFID↓ |
|---------------------|-----------|------|-------|
| MSE-based | All layers | 33.87 | 25.40 |
| MSE-based | Skip layers | 32.35 | 25.07 |
| **Voting Algorithm** | **Skip layers** | **30.37** | **22.72** |

### Key Findings

- W4A6 is a particularly difficult regime for diffusion model quantization, where existing methods typically fail (FID spikes dramatically); DMQ achieves stable quantization in this setting for the first time.
- Although quantization errors at early timesteps are smaller, their impact on final quality is disproportionately large due to error accumulation—simply prioritizing early timesteps is therefore suboptimal.
- Inter-layer variance in skip connection layers is far higher than in other layers, making them the primary quantization bottleneck.
- The Voting Algorithm is more robust than direct MSE optimization, since MSE tends to overfit on small calibration sets.

## Highlights & Insights

- **Complementary design of LES and PTS**: LES performs fine-grained adjustment (floating-point scaling factors, applied globally), while PTS provides coarser but effective outlier removal (power-of-two factors, applied only to extreme layers), with a clear division of responsibility.
- **Adaptive timestep weighting**: Rather than a simple early-step priority strategy, adaptive weighting based on dynamically accumulated losses accounts for the heterogeneous error behavior of different layers across timesteps.
- **Voting Algorithm**: Statistical consensus cleverly replaces direct optimization for selecting discrete scaling factors, resolving the small-sample overfitting problem.
- **Strong practicality**: $\tau$ is absorbed into quantization scales with no additional inference overhead, and the bit-shift implementation of PTS is hardware-friendly.

## Limitations & Future Work

- Validation is conducted primarily on UNet-based architectures (Stable Diffusion); applicability to DiT architectures is discussed but experimentally limited.
- PTS is applied only to skip connection layers; other layers containing outliers may also benefit.
- The effect of calibration set size on the Voting Algorithm warrants further investigation.
- Integration with QAT methods to achieve quantization at even lower bit-widths (e.g., W3A4) is a promising direction.

## Related Work & Insights

- The success of SmoothQuant in LLM quantization inspired the equivalent scaling approach, but important adaptations are required to address diffusion model characteristics (activations far exceeding weights, iterative error accumulation).
- Unlike DiT quantization methods such as ViDiT-Q, LES learns **static** scaling factors that can be absorbed into weights, rather than timestep-varying factors.
- Takeaway: Diffusion model quantization must simultaneously consider the spatial dimension (which layers/channels contain outliers) and the temporal dimension (which timesteps contribute most critically to accumulated error).

## Rating

- Novelty: ⭐⭐⭐⭐ The LES+PTS combination is novel, and the adaptive timestep weighting has theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple datasets, architectures, and bit-width settings, with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Thorough analysis with a clear logical progression from problem identification to proposed solution.
- Value: ⭐⭐⭐⭐⭐ Achieving stable W4A6 quantization for the first time carries significant practical implications for diffusion model deployment.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution](../../AAAI2026/image_generation/quantvsr_low-bit_post-training_quantization_for_real-world_video_super-resolutio.md)
- [\[ICCV 2025\] Text Embedding Knows How to Quantize Text-Guided Diffusion Models](text_embedding_knows_how_to_quantize_text-guided_diffusion_models.md)
- [\[CVPR 2026\] SegQuant: A Semantics-Aware and Generalizable Quantization Framework for Diffusion Models](../../CVPR2026/image_generation/segquant_a_semantics-aware_and_generalizable_quantization_framework_for_diffusio.md)
- [\[ICCV 2025\] Improved Noise Schedule for Diffusion Training](improved_noise_schedule_for_diffusion_training.md)
- [\[ICCV 2025\] IntroStyle: Training-Free Introspective Style Attribution using Diffusion Features](introstyle_training-free_introspective_style_attribution_using_diffusion_feature.md)

<!-- RELATED:END -->
