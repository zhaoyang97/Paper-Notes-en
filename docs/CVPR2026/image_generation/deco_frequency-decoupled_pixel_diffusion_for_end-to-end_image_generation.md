---
title: >-
  [Paper Note] DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation
description: >-
  [CVPR 2026][Image Generation][Pixel diffusion] DeCo proposes a frequency-decoupled pixel diffusion framework that delegates high-frequency detail synthesis to a lightweight pixel decoder while allowing the DiT to focus on low-frequency semantic modeling. Combined with a frequency-aware flow matching loss, it achieves FID 1.62 (256×256) and 2.22 (512×512) on ImageNet, substantially narrowing the gap between pixel diffusion and latent diffusion models.
tags:
  - CVPR 2026
  - Image Generation
  - Pixel diffusion
  - frequency decoupling
  - end-to-end generation
  - Diffusion Transformer
  - frequency-aware loss
date: 2026-05-08
content_hash: e03cf95daa6f1812
---

# DeCo: Frequency-Decoupled Pixel Diffusion for End-to-End Image Generation

**Conference**: CVPR 2026
**arXiv**: [2511.19365](https://arxiv.org/abs/2511.19365)
**Code**: [https://github.com/Zehong-Ma/DeCo](https://github.com/Zehong-Ma/DeCo)
**Area**: Image Generation
**Keywords**: Pixel diffusion, frequency decoupling, end-to-end generation, Diffusion Transformer, frequency-aware loss

## TL;DR
DeCo proposes a frequency-decoupled pixel diffusion framework that delegates high-frequency detail synthesis to a lightweight pixel decoder while allowing the DiT to focus on low-frequency semantic modeling. Combined with a frequency-aware flow matching loss, it achieves FID 1.62 (256×256) and 2.22 (512×512) on ImageNet, substantially narrowing the gap between pixel diffusion and latent diffusion models.

## Background & Motivation
1. **State of the Field**: Latent diffusion models (LDMs) constitute the dominant paradigm, yet their two-stage pipeline relying on a VAE introduces lossy reconstruction and distribution shift. Pixel diffusion models end-to-end generation directly in pixel space, bypassing VAE limitations, but suffer from low training and inference efficiency.
2. **Limitations of Prior Work**: Existing pixel diffusion models employ a single DiT to model high-frequency signals and low-frequency semantics simultaneously. High-frequency noise is difficult to learn and interferes with low-frequency semantic learning, resulting in slow convergence and suboptimal generation quality.
3. **Root Cause**: DiTs excel at capturing low-frequency semantics but are ill-suited for handling high-frequency signals, while pixel space inherently contains both.
4. **Paper Goals**: Design a more efficient pixel diffusion paradigm that decouples the modeling of high- and low-frequency components.
5. **Starting Point**: Motivated by the observation that high-frequency signals are more easily reconstructed at high resolution, whereas low-frequency semantics are more easily modeled at low resolution.
6. **Core Idea**: The DiT operates on downsampled inputs to focus on low-frequency semantics, while a lightweight pixel decoder generates high-frequency details at full resolution conditioned on the DiT output.

## Method

### Overall Architecture
The input image is downsampled and processed by the DiT for low-frequency semantic modeling: $x_{\text{low}} = \text{DiT}(\bar{x}_t, t, y)$. A pixel decoder then predicts the pixel-space velocity at full resolution conditioned on $x_{\text{low}}$: $v_\theta(x_t, t, y) = \text{Dec}(x_t, t, x_{\text{low}})$. The training objective combines a standard flow matching loss, a frequency-aware flow matching loss, and a REPA alignment loss.

### Key Designs

1. **Frequency-Decoupled Architecture**:
    - Function: Assigns high-frequency and low-frequency modeling to separate modules.
    - Mechanism: The DiT models low-frequency semantics from downsampled inputs. The lightweight pixel decoder is an attention-free linear network consisting of $N$ linear blocks and projection layers, operating directly on full-resolution noisy images and conditioned on DiT outputs to synthesize high-frequency details. The multi-scale input strategy is central — the full-resolution input to the pixel decoder makes it inherently suited for high-frequency modeling.
    - Design Motivation: DCT energy analysis confirms that DeCo successfully transfers high-frequency components from the DiT to the pixel decoder; the high-frequency energy in DiT outputs is substantially lower than in the baseline.

2. **AdaLN Conditional Interaction**:
    - Function: Injects the DiT's low-frequency semantics into the pixel decoder.
    - Mechanism: The DiT output is upsampled to full resolution, and an MLP produces modulation parameters $\alpha, \beta, \gamma$, which modulate the dense queries in the decoder via AdaLN-Zero: $h_N = h_{N-1} + \alpha \cdot \text{MLP}((1+\gamma) \cdot h_{N-1} + \beta)$.
    - Design Motivation: AdaLN provides more effective conditional injection than simple addition; experiments confirm it outperforms UNet-style upsampling summation.

3. **Frequency-Aware Flow Matching Loss**:
    - Function: Emphasizes visually salient frequencies while suppressing perceptually unimportant high-frequency components.
    - Mechanism: Predicted and ground-truth velocities are transformed to YCbCr color space followed by an 8×8 DCT. Normalized reciprocals of the JPEG quantization table serve as adaptive weights: frequencies with smaller quantization intervals are deemed more important. A weighted MSE is then computed in the frequency domain: $\mathcal{L}_{\text{FreqFM}} = \mathbb{E}[w\|\mathbb{V}_\theta - \mathbb{V}_t\|^2]$.
    - Design Motivation: Standard flow matching loss treats all frequencies equally, yet human visual sensitivity varies substantially across frequencies. The JPEG quantization table encodes a robust prior over perceptual importance.

### Loss & Training
$\mathcal{L} = \mathcal{L}_{\text{FM}} + \mathcal{L}_{\text{FreqFM}} + \mathcal{L}_{\text{REPA}}$. Inference uses 50-step Euler sampling.

## Key Experimental Results

### Main Results

| Method | Type | FID↓ (256) | FID↓ (512) | IS↑ | Notes |
|--------|------|-----------|-----------|-----|-------|
| DeCo | Pixel diffusion | 1.62 | 2.22 | 294.6 | Pixel diffusion SOTA |
| DiT-XL/2 | Latent diffusion | 2.27 | - | 278.2 | Requires VAE |
| REPA-XL/2 | Latent diffusion | 1.42 | - | 305.5 | Best LDM |
| PixelFlow | Pixel diffusion | 54.33 | - | 24.67 | Multi-scale cascaded |
| Baseline | Pixel diffusion | 61.10 | - | 16.81 | No decoupling |

### Ablation Study

| Configuration | FID↓ | Notes |
|---------------|-----|-------|
| DeCo (full) | 31.35 | 200K iterations |
| w/o FreqFM | 34.12 | Frequency loss is effective |
| w/o REPA | 67.55 | REPA alignment is critical |
| Baseline | 61.10 | No decoupling |

### Key Findings
- DeCo reaches FID 2.57 at 400K iterations, converging approximately 10× faster than the baseline.
- The multi-scale input strategy and AdaLN interaction are both essential to effective frequency decoupling.
- The pixel decoder is extremely lightweight (attention-free), adding only 3% additional parameters while yielding substantial quality gains.
- DeCo also demonstrates strong text-to-image performance: GenEval 0.86, DPG-Bench 81.4.

## Highlights & Insights
- **The frequency decoupling principle** is concise yet powerful: let each module do what it does best.
- **Using the JPEG quantization table as a perceptual prior** is an elegant zero-cost trick that incorporates human visual knowledge.
- Pixel diffusion can now compete with latent diffusion, demonstrating that a VAE is not a prerequisite for high-quality generation.

## Limitations & Future Work
- Performance at 512×512 still slightly trails the strongest LDMs, though the gap is narrowing.
- The hidden dimension and number of layers of the pixel decoder require tuning.
- Future work may explore stronger frequency decoupling schemes or integration with concurrent works such as JiT.

## Related Work & Insights
- **vs. PixelFlow**: Employs cascaded multi-resolution stages, yet each stage still handles all frequency components simultaneously. DeCo performs within-timestep decoupling instead.
- **vs. DDT**: Performs single-scale frequency decoupling in latent space; DeCo is a multi-scale counterpart operating in pixel space.

## Rating
- Novelty: ⭐⭐⭐⭐ — The frequency decoupling idea is clear but not revolutionary.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Validated across 256/512/T2I settings with in-depth ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ — Analysis is thorough and figures are convincing.
- Value: ⭐⭐⭐⭐⭐ — Revitalizes pixel diffusion as a competitive alternative.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] PixelDiT: Pixel Diffusion Transformers for Image Generation](pixeldit_pixel_diffusion_transformers_for_image_generation.md)
- [\[ICCV 2025\] End-to-End Multi-Modal Diffusion Mamba](../../ICCV2025/image_generation/end-to-end_multi-modal_diffusion_mamba.md)
- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](dip_taming_diffusion_models_in_pixel_space.md)
- [\[CVPR 2026\] Frequency-Aware Flow Matching for High-Quality Image Generation](freqflow_frequency_aware_flow_matching.md)
- [\[ICCV 2025\] REPA-E: Unlocking VAE for End-to-End Tuning with Latent Diffusion Transformers](../../ICCV2025/image_generation/repa-e_unlocking_vae_for_end-to-end_tuning_of_latent_diffusion_transformers.md)

<!-- RELATED:END -->
