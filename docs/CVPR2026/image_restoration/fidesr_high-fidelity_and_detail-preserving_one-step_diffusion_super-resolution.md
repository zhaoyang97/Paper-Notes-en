---
title: >-
  [Paper Note] FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution
description: >-
  [CVPR 2026][Image Restoration][one-step diffusion SR] This paper proposes FiDeSR, a high-fidelity and detail-preserving one-step diffusion super-resolution framework that simultaneously addresses structural fidelity degr…
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "one-step diffusion SR"
  - "frequency-aware"
  - "residual refinement"
  - "detail weighting"
  - "high fidelity"
date: 2026-05-08
content_hash: 58e42f88222db9d7
---

# FiDeSR: High-Fidelity and Detail-Preserving One-Step Diffusion Super-Resolution

**Conference**: CVPR 2026
**arXiv**: [2603.02692](https://arxiv.org/abs/2603.02692)  
**Code**: [GitHub](https://github.com/Ar0Kim/FiDeSR)  
**Area**: Image Super-Resolution
**Keywords**: one-step diffusion SR, frequency-aware, residual refinement, detail weighting, high fidelity

## TL;DR

This paper proposes FiDeSR, a high-fidelity and detail-preserving one-step diffusion super-resolution framework that simultaneously addresses structural fidelity degradation and insufficient high-frequency detail recovery in one-step diffusion SR through three complementary components: Detail-Aware Weighting (DAW), Latent Residual Refinement Block (LRRB), and Latent Frequency Injection Module (LFIM).

## Background & Motivation

Diffusion models have demonstrated strong performance in real-world image super-resolution (Real-ISR), but multi-step diffusion inference is computationally expensive. One-step diffusion methods (SinSR, OSEDiff) compress the iterative process via distillation, yet face two core challenges:

**Difficulty maintaining high fidelity**: VAE-encoded conditioning introduces structural distortion and low-frequency inconsistencies (e.g., structural warping in AddSR).

**Insufficient high-frequency detail recovery**:
   - Multi-step diffusion progressively recovers high-frequency details through iterative denoising; compressing this into a single step leads to inadequate high-frequency reconstruction (e.g., over-smoothing in OSEDiff).
   - Recent residual learning methods (PiSA-SR) predict only a single global residual, resulting in unstable high-frequency reconstruction and residual artifacts (e.g., excessive detail generation in PiSA-SR).

The paper addresses fidelity and detail recovery separately at three stages: training (DAW), model architecture (LRRB), and inference (LFIM).

## Method

### Overall Architecture

FiDeSR is built on SD 2.1-base with LoRA fine-tuning in a one-step diffusion framework:
1. **Training stage**: The LQ image is encoded into $z_L$; the U-Net predicts a coarse residual $r$; LRRB refines it to $z_r$; DAW guides loss weighting.
2. **Inference stage**: After one-step diffusion, LRRB refines the residual, LFIM injects frequency enhancements, and the VAE decoder produces the output.

### Key Designs

1. **Detail-Aware Weighting (DAW)**: Adaptively emphasizes detail-rich regions where the model performs poorly. Two spatial maps are constructed:

    - **Detail map $D$**: Fuses three spatial operators — Sobel (edge sharpness), Laplacian (local contrast), and Variance (texture variance):
    $D = \frac{Sobel(x_H) + Laplacian(x_H) + Variance(x_H)}{3}$
    - **Error map $E$**: Fuses pixel-level and perceptual errors: $E = (1-p)E_{pix} + pE_{perc}$
    - **Difficulty weight map**: $W_{DAW} = D \odot E$, applied as spatial weighting to both reconstruction and CSD losses.
    - **Design Motivation**: Directs the model's focus toward visually important regions such as edges and textures, preventing overfitting on already well-reconstructed smooth areas.

2. **Latent Residual Refinement Block (LRRB)**: Compensates for the instability of residual prediction and insufficient high-frequency recovery in one-step diffusion. Based on the RRDB architecture, operating in the latent space:

    - Input: concatenation of $z_L$ and the U-Net's initial residual $r$.
    - Learns a correction $\Delta r$ to refine the residual: $r' = r + \Delta r$.
    - Final latent variable: $z_r = z_L - r'$.
    - **Design Motivation**: Treats the U-Net prediction as a strong initial estimate; LRRB learns a more precise correction. Unlike pixel-domain ESRGAN, LRRB specifically targets residual instability in the diffusion latent space.

3. **Latent Frequency Injection Module (LFIM)**: Flexibly enhances frequency components at inference time without retraining. Applies FFT decomposition to the refined latent $z_r$:

    - Separates low-frequency $\Delta_{LP}$ and high-frequency $\Delta_{HP}$ components via a Butterworth filter.
    - **Spatial gate $M_{sp}$**: Identifies detail-rich and flat regions based on the detail map (Sobel, Laplacian, Variance).
    - **Channel gate $M_{ch}$**: Analyzes the frequency energy ratio of each channel.
    - Selective injection: low-frequency components enhance structure; high-frequency components enhance texture.
    - **Design Motivation**: Enhancement intensity can be flexibly controlled at inference time without retraining.

### Loss & Training

Total loss $\mathcal{L}_{total} = \mathcal{L}_{rec} + \mathcal{L}_{reg}$:
- **Reconstruction loss**: $\mathcal{L}_{rec} = \lambda_{mse} \cdot W_{DAW} \cdot \text{MSE} + \lambda_{lpips} \cdot W'_{DAW} \cdot \text{LPIPS}$
- **Regularization loss**: DAW-weighted CSD loss (distilling semantic priors from the pretrained diffusion model).
- $\lambda_{mse} = 1$, $\lambda_{lpips} = 2$
- Backbone: SD 2.1-base with frozen VAE and U-Net; LoRA rank=8.
- Training: 2× H100, batch size 8, AdamW, lr $5 \times 10^{-5}$, 200K steps.
- Text prompts extracted via RAM.

## Key Experimental Results

### Main Results

| Dataset | Metric | FiDeSR (1-step) | PiSA-SR (1-step) | OSEDiff (1-step) | SeeSR (50-step) |
|--------|------|-------------|--------------|--------------|-------------|
| DRealSR | PSNR↑ | **28.90** | 28.32 | 27.92 | 28.14 |
| DRealSR | LPIPS↓ | **0.2836** | 0.2960 | 0.2967 | 0.3141 |
| DRealSR | MANIQA↑ | **0.6239** | 0.6161 | 0.5898 | 0.6016 |
| DRealSR | FID↓ | **127.97** | 130.48 | 135.45 | 146.98 |
| RealSR | LPIPS↓ | **0.2626** | 0.2672 | 0.3194 | 0.3004 |
| RealSR | FID↓ | **109.68** | 124.18 | 123.49 | 125.09 |
| DIV2K | DISTS↓ | **0.1845** | 0.1934 | 0.1975 | 0.1966 |

Note: FiDeSR uses only **1-step** inference and outperforms most one-step and several multi-step methods on both full-reference and no-reference metrics, achieving the lowest FID among all compared methods.

### Ablation Study

| Configuration | CLIPIQA↑ | NIQE↓ | MUSIQ↑ | MANIQA↑ | Note |
|------|----------|-------|--------|---------|------|
| w/o LRRB + w/o DAW | 0.6611 | 4.7381 | 67.60 | 0.6237 | Baseline |
| DAW only | 0.6641 | 4.7129 | 67.63 | 0.6236 | Marginal improvement |
| LRRB only | 0.6626 | 4.7340 | 67.95 | 0.6278 | More significant gain |
| DAW + LRRB | **0.6699** | **4.6300** | **68.29** | **0.6285** | Best complementary effect |

### Key Findings

- FiDeSR is the first one-step diffusion SR method to achieve an optimal balance on both full-reference and no-reference metrics simultaneously.
- LRRB reduces the average high-frequency noise prediction error by 1.62% (DIV2K: 1.24%, DRealSR: 1.99%, RealSR: 1.62%).
- LFIM's low-frequency injection improves PSNR/SSIM (structural fidelity) while high-frequency injection improves MUSIQ/MANIQA (perceptual quality), enabling flexible trade-off control.
- FiDeSR achieves the lowest FID across all datasets, indicating that its generated distribution most closely resembles the real image distribution.

## Highlights & Insights

1. **Precise problem analysis**: The paper clearly identifies two core bottlenecks in one-step diffusion SR (fidelity vs. detail) and proposes targeted designs at the training, architecture, and inference stages respectively.
2. **Dual guidance in DAW**: Simultaneously leverages the detail map ("where matters") and the error map ("where performance is poor"), yielding a more intelligent weighting strategy than purely frequency-based approaches.
3. **Principled design of LRRB**: Adapts the residual refinement concept from RRDB into the diffusion latent space, specifically targeting the instability of diffusion residual prediction.
4. **Flexibility of LFIM**: Enhancement intensity can be adjusted at inference time without retraining, offering strong practical utility.
5. **Breaking the perception–distortion trade-off**: FiDeSR achieves a better balance between perceptual quality and distortion than existing methods.

## Limitations & Future Work

1. Built on SD 2.1-base, the method may be constrained by the generative capacity of the backbone model.
2. Frequency separation in LFIM relies on manually set Butterworth filter parameters.
3. Computing the error map in DAW introduces additional training overhead.
4. More efficient one-step distillation strategies (e.g., consistency models) remain unexplored.
5. The framework could be extended to video super-resolution or multimodal restoration tasks.

## Related Work & Insights

- **PiSA-SR**: A one-step diffusion SR method based on residual learning; LRRB in FiDeSR directly improves upon its coarse residual prediction.
- **OSEDiff**: One-step SR based on VSD + LoRA; suffers from insufficient high-frequency recovery.
- **GuideSR**: Improves fidelity via full-resolution guidance, but perceptual quality is limited.
- **TFDSR**: Incorporates frequency information into multi-step diffusion; FiDeSR compresses a similar idea into a single step.
- **Insight**: The bottleneck of one-step diffusion lies not in the number of steps, but in the precision of residual prediction and the control of frequency components.

## Rating

- Novelty: ⭐⭐⭐⭐ All three components offer original contributions; the dual guidance in DAW and the diffusion latent-space residual refinement in LRRB are particularly distinctive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on 3 datasets with 9 metrics, compared against 8 methods, with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear; the motivation and interconnection of the three components are presented coherently.
- Value: ⭐⭐⭐⭐ One-step diffusion SR has strong practical value; simultaneously addressing fidelity and detail recovery constitutes an important contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Disentangled Textual Priors for Diffusion-based Image Super-Resolution](disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)
- [\[CVPR 2026\] Statistical Characteristic-Guided Denoising for Rapid High-Resolution Transmission Electron Microscopy Imaging](statistical_characteristic-guided_denoising_for_rapid_high-resolution_transmissi.md)
- [\[CVPR 2026\] Bridging the Perception Gap in Image Super-Resolution Evaluation](bridging_the_perception_gap_in_image_super-resolution_evaluation.md)
- [\[CVPR 2026\] SAT: Selective Aggregation Transformer for Image Super-Resolution](sat_selective_aggregation_transformer_for_image_super_resolution.md)
- [\[ICLR 2026\] Trust but Verify: Adaptive Conditioning for Reference-Based Diffusion Super-Resolution](../../ICLR2026/image_restoration/trust_but_verify_adaptive_conditioning_for_reference-based_diffusion_super-resol.md)

</div>

<!-- RELATED:END -->
