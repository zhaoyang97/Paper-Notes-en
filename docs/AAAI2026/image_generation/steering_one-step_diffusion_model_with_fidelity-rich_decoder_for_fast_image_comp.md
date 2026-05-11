---
title: >-
  [Paper Note] Steering One-Step Diffusion Model with Fidelity-Rich Decoder for Fast Image Compression
description: >-
  [AAAI 2026][Image Generation][Image Compression] This paper proposes SODEC, a one-step diffusion-based image compression model that injects the prior of a high-fidelity VAE decoder into the diffusion generation process via a Fidelity Guidance Module (FGM). Combined with a rate annealing training strategy, SODEC achieves high-quality compression at extremely low bitrates, with decoding speed more than 20× faster than multi-step diffusion methods, while reaching state-of-the-art rate-distortion-perception trade-offs.
tags:
  - AAAI 2026
  - Image Generation
  - Image Compression
  - Diffusion Model
  - One-Step Decoding
  - Fidelity Guidance
  - Rate Annealing
date: 2026-05-08
content_hash: 6c821bc59eeca94c
---

# Steering One-Step Diffusion Model with Fidelity-Rich Decoder for Fast Image Compression

**Conference**: AAAI 2026
**arXiv**: [2508.04979](https://arxiv.org/abs/2508.04979)
**Code**: [https://github.com/zhengchen1999/SODEC](https://github.com/zhengchen1999/SODEC)
**Area**: Image Generation
**Keywords**: Image Compression, Diffusion Model, One-Step Decoding, Fidelity Guidance, Rate Annealing

## TL;DR

This paper proposes SODEC, a one-step diffusion-based image compression model that injects the prior of a high-fidelity VAE decoder into the diffusion generation process via a Fidelity Guidance Module (FGM). Combined with a rate annealing training strategy, SODEC achieves high-quality compression at extremely low bitrates, with decoding speed more than 20× faster than multi-step diffusion methods, while reaching state-of-the-art rate-distortion-perception trade-offs.

## Background & Motivation

### State of the Field

**Traditional codecs** (JPEG2000, VVC) perform reliably at medium-to-high bitrates, but produce severe blocking artifacts, blurring, and structural distortion when the bitrate drops to extremely low levels (<0.1 bpp).

**VAE-based compression** methods (e.g., HiFiC, MS-ILLM) surpass traditional codecs on rate-distortion metrics (PSNR/MS-SSIM) through probabilistic modeling innovations such as hyperpriors, and further optimize the rate-distortion-perception framework by incorporating perceptual losses. However, these methods still struggle to reconstruct fine details at extremely low bitrates — reconstructions are technically correct but lack perceptual realism.

**Diffusion-based compression** methods (e.g., CDC, PerCo, DiffEIC) leverage the powerful generative priors of diffusion models to achieve excellent rate-perception trade-offs, synthesizing highly realistic textures and details under extreme compression.

### Root Cause

Nevertheless, diffusion-based compression suffers from two critical drawbacks:

**High latency**: The multi-step denoising process incurs substantial decoding delay and computational cost (PerCo: 6.2 s, DiffEIC: 7.8 s), making these methods unsuitable for real-time or resource-constrained scenarios.

**Low fidelity**: Diffusion models rely heavily on pretrained priors rather than the input itself, causing reconstructions to deviate from the original content.

### Starting Point

The key argument is: **in image compression, a sufficiently information-rich latent renders multi-step refinement unnecessary**. By leveraging the information-rich latents produced by a pretrained VAE-based model, one-step decoding can replace iterative denoising. A fidelity guidance mechanism is additionally introduced to compensate for generative drift.

### Core Idea

Three core designs:

**One-step decoding**: Information-rich latent + single-step diffusion = high-quality reconstruction (no multi-step required).

**Fidelity Guidance Module (FGM)**: A VAE decoder generates a high-fidelity preliminary reconstruction that serves as a visual guidance condition.

**Rate annealing training strategy**: The model is first pretrained at high bitrates to learn rich representations, then progressively annealed to the target low bitrate.

## Method

### Overall Architecture

The complete pipeline of SODEC:

**Encoding side**:
1. Input image $x \in \mathcal{R}^{H \times W \times 3}$ is downsampled 16× by VAE encoder $\mathcal{E}$ to latent $y \in \mathcal{R}^{H/16 \times W/16 \times C}$ (typically $C=220$).
2. Hyper-encoder $\mathcal{H}_a$ extracts hyperprior $z = \mathcal{H}_a(y)$.
3. Quantized representations $\hat{y} = \mathcal{Q}(y)$, $\hat{z} = \mathcal{Q}(z)$ are entropy-coded via probability model $\mathcal{P}$.

**Decoding side**:
1. $\hat{y}$ and $\hat{z}$ are recovered.
2. Transform module $\mathcal{T}_s$ converts them into a content variable $\hat{y}_t \in \mathcal{R}^{64 \times 64 \times 4}$ suitable for diffusion processing.
3. The **one-step diffusion** model performs a single denoising step to obtain $\hat{y}_0$.
4. Simultaneously, **FGM** uses the VAE decoder to generate a high-fidelity preliminary reconstruction $\hat{x}_f$; features extracted by a ViT are projected into guidance condition $c_g$ and injected into the diffusion process.

### Key Designs

#### 1. **One-Step Diffusion Model**

The hyper-synthesis network $\mathcal{H}_s$ first extracts global information $w$ from $\hat{z}$, which is merged with $\hat{y}$ and transformed into the content variable:

$$\hat{y}_t = \sqrt{\bar{\alpha}_t}\hat{y}_0 + \sqrt{1 - \bar{\alpha}_t}\epsilon$$

A UNet architecture $\epsilon_\theta$ based on Stable Diffusion 2.1 performs single-step prediction:

$$\hat{y}_0 = \frac{\hat{y}_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(\hat{y}_t, t, c_g)}{\sqrt{\bar{\alpha}_t}}$$

The timestep $t$ is fixed at 999 (maximum noise), and the diffusion model is adapted to the compression task via LoRA fine-tuning.

**Design Motivation**: Unlike standard text-to-image generation, the latent in compression carries rich information about the original image — the structural information encoded by the VAE far exceeds that of random noise. Consequently, a single step suffices to generate high-quality reconstructions without multi-step iteration.

#### 2. **Fidelity Guidance Module (FGM)**

FGM compensates for the generative bias of diffusion models:

$$\hat{x}_f = \mathcal{D}_a(\hat{y})$$

where $\mathcal{D}_a$ is a fidelity decoder initialized from a HiFiC-pretrained VAE decoder and further fine-tuned. $\hat{x}_f$ is highly faithful to the original image but may lack perceptual richness.

A pretrained ViT then extracts deep visual features, which are mapped into the diffusion model's conditioning space via a projection network:

$$c_g = \mathcal{F}_p(\mathcal{F}(\hat{x}_f)) \in \mathcal{R}^{L \times D}$$

where $L=77$, $D=1024$. $c_g$ is injected into the UNet via cross-attention.

**Design Motivation**: The VAE decoder excels at fidelity but lacks perceptual quality; the diffusion model excels at synthesizing realistic textures but lacks knowledge of the source image. FGM combines the strengths of both — using the high-fidelity reconstruction as a strong conditioning signal to guide the diffusion generator toward outputs that are both realistic and faithful.

Ablation studies confirm that FGM outperforms text-prompt guidance (PerCo) and hyperprior guidance (DiffEIC):
- No guidance: MS-SSIM = 0.8212
- Text-prompt guidance: MS-SSIM = 0.8185
- Hyperprior guidance: MS-SSIM = 0.8258
- **FGM**: **MS-SSIM = 0.8481** (+2.7%)

#### 3. **Rate Annealing Training Strategy**

Three-stage training design:

**Stage 1: High-Bitrate VAE Pretraining**
- Train HiFiC with a small $\lambda$ (low rate penalty) to encourage learning of rich, comprehensive latent representations.
- High bitrate → information-rich encoder–decoder pair.

**Stage 2: Diffusion Path Warm-Up**
- Freeze the entire VAE encoding module; train only the diffusion generation path.
- ViT and diffusion decoder $\mathcal{D}_m$ are frozen; UNet is fine-tuned with LoRA.
- Full-parameter training: $\mathcal{H}_s$, $\mathcal{T}_s$, $\mathcal{D}_a$, $\mathcal{F}_p$.
- Distortion loss only; no rate penalty or alignment loss.
- Goal: teach the one-step diffusion generator to effectively map fixed latents to high-quality reconstructions.

**Stage 3: Joint Training + Rate Annealing**
- End-to-end optimization with increasing rate penalty $\lambda$ (rate annealing).
- Alignment loss is introduced to ensure $\mathcal{D}_a$ maintains high fidelity as the latent varies:

$$\mathcal{L}_{align} = \mathbb{E}[\|x - \hat{x}_f\|_2^2]$$

Overall loss:

$$\mathcal{L}_{overall} = d(x, \hat{x}) + \lambda \cdot r(\hat{y}, \hat{z}) + \alpha \cdot \mathcal{L}_{align}$$

A GAN loss is subsequently added to further enhance detail synthesis:

$$\mathcal{L}_{finetune} = d(x, \hat{x}) + \lambda \cdot r(\hat{y}, \hat{z}) + \alpha \cdot \mathcal{L}_{align} + \beta \cdot \mathcal{L}_g$$

**Design Motivation**: The core intuition is that **selectively discarding information from rich representations is easier than hallucinating details from impoverished ones**. By first learning comprehensive features under relaxed constraints and then progressively tightening the rate constraint, the model can intelligently retain the most important information.

### Loss & Training

Distortion loss: $d(x, \hat{x}_f) = k_M \cdot \text{MSE} + k_P \cdot \text{LPIPS}$

Each of the three stages employs a distinct loss configuration, progressively increasing in complexity from distortion-only to rate–distortion–alignment–GAN.

## Key Experimental Results

### Main Results

Inference efficiency comparison (DIV2K-Val, 512×512, A6000):

| Method | Total (ms) | Encode (ms) | Decode (ms)↓ | bpp↓ |
|--------|-----------|-------------|-------------|------|
| HiFiC | 9.3 | 5.4 | 3.9 | 0.0310 |
| MS-ILLM | 93.7 | 54.5 | 84.4 | 0.0395 |
| PerCo | 6,242 | 1,540 | **4,702** | 0.0313 |
| DiffEIC | 7,828 | 266 | **7,561** | 0.0391 |
| **SODEC** | **233** | 5.0 | **228** | **0.0314** |

SODEC is **26×** faster than PerCo and **33×** faster than DiffEIC.

Perception–rate–latency comparison (DIV2K-Val):

| Method | bpp↓ | LPIPS↓ | Decode Speed |
|--------|------|--------|-------------|
| HiFiC | 0.0420 | High | Very fast (3.9 ms) |
| PerCo | 0.0535 | Low | Very slow (4702 ms) |
| DiffEIC | 0.0457 | Low | Very slow (7561 ms) |
| **SODEC** | **0.0322** | **Lowest** | Fast (228 ms) |

### Ablation Study

Fidelity guidance strategy ablation:

| Guidance Strategy | MS-SSIM↑ | LPIPS↓ | bpp↓ |
|------------------|---------|--------|------|
| No guidance | 0.8212 | 0.3625 | 0.0424 |
| Text-prompt guidance (PerCo) | 0.8185 | 0.3631 | 0.0412 |
| Hyperprior guidance (DiffEIC) | 0.8258 | 0.3527 | 0.0385 |
| **Auxiliary fidelity guidance (Ours)** | **0.8481** | **0.3351** | **0.0368** |

Training strategy ablation:

| Training Strategy | MS-SSIM↑ | LPIPS↓ | bpp↓ |
|------------------|---------|--------|------|
| Frozen VAE module | 0.8512 | 0.3761 | 0.0695 |
| Joint training (matched bpp) | 0.8621 | 0.3750 | 0.0678 |
| Low-to-high bpp curriculum | 0.8643 | 0.3451 | 0.0593 |
| **Rate annealing (Ours)** | **0.8951** | **0.3113** | **0.0604** |

Alignment loss ablation:

| Alignment Loss Configuration | MS-SSIM↑ | LPIPS↓ | bpp↓ |
|-----------------------------|---------|--------|------|
| No alignment loss | 0.7490 | 0.4210 | 0.0203 |
| MSE + LPIPS | 0.7481 | 0.3961 | 0.0199 |
| Merged into main loss | 0.7984 | 0.4023 | 0.0232 |
| **MSE-only alignment (Ours)** | **0.7948** | **0.3827** | **0.0227** |

### Key Findings

1. **One-step diffusion is sufficient for compression**: Information-rich VAE latents render multi-step refinement redundant.
2. **Rate annealing outperforms all other training strategies**: It saves ~30% bitrate at a given quality level, or significantly improves quality at equal bitrate.
3. **FGM substantially outperforms text/hyperprior guidance**: MS-SSIM improves by 2.7% with concurrent LPIPS improvement.
4. **MSE-only alignment loss is optimal**: Adding LPIPS or merging into the main loss is less effective than pure MSE alignment.
5. **Order-of-magnitude decoding speedup**: 228 ms vs. 4–7.5 s, making diffusion-based compression viable for real-time applications for the first time.

## Highlights & Insights

1. The central argument — **"information-rich latents render multi-step refinement unnecessary"** — is compelling and fundamentally challenges the assumption that diffusion-based compression requires multiple steps.
2. The **complementary design of FGM** is elegant: it explicitly combines VAE fidelity with diffusion perceptual quality.
3. The **rate annealing** principle of "selective discarding from rich representations" is conceptually inspiring and draws an interesting parallel with knowledge distillation.
4. The **progressive three-stage training** design (VAE pretraining → diffusion path warm-up → joint training) avoids instability in early-stage joint optimization.
5. The **20–33× decoding speedup** is a critical step toward transforming diffusion-based compression from an academic concept into a practical technology.

## Limitations & Future Work

1. **The encoder remains a conventional VAE**: The 16× downsampling with 220 channels may limit the potential for extreme compression.
2. **Based on SD 2.1**: Performance on newer architectures such as SD3/FLUX has not been validated.
3. **512×512 resolution**: Behavior on high-resolution images is unknown.
4. **Fixed timestep t=999**: The optimal timestep may vary across different bitrates.
5. **Image compression only**: The temporal dimension of video compression may be even better suited to the one-step paradigm, warranting further exploration.

## Related Work & Insights

- **PerCo** (Careil et al., 2023): Multi-step diffusion compression with dual text and visual feature guidance.
- **DiffEIC** (Li et al., 2024): Diffusion image compression with hyperprior guidance.
- **HiFiC** (Mentzer et al., 2020): Generative adversarial compression; the source of SODEC's VAE backbone initialization.
- **One-step diffusion** (e.g., DMD, LCM, SDXL-Turbo): Related work on single-step generation.
- **Key Takeaway**: The fundamental distinction between image compression and image generation is that compression has the source image as a condition, while generation does not. This informational advantage makes one step sufficient, opening a new paradigm for diffusion models in low-latency visual tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of one-step diffusion compression, fidelity guidance, and rate annealing constitutes a complete and well-motivated design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Validated on three datasets with comprehensive rate–distortion–perception curves and detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-argued motivation, and rich figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ — 20× decoding speedup combined with SOTA quality marks a pivotal step toward practical diffusion-based compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Realism Control One-step Diffusion for Real-World Image Super-Resolution](realism_control_one-step_diffusion_for_real-world_image_super-resolution.md)
- [\[ICCV 2025\] Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal](../../ICCV2025/image_generation/compression-aware_one-step_diffusion_model_for_jpeg_artifact_removal.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](../../CVPR2026/image_generation/pixelrush_ultra-fast_training-free_high-resolution_image_generation_via_one-step.md)
- [\[AAAI 2026\] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)
- [\[AAAI 2026\] GEWDiff: Geometric Enhanced Wavelet-based Diffusion Model for Hyperspectral Image Super-resolution](gewdiff_geometric_enhanced_wavelet-based_diffusion_model_for_hyperspectral_image.md)

</div>

<!-- RELATED:END -->
