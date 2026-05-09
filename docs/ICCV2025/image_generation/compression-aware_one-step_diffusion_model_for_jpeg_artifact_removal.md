---
title: >-
  [Paper Note] Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal
description: >-
  [ICCV 2025][Image Generation][JPEG artifact removal] This paper proposes CODiff, a compression-aware one-step diffusion model for JPEG artifact removal. The core contribution is a Compression-aware Visual Embedder (CaVE) that extracts JPEG compression priors via an explicit–implicit dual learning strategy, guiding the diffusion model toward high-quality restoration. CODiff comprehensively outperforms existing methods on LIVE-1, Urban100, and DIV2K-Val while achieving extremely high inference efficiency.
tags:
  - ICCV 2025
  - Image Generation
  - JPEG artifact removal
  - one-step diffusion model
  - compression prior
  - dual learning
  - image restoration
date: 2026-05-08
content_hash: 807966ddc8aa402f
---

# Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal

**Conference**: ICCV 2025
**arXiv**: [2502.09873](https://arxiv.org/abs/2502.09873)
**Code**: [github.com/jp-guo/CODiff](https://github.com/jp-guo/CODiff)
**Area**: Image Generation / Image Restoration
**Keywords**: JPEG artifact removal, one-step diffusion model, compression prior, dual learning, image restoration

## TL;DR

This paper proposes CODiff, a compression-aware one-step diffusion model for JPEG artifact removal. The core contribution is a Compression-aware Visual Embedder (CaVE) that extracts JPEG compression priors via an explicit–implicit dual learning strategy, guiding the diffusion model toward high-quality restoration. CODiff comprehensively outperforms existing methods on LIVE-1, Urban100, and DIV2K-Val while achieving extremely high inference efficiency.

## Background & Motivation

### Problem Definition

JPEG artifact removal aims to eliminate compression distortions such as blocking and banding from compressed images, recovering lost visual information. At high compression rates (e.g., QF=5), information loss is severe, and conventional CNN/Transformer-based methods struggle to cope.

### Limitations of Prior Work

**CNN/Transformer methods** (FBCNN, PromptCIR, etc.): Limited effectiveness at high compression rates, as lost information cannot be fully recovered from the remaining signal.

**Multi-step diffusion models** (DiffBIR, SUPIR): While possessing strong generative priors, their multi-step denoising incurs enormous computational overhead (188T MACs, ~50 seconds for 50 steps).

**Existing one-step diffusion models** (OSEDiff): Inference-efficient but agnostic to JPEG compression priors, unable to distinguish compression artifacts from natural image features.

**Insufficient use of compression priors**: Prior quality factor (QF) learning methods treat QF (a single integer) as the sole learning target, capturing very limited information; quantization table methods provide only static numerical values.

### Root Cause

How can JPEG compression priors be effectively extracted and leveraged to guide a diffusion model while preserving one-step inference efficiency?

**Core Idea**: Design a Compression-aware Visual Embedder (CaVE) that comprehensively captures JPEG compression characteristics via a dual strategy—*explicit learning* (QF prediction) and *implicit learning* (high-quality image reconstruction)—and inject the extracted priors into a one-step diffusion model.

## Method

### Overall Architecture

CODiff adopts a two-stage training pipeline:
- **Stage 1**: Train CaVE to extract JPEG compression prior embeddings.
- **Stage 2**: Inject the priors extracted by CaVE into a pretrained Stable Diffusion model (fine-tuned via LoRA), training the generator with perceptual loss and GAN loss.

### Key Designs

#### 1. Compression-aware Visual Embedder (CaVE)

- **Function**: Encodes the low-quality image $\mathbf{I}_L$ into a set of feature vectors $\mathbf{c}_L = \{\mathbf{c}_{L_k} \in \mathbb{R}^d\}_{k=1}^K$ serving as JPEG compression priors.
- **Core Architecture**: UNet encoder + lightweight QF predictor + UNet decoder.
- **Design Motivation**: Exploits the multi-scale feature extraction capacity of UNet to capture compression-related information across multiple resolutions.

#### 2. Dual Learning Strategy

- **Explicit learning**: Trains CaVE to predict QF from the embedding using an $\ell_1$ loss:
$$\mathcal{L}_{QF} = \frac{1}{B}\sum_{i=1}^{B}\|QF_{pred}^i - QF_{gt}^i\|_1$$
  - Motivation: Encourages the embedding to explicitly distinguish different compression levels.
  - Limitation: QF prediction alone fails to generalize to unseen compression levels (confirmed by t-SNE visualization).

- **Implicit learning**: Trains CaVE to reconstruct the high-quality image from the embedding using an $\ell_1$ loss:
$$\mathcal{L}_{rec} = \frac{1}{B}\sum_{i=1}^{B}\|\hat{\mathbf{I}}_H^i - \mathbf{I}_H^i\|_1$$
  - Motivation: The reconstruction objective forces the embedding to capture the complete information of the compression process, rather than a single QF integer.

- **Joint objective**: $\mathcal{L}_{CaVE} = \mathcal{L}_{QF} + \lambda \cdot \mathcal{L}_{rec}$, where $\lambda=1000$.
  - Key finding: After dual learning, CaVE can effectively distinguish **unseen** compression levels (QF=1, 5) at test time, whereas purely explicit learning cannot.

#### 3. One-Step Diffusion Generator

- **Function**: Takes the latent representation of the low-quality image as input (replacing Gaussian noise) and recovers the high-quality image in a single denoising step.
- **Core formula**:
$$\hat{\mathbf{z}}_H = \frac{\mathbf{z}_L - \sqrt{1-\bar{\alpha}_{T_L}} \varepsilon_\theta(\mathbf{z}_L; \mathbf{c}_L, T_L)}{\sqrt{\bar{\alpha}_{T_L}}}$$
- **Training**: The VAE encoder and UNet are fine-tuned via LoRA (rank=16); the VAE decoder is frozen.
- **Discriminator**: Pretrained SD UNet encoder + lightweight MLP.

### Loss & Training

Stage 2 total loss: $\mathcal{L} = \mathcal{L}_{per} + \lambda_G \mathcal{L}_{\mathcal{G}}$
- Perceptual loss: $\mathcal{L}_{per} = \mathcal{L}_2(\hat{\mathbf{I}}_H, \mathbf{I}_H) + \lambda_D \mathcal{L}_{DISTS}(\hat{\mathbf{I}}_H, \mathbf{I}_H)$
- GAN loss: Standard adversarial loss with $\lambda_G = 5 \times 10^{-3}$
- Distillation is not used (unlike OSEDiff); GAN is instead employed to enhance perceptual realism.

Training details:
- Stage 1: 200K iterations, 4× A6000
- Stage 2: 100K iterations, 4× A6000, AdamW, lr=5e-5
- Training QF range: 8–95; patch size: 256×256

## Key Experimental Results

### Main Results

LIVE-1 dataset (QF=5, perceptual quality metrics):

| Method | Steps | LPIPS↓ | DISTS↓ | MUSIQ↑ | MANIQA↑ | CLIPIQA↑ |
|--------|-------|--------|--------|--------|---------|----------|
| JPEG | — | 0.4384 | 0.3242 | 40.33 | 0.2294 | 0.1716 |
| FBCNN | 1 | 0.3736 | 0.2353 | 63.56 | 0.3425 | 0.2763 |
| PromptCIR | 1 | 0.3797 | 0.2334 | 60.34 | 0.2790 | 0.2655 |
| DiffBIR* | 50 | 0.3509 | 0.2035 | 58.09 | 0.2812 | 0.3776 |
| OSEDiff* | 1 | 0.2675 | 0.1653 | 65.51 | 0.3417 | 0.5623 |
| **CODiff** | **1** | **0.2062** | **0.1121** | **73.16** | **0.5321** | **0.7212** |

Computational efficiency (1024×1024 input):

| Method | Steps | Params (G) | MACs (T) | Time (s) |
|--------|-------|-----------|---------|---------|
| DiffBIR | 50 | 1.52 | 188.24 | 50.81 |
| SUPIR | 50 | 4.49 | 464.29 | 24.33 |
| OSEDiff | 1 | 1.40 | 10.39 | 0.65 |
| **CODiff** | **1** | **1.00** | **9.46** | **0.57** |

### Ablation Study

Comparison of prompt generation strategies (LIVE-1, QF=5):

| Method | LPIPS↓ | MUSIQ↑ | MANIQA↑ |
|--------|--------|--------|---------|
| Empty string | 0.3485 | 62.56 | 0.3793 |
| Learnable | 0.3471 | 63.39 | 0.3900 |
| DAPE | 0.3463 | 62.54 | 0.3793 |
| **CaVE (Ours)** | **0.3426** | **67.13** | **0.4584** |

The effectiveness of the dual learning strategy is clearly demonstrated through t-SNE visualization: CaVE trained with only explicit learning fails to distinguish unseen QF values (1, 5), whereas dual learning yields well-separated clusters.

### Key Findings

- CODiff comprehensively outperforms existing methods across all three datasets and all QF levels, including 50-step DiffBIR and SUPIR.
- The advantage is most pronounced at extreme compression (QF=5): LPIPS drops from 0.2675 to 0.2062, and MANIQA rises from 0.3417 to 0.5321.
- Inference is 89× faster than DiffBIR (0.57s vs. 50.81s) with 34% fewer parameters.
- CaVE is the core driver of performance gains (MANIQA 18% higher than DAPE).
- Dual learning substantially outperforms purely explicit or purely implicit learning.

## Highlights & Insights

- **Elegant exploitation of compression priors**: Rather than merely predicting QF, the model also employs a reconstruction objective to understand the full compression process—a generalizable idea of using multi-task learning to enrich representations.
- **Lightweight design**: CaVE relies only on a UNet encoder (without a full UNet), making it far more lightweight than auxiliary modules such as ControlNet or DAPE.
- **No distillation required**: Unlike OSEDiff, CODiff replaces distillation with GAN training, removing the performance ceiling imposed by a teacher model.
- **t-SNE visualization**: Intuitively demonstrates how dual learning enables generalization to unseen compression levels.

## Limitations & Future Work

- The training QF range is 8–95; generalization to extremely low QF values (1–7) relies on the implicit extrapolation capacity of dual learning.
- The method is designed specifically for JPEG compression and does not explore modern formats such as WebP or HEIF.
- GAN training may introduce mode collapse risks.
- Evaluation is not conducted on real-world compressed images (as opposed to synthetically degraded ones).
- The UNet decoder of CaVE is used only during Stage 1 training and discarded in Stage 2, resulting in some computational overhead.

## Related Work & Insights

- FBCNN first proposed predicting an adjustable QF as a compression prior; CODiff substantially improves upon this via dual learning.
- OSEDiff demonstrated the viability of one-step diffusion for image restoration; CODiff further demonstrates the importance of domain-specific priors.
- The design philosophy of CaVE (multi-task representation extraction) is generalizable to other degradation types (blur, noise).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual learning strategy and injection of compression priors into a diffusion model constitute an effective and novel design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three datasets, multiple QF levels, comprehensive metric suite, and complete ablation study.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear figures; t-SNE visualizations are persuasive.
- **Value**: ⭐⭐⭐⭐⭐ — High practical value with fast inference, strong performance, and open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](sanasprint_onestep_diffusion_with_continuoustime_consistency.md)
- [\[AAAI 2026\] Steering One-Step Diffusion Model with Fidelity-Rich Decoder for Fast Image Compression](../../AAAI2026/image_generation/steering_one-step_diffusion_model_with_fidelity-rich_decoder_for_fast_image_comp.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[ICCV 2025\] Fewer Denoising Steps or Cheaper Per-Step Inference: Towards Compute-Optimal Diffusion Model Deployment](fewer_denoising_steps_or_cheaper_per-step_inference_towards_compute-optimal_diff.md)
- [\[ICCV 2025\] Structure-Guided Diffusion Models for High-Fidelity Portrait Shadow Removal](structure-guided_diffusion_models_for_high-fidelity_portrait_shadow_removal.md)

</div>

<!-- RELATED:END -->
