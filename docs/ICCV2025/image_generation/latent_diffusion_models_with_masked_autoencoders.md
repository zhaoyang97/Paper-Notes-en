---
title: >-
  [Paper Note] Latent Diffusion Models with Masked AutoEncoders
description: >-
  [ICCV 2025][Image Generation][Latent Diffusion Models] This paper systematically analyzes three key properties that autoencoders in LDMs should possess (latent space smoothness, perceptual compression quality, and reconstruction quality), identifies that existing autoencoders fail to satisfy all three simultaneously, and proposes Variational Masked AutoEncoders (VMAEs). By combining MAE's hierarchical features with VAE's probabilistic encoding, VMAEs achieve significant improvements in generation quality (ImageNet-1K gFID: 5.98 vs. 6.49 for SD-VAE) using only 13.4% of the parameters and 4.1% of the GFLOPs.
tags:
  - ICCV 2025
  - Image Generation
  - Latent Diffusion Models
  - Masked AutoEncoders
  - Variational AutoEncoders
  - Perceptual Compression
  - Latent Space Smoothness
date: 2026-05-08
content_hash: 297c55c0c9d6fa8e
---

# Latent Diffusion Models with Masked AutoEncoders

**Conference**: ICCV 2025
**arXiv**: [2507.09984](https://arxiv.org/abs/2507.09984)
**Code**: [https://github.com/isno0907/ldmae](https://github.com/isno0907/ldmae)
**Area**: Image Generation
**Keywords**: Latent Diffusion Models, Masked AutoEncoders, Variational AutoEncoders, Image Generation, Perceptual Compression, Latent Space Smoothness

## TL;DR

This paper systematically analyzes three key properties that autoencoders in LDMs should possess (latent space smoothness, perceptual compression quality, and reconstruction quality), identifies that existing autoencoders fail to satisfy all three simultaneously, and proposes Variational Masked AutoEncoders (VMAEs). By combining MAE's hierarchical features with VAE's probabilistic encoding, VMAEs achieve significant improvements in generation quality (ImageNet-1K gFID: 5.98 vs. 6.49 for SD-VAE) using only 13.4% of the parameters and 4.1% of the GFLOPs.

## Background & Motivation

**Why is autoencoder design critical for LDMs?** The core idea of Latent Diffusion Models is to transfer the denoising process from pixel space to a compressed latent space, and the autoencoder determines the properties of this latent space. However, existing research has insufficiently explored what properties autoencoders should possess and how different design choices affect the overall LDM framework.

**Three key properties are proposed:**

**Latent Space Smoothness**: Small perturbations in the latent representation should not lead to drastic changes in the generated output. The denoising process in diffusion models introduces prediction errors, and a smooth latent space can tolerate these errors.

**Perceptual Compression Quality**: Effectively compresses perceptual details while retaining semantic information. However, there is no clear boundary between "semantics" and "perceptual details" — they form a continuous spectrum from pixel level to object level.

**Reconstruction Quality**: The decoder should accurately reconstruct the original image, evaluated at both perceptual level (rFID, LPIPS) and pixel level (PSNR, SSIM).

**Limitations of existing autoencoders:**
- **AE/DAE** (deterministic encoding): Sparse latent space; fails to satisfy smoothness.
- **VAE**: Smoothest latent space but poor reconstruction quality.
- **SD-VAE**: Overly aggressive perceptual compression (compressed to object level), losing fine-grained features.

## Method

### Overall Architecture

VMAE adopts a symmetric ViT architecture (encoder–decoder), incorporating probabilistic encoding, masked prediction loss, and perceptual loss to simultaneously satisfy all three properties.

### Smooth Latent Space (Sec. 4.1)

Probabilistic encoding $q_\phi(\mathbf{z}|\mathbf{x})$ is adopted instead of fixed vectors, with KL divergence constraining the latent distribution toward a Gaussian prior:

$$\mathcal{L}_{\text{reg}} = \mathbb{E}_{p_{\text{data}}(\mathbf{x}), p(\mathbf{x}_v|\mathbf{x})} [D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}_v) | p(\mathbf{z}))]$$

**Key design**: The predicted mean is learnable (not forced to zero); only the variance is constrained to unit variance. This satisfies the Variance Preserving (VP) condition while retaining discriminative features.

Deterministic encoders (AE/DAE) map inputs to discrete sparse points, leaving most of the latent space undecodable. Probabilistic encoders map inputs to distributions, forming a continuous latent space where points in the noise neighborhood remain decodable.

### Hierarchical Perceptual Compression (Sec. 4.2)

MAE's masked prediction objective is employed to achieve hierarchical compression:

$$\mathcal{L}_{\text{M}} = \mathbb{E}[-\log p_\theta(\mathbf{x}_m | \mathbf{z})]$$

where the encoder receives only the visible region $\mathbf{x}_v$, and the decoder predicts the masked region $\mathbf{x}_m$ based on the latent variable $\mathbf{z}$.

**Why does MAE enable hierarchical compression?** Recent research demonstrates that MAE's masked prediction training causes encoded features to form hierarchical clusters in the embedding space — progressively differentiating from abstract object level to simpler visual pattern level. This hierarchical structure both facilitates diffusion model training (high-level clusters simplify learning) and preserves fine-grained information (multi-level discriminative features support high reconstruction quality).

By contrast, SD-VAE's compression is overly aggressive: features clustered at the object level cannot further distinguish different parts (e.g., the fur pattern of a giraffe), leading to detail loss during decoding.

### Perceptual Reconstruction (Sec. 4.3)

Reconstruction loss and LPIPS perceptual loss are applied to the visible region:

$$\mathcal{L}_{\text{R}} = \mathbb{E}[-\log p_\theta(\mathbf{x}_v | \mathbf{z})]$$

$$\mathcal{L}_{\text{P}} = \mathbb{E}\left[\sum_l w_l \|\psi_l(\mathbf{x}) - \psi_l(\hat{\mathbf{x}})\|_2^2\right]$$

where $\psi_l$ denotes the feature extraction at the $l$-th layer of a pretrained VGG network.

### Full Training Objective

$$\mathcal{L}_{\text{VMAE}} = \mathcal{L}_{\text{R}} + \lambda_{\text{M}} \cdot \mathcal{L}_{\text{M}} + \lambda_{\text{P}} \cdot \mathcal{L}_{\text{P}} + \lambda_{\text{reg}} \cdot \mathcal{L}_{\text{reg}}$$

## Key Experimental Results

### Main Results: Generation Performance Comparison

| Autoencoder | ImageNet gFID↓ | sFID↓ | IS↑ | Prec↑ | Rec↑ | CelebA gFID↓ |
|-------------|---------------|-------|-----|-------|------|-------------|
| AE | 12.92 | 12.65 | 124.0 | 0.724 | 0.339 | 24.80 |
| DAE | 8.60 | 12.12 | 160.3 | 0.797 | 0.402 | 21.42 |
| VAE | 34.60 | 22.32 | 54.6 | 0.517 | 0.415 | 32.33 |
| SD-VAE | 6.49 | 5.60 | 173.3 | 0.819 | 0.429 | 9.00 |
| **VMAE** | **5.98** | **5.16** | **185.5** | **0.844** | **0.435** | **7.61** |

VMAE surpasses SD-VAE across all generation metrics: ImageNet gFID improves by 0.51, IS by 12.2, and CelebA gFID by 1.39. Deterministic autoencoders (AE/DAE) show significantly degraded performance due to the lack of a smooth latent space.

### Reconstruction Performance Comparison

| Model | L1↓ | PSNR↑ | SSIM↑ | LPIPS↓ | rFID↓ |
|-------|-----|-------|-------|--------|-------|
| AE | 0.0218 | 32.18 | 0.895 | 0.172 | 6.21 |
| DAE | 0.0237 | 31.31 | 0.887 | 0.175 | 3.97 |
| VAE | 0.0281 | 29.40 | 0.825 | 0.281 | 17.41 |
| SD-VAE | 0.0223 | 29.85 | 0.853 | 0.099 | 1.89 |
| **VMAE** | **0.0221** | **31.52** | **0.890** | **0.062** | **0.89** |

VMAE achieves state-of-the-art performance at both pixel level (PSNR 31.52 vs. SD-VAE's 29.85) and perceptual level (LPIPS 0.062 vs. 0.099; rFID 0.89 vs. 1.89).

### Ablation Study: Contribution of Each Loss Term

| Loss Combination | PSNR↑ | SSIM↑ | rFID↓ | LPIPS↓ | gFID↓ |
|-----------------|-------|-------|-------|--------|-------|
| Baseline (MSE only) | 32.18 | 0.906 | 6.21 | 0.172 | 12.92 |
| + Masking loss $\mathcal{L}_M$ | 32.01 | 0.913 | 4.66 | 0.130 | 8.92 |
| + Latent regularizer $\mathcal{L}_{reg}$ | 31.14 | 0.881 | 1.59 | 0.112 | 6.32 |
| + Perceptual loss $\mathcal{L}_P$ | **31.52** | **0.889** | **0.89** | **0.062** | **5.98** |

Each component contributes distinctly: masking loss introduces hierarchical compression (gFID −4.0) → latent regularization provides smoothness (gFID −2.6) → perceptual loss recovers visual details (LPIPS −0.050).

### Model Efficiency Comparison

| Metric | AE/DAE/VAE/SD-VAE | VMAE |
|--------|-------------------|------|
| Model Size | 319.7 MB | 42.7 MB (13.4%) |
| GFLOPs | 17,331.3 | 703.9 (4.1%) |
| Training Time | 24 hr | 9 hr (37.5%) |

VMAE substantially outperforms prior autoencoders in parameter count, computational cost, and training time, while also reducing encoding overhead per LDM training iteration due to faster inference.

## Highlights & Insights

1. **Systematic Analysis Framework**: The proposed three-property framework (smoothness / compression / reconstruction) and its quantitative evaluation methodology provide a clear assessment system for autoencoder design.
2. **Theoretical Insight on Hierarchical Compression**: SD-VAE's over-compression (object-level clustering with entangled internal features) is identified as the root cause of its reconstruction detail loss — a finding highly valuable for understanding LDM bottlenecks.
3. **Remarkable Efficiency Gains**: Using only 4.1% of the GFLOPs, VMAE surpasses SD-VAE, attributed to the efficient patchification and lightweight design of the ViT architecture.
4. **Lessons from VAE**: The smoothest latent space (VAE) yields the worst generation quality, demonstrating that smoothness is necessary but not sufficient — reconstruction quality is equally critical.

## Limitations & Future Work

- Experiments are conducted primarily at 256×256 resolution; high-resolution scenarios remain unvalidated.
- Only the autoencoder in LDMs is replaced; the diffusion model backbone (DiT/UNet) is not co-optimized.
- MAE's random masking strategy may not be optimal for certain structured data types.
- Comparison with the latest VAE variants (e.g., FLUX/SD3 with 16 channels) is insufficient.

## Related Work & Insights

- **SD-VAE (StableDiffusion3)**: The standard autoencoder for current LDMs; VMAE surpasses it across all dimensions.
- **MAE**: Provides the foundation for hierarchical features; VMAE extends it to probabilistic encoding.
- **DC-AE**: Pursues high spatial compression ratios; its goals are complementary to VMAE.
- **VA-VAE**: Improves VAE by aligning with visual foundation models; represents an alternative enhancement direction.
- **Insight**: Autoencoders are not merely "preprocessing tools" for LDMs — the properties of their latent spaces directly determine diffusion training efficiency and generation quality, making them a component worthy of deep investment.

## Rating ⭐⭐⭐⭐

- Novelty: ⭐⭐⭐⭐ — The three-property analysis framework and the MAE–VAE combination represent a novel design direction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage including autoencoder property analysis, generation comparison, reconstruction comparison, ablation study, and efficiency evaluation.
- Value: ⭐⭐⭐⭐ — Can directly replace SD-VAE for LDM training with substantial efficiency improvements.
- Writing Quality: ⭐⭐⭐⭐⭐ — Analysis is thorough and systematic; visualizations such as radar charts are highly effective.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)
- [\[ICCV 2025\] HypDAE: Hyperbolic Diffusion Autoencoders for Hierarchical Few-shot Image Generation](hypdae_hyperbolic_diffusion_autoencoders_for_hierarchical_few-shot_image_generat.md)
- [\[NeurIPS 2025\] Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders](../../NeurIPS2025/image_generation/learning_interpretable_features_in_audio_latent_spaces_via_sparse_autoencoders.md)
- [\[NeurIPS 2025\] OmniCast: A Masked Latent Diffusion Model for Weather Forecasting Across Time Scales](../../NeurIPS2025/image_generation/omnicast_a_masked_latent_diffusion_model_for_weather_forecasting_across_time_sca.md)
- [\[ICCV 2025\] What's in a Latent? Leveraging Diffusion Latent Space for Domain Generalization](whats_in_a_latent_leveraging_diffusion_latent_space_for_domain_generalization.md)

<!-- RELATED:END -->
