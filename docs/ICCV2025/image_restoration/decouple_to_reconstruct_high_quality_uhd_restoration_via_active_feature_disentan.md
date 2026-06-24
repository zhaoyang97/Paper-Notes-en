---
title: >-
  [Paper Note] Decouple to Reconstruct: High Quality UHD Restoration via Active Feature Disentanglement and Reversible Fusion
description: >-
  [ICCV 2025][Image Restoration][UHD image restoration] This paper proposes D²R-UHDNet, a framework that employs a Controlled Differential Disentangled VAE (CD²-VAE) to actively decompose degraded images into a degradation-dominant latent space and background-dominant features, and processes the background features via a complex-domain invertible multi-scale fusion network. The method achieves state-of-the-art performance across six UHD restoration tasks with only 1M parameters…
tags:
  - "ICCV 2025"
  - "Image Restoration"
  - "UHD image restoration"
  - "VAE"
  - "feature disentanglement"
  - "invertible networks"
  - "complex-domain fusion"
date: 2026-05-08
content_hash: 430ae9884a689b4d
---

# Decouple to Reconstruct: High Quality UHD Restoration via Active Feature Disentanglement and Reversible Fusion

**Conference**: ICCV 2025
**arXiv**: [2503.12764](https://arxiv.org/abs/2503.12764)  
**Code**: N/A  
**Area**: Image Restoration / Ultra-High-Definition Image Processing
**Keywords**: UHD image restoration, VAE, feature disentanglement, invertible networks, complex-domain fusion

## TL;DR

This paper proposes D²R-UHDNet, a framework that employs a Controlled Differential Disentangled VAE (CD²-VAE) to actively decompose degraded images into a degradation-dominant latent space and background-dominant features, and processes the background features via a complex-domain invertible multi-scale fusion network. The method achieves state-of-the-art performance across six UHD restoration tasks with only 1M parameters.

## Background & Motivation

Ultra-high-definition (4K) image restoration presents two fundamental tensions:

**Computational efficiency vs. information preservation**: Processing images at native resolution incurs prohibitive computational costs. Existing methods (e.g., DreamUHD) transfer computation to a low-dimensional latent space via VAE to improve efficiency, but the compression process uncontrollably discards multi-band signals.

**Degradation removal vs. background preservation**: Degradation and background components are deeply entangled in corrupted images. The global compression of VAE causes high-frequency injection (HFI) strategies to inadvertently reintroduce residual degradation when compensating for lost information.

The core insight of this work is: rather than suppressing information loss by increasing VAE capacity, it is more effective to **guide the VAE to actively discard easily recoverable background information**, while encoding hard-to-recover degradation information into the latent space — an *Active Discarding-Targeted Restoration* paradigm.

## Method

### Overall Architecture

D²R-UHDNet is trained in three stages:
1. **Stage 1**: A CleanVAE is trained on clean images for image reconstruction.
2. **Stage 2**: Feature disentanglement constraints are introduced to train CD²-VAE, decomposing degraded inputs into a degradation-dominant latent code $z_{\text{deg}}$ and multi-scale background-dominant features $\{F_{\text{bg}}^l\}_{l=1}^L$.
3. **Stage 3**: A dual-path restoration network is constructed — LaReNet processes the degradation latent code, CIMF-Net processes the background features, and a decoder synthesizes the restored image.

### Key Designs

1. **Hierarchical Contrastive Disentanglement Learning (Hi-CDL)**: At each encoder layer $i$, a contrastive loss is constructed using cross-layer similarity between clean and degraded features:

$$\mathcal{L}_{\text{contrast}}^i = -\log \frac{\exp(s_{\text{pos}}^i / \tau_i)}{\exp(s_{\text{pos}}^i / \tau_i) + \exp(s_{\text{neg}}^i / \tau_i) + \epsilon}$$

The positive sample $s_{\text{pos}}^i = \text{sim}(E_{\text{deg}}^{i-1} - \text{UP}(E_{\text{deg}}^i), E_{\text{clean}}^{i-1})$ ensures that information discarded at each encoder layer is primarily background content, while the negative sample $s_{\text{neg}}^i = \text{sim}(E_{\text{deg}}^i, E_{\text{clean}}^i)$ reduces the divergence between degraded and clean features at the encoding output. The temperature $\tau_i$ decreases with depth, steering disentanglement from coarse to fine.

2. **Orthogonal Gating Projection Module (OrthoGate)**: An orthogonal pointwise convolution $W_{\text{ortho}}$ is constructed on the Stiefel manifold satisfying $W^\top W = I$, mathematically guaranteeing minimal mutual information between disentangled feature subspaces. OrthoGate operates in two steps:

    - **Channel disentanglement gating**: orthogonal convolution + depthwise convolution → global average pooling → Sigmoid → channel gating factor $C_{\text{gate}}$;
    - **Spatial disentanglement gating**: permute feature dimensions to bring spatial dimensions into the channel dimension → orthogonal convolution → restore original dimensions → spatial gating factor $S_{\text{gate}}$.

   Weight updates are performed on the Stiefel manifold using a Riemannian gradient strategy: $W_{\text{ortho}} \leftarrow \text{Retr}_{W_{\text{ortho}}}(-\eta \cdot \text{Proj}_{T_W \mathcal{W}}(\nabla_W \mathcal{L}))$.

3. **Complex-domain Invertible Multi-scale Fusion Network (CIMF-Net)**: Multi-scale background features are fused across scales via invertible computation. The first layer applies Real-GLOW to the largest-scale features; subsequent layers combine features passed from the previous layer with current-scale features as the real and imaginary parts of a complex number, feeding them into Complex-GLOW. Core components of Complex-GLOW include:

    - Complex-domain invertible 1×1 convolution (invertibility guaranteed via unitary matrix constraints);
    - Complex-domain ActNorm (separate normalization of real and imaginary parts);
    - Complex-domain affine coupling layers (scale/shift parameters predicted by complex convolutional networks).

   Invertibility ensures information consistency of background features throughout multi-scale fusion.

### Loss & Training

- Stage 1: Standard VAE reconstruction loss.
- Stage 2: Hi-CDL contrastive loss + reconstruction loss + OrthoGate orthogonality constraint.
- Stage 3: Pixel-level L1 loss + perceptual loss + SSIM loss.
- LaReNet uses SFHformer as a lightweight backbone, replaceable with Restormer/NAFNet, etc.

## Key Experimental Results

### Main Results

| Task | Dataset | Metric | D²R-UHD | Prev. SOTA | Gain |
|------|---------|--------|---------|------------|------|
| Low-light enhancement | UHD-LL | PSNR/SSIM | 27.94/0.934 | 27.72/0.928 (DreamUHD) | +0.22 |
| Dehazing | UHD-Haze | PSNR/SSIM | 25.37/0.955 | 24.69/0.952 (UHDDIP) | +0.68 |
| Deblurring | UHD-Blur | PSNR/SSIM | 29.84/0.861 | 29.51/0.858 (UHDDIP) | +0.33 |
| Demoiréing | UHDM | PSNR/SSIM | 23.92/0.851 | 23.24/0.843 (DreamUHD) | +0.68 |
| All-in-One (4 degradations) | Combined | Avg PSNR | 28.20 | 27.43 (DreamUHD) | +0.77 |

The model contains only **1.0M** parameters, significantly fewer than non-UHD-specific methods (Restormer 26.1M, PromptIR 33M), while supporting full 4K inference.

### Ablation Study

| Configuration | PSNR | SSIM | Note |
|---------------|------|------|------|
| Baseline VAE | 28.89 | 0.836 | Baseline |
| + Hi-CDL | 29.48 | 0.852 | Significant gain from contrastive disentanglement |
| + OrthoGate | 29.32 | 0.853 | Orthogonal gating also effective |
| + Hi-CDL + OrthoGate | **29.84** | **0.861** | Best synergistic performance |

**CIMF-Net Ablation** (deblurring task):

| Configuration | PSNR | Params (M) | Note |
|---------------|------|------------|------|
| Set1 (non-invertible) | 27.86 | 1.103 | Information loss due to non-invertibility |
| Set2 (independent INN) | 29.26 | 2.212 | No cross-scale interaction |
| Set3 (INN + Cross-Attn) | 29.52 | 4.289 | Excessive parameter count |
| CIMF-Net (complex-domain) | **29.84** | **1.008** | Best performance with fewest parameters |

### Key Findings

- Spectral residual map visualizations demonstrate that CD²-VAE preserves information across the full frequency spectrum (high/mid/low) more effectively than conventional downsampling and DreamUHD's HFI strategy.
- LaReNet is compatible with different backbones (Restormer +4.57 dB, NAFNet +3.44 dB, SFHformer +4.05 dB), validating the generality of the framework.

## Highlights & Insights

1. **Counterintuitive paradigm**: Rather than augmenting VAE capacity to reduce information loss, the method guides the VAE to actively select what to discard — converting a "deficiency" into a "feature."
2. **Manifold-constrained disentanglement**: The orthogonality constraint of OrthoGate on the Stiefel manifold provides a mathematical guarantee for feature disentanglement, beyond implicit regularization via loss functions alone.
3. **Complex-domain invertible fusion**: Encoding features from different scales as the real and imaginary parts of complex numbers is a novel approach that simultaneously enables cross-scale interaction and preserves information integrity.
4. **Extreme lightweight design**: Processing 4K images with only 1M parameters demonstrates genuine practical deployment value.

## Limitations & Future Work

- The three-stage training pipeline is relatively complex (CleanVAE → CD²-VAE → full restoration network); end-to-end training could be more efficient.
- Clean-degraded paired data are required to train the disentanglement mechanism of CD²-VAE.
- The All-in-One setting lacks adaptive design for different degradation types.
- The computational efficiency of complex-domain operations requires validation at larger scales.

## Related Work & Insights

- Compared to DreamUHD's high-frequency injection, the proposed feature disentanglement scheme addresses information loss more fundamentally.
- The GLOW invertible network is transferred from image generation to cross-scale feature fusion in image restoration.
- The application of Stiefel manifold orthogonality constraints in lightweight network design warrants further attention.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The "active discarding" paradigm and complex-domain invertible fusion are proposed for the first time in UHD restoration.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Six tasks, All-in-One setting, detailed ablations, and visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clearly articulated; methodology is systematically presented.
- **Value**: ⭐⭐⭐⭐ Strong practical utility with 1M-parameter SOTA; the feature disentanglement paradigm is generalizable to other compression-restoration scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Towards a Universal Image Degradation Model via Content-Degradation Disentanglement](towards_a_universal_image_degradation_model_via_content-degradation_disentanglem.md)
- [\[NeurIPS 2025\] Real-World Adverse Weather Image Restoration via Dual-Level Reinforcement Learning with High-Quality Cold Start](../../NeurIPS2025/image_restoration/real-world_adverse_weather_image_restoration_via_dual-level_reinforcement_learni.md)
- [\[NeurIPS 2025\] Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement](../../NeurIPS2025/image_restoration/latent_harmony_synergistic_unified_uhd_image_restoration_via_latent_space_regula.md)
- [\[CVPR 2025\] Degradation-Aware Feature Perturbation for All-in-One Image Restoration](../../CVPR2025/image_restoration/degradation-aware_feature_perturbation_for_all-in-one_image_restoration.md)
- [\[NeurIPS 2025\] DynaGuide: Steering Diffusion Policies with Active Dynamic Guidance](../../NeurIPS2025/image_restoration/dynaguide_steering_diffusion_polices_with_active_dynamic_guidance.md)

</div>

<!-- RELATED:END -->
