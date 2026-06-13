---
title: >-
  [Paper Note] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation
description: >-
  [CVPR 2026][Medical Imaging][MRI synthesis] This paper proposes MSG-LDM, which introduces a multiscale structure-style disentanglement mechanism into a latent diffusion model. Through high-frequency injection…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "MRI synthesis"
  - "latent diffusion model"
  - "structure guidance"
  - "style-structure disentanglement"
  - "missing modality"
date: 2026-05-08
content_hash: 50534cdf3a82fa50
---

# Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation

**Conference**: CVPR 2026
**arXiv**: [2603.12581](https://arxiv.org/abs/2603.12581)  
**Code**: [Available](https://github.com/ziyi-start/MSG-LDM)  
**Area**: Medical Imaging
**Keywords**: MRI synthesis, latent diffusion model, structure guidance, style-structure disentanglement, missing modality

## TL;DR

This paper proposes MSG-LDM, which introduces a multiscale structure-style disentanglement mechanism into a latent diffusion model. Through high-frequency injection, multimodal structural feature fusion, and structure-aware losses, MSG-LDM achieves multimodal MRI synthesis that preserves anatomical structures and fine-grained details under missing-modality scenarios.

## Background & Motivation

### 1. State of the Field
Multimodal MRI (T1, T2, T1CE, FLAIR) provides complementary anatomical and pathological information, and is widely used in brain tumor segmentation and lesion analysis. However, complete multimodal data are often unavailable in clinical settings due to lengthy acquisition times, poor patient tolerance, and equipment constraints.

### 2. Limitations of Prior Work
Diffusion models have surpassed GANs in MRI synthesis, yet existing methods still suffer from three issues: (1) potential distortion of anatomical structures; (2) degradation of high-frequency details (edges and textures); and (3) entanglement between structural information and modality-specific style, which limits synthesis fidelity and consistency.

### 3. Root Cause
Conventional diffusion models lack structure-awareness — as illustrated in Fig. 1 of the paper, structural reconstruction during standard diffusion denoising is unstable and inefficient. Explicit structural priors are needed to accelerate generation and maintain anatomical fidelity.

### 4. Starting Point
The paper explicitly disentangles modality-invariant structural features from modality-specific style features in the latent space, and injects structural priors into the diffusion process.

## Method

### Overall Architecture

MSG-LDM operates in the VAE latent space and consists of four core components:
1. Per-modality **structure encoder** $E_j^{\mathrm{str}}$ (with HFIB) + **style encoder** $E_j^{\mathrm{sty}}$ + **reconstruction decoder** $D_j^{\mathrm{rec}}$
2. A shared **segmentation decoder** $D_{\mathrm{seg}}$ across all modalities (enforcing modality-invariant structural features)
3. **Multimodal Structure Feature Fusion (MMSF)** + **Multiscale Structure Feature Enhancement (MSSE)** → unified structural representation $F_s$
4. A **latent diffusion model** conditioned on $F_s$ for denoising

Pipeline: partial available modalities → per-modality structure/style encoding → MMSF cross-modal fusion → MSSE multiscale enhancement → $F_s$-guided LDM denoising → synthesized missing modality.

### Key Designs

#### 1. **High-Frequency Injection Block (HFIB)**

**Function**: Enhances high-frequency structural information (edges, textures) at each scale of the structure encoder.

**Mechanism**: Given the content feature $C^l$ at layer $l$, a learnable dynamic Gaussian filter extracts the low-frequency component; the residual yields the high-frequency component $C_{\mathrm{high}}^l$, which is then re-injected into the original feature:

$$C_{\mathrm{high}}^l = C^l - \mathcal{G}_{\theta_l}(C^l), \quad S_j^l = C^l + C_{\mathrm{high}}^l$$

**Design Motivation**: ViT/CNN encoders tend to suppress high-frequency information, whereas edges and textures are diagnostically critical in medical images. Learnable dynamic Gaussian filters offer greater flexibility than fixed filters.

#### 2. **Multimodal Structure Feature Fusion (MMSF)**

**Function**: Fuses structural features from $M$ available modalities at each scale $l$.

**Mechanism**: Attention weights $w_j \in [0,1]$ for each modality are computed via a Sigmoid gating network; the weighted sum is passed through a learnable convolution to produce the fused feature:

$$F_l = \mathrm{Fusion}\left(\sum_{j=1}^{M} w_j S_j^{(l)}\right)$$

**Design Motivation**: Different modalities provide complementary structural information (e.g., T1 vs. FLAIR have different tissue sensitivity); adaptive weighting prevents any single modality from dominating.

#### 3. **Multiscale Structure Feature Enhancement (MSSE)**

**Function**: Injects multiscale structural information into the highest-level representation to form the unified structural representation $F_s$.

**Mechanism**: Lower-scale features ($F_1$ to $F_{L-1}$) are aligned to the highest scale via $1\times1$ convolution and upsampling, then used to enhance the top-level representation through cross-attention:

$$F_s = F_L + \alpha \, \mathrm{Attn}\left(F_L, \sum_{l=1}^{L-1} \mathrm{Up}(\mathrm{Proj}(F_l))\right)$$

**Design Motivation**: Lower scales capture global anatomical layout while higher scales retain fine-grained structure; cross-attention enables the top level to selectively draw structural guidance from lower levels.

### Loss & Training

Total loss: $L_{\text{total}} = L_{\text{seg}} + \lambda_1 L_{\text{sc}} + \lambda_2 L_{\text{sa}} + \lambda_3 L_{\text{ldm}}$

- **$L_{\text{seg}}$**: Auxiliary segmentation loss, enforcing modality-invariant structural features.
- **$L_{\text{sc}}$ (style consistency loss)**: Contrastive-style objective — pulling same-modality style features closer and pushing cross-modality style features apart, suppressing contamination of structural features by modality-specific style:

$$L_{\text{sc}} = -\frac{1}{(M \times B)^2} \sum_{p,q} [T_{pq} \log \sigma(z_{pq}) + (1-T_{pq}) \log \sigma(-z_{pq})]$$

- **$L_{\text{sa}}$ (structure-aware loss)**: $L_1$ reconstruction loss + frequency-domain SSIM loss (comparing amplitude spectrum consistency after DCT transform):

$$L_{\text{sa}} = L_{\text{rec}} + L_{\text{freq}}, \quad L_{\text{freq}} = 1 - \text{SSIM}(|\mathcal{D}(\hat{X}_j)|, |\mathcal{D}(X_j)|)$$

- **$L_{\text{ldm}}$**: Standard denoising diffusion loss.

Training configuration: PyTorch 2.1.0, Adam (lr=$1\times10^{-4}$), batch size 9, 3× NVIDIA 4090, 100 epochs.

## Key Experimental Results

### Main Results

**Table 1: BraTS2020 dataset ($\bar{M}=3$, three available modalities used to synthesize the fourth)**

| Method | T1 PSNR/SSIM | T2 PSNR/SSIM | T1CE PSNR/SSIM | FLAIR PSNR/SSIM |
|--------|-------------|-------------|---------------|----------------|
| MM-GAN | 27.35/92.32 | 27.85/93.18 | 28.65/94.19 | 27.95/92.95 |
| SynDiff | 28.95/93.34 | 29.36/93.95 | 30.65/94.86 | 29.62/93.23 |
| MISA-LDM | 29.01/93.86 | 29.66/94.12 | 30.68/95.62 | 29.66/93.28 |
| **MSG-LDM** | **30.26/94.37** | **30.33/94.38** | **31.35/96.29** | **29.68/93.62** |

**Table 2: WMH dataset**

| Method | FLAIR→T1 PSNR/SSIM | T1→FLAIR PSNR/SSIM |
|--------|---------------------|---------------------|
| MISA-LDM | 28.86/95.23 | 28.10/94.65 |
| **MSG-LDM** | **29.16/96.80** | **28.38/95.55** |

MSG-LDM achieves state-of-the-art performance across all settings. On BraTS2020, the average PSNR improvement is ~1 dB and SSIM improvement is ~0.5%.

### Ablation Study

| Configuration | PSNR | SSIM% | Dice% |
|---------------|------|-------|-------|
| w/o disentanglement+MMSF | 27.92 | 92.41 | 85.03 |
| w/o HFIB | 28.17 | 92.68 | 85.41 |
| w/o MSSE | 29.04 | 93.28 | 86.55 |
| w/o $L_{\text{sa}}$ | 27.36 | 91.82 | 84.27 |
| w/o $L_{\text{sc}}$ | 27.11 | 91.54 | 83.89 |
| **Full model** | **29.68** | **93.62** | **87.60** |

### Key Findings

1. **Style consistency loss contributes most**: Removing $L_{\text{sc}}$ causes a 2.57 dB drop in PSNR (29.68→27.11), indicating that style interference is the central challenge in MRI synthesis.
2. **Structure-aware loss is equally critical**: Removing $L_{\text{sa}}$ leads to a 3.33% drop in Dice, demonstrating that frequency-domain constraints are indispensable for structural consistency.
3. **HFIB improves detail fidelity**: Its removal causes a 0.94% SSIM drop, confirming that high-frequency injection is important for texture and edge quality.
4. **More available modalities yield better synthesis**: Quality improves consistently as the number of available modalities increases from 1 to 3.
5. **Structure guidance accelerates denoising**: As shown in Fig. 1, denoising with structural priors reconstructs clear structures at intermediate steps, whereas without priors the intermediate results remain blurry.

## Highlights & Insights

1. **Systematic design for style-structure disentanglement**: Disentanglement is enforced both at the encoding stage (separate encoders) and at the loss level ($L_{\text{sc}}$ repels style features; $L_{\text{sa}}$ preserves structure), forming a closed-loop design.
2. **HFIB is concise yet effective**: Learnable dynamic Gaussian filtering → residual high-frequency extraction → re-injection, with negligible additional parameters and plug-and-play applicability.
3. **Clever use of the shared segmentation decoder**: As an auxiliary task, it forces structural features to be modality-invariant — an indirect but effective regularization strategy.
4. **Novel frequency-domain SSIM loss**: Computing SSIM on DCT amplitude spectra simultaneously constrains global layout and frequency distribution.

## Limitations & Future Work

1. **Only evaluated on brain MRI**: Validated on BraTS2020 (tumors) and WMH (white matter hyperintensities); other anatomical regions and diseases remain untested.
2. **2D processing**: 3D MRI volumes are sliced into 2D patches (192×192), losing volumetric context.
3. **Fixed modality set**: Assumes a fixed modality set (T1/T2/T1CE/FLAIR); the framework cannot dynamically adapt to new modalities.
4. **Training cost**: Separate encoders/decoders per modality result in parameter counts that scale linearly with $M$.
5. **Limited downstream evaluation**: Dice only assesses segmentation; radiologist subjective evaluation and clinical diagnostic task validation are absent.

## Related Work & Insights

- **Evolution from GANs to diffusion models**: MM-GAN → SynDiff → MISA-LDM → MSG-LDM, showing continuous progress in structural fidelity through diffusion models.
- **Structure guidance paradigm**: Injecting structural priors into the generative process is key for medical image synthesis — unconstrained generation must be avoided in favor of anatomical consistency constraints.
- **Inspiration from frequency-domain losses**: The DCT + SSIM combination is potentially transferable to other medical image generation and super-resolution tasks.

## Rating

- Novelty: ⭐⭐⭐ — Individual components (HFIB, MMSF, MSSE) are moderately designed in isolation, but the overall system integration and completeness of the style-structure disentanglement framework are commendable.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Two datasets, multiple missing-modality scenarios, complete ablation study, and visualization.
- Writing Quality: ⭐⭐⭐⭐ — Method description is clear; ablation analysis is systematic.
- Value: ⭐⭐⭐ — An incremental advance in MRI synthesis; the core insight (structure-guided diffusion) has moderate generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis](../../AAAI2026/medical_imaging/cocolit_controlnet-conditioned_latent_image_translation_for_mri_to_amyloid_pet_s.md)
- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)
- [\[CVPR 2026\] PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)
- [\[CVPR 2026\] Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](accelerating_stroke_mri_with_diffusion_probabilist.md)
- [\[ICLR 2026\] Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation](../../ICLR2026/medical_imaging/adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation.md)

</div>

<!-- RELATED:END -->
