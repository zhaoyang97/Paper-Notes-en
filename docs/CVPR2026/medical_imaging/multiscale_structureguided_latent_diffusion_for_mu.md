---
title: >-
  [Paper Note] Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation
description: >-
  [CVPR 2026][Medical Imaging][Multimodal MRI translation] This paper proposes MSG-LDM, a latent diffusion model-based framework for multimodal MRI translation. By explicitly disentangling style and structural information in the latent space and incorporating High-Frequency Injection Blocks (HFIB), Multi-Modal Structural Feature Fusion (MMSF), and Multi-Scale Structure Enhancement (MSSE) modules, the framework extracts modality-agnostic structural priors to guide diffusion denoising. MSG-LDM outperforms existing methods on the BraTS2020 and WMH datasets.
tags:
  - CVPR 2026
  - Medical Imaging
  - Multimodal MRI translation
  - latent diffusion model
  - style-structure disentanglement
  - multiscale feature enhancement
  - missing modality
date: 2026-05-08
content_hash: 36b5df9db304a025
---

# Multiscale Structure-Guided Latent Diffusion for Multimodal MRI Translation

**Conference**: CVPR 2026
**arXiv**: [2603.12581](https://arxiv.org/abs/2603.12581)
**Code**: [https://github.com/ziyi-start/MSG-LDM](https://github.com/ziyi-start/MSG-LDM)
**Area**: Medical Imaging / MRI Synthesis / Diffusion Models
**Keywords**: Multimodal MRI translation, latent diffusion model, style-structure disentanglement, multiscale feature enhancement, missing modality

## TL;DR
This paper proposes MSG-LDM, a latent diffusion model-based framework for multimodal MRI translation. By explicitly disentangling style and structural information in the latent space and incorporating High-Frequency Injection Blocks (HFIB), Multi-Modal Structural Feature Fusion (MMSF), and Multi-Scale Structure Enhancement (MSSE) modules, the framework extracts modality-agnostic structural priors to guide diffusion denoising. MSG-LDM outperforms existing methods on the BraTS2020 and WMH datasets.

## Background & Motivation
Multimodal MRI (T1, T2, T1CE, FLAIR) provides complementary information for brain tumor segmentation and lesion analysis. However, missing modalities are common in clinical settings due to long acquisition times, poor patient tolerance, and hardware limitations. Although diffusion model-based MRI synthesis methods outperform GANs, they still exhibit the following limitations:

1. **Anatomical inconsistency**: Conventional diffusion models lack structural awareness and may produce structural distortions when handling arbitrary missing-modality scenarios.
2. **High-frequency detail degradation**: High-frequency information such as edges and textures is easily lost during iterative denoising.
3. **Style-structure entanglement**: Different MRI sequences have distinct contrast styles (e.g., T1 with bright gray matter and dark white matter, T2 with the opposite), and modality-specific style information is entangled with structural content, limiting synthesis fidelity.

## Core Problem
How to effectively disentangle structural information from modality-specific style in multimodal MRI translation, and leverage complete structural priors—encompassing both low-frequency anatomical layout and high-frequency boundary details—to guide the diffusion process, so as to generate anatomically consistent and detail-preserving MRI images under arbitrary missing-modality conditions?

## Method

### Overall Architecture
MSG-LDM operates in the latent space of a VAE. Given multimodal inputs $\{X_j\}_{j=1}^M$:
- A partial masking strategy is first applied to simulate missing-modality scenarios.
- Each modality is equipped with an independent **structure encoder** $E_j^{str}$ (with HFIB), a **style encoder** $E_j^{sty}$, and a **reconstruction decoder** $D_j^{rec}$.
- All modalities share a single **segmentation decoder** $D_{seg}$ to ensure that structural features are modality-agnostic.
- Multiscale structural features are fused across modalities via MMSF and enhanced via MSSE to produce a unified structural representation $F_s$.
- $F_s$ serves as the conditioning signal to guide the LDM denoising process: $\mathcal{L}_{LDM} = \mathbb{E}\|\epsilon - \epsilon_\theta(z_t, t | F_s)\|^2$

### Key Designs

1. **HFIB (High-Frequency Injection Block)**: Inserted at each of the four scales within the structure encoder. A learnable dynamic Gaussian filter decomposes features into low- and high-frequency components: $S_j^l = C_l + (C_l - \mathcal{G}_{\theta_l}(C_l))$. High-frequency residuals (edges, textures) are re-injected into the features, enhancing structural detail while preserving global anatomical layout. Crucially, the Gaussian filter is learnable and input-adaptive rather than fixed.

2. **MMSF (Multi-Modal Structural Feature Fusion)**: At each scale, a Sigmoid-gated network computes attention weights $w_j \in [0,1]$ for each available modality's structural features, enabling adaptive weighted fusion: $F_l = \text{Fusion}(\sum_j w_j S_j^{(l)})$. This allows maximal aggregation of structural information from remaining modalities even when some are missing, with gating weights automatically down-weighting low-quality modality contributions.

3. **MSSE (Multi-Scale Structure Feature Enhancement)**: Shallow-scale (scales 1 to $L-1$) structural features are upsampled and projected to the highest scale, and a cross-attention mechanism enhances the high-level representation: $F_s = F_L + \alpha \cdot \text{Attn}(F_L, \sum_{l=1}^{L-1}\text{Up}(\text{Proj}(F_l)))$. The resulting unified structural representation captures both low-frequency global anatomy and high-frequency local detail, serving as the diffusion conditioning signal.

### Loss & Training
- **Segmentation loss** $\mathcal{L}_{seg}$: Auxiliary supervision via a shared segmentation decoder to enforce modality-agnostic structural features.
- **Style consistency loss** $\mathcal{L}_{sc}$: A contrastive BCE objective that pulls style features from the same modality closer ($T_{pq}=1$) and pushes those from different modalities apart ($T_{pq}=0$), scaled by a learnable temperature parameter.
- **Structure-aware loss** $\mathcal{L}_{sa}$: L1 reconstruction loss combined with frequency-domain SSIM computed on DCT-transformed magnitude spectra, jointly constraining pixel-level fidelity and global frequency distribution consistency.
- **Diffusion loss** $\mathcal{L}_{ldm}$: Standard LDM denoising loss.
- Total loss: $\mathcal{L}_{total} = \mathcal{L}_{seg} + \lambda_1 \mathcal{L}_{sc} + \lambda_2 \mathcal{L}_{sa} + \lambda_3 \mathcal{L}_{ldm}$
- Adam optimizer, lr=1e-4, batch size=9, 3×NVIDIA 4090, 100 epochs.

## Key Experimental Results

**WMH Dataset**:

| Method | T1 PSNR↑ | T1 SSIM%↑ | T1 Dice% | FLAIR PSNR↑ | FLAIR SSIM%↑ | FLAIR Dice% |
|------|---------|----------|----------|------------|-------------|-------------|
| MM-GAN | 27.66 | 93.68 | 0.801 | 26.88 | 92.78 | 0.576 |
| SynDiff | 28.42 | 94.53 | 0.810 | 27.89 | 93.56 | 0.582 |
| MISA-LDM | 28.86 | 95.23 | 0.813 | 28.10 | 94.65 | 0.588 |
| **MSG-LDM** | **29.16** | **96.80** | **0.818** | **28.38** | **95.55** | **0.595** |

**BraTS2020 Ablation (FLAIR reconstruction)**:

| Configuration | PSNR↑ | SSIM%↑ | Dice%↑ |
|------|-------|--------|--------|
| w/o Disentanglement+MMSF | 27.92 | 92.41 | 85.03 |
| w/o HFIB | 28.17 | 92.68 | 85.41 |
| w/o MSSE | 29.04 | 93.28 | 86.55 |
| w/o $\mathcal{L}_{sa}$ | 27.36 | 91.82 | 84.27 |
| w/o $\mathcal{L}_{sc}$ | 27.11 | 91.54 | 83.89 |
| **Full model** | **29.68** | **93.62** | **87.60** |

### Ablation Study
- **$\mathcal{L}_{sc}$ has the largest impact**: Removing it causes a 2.57 dB drop in PSNR (29.68→27.11), demonstrating that without style encoder constraints, severe style-structure entanglement occurs.
- **$\mathcal{L}_{sa}$ ranks second**: Removal leads to a 2.32 dB PSNR drop; frequency-domain SSIM is critical for structural preservation.
- **Disentanglement+MMSF is central**: Removal causes a 1.76 dB PSNR drop and a 2.57% Dice drop.
- **HFIB contributes substantially**: Removal leads to a 1.51 dB PSNR drop, confirming the importance of high-frequency injection for detail preservation.
- **MSSE has a relatively smaller but non-negligible contribution**: Removal causes only a 0.64 dB PSNR drop, yet still results in a 1.05% Dice decrease.

## Highlights & Insights
- **Structure-guided diffusion acceleration**: Fig. 1 intuitively demonstrates that incorporating structural priors yields cleaner and more structurally stable denoising results at equivalent timesteps, implying faster convergence.
- **Frequency-domain SSIM loss**: Computing SSIM in DCT space is an elegant design that simultaneously constrains frequency distribution and structural consistency.
- **Learnable high-frequency separation**: Using a data-driven dynamic Gaussian filter rather than a fixed high-pass filter allows more flexible adaptation to high-frequency patterns across different scales.
- **Shared segmentation decoder for modality-agnostic constraint**: Enforcing structural feature consistency across modalities via an auxiliary segmentation task is a simple yet effective supervision signal.

## Limitations & Future Work
- Validation is limited to BraTS2020 (369 cases) and WMH, both relatively small datasets.
- Processing is performed at the 2D slice level (192×192), without exploiting 3D volumetric continuity—extension to 2.5D or 3D is a natural direction.
- Each modality requires independent structure and style encoders, causing parameter count to scale linearly with the number of modalities.
- Only three baselines are compared (MM-GAN, SynDiff, MISA-LDM); comparisons with a broader set of recent methods are lacking.
- No comparison with Transformer-based diffusion architectures (e.g., DiT).

## Related Work & Insights
- **vs. MISA-LDM (MICCAI 2025)**: Also LDM-based for multimodal MRI synthesis with missing modality handling, but MISA-LDM's structural modeling is less explicit. MSG-LDM systematically addresses multiscale structure via HFIB+MMSF+MSSE, achieving a 1.57% SSIM improvement on WMH T1 (95.23→96.80).
- **vs. SynDiff**: SynDiff is an unsupervised adversarial diffusion translation method; MSG-LDM achieves superior structural fidelity owing to explicit structural prior conditioning.
- **vs. MM-GAN**: The GAN baseline is substantially outperformed across all metrics, with the performance gap widening as more modalities are missing.

**Transferable insights**:
- The style-structure disentanglement paradigm is broadly applicable to any medical imaging task involving domain adaptation.
- The combination of high-frequency injection and frequency-domain loss can be transferred to other medical image synthesis tasks (CT synthesis, PET synthesis, etc.).
- Using a shared segmentation decoder as a modality-agnostic constraint is a design pattern worth adopting in related work.

## Rating
- **Novelty**: ⭐⭐⭐ — Each module is well-motivated but not individually groundbreaking; the contribution is primarily a systematic combination of established techniques.
- **Experimental Thoroughness**: ⭐⭐⭐ — Two datasets with ablation studies, but only three baselines are compared and dataset scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, intuitive figures, and precise method descriptions.
- **Value**: ⭐⭐⭐⭐ — Addresses a clinically relevant problem of missing MRI modalities; open-source code facilitates reproducibility.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)
- [\[AAAI 2026\] CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis](../../AAAI2026/medical_imaging/cocolit_controlnet-conditioned_latent_image_translation_for_mri_to_amyloid_pet_s.md)
- [\[CVPR 2026\] PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation](pgr-net_prior-guided_roi_reasoning_network_for_brain_tumor_mri_segmentation.md)
- [\[CVPR 2026\] CLoE: Expert Consistency Learning for Missing Modality Segmentation](cloe_expert_consistency_learning_for_missing_modality_segmentation.md)
- [\[CVPR 2026\] Accelerating Stroke MRI with Diffusion Probabilistic Models through Large-Scale Pre-training and Target-Specific Fine-Tuning](accelerating_stroke_mri_with_diffusion_probabilist.md)

<!-- RELATED:END -->
