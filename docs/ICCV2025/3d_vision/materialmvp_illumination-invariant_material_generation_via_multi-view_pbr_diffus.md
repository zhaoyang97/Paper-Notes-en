---
title: >-
  [Paper Note] MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion
description: >-
  [ICCV 2025][3D Vision][PBR texture] MaterialMVP is an end-to-end multi-view PBR texture generation model that decouples illumination via consistency-regularized training and employs a dual-channel material generation framework (MCAA + Learnable Material Embeddings) to align albedo and metallic-roughness maps, enabling single-pass generation of high-quality, illumination-invariant, multi-view-consistent PBR materials from a 3D mesh and an image prompt.
tags:
  - ICCV 2025
  - 3D Vision
  - PBR texture
  - multi-view diffusion
  - illumination invariance
  - material generation
  - dual-channel
date: 2026-05-08
content_hash: 1c3e07c5d8b15673
---

# MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion

**Conference**: ICCV 2025
**arXiv**: [2503.10289](https://arxiv.org/abs/2503.10289)
**Code**: [GitHub](https://github.com/ZebinHe/MaterialMVP)
**Area**: 3D Vision / PBR Material Generation
**Keywords**: PBR texture, multi-view diffusion, illumination invariance, material generation, dual-channel

## TL;DR

MaterialMVP is an end-to-end multi-view PBR texture generation model that decouples illumination via consistency-regularized training and employs a dual-channel material generation framework (MCAA + Learnable Material Embeddings) to align albedo and metallic-roughness maps, enabling single-pass generation of high-quality, illumination-invariant, multi-view-consistent PBR materials from a 3D mesh and an image prompt.

## Background & Motivation

**State of the Field**: PBR texture generation is a core task in 3D asset creation. Existing approaches fall into two categories: (1) SDS-based optimization methods (Text2Tex, Paint-it, etc.), which yield high quality but require minutes of inference; and (2) generative methods (SuperMat, RGB↔X), which are fast but limited to single-view inputs or lack precise cross-channel alignment.

**Limitations of Prior Work**:

1. Optimization-based methods such as TextureDreamer and HyperDreamer are computationally expensive and unsuitable for large-scale production.

2. Single-view generation methods cannot guarantee multi-view consistency and are prone to seams and the Janus effect.

3. CLAY uses IP-Adapter to incorporate reference images, but the alignment between generated textures and the input remains insufficiently precise.

4. Diffusion model outputs tend to "bake" the illumination information from reference images, producing non-physical artifacts.

**Core Idea**: Construct an end-to-end multi-view PBR diffusion framework that simultaneously addresses illumination disentanglement, multi-channel alignment, and reference fidelity through three key designs: consistency-regularized training, dual-channel material generation, and reference attention.

## Method

### Overall Architecture

Input: 3D mesh (normal maps + position maps encoded into latent space and concatenated with noise) + reference image → Reference Attention extracts reference information → U-Net dual-channel parallel denoising → consistency-regularized training enforces illumination invariance → Output: 6-view PBR materials (albedo / metallic / roughness). Initialized from the SD 2.1 ZSNR checkpoint; optimized with AdamW, lr=$5 \times 10^{-5}$, 2000-step warmup, ~180 GPU-days.

### Key Designs

1. **Consistency-Regularized Training**

    - Design Motivation: Addresses view sensitivity and illumination entanglement — small camera pose changes lead to drastically different material outputs, and reference image illumination is "baked" into the output.
    - Core Idea: Each training step uses a pair of reference images $(I_1, I_2)$ that differ slightly in viewpoint/illumination but are required to produce identical network outputs.
    - Reference pair selection: Image pairs at adjacent azimuths ($\pm 15°$) are sampled from 312 renderings (4 elevations × 24 azimuths × multiple lighting conditions).
    - Training loss: $\mathcal{L} = (1-\lambda)\mathcal{L}_{pbr} + \lambda\mathcal{L}_{cons}$, where $\mathcal{L}_{cons} = \mathbb{E}_t[\|\epsilon_t^1 - \epsilon_t^2\|_2^2]$, $\lambda=0.1$.
    - Effect: Forces the model to learn illumination-invariant representations, eliminating the influence of input illumination on the output materials.

2. **Dual-Channel Material Generation + MCAA**

    - **Multi-Channel Aligned Attention (MCAA)**: The albedo channel retains standard cross-attention $\text{Attn}_{albedo} = \text{Softmax}(Q_{albedo}K_{ref}^T/\sqrt{d}) \cdot V_{ref}$; the MR channel is not conditioned directly on the reference image (due to large distribution gap) but instead inherits spatial information from the albedo channel via a residual connection: $z_{MR}^{new} = z_{MR} + \text{Attn}_{albedo}$.
    - **Learnable Material Embeddings**: Separate $16 \times 1024$ learnable embeddings are introduced for albedo and MR respectively, injected into each channel via cross-attention to capture the distinct distributions of the two texture types.
    - Design advantage: No additional trainable parameters are introduced (only cross-attention is reused), avoiding the difficulty of semantic alignment between reference images and MR maps.

### Loss & Training

- PBR loss: $\mathcal{L}_{pbr} = \mathbb{E}_{\epsilon, t}[\|\epsilon - \epsilon_t^1\|_2^2]$
- Consistency loss: $\mathcal{L}_{cons} = \mathbb{E}_t[\|\epsilon_t^1 - \epsilon_t^2\|_2^2]$, $\lambda = 0.1$
- Training data: 70,000 Objaverse/Objaverse-XL 3D assets, each rendered at 4 elevations × 24 azimuths, 512×512 resolution.
- Each step samples 6 PBR image groups at the same elevation + 2 reference images.

## Key Experimental Results

### Main Results

Quantitative comparison on 176 Objaverse evaluation objects:

| Method | Condition | CLIP-FID↓ | FID↓ | CMMD↓ | CLIP-I↑ | LPIPS↓ |
|--------|-----------|-----------|------|-------|---------|--------|
| Text2Tex | Text | 31.83 | 187.7 | 2.738 | - | 0.1448 |
| SyncMVD | Text | 29.93 | 189.2 | 2.584 | - | 0.1411 |
| Paint-it | Text | 33.54 | 179.1 | 2.629 | - | 0.1538 |
| Paint3D (text) | Text | 30.17 | 185.7 | 2.755 | - | 0.1388 |
| Paint3D (image) | Image | 26.86 | 176.9 | 2.400 | 0.8871 | 0.1261 |
| TexGen | Text+Image | 28.23 | 178.6 | 2.447 | 0.8818 | 0.1331 |
| **MaterialMVP** | Image | **24.78** | **168.5** | **2.191** | **0.9207** | **0.1211** |

### Ablation Study

Qualitative ablation validating the contribution of each component:

| Ablation Setting | Effect |
|-----------------|--------|
| Two-stage method (RGB first, then material estimation) | Glass/metal surfaces appear plastic-like; error accumulation is severe |
| Without consistency loss ($\lambda=0$) | Metallic channel frequently over-predicts; multiple material types are erroneously assigned metallic appearance |
| Without MCAA (standard weight sharing) | Spatial misalignment between albedo and MR; texture details are blurred in fine-grained regions |

### Key Findings

- MaterialMVP outperforms all baselines across all five metrics; FID is reduced by 8.4 and CLIP-I improved by 3.4% relative to Paint3D (image).
- The consistency loss is critical for eliminating illumination artifacts — removing it causes severe metallic over-prediction.
- End-to-end generation substantially outperforms two-stage approaches due to error accumulation in the latter.
- MCAA avoids semantic alignment difficulties in the MR channel by using residual connections rather than direct reference conditioning.

## Highlights & Insights

- The dual-reference consistency regularization is an elegant design: reference pairs with subtle differences compel the model to "ignore" illumination variation, effectively implementing invariance learning driven by data augmentation.
- MCAA avoids forcing cross-attention between albedo and MR channels where the distribution gap is large, instead using residual connections for implicit alignment — a practical design that introduces no additional parameters.
- Single-pass end-to-end generation of a complete PBR material set (including metallic/roughness) offers substantially greater practical utility than SDS-based methods requiring minutes of optimization.

## Limitations & Future Work

- Quantitative evaluation covers only 176 objects, limiting the scale of assessment; ablation experiments are purely qualitative.
- Training cost is high (~180 GPU-days); inference time is not reported.
- Generalization to out-of-distribution 3D assets (e.g., scanned data) is not evaluated.
- The MR channel relies entirely on spatial information from the albedo channel, which may propagate errors when albedo estimation is inaccurate.

## Related Work & Insights

- **vs. CLAY**: CLAY employs IP-Adapter, resulting in insufficient alignment precision; MaterialMVP achieves more accurate pixel-level alignment via Reference Attention + MCAA.
- **vs. Paint3D**: Paint3D is a single-view method; multi-view generation naturally eliminates seams and inconsistencies.
- **vs. SuperMat**: SuperMat's two-stage pipeline introduces error accumulation that degrades material estimation accuracy.
- **Insight**: Learnable Material Embeddings are inspired by IC-Light and merit broader application in other multi-channel generation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The consistency-regularized training and MCAA dual-channel design are genuinely novel.
- Experimental Thoroughness: ⭐⭐⭐ Quantitative results are comprehensively superior, but ablation studies are qualitative only.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear and visualizations are compelling.
- Value: ⭐⭐⭐⭐ A practical end-to-end PBR generation solution with direct applicability to 3D asset production pipelines.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] SuperMat: Physically Consistent PBR Material Estimation at Interactive Rates](supermat_physically_consistent_pbr_material_estimation_at_interactive_rates.md)
- [\[ICCV 2025\] SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models](spinmeround_consistent_multi-view_identity_generation_using_diffusion_models.md)
- [\[ICCV 2025\] MVGBench: a Comprehensive Benchmark for Multi-view Generation Models](mvgbench_a_comprehensive_benchmark_for_multi-view_generation_models.md)
- [\[ICCV 2025\] Bootstrap3D: Improving Multi-view Diffusion Model with Synthetic Data](bootstrap3d_improving_multiview_diffusion_model_with_synthet.md)
- [\[ICCV 2025\] FlexGen: Flexible Multi-View Generation from Text and Image Inputs](flexgen_flexible_multi-view_generation_from_text_and_image_inputs.md)

<!-- RELATED:END -->
