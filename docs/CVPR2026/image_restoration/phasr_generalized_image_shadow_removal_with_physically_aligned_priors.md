---
title: >-
  [Paper Note] PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors
description: >-
  [CVPR 2026][Image Restoration][Shadow Removal] PhaSR introduces a dual-level physically aligned prior framework: at the global level, PAN performs parameter-free Retinex decomposition to suppress color bias…
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "Shadow Removal"
  - "Retinex Decomposition"
  - "Differential Attention"
  - "Geometry-Semantic Prior Alignment"
  - "Ambient Light Normalization"
date: 2026-05-08
content_hash: abde7e7bad7940c0
---

# PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors

**Conference**: CVPR 2026
**arXiv**: [2601.17470](https://arxiv.org/abs/2601.17470)  
**Code**: [https://github.com/ming053l/PhaSR](https://github.com/ming053l/PhaSR)  
**Area**: Image Restoration
**Keywords**: Shadow Removal, Retinex Decomposition, Differential Attention, Geometry-Semantic Prior Alignment, Ambient Light Normalization

## TL;DR
PhaSR introduces a dual-level physically aligned prior framework: at the global level, PAN performs parameter-free Retinex decomposition to suppress color bias; at the local level, GSRA employs differential attention to align DepthAnything depth priors with DINO-v2 semantic embeddings. This enables generalized shadow removal spanning from single-source direct illumination to multi-source ambient lighting scenes, achieving state-of-the-art performance on WSRD+ and Ambient6K with the lowest FLOPs.

## Background & Motivation

**Background**: Shadow removal is a fundamental computer vision task. The core challenge lies in accurately distinguishing shadows from intrinsically dark object regions and performing physically plausible color correction. Learning-based approaches have progressed from CNNs to Transformers to diffusion models, but most are evaluated only on single-source direct-illumination benchmarks.

**Limitations of Prior Work**: (1) Relying solely on RGB cues causes shadows to be confused with intrinsic material properties, resulting in color distortion near texture boundaries. (2) Existing methods perform well on single-source benchmarks but generalize poorly to multi-source indoor ambient lighting scenarios characterized by color shifts and diffuse indirect illumination. (3) Conventional encoder-decoder architectures fail to effectively propagate physical priors; uniform feature fusion ignores spatially varying degradation characteristics, leading to edge blurring.

**Key Challenge**: Prior misalignment. Geometric features (local shading variations, surface normals) are sensitive to illumination geometry but noisy, while semantic features (object categories, materials) are stable across illumination conditions but spatially coarse. Without proper alignment, geometric noise corrupts semantic consistency, or semantic over-smoothing erases illumination boundaries — a problem particularly severe under indirect illumination.

**Goal**: (1) Suppression of global color bias; (2) Resolution of cross-modal conflicts between geometric and semantic priors; (3) Generalization from single-source to multi-source lighting scenarios.

**Key Insight**: Unifying the problem from an "alignment" perspective — global-level alignment (PAN for illumination-reflectance decomposition) and local-level alignment (GSRA using differential attention to reconcile geometry and semantics).

**Core Idea**: Through dual-level physically aligned priors — global parameter-free Retinex normalization combined with local differential attention cross-modal correction — the shadow removal system generalizes from single-source to complex multi-source lighting scenes.

## Method

### Overall Architecture
PhaSR consists of two stages. Stage 1 is PAN — a preprocessing module with no learnable parameters — that performs Gray-world color normalization → log-domain Retinex decomposition → dynamic range recombination, yielding an illumination-consistent image. Stage 2 is a multi-scale Transformer encoder-decoder: frozen DINO-v2 semantic embeddings are injected at the encoder stage, DepthAnything-v2 geometric priors (depth and normals) are injected at the bottleneck, and GSRA cross-modal differential attention aligns the two priors. The entire pipeline requires no shadow mask.

### Key Designs

1. **Physically Aligned Normalization (PAN)**:

    - **Function**: Parameter-free preprocessing to suppress global color bias and provide illumination-consistent input.
    - **Mechanism**: A three-step pipeline — (a) **Gray-world color normalization**: $\mathbf{I}_{\text{norm}} = \mathbf{I} \cdot \frac{\mathbb{E}[\mathbf{I}]}{\mathbb{E}_c[\mathbf{I}]+\varepsilon}$, balancing channel illumination to remove color cast; (b) **Log-domain Retinex decomposition**: decomposing the image into reflectance and illumination components in the log domain — $\log\hat{\mathbf{S}} = \mathbb{E}_{H,W}[\log(\mathbf{I}_{\text{norm}}+\varepsilon)]$, $\log\hat{\mathbf{R}} = \log(\mathbf{I}_{\text{norm}}+\varepsilon) - \log\hat{\mathbf{S}}$ — exploiting the additive separability of the log domain for a closed-form solution; (c) **Recombination normalization**: $\hat{\mathbf{I}} = \frac{\hat{\mathbf{R}} \otimes \hat{\mathbf{S}} - \min}{\max - \min + \varepsilon}$.
    - **Design Motivation**: Unlike learned Retinex decomposition, PAN is a closed-form operation requiring no trainable parameters and can be embedded as a plug-and-play module in any framework. Experiments demonstrate that as a plugin it improves OmniSR, DenseSR, and other methods by 0.15–0.34 dB.

2. **Geometry-Semantic Rectification Attention (GSRA)**:

    - **Function**: Aligns depth geometric priors and DINO-v2 semantic embeddings to resolve cross-modal conflicts.
    - **Mechanism**: (a) **Multi-modal prior injection**: a shared query feature is added to geometric and semantic priors respectively (with learnable scaling factor $\alpha$) to generate modality-specific key-value pairs; (b) **Differential rectification**: two attention maps $\mathbf{A}_{\text{geo}}$ and $\mathbf{A}_{\text{sem}}$ are computed using the shared query, followed by rectification $\mathbf{A}_{\text{rect}} = \mathbf{A}_{\text{sem}} - \lambda \cdot \mathbf{A}_{\text{geo}}$, where learnable $\lambda$ balances illumination-change sensitivity against geometric regularization strength; (c) Final output: $\mathbf{F}_{\text{output}} = \text{Concat}(\mathbf{A}_{\text{rect}}\mathbf{V}_{\text{geo}}, \mathbf{A}_{\text{rect}}\mathbf{V}_{\text{sem}})$.
    - **Design Motivation**: Geometric features are precise at shadow boundaries but noisy in uniformly lit regions; semantic features are stable but spatially coarse. The subtractive structure of differential attention naturally realizes physically interpretable gating — preserving geometric precision at true illumination boundaries while suppressing geometric noise in uniform regions. Unlike the original DiffTransformer (subtraction within a single self-attention head), GSRA performs cross-modal subtraction.

3. **Multi-Scale Transformer Backbone**:

    - **Function**: Mask-free shadow removal encoder-decoder backbone.
    - **Mechanism**: A hierarchical architecture with base channel dimension $C=32$ and 2 layers per Transformer block. Semantic priors are injected via frozen DINO-v2 at the encoder stage; geometric priors are injected via DepthAnything-v2 at the bottleneck. GSRA aligns both priors at the bottleneck.
    - **Design Motivation**: Injecting physical priors at different network stages — semantics at encoding, geometry at the bottleneck — matches each prior to its most appropriate level of abstraction.

### Loss & Training
Total loss: $\mathcal{L}_{\text{total}} = 0.95\mathcal{L}_{\text{Charb}} + 0.05\mathcal{L}_{\text{SSIM}}$, combining Charbonnier loss for fidelity and SSIM loss for structural consistency. AdamW optimizer, batch size 9, 1400 training epochs, learning rate $2\times10^{-4}$ with cosine annealing.

## Key Experimental Results

### Main Results

| Dataset | Metric | PhaSR | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| ISTD | PSNR/SSIM | 30.73/0.960 | 30.64 (DenseSR) | +0.09 |
| ISTD+ | PSNR/SSIM | 34.48/0.960 | 35.19 (StableSD) | −0.71 |
| INS | PSNR/SSIM | 30.38/0.961 | 30.64 (DenseSR) | −0.26 |
| WSRD+ | PSNR/SSIM | **28.44/0.842** | 26.28 (DenseSR) | **+2.16** |
| Ambient6K | PSNR/SSIM | **23.32/0.834** | 22.54 (DenseSR) | **+0.78** |

Note: The largest gains are observed on the most challenging benchmarks, WSRD+ and Ambient6K (multi-source ambient lighting), validating the generalization capability of the proposed approach.

### Ablation Study (PAN as a Plug-and-Play Module)

| Framework + PAN | ISTD (PSNR) | WSRD+ (PSNR) | Ambient6K (PSNR) |
|------|---------|------|------|
| OmniSR | 30.45 → 30.60 | 26.07 → 26.22 | 23.01 → 23.15 |
| DenseSR | 30.64 → 30.98 | 26.28 → 26.47 | 22.54 → 22.73 |
| PhaSR (Ours) | 30.73 | 28.44 | 23.32 |

### Complexity Comparison

| Method | FLOPs (G) | Params (M) |
|------|----------|-----------|
| OmniSR | 118.67 | 21.02 |
| DenseSR | 109.32 | 24.70 |
| PhaSR | **55.63** | **18.95** |

### Key Findings
- **PhaSR substantially outperforms competing methods on ambient lighting scenes (Ambient6K)** — surpassing the dedicated ambient normalization method IFBlend (21.44 dB) by 1.88 dB, underscoring the critical role of physically aligned priors in multi-source scenarios.
- **PAN as a plug-and-play module** consistently improves multiple frameworks, with error reduction reaching up to 26.4% on the ISTD dataset.
- **PhaSR achieves the highest computational efficiency** — FLOPs of only 55.63G, approximately 47% of OmniSR and 51% of DenseSR, while also having the fewest parameters (18.95M).
- PhaSR falls below StableShadowDiffusion on ISTD+, though the latter is a diffusion-based method with substantially higher computational cost.
- PAN outperforms traditional color correction methods (ACE, White-balance, etc.) across all metrics.

## Highlights & Insights
- **The plug-and-play nature of PAN** is the most notable contribution — a parameter-free, closed-form preprocessing module that consistently improves various shadow removal methods by 0.15–0.34 dB, indicating that color bias at the input stage is a widespread but often overlooked problem. This module is transferable to any image restoration task.
- **GSRA's cross-modal differential attention**, $\mathbf{A}_{\text{sem}} - \lambda \cdot \mathbf{A}_{\text{geo}}$, offers strong physical interpretability: the semantic attention map serves as a "globally stable base," while the geometric attention map captures "locally illumination-sensitive perturbations." The subtraction operation corrects semantic over-smoothing while suppressing geometric noise. This cross-modal differential paradigm is transferable to other tasks requiring fusion of heterogeneous priors.
- **The generalization strategy from single-source to multi-source scenes** relies on physical alignment rather than data-driven scaling, offering a more principled alternative to simply enlarging training sets.

## Limitations & Future Work
- PAN is grounded in the Gray-world assumption and may introduce bias for images with highly non-uniform color distributions (e.g., large monochromatic backgrounds).
- The log-domain Retinex estimates illumination using a global mean, which cannot handle spatially complex lighting with strong variation (e.g., multi-directional spotlights).
- PhaSR underperforms diffusion-based methods on ISTD+, leaving room for improvement in fine texture recovery.
- The $\lambda$ parameter in GSRA is a globally learnable scalar; a spatially adaptive $\lambda(x, y)$ may yield superior results when handling locally complex shadow patterns.

## Related Work & Insights
- **vs. OmniSR**: OmniSR also leverages geometry-semantic priors but its fusion strategy fails to properly align complementary modal strengths; PhaSR explicitly rectifies this through differential attention.
- **vs. DenseSR**: DenseSR reformulates shadow removal as dense prediction with adaptive fusion, yet still falls significantly behind PhaSR on Ambient6K, demonstrating that fusion without physical alignment is insufficient in multi-source scenarios.
- **vs. ReHiT**: ReHiT employs Retinex-guided dual-branch decomposition for mask-free shadow removal, but performance degrades in ambient lighting scenes (only 19.98 dB on Ambient6K); PhaSR achieves better generalization through the combined PAN and GSRA design.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The dual-level physical alignment design (PAN + GSRA) is systematic and physically intuitive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers five benchmarks including ambient lighting scenes; PAN plug-in experiments, comparisons with traditional methods, and complexity analysis are all comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Physical motivation is clearly articulated and well supported by figures and tables.
- **Value**: ⭐⭐⭐⭐ PAN has broad applicability as a plug-and-play module; the cross-modal alignment paradigm of GSRA is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Winner of CVPR2026 NTIRE Challenge on Image Shadow Removal: Semantic and Geometric Guidance for Shadow Removal via Cascaded Refinement](shadow_removal_cascaded_refinement.md)
- [\[CVPR 2026\] Disentangled Textual Priors for Diffusion-based Image Super-Resolution](disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)
- [\[CVPR 2026\] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration](beyond_ground-truth_leveraging_image_quality_priors_for_real-world_image_restora.md)
- [\[CVPR 2026\] Flickerformer: A Duet of Periodicity and Directionality for Burst Flicker Removal](it_takes_two_a_duet_of_periodicity_and_directionality_for_burst_flicker_removal.md)
- [\[CVPR 2026\] UniBlendNet: Unified Global, Multi-Scale, and Region-Adaptive Modeling for Ambient Lighting Normalization](uniblendnet_unified_global_multi_scale_and_region_adaptive_modeling_for_ambient_lighting_normalization.md)

</div>

<!-- RELATED:END -->
