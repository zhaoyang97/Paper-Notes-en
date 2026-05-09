---
title: >-
  [Paper Note] Balanced Image Stylization with Style Matching Score
description: >-
  [ICCV 2025][Image Generation][Style Transfer] This paper proposes Style Matching Score (SMS), which recasts image stylization as a style distribution matching problem. Through progressive spectrum regularization and semantic-aware gradient refinement, SMS achieves a superior balance between style alignment and content preservation, and can be distilled into a lightweight feed-forward network for one-step stylization.
tags:
  - ICCV 2025
  - Image Generation
  - Style Transfer
  - Score Distillation
  - diffusion model
  - LoRA
  - Frequency Domain Regularization
date: 2026-05-08
content_hash: eb85ed1ed6f31b25
---

# Balanced Image Stylization with Style Matching Score

**Conference**: ICCV 2025
**arXiv**: [2503.07601](https://arxiv.org/abs/2503.07601)
**Code**: [https://github.com/showlab/SMS](https://github.com/showlab/SMS)
**Area**: Image Generation / Style Transfer
**Keywords**: Style Transfer, Score Distillation, diffusion model, LoRA, Frequency Domain Regularization

## TL;DR

This paper proposes Style Matching Score (SMS), which recasts image stylization as a style distribution matching problem. Through progressive spectrum regularization and semantic-aware gradient refinement, SMS achieves a superior balance between style alignment and content preservation, and can be distilled into a lightweight feed-forward network for one-step stylization.

## Background & Motivation

The central challenge in image stylization is balancing effective style transfer with content preservation. Existing methods each exhibit distinct shortcomings:

**Zero-shot text-driven methods** (e.g., FreeStyle): Text struggles to precisely describe complex styles — "a picture is worth a thousand words."

**Single-exemplar-driven methods** (e.g., StyleID): Over-reliance on a single style image tends to produce texture overlay rather than genuine style transfer.

**Set-based fine-tuning methods** (e.g., Style-LoRA + ControlNet): While capable of capturing style distributions, the edge conditioning in ControlNet limits content preservation.

**SDS-based editing methods** (e.g., DDS): Lack explicit identity preservation mechanisms, making them prone to diverging from the source image.

**Core Idea of SMS**: A style-LoRA-integrated diffusion model is used to estimate the score function of the target style distribution. Style matching is achieved by minimizing the KL divergence between the generated distribution and the style distribution, while frequency-domain regularization and semantic gradient correction preserve content.

## Method

### Overall Architecture

SMS comprises three core components:
1. **Style Matching Objective**: Style distribution matching via score distillation
2. **Progressive Spectrum Regularization**: Frequency-domain progressive regularization for content protection
3. **Semantic-Aware Gradient Refinement**: Semantics-guided gradient correction

### Key Designs

1. **Style Matching Objective**

    - A parameterized generator $G_\theta$ (which may represent the image itself or the parameters of a feed-forward network)
    - Minimizes the KL divergence between the generated distribution $p_{G_\theta}$ and the target style distribution $p_{style}$
    - Two noise prediction models are used to estimate the score functions:
        - $\epsilon_{style}$: a pretrained diffusion model integrated with style LoRA (frozen), estimating $p_{style}$
        - $\epsilon_{fake}^\phi$: dynamically learned, estimating the current $p_{G_\theta}$
    - KL divergence gradient:
    $\nabla_\theta D_{KL} \approx \mathbb{E}_{t,\epsilon}[w_t(\epsilon_{style}(z_t^{tgt}; y_{src}, t) - \epsilon_{fake}^\phi(z_t^{tgt}; y_{src}, t))\frac{\partial G_\theta}{\partial \theta}]$
    - Distinction from DMD: DMD employs a generic pretrained model $\epsilon_{real}$; SMS replaces it with the style LoRA model $\epsilon_{style}$

2. **Progressive Spectrum Regularization**

    - Observation: The primary difference between real and stylized images lies in high-frequency components.
    - Core Idea: At high-noise timesteps (large $t$), more low-frequency components are protected to preserve structure; at low-noise timesteps (small $t$), high-frequency modifications are permitted to introduce style details.
    - Loss: $L_{freq} = \|\mathcal{F}_{low}(z_0^{tgt}, t) - \mathcal{F}_{low}(z_0^{src}, t)\|_2^2$
    - Where $\mathcal{F}_{low}(z, t) = \text{LPF}(\text{DCT}(z), \text{thld}(t))$
    - The cutoff frequency $\text{thld}(t)$ decreases with $t$: broad protection (strict) at large $t$, narrow protection (relaxed) at small $t$
    - More elegant than spatial-domain regularization, which either under-constrains or over-constrains content preservation

3. **Semantic-Aware Gradient Refinement**

    - Design Motivation: Different pixels require different degrees of stylization — foreground subjects warrant stronger stylization, while backgrounds require less.
    - A relevance map is computed using the semantic prior of the diffusion model:
    $\mathcal{R}(z_t^{src}, t) = \text{Norm}(|\epsilon_{real}(z_t^{src}; y_{edit}, t) - \epsilon_{real}(z_t^{src}; y_\emptyset, t)|)$
    - $y_{edit}$ = "Turn it into {target style}"; the difference from the unconditional prediction highlights semantically important regions.
    - Applied as an element-wise weight to modulate the gradient: emphasizing style in semantically important regions while suppressing changes in non-critical areas.
    - $\mathcal{R}$ is adaptive and timestep-dependent, adjusting naturally throughout the diffusion process.

### Loss & Training

Overall objective: $L_{SMS} = L_{style} + \lambda \cdot L_{freq}$

Where $L_{style}$ incorporates semantic weighting:
$$L_{style} = \mathbb{E}_{t,\epsilon}[\|\mathcal{R}(z_t^{src}, t) \odot w_t (\epsilon_{style} - \epsilon_{fake}^\phi)\|_2^2]$$

**Adaptive Narrowing Sampling Strategy**:
- The upper bound of the timestep sampling range is progressively reduced as training proceeds:
- $t \sim \mathcal{U}(t_{min}, t_{upper})$, $t_{upper} = (1 - \frac{iter_{cur}}{iter_{total}}) \cdot t_{max}$
- Avoids the inconsistent regularization strength of uniform sampling and the content drift caused by linear annealing.

**Feed-Forward Extension**:
- A lightweight network $G_\theta$ (~43 MB) replaces per-image optimization.
- Training proceeds with reconstruction warm-up, per-batch variable-timestep sampling, and the SMS loss.

## Key Experimental Results

### Main Results (Single-Image Stylization — Ghibli Style)

| Method | LPIPS ↓ | FID ↓ | ArtFID ↓ | PickScore ↑ |
|------|---------|-------|----------|-------------|
| FreeStyle | 0.690 | 12.361 | 22.582 | 0.683 |
| StyleID | 0.608 | 19.007 | 32.169 | 0.405 |
| InstantStyle+ | 0.538 | 14.949 | 24.532 | 1.019 |
| Style-LoRA | 0.438 | 12.267 | 19.077 | 2.067 |
| DDS | 0.513 | 15.233 | 24.554 | 0.537 |
| **SMS** | **0.326** | **13.089** | **18.686** | 1.487 |

User preference study (300 comparisons):

| Metric | FreeStyle | StyleID | InstantStyle+ | Style-LoRA | DDS | **SMS** |
|------|-----------|---------|---------------|-----------|-----|---------|
| Style | 0.060 | 0.147 | 0.083 | 0.100 | 0.033 | **0.577** |
| Content | 0.003 | 0.127 | 0.136 | 0.090 | 0.017 | **0.627** |
| Overall | 0.013 | 0.110 | 0.127 | 0.077 | 0.020 | **0.653** |

### Ablation Study

| Configuration | LPIPS ↓ | ArtFID ↓ | Note |
|------|---------|----------|------|
| Style matching only (no $L_f$, no $\mathcal{R}$) | 0.703 | 26.403 | Introduces noise and spurious details |
| + Spectrum regularization (no $\mathcal{R}$) | 0.505 | 24.132 | Improved structure but high-frequency artifacts remain |
| + Semantic refinement (no $L_f$) | 0.536 | 27.514 | Selective stylization but lacks direct constraint |
| **Full SMS** | **0.326** | **18.686** | Optimal balance |
| Random $t$ sampling | 0.389 | 32.936 | Blurry images |
| Linear annealing $t$ | 0.408 | 23.524 | Local identity drift |

Feed-forward stylization (~43 MB model, real-time inference):

| Method | LPIPS ↓ | ArtFID ↓ |
|------|---------|----------|
| Scenimefy | 0.422 | 18.561 |
| DDS | 0.321 | 18.338 |
| PDS | 0.427 | 24.590 |
| **SMS** | **0.268** | **17.079** |

### Key Findings

- SMS substantially outperforms baselines in content preservation (LPIPS) while maintaining competitive style alignment (FID).
- SMS achieves the best overall performance on the ArtFID = (LPIPS+1)·(FID+1) composite metric, demonstrating superior balance.
- Spectrum regularization and semantic refinement are each individually effective; their combination yields the best results.
- Adaptive narrowing sampling outperforms both uniform sampling and linear annealing.
- The feed-forward variant is equally effective, validating the scalability of SMS from pixel space to parameter space.

## Highlights & Insights

- **Problem Reformulation**: Stylization is elevated to a distribution matching problem rather than a simple image transformation.
- **Elegant Frequency-Domain Regularization**: DCT combined with timestep-aware low-pass filtering offers greater flexibility than spatial-domain regularization.
- **Novel Use of Semantic Priors**: The difference between conditional and unconditional diffusion predictions is repurposed as a semantic importance map.
- **Unified Pixel-to-Parameter Framework**: A single framework supports both per-image optimization and batch-trained feed-forward models.

## Limitations & Future Work

- Relies on off-the-shelf Style-LoRA; the quality of the LoRA directly affects stylization performance.
- Per-image optimization still requires 500 iterations (~several minutes), leaving room for speed improvement.
- Built on SD 1.5; stronger backbones such as SDXL have not been explored.
- Video and 3D stylization are not addressed (the authors note NeRF/3DGS as future directions).

## Related Work & Insights

- Shares high-level motivation with SDS/VSD/DMD but focuses on style distillation rather than 3D generation.
- Comparison with DDS: DDS lacks explicit identity preservation and its noise denoising direction is not conditioned on the current optimized image.
- The frequency-domain regularization paradigm is generalizable to other tasks requiring content-style disentanglement (e.g., video editing, 3D texture generation).
- The effectiveness of Style-LoRA as a style representation is validated, inspiring broader LoRA-based applications.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of style distribution matching, spectrum regularization, and semantic gradient refinement is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensively covers quantitative, qualitative, user study, ablation, and feed-forward extension evaluations.
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear and visualizations are rich.
- Value: ⭐⭐⭐⭐ Highly practical, open-source, and applicable to diverse styles.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)
- [\[ICCV 2025\] Domain Generalizable Portrait Style Transfer](domain_generalizable_portrait_style_transfer.md)
- [\[ICCV 2025\] LUSD: Localized Update Score Distillation for Text-Guided Image Editing](lusd_localized_update_score_distillation_for_text-guided_image_editing.md)
- [\[NeurIPS 2025\] A Connection Between Score Matching and Local Intrinsic Dimension](../../NeurIPS2025/image_generation/a_connection_between_score_matching_and_local_intrinsic_dimension.md)
- [\[ICCV 2025\] AIComposer: Any Style and Content Image Composition via Feature Integration](aicomposer_any_style_and_content_image_composition_via_feature_integration.md)

<!-- RELATED:END -->
