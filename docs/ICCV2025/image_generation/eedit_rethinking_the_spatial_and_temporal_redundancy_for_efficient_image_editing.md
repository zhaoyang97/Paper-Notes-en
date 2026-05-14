---
title: >-
  [Paper Note] EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing
description: >-
  [Image Generation] This paper proposes EEdit, an efficient image editing framework that achieves an average 2.46× speedup without quality degradation across diverse editing tasks—including prompt-guided, drag-based…
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 17e7584577ed051d
---

# EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing

## Metadata
- **Conference**: ICCV 2025
- **arXiv**: [2503.10270](https://arxiv.org/abs/2503.10270)
- **Code**: [yuriYanZeXuan/EEdit](https://github.com/yuriYanZeXuan/EEdit)
- **Area**: Image Generation / Image Editing
- **Keywords**: Diffusion Model, Image Editing, Inversion, Cache Acceleration, Spatial Redundancy, Temporal Redundancy

## TL;DR

This paper proposes EEdit, an efficient image editing framework that achieves an average 2.46× speedup without quality degradation across diverse editing tasks—including prompt-guided, drag-based, and image composition editing—via three components: Spatial Locality Caching (SLoC) to skip computation in unedited regions, Token Index Preprocessing (TIP) for lossless acceleration of caching operations, and Inversion Step Skipping (ISS) to reduce inversion redundancy.

## Background & Motivation

- **Cost of inversion-based editing**: Current diffusion model editing follows a two-stage pipeline—inversion (mapping images to noise space) followed by denoising to produce edited results. Both stages each account for roughly half the total computation, leading to significant overhead.
- **Two categories of redundancy**:
    - **Spatial redundancy**: Editing typically modifies only a small region of the image, yet the pipeline computes over all pixels. Unedited regions exhibit very high cosine similarity between pre- and post-edit latents, contributing negligibly to the output.
    - **Temporal redundancy**: Redundancy in the inversion process greatly exceeds that in denoising. Experiments show that skipping inversion steps has almost no effect on editing quality, whereas skipping denoising steps rapidly causes texture and structural degradation.
- **Limitations of prior caching methods**:
    - Layer-level caching granularity is too coarse and ignores asymmetric importance at the token level.
    - Some methods require access to attention maps or storage of KV matrices, making them incompatible with FlashAttention.
    - Prior methods do not exploit task-specific priors (i.e., known edit region locations).

## Method

### Overall Architecture

EEdit is built upon FLUX-Dev (a 12B-parameter flow matching model) and introduces three training-free acceleration modules.

### 1. Spatial Locality Caching (SLoC)

**Core Idea**: During both denoising and inversion, full computation is performed for tokens within and adjacent to the edit region, while other tokens reuse cached features from the previous timestep.

**Score Bonus for edit regions**:

$$\mathbf{S_E}(\mathbf{x}) = \begin{cases} 1 + b \cdot r^k, & \mathbf{x} \in \mathcal{N}_k(M_s), k \in 0,1,...,K \\ 1, & \mathbf{x} \notin \bigcup_{k=0}^{K} \mathcal{N}_k(M_s) \end{cases}$$

- $b > 1$: bonus factor; $0 < r < 1$: decay ratio
- $\mathcal{N}_k(M_s)$: the neighborhood at L1 distance $k$ from the edit mask $M_s$
- Tokens in and near the edit region receive higher priority for full computation

**Cache frequency control**: A reuse counter $\mathcal{M}_{freq}$ tracks how many times each token has been served from cache; tokens reused more frequently are prioritized for refresh, preventing error accumulation while suppressing redundant recomputation.

Composite score: $\mathcal{S}_l \leftarrow (\mathcal{R} \odot \mathbf{S_E}) \oplus \mathcal{M}_{freq}$

The top-R% of tokens by score undergo full computation; the remainder are retrieved from cache.

### 2. Token Index Preprocessing (TIP)

**Key Insight**: The score update and token selection logic in SLoC can be converted from online to offline precomputation while remaining mathematically equivalent:

$$\mathcal{I}^{(t)}_{\text{topR\%}}(\text{offline}) = \mathcal{I}^{(t)}_{\text{topR\%}}(\text{online}) \quad \forall t \in [1...T]$$

Token indices for all timesteps are precomputed and stored; at inference time, only a single read operation is needed, eliminating score computation, sorting, and selection. This yields an additional measured speedup of over 15% with zero quality loss.

### 3. Inversion Step Skipping (ISS)

Inspired by DDIM, ISS introduces a skip interval $m$ during rf-inversion:

$$\mathbf{Z}_t \leftarrow \begin{cases} \mathbf{Z}_{t-1} & \text{if } t \bmod m \neq 1 \text{ and } m \neq T \\ \text{RF-inversion}(\mathbf{Z}_{t-1}, t-1, \phi) & \text{otherwise} \end{cases}$$

- Inversion steps can safely be reduced to 33.3% of denoising steps ($m=3$) with negligible quality degradation.
- The final inversion step is always computed in full to ensure noise quality.
- This finding is the first to reveal a **strong asymmetry** in importance between inversion and denoising.

## Key Experimental Results

### Main Results: Quality and Efficiency Comparison (PIE-Bench Prompt-Guided Editing)

| Method | Inversion | PSNR↑ | LPIPS↓ | SSIM↑ | CLIP-T↑ | FLOPs(T) | Time(s) |
|--------|-----------|-------|--------|-------|---------|----------|---------|
| P2P | DDIM | 17.87 | 20.88 | 0.72 | 25.13 | 334.4 | 18.75 |
| InfEdit | Inv-free | 28.11 | 5.61 | 0.85 | 25.86 | 124.6 | 2.90 |
| RF-inv | RF-inv | 17.74 | 24.40 | 0.66 | 26.31 | 1111.6 | 13.56 |
| RF-Edit | RF-Solver | 20.17 | 18.50 | 0.77 | 26.64 | 2223.2 | 26.23 |
| Flow-Edit | - | 22.20 | 10.49 | 0.85 | 25.80 | 952.8 | 11.84 |
| **SLoC** | RF-inv | **31.97** | **1.96** | **0.94** | 25.37 | 384.0 | 5.96 |
| **SLoC+ISS** | ISS | **31.97** | **1.95** | **0.94** | **25.38** | **264.5** | **4.60** |

- SLoC **decisively outperforms** all methods on background consistency (PSNR 31.97 vs. second-best 28.11).
- Achieves 2.95× speedup over full RF-inversion and 5.70× over RF-Edit.

### Ablation Study

**ISS Ablation**:

| Inversion Setting | Denoising Setting | BG LPIPS↓ | FG LPIPS↓ | FG PSNR↑ | Time(s) |
|-------------------|-------------------|-----------|-----------|----------|---------|
| Full step | Full step | 1.98 | - | - | 13.27 |
| 2-step skip | Full step | 1.98 | 5.46 | 43.77 | 10.16 |
| **3-step skip** | Full step | **1.98** | **5.29** | **43.99** | **9.31** |
| 4-step skip | Full step | 1.98 | 5.29 | 43.80 | 8.76 |

**TIP + ISS Cross-Task Ablation**:

| Task | TIP | ISS | FG FID↓ | FG PSNR↑ | Time(s) |
|------|-----|-----|---------|----------|---------|
| Prompt | × | × | 39.50 | 31.75 | 5.96 |
| Prompt | ✓ | ✓ | 39.21 | 31.76 | 4.60 |
| Dragging | × | × | 20.61 | 33.47 | 7.12 |
| Dragging | ✓ | ✓ | 22.07 | 33.68 | 5.66 |
| Composition | × | × | 12.33 | 39.78 | 7.25 |
| Composition | ✓ | ✓ | 12.35 | 39.80 | 5.66 |

TIP + ISS preserve performance across all tasks while delivering an additional average speedup of over 20%.

### Key Findings

- **Inversion redundancy far exceeds denoising redundancy**: Inversion steps can be reduced to 33% without quality loss, whereas denoising steps cannot—this is the first quantitative validation of this asymmetry.
- **SLoC leverages editing priors**: The known mask region serves as a natural, lossless guide for acceleration, surpassing general-purpose caching strategies.
- **Extreme speedup**: Up to 10.96× latency reduction compared to the state-of-the-art RF-Edit method.

## Highlights & Insights

1. **Clear problem decomposition**: Editing redundancy is decomposed into the orthogonal dimensions of spatial and temporal redundancy, each addressed independently.
2. **Training-free**: All modules are plug-and-play without any fine-tuning, readily applicable to diverse editing tasks.
3. **Mathematically equivalent preprocessing**: TIP's offline precomputation is proven strictly equivalent to the online version, constituting a genuinely lossless acceleration.
4. **Exploitation of editing priors**: Existing caching methods are designed for generation and ignore the mask priors available in editing tasks—this constitutes a key differentiator.
5. **First quantification of inversion/denoising asymmetry**: This finding carries direct implications for the design of future editing pipelines.

## Limitations & Future Work

- **Mask dependency**: The method requires a user-provided edit region mask and is not directly applicable to global style editing without masks.
- **FLUX-12B only**: Main experiments are conducted solely on FLUX-Dev; results on SD-series models are provided only qualitatively.
- **Editing quality bounded by the base method**: EEdit accelerates existing editing pipelines; the editing fidelity itself is constrained by the underlying method.
- **Manual cache ratio R%**: Adaptive cache ratio strategies are not explored.

## Related Work & Insights

- Layer-level caching methods such as DeepCache and L2C inspire the caching design; EEdit innovates by introducing token-level caching guided by editing priors.
- Token-wise cache methods such as Toca and Duca are closely related recent works, but are not tailored for editing tasks.
- RF-inversion serves as the base inversion method in this work; ISS further accelerates it.
- The asymmetric importance analysis presented here may generalize to video editing, and similar analyses of forward vs. backward pass redundancy warrant further investigation.

## Rating ⭐⭐⭐⭐

The framework is elegantly designed, with three modules that each address a well-defined problem and compose multiplicatively. The dual-axis analysis of spatial and temporal redundancy is novel, experiments span three task types (prompt / drag / composition) with substantial speedups, and the mathematical equivalence proof for TIP demonstrates solid theoretical rigor. Weaknesses include the mask dependency and insufficient validation on a broader set of base models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Rethinking Layered Graphic Design Generation with a Top-Down Approach](rethinking_layered_graphic_design_generation_with_a_top-down_approach.md)
- [\[ICCV 2025\] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities](rethinking_the_embodied_gap_in_vision-and-language_navigation_a_holistic_study_o.md)
- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[ICCV 2025\] FlowTok: Flowing Seamlessly Across Text and Image Tokens](flowtok_flowing_seamlessly_across_text_and_image_tokens.md)

</div>

<!-- RELATED:END -->
