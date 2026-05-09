---
title: >-
  [Paper Note] DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers
description: >-
  [ICCV 2025][Image Generation][Diffusion Transformer] DiTFastAttnV2 is proposed for multi-modality diffusion Transformers (MMDiT), achieving fine-grained attention compression via Head-wise Arrow Attention and Head-wise Caching mechanisms. It reduces attention FLOPs by 68% and achieves 1.5× end-to-end speedup on 2K image generation without visual quality degradation.
tags:
  - ICCV 2025
  - Image Generation
  - Diffusion Transformer
  - Attention Compression
  - MMDiT
  - Sparse Attention
  - Inference Acceleration
date: 2026-05-08
content_hash: e8d1966491ac194b
---

# DiTFastAttnV2: Head-wise Attention Compression for Multi-Modality Diffusion Transformers

**Conference**: ICCV 2025
**arXiv**: [2503.22796](https://arxiv.org/abs/2503.22796)
**Code**: None
**Area**: Image Generation / Diffusion Model Acceleration
**Keywords**: Diffusion Transformer, Attention Compression, MMDiT, Sparse Attention, Inference Acceleration

## TL;DR

DiTFastAttnV2 is proposed for multi-modality diffusion Transformers (MMDiT), achieving fine-grained attention compression via Head-wise Arrow Attention and Head-wise Caching mechanisms. It reduces attention FLOPs by 68% and achieves 1.5× end-to-end speedup on 2K image generation without visual quality degradation.

## Background & Motivation

MMDiT architectures (e.g., SD3, FLUX) are the dominant paradigm for text-to-image generation, performing joint self-attention over concatenated visual and text tokens. However, attention computation remains the primary inference bottleneck. Existing acceleration methods (e.g., DiTFastAttn) exhibit three critical limitations:

**Cross-modal attention pattern complexity**: In MMDiT, visual tokens exhibit diagonal locality while text token interactions are highly semantics-dependent. A uniform sliding-window attention cannot capture this disparity, and forced application truncates text information.

**Inter-head redundancy heterogeneity**: Attention heads within the same layer exhibit drastically different behaviors—some approximate global attention while others are highly localized. Layer-level uniform caching or sparsity strategies discard critical information.

**Prohibitive search cost**: DiTFastAttn requires over 10 hours to search a compression scheme (50-step 2K FLUX generation), which would exceed 200 hours when extended to head-level granularity.

## Method

### Overall Architecture

DiTFastAttnV2 is a post-training compression framework comprising three components: Head-wise Arrow Attention (addressing spatial redundancy heterogeneity), Head-wise Caching (addressing timestep redundancy heterogeneity), and an efficient fused kernel (enabling practical acceleration), along with an efficient compression scheme search algorithm.

### Key Designs

1. **Head-wise Arrow Attention**:

    - The joint attention map is partitioned into four regions: visual-visual, visual-text, text-visual, and text-text.
    - Local attention is applied to the visual-visual region (retaining only attention scores near the diagonal), discarding long-range token interactions.
    - Full attention is preserved for the three regions involving text tokens without any compression.
    - This pattern resembles an arrow, hence the name "arrow attention."
    - Each attention head can independently select full attention or arrow attention (mixed attention design).
    - **Design Motivation**: The diagonal locality of visual tokens is consistent across prompts, whereas text interactions are highly semantics-dependent and incompressible.

2. **Head-wise Caching**:

    - Analysis reveals that inter-head similarity across adjacent timesteps varies significantly within the same layer.
    - For heads with high inter-step similarity, attention computation at the current timestep is skipped and the cached output from the previous step is reused directly.
    - Each head independently determines whether to use caching.
    - **Design Motivation**: Timestep redundancy is exploited at head-level granularity to avoid discarding critical information from rapidly changing heads.

3. **Fused Kernel**:

    - Integrates arrow attention and caching; each head can independently select one of three modes: full attention, computation skip (cache reuse), or arrow attention (with a specified window size).
    - Implemented based on FlashAttention2 with a block-sparse pattern to ensure each computation block is dense, minimizing irregular memory access overhead.
    - Mixed blocks are converted to dense blocks to reduce memory access costs.

### Loss & Training

**Efficient Compression Scheme Search**:

- **Per-layer RSE Metric**: The single-layer relative squared error (RSE), rather than the final output MSE, is used to measure the impact of each compression method, reducing calibration cost from $T \times L \times M \times H$ full inferences to $T \times M$:
  $$\mathcal{I}(m) = \frac{\sum(y_m - \bar{y}_o)^2}{\sum(y_o - \bar{y}_o)^2}$$

- **Head-wise Compression Scheme Optimization**: The compression configuration per layer per timestep is modeled as an integer optimization problem, minimizing latency subject to an RSE budget constraint $\delta$.
- **Head Constraint Coefficient $c$**: A constraint $\mathcal{I}(h,m) \leq \frac{c}{n}\delta$ is introduced to prevent any single head from absorbing a disproportionate share of the compression budget; default $c=1.5$.
- A progressive update search strategy is adopted, proceeding timestep-by-timestep and layer-by-layer.

## Key Experimental Results

### Main Results — SD3 and FLUX Generation Quality (Table)

| Model | Resolution | Threshold δ | Attention Sparsity | LPIPS↓ | SSIM↑ | HPSv2↑ | CLIP↑ |
|-------|-----------|-------------|-------------------|--------|-------|--------|-------|
| SD3 | 1024 | Original | 0 | - | - | 0.2926 | 0.3254 |
| SD3 | 1024 | δ=0.2 | 0.41 | 0.182 | 0.716 | 0.2933 | 0.3251 |
| SD3 | 1024 | δ=0.6 | 0.63 | 0.266 | 0.616 | 0.2933 | 0.3246 |
| FLUX | 2048 | Original | 0 | - | - | 0.2862 | 0.3169 |
| FLUX | 2048 | δ=0.2 | 0.43 | 0.242 | 0.646 | 0.2883 | 0.3164 |
| FLUX | 2048 | δ=1.0 | **0.68** | 0.393 | 0.497 | 0.2852 | 0.3163 |

### Ablation Study — Method Combinations and Constraint Coefficients (Table)

| Method Set | Attention Sparsity | LPIPS↓ | SSIM↑ | HPSv2↑ |
|------------|-------------------|--------|-------|--------|
| Original | 0 | - | - | 0.2926 |
| AA only | 0.30 | 0.275 | 0.608 | 0.2943 |
| **AA + OC** | **0.55** | **0.238** | **0.644** | **0.2935** |
| + CFG Sharing | 0.54 | 0.249 | 0.649 | 0.2913 |
| + Residual Sharing | 0.56 | 0.196 | 0.704 | 0.2906 |

| Constraint Coefficient $c$ | Sparsity | LPIPS↓ | SSIM↑ |
|---------------------------|---------|--------|-------|
| No constraint | 0.50 | 0.249 | 0.640 |
| $c=1$ | 0.55 | 0.240 | 0.641 |
| **$c=1.5$** | **0.55** | **0.238** | **0.644** |
| $c=2$ | 0.55 | 0.253 | 0.627 |

### Key Findings

- **Significant speedup**: 1.5× end-to-end acceleration on FLUX 2K image generation with up to 68% reduction in attention FLOPs.
- **Lossless quality**: At δ=0.2/0.6, HPSv2 and CLIP scores are comparable to or even exceed those of the original model.
- **Substantial improvement over predecessor**: At equivalent attention sparsity on SD3, DiTFastAttnV2 outperforms DiTFastAttn and its variants on all metrics.
- **Search efficiency gain**: Compression scheme search for 2K image generation is reduced from 10 hours to 15 minutes.
- **CFG Sharing ineffective for MMDiT**: The MMDiT design eliminates the need for CFG, leaving negligible redundancy for CFG sharing to exploit.
- **Fused kernel performance**: Achieves 3.55× speedup at 75% sparsity, approaching or exceeding the theoretical upper bound.

## Highlights & Insights

1. **Re-examining attention compression at head granularity**: The paper reveals high heterogeneity among attention heads in MMDiT and designs fine-grained strategies accordingly.
2. **Elegant Arrow Attention design**: Precisely captures the locality of visual tokens and the global dependency of text tokens.
3. **Two-order-of-magnitude search efficiency improvement**: The per-layer RSE metric combined with head-level optimization enables minute-scale search.
4. **High practical deployment value**: The fused kernel achieves genuine 1.5× speedup rather than merely theoretical FLOPs reduction.

## Limitations & Future Work

- Validation is limited to SD3 and FLUX; applicability to other MMDiT architectures (e.g., CogVideoX, HunyuanVideo, and other video models) remains unverified.
- The window size of Arrow Attention is determined via search and lacks an adaptive mechanism.
- The current implementation targets A100 GPUs; speedup ratios may differ on other hardware platforms.
- The combination of the proposed method with other compression approaches such as quantization and distillation is not explored.
- At high compression rates (δ=1.0), details and backgrounds may change, potentially unsuitable for scenarios requiring strict consistency.

## Related Work & Insights

- DiTFastAttnV2 is a direct extension of DiTFastAttn, expanding from layer-level to head-level granularity and resolving MMDiT adaptation issues.
- Arrow Attention draws inspiration from Attention Sink and sparse attention research in LLMs.
- The per-layer metric calibration approach is generalizable to other scenarios requiring compression configuration search.
- The finding of head-level heterogeneity may also serve as a reference for ViT pruning and distillation research.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The head-wise combination of arrow attention and caching is novel, with significant search efficiency improvements.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-metric, multi-threshold, and thorough ablations, though evaluation on only two models is somewhat limited.
- **Writing Quality**: ⭐⭐⭐⭐ In-depth analysis, clear illustrations, and complete algorithmic pseudocode.
- **Value**: ⭐⭐⭐⭐⭐ Addresses practical pain points in MMDiT inference acceleration with high real-world applicability.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Head Pursuit: Probing Attention Specialization in Multimodal Transformers](../../NeurIPS2025/image_generation/head_pursuit_probing_attention_specialization_in_multimodal.md)
- [\[ICCV 2025\] EDiT: Efficient Diffusion Transformers with Linear Compressed Attention](edit_efficient_diffusion_transformers_with_linear_compressed_attention.md)
- [\[ICCV 2025\] FreeCus: Free Lunch Subject-driven Customization in Diffusion Transformers](freecus_free_lunch_subject-driven_customization_in_diffusion_transformers.md)
- [\[ICCV 2025\] M2SFormer: Multi-Spectral and Multi-Scale Attention with Edge-Aware Difficulty Guidance for Image Forgery Localization](m2sformer_multi-spectral_and_multi-scale_attention_with_edge-aware_difficulty_gu.md)
- [\[ICCV 2025\] UniCombine: Unified Multi-Conditional Combination with Diffusion Transformer](unicombine_unified_multi-conditional_combination_with_diffusion_transformer.md)

<!-- RELATED:END -->
