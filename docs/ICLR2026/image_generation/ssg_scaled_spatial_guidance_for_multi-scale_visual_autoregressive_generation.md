---
title: >-
  [Paper Note] SSG: Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation
description: >-
  [ICLR 2026][Image Generation][VAR] This paper proposes Scaled Spatial Guidance (SSG), a training-free inference-time guidance method that enhances the coarse-to-fine hierarchical generation quality of visual autoregressi…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "VAR"
  - "next-scale prediction"
  - "information bottleneck"
  - "frequency-domain guidance"
  - "training-free"
date: 2026-05-08
content_hash: 256b406aff81fab6
---

# SSG: Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation

**Conference**: ICLR 2026
**arXiv**: [2602.05534](https://arxiv.org/abs/2602.05534)
**Code**: [GitHub](https://github.com/Youngwoo-git/SSG)
**Area**: Visual Autoregressive Models / Image Generation / Inference-Time Guidance
**Keywords**: VAR, next-scale prediction, information bottleneck, frequency-domain guidance, training-free

## TL;DR

This paper proposes Scaled Spatial Guidance (SSG), a training-free inference-time guidance method that enhances the coarse-to-fine hierarchical generation quality of visual autoregressive models through frequency-domain prior construction and semantic residual amplification.

## Background & Motivation

Visual autoregressive (VAR) models generate images via next-scale prediction, naturally achieving coarse-to-fine hierarchical synthesis. However:

**Training–inference discrepancy**: Limited model capacity and accumulated errors cause the model to deviate from its coarse-to-fine nature at inference time, leading to redundant prediction of low-frequency information.

**Limitations of existing improvements**:
   - Auxiliary refinement modules (CoDe, HMAR) require retraining.
   - Flow matching integration introduces additional overhead.
   - Self-correction mechanisms require architectural modifications.

**Core Problem**: How can one guide each generation step to produce novel, scale-specific high-frequency information without modifying model parameters?

## Method

### 1. Derivation from an Information-Theoretic Perspective

Starting from the Information Bottleneck (IB) principle, the stepwise generation of VAR is reformulated as a variational optimization problem:

$$\mathcal{L}_{\text{VAR-IB}} = \max_{z_k} \beta I(z_k; H(\hat{f}_K)) - I(z_k; L(\hat{f}_K))$$

- **Target information term**: Maximizes mutual information with high-frequency details.
- **State redundancy term**: Minimizes redundancy with already-established coarse structures.

### 2. SSG Formulation

The optimization objective is converted into a MAP-style surrogate function, yielding a closed-form solution:

$$\ell_k^{\text{SSG}} = \ell_k + \beta_k \Delta_k = \ell_k + \beta_k (\ell_k - \ell_{\text{prior}})$$

where:
- $\ell_k$: residual logits at step $k$
- $\ell_{\text{prior}}$: coarse-grained prior constructed from the previous step
- $\Delta_k = \ell_k - \ell_{\text{prior}}$: semantic residual (high-frequency details)
- $\beta_k$: step-wise scaling factor

### 3. Discrete Space Enhancement (DSE)

Frequency-domain prior construction procedure:
1. Spatially interpolate the previous-step logits $\ell_{k-1}$ to obtain $\ell'_{\text{interp}}$.
2. Apply DCT to both.
3. Fuse the low-frequency coefficients of $\ell_{k-1}$ with the high-frequency coefficients of $\ell'_{\text{interp}}$.
4. Apply IDCT to recover the prior $\ell_{\text{prior}}$.

**Advantages over simple interpolation**:
- Linear interpolation over-smooths and attenuates the prior.
- Nearest-neighbor interpolation introduces blocky discontinuities and spurious high frequencies.
- DCT frequency-domain fusion preserves energy conservation and achieves precise band separation.

### 4. Efficient Implementation

- No additional forward passes required (cached logits are reused).
- Implemented in only a few lines of code.
- Negligible computational and memory overhead.

## Key Experimental Results

### ImageNet 256×256 Class-Conditional Generation

| Model | FID↓ | sFID↓ | IS↑ | Pre↑ | Rec↑ |
|------|------|-------|-----|------|------|
| VAR-d16 | 3.42 | 8.70 | 275.6 | 0.84 | 0.51 |
| +SSG | **3.27** | **8.39** | **285.3** | **0.85** | 0.50 |
| VAR-d20 | 2.67 | 7.97 | 299.8 | 0.83 | 0.55 |
| +SSG | **2.49** | **7.60** | **305.2** | 0.83 | **0.56** |
| VAR-d24 | 2.39 | 8.18 | 314.7 | 0.82 | 0.58 |
| +SSG | **2.20** | **6.95** | **324.0** | **0.83** | **0.59** |
| VAR-d30 | 2.02 | 8.52 | 302.9 | 0.82 | 0.60 |
| +SSG | **1.68** | **8.50** | **313.2** | 0.81 | **0.62** |

### Cross-Model Generalization

SSG proves effective across different tokenization schemes:
- Standard VAR (Tian et al.)
- HART (hybrid tokens)
- Infinity (bitwise tokens)

### Comparison with Other Generative Models

VAR-d30 + SSG (FID 1.68) is competitive with diffusion and masked generative models while retaining VAR's low-latency advantage (10-step inference).

### Ablation Study

| Component | FID | IS |
|------|-----|-----|
| w/o SSG (baseline) | 2.02 | 302.9 |
| SSG + linear interpolation prior | limited improvement | — |
| SSG + nearest-neighbor prior | possible degradation | — |
| SSG + DSE (frequency-domain fusion) | **1.68** | **313.2** |

## Highlights & Insights

1. **Elegant information-theoretic design**: SSG's closed-form solution is rigorously derived from the IB principle.
2. **Completely training-free**: No modification of model weights, no additional data, and no fine-tuning required.
3. **Theoretically grounded frequency-domain prior (DSE)**: Exploits DCT orthogonality to achieve lossless energy-preserving band fusion.
4. **Strong consistency**: Effective across VAR models of varying scales and tokenization designs.
5. **Minimal implementation**: Integrable in just a few lines of code.

## Limitations & Future Work

1. The effectiveness of SSG depends on a well-designed $\beta_k$ schedule, requiring per-model tuning.
2. SSG has no effect at the first step (coarsest scale) due to the absence of a prior.
3. SSG is fundamentally a posterior correction and cannot compensate for information loss inherent to the tokenizer.
4. Applicable only to VAR models that operate on discrete visual tokens.

## Related Work & Insights

- **VAR models**: VAR (Tian 2024), HART (Tang 2025), Infinity (Han 2025)
- **Visual guidance**: CFG, SAG, PAG, STG — none designed specifically for VAR
- **Training–inference discrepancy mitigation**: CoDe, HMAR — both require retraining

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Elegant bridging from information theory to practice
- **Practicality**: ⭐⭐⭐⭐⭐ — Zero-cost integration, plug-and-play
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated across multiple models and settings
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are clear and intuitions are well-explained

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Laplacian Multi-scale Flow Matching for Generative Modeling](laplacian_multi-scale_flow_matching_for_generative_modeling.md)
- [\[ICLR 2026\] Visual Autoregressive Modeling for Instruction-Guided Image Editing](visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [\[AAAI 2026\] HACK: Head-Aware KV Cache Compression for Efficient Visual Autoregressive Modeling](../../AAAI2026/image_generation/head-aware_kv_cache_compression_for_efficient_visual_autoreg.md)
- [\[ICCV 2025\] Randomized Autoregressive Visual Generation](../../ICCV2025/image_generation/randomized_autoregressive_visual_generation.md)
- [\[ICLR 2026\] Next Visual Granularity Generation](next_visual_granularity_generation.md)

</div>

<!-- RELATED:END -->
