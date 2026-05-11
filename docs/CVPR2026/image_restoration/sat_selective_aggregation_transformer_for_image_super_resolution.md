---
title: >-
  [Paper Note] SAT: Selective Aggregation Transformer for Image Super-Resolution
description: >-
  [CVPR 2026][Image Restoration][super-resolution] This paper proposes the Selective Aggregation Transformer (SAT), which reduces Key-Value matrix token count by 97% through density-driven token aggregation while preserving full-resolution Queries, enabling efficient global attention modeling. SAT surpasses the state-of-the-art PFT by 0.22 dB while reducing FLOPs by 27%.
tags:
  - CVPR 2026
  - Image Restoration
  - super-resolution
  - transformer
  - token aggregation
  - efficient attention
  - global modeling
date: 2026-05-08
content_hash: c974bc6057aefc9c
---

# SAT: Selective Aggregation Transformer for Image Super-Resolution

**Conference**: CVPR 2026
**arXiv**: [2604.07994](https://arxiv.org/abs/2604.07994)
**Code**: [https://github.com/PhuTran1005/SAT](https://github.com/PhuTran1005/SAT)
**Area**: Image Super-Resolution
**Keywords**: super-resolution, transformer, token aggregation, efficient attention, global modeling

## TL;DR

This paper proposes the Selective Aggregation Transformer (SAT), which reduces Key-Value matrix token count by 97% through density-driven token aggregation while preserving full-resolution Queries, enabling efficient global attention modeling. SAT surpasses the state-of-the-art PFT by 0.22 dB while reducing FLOPs by 27%.

## Background & Motivation

Transformer-based super-resolution methods can capture long-range dependencies but suffer from quadratic computational complexity. Window attention methods restrict the receptive field, while recent approaches each have their own limitations: IPG's graph operations are hardware-unfriendly, ATD's external dictionary provides limited additional information, and PFT's cross-layer attention linkage may propagate errors from early layers.

A core observation is that high-frequency regions (edges, textures) in SR require more computation, whereas low-frequency regions (smooth areas) can be safely aggregated. Existing methods apply uniform processing across the entire image, resulting in inefficient computational allocation.

## Method

### Overall Architecture

SAT adopts a residual group structure that alternates between Local Transformer Blocks (LTB, window attention) and Selective Aggregation Transformer Blocks (SATB, global attention), forming a complementary global–local architecture.

### Key Designs

1. **Selective Aggregation Attention (SAA)**: Asymmetric compression — Queries are kept at full resolution (required for per-pixel reconstruction), while only Keys and Values are compressed. $N$ tokens are aggregated into $K$ representative tokens ($K \approx 3\% \times N$), reducing complexity from $O(N^2d)$ to $O(NKd)$.

2. **Density-driven Token Aggregation (DTA)**: Aggregation centers are selected based on the density peak clustering principle. Each token's local density (cosine similarity over $k$-nearest neighbors) and minimum distance to a higher-density point are computed; tokens with higher products are selected as centers. Hierarchical subsampling reduces center selection complexity from $O(N^2)$ to $O(K^2)$. Similarity-weighted aggregation combined with Feature Norm Recovery (FNR) maintains feature distribution consistency.

3. **Global–Local Alternating Structure**: SAA focuses on global modeling (capturing long-range dependencies) and is interleaved with Rwin-SA local attention, enabling complementary extraction of deep features.

### Loss & Training

Standard $L_1$ pixel loss is used for training. The paper provides rigorous complexity guarantees (Theorem 3.1) and approximation bound analysis (Theorem 3.2), demonstrating that the method achieves substantial acceleration with controllable quality degradation.

## Key Experimental Results

### Main Results

| Dataset | Metric | SAT | Prev. SOTA (PFT) | Gain |
|---------|--------|-----|------------------|------|
| Urban100 ×4 | PSNR | +0.22 dB | baseline | Significant |
| Multiple datasets | FLOPs | −27% | baseline | Large efficiency gain |

### Ablation Study

| Configuration | PSNR | Note |
|---------------|------|------|
| w/o FNR (Feature Norm Recovery) | Decrease | FNR is critical for training stability |
| Uniform aggregation vs. density-driven | Decrease | Density-aware center selection is superior |
| Local attention only | Decrease | Global modeling is indispensable |

### Key Findings

- A 97% reduction in token count can still maintain or even improve reconstruction quality.
- Density-driven selection naturally preserves fine-grained tokens in high-frequency regions while merging low-frequency regions.
- FNR is essential for maintaining the feature norm distribution after weighted averaging.

## Highlights & Insights

- Asymmetric Query–KV compression perfectly matches SR task requirements: Queries remain per-pixel while KVs can be aggregated.
- Density-driven selection adapts to image content, preserving high-frequency detail and aggregating low-frequency regions.
- Complete theoretical analysis (complexity bounds and approximation bounds) strengthens methodological credibility.
- The global–local alternating structure is validated as the optimal choice through thorough ablation studies.

## Limitations & Future Work

- The aggregation ratio ($k = 3\%$) and subsampling factor $\beta$ require tuning.
- The $k$-nearest neighbor search in DTA still incurs non-trivial computational overhead.
- Effectiveness on highly irregular textures remains to be verified.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of asymmetric KV compression and density-driven aggregation is novel.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Rigorous theoretical analysis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive comparisons and thorough ablation studies.
- **Practical Value**: ⭐⭐⭐⭐ — Significantly reduces FLOPs while improving performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance](../../ICCV2025/image_restoration/enhancing_image_restoration_transformer_via_adaptive_translation_equivariance.md)
- [\[CVPR 2026\] Bridging the Perception Gap in Image Super-Resolution Evaluation](bridging_the_perception_gap_in_image_super-resolution_evaluation.md)
- [\[CVPR 2026\] Disentangled Textual Priors for Diffusion-based Image Super-Resolution](disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)
- [\[CVPR 2026\] RAW-Domain Degradation Models for Realistic Smartphone Super-Resolution](rawdomain_degradation_models_smartphone_sr.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](real_iisr_infrared_image_super_resolution_autoregressive.md)

</div>

<!-- RELATED:END -->
