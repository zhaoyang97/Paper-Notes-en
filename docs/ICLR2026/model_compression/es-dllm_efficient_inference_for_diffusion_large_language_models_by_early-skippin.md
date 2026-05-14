---
title: >-
  [Paper Note] ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping
description: >-
  [ICLR2026][Model Compression][Diffusion LLM] To address the extensive computational redundancy in diffusion large language model (dLLM) inference, this paper proposes ES-dLLM…
tags:
  - "ICLR2026"
  - "Model Compression"
  - "Diffusion LLM"
  - "Inference Acceleration"
  - "Token Skipping"
  - "KV Cache"
  - "training-free"
date: 2026-05-08
content_hash: eaf4788a05a7c957
---

# ES-dLLM: Efficient Inference for Diffusion Large Language Models by Early-Skipping

**Conference**: ICLR2026
**arXiv**: [2603.10088](https://arxiv.org/abs/2603.10088)
**Code**: [zhuzj19/ES-dLLM](https://github.com/zhuzj19/ES-dLLM)
**Area**: Model Compression
**Keywords**: Diffusion LLM, Inference Acceleration, Token Skipping, KV Cache, training-free

## TL;DR

To address the extensive computational redundancy in diffusion large language model (dLLM) inference, this paper proposes ES-dLLM, a training-free Early-Skipping acceleration framework. By estimating token importance and skipping low-importance positions in early layers, ES-dLLM achieves 5.6×–16.8× speedup on LLaDA-8B and Dream-7B without degrading generation quality.

## Background & Motivation

Diffusion large language models (dLLMs) such as LLaDA and Dream generate text through iterative denoising, supporting bidirectional attention and parallel decoding as a compelling alternative to autoregressive models (ARMs). However, the inference efficiency of open-source dLLMs is substantially lower than that of ARMs of comparable scale, with three core bottlenecks:

1. **Full-sequence processing per iteration**: Each dLLM iteration performs a forward pass over the entire input sequence, incurring enormous computational cost.
2. **Massive ineffective computation**: Each iteration unmasks only a small number of high-confidence tokens, leaving the computation of most masked tokens unused.
3. **Near-identical inputs across adjacent iterations**: Adjacent iterations differ only at newly unmasked positions, with minimal change in intermediate states.

Through empirical analysis, the authors find that: (a) confidence changes at most positions follow an approximately exponential distribution concentrated near zero, with over 90% of positions exhibiting changes below 0.05; and (b) hidden states vary only marginally between consecutive iterations. These observations reveal substantial eliminable redundancy.

## Core Problem

How can low-importance token computations in dLLM inference be identified and skipped without introducing additional training, thereby significantly reducing the per-iteration computational cost?

## Method

ES-dLLM comprises two core components:

### 1. Importance Score Estimation

For each token position $i$ at layer $l$, the importance score is computed as:

$$I_{l,i} = \alpha \cdot c_i^{(t-1)} + (1-\alpha) \cdot \frac{\| H_{l,i}^{(t)} - H_{l,i}^{(t-1)} \|_1}{\sqrt{d} \cdot \| H_{l,i}^{(t-1)} \|_2}$$

- **Confidence term** $c_i^{(t-1)}$: the maximum softmax probability at position $i$ from the previous iteration; tokens with higher confidence are more likely to be unmasked in the current step.
- **Change term**: the L1 difference between the current-layer hidden state and the cached state from the previous iteration, normalized by L2 norm, reflecting the token's semantic dependency on newly generated content.
- **Weight** $\alpha$: a hyperparameter balancing the two terms, defaulting to 0.5.

The top-$k$ positions ($k = (1 - r_l) \cdot |S|$) ranked by importance score are retained for further computation; the remaining positions are skipped.

### 2. Partial Cache Update & Early Skip

The inference procedure within each Transformer block proceeds as follows:

1. Apply normalization and QKV projection to tokens in the current position set $S$.
2. Update the corresponding entries in the KV cache (via scatter), then read the full KV cache to perform attention.
3. Apply the FFN to obtain hidden states $H_S$ and update the hidden state cache.
4. Compute importance scores using the formula above and select the top-$k$ positions to form the new set $S'$.
5. Pass only the hidden states corresponding to $S'$ to the next layer.

**Key design considerations**:
- Skipping positions at earlier layers saves more computation (reducing the matrix multiplication size in all subsequent layers), but overly early skipping may compromise the reliability of change estimation.
- By default, skip ratios of 0.5 are applied at depths of 1/8 and 1/4, reducing FLOPs by approximately 60%.
- To prevent error accumulation, the KV cache for prompt tokens or the current block is periodically refreshed at fixed intervals.

### 3. Comparison with DualCache

DualCache (Fast-dLLM) caches KV states outside the current block but still computes the entire block. ES-dLLM further skips unimportant positions within the block, achieving deeper computational savings.

## Key Experimental Results

Main results on NVIDIA H200 GPU:

**LLaDA-8B-Instruct**:

| Benchmark | Baseline TPS | ES-dLLM TPS | Speedup | Accuracy Change |
|-----------|-------------|-------------|---------|----------------|
| GSM8K | 8.56 | 143.93 | 16.8× | +0.23 |
| MATH | 14.04 | 103.63 | 7.4× | -0.90 |
| BBH | 11.06 | 159.89 | 14.5× | -2.24 |
| HumanEval | 23.65 | 226.57 | 9.6× | +1.21 |
| MBPP | 8.98 | 145.99 | 16.3× | -1.60 |

**Dream-7B-Instruct**:

| Benchmark | Baseline TPS | ES-dLLM TPS | Speedup | Accuracy Change |
|-----------|-------------|-------------|---------|----------------|
| GSM8K | 19.80 | 267.13 | 13.5× | -1.06 |
| MATH | 26.38 | 147.44 | 5.6× | -0.50 |
| HumanEval | 44.34 | 308.51 | 7.0× | -1.83 |
| MBPP | 21.68 | 276.12 | 12.7× | -3.40 |

Compared to DualCache, ES-dLLM achieves an additional 1.20×–1.85× speedup, with superior accuracy on several benchmarks.

**Ablation Study highlights**:
- $\alpha=0.5$ (equal weighting of both terms) yields the best overall performance; using confidence alone ($\alpha=1$) leads to noticeable degradation.
- Hidden states serve as slightly better change indicators than QKV tensors, though the latter incur lower memory overhead.
- Additional memory overhead is negligible: only 528KB per output token for LLaDA-8B and 70KB for Dream-7B.

## Highlights & Insights

1. **Training-free**: A pure inference-stage optimization that is plug-and-play and compatible with existing dLLMs.
2. **Observation-driven design**: Motivated by systematic analysis of confidence and hidden state dynamics during dLLM generation.
3. **Significant speedup**: Up to 16.8× acceleration with accuracy changes within ±2% on most benchmarks.
4. **Orthogonal to existing methods**: Can be combined with sparse attention, parallel decoding, and other techniques.
5. **Negligible memory overhead**: The additional cache requires only a few hundred MB compared to 10GB+ model weights.

## Limitations & Future Work

1. **Heuristic importance estimation**: The linear combination of confidence and L1 change may lack precision; a lightweight trainable predictor could improve importance estimation.
2. **Deviation from training assumptions**: dLLMs are trained assuming full state updates; skipping positions may introduce a distribution shift.
3. **Practical speedup falls short of theoretical gains**: A 60% FLOPs reduction translates to only 1.2×–1.85× additional speedup over DualCache, as inference becomes memory-bound with weight and KV memory accesses undiminished.
4. **Manual tuning of skip ratios**: Different tasks may require different skipping rates, and no adaptive adjustment mechanism is provided.

## Related Work & Insights

| Method | Type | Training Required | Speedup Source | Limitation |
|--------|------|-------------------|---------------|------------|
| Semi-AR (LLaDA) | Generation strategy | No | Block-wise sequential generation | Constrains generation order |
| BD3-LM | Model design | Yes | Unidirectional attention + KV Cache | Loses bidirectional modeling |
| dKV-Cache | KV caching | No | Delayed update of newly unmasked KV | Does not exploit semi-AR |
| DualCache | KV caching | No | KV reuse outside block | Still computes the full block |
| Sparse-dLLM | Sparse attention | No | Sparsification over historical tokens | Orthogonal; targets attention range |
| **ES-dLLM** | **Token skipping** | **No** | **Skipping low-importance positions within block** | **Memory-bound bottleneck** |

**Broader insights**:
- **Complementarity of skipping and caching**: ES-dLLM and DualCache are composable—the former reduces intra-block computation while the latter reduces inter-block computation; this paradigm generalizes to other iterative generative models.
- **Generality of the importance score**: The dual-factor evaluation combining confidence and change magnitude is transferable to token pruning in diffusion image generation.
- **Rapid progress in dLLM inference optimization**: Techniques spanning KV caching → sparse attention → token skipping → parallel decoding are orthogonal and complementary; system-level integration represents a key future direction.

## Rating
- Novelty: 7/10 (the core intuition is clear, but the technical design is relatively straightforward)
- Experimental Thoroughness: 8/10 (multiple models and benchmarks, comprehensive ablations, measured throughput on H200)
- Writing Quality: 8/10 (clear motivation, well-organized experiments)
- Value: 7/10 (strong practical utility, but actual speedup is constrained by memory-bound bottlenecks, leaving theoretical acceleration potential unrealized)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping](../../AAAI2026/model_compression/skipcat_rank-maximized_low-rank_compression_of_large_language_models_via_shared_.md)
- [\[ICLR 2026\] Knowledge Fusion of Large Language Models Via Modular Skillpacks](knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)
- [\[ICLR 2026\] Distillation of Large Language Models via Concrete Score Matching](distillation_of_large_language_models_via_concrete_score_matching.md)
- [\[ICLR 2026\] Is Finer Better? The Limits of Microscaling Formats in Large Language Models](is_finer_better_the_limits_of_microscaling_formats_in_large_language_models.md)
- [\[ICLR 2026\] Unveiling Super Experts in Mixture-of-Experts Large Language Models](unveiling_super_experts_in_mixture-of-experts_large_language_models.md)

</div>

<!-- RELATED:END -->
