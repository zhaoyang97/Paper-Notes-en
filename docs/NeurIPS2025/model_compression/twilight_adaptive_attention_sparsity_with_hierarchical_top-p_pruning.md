---
title: >-
  [Paper Note] Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning
description: >-
  [NeurIPS 2025][Model Compression][Attention Sparsity] This paper proposes Twilight, which replaces fixed-budget top-k attention sparsity with a top-p (nucleus sampling) inspired approach — dynamically selecting the minimum set of tokens whose cumulative attention weights reach p%, adapting to the distribution characteristics of different attention heads. Twilight achieves up to 1.4× additional speedup over state-of-the-art sparse attention methods while maintaining accuracy.
tags:
  - NeurIPS 2025
  - Model Compression
  - Attention Sparsity
  - Adaptive Budget
  - Top-p Sampling
  - Long-Context Acceleration
  - KV Cache Compression
date: 2026-05-08
content_hash: 0b3f2cbfdf4ea6de
---

# Twilight: Adaptive Attention Sparsity with Hierarchical Top-p Pruning

**Conference**: NeurIPS 2025
**arXiv**: [2502.02770](https://arxiv.org/abs/2502.02770)
**Code**: [https://github.com/tsinghua-ideal/Twilight](https://github.com/tsinghua-ideal/Twilight)
**Area**: Model Compression
**Keywords**: Attention Sparsity, Adaptive Budget, Top-p Sampling, Long-Context Acceleration, KV Cache Compression

## TL;DR
This paper proposes Twilight, which replaces fixed-budget top-k attention sparsity with a top-p (nucleus sampling) inspired approach — dynamically selecting the minimum set of tokens whose cumulative attention weights reach p%, adapting to the distribution characteristics of different attention heads. Twilight achieves up to 1.4× additional speedup over state-of-the-art sparse attention methods while maintaining accuracy.

## Background & Motivation

**State of the Field**: Exploiting attention sparsity to accelerate long-context LLM inference is an active research direction. Methods such as H2O, StreamingLLM, Quest, and Double Sparsity select subsets of "important tokens" for attention computation.

**Limitations of Prior Work**: All existing methods rely on a **fixed-budget** top-k strategy — retaining the k tokens with the highest attention weights. However, the optimal k varies across layers, attention heads, sequence positions, and input content:
   - **Focused attention**: weights concentrate on a few tokens; fixed k leads to **over-selection** — retaining too many unimportant tokens.
   - **Diffuse attention**: weights are uniformly distributed; fixed k leads to **under-selection** — missing important tokens.
   - Different algorithms require varying degrees of "over-selection redundancy" to compensate for estimation error.

**Root Cause**: A fixed budget cannot adapt to the dynamic variation in attention distributions.

**Starting Point**: This issue is a direct analogy to the top-k vs. top-p dilemma in LLM sampling — top-k sampling was similarly replaced by nucleus sampling due to its inability to adapt to varying token distributions.

**Core Idea**: Introduce top-p into attention sparsity — instead of fixing the number of retained tokens to k, retain the **minimum** number of tokens whose cumulative attention weights reach p, automatically adapting to each head's distribution.

## Method

### Overall Architecture
Twilight adopts a Select-then-Prune two-stage architecture: (1) **Token Selector**: any existing sparse attention algorithm that performs initial selection with a conservative, larger budget (e.g., 1/4 sparsity); (2) **Twilight Pruner**: applies top-p pruning to the selected set, retaining only the minimum token subset whose cumulative attention weights reach p. The final sparse attention is computed over only the top-p tokens.

### Key Designs

1. **Formalization of Oracle Top-p Sparse Attention**:

    - Function: Given threshold p, select the minimum token set such that the sum of attention weights ≥ p.
    - Formulation: $\mathcal{I} = \arg\min_\mathcal{I} |\mathcal{I}|$ s.t. $\sum_{i \in \mathcal{I}} \mathbf{W}[i] \geq p$
    - Theoretical guarantee: The output error is upper-bounded by $(1-p) \cdot \|\mathbf{V}\|_F$ — independent of token count, depending only on p.
    - vs. top-k: The error of top-k depends on the distribution — small for focused distributions, large for diffuse ones. Top-p adapts automatically.

2. **Efficient Top-p Implementation (Binary Search)**:

    - Function: Parallelly and efficiently implement top-p selection on GPU.
    - Mechanism: Rather than sorting (which is ill-suited for GPU parallelism), binary search is used to find a threshold $m$ such that $\sum_{W[i] \geq m} W[i] \geq p$. Implemented by modifying the top-p sampling kernel in FlashInfer.
    - Design Motivation: Avoids $O(n \log n)$ sorting overhead; binary search requires only $O(\log (\max W / \epsilon))$ rounds, each with $O(n)$ parallel operations.

3. **4-bit K Cache Quantization**:

    - Function: Use INT4-quantized K cache to rapidly estimate attention weights.
    - Mechanism: Top-p requires **numerical precision** of attention weights (not merely correct ranking). Experiments show that 4-bit quantization achieves the optimal accuracy-efficiency trade-off — 2-bit causes significant degradation in weight sums, while 4-bit and 8-bit perform nearly identically.
    - Memory overhead: 1/8 of KV cache (16-bit → 4-bit, applied to only half of K).

4. **Load Balancing (Head Dynamics-Aware)**:

    - Function: Address computational load imbalance caused by varying per-head budgets.
    - Mechanism: Reuses FlashInfer's load balancing algorithm, flattening the head dimension and redistributing computational resources accordingly.
    - Design Motivation: Some heads may require only 2% of tokens while others require ~100% — conventional implementations allocate equal resources to all heads, leading to significant waste.

### Loss & Training
- **Fully training-free** — Twilight is an inference-time acceleration framework.
- $p$ is the only hyperparameter, typically set to 0.8–0.9.
- Can augment any existing top-k sparse attention algorithm.

## Key Experimental Results

### Accuracy Evaluation

| Benchmark | Method | Base Budget | Effective Budget (after top-p) | Performance Retention |
|------|------|-----------|------------------|---------|
| PG-19 (PPL) | Quest | 1024 | ~100 (p=0.85) | ~99.5% |
| RULER (128K) | DS | 1/4 | ~1/64 | ~98% |
| LongBench | H2O+Twilight | 1/4 | Adaptive | **Outperforms fixed H2O** |

### Efficiency Evaluation (End-to-End Decoding Latency)

| Method | Single-Layer Attention Speedup | End-to-End Speedup | Max Speedup |
|------|-------------|---------|---------|
| Full Attention | 1× | 1× | — |
| Quest (top-k) | ~8× | ~3× | — |
| **Quest + Twilight (top-p)** | **~11×** | **~4×** | **15.8×** |
| Relative to SOTA Sparse Attention | **1.4×** | **1.35×** | — |

### Ablation Study

| Configuration | Accuracy | Efficiency | Notes |
|------|------|------|------|
| p=0.7 | Slight degradation | Highest | Too aggressive |
| p=0.85 | **Near-lossless** | **High** | Recommended |
| p=0.95 | Best | Moderate | Conservative |
| 2-bit K cache | Degraded | — | Insufficient precision |
| 4-bit K cache | **Near-lossless** | **Good** | Optimal trade-off |

### Key Findings
- Budget variation across attention heads within the same model can reach **50×** (2% vs. ~100% of tokens) — fixed budgets are inherently wasteful or insufficient.
- Top-p nearly matches oracle top-k (top-k with perfect attention weights known in advance) in correctness, while far surpassing manually tuned top-k in adaptability.
- Twilight consistently accelerates all tested base algorithms (Quest, DS, H2O) — serving as a plug-and-play enhancement.
- Gains are larger in long-context settings, as attention distributions vary more drastically over longer sequences.

## Highlights & Insights
- The **top-p → attention pruning analogy** is the paper's deepest insight — two seemingly unrelated problems (sampling and attention selection) share the same mathematical structure: selecting the minimum subset of a probability distribution whose cumulative mass reaches a threshold.
- The positioning as an **"enhancer rather than a replacement"** is particularly strategic — rather than competing with existing methods, Twilight stacks adaptivity on top of them, allowing it to benefit from all future advances in the field.
- The theoretical error bound $(1-p) \cdot \|\mathbf{V}\|_F$ is concise and token-count-independent — providing a universal accuracy-efficiency trade-off knob for attention approximation.

## Limitations & Future Work
- Top-p requires (approximate) attention weight values rather than merely their ranking — placing higher demands on weight estimation accuracy than top-k.
- The additional INT4 K cache introduces 1/8 memory overhead, which may be unacceptable in memory-constrained settings.
- The parameter $\epsilon$ in the binary search implementation requires careful selection — too large a value may cause the actual selection to violate the p constraint.
- Top-p sparsity during the prefill phase is unexplored — the current work focuses primarily on the decoding phase.

## Related Work & Insights
- **vs. Quest/DS (top-k)**: These methods require offline calibration to find the optimal k; Twilight uses p to determine budgets automatically — a single p generalizes across all heads and layers.
- **vs. SampleAttention**: Focuses on the prefill phase with some adaptivity, but Twilight is more general.
- **vs. Tactic (concurrent)**: Tactic also applies top-p but estimates the distribution via function fitting — prone to budget overestimation; Twilight uses binary search on actual weights for greater precision.
- Inspiration: The top-p idea can be generalized to other "subset selection from a distribution" settings — such as token pruning and expert selection in MoE.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The conceptual transfer of top-p to attention sparsity is profound and natural.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Long/medium-context accuracy + latency/throughput efficiency + multiple base algorithms + detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formalization, analogical reasoning, and system implementation are all clearly presented.
- Value: ⭐⭐⭐⭐⭐ Addresses the fundamental limitation of fixed budgets; serves as a plug-and-play enhancement to all sparse attention methods.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Spark Transformer: Reactivating Sparsity in FFN and Attention](spark_transformer_reactivating_sparsity_in_ffn_and_attention.md)
- [\[NeurIPS 2025\] MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference](mustafar_promoting_unstructured_sparsity_for_kv_cache_pruning_in_llm_inference.md)
- [\[NeurIPS 2025\] DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs](duogpt_training-free_dual_sparsity_through_activation-aware_pruning_in_llms.md)
- [\[ICLR 2026\] AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in LVLMs](../../ICLR2026/model_compression/agilepruner_an_empirical_study_of_attention_and_diversity_for_adaptive_visual_to.md)
- [\[NeurIPS 2025\] 4DGCPro: Efficient Hierarchical 4D Gaussian Compression for Progressive Volumetric Video Streaming](4dgcpro_efficient_hierarchical_4d_gaussian_compression_for_p.md)

<!-- RELATED:END -->
