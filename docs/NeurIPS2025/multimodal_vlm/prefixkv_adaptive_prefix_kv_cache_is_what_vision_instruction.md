---
title: >-
  [Paper Note] PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation
description: >-
  [NeurIPS 2025][Multimodal VLM][KV cache compression] PrefixKV identifies that the importance distributions of KV caches vary substantially across layers, and formalizes the per-layer cache sizing problem as a global prefix configuration search. A binary search is employed to find the optimal cumulative priority threshold that maximizes contextual information retention in each layer. At a 20% retention ratio, PrefixKV incurs only a 0.49 PPL degradation while delivering a 1.8× inference speedup.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - KV cache compression
  - adaptive layer-wise allocation
  - prefix configuration
  - binary search
  - vision-language models
date: 2026-05-08
content_hash: 0e0e5e53f88ac50e
---

# PrefixKV: Adaptive Prefix KV Cache is What Vision Instruction-Following Models Need for Efficient Generation

**Conference**: NeurIPS 2025
**arXiv**: [2412.03409](https://arxiv.org/abs/2412.03409)
**Code**: [https://github.com/THU-MIG/PrefixKV](https://github.com/THU-MIG/PrefixKV)
**Area**: Multimodal VLM / Inference Efficiency
**Keywords**: KV cache compression, adaptive layer-wise allocation, prefix configuration, binary search, vision-language models

## TL;DR

PrefixKV identifies that the importance distributions of KV caches vary substantially across layers, and formalizes the per-layer cache sizing problem as a global prefix configuration search. A binary search is employed to find the optimal cumulative priority threshold that maximizes contextual information retention in each layer. At a 20% retention ratio, PrefixKV incurs only a 0.49 PPL degradation while delivering a 1.8× inference speedup.

## Background & Motivation

**State of the Field**: Large vision-language models (LVLMs) demonstrate strong multimodal generation and reasoning capabilities; however, Transformer-based autoregressive decoding requires storing KV caches for all processed tokens, resulting in memory consumption that grows linearly with sequence length and constitutes a critical inference efficiency bottleneck.

**Limitations of Prior Work**: Existing KV cache compression methods—such as H2O, which evicts less important KVs based on attention scores, and Elastic Cache, which merges unimportant KVs into anchor positions—are effective but universally adopt a strategy of **retaining an equal number of KVs across all layers**. This ignores a critical observation: the importance distributions of KV caches differ dramatically across layers.

**Root Cause**: Some layers exhibit highly concentrated importance distributions (a small number of KVs encapsulates most of the information), while others are relatively uniform (requiring more KVs to preserve sufficient information). Uniform allocation wastes cache budget in concentrated layers while causing severe information loss in dispersed ones—a phenomenon the paper quantifies using Lorenz curves and Gini coefficients.

**Paper Goals**: How can the optimal number of KVs to retain be adaptively determined for each layer, such that each layer preserves as much contextual information as possible while satisfying a global compression budget?

**Starting Point**: The authors treat KVs sorted by importance as a priority sequence—retaining the most important KVs is equivalent to retaining a "prefix" of this sequence. The per-layer cache sizing problem is thereby reformulated as searching for the optimal global prefix configuration.

**Core Idea**: A binary search is used to identify a unified cumulative priority threshold $p$, under which each layer retains a prefix of sufficient length to meet this threshold, thereby maximizing per-layer information retention while satisfying the compression budget.

## Method

### Overall Architecture

The method follows the standard two-stage KV cache compression pipeline: (1) after prefilling, the importance of each KV is computed from attention scores, and the most important KVs in each layer are retained according to the global prefix configuration; (2) during decoding, as new tokens are generated and KV entries are updated, fixed-interval pruning maintains the per-layer cache size ratios. The core contribution lies in determining the per-layer cache size ratios $\{R_1, R_2, \ldots, R_L\}$.

### Key Designs

1. **Priority Sequence and Prefix Cumulative Priority**

   - **Function**: Formalizes KV compression as retaining the prefix of a priority sequence.
   - **Mechanism**: For layer $l$, the importance scores $I_l^n$ (sum of attention scores averaged across heads) for all KVs are normalized and sorted in descending order to form a priority sequence. Retaining the top-$o$ fraction of KVs is equivalent to retaining a prefix of the priority sequence, with cumulative priority $P_l^o$ representing the amount of contextual information preserved in that layer. Visualization via Lorenz curves reveals substantial variation in curve shapes across layers (with Gini coefficients ranging widely), demonstrating that uniform retention leads to severe information imbalance.
   - **Design Motivation**: Translates the qualitative observation that "different layers require different cache sizes" into a mathematically tractable optimization framework.

2. **Global Prefix Configuration and Binary Search**

   - **Function**: Efficiently identifies the optimal per-layer cache allocation satisfying the compression budget.
   - **Mechanism**: An information retention threshold $p$ is introduced; for each layer, the minimum prefix size ratio $R_l$ such that cumulative priority $P_l^o \geq p$ is determined. The constraint $\sum_l R_l = r \cdot L$ must be satisfied, where $r$ is the compression ratio. Given the large number of candidate values for $p$, binary search is applied over the interval $[0, 1]$: the total retention corresponding to the midpoint $p$ is computed, the difference $\delta$ from the budget is evaluated, and the search interval is refined accordingly. Upon convergence, the configuration that maximizes cumulative priority in each layer is obtained.
   - **Design Motivation**: Directly enumerating all possible per-layer allocation combinations is intractable; binary search converges efficiently in $O(\log N)$ steps.

3. **Offline Estimation and Cross-Sample Robustness**

   - **Function**: Eliminates the additional inference overhead of online search.
   - **Mechanism**: The authors find that cumulative priority sequences across different samples are highly similar within each layer (as shown in Figure 4), enabling the global prefix configuration to be estimated offline using a small number of random samples (even just one). At inference time, the precomputed configuration is directly applied. Experiments confirm that offline and online performance are nearly identical.
   - **Design Motivation**: Per-sample online search would introduce additional computational overhead; offline estimation enables seamless integration into existing inference pipelines.

### Loss & Training

PrefixKV is a training-free method and requires no additional training. It is complementary to prefilling acceleration techniques such as FastV.

## Key Experimental Results

### Main Results

Evaluated on LLaVA-1.5-7B and LLaVA-1.5-13B using the LLaVA-Description and MM-Vet datasets:

| Dataset | Model | Retention Ratio | PrefixKV PPL | Elastic PPL | H2O PPL | No Compression PPL |
|---------|-------|----------------|-------------|-------------|---------|-------------------|
| LLaVA-Desc | 7B | 20% | **3.69** | 14.0 | 48.3 | 3.20 |
| LLaVA-Desc | 7B | 50% | **3.41** | 6.31 | 12.9 | 3.20 |
| LLaVA-Desc | 13B | 20% | **3.17** | 5.75 | 10.4 | 2.73 |
| MM-Vet | 7B | 20% | **5.97** | 21.0 | 120 | 5.28 |
| MM-Vet | 13B | 40% | **4.78** | 6.31 | 10.6 | 4.72 |

Inference speedup: At a 20% retention ratio, LLaVA-1.5-7B achieves a 1.8× throughput improvement (363.3 vs. 266.6 tokens/s at batch size 16).

### Ablation Study

| Configuration | 10% PPL | 20% PPL | 30% PPL | 50% PPL |
|---------------|---------|---------|---------|---------|
| Uniform Allocation (Baseline) | 41.8 | 26.6 | 20.4 | 11.8 |
| PyramidKV (manual allocation) | 20.8 | 10.4 | 7.50 | 5.63 |
| **PrefixKV** | **7.38** | **5.97** | **5.72** | **5.50** |
| Offline Estimation | 7.38 | 5.97 | 5.72 | 5.50 |
| Online Search | 7.38 | 5.97 | 5.66 | 5.50 |

### Key Findings

- **Decisive advantage of global prefix configuration**: At a 30% retention ratio, PrefixKV (5.72) outperforms uniform allocation (20.4) by 14.7 PPL—a substantial margin.
- **Amplified gains at low retention ratios**: The performance gap between PrefixKV and competing methods widens as the retention ratio decreases (10%, 20%), underscoring the criticality of adaptive allocation under extreme compression.
- **Highly reliable offline estimation**: A configuration estimated from a single random sample yields PPL virtually identical to that obtained from 10–20 samples.
- **Cross-domain robustness**: Configurations estimated using samples from different domains (VQA, OCR, reasoning) produce stable performance with no significant variance.
- **Greater benefit for 13B models**: LLaVA-1.5-13B experiences almost no performance degradation at retention ratios above 30%.

## Highlights & Insights

- **Elegant problem formulation**: Recasting "per-layer KV allocation" as "global prefix configuration search" and employing Lorenz curves and Gini coefficients for intuitive visualization and quantification is highly original. Borrowing an inequality measurement tool from economics to analyze internal resource allocation in neural networks is a particularly creative perspective.
- **Elegant solution via binary search**: The approach avoids brute-force enumeration or heuristic design, converging in $O(\log N)$ steps. The binary search objective—a unified cumulative priority threshold $p$—is semantically transparent: it ensures each layer retains an "equal proportion" of contextual information.
- **High practical value**: Offline estimation makes the method ready to use out of the box, requiring no modifications to model architecture or retraining, and enabling direct integration into existing LVLM inference pipelines.
- **Transferable to pure LLMs**: Although validated on LVLMs, the core idea of adaptive per-layer KV retention is fully applicable to inference acceleration in text-only LLMs.

## Limitations & Future Work

- **Validation limited to LLaVA-1.5**: Experiments do not cover more recent LVLMs (e.g., LLaVA-Next, InternVL, Qwen-VL), leaving generalizability to be confirmed.
- **Single importance metric**: Only attention scores are used to measure KV importance; alternative metrics (e.g., gradient signals, activation magnitudes) are not explored.
- **No dynamic adjustment**: The prefix configuration is fixed after offline estimation and cannot adapt dynamically to input content, potentially yielding suboptimal results for highly heterogeneous inputs.
- **GQA/MQA architectures not considered**: Modern large models commonly employ GQA/MQA, whose inter-layer KV heterogeneity may differ from standard MHA.
- **Lack of end-to-end task evaluation**: Evaluation relies solely on PPL and ROUGE, without direct assessment of accuracy on downstream visual reasoning tasks.

## Related Work & Insights

- **vs. H2O (Zhang et al., 2024)**: H2O retains important KVs based on attention scores but applies equal retention across all layers; PrefixKV builds upon H2O's importance measure and adds adaptive per-layer allocation, yielding significant advantages at low retention ratios.
- **vs. Elastic Cache (Liu et al., 2024)**: Elastic Cache performs bucket-based merging with equal per-layer allocation; PrefixKV applies pruning without merging, yet surpasses Elastic Cache through superior allocation strategy.
- **vs. PyramidKV (Zhang et al., 2024)**: PyramidKV manually designs a pattern assigning more cache to shallow layers and less to deep layers; PrefixKV automatically determines the allocation via data-driven binary search, achieving superior results.
- **vs. StreamingLLM (Xiao et al., 2023)**: StreamingLLM retains initial and recent tokens via a position-based heuristic, which is complementary to PrefixKV's importance-based approach.

## Rating

- Novelty: ⭐⭐⭐⭐ — The perspective of analyzing inter-layer heterogeneity via Lorenz curves is original; the binary search solution is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage of multiple retention ratios, model scales, datasets, offline/online comparisons, and cross-domain robustness analysis.
- Writing Quality: ⭐⭐⭐⭐ — Problem definition is clear, method derivation is rigorous, and figures are informative.
- Value: ⭐⭐⭐⭐ — A plug-and-play KV compression scheme with direct practical value for LVLM deployment.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching](vla-cache_efficient_vision-language-action_manipulation_via_adaptive_token_cachi.md)
- [\[ICCV 2025\] MM-IFEngine: Towards Multimodal Instruction Following](../../ICCV2025/multimodal_vlm/mm-ifengine_towards_multimodal_instruction_following.md)
- [\[ICCV 2025\] AirCache: Activating Inter-Modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference](../../ICCV2025/multimodal_vlm/aircache_activating_inter-modal_relevancy_kv_cache_compression_for_efficient_lar.md)
- [\[ICLR 2026\] Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models](../../ICLR2026/multimodal_vlm/mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_.md)
- [\[CVPR 2026\] FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression](../../CVPR2026/multimodal_vlm/flashcache_frequency_kv_cache_compression.md)

<!-- RELATED:END -->
