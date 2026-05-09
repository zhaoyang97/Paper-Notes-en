---
title: >-
  [Paper Note] Revisiting Multimodal KV Cache Compression: A Frequency-Domain-Guided Outlier-KV-Aware Approach
description: >-
  [CVPR2026][Multimodal VLM][KV Cache Compression] This paper proposes FlashCache — the first training-free multimodal KV Cache compression framework that requires no attention scores. By identifying Outlier KVs via frequency-domain low-pass filtering and dynamically allocating per-layer budgets, FlashCache achieves 80% memory reduction and 1.69× decoding speedup while preserving model performance.
tags:
  - CVPR2026
  - Multimodal VLM
  - KV Cache Compression
  - Frequency-Domain Analysis
  - Discrete Cosine Transform
  - Outlier KV
  - Multimodal Inference Acceleration
  - FlashAttention Compatibility
date: 2026-05-08
content_hash: adc0801e7b5e0bf7
---

# Revisiting Multimodal KV Cache Compression: A Frequency-Domain-Guided Outlier-KV-Aware Approach

**Conference**: CVPR2026
**arXiv**: [2511.16786](https://arxiv.org/abs/2511.16786)
**Code**: TBD
**Area**: Multimodal VLM
**Keywords**: KV Cache Compression, Frequency-Domain Analysis, Discrete Cosine Transform, Outlier KV, Multimodal Inference Acceleration, FlashAttention Compatibility

## TL;DR

This paper proposes FlashCache — the first training-free multimodal KV Cache compression framework that requires no attention scores. By identifying Outlier KVs via frequency-domain low-pass filtering and dynamically allocating per-layer budgets, FlashCache achieves 80% memory reduction and 1.69× decoding speedup while preserving model performance.

## Background & Motivation

1. **Multimodal long-context inference bottleneck**: MLLMs face explosive growth in visual tokens under multi-image, high-resolution, and video scenarios, causing KV Cache to grow linearly and imposing significant GPU memory overhead and decoding latency.
2. **Existing methods rely on attention scores**: Methods such as LOOK-M and MEDA select KV pairs based on attention scores; however, efficient attention kernels like FlashAttention do not explicitly output full attention score matrices, and recomputing them introduces additional overhead.
3. **Neglect of Value matrix contributions**: Attention scores are determined solely by Query-Key dot products, so using them directly for KV Cache compression ignores the informational contribution of Value vectors to the attention output.
4. **Incompatibility with efficient attention kernels**: Attention-score-based methods cannot natively support FlashAttention, limiting practical deployment efficiency.
5. **Uniform compression ignores inter-layer differences**: KV matrix information redundancy varies across Transformer layers, and applying a uniform compression ratio leads to suboptimal results.
6. **Inspiration from the frequency domain**: Frequency-domain analysis is widely used in image processing, and outlier removal in model quantization is known to cause severe performance degradation. The authors transfer these two intuitions to KV Cache compression, finding that KV matrix energy concentrates at low frequencies and that KV pairs deviating from the dominant trend are more critical.

## Method

### Overall Architecture: FlashCache

After the prefill stage, FlashCache performs a one-shot compression of the multimodal KV Cache. It consists of two core modules: the **Outlier KV Recognition Module** and the **Dynamic Budget Allocation Module**.

### Outlier KV Recognition Module

1. **Frequency-domain transform**: Discrete Cosine Transform (DCT) is applied to the Key/Value matrices $K^l, V^l$ at each layer, yielding frequency-domain representations $C_k^l[m], C_v^l[m]$.
2. **Low-pass filtering**: A cutoff factor $\gamma$ (optimally 0.1–0.2) is set; components with frequency $m \leq \omega = \gamma \cdot N$ are retained while high-frequency components are zeroed out.
3. **Inverse transform to obtain Base KV**: IDCT is applied to the filtered frequency-domain representation to produce smooth Base KVs $K_{base}^l, V_{base}^l$, capturing the dominant trend of the KV matrices.
4. **Deviation measurement**: The mean squared error between each KV pair and its Base KV counterpart is computed as $Dev[x] = \text{MSE}(K^l[x], K_{base}^l[x]) + \text{MSE}(V^l[x], V_{base}^l[x])$.
5. **Outlier KV retention**: KV pairs are ranked by deviation in descending order, and those with the largest deviations are preferentially retained — these "Outlier KVs" are more likely to encode critical retrieval features.

### Dynamic Budget Allocation Module

1. **Per-layer energy analysis**: Using Parseval's theorem, the power spectrum $P_k^l[m] = |C_k^l[m]|^2$ of each layer's KV matrices is computed in the frequency domain.
2. **Outlier energy ratio**: The ratio of high-frequency (outlier) energy to total energy, $R^l = R_k^l + R_v^l$, is computed for each layer.
3. **Normalized allocation**: The per-layer ratios are normalized into weights, and under a global budget constraint, each layer is assigned a different KV Cache retention quota — layers with higher outlier energy ratios receive larger budgets.

### Loss & Training

FlashCache is a **training-free** method requiring no loss functions or fine-tuning; compression is performed once at inference time.

## Key Experimental Results

### Multi-Image Understanding (MileBench, ρ=0.2)

| Method | Task T | Task S | NH | IR |
|------|--------|--------|------|------|
| Full Cache | 55.59 | 69.17 | 27.35 | 14.17 |
| StreamingLLM | 55.59 | 67.51 | 9.69 | 14.00 |
| SnapKV | 55.59 | 68.27 | 13.59 | 15.33 |
| LOOK-M | 55.55 | 67.50 | 11.88 | 11.83 |
| **FlashCache** | **55.59** | **68.85** | **26.72** | **15.50** |

> On Qwen2.5-VL-7B, FlashCache achieves 26.72 on the Needle-in-a-Haystack task, substantially outperforming the second-best method SnapKV (13.59, +13.13) and approaching Full Cache (27.35).

### High-Resolution & Video Benchmarks

| Benchmark | Full Cache | Best Competing Method | FlashCache (ρ=0.1) |
|------|-----------|-------------|-------------------|
| V* | 80.23 | 79.56 (SnapKV) | **80.23** |
| HR-Bench | 70.75 | 71.12 (SnapKV) | **71.25** |
| FAVOR-Bench (all) | 40.91 | 35.78 (H2O) | **36.49** |

> On V*, FlashCache with ρ=0.1 **exactly matches** Full Cache; on HR-Bench it even marginally exceeds Full Cache.

### Ablation Study

| Ablation | INIAH | GPR1200 | CLEVR-Change |
|--------|-------|---------|-------------|
| w/o DBA | 24.69 | 14.67 | 35.85 |
| w/ DBA | **29.69** | **15.50** | **41.04** |

> The Dynamic Budget Allocation module contributes significantly, yielding a +5.19 gain on CLEVR-Change.
> The optimal low-pass cutoff factor $\gamma$ is 0.1–0.2; excessively large values prevent the Base KV from effectively capturing the dominant trend.

### Efficiency Analysis

- **Decoding speedup**: Up to **1.69×** speedup at ρ=0.2, with decoding latency showing minimal growth as input length increases.
- **Negligible method overhead**: At 8K input length, FlashCache incurs only 6.77ms of additional cost, far lower than LOOK-M (53.97ms) and MEDA (83.75ms).
- **OOM avoidance**: H2O, LOOK-M, and MEDA all run out of memory on MUIRBench, while FlashCache operates normally due to FlashAttention compatibility.

## Highlights & Insights

1. **First multimodal KV compression method without attention scores**: Relying entirely on the distributional properties of KV matrices, it is inherently compatible with efficient attention implementations such as FlashAttention.
2. **Novel frequency-domain perspective**: Frequency-domain analysis from signal processing is introduced into KV Cache compression, revealing the "Outlier KV" phenomenon — KV pairs deviating from the low-frequency dominant trend carry greater importance.
3. **Dynamic per-layer budget allocation**: Compression budgets are adaptively assigned based on the outlier energy intensity of each layer, avoiding a one-size-fits-all approach.
4. **Minimal additional overhead**: CuPy-accelerated DCT computation results in only 6.77ms overhead at 8K input, 12.4× faster than MEDA.
5. **Robustness under extreme compression**: At ρ=0.05, FlashCache still significantly outperforms competing methods, with particularly pronounced advantages on the NH task.

## Limitations & Future Work

1. **One-shot compression after prefill only**: The compression strategy is not dynamically updated during decoding, which may be suboptimal for long-generation scenarios.
2. **Manual specification of cutoff factor γ**: While experiments indicate 0.1–0.2 is optimal, the best value may vary across models and tasks, and an adaptive selection mechanism is lacking.
3. **Less pronounced advantage on video benchmarks**: On FAVOR-Bench, FlashCache outperforms competing methods but still falls notably short of Full Cache, unlike its stronger performance on multi-image and high-resolution tasks.
4. **Validation limited to two model families**: Experiments cover only LLaVA-OneVision and Qwen2.5-VL; generalization to additional architectures (e.g., InternVL, Gemini) remains unverified.
5. **No differentiated compression for Keys and Values**: The same compression strategy is currently applied to both K and V, despite their distinct roles in the attention mechanism; asymmetric treatment may yield further improvements.

## Related Work & Insights

| Method | Requires Attention Scores | FlashAttention Compatible | Training Required | Dynamic Layer Budget |
|------|:---:|:---:|:---:|:---:|
| StreamingLLM | ✓ | ✗ | ✗ | ✗ |
| H2O | ✓ | ✗ | ✗ | ✗ |
| SnapKV | ✓ | ✗ | ✗ | ✗ |
| LOOK-M | ✓ | ✗ | ✗ | ✗ |
| MEDA | ✓ | ✗ | ✗ | ✓ |
| **FlashCache** | **✗** | **✓** | **✗** | **✓** |

FlashCache is the only method that requires no attention scores and is natively compatible with FlashAttention, circumventing the dependency on full attention matrices through frequency-domain analysis.

## Rating

- Novelty: ⭐⭐⭐⭐ — The frequency-domain Outlier KV perspective is novel; applying signal processing ideas to KV compression is a first.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 6 benchmarks, 3 models, multiple compression ratios, with comprehensive ablation and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is developed progressively, and key findings are well-supported by visualizations.
- Value: ⭐⭐⭐⭐ — Highly practical; training-free design and FlashAttention compatibility make it deployment-friendly.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] FlashCache: Frequency-Domain-Guided Outlier-KV-Aware Multimodal KV Cache Compression](flashcache_frequency_kv_cache_compression.md)
- [\[ICLR 2026\] Mixing Importance with Diversity: Joint Optimization for KV Cache Compression in Large Vision-Language Models](../../ICLR2026/multimodal_vlm/mixing_importance_with_diversity_joint_optimization_for_kv_cache_compression_in_.md)
- [\[CVPR 2026\] ApET: Approximation-Error Guided Token Compression for Efficient VLMs](apet_approximation_error_token_compression.md)
- [\[CVPR 2026\] Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)
- [\[CVPR 2026\] Uncertainty-Aware Knowledge Distillation for Multimodal Large Language Models](uncertainty-aware_knowledge_distillation_for_multimodal_large_language_models.md)

<!-- RELATED:END -->
