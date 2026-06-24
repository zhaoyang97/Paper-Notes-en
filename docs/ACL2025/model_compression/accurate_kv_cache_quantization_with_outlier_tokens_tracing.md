---
title: >-
  [Paper Note] Accurate KV Cache Quantization with Outlier Tokens Tracing
description: >-
  [ACL 2025][Model Compression][KV Cache] It is discovered that a small number of outlier tokens in the outlier channels of KV Cache deviate from the previously assumed uniform distribution. To address this, the Outlier Tokens Tracing (OTT) method is proposed to dynamically trace and exclude these tokens during the quantization process. Under 2-bit quantization, this approach achieves a 6.4x memory compression and a 2.3x throughput speedup while significantly improving accuracy…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "KV Cache"
  - "quantization"
  - "Outlier Tokens"
  - "LLM Inference"
  - "Memory Efficiency"
date: 2026-05-08
content_hash: 61f13b72bfba1b12
---

# Accurate KV Cache Quantization with Outlier Tokens Tracing

**Conference**: ACL 2025  
**arXiv**: [2505.10938](https://arxiv.org/abs/2505.10938)  
**Code**: [https://github.com/yisunlp/OTT](https://github.com/yisunlp/OTT)  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: KV Cache, quantization, Outlier Tokens, LLM Inference, Memory Efficiency

## TL;DR

It is discovered that a small number of outlier tokens in the outlier channels of KV Cache deviate from the previously assumed uniform distribution. To address this, the Outlier Tokens Tracing (OTT) method is proposed to dynamically trace and exclude these tokens during the quantization process. Under 2-bit quantization, this approach achieves a 6.4x memory compression and a 2.3x throughput speedup while significantly improving accuracy.

## Background & Motivation

In autoregressive LLM inference, although KV Cache reduces the computational complexity from $O(n^2)$ to $O(n)$, it introduces significant GPU memory overhead. For LLaMA-3-8B, with a batch size of 64 and a sequence length of 8192, the KV Cache requires 256GB of memory.

KV Cache quantization is an effective compression method. Prior work (KIVI) discovered:
- **Keys are distributed channel-wise**: Certain channels have extremely large magnitudes, while the distribution within the same channel is relatively uniform $\rightarrow$ suitable for channel-wise quantization.
- **Values are distributed token-wise** $\rightarrow$ suitable for token-wise quantization.

However, further exploration reveals that **in the outlier channels, several outlier tokens have extremely small Key values, which severely deviates from the "uniform distribution within the same channel" assumption**. These outlier tokens greatly expand the quantization range $(X_{max} - X_{min})$, severely undermining the quantization accuracy.

## Method

### Overall Architecture

Building upon the channel-wise Key + token-wise Value quantization of KIVI, OTT introduces an **outlier pool** to store the full-precision KV representations of the identified outlier tokens. During inference, three types of KV Cache are maintained: quantized, full-precision (group tokens + recent tokens), and those in the outlier pool.

### Key Designs

**1. Identification and Characteristics of Outlier Tokens**

- In the outlier channels (channels with extremely large magnitudes), a small number of tokens have abnormally small Key values, breaking the original uniform distribution.
- Sorting reveals a clear "cliff-like" jump: a few extremely low values suddenly rise to very high values.
- These tokens also tend to have smaller Key magnitudes **across all channels**, making the overall Key magnitude a viable identification metric.
- The outlier token phenomenon is **absent in shallow layers** (the first 1-2 layers).

**2. Outlier Token Identification (Quantization Stage)**

- Define a fixed-size outlier pool (capacity of `outlier_num`, default is 3).
- Perform quantization every $G$ steps (group size).
- For the $G$ tokens to be quantized and the tokens in the outlier pool, calculate the Key magnitude of each token.
- **Tokens with the smallest magnitudes are selected into the outlier pool** (as they cause the largest quantization error).
- The Keys and Values of the selected outlier tokens are replaced with the mean value of all tokens to eliminate their impact on quantization.
- An extra pool (size 32) is maintained to store tokens evicted from the outlier pool; tracking stops once this pool is full.

**3. Decoding Stage**

- Queries are multiplied with the three types of Keys separately and then concatenated to obtain the attention scores.
- The attention scores are multiplied by the corresponding three types of Values and summed to produce the final output.
- A **CUDA fused kernel** is used to accelerate the mixed multiplication of full-precision and quantized matrices.

**4. Key Differences from KIVI**

- KIVI compresses Values at every step and Keys every $G$ steps.
- OTT compresses both Keys and Values every $G$ steps, leading to more consistent processing.
- OTT's residual length (the sliding window for recent tokens) is set to 32 (compared to 128 in KIVI), as the outlier pool already compensates for accuracy.

### Loss & Training

OTT is a **training-free** method that does not require any training or fine-tuning. All operations are conducted online during inference, primarily relying on simple sorting and bucketing based on Key magnitudes.

## Key Experimental Results

### Main Results

**Normal Context Evaluation (GSM8K, BBH, HumanEval)**:

| Dataset | LLaMA-3-8B FP16 | KIVI 2-bit | OTT 2-bit |
|--------|-----------------|------------|-----------|
| GSM8K (8-shot) | 74.91 | 63.15 | **72.55** |
| BBH (3-shot CoT) | 68.18 | 47.38 | **60.31** |
| HumanEval (p@1) | 40.24 | 28.05 | **40.85** |

On BBH (3-shot CoT, LLaMA-3-8B), OTT improves performance by **12.93%** compared to KIVI, almost recovering to the FP16 baseline.

**Long Context Evaluation (LongBench)**:

| Model | KIVI Avg | OTT Avg | FP16 Avg |
|------|---------|---------|----------|
| LLaMA-3-8B-Instruct | 47.13 | **49.18** | 50.22 |
| LLaMA-2-13B-chat | 43.06 | **44.10** | 44.30 |

KIVI suffers a significant performance drop on the LCC task of LLaMA-3-8B (56.58 $\rightarrow$ 44.42), whereas OTT (52.37) avoids such a collapse.

**Efficiency Comparison**:
- Memory compression: Both OTT and KIVI achieve **6.4x** (when sequences are sufficiently long).
- Throughput improvement: OTT achieves up to **2.3x** (superiority is more pronounced with large batch sizes).
- OTT is faster than KIVI under any batch size because it avoids compressing Values at each step.

### Ablation Study

**1. Group Size and Residual Length**

- Increasing residual length significantly improves accuracy (with fixed G=128, as R increases from 0 to 128, GSM8K 8-shot performance rises from 70.96 to 73.24).
- Changes in group size have minimal impact (once the distribution is uniform enough, increasing G has a limited effect).
- The main experiments use G=128, R=32 to balance performance and compression ratio.

**2. Outlier Num**

| outlier_num | GSM8K (8-shot) | GSM8K (8-CoT) |
|-------------|---------------|--------------|
| 0 | 62.09 | 68.31 |
| 1 | 71.80 | 75.74 |
| 3 | 72.55 | 75.06 |
| 6 | 72.18 | 75.89 |

Keeping just 1 outlier token yields a ~10% improvement, after which marginal returns diminish.

**3. Shallow Layer Handling**

Setting `outlier_num=0` for the first 1-2 layers yields the best results (confirming the absence of outlier tokens in shallow layers). Setting more layers to 0 degrades performance.

### Key Findings

1. The Keys of outlier tokens have extremely small magnitudes in the outlier channels, creating a "cliff" compared to other tokens.
2. These tokens also have the smallest **sum of Key magnitudes across all channels**, allowing them to be efficiently and accurately identified using the Key magnitude.
3. Retaining tokens with the smallest Key magnitude in full precision works best (significantly better than retaining the largest).
4. No outlier token phenomenon occurs in the shallow layers (first 2 layers).

## Highlights & Insights

1. **Observation-Driven Simple Design**: Starting with detailed observations of the token distribution in outlier channels, the method is remarkably simple (requiring tracking of only 3 tokens) yet surprisingly effective.
2. **Training-free**: Directly deployable without requiring any additional training.
3. **Hardware-friendly**: Utilizing a CUDA fused kernel, the overhead of handling outliers during actual inference is negligible.
4. **Strong Complementarity**: Can be combined with weight quantization, token eviction, and other methods.

## Limitations & Future Work

1. The compression ratio decreases when the sequence is very short (< group size) and the batch size is very large.
2. A minor performance loss still exists under 2-bit quantization on certain challenging datasets, and errors might accumulate during long generation.
3. The performance under 3-bit or 4-bit quantization has not been explored (where performance might already be close enough to FP16 to render this method unnecessary).
4. The strategy of stopping tracking once the outlier pool is full is relatively simple.

## Related Work & Insights

- **KIVI**: The foundational work on channel-wise Key + token-wise Value quantization, upon which OTT is directly built.
- **StreamingLLM**: The idea of retaining initial and local tokens shares similarities with the outlier pool.
- **Massive Activations**: The massive activation patterns discovered by Sun et al. have a potential connection with the outlier token phenomenon.
- **GPTQ / AWQ**: Weight quantization methods that can complement KV Cache quantization.

## Rating

- **Novelty**: ★★★★☆ — Insightful and meticulous observation of outlier tokens, leading to a simple and effective method.
- **Value**: ★★★★★ — Training-free, hardware-friendly, and offers significant memory savings, bearing high engineering value.
- **Experimental Thoroughness**: ★★★★☆ — Evaluated across 4 model families and various benchmarks, with sufficient ablation studies and efficiency comparisons.
- **Writing Quality**: ★★★★☆ — Clearly structured, naturally motivated, and rich in figures/tables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)
- [\[ACL 2025\] UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](uniquanf_unified_quantization.md)
- [\[ACL 2025\] Outlier-Safe Pre-Training for Robust 4-Bit Quantization of Large Language Models](outlier-safe_pre-training_for_robust_4-bit_quantization_of_large_language_models.md)
- [\[ICML 2026\] OSAQ: Outlier Self-Absorption for Accurate Low-bit LLM Quantization](../../ICML2026/model_compression/osaq_outlier_self-absorption_for_accurate_low-bit_llm_quantization.md)
- [\[ACL 2025\] Quaff: Quantized Parameter-Efficient Fine-Tuning under Outlier Spatial Stability Hypothesis](quaff_quantized_peft.md)

</div>

<!-- RELATED:END -->
