---
title: >-
  [Paper Note] LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training
description: >-
  [ACL 2025][LLM Efficiency][Positional Encoding] Ours proposes LaMPE (Length-aware Multi-grained Positional Encoding), which adaptively determines the optimal mapping length using a **parameterized scaled sigmoid function**, and designs a **three-region multi-grained attention mechanism** (fine-grained local head + linearly normalized and compressed middle + tail that restores long-range dependencies) to achieve training-free plug-and-play context window extrapolation for LLMs…
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "Positional Encoding"
  - "RoPE"
  - "Context Window Extension"
  - "training-free"
  - "Length Extrapolation"
  - "Multi-grained Attention"
date: 2026-05-08
content_hash: a5dcc2edfdab6e53
---

# LaMPE: Length-aware Multi-grained Positional Encoding for Adaptive Long-context Scaling Without Training

**Conference**: ACL 2025  
**arXiv**: [2508.02308](https://arxiv.org/abs/2508.02308)  
**Code**: [https://github.com/scar-on/LaMPE](https://github.com/scar-on/LaMPE)  
**Authors**: Sikui Zhang, Guangze Gao, Ziyun Gan, Chunfeng Yuan, Zefeng Lin, Houwen Peng, Bing Li, Weiming Hu  
**Area**: LLM / NLP — Long-context Modeling, Positional Encoding  
**Keywords**: Positional Encoding, RoPE, Context Window Extension, training-free, Length Extrapolation, Multi-grained Attention

## TL;DR

Ours proposes LaMPE (Length-aware Multi-grained Positional Encoding), which adaptively determines the optimal mapping length using a **parameterized scaled sigmoid function**, and designs a **three-region multi-grained attention mechanism** (fine-grained local head + linearly normalized and compressed middle + tail that restores long-range dependencies) to achieve training-free plug-and-play context window extrapolation for LLMs, comprehensively outperforming existing methods on five major long-context benchmarks.

## Background & Motivation

- **Background**: RoPE (Rotary Position Embedding) has become the standard positional encoding method for mainstream LLMs (such as Llama, Qwen, Mistral). However, the effective context of models is limited by the window length during the pre-training phase (e.g., 4K for Llama 2, 8K for Llama 3).
- **Limitations of Prior Work**: When the input exceeds the pre-training window, RoPE encounters out-of-distribution (OOD) relative positions, leading to attention collapse. Existing extrapolation methods (such as SelfExtend and DCA) adopt a **fixed mapping strategy**—regardless of how long the input is, the group size $G$ and the mapping range are manually preset constants.
- **Key Challenge**: ① Fixed mapping ignores the **left-skewed frequency distribution** of relative positions during the training phase (short-range positions are fully trained, while long-range positions are severely under-trained), treating all positions equally; ② A fixed group size cannot adapt to different input lengths, as the same $G$ over-compresses short sequences and under-compresses long sequences.
- **Goal**: How to dynamically determine the optimal mapping length based on the input length, and design a multi-grained attention mechanism where spatial resolution varies across different regions?
- **Key Insight**: Through systematic experiments, it is discovered that perplexity exhibits V-shaped or monotonically decreasing patterns with changes in mapping length, and the relationship between the optimal mapping length and input length follows an S-shaped curve, which can be precisely modeled by a sigmoid function.
- **Core Idea**: Adaptively determining the mapping length using a scaled sigmoid function + utilizing a three-region multi-grained positional encoding to capture both fine-grained local information and long-range dependencies.

## Method

### Overall Architecture

LaMPE is a **position indices modification method** applied during the RoPE attention computation stage, consisting of two core components:

1. **Length-aware Dynamic Mapping Strategy**: Computes the optimal mapping length $m$ through a scaled sigmoid function based on the input length $l$.
2. **Multi-grained Attention Mechanism**: Divides the sequence into three regions (head, middle, and tail), applying different granularities of positional encoding to each region.

During inference, it only replaces position indices, **requiring no model parameter modification, no training data, and no extra fine-tuning**, and can be directly integrated with FlashAttention2.

### Key Designs

#### Key Design 1: Length-aware Dynamic Mapping Strategy

The paper systematically explores the **relationship between mapping length and perplexity** on the PG-19 dataset, revealing two key patterns:

- **V-shaped pattern for short inputs**: Perplexity decreases first and then increases, indicating the existence of an optimal mapping length.
- **Monotonically decreasing pattern for long inputs**: Perplexity continues to decrease as the mapping length increases, with the optimal value being the upper bound of the pre-training window.

Based on this, the relationship between the optimal mapping length $m$ and the input length $l$ displays an **S-shaped growth trend**, which can be precisely modeled by a scaled sigmoid function:

$$m = \frac{L}{1 + e^{-(al + b)}}$$

where $L$ is the maximum mapping length (set to 3/4 of the pre-training window), and $a$ and $b$ are parameters obtained through curve fitting on a few sampled points. The core advantages of this design are:

- For short inputs: The mapping length is small, avoiding the waste of positional space.
- For long inputs: The mapping length automatically increases, making full use of the pre-trained positions.
- **Fully adaptive**, eliminating the burden of manual parameter tuning.

#### Key Design 2: Multi-grained Attention Mechanism

After obtaining the optimal mapping length $m$, LaMPE divides the sequence into three regions, each applying a different positional encoding strategy:

**① Head Region ($i-j \leq s_1$)**: Maintains the original 1:1 precise positions, where $PE[i][j] = i - j$. This ensures that the current token maintains a fine-grained positional distinction with its nearest neighbors, which is crucial for fluent text generation.

**② Middle Region ($s_1 < i-j < l - s_2$)**: Adopts linearly normalized compression to map positions into the range $[s_1, m - s_2]$:

$$PE[i][j] = \left\lfloor \frac{m - s_1 - s_2}{l - s_1 - s_2} (i - j - s_1) + s_1 \right\rfloor$$

The compression ratio $m/l$ automatically adjusts with the input length, naturally making the positional granularity of long-range tokens coarser.

**③ Tail Region ($i-j \geq l - s_2$)**: Restores precise positions, where $PE[i][j] = m - l + (i - j)$. This is based on the observation that key instructions or questions often appear at the beginning or end of a sequence, preserving the precise positional relationship between the current token and the starting tokens of the sequence.

The paper proves that the boundaries of the three regions satisfy **monotonicity and continuity**, preventing sudden positional jumps. The optimal hyperparameters are $s_1$ set to 1/8 to 1/16 of the pre-training window, and $s_2$ set to a small value between 8 and 1024.

#### Key Design 3: Seamless Integration with FlashAttention2

The three regions of LaMPE are implemented using different Q/K position indices:
- Head region: Standard sliding-window attention (window_size = $s_1$).
- Middle region: Sliding-window attention computed after modifying the position indices of Q and K.
- Tail region: Modifies only the position indices of Q while K keeps its original positions, using full attention with a lower-triangular mask.

The three parts are merged using the log-sum-exp trick, requiring no modification to the core implementation of FlashAttention2.

## Key Experimental Results

### Main Results

#### Table 1: LongBench (16 tasks) + L-Eval (5 tasks)

| Model | Method | LongBench Avg. | L-Eval Avg. |
|------|------|----------------|-------------|
| Llama2-7B-Chat | Original RoPE | 31.52 | 39.53 |
| | + SelfExtend (25K) | 34.30 | 44.27 |
| | + DCA (25K) | 32.48 | 45.59 |
| | + YaRN (25K) | 31.35 | 41.01 |
| | + NTK (25K) | 25.03 | 35.91 |
| | **+ LaMPE** | **35.07** | **48.13** |
| Llama3-8B-Ins | Original RoPE | 42.38 | 67.07 |
| | + SelfExtend (32K) | 42.22 | 69.39 |
| | + DCA (32K) | 44.70 | 69.93 |
| | + YaRN (32K) | 45.90 | 70.79 |
| | + NTK (32K) | 44.24 | 68.75 |
| | **+ LaMPE** | **46.99** | **71.78** |

LaMPE outperforms the best baselines on both models by 0.45/1.09 (LongBench) and 2.54/0.99 (L-Eval).

#### Table 2: ∞Bench (Ultra-long context, all inputs >64K tokens)

| Model | Method | En.MC | En.QA | En.sum | Code | Re.KV | Re.Num | Re.Pass | Avg. |
|------|------|-------|-------|--------|------|-------|--------|---------|------|
| Llama3 (32K) | SelfExtend | 50.66 | 14.06 | 15.13 | 24.87 | 3.60 | 27.12 | 27.12 | 23.22 |
| | DCA | 52.84 | 13.90 | 18.79 | 25.38 | 4.40 | 27.12 | 27.12 | 24.22 |
| | **LaMPE** | **55.02** | **16.36** | **20.49** | 25.63 | **17.40** | 27.12 | 27.12 | **27.02** |
| Llama3 (64K) | SelfExtend | 53.71 | 15.10 | 15.22 | 21.57 | 2.80 | 54.24 | 54.24 | 30.98 |
| | DCA | 50.66 | 14.35 | 18.98 | 24.11 | 2.00 | 52.88 | 54.24 | 31.03 |
| | **LaMPE** | **55.90** | **15.49** | **23.10** | 24.11 | **10.80** | 54.24 | 54.24 | **33.98** |
| Llama3.1 (128K) | Original RoPE | 67.25 | 14.57 | 25.42 | 22.08 | 54.80 | 99.49 | 100.00 | 54.80 |
| | STRING | 71.18 | 14.39 | 27.81 | 30.46 | 81.40 | 99.83 | 100.00 | 60.72 |
| | **LaMPE** | 70.30 | **19.51** | **28.54** | 29.19 | **92.60** | 99.83 | 100.00 | **62.85** |

On the KV retrieval task, LaMPE applied to Llama 3.1 outperforms original RoPE by **37.8 points** (92.60 vs 54.80).

### Key Findings

#### PG-19 Perplexity (PPL) and RULER Benchmark

| Model | Method | PPL Avg. (4K-64K) | RULER 8K | RULER 16K | RULER 64K | RULER 128K |
|------|------|-------------------|----------|-----------|-----------|------------|
| Llama2 | DCA | 7.23 | - | - | - | - |
| | **LaMPE** | **7.00** | - | - | - | - |
| Llama3 | Original RoPE | - | 88.76 | - | - | - |
| | SelfExtend | 7.60 | 87.59 | 75.44 | 61.95 | 35.97 |
| | DCA | 7.43 | 89.35 | 72.28 | 47.01 | 15.96 |
| | YaRN | 7.41 | - | 62.93 | 5.02 | 12.17 |
| | **LaMPE** | **7.23** | **90.57** | **87.32** | **69.46** | **59.48** |

LaMPE reaches 59.48 on RULER 128K, which is **1.65 times** higher than the runner-up SelfExtend (35.97). YaRN collapses to 5.02 at 64K.

## Highlights & Insights

1. **Empirical-driven theoretical discovery**: Through systematic experiments, the V-shape and monotonic decay modes of PPL are discovered. The S-shaped relationship between the optimal mapping length and the input length is precisely modeled using a sigmoid function, converting heuristic tuning into a curve-fitting problem based on a few sampling points.

2. **Elegant division of labor in the three-region design**: The head preserves local coherence, the middle performs adaptive compression, and the tail restores long-range dependencies. Each region's design has a clear cognitive motivation (adjacent tokens require fine distinction, middle tokens tolerate coarse granularity, while instructions/questions at the start or end of a sequence require precise positions).

3. **Outperforming native long-context models**: On the KV retrieval task of ∞Bench, LaMPE applied to Llama3.1-8B-Instruct-128K achieves 92.60, greatly surpassing native RoPE (54.80). This suggests that optimizing positional mapping can be more effective than simply expanding the training window.

4. **Performance gains within the pre-training window**: LaMPE not only performs well in extrapolation but also improves performance within the original pre-training window (RULER 8K: 90.57 vs 88.76), benefiting from leveraging the left-skewed position frequency distribution.

5. **Free from manual parameter tuning**: Completely eliminates the burden of manually tuning $G$ and $w$ as in SelfExtend. The sigmoid function parameters are automatically determined using simple curve-fitting.

## Limitations & Future Work

1. **Validated only on Llama family**: Experiments cover Llama2-7B-Chat, Llama3-8B-Instruct, and Llama3.1-8B-Instruct, but exclude other RoPE-based architectures like Qwen, Mistral, and Phi. Generalization remains to be fully verified.

2. **Model-dependence of sigmoid parameters**: Although sigmoid parameters can be fitted with a few samples, they need to be re-fitted for different models, introducing a one-time exploration cost.

3. **Remaining bottlenecks in precise retrieval tasks like KV retrieval**: Although LaMPE significantly improves KV retrieval, linear compression in the middle region still causes loss of precise location information. Performance on ultra-long multi-needle retrieval tasks (NIAH_M3) remain low (only 1.20 at 128K).

4. **Combination with base-modified methods**: The paper points out that methods like NTK-RoPE and YaRN are orthogonal and combinable with LaMPE, but does not provide actual experimental results of such combinations.

5. **Setting of the tail region size $s_2$**: Although experiments show $s_2 = 8$ is sufficient to recover most of the performance, finding the optimal value still relies on empirical judgment and lacks theoretical guidance.

## Related Work & Insights

| Method | Type | Requires Training | Mapping Strategy | Length-Adaptive | Core Limitation |
|------|------|-----------|---------|----------|---------|
| NTK-RoPE | base-modified | No (Optional FT) | Modify frequency base | No | Has extrapolation upper bound, performance collapses at 64K+ |
| YaRN | base-modified | No (Optional FT) | Frequency scaling | No | Extrapolation upper bound is lower than indices-modification methods |
| SelfExtend | indices-modified | No | Fixed group size $G$ | No (Manual $G$) | Crude uniform grouping, performance within window may drop |
| DCA | indices-modified | No | Chunk-based mapping | No (Manual setup) | Stable but cannot leverage position frequency distribution |
| STRING | indices-modified | No | Leverage frequency distribution | Partial | Mainly enhances performance within window |
| **LaMPE** | indices-modified | **No**| **Sigmoid dynamic mapping + Three regions** | **Yes** | Validated only on Llama family |

**Inspirational Directions**:
- The three-region concept of LaMPE can be transferred to **KV Cache compression**: Maintaining original precision for near-range KV and compressing or merging far-range KV by region.
- The concept of sigmoid modeling for mapping length can be applied to **multimodal long sequences** (e.g., adaptive compression of inter-frame positional encodings in video understanding).
- The discovery of performance gains within the pre-training window indicates that **optimizing positional encoding utilization efficiency** is itself a promising research direction, even without extrapolation.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of sigmoid dynamic mapping and three-region multi-grained mechanism is novel, and the empirical discovery of V-shaped/monotonic decay patterns is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage of five major benchmarks (LongBench/L-Eval/∞Bench/RULER/PG-19) with complete ablation studies and hyperparameter analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, with a smooth logical flow from empirical observation $\rightarrow$ mathematical modeling $\rightarrow$ experimental validation, accompanied by intuitive charts.
- Value: ⭐⭐⭐⭐ Training-free long-context scaling is highly demanded. The properties of plug-and-play, adaptiveness, and FlashAttention2 compatibility offer high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Mitigating Posterior Salience Attenuation in Long-Context LLMs with Positional Contrastive Decoding](mitigating_posterior_salience_attenuation_in_long-context_llms_with_positional_c.md)
- [\[ACL 2025\] Scaling Context, Not Parameters: Training a Compact 7B Language Model for Efficient Long-Context Processing](scaling_context_not_parameters_training_a_compact_7b_language_model_for_efficien.md)
- [\[ICML 2025\] NExtLong: Toward Effective Long-Context Training without Long Documents](../../ICML2025/llm_efficiency/nextlong_toward_effective_long-context_training_without_long_documents.md)
- [\[ACL 2025\] Squeezed Attention: Accelerating Long Context Length LLM Inference](squeezed_attention_accelerating_long_context_length_llm_inference.md)
- [\[ACL 2025\] SEAL: Scaling to Emphasize Attention for Long-Context Retrieval](seal_scaling_to_emphasize_attention_for_long-context_retrieval.md)

</div>

<!-- RELATED:END -->
