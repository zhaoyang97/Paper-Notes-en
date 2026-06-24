---
title: >-
  [Paper Note] SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers
description: >-
  [ACL 2025][LLM Efficiency][KV cache compression] SpindleKV proposes a layer-aware KV cache compression strategy: active attention-driven token eviction in deep layers (leveraging sparse attention), and similarity-based codebook substitution in shallow layers (leveraging high token similarity), while resolving GQA compatibility issues to achieve a 50% KV cache reduction without performance loss.
tags:
  - "ACL 2025"
  - "LLM Efficiency"
  - "KV cache compression"
  - "token eviction"
  - "codebook"
  - "GQA"
  - "layer-aware"
date: 2026-05-08
content_hash: 90ec3c9846b3c633
---

# SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers

**Conference**: ACL 2025  
**arXiv**: [2507.06517](https://arxiv.org/abs/2507.06517)  
**Code**: [https://github.com/tyxqc/SpindleKV](https://github.com/tyxqc/SpindleKV)  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: KV cache compression, token eviction, codebook, GQA, layer-aware  

## TL;DR
SpindleKV proposes a layer-aware KV cache compression strategy: active attention-driven token eviction in deep layers (leveraging sparse attention), and similarity-based codebook substitution in shallow layers (leveraging high token similarity), while resolving GQA compatibility issues to achieve a 50% KV cache reduction without performance loss.

## Background & Motivation

**Background**: KV cache is the primary memory bottleneck in long-context LLM inference. Existing compression methods include: token eviction (H2O/SnapKV/PyramidKV—removing unimportant tokens based on attention scores), token merging (CaM/D2O—merging similar tokens), and quantization (KIVI/Atom—low-precision representation).

**Limitations of Prior Work**: (1) All existing methods perform well in deep layers but poorly in shallow layers—deep layers have sparse attention and are easy to compress, while shallow layers have dispersed attention, making it difficult to determine which tokens are unimportant; (2) attention score-based eviction methods are incompatible with GQA—a single KV head in GQA serves multiple Q heads, and importance evaluations from different Q heads can conflict; (3) the redundancy in shallow KV cache remains unexploited.

**Key Challenge**: Redundancy patterns differ between deep and shallow layers—deep layers exhibit inter-token redundancy (attention concentrated on a few tokens), whereas shallow layers exhibit inner-token composition redundancy (token vectors are highly similar and can be decomposed into base vectors). A uniform approach cannot optimize both types of redundancy simultaneously.

**Goal**: To balance KV cache compression between deep and shallow layers, thereby improving the overall compression rate.

**Key Insight**: It is observed that the cosine similarity between token vectors in the shallow KV cache is significantly high (since shallow Transformers have fewer encoding steps, resulting in limited contextual differentiation), which allows for codebook substitution.

**Core Idea**: Use attention-based eviction to remove redundant tokens in deep layers, and use learned codebooks to replace highly similar tokens in shallow layers, forming a "spindle-like" compression pattern.

## Method

### Overall Architecture
The KV cache is divided into two parts by layer: deep layers (sparse attention) $\to$ attention score-based token eviction (retaining high-attention tokens); shallow layers (high token similarity) $\to$ similarity learning-based codebook substitution (using a small number of base vectors to approximate the original tokens). The overall compression pattern is "spindle-shaped"—retaining the most in the middle layers, while compressing both shallow and deep layers.

### Key Designs

1. **Deep Layer Token Eviction**:

    - **Function**: Use cumulative attention scores in deep layers to remove unimportant tokens.
    - **Mechanism**: Calculate the cumulative attention score $ac_{i,a}$ within a window and retain the tokens with the highest scores.
    - GQA compatibility improvement: Instead of averaging over Q heads (as done in PyramidInfer), a GQA-aware scoring strategy is designed to avoid conflicts among multiple Q heads.

2. **Shallow Layer Codebook Substitution**:

    - **Function**: Replace highly similar tokens in shallow layers with JIT (Just-in-Time) learned codebook base vectors.
    - Core Observation: The cosine similarity between shallow KV cache tokens is significantly higher than that in deep layers (due to fewer encoding steps and weaker contextual differentiation).
    - Method: Cluster or merge shallow KV cache tokens to learn a set of codebook base vectors, which are then used to approximately replace the original tokens.
    - **Design Motivation**: Eviction is inapplicable because shallow attention is not sparse, but the high similarity among tokens implies that they can be approximated with fewer representative vectors—this is a redundancy exploitation strategy distinct from eviction.

3. **GQA Compatibility**:

    - **Function**: Solve the dilemma of attention-based eviction on GQA models.
    - Problem: A single KV head serves multiple Q heads in GQA, and different Q heads may consider different tokens to be important.
    - The SpindleKV scheme ensures effective compression under GQA settings.

## Key Experimental Results

### Main Results: LongBench + Needle-in-a-Haystack

| Method | KV Compression Rate | LongBench Avg | Needle Accuracy |
|------|----------|-------------|-------------|
| No Compression | 100% | Baseline | Baseline |
| PyramidKV | 50% | Degraded | Degraded |
| PyramidInfer | 50% | Degraded | Degraded |
| **SpindleKV** | **50%** | **≈ Baseline** | **Better than Pyramid** |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| SpindleKV Full | Best | Deep layer eviction + shallow layer codebook |
| Deep eviction only | Limited compression | Shallow layers uncompressed, limiting overall compression rate |
| All-layer eviction | Degraded performance in shallow layers | Eviction is not applicable to shallow layers |
| Shallow codebook only | Limited compression | Deep layers can compress more but are unexploited |

### Key Findings
- **50% KV compression rate is basically lossless**: Validated on multiple LLMs (LLaMA-2/3, Qwen2).
- **SpindleKV shows clear advantages on GQA models**: Resolves compatibility issues of other eviction methods on GQA.
- **Cosine similarity between shallow tokens is indeed significantly higher than in deep layers**: Validates the theoretical basis of shallow codebook substitution.
- **Maintains long-sequence retrieval capability on the Needle-in-a-Haystack task**: Does not degrade performance on information retrieval-sensitive tasks.

## Highlights & Insights
- **Intuition of "spindle" shape**: Deep layer sparsity (evictable), shallow layer similarity (replaceable), and middle layers retained the most—this layer-aware compression strategy is more rational than a uniform strategy.
- **New perspective on shallow redundancy**: Prior work deemed shallow layers hard to compress; SpindleKV discovers that the redundancy pattern of shallow layers (token similarity) differs from deep layers but is equally exploitable.
- **GQA Compatibility**: As GQA becomes mainstream (LLaMA-3, Qwen2, etc.), this compatibility advantage yields practical deployment value.

## Limitations & Future Work
- Codebook learning introduces extra computational overhead (JIT learning).
- Codebook size is a hyperparameter that needs tuning for different models/tasks.
- The boundary between shallow/deep layers is predefined and may benefit from adaptive strategies.
- Has not been explored in combination with quantization methods (KIVI/Atom)—combining quantization + eviction + codebook might yield better results.

## Related Work & Insights
- **vs PyramidKV (Cai et al., 2024)**: PyramidKV finds that deep layers can be compressed more in a pyramid fashion but ignores shallow layers. SpindleKV complements shallow compression via codebook substitution.
- **vs H2O (Zhang et al., 2023)**: H2O performs eviction based on accumulated attention scores; SpindleKV is similar in deep layers but employs a different strategy in shallow layers.
- **vs SnapKV (Li et al., 2024)**: SnapKV performs observation-driven eviction based on attention window features; SpindleKV further distinguishes the behavioral differences across layers.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The idea of layer-aware processing + shallow codebook is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multiple models and benchmarks, including Needle-in-a-Haystack long-sequence evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic from observation to method.
- **Value**: ⭐⭐⭐⭐ Real contribution to the domain of KV cache compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] KV-Latent: Dimensional-level KV Cache Reduction with Frequency-aware Rotary Positional Embedding](kv_latent_cache_reduction.md)
- [\[ACL 2025\] RefreshKV: Updating Small KV Cache During Long-form Generation](refreshkv_updating_small_kv_cache_during_long-form_generation.md)
- [\[ICLR 2026\] DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM Serving](../../ICLR2026/llm_efficiency/dualmap_enabling_both_cache_affinity_and_load_balancing_for_distributed_llm_serv.md)
- [\[ICLR 2026\] ThinKV: Thought-Adaptive KV Cache Compression for Efficient Reasoning Models](../../ICLR2026/llm_efficiency/thinkv_thought-adaptive_kv_cache_compression_for_efficient_reasoning_models.md)
- [\[ACL 2025\] Squeezed Attention: Accelerating Long Context Length LLM Inference](squeezed_attention_accelerating_long_context_length_llm_inference.md)

</div>

<!-- RELATED:END -->
