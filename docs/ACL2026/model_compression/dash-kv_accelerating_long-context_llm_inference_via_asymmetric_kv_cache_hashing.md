---
title: >-
  [Paper Note] DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing
description: >-
  [ACL 2026][Model Compression][KV Cache] The DASH-KV framework is proposed to reformulate the attention mechanism as an approximate nearest neighbor search problem. By mapping high-dimensional floating-point similarity co…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "KV Cache"
  - "Deep Hashing"
  - "Asymmetric Encoding"
  - "Attention Acceleration"
  - "Long-context Inference"
date: 2026-05-08
content_hash: 3c636f122a2ef0ef
---

# DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.19351](https://arxiv.org/abs/2604.19351)  
**Code**: [https://github.com/Zhihan-Zh/DASH-KV](https://github.com/Zhihan-Zh/DASH-KV)  
**Area**: Model Compression  
**Keywords**: KV Cache, Deep Hashing, Asymmetric Encoding, Attention Acceleration, Long-context Inference

## TL;DR

The DASH-KV framework is proposed to reformulate the attention mechanism as an approximate nearest neighbor search problem. By mapping high-dimensional floating-point similarity computations to efficient Hamming distance bit operations via asymmetric deep hashing, and utilizing a dynamic mixed-precision mechanism, the complexity of long-context inference is reduced from $O(N^2)$ to $O(N)$ while matching full-attention performance.

## Background & Motivation

**Background**: The quadratic complexity of the standard attention mechanism is a fundamental bottleneck for long-context inference in LLMs. Existing KV cache optimization methods, including quantization, selective eviction, and structured sharing, do not change the underlying floating-point computation paradigm.

**Limitations of Prior Work**: (1) Quantization methods suffer from severe performance degradation at ultra-low bits (1-2 bits), and dequantization introduces additional overhead. (2) Selective eviction leads to irreversible information loss, harming long-range dependency tasks. (3) Structured sharing ignores the heterogeneous characteristics across different heads and layers. Crucially, all these methods still operate within a floating-point calculation framework without fundamentally addressing the computation paradigm.

**Key Challenge**: Query-Key similarity computation requires billions of high-precision floating-point operations. All existing optimizations operate within this floating-point framework. A completely new computational paradigm is needed to replace floating-point similarity calculations.

**Goal**: Transform attention computation from a floating-point paradigm to a binary bit manipulation paradigm to achieve fundamental acceleration.

**Key Insight**: Query-Key similarity matching in attention is highly analogous to relevance matching in information retrieval. Deep hashing has proven effective in large-scale retrieval by encoding high-dimensional vectors into compact binary codes and replacing dot products with Hamming distance calculations.

**Core Idea**: Reformulate attention computation as an approximate nearest neighbor search by asymmetrically encoding the Query (via a high-precision MLP) and the Key (via light linear projection). Hamming distances are calibrated using cross-head consensus and cross-layer momentum, while full-precision computation is preserved for critical tokens.

## Method

### Overall Architecture

DASH-KV consists of three core components: (1) Asymmetric Hashing—Queries are encoded via a 3-layer MLP and Keys via linear projection into binary hash codes; (2) Calibrated Hamming Distance Retrieval—Coarse-grained hashing distances are corrected using cross-head consensus and cross-layer momentum; (3) Dynamic Mixed-Precision Attention—Keys are categorized into three levels (high correlation: full precision; medium correlation: hash + residual compensation; low correlation: skip) based on calibrated distances.

### Key Designs

1. **Asymmetric Deep Hashing Encoding**:

    - **Function**: Differentially maps Queries and Keys into a binary hash space.
    - **Mechanism**: Queries must capture dynamic semantics with high precision, thus a 3-layer MLP ($d \to 256 \to 256 \to l$) is used. During training, a progressive annealing strategy $\tanh(\beta \cdot v_q)$ (with $\beta$ increasing from 1 to 10) simulates the sign function, which is directly taken during inference. Keys require efficient encoding for reuse and use a single-layer linear projection $h_k = \text{sign}(W_k K)$. This asymmetric design ensures Query precision while prioritizing Key efficiency.
    - **Design Motivation**: Queries are dynamically generated and unique per step, requiring precise encoding; Keys are extensively reused once cached, prioritizing encoding speed and storage efficiency. Symmetric encoding fails to satisfy these distinct requirements simultaneously.

2. **Cross-head Consensus and Cross-layer Momentum Calibration**:

    - **Function**: Corrects biases in coarse-grained Hamming distances.
    - **Mechanism**: Cross-head Consensus—Statistics are gathered on how many attention heads select the same Key (exceeding threshold $T_{\text{vote}}$), providing a distance discount $\Delta_{\text{spatial}}$ for Keys with multi-head consensus. Cross-layer Momentum—Prior information from the previous layer's attention distribution is utilized, providing a distance discount $\Delta_{\text{temporal}}$ for Keys under continuous attention. The final distance is $D_{\text{final}} = D_{\text{raw}} + \Delta_{\text{spatial}} + \Delta_{\text{temporal}}$, where discount coefficients are learnable.
    - **Design Motivation**: Pure Hamming distance is a coarse approximation; structural Transformer priors (multi-head and multi-layer) are needed for correction.

3. **Dynamic Importance Mixed-Precision Attention**:

    - **Function**: Performs instance-level fine-grained trade-offs between efficiency and accuracy.
    - **Mechanism**: Adaptive percentile thresholds (rather than fixed ones) categorize Keys into three tiers: high correlation ($D \leq t_1$) retains full precision; medium correlation ($t_1 < D \leq t_2$) uses "hashing + residual compensation" where hashing inner products provide a coarse estimate and a lightweight MLP fits the residual $\Delta(h_q, h_k; \phi)$; low correlation ($D > t_2$) skips computation without discarding the data (unlike eviction). Special tokens (CLS/SEP/sink/neighboring tokens) are forced to full precision.
    - **Design Motivation**: Not all tokens are equally important; while hashing everything loses critical information, using full precision everywhere sacrifices acceleration. Tiered stratification allocates precision as needed.

### Loss & Training

The main loss is list-wise distillation $\mathcal{L}_{\text{distill}} = \text{KL}(P_{\text{student}} \| P_{\text{teacher}})$, combined with a bit balance loss $\mathcal{L}_{\text{bal}}$ and a quantization loss $\mathcal{L}_{\text{quant}}$, each with an auxiliary coefficient of 0.1. Asymmetric temperature scaling is employed to manage the over-smoothed distribution of hashing inner products.

## Key Experimental Results

### Main Results

| Method | Qwen2-7B LongBench Avg. | Complexity |
|------|------------------------|--------|
| Full Attention | Baseline | $O(N^2)$ |
| H2O (Eviction) | Below Baseline | Linear |
| SnapKV (Eviction) | Below Baseline | Linear |
| KIVI (Quantization) | Below Baseline | Near Linear |
| DASH-KV | **Matches Baseline** | **$O(N)$** |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Hashing Only (No Calibration) | Perf. Drop | Inaccurate coarse-grained distance |
| + Cross-head Consensus | Gain | Multi-head voting reduces misjudgments |
| + Cross-layer Momentum | Further Gain | Temporal priors are effective |
| + Mixed Precision | Optimal | Critical tokens maintain precision |

### Key Findings

- DASH-KV matches Full Attention performance on LongBench while achieving linear complexity.
- Eviction methods suffer performance degradation due to irreversible information loss, whereas DASH-KV discards no information.
- Asymmetric encoding outperforms symmetric encoding, validating the need for differential treatment of Queries and Keys.
- A hash code length $l$ between 32-64 bits achieves the optimal balance between performance and efficiency.

## Highlights & Insights

- **The "Attention as Retrieval" perspective is highly inspiring**: Transforming attention from a computation problem to a retrieval problem allows the adoption of mature Information Retrieval techniques (deep hashing), opening a new path for optimization.
- **Asymmetric design reflects a deep understanding of Q/K roles**: Queries are transient and require precision, while Keys are recurrently reused and require efficiency. This distinction was often ignored in prior work.
- **Information-retaining philosophy**: Unlike eviction methods, low-correlation Keys are merely skipped rather than permanently deleted, allowing them to be "awakened" in subsequent steps.

## Limitations & Future Work

- Requires training hash encoders (though lightweight, they still represent a cost) and is not strictly plug-and-play.
- The two learnable parameters for consensus and momentum require tuning.
- Validation is currently limited to LongBench; other benchmarks (e.g., RULER, ∞-Bench) have not been tested.
- The residual compensation MLP may require specific adjustments for different models.

## Related Work & Insights

- **vs H2O/SnapKV (Eviction)**: Eviction permanently loses information, while DASH-KV retains all Keys and only skips low-relevance computations, achieving a superior balance between information retention and efficiency.
- **vs KIVI/Atom (Quantization)**: Quantization still operates within the floating-point framework; DASH-KV fundamentally changes the paradigm via bit manipulation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Introducing deep hashing into attention is a first; asymmetric design and three-tier mixed precision are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three models + LongBench + detailed ablation, though benchmark coverage is somewhat limited.
- Writing Quality: ⭐⭐⭐⭐ The method description is systemic and detailed, although formulas are dense and some descriptions are lengthy.

## Related Papers

<div class="related-papers" markdown="1">

</div>

## Related Papers

- [\[ACL 2026\] HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)
- [\[ICML 2025\] RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression](../../ICML2025/model_compression/rocketkv_accelerating_long-context_llm_inference_via_two-stage_kv_cache_compress.md)
- [\[ACL 2026\] FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration](fastkv_decoupling_of_context_reduction_and_kv_cache_compression_for_prefill-deco.md)
- [\[ACL 2026\] The Pitfalls of KV Cache Compression](the_pitfalls_of_kv_cache_compression.md)
- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](../../NeurIPS2025/model_compression/chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)
- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](../../NeurIPS2025/model_compression/chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[NeurIPS 2025\] KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments](../../NeurIPS2025/model_compression/keydiff_key_similarity-based_kv_cache_eviction_for_long-context_llm_inference_in.md)
- [\[ACL 2026\] FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration](fastkv_decoupling_of_context_reduction_and_kv_cache_compression_for_prefill-deco.md)
- [\[ICML 2026\] A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints](../../ICML2026/model_compression/a_queueing-theoretic_framework_for_stability_analysis_of_llm_inference_with_kv_c.md)

</div>

<!-- RELATED:END -->
