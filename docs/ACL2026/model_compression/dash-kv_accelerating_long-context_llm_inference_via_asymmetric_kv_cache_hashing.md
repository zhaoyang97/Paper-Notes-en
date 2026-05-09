---
title: >-
  [Paper Note] DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing
description: >-
  [ACL 2026][Model Compression][To be supplemented] To be supplemented after thorough reading.
tags:
  - ACL 2026
  - Model Compression
  - To be supplemented
date: 2026-05-08
content_hash: 3336a352660fbe84
---

# DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing

**Conference**: ACL 2026
**arXiv**: [2604.19351](https://arxiv.org/abs/2604.19351)
**Code**: [https://github.com/Zhihan-Zh/DASH-KV](https://github.com/Zhihan-Zh/DASH-KV)
**Area**: Model Compression
**Keywords**: KV cache, deep hashing, asymmetric encoding, attention acceleration, long-context inference

## TL;DR

This paper proposes DASH-KV, a framework that reformulates the attention mechanism as an approximate nearest neighbor search problem. By employing asymmetric deep hashing to encode queries and keys into binary codes, high-dimensional floating-point similarity computation is replaced with efficient Hamming distance bit operations. Combined with a dynamic mixed-precision mechanism, the approach reduces long-context inference complexity from $O(N^2)$ to $O(N)$ while matching the performance of full attention.

## Background & Motivation

**Background**: The quadratic complexity of standard attention is the fundamental bottleneck for long-context LLM inference. Existing KV cache optimization approaches include quantization, selective eviction, and structured sharing, yet none fundamentally alter the underlying floating-point computation paradigm.

**Limitations of Prior Work**: (1) Quantization methods suffer severe performance degradation at ultra-low bit-widths (1–2 bits), and dequantization introduces additional overhead; (2) selective eviction causes irreversible information loss, impairing long-range dependency tasks; (3) structured sharing ignores the heterogeneous characteristics across different heads and layers. More critically, all these methods still operate within the floating-point computation framework and do not fundamentally address the computational paradigm problem.

**Key Challenge**: Query–Key similarity computation requires billions of high-precision floating-point operations, yet all existing optimizations merely work within this floating-point framework. A fundamentally new computational paradigm is needed to replace floating-point similarity computation.

**Goal**: To shift attention computation from a floating-point paradigm to a binary bit-operation paradigm, achieving fundamental acceleration.

**Key Insight**: Query–Key similarity matching in attention closely resembles relevance matching in information retrieval. Deep hashing has proven effective in large-scale retrieval — encoding high-dimensional vectors into compact binary codes and replacing dot-product computation with Hamming distance.

**Core Idea**: The attention computation is reformulated as approximate nearest neighbor search. Queries and keys are encoded asymmetrically — queries via a high-precision MLP and keys via a lightweight linear projection — into binary hash codes. Cross-head consensus and cross-layer momentum are used to calibrate hash distances, and full-precision computation is preserved for critical tokens.

## Method

### Overall Architecture

DASH-KV comprises three core components: (1) **Asymmetric Hashing** — queries are encoded via a 3-layer MLP and keys via a linear projection, both mapped to binary hash codes; (2) **Calibrated Hamming Distance Retrieval** — cross-head consensus and cross-layer momentum are applied to refine the coarse-grained hash distances; (3) **Dynamic Mixed-Precision Attention** — based on calibrated distances, keys are partitioned into three tiers: highly relevant (full precision), moderately relevant (hashing with residual compensation), and low relevant (computation skipped).

### Key Designs

1. **Asymmetric Deep Hash Encoding**

    - **Function**: Differentially maps queries and keys into binary hash space.
    - **Mechanism**: Queries require high-precision capture of dynamic semantics and are encoded by a 3-layer MLP ($d \to 256 \to 256 \to l$); during training, a progressive annealing strategy using $\tanh(\beta \cdot v_q)$ (with $\beta$ increasing from 1 to 10) approximates the sign function, while at inference the sign function is applied directly. Keys require efficient encoding and are reused repeatedly, so a single-layer linear projection $h_k = \text{sign}(W_k K)$ is employed. The asymmetric design preserves precision on the query side and prioritizes efficiency on the key side.
    - **Design Motivation**: Queries are generated dynamically and differ at each step, necessitating precise encoding. Keys, once written to the cache, are reused extensively, making encoding speed and storage efficiency the priority. Symmetric encoding cannot simultaneously satisfy these differing requirements.

2. **Cross-Head Consensus and Cross-Layer Momentum Calibration**

    - **Function**: Corrects the bias introduced by coarse-grained Hamming distances.
    - **Mechanism**: *Cross-head consensus* — the number of attention heads that select a given key is counted; keys selected by more than a threshold $T_{\text{vote}}$ receive a distance discount $\Delta_{\text{spatial}}$. *Cross-layer momentum* — the attention distribution from the previous layer serves as a prior; keys that are consistently attended to receive a discount $\Delta_{\text{temporal}}$. The final distance is $D_{\text{final}} = D_{\text{raw}} + \Delta_{\text{spatial}} + \Delta_{\text{temporal}}$, where the discount coefficients are learnable.
    - **Design Motivation**: Raw Hamming distance is a coarse approximation; the multi-head and multi-layer structure of Transformers provides useful priors that can be leveraged for correction.

3. **Dynamic Importance-Aware Mixed-Precision Attention**

    - **Function**: Enables instance-level fine-grained trade-offs between efficiency and precision.
    - **Mechanism**: Adaptive percentile-based thresholds (rather than fixed thresholds) partition keys into three tiers: highly relevant ($D \leq t_1$) — full-precision computation retained; moderately relevant ($t_1 < D \leq t_2$) — coarse estimation via hash inner product followed by residual compensation via a lightweight MLP $\Delta(h_q, h_k; \phi)$; low relevant ($D > t_2$) — computation skipped but keys not discarded (unlike eviction methods). Special tokens (CLS/SEP/sink/neighboring tokens) are forced to full precision.
    - **Design Motivation**: Not all tokens are equally important. Applying hashing uniformly would sacrifice critical information, while applying full precision universally would negate the acceleration benefit. Three-tier stratification enables on-demand allocation of computational precision.

### Loss & Training

The primary loss is a listwise distillation objective $\mathcal{L}_{\text{distill}} = \text{KL}(P_{\text{student}} \| P_{\text{teacher}})$, supplemented by a bit-balance loss $\mathcal{L}_{\text{bal}}$ and a quantization loss $\mathcal{L}_{\text{quant}}$, each with an auxiliary coefficient of 0.1. Asymmetric temperature scaling is applied to address the overly smooth distribution of hash inner products.

## Key Experimental Results

### Main Results

| Method | Qwen2-7B LongBench Avg. | Complexity |
|--------|------------------------|------------|
| Full Attention | Baseline | $O(N^2)$ |
| H2O (eviction) | Below baseline | Linear |
| SnapKV (eviction) | Below baseline | Linear |
| KIVI (quantization) | Below baseline | Near-linear |
| DASH-KV | **Matches baseline** | **$O(N)$** |

### Ablation Study

| Configuration | Effect | Notes |
|---------------|--------|-------|
| Hash only (no calibration) | Performance degradation | Coarse-grained distances are inaccurate |
| + Cross-head consensus | Improvement | Multi-head voting reduces misclassification |
| + Cross-layer momentum | Further improvement | Temporal prior is beneficial |
| + Mixed precision | Best performance | Critical tokens maintain precision |

### Key Findings

- DASH-KV matches Full Attention performance on LongBench while achieving linear complexity.
- Eviction-based methods degrade due to irreversible information loss; DASH-KV discards no information.
- Asymmetric encoding outperforms symmetric encoding, validating that queries and keys genuinely require differentiated treatment.
- Hash code length $l$ achieves the optimal efficiency–performance trade-off in the range of 32–64 bits.

## Highlights & Insights

- **The "attention as retrieval" reformulation is highly inspiring**: Recasting attention as a retrieval problem rather than a computation problem introduces mature techniques from information retrieval (deep hashing), opening an entirely new optimization pathway.
- **Asymmetric design reflects a deep understanding of the distinct roles of Q and K**: Queries are one-time and require precision; keys are reused many times and require efficiency. This distinction has been overlooked in prior methods.
- **The principle of not discarding information**: Unlike eviction methods, low-relevant keys are only skipped during computation rather than permanently removed, preserving the possibility of being "reactivated" in subsequent steps.

## Limitations & Future Work

- Training a hash encoder is required (lightweight but not zero-cost), precluding plug-and-play deployment.
- The two learnable parameters introduced by cross-head consensus and cross-layer momentum require tuning.
- Evaluation is limited to LongBench; other long-context benchmarks (e.g., RULER, ∞-Bench) have not been tested.
- The residual compensation MLP design may require adaptation for different model architectures.

## Related Work & Insights

- **vs. H2O/SnapKV (eviction methods)**: Eviction permanently loses information, whereas DASH-KV retains all keys and only skips computation for low-relevant ones, achieving a better balance between information preservation and efficiency.
- **vs. KIVI/Atom (quantization methods)**: Quantization still operates within the floating-point framework; DASH-KV fundamentally changes the computational paradigm through bit operations.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Introducing deep hashing into the attention mechanism is pioneering; the asymmetric design and three-tier mixed-precision strategy are both well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three models, LongBench, and detailed ablations are provided, though benchmark coverage is limited.
- **Writing Quality**: ⭐⭐⭐⭐ The method is described systematically and thoroughly, though the heavy use of formulas makes some sections verbose.
**Code**: To be confirmed
**Area**: model_compression
**Keywords**: To be supplemented

## TL;DR
To be supplemented after thorough reading.

## Background & Motivation
To be supplemented after thorough reading.

## Method
To be supplemented after thorough reading.

## Key Experimental Results
To be supplemented after thorough reading.

## Highlights & Insights
To be supplemented after thorough reading.

## Limitations & Future Work
To be supplemented after thorough reading.

## Related Work & Insights
To be supplemented after thorough reading.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Latent-Condensed Transformer for Efficient Long Context Modeling](latent-condensed_transformer_for_efficient_long_context_modeling.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](representation-guided_parameter-efficient_llm_unlearning.md)
- [\[ACL 2026\] HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)
- [\[ACL 2026\] FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration](fastkv_decoupling_of_context_reduction_and_kv_cache_compression_for_prefill-deco.md)
- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](../../NeurIPS2025/model_compression/chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)

<!-- RELATED:END -->
