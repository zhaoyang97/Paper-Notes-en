---
title: >-
  [Paper Note] A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints
description: >-
  [ICML 2026][Model Compression][Queueing theory] This paper establishes the first queueing model for LLM inference that explicitly incorporates KV cache memory dynamics…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Queueing theory"
  - "KV cache"
  - "memory constraints"
  - "stability conditions"
  - "throughput prediction"
date: 2026-05-08
content_hash: fe57c31f29f3ebfc
---

# A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints

**Conference**: ICML 2026  
**arXiv**: [2605.04595](https://arxiv.org/abs/2605.04595)  
**Code**: Provided in the paper appendix  
**Area**: LLM Inference Efficiency / Systems  
**Keywords**: Queueing theory, KV cache, memory constraints, stability conditions, throughput prediction

## TL;DR
This paper establishes the first queueing model for LLM inference that explicitly incorporates KV cache memory dynamics, providing a closed-form stability condition $\lambda < \mu(1-\delta)$. This allows operators to directly calculate the required number of GPUs; validation on single GPU, 8-GPU clusters, and real LongBench data shows errors $\leq 10\%$.

## Background & Motivation

**Background**: LLM inference services are simultaneously constrained by computation and memory. KV caching accelerates decoding but consumes significant memory in long-context scenarios. System design necessitates balancing throughput, latency, and hardware costs.

**Limitations of Prior Work**: Classic queueing theory only models computational constraints; existing LLM system papers (Wu/Yang/Li et al.) focus on scheduling algorithms but lack closed-form stability criteria. KV cache memory follows non-linear dynamics—increasing by chunks in the prompt phase and token-by-token in the decode phase—making it difficult to apply standard bin-packing frameworks.

**Key Challenge**: Memory is not a static constraint but evolves over time. Different requests occupy different stages, and shared memory is highly coupled, causing simple average-rate approximations to fail.

**Goal**: Provide strict, computable stability conditions that allow designers to estimate the required number of GPUs based on $\lambda$ and system parameters.

**Key Insight**: Construct a discrete-time Markov chain where states are defined as the "set of ongoing requests + respective progress"; define a Lyapunov potential function as "remaining lifetime memory $\times$ time" and estimate the lower bound for drift.

**Core Idea**: The "memory $\times$ time" cost of each request can be written as an explicit function $g(s,o)$. Thus, the service rate $\mu = M / (\bar b \mathbb E[g(s,o)])$, and the stability condition is $\lambda < \mu(1-\delta)$, where $\delta = \text{ess sup}(s+o)/M$ is a slack term.

## Method

### Overall Architecture

Two layers:

- **System Layer**: Requests are scheduled via FCFS/SJF with hybrid continuous batching—prompt chunks and decode tokens from multiple requests are processed in the same batch. Memory constraint: $\sum_{i\in S(t)} c_i^{(t)} \hat s + d_i^{(t)} \leq M$, where $c_i^{(t)}$ is the number of processed chunks for request $i$, and $d_i^{(t)}$ is the number of generated tokens.
- **Stability Layer**: The Markov chain state includes the remaining progress of each request. The potential function $V(t) = \sum_i g(s_i, o_i)$ represents the total "memory $\times$ time" of all unfinished requests. The first-order drift is $\mathbb E[V(t+1)-V(t)] = -M(1-\delta) + \lambda \bar b \mathbb E[g(s,o)]$.

### Key Designs

1. **Segmented Memory Function and Lifetime Perspective**:
    - **Function**: Consolidates complex memory dynamics of prompt and decode phases into an analytical quantity.
    - **Mechanism**: Defines $g(s,o) = \frac{(1+s/\hat s)s}{2} + s\cdot o + \frac{(1+o)o}{2}$, corresponding to prompt accumulation, prompt-decode overlap, and decode accumulation. Regardless of the stage, $V(t)$ decreases by $M(1-\delta)$ per step under full load, converting dynamic non-linear constraints into linear "area erasure."
    - **Design Motivation**: Use a single scalar function to summarize lifetime occupancy, simplifying drift analysis.

2. **Chunked Prefill and Continuous Batching**:
    - **Function**: Controls the computational complexity per batch, allowing flexible mixing of multiple requests.
    - **Mechanism**: Prompts are divided into fixed chunk sizes $\hat s$, ensuring attention calculation per batch is $O(M)$ (proportional to total KV cache, rather than the square of prompt length). Continuous batching dynamically admits new requests when memory permits; overflows trigger CPU swapping (assumed rare in this paper).
    - **Design Motivation**: Eliminates quadratic overhead for long prompts, making per-step processing time approximately constant, thus enabling queueing analysis.

3. **Work-Conserving Scheduling + Slack Term**:
    - **Function**: Ensures GPU readiness while reserving space for bursty requests.
    - **Mechanism**: Any work-conserving strategy is stable under the slack $\delta = \text{ess sup}(s+o)/M$. The slack corresponds to "accommodating one maximum request even in the worst case." This aligns with practical settings like vLLM's `gpu_memory_utilization=0.9`.
    - **Design Motivation**: Explicitly incorporates the conservative memory utilization of real systems into the theory.

## Key Experimental Results

### Single GPU Validation (Synthetic P:D Ratios)

| Prompt:Decode | $\mu_{\text{gpu}}$ (req/s) | $\mu_{\text{theory}}$ | Gap |
|--------------|-------|-------|------|
| 1:1 | 3.387 | 3.263 | 3.66% |
| 2:1 | 3.650 | 3.956 | 8.38% |
| 1:2 | 2.969 | 2.902 | 2.25% |
| Mixed (2:1→1:2 time-varying) | 3.137 | 3.385 | 7.90% |

### Real Dataset (LongBench v2, Single GPU)

| Metric | Value | Description |
|------|----|------|
| $\mu_{\text{gpu}}$ | 0.610 req/s | Measured |
| $\mu_{\text{theory}}$ | 0.561 req/s | Predicted |
| Gap | 8.03% | Real-world long context scenario |

### 8-GPU Cluster

| Configuration | $\mu_{\text{gpu}}$ | $8\mu_{\text{theory}}$ | Gap |
|------|------|------|------|
| 1:1 P/D | 26.71 | 25.81 | 3.38% |

### Stability Experiment (Single GPU, 1:1)

| $\lambda$ | Relationship | Queue Behavior | Theoretical Prediction |
|---------|------|--------|--------|
| 1, 3 | $\lambda < \mu$ | Bounded (<5) | ✓ Stable |
| 5, 20, 50 | $\lambda > \mu$ | Linear growth | ✓ Unstable |

### Key Findings
- **Theory-Measurement Gap always $\leq 10\%$**: Covers synthetic/real and single/8-GPU scenarios; prediction accuracy is remarkably high.
- **Linear Scalability**: 8-GPU cluster Gap is 3.38%, matching single GPU magnitude; the formula $\mu_{\text{multi}} = 8\mu_{\text{single}}$ holds.
- **P:D Ratio Impact**: 1:2 (long generation) has lower throughput than 2:1 (long prompt), consistent with theoretical signs.
- **Clear Stability Transitions**: Queue length diverges linearly once $\lambda$ crosses the $\mu$ threshold, validating the drift argument.

## Highlights & Insights
- **First KV Memory Queueing Model**: Fills the gap between queueing theory and LLM inference with clear theoretical contributions.
- **Practical Closed-form Condition**: Operators can directly use $\lceil \lambda/\mu \rceil$ to estimate GPU counts without simulation.
- **Elegant Potential Function Construction**: Uses "memory $\times$ time area" to unify two-phase dynamics, making drift analysis minimalist.
- **Rigorous Cross-scenario Validation**: Validated across synthetic, real, single-card, and multi-card setups; $\leq 10\%$ gap provides high confidence.

## Limitations & Future Work
- Assumes constant batch time, ignoring additional fluctuations from CPU-GPU I/O or attention irregularities.
- Does not support parallel strategies like TP/PP; requires replacing $M, \bar b$ with TP-effective values.
- High-level scheduling like dynamic batch sizes, proactive preloading, or speculative decoding is not considered.
- Average arrival rate assumptions may be inaccurate under long-tail or bursty real-world traffic.
- Future directions: Online control for dynamic chunk sizes; fine-grained modeling of CPU cache swapping; multi-model co-location scenarios.

## Related Work & Insights
- **vs. Classic Queueing Theory**: M/M/1, M/G/1, etc., do not model shared resources; this work introduces memory as a coupling variable.
- **vs. LLM Scheduling Literature (Wu/Yang/Li)**: Those works focus on maximizing efficiency; this work provides stability upper bounds, forming a basis for scheduling analysis.
- **Insight**: The "lifetime resource occupancy" model can be generalized to other constrained systems like data center power or cooling.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to strictly incorporate KV memory into a queueing framework with clear theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes synthetic + real, single GPU + cluster, and stability validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivation and clear system modeling.
- Value: ⭐⭐⭐⭐⭐ Directly guides resource planning for LLM services.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DASH-KV: Accelerating Long-Context LLM Inference via Asymmetric KV Cache Hashing](../../ACL2026/model_compression/dash-kv_accelerating_long-context_llm_inference_via_asymmetric_kv_cache_hashing.md)
- [\[NeurIPS 2025\] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference](../../NeurIPS2025/model_compression/ada-kv_optimizing_kv_cache_eviction_by_adaptive_budget_allocation_for_efficient_.md)
- [\[ICML 2026\] Stochastic Sparse Attention for Memory-Bound Inference](stochastic_sparse_attention_for_memory-bound_inference.md)
- [\[ACL 2026\] HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference](../../ACL2026/model_compression/heterocache_a_dynamic_retrieval_approach_to_heterogeneous_kv_cache_compression_f.md)
- [\[NeurIPS 2025\] MUSTAFAR: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference](../../NeurIPS2025/model_compression/mustafar_promoting_unstructured_sparsity_for_kv_cache_pruning_in_llm_inference.md)

</div>

<!-- RELATED:END -->
