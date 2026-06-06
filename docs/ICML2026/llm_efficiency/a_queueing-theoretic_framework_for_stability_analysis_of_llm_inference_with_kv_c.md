---
title: >-
  [Paper Note] A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints
description: >-
  [ICML 2026][LLM Efficiency][Queueing Theory] This work establishes the first queueing model for LLM inference that explicitly incorporates the dynamic behavior of KV cache memory…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "Queueing Theory"
  - "KV Cache"
  - "GPU Memory Constraints"
  - "Stability Condition"
  - "Throughput Prediction"
date: 2026-05-08
content_hash: 880728620e9254fd
---

# A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints

**Conference**: ICML 2026  
**arXiv**: [2605.04595](https://arxiv.org/abs/2605.04595)  
**Code**: Provided in paper appendix  
**Area**: LLM Inference Efficiency / Systems  
**Keywords**: Queueing Theory, KV Cache, GPU Memory Constraints, Stability Condition, Throughput Prediction

## TL;DR
This work establishes the first queueing model for LLM inference that explicitly incorporates the dynamic behavior of KV cache memory, deriving a closed-form stability condition $\lambda < \mu(1-\delta)$, enabling operators to directly compute the required number of GPUs. Validation on single GPU, 8-GPU clusters, and LongBench real data shows prediction error within $10\%$.

## Background & Motivation

**Background**: LLM inference services are constrained by both computation and GPU memory. KV cache accelerates decoding but consumes significant memory for long contexts. System design must balance throughput, latency, and hardware cost.

**Limitations of Prior Work**: Classical queueing theory only models computational constraints; existing LLM systems literature (Wu/Yang/Li et al.) focuses on scheduling algorithms but lacks closed-form stability criteria. The memory dynamics of KV cache are nonlinear—prompt phase increases by chunk, decode phase increases by token—making standard bin packing frameworks inapplicable.

**Key Challenge**: Memory is not a static constraint but evolves over time; requests at different stages share and couple memory usage, invalidating simple average-rate approximations.

**Goal**: Provide a rigorous, computable stability condition so designers can estimate required GPU count directly from $\lambda$ and system parameters.

**Key Insight**: Construct a discrete-time Markov chain, defining the state as "the set of ongoing requests and their progress"; define a Lyapunov potential function as "memory × time over remaining lifetime", and lower-bound the drift.

**Core Idea**: The "memory × time" cost for each request can be written as an explicit function $g(s,o)$, so the service rate $\mu = M / (\bar b \mathbb E[g(s,o)])$, and the stability condition is $\lambda < \mu(1-\delta)$, where $\delta = \text{ess sup}(s+o)/M$ is a relaxation term.

## Method

### Overall Architecture

Two layers:

- **System Layer**: Requests are scheduled via FCFS/SJF, with hybrid continuous batching—prompt chunks and decode tokens from multiple requests are batched together. The memory constraint is $\sum_{i\in S(t)} c_i^{(t)} \hat s + d_i^{(t)} \leq M$, where $c_i^{(t)}$ is the number of prompt chunks processed for request $i$, and $d_i^{(t)}$ is the number of generated tokens.
- **Stability Layer**: The Markov chain state includes the remaining progress of each request. The potential function $V(t) = \sum_i g(s_i, o_i)$ represents the total "memory × time" of all unfinished requests. The first-order drift is $\mathbb E[V(t+1)-V(t)] = -M(1-\delta) + \lambda \bar b \mathbb E[g(s,o)]$.

### Key Designs

1. **Piecewise Memory Function & Lifetime Perspective**:

    - **Function**: Integrates the complex memory dynamics of prompt and decode phases into a single analytic quantity.
    - **Mechanism**: Defines $g(s,o) = \frac{(1+s/\hat s)s}{2} + s\cdot o + \frac{(1+o)o}{2}$, corresponding to prompt accumulation, prompt-decode overlap, and decode accumulation. Regardless of the request phase, $V(t)$ decreases by $M(1-\delta)$ per step under full load, converting dynamic nonlinear constraints into linear "area erasure".
    - **Design Motivation**: Summarizes lifetime memory usage with a single scalar function, simplifying drift analysis.

2. **Chunked Pre-Fill & Continuous Batching**:

    - **Function**: Controls per-batch computational complexity, allowing flexible mixing of multiple requests.
    - **Mechanism**: Prompts are chunked with fixed size $\hat s$, so each batch's attention computation is $O(M)$ (proportional to total KV cache, not prompt length squared). Continuous batching dynamically admits new requests as memory allows; overflow falls back to CPU swapping (assumed rare here).
    - **Design Motivation**: Eliminates quadratic cost of long prompts, making per-step processing time nearly constant, thus enabling queueing analysis.

3. **Work-Conserving Scheduling + Relaxation Term**:

    - **Function**: Ensures GPU is never idle and reserves space for bursty requests.
    - **Mechanism**: Any work-conserving policy is stable under relaxation $\delta = \text{ess sup}(s+o)/M$; the relaxation corresponds to "being able to accommodate the largest possible request in the worst case". This matches practical settings like vLLM's `gpu_memory_utilization=0.9`.
    - **Design Motivation**: Explicitly incorporates conservative memory utilization in real systems into the theory.

## Key Experimental Results

### Single GPU Validation (Synthetic P:D Ratios)

| Prompt:Decode | $\mu_{\text{gpu}}$ (req/s) | $\mu_{\text{theory}}$ | Gap |
|--------------|-------|-------|------|
| 1:1 | 3.387 | 3.263 | 3.66% |
| 2:1 | 3.650 | 3.956 | 8.38% |
| 1:2 | 2.969 | 2.902 | 2.25% |
| Mixed (2:1→1:2 time-varying) | 3.137 | 3.385 | 7.90% |

### Real Dataset (LongBench v2, Single GPU)

| Metric | Value | Note |
|------|----|------|
| $\mu_{\text{gpu}}$ | 0.610 req/s | Measured |
| $\mu_{\text{theory}}$ | 0.561 req/s | Predicted |
| Gap | 8.03% | Real long-context scenario |

### 8-GPU Cluster

| Config | $\mu_{\text{gpu}}$ | $8\mu_{\text{theory}}$ | Gap |
|------|------|------|------|
| 1:1 P/D | 26.71 | 25.81 | 3.38% |

### Stability Experiment (Single GPU, 1:1)

| $\lambda$ | Relation | Queue Behavior | Theoretical Prediction |
|---------|------|--------|--------|
| 1, 3 | $\lambda < \mu$ | Bounded (<5) | ✓ Stable |
| 5, 20, 50 | $\lambda > \mu$ | Linear growth | ✓ Unstable |

### Key Findings
- **Theory-experiment gap always $\leq 10\%$**: Covers synthetic/real, single/8-GPU scenarios, with surprisingly high prediction accuracy.
- **Linearly scalable**: 8-GPU cluster gap is 3.38%, same order as single GPU; formula $\mu_{\text{multi}} = 8\mu_{\text{single}}$ holds.
- **P:D ratio has significant impact**: 1:2 (long generation) yields lower throughput than 2:1 (long prompt), consistent with theoretical sign.
- **Sharp stable/unstable transition**: When $\lambda$ crosses the $\mu$ threshold, queue length diverges linearly, validating the drift argument.

## Highlights & Insights
- **First KV memory queueing model**: Fills the gap between queueing theory and LLM inference, with clear theoretical contribution.
- **Practical closed-form stability condition**: Operators can estimate GPU count directly via $\lceil \lambda/\mu \rceil$, no simulation needed.
- **Elegant potential function construction**: Unifies two-phase dynamics via "memory × time area", enabling minimal drift analysis.
- **Rigorous cross-scenario validation**: Synthetic, real, single, and multi-GPU all validated, with gap $\leq 10\%$ for high confidence.

## Limitations & Future Work
- Assumes constant batch processing time, ignoring CPU-GPU I/O and attention irregularities.
- Does not support TP/PP parallelism; $M, \bar b$ need to be replaced with TP-effective values.
- Does not consider dynamic batch size, proactive prefetching, or speculative decoding.
- For real-world long-tail/bursty traffic, average arrival rate assumption may be inaccurate.
- Future directions: online control for dynamic chunk size; fine-grained modeling of CPU cache swapping; multi-model colocation scenarios.

## Related Work & Insights
- **vs Classical Queueing Theory**: M/M/1, M/G/1, etc., do not model shared resources; this work introduces memory as a coupling variable.
- **vs LLM Scheduling Literature (Wu/Yang/Li)**: They design efficiency-maximizing schedulers; this work provides an upper bound for stability, forming a basis for scheduler analysis.
- **Insights**: The "lifetime resource usage" model can be extended to other constrained systems such as datacenter power/thermal management.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to rigorously incorporate KV memory into a queueing framework, with clear theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Synthetic + real, single GPU + cluster, stability all validated.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous mathematical derivation, clear system model.
- Value: ⭐⭐⭐⭐⭐ Directly guides resource planning for LLM services.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Understand and Accelerate Memory Processing Pipeline for Large Language Model Inference](understand_and_accelerate_memory_processing_pipeline_for_disaggregated_llm_infer.md)
- [\[ICML 2026\] PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](pipesd_an_efficient_cloud-edge_collaborative_pipeline_inference_framework_with_s.md)
- [\[AAAI 2026\] Judge Q: Trainable Queries for Optimized Information Retention in KV Cache Eviction](../../AAAI2026/llm_efficiency/judge_q_trainable_queries_for_optimized_information_retention_in_kv_cache_evicti.md)
- [\[ICML 2026\] Training-Inference Consistent Segmented Execution for Long-Context LLMs](training-inference_consistent_segmented_execution_for_long-context_llms.md)
- [\[ICLR 2026\] When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework](../../ICLR2026/llm_efficiency/when_does_divide_and_conquer_work_for_long_context_llm_a_noise_decomposition_fra.md)

</div>

<!-- RELATED:END -->
