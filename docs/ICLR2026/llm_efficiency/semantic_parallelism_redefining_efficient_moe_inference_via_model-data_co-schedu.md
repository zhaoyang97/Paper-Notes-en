---
title: >-
  [Paper Note] Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling
description: >-
  [ICLR 2026][LLM Efficiency][Mixture-of-Experts] This paper proposes the Semantic Parallelism (SP) paradigm, which predicts token-expert routing paths and co-schedules model placement with data dispatch to substantially reduce all-to-all communication overhead in MoE inference under expert parallelism. It achieves up to 2.78× throughput improvement in Attention-DP settings and up to 24.9% latency reduction in Attention-TP settings.
tags:
  - ICLR 2026
  - LLM Efficiency
  - Mixture-of-Experts
  - Expert Parallelism
  - all-to-all communication
  - model-data co-scheduling
  - token-expert affinity
date: 2026-05-08
content_hash: b6f4b5edb9c06236
---

# Semantic Parallelism: Redefining Efficient MoE Inference via Model-Data Co-Scheduling

**Conference**: ICLR 2026
**arXiv**: [2503.04398](https://arxiv.org/abs/2503.04398)
**Code**: Implemented on SGLang (~5000 lines of Python + Triton kernels)
**Area**: LLM Efficiency
**Keywords**: Mixture-of-Experts, Expert Parallelism, all-to-all communication, model-data co-scheduling, token-expert affinity

## TL;DR
This paper proposes the Semantic Parallelism (SP) paradigm, which predicts token-expert routing paths and co-schedules model placement with data dispatch to substantially reduce all-to-all communication overhead in MoE inference under expert parallelism. It achieves up to 2.78× throughput improvement in Attention-DP settings and up to 24.9% latency reduction in Attention-TP settings.

## Background & Motivation
**MoE inference is bottlenecked by all-to-all communication**: Expert Parallelism (EP) distributes experts across multiple GPUs but requires two all-to-all collective communications to route tokens to remote experts and back. Even on 400 GB/s high-speed interconnects, this accounts for 59.2% of MoE layer forward latency.

**Existing approaches decouple model placement from data scheduling**: Expert placement and token dispatch are treated as independent problems, resulting in substantial unnecessary cross-device communication.

**Tokens exhibit context-independent expert affinity**: Profiling reveals that tokens consistently activate a concentrated and stable set of top-k experts across different contexts (median cumulative activation probability of top-k experts: 0.833–0.976), providing a basis for predictive routing.

**Widespread deployment of MoE models such as DeepSeek-V3/R1 and Qwen3** makes EP communication optimization a critical industrial need.

## Method

### Core Insight: Context-Independent Token-Expert Affinity
- Profiling DeepSeek-V2-Lite on ShareGPT reveals that each token consistently routes to the same top-k expert subset across different contexts.
- The median F1-score across all layers reaches 0.833–1.000, and the maximum activation frequency of non-top-k experts is only ~0.05.
- Based on this, a token-expert activation frequency table $T^{(L)} \in \mathbb{N}^{t \times N}$ is constructed to estimate routing probabilities.

### Offline Model Scheduling: Expert Clustering and Placement
- Model-data co-scheduling is formulated as a 0-1 integer linear programming (ILP) problem.
- Objective: $L = \theta \cdot \text{load balance term} + (1-\theta) \cdot \text{remote activation minimization term}$
- Constraints: each token/expert belongs to exactly one cluster; each cluster contains an equal number of experts.
- An alternating optimization algorithm is used for efficient solving, placing co-activated experts on the same GPU.

### Online Data Scheduling (Two Settings)

**Attention-DP Setting — Inter-request Scheduling**:
- Predicts the target DP rank for an entire request based on token-expert affinity.
- $S_r = \arg\max_j \sum_{i \in r} R_{ij}$, assigning each request to the device hosting the experts most activated by its tokens.
- Combined with workload-aware balanced scheduling to ensure load balance across ranks.

**Attention-TP Setting — Intra-request Token Scheduling**:
- Leverages Markov dependencies in inter-layer expert selection (2-gram device transition model) to enhance prediction.
- Designs Shuffled-Reduce-Scatter (SRS) and Shuffled-AllGather (SAG) fused communication primitives.
- Embeds speculative token reordering seamlessly into existing TP communication phases with only ~1% overhead.

### System Implementation
- Built on SGLang with ~5000 lines of Python and custom Triton kernels.
- Optimized argsort kernel is 25% faster than PyTorch's native implementation.
- Integrates the DeepEP communication library for efficient all-to-all operations.

## Key Experimental Results

### Attention-DP Setting (Throughput under SLO Constraints)

| Model | vs SGLang (TTFT SLO) | vs SGLang (E2E SLO) | vs MoETuner (TTFT) | vs MoETuner (E2E) |
|------|---------------------|--------------------|--------------------|-------------------|
| DeepSeek-V2-Lite | +31% | +221% | +32% | +278% |
| Qwen3-30B-A3B | +98% | +11% | +35% | +32% |

### Attention-TP Setting (Latency Reduction)

| Model | Input Length 256 | Input Length 512 | Input Length 1024 |
|------|-----------|-----------|------------|
| DeepSeek-V2-Lite p99 TTFT | -12.21% | -10.60% | -18.89% |
| Qwen3-30B-A3B p99 TTFT | -17.16% | -24.90% | -3.80% |

### Key Findings
- Local Activation Rate (LAR) improves by 37–43% over vanilla EP, corresponding to a 41.8–46.6% reduction in EP layer latency.
- The co-scheduling algorithm achieves 15.4% higher LAR than the best baseline (MoETuner) with lower load imbalance.
- Zero-shot cross-dataset transfer results validate the generalizability of the scheduling strategy.

## Highlights & Insights
- Introduces the "Semantic Parallelism" paradigm, shifting communication optimization from reactive mitigation to proactive prevention.
- Reveals the context-independent nature of token-expert affinity, providing a theoretical foundation for predictive scheduling.
- Jointly optimizes both model placement and data scheduling, offering a systematic rather than a localized solution.
- The SRS/SAG fused primitive design is elegant, embedding token reordering into existing communication flows with only ~1% additional overhead.

## Limitations & Future Work
- Validation is limited to single-node 8-GPU setups; effectiveness in cross-node or low-bandwidth interconnect settings remains to be verified.
- The prediction model requires offline profiling data, making it ineffective during cold starts.
- MoE variants with highly dynamic routing mechanisms require re-profiling after changes to the gating function.
- Joint evaluation with KV cache optimization or quantization techniques has not been conducted.

## Related Work & Insights
- **Expert placement**: MoETuner (ILP-based placement), ExFlow (inter-layer expert affinity), EPLB (DeepSeek's load balancing).
- **MoE inference systems**: DeepSpeed-MoE, Tutel, vLLM, SGLang.
- **Prefetching/offloading**: Pre-gated MoE (architecture modification for next-layer expert prediction) — Sem-MoE requires no architectural changes.
- This work is the first to jointly optimize both model scheduling and data scheduling.

## Rating ⭐⭐⭐⭐⭐
- **Novelty**: 5/5 — The Semantic Parallelism paradigm and co-scheduling concept are highly original.
- **Experimental Thoroughness**: 4/5 — Covers two models and two settings, but limited to single-node evaluation.
- **Writing Quality**: 4/5 — System description is clear with high-quality figures.
- **Value**: 5/5 — Addresses a core bottleneck in MoE inference with significant industrial relevance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Expert Divergence Learning for MoE-based Language Models](expert_divergence_learning_for_moe-based_language_models.md)
- [\[AAAI 2026\] How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](../../AAAI2026/llm_efficiency/how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)
- [\[ICLR 2026\] LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](lycheedecode_accelerating_long-context_llm_inference_via_hybrid_speculative_deco.md)
- [\[ICLR 2026\] Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](fast_catch-up_late_switching_optimal_batch_size_scheduling_via_functional_scalin.md)
- [\[NeurIPS 2025\] Mozart: Modularized and Efficient MoE Training on 3.5D Wafer-Scale Chiplet Architectures](../../NeurIPS2025/llm_efficiency/mozart_modularized_and_efficient_moe_training_on_35d_wafer-scale_chiplet_archite.md)

</div>

<!-- RELATED:END -->
