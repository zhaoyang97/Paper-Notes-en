---
title: >-
  [Paper Note] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving
description: >-
  [ICML 2026][LLM Efficiency][PD disaggregation] This paper identifies that in multi-turn dialogue scenarios, the traditional Prefill-Decode (PD) disaggregation architecture is highly inefficient due to repeated P→D recomp…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "PD disaggregation"
  - "multi-turn dialogue"
  - "KV cache reuse"
  - "dynamic routing"
  - "SLO"
date: 2026-05-08
content_hash: 65a50f32cbe0c286
---

# Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving

**Conference**: ICML 2026  
**arXiv**: [2603.13358](https://arxiv.org/abs/2603.13358)  
**Code**: None (based on vLLM disaggregated serving prototype)  
**Area**: LLM inference service / dialogue systems / system optimization  
**Keywords**: PD disaggregation, multi-turn dialogue, KV cache reuse, dynamic routing, SLO

## TL;DR
This paper identifies that in multi-turn dialogue scenarios, the traditional Prefill-Decode (PD) disaggregation architecture is highly inefficient due to repeated P→D recomputation and KV transmission at every turn. It proposes the PPD (Prefill-capable Decode) dynamic routing system, allowing decode nodes to decide—based on SLO weights—whether to locally process Turn 2+ append-prefill. This reduces Turn 2+ TTFT by approximately 68%.

## Background & Motivation

**Background**: Modern LLM inference engines (vLLM, SGLang, TensorRT-LLM, DeepSeek, Gemini, etc.) commonly adopt a Prefill-Decode (PD) disaggregation architecture—placing compute-intensive prefill and bandwidth-constrained decode on separate GPU pools to avoid workload interference and enable independent scaling. The KV cache is strictly passed one-way from P nodes to D nodes.

**Limitations of Prior Work**: PD is designed for single-turn queries, but real-world chatbot and agent systems are almost always multi-turn. In multi-turn settings, each new turn must resend the entire history (previous prompt + reply + new prompt) to the P node for KV recomputation, then transmit back to the D node. Empirical results show this recomputation accounts for 99% of multi-turn prefill cost; KV transmission saturates network bandwidth, causing persistently high Turn 2+ TTFT and, under heavy load, service degradation.

**Key Challenge**: The PD KV channel is unidirectional (P produces, D consumes, no reverse link). Even if the previous reply's KV is already on D, P cannot access it. Addressing this trade-off requires either breaking the unidirectional contract (high engineering cost) or adding distributed KV storage (Mooncake, MemServe, etc.)—but neither changes the routing decision itself.

**Goal**: Without modifying the KV protocol of mainstream engines like vLLM, design a dynamic routing strategy that simultaneously optimizes Turn 2+ TTFT, TPOT, and system throughput, while maintaining robustness across different P:D ratios.

**Key Insight**: Microbenchmarks on H100 reveal that **not all prefill operations interfere equally**—full prefill (no cache) at batch=200 slows decode TPOT by 48%, while append-prefill (only new tokens, reusing cached KV) slows it by just ~2%, a one-order-of-magnitude difference. This indicates that the cost for D nodes to locally process append-prefill is much lower than previously assumed.

**Core Idea**: Formalize the decision of "whether to route Turn 2+ append-prefill to D node for local processing" as a weighted binary decision $x \in \{0,1\}$, scored offline and looked up online according to SLO weights $\mathbf{w}=(w_{ttft},w_{tpot})$. Traditional PD is the special case $x \equiv 0$.

## Method

### Overall Architecture
PPD modifies the scheduling layer atop vLLM's disaggregated serving, in two phases: (1) **Offline Table Construction**—on a coarse workload grid (accumulated context length × input/output ratio × system QPS), directly measure Turn 2 TTFT and TPOT for $x{=}0$ and $x{=}1$, compute the score $S(\psi;\pi,\mathbf{w}) = w_{ttft}\Delta_{ttft} - w_{tpot}\Delta_{tpot}$, and store $x^*(\hat\psi)=\mathbb{1}[S>0]$; (2) **Online Lookup**—for each incoming request, Turn 1 is forced to $x{=}0$ (no cache), Turn 2+ finds the nearest grid cell by three features and returns the precomputed decision, with lookup time <1ms. Except for routing, KV transmission protocol and prefix cache fully reuse vLLM's original implementation.

### Key Designs

1. **Asymmetric Interference of Append-prefill vs Full-prefill**:

    - Function: The theoretical foundation for PPD decisions, quantifying the differing impact of the two prefill types on decode within the same GPU.
    - Mechanism: Full prefill processes $n$ new tokens with attention complexity $O(n^2)$; append-prefill computes attention only for $m$ new tokens (each attends to $n+m$ keys), complexity $O(m(n+m))$. When $m \ll n$, this is $n/m$ times cheaper than full prefill. Microbenchmarks on H100 with Llama-3.1-8B show that at batch=200, full prefill slows TPOT by ~48%, append-prefill by only ~2%; at 4-way concurrency, full +57% vs append +21%, with the gap widening to 3-4× at 32K/64K context lengths.
    - Design Motivation: The original PD separation assumes "all prefill heavily interferes with decode." The authors' precise measurements overturn this, opening the design space for D nodes to locally process certain prefill types.

2. **Optimization Problem Formalized via Scoring Function $S$**:

    - Function: Unifies the "should the request go to P or be processed locally" decision into a tunable objective, making PD $x \equiv 0$ and "all AP-to-D" $x \equiv 1$ special cases.
    - Mechanism: For each Turn 2+ request, define the local processing benefit relative to PD as $S(\psi;\pi,\mathbf{w}) = w_{ttft}\Delta_{ttft} - w_{tpot}\Delta_{tpot}$, where $\Delta_{ttft}$ is TTFT improvement and $\Delta_{tpot}$ is TPOT degradation. If $S>0$, process locally; otherwise, route to P. System throughput is not directly optimized but improves as a byproduct of reduced KV transmission.
    - Design Motivation: In a system scan of 3060 configurations (17 configs × 18 workloads × 10 QPS), **92.2% of (workload, QPS) pairs** have different optimal settings for Turn 2 TTFT and TPOT—there is no static optimum, necessitating per-request dynamic decisions.

3. **Two-stage Routing: Offline Table + Online Lookup**:

    - Function: Moves expensive optimization computation offline; online, only millisecond-level table lookup is needed, imposing near-zero overhead on latency-sensitive paths.
    - Mechanism: Offline, for each grid cell, directly measure TTFT/TPOT for $x{=}0$ and $x{=}1$, store Boolean decision by the sign of $S$; online, quantize each request by three features (accumulated context length, input/output ratio, system QPS) to the nearest cell, and retrieve $x^*$ in <1ms. Turn 1 is always $x=0$ (no KV cache available) for consistency. The system can revert to any static strategy (including traditional PD) by adjusting weights $\mathbf{w}$.
    - Design Motivation: Decouples "configuration scale" (P:D ratio, determines Turn 1 capacity) from "Turn 2+ SLO tuning" (weights, determines Pareto point) into two independent knobs—whereas traditional PD couples both, forcing multi-objective trade-offs in operations.

### Loss & Training
PPD does not involve model training; all decisions are system-level scheduling strategies. All decisions are driven by offline measurements, with main parameters being user-specified SLO weights $w_{ttft}, w_{tpot}$ and workload grid discretization thresholds.

## Key Experimental Results

### Main Results
Hardware: 4× H100 80GB + NVLink; primary model Llama-3.1-8B (Qwen2.5-14B/Qwen3-30B for validation); synthetic 18 workloads × 10 QPS × 17 configs = 3060 data points; real datasets: ShareGPT and WildChat.

| Config | Metric | $x=0$ Baseline | $x=1$ / PPD | Gain |
|--------|--------|----------------|-------------|------|
| 1P_3D Long Context High QPS | Turn 2 TTFT | Baseline | $x=1$ improved | -73.3% |
| 2P_2D Long Context High QPS | Turn 2 TTFT | Baseline | $x=1$ improved | -56.2% |
| 3P_1D Long Context High QPS | Turn 2 TTFT | Baseline | $x=1$ improved | -24.9% |
| 1P_3D ShareGPT | Avg. Query Latency | Baseline | PPD | -15~25% |
| 2P_2D / 3P_1D ShareGPT Multi-QPS | Success Rate | <95% (degraded) | PPD 100% | Recovers unavailable configs |

### Ablation Study

| Config Type | TTFT Win Rate | TPOT Win Rate | Throughput Win Rate | Avg. Win Rate |
|-------------|--------------|--------------|---------------------|---------------|
| Replica (4R) | 63.3% | 0.6% | 0% | 21.3% |
| $x=0$ (Traditional PD) | 0% | 38.3% | 4.4% | 14.2% |
| $0<x<1$ Partial Routing | 3.3% | 33.3% | 27.8% | 21.5% |
| $x=1$ (Full AP-to-D) | 27.2% | 15.6% | 38.3% | 27.0% |

### Key Findings
- **Greater local processing benefit when P resources are tight**: 1P_3D achieves up to 73.3% Turn 2 TTFT improvement, while 3P_1D only 24.9%—when P is the bottleneck, $x=1$ bypasses it directly.
- **No static optimum**: For 92.2% of workload-QPS pairs, the TTFT-optimal config ≠ TPOT-optimal config, validating the need for dynamic routing.
- **PPD recovers unavailable configs**: In 2P_2D and 3P_1D, many QPS points have <95% success rate under $x=0$ (KV transmission saturated); enabling PPD stabilizes to 100%.
- **Improvements persist across turn count and model size**: For 2-16 turns and 8B/14B/30B models, Turn 2+ TTFT improvement remains ~70%, indicating the benefit is architectural, not model-specific.

## Highlights & Insights
- **Challenges the meta-assumption of PD**: PD design has long assumed "all prefill heavily interferes with decode." This work splits that into full vs append types and quantifies a one-order-of-magnitude difference, opening new design dimensions for the disaggregation family.
- **Elegant unification via single parameter $x$**: Traditional PD, Replica, partial routing, and full local all become special cases of $x \in \{0, \text{frac}, 1\}$, making comparisons and analysis clear.
- **Decoupling configuration scale and SLO tuning**: In traditional PD, the P:D ratio governs both "Turn 1 capacity planning" and "Turn 2+ latency tuning." PPD separates Turn 2+ tuning via weights $\mathbf{w}$, a transferable approach for other multi-objective scheduling problems.
- **Offline table + online lookup pattern**: Shifting expensive decisions offline and using <1ms lookup online is highly valuable for latency-sensitive engineering scenarios.

## Limitations & Future Work
- **Coverage issues with grid discretization**: The three-dimensional grid's precision and thresholds are empirically chosen; new workload patterns require rebuilding the table. The paper does not discuss adaptive update mechanisms.
- **Hybrid R+P/D configs excluded**: The authors acknowledge that seven hybrid configs generally underperform pure PD, but provide no theoretical explanation and do not explore R's potential value in edge cases.
- **Experiments mainly within 4×H100 NVLink**: Cross-node RDMA/Ethernet slow links are only bandwidth-simulated; real multi-node deployment portability needs further validation.
- **Prefix cache hit rate drift not considered**: When multiple sessions compete for D node local KV slots, local processing advantages may be offset by cache thrashing, which is not deeply explored.

## Related Work & Insights
- **vs AMPD (he2026)**: Concurrent work also routes AP to D, but bases decisions on real-time queue state; this paper uses an offline optimization framework for table construction, offering greater stability and theoretical clarity.
- **vs Mooncake / MemServe / LMCache**: These approaches add distributed KV storage without altering the PD unidirectional protocol; PPD is complementary—achieving most of the benefit via routing alone, without new storage layers.
- **vs DuetServe / Nexus / TaiChi**: These works perform SM partitioning or dynamic resource allocation within the GPU; PPD operates at the request level and can be combined with them.
- **vs Chunked-prefill (Splitwise / FastGen)**: Chunking mitigates prefill-decode interference; PPD shows that for append-prefill (small chunks), interference is already minimal, aligning with the underlying motivation of chunked approaches.

## Rating
- Novelty: ⭐⭐⭐⭐ Re-examines the core PD assumption and formalizes the finding into a schedulable optimization framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3060 config scan + synthetic + real data + multi-model/multi-turn validation—an unusually comprehensive system evaluation in this field.
- Writing Quality: ⭐⭐⭐⭐ Clear logical flow from microbenchmark → formalization → algorithm → empirical results, though some formula notation is dense.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play improvement for production LLM serving, delivering substantial TTFT gains without changing models or protocols.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration](oserve_accelerating_llm_serving_via_spatial-temporal_workload_orchestration.md)
- [\[ICML 2026\] Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving.md)
- [\[NeurIPS 2025\] Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](../../NeurIPS2025/llm_efficiency/efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)
- [\[ICML 2026\] A Queueing-Theoretic Framework for Stability Analysis of LLM Inference with KV Cache Memory Constraints](a_queueing-theoretic_framework_for_stability_analysis_of_llm_inference_with_kv_c.md)
- [\[ACL 2026\] Task-Aware LLM Routing with Multi-Level Task-Profile-Guided Data Synthesis for Cold-Start Scenarios](../../ACL2026/llm_efficiency/task-aware_llm_routing_with_multi-level_task-profile-guided_data_synthesis_for_c.md)

</div>

<!-- RELATED:END -->
