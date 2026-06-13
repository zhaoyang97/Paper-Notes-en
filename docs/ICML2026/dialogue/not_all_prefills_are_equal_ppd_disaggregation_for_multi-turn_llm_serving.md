---
title: >-
  [Paper Note] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving
description: >-
  [ICML 2026][Dialogue Systems][PD Disaggregation] This paper points out that traditional Prefill-Decode (PD) disaggregated architectures are significantly inefficient in multi-turn dialogue scenarios because they require…
tags:
  - "ICML 2026"
  - "Dialogue Systems"
  - "PD Disaggregation"
  - "Multi-turn Dialogue"
  - "KV cache reuse"
  - "Dynamic Routing"
  - "SLO"
date: 2026-05-08
content_hash: 31ec16819d61c287
---

# Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving

**Conference**: ICML 2026  
**arXiv**: [2603.13358](https://arxiv.org/abs/2603.13358)  
**Code**: None (Based on vLLM disaggregated serving prototype)  
**Area**: LLM Inference Serving / Dialogue Systems / System Optimization  
**Keywords**: PD Disaggregation, Multi-turn Dialogue, KV cache reuse, Dynamic Routing, SLO

## TL;DR
This paper points out that traditional Prefill-Decode (PD) disaggregated architectures are significantly inefficient in multi-turn dialogue scenarios because they require KV recomputation and transmission for every turn. It proposes PPD (Prefill-capable Decode), a dynamic routing system that allows decode nodes to decide whether to process Turn 2+ append-prefills locally based on SLO weights, reducing Turn 2+ TTFT by approximately 68%.

## Background & Motivation

**Background**: Modern LLM inference engines (vLLM, SGLang, TensorRT-LLM, DeepSeek, Gemini, etc.) commonly adopt Prefill-Decode (PD) disaggregated architectures—placing compute-intensive prefill and bandwidth-constrained decode tasks into different GPU pools to avoid interference and support independent scaling. KV cache is strictly transmitted unidirectionally from P nodes to D nodes.

**Limitations of Prior Work**: PD is designed based on single-turn independent queries. However, in real-world deployments, chatbots and agent systems are almost entirely multi-turn. In multi-turn scenarios, every new turn must send the entire history (previous prompts + responses + new prompt) back to the P node to recompute KV, which is then sent back to the D node. Empirical measurements show that this recomputation accounts for 99% of multi-turn prefill costs; meanwhile, KV transmission saturates network bandwidth, leading to high Turn 2+ TTFT and triggering service degradation under high loads.

**Key Challenge**: The KV channel in PD is unidirectional (P produces, D consumes, no reverse link). Even if the KV for the previous response is already on D, P cannot access it. To resolve this trade-off, one must either break the unidirectional contract (high engineering cost) or use external distributed KV storage (Mooncake, MemServe, etc.)—but neither changes the routing decision itself.

**Goal**: Design a dynamic routing strategy that optimizes Turn 2+ TTFT, TPOT, and system throughput simultaneously without modifying the KV protocols of mainstream engines like vLLM, while remaining robust to different P:D ratios.

**Key Insight**: The authors conducted micro-benchmarks on H100 and discovered that **not all prefills cause the same degree of interference**. Specifically, a full prefill (no cache) at batch=200 slows down decode TPOT by 48%, while an append-prefill (only new tokens, reusing cached KV) only slows it by about 2%, representing an order of magnitude difference. This implies that the cost of processing append-prefills locally on D nodes is much lower than intuition suggests.

**Core Idea**: Formalize the decision of "whether to route Turn 2+ append-prefills to the local D node" as a weighted binary decision $x \in \{0,1\}$. Scores are calculated offline based on SLO weights $\mathbf{w}=(w_{ttft},w_{tpot})$ and used via online table look-up. Traditional PD is a special case where $x \equiv 0$.

## Method

### Overall Architecture
PPD modifies the scheduling layer on top of vLLM's disaggregated serving in two phases: (1) **Offline Table Construction**—On a coarse-grained workload grid (cumulative context length × input/output ratio × system QPS), Turn 2 TTFT and TPOT are measured for $x{=}0$ and $x{=}1$. Scores are calculated using the formula $S(\psi;\pi,\mathbf{w}) = w_{ttft}\Delta_{ttft} - w_{tpot}\Delta_{tpot}$, and $x^*(\hat\psi)=\mathbb{1}[S>0]$ is stored. (2) **Online Look-up**—For every query entering the system, Turn 1 is forced to $x{=}0$ (no cache). For Turn 2+, the nearest grid cell is found based on three-dimensional features to return the pre-stored decision. The lookup takes <1ms. Aside from routing, the KV transmission protocol and prefix caching reuse the original vLLM implementation.

### Key Designs

1.  **Asymmetric Interference of Append-prefill vs Full-prefill**:
    - **Function**: Serves as the theoretical foundation for PPD decisions by quantifying the different impacts of two prefill types on co-located GPU decodes.
    - **Mechanism**: Full prefill processes $n$ new tokens with $O(n^2)$ attention complexity. Append-prefill only computes attention for $m$ new tokens (each token attends to $n+m$ keys), with complexity $O(m(n+m))$. When $m \ll n$, it is $n/m$ times cheaper than full prefill. Llama-3.1-8B benchmarks on H100 showed that at batch=200, full prefill slows TPOT by ~48%, whereas append-prefill only causes ~2% slowdown.
    - **Design Motivation**: The original assumption of PD disaggregation is that "all prefills heavily interfere with decode." The authors disprove this premise through fine-grained measurement, opening the design space for "local prefill processing on D nodes."

2.  **Optimization Problem Formalized by Scoring Function $S$**:
    - **Function**: Unifies the decision of whether to send a request to P or process locally into a weight-adjustable objective function, making PD a special case of $x \equiv 0$ and "all AP-to-D" a special case of $x \equiv 1$.
    - **Mechanism**: For each Turn 2+ request, define the benefit score of local processing relative to the PD path as $S(\psi;\pi,\mathbf{w}) = w_{ttft}\Delta_{ttft} - w_{tpot}\Delta_{tpot}$, where $\Delta_{ttft}$ is the relative TTFT improvement and $\Delta_{tpot}$ is the relative TPOT degradation. If $S>0$, process locally; otherwise, go to P. System throughput is not directly optimized but improves naturally as a byproduct of reduced KV transmission.
    - **Design Motivation**: In a system scan of 3060 configurations (17 configs × 18 workloads × 10 QPS), the authors found that in **92.2% of (workload, QPS) combinations**, the optimal configurations for Turn 2 TTFT and TPOT differ—there is no static optimal solution, so per-request dynamic decisions are necessary.

3.  **Two-phase Routing: Offline Table + Online Look-up**:
    - **Function**: Shifts expensive optimization calculations to the offline phase, leaving only millisecond-level lookups online, ensuring near-zero overhead on latency-sensitive service paths.
    - **Mechanism**: Offline, TTFT/TPOT for $x{=}0$ and $x{=}1$ are measured for each grid cell, and boolean decisions are stored based on the sign of $S$. Online, requests are quantized to the nearest cell based on context length, I/O ratio, and QPS to retrieve $x^*$ in <1ms. Turn 1 always uses $x=0$ to ensure consistency. The system can revert to any static strategy (including traditional PD) by adjusting $\mathbf{w}$.
    - **Design Motivation**: Decouples "configuration scale" (P:D ratio, determining Turn 1 capacity) and "Turn 2+ SLO tuning" (weights, determining Pareto point) into two independent knobs. In traditional PD, these are forced together by the P:D ratio, forcing operators into multi-objective balancing.

### Loss & Training
Ours does not involve model training; it consists entirely of system-level scheduling strategies. All decisions are driven by offline measurements. Main parameters include user-provided SLO weights $w_{ttft}, w_{tpot}$ and discretization thresholds for the workload grid.

## Key Experimental Results

### Main Results
Hardware: 4× H100 80GB + NVLink. Models: Primarily Llama-3.1-8B (validated with Qwen2.5-14B/Qwen3-30B). Synthesis: 18 workloads × 10 QPS × 17 configs = 3060 data points. Real datasets: ShareGPT and WildChat.

| Configuration | Metric | $x=0$ Baseline | $x=1$ / PPD | Gain |
| :--- | :--- | :--- | :--- | :--- |
| 1P_3D Long Context High QPS | Turn 2 TTFT | Baseline | $x=1$ Improved | -73.3% |
| 2P_2D Long Context High QPS | Turn 2 TTFT | Baseline | $x=1$ Improved | -56.2% |
| 3P_1D Long Context High QPS | Turn 2 TTFT | Baseline | $x=1$ Improved | -24.9% |
| 1P_3D ShareGPT | Avg Query Latency | Baseline | PPD | -15~25% |
| 2P_2D / 3P_1D Multi-QPS | Success Rate | <95% (Degraded) | PPD 100% | Revived unusable configs |

### Ablation Study

| Config Category | TTFT Win Rate | TPOT Win Rate | Throughput Win Rate | Avg Win Rate |
| :--- | :--- | :--- | :--- | :--- |
| Replica (4R) | 63.3% | 0.6% | 0% | 21.3% |
| $x=0$ (Traditional PD) | 0% | 38.3% | 4.4% | 14.2% |
| $0<x<1$ Partial Routing | 3.3% | 33.3% | 27.8% | 21.5% |
| $x=1$ (Full AP-to-D) | 27.2% | 15.6% | 38.3% | 27.0% |

### Key Findings
- **Tighter P resources lead to higher local processing gains**: 1P_3D achieved up to 73.3% Turn 2 TTFT improvement, while 3P_1D achieved 24.9%—when P is the bottleneck, $x=1$ simply bypasses it.
- **No static optimum**: In 92.2% of workload-QPS combinations, the optimal TTFT configuration is not the same as the optimal TPOT configuration, validating the need for dynamic routing.
- **PPD revives unusable configs**: Under $x=0$, many QPS points for 2P_2D and 3P_1D had success rates <95% due to KV transmission saturation; PPD stabilized these to 100%.
- **Improvements hold across turns and model sizes**: Turn 2+ TTFT improvements remained ~70% across 2-16 turns and 8B/14B/30B models, indicating gains stem from architectural properties.

## Highlights & Insights
- **Challenging the Meta-hypothesis of PD**: For a long time, PD design was based on the implicit premise that "all prefills heavily interfere with decode." This paper breaks this down into full vs. append types using a 1024-token benchmark, quantifying an order of magnitude difference and opening a new design dimension for the disaggregated family.
- **Elegant Unification via Parameter $x$**: Traditional PD, Replica, partial routing, and full local processing all become special cases of $x \in \{0, \text{frac}, 1\}$. This "one parameter for all" formalization makes comparison and analysis highly clear.
- **Decoupling Scale and SLO Tuning**: In traditional PD, the P:D ratio serves both "Turn 1 capacity planning" and "Turn 2+ latency tuning." PPD isolates Turn 2+ tuning using weights $\mathbf{w}$, which can be migrated to other multi-objective system scheduling problems.
- **Offline-Online Pattern**: Shifting expensive decisions to the offline phase while maintaining <1ms online lookups has significant engineering value for latency-sensitive scenarios.

## Limitations & Future Work
- **Grid Discretization Coverage**: Accuracy and thresholds for the 3D grid are chosen empirically; new workload patterns may require rebuilding the table. The paper does not discuss adaptive update mechanisms.
- **Exclusion of Hybrid R+P/D Configs**: The authors admit that 7 hybrid configs were generally inferior to pure PD but do not provide a theoretical explanation or explore the potential of R in boundary cases.
- **Experiments limited to 4×H100 NVLink**: Cross-node slow links (RDMA/Ethernet) were only bandwidth-simulated; portability to real multi-node deployments needs more validation.
- **Ignore Prefix Cache Hit Rate Drift**: When multiple sessions compete for local KV slots on D nodes, the benefit of local processing might be offset by cache thrashing, which was not analyzed in depth.

## Related Work & Insights
- **vs AMPD (he2026)**: Concurrent work sharing the "route AP to D" intuition but uses real-time queue states. Ours uses an offline framework for more stable/predictable results and offers theoretical clarity via the optimization formalization.
- **vs Mooncake / MemServe / LMCache**: These use external distributed KV storage without changing the unidirectional PD protocol. PPD is complementary—obtaining most gains through routing alone without a new storage layer.
- **vs DuetServe / Nexus / TaiChi**: These perform SM slicing or dynamic resource reallocation within the GPU; PPD schedules at the higher request level and can be stacked with them.
- **vs Chunked-prefill (Splitwise / FastGen)**: Uses chunking to mitigate prefill-decode interference. PPD proves that append-prefills are essentially "small chunks" that inherently cause little interference, aligning with the bottom-up motivation of chunking.

## Rating
- Novelty: ⭐⭐⭐⭐ Re-examined the core hypothesis of PD and formalized the findings into a schedulable optimization framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3060 configuration scan + synthetic + real data + multi-model/multi-turn validation; rare for complete system evaluations in this field.
- Writing Quality: ⭐⭐⭐⭐ Clear chain of argument from micro-benchmark to formalization to algorithm to measurement, though some notations are a bit dense.
- Value: ⭐⭐⭐⭐⭐ A plug-and-play improvement for production LLM serving that yields significant TTFT gains without modifying models or protocols.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)
- [\[ACL 2026\] ETHICMIND: A Risk-Aware Framework for Ethical-Emotional Alignment in Multi-Turn Dialogue](../../ACL2026/dialogue/ethicmind_a_risk-aware_framework_for_ethical-emotional_alignment_in_multi-turn_d.md)
- [\[NeurIPS 2025\] HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location](../../NeurIPS2025/dialogue/hygen_efficient_llm_serving_via_elastic_online-offline_request_co-location.md)
- [\[ACL 2026\] SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation](../../ACL2026/dialogue/spasm_stable_persona-driven_agent_simulation_for_multi-turn_dialogue_generation.md)
- [\[ICML 2026\] Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives](is_your_llm_overcharging_you_tokenization_transparency_and_incentives.md)

</div>

<!-- RELATED:END -->
