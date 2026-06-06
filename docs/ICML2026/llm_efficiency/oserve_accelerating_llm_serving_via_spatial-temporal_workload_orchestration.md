---
title: >-
  [Paper Note] OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration
description: >-
  [ICML 2026][LLM Efficiency][LLM inference serving] OServe jointly models LLM serving’s “resource allocation + parallelism strategy + request routing” as a two-level max-flow problem on a flow network. It leverages LSTM-b…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "LLM inference serving"
  - "heterogeneous deployment"
  - "flow network scheduling"
  - "workload prediction"
  - "online model switching"
date: 2026-05-08
content_hash: a3f36966c2fb5139
---

# OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration

**Conference**: ICML 2026  
**arXiv**: [2602.12151](https://arxiv.org/abs/2602.12151)  
**Code**: None  
**Area**: LLM Efficiency / Inference Serving Systems  
**Keywords**: LLM inference serving, heterogeneous deployment, flow network scheduling, workload prediction, online model switching  

## TL;DR
OServe jointly models LLM serving’s “resource allocation + parallelism strategy + request routing” as a two-level max-flow problem on a flow network. It leverages LSTM-based workload prediction and ad hoc model switching via GPU interconnects to address real-world traffic heterogeneity in both spatial (different request types) and temporal (composition shifts over time) dimensions. Compared to vLLM, OServe achieves an average 1.5× and up to 2× improvement in end-to-end P99 latency and throughput.

## Background & Motivation

**Background**: Existing LLM inference systems (vLLM, Llumnix, Dynamo+vLLM, etc.) mostly assume spatially homogeneous and temporally static workloads, thus deploying N identical model replicas with a single parallelism strategy and uniform resource allocation.

**Limitations of Prior Work**: Real-world traffic exhibits two types of heterogeneity—(i) **Spatial heterogeneity**: Concurrent requests include short-input/short-output (chat, summarization), which are compute-intensive, and long-input/long-output (generation, code), which are memory bandwidth-intensive; (ii) **Temporal heterogeneity**: Traffic composition shifts hourly or even minutely, with business hours dominated by short outputs and nighttime by long outputs. On Azure public traces, input lengths range from 1–7999 and output lengths from 1–5000, showing extreme distributions.

**Key Challenge**: Compute-intensive loads prefer more replicas (data parallelism, DP) to maximize compute utilization; memory-intensive loads prefer higher parallelism (tensor parallelism, TP / pipeline parallelism, PP) to spread out the KV cache. A single static deployment cannot be optimal for all workloads, and traditional systems lack even the ability to “switch deployments by time period” since reloading a 70B model takes minutes.

**Goal**: (a) Given a traffic profile, find a **heterogeneous deployment**—allowing different replicas to use different DP/TP/PP configurations; (b) Simultaneously determine the optimal “request→replica” assignment; (c) When traffic changes, **rapidly switch** deployments instead of cold-start reloading.

**Key Insight**: Jointly model heterogeneous deployment and request assignment as a max-flow problem on a directed flow network: “how many GPUs to allocate, which parallelism to use” as the upper-level discrete search, and “which replica to assign each request” as the lower-level max-flow; use LSTM to predict the next minute’s request composition; during switching, migrate parameter shards directly between GPUs via NVLink/InfiniBand instead of loading from disk.

**Core Idea**: “Flow-network-driven two-level scheduling + GPU-interconnect-based hot switching” jointly addresses spatial and temporal heterogeneity.

## Method

### Overall Architecture
OServe consists of three components: (1) **Workload Predictor** reads historical traces, clusters requests by (input length, output length) using k-means, and trains an LSTM for each class to predict the next minute’s arrival rate; (2) **OServe Scheduler** receives cluster specs, model configs, and predicted traffic, then uses the two-level algorithm (§4) to search for the optimal serving strategy (including model deployment $\{d_r, s_r\}$ and request assignment $\{x_{k,j}\}$); (3) **Switch Planner** receives the “source→target strategy,” generates a parameter migration plan via a greedy algorithm, and executes ad hoc switching over GPU interconnects. These modules run in a loop every minute to keep deployment aligned with predicted traffic.

### Key Designs

1. **Two-Level Flow Network Scheduling (Spatial Heterogeneity)**:

    - **Function**: Given total GPU count $D$ and target replica count $R$, jointly optimize “number of GPUs per replica + parallelism strategy” and “which replica serves each request class.”
    - **Mechanism**: **Lower level** models request routing as a flow network—each workload edge $w_j$ from source $\mathcal{S}$ has capacity equal to the arrival rate $\lambda_j$; replica $k$ is split into input node $c_k^{in}$ and output node $c_k^{out}$, with an intermediate least common multiple $M_k = \mathrm{lcm}\{n_{k,j}\}$ as the normalized capacity for mixed workloads; each type-$j$ request consumes $M_k/n_{k,j}$ units; preflow-push algorithm finds the max flow for optimal assignment. **Upper level** uses a flow-network-guided heuristic: based on bottleneck (fully utilized) and redundant (underutilized) replicas from the lower level, reallocates GPUs and enumerates new parallelism combinations, stopping after 20 iterations without improvement.
    - **Design Motivation**: Reduces the discrete $\{d_r, s_r\}$ search from exponential to a few dozen heuristic rounds; flow network bottleneck signals are much more informative than blind enumeration—on 16 GPUs, brute-force search takes 50s, this method only 12s with just 6% P99 loss.

2. **LSTM-Type Workload Prediction (Temporal Heterogeneity)**:

    - **Function**: Predicts the next period’s arrival count for each request class at 1-minute granularity, serving as scheduler input.
    - **Mechanism**: First, k-means clusters historical requests by (input length, output length) (typically 4 classes), then trains an independent LSTM for **each class** (sequence length 50, i.e., past 50 minutes to predict the next minute). The key trick is **not predicting individual request input/output lengths**—LSTM cannot learn such high-variance (1–7999) signals; instead, it predicts “class arrival rates,” turning the task into 4 stable low-dimensional sequences.
    - **Design Motivation**: Ablation shows that using LSTM to directly predict aggregate arrival rates without class decomposition yields RRMSE up to 40% and non-convergent training; with class decomposition, RRMSE drops to 5.045% and single prediction takes only 30ms, matching the 1-minute switching cadence.

3. **Ad hoc Model Switching (Greedy + Locality Heuristic)**:

    - **Function**: Minimizes total communication time between serving strategy switches, avoiding 50s+ cold loading.
    - **Mechanism**: Source and target strategies differ in parameter sharding; each shard maps to a set of source and target GPUs. For each target shard, the algorithm iterates over feasible source GPUs, selecting the one with the lowest communication load. The heuristic first seeks sources **within the same node** (NVLink 400GB/s), falling back to **cross-node** (InfiniBand/RoCE 10–200GB/s) if needed; KV cache migration: short-sequence KV is drained at the source, long-sequence KV is greedily moved to the target GPU, with a buffer preallocated with 10–20% headroom to prevent OOM jitter.
    - **Design Motivation**: On trace 2, the minimum switching interval is only 5 minutes; if each switch took 50s cold loading, average latency would increase by ~17%. Ad hoc switching reduces overhead to under 10s, lowering P99 by an average of 12% and up to 17%.

### Loss & Training
This is a pure systems work with no training loss. LSTM is trained on two weeks of Azure real traces with a 9:1 train/test split; the scheduling algorithm is deterministic max-flow + heuristic search; the switching algorithm is greedy + heuristic pruning.

## Key Experimental Results

### Main Results
Experiments were conducted on 4 servers with 8×H100-80GB each, intra-node NVLink 400GB/s, inter-node InfiniBand 200GB/s. Models include OPT-30B/66B, LLaMA-30B, LLaMA2-70B; traces are from two segments (800 min and 50 min) of the Azure Public Dataset, compared across slices P1–P6.

| Baseline | P99 Latency / Throughput Relative Gain | Average Gain |
|---|---|---|
| vLLM (static) | up to 2.0× | 1.5× |
| vLLM (reload) | up to 1.5× | 1.3× |
| Llumnix | up to 1.51× | 1.32–1.51× |
| Dynamo+vLLM | -- | 12–20% |
| 32-GPU Cluster (LLaMA2-70B) | up to 1.9× | -- |

For spatial sensitivity, the authors constructed S1–S5 with varying workload distribution CV; OServe’s speedup over vLLM(static) increases monotonically from 1.14× at CV=0.112 to 2.66× at CV=0.688. For temporal sensitivity, T1→T4 increases from 1.23× to 1.98×.

### Ablation Study

| Configuration | LLaMA2-70B/OPT-66B P99 Improvement | Notes |
|---|---|---|
| vLLM (reload) baseline | -- | Starting point |
| + Heterogeneous Deployment | avg 34% / max 52% | Different replicas use different parallel configs |
| + Optimal Request Assignment | avg 64% / max 109% | Route loads to best-matched replicas |
| + ad hoc Switching | further P99 drop avg 12% / max 17% | Saves cold loading |
| LSTM Prediction (by type) | RRMSE 5.045% | -- |
| Moving Average | RRMSE 43.375%, throughput -41% | Simple baseline |
| LSTM w/o type separation | RRMSE ~40%, non-convergent | Shows necessity of type separation |

### Key Findings
- The benefit of heterogeneous deployment correlates with workload heterogeneity: the more skewed the traffic, the greater OServe’s advantage, up to 2.66×.
- Heuristic search is over 4× faster than brute-force on 16 GPUs, with only 6% P99 loss, indicating the flow-network-guided signal is sufficiently accurate.
- The relative gain of ad hoc switching is most significant in **high-frequency fluctuation scenarios**; under stable loads, switching is rarely needed, aligning with the system’s philosophy: “switch only when necessary.”

## Highlights & Insights
- Unifying heterogeneous resource allocation and request assignment into a max-flow framework transforms an apparently NP-hard joint scheduling problem into a “two-level LP/max-flow” solvable form—structurally elegant.
- The idea of “predicting only class arrival rates, not per-request lengths” is a general trick to reduce prediction difficulty, directly transferable to other systems prediction tasks (e.g., GPU scheduling, storage scheduling).
- The ad hoc switching approach leveraging GPU interconnects for parameter migration is valuable for multi-tenant GPU clusters, Mixture-of-Experts routing, LoRA hot swapping, etc.

## Limitations & Future Work
- Two-level scheduling requires offline profiling of each (replica, workload type) processing rate $n_{k,j}$ and edge capacity $e_{k,j}$, incurring high initial deployment cost for new models/hardware.
- Prediction errors are inevitable; the paper uses 1-minute granularity + fast switching as a safeguard, but extreme bursty traffic (sub-minute spikes) will only be corrected in the next cycle.
- Currently only dense decoder LLMs are considered; adaptation to MoE routing models, speculative decoding, disaggregated prefill/decode, and other new paradigms is not addressed.

## Related Work & Insights
- **vs vLLM**: vLLM excels at paged KV cache and continuous batching, but deployment is static and single-strategy; OServe uses vLLM as the backend engine, handling scheduling at the “strategy layer.”
- **vs Llumnix**: Llumnix supports request-level dynamic migration, but all instances are homogeneous; OServe jointly tunes replica configs and assignment, yielding 1.32–1.51× better P99.
- **vs Dynamo**: Dynamo decouples prefill/decode for autoscaling, but each worker’s parallelism is fixed; OServe allows parallelism to adapt to workload, achieving 12–20% end-to-end improvement.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of flow network, two-level heuristic, and ad hoc switching is a rare, comprehensive solution in LLM serving.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 major baselines / 4 models / 2 trace segments / 8–32 GPUs / spatial & temporal sensitivity; substantial scale.
- Writing Quality: ⭐⭐⭐⭐ System diagrams are clear, algorithm descriptions are concrete, but mathematical notation is somewhat fragmented, making initial reading challenging.
- Value: ⭐⭐⭐⭐⭐ Industrial-grade serving system; 1.5× average acceleration is directly deployable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving.md)
- [\[ICML 2026\] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)
- [\[ICLR 2026\] LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](../../ICLR2026/llm_efficiency/lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)
- [\[NeurIPS 2025\] Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](../../NeurIPS2025/llm_efficiency/efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)
- [\[ICLR 2026\] MVAR: Visual Autoregressive Modeling with Scale and Spatial Markovian Conditioning](../../ICLR2026/llm_efficiency/mvar_visual_autoregressive_modeling_with_scale_and_spatial_markovian_conditionin.md)

</div>

<!-- RELATED:END -->
