---
title: >-
  [Paper Note] OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration
description: >-
  [ICML 2026][LLM Efficiency][LLM inference serving] OServe models the joint optimization of "resource allocation + parallel strategy + request routing" for LLM serving as a bilevel maximum flow problem on a flow network.…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "LLM inference serving"
  - "heterogeneous deployment"
  - "flow network scheduling"
  - "workload prediction"
  - "online model switching"
date: 2026-05-08
content_hash: 490b188ff9dc6b25
---

# OServe: Accelerating LLM Serving via Spatial-Temporal Workload Orchestration

**Conference**: ICML 2026  
**arXiv**: [2602.12151](https://arxiv.org/abs/2602.12151)  
**Code**: None  
**Area**: LLM Efficiency / Inference Serving Systems  
**Keywords**: LLM inference serving, heterogeneous deployment, flow network scheduling, workload prediction, online model switching  

## TL;DR
OServe models the joint optimization of "resource allocation + parallel strategy + request routing" for LLM serving as a bilevel maximum flow problem on a flow network. Combined with LSTM workload prediction and ad hoc model switching based on GPU interconnects, it addresses the heterogeneity of real-world traffic in both spatial (different request types) and temporal (varying composition over time) dimensions. Compared to vLLM, it improves end-to-end P99 latency and throughput by an average of 1.5$\times$ and up to 2$\times$.

## Background & Motivation

**Background**: Most existing LLM inference systems (vLLM, Llumnix, Dynamo+vLLM, etc.) assume that workloads are spatially homogeneous and temporally static. Consequently, they deploy $N$ identical model replicas using a single parallel strategy and uniform resource allocation.

**Limitations of Prior Work**: Real-world traffic exhibits two types of heterogeneity: (i) **Spatial Heterogeneity**: Concurrent requests include short-input/short-output (chat, summarization) which are compute-intensive, and long-input/long-output (generation, code) which are memory-bandwidth intensive. (ii) **Temporal Heterogeneity**: Traffic composition changes hourly or even minutely; business hours are dominated by short outputs, while the proportion of long outputs increases at night. Using Azure public traces, the authors measured an extreme distribution with input lengths from 1–7999 and output lengths from 1–5000.

**Key Challenge**: Compute-intensive workloads perform best with more replicas (Data Parallelism, DP) to saturate compute units, whereas memory-intensive workloads prefer higher parallelism degrees (Tensor Parallelism TP / Pipeline Parallelism PP) to distribute the KV cache. A static deployment cannot be optimal for all workloads, but traditional systems lack the capability for "scheduled deployment switching" because reloading a 70B model takes minutes.

**Goal**: (a) Determine a **heterogeneous deployment** given a traffic profile—where different replicas can use different DP/TP/PP configurations; (b) provide optimal "request-to-replica" dispatching; (c) perform **fast switching** of deployments instead of cold-start reloading when traffic changes.

**Key Insight**: Model heterogeneous deployment and request dispatching simultaneously as a maximum flow problem on a directed flow network, treating "GPU allocation and parallel strategy" as an upper-level discrete search and "request dispatching" as a lower-level maximum flow problem. Use LSTM to predict next-minute request composition and migrate parameter shards directly across GPUs via NVLink/InfiniBand during switching instead of loading from disk.

**Core Idea**: Jointly solve spatial and temporal heterogeneity using "flow-network-driven bilevel scheduling + GPU-interconnect-based hot switching."

## Method

### Overall Architecture
OServe consists of three components: (1) **Workload Predictor** reads historical traces, clusters requests into types based on (input length, output length) via k-means, and trains an LSTM for each type to predict the arrival rate for the next minute; (2) **OServe Scheduler** takes cluster specifications, model configurations, and predicted traffic to search for the optimal serving strategy (including model deployment $\{d_r, s_r\}$ and request dispatching $\{x_{k,j}\}$); (3) **Switch Planner** takes the "source strategy $\rightarrow$ target strategy" and generates a parameter migration plan using a greedy algorithm, executed by the engine over GPU interconnects. These modules run in a loop every minute to keep the deployment matched with the predicted traffic.

### Key Designs

1.  **Bilevel Flow Network Scheduling (Spatial Heterogeneity)**:
    *   **Function**: Simultaneously optimizes "GPU count per replica + parallel strategy" and "assignment of request types to replicas" under the constraints of total GPUs $D$ and target replica count $R$.
    *   **Mechanism**: The **lower level** models request routing as a flow network—each workload edge $w_j$ from source $\mathcal{S}$ has a capacity equal to the arrival rate $\lambda_j$. Each replica $k$ is split into an entry node $c_k^{in}$ and exit node $c_k^{out}$, with a least common multiple $M_k = \mathrm{lcm}\{n_{k,j}\}$ as a "normalized capacity for mixed workloads," where each type-$j$ request consumes $M_k/n_{k,j}$ units. The preflow-push algorithm finds the optimal dispatch via maximum flow. The **upper level** uses a flow-network-guided heuristic search: it reallocates GPUs based on "bottleneck replicas (full flow)" and "redundant replicas (incomplete flow)" from the lower level, then enumerates parallel strategy combinations for the new configuration, stopping after 20 iterations without improvement.
    *   **Design Motivation**: Reduces the discrete $\{d_r, s_r\}$ search from exponential complexity to a few dozen heuristic rounds. The bottleneck signal from the flow network is much more directed than blind enumeration—on a 16-GPU setup, it takes only 12s compared to 50s for brute force, with only a 6% P99 difference.

2.  **Typed LSTM Workload Prediction (Temporal Heterogeneity)**:
    *   **Function**: Predicts the arrival count of each request type for the next period at 1-minute granularity as input for the scheduler.
    *   **Mechanism**: Clusters historical requests into types (typically 4) based on (input, output) lengths via k-means, then trains an independent LSTM for **each category** (sequence length 50, using the past 50 minutes to predict the next). A key trick is **not trying to predict the specific input/output length of each request**—LSTMs struggle with such extreme high-variance signals (1–7999); instead, it predicts "categorical arrival rates," transforming the task into 4 stable low-dimensional sequences.
    *   **Design Motivation**: Ablation shows that predicting aggregated arrival rates without type decomposition results in RRMSE soaring to 40% and training failure. Categorical prediction reduces RRMSE to 5.045% with a per-prediction time of only 30ms, aligning with the 1-minute switching cadence.

3.  **Ad hoc Model Switching (Greedy + Local-First Heuristic)**:
    *   **Function**: Minimizes total communication time between two serving strategy switches to avoid 50s+ cold loads.
    *   **Mechanism**: Source and target strategies shard parameters differently; each parameter shard corresponds to a set of source GPUs and target GPUs. The algorithm iterates through feasible source GPUs for each target shard, choosing the one with the lowest communication load. A local-first heuristic prioritizes sources **intra-node** (NVLink 400GB/s) before searching **inter-node** (InfiniBand/RoCE 10–200GB/s). KV caches are migrated synchronously—short sequence KVs are drained at the source, while long sequence KVs are moved to the target using the same greedy approach with a 10–20% headroom buffer to prevent OOM jitter.
    *   **Design Motivation**: In Trace 2, the minimum switching interval is 5 minutes; a 50s cold load would increase average latency by ~17%. Ad hoc switching reduces overhead to under 10s, decreasing average P99 by 12% (up to 17%).

### Loss & Training
This is a pure systems work with no training loss. LSTMs were trained on two weeks of real Azure traces with a 9:1 train/test split. The scheduling algorithm uses deterministic max flow + heuristic search, while the switching algorithm uses greedy + heuristic pruning.

## Key Experimental Results

### Main Results
The experimental platform consists of 4 nodes of 8$\times$H100-80GB servers (Intra-node NVLink 400GB/s, Inter-node InfiniBand 200GB/s). Models include OPT-30B/66B, LLaMA-30B, LLaMA2-70B. Traces are taken from the Azure Public Dataset (800 min and 50 min), compared across P1–P6 slices.

| Baseline | Relative P99 Latency / Throughput Gain | Average Gain |
|---|---|---|
| vLLM (static) | Up to 2.0$\times$ | 1.5$\times$ |
| vLLM (reload) | Up to 1.5$\times$ | 1.3$\times$ |
| Llumnix | Up to 1.51$\times$ | 1.32–1.51$\times$ |
| Dynamo+vLLM | -- | 12–20% |
| 32-GPU Cluster (LLaMA2-70B) | Up to 1.9$\times$ | -- |

In terms of spatial sensitivity, the authors constructed S1–S5 by increasing the coefficient of variation (CV) of the workload distribution. OServe's acceleration over vLLM (static) rises monotonically from 1.14$\times$ (CV=0.112) to 2.66$\times$ (CV=0.688). Regarding temporal sensitivity, T1$\rightarrow$T4 improvement rises from 1.23$\times$ to 1.98$\times$.

### Ablation Study

| Configuration | LLaMA2-70B/OPT-66B P99 Improvement | Note |
|---|---|---|
| vLLM (reload) baseline | -- | Starting point |
| + Heterogeneous Deployment | Avg. 34% / Max 52% | Different replicas use different parallel configs |
| + Optimal Request Dispatching | Avg. 64% / Max 109% | Route loads to matching replicas |
| + ad hoc Switching | Further P99 reduction: Avg. 12% / Max 17% | Saves cold loading overhead |
| LSTM Prediction (by type) | RRMSE 5.045% | -- |
| Moving Average | RRMSE 43.375%, Throughput -41% | Simple baseline |
| LSTM w/o Types | RRMSE ~40%, non-convergent | Proves necessity of categorization |

### Key Findings
- Gains from heterogeneous deployment correlate positively with traffic heterogeneity: the more skewed the traffic, the greater the OServe advantage, reaching up to 2.66$\times$.
- Heuristic search is over 4$\times$ faster than brute force on 16-GPUs with only a 6% P99 loss, indicating flow-network-guided signals are sufficiently accurate.
- The relative benefit of ad hoc switching is most significant in **high-frequency fluctuation scenarios**; stable loads require few switches, aligning with the system's "switch only when needed" philosophy.

## Highlights & Insights
- Unifying heterogeneous resource allocation and request dispatching into a flow network framework transforms a seemingly NP-hard joint scheduling problem into a solvable "bilevel max-flow" form, which is structurally elegant.
- The concept of "not predicting request-level length, but categorical arrival rates" is a universal trick to reduce prediction difficulty, applicable to other system-level prediction tasks (e.g., GPU scheduling, storage scheduling).
- Ad hoc switching leveraging GPU interconnects for parameter migration is a methodology worth referencing for multi-tenant GPU clusters, Mixture-of-Experts routing, and LoRA hot swapping.

## Limitations & Future Work
- Bilevel scheduling requires offline profiling of processing rates $n_{k,j}$ and capacities $e_{k,j}$ for each (replica, load type); first-time deployment cost for new models or hardware is high.
- Prediction errors are inevitable; the paper uses 1-minute granularity and fast switching to mitigate this, but extreme bursts (second-level spikes) still lag by one cycle before correction.
- Currently only considers dense decoder LLMs; compatibility with new paradigms like MoE, speculative decoding, or disaggregated prefill/decode was not provided.

## Related Work & Insights
- **vs vLLM**: vLLM excels in paged KV cache and continuous batching but uses static single-strategy deployment; OServe treats vLLM as a backend engine and performs scheduling at the "strategy layer."
- **vs Llumnix**: Llumnix performs request-level dynamic migration but assumes homogeneous instance configurations; OServe adjusts both replica configurations and dispatching, outperforming it by 1.32–1.51$\times$.
- **vs Dynamo**: Dynamo performs autoscaling for prefill/decode decoupling, but worker parallelism is fixed; OServe allows parallelism to change with the load, improving end-to-end performance by another 12–20%.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of flow network + bilevel heuristics + ad hoc switching is a rare complete solution in LLM serving.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 baselines / 4 models / 2 traces / 8-32 GPUs / spatial & temporal sensitivity; very solid.
- Writing Quality: ⭐⭐⭐⭐ System diagrams are clear and algorithms are specific, though mathematical notation is slightly fragmented.
- Value: ⭐⭐⭐⭐⭐ Industrial-grade serving system; 1.5$\times$ average acceleration is ready for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration](team_temporal-spatial_consistency_guided_expert_activation_for_moe_diffusion_lan.md)
- [\[ICML 2026\] Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving](theoretically_optimal_attentionffn_ratios_in_disaggregated_llm_serving.md)
- [\[ICML 2026\] GraphFlow: A Graph-Based Workflow Management for Efficient LLM-Agent Serving](graphflow_a_graph-based_workflow_management_for_efficient_llm-agent_serving.md)
- [\[ICLR 2026\] LycheeDecode: Accelerating Long-Context LLM Inference via Hybrid-Head Sparse Decoding](../../ICLR2026/llm_efficiency/lycheedecode_accelerating_long-context_llm_inference_via_hybrid-head_sparse_deco.md)
- [\[NeurIPS 2025\] Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving](../../NeurIPS2025/llm_efficiency/efficient_training-free_online_routing_for_high-volume_multi-llm_serving.md)

</div>

<!-- RELATED:END -->
