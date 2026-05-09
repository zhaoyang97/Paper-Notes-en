---
title: >-
  [Paper Note] HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location
description: >-
  [NeurIPS 2025][Dialogue Systems][online-offline co-location] This paper proposes HyGen, an interference-aware LLM inference system that achieves elastic co-location of online and offline workloads through an accurate batch latency predictor, an SLO-aware performance profiler, and a prefix-sharing-maximization scheduling strategy, delivering 3.87–5.84× throughput gains while strictly guaranteeing SLO compliance.
tags:
  - NeurIPS 2025
  - Dialogue Systems
  - online-offline co-location
  - latency prediction
  - SLO guarantee
  - prefix sharing
  - elastic scheduling
date: 2026-05-08
content_hash: 76e7097a655a9e31
---

# HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location

**Conference**: NeurIPS 2025
**arXiv**: [2501.14808](https://arxiv.org/abs/2501.14808)
**Code**: [https://github.com/UIUC-MLSys/HyGen](https://github.com/UIUC-MLSys/HyGen)
**Area**: Dialogue Systems
**Keywords**: online-offline co-location, latency prediction, SLO guarantee, prefix sharing, elastic scheduling

## TL;DR
This paper proposes HyGen, an interference-aware LLM inference system that achieves elastic co-location of online and offline workloads through an accurate batch latency predictor, an SLO-aware performance profiler, and a prefix-sharing-maximization scheduling strategy, delivering 3.87–5.84× throughput gains while strictly guaranteeing SLO compliance.

## Background & Motivation

**Two workload patterns in LLM serving**: LLM applications can be categorized into online serving (interactive tasks such as chatbots and code assistants that require low latency) and offline serving (batch processing tasks such as data synthesis, document summarization, and model evaluation that prioritize throughput). Their SLO requirements differ fundamentally—online serving emphasizes P99 TTFT (time-to-first-token) and TBT (time-between-tokens), whereas offline serving treats QPS/TPS throughput as the primary metric.

**Root cause of resource inefficiency**: The prevailing deployment practice allocates separate GPU clusters to each workload type. However, analysis of production traces from Azure LLM services reveals a severe efficiency problem—online request load is highly time-varying, exhibiting not only diurnal periodicity but also **burst fluctuations of up to 3× within minutes**. To meet latency requirements under peak load, operators must provision GPUs according to peak demand (minute-level elastic scaling is practically infeasible), resulting in substantial GPU idleness during off-peak periods.

**Opportunity and challenges of co-location**: Co-locating online and offline requests on the same inference instance can utilize idle resources for offline tasks during off-peak periods, but introduces three core challenges: (1) **Diverse SLOs**—different applications, and even different users of the same application, impose different latency requirements (P99 TBT vs. mean TTFT vs. hard upper bounds), making it difficult to formulate a unified resource-sharing policy; (2) **Pervasive uncertainty**—the burstiness of request arrivals combined with the unpredictability of output lengths makes scheduling decisions highly uncertain; (3) **Performance interference**—large offline batches may severely block online requests (head-of-line blocking), and mixing long inputs with short queries degrades latency for all requests by an order of magnitude.

**Core insight**: By precisely modeling batch execution time and quantifying co-location interference costs, fine-grained opportunistic scheduling becomes feasible—maximizing idle compute utilization for offline tasks while strictly guaranteeing online SLOs.

## Method

### Overall Architecture
HyGen adopts a dual-queue architecture that manages latency-sensitive online requests and throughput-oriented offline requests separately. Its two-phase scheduling pipeline operates as follows: the **online phase** first forms an initial batch using FCFS or fair scheduling, and evicts offline requests via a priority preemption mechanism when necessary to protect online performance; the **offline phase** then uses the latency predictor to assess residual capacity and fills in offline requests without violating SLOs. HyGen operates as an instance-level scheduler receiving requests from an upstream system-level router (e.g., Preble), so per-instance request concurrency and scheduling overhead are naturally bounded.

### Key Designs

1. **Latency Predictor**:

    - *Function*: Accurately estimates execution time for different request batch compositions to support real-time scheduling decisions.
    - *Mechanism*: Models batch execution time based on the distinct computational patterns of the two LLM inference phases—prefill exhibits quadratic complexity in attention computation, while decode is linear. The prediction model is $T_{batch} = f(S_p, S_d, S_p^2, S_d^2, N_p, N_d)$, where $S_p, S_d$ denote total prefill/decode token counts and $N_p, N_d$ denote request counts; quadratic terms capture non-linear scaling effects. Linear regression is used as the prediction model, with training data collected through systematic profiling.
    - *Design Motivation*: The choice of linear regression is deceptively simple yet well-motivated: (1) inference is extremely fast (~18 μs per call), enabling real-time scheduling; (2) the compact feature set ensures stable generalization across diverse workload patterns; (3) training is lightweight (80k samples in 15 ms), making adaptation to different hardware trivial. The measured MAPE is only 1.07%–1.78%, outperforming more complex models in reliability.

2. **SLO-aware Profiler**:

    - *Function*: Translates latency predictions into concrete scheduling constraints—determining the maximum permissible offline load under a given SLO.
    - *Mechanism*: The profiler first determines a feasible latency budget range based on workload characteristics and SLO requirements, then uses binary search within this range to find the tightest batch latency budget that satisfies the SLO. A key distinction is that the latency budget for a single batch and the overall SLO (e.g., mean TBT or P99 TTFT) are separated by a statistical gap; the profiler bridges this gap through test-run validation. At deployment time, the resulting budget serves as the per-batch latency limit in the two-phase scheduler.
    - *Design Motivation*: A naïve approach of directly using the SLO value as the batch latency budget leads to either over-conservative or under-conservative behavior. The SLO-aware profiler establishes an accurate mapping between batch-level latency and end-to-end SLO through offline profiling, enabling more precise exploitation of residual capacity.

3. **Prefix Sharing Maximization (PSM)**:

    - *Function*: Optimizes the scheduling order of offline requests to maximize KV cache prefix reuse and improve throughput.
    - *Mechanism*: All offline requests are organized into a Trie structure, where leaf nodes correspond to requests and shared prefixes form common ancestor paths. DFS traversal determines scheduling priority—requests sharing the most prefixes are grouped together. For example, given the queue ("What is ML", "How to code", "What is AI", "How to debug"), PSM reorders to ("What is ML", "What is AI"), ("How to code", "How to debug"), maximizing cache reuse of the "What is" and "How to" prefixes.
    - *Design Motivation*: Naïve FCFS scheduling completely ignores prefix-sharing opportunities. PSM introduces cache efficiency optimization orthogonally within the SLO-aware scheduling framework. To prevent starvation of requests with low prefix-sharing affinity, an extended PSM variant incorporates a utility ratio that accounts for request freshness—balancing efficiency and fairness by comparing DFS order against the oldest request in a self-balancing BST.

### Loss & Training
The latency predictor is trained via linear regression: training data are collected by systematically profiling the target hardware across varying request counts, sequence length distributions, and batch compositions. Approximately 80k training samples are used, with training taking only 15 ms on CPU. The SLO-aware profiler determines latency budgets offline via binary search, requiring no learning at runtime. The priority preemption mechanism preserves the execution state of evicted requests to support subsequent resumption.

## Key Experimental Results

### Main Results
**Online SLO compliance and throughput gains (Llama2-7B, Azure trace, A100)**:

| Method | Mean TBT SLO | P99 TBT SLO | Mean TTFT SLO | P99 TTFT SLO | Offline Throughput Gain |
|--------|-------------|-------------|--------------|-------------|------------------------|
| Sarathi (online only) | ✓ | ✓ | ✓ | ✓ | 1.0× |
| Sarathi++ (mixed) | ✗ (no SLO control) | ✗ | ✗ | ✗ | - |
| HyGen* | ✓ | ✓ | ✓ | ✓ | 1.82× |
| **HyGen** | **✓** | **✓** | **✓** | **✓** | **3.87–5.84×** |

**Cross-model/hardware validation**:

| Model | Hardware | Offline Throughput Gain | Total Throughput vs. Offline-only |
|-------|----------|------------------------|----------------------------------|
| Llama2-7B | A100×4 | 3.87× | 84.3% |
| Qwen-14B | A40×4 | 5.84× | - |
| Yi-34B (TP2+PP2) | A40×4 | 1.89× | - |
| Sheared-LLaMA-2.7B | A5000×1 | 2.18× | - |

### Ablation Study

| Component | Offline Throughput Gain | Notes |
|-----------|------------------------|-------|
| Without SLO-aware profiler (direct SLO as budget) | Lower | Over-conservative or SLO violations |
| **With SLO-aware profiler** | **Significantly higher** | Accurate mapping: batch latency → end-to-end SLO |
| Without PSM prefix sharing (FCFS) | 1.0× | No prefix reuse |
| **With PSM prefix sharing** | **Up to 4×** | Validated via simulation on MMLU dataset |
| Latency predictor MAPE = 1% | Baseline | Optimal |
| Latency predictor MAPE = 5% | Slight decrease, SLO still met | System is robust to prediction accuracy |
| Latency predictor MAPE = 10% | Gradual decrease | Impact of accuracy degradation remains limited |

### Key Findings
- The latency predictor achieves high accuracy: MAPE of 1.78% on Llama2-7B and 1.07% on Qwen-14B.
- HyGen's offline throughput reaches 84.3% of that of Sarathi-offline (pure offline mode), indicating that co-location wastes virtually no compute.
- The system can simultaneously satisfy multiple SLO constraints (e.g., P99 TTFT 8% + mean TBT 10–50%); overall performance is governed by the most stringent SLO.
- HyGen remains effective on the Mooncake trace, which features more extreme load fluctuations.
- The system is robust to predictor accuracy—increasing MAPE from 1% to 5% causes only modest throughput reduction while maintaining SLO compliance.

## Highlights & Insights
- **Integration of production-grade engineering depth and academic innovation**: The latency predictor design appears simple (linear regression), yet achieves ~1% prediction error by leveraging principled understanding of prefill quadratic / decode linear complexity to select features. This paradigm of "using the right prior knowledge to make simple models outperform complex ones" is broadly instructive.
- **The SLO-aware profiler bridges the gap between per-batch and statistical SLOs**: This is a critical detail overlooked by many systems papers—a P99 TBT SLO constrains the 99th percentile of the TBT distribution across all requests, not the latency of each individual batch. The profiler establishes this mapping through profiling, enabling batch-level latency predictions to directly serve statistical SLO guarantees.
- **Unified framework for prefix sharing and SLO-aware scheduling**: Cache optimization and latency management are addressed jointly within the same scheduling pipeline, avoiding potential conflicts that could arise from optimizing each independently.

## Limitations & Future Work
- The linear regression predictor may require re-profiling when model architectures or use cases differ substantially, limiting cross-architecture adaptability.
- Evaluation focuses on decoder-only models (Llama/Qwen/Yi/Mistral); applicability to encoder-decoder architectures (e.g., T5) or MoE models (e.g., Mixtral) remains unexplored.
- The fairness extension of PSM (the utility ratio mechanism for starvation prevention) has theoretical justification but lacks large-scale, long-horizon experimental validation.
- When online load is sustained near peak capacity, offline requests can rarely be scheduled, reducing the benefit of co-location to near zero; integration with cluster-level load balancing is necessary in such scenarios.
- Output length unpredictability remains incompletely resolved—the current system absorbs this uncertainty through margins in the SLO profiler, but SLO violations may occur in extreme cases.
- Multi-tenant scenarios involving differentiated priority management across users are not addressed.

## Related Work & Insights
- **vs. Sarathi-Serve**: Sarathi implements iteration-level scheduling and chunked prefill, serving as HyGen's underlying engine. However, Sarathi does not distinguish between online/offline priorities and cannot perform SLO-aware mixed scheduling. HyGen adds three layers of optimization on top of Sarathi—latency prediction, SLO profiling, and prefix sharing—achieving 3.87–5.84× throughput gains.
- **vs. BlendServe**: BlendServe also explores workload co-location, but targets co-location of different tasks across different models (e.g., simultaneously serving chat and summarization models). HyGen focuses on mixed scheduling of online and offline requests for the same model within a single inference engine, with a more specific optimization objective.
- **vs. FlexGen**: FlexGen supports large-model offline inference through hierarchical memory offloading across CPU/GPU/NVMe. HyGen does not perform offloading but co-locates two request types on GPU; the two systems address entirely different problems—FlexGen solves "model too large to fit on a single GPU," while HyGen solves "low GPU utilization."
- **vs. Splitwise/DynamoLLM**: These systems separate prefill and decode to different machines at the cluster level. HyGen mixes online and offline requests at the single-instance level; the two approaches are complementary—HyGen can serve as the internal scheduler for each instance within a Splitwise cluster.

## Rating
- Novelty: ⭐⭐⭐⭐ Online-offline co-location has been discussed previously, but HyGen is the first to propose a complete systematic solution integrating latency prediction, SLO profiling, and prefix sharing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across multiple models (7B/14B/34B/2.7B), hardware platforms (A100/A40/A5000), datasets (Azure/Mooncake/arXiv/MMLU/CNN), and SLO metrics.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and solution design is logically rigorous, though some algorithmic details require consulting the appendix.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to industrial LLM inference clusters, with open-sourced code and practically significant throughput gains of 3.87–5.84×.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Bridging Human and LLM Judgments: Understanding and Narrowing the Gap](bridging_human_and_llm_judgments_understanding_and_narrowing_the_gap.md)
- [\[AAAI 2026\] Auto-PRE: An Automatic and Cost-Efficient Peer-Review Framework for Language Generation Evaluation](../../AAAI2026/dialogue/auto-pre_an_automatic_and_cost-efficient_peer-review_framework_for_language_gene.md)
- [\[NeurIPS 2025\] AC-LoRA: (Almost) Training-Free Access Control-Aware Multi-Modal LLMs](aclora_almost_trainingfree_access_controlaware_multimodal_ll.md)
- [\[NeurIPS 2025\] MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems](metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems.md)
- [\[NeurIPS 2025\] SciArena: An Open Evaluation Platform for Non-Verifiable Scientific Literature-Grounded Tasks](sciarena_an_open_evaluation_platform_for_non-verifiable_scientific_literature-gr.md)

<!-- RELATED:END -->
