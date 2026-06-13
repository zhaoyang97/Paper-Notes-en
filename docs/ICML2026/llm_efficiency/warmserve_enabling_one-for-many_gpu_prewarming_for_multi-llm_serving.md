---
title: >-
  [Paper Note] WarmServe: A Multi-Model Loading GPU Warm-up Mechanism
description: >-
  [ICML 2026][LLM Efficiency][GPU Warm-up] WarmServe proactively pre-loads multiple model parameters onto GPUs by analyzing long-term periodic patterns of LLM workloads. Combined with optimized placement algorithms and dyn…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "GPU Warm-up"
  - "Multi-LLM Serving"
  - "Workload Prediction"
  - "Cold Start"
  - "Resource Efficiency"
date: 2026-05-08
content_hash: b465ae05990b8466
---

# WarmServe: A Multi-Model Loading GPU Warm-up Mechanism

**Conference**: ICML 2026  
**arXiv**: [2512.09472](https://arxiv.org/abs/2512.09472)  
**Code**: https://github.com/LLMServe/WarmServe  
**Area**: LLM Efficiency / Multi-model Serving  
**Keywords**: GPU Warm-up, Multi-LLM Serving, Workload Prediction, Cold Start, Resource Efficiency

## TL;DR
WarmServe proactively pre-loads multiple model parameters onto GPUs by analyzing long-term periodic patterns of LLM workloads. Combined with optimized placement algorithms and dynamic KV cache reservation strategies, the system enables rapid instantiation of new instances during request bursts—reducing tail TTFT by 50.8× compared to existing systems.

## Background & Motivation

**Background**: Multi-LLM serving systems must concurrently deploy multiple models in shared GPU clusters to improve resource utilization. Two mainstream approaches exist—(1) Auto-scaling: Dynamically creates instances based on current load but suffers from high cold-start latency; (2) GPU Sharing: Colocates multiple models on the same GPU but is severely limited by KV cache capacity.

**Limitations of Prior Work**: Auto-scaling requires on-the-fly loading of model parameters during request bursts, leading to severe TTFT. While GPU sharing avoids initialization delay, each model receives very little KV cache.

**Key Challenge**: Existing systems lack awareness of future workload characteristics—auto-scaling can only respond passively, and the placement strategies for GPU sharing must remain stable over time.

**Key Observations**: Although short-term request arrivals are stochastic, the long-term statistical properties of LLM services in actual production environments exhibit strong periodicity—peak loads within a 5-minute window can be predicted with an average relative error of 7.3%.

**Key Insight**: Fully exploit this predictability by employing a proactive warm-up strategy—actively loading standby model copies onto idle GPUs before a predicted future load surge.

**Core Idea**: Introducing a "load multiple models once" mechanism—simultaneously loading multiple model parameters into a single GPU's memory. When a model encounters a request burst, it immediately utilizes the warmed-up parameters to start an active instance and then quickly evicts other model parameters. Evicting weights is significantly faster than loading them on demand.

## Method

### Overall Architecture
GPU cluster worker nodes are categorized into three types—idle, universal, and dedicated. The system warms up multiple LLMs on idle nodes to convert them into universal nodes. When a warmed-up model receives a burst of requests, the node is upgraded to a dedicated node while other models are evicted. Warm-up is also permitted within unused KV cache space on dedicated nodes.

### Key Designs

1. **Workload Prediction (Corrective Seasonal Predictor, CSP)**:

    - **Function**: Predicts the average and peak loads of each model in the next time window based on historical data.
    - **Mechanism**: Combines a seasonal pattern $P_{k,i} = \frac{1}{D}\sum_{d=1}^{D}L_{k-d,i}$ with a correction term $\Delta_{k,i} = \frac{\sum_{w=1}^{\min(i,N)}(L_{k,i-w}-P_{k,i-w})\cdot 2^{w-1}}{2^{\min(i,N)}-1}$, resulting in the final prediction $\hat{L}_{k,i} = P_{k,i} + \Delta_{k,i}$. The correction term assigns higher weight to recent errors.
    - **Design Motivation**: While LLM workloads are unpredictable in the short term, they exhibit long-term periodicity; adding a correction term allows for rapid adaptation to current trends, achieving 92.7% prediction accuracy.

2. **Model Placement Algorithm**:

    - **Function**: Decides which model copies need to be warmed up and where to place them on GPUs to minimize cross-model warm-up interference.
    - **Mechanism**: Calculates a priority score for each copy to be warmed up (based on the gap between expected load and current instances, cold start latency, etc.) and sorts them in descending order. For each copy, it greedily selects the optimal GPU group—prioritizing GPU groups where high-score copies can be protected (not evicted by low-score copies).
    - **Design Motivation**: LLMs are distributed across multiple GPUs (tensor parallelism), and releasing a single GPU can trigger a chain reaction of evictions across multiple models (cross-model interference). The placement algorithm ensures that important models are not disrupted by minor ones through priority isolation.

3. **Proactive Warm-up + KV Cache Reservation**:

    - **Function**: Utilizes unused KV cache space to pre-load new model parameters before load decreases and model instances are released.
    - **Mechanism**: When the auto-scaler detects a load drop and prepares to shut down instances, these instances usually have ample unused KV cache. The system calculates the required KV cache to reserve as $R = \max(C \cdot Q/B, T + C/B)$; anything exceeding this can be used for warm-up. If space is insufficient, warmed-up weights are dynamically evicted.
    - **Design Motivation**: LLM checkpoints are massive (128GB+), and traditional warm-up often fails within short windows. By "stealthily" pre-loading on GPUs about to be released, I/O is spread across idle periods while the GPU is still operational.

## Key Experimental Results

### Main Results

| System | P95 TTFT(s) | P99 TTFT(s) | Gain | Max RPS |
|------|-----------|-----------|--------|--------|
| SLLM-GPU | 1.23 | 3.45 | - | 10 |
| MuxServe | 0.89 | 2.34 | - | 6 |
| WarmServe (w/o Proactive) | 0.18 | 0.31 | 6.8×-11.1× vs SLLM | 20 |
| WarmServe (Full) | 0.17 | 0.29 | 7.2×-11.9× vs SLLM / 5.2× vs MuxServe | 25 |

Under a setting of 15 RPS and $\alpha$=0.5, WarmServe achieves a 1.53-50.79× reduction in P99 TTFT compared to SLLM-GPU.

### Ablation Study

| Configuration | Ratio of TTFT < 100ms | Description |
|------|---------------|------|
| Full Model | 100% | baseline |
| w/o Model Warm-up | 15% | Performance collapse |
| w/o Placement Algorithm | 29% | Interference surge |
| w/o Proactive Warm-up | 88% | Still improved but 32.87× worse than Full |
| 3-min Warm-up Window | 46% | Window too small, prediction unstable |
| 40-min Warm-up Window | 30% | Window too large, fails to capture short-term changes |

### Key Findings
- Model warm-up provides dozens of times improvement in TTFT.
- The proactive warm-up strategy brings the most significant improvement (up to 32×).
- The placement algorithm prevents model interference avalanches under high load.
- A 5-minute warm-up window is optimal.
- Workload Prediction: Average load prediction accuracy of 94.7% and peak accuracy of 92.7% under a 5-minute window.

## Highlights & Insights
- **Discovery of long-term periodicity in LLM workloads**: Challenges the perception that LLM requests are completely unpredictable.
- **Innovation of loading multiple models once**: Perfectly balances resource efficiency and performance advantages.
- **Dual-purpose KV cache**: Expands the role of KV cache from pure activation storage to temporary storage for warm-ups.
- **Priority isolation in greedy placement**: Simple and efficient, eliminating the need to solve complex integer programs at runtime.

## Limitations & Future Work
- Boundary of workload prediction applicability—may fail for entirely new models or special business events.
- Lack of multi-datacenter/multi-tenancy scenarios.
- Insufficient handling of model size disparities—limited effectiveness when colocating 7B + 70B models.
- Future improvements: Integration of online learning, multi-model ensemble forecasting, and detailed analysis of warm-up failure rates and energy consumption impacts.

## Related Work & Insights
- **vs ServerlessLLM/MuxServe**: WarmServe finds a new design space between the two by using a warm-up intermediate layer.
- **vs Serverless Warm-up**: WarmServe is specialized for three major LLM challenges (cross-multi-GPU dependencies, extreme model sizes, and KV cache).
- **vs KV Cache Optimization**: Utilizing unused cache space as temporary warm-up storage reflects the philosophy of "maximizing system resource utilization."

## Rating
- Novelty: ⭐⭐⭐⭐⭐  Identifies long-term predictability of LLM workloads and introduces an innovative multi-model loading mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐  Includes single-machine tests, large-scale simulations, ablation studies, and prediction accuracy verification.
- Writing Quality: ⭐⭐⭐⭐⭐  Clear logic with a natural progression of motivations.
- Value: ⭐⭐⭐⭐⭐  A 50× improvement in TTFT holds significant practical deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MTRouter: Cost-Aware Multi-Turn LLM Routing with History-Model Joint Embeddings](../../ACL2026/llm_efficiency/mtrouter_cost-aware_multi-turn_llm_routing_with_history-model_joint_embeddings.md)
- [\[ICML 2026\] Efficient Training-Free Multi-Token Prediction via Embedding-Space Probing](efficient_training-free_multi-token_prediction_via_embedding-space_probing.md)
- [\[ICLR 2026\] Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](../../ICLR2026/llm_efficiency/fast_catch-up_late_switching_optimal_batch_size_scheduling_via_functional_scalin.md)
- [\[AAAI 2026\] Resource Efficient Sleep Staging via Multi-Level Masking and Prompt Learning](../../AAAI2026/llm_efficiency/resource_efficient_sleep_staging_via_multi-level_masking_and_prompt_learning.md)
- [\[ACL 2026\] Multi-Drafter Speculative Decoding with Alignment Feedback](../../ACL2026/llm_efficiency/multi-drafter_speculative_decoding_with_alignment_feedback.md)

</div>

<!-- RELATED:END -->
