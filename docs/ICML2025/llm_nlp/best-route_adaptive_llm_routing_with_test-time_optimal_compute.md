---
title: >-
  [Paper Note] BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute
description: >-
  [ICML 2025][LLM (Other)][LLM routing] The paper proposes BEST-Route (Best-of-$n$ Enhanced Sampling and Test-time Route Optimization). It introduces the best-of-$n$ sampling strategy into traditional query routing, allowing the router to not only select the model but also adaptively decide the sampling number $n$. By replacing a single invocation of a large model with multiple samplings and selections from a small model, it reduces inference costs by up to 60% with less than 1…
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "LLM routing"
  - "best-of-n sampling"
  - "test-time compute"
  - "cost optimization"
  - "adaptive inference"
date: 2026-05-08
content_hash: 56c4ca3e3cf28174
---

# BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute

**Conference**: ICML 2025  
**arXiv**: [2506.22716](https://arxiv.org/abs/2506.22716)  
**Authors**: Dujian Ding (UBC/Microsoft), Ankur Mallick (Microsoft), Shaokun Zhang (Penn State), Chi Wang (Google DeepMind), Victor Rühle (Microsoft) et al.
**Affiliations**: University of British Columbia, Microsoft, Pennsylvania State University, Google DeepMind, AG2AI
**Area**: LLM/NLP  
**Keywords**: LLM routing, best-of-n sampling, test-time compute, cost optimization, adaptive inference

## TL;DR

The paper proposes BEST-Route (Best-of-$n$ Enhanced Sampling and Test-time Route Optimization). It introduces the best-of-$n$ sampling strategy into traditional query routing, allowing the router to not only select the model but also adaptively decide the sampling number $n$. By replacing a single invocation of a large model with multiple samplings and selections from a small model, it reduces inference costs by up to 60% with less than 1% performance loss.

## Background & Motivation

### Key Challenge

LLM inference faces a fundamental trade-off between quality and cost: large models (e.g., GPT-4) offer high quality but are expensive, while small models (e.g., the Llama family) are cheap but lack sufficient quality in a single response. Three main strategies have been proposed to alleviate this tension:

**Query Routing**: Allocating requests to models of different sizes based on query difficulty (Ong et al., 2024).

**Speculative Decoding**: Token-level collaboration where a small model drafts and a large model verifies (Kim et al., 2023).

**Model Cascades**: Progressively escalating from cheaper models to more expensive ones until a satisfactory response is obtained (Chen et al., 2023).

### Limitations of Prior Work

Previous query routing methods suffer from a core issue: **each model generates only one response**. This implies that:

- The single-response quality of small models is often inadequate to compete with large models.
- Routers are forced to allocate a large portion of queries (except for the simplest ones) to large models.
- The actual cost savings are far below theoretical expectations.

### Key Insight

The authors observe an important but overlooked phenomenon: **small models can significantly improve response quality through best-of-$n$ sampling**. Specifically, sampling a small model $n$ times and selecting the best response can yield a quality close to or even exceeding a single response from a large model, while the total cost (of $n$ small model calls) remains lower than a single large model call. This observation stems from recent research on test-time compute scaling (Snell et al., 2024), which demonstrates that increasing compute during inference can effectively boost model performance.

### Design Motivation

Let the single-call cost of a large model be $C_L$ and that of a small model be $C_S$. If $n \cdot C_S < C_L$, performing $n$ samplings on the small model remains cheaper than a single call to the large model. When the best-of-$n$ selection strategy enables the best response from the small model's $n$ samples to approach the quality of the large model, the router can route more queries to the small model, thereby significantly reducing the overall cost.

## Method

### Overall Architecture

The system architecture of BEST-Route consists of three core components:

1. **Query Difficulty Estimator**: Scores the difficulty of the input query to determine the required model capability.
2. **Model Selector/Router**: Selects whether to call a large model or a small model based on the difficulty score.
3. **Sampling Number Decider**: Adaptively decides the sampling count $n$ when the small model is selected, minimizing the cost while meeting the quality threshold.

The workflow is as follows:
- Input query $q$ $\rightarrow$ estimate difficulty $d(q)$.
- If $d(q)$ exceeds the threshold, directly route to the large model to generate 1 response.
- If $d(q)$ is in the medium range, route to the small model, sample $n(q)$ responses, and select the best one using the selection strategy.
- If $d(q)$ is extremely low, route to the small model and sample only 1 response.

### Key Designs

#### Best-of-$n$ Sampling and Selection

BEST-Route introduces test-time compute into the routing framework through the following core mechanisms:

- **Sampling Phase**: Generates $n$ independent responses $\{r_1, r_2, \dots, r_n\}$ from the small model at a temperature $T > 0$.
- **Selection Phase**: Uses a reward model or a lightweight scorer to score the $n$ responses and selects the highest-scoring one as the final output.
- **Adaptive $n$**: The value of $n$ is not fixed but dynamically adjusted based on query difficulty—medium-difficulty queries require a larger $n$, while simple queries only need $n=1$.

#### Optimization Objective

The routing decision can be formulated as a constrained optimization problem:

$$\min_{f} \mathbb{E}_{q \sim Q}\left[\text{Cost}(f(q))\right] \quad \text{s.t.} \quad \mathbb{E}_{q \sim Q}\left[\text{Quality}(f(q))\right] \geq \tau$$

where $f(q)$ represents the routing decision for query $q$ (which model to choose + how many times to sample) and $\tau$ is the quality threshold. By expanding the decision space—from "selecting a model" to "selecting a model $\times$ sampling count"—BEST-Route finds lower-cost solutions while adhering to quality constraints.

#### Quality Threshold Mechanism

The system allows users to set different quality thresholds $\tau$ to achieve flexible quality-cost trade-offs:
- High $\tau$: More queries are routed to the large model, prioritizing quality.
- Low $\tau$: More queries are handled by the small model using best-of-$n$, prioritizing cost.
- Adaptive $\tau$: Dynamically adjusted based on application scenarios.

### Loss & Training

#### Router Training

The training of the router needs to address two core issues:

1. **Difficulty Annotation Acquisition**: By collecting response quality data for both large and small models across various $n$ values on the training set, the optimal routing label for each query is constructed.
2. **Cost-Aware Learning**: Both quality and cost are taken into account in the training objective, enabling the router to learn the optimal trade-off between the two.

#### Integration with Test-Time Compute Scaling

The core innovation of BEST-Route lies in systematically integrating the concepts of test-time compute scaling (Snell et al., 2024) into the routing framework:

- Traditional routing: The decision space is $\{M_1, M_2, \dots, M_K\}$ (selection among $K$ models).
- BEST-Route: The decision space is $\{(M_i, n_j) | i \in [K], j \in [N_{max}]\}$ (combinations of model $\times$ sampling count).

This expansion enables the router to find Pareto-optimal combinations—for example, selecting a small model with 5 samplings may offer quality comparable to a single large model call while costing only 1/3.

## Key Experimental Results

### Main Results

Based on the core experimental data reported in the paper's abstract (tested on real-world datasets):

| Method | Cost Reduction | Performance Loss | Routing Strategy |
|------|---------|---------|---------|
| Single Large Model (baseline) | 0% | 0% | All queries $\rightarrow$ Large model |
| Traditional Routing (prior work) | 10-25% | 1-3% | Simple $\rightarrow$ Small, Difficult $\rightarrow$ Large |
| **BEST-Route** | **Up to 60%** | **< 1%** | Simple $\rightarrow$ Small ($n=1$), Medium $\rightarrow$ Small (best-of-$n$), Difficult $\rightarrow$ Large |
| Single Small Model (baseline) | Maximum | Significant | All queries $\rightarrow$ Small model |

### Comparison with Prior Routing Methods

| Dimension | Traditional Routing Methods | BEST-Route |
|------|------------|-----------|
| Decision Space | Model selection | Model selection $\times$ Sampling count |
| Small Model Utilization | Low (only handles simple queries) | High (can handle medium-difficulty queries) |
| Test-time Compute | Unused | Core mechanism |
| Cost Savings | Limited | Significant (up to 60%) |
| Quality Guarantee | Depends on routing accuracy | Quality threshold + best-of-$n$ safety net |

### Key Findings

1. **Diminishing Marginal Utility of Best-of-$n$ with High Early Gains**: The quality improvement for small models is most significant from $n=1$ to $n=3$ or $5$, after which marginal gains diminish. This indicates that moderate extra compute (3-5 times the cost of a small model) can substantially bridge the gap with the large model.

2. **Long-Tail Distribution of Query Difficulty**: In real-world workloads, most queries are of medium or lower difficulty, with only a minority of difficult queries genuinely requiring a large model. This provides substantial headroom for BEST-Route's cost optimization.

3. **Expansion of the Cost-Quality Pareto Frontier**: By introducing the joint decision space of (model, $n$), BEST-Route markedly expands the achievable Pareto frontier, yielding lower-cost feasible solutions for equivalent quality targets.

## Highlights & Insights

1. **Simple yet Powerful Concept**: The core idea is highly intuitive—"trying multiple times with a small model" can be more cost-effective than "trying once with a large model." Systematizing this observation into a routing framework and providing an adaptive selection mechanism for $n$ represents the key contribution of this paper.

2. **Bridging the Gap Between Two Fields**: This work organically combines test-time compute scaling (initially focused on computing allocation within a single model) with LLM routing (focused on task allocation across multiple models), broadening the perspectives of both research tracks.

3. **High Practical Value**: A 60% cost reduction has direct financial implications for large-scale LLM deployments. For API providers and enterprise users, this translates to slashing inference budgets in half while maintaining service quality.

4. **Consistency with Scaling Law Observations**: The effectiveness of the small-model best-of-$n$ strategy aligns with findings from inference-time scaling laws—investing more compute at the inference stage can compensate for a smaller model parameter footprint.

## Limitations & Future Work

1. **Dependency on Selection Strategies**: The effectiveness of best-of-$n$ heavily relies on the quality of the selector (reward model or verifier). If the selector is unreliable, multiple samplings may fail to identify the truly optimal response.

2. **Latency Issues**: Although overall costs are reduced, $n$ sequential samplings increase end-to-end latency. For latency-sensitive scenarios, parallel sampling is required, which in turn escalates system complexity.

3. **Static Capacity Assumption**: The paper assumes that model performance distributions remain static during inference, whereas in practical deployments, models may be updated, and input distributions may drift, requiring continuous calibration of the router.

4. **Metric Limitations**: "Quality" is defined differently across applications (factual accuracy vs. creativity vs. code correctness); a unified quality threshold may not suit all scenarios.

5. **Scalability of Model Combinations**: As the number of candidate models grows, the joint decision space of (model, $n$) expands rapidly, increasing the complexity of router training and inference.

## Related Work & Insights

### LLM Query Routing
- **RouteLLM** (Ong et al., 2024): A classifier-based routing method, but each model generates only a single response.
- **AutoMix** (Ding et al., 2024): Automatically mixes responses from different models, but does not incorporate test-time compute.
- **FrugalGPT** (Chen et al., 2023): A cascade-style model invocation strategy that attempts models from cheapest to most expensive.

### Test-Time Compute Scaling
- **Scaling LLM Test-Time Compute** (Snell et al., 2024): Systematically studies the impact of inference computational budget on performance.
- **Self-Consistency** (Wang et al., 2023): An inference strategy employing multiple samples and voting.
- **Re-ranking** (Chuang et al., 2023): Sorts and selects the best response from several candidates.

### Speculative Decoding
- **Speculative Decoding** (Kim et al., 2023): A small model generates draft tokens and a large model verifies them (accept/reject), accelerating autoregressive generation.

## Rating

| Dimension | Score (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | Combining best-of-$n$ with routing is simple yet effective; although the individual concepts are not entirely new, their combination is novel. |
| Value | 5 | A 60% cost reduction offers direct and significant economic value for LLM deployments. |
| Technical Depth | 3 | The method is relatively intuitive; the core contribution lies in the systematic framework rather than a major theoretical breakthrough. |
| Experimental Thoroughness | 3 | The abstract mentions verification on real-world datasets, but detailed experimental setups and ablation information are sparser in the overview. |
| Writing Quality | 4 | Well-motivated problem formulation, systematic framework description, and intuitive illustrations. |

**Overall**: 4/5 — A highly practical piece of work with a clean and elegant core idea. By organically combining test-time compute scaling with LLM routing, it achieves remarkable Pareto improvements in the cost-quality trade-off. Despite its low technical barrier to entry, its engineering significance is profound, offering direct guidance for the large-scale deployment of LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Best-of-∞: Asymptotic Performance of Test-Time LLM Ensembling](../../ICLR2026/llm_nlp/best-of-infinity_asymptotic_performance_of_test-time_llm_ensembling.md)
- [\[NeurIPS 2025\] Wider or Deeper: Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search](../../NeurIPS2025/llm_nlp/wider_or_deeper_scaling_llm_inference-time_compute_with_adaptive_branching_tree_.md)
- [\[ICML 2025\] Breaking Silos: Adaptive Model Fusion Unlocks Better Time Series Forecasting](breaking_silos_adaptive_model_fusion_unlocks_better_time_series_forecasting.md)
- [\[ICLR 2026\] Near-Optimal Online Deployment and Routing for Streaming LLMs](../../ICLR2026/llm_nlp/near-optimal_online_deployment_and_routing_for_streaming_llms.md)
- [\[CVPR 2025\] Test-Time Visual In-Context Tuning](../../CVPR2025/llm_nlp/test-time_visual_in-context_tuning.md)

</div>

<!-- RELATED:END -->
