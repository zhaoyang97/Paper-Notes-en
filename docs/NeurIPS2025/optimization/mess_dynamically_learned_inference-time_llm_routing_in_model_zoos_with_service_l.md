---
title: >-
  [Paper Note] MESS+: Dynamically Learned Inference-Time LLM Routing in Model Zoos with Service Level Guarantees
description: >-
  [NeurIPS 2025][Optimization][Cost Optimization] MESS+ is the first framework to formalize LLM request routing as a constrained stochastic optimization problem with SLA guarantees. By combining an online-learned request s…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Cost Optimization"
  - "LLM Routing"
  - "Virtual Queue"
  - "SLA Guarantee"
  - "Online Learning"
date: 2026-05-08
content_hash: 4a10733a50b7bd1c
---

# MESS+: Dynamically Learned Inference-Time LLM Routing in Model Zoos with Service Level Guarantees

**Conference**: NeurIPS 2025
**arXiv**: [2505.19947](https://arxiv.org/abs/2505.19947)  
**Code**: [GitHub](https://github.com/laminair/mess-plus)  
**Area**: Optimization
**Keywords**: Cost Optimization, LLM Routing, Virtual Queue, SLA Guarantee, Online Learning

## TL;DR

MESS+ is the first framework to formalize LLM request routing as a constrained stochastic optimization problem with SLA guarantees. By combining an online-learned request satisfaction predictor with a virtual queue mechanism, it dynamically selects models per request. Across 3 reasoning and 5 question-answering benchmarks, MESS+ achieves an average 2× cost reduction while satisfying SLA constraints, with theoretical guarantees on both cost optimality and constraint satisfaction.

## Background & Motivation

**Background**: The open-source LLM ecosystem (Llama, Qwen, Granite) provides multiple models at varying scales, forming a "model zoo." Each model family contains at least three variants (e.g., 1B/8B/70B) with vastly different performance and cost profiles. Users face difficulty selecting appropriate models and typically default to the largest available—wasting resources and incurring uncontrolled costs.

**Limitations of Prior Work**: Existing routing approaches each have notable shortcomings—RouteLLM supports routing between only two models; Zooter and RouterDC lack formalized cost guarantees; and none provide theoretical guarantees for SLA compliance. What users require is a hard commitment such as "at least X% of requests are satisfactorily answered."

**Key Challenge**: A three-way conflict of requirements: (1) users demand high-quality responses without technical expertise; (2) service providers aim to minimize operational costs; (3) enterprise clients require SLA guarantees. These three objectives must be simultaneously optimized within a unified framework.

**Goal**: How to design an LLM routing algorithm that strictly guarantees SLA compliance (i.e., a minimum request satisfaction rate over time) while minimizing operational cost?

**Key Insight**: Drawing on the Lyapunov drift-plus-penalty framework, SLA constraints are encoded as virtual queues, and request satisfaction prediction is integrated into per-request optimization as an online learning problem.

**Core Idea**: Virtual queues track cumulative SLA violations; an online-learned satisfaction predictor estimates per-model performance; and a lightweight optimization problem is solved for each request to achieve cost-optimal model selection.

## Method

### Overall Architecture

The problem is formalized as a constrained stochastic optimization: the objective is to minimize average operational cost $\frac{1}{T}\sum_t\sum_m \mathbb{E}[y_{m,t}E_{m,t}]$, subject to a request satisfaction rate no lower than a target $\alpha$: $\frac{1}{T}\sum_t\sum_m \mathbb{E}[y_{m,t}s_{m,t}] \geq \alpha$, where $y_{m,t} \in \{0,1\}$ is the model selection variable, $E_{m,t}$ is the cost, and $s_{m,t} \in \{0,1\}$ is the unknown satisfaction label. The system operates online, making routing decisions upon each request arrival.

### Key Designs

1. **Virtual Queue Mechanism**:

    - Function: Transforms long-term SLA constraints into step-wise actionable signals.
    - Mechanism: Maintains a virtual queue $Q_{t+1} = \max\{0, Q_t + \alpha - s_{m^*,t}\}$, representing the cumulative SLA violation. When $Q_t$ is large (i.e., SLA is frequently violated), the optimization automatically favors larger models with higher satisfaction rates; when $Q_t$ is small (i.e., SLA headroom exists), cheaper smaller models are permitted.
    - Design Motivation: Inspired by the Lyapunov drift-plus-penalty framework, with a critical extension—integration of an online-learned satisfaction predictor, which is absent from the original framework.

2. **Online Request Satisfaction Predictor**:

    - Function: Prior to dispatching a request to any model, predicts whether each candidate model will satisfy the request.
    - Mechanism: Uses ModernBERT as a frozen backbone with a trainable multi-label classifier (trained via SGD). Outputs a satisfaction probability $\hat{s}_{m,t} \in [0,1]$ per model. A probabilistic exploration strategy (with probability $p_t = \min(1, c/\sqrt[4]{t})$ decaying over time) balances exploration and exploitation—during exploration, all models are queried to obtain ground-truth labels for updating the predictor. Training uses regularized cross-entropy loss.
    - Design Motivation: The decay rate $p_t \propto 1/\sqrt[4]{t}$ is critical—too fast leads to an inaccurate predictor; too slow incurs excessive exploration costs. The fourth-root decay achieves an optimal balance between prediction accuracy and exploration overhead, with theoretical guarantees.

3. **Per-Request Optimization Problem**:

    - Function: Makes a cost-optimal routing decision for each incoming request.
    - Mechanism: $m^* = \arg\min_m V \cdot E_{m,t} + Q_t(\alpha - \hat{s}_{m,t})$. The first term is the weighted cost ($V$ controls cost emphasis); the second term penalizes SLA violations (queue length multiplied by predicted satisfaction gap). When the queue is long, the penalty term dominates and favors high-satisfaction large models; when the queue is short, the cost term dominates and favors cheaper small models.
    - Design Motivation: The parameter $V$ controls the tradeoff between convergence speed and cost—larger $V$ yields lower cost but slower SLA convergence; smaller $V$ achieves SLA compliance faster but at higher cost. Theoretically, constraint violation is $O(V/T + 1/\sqrt{T})$ and the cost gap is $O(M/\sqrt[4]{T} + 1/V)$.

### Loss & Training

- Satisfaction predictor: Regularized cross-entropy loss with SGD-based online learning.
- Hyperparameter settings: $V=0.0001$, $c=0.1$, held fixed across all benchmarks without per-task tuning.
- Cost metric: Energy consumption per request (MJ), obtained via offline profiling.

## Key Experimental Results

### Main Results

Representative results across 8 benchmarks (Llama 1B/8B/70B model zoo):

| Method | ARC-C (α=50%) Cost (MJ) | Satisfaction % | BoolQ (α=80%) Cost (MJ) | Satisfaction % |
|--------|-------------------------|----------------|--------------------------|----------------|
| L70B only | 2.35 | 60.8% | 3.40 | 88.8% |
| L8B only | 0.46 | 54.4% | 0.43 | 84.2% |
| RouteLLM | 1.24 | 51.2% | 2.96 | 86.8% |
| RouterDC | 2.09 | 60.9% | 2.14 | 87.1% |
| **MESS+** | **0.83** | **53.6%** | **0.90** | **82.2%** |

Average across 8 benchmarks (α=66%):

| Method | Avg. Cost (MJ) | Avg. Satisfaction % | Model Call Ratio (70B/8B/1B) |
|--------|----------------|---------------------|------------------------------|
| L70B only | 2.79 | 77.3% | 100/0/0 |
| RouteLLM | 2.11 | 74.2% | 75/0/25 |
| RouterDC | 2.09 | 76.1% | 76/22/2 |
| **MESS+** | **1.07** | **67.8%** | 35/38/27 |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| Effect of exploration constant $c$ | $c=0.1$ is optimal; larger $c$ incurs excessive exploration cost, smaller $c$ yields an inaccurate predictor |
| Tradeoff under parameter $V$ | Larger $V$ lowers cost but slows SLA convergence; smaller $V$ achieves SLA compliance quickly but at higher cost |
| Predictor accuracy | Prediction accuracy converges rapidly after the initial exploration phase, supporting theoretical assumptions |
| Non-i.i.d. setting | Although theory assumes i.i.d. inputs, experiments show robustness under non-i.i.d. (topic-sorted) settings |

### Key Findings

- **MESS+ achieves approximately 2× cost reduction**: Under SLA-compliant conditions, MESS+ incurs an average cost of only 1.07 MJ, compared to approximately 2.09–2.11 MJ for the next-best methods (RouterDC and RouteLLM)—a near-halving of cost.
- **Full utilization of model zoo diversity**: MESS+ distributes calls at 35%/38%/27% (large/medium/small), achieving true demand-driven allocation. In contrast, RouteLLM routes to only two models (75%/0%/25%), and RouterDC nearly always selects the largest model (76%/22%/2%).
- **SLA guarantees are met within a finite number of requests**: Although the theoretical guarantee requires $T \to \infty$, in practice SLA compliance stabilizes after approximately 1,000 requests, demonstrating strong practical utility.
- **Fixed hyperparameters generalize across tasks**: $V=0.0001$ and $c=0.1$ require no per-benchmark tuning across all 8 tasks, enabling deployment-friendly operation.

## Highlights & Insights

- **Theoretical guarantees as the core contribution**: Theorem 1 proves constraint violation of $O(V/T + 1/\sqrt{T})$; Theorem 2 proves a cost gap of $O(M/\sqrt[4]{T} + 1/V + M \cdot F_{\min})$. This is the first LLM routing method with rigorous theoretical guarantees, enabling service providers to contractually commit to SLAs with mathematical assurance of compliance.
- **Integration of virtual queues and online learning**: The framework unifies Lyapunov optimization with online predictor training and analyzes how predictor error propagates into overall optimization—representing a substantive theoretical extension of the classical framework.
- **Energy consumption as cost metric**: Beyond API call fees, the framework accounts for energy consumption (MJ), aligning with the energy monitoring requirements of EU AI Act Article 95 and reinforcing its practical deployment relevance.

## Limitations & Future Work

- **Limited model zoo scale**: Experiments use only three Llama models; performance in larger-scale or heterogeneous zoos (mixing different model families) remains unvalidated.
- **Satisfaction label acquisition**: The framework assumes users provide feedback immediately; in practice, delayed and sparse feedback is common.
- **Simplified cost model**: Costs are assumed known upon request receipt (estimated via token count); in reality, cost uncertainty arises from variable generation lengths across models.
- **I.i.d. assumption**: Theoretical analysis relies on i.i.d. inputs; although experiments demonstrate robustness under non-i.i.d. conditions, formal non-i.i.d. guarantees are absent.
- **Uniform SLA targets**: All requests share the same SLA objective; differentiated service for requests with varying priorities is not supported.
- Future directions include extending to heterogeneous model zoos, supporting delayed feedback, incorporating request prioritization, and modeling generation-length cost uncertainty.

## Related Work & Insights

- **vs. RouteLLM**: RouteLLM trains a router on human preference data and supports only two models, with no SLA guarantees. MESS+ supports an arbitrary number of models, provides rigorous theoretical guarantees, and requires no human preference data (relying instead on binary satisfaction signals).
- **vs. RouterDC**: RouterDC employs contrastive learning to train a multi-model router but overwhelmingly selects the largest model (76% call share), with no cost optimization or SLA guarantees. MESS+ distributes load across all models (35%/38%/27%).
- **vs. Lyapunov optimization framework**: The classical framework assumes constraint signals are known. MESS+ extends this to settings where constraint signals must be predicted via online learning—a substantive theoretical contribution.

## Rating

- Novelty: ⭐⭐⭐⭐ — First framework to formalize LLM routing as a constrained optimization problem with SLA guarantees; theoretically rigorous.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers 8 benchmarks with multiple baselines and thorough hyperparameter analysis, though the model zoo scale is limited.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clearly framed through a three-party needs analysis; theoretical derivations are rigorous.
- Value: ⭐⭐⭐⭐⭐ — Addresses a core practical challenge in LLM deployment (cost vs. quality), with strong theoretical and empirical contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Preference Learning with Response Time: Robust Losses and Guarantees](preference_learning_with_response_time_robust_losses_and_guarantees.md)
- [\[NeurIPS 2025\] Least Squares Variational Inference](least_squares_variational_inference.md)
- [\[NeurIPS 2025\] Brain-like Variational Inference](brain-like_variational_inference.md)
- [\[NeurIPS 2025\] Natural Gradient VI: Guarantees for Non-Conjugate Models](natural_gradient_vi_guarantees_for_non-conjugate_models.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)

</div>

<!-- RELATED:END -->
