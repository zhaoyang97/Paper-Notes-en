---
title: >-
  [Paper Note] RCCDA: Adaptive Model Updates in the Presence of Concept Drift under a Constrained Resource Budget
description: >-
  [NeurIPS 2025][Model Compression][concept drift] This paper proposes RCCDA, a lightweight model update policy based on the Lyapunov drift-plus-penalty framework. Under concept drift scenarios where the data distribution shifts over time, RCCDA greedily determines when to retrain the model using only historical inference loss and a tunable threshold, while provably satisfying strict resource budget constraints.
tags:
  - NeurIPS 2025
  - Model Compression
  - concept drift
  - resource-constrained
  - model update policy
  - Lyapunov optimization
  - online learning
date: 2026-05-08
content_hash: 68669d5c4271017f
---

# RCCDA: Adaptive Model Updates in the Presence of Concept Drift under a Constrained Resource Budget

**Conference**: NeurIPS 2025
**arXiv**: [2505.24149](https://arxiv.org/abs/2505.24149)
**Code**: None
**Area**: Model Compression
**Keywords**: concept drift, resource-constrained, model update policy, Lyapunov optimization, online learning

## TL;DR

This paper proposes RCCDA, a lightweight model update policy based on the Lyapunov drift-plus-penalty framework. Under concept drift scenarios where the data distribution shifts over time, RCCDA greedily determines when to retrain the model using only historical inference loss and a tunable threshold, while provably satisfying strict resource budget constraints.

## Background & Motivation

1. **Concept drift is ubiquitous**: ML models deployed in real-world environments face continuously shifting data distributions (concept drift), leading to persistent degradation of inference performance and necessitating timely retraining to adapt to new distributions.
2. **Resource constraints cannot be ignored**: In resource-constrained settings such as mobile and edge devices, the computational, energy, and time overhead of frequent retraining is prohibitive, requiring a balance between performance recovery and resource consumption.
3. **High overhead of existing drift detection methods**: Sliding-window detectors such as ADWIN and statistics-based drift detection methods themselves consume substantial computational resources, making them unsuitable for resource-scarce deployment environments.
4. **Lack of resource guarantees**: Most existing adaptive learning methods offer only empirical validation and cannot provide strict upper bounds on resource usage; artificially capping the number of updates leads to suboptimal performance.
5. **Gap between theory and practice**: Existing convergence analyses rely on unconstrained updates or strong assumptions about drift patterns, and cannot be directly applied to online decision-making under resource budgets.
6. **An open problem**: Optimally handling concept drift while satisfying hard resource constraints is a long-standing unsolved problem with no systematic solution prior to this work.

## Method

### Overall Architecture: Online Update Policy under the Lyapunov Drift-Plus-Penalty Framework

- **Function**: The problem of "when to update the model" is formulated as a resource-constrained online optimization problem, with the objective of minimizing the time-average loss subject to the constraint that the time-average update cost does not exceed budget $\bar{\lambda}$.
- **Design Motivation**: Solving this directly requires global knowledge of future data distribution evolution, which is infeasible. The Lyapunov framework decomposes the long-horizon optimization problem into greedy per-step decisions, each requiring only historical information and the current virtual queue state.
- **Mechanism**:
  1. Derive a convergence upper bound on the model loss evolution under concept drift (Theorem 5.1), obtaining the per-step penalty term.
  2. Introduce a virtual queue $Q(t)$ to track cumulative resource consumption deviation: $Q(t+1) = \max\{0, Q(t) + \lambda(t)\pi(t) - \bar{\lambda}\}$.
  3. Greedily minimize the Lyapunov drift-plus-penalty expression at each step: $\mathbb{E}[\Delta Q(t)] + V\Psi(t)$, where $V$ is a performance-resource trade-off parameter.
  4. Threshold decision rule: execute an update ($\pi(t)=1$) when $V\hat{\mathcal{G}}_t \geq \mathcal{C}(t)$; otherwise keep the model unchanged.

### Key Design 1: Causally Realizable Estimator

- **Function**: A causal estimator $\hat{\mathcal{G}}(\mathcal{H}_t)$ approximates the theoretically non-causal performance gain $\mathcal{G}(t)$, enabling the policy to be executed online.
- **Design Motivation**: The theoretically optimal threshold depends on the future data distribution $\mathcal{D}_{t+1}$ and the recursively defined optimal future action $\pi(t+1)$, both of which are uncomputable in practice.
- **Mechanism**: A PD-controller-style estimator is adopted: $\hat{\mathcal{G}}(\mathcal{H}_t) = K_p(f_t - f_{\min}) + K_d(f_t - f_{t-1})$, where $K_p, K_d$ are tunable constants and $f_{\min}$ is the historical minimum inference loss. The proportional term captures cumulative performance degradation, while the derivative term captures abrupt loss change trends.

### Key Design 2: Theoretical Guarantees on Convergence and Stability

- **Function**: Prove the convergence upper bound and resource constraint violation bound of RCCDA.
- **Design Motivation**: These results provide theoretical foundations for the policy design, ensuring that RCCDA is not only empirically effective but also provably compliant with resource budgets.
- **Mechanism**:
  - Theorem 5.1: Under smoothness and bounded variance assumptions, the time-average gradient norm is bounded by $\mathcal{O}(\sqrt{\delta}) + \mathcal{O}(\sigma^2)$, where $\delta$ is the drift upper bound.
  - Theorem 5.3: The resource constraint violation vanishes at rate $\mathcal{O}(1/\sqrt{T})$ as $T \to \infty$, guaranteeing long-term resource compliance.

## Key Experimental Results

### Experimental Setup
- **Datasets**: PACS, DigitsDG, OfficeHome, MEMD-ABSA (4 domain generalization datasets)
- **Drift patterns**: Burst (sudden high-intensity drift), Step (gradual stepwise drift), Wave (periodic low-intensity drift), Spikes (random timing and random rate)
- **Baselines**: Uniform Random, Periodic, Budget-Increase, Budget-Threshold
- **Constraint setting**: Average update rate constraint $\bar{\lambda}/\lambda = 0.1$

### Main Results

| Policy | PACS-Burst | PACS-Step | PACS-Wave | PACS-Spikes | Digits-Burst | Digits-Spikes |
|--------|-----------|----------|----------|------------|-------------|--------------|
| **RCCDA** | **72.8±4.5** | **72.0±8.1** | **67.0±4.7** | **73.0±3.6** | **77.6±6.3** | **74.3±7.3** |
| Uniform | 64.0±2.2 | 65.3±6.3 | 61.8±7.3 | 60.8±7.7 | 71.4±5.0 | 67.0±3.6 |
| Periodic | 65.1±2.4 | 65.5±7.1 | 61.0±7.8 | 55.3±4.0 | 69.8±3.6 | 68.1±6.8 |
| Budget-Threshold | 67.4±14.9 | 71.7±8.3 | 54.5±7.0 | 58.4±12.3 | 71.9±3.7 | 67.3±6.2 |

| Metric | RCCDA vs. Best Baseline |
|--------|------------------------|
| Burst accuracy gain | +5.4pp (PACS), +5.7pp (DigitsDG) |
| Resource constraint | Update rate converges to target 0.1 |
| Recovery speed | Fastest accuracy recovery after drift |

### Key Findings
1. RCCDA consistently outperforms all baselines across all drift patterns, with the largest advantage in high-drift scenarios such as Burst and Spikes (fast response combined with rational resource allocation).
2. Budget-Threshold exhibits some reactivity in burst scenarios but fails to utilize the update budget under low-drift conditions, resulting in resource waste.
3. The update rate of RCCDA converges to the constraint value 0.1 over time, empirically validating Theorem 5.3 — the policy maximally exploits the available resource budget.

## Highlights & Insights

- First systematic solution to the problem of model updates under strict resource constraints in the presence of concept drift, filling a theoretical gap.
- The policy is extremely lightweight, requiring only historical inference loss with no additional drift detection computation.
- Provides dual theoretical guarantees on both convergence and resource compliance.
- The framework is general; the estimator $\hat{\mathcal{G}}$ can be customized for different problem settings.

## Limitations & Future Work

- Approximation error in the estimator $\hat{\mathcal{G}}$ may affect threshold policy performance; a rigorous characterization of estimation error is not provided.
- Experiments primarily simulate concept drift using domain generalization datasets, which may differ from continuous drift in real-world online systems.
- The bounded drift assumption (Assumption 3) may limit applicability to scenarios with extreme distributional shifts.
- Each update uses a fixed number of SGD steps; adaptive update granularity is not explored.

## Related Work & Insights

| Aspect | Ours (RCCDA) | Drift Detection Methods (e.g., ADWIN) |
|--------|-------------|--------------------------------------|
| Detection overhead | No drift detection needed; uses only inference loss | Requires additional statistical computation |
| Resource guarantee | Provably satisfies long-term resource budget | Does not consider resource constraints |
| Theoretical basis | Lyapunov framework + convergence analysis | Statistical detection thresholds |

| Aspect | Ours (RCCDA) | Temporal Domain Generalization (TDG) |
|--------|-------------|--------------------------------------|
| Core objective | Optimality on current distribution + resource constraints | Generalization to future distributions |
| Resource consideration | Explicit budget constraints | Does not account for computational cost |
| Applicable scenarios | Non-periodic drift | Exploits temporal regularity |

## Rating

- ⭐⭐⭐⭐ **Novelty**: First application of Lyapunov optimization to resource-constrained model updates under concept drift.
- ⭐⭐⭐⭐ **Theoretical Depth**: Dual guarantees via convergence and stability analysis with rigorous derivations.
- ⭐⭐⭐ **Experimental Thoroughness**: 4 datasets × 4 drift patterns, but all use simulated drift; validation on real online systems is lacking.
- ⭐⭐⭐⭐ **Value**: Lightweight, easy to implement, resource-controllable, well-suited for edge ML deployment.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] A Granular Study of Safety Pretraining under Model Abliteration](a_granular_study_of_safety_pretraining_under_model_abliteration.md)
- [\[NeurIPS 2025\] KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments](keydiff_key_similarity-based_kv_cache_eviction_for_long-context_llm_inference_in.md)
- [\[NeurIPS 2025\] Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference](ada-kv_optimizing_kv_cache_eviction_by_adaptive_budget_allocation_for_efficient_.md)
- [\[NeurIPS 2025\] REP: Resource-Efficient Prompting for Rehearsal-Free Continual Learning](rep_resource-efficient_prompting_for_rehearsal-free_continual_learning.md)
- [\[NeurIPS 2025\] A*-Thought: Efficient Reasoning via Bidirectional Compression for Low-Resource Settings](a-thought_efficient_reasoning_via_bidirectional_compression_for_low-resource_set.md)

<!-- RELATED:END -->
