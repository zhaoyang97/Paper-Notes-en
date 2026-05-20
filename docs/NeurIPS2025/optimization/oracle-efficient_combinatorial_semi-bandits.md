---
title: >-
  [Paper Note] Oracle-Efficient Combinatorial Semi-Bandits
description: >-
  [NeurIPS 2025][Optimization][combinatorial semi-bandits] Two oracle-efficient frameworks (adaptive and scheduled) are proposed for combinatorial semi-bandits…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "combinatorial semi-bandits"
  - "oracle efficiency"
  - "regret bounds"
  - "online learning"
  - "covariance-adaptive"
date: 2026-05-08
content_hash: cb06d2885c07ac7b
---

# Oracle-Efficient Combinatorial Semi-Bandits

**Conference**: NeurIPS 2025
**arXiv**: [2510.21431](https://arxiv.org/abs/2510.21431)  
**Code**: [GitHub](https://github.com/junghunkim7786/OracleEfficientCombinatorialBandits)  
**Area**: Optimization
**Keywords**: combinatorial semi-bandits, oracle efficiency, regret bounds, online learning, covariance-adaptive

## TL;DR

Two oracle-efficient frameworks (adaptive and scheduled) are proposed for combinatorial semi-bandits, reducing oracle calls from $\Theta(T)$ to $O(\log\log T)$ while maintaining near-optimal regret bounds.

## Background & Motivation

The Combinatorial Semi-Bandit (CMAB) is a generalization of the classical multi-armed bandit problem, where an agent selects a subset of base arms each round and observes feedback from each selected arm. This setting has broad applications in product recommendation, ad allocation, and network routing.

**Core Problem**: Existing algorithms such as CUCB achieve near-optimal regret $\widetilde{O}(\sqrt{mdT})$ but require one combinatorial optimization oracle call per round. Since combinatorial optimization is generally NP-hard, this results in $\Theta(T)$ oracle calls and substantial computational overhead.

**Limitations of Prior Work**: Cuvelier et al. proposed approximation-based methods to reduce computational complexity, but introduced a trade-off between regret and computation — achieving optimal regret requires increasingly precise approximations, potentially leading to unbounded computation time. Combes et al. heuristically use $O(\log T)$ oracle calls but provide no theoretical regret guarantees.

**Key Insight**: Drawing on ideas from batch learning, the paper designs epoch-based oracle query strategies that concentrate oracle calls at critical time points and reuse decisions from the previous round for the majority of steps. The core idea is to optimize oracle efficiency along two dimensions: adaptive complexity (number of sequential query rounds) and query complexity (total number of queries).

## Method

### Overall Architecture

Two families of frameworks are proposed to reduce oracle calls:

1. **Adaptive Rare Oracle Query (AROQ)**: Dynamically triggers oracle updates based on arm selection counts.
2. **Scheduled Rare Oracle Query (SROQ)**: Executes oracle queries in batches at predetermined time points, supporting parallelization.

Both frameworks are applied across three reward models: worst-case linear rewards, covariance-dependent linear rewards, and general (nonlinear) rewards.

### Key Designs

1. **Adaptive Epoch Update Condition (AROQ-CMAB)**: Employs a UCB strategy but invokes the oracle update only when a threshold condition is satisfied. Specifically, for each base arm $i$, an epoch counter $\tau_i$ is maintained, and an update is triggered when the number of times arm $i$ is selected within the current epoch satisfies

$$|{\mathcal{T}}_i(\tau_i)| \geq 1 + \sqrt{Tm \cdot |{\mathcal{T}}_i(\tau_i - 1)| / d}$$

The design motivation is that the regret within each epoch is bounded by $\sqrt{1/|{\mathcal{T}}_i(\tau_i-1)|}$, which when multiplied by the epoch length keeps the overall regret controlled within $\sqrt{Tm/d}$. When the update condition is not met, the previous action $a_t = a_{t-1}$ is reused directly, avoiding an oracle call.

2. **Scheduled Batch Queries (SROQ-CMAB)**: Employs an elimination-based strategy. Oracle queries are executed at predetermined time points $\mathcal{T} = \{t_1, \ldots, t_M\}$, where $t_\tau = \eta\sqrt{t_{\tau-1}}$ and $M = \Theta(\log\log T)$. A key innovation is the construction of representative actions $a_\tau^{(i)}$ for each base arm $i$, using UCB/LCB indices for elimination:

$$r_\tau^{UCB}(a) = \sum_{i \in a}\left(\hat{\mu}_{\tau,i} + \sqrt{\frac{C\log T}{n_{\tau,i}}}\right)$$

If the UCB value of the best representative action for arm $i$ falls below the LCB value of the global optimum, that arm is eliminated. This design progressively reduces the candidate set $\mathcal{N}_\tau$, thereby reducing the search space for subsequent oracle queries.

3. **Covariance-Adaptive Extensions (AROQ-C-CMAB / SROQ-C-CMAB)**: A covariance estimator $\hat{\Sigma}_t$ is introduced, with UCB indices based on confidence ellipsoids:

$$r_t^{UCB}(a) = \langle a, \hat{\mu}_t \rangle + f_t \|D_{n_t}^{-1} a\|_{\bar{G}_t}$$

where $\bar{G}_t$ is a Gram matrix constructed from an upper bound on the covariance. An initial phase of uniform exploration stabilizes the covariance estimator, after which the main phase proceeds using confidence ellipsoid-based indices. Under independent rewards ($\Sigma = I$), the regret bound improves from $\widetilde{O}(\sqrt{mdT})$ to $\widetilde{O}(\sqrt{dT})$, a $\sqrt{m}$-factor tighter.

### Loss & Training

The paper operates within an online learning framework. The core optimization objective is to minimize cumulative regret:

$$\mathcal{R}(T) = \mathbb{E}\left[\sum_{t=1}^{T}(\bar{r}(a^*) - \bar{r}(a_t))\right]$$

All algorithms balance exploration and exploitation through carefully designed epoch update conditions or time schedules.

## Key Experimental Results

### Main Results

Synthetic data experiments under the linear reward setting with $d=20$, $m=3$:

| Algorithm | Regret Bound | Adaptive Complexity | Query Complexity | Runtime |
|-----------|-------------|---------------------|-----------------|---------|
| CUCB | $\widetilde{O}(\sqrt{mdT})$ | $\Theta(T)$ | $\Theta(T)$ | Slowest |
| AROQ-CMAB | $\widetilde{O}(\sqrt{mdT})$ | $O(d\log\log T)$ | $O(d\log\log T)$ | Faster |
| SROQ-CMAB | $\widetilde{O}(m\sqrt{dT})$ | $\Theta(\log\log T)$ | $O(d\log\log T)$ | Fastest |

### Ablation Study

| Framework | Adaptive Complexity | Regret Overhead | Parallelism-Friendly |
|-----------|---------------------|-----------------|----------------------|
| Adaptive (AROQ) | $O(d\log\log T)$ | $\log\log T$ | Moderate |
| Scheduled (SROQ) | $\Theta(\log\log T)$ | $\sqrt{m} \cdot \sqrt{\log\log T}$ | Strong |
| Covariance-Adaptive AROQ | $O(d^2\log(Tm))$ | Near-optimal | Moderate |
| Covariance-Adaptive SROQ | $\Theta(\log\log T)$ | $\sqrt{d}$ factor | Strong |

### Key Findings

- AROQ-CMAB incurs slightly higher regret than CUCB, but reduces oracle calls from linear to doubly logarithmic, significantly reducing runtime.
- SROQ-CMAB introduces an extra $\sqrt{m}$ factor in regret but achieves the fastest runtime in practice due to its lower adaptive complexity enabling efficient parallelization.
- The scheduled framework progressively narrows the candidate arm set during elimination, further reducing the actual computational cost of each oracle query.

## Highlights & Insights

- **Distinguishing adaptive complexity from query complexity**: This two-dimensional measure of oracle efficiency more accurately reflects actual computational overhead, particularly in parallel settings where adaptive complexity is often the true bottleneck.
- **Doubly logarithmic scale**: $O(\log\log T)$ oracle calls are practically negligible (e.g., $\log\log T \approx 3$ for $T=10^6$), representing a significant theoretical advance.
- **Generality of the framework**: The same core ideas extend seamlessly to covariance-adaptive and general reward function variants, demonstrating strong theoretical unity.

## Limitations & Future Work

- The SROQ framework incurs an extra $\sqrt{m}$ factor in regret compared to AROQ, indicating that lower adaptive complexity comes at the cost of higher regret.
- The covariance-adaptive variant requires $O(d^2)$-level oracle complexity, which may become a bottleneck when the number of arms $d$ is large.
- Experiments are conducted solely on synthetic datasets, lacking validation on real-world application scenarios.
- Although the use of approximate oracles ($\alpha$-approximation) is discussed, no empirical evaluation is provided.

## Related Work & Insights

- The epoch-based design is closely related to batched bandit learning, but is innovatively applied to the oracle call setting in combinatorial optimization.
- Unlike oracle-efficient algorithms in submodular optimization, this work must additionally handle the complexity of learning a latent model from stochastic feedback.
- Insight: The principle of "reducing the frequency of expensive operations while maintaining performance guarantees" may generalize to other online learning settings that require frequent calls to NP-hard solvers.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to reduce oracle call complexity to the doubly logarithmic level; outstanding theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐ Only synthetic data experiments; real-world applications are absent.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, though tracking multiple algorithm variants is challenging.
- Value: ⭐⭐⭐⭐ Significant contribution to computational efficiency in combinatorial online learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous Federated Learning](fedqs_optimizing_gradient_and_model_aggregation_for_semi-asynchronous_federated_.md)
- [\[NeurIPS 2025\] Probing Neural Combinatorial Optimization Models](probing_neural_combinatorial_optimization_models.md)
- [\[NeurIPS 2025\] Efficient Adaptive Experimentation with Noncompliance](efficient_adaptive_experimentation_with_noncompliance.md)
- [\[NeurIPS 2025\] Efficient Adaptive Federated Optimization](efficient_adaptive_federated_optimization.md)
- [\[NeurIPS 2025\] FedRTS: Federated Robust Pruning via Combinatorial Thompson Sampling](fedrts_federated_robust_pruning_via_combinatorial_thompson_sampling.md)

</div>

<!-- RELATED:END -->
