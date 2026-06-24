---
title: >-
  [Paper Note] Enhancing Parallelism in Decentralized Stochastic Convex Optimization
description: >-
  [ICML2025][Optimization][Decentralized SGD] Proposes Decentralized Anytime SGD (DAT-SGD), which alleviates consensus distance bias by calculating gradients on slowly-varying averaged query points. This improves the parallelism upper bound of decentralized stochastic convex optimization from $\mathcal{O}(\rho^{1/2} N^{1/4})$ to $\mathcal{O}(\rho \sqrt{N})$, matching the rate of centralized learning for the first time under highly connected topologies.
tags:
  - "ICML2025"
  - "Optimization"
  - "Decentralized SGD"
  - "Stochastic Convex Optimization"
  - "Upper Bound of Parallelism"
  - "Gossip Communication"
  - "Anytime SGD"
date: 2026-05-08
content_hash: 79480d177dce1c7c
---

# Enhancing Parallelism in Decentralized Stochastic Convex Optimization

**Conference**: ICML2025  
**arXiv**: [2506.00961](https://arxiv.org/abs/2506.00961)  
**Code**: None  
**Area**: Decentralized Optimization / Distributed Learning  
**Keywords**: Decentralized SGD, Stochastic Convex Optimization, Upper Bound of Parallelism, Gossip Communication, Anytime SGD

## TL;DR

Proposes Decentralized Anytime SGD (DAT-SGD), which alleviates consensus distance bias by calculating gradients on slowly-varying averaged query points. This improves the parallelism upper bound of decentralized stochastic convex optimization from $\mathcal{O}(\rho^{1/2} N^{1/4})$ to $\mathcal{O}(\rho \sqrt{N})$, matching the rate of centralized learning for the first time under highly connected topologies.

## Background & Motivation

**Parallelism Bottleneck in Distributed Learning**: Distributed learning accelerates training by processing data in parallel across multiple machines. However, increasing the number of machines $M$ beyond a critical point degrades convergence efficiency. This issue is particularly severe in decentralized systems where nodes communicate via gossip protocols, and sparse topologies amplify performance degradation.

**Gap between Centralized and Decentralized**: Centralized Minibatch SGD (Dekel et al., 2012) allows using up to $\mathcal{O}(\sqrt{N})$ machines without losing statistical efficiency. However, existing decentralized methods (such as D-SGD, Koloskova et al., 2020) only achieve a parallelism upper bound of $\mathcal{O}(\rho^{1/2} N^{1/4})$, showing a significant gap even in near-complete graph topologies ($\rho \approx 1$).

**Root Cause — Consensus Distance**: In decentralized training, models at different nodes are asynchronous, leading to gradient estimation biases relative to the global consensus. In D-SGD, the consensus distance $\Xi_t = \frac{1}{M}\sum_i \|w_t^i - \bar{w}_t\|^2$ is bounded by $\mathcal{O}(\eta^2/\rho)$, which constitutes the core bottleneck restricting parallelism.

**Core Problem**: Are decentralized methods fundamentally unable to fully exploit parallelism even under ideal communication conditions? This work provides a negative answer.

## Method

### Framework: Anytime SGD → Decentralized Extension

**Review of Anytime SGD** (Cutkosky, 2019): Unlike standard SGD, which computes gradients at the current iterate $w_t$, Anytime SGD computes gradients at a weighted average $x_t$ of past iterates. Given non-negative weights $\{\alpha_t\}$, the update rules are:

$$w_{t+1} = w_t - \eta \alpha_t g_t$$

$$x_{t+1} = \frac{\alpha_{1:t-1}}{\alpha_{1:t}} x_t + \frac{\alpha_t}{\alpha_{1:t}} w_{t+1}$$

where $\alpha_{1:t} = \sum_{\tau=1}^t \alpha_\tau$, and $g_t$ is the stochastic gradient evaluated at $x_t$ (instead of $w_t$). **Key Insight**: The averaged query point $x_t$ varies more slowly than the iterate $w_t$, which is naturally suited for controlling the consensus distance in decentralized settings.

### DAT-SGD Algorithm

Extending Anytime SGD to the decentralized setting, each machine $i$ performs the following in each round:

1. **Sampling and Gradient Computation**: $g_t^i = \nabla f_i(x_t^i, z_t^i)$ (evaluated at the query point $x_t^i$ instead of $w_t^i$)
2. **Local Update**:
    - $w_{t+1/2}^i = w_t^i - \eta \alpha_t g_t^i$
    - $x_{t+1/2}^i = \frac{\alpha_{1:t-1}}{\alpha_{1:t}} x_t^i + \frac{\alpha_t}{\alpha_{1:t}} w_{t+1/2}^i$
3. **Gossip Communication** (exchanging both $w$ and $x$ simultaneously):
    - $w_{t+1}^i = \sum_j w_{t+1/2}^j P_{ij}$
    - $x_{t+1}^i = \sum_j x_{t+1/2}^j P_{ij}$

where $P$ is the gossip matrix (a symmetric doubly stochastic matrix) and $\rho = 1 - |\lambda_2| \in (0,1]$ is the spectral gap.

### Key Designs

In D-SGD, the recurrence relation of the consensus distance is $\Xi_{t+1} \leq (1-\rho/2)\Xi_t + C \cdot G^2\eta^2/\rho$, which yields the bound $\Xi_t \leq \mathcal{O}(\eta^2/\rho)$.

DAT-SGD introduces the consensus distance of the query points $\Gamma_t = \frac{1}{M}\sum_i \|x_t^i - \bar{x}_t\|^2$, whose recurrence relation is:

$$\Gamma_{t+1} \leq \left(1-\frac{\rho}{2}\right)\Gamma_t + \frac{C}{\rho t^2}(\Xi_t + G^2\eta^2)$$

Due to the $1/t^2$ decay factor of the weighted average, the consensus distance of the query points experiences stronger contraction control. Substituting the bound of $\Xi_t$ yields:

$$\Gamma_{t+1} \leq \left(1-\frac{\rho}{2}\right)\Gamma_t + \frac{2C \cdot G^2\eta^2}{\rho^2 t^2}$$

Compared to the $\mathcal{O}(\eta^2/\rho)$ bound in D-SGD, the query-point consensus distance of DAT-SGD is significantly smaller, thereby relaxing the constraint on the learning rate $\eta$.

### Weight Choice and Learning Rate

Adopting linear weights $\alpha_t = t$, the learning rate is scheduled as:

$$\eta = \min\left\{\frac{1}{24LT}, \frac{\rho^2}{K}, \frac{D_1\sqrt{M}}{\sqrt{3}\sigma T^{3/2}}, \sqrt{\frac{D_1}{2K\tilde{\sigma}}} \frac{\rho}{T}\right\}$$

where $D_1 = \|w_1 - x^*\|$, $K^2 = 5120L^2$, and $\tilde{\sigma}^2 = 2\sigma^2 + \zeta^2$.

## Theoretical Results

### Main Theorem (Theorem 4.1)

Under the assumptions of convex and smooth functions, bounded noise variance $\sigma^2$, and bounded heterogeneity $\zeta^2$:

$$\mathbb{E}[f(\bar{x}_T) - f^*] \leq \mathcal{O}\left(\frac{\sigma D_1}{\sqrt{MT}} + \frac{D_1^{3/2}\sqrt{L\tilde{\sigma}}}{\rho T} + \frac{LD_1^2}{T}\right)$$

| Metric | D-SGD (Koloskova+2020) | DAT-SGD (Ours) |
|------|----------------------|---------------|
| Convergence Rate | $\mathcal{O}\left(\frac{\sigma^2}{M\epsilon^2}+\frac{\sigma\sqrt{\rho}+\zeta}{\rho\epsilon^{3/2}}+\frac{1}{\rho\epsilon}\right)$ | $\mathcal{O}\left(\frac{\sigma^2}{M\epsilon^2}+\frac{\sqrt{\sigma}+\sqrt{\zeta}}{\rho\epsilon}+\frac{1}{\epsilon}\right)$ |
| Parallelism Upper Bound | $\mathcal{O}(\rho^{1/2} N^{1/4})$ | $\mathcal{O}(\rho\sqrt{N})$ |

### Comparison of Parallelism Under Different Topologies

| Topology | $1/\rho$ | D-SGD | DAT-SGD |
|------|----------|-------|---------|
| Ring | $\mathcal{O}(M^2)$ | $\mathcal{O}(N^{1/8})$ | $\mathcal{O}(N^{1/6})$ |
| Torus | $\mathcal{O}(M)$ | $\mathcal{O}(N^{1/6})$ | $\mathcal{O}(N^{1/4})$ |
| Near-Complete Graph | $\approx 1$ | $\mathcal{O}(\rho^{1/2} N^{1/4})$ | $\mathcal{O}(\rho\sqrt{N})$ |

### Key Findings

- **Near-Complete Graph**: DAT-SGD achieves $\mathcal{O}(\rho\sqrt{N})$, recovering the centralized parallelism of $\mathcal{O}(\sqrt{N})$ when $\rho = \Omega(1)$, which **closes the gap between decentralized and centralized settings for the first time**.
- **General Topologies**: The parallelism upper bounds are comprehensively improved (from $N^{1/8}$ to $N^{1/6}$ for Ring, and from $N^{1/6}$ to $N^{1/4}$ for Torus).
- **Transient Iteration Complexity**: $\mathcal{O}(M/\rho^2)$, which improves upon D-SGD by a factor of $M^2$.
- **Local Iterate Convergence** (Corollary 4.2): The local iterates of each machine also converge, with an extra term $M\tilde{\sigma}D_1/\rho^2 T^2$ that does not affect the parallelism upper bound.

## Highlights & Insights

- **Simple and Elegant**: The algorithmic modification is minimal—only replacing the gradient query point from the current iterate to a weighted average—yet it achieves a fundamental improvement in parallelism.
- **Theoretical Breakthrough**: It matches the centralized learning rate under highly connected topologies for the first time, answering the open question of whether decentralized learning is fundamentally limited.
- **Generality**: The Anytime SGD framework was previously utilized in asynchronous and local training; this work extends its advantages to the decentralized setting.
- **Clear Intuition**: The improvement in consensus distance stems from the slower variation of the query points (due to the $1/t^2$ recurrence factor), which breaks the bottleneck of D-SGD.

## Limitations & Future Work

- **Restricted to Convex Settings**: The analysis is conducted under the SCO framework and is not extended to non-convex optimization (the primary scenario for practical deep learning).
- **Doubled Communication Overhead**: Both $w_t^i$ and $x_t^i$ must be exchanged, which doubles the communication volume compared to D-SGD.
- **No Empirical Verification**: The work is purely theoretical and lacks validation on real-world deep learning tasks.
- **Strong Heterogeneity Assumption**: A globally bounded heterogeneity $\zeta^2$ is assumed, which is not relaxed to a weaker assumption bounded only at the optimum.
- **Lack of Lower Bounds**: It remains unclear whether the parallelism upper bound $\mathcal{O}(\rho\sqrt{N})$ is tight (i.e., whether a matching lower bound exists).
- **Fixed Topology**: The analysis assumes a static communication graph, without covering time-varying topologies or randomized gossip.

## Related Work & Insights

- **D-SGD** (Lian+2017, Koloskova+2020): Standard decentralized SGD, serving as the baseline method.
- **Gradient Tracking** (Koloskova+2021): Eliminates the effect of data heterogeneity, but the parallelism remains limited.
- **Anytime SGD** (Cutkosky+2019): The core tool used in this work, representing a slowly-varying query point framework.
- **Asynchronous Training** (Aviv+2021): Applications of the Anytime framework in asynchronous environments.
- **Local SGD** (Dahan & Levy, 2024): Applications of the Anytime framework in local training.
- **Decentralized Momentum** (He+2022): Replaces SGD with momentum in non-convex settings, achieving a parallelism of $\mathcal{O}((\rho\sqrt{N})^{2/3})$.

## Rating ⭐⭐⭐⭐

Solid theoretical contribution that achieves a significant breakthrough on the classical problem of parallelism in decentralized convex optimization. The algorithmic design is elegant and the analysis is profound. However, it is limited to convex settings and lacks empirical validation; its practical contribution awaits further validation through non-convex extensions.

## Rating
- Novelty: To be evaluated
- Experimental Thoroughness: To be evaluated
- Writing Quality: To be evaluated
- Value: To be evaluated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](../../NeurIPS2025/optimization/isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)
- [\[ICML 2025\] A Near-Optimal Single-Loop Stochastic Algorithm for Convex Finite-Sum Coupled Compositional Optimization](a_near-optimal_single-loop_stochastic_algorithm_for_convex_finite-sum_coupled_co.md)
- [\[ICLR 2026\] Derandomized Online-to-Non-convex Conversion for Stochastic Weakly Convex Optimization](../../ICLR2026/optimization/derandomized_online-to-non-convex_conversion_for_stochastic_weakly_convex_optimi.md)
- [\[ICLR 2026\] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum](../../ICLR2026/optimization/high_probability_bounds_for_non-convex_stochastic_optimization_with_momentum.md)
- [\[NeurIPS 2025\] Problem-Parameter-Free Decentralized Bilevel Optimization](../../NeurIPS2025/optimization/problem-parameter-free_decentralized_bilevel_optimization.md)

</div>

<!-- RELATED:END -->
