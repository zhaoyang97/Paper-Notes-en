---
title: >-
  [Paper Note] Extreme Value Policy Optimization for Safe Reinforcement Learning
description: >-
  [ICML2025][Reinforcement Learning][Safe RL] The EVO algorithm is proposed to introduce Extreme Value Theory (EVT) into constrained reinforcement learning. It models extreme samples in the tail of the cost distribution using the Generalized Pareto Distribution (GPD) and designs extreme quantile constraints along with an extreme prioritization replay mechanism, achieving zero constraint violations during training while maintaining competitive policy performance.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Safe RL"
  - "Extreme Value Theory"
  - "Constraint Satisfaction"
  - "Generalized Pareto Distribution"
  - "Prioritized Experience Replay"
date: 2026-05-08
content_hash: 393424d174962a3f
---

# Extreme Value Policy Optimization for Safe Reinforcement Learning

**Conference**: ICML2025  
**arXiv**: [2601.12008](https://arxiv.org/abs/2601.12008)  
**Code**: [ShiqingGao/EVO](https://github.com/ShiqingGao/EVO)  
**Area**: Safe Reinforcement Learning / Constrained Reinforcement Learning  
**Keywords**: Safe RL, Extreme Value Theory, Constraint Satisfaction, Generalized Pareto Distribution, Prioritized Experience Replay

## TL;DR

The EVO algorithm is proposed to introduce Extreme Value Theory (EVT) into constrained reinforcement learning. It models extreme samples in the tail of the cost distribution using the Generalized Pareto Distribution (GPD) and designs extreme quantile constraints along with an extreme prioritization replay mechanism, achieving zero constraint violations during training while maintaining competitive policy performance.

## Background & Motivation

The goal of constrained reinforcement learning (CRL) is to maximize cumulative rewards while satisfying predefined constraints. Existing methods are mainly divided into two categories:

**Expectation-based constraint methods** (e.g., CPO, PID Lagrangian): Use the expectation of cumulative cost as the constraint, which only guarantees "average" constraint satisfaction. They ignore the variability of the cost distribution, particularly extreme events in the tail (black swan events), leading to frequent constraint violations.

**Probability-based constraint methods** (e.g., WCSAC, QCPO): Use CVaR or quantile constraints. However, WCSAC employs Gaussian approximation for the cost distribution, failing to capture the tail decay behavior accurately, while QCPO neglects the critical impact of extreme samples during training.

**Core Problem**: Extreme samples (low probability, high impact) are critical in safety-critical scenarios but are naturally scarce and exhibit high variance, making them difficult to model accurately and utilize effectively.

## Method

### 1. Extreme Quantile Constraint

The distribution of cumulative cost $C = \sum_{t=0}^{\infty} \gamma^t c$ is divided into a body and a tail:

- **Safety boundary** $q_\mu$: Determined by the expected cumulative cost, separating the body and the tail.
- **Risk boundary** $q_{\mu+\nu}$: The constraint boundary integrating extreme values.

According to Pickands' theorem in EVT, the conditional excess distribution above the threshold $q_\mu$ asymptotically follows a GPD:

$$q_{\mu+\nu} \simeq q_\mu + q^H_{\frac{\nu}{1-\mu}}$$

where $q^H_{\frac{\nu}{1-\mu}}$ is the quantile under the GPD. The optimization objective is:

$$\arg\max_{\pi \in \Pi} J_R(\pi) \quad \text{s.t.} \quad q_\mu + q^H_{\frac{\nu}{1-\mu}} \leq d$$

The surrogate optimization objective within the trust region is:

$$\pi_{k+1} = \arg\max_{\pi} \mathbb{E}_{s \sim d^{\pi_k}, a \sim \pi}[A_R^{\pi_k}(s,a)]$$

$$\text{s.t.} \quad J_C(\pi_k) + \frac{1}{1-\gamma}\mathbb{E}[A_C^{\pi_k}(s,a)] + q^H_{\frac{\nu}{1-\mu}} \leq d, \quad D(\pi \| \pi_k) \leq \delta$$

### 2. GPD Parameter Estimation

Samples exceeding the safety boundary $q_\mu$ (peak set $Y_\mu$) are used to fit the shape parameter $\xi$ and scale parameter $\sigma$ of the GPD via MLE:

$$\log \mathcal{L}(\xi, \sigma) = -N_\mu \log \sigma - (1 + \frac{1}{\xi})\sum_{i=1}^{N_\mu} \log(1 + \frac{\xi}{\sigma} Y_i)$$

Risk boundary calculation:

$$q_\mu + q^H_{\frac{\nu}{1-\mu}} = q_\mu + \frac{\sigma}{\xi}\left((1 - \frac{\nu n}{N_\mu})^{-\xi} - 1\right)$$

### 3. Extreme Prioritization

The GPD is modeled separately for extreme samples of rewards and costs to construct extreme sets:

- Extreme cost set $Z_C: \{C > q_\mu + q^H_{\frac{\nu}{1-\mu}}\}$
- Extreme reward set $Z_R: \{A_R > q^r_\mu + q^{H,r}_{\frac{\nu}{1-\mu}}\}$

The priority score is determined by the GPD quantile level:

$$p = \omega_r + \omega_c, \quad P(s_i) = \frac{p(s_i)}{\sum_{k=1}^N p(s_k)}$$

The higher the quantile level (extreme samples with lower probability) $\to$ the higher the replay priority.

### 4. Off-Policy Importance Resampling

To mitigate the high variance caused by the scarcity of extreme samples, importance sampling correction is performed using stored samples from the old policy $\pi_0$:

$$A'_R = \frac{\pi(a|s)}{\pi_0(a|s)} A_R, \quad C' = \frac{\pi(a|s)}{\pi_0(a|s)} C$$

This expands the size of extreme samples and improves GPD fitting stability.

## Key Experimental Results

### Experimental Setup

- **Environments**: Safety Gymnasium (navigation and obstacle avoidance) + Safety MuJoCo (locomotion control)
- **Training Steps**: $10^7$ steps, maximum trajectory length 1000
- **Random Seeds**: 6 seeds per method
- **Baselines**: CPO, WCSAC, Saute, Simmer
- **Constraint Threshold**: 25

### Main Results

| Dimension | EVO Performance |
|------|---------|
| Constraint Satisfaction | Rapid convergence to the feasible region followed by zero constraint violations throughout |
| Policy Performance | Comparable to CPO, superior to Saute and WCSAC |
| Variance | Lower than quantile regression methods (theoretically proven + experimentally verified) |
| Violation Probability | Lower than expectation-based methods by a margin of $\nu_0$ |

### Ablation Study

| Ablating Component | Impact |
|---------|------|
| Remove EVT constraint $\to$ Constant quantile constraint | Policy performance degrades, satisfying constraints by sacrificing returns |
| Remove extreme prioritization | Performance degrades, failing to fully utilize learning signals from extreme samples |
| Remove off-policy resampling | GPD variance increases, leading to inaccurate tail modeling |

### Other Findings

- GPD outperforms Gaussian fitting across various distribution shapes (lower KS test values).
- EVO functions effectively with a sample size of only 10-20.
- EVO adapts robustly across different cost thresholds (0/25/35).

## Highlights & Insights

1. **Solid Theoretical Contribution**: Provides three theoretical guarantees: an upper bound on constraint violation (Theorem 4.1), a lower bound on violation probability (Theorem 4.2), and a lower bound on variance (Theorem 4.3).
2. **Novel Combination of EVT and CRL**: First to systematically introduce Extreme Value Theory into constrained RL, using GPD to model the tail instead of Gaussian approximation.
3. **Zero Violation Guarantee**: Theoretically guarantees that the expected cost of the updated policy strictly satisfies constraints through a zero-violation exploration margin $\nu_0$.
4. **Sample Efficiency**: EVO remains effective even when extreme samples are scarce (10-20 samples).
5. **Dual Extreme Utilization**: Simultaneously models extreme values on both reward and cost sides, balancing performance and safety.

## Limitations & Future Work

1. **GPD Applicability**: When the difference between extreme and normal values is small, the GPD fitting quality degrades (requiring a sufficiently large threshold $t$). The paper suggests using non-linear transformations to amplify differences, but this was not experimentally verified.
2. **I.I.D. Assumption**: EVT requires independent and identically distributed samples, but samples in RL across adjacent time steps are naturally correlated.
3. **Environment Complexity**: Only validated in Safety Gymnasium and MuJoCo; more complex real-world scenarios (e.g., real robots, autonomous driving) have not been tested.
4. **Multi-Constraint Expansion**: Currently only handles a single constraint. How to model the GPD for each constraint separately in multi-constraint scenarios remains to be discussed.
5. **Computational Overhead**: GPD fitting + importance resampling + prioritized replay introduce additional computational costs, which were not reported in terms of time overhead.

## Related Work & Insights

- **CPO** (Achiam et al., 2017): A classic expectation-based trust region method, upon which EVO introduces EVT tail constraints.
- **WCSAC** (Yang et al., 2021): Probabilistic constraint using Gaussian approximation + CVaR, which EVO improves upon by using GPD for more precise tail modeling instead of the Gaussian approximation.
- **QCPO** (Jung et al., 2022): A quantile constraint method, where EVO theoretically proves a lower variance compared to quantile regression.
- **EVAC** (NS et al., 2023): Applies EVT to reduce the variance of extreme rewards but does not address constraint satisfaction.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of EVT + CRL is novel; using GPD to model the tail is intuitive and clear)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive ablation, sensitivity analysis, GPD fitting validation, and sample size experiments are all covered)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, complete theoretical derivations, and intuitive figures)
- Value: ⭐⭐⭐⭐ (Tail risk is a core problem in Safe RL; the method is practical and theoretically guaranteed)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Scaling Value Iteration Networks to 5000 Layers for Extreme Long-Term Planning](scaling_value_iteration_networks_to_5000_layers_for_extreme_long-term_planning.md)
- [\[ICML 2026\] CSPO: Constraint-Sensitive Policy Optimization for Safe Reinforcement Learning](../../ICML2026/reinforcement_learning/cspo_constraint-sensitive_policy_optimization_for_safe_reinforcement_learning.md)
- [\[ICML 2025\] Wasserstein Policy Optimization](wasserstein_policy_optimization.md)
- [\[NeurIPS 2025\] Online Optimization for Offline Safe Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/online_optimization_for_offline_safe_reinforcement_learning.md)
- [\[ICLR 2026\] Off-Policy Safe Reinforcement Learning with Constrained Optimistic Exploration](../../ICLR2026/reinforcement_learning/off-policy_safe_reinforcement_learning_with_cost-constrained_optimistic_explorat.md)

</div>

<!-- RELATED:END -->
