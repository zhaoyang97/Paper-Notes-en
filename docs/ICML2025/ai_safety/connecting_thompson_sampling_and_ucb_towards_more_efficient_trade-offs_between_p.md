---
title: >-
  [Paper Note] Connecting Thompson Sampling and UCB: Towards More Efficient Trade-offs Between Privacy and Regret
description: >-
  [ICML2025][AI Safety][Thompson Sampling] The authors propose the DP-TS-UCB algorithm, which establishes a connection between Thompson Sampling and UCB by restricting the number of Gaussian samples and reusing the maximum model value. This achieves a parameterized tradeoff between $\tilde{O}(T^{0.25(1-\alpha)})$-GDP privacy guarantees and an $O(K\ln^{\alpha+1}(T)/\Delta)$ regret upper bound.
tags:
  - "ICML2025"
  - "AI Safety"
  - "Thompson Sampling"
  - "UCB"
  - "Differential Privacy"
  - "Gaussian Differential Privacy (GDP)"
  - "Stochastic Multi-Armed Bandits"
  - "Privacy-Regret Tradeoff"
  - "anti-concentration bounds"
date: 2026-05-08
content_hash: c6f4ab4004f070f2
---

# Connecting Thompson Sampling and UCB: Towards More Efficient Trade-offs Between Privacy and Regret

**Conference**: ICML2025  
**arXiv**: [2505.02383](https://arxiv.org/abs/2505.02383)  
**Code**: None  
**Area**: AI Safety / Differential Privacy and Online Learning  
**Keywords**: Thompson Sampling, UCB, Differential Privacy, Gaussian Differential Privacy (GDP), Stochastic Multi-Armed Bandits, Privacy-Regret Tradeoff, anti-concentration bounds

## TL;DR

The authors propose the DP-TS-UCB algorithm, which establishes a connection between Thompson Sampling and UCB by restricting the number of Gaussian samples and reusing the maximum model value. This achieves a parameterized tradeoff between $\tilde{O}(T^{0.25(1-\alpha)})$-GDP privacy guarantees and an $O(K\ln^{\alpha+1}(T)/\Delta)$ regret upper bound.


## Background & Motivation

### Background

**Background: Private Multi-Armed Bandits**: In online learning, learners need to make decisions using historical data, while privacy constraints require limiting the amount of information leakage.

### Limitations of Prior Work & Goal

**Goal**: **Key Challenge**: **Limitations of Prior GDP Relaxation**: Although TS-Gaussian (Agrawal & Goyal, 2017) inherently satisfies $O(\sqrt{T})$-GDP, sampling Gaussian models for every arm in each round results in severe privacy budget waste.

### Proposed Solution

**Core Idea / Key Insight**: (1) The Gaussian distribution only changes when an arm is pulled; sampling in intermediate rounds causes redundant privacy leakage. (2) An arm-specific epoch structure is utilized to ensure each observation is used only once, further reducing privacy loss. (3) The maximum of $\phi$ samples is probabilistically equivalent to the optimistic upper bound of UCB.

## Method

### Overall Architecture

DP-TS-UCB is a two-stage algorithm:
1. **Mandatory TS-Gaussian Phase** ($h_i \geq 1$): Sample at most $\phi$ model values from the Gaussian distribution $\mathcal{N}(\hat{\mu}_{i,n_i}, \ln^\alpha(T)/n_i)$.
2. **Optional UCB Phase** ($h_i = 0$): Reuse the maximum of the $\phi$ model values, $\text{MAX}_i = \max_{h \in [\phi]} \theta_{i,n_i}^{(h)}$, as the optimistic estimate.

### Key Designs

- **Sampling Budget**: $\phi = c_0 T^{0.5(1-\alpha)} \ln^{0.5(3-\alpha)}(T)$, determined by the tradeoff parameter $\alpha \in [0,1]$ and the time horizon $T$.
- **Lemma 4.1 (Sufficient Exploration Guarantee)**: The maximum of $\phi$ samples satisfies $\max_{h \in [\phi]} \theta_{i,s}^{(h)} \geq \mu_i$ with probability at least $1 - O(1/T)$, which is equivalent to UCB-style optimism.
- **Arm-specific Epoch Structure**: The $r$-th epoch uses the most recent $2^r$ observations to update the empirical mean, ensuring that each observation is used only once.

### Privacy Analysis

- The privacy of each round of the TS-Gaussian phase is $\sqrt{1/\ln^\alpha(T)}$-GDP.
- The composition of $\phi$ rounds yields $\sqrt{\phi/\ln^\alpha(T)}$-GDP (Theorem 4.7).
- The UCB phase constitutes post-processing and does not incur additional privacy loss (Theorem 4.8).
- **Overall (Theorem 4.4)**: $\sqrt{2c_0 T^{0.5(1-\alpha)} \ln^{1.5(1-\alpha)}(T)}$-GDP.

## Key Experimental Results

### Summary of Privacy-Regret Tradeoff


### Main Results

| Algorithm | Regret Upper Bound | GDP Guarantee |
|---|---|---|
| TS-Gaussian | $O(K\ln(T\Delta^2)/\Delta)$ | $O(T^{0.5})$ |
| M-TS-Gaussian (Tuning $b,c=\ln^\alpha T$) | $O(K\ln^\alpha(T)\ln(T\Delta^2)/\Delta)$ | $O(T^{0.5}/\ln^\alpha T)$ |
| M-TS-Gaussian (Tuning $b,c=T^\gamma$) | $O(KT^\gamma\ln(T\Delta^2)/\Delta)$ | $O(T^{0.5-\gamma})$ |
| **DP-TS-UCB ($\alpha=0$)** | $O(K\ln(T^{1.5}\Delta^2)/\Delta)$ | $\tilde{O}(T^{0.25})$ |
| **DP-TS-UCB ($\alpha=1$)** | $O(K\ln(T\Delta^2)\ln T/\Delta)$ | $O(1)$ |

### Key Findings

- When $\alpha=0$: The regret is near-optimal (with only an additional $\log^{0.5}$ factor), while the GDP is reduced from $O(\sqrt{T})$ to $\tilde{O}(T^{0.25})$, showing a significant improvement.
- When $\alpha=1$: Constant GDP is achieved, with the regret incurring only an additional $\ln T$ factor, meaning that extending the time horizon does not increase the privacy cost.
- Compared to M-TS-Gaussian, the GDP improvement is more prominent under the same regret magnitude—M-TS-G can only reduce $\sqrt{T}$ to $\sqrt{T}/\text{polylog}$ at most, whereas DP-TS-UCB directly drops it to the $T^{0.25}$ scale.
- $(\varepsilon,\delta)$-DP Conversion (Theorem 4.6): Through the primal-dual conversion in Theorem 2.4, a smaller $\phi$ leads to a smaller $\delta(\varepsilon)$.

## Highlights & Insights

1. **Theoretical Value of the TS-UCB Connection**: This work reveals that the maximum of $\phi$ Gaussian samples is equivalent to the UCB optimistic upper bound, bridging the theories of randomized and deterministic exploration.
2. **Smoothly Adjustable Privacy-Regret Tradeoff**: A single parameter $\alpha$ controls the continuous spectrum from optimal regret to constant privacy.
3. **Practical Epoch Structure**: The arm-specific epoch structure serves not only as an analytical tool but also as a core mechanism to reduce privacy loss.
4. **Advantages of the GDP Framework**: The composition theorems of Gaussian Differential Privacy make the analysis tighter and more natural.
5. **Worst-case Regret**: The worst-case regret of DP-TS-UCB is $O(\sqrt{KT}\ln^{0.5(1+\alpha)}(T))$, which is close to the optimal bound of $O(\sqrt{KT\ln K})$.
6. **Privacy Preservation via Post-Processing**: The UCB phase uses the maximum of existing samples, which is a post-processing step and does not incur further privacy cost—a critical design technique.

### Detailed Comparison of Worst-case Regret

| Algorithm | Worst-case Regret | GDP |
|---|---|---|
| TS-Gaussian | $O(\sqrt{KT\ln K})$ | $O(\sqrt{T})$ |
| DP-TS-UCB ($\alpha=0$) | $O(\sqrt{KT\ln T})$ | $\tilde{O}(T^{0.25})$ |
| DP-TS-UCB ($\alpha=1$) | $O(\sqrt{KT}\ln T)$ | $O(1)$ |

## Limitations & Future Work

- The paper only considers stochastic multi-armed bandits, without generalizing to linear/contextual bandits or adversarial environments.
- No new lower bound results are provided, and there may still be a gap between the existing upper bound and the known lower bound.
- The selection of $\phi$ relies on $T$. In practice, $T$ might be unknown, requiring techniques like the doubling trick.
- The experimental evaluation is limited, as the paper is primarily theoretical and lacks validation on real-world datasets.
- The assumption of reward distributions with $[0,1]$ support might be restrictive in practice.
- The relationship with local DP or shuffle DP is not discussed.

## Related Work & Insights

- **TS-Gaussian (Agrawal & Goyal, 2017)**, **UCB1 (Auer et al., 2002)**: These are the two foundational algorithms of this work.
- **Ou et al. (2024)**: First associated TS-Gaussian with GDP and proved it to be $O(\sqrt{T})$-GDP. Based on this, this work reduces the GDP to the $T^{0.25}$ scale by bounding the number of sampling steps.
- **Sajed & Sheffet (2019), Azize & Basu (2022)**: Pioneered the epoch structure for private online learning.
- **Hu et al. (2021), Hu & Hegde (2022)**: Designed optimal $(\varepsilon,0)$-DP multi-armed bandit algorithms.
- **Lower Bound Reference**: The lower bound $\Omega(\sum \ln(T)/\Delta_i + K\ln(T)/\varepsilon)$ from Shariff & Sheffet (2018) is compatible with the upper bound in this paper.
- **Insights**: The core insight that the maximum of TS samples is equivalent to the UCB upper bound can be extended to broader Bayesian optimization scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — The insight regarding the connection between TS and UCB holds independent research value.
- Experimental Thoroughness: ⭐⭐⭐ — Primarily theoretical, with limited experimental validation.
- Writing Quality: ⭐⭐⭐⭐ — Well-presented and clear, with concise proof sketches.
- Value: ⭐⭐⭐⭐ — Provides a superior privacy-regret tradeoff for private online learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rethinking Pareto Frontier: On the Optimal Trade-offs in Fair Classification](../../ICLR2026/ai_safety/rethinking_pareto_frontier_on_the_optimal_trade-offs_in_fair_classification.md)
- [\[ICML 2025\] Clients Collaborate: Flexible Differentially Private Federated Learning with Guaranteed Improvement of Utility-Privacy Trade-off](clients_collaborate_flexible_differentially_private_federated_learning_with_guar.md)
- [\[NeurIPS 2025\] Mitigating Privacy-Utility Trade-off in Decentralized Federated Learning via f-Differential Privacy](../../NeurIPS2025/ai_safety/mitigating_privacy-utility_trade-off_in_decentralized_federated_learning_via_f-d.md)
- [\[ICML 2025\] Breaking the n^{1.5} Additive Error Barrier for Private and Efficient Graph Sparsification](breaking_the_n15_additive_error_barrier_for_private_and_efficient_graph_sparsifi.md)
- [\[NeurIPS 2025\] Position: Bridge the Gaps between Machine Unlearning and AI Regulation](../../NeurIPS2025/ai_safety/position_bridge_the_gaps_between_machine_unlearning_and_ai_regulation.md)

</div>

<!-- RELATED:END -->
