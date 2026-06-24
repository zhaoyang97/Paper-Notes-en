---
title: >-
  [Paper Note] Pessimism Principle Can Be Effective: Towards a Framework for Zero-Shot Transfer RL
description: >-
  [ICML 2025][Reinforcement Learning][transfer RL] This paper proposes a transfer RL framework based on the pessimism principle. By constructing a conservative lower bound of target domain performance as an optimization surrogate using robust MDPs, the authors design two surrogates, Averaged Operator and Minimal Pessimism, along with distributed algorithms, to ensure safe transfer and avoid negative transfer.
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "transfer RL"
  - "pessimism"
  - "robust MDP"
  - "distributed learning"
  - "negative transfer"
date: 2026-05-08
content_hash: bcf22f57afd03650
---

# Pessimism Principle Can Be Effective: Towards a Framework for Zero-Shot Transfer RL

**Conference**: ICML 2025  
**arXiv**: [2505.18447](https://arxiv.org/abs/2505.18447)  
**Code**: None  
**Area**: Reinforcement Learning / Transfer Learning  
**Keywords**: transfer RL, pessimism, robust MDP, distributed learning, negative transfer

## TL;DR
This paper proposes a transfer RL framework based on the pessimism principle. By constructing a conservative lower bound of target domain performance as an optimization surrogate using robust MDPs, the authors design two surrogates, Averaged Operator and Minimal Pessimism, along with distributed algorithms, to ensure safe transfer and avoid negative transfer.

## Background & Motivation

### Background

**Background**: Background**: Transfer RL leverages source domain data to train policies for data-scarce target domains. Domain Randomization (DR) is a commonly used method but optimizes an average performance surrogate, while multi-task learning jointly optimizes multiple tasks.

**Limitations of Prior Work**: (1) Lack of performance guarantees—DR may overestimate performance in the target domain, leading to severe consequences after deployment; (2) Inability to avoid negative transfer—source domains with large discrepancies can degrade overall performance.

**Key Challenge**: There is no theoretical connection between the surrogate objective and the target domain performance; being too optimistic is unsafe, while being too conservative yields poor performance.

**Goal**: Design a conservative surrogate $f(\pi) \leq V_{P_0}^\pi$ to provide performance guarantees with controllable pessimism for policy transfer.

**Key Insight**: Robust RL is inherently pessimistic—the robust value function serves as a lower bound when the target domain lies within the uncertainty set.

**Core Idea**: The pessimism principle guarantees safety and performance monotonicity; the AO surrogate outperforms Proximal DR; the MP surrogate prevents negative transfer.

## Method

### Overall Architecture
Target domain $\mathcal{M}_0$, $K$ source domains, $D(P_0, P_k) \leq \Gamma$. For each source domain, a local uncertainty set $\mathcal{P}_k$ is constructed such that $P_0 \in \mathcal{P}_k$, designing a conservative surrogate $\to$ distributed optimization $\to$ transferable policy.

### Key Designs

1. **Effectiveness of Pessimism Principle (Lemma 4.1)**: $V_{P_0}^{\pi^*} - V_{P_0}^{\pi_f} \leq \|\zeta\| = \max_\pi(V_{P_0}^\pi - f(\pi))$. Smaller pessimism $\to$ smaller suboptimality gap $\to$ better policy. Improving the surrogate equals improving the transfer.

2. **Averaged Operator Surrogate (Sec 5)**: $\mathbf{T}^\pi_{\text{AO}}Q = \frac{1}{K}\sum_k \mathbf{T}^\pi_k Q$. Conservativeness (Thm 5.2): $V^\pi_{\text{AO}} \leq V^\pi_{P_0}$. Outperforms Proximal DR (Prop 5.4): $V^\pi_{\bar{\mathcal{P}}} \leq V^\pi_{\text{AO}}$.

3. **MDTL-Avg Algorithm**: Local robust Bellman updates for $E$ steps at each source domain $\to$ global averaged synchronization. Convergence (Thm 5.7): $\tilde{O}(1/(TK) + (E-1)\Gamma/T)$, achieving partial linear speedup.

4. **Minimal Pessimism Surrogate (Sec 6)**: $\mathbf{T}^\pi_{\text{MP}}Q = \max_k \mathbf{T}^\pi_k Q$. Conservative and outperforms AO (Thm 6.1): $V^\pi_{\mathcal{P}_k} \leq V^\pi_{\text{AO}} \leq V^\pi_{\text{MP}} \leq V^\pi_{P_0}$. Outlier source domains are automatically covered $\to$ avoiding negative transfer.

5. **MDTL-Max + MLMC**: The bias in $\mathbb{E}[\max_k Q_k] \neq \max_k \mathbb{E}[Q_k]$ is addressed by constructing an Unbiased Estimator using MLMC (Lemma 6.2).

## Key Experimental Results

### Comparison of Surrogate Methods


### Main Results

| Surrogate | Conservative | Performance | Prevent Negative Transfer | Distributed |
|------|------|------|-----------|--------|
| Single Source Domain $V^\pi_{\mathcal{P}_k}$ | ✓ | Worst | ✗ | ✓ |
| Proximal DR | ✓ | Poor | ✗ | Requires Model Sharing |
| **AO** | ✓ | Moderate | ✗ | ✓ |
| **MP** | ✓ | Best | ✓ | ✓ (Requires MLMC) |

### Convergence Rate


### Ablation Study

| Algorithm | Convergence |
|------|------|
| MDTL-Avg | $\tilde{O}(1/(TK) + (E-1)\Gamma/T)$ |
| MDTL-Max | Same as above |

### Key Findings
- AO > Proximal DR: "average then pessimise" outperforms "pessimise then average".
- MP > AO: taking the max naturally filters/selects the most similar source domains.
- Conservative surrogates simultaneously guarantee performance in perturbed environments near the target domain (robustness bonus).

## Highlights & Insights
- **Performance Monotonicity** (Lemma 4.1) is the cornerstone of the framework—improving the surrogate equals improving the transfer, which DR lacks.
- **Automatic Domain Selection of MP**: No extra domain similarity estimation is required; taking the max intrinsically filters them.
- **Distributed Privacy Protection**: Only Q-tables are shared, without exposing raw data.

## Limitations & Future Work
- It requires a known bound $\Gamma \geq D(P_0, P_k)$. An excessively large $\Gamma$ leads to over-conservative behavior; in extreme cases, $\Gamma=1$ (TV distance) can be set, but the policy may become overly conservative.
- MLMC introduces additional computational overhead. Although mitigated by threshold-MLMC, it increases implementation complexity; sample complexity is the trade-off for improved performance.
- Tabular MDP setting, not yet extended to function approximation, continuous state spaces, or deep RL scenarios.
- Lack of numerical comparisons with real sim-to-real transfer benchmarks, wanting empirical validation on domains like robotics or autonomous driving.
- Assumes all source domains share the same state space, action space, and reward function, differing only in the transition kernel—in practice, rewards may also differ.
- The choice of $E$ (communication efficiency vs. convergence) requires tuning for specific problems; the theoretically derived bounds may be conservative.
- When the number of source domains $K$ is large, the max aggregation of the MP surrogate might be highly sensitive to noise in individual source domains.

## Related Work & Insights
- **vs DR (Tobin et al., 2017)**: DR lacks theoretical guarantees and can be overly optimistic; the pessimism principle in this paper ensures safety.
- **vs Offline RL Pessimism (Jin et al., 2020)**: Similar concept but applied differently (out-of-domain transfer vs. out-of-distribution evaluation).
- **vs Federated Learning**: Borrows the local update + global aggregation paradigm, but the MP aggregation scheme is unique to transfer RL.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic application of the pessimism principle in transfer RL
- Experimental Thoroughness: ⭐⭐ Experiments are present but on a small scale
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, progressively advancing theoretical development
- Value: ⭐⭐⭐⭐ Provides a theoretically guaranteed framework for transfer RL

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Zero-Shot Generalization of Vision-Based RL Without Data Augmentation](zero-shot_generalization_of_vision-based_rl_without_data_augmentation.md)
- [\[ICML 2025\] Actor-Critics Can Achieve Optimal Sample Efficiency](actor-critics_can_achieve_optimal_sample_efficiency.md)
- [\[ICML 2025\] Solving Zero-Sum Convex Markov Games](solving_zero-sum_convex_markov_games.md)
- [\[NeurIPS 2025\] Dynamics-Aligned Latent Imagination in Contextual World Models for Zero-Shot Generalization](../../NeurIPS2025/reinforcement_learning/dynamics-aligned_latent_imagination_in_contextual_world_models_for_zero-shot_gen.md)
- [\[ICML 2026\] Unlocking Zero-Shot Geospatial Reasoning via Indirect Rewards](../../ICML2026/reinforcement_learning/unlocking_zero-shot_geospatial_reasoning_via_indirect_rewards.md)

</div>

<!-- RELATED:END -->
