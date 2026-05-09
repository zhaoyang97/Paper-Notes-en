---
title: >-
  [Paper Note] A Generalized Bisimulation Metric of State Similarity between Markov Decision Processes: From Theoretical Propositions to Applications
description: >-
  [NeurIPS 2025 (Poster)][Reinforcement Learning][bisimulation metric] This paper extends the classical bisimulation metric (BSM), which is limited to measuring state similarity within a single MDP, to cross-MDP settings by proposing a Generalized Bisimulation Metric (GBSM). The authors rigorously prove three fundamental metric properties — symmetry, cross-MDP triangle inequality, and an upper bound on same-state distances — and derive tighter error bounds and closed-form sample complexities than standard BSM in three applications: policy transfer, state aggregation, and sampling-based estimation.
tags:
  - NeurIPS 2025 (Poster)
  - Reinforcement Learning
  - bisimulation metric
  - MDP state similarity
  - policy transfer
  - state aggregation
  - Wasserstein distance
date: 2026-05-08
content_hash: bea747ddfaabfeb1
---

# A Generalized Bisimulation Metric of State Similarity between Markov Decision Processes: From Theoretical Propositions to Applications

**Conference**: NeurIPS 2025 (Poster)
**arXiv**: [2509.18714](https://arxiv.org/abs/2509.18714)
**Code**: N/A
**Area**: Reinforcement Learning / MDP Theory
**Keywords**: bisimulation metric, MDP state similarity, policy transfer, state aggregation, Wasserstein distance

## TL;DR
This paper extends the classical bisimulation metric (BSM), which is limited to measuring state similarity within a single MDP, to cross-MDP settings by proposing a Generalized Bisimulation Metric (GBSM). The authors rigorously prove three fundamental metric properties — symmetry, cross-MDP triangle inequality, and an upper bound on same-state distances — and derive tighter error bounds and closed-form sample complexities than standard BSM in three applications: policy transfer, state aggregation, and sampling-based estimation.

## Background & Motivation
The bisimulation metric (BSM) is a powerful tool in RL theory for measuring state similarity within a single MDP — states with smaller BSM distance have closer optimal value functions. BSM has been successfully applied to state representation learning and policy exploration. However, in **multi-MDP scenarios** (e.g., sim-to-real policy transfer, multi-task RL), where one needs to measure similarity between states from different MDPs, standard BSM is inapplicable.

Song et al. (2016) attempted to assess inter-MDP similarity using BSM, but only conducted empirical evaluations (via Hausdorff distance) without rigorous mathematical analysis. As noted in the survey by García et al. (2022), existing MDP similarity measures lack "theoretical guarantees that the most similar MDP is similar enough to produce positive transfer." This theoretical gap hinders further theoretical development and practical application.

## Core Problem
**How can BSM be rigorously extended from measuring state similarity within a single MDP to a cross-MDP state similarity metric, and what mathematical properties must be established to support theoretical analyses of policy transfer, state aggregation, and related tasks?**

The core challenge is that a cross-MDP metric is no longer a pseudometric on a single state space. The same state $s$ in different MDPs $M_1$ and $M_2$ is an essentially different object (due to differing transition probabilities $P$ and rewards $R$), which means existing BSM proof techniques (e.g., triangle inequality) cannot be directly applied.

## Method

### Overall Architecture
GBSM is defined over **state–MDP pairs**: $d((s, M_1), (s', M_2))$, abbreviated as $d_{1-2}(s, s')$. The inputs are two MDPs $M_1 = \langle S_1, A, P_1, R_1, \gamma \rangle$ and $M_2 = \langle S_2, A, P_2, R_2, \gamma \rangle$ sharing the action space $A$ but potentially having different state spaces; the output is the distance between any cross-MDP state pair $(s \in S_1, s' \in S_2)$.

The framework proceeds in three steps: (1) define the recursive fixed-point equation for GBSM and prove its existence and convergence; (2) rigorously establish three metric properties; (3) derive theoretical bounds for policy transfer, aggregation, and estimation based on these properties.

### Key Designs

1. **GBSM Fixed-Point Definition (Theorem 3.2)**: The operator is defined as $F(d|s,s') = \max_a \{|R_1(s,a) - R_2(s',a)| + \gamma W_1(P_1(\cdot|s,a), P_2(\cdot|s',a); d)\}$, where $W_1$ denotes the 1-Wasserstein distance. The Knaster–Tarski fixed-point theorem is invoked to prove that $F$ admits a unique fixed point $d_{1-2}$, with iterative convergence guaranteed. The elegance of this definition lies in naturally extending the BSM framework to the cross-MDP setting, coupling the transition distributions of two MDPs via the Wasserstein distance.

2. **Three Metric Properties**:

    - **GBSM Symmetry (Theorem 3.4)**: $d_{1-2}(s, s') = d_{2-1}(s', s)$. Although seemingly natural, the proof requires exploiting the symmetry of the Wasserstein distance, since $d_{1-2}$ and $d_{2-1}$ involve different combinations of probabilities and rewards.
    - **Cross-MDP Triangle Inequality (Theorem 3.5)**: $d_{1-3}(s_1, s_3) \leq d_{1-2}(s_1, s_2) + d_{2-3}(s_2, s_3)$. This is the most fundamental property and the most technically demanding to prove. Since GBSM is not a pseudometric on a single space, the authors introduce a third MDP $M_3$ and employ the **Gluing Lemma** to construct a joint transport plan for the inductive proof.
    - **Same-State-Space Distance Upper Bound (Theorem 3.6)**: When $M_1$ and $M_2$ share the state space $S$, $d_{1-2}(s, s')$ is bounded above by a quantity characterized by the total variation distance. Notably, $d_{1-2}(s, s) \neq 0$ in general (since the same state behaves differently in different MDPs); the equality $d_{1-1}(s,s) = 0$ holds only when $M_1 = M_2$.

3. **Value Function Difference Bound (Theorem 3.3)**: $|V_1^*(s) - V_2^*(s')| \leq d_{1-2}(s, s')$. The GBSM distance directly upper-bounds the difference in optimal value functions across MDPs, forming the foundation for all application analyses.

### Application Theory

1. **Policy Transfer (Theorem 4.1, Corollary 4.2)**: Given a state mapping $f: S_2 \to S_1$, when a policy $\pi$ trained on $M_1$ is transferred to $M_2$, the regret $V_2^* - V_2^\pi$ is provably bounded in terms of GBSM. In contrast to Song et al. (2016), who only demonstrate positive correlation, GBSM yields a **provable upper bound** applicable to arbitrary policies and state mappings.

2. **State Aggregation (Theorem 4.4)**: The error bound for aggregating a continuous or large-scale MDP into a smaller one is tightened from the existing $\frac{2\sigma(2+\gamma)}{1-\gamma}$ to $\frac{2\sigma}{1-\gamma}$. The key to this improvement is that the cross-MDP triangle inequality enables **decoupled error analysis**.

3. **Sampling-Based Estimation (Theorem 4.5)**: In settings where transition probabilities are unknown and must be estimated from samples, the paper provides a **closed-form sample complexity** of $K = \frac{-\ln(\alpha/2) \gamma^2 \bar{R}^2 |S|^2}{2\epsilon^2(1-\gamma)^4}$, improving upon the asymptotic result of Ferns et al. (2011).

### Decoupled Error Analysis
The core practical value of the cross-MDP triangle inequality lies in enabling separation of different error sources. For example, the approximation error $|d_{1-1} - d_{\hat{[1]}-\hat{[1]}}|$ can be decomposed into an aggregation error $2d_{1-[1]}$ and an estimation error $2d_{[1]-\hat{[1]}}$, each analyzed independently before being combined. This avoids the relaxation incurred by the joint analysis required in Ferns et al. (2011).

## Key Experimental Results

Experiments are conducted on randomly generated Garnet-style MDPs ($|S|=20, |A|=5$, 50% branching factor) and a real-world 5G wireless network testbed.

### State Aggregation Experiment

| Discount factor $\gamma$ | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.55 | 0.65 | 0.75 | 0.8 |
|---|---|---|---|---|---|---|---|---|---|
| BSM-based bound | 0.111 | 0.181 | 0.255 | 0.389 | 0.669 | 0.818 | 1.311 | 2.374 | 4.159 |
| GBSM-based bound | 0.101 | 0.156 | 0.187 | 0.275 | 0.390 | 0.455 | 0.618 | 1.078 | 1.536 |

- The advantage of GBSM over BSM becomes increasingly pronounced as $\gamma$ grows; at $\gamma=0.8$, the GBSM bound is approximately 37% of the BSM bound.
- Overall, the GBSM-based bound improves over the BSM-based bound by **60–80%**.
- In the 5G Digital Twin experiment, GBSM distance exhibits a strong linear correlation with worst-case post-transfer performance.

### Ablation Study Highlights
- Policy transfer: The metric of Song et al. (2016) fails to serve as an upper bound on regret, whereas the GBSM bound remains valid throughout.
- Estimation experiment: Estimation error bounds are validated under Gaussian noise with standard deviations of 0.1–0.3 added to transition probabilities.
- The advantage of decoupled analysis is particularly pronounced for larger values of $\gamma$.

## Highlights & Insights
- **Rigorous theoretical foundation for cross-MDP metrics**: This work is the first to establish a framework with complete metric properties for cross-MDP state similarity, filling a theoretical gap in the field.
- **Elegant use of the Gluing Lemma**: Introducing a third MDP and employing the Gluing Lemma to construct a joint transport plan for proving the cross-MDP triangle inequality is a technically elegant proof strategy.
- **Decoupled error analysis**: The greatest practical benefit of the cross-MDP triangle inequality — enabling independent analysis of distinct error sources before combination — yields tighter bounds.
- **Compatibility with lax BSM and on-policy BSM**: The framework is flexible enough to accommodate modern variants of BSM, addressing known limitations of the max-over-actions formulation.
- **First closed-form sample complexity**: Compared to prior asymptotic results, the paper provides an explicit sample count requirement.

## Limitations & Future Work
- **Shared action space requirement**: The current GBSM requires both MDPs to share the same action space $A$. Although the authors demonstrate in the rebuttal that the framework can be extended to lax GBSM to support different action spaces, this extension is not fully developed in the main text.
- **Incompatibility with average-reward settings**: All bounds contain a $\frac{1}{1-\gamma}$ factor, which diverges as $\gamma \to 1$, making the framework inapplicable to average-reward MDPs.
- **Limited experimental scale**: Main experiments are conducted on random MDPs with $|S|=20$; scalability to large-scale or high-dimensional settings requires further investigation.
- **Computational complexity**: Exact computation of GBSM requires iteratively solving Wasserstein distances over a space of size $|S_1| \times |S_2|$, which is impractical for large-scale environments.
- **Absence of deep RL experiments**: No evaluation is performed on standard RL benchmarks such as Atari (reviewers suggested experiments such as Pac-Man color permutation).

## Related Work & Insights

| Method | Scope | Theoretical Bound | Sample Complexity |
|------|---------|--------|-----------|
| BSM (Ferns et al., 2011) | Within single MDP | $\frac{2\sigma(2+\gamma)}{1-\gamma}$ (aggregation) | Asymptotic |
| Song et al. (2016) | Cross-MDP (empirical) | No rigorous upper bound | N/A |
| Kemertas et al. (2021) | Within single MDP | $\frac{2\sigma}{1-\gamma}$ (aggregation) | No closed form |
| **GBSM (Ours)** | **Cross-MDP** | **Tighter** | **Closed-form** |

The core distinction from Song et al. (2016): Song's Hausdorff metric is a simplified version of the first term in the GBSM policy transfer bound (omitting the $\frac{1}{1-\gamma}$ factor) and applies only to the idealized scenario of optimal policies with optimal mappings. GBSM provides rigorous bounds under arbitrary policies and mappings.

**Additional connections**:
- **Theoretical tool for sim-to-real transfer**: GBSM can be used to predict worst-case performance degradation prior to policy deployment, offering practical value for safety-critical systems (e.g., 5G networks, autonomous driving).
- **Multi-task RL**: GBSM can guide task clustering and gradient conflict detection — tasks with large GBSM distance may require gradient surgery, while conflicts between tasks with small GBSM distance may be attributable to sampling noise.
- **Digital Twin evaluation**: GBSM provides a theoretically grounded metric for assessing digital twin fidelity, informing the design and optimization of DTs.
- The decoupled error analysis paradigm may inspire other RL theory work requiring multi-source error decomposition.

## Rating
- Novelty: ⭐⭐⭐⭐ — The problem formulation (cross-MDP metric) is novel, though proof techniques largely extend existing BSM theory.
- Experimental Thoroughness: ⭐⭐⭐ — Primarily validated on small-scale random MDPs; the 5G experiment was added after the rebuttal.
- Writing Quality: ⭐⭐⭐⭐ — Theoretical exposition is clear and rigorous with good structure, though experimental descriptions lack detail.
- Value: ⭐⭐⭐⭐ — Fills a theoretical gap in cross-MDP analysis with practical relevance to sim-to-real transfer.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[NeurIPS 2025\] Prompt Tuning Decision Transformers with Structured and Scalable Bandits](prompt_tuning_decision_transformers_with_structured_and_scalable_bandits.md)
- [\[NeurIPS 2025\] Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions](certifying_stability_of_reinforcement_learning_policies_using_generalized_lyapun.md)
- [\[NeurIPS 2025\] Generalized Linear Bandits: Almost Optimal Regret with One-Pass Update](generalized_linear_bandits_almost_optimal_regret_with_one-pass_update.md)
- [\[NeurIPS 2025\] Feel-Good Thompson Sampling for Contextual Bandits: a Markov Chain Monte Carlo Showdown](feel-good_thompson_sampling_for_contextual_bandits_a_markov_chain_monte_carlo_sh.md)

<!-- RELATED:END -->
