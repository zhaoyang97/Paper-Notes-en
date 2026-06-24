---
title: >-
  [Paper Note] Decoding Rewards in Competitive Games: Inverse Game Theory with Entropy Regularization
description: >-
  [ICML2025][Reinforcement Learning][Inverse Game Theory] A unified framework for inverse problems in zero-sum games based on entropy regularization is proposed. Under linear assumptions, the identifiability conditions of reward functions are established using Quantal Response Equilibrium (QRE). An algorithm for constructing confidence sets is provided to recover reward functions from observed actions, with a guaranteed convergence rate of $\mathcal{O}(T^{-1/2})$.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Inverse Game Theory"
  - "Zero-Sum Games"
  - "Entropy Regularization"
  - "Quantal Response Equilibrium"
  - "Markov Games"
  - "Reward Recovery"
  - "Identifiability"
date: 2026-05-08
content_hash: 7aa1e309bd339318
---

# Decoding Rewards in Competitive Games: Inverse Game Theory with Entropy Regularization

**Conference**: ICML2025  
**arXiv**: [2601.12707](https://arxiv.org/abs/2601.12707)  
**Code**: None  
**Area**: Inverse Game Theory / Inverse Reinforcement Learning  
**Keywords**: Inverse Game Theory, Zero-Sum Games, Entropy Regularization, Quantal Response Equilibrium, Markov Games, Reward Recovery, Identifiability

## TL;DR
A unified framework for inverse problems in zero-sum games based on entropy regularization is proposed. Under linear assumptions, the identifiability conditions of reward functions are established using Quantal Response Equilibrium (QRE). An algorithm for constructing confidence sets is provided to recover reward functions from observed actions, with a guaranteed convergence rate of $\mathcal{O}(T^{-1/2})$.

## Background & Motivation

**Inverse Reinforcement Learning (IRL)** aims to infer the underlying reward function driving agent decisions from observed behaviors. Classical IRL focuses on single-agent scenarios, whereas in **competitive games** (such as zero-sum games), an agent's strategy depends not only on its own reward but also on the opponent's strategy, posing greater complexity.

Existing methods face three core challenges:

**Ill-posedness of the Inverse Problem**: Multiple reward functions can yield the same equilibrium policy, requiring the identification of the entire feasible reward set rather than a single solution.

**Insufficient Offline Data Coverage**: Observed policies may fail to traverse the entire state-action space, rendering reward recovery non-robust.

**Complexity of Dynamic Games**: In Markov games, policies evolve over time, increasing the difficulty of identifiability and estimation.

Application scenarios cover economic market pricing analysis, cybersecurity defense strategy inference, competitive routing in transportation and logistics, etc.

## Method

### 1. Entropy-Regularized Zero-Sum Matrix Games

Consider a two-player zero-sum matrix game $(\mathcal{A}, \mathcal{B}, Q)$, where $|\mathcal{A}|=m$ and $|\mathcal{B}|=n$. The minimax objective after introducing entropy regularization is:

$$\max_{\mu} \min_{\nu} \mu^\top Q \nu + \eta^{-1}\mathcal{H}(\mu) - \eta^{-1}\mathcal{H}(\nu)$$

where $\eta > 0$ is the regularization parameter and $\mathcal{H}(\pi) = -\sum_i \pi_i \log(\pi_i)$ is the Shannon entropy. The solution to this problem is called the **Quantal Response Equilibrium (QRE)**, which satisfies the softmax-style fixed-point equations:

$$\mu^*(a) = \frac{e^{\eta Q(a,\cdot)\nu^*}}{\sum_{a' \in \mathcal{A}} e^{\eta Q(a',\cdot)\nu*}}, \quad \nu^*(b) = \frac{e^{-\eta Q(\cdot,b)^\top \mu^*}}{\sum_{b' \in \mathcal{B}} e^{-\eta Q(\cdot,b')^\top \mu^*}}$$

### 2. Identifiability Under Linear Parameterization

**Linear Assumption (Assumption 2.1)**: There exists a feature function $\phi: \mathcal{A} \times \mathcal{B} \to \mathbb{R}^d$ and parameters $\theta^* \in \mathbb{R}^d$ such that $Q(a,b) = \langle \phi(a,b), \theta^* \rangle$.

Taking the logarithm of the QRE fixed-point equations yields the linear system:

$$\begin{bmatrix} A(\nu^*) \\ B(\mu^*) \end{bmatrix} \theta = \begin{bmatrix} c(\mu^*) \\ d(\nu^*) \end{bmatrix}$$

**Strong Identifiability Condition (Proposition 2.2)**: $\theta^*$ is uniquely identifiable if and only if the above coefficient matrix is of full rank, i.e., $\text{rank}\left(\begin{bmatrix} A(\nu^*) \\ B(\mu^*) \end{bmatrix}\right) = d$.

### 3. Two-Step Estimation Algorithm

**Step 1**: Estimate the QRE using a frequency estimator from $N$ i.i.d. samples: $\hat{\mu}(a) = \frac{1}{N}\sum_{k=1}^N \mathbf{1}_{\{a^k=a\}}$

**Step 2**: Estimate the parameters via ordinary least squares: $\hat{\theta} = \arg\min_\theta \left\| \begin{bmatrix} A(\hat{\nu}) \\ B(\hat{\mu}) \end{bmatrix} \theta - \begin{bmatrix} c(\hat{\mu}) \\ d(\hat{\nu}) \end{bmatrix} \right\|^2$

**Finite-Sample Error Bound (Theorem 2.4)**: With probability $\geq 1-\delta$,

$$\|\hat{Q} - Q\|_F^2 \lesssim \mathcal{O}\left(\frac{m^2 + n^2 + (m+n)\log(1/\delta)}{N}\right)$$

### 4. Partial Identifiability and Confidence Sets

When the rank condition is not satisfied, a confidence set $\hat{\Theta}$ is constructed as an alternative to point estimation. By choosing an appropriate threshold $\kappa$ such that $\Theta \subseteq \hat{\Theta}$, the Hausdorff distance $d_H(\Theta, \hat{\Theta}_N) \lesssim \sqrt{\kappa}$ converges at a rate of $\mathcal{O}(N^{-1/2})$ (Theorem 2.7).

### 5. Generalization to Markov Games

The framework is generalized to two-player zero-sum Markov games with state space $\mathcal{S}$, time steps $H$, and discount factor $\gamma$. Under the **linear MDP assumption**, the algorithm consists of four steps:
- Frequency estimation of QRE → Construction of Q-function confidence sets → Ridge regression estimation of the transition kernel → Reward recovery using the Bellman equation

**Sample Complexity (Theorem 3.12)**: With probability $\geq 1-3\delta$,

$$D(\mathcal{R}, \hat{\mathcal{R}}) \lesssim \frac{1}{\sqrt{T}}\left(\sqrt{S(m+n)\log\frac{HS}{\delta}}\left(\sqrt{S(m+n)} + \log T\right) + \left(\sqrt{Sd} + \sqrt{d\log T}\right)\log(mn)\right)$$

### 6. MLE Alternative to Frequency Estimation

To relax the assumption that "all states are fully visited," a linearly parameterized QRE (Assumption 3.13) is introduced, utilizing MLE for policy estimation. The Hellinger distance convergence rate of MLE is $\mathcal{O}(1/T)$, leading to an error bounds rate of $\mathcal{O}(T^{-1/2})$ for the final reward set.

## Key Experimental Results

### Matrix Game Experiments

| Setup | Parameter Dimension $d$ | Action Space $m \times n$ | Identifiability | Sample Range |
|------|-------------|---------------------|---------|---------|
| Setup I | 2 | 4×6 | Strongly Identifiable | $10^3$–$10^6$ |
| Setup II | 6 | 6×6 | Partially Identifiable | $10^3$–$10^6$ |

- **Setup I** (Strongly Identifiable): $\|\hat{\theta} - \theta^*\|$, $\|\hat{Q} - Q^*\|_F$, and $\text{TV}(\hat{\mu},\mu^*) + \text{TV}(\hat{\nu},\nu^*)$ all converge at a rate of ~$\mathcal{O}(N^{-1/2})$, aligning with the theory.
- **Setup II** (Partially Identifiable): Parameters and rewards do not converge to ground-truth values, but the QRE error still converges to zero.

### Markov Game Experiments ($m=n=5$, $S=4$, $H=6$, $\eta=0.5$)

| Sample Size | Reward Error (Mean ± 95% CI) | QRE Error (Mean ± 95% CI) |
|-------|----------------------|---------------------|
| 10,000 | 2.46 ± 0.16 | 7.08×10⁻³ ± 4.61×10⁻⁴ |
| 20,000 | 1.90 ± 0.10 | 5.11×10⁻³ ± 3.11×10⁻⁴ |
| 50,000 | 1.56 ± 0.07 | 3.28×10⁻³ ± 1.70×10⁻⁴ |
| 100,000 | 1.44 ± 0.05 | 2.41×10⁻³ ± 1.41×10⁻⁴ |

Key Observation: Even when the estimation error of the reward function is relatively large, the recovered QRE remains highly consistent with the ground-truth QRE, validating the statistical consistency of the method.

## Highlights & Insights

1. **First to establish a complete identifiability theory for competitive game inverse problems**: Providing the necessary and sufficient conditions for strong identifiability (the rank condition) and addressing partial identifiability scenarios.
2. **Unified framework covering both static and dynamic settings**: Under the same theoretical system, both matrix games and Markov games are processed.
3. **Confidence sets rather than point estimates**: When the inverse problem has multiple solutions, constructing a confidence set containing the entire feasible set is more reasonable than imposing a unique solution.
4. **MLE extension**: Relaxing the strong assumption of frequency estimators requiring all states to be fully covered, rendering the method more practical.
5. **Optimal convergence rate of $\mathcal{O}(T^{-1/2})$**: All theoretical guarantees achieve the minimax optimal statistical rate of classical empirical risk minimization.

## Limitations & Future Work

1. **Strong linear parameterization assumption**: Reward/transition functions in real-world games are often non-linear, limiting the representational power of the method.
2. **Limited to zero-sum games**: General-sum games or multi-player games are not addressed.
3. **Limited experimental scale**: Verification is conducted in small-scale environments (state space $S=4$, action space $m=n=5$) and remains unverified in large-scale scenarios.
4. **Perfect information assumption**: Partially observable stochastic games (POSGs) are not considered.
5. **Offline setting**: The integration with online learning/exploration strategies is not explored.
6. **Computational complexity unaddressed**: Constructing confidence sets in high-dimensional parameter spaces can be computationally expensive.

## Related Work & Insights

- **Inverse Reinforcement Learning**: Inherits the entropy regularization approach from Maximum Entropy IRL (Ziebart et al., 2008) and extends it to game theory.
- **Zero-Sum Markov Games**: Complements forward learning algorithms such as Cen et al. (2021, 2022) and Xie et al. (2023).
- **Inverse Game Theory**: Extends the work of Lin et al. (2014) and Yu et al. (2019), providing the first complete sample complexity analysis.
- **Insights**: The softmax structure of QRE serves as a critical bridge connecting entropy-regularized RL and game theory, potentially inspiring a broader range of research in multi-agent IRL.

## Rating
- Novelty: ⭐⭐⭐⭐ — First to systematically extend the identifiability theory of entropy-regularized IRL to competitive games, presenting significant theoretical contributions.
- Experimental Thoroughness: ⭐⭐⭐ — Thorough theoretical validation but on a small scale, lacking experiments on real-world tasks.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, progressing logically from matrix games to Markov games, with rigorous theoretical derivations.
- Value: ⭐⭐⭐⭐ — Laying a solid theoretical foundation for reward inference in competitive environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory](../../ICML2026/reinforcement_learning/game_of_thought_robust_information_seeking_with_large_language_models_using_game.md)
- [\[ICML 2025\] Learning to Trust Bellman Updates: Selective State-Adaptive Regularization for Offline RL](learning_to_trust_bellman_updates_selective_state-adaptive_regularization_for_of.md)
- [\[ICML 2025\] Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization](robust_offline_reinforcement_learning_with_linearly_structured_f-divergence_regu.md)
- [\[ICLR 2026\] Beyond Softmax and Entropy: Convergence Rates of Policy Gradients with $f$-SoftArgmax Parameterization & Coupled Regularization](../../ICLR2026/reinforcement_learning/beyond_softmax_and_entropy_convergence_rates_of_policy_gradients_with_boldsymbol.md)
- [\[ICML 2025\] Heterogeneous Data Game: Characterizing the Model Competition Across Multiple Data Sources](heterogeneous_data_game_characterizing_the_model_competition_across_multiple_dat.md)

</div>

<!-- RELATED:END -->
