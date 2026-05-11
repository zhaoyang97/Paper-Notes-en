---
title: >-
  [Paper Note] Near-Optimal Second-Order Guarantees for Model-Based Adversarial Imitation Learning
description: >-
  [ICLR 2026][Reinforcement Learning][Adversarial imitation learning] This paper proposes MB-AIL (Model-Based Adversarial Imitation Learning), establishing horizon-free second-order sample complexity upper bounds under gen…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Adversarial imitation learning"
  - "model-based methods"
  - "second-order bounds"
  - "sample complexity"
  - "information-theoretic lower bounds"
date: 2026-05-08
content_hash: 0e92c7d489aa3134
---

# Near-Optimal Second-Order Guarantees for Model-Based Adversarial Imitation Learning

**Conference**: ICLR 2026
**arXiv**: [2510.09487](https://arxiv.org/abs/2510.09487)
**Code**: None
**Area**: Reinforcement Learning / Imitation Learning
**Keywords**: Adversarial imitation learning, model-based methods, second-order bounds, sample complexity, information-theoretic lower bounds

## TL;DR

This paper proposes MB-AIL (Model-Based Adversarial Imitation Learning), establishing horizon-free second-order sample complexity upper bounds under general function approximation. Combined with information-theoretic lower bounds on constructed hard instances, MB-AIL is shown to be minimax optimal (up to logarithmic factors) in terms of online interaction sample complexity.

## Background & Motivation

Imitation learning (IL) aims to learn a policy from expert demonstrations without access to reward signals. The main approaches fall into two categories:

**Behavioral Cloning (BC)**: Directly fits the expert policy via supervised learning.

**Adversarial Imitation Learning (AIL)**: Aligns the state-action distributions of the expert and learner through an adversarial framework.

### Core Problem

Empirically, AIL typically outperforms BC with limited expert demonstrations, yet the theoretical understanding remains incomplete. This paper focuses on two key questions:

**Precise benefit of online interaction**: How much sample efficiency does online interaction actually provide for imitation learning?

**Effect of stochasticity**: How do the stochasticity of the expert policy and environment dynamics affect sample complexity?

### Limitations of Prior Work

| Prior Work | Limitations |
|-----------|-------------|
| Xu et al. (2023) | Restricted to tabular MDPs and deterministic experts |
| Viano et al. (2024) | Restricted to linear MDPs |
| Xu et al. (2024) OPT-AIL | General function approximation, but not model-based; no second-order bounds; higher online interaction complexity |
| Foster et al. (2024) | BC theory; no lower bounds for online interaction |

No prior work simultaneously provides: (1) results under general function approximation, (2) second-order (variance-dependent) bounds, and (3) information-theoretic lower bounds for online interaction.

## Method

### Overall Architecture

MB-AIL's core assumption is that the policy space $\Pi$ can be factored into a reward class $\mathcal{R}$ and a model class $\mathcal{P}$. Based on this decomposition, the algorithm separates reward learning from model learning:

- **Reward learning**: Adversarially estimates the reward function using expert demonstrations.
- **Model learning**: Learns the transition kernel via maximum likelihood estimation.
- **Policy optimization**: Performs optimistic planning based on the learned reward and model.

### Key Designs

1. **Adversarial Reward Learning (Procedure A)** → Estimates the unknown reward → Design motivation: eliminates the need for true rewards within the AIL framework.

   Given online-collected trajectories and offline expert data, the reward is learned by minimizing the following empirical loss:
   $$\mathcal{L}_{k-1}(r) = \hat{V}_{1,P^*,r}^{\pi_{k-1}}(s_1) - \hat{V}_{1,P^*,r}^{\pi_E}(s_1)$$

   Follow-the-Regularized-Leader (FTRL) is used as the no-regret online optimization algorithm, achieving $O(1/\sqrt{K})$ optimization error.

2. **Model-Based Learning (Procedure B)** → Exploits problem structure decomposition → Design motivation: maximizes data efficiency of environment samples for model learning.

   A version space is constructed via maximum likelihood estimation (MLE):
   $$\hat{\mathcal{P}}_k = \{P \in \mathcal{P}: \sum_{(s,a,s') \in \mathcal{D}_k} \log P(s'|s,a) \geq \max_{\tilde{P}} \sum \log \tilde{P}(s'|s,a) - \beta\}$$

   Optimistic planning is then performed within the version space:
   $$(\pi_k, P_k) = \arg\max_{\pi, P \in \hat{\mathcal{P}}_k} V_{1,P,r_k}^\pi(s_1)$$

3. **Second-Order Analysis Technique** → Obtains variance-dependent bounds → Design motivation: precisely characterizes the effect of stochasticity.

   Core analysis steps:
   - Decompose regret into reward estimation error and policy error.
   - Apply Bernstein-type concentration inequalities to obtain variance-dependent bounds on reward error.
   - Leverage Eluder dimension and the Variance Conversion Lemma to obtain second-order bounds on policy error.
   - The final bound has only logarithmic dependence on the horizon $H$ (horizon-free).

### Loss & Training

At the theoretical algorithm level:
- **Reward**: Adversarial optimization (FTRL), self-supervisedly maximizing the value gap between expert and learner policies.
- **Model**: Maximum likelihood estimation.
- **Policy**: Optimistic exploration (version-space-based optimistic planning).

At the practical implementation level (Section 6):
- **Reward network**: Wasserstein GAN-style training with gradient penalty.
- **Model ensemble**: Ensemble of 7 world models trained with MLE.
- **Policy optimization**: Soft Actor-Critic (SAC).

## Key Experimental Results

### Theoretical Comparison

| Method | Expert Demo Complexity | Online Interaction Complexity | Second-Order? |
|--------|----------------------|------------------------------|--------------|
| MB-TAIL (Xu, 2023) | $\tilde{O}(H^{3/2}|S|/\epsilon)$ | $\tilde{O}(H^3|S|^2|A|/\epsilon)$ | No |
| OPT-AIL (Xu, 2024) | $\tilde{O}(H^2 \log\mathcal{N_R}/\epsilon^2)$ | $\tilde{O}(H^4 d_{GEC} \log(\mathcal{N_R}\mathcal{N_Q})/\epsilon^2)$ | No |
| **MB-AIL (Ours)** | $\tilde{O}(\sigma^2 \log\mathcal{N_R}/\epsilon^2)$ | $\tilde{O}(\sigma^2 (d_E \log\mathcal{N_P} + \log\mathcal{N_R})/\epsilon^2)$ | **Yes** |
| **Lower Bound (Ours)** | $\Omega(\sigma^2/\epsilon^2)$ | $\Omega(\sigma^2 \log^2|\mathcal{P}| e^{-N}/\epsilon^2)$ | **Yes** |

### GridWorld Experiments

| Experimental Setting | Finding |
|---------------------|---------|
| Varying reward space size | AIL significantly outperforms BC with small reward spaces |
| Varying environment stochasticity | Both methods improve in more deterministic environments; AIL consistently outperforms BC |

### MuJoCo Experiments

| Environment | Expert | BC | GAIL | OPT-AIL | **MB-AIL** |
|------------|--------|----|------|---------|-----------|
| Hopper | 3609 | 2857 | 3212 | 3439 | **3451** |
| Walker2d | 4637 | 2697 | 3777 | 4238 | 4170 |
| Humanoid | 5885 | 343 | 1614 | 2014 | **5816** |

### Interaction Efficiency Comparison

| Environment | OPT-AIL | **MB-AIL** | Gain |
|------------|---------|-----------|------|
| Hopper | 210K | **60K** | 3.5x |
| Walker2d | 320K | **120K** | 2.7x |
| Humanoid | 220K | **90K** | 2.4x |

### Key Findings

1. **Significance of second-order bounds**: As the system approaches determinism ($\sigma^2 \to 0$), sample complexity can improve from $O(1/\epsilon^2)$ to $O(1/\epsilon)$, precisely quantifying the effect of stochasticity.
2. **Horizon-free**: Unlike prior work, the upper bound has only logarithmic dependence on $H$, eliminating the exponential penalty for long-horizon problems.
3. **Minimax optimality of online interaction**: When expert data is limited ($N \ll \log^2|\mathcal{P}|$), MB-AIL's online interaction complexity $\Omega(\sigma^2 \log^2|\mathcal{P}|/\epsilon^2)$ matches the lower bound up to logarithmic factors.
4. **Precise separation between BC and AIL**:
   - AIL is preferred: when the reward class $\mathcal{R}$ has simple structure (small $\log\mathcal{N_R}$).
   - BC is preferred: when the expert policy is deterministic but the environment is highly stochastic.
5. **Breakthrough on Humanoid**: MB-AIL achieves near-expert performance on high-dimensional Humanoid (5816 vs. 5885), far surpassing all baselines.

## Highlights & Insights

1. **Power of decomposition**: Factoring the policy space as $\Pi = \mathcal{R} \times \mathcal{P}$ is the central insight. When the complexity of $\mathcal{R}$ or $\mathcal{P}$ is much lower than that of $\Pi$, model-based methods achieve an intrinsic statistical advantage.
2. **Naturalness of second-order analysis**: Variance-dependent bounds unify the deterministic and stochastic settings, avoiding artificial case distinctions.
3. **Theory-practice consistency**: GridWorld experiments precisely validate theoretical predictions (small reward space → AIL advantage; deterministic environment → improvement for both).
4. **Elegance of hard instance construction**: By combining two scenarios (hard-to-learn policy vs. hard-to-learn model), the construction separates the quantities estimated by expert demonstrations and online interaction respectively.
5. **Simplicity of the practical algorithm**: The practical implementation requires only standard components (world model ensemble + SAC + GAN discriminator).

## Limitations & Future Work

1. **Logarithmic gap in expert demonstration complexity**: The upper bound is $\tilde{O}(\sigma^2 \log\mathcal{N_R}/\epsilon^2)$ while the lower bound is $\Omega(\sigma^2/\epsilon^2)$, leaving a gap of $\log|\mathcal{R}|$. The authors conjecture this gap may be fundamental.
2. **Time-homogeneity assumption**: The theoretical analysis assumes the transition kernel and reward do not vary over time, limiting applicability to more general MDPs.
3. **Approximations in implementation**: Optimistic planning in the practical algorithm is approximated via model ensembles, reducing the strictness of theoretical guarantees.
4. **Limited MuJoCo experiments**: Only 3 environments and 64 expert trajectories are used, limiting the scale of empirical evaluation.
5. **Fairness of comparison with offline BC**: BC requires no online interaction, making direct comparisons not entirely equivalent.

## Related Work & Insights

### Theoretical

- **Foster et al. (2024)**: Second-order bounds and information-theoretic lower bounds for BC; the corresponding counterpart to this paper in the AIL direction.
- **Wang et al. (2024)**: Second-order analysis framework for model-based RL; the primary basis for this paper's analytical techniques.
- **Xu et al. (2024) OPT-AIL**: AIL upper bounds under general function approximation, but not model-based.

### Practical

- **GAIL (Ho & Ermon, 2016)**: Classic adversarial imitation learning method.
- **SAC (Haarnoja et al., 2018)**: Policy optimization method used in the practical algorithm.
- **World model ensemble**: Method for quantifying model uncertainty in the practical implementation.

### Insights for Research

1. Model-based methods have a theoretically guaranteed intrinsic advantage in sample efficiency.
2. Second-order analysis is the appropriate framework for understanding the effect of stochasticity.
3. Simultaneously establishing upper and lower bounds is essential for understanding the fundamental difficulty of a problem.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First second-order upper and lower bounds for AIL under general function approximation; first information-theoretic lower bound for online interaction in AIL.
- **Experimental Thoroughness**: ⭐⭐⭐ — GridWorld validates the theory; MuJoCo demonstrates practical utility; however, the number of environments and scale are limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretically rigorous and thorough across 39 pages, though the presentation is dense.
- **Value**: ⭐⭐⭐⭐⭐ — Makes important contributions to the theoretical foundations of imitation learning and clearly answers the core question of "the value of online interaction."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Model Predictive Adversarial Imitation Learning for Planning from Observation](model_predictive_adversarial_imitation_learning_for_planning_from_observation.md)
- [\[ICLR 2026\] On Discovering Algorithms for Adversarial Imitation Learning](on_discovering_algorithms_for_adversarial_imitation_learning.md)
- [\[ICLR 2026\] Latent Wasserstein Adversarial Imitation Learning](latent_wasserstein_adversarial_imitation_learning.md)
- [\[NeurIPS 2025\] A Near-optimal, Scalable and Parallelizable Framework for Stochastic Bandits Robust to Adversarial Corruptions and Beyond](../../NeurIPS2025/reinforcement_learning/a_nearoptimal_scalable_and_parallelizable_framework_for_stoc.md)
- [\[ICLR 2026\] ROMI: Model-based Offline RL via Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting](model-based_offline_rl_via_robust_value-aware_model_learning_with_implicitly_dif.md)

</div>

<!-- RELATED:END -->
