---
title: >-
  [Paper Note] ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation
description: >-
  [ICLR 2026][Reinforcement Learning][Offline RL] ReFORM is proposed to manipulate the source distribution of a behavior cloning (BC) flow policy by learning a reflected flow noise generator…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Offline RL"
  - "Flow Matching"
  - "Support Constraint"
  - "Reflected Flows"
  - "OOD Problem"
date: 2026-05-08
content_hash: 7536dd252d950efa
---

# ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation

**Conference**: ICLR 2026
**arXiv**: [2602.05051](https://arxiv.org/abs/2602.05051)  
**Code**: [Project Page](https://mit-realm.github.io/reform/)  
**Area**: Reinforcement Learning
**Keywords**: Offline RL, Flow Matching, Support Constraint, Reflected Flows, OOD Problem

## TL;DR

ReFORM is proposed to manipulate the source distribution of a behavior cloning (BC) flow policy by learning a reflected flow noise generator, achieving support constraints in a **constructive manner** that avoids OOD issues while preserving policy expressiveness, without requiring hyperparameter tuning.

## Background & Motivation

Offline reinforcement learning faces two core challenges: (1) **OOD problem**—policies generate actions absent from the dataset, causing overly optimistic Q-function estimates; (2) **multimodal action distributions**—conventional unimodal Gaussian policies fail to represent complex multimodal behaviors in the dataset.

Prior methods primarily constrain the learned policy to remain close to the behavior policy by regularizing statistical distances (KL divergence, Wasserstein distance, etc.), but suffer from fundamental limitations:
- **KL divergence is overly restrictive** (Proposition 1): KL constraints provide sufficient but not necessary conditions for support constraints, potentially over-restricting policy improvement.
- **Wasserstein distance is insufficient** (Proposition 2): Wasserstein constraints cannot guarantee support constraints.
- Both introduce a hyperparameter $\alpha$ that requires separate tuning for different tasks and datasets.

The core idea of this paper: rather than constraining statistical distances, directly ensure the support constraint $\text{supp}(\pi_\theta(\cdot|s)) \subseteq \text{supp}(\pi_\beta(\cdot|s))$ via a **constructive approach** by optimizing within the bounded source distribution space of the BC flow policy, naturally satisfying the constraint.

## Method

### Overall Architecture

ReFORM consists of two stages: (1) learning a BC flow policy that maps a bounded uniform distribution to the behavior policy distribution; (2) learning a reflected flow noise generator that manipulates the input noise of the BC policy to maximize Q-values while preserving the support.

### Key Designs

1. **BC Flow Policy with Bounded Source Distribution**:

    - Function: Learn a flow policy $\psi_{\theta_1}(t,z;s)$ that maps a bounded uniform distribution $q_{BC} = \mathcal{U}(\mathcal{B}_l^d)$ to the behavior policy.
    - Mechanism: A uniform distribution over the $d$-dimensional ball is selected as the source distribution, such that $\text{supp}(q_{BC}) = \mathcal{B}_l^d = \{z \in \mathbb{R}^d \mid \|z\| \leq l\}$. A linear flow matching loss $\mathcal{L}_{BC}(\theta_1) = \mathbb{E}[\|v_{\theta_1}(t,x_t;s) - (a-z)\|^2]$ is employed.
    - Design Motivation: The bounded source distribution ensures that its image can approximate the support of the behavior policy, laying the foundation for the constructive support constraint.

2. **Reflected Flow Noise Generator**:

    - Function: Learn a second flow model $\psi_{\theta_2}(t,w;s): \mathcal{B}_l^d \to \mathcal{B}_l^d$ that generates multimodal noise within the ball.
    - Mechanism: A reflected ODE $d\psi_{\theta_2} = v_{\theta_2}dt + dL_t$ is used, where the reflection term $dL_t$ compensates velocity components that would exceed the ball boundary. The reflected Euler method is implemented via projection: when $\hat{z}_{k+1} \notin \mathcal{B}_l^d$, the normal component is subtracted.
    - Design Motivation: Ordinary flow models have unbounded support and cannot guarantee that generated noise remains within the ball; reflected flows both guarantee the support constraint (Theorem 1) and maintain multimodal expressiveness (superior to truncated Gaussians or tanh squashing).

3. **Policy Distillation**:

    - Function: Distill the multi-step BC flow policy into a single-step mapping $\hat{\mu}_{\hat{\theta}_1}$.
    - Mechanism: $\mathcal{L}_{\text{Distill}}(\hat{\theta}_1) = \mathbb{E}[\|\mu_{\hat{\theta}_1}(z;s) - \mu_{\theta_1}(z;s)\|^2]$
    - Design Motivation: Reduces the length of the backpropagation chain, accelerating gradient computation for the noise generator.

### Loss & Training

The noise generator optimization objective is: $\mathcal{L}_{NG}(\theta_2) = \mathbb{E}_{s,w}[-Q^{\mu_\theta}(s, \mu_{\theta_1}(\mu_{\theta_2}(w;s);s))]$, i.e., maximizing the Q-value of the composed policy without any regularization term.

## Key Experimental Results

### Main Results (OGBench, 40 Tasks, Performance Profile)

| Method | Clean Dataset | Noisy Dataset | Hyperparameter Tuning |
|--------|--------------|---------------|-----------------------|
| ReFORM | **Best** | **Best** | Fixed hyperparameters |
| FQL(M) | Second | Significant drop | Manual tuning |
| DSRL | Third | Significant drop | Manual tuning |
| FQL(S) | Moderate | Second | Manual tuning |
| IFQL | Poor | Poor | — |

### Ablation Study

| Configuration | Normalized Score | Notes |
|---------------|-----------------|-------|
| ReFORM (Full) | Highest | Bounded source + reflected flow + distillation |
| ReFORM(U): Gaussian source | Near zero | Unbounded source causes severe OOD |
| ReFORM(MLP): MLP noise generator | Notable drop | Cannot represent multimodal distributions |
| ReFORM(tanh): tanh squashing | Drop | Gradient vanishing issues |
| ReFORM(Gaussian): truncated Gaussian | Drop | Unimodal limitation |
| ReFORM(NoDistill) | Slight drop | Long BPTT chain is detrimental |

### Key Findings
- ReFORM achieves the highest proportion of tasks with normalized scores near 1.0, indicating no upper bound restriction on policy improvement.
- A toy example clearly demonstrates that ReFORM simultaneously reaches both Q-value peaks without boundary violation, whereas DSRL collapses to a single mode.
- The bounded source distribution is the core design—switching to a Gaussian source causes performance collapse.

## Highlights & Insights

- **Theoretical proofs** establish that KL divergence is overly restrictive and Wasserstein distance is insufficient, with support constraints representing a more principled intermediate choice.
- The **constructive approach** completely eliminates the burden of regularization hyperparameter tuning—all 40 tasks share the same set of hyperparameters.
- The introduction of **reflected flows** not only resolves the constraint problem but also preserves the multimodal expressiveness of flow models.

## Limitations & Future Work

- Training the noise generator still requires BPTT through the BC policy, incurring significant computational cost.
- The quality of the support constraint depends on the accuracy of the support learned by the BC model.
- When the dataset contains expert demonstrations, the method converges more slowly than statistical distance approaches due to the absence of explicit regularization.

## Related Work & Insights

- Forms a direct comparison with Wagenmaker et al. (2025)'s DSRL method: both manipulate the noise space, but ReFORM eliminates the hyperparameter requirement through a bounded source distribution.
- Reflected flows (Xie et al., 2024) are applied to constraint satisfaction in RL for the first time.
- The noise manipulation paradigm generalizes to safety constraints in online RL and fine-tuning of diffusion policies.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of reflected flows and bounded source distribution for constructive support constraints is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 40 tasks × two dataset types, detailed ablations, and clear toy example visualizations.
- Writing Quality: ⭐⭐⭐⭐⭐ Logically coherent progression from theory to method to experiments.
- Value: ⭐⭐⭐⭐⭐ Provides a solution to the OOD problem in offline RL with both theoretical guarantees and practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Value Flows](value_flows.md)
- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](flow_actor-critic_for_offline_reinforcement_learning.md)
- [\[ICLR 2026\] BA-MCTS: Bayes Adaptive Monte Carlo Tree Search for Offline Model-based RL](bayes_adaptive_monte_carlo_tree_search_for_offline_model-based_reinforcement_lea.md)
- [\[ICLR 2026\] ROMI: Model-based Offline RL via Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting](model-based_offline_rl_via_robust_value-aware_model_learning_with_implicitly_dif.md)
- [\[ICLR 2026\] Less is More: Clustered Cross-Covariance Control for Offline RL](less_is_more_clustered_cross-covariance_control_for_offline_rl.md)

</div>

<!-- RELATED:END -->
