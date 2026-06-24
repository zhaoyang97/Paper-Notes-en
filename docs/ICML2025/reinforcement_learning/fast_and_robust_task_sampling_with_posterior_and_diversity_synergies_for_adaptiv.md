---
title: >-
  [Paper Note] Fast and Robust: Task Sampling with Posterior and Diversity Synergies for Adaptive Decision-Makers in Randomized Environments
description: >-
  [ICML2025][Reinforcement Learning][Robust Active Task Sampling (RATS)] Proposes PDTS (Posterior and Diversity Synergized Task Sampling), modeling robust active task sampling as an infinite-armed bandit problem. By replacing UCB with posterior sampling and introducing diversity regularization, it minimalistly achieves near-worst-case robust adaptation performance in Domain Randomization and Meta-RL.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "Robust Active Task Sampling (RATS)"
  - "posterior sampling"
  - "diversity regularization"
  - "CVaR"
  - "infinite-armed bandit"
date: 2026-05-08
content_hash: 97cc62101fae5ab5
---

# Fast and Robust: Task Sampling with Posterior and Diversity Synergies for Adaptive Decision-Makers in Randomized Environments

**Conference**: ICML2025  
**arXiv**: [2504.19139](https://arxiv.org/abs/2504.19139)  
**Code**: [Project Page](https://thu-rllab.github.io/PDTS_project_page)  
**Area**: Meta-Learning / Reinforcement Learning (Meta-RL, Domain Randomization)  
**Keywords**: Robust Active Task Sampling (RATS), posterior sampling, diversity regularization, CVaR, infinite-armed bandit

## TL;DR
Proposes PDTS (Posterior and Diversity Synergized Task Sampling), modeling robust active task sampling as an infinite-armed bandit problem. By replacing UCB with posterior sampling and introducing diversity regularization, it minimalistly achieves near-worst-case robust adaptation performance in Domain Randomization and Meta-RL.

## Background & Motivation

In DR (Domain Randomization) and Meta-RL, policies need to perform zero-shot or few-shot adaptation on randomized MDPs. Risk-averse scenarios (e.g., robotic control, autonomous driving) place extremely high requirements on adaptation robustness, and tail-risk principles like CVaR are used to prioritize training on hard tasks.

**Core Bottleneck**: CVaR policies require expensive policy evaluation on a large number of MDPs. Previous MPTS methods use risk prediction models as proxies for evaluation, but suffer from three issues:

**Lack of Theoretical Tools**: Inability to formally analyze robustness concepts

**Concentration Issue**: As the pseudo-batch $\hat{\mathcal{B}}$ increases, Top-$\mathcal{B}$ selection clusters the subset into a narrow region, causing performance collapse

**Hyperparameter Sensitivity**: The UCB acquisition function requires careful tuning of the exploration-exploitation trade-off parameter

## Method

### 1. Task Selection MDP Modeling

Robust task sampling is modeled as a finite-horizon MDP $\mathcal{M} = \langle \mathbf{S}, \mathbf{A}, \mathbf{P}, \mathbf{R} \rangle$:

- **State Space**: Policy parameters $\mathbf{S} = \{\boldsymbol{\theta} \in \boldsymbol{\Theta}\}$
- **Action Space**: Task subsets $\mathbf{A}_t = \{\mathcal{T}_t^{\mathcal{B}} \subseteq \mathcal{T}_t^{\hat{\mathcal{B}}} : |\mathcal{T}_t^{\mathcal{B}}| = \mathcal{B}\}$
- **Reward**: Robustness improvement $R(\boldsymbol{\theta}_t, \mathcal{T}_{t+1}^{\mathcal{B}}) := \text{CVaR}_\alpha(\boldsymbol{\theta}_t) - \text{CVaR}_\alpha(\boldsymbol{\theta}_{t+1})$

This is further simplified to an **infinite-armed bandit (i-MAB)**, where each feasible subset is an arm, and MPTS is proven to be a particular UCB solution of it.

### 2. Diversity-Regularized Acquisition Function

The concentration issue of Top-$\mathcal{B}$ selection is validated through theoretical analysis (Proposition 3.3): as $\hat{\mathcal{B}} \to \infty$, all selected samples concentrate in the $\epsilon$-neighborhood of the global optimum with probability 1. The solution is to introduce a diversity regularization term:

$$\max_{\mathcal{T}^{\mathcal{B}} \subseteq \mathcal{T}^{\hat{\mathcal{B}}}: |\mathcal{T}^{\mathcal{B}}| = \mathcal{B}} \mathcal{A}(\mathcal{T}^{\mathcal{B}}) + \gamma \mathcal{S}[\{d(\boldsymbol{\tau}_i, \boldsymbol{\tau}_j)\}]$$

where $\mathcal{S}$ measures subset diversity (e.g., sum of pairwise distances $\sum_{i,j}\|\boldsymbol{\tau}_i - \boldsymbol{\tau}_j\|_2^2$), and $\gamma$ controls the regularization strength.

**Proposition 3.4**: When $\hat{\mathcal{B}}$ is sufficiently large, this regularized acquisition rule achieves **near-worst-case optimization**.

### 3. Replacing UCB with Posterior Sampling

UCB requires multiple stochastic forward passes to estimate mean and variance, which incurs computational overhead scaling with $\hat{\mathcal{B}}$. PDTS adopts posterior sampling:

$$\boldsymbol{z}_t \sim q_{\boldsymbol{\phi}}(\boldsymbol{z}_t | H_t), \quad \hat{\ell}_{t+1,i} \sim p_{\boldsymbol{\psi}}(\ell | \hat{\boldsymbol{\tau}}_i, \boldsymbol{z}_t)$$

Only **a single forward pass** is required to complete the evaluation of all candidate tasks, followed by solving the diversity-regularized subset selection problem. Posterior sampling naturally possesses randomized optimism, avoiding UCB's over-exploitation of inaccurate uncertainty estimates.

### 4. Overall Workflow

1. Optimize the risk prediction module: Maximize ELBO using historical data $H_{t-1}$
2. Posterior sampling evaluation: Randomly sample $\hat{\mathcal{B}}$ candidate tasks from $p(\tau)$, followed by a single forward pass to obtain risk predictions
3. Diversity-guided subset search: Solve the regularized combinatorial optimization problem using an approximation algorithm, returning $\mathcal{B}$ tasks

## Key Experimental Results

### Meta-RL (MuJoCo, MAML Backbone)

| Scenario | Metric | PDTS | MPTS | DRM | ERM |
|------|------|------|------|-----|-----|
| ReacherPos | CVaR₀.₉ Return | **Best** | Suboptimal | Poor | Baseline |
| Walker2dVel | CVaR₀.₉ Return | **Best, >15%** | Suboptimal | Poor | Baseline |
| HalfCheetahVel | CVaR₀.₉ Return | **Best, >15%** | Suboptimal | Poor | Baseline |

### Compatibility with PEARL Backbone

| Scenario | PDTS | MPTS | RoML | PEARL |
|------|------|------|------|-------|
| HalfCheetahBody (CVaR₀.₉₅) | **993±26** | 945±26 | 855±35 | 847±42 |
| HalfCheetahMass (CVaR₀.₉₅) | **1296±41** | 1209±45 | 1197±59 | 1118±51 |

### Domain Randomization (Physical Robot)

- LunarLander: CVaR₀.₉ is **73%** higher than ERM
- Pusher: Training acceleration by **2.4×**, LunarLander by **1.3×**
- CVaR₀.₉ is **>8%** higher than ERM across all scenarios
- OOD Adaptation: PDTS shows the minimal performance degradation in out-of-distribution (OOD) tasks

### Visual Robotic DR (ManiSkill3)

- Two scenarios: LiftPegUpright_Light (lighting randomization) and AnymalCReach_Goal (target randomization)
- PDTS is optimal in CVaR₀.₅ success rate while demonstrating a training acceleration trend
- Risk prediction model PCC > 0.5, with PDTS even exceeding MPTS in prediction accuracy
- Computational overhead is comparable to ERM (optimal except for DRM)

## Highlights & Insights

1. **Theoretical Elegance**: Modeling the RATS pipeline as a task-selection MDP -> i-MAB, unifying the theoretical foundations of MPTS and PDTS
2. **Simplicity and Efficiency**: PDTS implementation is highly minimalist (posterior sampling + diversity regularization), avoiding multiple forward passes and hyperparameter tuning required by UCB
3. **High Scalability**: Large pseudo-batches of $\hat{\mathcal{B}} = 64\mathcal{B}$ do not lead to performance collapse, which is a major limitation for MPTS
4. **Plug-and-Play**: Compatible with various Meta-RL backbones such as MAML and PEARL
5. **Unexpected Gains**: Not only improves robustness but also accelerates training in some scenarios (Pusher by 2.4×)
6. **OOD Generalization**: Coupled with diversity regularization, it minimizes performance degradation on out-of-distribution tasks

## Limitations & Future Work

1. **Dependency on Risk Prediction Model Quality**: Assumes the smoothness of the adaptation risk function and the availability of identifier information, which may not hold in certain restricted scenarios
2. **NP-hard Combinatorial Optimization in Diversity Regularization**: Although approximation algorithms exist, the gap between theoretical guarantees and practical quality has not been discussed in depth
3. **Experimental Limitations**: Validation is predominantly conducted in continuous control/robotics domains, lacking validation on discrete action spaces or more complex real-world applications
4. **Directions for Improvement**: Designing more precise risk prediction models; integrating stronger robust optimization techniques into the i-MAB framework

## Related Work & Insights

- **MPTS** (Wang et al., 2025b): The core baseline of this work, with PDTS improving upon it both theoretically and practically
- **RoML** (Greenberg et al., 2024): Hard MDP prioritization methods in Meta-RL
- **CVaR/DRM**: Classic paradigms of tail-risk optimization, where PDTS approximates worst-case optimization via a large $\hat{\mathcal{B}}$
- **Thompson Sampling / Posterior Sampling**: Classic bandit strategy, systematically introduced to robust task sampling for the first time

## Rating
- Novelty: ⭐⭐⭐⭐ (i-MAB modeling unifies the theoretical framework + the idea of replacing UCB with posterior sampling is novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Validated across Meta-RL + physical DR + visual DR + OOD + multiple backbones, with 7 seeds)
- Writing Quality: ⭐⭐⭐⭐ (Clear theoretical derivations, complete notation system, well-structured)
- Value: ⭐⭐⭐⭐ (Provides an efficient and simple task sampling scheme for risk-averse RL)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Q-learning with Posterior Sampling](../../ICLR2026/reinforcement_learning/q-learning_with_posterior_sampling.md)
- [\[ICML 2025\] Mastering Massive Multi-Task Reinforcement Learning via Mixture-of-Expert Decision Transformer](mastering_massive_multi-task_reinforcement_learning_via_mixture-of-expert_decisi.md)
- [\[ICML 2025\] Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs](robust_noise_attenuation_via_adaptive_pooling_of_transformer_outputs.md)
- [\[ICML 2025\] Exploring Large Action Sets with Hyperspherical Embeddings using von Mises-Fisher Sampling](exploring_large_action_sets_with_hyperspherical_embeddings_using_von_mises-fishe.md)
- [\[ICML 2025\] Enhancing Decision-Making of Large Language Models via Actor-Critic](enhancing_decision-making_of_large_language_models_via_actor-critic.md)

</div>

<!-- RELATED:END -->
