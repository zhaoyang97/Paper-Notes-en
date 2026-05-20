---
title: >-
  [Paper Note] Scalable In-Context Q-Learning
description: >-
  [ICLR 2026][Reinforcement Learning][In-Context RL] This paper proposes S-ICQL, which introduces dynamic programming (Q-learning) and world models into the supervised ICRL framework. A multi-head Transformer simultaneousl…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "In-Context RL"
  - "Q-Learning"
  - "World Model"
  - "Dynamic Programming"
  - "Efficient Prompting"
date: 2026-05-08
content_hash: a00a09afafc98dd8
---

# Scalable In-Context Q-Learning

**Conference**: ICLR 2026
**arXiv**: [2506.01299](https://arxiv.org/abs/2506.01299)  
**Code**: [GitHub](https://github.com/NJU-RL/SICQL)  
**Area**: Reinforcement Learning / In-Context Learning
**Keywords**: In-Context RL, Q-Learning, World Model, Dynamic Programming, Efficient Prompting

## TL;DR

This paper proposes S-ICQL, which introduces dynamic programming (Q-learning) and world models into the supervised ICRL framework. A multi-head Transformer simultaneously predicts the policy and in-context value functions, a pretrained world model constructs lightweight yet accurate prompts, and advantage-weighted regression is used for policy extraction. S-ICQL consistently outperforms all baselines when learning from suboptimal data in both discrete and continuous environments.

## Background & Motivation

**Background**: In-context RL (ICRL) extends the in-context learning capabilities of language models to decision-making — pretraining a Transformer on multi-task offline data and adapting to new tasks at test time via prompts without parameter updates. Existing methods fall into two branches: Algorithm Distillation (AD, which uses learning histories as context for autoregressive action prediction) and Decision-Pretrained Transformer (DPT, which predicts optimal actions from interaction trajectory sequences).

**Limitations of Prior Work**:

- **Inherent limitations of supervised pretraining**: AD and DPT are essentially imitation learning approaches and cannot surpass the quality of the data they are trained on — they lack *stitching* ability (the ability to combine suboptimal trajectory segments into globally optimal behavior).
- **AD requires long-horizon context**: It relies on complete learning histories as prompts and inherits the gradient update rules of suboptimal behavior.
- **DPT requires oracle optimal action annotations**: This is often infeasible in practice.
- **Raw trajectories as prompts are inefficient**: They contain a large number of tokens with high redundancy, and behavioral policy information is entangled with task information, leading to biased task inference.

**Key Challenge**: The essence of RL lies in reward maximization through dynamic programming updates of value functions. However, existing ICRL methods remain entirely within the supervised learning paradigm, forgoing the core advantages of RL. The key challenge is how to introduce dynamic programming to unlock the potential for learning from suboptimal data while preserving the scalability and stability of supervised pretraining.

**Goal**: The paper leverages two fundamental properties of RL — (1) the stitching capability of dynamic programming (Bellman backup) and (2) the precise characterization of environment dynamics via world models — to design a scalable ICRL framework that simultaneously achieves efficient reward maximization and accurate task generalization.

## Method

### Overall Architecture

S-ICQL comprises three core components: (a) a pretrained universal world model that compresses raw trajectories into lightweight task prompts $\beta$; (b) a multi-head Transformer that simultaneously predicts the policy $\pi_\theta(a|s;\beta)$, state value function $V_\theta(s;\beta)$, and action value function $Q_\theta(s,a;\beta)$; and (c) a joint optimization objective combining Bellman backup (Q-learning) and advantage-weighted regression (policy extraction).

The problem is formulated as multi-task offline RL: tasks $M^i = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}^i, \mathcal{R}^i, \gamma \rangle \sim P(M)$ share the state-action space but differ in reward functions or transition dynamics. Each task's offline dataset $\mathcal{D}^i$ is collected by an arbitrary behavioral policy.

### Key Design 1: World Model → Lightweight and Accurate Prompts

The core insight is that environment dynamics $p(s', r | s, a)$ fully characterize a decision-making task and are inherently unaffected by the behavioral policy. Therefore, encoding task information via a world model is more precise and compact than using raw trajectories directly.

**World Model Architecture**: Consists of a context encoder $E_\phi$ and a dynamics decoder $D_\varphi$:

- Context encoder: compresses the recent $k$-step experience $\eta_t^i = (s_{t-k}, a_{t-k}, r_{t-k}, \ldots, s_t, a_t)^i$ into a task representation $z_t^i = E_\phi(\eta_t^i)$
- Dynamics decoder: conditioned on the task representation, predicts the immediate reward and next state $[\hat{r}_t, \hat{s}_{t+1}] = D_\varphi(s_t, a_t; z_t^i)$

**Pretraining objective** minimizes prediction errors for rewards and state transitions:

$$\mathcal{L}(\phi, \varphi) = \mathbb{E}_{\eta_t^i \sim M^i} \left[ \| [r_t, s_{t+1}] - D_\varphi(s_t, a_t; z_t^i) \|_2^2 \mid z_t^i = E_\phi(\eta_t^i) \right]$$

After pretraining, the world model is frozen and converts $h$-step trajectories into lightweight prompts:

$$\beta^i := [z_1^i, z_2^i, \ldots, z_h^i] = [E_\phi(\eta_1^i), E_\phi(\eta_2^i), \ldots, E_\phi(\eta_h^i)]$$

Compared to AD, which requires long learning histories as context, this prompt structure is more compact and encodes more precise task information.

### Key Design 2: In-Context Q-Learning (Bellman Backup + Expectile Regression)

**Q-function training** — minimizes Bellman error to introduce stitching capability:

$$\mathcal{L}_Q(\theta) = \mathbb{E}_{(s_t^i, a_t^i, s_{t+1}^i) \sim \mathcal{D}^i} \left[ \left( r(s_t^i, a_t^i) + \gamma V_\theta(s_{t+1}^i; \beta^i) - Q_\theta(s_t^i, a_t^i; \beta^i) \right)^2 \right]$$

**State value function training** — uses expectile regression to fit the upper quantile of the Q-function:

$$\mathcal{L}_V(\theta) = \mathbb{E}_{(s_t^i, a_t^i) \sim \mathcal{D}^i} \left[ L_2^\omega \left( Q_{\hat{\theta}}(s_t^i, a_t^i; \beta^i) - V_\theta(s_t^i; \beta^i) \right) \right]$$

where $L_2^\omega(u) = |\omega - \mathbb{1}(u < 0)| \cdot u^2$ is an asymmetric loss function with $\omega \in (0.5, 1)$. When $Q > V$, a larger weight $\omega$ is assigned; when $Q < V$, the weight is only $1-\omega$, thereby approximating $\max_a Q(s, a)$.

### Key Design 3: Advantage-Weighted Regression Policy Extraction

The in-context value functions are distilled into policy extraction via advantage-weighted regression:

$$\mathcal{L}_\pi(\theta) = -\mathbb{E}_{(s_t^i, a_t^i) \sim \mathcal{D}^i} \left[ \exp\left( \frac{1}{\lambda} \left( Q_{\hat{\theta}}(s_t^i, a_t^i; \beta^i) - V_\theta(s_t^i; \beta^i) \right) \right) \cdot \log \pi_\theta(a_t^i | s_t^i; \beta^i) \right]$$

Actions with larger advantage values $A = Q - V$ receive larger training weights. This is not simple behavior cloning but rather learning a policy that maximizes Q-values subject to dataset constraints. The total loss is a weighted sum of all three terms:

$$\mathcal{L}(\theta) = \mathsf{c}_1 \mathcal{L}_\pi(\theta) + \mathsf{c}_2 \mathcal{L}_Q(\theta) + \mathsf{c}_3 \mathcal{L}_V(\theta)$$

Coefficients are set to $(1:1:1)$, and the entire multi-head Transformer is jointly optimized end-to-end.

## Key Experimental Results

### Main Results: Few-Shot Evaluation on Mixed Datasets

| Method | DarkRoom | Push | Reach | Cheetah-Vel | Walker-Param | Ant-Dir |
|:-------|:---------|:-----|:------|:------------|:-------------|:--------|
| DPT | 22.12 | 362.74 | 736.72 | -78.35 | 257.11 | 591.31 |
| AD | 42.72 | 604.50 | 738.96 | -67.37 | 424.82 | 215.01 |
| IDT | 40.70 | 621.58 | 790.68 | -59.46 | 343.01 | 631.83 |
| DICP | 59.76 | 487.28 | 706.46 | -66.53 | 403.90 | 745.05 |
| DIT | 30.90 | 633.58 | 758.92 | -74.50 | 253.94 | 723.49 |
| IC-IQL | 60.12 | 646.08 | 773.33 | -56.53 | 391.38 | 713.26 |
| **S-ICQL** | **66.05** | **653.04** | **806.97** | **-35.48** | **466.72** | **813.34** |

S-ICQL achieves the best performance across all 6 environments. The advantage is particularly pronounced in complex environments (Cheetah-Vel, Ant-Dir), reducing the error from -56.53 to -35.48 (a 37% improvement) and improving from 745.05 to 813.34, respectively.

### Ablation Study: Component Contribution Analysis

| Ablation Configuration | Reach | Cheetah-Vel | Ant-Dir |
|:-----------------------|:------|:------------|:--------|
| w/o\_cq (no world model + no Q-learning = DPT) | 736.72 | -78.35 | 591.31 |
| w/o\_c (no world model) | 792.09 | -56.19 | 693.87 |
| w/o\_q (no Q-learning) | 752.41 | -63.66 | 784.07 |
| **S-ICQL (full)** | **806.97** | **-35.48** | **813.34** |

Both the world model and Q-learning components contribute significant gains individually. Removing either component leads to performance degradation, and removing both degrades the model to DPT, yielding the worst performance. On Ant-Dir, Q-learning contributes a larger gain (+29 vs. w/o\_q), highlighting the importance of stitching in complex tasks.

### OOD Generalization

| Method | Cheetah-Vel (OOD) | Ant-Dir (OOD) |
|:-------|:-----------------|:-------------|
| DPT | -137.26 | 205.29 |
| IC-IQL | -101.89 | 540.20 |
| **S-ICQL** | **-83.45** | **664.95** |

S-ICQL also substantially outperforms baselines on out-of-distribution tasks, surpassing the second-best method IC-IQL by approximately 23% on Ant-Dir, validating the OOD generalization capability conferred by the world model.

## Rating

**Rating**: ⭐⭐⭐⭐⭐

**Strengths**:

- The paper innovatively incorporates two core RL concepts — dynamic programming (Q-learning stitching) and world models — into ICRL, addressing the fundamental limitation that supervised pretraining cannot surpass the quality of the collected data.
- The multi-head Transformer architecture is elegantly designed — adding only two lightweight heads enables simultaneous prediction of the policy and value functions with negligible parameter overhead.
- The world-model-driven prompt construction method is principled and theoretically grounded, as environment dynamics are inherently unaffected by the behavioral policy.
- The experiments are highly comprehensive: 6 standard environments + 2 complex environments + OOD generalization + stitching verification + 7 competitive baselines.

**Limitations**:

- Prompt length is tied to the sampled trajectory length, which may become excessively long in long-horizon interaction problems.
- The world model requires an additional pretraining stage, increasing the complexity of the overall training pipeline.
- Validation is limited to standard RL benchmarks, with no evaluation on more complex real-world decision-making scenarios (e.g., sim-to-real transfer for robot manipulation).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)
- [\[NeurIPS 2025\] Towards Provable Emergence of In-Context Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/towards_provable_emergence_of_in-context_reinforcement_learning.md)
- [\[ICLR 2026\] Chain-of-Context Learning: Dynamic Constraint Understanding for Multi-Task VRPs](chain-of-context_learning_dynamic_constraint_understanding_for_multi-task_vrps.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)

</div>

<!-- RELATED:END -->
