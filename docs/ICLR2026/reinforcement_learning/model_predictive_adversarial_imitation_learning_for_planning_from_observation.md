---
title: >-
  [Paper Note] Model Predictive Adversarial Imitation Learning for Planning from Observation
description: >-
  [ICLR 2026][Reinforcement Learning][Adversarial Imitation Learning] This paper proposes MPAIL (Model Predictive Adversarial Imitation Learning), which embeds an MPPI planner natively into the adversarial imitation learning loop, achieving the first end-to-end Planning-from-Observation (PfO) framework. MPAIL comprehensively outperforms policy-based AIL methods in generalization, robustness, interpretability, and sample efficiency, and is successfully deployed on a real-world robot navigation task from a single observed demonstration.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Adversarial Imitation Learning
  - Model Predictive Control
  - Inverse Reinforcement Learning
  - Learning from Observation
  - MPPI
date: 2026-05-08
content_hash: 07dcbf26e610013e
---

# Model Predictive Adversarial Imitation Learning for Planning from Observation

**Conference**: ICLR 2026
**arXiv**: [2507.21533](https://arxiv.org/abs/2507.21533)
**Code**: None
**Area**: Imitation Learning / Robot Planning
**Keywords**: Adversarial Imitation Learning, Model Predictive Control, Inverse Reinforcement Learning, Learning from Observation, MPPI

## TL;DR
This paper proposes MPAIL (Model Predictive Adversarial Imitation Learning), which embeds an MPPI planner natively into the adversarial imitation learning loop, achieving the first end-to-end Planning-from-Observation (PfO) framework. MPAIL comprehensively outperforms policy-based AIL methods in generalization, robustness, interpretability, and sample efficiency, and is successfully deployed on a real-world robot navigation task from a single observed demonstration.

## Background & Motivation

- **Background**: Inverse reinforcement learning (IRL) enables imitation learning by inferring reward functions from expert behavior, and has been widely applied to autonomous driving, social navigation, and path planning. In high-dimensional continuous control, learned IRL rewards are typically deployed via model predictive control (MPC) — first solving IRL offline with RL, then planning online with MPC — forming the dominant "IRL-then-MPC" paradigm. Meanwhile, adversarial imitation learning (AIL, e.g., GAIL) has achieved notable advances in algorithmic complexity and sample efficiency.

- **Limitations of Prior Work**: (1) IRL-then-MPC is a two-stage disjoint process: the inner-loop policy used during training is completely decoupled from the MPC planner used at deployment, so the learned reward is never optimized for MPC-based deployment and requires additional manual tuning; (2) Policy-based AIL methods (e.g., GAIL, AIRL) rely on black-box RL policy networks, making it difficult to impose safety constraints, lack interpretability, and are brittle in partially observable real-world settings; (3) Learned reward and value functions are severely underutilized in policy-based AIL — only the policy network is used at deployment, while the reward function is entirely discarded.

- **Key Challenge**: There is a fundamental disconnect between the theoretical advantages of AIL (unified reward learning and policy optimization) and the practical requirements of robotic deployment (safety, interpretability, and online optimization capability provided by MPC).

- **Goal**: To natively embed planning (MPC) into the AIL loop, realizing an end-to-end "learning planner" that simultaneously learns a reward function and improves a planning-based agent, requiring only observed states (no expert actions).

- **Key Insight**: The observation that the objective of MPPI (Model Predictive Path Integral) control is naturally a KL-constrained cost minimization problem, which is mathematically equivalent to the maximum-entropy RL objective of the AIL inner loop — implying that MPPI can directly replace the RL policy as the "generator" in AIL.

- **Core Idea**: Replace the RL policy in AIL with an MPPI planner that solves for a new policy online at each timestep (a "deconstructed policy"), while learning a discriminator as a cost function and a value function for reasoning beyond the planning horizon. No persistent policy network is required; instead, the reward function must generalize. Since reward functions are typically simpler and more structurally regularized than policy functions, this naturally yields better out-of-distribution generalization.

## Method

### Overall Architecture
The training pipeline of MPAIL resembles GAIL but differs fundamentally: (1) the MPPI planner samples trajectories in the environment — rolling out a set of action sequences, evaluating each trajectory using the discriminator cost, and computing the optimal action via importance-weighted averaging; (2) the discriminator is trained with BCE loss to distinguish agent state transitions from expert state transitions; (3) the value network is fitted to Monte Carlo returns as a terminal cost. Unlike GAIL, no policy update step is needed after updating the value network — the policy is solved online by MPPI at every state.

### Key Designs

1. **MPPI as the AIL Generator (Core Theoretical Contribution)**:

    - Function: Embed MPPI into AIL to replace the RL policy, making the planner itself the generator in adversarial learning.
    - Mechanism: The AIL inner-loop RL objective is $\min_\pi \mathbb{E}_\pi[c(s,s')] + \beta \text{KL}(\pi || \bar{\pi})$, whose closed-form solution is $\pi^*(a|s) \propto \bar{\pi}(a|s) e^{-\frac{1}{\beta}\bar{c}(s,a)}$. MPPI solves the trajectory-level equivalent: $\min_\pi \mathbb{E}_{\tau \sim \pi}[C(\tau) + \beta \text{KL}(\pi(\tau) || \bar{\pi}(\tau))]$, which is equivalent to the AIL objective under uniformly ergodic MDP conditions.
    - Design Motivation: MPPI's zeroth-order optimization requires no gradient backpropagation through a policy network; instead, it samples a large number of trajectories at each timestep and computes the optimal action via weighted averaging. This "online policy solving" approach eliminates the need for policy network generalization — only the reward function needs to generalize. Since reward functions are typically simpler and more structurally regularized than policy functions, this naturally yields better OOD generalization.

2. **Observation-Only State-Transition Cost Function**:

    - Function: Define the cost function over state transitions $(s, s')$ rather than state-action pairs $(s, a)$, enabling learning from observation only.
    - Mechanism: The discriminator is $D(s, s') = \sigma \circ d_\theta(s, s')$, and the reward is defined as the discriminator logit: $r(s, s') = \log D(s,s') - \log(1 - D(s,s')) = d_\theta(s,s')$. An AIRL-style reward formulation is adopted, which provides greater stability when combined with the value function.
    - Design Motivation: In real-world robotics, expert actions are often unobservable or difficult to obtain (e.g., learning from video), making observation-only learning the most general setting. Moreover, under partial observability, $(s, s')$ can encode information such as movement direction that a single state $s$ cannot capture.

3. **Infinite-Horizon MPPI with Value Guidance**:

    - Function: Use the learned value function $V_\phi(s)$ as a terminal cost for MPPI rollouts, extending short-horizon planning to long-horizon reasoning.
    - Mechanism: The value function estimates $G_t = \mathbb{E}_\pi[R_{t+1} + \gamma R_{t+2} + ... | S_t = s_t]$ and is updated via TD learning: $\nabla_\phi \mathbb{E}[(G_t - V_\phi(s))^2]$. Adding $V_\phi$ to the cost of terminal states in MPPI allows the planner to "see" beyond the rollout horizon.
    - Design Motivation: A pure MPPI rollout has limited reach (e.g., 3 meters), while the task may require navigating to a goal 40 meters away. The value function provides experience-based evaluation beyond the planning horizon, endowing the short-sighted planner with global awareness.

### Loss & Training
- **Discriminator Loss**: Standard BCE — $\nabla_\theta [\mathbb{E}_{d^\pi}[\log D_\theta(s,s')] + \mathbb{E}_{d^{\pi_E}}[\log(1 - D_\theta(s,s'))]]$
- **Value Function Loss**: MSE against Monte Carlo returns — $\nabla_\phi \mathbb{E}[(G_t - V_\phi(s))^2]$; GAE-$\lambda$ estimation is uniformly applied across all methods.
- **No Policy Update Required**: Unlike GAIL/AIRL, MPAIL proceeds directly to online MPPI solving after updating the discriminator and value network, without any policy gradient step.
- **Temperature Annealing**: The MPPI temperature $\lambda$ can be gradually reduced during training to prevent premature distribution collapse.
- **Stabilization**: Spectral normalization is applied to the discriminator; hyperparameters are kept consistent across all experiments.

## Key Experimental Results

### Main Results

**Real-World Navigation (Real-Sim-Real, learning from a single observed trajectory)**:

| Method | Max CTE (m) | Mean CTE (m) | Mean Speed (m/s) |
|--------|-------------|--------------|------------------|
| Expert | - | - | 1.0 |
| GAIL | 1.29 | 0.56 | 0.37 |
| IRL-MPC | 1.28 | 0.37 | 0.30 |
| **MPAIL** | **0.76** | **0.17** | **0.70** |

MPAIL achieves a mean cross-track error of only 0.17 m, 70% lower than GAIL, while maintaining a speed of 0.70 m/s — nearly twice that of GAIL and closest to the expert's 1.0 m/s. GAIL consistently drifts off-track or spins in place during real-world deployment, failing across multiple initial configurations.

### Ablation Study

**OOD Generalization (Navigation task, initial position expanded from 1×1 to 40×40 m)**:

| Method | ID Performance | Near OOD | Far OOD | Extreme OOD |
|--------|---------------|----------|---------|-------------|
| GAIL (policy-based) | Good | Poor | Very Poor | Random |
| BC (requires actions) | Moderate | Poor | Very Poor | Random |
| MPAIL (prior model) | Good | Good | Good | Still navigable |
| MPAIL (online model) | Good | Good | Moderate | Reachable but longer path |

MPAIL's planning horizon is only 3 meters, yet the task horizon can be up to 15× larger — demonstrating that the learned cost and value functions successfully generalize to OOD states. Policy networks fail even when the agent starts facing the goal but slightly outside the data distribution, revealing extremely brittle representations.

**Efficiency Comparison (Navigation + CartPole)**:

| Method | Navigation-4 demos | Navigation-convergence | CartPole |
|--------|--------------------|------------------------|----------|
| GAIL | Converges | Slow (2× interactions) | Fastest |
| AIRL | Does not converge | - | Comparable to MPAIL |
| MPAIL | Converges | Fast (<50% interactions) | Comparable |

### Key Findings
- **Reward Deployment Is Critical**: Policy-based AIL learns a reward but completely discards it at deployment — a fundamental limitation. MPAIL reintroduces the reward online, shifting the generalization burden from the policy to the reward function.
- **End-to-End Training Outperforms Disjoint Deployment**: IRL-MPC uses the same reward and value as GAIL but switches to MPPI at deployment — already significantly outperforming GAIL, yet still inferior to MPAIL. The reason is that MPAIL's end-to-end training forces the discriminator to reach a higher level of quality.
- **Policy-Based AIL Is Extremely Brittle in the Real World**: The performance gap of GAIL between simulation and real-world deployment far exceeds expectations. Partial observability causes the reward signal to be extremely weak (cost range $(-0.022, -0.018)$ vs. $(-3, 3)$ in simulation), which the policy network cannot handle.
- **Sample Efficiency of MPPI Zeroth-Order Optimization**: Although MPPI does not backpropagate gradients to a policy, it converges more than twice as fast as GAIL on the navigation task — validating MPAIL's sample efficiency advantage as a model-based method.
- **Wall Clock Time**: MPPI with 2 iterations is approximately 10% faster than GAIL (PPO); with 5 iterations, approximately 10% slower — the practical computational overhead is manageable.

## Highlights & Insights
- **Mathematical Unification of IRL and MPC**: MPPI's KL-constrained trajectory optimization objective is strictly equivalent to the maximum-entropy RL objective of the AIL inner loop under uniformly ergodic MDP conditions. This is not merely an engineering integration, but reveals a deep mathematical connection between control theory and adversarial learning — enabling the previously disjoint training and deployment pipelines to be unified into a single framework.
- **The Philosophy of the "Deconstructed Policy"**: MPAIL deconstructs the policy into more fundamental components (cost + value + model + online optimizer), each of which can be inspected and modified independently. This transparency is essential for safety-critical systems — one can directly examine why a planned path incurs low cost and why a particular decision is made.
- **A Paradigm Shift in Generalization**: Conventional AIL requires the policy network to generalize to unseen states, whereas MPAIL instead requires the reward function to generalize. Since reward functions encode "intent" rather than "execution," they tend to have lower complexity and better structure, making them inherently easier to generalize.

## Limitations & Future Work
- **No Latent-Space Planning**: The current MPAIL performs MPPI rollouts directly in state space without a latent dynamics model. In high-dimensional spaces (e.g., image observations), MPPI sampling efficiency degrades sharply, requiring extensions such as latent-state planning as in TD-MPC2.
- **Lack of Theoretical Grounding for Temperature Annealing**: The paper acknowledges that the temperature annealing schedule is currently heuristic — effective in practice but without theoretical justification.
- **No Efficiency Advantage on CartPole**: MPAIL with an online-learned dynamics model is less sample-efficient than GAIL on CartPole, likely due to the combined effect of sparse reward signals, model bias, and additional exploration requirements.
- **Validation Limited to Simple Navigation Tasks**: Real-world experiments are conducted solely on RC car navigation; more complex manipulation tasks (e.g., grasping, bimanual coordination) are not evaluated.
- **No Policy Prior**: The current MPAIL does not use a policy-like sampling prior, limiting MPPI scalability to high-dimensional action spaces.

## Related Work & Insights
- **vs. GAIL**: GAIL uses a PPO policy as the AIL generator and discards the reward at deployment, retaining only the policy. MPAIL demonstrates that this approach has a fundamental flaw — the learned reward is severely wasted and the policy network fails to generalize to OOD states. GAIL fails completely in real-world experiments while MPAIL succeeds.
- **vs. IRL-MPC**: IRL-MPC represents the current dominant paradigm — training a reward via GAIL/IRL and then manually porting it to MPC for deployment. MPAIL shows that end-to-end training is substantially superior to disjoint deployment: IRL-MPC's reward and value are derived directly from GAIL training, which is insufficiently optimized because the reward was never challenged by MPPI during training.
- **vs. TD-MPC2**: TD-MPC2 is a state-of-the-art model-based RL method using latent-state planning. MPAIL currently operates in state space, but its framework is naturally compatible with latent dynamics extensions, which the paper explicitly identifies as a future direction.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The mathematical equivalence between MPPI and AIL and the end-to-end PfO framework constitute entirely original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers simulation navigation, real-world RC car deployment, OOD evaluation, efficiency comparisons, and wall clock time analysis, though more complex tasks are absent.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear and rigorous; the logical chain connecting experimental motivation and conclusions is complete.
- Value: ⭐⭐⭐⭐⭐ Provides direct and practical value for imitation learning in safety-critical systems; open-source implementation lowers the barrier to reproduction.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Near-Optimal Second-Order Guarantees for Model-Based Adversarial Imitation Learning](near-optimal_second-order_guarantees_for_model-based_adversarial_imitation_learn.md)
- [\[ICLR 2026\] On Discovering Algorithms for Adversarial Imitation Learning](on_discovering_algorithms_for_adversarial_imitation_learning.md)
- [\[ICLR 2026\] Latent Wasserstein Adversarial Imitation Learning](latent_wasserstein_adversarial_imitation_learning.md)
- [\[AAAI 2026\] Language Model Distillation: A Temporal Difference Imitation Learning Perspective](../../AAAI2026/reinforcement_learning/language_model_distillation_a_temporal_difference_imitation_learning_perspective.md)
- [\[ICLR 2026\] One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)

<!-- RELATED:END -->
