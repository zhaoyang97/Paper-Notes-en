---
title: >-
  [Paper Note] Confounding Robust Deep Reinforcement Learning: A Causal Approach
description: >-
  [NeurIPS 2025][Reinforcement Learning][confounded MDP] This paper extends DQN via partial identification theory, proposing Causal DQN to learn robust policies from offline data with unobserved confounders—by optimizing a worst-case lower bound on the value function to obtain safe policies—and consistently outperforms standard DQN across 12 confounded Atari games.
tags:
  - NeurIPS 2025
  - Reinforcement Learning
  - confounded MDP
  - off-policy learning
  - partial identification
  - causal DQN
  - Atari
date: 2026-05-08
content_hash: dd5e5b2cdccc4966
---

# Confounding Robust Deep Reinforcement Learning: A Causal Approach

**Conference**: NeurIPS 2025
**arXiv**: [2510.21110](https://arxiv.org/abs/2510.21110)
**Code**: Available (supplementary materials include gameplay videos)
**Area**: Reinforcement Learning / Causal Inference
**Keywords**: confounded MDP, off-policy learning, partial identification, causal DQN, Atari

## TL;DR
This paper extends DQN via partial identification theory, proposing Causal DQN to learn robust policies from offline data with unobserved confounders—by optimizing a worst-case lower bound on the value function to obtain safe policies—and consistently outperforms standard DQN across 12 confounded Atari games.

## Background & Motivation
**Background**: Deep RL methods such as DQN excel in high-dimensional state spaces, but implicitly assume No Unmeasured Confounders (NUC)—i.e., that the behavioral policy's data contains no unobserved confounding factors. Off-policy learning relies on this assumption to directly equate conditional distributions in observational data with causal transition distributions.

**Limitations of Prior Work**: When learning from an offline demonstrator's data, the learner has no control over the data collection process. If the demonstrator uses information unobservable to the learner (e.g., in Pong the demonstrator can see the opponent's position while the learner only observes a partial view), standard DQN cannot distinguish causal effects from spurious correlations, resulting in ineffective policies.

**Key Challenge**: In confounded settings, the effect of a target policy is generally not identifiable, and traditional methods cannot uniquely determine the value function from data. However, partial identification methods can derive informative upper and lower bounds on the value function.

**Goal**: To extend partial identification methods to complex, high-dimensional domains (image-based inputs) and construct a confounding-robust deep RL algorithm.

**Key Insight**: In a confounded MDP (CMDP), derive a causal Bellman optimality equation for the optimal value function (in lower-bound form), then approximate this lower bound with a neural network.

**Core Idea**: Replace the standard DQN Q-value update with a pessimistic lower-bound update via the causal Bellman equation—following the standard path when the observed action matches the target action, and using worst-case reward and worst-case next state otherwise.

## Method

### Overall Architecture
Causal DQN follows the standard DQN experience replay framework, but replaces the standard Bellman equation in the Q-value update step with the causal Bellman optimality equation (Proposition 3.1). The learner observes demonstrator trajectories (which may exploit information invisible to the learner), stores them in a replay buffer, and optimizes the Q-network lower bound via mini-batch gradient descent.

### Key Designs

1. **Confounded MDP (CMDP) Formalization**:

    - Function: Explicitly models unobserved confounders in the MDP—$\langle \mathcal{S}, \mathcal{X}, \mathcal{Y}, \mathcal{U}, \mathcal{F}, P \rangle$, where $\mathcal{U}$ is the unobserved noise space.
    - Mechanism: In the causal graph, bidirectional arrows $X_t \leftrightarrow Y_t$ and $X_t \leftrightarrow S_{t+1}$ indicate that the unobserved confounder $U_t$ jointly influences actions, rewards, and next states.
    - Design Motivation: Standard MDPs assume $P(s'|s,a) = \mathcal{T}(s,a,s')$ (identifying transition distributions directly from conditional distributions under NUC), but this equality does not hold under confounding.

2. **Causal Bellman Optimality Equation (Proposition 3.1)**:

    - Function: Derives an identifiable lower bound $\underline{Q_*}(s,x)$ on the optimal Q-value function.
    - Mechanism: Updates proceed in two cases:
        - When the observed action $x_t = x$ (matched): use the standard update $y_t + \gamma \max_{x'} \underline{Q_*}(s_{t+1}, x')$.
        - When $x_t \neq x$ (unmatched): use the worst-case estimate $a + \gamma \min_{s'} \max_{x'} \underline{Q_*}(s', x')$, where $a$ is the reward lower bound.
    - Design Motivation: In the unmatched case, the presence of confounders precludes knowledge of the true transition upon executing target action $x$, necessitating a worst-case estimate to guarantee safety.

3. **Q-Network Lower Bound Optimization**:

    - Function: Approximates the lower bound function with a neural network $\underline{Q_*}(s,x;\theta)$.
    - Mechanism: The loss function $L_i(\theta_i) = \mathbb{E}_{s \sim \rho(\cdot)}[\sum_x (W_i(x) - \underline{Q_*}(s,x;\theta_i))^2]$ simultaneously updates all actions (not only the matched action), since unmatched actions also provide information about the lower bound under the causal update.
    - Design Motivation: Standard DQN updates only the Q-value for the sampled action, but in the causal Bellman equation every action's update depends on the current observation, requiring simultaneous optimization across all actions.

4. **Confounded Atari Game Design**:

    - Function: Designs confounded variants of 12 Atari games by occluding portions of the screen so that the learner cannot observe information used by the demonstrator.
    - Mechanism: Saliency maps of the demonstrator are used to localize the visual regions it relies on; these regions are occluded to serve as unobserved confounders.
    - Design Motivation: Constructs a controlled experimental environment that precisely governs the presence of confounding factors.

### Loss & Training
- 1M environment steps, 20 parallel environments
- Batch size 512, replay buffer 100K
- Double DQN used for training stability
- Worst-case states estimated by random sampling from the replay buffer

## Key Experimental Results

### Comparison Across 12 Confounded Atari Games

| Game | Demonstrator | Conf. DQN | Conf. LSTM-DQN | Interv. DQN | **Causal DQN** |
|------|-------------|---------|--------------|-----------|--------------|
| Pong | 21.0 | -19.4 | -20.5 | -19.7 | **-1.8** |
| Boxing | 99.8 | 1.3 | -2.6 | -1.7 | **97.8** |
| Gopher | 6780 | 380 | 420 | 350 | **8140** |
| ChopperCommand | 4560 | 800 | 920 | 750 | **5280** |
| Amidar | 232.4 | 37.8 | 59.0 | 44.0 | **282.6** |

### Ablation Study (Aggregated Normalized Return)

| Method | Normalized Return |
|------|-------------|
| Conf. DQN | ~0.05 |
| Conf. LSTM-DQN | ~0.08 |
| Interv. DQN | ~0.06 |
| **Causal DQN** | **~0.85** |

### Key Findings
- **Consistent Superiority**: Causal DQN outperforms all standard DQN variants across all 12 confounded games by large margins.
- **Emergence of Conservative Policies**: Causal DQN learns reasonable conservative behaviors under information deficiency—tracking only the ball in Pong (rather than the opponent), and adopting a rope-a-dope defensive strategy in Boxing.
- **Surpassing the Demonstrator**: In Gopher and ChopperCommand, Causal DQN even exceeds the fully informed demonstrator, as the conservative strategy proves more effective in certain games.
- Standard DQN variants (including LSTM-DQN) completely fail under confounding and exhibit near-zero convergence.

## Highlights & Insights
- **Elegant Integration of Causal Inference and Deep RL**: The rigorous mathematical framework of partial identification theory is seamlessly incorporated into the practical DQN algorithm, demonstrating the feasibility of causal methods in high-dimensional domains.
- **Natural Emergence of Conservative Policies**: The algorithm is not explicitly programmed for conservatism, yet pessimistic lower-bound optimization naturally yields safe and reasonable policies—a byproduct analogous to safe RL.
- **Confounded Atari Games as a Benchmark**: The saliency-map-guided occlusion approach for constructing controlled confounded environments is generalizable to other confounded RL benchmark designs.
- **Simultaneous Update of All Actions**: The causal Bellman equation reveals that unmatched actions also carry information—a sharp contrast to standard off-policy RL, which updates only the matched action.

## Limitations & Future Work
- **Excessive Pessimism of Worst-Case Assumptions**: In some games, worst-case assumptions may lead to overly conservative behavior, failing to exploit signals in the data that, while imprecise, remain informative.
- **Worst-Case State Estimation via Replay Buffer Sampling**: This estimate may be inaccurate, particularly in large state spaces.
- **Evaluation Limited to Atari**: More complex continuous control tasks and real-world scenarios have not been assessed.
- **Computational Overhead**: Simultaneous updates across all actions combined with worst-case state search increase training time.

## Related Work & Insights
- **vs. Standard DQN**: Standard DQN assumes NUC and fails entirely under confounding; Causal DQN ensures robustness through pessimistic lower bounds.
- **vs. Zhang & Bareinboim (2025) Causal Bellman Equation**: This work extends their tabular formulation to deep network implementations, demonstrating feasibility under high-dimensional visual inputs.
- **vs. Pessimistic Offline RL (e.g., CQL)**: CQL is pessimistic with respect to distributional shift, while this work is pessimistic with respect to causal confounding—the sources and treatments of pessimism differ fundamentally.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First high-dimensional implementation of causal partial identification × DQN
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 confounded Atari games + detailed ablations + saliency visualizations + gameplay videos
- Writing Quality: ⭐⭐⭐⭐⭐ Narrative flow from problem motivation → theory → algorithm → experiments is highly coherent
- Value: ⭐⭐⭐⭐⭐ Significant contributions to both causal RL and safe RL

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] A Differential and Pointwise Control Approach to Reinforcement Learning](a_differential_and_pointwise_control_approach_to_reinforceme.md)
- [\[NeurIPS 2025\] Mind the GAP! The Challenges of Scale in Pixel-based Deep Reinforcement Learning](mind_the_gap_the_challenges_of_scale_in_pixel-based_deep_reinforcement_learning.md)
- [\[NeurIPS 2025\] Multi-Objective Reinforcement Learning with Max-Min Criterion: A Game-Theoretic Approach](multi-objective_reinforcement_learning_with_max-min_criterion_a_game-theoretic_a.md)
- [\[NeurIPS 2025\] Enhancing Interpretability in Deep Reinforcement Learning through Semantic Clustering](enhancing_interpretability_in_deep_reinforcement_learning_through_semantic_clust.md)
- [\[NeurIPS 2025\] Robust Adversarial Reinforcement Learning in Stochastic Games via Sequence Modeling](robust_adversarial_reinforcement_learning_in_stochastic_games_via_sequence_model.md)

<!-- RELATED:END -->
