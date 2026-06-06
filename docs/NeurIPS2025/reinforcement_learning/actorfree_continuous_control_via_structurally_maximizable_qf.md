---
title: >-
  [Paper Note] Actor-Free Continuous Control via Structurally Maximizable Q-Functions
description: >-
  [NeurIPS 2025][Reinforcement Learning][actor-free Q-learning] This paper proposes Q3C (Q-learning for Continuous Control with Control-points)…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "actor-free Q-learning"
  - "continuous control"
  - "control points"
  - "wire-fitting"
  - "structural maximization"
date: 2026-05-08
content_hash: d77545206908572b
---

# Actor-Free Continuous Control via Structurally Maximizable Q-Functions

**Conference**: NeurIPS 2025
**arXiv**: [2510.18828](https://arxiv.org/abs/2510.18828)  
**Code**: [https://github.com/USC-Lira/Q3C](https://github.com/USC-Lira/Q3C)  
**Area**: Reinforcement Learning
**Keywords**: actor-free Q-learning, continuous control, control points, wire-fitting, structural maximization

## TL;DR

This paper proposes Q3C (Q-learning for Continuous Control with Control-points), which approximates the Q-function via a learned set of control points such that the maximum value is structurally attained at one of those points. Combined with action-conditioned Q-value generation, a control-point diversity loss, and scale normalization, Q3C matches TD3 on standard benchmarks and substantially outperforms all actor-critic methods in constrained action spaces.

## Background & Motivation

**Background**: Continuous-action RL conventionally relies on actor-critic methods (e.g., DDPG, TD3, SAC), in which a critic estimates Q-values and an actor searches for the optimal action via gradient ascent over the Q landscape. Pure value-based methods (e.g., DQN) excel in discrete spaces but are generally considered inapplicable to continuous control because exhaustive enumeration of actions is infeasible.

**Limitations of Prior Work**: Actor-critic methods suffer from fundamental issues: (1) coupled training of the actor and critic leads to instability; (2) gradient ascent only finds locally optimal actions and fails when the Q-function is non-convex (e.g., in constrained action spaces); (3) the additional actor network introduces extra hyperparameters and computational overhead. Earlier actor-free attempts—such as NAF, which restricts Q to a quadratic form, and RBF-DQN, which uses radial basis functions—either lack expressive power or cannot guarantee that the maximum is attained at the basis points.

**Key Challenge**: The fundamental difficulty in continuous Q-learning is the $\max_a Q(s,a)$ operation in the Bellman equation, which admits no exact solution in continuous spaces. The actor is an approximation scheme that introduces its own pathologies. What is needed is a Q-function representation for which the maximization can be performed exactly and efficiently.

**Goal**: Design a structurally maximizable Q-function representation that enables exact identification of the optimal action in continuous spaces without an actor.

**Key Insight**: The paper revisits the wire-fitting framework—anchoring the Q-function approximation with a set of control points so that the global maximum structurally coincides with one of them. This direction was largely abandoned after yielding poor results in early deep RL, but the authors show that combining it with modern deep RL techniques revitalizes its performance.

**Core Idea**: Construct a structurally maximizable Q-function via control-point interpolation, augmented with a series of architectural and algorithmic innovations that bring it to state-of-the-art performance in deep RL.

## Method

### Overall Architecture

Q3C consists of three components: (1) a control-point generator $g_\phi(s)$ that produces $N$ candidate actions $\hat{a}_i(s)$; (2) a Q-estimator $h_\psi(s, \hat{a}_i)$ that evaluates the Q-value $\hat{Q}_i(s)$ at each control point; and (3) a wire-fitting interpolator that computes the Q-value for an arbitrary action $a$ from the control-point locations and their Q-values. The optimal action is selected directly as $\arg\max_i \hat{Q}_i$ over $N$ scalars, requiring no gradient ascent.

### Key Designs

1. **Wire-Fitting Interpolation for Structural Maximization**

    - **Function**: Construct the Q-function such that its maximum is guaranteed to be attained at a control point.
    - **Mechanism**: $Q(s,a) = \frac{\sum_i \hat{Q}_i w_i}{\sum_i w_i}$, where weights $w_i = \frac{1}{|a - \hat{a}_i|^2 + c_i(\hat{Q}_{\max} - \hat{Q}_i)}$. As $a$ approaches the control point with the highest Q-value, the corresponding weight diverges and $Q$ converges to that value. The authors prove this interpolation retains universal approximation capability (Proposition).
    - **Design Motivation**: Compared with NAF's quadratic restriction and RBF-DQN's failure to guarantee the maximum at basis points, wire-fitting provides both sufficient expressive power and a structural guarantee on the location of the maximum.

2. **Action-Conditioned Q-Value Generation**

    - **Function**: Ensure consistency between Q-value estimates and control-point locations.
    - **Mechanism**: The architecture is decomposed into two stages—the control-point generator $g_\phi(s)$ outputs $N$ actions, and a shared Q-estimator $h_\psi(s, \hat{a}_i)$ evaluates each control point independently. Sharing the Q-estimator across all control points enforces consistent Q-values for identical or nearby actions.
    - **Design Motivation**: In the original wire-fitting formulation, Q-values and control-point locations are predicted independently, potentially assigning entirely different Q-values to identically located control points, which destabilizes training.

3. **Control-Point Diversity and Scale Normalization**

    - **Function**: Prevent control-point collapse and ensure robustness to scale variation across tasks.
    - **Mechanism**: A separation loss $L_{\text{sep}} = \frac{1}{N(N-1)} \sum_{i \neq j} \frac{1}{\|\hat{a}_i - \hat{a}_j\|_2 + \epsilon}$ encourages uniform spread of the control points. The Q-value difference term in the wire-fitting weights is normalized as $\tilde{Q}_i = (\hat{Q}_i - \hat{Q}_{\min})/(\hat{Q}_{\max} - \hat{Q}_{\min})$, and the smoothing coefficient $c_i$ is decayed exponentially, rendering the method robust to varying reward scales and action ranges.
    - **Design Motivation**: Without regularization, control points tend to collapse toward the boundaries of the action space (as observed empirically), degrading the expressive capacity of the Q-function.

### Loss & Training

The method builds on the TD3 framework: dual Q-networks to mitigate overestimation, target networks for stable Bellman targets, and Gaussian noise for exploration. The total loss is the Bellman loss plus $\lambda \cdot L_{\text{sep}}$. A delayed exponential learning-rate schedule is used, decaying to 10% of the initial rate. Default hyperparameters are $N=20$ control points and $k=10$ nearest neighbors for Q-value computation.

## Key Experimental Results

### Main Results

| Environment | TD3 | NAF | Wire-Fitting | RBF-DQN | **Q3C** |
|---|---|---|---|---|---|
| Pendulum | -144.6 | -252.4 | -351.5 | -143.9 | **-159.5** |
| Swimmer | 300.7 | 20.6 | 313.6 | 92.4 | **316.4** |
| Hopper | 3113.4 | 500.8 | 1987.5 | 2189.4 | **3206.1** |
| Walker2d | **4770.8** | 2179.6 | 2462.3 | 781.6 | 3977.4 |
| HalfCheetah | **9984.7** | 3531.5 | 7546.2 | 6175.6 | 9468.7 |
| Ant | **5167.7** | -18.1 | 1154.6 | 1674.0 | 3698.4 |

Constrained environments (non-convex Q-functions):

| Environment | TD3 | NAF | Wire-Fitting | RBF-DQN | **Q3C** |
|---|---|---|---|---|---|
| InvPendulumBox | 782.8 | 909.7 | 386.4 | 862.0 | **1000.0** |
| HalfCheetahBox | 2276.7 | **4867.1** | -2139.8 | 2238.4 | 4357.8 |
| HopperBox | 1406.8 | 461.5 | 169.8 | 1641.2 | **1974.3** |

### Ablation Study

| Configuration | Hopper | BipedalWalker | HalfCheetah |
|---|---|---|---|
| **Q3C (full)** | **3206** | **290** | **9469** |
| − CondQ | 2330 | 286 | 8386 |
| − Ranking | 3037 | 180 | 8961 |
| − Div | 1921 | -68 | 5283 |
| − Norm | 2915 | 262 | 8746 |
| Wire-Fitting | 1988 | 70 | 7546 |

### Key Findings

- Q3C matches TD3 in standard environments but is substantially superior in constrained/non-convex settings—achieving a perfect score of 1000 on InvPendulumBox versus 783 for TD3.
- The control-point diversity component (Div) is the most critical: its removal causes BipedalWalker performance to plummet from 290 to −68 and reduces Hopper performance by 40%.
- Vanilla wire-fitting performs poorly in deep RL; the improvements in Q3C yield a 2–5× performance gain over the baseline.
- Q3C matches TD3 on the 26-dimensional Adroit tasks, demonstrating scalability to high-dimensional action spaces.

## Highlights & Insights

- **Structural maximization** elegantly transforms the continuous $\max$ into an $\arg\max$ over $N$ scalars, completely eliminating the local-optimum problem inherent in gradient-ascent-based action selection. This advantage is particularly pronounced when the Q-function is non-convex.
- **Actor-free simplicity**: Q3C simultaneously serves as actor and critic, reducing the hyperparameter burden (no separate actor learning rate, update frequency, etc.) and improving training stability.
- The number of control points does not need to scale linearly with action dimensionality (only 70 control points suffice for a 26-dimensional space), because the action-conditioned Q-estimator is shared across all control points, preventing parameter count from growing linearly with $N$.

## Limitations & Future Work

- Q3C still trails TD3 by approximately 20–30% on Ant-v4 and Walker2d in standard settings.
- The exploration strategy directly inherits TD3's Gaussian noise without task-specific design (e.g., Boltzmann exploration guided by control-point Q-values).
- Validation is limited to deterministic policies; extension to stochastic policies (e.g., SAC-style soft Q-learning) remains unexplored.
- The offline RL setting warrants investigation—the Q-value constraints imposed by control-point interpolation may naturally alleviate overestimation.

## Related Work & Insights

- **vs. TD3**: TD3 is the leading deterministic actor-critic method, but its gradient-ascent actor finds only locally optimal actions. Q3C matches TD3 in standard environments and is significantly superior in constrained settings, at the cost of somewhat lower performance in high-dimensional environments such as Ant.
- **vs. NAF (Gu et al. 2016)**: NAF restricts the Q-function to a quadratic form in action, enabling analytic maximization but severely limiting expressiveness. Q3C preserves universal approximation capability via control-point interpolation.
- **vs. RBF-DQN (Asadi et al. 2021)**: RBF interpolation does not guarantee that the maximum is attained at a basis point and requires a large number of centers (~100). Q3C's control-point interpolation structurally guarantees the maximum at a control point and is more efficient.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Revives the abandoned wire-fitting direction; the key innovation lies in making it viable within modern deep RL.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Standard and constrained environments, multiple baselines, comprehensive ablations, high-dimensional tests, and visualization analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic; motivation and contributions are well articulated; theory and experiments are tightly integrated.
- **Value**: ⭐⭐⭐⭐ — Offers clear advantages in constrained action-space scenarios and opens a new avenue for continuous Q-learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Test-driven Reinforcement Learning in Continuous Control](../../AAAI2026/reinforcement_learning/test-driven_reinforcement_learning_in_continuous_control.md)
- [\[NeurIPS 2025\] Parameter-Free Algorithms for the Stochastically Extended Adversarial Model](parameter-free_algorithms_for_the_stochastically_extended_adversarial_model.md)
- [\[NeurIPS 2025\] When Can Model-Free Reinforcement Learning be Enough for Thinking?](when_can_model-free_reinforcement_learning_be_enough_for_thinking.md)
- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](../../ICLR2026/reinforcement_learning/wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [\[NeurIPS 2025\] A Differential and Pointwise Control Approach to Reinforcement Learning](a_differential_and_pointwise_control_approach_to_reinforceme.md)

</div>

<!-- RELATED:END -->
