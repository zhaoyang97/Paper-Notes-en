---
title: >-
  [Paper Note] Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online Reinforcement Learning
description: >-
  [ICML2025][Robotics][Bellman operators] Proposes Annealed Q-learning (AQ-L), which smoothly transitions from the Bellman optimality operator to the Bellman operator by annealing the parameter $\tau$ of the expectile loss from close to 1 down to 0.5. In continuous action spaces, this both accelerates early learning and suppresses late-stage overestimation bias. When integrated with TD3/SAC, it significantly outperforms baselines on various locomotion and robotic manipulation t…
tags:
  - "ICML2025"
  - "Robotics"
  - "Bellman operators"
  - "overestimation bias"
  - "Actor-Critic"
  - "annealing strategy"
  - "expectile loss"
  - "continuous action spaces"
date: 2026-05-08
content_hash: a1224bac9f0ba475
---

# Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online Reinforcement Learning

**Conference**: ICML2025  
**arXiv**: [2506.05968](https://arxiv.org/abs/2506.05968)  
**Authors**: Motoki Omura, Kazuki Ota, Takayuki Osa, Yusuke Mukuta, Tatsuya Harada (The University of Tokyo)  
**Code**: [GitHub](https://github.com/motokiomura/annealed-q-learning)  
**Area**: Reinforcement Learning  
**Keywords**: Bellman operators, overestimation bias, Actor-Critic, annealing strategy, expectile loss, continuous action spaces

## TL;DR

Proposes Annealed Q-learning (AQ-L), which smoothly transitions from the Bellman optimality operator to the Bellman operator by annealing the parameter $\tau$ of the expectile loss from close to 1 down to 0.5. In continuous action spaces, this both accelerates early learning and suppresses late-stage overestimation bias. When integrated with TD3/SAC, it significantly outperforms baselines on various locomotion and robotic manipulation tasks.

## Background & Motivation

### Background

In online reinforcement learning within discrete action spaces (such as Atari games), Q-learning-like algorithms directly estimate the optimal Q-values via the **Bellman optimality operator**, resulting in high learning efficiency. Its mathematical form is:

$$T^*Q(s,a) = r + \gamma \max_{a'} Q(s',a')$$

However, in continuous action spaces (such as robotic control), the $\max_{a'}$ operation is computationally intractable over infinite action sets. Therefore, Actor-Critic methods instead employ the **Bellman operator**:

$$T^\pi Q(s,a) = r + \gamma \mathbb{E}_{a'\sim \pi} Q(s',a')$$

which only models the value function of the current policy and relies on policy updates to achieve improvement.

### Limitations of Prior Work

- **Actor-Critic methods (TD3, SAC)**: By only utilizing the Bellman operator, policy improvement completely depends on the gradient updates of the actor. The improvement of Q-values lags behind policy improvement, leading to **low sample efficiency**.
- **Attempts to directly use the Bellman optimality operator** (Haarnoja et al., 2017; Garg et al., 2023; Kalashnikov et al., 2018): These methods face high computational costs and training instability in continuous spaces.
- **Fixed expectile estimation** (e.g., IQL, XQL): These approximate the optimal value using a fixed $\tau$ value, but exhibit high hyperparameter sensitivity. An excessively large $\tau$ leads to severe overestimation, while a too small $\tau$ loses the acceleration benefits of optimality.

### Key Insight

Through preliminary experiments with tabular Actor-Critic, the authors reveal a key trade-off:

**Q-learning-style updates** (Bellman optimality operator): Q-values converge directly toward optimal values, learning quickly but introducing overestimation bias.

**SARSA-style updates** (Bellman operator): Q-values must wait for policy improvement to increase, learning slowly but exhibiting minimal bias.

This trade-off naturally leads to an annealing idea of "**fast first, stable later**"—utilizing the optimality operator in the early stages of training to accelerate exploration, and switching to the standard operator in the later stages to eliminate bias.

## Method

### Overall Architecture: Annealed Q-learning (AQ-L)

The core idea of AQ-L is remarkably simple: in the critic learning of existing Actor-Critic methods (such as TD3 or SAC), the standard L2 loss is replaced with the **expectile loss**, and the degree of optimality is controlled by annealing the parameter $\tau$.

#### Expectile Loss Definition

The expectile loss is defined as:

$$L_\tau(u) = |\tau - \mathbb{I}(u < 0)| \cdot u^2$$

where $u = Q_\theta(s,a) - y$ is the TD error, and $y$ is the target value. When $\tau > 0.5$, positive residuals (underestimation) receive higher weight than negative residuals (overestimation), biasing the Q-value estimation toward higher quantiles to approximate the $\max$ operation. When $\tau = 0.5$, it degenerates to the standard L2 loss, which is the standard Bellman operator.

#### Annealing Strategy

The parameter $\tau$ is linearly annealed from an initial value $\tau_0$ (close to 1, e.g., 0.9 or 0.99) to 0.5:

$$\tau_t = \tau_0 + (0.5 - \tau_0) \cdot \frac{t}{T}$$

where $T$ is the timestep at which annealing ends. Once annealing is complete, $\tau$ remains constant at 0.5, and the critic learning reverts to standard Actor-Critic updates.

### Key Designs

#### 1. Integration with TD3 (AQ-L + TD3)

TD3 utilizes double critic networks $Q_{\theta_1}, Q_{\theta_2}$ and delayed policy updates. The modification of AQ-L only involves the critic loss function—replacing the MSE loss with the expectile loss $L_{\tau_t}$. The computation of the target value $y$ remains identical to the original TD3 (using clipped double Q-learning).

#### 2. Integration with SAC (AQ-L + SAC)

SAC maximizes policy entropy alongside the return. Similarly, AQ-L only replaces the critic loss function, with the target value incorporating entropy regularization: $y = r + \gamma (\min Q(s', a') - \alpha \log \pi(a'|s'))$. The remainder of the training pipeline is identical to original SAC.

#### 3. Dual Advantages of Annealing

- **Early stage ($\tau$ close to 1)**: Approximates the Bellman optimality operator, forcing Q-values directly toward optimal values to accelerate learning; the overestimation bias during this phase actually facilitates exploration.
- **Late stage ($\tau = 0.5$)**: Recovers the standard Bellman operator to eliminate overestimation bias and guarantee convergence stability; by this point, the policy has already improved, eliminating the need for extra bias-driven improvement.

#### 4. Simplicity of Implementation

The entire method requires only two lines of code modifications: (1) replacing the critic's MSE loss with the expectile loss; (2) adding a linear annealing schedule for $\tau$. It requires no extra networks, additional data sampling, or complex training pipeline modifications.

### Theoretical Analysis

#### Acceleration Mechanism of the Bellman Optimality Operator

In the tabular setting, the Bellman optimality operator directly applies the $\max$ operator over all actions, making Q-value updates independent of the quality of the current policy. Conversely, Q-value improvement with the Bellman operator is constrained by the current policy—Q-values can only follow and improve after policy improvement, establishing a delayed policy-value coupling.

#### Sources of Overestimation Bias

When utilizing a function approximator (neural network) to model Q-values, the Q-values suffer from stochasticity (originating from sampling noise and function approximation error). Taking the $\max$ over noisy Q-values systematically overestimates the true value: $\mathbb{E}[\max_a Q(s,a)] \ge \max_a \mathbb{E}[Q(s,a)]$ (a corollary of Jensen's inequality). This phenomenon is termed "overestimation bias" and represents the core challenge of Q-learning under function approximation settings.

#### Expectile as a Soft Approximation of Max

For a random variable $X$, as $\tau \to 1$, the $\tau$-expectile approaches the essential supremum of the distribution, $\text{ess sup}(X)$. This provides a differentiable and computationally tractable alternative for approximating $\max_{a'} Q(s', a')$ in continuous action spaces, while $\tau$ provides a continuous interpolation from the "exact max" to the "mean." This smoothness makes the annealing strategy possible.

## Key Experimental Results

### Experiment 1: Preliminary Experiment—Comparison of Two Operators in Tabular Environments

Using a 5x5 Grid World environment, the tabular Actor-Critic under SARSA-style (Bellman operator) and Q-learning-style (Bellman optimality operator) updates are compared.

| Method | Q-value Convergence Speed | Overestimation | Noise Sensitivity | Post-annealing Performance |
|------|------------|-----------|-----------|-----------|
| SARSA-style (Bellman operator) | Slow (depends on policy improvement) | No | Low | — |
| Q-learning-style (Bellman optimality operator) | Fast (directly toward optimal) | Yes (significant) | High | — |
| Linear Interpolation + Annealing | Fast (early acceleration) | Mild (early transient) | Medium | Converges to unbiased estimation |

The preliminary experiment injects Gaussian noise into Q-values to simulate function approximation error: Q-learning-style updates cause Q-values to converge to an overestimated value higher than the optimal value, whereas SARSA-style updates are largely unaffected. The annealing method combines the strengths of both approaches.

### Experiment 2: Performance Comparison on Continuous Control Benchmark Tasks

Comparison of AQ-L against baseline methods on MuJoCo locomotion and robotic manipulation tasks:

| Task | TD3 | AQ-L + TD3 | SAC | AQ-L + SAC | Key Observations |
|------|-----|------------|-----|------------|---------|
| HalfCheetah | Baseline | Significantly Outperforms | Baseline | Significantly Outperforms | Clear sample efficiency improvement |
| Hopper | Baseline | Significantly Outperforms | Baseline | Significantly Outperforms | Faster early convergence |
| Walker2d | Baseline | Significantly Outperforms | Baseline | Significantly Outperforms | Higher final return |
| Ant | Baseline | Significantly Outperforms | Baseline | Significantly Outperforms | More stable training |
| Humanoid | Baseline | Significantly Outperforms | Baseline | Significantly Outperforms | Substantial improvement margin |
| Manipulation Tasks | Baseline | Significantly Outperforms | Baseline | Significantly Outperforms | Good generalization |

AQ-L significantly outperforms the corresponding baselines on all tested tasks, especially showcasing a clear sample efficiency advantage in the early stages of training.

### Experiment 3: Hyperparameter Robustness Analysis

| $\tau_0$ Setting | Fixed $\tau$ (unannealed) | AQ-L (annealed to 0.5) |
|---------|-----------------|------------------|
| $\tau_0 = 0.7$ | Moderate performance | Consistently high performance |
| $\tau_0 = 0.9$ | High performance volatility | Consistently high performance |
| $\tau_0 = 0.99$ | Severe overestimation, performance collapse | Consistently high performance |
| $\tau_0 = 0.999$ | Training failure | Maintains reasonable performance |

The annealing mechanism significantly enhances robustness to the $\tau_0$ hyperparameter: under a fixed $\tau$ setting, excessively large $\tau$ values lead to overestimation and training collapse; in contrast, under the annealing strategy, even with aggressive initial $\tau_0$ selections, the final performance remains robust.

## Highlights & Insights

- **Intuitive, Simple yet Deep**: Introduces the success of discrete Q-learning (fast convergence rate of the Bellman optimality operator) into continuous action spaces, solving the overestimation problem elegantly via annealing. This "perfect pairing" of two classic operators is highly commendable.
- **Extremely Simple Implementation**: Requires only modifying the critic loss function (MSE to expectile loss) and adding an annealing schedule. It demands no extra networks or training tricks, making it highly engineering-friendly.
- **Natural Mapping to Exploration-Exploitation**: The annealing strategy aligns perfectly with the "exploration first, exploitation later" paradigm in RL—early overestimation facilitates exploration, while late-stage bias elimination secures stable exploitation.
- **Universal Plug-and-Play Module**: The method can be easily embedded into any Actor-Critic-based algorithm (such as TD3, SAC, etc.) without altering the original framework architecture, offering extremely low migration cost.
- **Clever Reuse of Expectile Loss**: Borrows the expectile regression concept from IQL, yet escapes the constraints of offline RL to uncover a novel use case within online RL.

## Limitations & Future Work

- **Annealing Schedule Fixed to Linear**: Currently, only linear annealing is explored. Nonlinear schedules, such as cosine or exponential annealing, might further improve performance but have not been systematically investigated.
- **Selection of Annealing End Time $T$**: The timestep where annealing ends must be pre-specified, lacking an adaptive adjustment mechanism (e.g., dynamically adjusting $\tau$ based on the degree of overestimation).
- **Limited Theoretical Guarantees**: The motivation relies on intuitive observations from tabular experiments, lacking rigorous theoretical analysis regarding convergence or bias-variance trade-offs under function approximation settings.
- **Evaluated Only on MuJoCo Environments**: Although it covers multiple locomotion and manipulation tasks, it has not been validated in more complex scenarios like pixel inputs, sparse rewards, or long-horizon planning.
- **Integration with Other Overestimation Mitigation Methods**: The potential complementarity between AQ-L and existing anti-overestimation techniques (such as MaxMin Q-learning, Averaged DQN, or ensemble methods) is not analyzed in depth.
- **Lack of Computational Overhead Analysis**: Despite its implementation simplicity, the extra computational cost of expectile loss compared to MSE and its impact on training time are not quantitatively discussed.

## Related Work & Insights

- **IQL (Kostrikov et al., 2022)**: Employs expectile regression in offline RL to avoid querying Q-values of OOD actions; this work introduces expectile loss into a different context of online RL.
- **XQL (Garg et al., 2023)**: Uses exponential loss to approximate the Bellman optimality operator, but is computationally expensive and unstable; AQ-L offers a more practical alternative via its annealing strategy.
- **QT-Opt (Kalashnikov et al., 2018)**: Employs CEM to approximate the continuous action $\max$ in robotic grasping, which is computationally expensive; expectile loss provides a more efficient alternative.
- **OptAC (Ji et al., 2024)**: Analyzes the inefficiency of using the optimality operator in actor-critic, providing theoretical motivation for this work.
- **Clipped Double Q-learning (Fujimoto et al., 2018, TD3)**: Mitigates overestimation by taking the minimum of double critics; AQ-L builds upon this by further leveraging the annealing mechanism.
- **SQL (Haarnoja et al., 2017)**: An early attempt to use a soft Bellman optimality operator in continuous spaces, which suffered from training instability.

## Rating

- Novelty: ⭐⭐⭐⭐ — Formulates the annealed transition between two classic Bellman operators as a unified framework; the perspective is novel, though the technical components (expectile loss, annealing) are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers various MuJoCo tasks and two baseline frameworks with comprehensive hyperparameter robustness analysis, though validation in more complex scenarios is missing.
- Writing Quality: ⭐⭐⭐⭐⭐ — The motivation derived from the preliminary experiment is clear and logical, featuring an excellent narrative structure from simple to complex.
- Value: ⭐⭐⭐⭐ — High practical value (plug-and-play, engineering-friendly), but theoretical depth is relatively shallow; the long-term impact on the community remains to be proven.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Snapshot of Influence: A Local Data Attribution Framework for Online Reinforcement Learning](../../NeurIPS2025/robotics/a_snapshot_of_influence_a_local_data_attribution_framework_f.md)
- [\[ICML 2025\] Maximum Total Correlation Reinforcement Learning](maximum_total_correlation_reinforcement_learning.md)
- [\[ICML 2025\] Graph-Assisted Stitching for Offline Hierarchical Reinforcement Learning](graph-assisted_stitching_for_offline_hierarchical_reinforcement_learning.md)
- [\[NeurIPS 2025\] Reinforcement Learning with Action Chunking](../../NeurIPS2025/robotics/reinforcement_learning_with_action_chunking.md)
- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](../../NeurIPS2025/robotics/learning_interactive_world_model_for_object-centric_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
