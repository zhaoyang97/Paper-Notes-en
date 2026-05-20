---
title: >-
  [Paper Note] RLSLM: A Hybrid Reinforcement Learning Framework Aligning Rule-Based Social Locomotion Model with Human Social Norms
description: >-
  [AAAI 2026][Reinforcement Learning][Social Navigation] This paper proposes RLSLM, a hybrid framework that embeds a psychology-experiment-driven rule-based Social Locomotion Model (SLM) into the reward function of reinfor…
tags:
  - "AAAI 2026"
  - "Reinforcement Learning"
  - "Social Navigation"
  - "Social Locomotion Model"
  - "VR Experiment"
  - "Human-Robot Interaction"
date: 2026-05-08
content_hash: 9790c688ec327ebd
---

# RLSLM: A Hybrid Reinforcement Learning Framework Aligning Rule-Based Social Locomotion Model with Human Social Norms

**Conference**: AAAI 2026
**arXiv**: [2511.11323](https://arxiv.org/abs/2511.11323)  
**Code**: [github.com/kouyitian/RLSLM](https://github.com/kouyitian/RLSLM)  
**Area**: Reinforcement Learning
**Keywords**: Social Navigation, Reinforcement Learning, Social Locomotion Model, VR Experiment, Human-Robot Interaction

## TL;DR

This paper proposes RLSLM, a hybrid framework that embeds a psychology-experiment-driven rule-based Social Locomotion Model (SLM) into the reward function of reinforcement learning, enabling agents to efficiently learn navigation policies aligned with human social norms in crowd environments. VR experiments demonstrate that RLSLM achieves significantly higher comfort ratings than existing rule-based baselines.

## Background & Motivation

### State of the Field

Navigating in human-dense environments without causing discomfort is a core capability for social agents. Existing approaches fall into two categories:

**Rule-based methods**: Deterministic rules designed from psychological principles (e.g., proxemics, social force models)
- Strengths: High interpretability, low computational overhead
- Limitations: Difficult to quantify precisely, poor generalization, may produce unnatural oscillatory trajectories

**Data-driven methods**: RL or imitation learning from large-scale datasets
- Strengths: Capable of learning complex behaviors, high expressiveness
- Limitations: Highly dependent on data quality, expensive to train, lack interpretability, difficult to align with human intuition

### Core Motivation

**Key question**: Can both paradigms be integrated to build a model that is efficient, adaptable, interpretable, and aligned with real human social behavior?

Social rule design in existing navigation research is largely based on intuition or data statistics, rather than rigorously controlled human behavior experiments. Third-person user studies also suffer from ecological validity issues. The authors propose **directly embedding quantitative findings from state-of-the-art psychology research into the RL training pipeline**, with validation via immersive VR experiments.

### Novelty

- First work to use a quantitative social locomotion model derived from psychology behavior experiments as an RL reward signal
- Comfort evaluation via first-person immersive VR experiments (high ecological validity)
- Learns socially aligned navigation policies within as few as 10,000 training steps

## Method

### Overall Architecture

The RLSLM framework follows a three-stage decision loop:
1. **Environment observation**: Captures the agent's own position along with relative positions and orientations of surrounding pedestrians
2. **Action selection**: Generates navigation actions via an Actor-Critic network
3. **Policy update**: Updates the policy through a multi-dimensional feedback mechanism (mechanical energy + goal progress + social influence)

The core mechanism is: top-down rule-based methods provide prior knowledge → encoded as a reward function → bottom-up RL optimizes the policy on real-scenario data.

### Key Designs

#### 1. **Environment Observation Module**

The observation vector contains the agent's own position and the relative positions and orientations of $n$ surrounding pedestrians, concatenated into a structured input:

$$s_t \in \mathbb{R}^{3n+2}$$

Each pedestrian is described by 3 parameters (relative x/y coordinates + heading angle), and the agent itself by 2 parameters (position coordinates).

#### 2. **Actor-Critic Action Selection**

The A2C (Advantage Actor-Critic) algorithm is adopted:
- **Actor network**: Represents policy $\pi(a_t|s_t)$, outputs action probability distributions, supporting the exploration-exploitation balance
- **Critic network**: Estimates value function $V(s_t)$, predicting expected returns from the current state
- Network architecture: 5-layer MLP (64-128-256-128-64), RMSprop optimizer, learning rate $5 \times 10^{-4}$

#### 3. **Multi-Dimensional Feedback Mechanism (Core Contribution)**

The reward consists of three components, balancing social influence and mechanical energy consumption:

**a) Mechanical energy penalty $R_e$**:
$$R_e(s_t) = -\alpha$$
A fixed per-step energy consumption penalty ($\alpha=1$) that incentivizes the agent to reach the goal in the fewest steps.

**b) Goal progress reward $R_d$**:
$$R_d(s_t, s_{t-1}) = \frac{D_{t-1} - D_t}{l}$$
Proportional to the reduction in distance to the goal, encouraging progress toward the target.

**c) Social influence penalty $R_s$ (core)**:

Based on behavioral experiment data from Zhou et al., an orientation-sensitive asymmetric social comfort field is constructed. Social influence comprises three sub-components:

- **HRSC (Heading-Related Social Component)**: $m \times f(\theta_h)$, where $f(\theta_h) = \max(\cos(\theta_h), 0)$. Captures the psychological finding that discomfort is stronger in face-to-face encounters
- **HISC (Heading-Independent Social Component)**: Constant $n$, representing baseline personal space discomfort
- **CAC (Collision Avoidance Component)**: $c \times I_{CA}$, modeling physical collision risk by approximating the human body as an ellipse

Total social influence formula:
$$F = \frac{I_{\text{agent}} \times I_{\text{person}}}{d^2}$$
$$I_{\text{human}} = m \times f(\theta_h) + n + c \times I_{CA}$$

After normalization: $F' = \min(F/K, 1)$, where $K=10.180$ is the upper bound fitted from behavioral data.

All parameters ($m_a=0.321, n_a=0.856, m_p=0.438, n_p=0.630, a=0.285, b=0.175, c=1.430$) are derived from fitting results of psychology experiments.

**Total reward function**:
$$r_t = R_d(s_t, s_{t-1}) + R_e(s_t) + \sigma R_s(s_t), \quad 0 < t < T$$
$$r_T = \pm C \quad (\text{terminal reward/penalty})$$

where $\sigma=0.5$ (social influence weight), $C=500$ (terminal reward/penalty), and $\gamma=0.9$ (discount factor).

### Loss & Training

- Algorithm: A2C (Stable-Baselines3 implementation)
- Network: MLP policy, 5 layers (64-128-256-128-64)
- Optimizer: RMSprop, lr = $5\times10^{-4}$
- Discount factor: $\gamma = 0.8$
- Training budget: 10,000 steps per run (extremely low training cost)
- Hardware: NVIDIA 3090 GPU + CUDA
- Environment: OpenAI Gymnasium
- Separate models trained for single-person and multi-person scenarios

## Key Experimental Results

### VR Human-Robot Interaction Experiment

**Participants**: 30 university students and staff (11 male, 19 female, ages 18–29), normal or corrected-to-normal vision

**Equipment**: HTC Vive Pro headset (binocular resolution 2880×1600, refresh rate 90 Hz, field of view 110°)

**Experimental design**: 50 scenarios (25 single-person + 25 multi-person), each scenario × 3 algorithms = 150 trials. Participants experienced each scenario from a first-person perspective and rated comfort on a 1–5 scale.

### Main Results

**User comfort rating comparison:**

| Model | Single-Person Avg. | Multi-Person Avg. | Overall Avg. | Statistical Significance |
|------|-----------|-----------|-------|----------|
| **RLSLM** | ~3.8 | ~4.5 | **4.21/5** | - |
| COMPANION | ~2.8 | ~2.9 | ~3.09 | P < 0.001 |
| n-Body | ~2.5 | ~3.1 | ~2.80 | P < 0.001 |

**Statistical analysis**: Repeated-measures ANOVA revealed a significant main effect of model type on comfort ($F_{(2,58)}=219.589$, $P<0.001$, $\eta_G^2=0.525$). RLSLM significantly outperformed both baselines in both single-person and multi-person scenarios (Bonferroni-corrected $P<0.001$).

**Key finding**: RLSLM achieves a comfort gain of $\Delta\text{rating}=1.12$, a 36% improvement over the best baseline.

### Ablation Study

**Sensitivity analysis on social influence weight $\sigma$:**

| $\sigma$ value | Behavioral pattern | MLD (Maximum Lateral Deviation) |
|-------------|---------|-------------------|
| 0 | Strict shortest path | Minimum |
| 0.5 | Moderate detour | Moderate |
| 1.0 | Larger detour | Large |
| 2.0 | Excessively conservative | Maximum |

$\sigma$ effectively controls social sensitivity, validating the interpretability of the reward function.

**HRSC ablation (heading-related component):**

| Configuration | Times passing pedestrians head-on (42 scenarios) | Notes |
|------|--------------------------|------|
| Full model | 5 times (11.9%) | Orientation-aware |
| w/o HRSC | 23 times (57.76%) | Loses orientation awareness, random detours |

Removing HRSC causes the agent to lose awareness of pedestrian heading, with the proportion of head-on passes rising from 12% to 58%.

**HISC and CAC ablation (heading-independent components):**

| Configuration | MLD change | Notes |
|------|---------|------|
| Full model | Baseline | Stable navigation |
| w/o HISC | Decreased | Reduced personal space awareness |
| w/o CAC | Decreased | Reduced collision avoidance capability |

### Key Findings

1. **Hybrid framework outperforms purely rule-based methods**: RLSLM significantly surpasses COMPANION and n-Body in user comfort
2. **Psychological priors dramatically reduce training cost**: Convergence in only 10,000 steps, far fewer than purely data-driven methods
3. **Social influence weight $\sigma$ provides an intuitive control knob**: Social conservatism can be adjusted according to the application scenario
4. **Three social components are each indispensable**: Ablations confirm that each component is essential for specific social behaviors
5. **Pronounced advantage in multi-person scenarios**: RLSLM's gains are most significant in multi-person interaction settings
6. **High ecological validity**: First-person VR experiments are more realistic than conventional third-person video evaluations

## Highlights & Insights

1. **Exemplary interdisciplinary integration**: Quantitative psychology experiment results are directly embedded into the RL reward function, achieving an organic combination of cognitive science and machine learning
2. **Exceptional training efficiency**: Convergence in 10,000 steps demonstrates that well-designed prior knowledge can dramatically reduce data requirements
3. **Strong interpretability**: Each reward component and parameter carries explicit psychological meaning, rather than functioning as a black box
4. **Bidirectional value**: The framework not only serves robot navigation but can also function as a computational tool for psychology research
5. **Reusable VR evaluation pipeline**: The open-source VR evaluation toolchain can be used for standardized testing in other social navigation research

## Limitations & Future Work

1. **Static pedestrian assumption**: Pedestrians are stationary in current experiments; dynamic pedestrian avoidance is not considered
2. **Fixed parameters**: SLM parameters are derived from specific experiments and may not generalize across all cultures and contexts
3. **Simple action space**: Only outputs movement direction; velocity variation is not modeled
4. **Limited scenario scale**: 15m×15m virtual environment; performance in large-scale open scenes is unvalidated
5. **Single evaluation metric**: Primarily relies on subjective comfort ratings, lacking quantitative trade-off analysis of navigation efficiency
6. **No comparison with deep learning baselines**: Such as Social Force GAN, STGCNN, and other data-driven methods

## Related Work & Insights

- **Zhou et al. (2022)**: Provides the core behavioral experiment data for the SLM, forming the psychological foundation of this work
- **Social Force Model (Helbing 1995)**: A classic physics-based force model, but less grounded in human behavior experiments than the SLM
- **COMPANION (Kirby 2009)**: A constrained optimization navigation method; one of the baselines in this paper
- **n-Body (van den Berg 2011)**: A reciprocal collision avoidance method; one of the baselines in this paper
- Insight: **Encoding domain expert knowledge as reward functions is an effective strategy for improving RL training efficiency and interpretability**, particularly in data-scarce scenarios

## Rating

- Novelty: ⭐⭐⭐⭐ (The hybrid paradigm embedding psychological models into RL rewards is novel)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Comprehensive VR human-robot experiments + ablations + sensitivity analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear and accessible; interdisciplinary work is well communicated)
- Value: ⭐⭐⭐⭐ (Provides a reusable hybrid paradigm and evaluation toolchain for social navigation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Savoir: Learning Social Savoir-Faire via Shapley-based Reward Attribution](../../ACL2026/reinforcement_learning/savoir_learning_social_savoir-faire_via_shapley-based_reward_attribution.md)
- [\[ACL 2026\] The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games](../../ACL2026/reinforcement_learning/the_stackelberg_speaker_optimizing_persuasive_communication_in_social_deduction_.md)
- [\[AAAI 2026\] Distilling Deep Reinforcement Learning into Interpretable Fuzzy Rules: An Explainable AI Framework](distilling_deep_reinforcement_learning_into_interpretable_fuzzy_rules_an_explain.md)
- [\[AAAI 2026\] Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping](aligning_machiavellian_agents_behavior_steering_via_test-tim.md)
- [\[AAAI 2026\] MARS: A Meta-Adaptive Reinforcement Learning Framework for Risk-Aware Multi-Agent Portfolio Management](mars_a_meta-adaptive_reinforcement_learning_framework_for_risk-aware_multi-agent.md)

</div>

<!-- RELATED:END -->
