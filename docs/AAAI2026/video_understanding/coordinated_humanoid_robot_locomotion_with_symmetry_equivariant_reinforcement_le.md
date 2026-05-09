---
title: >-
  [Paper Note] Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy
description: >-
  [AAAI 2026][Video Understanding][Humanoid Robot] This paper proposes SE-Policy, which directly embeds strict symmetry equivariance (actor) and symmetry invariance (critic) into the neural network architecture without additional hyperparameters, enabling humanoid robots to produce spatiotemporally coordinated natural locomotion. The velocity tracking error is reduced by 40% compared to DreamWaQ, and the policy is successfully deployed on a physical Unitree G1 robot.
tags:
  - AAAI 2026
  - Video Understanding
  - Humanoid Robot
  - Symmetry Equivariance
  - Deep Reinforcement Learning
  - Motion Coordination
  - PPO
  - Sim-to-Real
date: 2026-05-08
content_hash: dfd2b8462fb6b251
---

# Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy

**Conference**: AAAI 2026
**arXiv**: [2508.01247](https://arxiv.org/abs/2508.01247)
**Code**: None
**Area**: Video Understanding
**Keywords**: Humanoid Robot, Symmetry Equivariance, Deep Reinforcement Learning, Motion Coordination, PPO, Sim-to-Real

## TL;DR

This paper proposes SE-Policy, which directly embeds strict symmetry equivariance (actor) and symmetry invariance (critic) into the neural network architecture without additional hyperparameters, enabling humanoid robots to produce spatiotemporally coordinated natural locomotion. The velocity tracking error is reduced by 40% compared to DreamWaQ, and the policy is successfully deployed on a physical Unitree G1 robot.

## Background & Motivation

Humanoid robots possess an inherently bilateral morphology—the left and right arms and legs are mirror-symmetric, consistent with the bilateral symmetry of the human nervous system. However, existing deep reinforcement learning (DRL)-based control policies typically treat the network as a black box, completely ignoring this morphological prior:

**Asymmetric actions from symmetric observations**: For mirror-symmetric state inputs (e.g., "left foot in the air" vs. "right foot in the air"), the policy network may produce entirely different action responses, leading to inconsistent motion styles between left and right joints.

**Incoordinated locomotion**: For example, under the DreamWaQ policy, the left and right foot stride lengths are 0.13m and 0.25m respectively, resulting in severely asymmetric gait that is neither natural nor tracking-accurate.

**Poor user experience**: Incoordinated motion patterns appear unnatural and degrade task performance.

Existing approaches to exploiting symmetry fall into three broad categories, each with drawbacks:

- **Temporal symmetry methods** (periodic signals, CPG): Only encourage periodicity of motion; the constraint is loose and symmetry is not guaranteed.
- **Data augmentation** (symmetric experience replay): Induces equivariance by training on collected transitions together with their mirrored counterparts, but this is a soft constraint and symmetry may still be violated at inference time.
- **Loss regularization**: Adds an equivariance penalty $\mathcal{L}_{reg} = \|\pi(\mathcal{F}_o(o)) - \mathcal{F}_a(\pi(o))\|^2$, which requires additional hyperparameter tuning and may interfere with policy optimization.
- **Strict equivariant networks**: Effective in classical control tasks, but their efficacy on real humanoid robots has not been thoroughly validated.

**Core Motivation**: Can one design a method that encodes strict symmetry equivariance directly into the network architecture—without additional hyperparameters—so that humanoid robots naturally produce spatiotemporally coordinated locomotion?

## Method

### Overall Architecture

SE-Policy is built on an actor-critic framework (PPO algorithm) with four components:

- **History encoder** $f_{en}$: Takes the observation sequence $o_{[t-h:t]}$ over the past $h$ steps as input and outputs a latent feature $z$.
- **Observation decoder** $f_{de}$: Predicts the next observation $\hat{o}_{t+1}$ for self-supervised training of the encoder.
- **Policy network** $f_\pi$ (actor): Outputs joint target positions based on the current observation $o_t$ and latent feature $z$.
- **Value network** $V$ (critic): Estimates state value using observations and privileged terrain information.

The key innovation is that **all network modules in the actor satisfy strict equivariance, while the critic satisfies strict invariance**—achieved through parameter sharing within the network architecture itself, rather than through soft constraints in the training loss.

### Mathematical Foundation of Symmetric MDPs

The humanoid robot MDP $\mathcal{M} = \langle \mathcal{S}, \mathcal{O}, \mathcal{A}, P, R, \gamma \rangle$ possesses reflection symmetry. Defining symmetry transformation functions $\mathcal{F}_s$, $\mathcal{F}_o$, $\mathcal{F}_a$ acting on states, observations, and actions respectively:

- **Transition invariance**: $P(\mathcal{F}_s(s')|\mathcal{F}_s(s), \mathcal{F}_a(a)) = P(s'|s,a)$
- **Reward invariance**: $R(\mathcal{F}_s(s), \mathcal{F}_a(a)) = R(s,a)$

From these, the equivariance of the optimal policy follows: $\pi^*(\mathcal{F}_s(s)) = \mathcal{F}_a(\pi^*(s))$, and the invariance of the optimal value function: $V(\mathcal{F}_s(s)) = V(s)$.

Intuitively, when the robot's left foot is in the air, the optimal action is to lower the left foot; in the symmetric state (right foot in the air), the optimal action is to lower the right foot—symmetric states correspond to symmetric actions.

### Symmetry Transformation Design for Observation and Action Spaces

The observation space has dimension 96, comprising 7 components, each with carefully designed symmetry transformation rules:

- **Angular velocity** $\omega$: $(ω_x, ω_y, ω_z) \to (-ω_x, ω_y, -ω_z)$, flipping the x and z axes.
- **Gravity projection** $g$: $(g_x, g_y, g_z) \to (g_x, -g_y, g_z)$, flipping the y axis.
- **Velocity command** $c$: $(c_x, c_y, c_\omega) \to (c_x, -c_y, -c_\omega)$, flipping lateral velocity and angular velocity.
- **Joint positions/velocities/previous actions**: Left-right swapped and negated, e.g., $\theta_{left}^{arm} \leftrightarrow -\theta_{right}^{arm}$.
- **Phase input** $\Phi$: $(\Phi_{sin}, \Phi_{cos}) \to (-\Phi_{sin}, -\Phi_{cos})$, a half-period phase shift.
- **Height map** $H$: Left and right sub-maps swapped; center unchanged.

The action space has dimension 27 (27 joint target positions), with the same transformation rules as joint positions.

### Equivariant Neural Network Construction

All networks are built using parameter-sharing linear layers with ReLU activations based on the ESCNN framework to construct equivariant MLPs. The core idea is that the network weight matrices are constrained to a specific structure satisfying equivariance, such that:

$$f(\mathcal{F}_{input}(x)) = \mathcal{F}_{output}(f(x))$$

For the latent feature $z$ (of even dimension), the symmetry transformation $\mathcal{F}_z$ is defined as swapping adjacent elements: $[\mathcal{F}_z(z)]_i = z_{i+1}$ (for odd $i$) or $z_{i-1}$ (for even $i$). This is a concise yet effective design that bridges the different symmetry transformations of the input and output.

### Loss & Training

Training is based on standard PPO with the following loss terms:

1. **PPO policy loss**: $\mathcal{L}_{PPO} = \mathbb{E}[\min(\rho_\pi A, \text{clip}(\rho_\pi, 1-\xi, 1+\xi) A)]$
2. **Autoencoder loss**: $\mathcal{L}_{AE} = \text{MSE}(\hat{o}, o_{t+1})$, for training the history encoder.
3. **Value loss**: $\mathcal{L}_V = \text{MSE}(V(H_t, o_t), y)$, where $y$ is the reward-to-go.

Auxiliary training techniques:
- **Curriculum learning**: Progressive increases in terrain difficulty (flat → rough → discrete → slopes), task command range, and sensor noise.
- **Domain randomization**: Friction coefficient $[0.7, 1.0]$, base mass $\pm5\,\text{kg}$, motor strength $[0.9, 1.1]$, motor delay $[0.02, 0.1]\,\text{s}$, etc., to reduce the sim-to-real gap.

The reward function covers 14 terms including velocity tracking (weight 2.0), survival reward (2.0), z-axis velocity penalty ($-1.0$), action smoothness penalty ($-0.01$), and joint regularization.

## Key Experimental Results

### Table 1: Core Metric Comparison (Simulation, Velocity Tracking Task)

| Metric | DreamWaQ | DreamWaQ-Regu | SE-Policy (actor only) | **SE-Policy** |
|--------|----------|---------------|----------------------|---------------|
| TE-V (cm/s) ↓ | 16.43±9.54 | 13.91±8.53 | 11.06±8.63 | **9.85±1.54** |
| Temp-S (×10⁻²rad) ↓ | 22.52±2.70 | 16.58±2.88 | 9.20±2.19 | **7.86±1.44** |
| Spat-S (×10⁻²rad) ↓ | 30.84±5.20 | 8.18±1.46 | 0.00±0.00 | **0.00±0.00** |

**Core findings**: SE-Policy achieves a velocity tracking error of only 9.85 cm/s, 40.0% lower than DreamWaQ and 29.2% lower than DreamWaQ-Regu. Spatial symmetry reaches a perfect 0.00—a theoretical guarantee of strict equivariance. Temporal symmetry is also optimal, indicating that the equivariant constraint simultaneously improves the periodic consistency of gait.

### Ablation Study

| Comparison Dimension | DreamWaQ | SE-Policy | Improvement |
|----------------------|----------|-----------|-------------|
| Mean left foot stride | 0.13m | Consistent | Symmetric gait |
| Mean right foot stride | 0.25m | Consistent | Symmetric gait |
| Trajectory symmetry | Noticeable left-turn bias | Mirror-symmetric | Trajectory accuracy ↑ |
| Cumulative position error (TE-P) | Diverges over time | Grows slowly | Long-term stability ↑ |
| Cumulative heading error (TE-O) | Diverges over time | Grows slowly | Directional control ↑ |

**Ablation findings**: Removing the invariant critic (SE-Policy actor only) raises TE-V by 12.2% and Temp-S by 17.0%, demonstrating that critic invariance is also important for policy optimization—an invariant critic provides consistent value estimation signals for the equivariant actor.

## Highlights & Insights

1. **Paradigm shift from soft to hard constraints**: Unlike data augmentation or regularization that yield "expected equivariance," SE-Policy guarantees "necessary equivariance" through the network architecture itself, eliminating the possibility of symmetry violation at inference time. This approach generalizes to any robot with morphological symmetry.

2. **Theoretical guarantee of zero Spat-S**: A spatial symmetry score of exactly zero is a direct mathematical consequence of the architectural constraint, not an approximation from training optimization—symmetric observations produce mathematically equivalent actions.

3. **Indirect benefits of equivariance**: Strict equivariance not only improves spatial symmetry but also significantly enhances temporal symmetry (Temp-S reduced by 65%), indicating that the equivariant constraint effectively restricts the policy search space, making it easier for optimization to find periodically coordinated gaits.

4. **Successful sim-to-real transfer**: The policy handles diverse terrains on a physical Unitree G1—grass, slopes, sand, gravel—demonstrating that equivariance remains robust in the presence of domain gaps.

5. **Elegant latent feature symmetry transformation design**: The simple adjacent-element-swapping rule for $\mathcal{F}_z$ satisfies group-theoretic requirements while being straightforward to implement, serving as a key bridge between different symmetry group representations of the input and output.

## Limitations & Future Work

1. **Only reflection symmetry validated**: The current method focuses on the left-right mirror symmetry ($\mathbb{Z}_2$ group) of humanoid robots and does not extend to richer symmetry groups (e.g., rotational symmetry), limiting applicability to non-bilateral robots.
2. **Task scope is limited**: Validation is confined to velocity tracking tasks; manipulation tasks (grasping, carrying) and highly dynamic tasks (running, jumping) are not addressed, and the symmetry assumption may be violated in such scenarios.
3. **Latent space symmetry design is hand-crafted**: The adjacent-swapping rule for $\mathcal{F}_z$ is manually specified; for more complex robot morphologies, the optimal latent space symmetry transformation may not be immediately apparent.
4. **No comparison with motion imitation methods**: Motion capture-based approaches such as AMP can already produce natural locomotion, yet no direct comparison with such methods is provided.
5. **Computational overhead not analyzed**: Parameter sharing in equivariant layers may reduce parameter count, but whether the equivariant constraints in ESCNN introduce additional computational cost is not discussed.

## Related Work & Insights

- **DreamWaQ** (nahrendra2023dreamwaq): The direct baseline. A model-free PPO-based method using a history encoder to extract motion context. SE-Policy is fully compatible with its architecture, replacing only the MLPs with equivariant MLPs.
- **ESCNN** (cesa2022program): Provides the implementation framework for equivariant networks, realizing group equivariance constraints on linear layers via parameter sharing.
- **Symmetry in MDPs** (zinkevich2001symmetry; van2020mdp): Theoretical foundation—in a symmetric MDP, the optimal policy is necessarily equivariant and the optimal value function is necessarily invariant.
- **Implications for quadruped robots**: The ideas behind SE-Policy transfer directly to quadruped robots (which possess richer $C_2$ or $C_4$ symmetry) and may yield even greater performance gains.
- **Implications for multi-agent systems**: Policy sharing among homogeneous agents is essentially a form of equivariance; the architectural design of SE-Policy offers relevant insights.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ⭐⭐⭐ |
| Theoretical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| **Overall** | **⭐⭐⭐⭐** |

The theoretical derivation is complete, naturally motivating the architectural design from the properties of symmetric MDPs; experiments cover both simulation and physical deployment with well-designed ablations. However, the core contribution lies in engineering the existing equivariant network technique (ESCNN) to fit humanoid robots, making the novelty more of an applied engineering contribution than a methodological breakthrough. The task scenarios are relatively narrow (velocity tracking only), and validation on more complex tasks would be desirable.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning](../../NeurIPS2025/video_understanding/adversarial_locomotion_and_motion_imitation_for_humanoid_policy_learning.md)
- [\[ICLR 2026\] Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning](../../ICLR2026/video_understanding/stabilizing_policy_gradients_for_sample-efficient_reinforcement_learning_in_llm_.md)
- [\[CVPR 2026\] VideoChat-M1: Collaborative Policy Planning for Video Understanding via Multi-Agent Reinforcement Learning](../../CVPR2026/video_understanding/videochatm1_collaborative_policy_planning_for_vide.md)
- [\[AAAI 2026\] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)
- [\[CVPR 2026\] Dual-Agent Reinforcement Learning for Adaptive and Cost-Aware Visual-Inertial Odometry](../../CVPR2026/video_understanding/dual-agent_reinforcement_learning_for_adaptive_and_cost-aware_visual-inertial_od.md)

<!-- RELATED:END -->
