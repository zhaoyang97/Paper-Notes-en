---
title: >-
  [Paper Note] Reward-free World Models for Online Imitation Learning
description: >-
  [ICML2025][Reinforcement Learning][imitation learning] Proposes IQ-MPC, a reward-free world model online imitation learning method that jointly learns the dynamics model and Q-function in latent space via inverse soft-Q learning, achieving stable expert-level imitation in high-dimensional observation and complex dynamics tasks using MPPI planning.
tags:
  - "ICML2025"
  - "Reinforcement Learning"
  - "imitation learning"
  - "world model"
  - "reward-free"
  - "inverse soft-Q learning"
  - "model predictive control"
  - "latent dynamics"
date: 2026-05-08
content_hash: 70f8dc15f4b542ca
---

# Reward-free World Models for Online Imitation Learning

**Conference**: ICML2025  
**arXiv**: [2410.14081](https://arxiv.org/abs/2410.14081)  
**Code**: To be confirmed  
**Area**: Imitation Learning / World Models / Model Predictive Control  
**Keywords**: imitation learning, world model, reward-free, inverse soft-Q learning, model predictive control, latent dynamics

## TL;DR

Proposes IQ-MPC, a reward-free world model online imitation learning method that jointly learns the dynamics model and Q-function in latent space via inverse soft-Q learning, achieving stable expert-level imitation in high-dimensional observation and complex dynamics tasks using MPPI planning.

## Background & Motivation

- **Limitations of Offline Imitation Learning**: Behavior Cloning (BC) methods (e.g., Diffusion Policy, Implicit BC) rely on massive expert data, struggle with out-of-distribution (OOD) states, and suffer from error accumulation and performance degradation.
- **Limitations of Prior Work**: Existing online IL methods (e.g., GAIL, IQ-Learn, CFIL) perform poorly in high-dimensional observation/action spaces and complex dynamics tasks; IRL-based min-max optimization is unstable in the reward-policy space.
- **Potential of World Models**: Decoder-free world models like the TD-MPC series demonstrate remarkable sample efficiency and planning capabilities in RL, but have not been effectively applied to reward-free imitation learning scenarios.
- **Goal**: Can the dynamic modeling of world models enhance online imitation learning performance while completely eliminating the reliance on explicit reward models?

## Method

### Overall Architecture: IQ-MPC

IQ-MPC consists of four core components, all operating in the latent space without reconstructing original observations:

1. **Encoder** $h$: $\mathbf{z} = h(\mathbf{s})$, mapping states to latent representations.
2. **Latent Dynamics Model** $d$: $\mathbf{z}' = d(\mathbf{z}, \mathbf{a})$, predicting the next latent state.
3. **Q-function** $Q$: $\hat{q} = Q(\mathbf{z}, \mathbf{a})$, estimating state-action values.
4. **Policy Prior** $\pi$: $\hat{\mathbf{a}} = \pi(\mathbf{z})$, guiding MPPI planning.

The system maintains two independent replay buffers: the expert buffer $\mathcal{B}_E$ and the behavior buffer $\mathcal{B}_\pi$.

### Core Idea: Reward-free Optimization in the Q-Policy Space

Key Insight: The inverse Bellman operator $\mathcal{T}^\pi$ establishes a bijective mapping between the Q-space and the reward space:

$$r(\mathbf{s}, \mathbf{a}) = Q(\mathbf{s}, \mathbf{a}) - \gamma \mathbb{E}_{\mathbf{s}' \sim \mathcal{P}(\cdot|\mathbf{s},\mathbf{a})} V^\pi(\mathbf{s}')$$

Consequently, there is no need to train a separate reward model; rewards can be directly decoded from Q-values and the policy prior. The optimization is shifted from the reward-policy space to the Q-policy space, bypassing the instability of min-max optimization.

### Joint Training Loss

The joint training objective for the encoder, dynamics model, and Q-function is:

$$\mathcal{L} = \sum_{t=0}^{H} \lambda^t \left( \mathbb{E}_{(\mathbf{s}_t, \mathbf{a}_t, \mathbf{s}'_t) \sim \mathcal{B}} \| \mathbf{z}_{t+1} - \text{sg}(h(\mathbf{s}'_t)) \|_2^2 \right) + \mathcal{L}_{iq}$$

- The first term is the **consistency loss**: ensuring the latent states predicted by the dynamics model are consistent with the actual next states encoded by the encoder (where sg denotes stop-gradient).
- The second term is the **inverse soft-Q loss** $\mathcal{L}_{iq}$: using $\chi^2$ regularization, which consists of three parts—Q-estimates on expert data, initial state value function terms, and a regularization penalty on Q-value magnitudes.

### Policy Prior Learning

The policy is learned via a maximum entropy RL objective:

$$\mathcal{L}_\pi = \sum_{t=0}^{H} \lambda^t \left[ \mathbb{E}_{(\mathbf{s}_t, \mathbf{a}_t) \sim \mathcal{B}} \left[ -Q(\mathbf{z}_t, \pi(\mathbf{z}_t)) + \beta \log(\pi(\cdot|\mathbf{z}_t)) \right] \right]$$

where $\beta$ is a fixed entropy coefficient. Policy learning utilizes a mixture of data from both the expert and behavior buffers.

### Gradient Penalty for Stable Training

To address the issue where policy learning fails due to an overly dominant critic, a Wasserstein-1 gradient penalty is introduced:

$$\mathcal{L}_{pen} = \sum_{t=0}^{H} \lambda^t \left[ \mathbb{E}_{(\hat{\mathbf{s}}_t, \hat{\mathbf{a}}_t) \sim \mathcal{B}} \left( \| \nabla Q(\hat{\mathbf{z}}_t, \hat{\mathbf{a}}_t) \|_2 - 1 \right)^2 \right]$$

The gradient penalty points are generated through linear interpolation between expert and behavior samples, enforcing the Lipschitz condition on the Q-function.

### MPPI Planning (Inference Phase)

During inference, gradient-free planning is performed using MPPI (Model Predictive Path Integral):

1. Encode the current state $\mathbf{z}_t = h(\mathbf{s}_t)$.
2. Sample $N$ and $N_\pi$ action trajectories from the Gaussian distribution and policy prior, respectively.
3. Roll out through the dynamics model, and decode rewards using the inverse Bellman operator: $r(\mathbf{z},\mathbf{a}) = Q(\mathbf{z},\mathbf{a}) - \gamma V^\pi(\mathbf{z}')$.
4. Accumulate soft returns and add the terminal value estimation $\gamma^H V^\pi(\mathbf{z}_H)$.
5. Iteratively update Gaussian parameters $(\mu, \sigma)$ and execute the first action.

## Key Experimental Results

### Locomotion Tasks (DMControl, State Input)

| Task | IQL+SAC | CFIL+SAC | HyPE | **IQ-MPC** |
|------|---------|----------|------|-----------|
| Hopper Hop | Unstable | Unstable | Moderate | **Stable Expert-level** |
| Walker Run | Moderate | Low | Moderate | **Stable Expert-level** |
| Humanoid Walk | Low | Low | Moderate | **Stable Expert-level** |
| Dog Walk | Low | Low | Moderate | **Best** |

- Low-dimensional tasks use 100 expert trajectories, Humanoid uses 500, and Dog uses 1000.

### Dexterous Manipulation Tasks (MyoSuite, Success Rate)

| Task | IQL+SAC | CFIL+SAC | HyPE | **IQ-MPC** |
|------|---------|----------|------|-----------|
| Key Turn | 0.72±0.04 | 0.65±0.08 | 0.55±0.09 | **0.87±0.03** |
| Object Hold | 0.00±0.00 | 0.01±0.01 | 0.13±0.10 | **0.96±0.03** |
| Pen Twirl | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | **0.73±0.05** |

- Only uses 100 expert trajectories (each of 100 steps).
- Baselines all achieve a 0% success rate in the Pen Twirl task, while IQ-MPC reaches 73%.

### Visual Input Experiments (DMControl, Image Observations)

- Only replaces the encoder with a shallow convolutional network, leaving the rest of the model unchanged.
- Outperforms the visual version of IQL+SAC significantly on Cheetah Run and Walker Run.
- Matches baseline performance on Walker Walk.

### Ablation Study on Number of Expert Trajectories

- Hopper Hop: Reaches expert-level performance with only 10 expert trajectories (100 trajectories converge faster).
- Object Hold: Reaches expert-level performance with only 5 expert trajectories.
- In Hopper Hop, instability arises with only 5 trajectories.

## Highlights & Insights

1. **Reward-free World Model**: Decodes rewards directly from Q-values using the inverse Bellman operator, completely eliminating reward model training and lowering system complexity.
2. **Q-Policy Space Optimization**: Reformulates the traditional min-max problem in the reward-policy space into optimization in the Q-policy space, yielding theoretically and experimentally proven training stability.
3. **Theoretical Guarantees**: Proves that the training objective simultaneously minimizes the upper bound of the policy return discrepancy (comprising T1: distribution matching + T2: dynamics consistency).
4. **Breakthrough in Dexterous Manipulation**: On high-dimensional musculoskeletal control tasks in MyoSuite, where baseline methods almost entirely fail (success rate ~0%), IQ-MPC still achieves a 73-96% success rate.
5. **Data Efficiency**: Achieves expert-level performance with only 5-10 expert trajectories, demonstrating exceptional sample efficiency.
6. **Modality-agnostic Design**: Highly flexible architecture requiring only an encoder replacement when transitioning from state to visual inputs.

## Limitations & Future Work

1. **Expert Data Acquisition**: Still requires sampling expert trajectories from a pre-trained TD-MPC2 model; obtaining high-quality expert demonstrations in real-world scenarios can be difficult.
2. **Computational Overhead**: MPPI planning requires multiple iterations of sampling and rollouts, incurring higher inference costs than policy-only methods.
3. **Deterministic Environment Assumptions**: Most experiment environments feature deterministic dynamics; performance in high-stochasticity environments remains to be verified.
4. **Single-task Design**: Trains an independent world model for each task, lacking cross-task generalization capabilities.
5. **Lack of Real-robot Validation**: All experiments are conducted in simulated environments; sim-to-real transfer is not discussed.

## Related Work & Insights

- **IQ-Learn** (Garg et al., 2021): Serves as the theoretical foundation for this work, introducing inverse soft-Q learning to shift IRL optimization from the reward space to the Q space.
- **TD-MPC / TD-MPC2** (Hansen et al., 2022/2023): Serves as the architectural blueprint for this work, offering a decoder-free world model + MPPI planning framework.
- **GAIL** (Ho & Ermon, 2016): Classic adversarial imitation learning; this work builds upon it by resolving the instability of min-max optimization.
- **CMIL** (Kolev et al., 2024): Conservative world model for imitation learning in visual manipulation; its bounded suboptimality lemma is referenced in the theoretical analysis.

## Rating

- Novelty: ⭐⭐⭐⭐ — Combining a decoder-free world model with inverse soft-Q learning for online imitation learning is a novel combination; the reward-free planning design is simple and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers three categories of tasks (locomotion, manipulation, and vision) with thorough ablation studies, though it lacks real-robot validation.
- Writing Quality: ⭐⭐⭐⭐ — The theoretical derivations are clear, experimental descriptions are comprehensive, and notations are consistent and standardized.
- Value: ⭐⭐⭐⭐ — The breakthrough performance in dexterous manipulation (baselines success rate ~0% vs. IQ-MPC 73-96%) demonstrates the practical value of the method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Continual Reinforcement Learning by Planning with Online World Models](continual_reinforcement_learning_by_planning_with_online_world_models.md)
- [\[ICML 2025\] Online Pre-Training for Offline-to-Online Reinforcement Learning](online_pre-training_for_offline-to-online_reinforcement_learning.md)
- [\[NeurIPS 2025\] Quantifying Generalisation in Imitation Learning](../../NeurIPS2025/reinforcement_learning/quantifying_generalisation_in_imitation_learning.md)
- [\[NeurIPS 2025\] Foundation Models as World Models: A Foundational Study in Text-Based GridWorlds](../../NeurIPS2025/reinforcement_learning/foundation_models_as_world_models_a_foundational_study_in_text-based_gridworlds.md)
- [\[ICML 2025\] PIGDreamer: Privileged Information Guided World Models for Safe Partially Observable RL](pigdreamer_privileged_information_guided_world_models_for_safe_partially_observa.md)

</div>

<!-- RELATED:END -->
