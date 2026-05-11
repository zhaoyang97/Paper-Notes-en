---
title: >-
  [Paper Note] Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control
description: >-
  [ICLR 2026][Reinforcement Learning][humanoid robot control] LIFT proposes a three-stage pretraining-finetuning framework: (i) large-scale parallel SAC pretraining for zero-shot deployment…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "humanoid robot control"
  - "large-scale pretraining"
  - "efficient finetuning"
  - "SAC"
  - "physics-prior world model"
  - "sim-to-real"
date: 2026-05-08
content_hash: 96325f192a5e3a5a
---

# Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control

**Conference**: ICLR 2026
**arXiv**: [2601.21363](https://arxiv.org/abs/2601.21363)
**Code**: [https://lift-humanoid.github.io](https://lift-humanoid.github.io)
**Area**: Reinforcement Learning
**Keywords**: humanoid robot control, large-scale pretraining, efficient finetuning, SAC, physics-prior world model, sim-to-real

## TL;DR
LIFT proposes a three-stage pretraining-finetuning framework: (i) large-scale parallel SAC pretraining for zero-shot deployment; (ii) offline pretraining of a physics-prior world model based on Lagrangian dynamics; (iii) efficient finetuning via deterministic action execution in the environment combined with stochastic exploration within the world model. The full sim-to-real pipeline is validated on Booster T1 and Unitree G1 humanoid robots.

## Background & Motivation

**Background**: PPO has become the dominant method for humanoid robot control due to its robust convergence under large-scale parallel GPU simulation, enabling zero-shot deployment. However, the low sample efficiency of on-policy methods limits safe adaptation to new environments.

**Limitations of Prior Work**: (1) Large-scale parallel training with off-policy methods has received insufficient attention; (2) stochastic exploration during finetuning risks actuator damage or unsafe states, particularly dangerous for humanoids with small support polygons; (3) training model-based methods from scratch is time-consuming and prone to local optima.

**Key Challenge**: Large-scale pretraining requires the stability and parallel efficiency of on-policy methods, while efficient finetuning demands the sample efficiency of off-policy methods and the data efficiency of model-based approaches.

**Goal**: To unify the algorithm choice across pretraining and finetuning stages while guaranteeing both safety and efficiency.

**Key Insight**: SAC is adopted as a unified backbone—pretraining employs high UTD with large-batch parallel training, while finetuning confines stochastic exploration to the world model, with only deterministic actions executed in the real environment.

**Core Idea**: SAC serves as the backbone throughout the pretraining-finetuning pipeline; a physics-prior world model bridges simulation and reality; deterministic execution combined with in-model exploration enables safe and efficient finetuning.

## Method

### Overall Architecture
LIFT consists of three stages: (i) large-scale SAC pretraining in JAX with 1024 parallel environments and high UTD=10, achieving rapid convergence in MuJoCo Playground; (ii) offline training of a physics-prior world model on pretraining data; (iii) collecting data in the new environment with a deterministic policy, generating synthetic trajectories via stochastic exploration within the world model, and alternately finetuning the policy and world model.

### Key Designs

1. **Large-Scale SAC Pretraining**:

    - Function: Parallel SAC training on GPU, achieving wall-clock efficiency comparable to PPO.
    - Mechanism: A fully compiled JAX implementation of SAC with fixed tensor shapes for efficient operator fusion, large-batch updates (batch=1024), and high UTD (=10) across 1024 parallel environments with no additional communication overhead.
    - Employs asymmetric actor-critic—the actor receives proprioceptive state $s_t$, while the critic receives privileged state $s_t^p$.
    - Design Motivation: The off-policy nature of SAC makes it more naturally compatible with model-based methods than PPO; the state-dependent stochastic policy provides greater exploration diversity during world model rollouts.

2. **Physics-Prior World Model**:

    - Function: A hybrid model combining rigid-body dynamics priors with learned residuals.
    - Mechanism: Built on the Lagrangian equation $M(q_t)\ddot{q_t} + C(q_t,\dot{q_t}) + G(q_t) = B\tau_t + J^\top F^e_t + \tau^d_t$, where $M, C, G, B$ are known (determined by robot geometry and inertial parameters), and a residual network $\tau_\phi(s_t,a_t) \approx J^\top F^e_t + \tau^d_t$ learns the unknown contact forces and dissipative terms.
    - Loss function: $\mathcal{L}_\phi = \frac{1}{B}\sum_{b=1}^{B}((\hat{s}^p_{b,t+\Delta t} - s^p_{b,t+\Delta t})^2 \odot \exp(-\log\sigma^2_{b,t}) + \log\sigma^2_{b,t})$
    - Design Motivation: Pure neural network world models generalize poorly under limited data and produce physically implausible predictions that cause critic loss divergence.

3. **Safe Finetuning Strategy**:

    - Function: Safely and efficiently finetune the policy in a new environment.
    - Mechanism: Deterministic actions (action mean) are executed in the environment; stochastic exploration is confined to world model rollouts. Initial states are sampled from the replay buffer, and $H_{wm}=20$-step trajectories are rolled out within the world model for actor-critic training.
    - Safe reset: Rollouts are immediately terminated when base height, velocity, orientation, or joint states exceed predefined thresholds.
    - Design Motivation: The small support polygon of humanoid robots makes them highly sensitive to perturbations during single-support phases, where stochastic exploration may cause falls.

### Loss & Training
- Pretraining: Standard SAC objective with Optuna hyperparameter search (~10 hours); Booster T1 training time reduced from 7 hours to 30 minutes.
- World model: Gaussian negative log-likelihood loss with end-to-end gradients backpropagated through normalization, coordinate transforms, PD controller, and Euler integration.
- Finetuning: Multi-epoch autoregressive training to enhance sample efficiency; autoregressive loss with length 2–4 stabilizes learning.

## Key Experimental Results

### Pretraining Results (6 Humanoid Tasks)
- LIFT achieves peak return comparable to PPO and FastTD3 on all flat terrain tasks.
- Converges to peak performance more rapidly on rough terrain.
- Booster T1 pretraining completes within 30 minutes on a single GPU (RTX 4090).

### Finetuning Results (Brax Environment, 8 Seeds)

| Scenario | LIFT | SAC | PPO | FastTD3 | SSRL |
|---|---|---|---|---|---|
| In-Distribution (0.6 m/s) | ✓ Converges | Diverges | Initially stable, then collapses | Strong oscillations, then collapses | Signs of convergence but below target |
| Long-Tail (1.0 m/s) | ✓ Converges | Diverges | Collapses | Collapses | Does not converge |
| OOD (1.5 m/s) | ✓ Converges | Diverges | Collapses | Collapses | Does not converge |

Finetuning requires only $4 \times 10^4$ environment steps (~800 seconds of online interaction).

### Ablation Study

| Configuration | Result |
|---|---|
| Full LIFT (SAC pretraining + WM pretraining) | Converges to target velocity within $4\times10^4$ steps |
| Without WM pretraining | Still converges but noticeably slower |
| Without SAC + WM pretraining (= SSRL) | Learns to stand only; near-zero forward velocity |
| MBPO ensemble replacing physics-prior WM | Does not converge; critic loss explodes |

### Real-World Finetuning
- After 80–590 seconds of real-world data collection, the robot exhibits a more upright posture, smoother gait, and more stable forward velocity.
- Limitation: Relies on Vicon motion capture to estimate base height; IMU integration suffers from drift.

## Highlights & Insights
- **Advantage of a unified backbone**: Using SAC throughout pretraining and finetuning avoids objective inconsistency and catastrophic forgetting caused by algorithm switching.
- **Critical role of physics priors**: Ablation experiments quantitatively demonstrate that pure neural network world models are entirely ineffective under limited data; physics priors supply the necessary inductive bias for generalization.
- **Safe exploration paradigm**: Deterministic execution combined with in-model stochastic exploration constitutes a generalizable paradigm applicable to any robotic system requiring safe finetuning.
- **Engineering contribution**: A state mapping error in the SSRL codebase was identified and corrected, and base height was added to the privileged state—a critical fix for humanoid robots.

## Limitations & Future Work
- Current implementation uses only proprioceptive observations without visual input.
- Real-world finetuning depends on external motion capture systems and IMU integration.
- The finetuning pipeline is synchronous (data collection → training); an asynchronous pipeline could substantially improve efficiency.
- Action corrections may be unbounded (cf. ASAP's delta-action approach).

## Related Work & Insights
- **vs. PPO**: PPO gradually degrades and collapses under deterministic data collection and limited data, making it unsuitable for finetuning scenarios.
- **vs. SSRL**: LIFT is essentially a pretraining-augmented version of SSRL, validating that training model-based methods from scratch on humanoids is infeasible.
- **vs. FastTD3**: FastTD3 achieves large-scale off-policy training but lacks finetuning validation and sim-to-real transfer.
- **vs. DreamerV3**: Dreamer employs a latent world model with learned rewards and is unstable under deterministic data collection.

## Rating
- Novelty: ⭐⭐⭐⭐ A systematic solution for humanoid robots combining a pretraining-finetuning framework with a physics-prior world model.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Simulation and real-world evaluation across multiple platforms (T1/G1), multiple scenarios (in/out-of-distribution), and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear and ablation design is well-reasoned, though the paper is lengthy.
- Value: ⭐⭐⭐⭐⭐ Provides a complete open-source pipeline with direct practical value for the humanoid robot learning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Regret-Guided Search Control for Efficient Learning in AlphaZero](regret-guided_search_control_for_efficient_learning_in_alphazero.md)
- [\[ICLR 2026\] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning](rethinking_policy_diversity_in_ensemble_policy_gradient_in_large-scale_reinforce.md)
- [\[ICLR 2026\] RLP: Reinforcement as a Pretraining Objective](rlp_reinforcement_as_a_pretraining_objective.md)
- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [\[ICLR 2026\] ReMix: Reinforcement Routing for Mixtures of LoRAs in LLM Finetuning](remix_reinforcement_routing_lora.md)

</div>

<!-- RELATED:END -->
