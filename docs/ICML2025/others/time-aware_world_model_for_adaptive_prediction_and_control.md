---
title: >-
  [Paper Note] Time-Aware World Model for Adaptive Prediction and Control
description: >-
  [ICML 2025][world model] Proposed the Time-Aware World Model (TAWM), which conditions the model on the time step $\Delta t$ as an explicit input and mixes various $\Delta t$ samples during training, enabling the model to adapt to arbitrary time resolutions at inference time via single-step prediction without increasing sample complexity.
tags:
  - "ICML 2025"
  - "world model"
  - "model-based RL"
  - "time-step conditioning"
  - "Nyquist-Shannon theorem"
  - "multi-scale dynamics"
date: 2026-05-08
content_hash: 7bdd8ca71744ee0e
---

tags:
  - ICML 2025
  - Others
  - world model
  - model-based RL
  - time-step conditioning
  - Nyquist-Shannon theorem
  - multi-scale dynamics
date: 2026-05-08
content_hash: a5c005b9a8e25dfa
---
# Time-Aware World Model for Adaptive Prediction and Control

---

**Conference**: ICML 2025  
**arXiv**: [2506.08441](https://arxiv.org/abs/2506.08441)  
**Code**: [https://github.com/anh-nn01/Time-Aware-World-Model](https://github.com/anh-nn01/Time-Aware-World-Model)  
**Area**: Other  
**Keywords**: world model, model-based RL, time-step conditioning, Nyquist-Shannon theorem, multi-scale dynamics

## TL;DR

Proposed the Time-Aware World Model (TAWM), which conditions the model on the time step $\Delta t$ as an explicit input and mixes various $\Delta t$ samples during training, enabling the model to adapt to arbitrary time resolutions at inference time via single-step prediction without increasing sample complexity.

---

## Background & Motivation

**Background**: Model-based reinforcement learning (MBRL) improves sample efficiency by learning a transition model of the environment (world model), with representative methods including the Dreamer series and TD-MPC2. These methods model state transitions $D:(s_t, a_t) \to s_{t+1}$ in latent space, and then utilize the learned dynamics model for planning or policy optimization.

**Limitations of Prior Work**: Existing world models are trained with a fixed time step $\Delta t$ (e.g., 2.5ms), leading to three key issues. (1) **Time-resolution overfitting**: Models trained on a fixed $\Delta t$ suffer from severe performance drops when deployed to real-world environments with different observation rates (e.g., from 400Hz down to 50Hz) because cumulative errors compound with the number of rollout steps. (2) **Inaccurate dynamics learning**: Models that are not conditioned on $\Delta t$ may fail to capture the true underlying system dynamics, only fitting the surface-level behavior at a specific sampling rate. (3) **Inefficient dynamics learning**: Real-world systems often consist of sub-systems operating on multiple time scales; sampling at a uniformly high frequency generates highly redundant data for slow-changing sub-systems, wasting computational resources.

**Key Challenge**: The tension between the observation rate (the inverse of $\Delta t$) and the efficiency of learning dynamics—according to the Nyquist-Shannon sampling theorem, the optimal sampling rate for signal reconstruction depends on its highest frequency component. However, multi-scale systems feature sub-systems with different frequencies, meaning no single optimal sampling rate exists.

**Goal**: How to efficiently train a world model to learn the underlying task dynamics across different time steps $\Delta t$ without increasing sample complexity?

**Key Insight**: Treat $\Delta t$ as an explicit conditional input to the world model rather than a fixed constant. Guided by the Nyquist-Shannon theorem—which implies that different sub-systems have different optimal sampling rates—mixing multiple values of $\Delta t$ during training allows each sub-system to be learned from data close to its optimal sampling rate.

**Core Idea**: Conditioning the world model explicitly on the time step, combined with a mixed-time-step training strategy, and replacing multi-step rollouts with single-step predictions to achieve robust control across different time resolutions.

## Method

### Overall Architecture

TAWM is built upon the TD-MPC2 architecture. The encoder $h$ maps the observation $o_t$ to a latent space $z_t$ (without conditioning on $\Delta t$). The dynamics model, reward model, value model, and policy prior are all conditioned on $\Delta t$ as an additional input. During training, each episode samples a $\Delta t$ from a log-uniform distribution $\text{Log-Uniform}(\Delta t_{min}, \Delta t_{max})$, interacts with the environment using this time step to collect data, and updates the model. The model predicts the state after an arbitrary $\Delta t$ in a single step, thereby avoiding the compounding errors of multi-step rollouts.

### Key Designs

1. **Time-Aware Latent Dynamics Model**:

    - Function: Predicts latent state transitions conditioned on $\Delta t$ in the latent space.
    - Mechanism: Modifies the baseline's direct mapping $z_{t+1} = D(z_t, a_t)$ into an Euler integration form: $\hat{z}_{t+\Delta t} = z_t + d(z_t, a_t, \Delta t) \cdot \tau(\Delta t)$, where $d(\cdot)$ is a latent space derivative function modeled by an MLP, and $\tau(\Delta t) = \max(0, \log_{10}(\Delta t) + 5)$ is a log-normalization function. This automatically ensures identity mapping when $\Delta t = 0$: $z_{t+0} = z_t$.
    - Design Motivation: Since $\Delta t$ spans multiple orders of magnitude ($10^{-3}$ to $5 \times 10^{-2}$), direct use of a linear $\Delta t$ causes numerical instability. The logarithmic transformation $\tau$ normalizes $\Delta t$ to a narrow range, making it easier for the MLP to learn. For tasks with complex dynamics, RK4 integration can be used instead of Euler integration.

2. **Mixed-Time-Step Training Strategy**:

    - Function: Enables the world model to learn multi-scale underlying dynamics within a fixed training budget.
    - Mechanism: At the start of each episode, $\Delta t$ is sampled from $\text{Log-Uniform}(\Delta t_{min}, \Delta t_{max})$, and data is collected at this $\Delta t$ to train the model. The log-uniform distribution allocates equal probability mass across each order of magnitude of time scales, ensuring sufficient training data for both high-frequency and low-frequency sub-systems. All training data is stored in a unified replay buffer $\mathcal{B}$, from which batches are sampled for updates at each step.
    - Design Motivation: According to the Nyquist-Shannon theorem, each sub-system $f_i$ has a maximum frequency $f_{max}^i$. By dynamically varying the sampling rate to cover the vicinity of the Nyquist rate for all frequencies, the strategy avoids under-sampling high-frequency sub-systems and over-sampling low-frequency sub-systems.

3. **Time-Aware Reward and Value Models**:

    - Function: Generalizes reward prediction and value estimation to varying time steps.
    - Mechanism: The reward model $\hat{r}_t = R(z_t, a_t, \Delta t)$ and value model $\hat{q}_t = Q(z_t, a_t, \Delta t)$ both take $\Delta t$ as input. This is because the same state-action pair can yield different immediate rewards and long-term returns under different $\Delta t$. The policy prior $\hat{a}_t = p(z_t, \Delta t)$ is also conditioned on $\Delta t$.
    - Design Motivation: If only the dynamics model is time-aware while the reward/value models are not, the planner cannot correctly evaluate action quality across different time resolutions.

### Theoretical Analysis

**Lemma 4.1**: When the environment dynamics are fully captured by $\Delta\bar{t}$ (i.e., smaller $\Delta t$ provides no additional information), the optimal dynamics function $d^*$ satisfies a simple interpolation relationship across different $\Delta t$.

**Lemma 4.2**: Reducing modeling error at $\Delta\bar{t}$ decreases the error upper bound for all $\Delta t < \Delta\bar{t}$—improvements at larger time scales can "transfer" to smaller time scales.

These two lemmas provide a theoretical explanation of why mixed $\Delta t$ training does not increase sample complexity.

## Key Experimental Results

### Main Results — Meta-World Control Tasks

| Task | Baseline ($\Delta t=2.5$ms) Success Rate | TAWM-Euler Success Rate | TAWM-RK4 Success Rate |
|------|------|------|------|
| Assembly @2.5ms | ~80% | ~85% | **~90%** |
| Assembly @20ms | ~10% | ~70% | **~80%** |
| Basketball @2.5ms | ~90% | **100%** | **100%** |
| Basketball @20ms | ~0% | ~60% | **~70%** |
| Faucet Open @50ms | ~0% | ~80% | **~90%** |

*The baseline drops performance sharply under non-default $\Delta t$, whereas TAWM remains robust.*

### Comparison with MTS3

| Method | Faucet Open @2.5ms | @10ms | @30ms | @50ms |
|------|-------------------|-------|-------|-------|
| MTS3 (H=2) | ~80% | ~40% | ~10% | ~0% |
| MTS3 (H=5) | ~80% | ~50% | ~20% | ~5% |
| TAWM-RK4 | **~95%** | **~95%** | **~90%** | **~90%** |

*MTS3's performance decreases rapidly as $\Delta t$ increases (compounding errors), whereas TAWM maintains ~90%.*

### Key Findings

- TAWM is competitive with, or even outperforms, baselines specifically trained on the default $\Delta t = 2.5$ms, while significantly leading at larger $\Delta t$.
- Baselines trained solely at low observation rates ($\Delta t \geq 10$ms) completely fail to converge (0% success rate on all tasks), whereas TAWM's mixed training consistently outperforms any single-rate fixed $\Delta t$ training.
- Euler integration is sufficient for most Meta-World tasks, while RK4 is more advantageous in PDE control tasks with complex dynamics.
- Learning curves show that TAWM's convergence speed is comparable to or faster than the baseline, confirming no increase in sample requirements.
- Log-uniform sampling is more effective for high-frequency dynamics tasks, while uniform sampling offers additional benefits for low-frequency dynamics tasks.

## Highlights & Insights

- Extremely simple Core Idea: Simply conditioning on $\Delta t$ as an extra input and performing mixed $\Delta t$ training yields substantial generalization across different time resolutions.
- The Nyquist-Shannon theorem provides an elegant theoretical motivation, bringing classical insights from signal processing into world model training.
- Although simple, the design of the log-normalization function $\tau(\Delta t)$ is crucial for training stability.
- Architecture-agnostic: TAWM can be applied out-of-the-box to any existing world model architecture.

## Limitations & Future Work

- The choice of $\Delta t_{max}$ still relies on task-specific heuristics, lacking an automated method to determine the maximum frequency $f_{max}$.
- The current dynamics model is deterministic. Since actual transitions under larger $\Delta t$ exhibit higher stochasticity, probabilistic extensions are worth exploring.
- The theoretical analysis relies on strong assumptions ("sufficient model capacity", "learnable interpolation relationships"), limiting its rigor.
- Evaluation is limited to Meta-World and 1D PDE control, without testing on complex control tasks with visual inputs.
- Fixing $\Delta t$ per episode might be sub-optimal compared to dynamically adjusting $\Delta t$ within an episode.

## Related Work & Insights

- TD-MPC2 (Hansen et al., 2024) is the direct baseline framework; TAWM only modifies its conditional input dimension.
- MTS3 (Shaj Kumar et al., 2023) focuses on multi-scale dynamics but is still trained on a fixed $\Delta t$ and utilizes multi-step rollouts; the proposed method bypasses compounding errors through single-step conditional prediction.
- Neural ODE-based methods also introduce continuous time into neural networks, but they incur higher computational costs and are not directly applicable to MBRL.

## Rating

⭐⭐⭐⭐ The proposed method is simple yet effective (introducing only $\Delta t$ input + mixed $\Delta t$ training). Experiments on 9 Meta-World tasks and 3 PDE control tasks fully demonstrate its practical value for generalization across different time resolutions. The Nyquist-Shannon theorem provides an elegant theoretical motivation, and Lemmas 4.1-4.2 offer a theoretical explanation of sample efficiency. The comparison with MTS3 clearly demonstrates the benefits of single-step conditional prediction over multi-step rollouts. However, the rigor of the theoretical analysis is limited by strong assumptions, and it has not been validated on visual-input scenarios. It holds potential importance for sim-to-real transfer.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Prediction-Powered Adaptive Shrinkage Estimation](prediction-powered_adaptive_shrinkage_estimation.md)
- [\[NeurIPS 2025\] Deep Learning for Continuous-Time Stochastic Control with Jumps](../../NeurIPS2025/others/deep_learning_for_continuous-time_stochastic_control_with_jumps.md)
- [\[ICML 2025\] Cross-regularization: Adaptive Model Complexity through Validation Gradients](cross-regularization_adaptive_model_complexity_through_validation_gradients.md)
- [\[ICML 2025\] Symmetry-Aware GFlowNets](symmetry-aware_gflownets.md)
- [\[ICML 2025\] General Agents Contain World Models](general_agents_contain_world_models.md)

</div>

<!-- RELATED:END -->
