---
title: >-
  [Paper Note] Neural Motion Simulator: Pushing the Limit of World Models in Reinforcement Learning
description: >-
  [CVPR 2025][Robotics][World Models] This paper proposes MoSim, a world model based on rigid-body dynamics priors and Neural ODEs. Operating in physical state spaces, it performs high-precision, long-horizon predictions, enabling zero-shot reinforcement learning—training policies without any real environment interactions—for the first time.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "World Models"
  - "Neural Motion Simulator"
  - "Neural ODE"
  - "Rigid-Body Dynamics"
  - "Zero-Shot Reinforcement Learning"
date: 2026-05-08
content_hash: 1b7c11288d76d92e
---

# Neural Motion Simulator: Pushing the Limit of World Models in Reinforcement Learning

**Conference**: CVPR 2025  
**arXiv**: [2504.07095](https://arxiv.org/abs/2504.07095)  
**Code**: [https://oamics.github.io/mosim_page/](https://oamics.github.io/mosim_page/)  
**Area**: Reinforcement Learning / World Models  
**Keywords**: World Models, Neural Motion Simulator, Neural ODE, Rigid-Body Dynamics, Zero-Shot Reinforcement Learning

## TL;DR

This paper proposes MoSim, a world model based on rigid-body dynamics priors and Neural ODEs. Operating in physical state spaces, it performs high-precision, long-horizon predictions, enabling zero-shot reinforcement learning—training policies without any real environment interactions—for the first time.

## Background & Motivation

**Background**: World models play a pivotal role in reinforcement learning. Mainstream methods, such as DreamerV3 (based on RSSM) and TD-MPC2, predict future states in latent spaces to improve sample efficiency through "dreaming". However, the prediction capabilities of these models have never been directly and systematically evaluated, as all evaluations are conducted indirectly through downstream tasks.

**Limitations of Prior Work**: Existing world models suffer from two key limitations: (1) Insufficient prediction accuracy, where errors accumulate rapidly during long-horizon ($>30$ steps) prediction, failing to support purely prediction-based policy learning; (2) Latent-space predictions, which tightly couple the world model with specific RL algorithms, preventing decoupling.

**Key Challenge**: If world models were sufficiently accurate and capable of long-horizon predictions, they could theoretically replace real environments entirely for policy training (zero-shot RL). However, currently no models possess this level of precision.

**Goal**: Construct a world model that performs precise, long-horizon predictions directly in the raw physical state space to demonstrate the feasibility of zero-shot/few-shot RL.

**Key Insight**: Rigid-body dynamics possess explicit mathematical structures (mass matrix, conservative forces, and contact forces). Constraining the network architecture with physical structural priors can significantly improve prediction accuracy. Meanwhile, dynamics modeling is decomposed into a smooth term (rigid-body components) and a non-smooth term (residual correction), trained in a multi-stage manner.

**Core Idea**: An architecture combining a Predictor, which embeds rigid-body dynamics inductive biases, with a residual Corrector, integrated with Neural ODEs for continuous-time integration. This realizes high-precision physical state predictions, thereby unlocking zero-shot RL.

## Method

### Overall Architecture

MoSim operates in a continuous physical state space $\mathcal{S}$. Given the current physical state $\boldsymbol{s}(t) = (\boldsymbol{q}, \dot{\boldsymbol{q}})$ (joint positions and velocities) and an action $\boldsymbol{a}(t)$ as inputs, the model outputs the physical state at the next time step. The core formulation is $\dot{\boldsymbol{s}}(t) = \boldsymbol{f}(\boldsymbol{s}(t), \boldsymbol{a}(t)) + \boldsymbol{\epsilon}(\boldsymbol{s}(t), \boldsymbol{a}(t))$, where $\boldsymbol{f}$ is the Predictor based on rigid-body dynamics, and $\boldsymbol{\epsilon}$ is the Corrector handling non-smooth terms like friction and collisions. The entire system is integrated over continuous time using a Neural ODE (specifically, the DOPRI5 integrator) to achieve precise long-horizon predictions from $t_0$ to $t_0+T$.

### Key Designs

1. **Predictor (Rigid-body Dynamics Prior Network)**:
    - **Function**: Models the smooth dynamics of ideal rigid-body motion.
    - **Mechanism**: Parametrizes the three physical quantities in the rigid-body dynamics equation $\ddot{\boldsymbol{q}} = M(\boldsymbol{q})[\boldsymbol{b}(\boldsymbol{s}) + \boldsymbol{\tau}(\boldsymbol{a})]$ through separate, independent networks. A Position Encoder outputs a lower-triangular matrix $L$ to ensure the positive-definiteness of the inverse mass matrix $M = LL^T$ via Cholesky decomposition; a State Encoder uses a ResNet to encode conservative forces $\boldsymbol{b}$; an Action Encoder uses an MLP to encode external force $\boldsymbol{\tau}$. Combining these three yields the acceleration $\ddot{\boldsymbol{q}}$, which is then concatenated with velocity to formulate $\dot{\boldsymbol{s}}$.
    - **Design Motivation**: Leverages the compositional structure of rigid-body dynamics as an inductive bias without needing explicit physical parameters, borrowing only its mathematical formulation. Ablation studies demonstrate that this structure trains faster and exhibits higher accuracy compared to a pure ResNet of equivalent parameter size.

2. **Corrector (Residual Correction Network)**:
    - **Function**: Captures non-smooth dynamics, such as friction, collisions, and contact forces, which are difficult to model explicitly.
    - **Mechanism**: Composed of one or more standard parallel ResNets, it introduces no physical priors and directly learns the residuals that the Predictor fails to fit. For complex robotic platforms, multiple layers of Correctors can be stacked for progressive refinement.
    - **Design Motivation**: Decouples "what can be solved with physical priors" from "what cannot," allowing each component network to specialize.

3. **Neural ODE Continuous Integration**:
    - **Function**: Converts discrete predictions into continuous-time integration, enabling precise predictions at arbitrary step sizes.
    - **Mechanism**: Uses the dynamics function $g_\theta$ composed of the Predictor and Corrector as the right-hand side of the ODE, solved via the DOPRI5 adaptive integrator. Backpropagation is implemented via the adjoint method, eliminating the need to store intermediate states.
    - **Design Motivation**: Continuous-time modeling is inherently suited for physical dynamics. Furthermore, the adaptive step size of DOPRI5 automatically refines time steps when rapid dynamic changes occur.

### Loss & Training

A multi-stage training strategy is adopted: first, only the Predictor is trained to convergence to capture smooth dynamics; then, the Predictor is frozen, and the Corrector is trained to learn the residuals. For complex robots, multiple Correctors can be stacked. The loss function is the MSE between the predicted and ground-truth states. Training data is generated via a random action policy (rather than an RL replay buffer), yielding superior generalization.

## Key Experimental Results

### Main Results

MoSim is evaluated against DreamerV3 (RSSM) and TD-MPC2 across 7 robotic environments (DM Control's Cheetah, Reacher, Acrobot, Hopper, Humanoid, a Panda robotic arm, and a Go2 quadruped):

| Environment | Steps | DreamerV3-r (5-step initialization) | MoSim-rm |
|-------------|-------|------------------------|----------|
| Cheetah     | 100   | 0.2297                 | **0.2185** |
| Reacher     | 100   | 0.0988                 | **0.0009** |
| Acrobot     | 100   | 4.8957                 | **0.1043** |
| Panda       | 100   | 0.0971                 | **0.0043** |
| Hopper      | 100   | 0.3199                 | **0.2507** |
| Go2         | 100   | 0.4165                 | **0.1282** |
| Humanoid    | 16    | 2.1291                 | **1.2737** |

In the TD-MPC2 latent-space evaluation, MoSim also completely outperforms TD-MPC2 itself (e.g., Reacher: 4.8e-5 → 2.9e-7).

### Ablation Study

| Configuration | Hopper MSE | Description |
|-----------|-----------|------|
| Pure ResNet (No rigid-body prior) | Slow convergence, low accuracy | Lack of inductive bias |
| Predictor only | Moderate | Cannot handle non-smooth dynamics |
| End-to-end joint training | Unstable convergence | Corrector interferes with Predictor learning |
| Multi-stage training | **Optimal** | Predictor learns the smooth system first; Corrector learns the residual |

### Key Findings
- **Inductive Biases are Crucial**: Under equivalent parameter sizes, the Predictor structured with rigid-body dynamics yields significantly faster training speed and higher final accuracy compared to a pure ResNet.
- **Training on Random Data Generalizes Better**: In out-of-distribution (OOD) evaluations (tested on TD-MPC2 policy data), models trained on random data exhibit lower errors than those trained on experienced expert data.
- **Zero-shot RL is Feasible but Limited**: On Reacher (Easy/Hard) and Cartpole tasks, zero-shot RL achieves performance close to that in actual environments, but tasks requiring long horizons ($>500$ steps), such as Cheetah-Run, remain intractable.
- **Longer Prediction Horizons Correlate with Better RL Performance**: Increasing the prediction horizons from $10 \rightarrow 50 \rightarrow 100$ steps scales policy performance continuously.

## Highlights & Insights

- **Elegant Embedding of Rigid-body Dynamics Inductive Biases**: Requires no explicit physical parameters (e.g., mass, friction coefficient); it solely leverages the mathematical structure of the Newton-Euler equation to constrain the network. Utilizing the Cholesky decomposition to guarantee the positive-definiteness of the inertia matrix is an ingenious design.
- **Predictor-Corrector Separation Concept**: Separating the training of analytically modelable smooth parts and non-analytically modelable non-smooth parts is a powerful paradigm that can be extended to other physical simulation scenarios (e.g., fluids, soft bodies).
- **First Validation of Zero-Shot RL**: While currently successful only on simple tasks, it validates the core hypothesis that "a sufficiently robust world model can substitute the real environment," charting a clear course for future world model research.
- **Distribution Shift Detection Mechanism**: Employs a residual flow to estimate the training data distribution, adding a density penalty term into the RL reward to prevent the policy from exploring regions where the world model's accuracy deteriorates.

## Limitations & Future Work

- **Limited Upper Bound of Zero-shot Capabilities**: Tasks like Cheetah require 500-step prediction horizons, whereas MoSim currently struggles beyond 100 steps, meaning practical zero-shot RL remains some distance away.
- **Restricted to Rigid Bodies**: The current framework relies on rigid-body assumptions, remaining unable to handle more complex physical systems such as soft bodies or fluids.
- **State Space Limitations**: Requires complete physical state representations (joint positions and velocities) as inputs, precluding direct deployment from raw pixel-based observations.
- **Distribution Shift remains Unsolved**: The density penalty using residual flow merely mitigates rather than eliminates distribution shift; policies may still degenerate during late-stage training.
- **Lack of Sim-to-Real Validation**: All experiments are conducted in simulation. Sensor noise and unmodeled real-world system dynamics present significantly greater challenges.

## Related Work & Insights

- **vs. DreamerV3**: DreamerV3 conducts predictions in a latent space, which couples it with specific RL algorithms, and its prediction accuracy deteriorates rapidly over long horizons. In contrast, MoSim predicts directly in the raw state space, yielding one to two orders of magnitude higher accuracy, allowing seamless integration with any model-free RL algorithm.
- **vs. TD-MPC2**: TD-MPC2 restricts itself to 3-step predictions, primarily targeting MPC planning. MoSim outperforms TD-MPC2 even in TD-MPC2's own latent space.
- **vs. Differentiable Simulators**: Differentiable physical engines require manual formulation of force/contact models. MoSim adopts a data-driven learning format, which is more flexible albeit at the cost of interpretability.

## Rating
- Novelty: ⭐⭐⭐⭐ While combining rigid-body priors with Neural ODEs is not entirely new, this work is the first to systematically validate the feasibility of zero-shot RL using a world model.
- Experimental Thoroughness: ⭐⭐⭐⭐ 7 environments, various evaluation protocols, and extensive ablations, though lacking real-robot experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and rigorous mathematical derivations, though some notations appear slightly redundant.
- Value: ⭐⭐⭐⭐ Opens up a new paradigm for the direct evaluation of world models and zero-shot RL, though current zero-shot capabilities remain constrained.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Learning Interactive World Model for Object-Centric Reinforcement Learning](../../NeurIPS2025/robotics/learning_interactive_world_model_for_object-centric_reinforcement_learning.md)
- [\[ICLR 2026\] World-In-World: World Models in a Closed-Loop World](../../ICLR2026/robotics/world-in-world_world_models_in_a_closed-loop_world.md)
- [\[NeurIPS 2025\] Real-World Reinforcement Learning of Active Perception Behaviors](../../NeurIPS2025/robotics/real-world_reinforcement_learning_of_active_perception_behaviors.md)
- [\[ICCV 2025\] TesserAct: Learning 4D Embodied World Models](../../ICCV2025/robotics/learning_4d_embodied_world_models.md)
- [\[CVPR 2026\] Dexterous World Models](../../CVPR2026/robotics/dexterous_world_models.md)

</div>

<!-- RELATED:END -->
