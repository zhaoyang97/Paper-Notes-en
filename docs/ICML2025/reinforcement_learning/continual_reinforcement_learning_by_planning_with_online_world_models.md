---
title: >-
  [Paper Note] Continual Reinforcement Learning by Planning with Online World Models
description: >-
  [ICML 2025 (Spotlight)][Reinforcement Learning][Continual Reinforcement Learning] This paper proposes the FTL Online Agent (OA), which achieves continual reinforcement learning through an online-learned Follow-The-Leader shallow world model combined with Model Predictive Control (MPC) planning. This world model is immune to catastrophic forgetting by construction, provides a theoretical regret bound guarantee of $\mathcal{O}(\sqrt{K^2 D \log(T)})$…
tags:
  - "ICML 2025 (Spotlight)"
  - "Reinforcement Learning"
  - "Continual Reinforcement Learning"
  - "Online World Models"
  - "Catastrophic Forgetting"
  - "Follow-The-Leader"
  - "Model Predictive Control"
date: 2026-05-08
content_hash: 7e20f3c9efcb414a
---

# Continual Reinforcement Learning by Planning with Online World Models

**Conference**: ICML 2025 (Spotlight)  
**arXiv**: [2507.09177](https://arxiv.org/abs/2507.09177)  
**Code**: None  
**Area**: Reinforcement Learning / Continual Learning  
**Keywords**: Continual Reinforcement Learning, Online World Models, Catastrophic Forgetting, Follow-The-Leader, Model Predictive Control

## TL;DR

This paper proposes the FTL Online Agent (OA), which achieves continual reinforcement learning through an online-learned Follow-The-Leader shallow world model combined with Model Predictive Control (MPC) planning. This world model is immune to catastrophic forgetting by construction, provides a theoretical regret bound guarantee of $\mathcal{O}(\sqrt{K^2 D \log(T)})$, and comprehensively outperforms deep world model-based methods on a specially designed Continual Bench.

## Background & Motivation

**Background**: Continual Reinforcement Learning (CRL) is a natural and important setting where an agent needs to continuously evolve and learn across sequentially presented tasks. This is closely related to human lifelong learning capabilities and is an essential capability for actual robot deployment.

**Limitations of Prior Work**: The primary obstacle facing CRL is **catastrophic forgetting**, where the agent forgets the solutions to previously learned tasks when acquiring new ones. Current mainstream approaches either employ deep neural networks as world models (such as DreamerV3) coupled with various continual learning techniques (such as EWC and experience replay) to mitigate forgetting, or utilize separate models/heads to isolate different tasks. These methods either deliver limited performance or scale poorly.

**Key Challenge**: The parameterization of deep world models is inherently the root of forgetting; when updating parameters via gradient descent to fit new tasks, information from older tasks is inevitably overwritten. Patch-like continual learning techniques only mitigate rather than fundamentally solve this issue. This raises the question: can a world model be designed that is **by construction** immune to forgetting?

**Goal**: To design a world model for continual RL that is naturally immune to catastrophic forgetting. The core requirements are: (1) learned world dynamics are never forgotten; (2) effective planning can be performed over this model to solve tasks specified by arbitrary reward functions; and (3) updates are incremental and computationally efficient.

**Key Insight**: Utilizing the Follow-The-Leader (FTL) online learning algorithm to maintain shallow (non-deep) world models. FTL updates fit all historical data globally (rather than local updates via gradient descent), mathematically guaranteeing that old data is not forgotten. Model Predictive Control (MPC) is then used to plan and execute actions on top of this model.

**Core Idea**: Construct a shallow world model using the FTL online learning algorithm to naturally render it immune to forgetting (as each update fits all historical data). This is combined with an MPC planner to solve sequential tasks in CRL, forming the FTL Online Agent (OA).

## Method

### Overall Architecture

```
Sequence of Tasks: Task 1 → Task 2 → ... → Task K
                    ↓
      [FTL Online World Model] —— Incremental updates, fitting all historical data
                    ↓
      [MPC Planner] —— Given the task reward function, searching for the optimal action sequence
                    ↓
                Environment Execution
```

- The world model solely learns the **environment dynamics** (state transitions), decoupled from task rewards.
- When switching tasks, only the reward function of the MPC needs to be changed, while the world model remains unaltered.
- The world model is updated incrementally without requiring retraining.

### Key Designs

1. **Follow-The-Leader (FTL) Shallow World Model**:

    - **Function**: Incrementally maintains a shallow model (such as a linear model or kernel method) to predict environment state transitions $s_{t+1} = f(s_t, a_t)$.
    - **Mechanism**: FTL is a classic algorithm in online learning. At each timestep $t$, the model parameters are selected to achieve the optimal fit over all historical data:
  
    $$\theta_t = \arg\min_\theta \sum_{i=1}^{t} \ell(\theta; s_i, a_i, s_{i+1})$$
   
    That is, each update performs a global optimization over **all historical data**, rather than a local update on new data as in gradient descent.
    - **Design Motivation**: This serves as the mathematical guarantee for being immune to forgetting. Because the update objective always encompasses all historical data, information from past data is never discarded. In contrast, gradient descent updates in deep models are essentially local adjustments on new data, which inevitably overwrites older information.

2. **Efficient Implementation of Incremental Updates**:

    - **Function**: Although the definition of FTL involves "refitting all history," incremental updates can be achieved by choosing an appropriate model class (such as linear models).
    - **Mechanism**: For a linear model $\theta_t = A_t^{-1} b_t$, only the incremental updates of the statistics $A_t$ and $b_t$ need to be maintained:
  
    $$A_t = A_{t-1} + \phi(s_t, a_t)\phi(s_t, a_t)^\top, \quad b_t = b_{t-1} + \phi(s_t, a_t) s_{t+1}$$
   
    where $\phi$ is a feature mapping. Each update requires only $O(D^2)$ computation, avoiding the need to re-traverse historical data.
    - **Design Motivation**: This makes the algorithm practically feasible. While theoretically equivalent to refitting all history, it only requires $O(1)$ incremental computation in practice.

3. **Model Predictive Control (MPC) Planner**:

    - **Function**: Given the reward function $r(\cdot)$ of the task, it searches for an action sequence that maximizes the expected cumulative reward using the FTL world model.
    - **Mechanism**: At each decision timestep, the world model is used to roll out multiple simulated trajectories to evaluate the cumulative rewards of different action sequences. The optimal first-step action is then executed:
  
    $$a_t^* = \arg\max_{a_{t:t+H}} \sum_{h=0}^{H} r(\hat{s}_{t+h}, a_{t+h})$$
   
    where $\hat{s}_{t+h}$ is the predicted state from the world model, and $H$ is the planning horizon.
    - **Design Motivation**: The advantage of MPC is that planning is entirely based on the latest world model, eliminating the need to train a separate policy network. When switching tasks, only the reward function $r$ needs to be changed, and the planner automatically adapts—achieving true "zero-forgetting" task switching.

4. **Theoretical Regret Bound**:

    - **Function**: Proves that the OA exhibits a sublinear regret bound under mild assumptions.
    - **Mechanism**: Under the setting with $K$ tasks, $T$ total timesteps, and a feature dimension $D$:
  
    $$\text{Regret}(T) = \mathcal{O}(\sqrt{K^2 D \log(T)})$$
   
    - **Design Motivation**: This theoretically guarantees the continual learning performance of OA—showing that the regret grows sublinearly over time, implying that the average performance loss per step approaches zero.

### Loss & Training

- The FTL world model fits state transitions using the least-squares loss.
- The MPC planner searches for actions using the Cross-Entropy Method (CEM) or Random Shooting.
- The entire system does not require gradient descent to train a policy network—the world model is updated incrementally, and the planner searches online.

## Key Experimental Results

### Main Results (Continual Bench)

This paper designs Continual Bench, an environment suite specifically tailored for evaluating CRL.

| Method | Forward Transfer | Forgetting Rate | Overall CRL Score | Description |
|------|---------|-------|--------------|------|
| DreamerV3 (naive) | Medium | High | Lower | Deep world model, no anti-forgetting measures |
| DreamerV3 + EWC | Medium | Medium | Medium | With Elastic Weight Consolidation |
| DreamerV3 + PackNet | Medium | Low | Medium | Parameter isolation |
| DreamerV3 + Experience Replay | Medium | Medium-Low | Medium | Old data replay |
| **OA (Ours)** | **High** | **Zero** | **Highest** | FTL shallow model + MPC |

### Ablation Study

| Configuration | Forgetting Rate | New Task Learning Efficiency | Description |
|------|-------|--------------|------|
| OA (Full) | Zero Forgetting | Efficient | Full FTL + MPC scheme |
| Replace with Deep Model + FTL | Non-zero Forgetting | Efficient | Deep model has approximation error even with FTL theory |
| Replace with Gradient Descent Updates | Significant Forgetting | Efficient | Proves that the FTL update method is crucial |
| Fixed World Model (No Updates) | Zero Forgetting | Poor | Unable to learn new dynamics |
| Varying MPC Horizon $H$ | Unchanged | Varies with $H$ | Longer horizon yields better performance on complex tasks |

### Key Findings

1. **OA achieves true zero-forgetting**: Across all experiments, OA experiences zero performance decline on older tasks after switching to new ones—a feat that all deep world model-based methods (even with EWC, PackNet, etc.) fail to achieve.
2. **Competitiveness of shallow models**: Despite using a shallow model, which has less expressive power than deep models like DreamerV3, OA achieves a higher overall CRL score due to its zero-forgetting advantage.
3. **Efficiency of incremental updates**: The per-step update time of OA is far lower than the training time of deep models, making it computationally much more efficient.
4. **FTL updates vs. gradient descent**: Even within the same shallow model architecture, the FTL update method significantly outperforms gradient descent updates; the former incurs zero forgetting while the latter displays substantial forgetting.

## Highlights & Insights

- **Resolving forgetting by construction**: Rather than mitigating forgetting via "patchwork" techniques (e.g., EWC, experience replay), this work fundamentally designs a learning algorithm that is mathematically immune to forgetting—representing a paradigm shift.
- **Decoupling of model and policy**: The world model exclusively learns the environment dynamics (independent of tasks), while the policy is generated online via MPC (decoupled from the world model). Task switching requires only changing the reward function, making it elegant and efficient.
- **Unification of theory and practice**: It provides rigorous theoretical guarantees for regret bounds while demonstrating practical superiority in actual CRL environments, which is rare in the CRL field.
- **ICML Spotlight**: Being selected as a Spotlight paper underscores the community's recognition of the direction of "fundamentally resolving forgetting."

## Limitations & Future Work

1. **Representation power limitations**: Shallow (linear) world models have restricted expressive capabilities, which may be insufficient in scenarios with highly non-linear dynamics. Enhancing model capacity while preserving the zero-forgetting property remains a critical challenge.
2. **Dependency on feature engineering**: Shallow models rely on high-quality feature mappings $\phi$, which are difficult to acquire in complex, high-dimensional environments.
3. **MPC computational overhead**: Performing rollouts and searching over multiple trajectories on the world model at each inference step may limit real-time capability.
4. **Complexity of the Continual Bench**: Although the designed CRL benchmark is tailored and reasonable, its environmental complexity still lags behind standard RL benchmarks such as Atari.
5. **Comparisons with more CRL methods**: The lack of comparison with methods like Progressive Neural Networks and CLEAR makes the evaluation less comprehensive.

## Related Work & Insights

- **DreamerV3**: A representative deep world model, serving as GLIDER's main baseline.
- **EWC / PackNet / Experience Replay**: Classic continual learning techniques; this paper compares against them to demonstrate the limitations of "patchwork" strategies.
- **Follow-The-Leader / Online Learning**: Classic online learning theory, introduced here to CRL.
- **Model Predictive Control**: A classic control methodology combined with online world models to enable "zero-forgetting" decision-making.
- **Insights**: In CRL, "not forgetting" may be more important than "learning fast"—a simple model that never forgets can outperform a complex, highly expressive model that is prone to forgetting.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Resolving forgetting by construction is a paradigm innovation, and the combination of FTL + MPC is entirely fresh in CRL.
- Experimental Thoroughness: ⭐⭐⭐⭐ Self-constructed Continual Bench with comparisons against multiple baselines, though environmental complexity could be further scaled.
- Writing Quality: ⭐⭐⭐⭐ Balanced in both theory and experiments, reflecting the high standard of a Spotlight paper.
- Value: ⭐⭐⭐⭐⭐ Proposes a fundamentally new direction for CRL, carrying long-term impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Reward-free World Models for Online Imitation Learning](reward-free_world_models_for_online_imitation_learning.md)
- [\[NeurIPS 2025\] Continual Knowledge Adaptation for Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/continual_knowledge_adaptation_for_reinforcement_learning.md)
- [\[ICML 2025\] Online Pre-Training for Offline-to-Online Reinforcement Learning](online_pre-training_for_offline-to-online_reinforcement_learning.md)
- [\[ICML 2025\] Position: Lifetime Tuning is Incompatible with Continual Reinforcement Learning](position_lifetime_tuning_is_incompatible_with_continual_reinforcement_learning.md)
- [\[NeurIPS 2025\] Temporal-Difference Variational Continual Learning](../../NeurIPS2025/reinforcement_learning/temporal-difference_variational_continual_learning.md)

</div>

<!-- RELATED:END -->
