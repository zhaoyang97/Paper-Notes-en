---
title: >-
  [Paper Note] Improving Planning and MBRL with Temporally-Extended Actions
description: >-
  [NeurIPS 2025][Reinforcement Learning][temporally-extended actions] This paper proposes treating action duration as an additional optimization variable in shooting-based planning and MBRL, combined with a multi-armed bandit (MAB) mechanism for automatic duration range selection. The approach significantly accelerates planning across multiple environments and solves challenging tasks that standard methods fail to handle.
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "temporally-extended actions"
  - "model-based RL"
  - "planning"
  - "action duration"
  - "multi-armed bandit"
date: 2026-05-08
content_hash: d381ba97e5e336e4
---

# Improving Planning and MBRL with Temporally-Extended Actions

**Conference**: NeurIPS 2025
**arXiv**: [2505.15754](https://arxiv.org/abs/2505.15754)  
**Code**: None  
**Area**: Reinforcement Learning
**Keywords**: temporally-extended actions, model-based RL, planning, action duration, multi-armed bandit

## TL;DR
This paper proposes treating action duration as an additional optimization variable in shooting-based planning and MBRL, combined with a multi-armed bandit (MAB) mechanism for automatic duration range selection. The approach significantly accelerates planning across multiple environments and solves challenging tasks that standard methods fail to handle.

## Background & Motivation

**Background**: Continuous systems are commonly approximated in discrete time, where a small simulation step $\delta_t$ requires a long planning horizon $D$, imposing substantial computational burden on CEM/MPPI.

**Limitations of Prior Work**: The optimal frame-skip value varies across environments. While model-free RL has explored learning frame-skip, this problem remains unaddressed in planning and MBRL. Long rollouts also exacerbate compounding errors.

**Key Challenge**: Small $\delta_t$ ensures accuracy but enlarges the search space; large frame-skip reduces search complexity but sacrifices flexibility.

**Goal**: Enable the planner to jointly optimize actions and duration $\delta t_k \in [\delta t_{\min}, \delta t_{\max}]$ at each step.

**Key Insight**: Treat $\delta t$ as a continuous optimization variable, with MAB automatically selecting $\delta t_{\max}$.

**Core Idea**: Action duration as a planner optimization variable, combined with a learned temporally-extended dynamics model and MAB-based automatic range selection.

## Method

### Overall Architecture

CEM outputs $(a_k, \delta t_k)$ at each decision step. The return is defined as $J_2 = \sum_k \gamma^{e_{<k}} \sum_t \gamma^{t-1} \mathcal{R}(s,a)$, where $e_k = \lfloor \delta t_k / \delta_t^{\text{env}} \rfloor$. Dual discount factors $\gamma_1, \gamma_2$ are supported.

### Key Designs

1. **Temporally-Extended Dynamics Model $\hat{F}_{\text{TE}}$**:

    - Takes $(s, a, \delta t)$ as input and outputs the next state distribution and reward.
    - Inference time is constant (versus the iterative model $F_{\text{IP}}$, which scales linearly with $\delta t$).
    - Shorter rollouts reduce compounding errors.

2. **MAB for Automatic $\delta t_{\max}$ Selection**:

    - $m = \log_2(T)$ exponentially spaced candidate values.
    - UCB with EMA: $\arg\max_i (\hat{R}_{i,T} + c\sqrt{2\log T / N(i,T)})$.
    - Each candidate maintains an independent dataset and model.

3. **Search Space Analysis**:

    - Standard: search space $|\mathcal{A}|^H$, optimizing $H|\mathcal{A}|$ variables.
    - TE ($m$-fold): search space $|\mathcal{A}|^{H/m}$, optimizing $(H/m)(|\mathcal{A}|+1)$ variables.

## Key Experimental Results

### Planning Experiments (Exact Dynamics)

| Environment | Standard | TE | Improvement |
|------|------|----|----|
| Mountain Car | Requires $D \geq 60$ | Solved with $D_{\text{TE}} \geq 4$ | 15× horizon reduction |
| Multi-hill MC (5 instances) | Solves 1/5 | Solves all | Resolves previously intractable tasks |
| Dubins Car (102-dim) | OOM at 4GB | 103MB, successful | 40× memory reduction |

### MBRL Experiments

| Environment | PETS | TE(F) | TE(D) | Max Gain |
|------|------|-------|-------|---------|
| Reacher | -6.5 | **-4.2** | -4.5 | +35% |
| HalfCheetah | ~4900 | **~6100** | ~5800 | +24% |
| Hopper | ~230 | **~680** | ~350 | +195% |
| Walker | ~420 | **~610** | ~540 | +45% |

### Ablation Study

| Configuration | Effect |
|------|------|
| Fixed $\gamma_1$, decreasing $\gamma_2$ | More decision steps, total steps unchanged |
| Fixed $\gamma_2$, decreasing $\gamma_1$ | Fewer total steps |
| Shared vs. independent models | Independent models perform better |
| TE(F) vs. TE(D) | MAB auto-selection approaches manual optimum |

### Key Findings
- Sparse-reward environments benefit most: shallow depth enables deep search.
- MBRL advantages: shorter rollouts reduce compounding errors and accelerate convergence.
- Hopper shows the largest improvement (+195%).
- MAB eliminates the need for manual hyperparameter tuning.

## Highlights & Insights
- **Simplicity and effectiveness**: Adding a single optimization dimension substantially extends the capability boundary of the planner.
- **MAB-based hyperparameter automation**: EMA + UCB addresses the non-stationary bandit problem.
- **Search space analysis**: The $2^H$ vs. $2^{H/m}$ comparison is intuitive and illuminating.
- The framework is transferable to any system involving a trade-off between temporal resolution and search depth.

## Limitations & Future Work
- Predicting states reached after $e_k$ steps from $s$ is inherently more difficult than single-step prediction, posing accuracy challenges for $\hat{F}_{\text{TE}}$ under long action durations.
- Maintaining independent models for each $\delta t_{\max}$ candidate increases memory and training time overhead.
- Validation is limited to shooting-based planners (CEM); gradient-based methods and tree search (MCTS) have not been explored.
- Theoretical analysis is absent — under what conditions do temporally-extended actions guarantee no loss of optimality, and how does error propagation relate to $\delta t$?
- The dual discount factors $\gamma_1, \gamma_2$ introduce additional degrees of freedom, though setting them equal suffices as a default.

### Comparison of Two Variants

| Property | $A_{\text{TE}}$(F) Fixed Range | $A_{\text{TE}}$(D) Dynamic Selection |
|------|--------------------------|----------------------------|
| $\delta t_{\max}$ | Manually specified | Automatically selected by MAB |
| Number of models | 1 | $m = \log_2(T)$ |
| Dataset | Shared | Independent per candidate |
| Tuning requirement | Requires tuning $\delta t_{\max}$ | None (MAB automated) |
| Performance | Slightly better when hand-tuned optimally | Approaches hand-tuned optimum; more robust |

## Related Work & Insights
- **vs. PETS (Chua et al. 2018)**: TE + MAB is integrated into PETS with minimal architectural changes.
- **vs. Ni & Jang (2022)**: Model-free timescale learning; this work directly optimizes duration in MBRL.
- **vs. Options (Sutton et al. 1999)**: Options require learning initiation and termination conditions; the proposed approach is substantially more lightweight.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of action duration optimization in MBRL and planning.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two major settings, 7+ environments, complete ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, consistent notation.
- Value: ⭐⭐⭐⭐ Highly practical; directly integrable into existing MBRL frameworks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Parameter-Free Algorithms for the Stochastically Extended Adversarial Model](parameter-free_algorithms_for_the_stochastically_extended_adversarial_model.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[NeurIPS 2025\] FedRAIN-Lite: Federated Reinforcement Algorithms for Improving Idealised Numerical Weather and Climate Models](fedrain-lite_federated_reinforcement_algorithms_for_improving_idealised_numerica.md)
- [\[NeurIPS 2025\] Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents](deep_rl_needs_deep_behavior_analysis_exploring_implicit_planning_by_model-free_a.md)
- [\[ACL 2026\] Controlling Multimodal Conversational Agents with Coverage-Enhanced Latent Actions](../../ACL2026/reinforcement_learning/controlling_multimodal_conversational_agents_with_coverage-enhanced_latent_actio.md)

</div>

<!-- RELATED:END -->
