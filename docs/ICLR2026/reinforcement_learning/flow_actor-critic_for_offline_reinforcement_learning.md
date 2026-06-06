---
title: >-
  [Paper Note] Flow Actor-Critic for Offline Reinforcement Learning (FAC)
description: >-
  [ICLR 2026][Reinforcement Learning][Offline RL] FAC is the first method to jointly leverage a continuous normalizing flow model to simultaneously construct an expressive actor policy and a critic penalty mechanism based…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Offline RL"
  - "Flow Matching"
  - "Actor-Critic"
  - "OOD Detection"
  - "Continuous Normalizing Flows"
date: 2026-05-08
content_hash: 5ac881ff7c08b009
---

# Flow Actor-Critic for Offline Reinforcement Learning (FAC)

**Conference**: ICLR 2026  
**arXiv**: [2602.18015](https://arxiv.org/abs/2602.18015)  
**Code**: None  
**Area**: Reinforcement Learning  
**Keywords**: Offline RL, Flow Matching, Actor-Critic, OOD Detection, Continuous Normalizing Flows  

## TL;DR
FAC is the first method to jointly leverage a continuous normalizing flow model to simultaneously construct an expressive actor policy and a critic penalty mechanism based on exact density estimation. By identifying OOD regions for selective conservative Q-value estimation, FAC achieves an average score of 60.3 across 55 OGBench tasks, substantially outperforming the previous best of 43.6.

## Background & Motivation

**Background**: Offline RL datasets typically exhibit complex multimodal behavioral distributions. Simple Gaussian policies lack sufficient expressiveness; while diffusion-based policies are highly expressive, their multi-step sampling renders policy optimization unstable.

**Limitations of Prior Work**: (a) CQL applies a global conservative penalty uniformly to all OOD actions, leading to excessive conservatism; (b) SVR identifies OOD actions via importance sampling ratios, but these ratios can explode when the behavioral policy is poorly approximated by a Gaussian model; (c) existing methods treat actor design and critic penalty independently, lacking co-design between the two components.

**Key Challenge**: How can strong policy expressiveness be maintained while precisely identifying OOD regions for conservative estimation?

**Key Insight**: Flow models offer both a highly expressive policy (actor) and exact density estimation (critic OOD penalty), addressing both challenges simultaneously.

**Core Idea**: A single flow model resolves both actor expressiveness and critic OOD detection.

## Method

### Overall Architecture
Two-stage training: (1) a behavioral proxy model $\hat{\beta}_\psi$ is trained via flow matching to provide exact density estimation; (2) the density estimates are used to construct a weight function that penalizes Q-values in OOD regions, while a one-step flow actor optimizes the policy.

### Key Designs

1. **Flow Behavior Proxy**:

    - Function: A continuous normalizing flow model is trained via flow matching to model the behavioral distribution $\hat{\beta}_\psi(a|s)$.
    - Mechanism: The flow matching objective is $\min_\psi \mathbb{E}[\|v_\psi(\tilde{a}_u; s, u) - (a-z)\|^2]$, where $\tilde{a}_u = (1-u)z + ua$. A key advantage is that the flow model provides **exact density estimates** $\log\hat{\beta}_\psi(a|s)$ via ODE integration, as opposed to the ELBO lower bound of VAEs or approximate likelihoods of diffusion models.
    - Design Motivation: Exact density estimation is a prerequisite for reliable OOD detection — approximate densities from VAEs or diffusion models lead to misclassification.

2. **Flow Critic Penalty**:

    - Function: Density estimates are used to construct a weight function that applies penalties exclusively to Q-values in OOD regions.
    - Mechanism: The weight function $w^{\hat{\beta}}(s,a) = \max(0, 1 - \hat{\beta}(a|s)/\epsilon)$ is zero within the data support ($\hat{\beta} \geq \epsilon$) and increases linearly in OOD regions. The critic loss incorporates an additional term $\alpha \cdot \mathbb{E}_{a \sim \pi}[w \cdot Q(s,a)]$.
    - Proposition 1 guarantees that the Bellman operator remains unbiased within the data distribution while strongly suppressing Q-values in OOD regions.

3. **One-Step Flow Actor**:

    - Function: A simplified one-step flow (direct mapping $z \mapsto a$) serves as the policy, avoiding the instability of multi-step sampling.
    - Mechanism: The actor loss is $\max \mathbb{E}[Q(s,a)] - \lambda \cdot \|a_\theta(s,z) - a_\psi(s,z)\|^2$ (Q-value maximization + behavioral regularization). Unlike multi-step diffusion or flow policies, the one-step mapping stabilizes gradient computation.

### Loss & Training
- Stage 1: Flow matching pre-training of the behavioral proxy.
- Stage 2: Alternating updates of the critic (TD loss + flow density-weighted penalty) and the actor (Q-value maximization + behavioral regularization).

## Key Experimental Results

### Main Results
OGBench (55 tasks, the most challenging offline RL benchmark):

| Method | State-based Avg. Score ↑ | Category |
|--------|--------------------------|----------|
| ReBRAC (Gaussian) | 31.0 | Gaussian Policy |
| FQL (Flow) | 43.6 | Flow Policy (actor only) |
| **FAC** | **60.3** | Flow Policy (actor+critic) |

Highlights: puzzle-3x3-play 100.0 (vs. FQL 29.6, +238%); antmaze-large 92.6 (vs. FQL 78.6).

D4RL AntMaze (6 tasks): average **90.5** (new SOTA, previous best FQL 83.5).

### Ablation Study

| Configuration | OGBench Avg. ↑ | Notes |
|---------------|----------------|-------|
| FAC (full) | 60.3 | Actor regularization + critic penalty |
| Actor regularization only (= FQL) | 43.6 | No critic penalty |
| Critic penalty only | ~48 | No actor regularization |
| VAE density replacing flow density | Large drop | Inaccurate density estimation causes OOD detection failure |

### Key Findings
- Flow-based density estimation substantially outperforms VAE/diffusion models for OOD detection (illustrated in synthetic experiments, Fig. 1).
- Both actor regularization and critic penalty are indispensable; the joint FAC approach improves over FQL (actor only) by +16.7.
- The one-step flow actor is more stable than multi-step flow policies (FAWAC, FBRAC).
- On D4RL MuJoCo, Gaussian methods remain competitive after extensive tuning, but the performance gap is substantial on the complex tasks of OGBench.

## Highlights & Insights
- The **one model, two uses** design is highly elegant: the flow model simultaneously provides an expressive policy and accurate OOD density estimation, which is more efficient and self-consistent than designing the two components separately.
- The **density-threshold penalty** (vs. CQL's global penalty) represents an important advancement — conservatism is applied only in genuinely OOD regions, maintaining unbiasedness within the data distribution and avoiding CQL's over-conservatism.
- The substantial improvement of 60.3 vs. 43.6 on OGBench suggests that, in complex multimodal tasks, precise OOD handling is more critical than policy expressiveness alone.

## Limitations & Future Work
- Two-stage training (pre-training the flow model before actor-critic training) increases overall training complexity.
- Density evaluation requires numerical ODE integration (10-step Euler), introducing additional inference overhead.
- The threshold $\epsilon$ introduces an additional design choice.
- Performance gains over baselines are less pronounced on D4RL MuJoCo than on OGBench.

## Related Work & Insights
- **vs. FQL**: FQL uses flow only for the actor; FAC additionally leverages it for critic penalty — achieving +38% improvement on OGBench.
- **vs. CQL**: CQL's uniform penalty over all OOD actions causes excessive conservatism; FAC uses exact density estimates to apply penalties only at genuinely OOD actions.
- **vs. DiffQL/IDQL**: Multi-step sampling in diffusion policies destabilizes policy optimization; FAC's one-step flow is more stable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to jointly leverage a flow model for both actor and critic; conceptually clean yet highly effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ OGBench (55 tasks) + D4RL (15 tasks) + pixel observations; comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Method motivation is clear; theoretical guarantee (Proposition 1) is concise.
- Value: ⭐⭐⭐⭐⭐ Achieves a major breakthrough on the most challenging offline RL benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Emergence of Spatial Representation in an Actor-Critic Agent with Hippocampus-Inspired Sequence Generator](emergence_of_spatial_representation_in_an_actor-critic_agent_with_hippocampus-in.md)
- [\[AAAI 2026\] Risk-Sensitive Exponential Actor Critic](../../AAAI2026/reinforcement_learning/risk-sensitive_exponential_actor_critic.md)
- [\[ICLR 2026\] ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation](reform_reflected_flows_for_on-support_offline_rl_via_noise_manipulation.md)
- [\[ICLR 2026\] Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](cross-embodiment_offline_reinforcement_learning_for_heterogeneous_robot_datasets.md)
- [\[AAAI 2026\] Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](../../AAAI2026/reinforcement_learning/actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)

</div>

<!-- RELATED:END -->
