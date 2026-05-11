---
title: >-
  [Paper Note] Offline Reinforcement Learning with Generative Trajectory Policies
description: >-
  [ICLR 2026][Reinforcement Learning][Generative Trajectory Policy] This paper proposes the Generative Trajectory Policy (GTP), which unifies diffusion models, flow matching…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "Generative Trajectory Policy"
  - "ODE Solution Mapping"
  - "Consistency Models"
  - "Flow Matching"
  - "D4RL"
date: 2026-05-08
content_hash: 22bbe69e72c983d1
---

# Offline Reinforcement Learning with Generative Trajectory Policies

**Conference**: ICLR 2026
**arXiv**: [2510.11499](https://arxiv.org/abs/2510.11499)
**Code**: Included with the paper
**Area**: Offline Reinforcement Learning / Generative Policies
**Keywords**: Generative Trajectory Policy, ODE Solution Mapping, Consistency Models, Flow Matching, D4RL

## TL;DR

This paper proposes the Generative Trajectory Policy (GTP), which unifies diffusion models, flow matching, and consistency models by learning the complete solution mapping of an ODE. Combined with two key adaptation techniques—score approximation and value-guided weighting—GTP achieves state-of-the-art performance on D4RL.

## Background & Motivation

- The central tension in generative policies for offline RL: **expressiveness vs. efficiency**
    - Diffusion policies: highly expressive but slow to sample (requiring hundreds of iterative steps)
    - Consistency policies: fast single-step generation but degraded performance
- **Key Insight**: Diffusion models, flow matching, consistency models, and related approaches can be unified as learning the solution mapping of a continuous-time ODE.

## Method

### Unified ODE Framework

The underlying ODE shared by modern generative models: $\frac{d\boldsymbol{x}_t}{dt} = f(\boldsymbol{x}_t, t)$

Flow map: $\Phi(\boldsymbol{x}_t, t, s) = \boldsymbol{x}_t + \int_t^s f(\boldsymbol{x}_\tau, \tau) d\tau$

Reparameterized function: $\phi(\boldsymbol{x}_t, t, s) = \boldsymbol{x}_t + \frac{t}{t-s}\int_t^s f(\boldsymbol{x}_\tau, \tau) d\tau$

Two fundamental training objectives:

1. **Instantaneous flow loss** (local anchoring): $\lim_{s \to t} \phi(\boldsymbol{x}_t, t, s) = \boldsymbol{x}_t - tf(\boldsymbol{x}_t, t)$
2. **Trajectory consistency loss** (global coherence): $\Phi(\boldsymbol{x}_t, t, s) \approx \Phi(\Phi(\boldsymbol{x}_t, t, u), u, s)$

### Key Design 1: Score Approximation

**Problem**: Self-referential supervision leads to expensive computation and unstable training (analogous to the vicious cycle in TD learning).

**Solution**: Replace the learned score function with a closed-form surrogate $\tilde{f}(\boldsymbol{x}_t, t) = (\boldsymbol{x}_t - \boldsymbol{x})/t$.

**Theorem 1**: The discrepancy between the practical training loss and the ideal loss is $O(h^p)$ (where $h$ is the step size and $p$ is the solver order), and intermediate points can be obtained in a single step: $\boldsymbol{x}_u = \boldsymbol{x} + u \cdot \boldsymbol{z}$.

### Key Design 2: Value-Guided Weighting

**Theorem 2**: The optimal solution to KL-regularized policy optimization satisfies $\pi^*(a|s) \propto \pi_{BC}(a|s) \exp(\eta A(s,a))$.

Weighted generative training objective:

$$\max_\theta \mathbb{E}_{(s,a) \sim \mathcal{D}} [\exp(\eta A(s,a)) \cdot \ell_{\text{gen}}(\pi_\theta; a|s)]$$

In practice, a clipped and normalized advantage weight is used: $w(s,a) = \exp\left(\eta \cdot \frac{\max(0, A(s,a))}{\text{std}(A) + \epsilon}\right)$

### GTP Actor-Critic Training

- **Actor**: $\mathcal{L}_{\text{actor}} = \mathcal{L}_{\text{Consistency}} + \lambda_{\text{Flow}} \cdot \mathcal{L}_{\text{Flow}}$
- **Critic**: Twin Q-networks with EMA target networks
- **Inference**: Starting from Gaussian noise, apply $K$ iterative steps via $a_{t_{i+1}} = \Phi_\theta(s, a_{t_i}, t_i, t_{i+1})$

## Key Experimental Results

### D4RL Behavior Cloning Performance

| Task | BC | D-BC | C-BC | GTP-BC |
|------|------|------|------|--------|
| halfcheetah-m | 42.6 | 45.4 | 31.0 | **48.6** |
| hopper-m | 52.9 | 65.3 | 71.7 | **83.7** |
| hopper-mr | 18.1 | 67.3 | 99.7 | **100.5** |
| **Gym Average** | - | 76.3 | 69.7 | **82.3** |
| **AntMaze Average** | - | 28.3 | 44.1 | **66.3** |

### D4RL Offline RL Performance (Selected Key Tasks)

| Task | IDQL | Diff-QL | CPQL | GTP |
|------|------|---------|------|-----|
| antmaze-large-play | 47.5 | 46.4 | 49.4 | **100.0** |
| antmaze-large-diverse | 45.9 | 36.0 | 42.0 | **100.0** |
| antmaze-ultra-play | 30.3 | 4.8 | 36.9 | **38.9** |

### Key Findings

1. GTP-BC under pure behavior cloning already substantially outperforms diffusion and consistency policies (AntMaze average: 66.3 vs. 44.1).
2. The full GTP achieves perfect scores (100.0) on multiple AntMaze tasks, far surpassing prior methods.
3. High performance is attained with only 5 sampling steps, resolving the efficiency–expressiveness trade-off.
4. Score approximation eliminates the need for multi-step ODE integration, enabling stable and efficient training.

## Highlights & Insights

1. **Unifying perspective**: Diffusion models, flow matching, and consistency models are unified under the framework of ODE solution mapping learning.
2. **Solid theoretical foundation**: Two theorems provide rigorous guarantees for score approximation and value-guided weighting, respectively.
3. **Engineering practicality**: Score approximation reduces training from multi-step ODE solving to single-step perturbation.
4. **Significant performance gains**: Breakthrough results on the most challenging AntMaze tasks.

## Limitations & Future Work

- Joint training with a critic is required, increasing overall training complexity.
- The $O(h^p)$ bound from score approximation may not be tight under extreme step sizes.
- Validation is limited to the standard D4RL benchmark; real-robot environments have not been tested.
- Performance may be sensitive to hyperparameter choices ($\lambda_{\text{Flow}}$, $\eta$, number of sampling steps $K$).

## Related Work & Insights

- **Generative policies**: Diffuser, Diffusion-QL, CPQL
- **Consistency models**: Consistency Models, CTM
- **Offline RL**: IQL, TD3+BC, AWAC

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The unified framework and GTP paradigm are highly original.
- Technical Depth: ⭐⭐⭐⭐⭐ — Rigorous theoretical derivations with complete theorem proofs.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive benchmarking and ablation analysis.
- Value: ⭐⭐⭐⭐ — Perfect scores on AntMaze tasks are impressive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Flow Actor-Critic for Offline Reinforcement Learning (FAC)](flow_actor-critic_for_offline_reinforcement_learning.md)
- [\[AAAI 2026\] One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow](../../AAAI2026/reinforcement_learning/one-step_generative_policies_with_q-learning_a_reformulation_of_meanflow.md)
- [\[AAAI 2026\] Know your Trajectory -- Trustworthy Reinforcement Learning Deployment through Importance-Based Trajectory Analysis](../../AAAI2026/reinforcement_learning/know_your_trajectory_--_trustworthy_reinforcement_learning_deployment_through_im.md)
- [\[ICLR 2026\] Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](cross-embodiment_offline_reinforcement_learning_for_heterogeneous_robot_datasets.md)
- [\[ICLR 2026\] ReFORM: Reflected Flows for On-support Offline RL via Noise Manipulation](reform_reflected_flows_for_on-support_offline_rl_via_noise_manipulation.md)

</div>

<!-- RELATED:END -->
