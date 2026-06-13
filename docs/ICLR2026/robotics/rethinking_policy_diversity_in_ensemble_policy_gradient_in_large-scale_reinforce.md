---
title: >-
  [Paper Note] Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning
description: >-
  [ICLR 2026][Robotics][Policy Ensemble] This paper theoretically analyzes how inter-policy diversity affects learning efficiency in ensemble policy gradient methods, and proposes Coupled Policy Optimization (CPO)…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "Policy Ensemble"
  - "Large-Scale Parallel RL"
  - "Policy Diversity"
  - "KL Constraint"
  - "Dexterous Manipulation"
date: 2026-05-08
content_hash: 7115446b2b43581b
---

# Rethinking Policy Diversity in Ensemble Policy Gradient in Large-Scale Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2603.01741](https://arxiv.org/abs/2603.01741)  
**Code**: [Project Page](https://naoki04.github.io/paper-cpo/)  
**Area**: Reinforcement Learning
**Keywords**: Policy Ensemble, Large-Scale Parallel RL, Policy Diversity, KL Constraint, Dexterous Manipulation

## TL;DR

This paper theoretically analyzes how inter-policy diversity affects learning efficiency in ensemble policy gradient methods, and proposes Coupled Policy Optimization (CPO), which regulates diversity via KL divergence constraints to achieve efficient and stable exploration in large-scale parallel environments.

## Background & Motivation

GPU-based parallel physics simulators (e.g., Isaac Gym, Genesis) enable data collection across tens of thousands of environments simultaneously. However, **simply increasing the number of parallel environments does not improve learning efficiency** (Singla et al., 2024) — a single policy generates highly similar trajectories across many parallel environments, leading to insufficient exploration diversity.

To address this, SAPG introduced a leader-follower framework: one leader policy and multiple follower policies collect data in separate environment blocks, and the leader aggregates all follower data via importance sampling (IS). Nevertheless, **excessive policy diversity is detrimental**: when follower policies deviate too far from the leader, IS ratios diverge from 1, causing the effective sample size (ESS) to drop and PPO clipping bias to increase, undermining training stability and sample efficiency.

The root cause is a fundamental tension: exploration diversity requires policy divergence, yet excessive divergence reduces the utilization efficiency of off-policy data. This paper proposes a method to regulate this "moderate diversity."

## Method

### Overall Architecture

CPO builds upon SAPG's leader-follower architecture and introduces two key mechanisms: (1) a KL divergence constraint on follower updates relative to the leader, maintaining a moderate distance; and (2) adversarial rewards that encourage dispersion among followers to prevent policy collapse.

### Key Designs

1. **KL-Constrained Follower Policy Update**:

    - Function: Models follower policy updates as a constrained optimization problem with a KL divergence bound.
    - Mechanism: $\pi_{F_i}^* = \arg\max_{\pi_{F_i}} A_{F_i}(\mathbf{s},\mathbf{a}) \quad \text{s.t.} \quad D_{KL}(\pi_{F_i}(\cdot|\mathbf{s}) \| \pi_L(\cdot|\mathbf{s})) \leq \varepsilon_{KL}$
    - An AWAC-style closed-form approximation yields the parameterized objective: $L_{\text{CPO},F_i}(\theta) = -\mathbb{E}_{\mathbf{a},\mathbf{s} \sim \pi_L}[\log \pi_{F_i,\theta}(\mathbf{a}|\mathbf{s}) \exp(\frac{1}{\lambda_f} A^{F_i})] + L_{\text{SAPG},F_i}$
    - Design Motivation: By Pinsker's inequality, the KL constraint directly controls the upper bound on IS ratio deviation $\mathbb{E}[|1-\frac{\pi_L}{\pi_F}|] \leq \sqrt{2D_{KL}}$, thereby guaranteeing ESS and training stability.

2. **Adversarial Reward for Policy Dispersion**:

    - Function: Trains a discriminator $D_\xi(y|\mathbf{s}_t,\mathbf{a}_t)$ to predict policy identity.
    - Mechanism: $r_t^{adv} = \lambda_{adv} \log D_\xi(y|\mathbf{s}_t,\mathbf{a}_t)$, rewarding each follower for exploring distinctly identifiable regions.
    - Design Motivation: The KL constraint implicitly pulls followers closer together; the adversarial reward provides a counterforce to prevent policy collapse.

### Loss & Training

The overall objective is: $L_{\text{CPO}}(\theta) = L_{\text{SAPG}}(\theta,j) + \beta \sum_{i} L_{\text{CPO},F_i,f}(\theta,\lambda_f)$, where $\beta$ balances the scale between the PPO objective and the KL regularization term. Adversarial rewards are provided only to followers; the leader is updated solely using true environment rewards.

## Key Experimental Results

### Main Results (Dexterous Manipulation Tasks, after $2\times10^{10}$ environment steps)

| Task | PPO | PBT | SAPG | **CPO** |
|------|-----|-----|------|---------|
| ShadowHand | 10661±1050 | 10294±1728 | 12882±343 | **13762±414** |
| AllegroHand | 10439±1282 | 13239±239 | 11989±817 | **14421±885** |
| Reorientation | 1.04±0.98 | 2.92±4.27 | 38.79±1.66 | **43.75±0.65** |
| Two-Arms | 1.41±0.80 | 26.43±11.12 | 5.11±3.41 | **35.30±2.77** |

### Ablation Study (IS Ratio Deviation and ESS at $5\times10^9$ steps)

| Method | Avg. IS Ratio Deviation↓ | ESS Rate↑ |
|--------|--------------------------|-----------|
| SAPG | 0.889 | 0.0223 |
| CPO($\lambda_f$=0.5) | Lower | Higher |

### Key Findings
- CPO reaches SAPG's final performance in approximately half the environment steps on most tasks.
- SAPG completely fails on the Two-Arms Reorientation task (5.11), whereas CPO successfully learns it (35.30).
- KL constraints bring IS ratios closer to 1, validating the theoretical analysis.
- Follower policies naturally form a structured distribution around the leader, exhibiting "emergent exploration behavior."

## Highlights & Insights

- **Strong theoretical grounding**: The harms of excessive diversity are demonstrated from two perspectives — IS ratio deviation and PPO clipping bias.
- **Simple and practical method**: Significant improvements are achieved by adding only KL constraints and adversarial rewards on top of SAPG.
- **In-depth comparative analysis**: Reveals SAPG's policy misalignment issue, where some follower policies deviate substantially from the leader.

## Limitations & Future Work

- The KL constraint strength $\lambda_f$ is robust but still requires manual specification; adaptive adjustment schemes are worth exploring.
- Performance gains are limited on simpler tasks (e.g., Locomotion).
- The current approach uses a one-dimensional conditioning vector to distinguish policies; richer conditioning schemes may further enhance diversity.

## Related Work & Insights

- SAPG (Singla et al., 2024) is the direct predecessor; CPO addresses its policy misalignment problem.
- The discriminator idea from DIAYN (Eysenbach et al., 2018) is cleverly repurposed to promote inter-policy dispersion.
- Takeaway: In large-scale distributed RL, "structured diversity" is more effective than "unconstrained diversity."

## Rating
- Novelty: ⭐⭐⭐⭐ Theory-driven design, though individual components (KL constraints, discriminators) have prior precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 10 tasks, detailed ablations, ISS/KL analysis, 5 random seeds.
- Writing Quality: ⭐⭐⭐⭐ Coherent logic from problem identification to theoretical analysis to method design.
- Value: ⭐⭐⭐⭐ Provides practical guidance for exploration strategies in large-scale parallel RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots](robocasa365_a_large-scale_simulation_framework_for_training_and_benchmarking_gen.md)
- [\[AAAI 2026\] Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy](../../AAAI2026/robotics/coordinated_humanoid_robot_locomotion_with_symmetry_equivariant_reinforcement_le.md)
- [\[ICLR 2026\] Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control](towards_bridging_the_gap_between_large-scale_pretraining_and_efficient_finetunin.md)
- [\[AAAI 2026\] Scalable Multi-Objective and Meta Reinforcement Learning via Gradient Estimation](../../AAAI2026/robotics/scalable_multi-objective_and_meta_reinforcement_learning_via_gradient_estimation.md)
- [\[NeurIPS 2025\] Opinion: Towards Unified Expressive Policy Optimization for Robust Robot Learning](../../NeurIPS2025/robotics/opinion_towards_unified_expressive_policy_optimization_for_robust_robot_learning.md)

</div>

<!-- RELATED:END -->
