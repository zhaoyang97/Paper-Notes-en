---
title: >-
  [Paper Note] SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning
description: >-
  [NeurIPS 2025][LLM Alignment][VLA safety] This work is the first to systematically apply the Constrained Markov Decision Process (CMDP) framework from Safe Reinforcement Learning (SafeRL) to safety alignment of Vision-Language-Action (VLA) models. Through a four-stage Integrated Safety Approach (ISA)—Model, Elicit, Constrain, and Assure—the method achieves an 83.58% reduction in safety violation costs on mobile manipulation tasks while maintaining task performance (+3.85%).
tags:
  - NeurIPS 2025
  - LLM Alignment
  - VLA safety
  - constrained MDP
  - safe reinforcement learning
  - embodied AI
  - robot safety
date: 2026-05-08
content_hash: 5e9baa178bf7cd89
---

# SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning

**Conference**: NeurIPS 2025
**arXiv**: [2503.03480](https://arxiv.org/abs/2503.03480)
**Code**: [Project Page](https://pku-safevla.github.io)
**Area**: LLM Alignment
**Keywords**: VLA safety, constrained MDP, safe reinforcement learning, embodied AI, robot safety

## TL;DR
This work is the first to systematically apply the Constrained Markov Decision Process (CMDP) framework from Safe Reinforcement Learning (SafeRL) to safety alignment of Vision-Language-Action (VLA) models. Through a four-stage Integrated Safety Approach (ISA)—Model, Elicit, Constrain, and Assure—the method achieves an 83.58% reduction in safety violation costs on mobile manipulation tasks while maintaining task performance (+3.85%).

## Background & Motivation

**Safety challenges of VLA models**: VLA models (e.g., RT-2, OpenVLA) are emerging as general-purpose robot policies, but face physical safety risks to the environment, the robot itself, and humans when deployed in the real world.

**Limitations of prior work**: Safety alignment methods for LLMs/VLMs (e.g., RLHF) focus on abstract, intent-level safety and cannot be directly applied to physical-world safety constraints. Existing VLA training paradigms (IL or standard RL) lack explicit safety constraint mechanisms.

**Core Problem**: How can safety constraints be explicitly integrated into VLA models without sacrificing task performance?

## Method

### Overall Architecture: Integrated Safety Approach (ISA)

ISA consists of four interrelated stages: Modeling (defining safety requirements) → Eliciting (provoking unsafe behaviors) → Constraining (applying SafeRL-based policy constraints) → Assuring (targeted safety evaluation).

### Problem Formulation: CMDP Framework

VLA safety alignment is formalized as a Constrained Markov Decision Process (CMDP), where the VLA policy maps observation histories to actions while constraining accumulated safety costs below a threshold:

$$\pi^* = \arg\max_{\pi_\theta \in \Pi_\mathcal{C}} \mathcal{J}(\pi_\theta)$$

Five categories of safety-critical components are defined:
- **Corners**: Narrow corners causing the robot to become stuck or repeatedly collide.
- **Blind Spots**: Collisions with previously seen but currently occluded obstacles due to short-term spatial perception failures.
- **Fragile Collections**: Collateral damage to nearby fragile objects during manipulation.
- **Critical Points**: Indirect actions causing unstable objects to fall.
- **Dangerous Equipment**: Prohibited interactions with hazardous devices.

### Risk Elicitation: Safety-CHORES Benchmark

150,000 diverse indoor scenes are generated via ProcTHOR, combined with 800,000 3D assets from Objaverse, to systematically elicit safety violations within the AI2THOR simulator.

### Constrained Policy Learning: Lagrangian Method

The constrained optimization is converted into an unconstrained min-max problem via Lagrangian relaxation:

$$\min_\theta \max_{\lambda \geq 0} [-\mathcal{J}_r(\theta) + \sum_{i=0}^n \lambda_i \mathcal{J}_{c_i}(\theta)]$$

Dynamic Lagrange multipliers adaptively balance reward and cost objectives, prioritizing safety before maximizing task performance.

### Safety Assurance

Evaluation is conducted along three dimensions: test-time safety (standard test set + OOD), long-tail safety (low-frequency dangerous events), and extreme-failure safety (behavior when tasks are infeasible).

## Key Experimental Results

### Main Results: ISA vs. Baselines

| Method | ObjNav SR↑/CC↓ | PickUp SR↑/CC↓ | Fetch SR↑/CC↓ |
|--------|----------------|----------------|---------------|
| SPOC-DINOv2 | 0.43/13.50 | 0.86/10.29 | 0.14/13.97 |
| FLaRe | 0.822/12.36 | 0.912/7.08 | 0.605/43.36 |
| **ISA** | **0.865/1.85** | **0.928/0.37** | **0.637/8.08** |

Average CC reduction: **83.58%**; average SR improvement: **+3.85%**.

### OOD Robustness

| Perturbation | ObjNav CC | PickUp CC | Fetch CC |
|--------------|-----------|-----------|----------|
| None | 1.85 | 0.37 | 8.98 |
| +Color | 3.10 | 1.82 | 15.34 |
| +All | 3.21 | 0.41 | 12.50 |

Safe behaviors remain stable under OOD conditions.

### Extreme Failure Scenarios

| Method | Avg. CC |
|--------|---------|
| FLaRe | 71.68 |
| SPOC | 14.63 |
| **ISA** | **2.20** |

Even when tasks completely fail (SR ≈ 0), ISA maintains safe behavior.

### Ablation Study

| Ablation | Key Findings |
|----------|--------------|
| Remove safety-critical components | CC increases from 1.854 to 5.01 |
| Fixed penalty coefficient | Dynamic Lagrange multipliers significantly outperform fixed coefficients |
| Different cost thresholds | 20% threshold yields the best balance |

### Key Findings

1. **Decoupling of safety and performance**: ISA's safety cost distribution is independent of task success or failure, whereas FLaRe exhibits a significant negative correlation.
2. **Elimination of catastrophic trajectories**: ISA completely eliminates high-risk trajectories with CC > 10.
3. **Cross-model generality**: The approach is effective across different VLA architectures including EmbCLIP and Embodied-Codebook.

## Highlights & Insights

1. **First systematic VLA safety alignment**: A complete pipeline from modeling to assurance, constituting an integrated safety approach.
2. **Pareto-optimal safety–performance tradeoff**: Explicit tradeoff achieved via the CMDP framework.
3. **Safety-CHORES benchmark**: Carefully designed safety-critical scenarios that expose VLA vulnerabilities more effectively than existing benchmarks (2× higher CC).
4. **Sim-to-Real validation**: Successfully deployed on a physical robot platform.

## Limitations & Future Work

1. Sim-to-Real transfer is validated only on the Safety-PickUp task.
2. Safety predicates require manual design; automated safety specification extraction is an important future direction.
3. Trajectory-level cost attribution assigns violations only to the last step of the offending segment; credit assignment warrants further investigation.
4. SafeRL training requires 15M–25M steps, entailing considerable computational overhead.

## Related Work & Insights

- **FLaRe**: Maximizes task reward via standard RL; ISA explicitly constrains safety costs under the CMDP framework.
- **Safe-RLHF**: Addresses intent-level safety for LLMs; ISA extends this to physical-world embodied safety.
- **Insight**: The CMDP framework provides a principled optimization paradigm for safety in embodied AI.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic application of SafeRL/CMDP to VLA safety alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers main results, ablations, OOD, extreme failures, cross-model generalization, and Sim-to-Real transfer.
- Writing Quality: ⭐⭐⭐⭐ The four-stage ISA framework is clearly presented, though the paper is lengthy.
- Value: ⭐⭐⭐⭐⭐ A landmark contribution to the field of embodied AI safety.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Alignment of Large Language Models with Constrained Learning](alignment_of_large_language_models_with_constrained_learning.md)
- [\[NeurIPS 2025\] Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons](towards_understanding_safety_alignment_a_mechanistic_perspective_from_safety_neu.md)
- [\[NeurIPS 2025\] Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis](limited_preference_data_learning_better_reward_model_with_latent_space_synthesis.md)
- [\[NeurIPS 2025\] Human-assisted Robotic Policy Refinement via Action Preference Optimization](human-assisted_robotic_policy_refinement_via_action_preference_optimization.md)
- [\[NeurIPS 2025\] LLM Safety Alignment is Divergence Estimation in Disguise](llm_safety_alignment_is_divergence_estimation_in_disguise.md)

<!-- RELATED:END -->
