---
title: >-
  [Paper Note] ELEMENTAL: Interactive Learning from Demonstrations and Vision-Language Models for Reward Design in Robotics
description: >-
  [ICML 2025][Multimodal VLM][Reward Design] ELEMENTAL integrates vision-language models (VLMs) with inverse reinforcement learning (IRL) to extract feature functions via VLMs, optimize weights via IRL, and iteratively improve via self-reflection, achieving a 42.3% improvement over EUREKA across 9 IsaacGym tasks.
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Reward Design"
  - "VLM"
  - "Inverse Reinforcement Learning"
  - "Learning from Demonstrations"
  - "Robotics"
date: 2026-05-08
content_hash: acea3476221cf347
---

# ELEMENTAL: Interactive Learning from Demonstrations and Vision-Language Models for Reward Design in Robotics

**Conference**: ICML 2025  
**arXiv**: [2411.18825](https://arxiv.org/abs/2411.18825)  
**Code**: None  
**Area**: Multimodal / VLM  
**Keywords**: Reward Design, VLM, Inverse Reinforcement Learning, Learning from Demonstrations, Robotics

## TL;DR
ELEMENTAL integrates vision-language models (VLMs) with inverse reinforcement learning (IRL) to extract feature functions via VLMs, optimize weights via IRL, and iteratively improve via self-reflection, achieving a 42.3% improvement over EUREKA across 9 IsaacGym tasks.

## Background & Motivation
**Background**: RL performs outstandingly in robotics tasks, but the core bottleneck remains reward function design—requiring extensive domain knowledge and manual parameter tuning.

**Limitations of Prior Work**: (a) LLM-based methods like EUREKA rely solely on text descriptions to generate reward functions, failing to precisely capture complex spatial tasks; (b) LLMs struggle to balance the weights of different features; (c) text-only task specifications cannot capture implicit user preferences.

**Key Challenge**: LLMs excel at semantic understanding and feature recognition but struggle with mathematical optimization and weight allocation; IRL excels at matching behaviors from demonstrations but requires predefined features. The two are highly complementary.

**Key Insight**: Let the VLM handle feature extraction and IRL handle weight optimization, while introducing visual demonstrations as auxiliary information.

**Core Idea**: A three-phase loop: initial prompt for VLM to generate feature functions $\to$ Approximate MaxEnt-IRL to learn reward weights and policies $\to$ self-reflection to compare differences in feature counts and iteratively improve.

## Method

### Overall Architecture
Input: Environment code + task text description + visual demonstration $\to$ Phase 1: VLM generates feature function $\phi(s)$ $\to$ Phase 2: IRL learns $R_\theta(s) = \theta^T \phi(s)$ $\to$ Phase 3: Compare feature count differences between policy and demonstrations $\to$ Feedback to VLM to modify features $\to$ Iterate.

### Key Designs

1. **Phase 1 - Initial Prompt (VLM Feature Extraction)**:

    - Inputs include: environment MDP code, task text description, and visual demonstrations (overlays / keyframes)
    - The VLM (GPT-4o) outputs feature functions $\phi: \mathcal{S} \to \mathbb{R}^n$ in the format of Python code
    - **Design Motivation**: Visual demonstrations compensate for the inadequacy of text descriptions; the VLM's code capabilities are restricted to "feature extraction" rather than "complete reward design".

2. **Phase 2 - Learning (Approximate MaxEnt-IRL)**:

    - Reward model: $R_\theta(s) = \theta^T \phi(s)$, initial $\theta = \{1/n\}^n$
    - Gradient: $\nabla_\theta \approx \mathbb{E}_{\tau \sim \mathcal{D}}[\sum_s \phi(s)] - \mathbb{E}_{\tau \sim \pi_\psi}[\sum_s \phi(s)]$
    - Alternatingly optimize $\theta$ (reward weights) and $\psi$ (PPO policy)
    - Key trick: Gradient L1 normalization + $\theta$ L1 normalization to ensure training stability
    - **Design Motivation**: Since direct computation of the partition function is intractable, policies are used for approximation.

3. **Phase 3 - Reflection (Self-Reflection)**:

    - Compute feature count vectors for policy trajectories and demonstration trajectories: $\vec{\Phi}_{\pi_\psi}$ vs $\vec{\Phi}_\mathcal{D}$
    - Feedback differences to the VLM to modify the feature functions
    - Automatically completed without requiring additional human input
    - **Design Motivation**: Simulates the "observe $\to$ execute $\to$ reflect $\to$ improve" loop in human learning.

### Loss & Training
- Reward weights: Gradient ascent $\theta \leftarrow \theta + \alpha \nabla_\theta'$, with L1-normalized $\theta$
- Policy: Optimize $\pi_\psi$ using PPO to maximize $J(\pi_\psi)$
- Perform $m$ rounds of IRL iterations alternately

## Key Experimental Results

### Main Results

| Task | ELEMENTAL | EUREKA | BC | IRL | GT Reward |
|------|-----------|--------|-----|-----|-----------|
| Cartpole | **233.92** | 215.91 | 149.85 | 28.15 | 260.14 |
| Ant | **8.49** | 6.88 | -0.05 | 0.88 | 7.00 |
| Humanoid | **4.70** | 3.78 | -0.43 | 2.13 | 5.07 |
| FrankaCabinet | **0.36** | 0.21 | 0.01 | 0.00 | 0.40 |
| AllegroHand | **22.97** | 11.12 | 0.04 | 0.01 | 23.70 |
| ShadowHand | **2.71** | 0.001 | 0.03 | 0.01 | 0.15 |
| Overall Gain | **+42.3%** | baseline | — | — | upper bound |

### Ablation Study

| Configuration | Average Performance | Explanation |
|------|---------|------|
| Full ELEMENTAL | Optimal | Full three-phase workflow |
| w/o Self-Reflection | Degraded | Lacks iterative improvement |
| w/o Visual Input | Degraded | Text only is insufficient for complex tasks |
| w/o Norm 1 (Gradient Normalization) | Degraded | Unstable training |
| w/o Norm 2 (Weight Normalization) | Degraded | Inconsistent reward scale |

### Key Findings
- The execution rate of GPT-4o's feature code (~80%) is significantly higher than EUREKA's reward code execution rate (<50%)
- Generalization experiments: ELEMENTAL improves performance by 41.3% over EUREKA across 4 Ant variants—EUREKA might have memorized standard IsaacGym rewards
- This is the first successful application of IRL to high-dimensional tasks in IsaacGym

## Highlights & Insights
- **Complementary Architecture**: VLM for feature recognition + IRL for weight optimization, playing to their respective strengths
- **First successful IRL application in IsaacGym**: Enabled by the structured feature space provided by the VLM
- **Self-Reflection Mechanism**: Feature count differences provide a more precise improvement signal than textual feedback

## Limitations & Future Work
- Runtime is approximately 2.5 times longer than EUREKA (168 vs 68 minutes)
- Not yet validated on real hardware
- The format of visual demonstrations (overlays/keyframes) needs manual selection based on the task type

## Related Work & Insights
- EUREKA (Ma et al. 2023) is a direct competitor
- RL-VLM-F uses VLM as a proxy reward but lacks interaction
- AIRL (Fu et al. 2018) provides the RL training paradigm
- Insight: LLMs/VLMs should not be expected to perform tasks they struggle with (such as mathematical optimization), but should instead focus on semantic understanding

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The integration of VLM+IRL and the self-reflection mechanism is very clever
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 9 IsaacGym tasks + 4 generalization variants + comprehensive ablation study
- **Writing Quality**: ⭐⭐⭐⭐ Clear methodology explanation, sound experimental design
- **Value**: ⭐⭐⭐⭐⭐ Provides a practical and powerful solution for robot reward design

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RoboSpatial: Teaching Spatial Understanding to 2D and 3D Vision-Language Models for Robotics](../../CVPR2025/multimodal_vlm/robospatial_teaching_spatial_understanding_to_2d_and_3d_vision-language_models_f.md)
- [\[ICML 2025\] The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models](the_devil_is_in_the_details_tackling_unimodal_spurious_correlations_for_generali.md)
- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](../../CVPR2026/multimodal_vlm/sapave_active_perception_manipulation_vla_roboti.md)
- [\[ICML 2025\] Learning Invariant Causal Mechanism from Vision-Language Models](learning_invariant_causal_mechanism_from_vision-language_models.md)
- [\[CVPR 2025\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](../../CVPR2025/multimodal_vlm/continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)

</div>

<!-- RELATED:END -->
