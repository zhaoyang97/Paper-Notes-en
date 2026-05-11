---
title: >-
  [Paper Note] Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models
description: >-
  [CVPR 2026][Multimodal VLM][Autonomous Driving VLA] This paper identifies an overlooked "Narrow Policy" bottleneck in driving VLA models—over-exploitation during the IL phase causes exploration collapse, which in turn constrains the RL phase. The proposed Curious-VLA framework achieves SOTA on Navsim (PDMS 90.3, Best-of-N 94.8) via feasible trajectory expansion and diversity-aware RL.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Autonomous Driving VLA
  - Narrow Policy
  - Exploration-Exploitation Dilemma
  - Reinforcement Learning
  - Trajectory Diversity
date: 2026-05-08
content_hash: 27c2ce6524b8970f
---

# Devil is in Narrow Policy: Unleashing Exploration in Driving VLA Models

**Conference**: CVPR 2026
**arXiv**: [2603.06049](https://arxiv.org/abs/2603.06049)
**Code**: [GitHub](https://github.com/Mashiroln/curious_vla.git)
**Area**: Multimodal VLM
**Keywords**: Autonomous Driving VLA, Narrow Policy, Exploration-Exploitation Dilemma, Reinforcement Learning, Trajectory Diversity

## TL;DR

This paper identifies an overlooked "Narrow Policy" bottleneck in driving VLA models—over-exploitation during the IL phase causes exploration collapse, which in turn constrains the RL phase. The proposed Curious-VLA framework achieves SOTA on Navsim (PDMS 90.3, Best-of-N 94.8) via feasible trajectory expansion and diversity-aware RL.

## Background & Motivation

**Two-stage paradigm for driving VLA**: Current driving VLA models universally adopt a two-stage IL (imitation learning) → RL (reinforcement learning) training pipeline, yet suffer from a fundamental exploration-exploitation imbalance.

**Discovery of the Narrow Policy problem**: The IL phase uses cross-entropy loss to imitate ground-truth trajectories, causing the policy distribution to collapse to a single mode; trajectories generated across multiple inference runs are nearly identical (mean-pFDE of only 0.20–0.33 m).

**Advantage collapse in the RL phase**: After policy collapse, rewards sampled by GRPO are nearly identical ($R(y_i) \approx \mu_R$), with standard deviation $\sigma_R \to 0$, leading to advantage estimates $A_i \to 0$ and vanishing gradients.

**Inherent deficiencies of cross-entropy loss**: CE treats all non-ground-truth tokens as equivalent errors, lacks any notion of spatial or functional proximity, and encourages overconfidence in a single mode.

**Temporal-scale mismatch**: Variance in far-horizon waypoints is orders of magnitude larger than that of near-horizon waypoints (variance at $t=4s$ vs. $t=0.5s$), causing far-horizon loss terms to dominate training.

**Lack of behavioral diagnostic metrics**: Prior work lacks quantitative tools for diagnosing the narrow policy phenomenon.

## Method

### Overall Architecture

Curious-VLA introduces improvements to both the IL and RL stages: the IL stage employs Feasible Trajectory Expansion (FTE) for training data diversification along with step-level normalization; the RL stage maintains exploration via Adaptive Diversity-Aware Sampling (ADAS) and Span Driving Reward (SDR).

### Key Designs

#### Feasible Trajectory Expansion (FTE)

1. **Exploratory data augmentation**: 12k challenging scenarios (multi-lane / intersection / occlusion) are selected from 103k NavTrain samples; ReCogDrive is used to perturb diffusion latents to generate diverse feasible trajectories, which are safety-filtered by a PDMS scorer, expanding the dataset to 142k samples.
2. **CoT data synthesis**: Qwen2.5-VL-72B is used to generate four-stage reasoning chains (perception → interpretation → meta-behavior → trajectory).
3. **Step-level normalization**: $\tilde{w}_t = (w_t - \mu_t) / \sigma_t$, independently normalizing each prediction step to balance gradient magnitudes across time steps.

#### Adaptive Diversity-Aware Sampling (ADAS)

The outcome of each scene is modeled as a Bernoulli process; success rate $\hat{p}$ is estimated via $M$ offline rollouts. Only scenes satisfying two diversity conditions are retained: (1) $\hat{p}^G + (1-\hat{p})^G < \epsilon_{\text{div}}$ (excluding all-success or all-failure scenes); (2) $|\sigma_R - \sqrt{\hat{p}(1-\hat{p})} R_{\text{range}}| < \epsilon_{\text{conf}}$ (ensuring the reward distribution conforms to theoretical expectations).

#### Span Driving Reward (SDR)

The original PDMS is reformulated in a focal-style manner:

$$R_{\text{span}} = \prod_{c \in C} c \cdot \frac{\sum w'_m (1-(1-m)^{\gamma_m})}{\sum w'_m}$$

This nonlinearly amplifies reward differences between suboptimal and optimal behaviors.

### Loss & Training

The IL stage employs standard CE loss (after step-level normalization); the RL stage uses the GRPO objective with SDR rewards.

## Key Experimental Results

### Main Results: Navsim V1 Benchmark

| Method | Backbone | PDMS↑ | NC↑ | EP↑ |
|--------|----------|-------|-----|-----|
| UniAD | - | 84.0 | 97.7 | 79.2 |
| ReCogDrive | InternVL2-8B | 89.6 | 98.2 | 83.5 |
| AutoVLA | Qwen2.5-VL-3B | 89.1 | 98.4 | 81.9 |
| AdaThinkDrive | InternVL3-8B | 90.3 | 98.4 | 84.4 |
| **Curious-VLA** | Qwen2.5-VL-3B | **90.3** | 98.4 | **88.5** |
| **Curious-VLA†(BoN)** | Qwen2.5-VL-3B | **94.8** | - | - |

### Ablation Study: Behavioral Diagnostics

| Method | Diversity (pFDE)↑ | Quality (min-FDE)↓ | PDMS↑ |
|--------|-------------------|-------------------|-------|
| Qwen2.5-VL | 0.20 m | 1.05 m | - |
| ReCogDrive | 0.33 m | - | - |
| **Curious-VLA** | **Best** | **Best** | **90.3** |

### Key Findings

- BoN PDMS of 94.8 directly demonstrates that exploration potential has been successfully unleashed.
- Applying GRPO directly to the post-IL model degrades performance, validating the obstruction posed by the narrow policy to RL.
- Step-level normalization significantly improves learning of near-horizon waypoints.
- ADAS effectively prevents early saturation during the RL phase.

## Highlights & Insights

- **Discovery and formalization of the Narrow Policy problem** constitutes a significant contribution, revealing a fundamental bottleneck in the IL→RL pipeline.
- The three-dimensional behavioral diagnostic framework (Diversity / Quality / Performance) is intuitive and effective.
- SOTA performance is achieved with only a 3B model and a single camera, demonstrating clear efficiency advantages.
- Best-of-N evaluation elegantly validates the exploration potential of the policy.
- The problem is systematically addressed across three dimensions: data augmentation, sampling strategy, and reward function design.

## Limitations & Future Work

- FTE relies on ReCogDrive's diffusion module to generate diverse trajectories, introducing an external dependency.
- Navsim is a closed-loop simulator; real-world performance remains to be verified.
- The offline rollout stage of ADAS incurs considerable computational overhead.
- The core analysis is grounded in the VLA-Token paradigm; the extent of narrow policy in the VLA-Planner paradigm is not thoroughly discussed.

## Related Work & Insights

- **Relation to DeepSeek-R1/GRPO**: This paper identifies the failure mode of GRPO in driving scenarios and proposes targeted remedies.
- **Comparison with DAPO**: DAPO improves advantage estimation, while Curious-VLA addresses the issue from the perspective of data diversity.
- The "Narrow Policy" concept generalizes to other IL→RL settings (e.g., robotic manipulation).

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] VILTA: A VLM-in-the-Loop Adversary for Enhancing Driving Policy Robustness](../../AAAI2026/multimodal_vlm/vilta_a_vlm-in-the-loop_adversary_for_enhancing_driving_poli.md)
- [\[CVPR 2026\] World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training](rehearsevla_simulated_post-training_for_vlas_with_physically-consistent_world_mo.md)
- [\[CVPR 2026\] TreeTeaming: Autonomous Red-Teaming of Vision-Language Models via Hierarchical Strategy Exploration](treeteaming_autonomous_red-teaming_of_vision-language_models_via_hierarchical_s.md)
- [\[CVPR 2026\] Prune2Drive: A Plug-and-Play Framework for Accelerating Vision-Language Models in Autonomous Driving](prune2drive_a_plug-and-play_framework_for_accelerating_vision-language_models_in.md)
- [\[CVPR 2026\] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](vggdrive_empowering_vision-language_models_with_cross-view_geometric_grounding_f.md)

</div>

<!-- RELATED:END -->
