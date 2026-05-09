---
title: >-
  [Paper Note] OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning
description: >-
  [ICLR 2026][Robotics][MLLM] This paper proposes OmniEVA, which addresses two critical gaps in spatial MLLMs — poor geometric adaptability (2D-only or hard-coded 3D injection) and the absence of embodiment constraints (plans that are theoretically feasible but physically unexecutable) — via a task-adaptive gated router that dynamically injects 3D positional encodings only when geometric reasoning is required, and an embodiment-aware reasoning framework that integrates physical constraints into the planning loop. OmniEVA achieves state-of-the-art performance on 7 out of 8 benchmarks.
tags:
  - ICLR 2026
  - Robotics
  - MLLM
  - Task-Adaptive 3D Grounding
  - Gated Routing
  - Embodiment-aware Reasoning
  - GRPO
date: 2026-05-08
content_hash: ec2002b7c3ce5670
---

# OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning

**Conference**: ICLR 2026
**arXiv**: [2509.09332](https://arxiv.org/abs/2509.09332)
**Code**: [Project Page](https://github.com/OmniEVA-Project)
**Area**: Embodied Intelligence / 3D Reasoning
**Keywords**: MLLM, Task-Adaptive 3D Grounding, Gated Routing, Embodiment-aware Reasoning, GRPO

## TL;DR
This paper proposes OmniEVA, which addresses two critical gaps in spatial MLLMs — poor geometric adaptability (2D-only or hard-coded 3D injection) and the absence of embodiment constraints (plans that are theoretically feasible but physically unexecutable) — via a task-adaptive gated router that dynamically injects 3D positional encodings only when geometric reasoning is required, and an embodiment-aware reasoning framework that integrates physical constraints into the planning loop. OmniEVA achieves state-of-the-art performance on 7 out of 8 benchmarks.

## Background & Motivation

### State of the Field

MLLMs are increasingly applied to embodied intelligence, requiring spatial understanding, reasoning, and action. Two dominant paradigms exist: (1) direct 2D RGB input, which lacks 3D geometric information; and (2) 3D-LLMs with hard-coded 3D injection, which lack flexibility.

### Limitations of Prior Work

**(1) Geometric Adaptability Gap**: 2D-only models fail on 3D reasoning tasks (e.g., stacking, occlusion handling, navigation); hard-coded 3D injection in 3D-LLMs introduces noise when 3D input is irrelevant or noisy.

### Root Cause

**(2) Embodiment Constraint Gap**: Models trained on web images and videos ignore robot physical constraints, producing plans that are theoretically valid but physically unexecutable (e.g., infeasible grasp poses, workspace violations, kinematic infeasibility).

**Key Insight**: (1) A gated router dynamically determines whether 3D information is needed, enabling on-demand injection; (2) TE-GRPO training teaches the model to respect physical constraints.

## Method

### Task-Adaptive Gated Router (TAGR)

1. **3D Positional Encoding**: Depth maps are unprojected to world coordinates, patch-level averaged, and encoded via sinusoidal functions to obtain $V^p \in \mathbb{R}^{N \times H_p \times W_p \times d_v}$.

2. **Gating Decision**:
   - Task condition: a sentence Transformer encodes the instruction to produce $V^T$
   - Scene condition: mean-pooled visual encoder output $V_{avg}^I$
   - Concatenated features are passed through an MLP to produce 2D gate logits, followed by Gumbel-Softmax for a binary decision

3. **Dynamic Injection**:
   - Gate = 1: $V^{final} = V^I + V^p$ (3D positional encoding added)
   - Gate = 0: $V^{final} = V^I$ (2D only)
   - The routing is automatically determined per task and scene, avoiding noise from unnecessary 3D injection

### Embodiment-aware Reasoning

1. **Primitive Skill Decomposition**:
   - Where2Go: navigation target selection
   - Where2Grasp: grasp pose estimation
   - Where2Approach: approach pose determination
   - Where2Fit: placement feasibility assessment

2. **TE-GRPO (Task- and Embodiment-aware GRPO)**:
   - Applied as a post-training stage using Group Relative Policy Optimization (GRPO)
   - Reward function accounts for: task objectives, object affordances, workspace boundaries, and kinematic feasibility
   - Ensures that generated plans are physically executable

### Two-Stage Training
- Stage 1: Supervised Fine-Tuning (SFT) on 2D + 3D VQA and embodied reasoning data
- Stage 2: TE-GRPO post-training via reinforcement learning to optimize executability

## Key Experimental Results

### 8 Benchmarks (2D + 3D + Video)

### Main Results

| Benchmark Type | Model | Performance |
|----------------|-------|-------------|
| 2D Spatial Reasoning | OmniEVA | **SOTA** |
| 3D Spatial Reasoning | OmniEVA | **SOTA (7/8)** |
| Object Navigation (HM3D) | OmniEVA | **Leaderboard #1** |
| Object Navigation (MP3D) | OmniEVA | **Leaderboard #1** |

### 4 Primitive Skill Benchmarks

### Ablation Study

| Skill | OmniEVA vs. SOTA | Description |
|-------|------------------|-------------|
| Where2Go | +5% | Navigation target selection |
| Where2Grasp | +8% | Grasp pose estimation |
| Where2Approach | +6% | Approach strategy |
| Where2Fit | +7% | Placement adaptation |

### Key Findings
- The gated router deactivates 3D injection for ~40% of tasks, confirming that these tasks genuinely do not require geometric reasoning — validating the adaptive strategy.
- Hard-coded 3D injection in baseline models degrades performance on 2D tasks that do not require 3D input, demonstrating the value of TAGR.
- TE-GRPO post-training raises the proportion of physically executable plans from ~65% to ~90% compared to SFT alone.

## Highlights & Insights
- **"3D on Demand" Design Philosophy**: Rather than applying 3D to all tasks, the model learns when 3D reasoning is necessary — a more flexible and accurate approach than hand-crafted rules.
- **Contribution of Primitive Skill Benchmarks**: The four new benchmarks (Where2Go / Where2Grasp / Where2Approach / Where2Fit) provide the first systematic evaluation of embodied plan executability.
- **TE-GRPO Bridges LLM Training and Robotics**: Combining GRPO — a mainstream LLM post-training method — with physics-constraint rewards represents a natural and effective integration of LLMs into robotics.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Dual innovation in task-adaptive 3D grounding and embodiment-aware reasoning
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 + 4 benchmarks, ablations, and leaderboard results
- Writing Quality: ⭐⭐⭐⭐ Architecture descriptions are clear and well-organized
- Value: ⭐⭐⭐⭐⭐ Significant contribution to embodied MLLMs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] REI-Bench: Can Embodied Agents Understand Vague Human Instructions in Task Planning?](rei-bench_can_embodied_agents_understand_vague_human_instructions_in_task_planni.md)
- [\[NeurIPS 2025\] MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning](../../NeurIPS2025/robotics/mesatask_towards_task-driven_tabletop_scene_generation_via_3d_spatial_reasoning.md)
- [\[ICLR 2026\] Sysformer: Safeguarding Frozen Large Language Models with Adaptive System Prompts](sysformer_safeguarding_frozen_large_language_models_with_adaptive_system_prompts.md)
- [\[ICLR 2026\] THOR: Tool-Integrated Hierarchical Optimization via RL for Mathematical Reasoning](thor_tool-integrated_hierarchical_optimization_via_rl_for_mathematical_reasoning.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](../../CVPR2026/robotics/geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)

</div>

<!-- RELATED:END -->
