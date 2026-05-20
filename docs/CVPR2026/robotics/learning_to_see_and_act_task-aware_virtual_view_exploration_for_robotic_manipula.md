---
title: >-
  [Paper Note] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation
description: >-
  [CVPR 2026][Robotics][View exploration] This paper proposes the TVVE framework, which employs a reinforcement learning-driven Multi-View Exploration Policy (MVEP) to select optimal virtual camera viewpoints and re-render…
tags:
  - "CVPR 2026"
  - "Robotics"
  - "View exploration"
  - "multi-task manipulation"
  - "Mixture-of-Experts"
  - "virtual view rendering"
  - "reinforcement learning"
date: 2026-05-08
content_hash: 082b774c72fc058e
---

# Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation

**Conference**: CVPR 2026
**arXiv**: [2508.05186](https://arxiv.org/abs/2508.05186)  
**Code**: [Available](https://github.com/) (TAVP)  
**Area**: Robotics / Embodied Intelligence
**Keywords**: View exploration, multi-task manipulation, Mixture-of-Experts, virtual view rendering, reinforcement learning

## TL;DR

This paper proposes the TVVE framework, which employs a reinforcement learning-driven Multi-View Exploration Policy (MVEP) to select optimal virtual camera viewpoints and re-render observations online. A task-aware MoE visual encoder (TaskMoE) is designed to mitigate cross-task feature interference. The framework achieves an average success rate of 86.6% across 18 tasks on RLBench.

## Background & Motivation

### 1. State of the Field

Vision-Language-Action (VLA) models have advanced rapidly in end-to-end robotic manipulation. Methods such as OpenVLA and π₀ have demonstrated strong performance on complex, fine-grained tasks. These approaches typically rely on single-viewpoint or a small number of fixed-viewpoint RGB-D observations to guide action prediction.

### 2. Limitations of Prior Work

Fixed viewpoints are severely problematic in cluttered or dynamic scenes. When target objects or end-effectors are repositioned during task execution, fixed cameras frequently cause occlusion. For example, when executing the instruction "put sugar in the cupboard," a front-facing camera can observe the cupboard but not the sugar, while left/right shoulder cameras can observe the grasped sugar but not the cupboard, resulting in incomplete scene understanding and failed action prediction.

### 3. Root Cause

- **Fixed viewpoints vs. dynamic scenes**: Fixed cameras cannot adaptively select optimal observation angles as a task progresses.
- **Shared encoder vs. task heterogeneity**: Prior methods such as RVT/RVT-2 use a shared visual encoder for all tasks, leading to severe feature interference across tasks with different characteristics (e.g., grasping an apple vs. opening a drawer).

### 4. Core Problem

(1) How to dynamically select virtual observation viewpoints that maximize coverage of task-critical information; (2) how to avoid interference between visual features and action policies in a multi-task setting.

### 5. Starting Point

The paper reconstructs a 3D point cloud from RGB-D inputs as a scene representation, then uses a reinforcement learning policy to explore optimal virtual camera poses over the point cloud, from which 2D observation images are re-rendered. A MoE architecture routes different tasks to specialized expert networks.

### 6. Core Idea

**"Learn to see before learning to act"** — viewpoint selection is formulated as a trainable RL policy problem. Pseudo-environment interaction avoids costly physical simulation, and a decoupled-gating TaskMoE enables selective parameter sharing across tasks.

## Method

### Overall Architecture

TVVE takes language instructions, multi-view RGB-D images, and the current gripper state as input, and processes them in four steps:

1. **3D Reconstruction**: Multiple RGB-D images are converted to point clouds and aggregated into a global point cloud in world coordinates.
2. **Coarse Grounding**: The approximate end-effector position is predicted, the point cloud is centered at that position, and the critical region is cropped.
3. **View Exploration (MVEP)**: An RL policy predicts $K$ optimal virtual camera poses over the global point cloud, and 2D images are re-rendered from the cropped point cloud.
4. **Fine Grounding**: The rendered images are fed into a TaskMoE-enhanced MVT to predict the final robot action (position, rotation, gripper state, and collision status).

### Key Designs

#### TaskMoE: Task-Aware Mixture-of-Experts Module

- **Function**: Dynamically selects different visual/action expert networks based on task semantics to avoid cross-task feature conflicts.
- **Mechanism**: (1) Cross-attention fuses instruction and scene visual information; a FiLM layer then incorporates the Task ID to generate context-aware routing features. (2) The number of gates $N_G$ is decoupled from the total number of tasks $N_J$ ($N_G < N_J$), allowing semantically similar tasks to share a gate while being routed to different experts.
- **Design Motivation**: Conventional MoE methods route based solely on task ID, ignoring intra-task visual variation across scenes. The decoupled gating allows semantically similar tasks (e.g., opening different drawers) to share routing paths while preserving independent channels for dissimilar tasks, improving generalization to unseen tasks.

#### MVEP: Multi-View Exploration Policy

- **Function**: Receives the global point cloud and its RGB features, and outputs $K$ optimal virtual camera poses.
- **Mechanism**: A look-at model parameterizes camera poses as 5-dimensional spherical coordinate vectors $(\theta, \phi, r, \theta_{up}, \phi_{up})$. An MLP predicts the mean and variance of a Gaussian distribution; camera poses are sampled via the reparameterization trick, with sigmoid constraints ensuring valid spherical coordinates.
- **Design Motivation**: Spherical coordinate parameterization naturally fits the semantics of "observing around a target." Probabilistic sampling introduces exploration while maintaining end-to-end differentiability.

### Loss & Training

Training proceeds in three stages:

**Stage 1 — Fixed-Viewpoint Pretraining**: A base model is trained using three default viewpoints (front, left, top).

$$\mathcal{L}_{s1} = \mathcal{L}_{hc} + \mathcal{L}_{hf} + \mathcal{L}_{rot} + \mathcal{L}_{gri} + \mathcal{L}_{col}$$

This includes coarse/fine heatmap cross-entropy losses, rotation loss, gripper state loss, and collision indicator loss.

**Stage 2 — PPO Optimization of MVEP**: All other components are frozen; only MVEP is trained. Three reward terms are designed:

- $r_0 = \mathcal{L}_{ref} - \mathcal{L}_{TVVE}$ (task loss improvement reward, using the fixed-viewpoint model as a lower bound)
- $r_1 =$ negative mean entropy of the heatmap (encouraging confident predictions)
- $r_2 =$ mean cosine distance between viewpoints (encouraging viewpoint diversity)

The total reward is adaptively normalized (Welford's algorithm) and clipped to $[-10, 10]$: $r = \sum_{i=0}^{2} w_i \cdot \mathcal{N}(r_i)$

**Stage 3 — Joint Fine-tuning**: MVEP is frozen; remaining modules are fine-tuned to coordinate perception and action.

## Key Experimental Results

### Main Results

**Table 1: RLBench Multi-View Setting (18 Tasks)**

| Method | Avg. SR (%) ↑ | Avg. Rank ↓ | Insert Peg | Sort Shape | Slide Block | Close Jar |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| PerAct | 49.4 | 7.06 | 5.6 | 16.8 | 74.0 | 55.2 |
| RVT | 62.9 | 5.28 | 11.2 | 36.0 | 81.6 | 52.0 |
| 3D Diffuser Actor | 81.3 | 3.0 | 65.6 | 44.0 | 97.6 | 96.0 |
| RVT2 | 81.4 | 2.89 | 40.0 | 35.0 | 92.0 | 100.0 |
| ARP | 81.6 | 2.83 | 53.2 | 35.2 | 98.4 | 97.6 |
| **TVVE (Ours)** | **86.6** | **2.17** | **98.0** | **62.0** | **100.0** | **100.0** |

TVVE outperforms the previous SOTA (ARP, 81.6%) by approximately **5 percentage points**, with a gain of +32.4% on the Insert Peg task (65.6→98.0).

**Table 2: RLBench-OG Robustness Evaluation (Occlusion + Generalization)**

| Method | Avg. SR (%) | Occlusion 1 | Occlusion 2 | Light | Table Texture | Camera Pose |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Diffusion Policy | 23.8 | 27.4 | 23.4 | 22.9 | 22.6 | 22.2 |
| ARP | 63.7 | 73.0 | 52.6 | 59.8 | 61.3 | 69.7 |
| RVT2 | 64.5 | 72.8 | 46.9 | 60.8 | 64.0 | 74.0 |
| **TVVE (Ours)** | **67.0** | **75.0** | **58.0** | **63.7** | **66.8** | 73.2 |

On the Occlusion 2 setting, TVVE improves over RVT2 by **+11.1%** (46.9→58.0), demonstrating the effectiveness of dynamic viewpoints against occlusion.

### Ablation Study

**Component Ablation (18 Tasks, Table 5)**:

| Configuration | Avg. SR (%) |
|------|:---:|
| TVVE (TaskMoE + MVEP) | **86.6** |
| w/o TaskMoE | 85.6 (−1.0) |
| Fixed Viewpoints (w/o MVEP) | 83.3 (−3.3) |
| Random Viewpoints | 8.9 (−77.7) |

- Random viewpoints cause a catastrophic drop to 8.9%, conclusively demonstrating that MVEP learns a meaningful viewpoint selection strategy.
- Removing dynamic viewpoints results in a 3.3% drop, confirming the significant benefit of dynamic view exploration.

**TaskMoE Generalization (Table 6)**: With TaskMoE, the average success rate on 5 seen tasks is 80.8% (vs. 72.0% without TaskMoE); on the unseen task Open Drawer, success rate is 72% vs. 60%, indicating that TaskMoE improves both in-distribution performance and out-of-distribution generalization.

**Ablation on Viewpoint Count $K$ and Radial Constraint $r$ (Table 7)**: Increasing $K$ from 2→3→4 raises success rate from 27.2%→49.6%→55.2%; a tighter distance constraint (0.90~1.04 m) further improves success rate from 49.6% to 56.0%. $K=3$ is selected as the final trade-off between accuracy and computational cost.

### Key Findings

1. **Dynamic viewpoints >> fixed viewpoints >> random viewpoints**: MVEP's viewpoint selection is not incidentally beneficial — the policy learns a task-relevant information maximization strategy.
2. **TaskMoE is critical for generalization**: The decoupled gating mechanism enables the model to automatically discover latent task clusters and facilitates cross-task transfer.
3. **Real-robot experiments validate sim-to-real transfer**: TVVE (88%) vs. DP (68%) on Dobot; TVVE (78%) vs. ARP (72%) on Franka.

## Highlights & Insights

- **Pseudo-environment interaction** substantially reduces the RL training cost of MVEP — no real-environment interaction is required; reward signals are generated solely from offline data and the reference model.
- The **three-stage training strategy** is elegantly designed: first learn basic capabilities with fixed viewpoints → then learn the viewpoint exploration policy → finally coordinate perception and action, avoiding the instability of joint training.
- The **decoupled gating design** in TaskMoE ($N_G < N_J$) balances parameter sharing with task isolation, offering clear engineering value in terms of system scalability.

## Limitations & Future Work

- Multi-view re-rendering introduces inference latency, limiting real-time applicability.
- The method depends on accurate global point cloud reconstruction and performs poorly on reflective or transparent objects.
- Ablation experiments are conducted on only 5 representative tasks, which is insufficient for comprehensive evaluation.
- Stage 2 PPO training requires 4 A800 GPUs, imposing high resource demands.
- No direct comparison with recent 3D diffusion policy methods (e.g., 3D Diffuser Actor) under out-of-distribution settings is provided.

## Related Work & Insights

- **RVT-2**: TVVE builds upon RVT-2's coarse grounding module, upgrading fixed multi-view rendering to dynamic view exploration.
- **ARP**: A baseline autoregressive action policy; TVVE enhances its action generation module with TaskMoE.
- **SDP**: Integrates MoE into diffusion policies for multi-task learning, but without task-aware routing design.
- **Insights**: The pseudo-environment interaction paradigm can be generalized to other settings where RL optimization is needed but environment interaction is costly (e.g., active perception, navigation).

## Rating

⭐⭐⭐⭐ The method is thoroughly designed (MVEP + TaskMoE + three-stage training), with experiments spanning simulation, OOD, and real-robot settings. Significant improvements are achieved across 18 RLBench tasks; however, inference efficiency and point cloud dependency remain practical deployment bottlenecks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](gecosrt_geometryaware_continual_adaptation_for_rob.md)
- [\[NeurIPS 2025\] Learning Spatial-Aware Manipulation Ordering](../../NeurIPS2025/robotics/learning_spatial-aware_manipulation_ordering.md)
- [\[CVPR 2026\] Language-Grounded Decoupled Action Representation for Robotic Manipulation (LaDA)](lada_robotic_manipulation.md)
- [\[CVPR 2026\] CycleManip: Enabling Cyclic Task Manipulation via Effective Historical Perception and Understanding](cyclemanip_enabling_cyclic_task_manipulation_via_effective_historical_percepti.md)

</div>

<!-- RELATED:END -->
