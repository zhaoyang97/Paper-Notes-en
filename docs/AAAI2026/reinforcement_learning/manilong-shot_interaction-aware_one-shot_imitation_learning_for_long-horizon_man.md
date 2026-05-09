---
title: >-
  [Paper Note] ManiLong-Shot: Interaction-Aware One-Shot Imitation Learning for Long-Horizon Manipulation
description: >-
  [AAAI 2026][Reinforcement Learning][One-shot imitation learning] This paper proposes ManiLong-Shot, a framework comprising three modules—interaction-aware task decomposition, invariant region prediction, and region matching—that generalizes to 20 unseen long-horizon manipulation tasks after training on only 10 short-horizon tasks, achieving a one-shot imitation success rate of 30.2%, a relative improvement of 22.8% over the prior state of the art.
tags:
  - AAAI 2026
  - Reinforcement Learning
  - One-shot imitation learning
  - long-horizon manipulation
  - interaction-awareness
  - invariant regions
  - task decomposition
date: 2026-05-08
content_hash: 775da845e0f738d2
---

# ManiLong-Shot: Interaction-Aware One-Shot Imitation Learning for Long-Horizon Manipulation

**Conference**: AAAI 2026
**arXiv**: [2512.16302](https://arxiv.org/abs/2512.16302)
**Code**: [Website](https://sites.google.com/view/manilong-shot)
**Area**: Reinforcement Learning
**Keywords**: One-shot imitation learning, long-horizon manipulation, interaction-awareness, invariant regions, task decomposition

## TL;DR

This paper proposes ManiLong-Shot, a framework comprising three modules—interaction-aware task decomposition, invariant region prediction, and region matching—that generalizes to 20 unseen long-horizon manipulation tasks after training on only 10 short-horizon tasks, achieving a one-shot imitation success rate of 30.2%, a relative improvement of 22.8% over the prior state of the art.

## Background & Motivation

### Problem Definition

One-Shot Imitation Learning (OSIL): learning new skills from a single demonstration without additional training. Robots must rapidly learn and execute diverse long-horizon manipulation tasks in everyday settings (e.g., "set the table," "tidy the kitchen"), which involve sequential interactions with multiple objects.

### Limitations of Prior Work

**Short-horizon restriction**: Most OSIL methods are designed for short-horizon skills only (e.g., IMOP, zhang2024oneshot) and cannot scale to multi-step long-horizon tasks.

**Task-variant dependency**: Some methods require new tasks to be minor variants of training tasks, or rely on known 3D object models.

**Predefined primitive libraries**: wu2024one depends on a predefined primitive library to compose long-horizon manipulation, limiting flexibility.

### Core Motivation — Inspiration from Human Learning

When faced with an unseen task (e.g., arranging tableware), humans naturally decompose it into short-horizon primitives and infer the key interaction regions:
1. Pick up the plate (at the edge) → 2. Place the plate (target location on the table) → 3. Pick up the fork (at the handle)

Each primitive is delimited by a physical interaction event (contact/release), and imitation is achieved by replicating actions in these regions. The core question is: can a robot infer subtask boundaries and key interaction regions from unannotated demonstrations?

## Method

### Overall Architecture

Three core modules are organized around physical interaction events:
1. **Interaction-aware task decomposition**: decomposes a demonstration into a sequence of primitives, each comprising pre-contact, grasping, and post-contact phases.
2. **Interaction-aware region prediction network**: predicts functionally invariant interaction regions for each primitive.
3. **Interaction-aware region matching network**: aligns predicted regions with the current observation to compute the target end-effector pose.

Inference pipeline: decompose demonstration → predict invariant regions → match current scene → pose regression → motion planning → execution → iterate until task completion.

### Key Designs

#### 1. **Interaction-Aware Task Decomposition**

The demonstration trajectory is organized into a primitive sequence based on physical interaction phases:

- **Pre-contact phase**: the gripper opens and approaches the object, ending when joint velocity drops to zero (alignment in place).
- **Grasping phase**: the gripper transitions from open to closed, grasping the object.
- **Post-contact phase**: after a successful grasp, the gripper transitions from closed to open, placing the object or interacting with another object.

Two interchangeable decomposition strategies are provided:
- **Rule-based**: infers phase boundaries by analyzing joint velocity and gripper state changes; stable and reliable.
- **VLM-based**: employs GPT-4o with structured trajectory representations to automatically identify interaction phases; semantically aware.

Short-horizon tasks contain one interaction cycle (3 phases); long-horizon tasks repeat multiple cycles.

**Design Motivation**: Physical interaction events serve as natural subtask boundaries that are more robust than semantic segmentation and are transferable across tasks and environments.

#### 2. **Interaction-Aware Region Prediction Network**

Identifies functionally and semantically invariant interaction regions for each interaction phase. An invariant region is defined as a 3D geometric subset that maintains $SE(3)$-equivariant structure across states sharing the same optimal policy.

Architecture:
- **Input**: RGB-D dense point clouds of consecutive state pairs $\{s_i, s_{i+1}\}$
- **Backbone**: Point Cloud Transformer V3 (PTV3) with progressive downsampling, cross-scene cross-attention, and within-scene self-attention
- **Output**: interaction probability distribution → activated region $\mathcal{I}(s_i)$

Key distinctions:
- **Pre-contact + grasping phases**: jointly trained (functionally similar); predict grasp surface regions.
- **Post-contact phase**: an additional **Positioning Network** is activated, using attention mechanisms to align the grasped object with the target region.

Training supervision: instance segmentation masks from simulation are used as ground truth. Training is performed exclusively on short-horizon tasks $\mathcal{T}^{\text{sh}}$.

**Design Motivation**: The invariant region concept enables knowledge transfer from short-horizon to long-horizon tasks—the same "grasp the cup rim" region remains consistent across different tasks.

#### 3. **Interaction-Aware Region Matching Network**

Aligns the predicted invariant regions from the demonstration with the current execution state to compute the target pose.

Pipeline:
1. **State routing network**: selects the frame in the demonstration trajectory most similar to the current state.
2. **Feature fusion**: cropped invariant region point cloud $\mathcal{I}(s_i)$ and current scene point cloud are jointly downsampled.
3. **Dual-stage attention**: a cross-self-cross attention module enhances spatial and geometric feature alignment.
4. **Correspondence matrix computation**: a dual softmax matching algorithm computes correspondence matrix $\mathbf{C}$.

Pose regression based on the correspondence matrix:

$$\mathbf{T}_j = \arg\min_{\mathbf{T} \in SE(3)} \|\mathbf{T} \mathbf{T}_i^{-1} P_{\mathcal{I}(s_i)} \mathbf{C} - P_{s_j}\|$$

Motion planning uses the RRT-Connect algorithm to find collision-free trajectories.

**Design Motivation**: Pose estimation based on correspondences rather than direct action prediction is more robust to object pose variations and scene layout changes.

### Loss & Training

- Training uses only 10 short-horizon tasks (100 demonstration trajectories per task).
- Observations from front, side, and wrist RGB-D cameras are used.
- Backbone: PTV3.
- Supervised learning with ground-truth instance masks and correspondence matrices.

## Key Experimental Results

### Experimental Setup

- **Simulation benchmark**: RLBench-Oneshot (30 tasks: 10 SH + 20 LH)
    - LH tasks at three difficulty levels: Level 1 (13 tasks, 6 interactions), Level 2 (4 tasks, 9 interactions), Level 3 (3 tasks, 12 interactions)
- **Real robot**: UFactory xArm7 + RealSense D435/D415, 3 LH tasks
- **Baselines**: ARP, 3DDA, RVT2 (SOTA IL models), IMOP (OSIL SOTA)
- 25 trials per task (5 random seeds); mean ± standard deviation reported

### Main Results — Short-Horizon Tasks

| Model | Avg. Success Rate (%) | Avg. Rank | # Best Tasks |
|---|---|---|---|
| IMOP | 65.24 | 3.9 | 1 |
| RVT2 | 79.2 | 3.1 | 0 |
| 3DDA | 85.0 | 2.9 | 2 |
| ARP | 86.6 | 2.7 | 3 |
| **ManiLong-Shot** | **90.4** (+3.8%) | **1.9** | **6** |

### Main Results — Unseen Long-Horizon Tasks (OSIL)

| Model | Avg. Success Rate (%) | Avg. Rank |
|---|---|---|
| RVT2+FT | 4.1 | 3.7 |
| 3DDA+FT | 4.5 | 3.3 |
| ARP+FT | 4.7 | 3.3 |
| IMOP | 7.4 | 2.6 |
| **ManiLong-Shot** | **30.2** (+22.8%) | **1.0** |

Representative task comparisons:

| Task | IMOP | ManiLong-Shot |
|---|---|---|
| Empty Container | 4.0% | **28.0%** |
| Empty Dishwasher | 1.3% | **42.7%** |
| Put Item in Drawer | 40.0% | **65.3%** |
| Take Item out Drawer | 38.7% | **76.0%** |
| Set Table | 2.7% | **17.3%** |
| Stack Blocks | 1.3% | **8.0%** |

### Ablation Study

| Configuration | Level 1 | Level 2 | Level 3 | Notes |
|---|---|---|---|---|
| ManiLong-Shot (Rule) | Highest | Highest | Highest | Rule-based decomposition |
| ManiLong-Shot (VLM) | Below Rule | Below Rule | Below Rule | VLM reasoning is unstable |
| w/o Positioning | Degraded | Notably degraded | Largest degradation | Inaccurate placement in post-contact phase |

### Real Robot Experiments

| Task | IMOP | ManiLong-Shot |
|---|---|---|
| Stack Blocks | 60% | **80%** |
| Stack Cups | 20% | **60%** |
| Place Cups | 20% | **40%** |
| Average | 33.3% | **60.0%** (+26.7%) |

### Key Findings

1. **Consistent superiority on short-horizon tasks**: 90.4% average success rate; best on 6 out of 10 training tasks.
2. **Substantial advantage on long-horizon tasks**: 30.2% vs. 7.4% (IMOP), an absolute gain of 22.8%; fine-tuned SOTA models achieve only 4–5%.
3. **Rule-based decomposition outperforms VLM-based**: VLM reasoning is unstable, and the gap widens with task complexity; the VLM variant still significantly outperforms all baselines.
4. **Positioning network is critical**: its removal causes inaccurate placement in the post-contact phase, disrupting the subsequent subtask execution chain.
5. **Effective sim-to-real transfer**: 60% average success rate, a 26.7% improvement over IMOP.

## Highlights & Insights

1. **Physical interaction as universal decomposition primitives**: the pre-contact/grasping/post-contact three-phase structure is a natural organization of grasping-based manipulation, more robust than semantic segmentation.
2. **Generalizing to long-horizon tasks trained only on short-horizon ones**: 10 short tasks → 20 unseen long tasks, demonstrating the compositional generalization capacity of interaction primitives.
3. **Elegant abstraction via invariant regions**: the functional property that "the cup rim is suitable for grasping" is formalized as an $SE(3)$-equivariant invariant region.
4. **Flexibility of dual decomposition strategies**: the rule-based approach offers stability while the VLM-based approach can capture semantic patterns; either can be selected as needed.
5. **End-to-end one-shot generalization**: no task-specific fine-tuning is required, enabling truly zero-shot transfer.

## Limitations & Future Work

1. **Restricted to grasping operations**: the three-phase decomposition is grounded in physical contact and does not apply to non-grasping behaviors (e.g., wiping, pouring, and other continuous interactions).
2. **Parallel gripper assumption**: the framework is designed around parallel grippers; dexterous hand manipulation is not addressed.
3. **Tabletop environment constraint**: experiments are conducted in tabletop settings; more complex environments (mobile manipulation, multi-room scenarios) remain unvalidated.
4. **Absolute success rate remains low**: the 30.2% average on long-horizon tasks, while far exceeding baselines, still limits practical applicability.
5. **VLM reasoning instability**: the inconsistency of GPT-4o as a task decomposer is amplified in more complex tasks.

## Related Work & Insights

- **IMOP (zhang2024oneshot)**: the pioneering work introducing the invariant region concept; the direct foundation of this paper.
- **GravMAD/DECO (chen2024/2025)**: subtask decomposition methods based on physical interactions.
- **RLBench (james2020)**: robotic manipulation simulation benchmark; this paper constructs the RLBench-Oneshot subset.
- **PTV3**: a point cloud Transformer used as the geometric feature extraction backbone.
- Insight for long-horizon manipulation research: composing short-horizon primitives is more effective than directly learning long-horizon sequences.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of interaction-aware three-phase decomposition and invariant region prediction is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across a 30-task benchmark, ablations, VLM comparisons, and real robot experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clear figures, rigorous problem formulation, and detailed method description.
- Value: ⭐⭐⭐⭐ — Provides a practical framework for long-horizon one-shot imitation learning.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)
- [\[AAAI 2026\] Where and What Matters: Sensitivity-Aware Task Vectors for Many-Shot Multimodal In-Context Learning](where_and_what_matters_sensitivity-aware_task_vectors_for_many-shot_multimodal_i.md)
- [\[AAAI 2026\] Learning to Generate and Extract: A Multi-Agent Collaboration Framework for Zero-shot Document-level Event Arguments Extraction](learning_to_generate_and_extract_a_multi-agent_collaboration_framework_for_zero-.md)
- [\[AAAI 2026\] One-Step Generative Policies with Q-Learning: A Reformulation of MeanFlow](one-step_generative_policies_with_q-learning_a_reformulation_of_meanflow.md)
- [\[ICLR 2026\] Strict Subgoal Execution: Reliable Long-Horizon Planning in Hierarchical Reinforcement Learning](../../ICLR2026/reinforcement_learning/strict_subgoal_execution_reliable_long-horizon_planning_in_hierarchical_reinforc.md)

<!-- RELATED:END -->
