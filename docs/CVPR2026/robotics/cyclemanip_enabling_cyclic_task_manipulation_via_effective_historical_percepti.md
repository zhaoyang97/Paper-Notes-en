---
title: >-
  [Paper Note] CycleManip: Enabling Cyclic Task Manipulation via Effective Historical Perception and Understanding
description: >-
  [CVPR 2026][Robotics][cyclic manipulation] CycleManip is the first work to systematically address cyclic robotic manipulation tasks (e.g., shaking a bottle N times). It enhances historical perception via a cost-aware history sampling strategy and improves historical understanding through multi-task learning auxiliary objectives, enabling controllable cycle-count manipulation in an end-to-end imitation learning framework.
tags:
  - CVPR 2026
  - Robotics
  - cyclic manipulation
  - robot manipulation
  - imitation learning
  - historical perception
  - multi-task learning
date: 2026-05-08
content_hash: d80aa0da5a06228b
---

# CycleManip: Enabling Cyclic Task Manipulation via Effective Historical Perception and Understanding

**Conference**: CVPR 2026
**arXiv**: [2512.01022](https://arxiv.org/abs/2512.01022)
**Code**: [https://isee-laboratory.github.io/CycleManip/](https://isee-laboratory.github.io/CycleManip/)
**Area**: Robotics
**Keywords**: cyclic manipulation, robot manipulation, imitation learning, historical perception, multi-task learning

## TL;DR
CycleManip is the first work to systematically address cyclic robotic manipulation tasks (e.g., shaking a bottle N times). It enhances historical perception via a cost-aware history sampling strategy and improves historical understanding through multi-task learning auxiliary objectives, enabling controllable cycle-count manipulation in an end-to-end imitation learning framework.

## Background & Motivation
1. **State of the Field**: Imitation learning and VLA models have demonstrated strong performance on sequential manipulation tasks, yet research on cyclic tasks—requiring repeated actions and accurate termination—remains nearly absent.
2. **Limitations of Prior Work**: (i) Policies with short observation windows cannot distinguish different phases of a cycle, as visual observations are nearly identical after each repetition; (ii) no benchmark exists with sufficient data and automated evaluation tools for cyclic tasks.
3. **Root Cause**: Cyclic tasks are non-Markovian processes where correct decisions depend not only on the current observation but also on accumulated progress. However, extending the observation horizon incurs substantial computational overhead.
4. **Paper Goals**: Design an end-to-end imitation learning framework that enables robots to execute cyclic actions and terminate at the correct moment.
5. **Starting Point**: Decompose observations into high-cost (visual) and low-cost (proprioceptive) modalities and sample them differently; leverage multi-task learning to promote understanding of cyclic phases.
6. **Core Idea**: Cost-aware sampling (sparse visual + dense proprioceptive) combined with a progress prediction auxiliary task yields a cycle-aware policy.

## Method

### Overall Architecture
Given a user instruction and robot observations, a cost-aware sampling strategy applies different sampling rates to high- and low-cost observations. All observations and language instructions are encoded as diffusion conditions for action prediction. Concurrently, observation features are used to predict task progress (auxiliary task), enhancing the model's understanding of cyclic phases.

### Key Designs

1. **Cost-Aware History Sampling Strategy**:
    - **Function**: Extends the observation horizon at low computational cost.
    - **Mechanism**: Observations are divided into low-cost (end-effector pose differences) and high-cost (point cloud/RGB) modalities. Low-cost observations are sampled densely in full (nearly free computationally). High-cost observations use heuristic frame sampling—half with binary sampling to cover the full history, half with exponential sampling ($t-2^k$) to preserve recent detail.
    - **Design Motivation**: The cyclic characteristics of the end-effector are more pronounced and easier to model than joint positions; using pose differences rather than absolute positions avoids positional bias.

2. **Multi-Task Learning Progress Prediction**:
    - **Function**: Enables the model to implicitly learn discriminative features for cyclic phases.
    - **Mechanism**: An auxiliary task is introduced to predict the current progress $b_t$ (current frame index / maximum frame index, discretized into a 10-class classification problem). Features are fused via a multi-layer MLP, and progress is predicted by a single-layer MLP. Total loss = MSE action loss + CE progress loss.
    - **Design Motivation**: The supervision signal in pure imitation learning is identical across every cycle (continue executing), preventing the model from distinguishing different phases. Progress prediction forces the model to learn discriminative representations.

3. **CycleManip Benchmark**:
    - **Function**: Provides an evaluation platform for cyclic manipulation tasks.
    - **Mechanism**: Eight cyclic manipulation tasks (hammering nails, shaking bottles, slicing carrots, etc.) are built on RoboTwin 2.0, with 200 demonstration trajectories per task and cycle counts ranging from 1 to 8. Automated evaluation deems a trial successful only when the manipulation succeeds and the cycle count is correct.
    - **Design Motivation**: The absence of a standardized benchmark has hindered progress in cyclic task research.

### Loss & Training
$$\mathcal{L} = \alpha \cdot \text{MSE}(a_t, a_t^*) + \beta \cdot \text{CE}(b_t, b_t^*)$$
Training is conducted within the diffusion policy framework.

## Key Experimental Results

### Main Results

| Task | CycleManip Success Rate | Baseline Success Rate | Cycle Accuracy |
|------|------------------------|-----------------------|----------------|
| Hammering Nails | High | Low | High |
| Shaking Bottles | High | Very Low | High |
| Slicing Carrots | Medium-High | Low | Medium-High |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Full CycleManip | Best | Complete framework |
| w/o Progress Prediction | Significant drop | Auxiliary task is critical |
| w/o Dense Proprioceptive Sampling | Drop | Historical perception matters |
| Visual Extension Only | High cost, limited gain | Validates necessity of cost-aware sampling |

### Key Findings
- The method generalizes well to general manipulation tasks beyond cyclic ones.
- It can be applied as a plug-and-play module to VLA models (e.g., Pi0).
- Cross-platform validation (dual-arm gripper, dexterous hand, humanoid robot) demonstrates generality.
- Dense proprioceptive sampling incurs negligible computational overhead and represents the most cost-effective approach to history modeling.

## Highlights & Insights
- **First systematic definition of cyclic manipulation tasks**, filling a gap in robot manipulation research.
- The **cost-aware sampling** design is intuitive: free proprioceptive signals replace expensive visual history to capture cyclic patterns.
- The progress prediction auxiliary task is a simple yet effective technique.

## Limitations & Future Work
- Discretizing progress prediction into 10 classes may lack sufficient granularity.
- The current framework only supports a fixed cycle count; dynamic termination conditions such as "until thoroughly mixed" remain unexplored.
- Complex physical interactions (e.g., friction from different materials) may require finer-grained force feedback.

## Related Work & Insights
- **vs. Diffusion Policy**: Standard diffusion policies rely on short observation windows and cannot handle cyclic tasks. CycleManip extends their capability through historical perception and understanding.
- **vs. VLA Models**: VLA models also depend on short-term observations; CycleManip's plug-and-play design can directly augment them.

## Rating
- Novelty: ⭐⭐⭐⭐ First formulation of the cyclic manipulation problem; practical but not overly complex methodology
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 tasks + 3 platforms + simulation + real-world + VLA integration
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and well-designed experiments
- Value: ⭐⭐⭐⭐ Fills an important gap with practical relevance for real-world robot deployment

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_active_perception_manipulation_vla_roboti.md)
- [\[CVPR 2026\] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [\[CVPR 2026\] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation](profocus_proactive_perception_and_focused_reasoning_in_vision-and-language_navig.md)
- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](../../NeurIPS2025/robotics/mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](geco-srt_geometry-aware_continual_adaptation_for_robotic_cross-task_sim-to-real_.md)

<!-- RELATED:END -->
