---
title: >-
  [Paper Note] GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning
description: >-
  [ICCV 2025][Image Generation][Reinforcement Learning] This paper proposes GenFlowRL, which integrates generative object-centric optical flow with reinforcement learning by shaping rewards using a δ-flow representation extracted from a flow generation model trained on cross-embodiment datasets. The approach enables robust and generalizable robot manipulation policy learning, significantly outperforming flow-based imitation learning and video-guided RL methods across 10 manipul…
tags:
  - "ICCV 2025"
  - "Image Generation"
  - "Reinforcement Learning"
  - "Object-Centric Flow"
  - "Reward Shaping"
  - "Robot Manipulation"
  - "Cross-Embodiment"
  - "Video Generative Models"
date: 2026-05-08
content_hash: 0c6de1e933de2f43
---

# GenFlowRL: Shaping Rewards with Generative Object-Centric Flow in Visual Reinforcement Learning

**Conference**: ICCV 2025
**arXiv**: [2508.11049](https://arxiv.org/abs/2508.11049)  
**Code**: [Project Page](https://colinyu1.github.io/genflowrl)  
**Area**: Image Generation
**Keywords**: Reinforcement Learning, Object-Centric Flow, Reward Shaping, Robot Manipulation, Cross-Embodiment, Video Generative Models

## TL;DR

This paper proposes GenFlowRL, which integrates generative object-centric optical flow with reinforcement learning by shaping rewards using a δ-flow representation extracted from a flow generation model trained on cross-embodiment datasets. The approach enables robust and generalizable robot manipulation policy learning, significantly outperforming flow-based imitation learning and video-guided RL methods across 10 manipulation tasks.

## Background & Motivation

Video generative foundation models have demonstrated great potential in robot learning—deriving actions from generated future frames via inverse dynamics. However, two fundamental problems persist in existing approaches:

**Open-loop policies lack robustness**: Policies that rely entirely on generated future frames without environment interaction perform poorly on fine-grained manipulation tasks.

**Video generation quality bottleneck**: Large-scale robot data collection is costly, and generated videos suffer from significant artifacts, limiting their effectiveness as RL reward signals.

While reinforcement learning provides robustness through environment interaction, directly applying video generative models to RL reward shaping is challenging: video is a high-dimensional signal from which fine-grained manipulation features are difficult to extract.

**Core observation**: Object-Centric Flow is a low-dimensional, cross-embodiment representation that retains critical manipulation features while abstracting away irrelevant details. Compared to raw video frames, end-effector keypoints, and other representations, optical flow offers comprehensive advantages in RL compatibility and geometric complexity modeling (see Table 1), with particular support for both deformable and articulated objects.

## Method

### Overall Architecture

GenFlowRL consists of three core stages:

1. **Task-conditioned object-centric flow generation**: Training a flow generation model on cross-embodiment datasets.
2. **Hybrid reward model**: Combining dense δ-flow matching rewards with sparse state-aware rewards.
3. **Flow-conditioned policy learning**: Training generalizable policies using the hybrid reward model.

### Flow Generation Process

Flow generation proceeds in three steps:

- **Flow dataset construction**: Grounding-DINO detects object bounding boxes in initial frames; CoTracker tracks 128 uniformly sampled keypoints to produce flow representations $\mathcal{F}_0 \in \mathbb{R}^{3 \times T \times H \times W}$.
- **Generative model adaptation**: AnimateDiff is fine-tuned in two stages—first fine-tuning the decoder to adapt to flow data, then injecting LoRA into motion modules to learn temporal dynamics.
- **Post-processing**: A motion filter removes static keypoints; a SAM semantic filter removes non-object keypoints.

### δ-flow Representation (Key Contribution)

The raw keypoint flow is compressed into three statistics:

$$\bar{\mathbf{P}}^t = \frac{1}{N}\sum_{i=1}^{N}\mathbf{P}_i^t, \quad \boldsymbol{\delta}_{tr}^t = \bar{\mathbf{P}}^t - \bar{\mathbf{P}}^1$$

$$\boldsymbol{\delta}_{rot}^t = \frac{1}{N}\sum_{i=1}^{N}\left[(\mathbf{P}_i^t - \bar{\mathbf{P}}^t) \times (\mathbf{P}_i^1 - \bar{\mathbf{P}}^1)\right]$$

The δ-flow is essentially a Monte Carlo estimator that compresses redundant multi-keypoint trajectories into statistical features of displacement and rotation, effectively reducing the influence of unreliable keypoints.

### Hybrid Reward Model

**Dense flow matching reward**: The generated and observed δ-flows are modeled as Gaussian distributions; their alignment is measured via KL divergence, simplified to mean matching:

$$R_{\delta}^t = 1 - \text{clip}\left(\frac{(\mathcal{T}_R^t - \mathcal{T}_G^t)^2}{C}, 0, 1\right)$$

**Overall reward design** (phased and task-agnostic):

$$R^t = \begin{cases} \alpha \cdot (1 - \tanh(\tau \cdot d_{grip})), & \text{approach phase} \\ \alpha, & \text{subgoal completion} \\ \alpha + \beta \cdot R_{\delta}^t, & \text{post-subgoal} \\ 1.0, & \text{task completion} \end{cases}$$

where $\alpha=0.25, \beta=0.75, \tau=10$.

### Policy Design

The policy takes six inputs: current robot state, current keypoint centroid, current observed δ-flow, $k$-step look-ahead generated centroid, $k$-step look-ahead generated δ-flow, and the 3D centroid position of the initial frame. The output is a 6D pose displacement converted to joint commands via inverse kinematics. Optimization is performed using the DrQv2 algorithm.

### Loss & Training

The policy is optimized by maximizing the hybrid reward using DrQv2's experience replay strategy:
- Learning rate $10^{-4}$, discount factor $\gamma=0.99$
- Exploration standard deviation linearly annealed from 1.0 to 0.1

## Key Experimental Results

### Main Results: Flow-based RL vs. Flow-based IL (Table 2)

| Method | PickNP. | Pour | Open | Fold | Pivot |
|--------|---------|------|------|------|-------|
| Heuristic | 70 | 50 | 30 | 0 | 0 |
| Im2Flow2Act | 100 | 95 | 95 | 90 | 60 |
| **GenFlowRL** | **100** | **100** | **100** | **95** | **90** |

The advantage is more pronounced under language conditioning: Fold improves from 35→80 (+45), and Pivot from 45→85 (+40).

### Comparison with Video Reward RL (Fig. 4)

On the 5 most challenging MetaWorld tasks:
- GenFlowRL significantly outperforms VIPER and Diffusion Reward on difficult tasks such as Assembly, Lever Pull, and Stick Pull.
- Faster convergence and higher success rates are observed.
- Pure Sparse Reward (PSR) and RND perform adequately on simple tasks but struggle on complex ones.

### Ablation Study (Fig. 6)

| Variant | Key Finding |
|---------|-------------|
| MLP replacing δ-flow | Performance degrades; δ-flow better captures spatiotemporal dynamics |
| Removing 3D initial centroid | Performance degrades; 3D spatial information benefits 6D action learning |
| 64 keypoints vs. 128 | Comparable performance; δ-flow is insensitive to keypoint count |

### Noise Robustness Analysis (Table 4)

| Noise Condition | PickNP. | Pour | Open | Fold | Pivot |
|-----------------|---------|------|------|------|-------|
| No noise | 95 | 95 | 95 | 80 | 85 |
| Large Gaussian (4×) | 95 | 90 | 90 | 75 | 80 |
| Large drift (2×) | 85 | 75 | 85 | 65 | 75 |

High performance is maintained even under large noise conditions, demonstrating the robustness of the δ-flow representation.

### Real-Robot Validation

Cross-embodiment flow matching between human and robot is validated on XArm7 across 4 tasks. The reward signal exhibits a monotonically increasing trend, indicating deployment feasibility.

## Highlights & Insights

1. **Deep insight into representation selection**: A systematic analysis of various manipulation-centric representations demonstrates that object-centric optical flow achieves an overall optimum in terms of low dimensionality, cross-embodiment transferability, reward compatibility, and support for geometric complexity.
2. **Monte Carlo nature of δ-flow**: Compressing multi-keypoint trajectories into statistical features is inherently a Monte Carlo estimation, which naturally confers noise robustness.
3. **Train-inference consistency**: Both training and inference use generated flows, avoiding the distribution shift introduced by relying on expert flows.
4. **Elegant hybrid reward design**: Sparse state-aware rewards provide task information, while dense δ-flow rewards provide motion priors—the two are complementary.

## Limitations & Future Work

1. Only 2D optical flow is used, which may be limiting for tasks involving out-of-plane rotations (e.g., unscrewing a bottle cap).
2. The cross-embodiment dataset scale (12K trajectories) is relatively small.
3. Real-world experiments only validate reward matching and do not include full end-to-end deployment.

## Related Work & Insights

- Unlike HuDor (Guzey et al., 2024), this work leverages generated flows (rather than a single expert flow) for dense reward shaping with cross-embodiment motion priors.
- The idea of combining flow generation with RL can be extended to broader embodied intelligence tasks (e.g., navigation, tool use).
- The δ-flow representation may serve as a general manipulation prior applicable to other robot learning paradigms.

## Rating ⭐⭐⭐⭐

Novelty ★★★★☆: The δ-flow representation and hybrid reward design are novel and theoretically grounded.
Experimental Thoroughness ★★★★☆: Broad coverage across 10 tasks with thorough ablations, though real-world evaluation is limited.
Writing Quality ★★★★☆: Clear structure with systematic comparison tables.
Value ★★★☆☆: Requires training both a flow generation model and an RL policy, resulting in relatively high deployment complexity.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CTRL-O: Language-Controllable Object-Centric Visual Representation Learning](../../CVPR2025/image_generation/ctrl-o_language-controllable_object-centric_visual_representation_learning.md)
- [\[CVPR 2025\] GLASS: Guided Latent Slot Diffusion for Object-Centric Learning](../../CVPR2025/image_generation/glass_guided_latent_slot_diffusion_for_object-centric_learning.md)
- [\[NeurIPS 2025\] Composite Flow Matching for Reinforcement Learning with Shifted-Dynamics Data](../../NeurIPS2025/image_generation/composite_flow_matching_for_reinforcement_learning_with_shifted-dynamics_data.md)
- [\[ICCV 2025\] GenHancer: Imperfect Generative Models are Secretly Strong Vision-Centric Enhancers](genhancer_imperfect_generative_models_are_secretly_strong_vision-centric_enhance.md)
- [\[ICCV 2025\] Deeply Supervised Flow-Based Generative Models](deeply_supervised_flow-based_generative_models.md)

</div>

<!-- RELATED:END -->
