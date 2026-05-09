---
title: >-
  [Paper Note] Learning Spatial-Aware Manipulation Ordering
description: >-
  [NeurIPS 2025][Robotics][manipulation ordering] This paper proposes OrderMind, a unified framework that learns manipulation ordering of objects in cluttered scenes directly from RGB-D images via a Spatial Context Understanding encoder and a Temporal Priority Structuring module. Training annotations are generated through VLM distillation with spatial priors. OrderMind significantly outperforms VLM baselines in both simulation and real-world environments while supporting real-time inference (5.6 FPS, 21.3 FPS for the lightweight variant).
tags:
  - NeurIPS 2025
  - Robotics
  - manipulation ordering
  - spatial graph
  - cluttered environment
  - VLM distillation
  - real-time inference
date: 2026-05-08
content_hash: 043334aa11335548
---

# Learning Spatial-Aware Manipulation Ordering

**Conference**: NeurIPS 2025
**arXiv**: [2510.25138](https://arxiv.org/abs/2510.25138)
**Authors**: Yuxiang Yan, Zhiyuan Zhou, Xin Gao, Guanghao Li, Shenglin Li, Jiaqi Chen, Qunyan Pu, Jian Pu (Fudan University, Stanford University)
**Code**: Not released
**Area**: Robotic Manipulation / Cluttered Scene Grasping / Spatial Reasoning
**Keywords**: manipulation ordering, spatial graph, cluttered environment, VLM distillation, real-time inference

## TL;DR

This paper proposes OrderMind, a unified framework that learns manipulation ordering of objects in cluttered scenes directly from RGB-D images via a Spatial Context Understanding encoder and a Temporal Priority Structuring module. Training annotations are generated through VLM distillation with spatial priors. OrderMind significantly outperforms VLM baselines in both simulation and real-world environments while supporting real-time inference (5.6 FPS, 21.3 FPS for the lightweight variant).

## Background & Motivation

Robotic manipulation in cluttered environments is a fundamental challenge. When objects are densely stacked, mutually occluded, or physically constrained, the manipulation order directly affects task efficiency and scene stability—an incorrect order may cause collisions or structural collapse.

Existing approaches exhibit notable limitations:

**Heuristic pipelines** (e.g., manual optimization post-detection): poor generalizability, prone to failure when spatial relationships vary across scenes.

**Two-stage frameworks** (detection followed by VLM-based order inference): high inference latency (typically several seconds), incompatible with real-time deployment requirements.

**Direct VLM inference**: even when granted privileged information (ground-truth object poses), GPT-4o achieves only 71.4% success rate on hard scenes, and Gemini-2.5 only 78.5%.

The root cause is that existing methods either fail to explicitly model manipulation ordering or rely on high-latency VLM inference. This work is motivated by the goal of designing a **unified framework** that learns spatially-aware manipulation ordering directly from visual input, achieving both high accuracy and real-time performance.

## Method

### Problem Formulation

Manipulation ordering is defined as a mapping $f: \mathcal{I} \times SE(3) \to \mathcal{O}$, where:
- $\mathcal{I} \subset \mathbb{R}^{H \times W \times 4}$ is the RGB-D image space
- $SE(3)$ is the pose space of the robot end-effector
- The output $\mathcal{O} = \{\mathbf{O}, \Sigma\}$ consists of an object representation set and a manipulation sequence

The model assigns a continuous priority score $s_i \in \mathbb{R}$ to each object; the manipulation sequence $\Sigma$ is derived by sorting these scores. Using continuous scores rather than discrete ranks enables fine-grained differentiation of manipulation priority.

### Spatial Context Understanding Module (SCU)

In cluttered scenes, occlusion and inter-object support relationships make visual appearance alone insufficient for reasoning about manipulation order; spatial and physical relationships must be explicitly modeled.

**Object representation**: Each object is represented by the center of its 3D bounding box along with intrinsic attributes (semantic category, physical dimensions) and extrinsic attributes (pose in world coordinates).

**Spatial graph construction**: A local spatial graph is constructed using a k-nearest-neighbor (kNN) strategy. Object centers form a sparse point cloud, with nodes representing objects and edges encoding geometric proximity. For each center point $p_i$, messages are aggregated from its spatial neighbors $\mathcal{N}_k(p_i)$:

$$\text{Fusion}(f_i, f_j) = \mathcal{M}(\text{Linear}(\text{Concat}(f_i, f_j - f_i))), \quad \forall p_j \in \mathcal{N}_k(p_i)$$

PointNet-style max pooling $\mathcal{M}$ aggregates neighborhood features to produce compact object-level embeddings.

**Robot–object relationship modeling**: The relative transformation between each object pose and the current end-effector state is computed, providing important cues for manipulation reachability.

### Temporal Priority Structuring Module (TPS)

This module transitions from structure-aware representation to manipulation-oriented ordering.

1. **Global scene representation**: Object ordering tokens extracted by the image encoder are globally max-pooled into a high-level scene representation $G$, encoding occlusion patterns, stacking relationships, and spatial symmetries.
2. **Self-attention**: Object tokens $Q$ interact with $G$ via Self-Attention to model inter-object dependencies.
3. **Cross-attention**: Updated tokens $Q'$ query global context $G$ and visual features $F$ via Cross-Attention:

$$[Q', G] \leftarrow \text{Self-Attn}([Q, G]), \quad Q'' \leftarrow \text{Cross-Attn}(Q', [G, F], [G, F])$$

The output $Q''$ encodes spatially-aware object priorities, implicitly modeling manipulation precedence under geometric and physical constraints.

### Preference Order Alignment (Loss Function)

Bipartite matching between predicted and ground-truth objects is first performed via the **Hungarian algorithm**. Given the established correspondences, a pairwise comparison loss is used to learn continuous scores:

$$\mathcal{L}_{\text{order}} = \sum_{j=1}^{N} \sum_{k=1}^{N} w_{jk} \cdot \mathbb{1}_{\{o_j < o_k\}} \log(1 + \exp(\hat{s}_{\hat{\sigma}(k)} - \hat{s}_{\hat{\sigma}(j)}))$$

where $w_{jk} = \log(1 + |o_j - o_k|)$ is a log-weighting term that emphasizes ranking consistency for object pairs with large ground-truth rank differences. This pairwise ranking loss enables the model to infer priority through relative comparisons in score space, rather than directly predicting exact rank values.

### Spatial Prior-guided Ordering Labels (SPOL)

Training annotations are generated by a VLM (Qwen2.5-VL), with two spatial priors introduced to improve annotation quality:

1. **Independence prior**: Encourages prioritizing objects that are spatially separated from others on the horizontal plane. The minimum distance between projected object areas is computed; when $\min_{j \neq i} d(A_i, A_j) \geq \tau$, an object is considered spatially independent and safe to manipulate first.
2. **Local optimality prior**: Identifies objects not occluded from above. The vertical space $V_{\text{above}}$ above each object is defined; when no other object intersects this space, the object can be directly approached from above, and prioritizing it helps maintain scene stability.

These two priors serve as auxiliary signals guiding the VLM to generate ordering annotations consistent with physical and operational constraints.

## Experimental Setup

### Dataset: Manipulation Ordering Benchmark

- **Simulation environment**: PyBullet engine + YCB object set (5 object categories)
    - Easy: 24 objects; Medium: 36 objects; Hard: 60 objects
    - Training set: 161,722 RGB-D images; validation set: 1,500 images
- **Real-world environment**: Training set: 26,324 images; validation set: 6,581 images; objects divided into box and bag categories
- The robot uses a suction-cup end-effector; RGB-D images are captured at 1408×1024 resolution from a top-down camera

### Evaluation Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| Success Rate (SR) | Successful grasps / total attempts | ↑ |
| Remaining Count (RC) | Objects remaining outside workspace at task end | ↓ |
| Object Disturbance (OD) | Total displacement of surrounding objects per manipulation | ↓ |

## Experimental Results

### Main Results in Simulation

| Method | Privileged Info | Easy SR | Hard SR | Params | FPS |
|--------|----------------|---------|---------|--------|-----|
| GPT-4o | ✓ | 90.3% | 71.4% | N/A | 0.1 |
| Gemini-2.5 | ✓ | 92.4% | 78.5% | N/A | 0.1 |
| Qwen2.5-VL | ✓ | 92.5% | 70.4% | 72B | 0.1 |
| UniDet3D+GPT-4o | ✗ | 42.4% | 33.4% | 15M+N/A | 0.1 |
| YOLOv11-det+SPH | ✗ | 75.5% | 74.9% | 31.7M | 11.9 |
| **OrderMind-Mini** | ✗ | **94.2%** | **90.4%** | 35.2M | **21.3** |
| **OrderMind** | ✗ | **96.5%** | **95.4%** | 41.8M | 5.6 |

Key findings:
- OrderMind surpasses all VLMs that use ground-truth poses without requiring privileged information (96.5% vs. best 92.5%).
- The advantage is more pronounced on hard scenes: OrderMind 95.4% vs. Gemini-2.5 78.5% (+16.9%).
- OrderMind-Mini achieves 21.3 FPS real-time inference with 35.2M parameters while maintaining 90.4% SR.
- Two-stage frameworks (UniDet3D+VLM) are not only slow (0.1 FPS) but also suffer a large accuracy drop (only 33–46%).

### Ordering Stability Analysis

Replanning stability is measured using Levenshtein Distance (LD). OrderMind consistently shows lower LD than heuristic baselines across all replanning intervals, attributed to the unified learning of spatial representations and manipulation ordering, which produces globally consistent and foresighted plans compared to greedy heuristic approaches.

### Robustness to Annotation Noise

| Noise Ratio | Easy SR | Hard SR |
|-------------|---------|---------|
| 0% | 85.35% | 83.66% |
| 10% | 85.10% | 79.13% |
| 20% | 80.08% | 76.30% |
| 50% | 78.88% | 73.81% |
| 70% | 75.99% | 67.31% |

Performance degrades only mildly under moderate noise levels, demonstrating the robustness of the annotation integration strategy and unified learning architecture. Under extreme 70% noise, performance degrades significantly as the model approaches a random strategy.

### Real-World Experiments

| Difficulty | RC | SR |
|------------|-----|------|
| Easy | 0.2 | 93.3% |
| Medium | 2.0 | 78.5% |
| Hard | 3.0 | 76.6% |

The model demonstrates understanding of object isolation and stacking relationships in both factory and laboratory environments.

### Ablation Study

| SCU | TPS | SPOL | RC↓ | OD↓ | SR↑ |
|-----|-----|------|-----|-----|------|
| | | | 5.0 | 5.4 | 76.1% |
| ✓ | | | 3.6 | 4.4 | 81.0% |
| | ✓ | | 4.5 | 4.9 | 80.5% |
| ✓ | ✓ | | 3.5 | 4.4 | 87.7% |
| ✓ | ✓ | ✓ | **1.0** | **1.4** | **95.3%** |

The three modules exhibit strong synergy: SCU and TPS each contribute approximately +5% SR, while the introduction of SPOL yields the largest single gain (87.7% → 95.3%), underscoring the critical importance of learning meaningful manipulation sequences in cluttered environments.

### Failure Analysis

Sources of failure during 30-minute real-world operation:
- Object disturbance due to incorrect manipulation order: 39%
- Inaccurate 3D rotation estimation: 21%
- Inability to find a suitable suction surface on deformable objects: 21%
- Misidentification of object center points: 15%
- Robot–camera coordination issues: 4%

## Highlights & Insights

### Strengths
1. **Unified framework design**: Integrating perception and ordering into a single inference pass avoids the error accumulation and latency issues inherent to two-stage methods.
2. **VLM distillation strategy**: The framework cleverly leverages VLM reasoning capabilities to generate training annotations while improving annotation quality through spatial prior constraints; the resulting lightweight model surpasses the VLM teacher.
3. **Comprehensive benchmarking**: The first large-scale manipulation ordering benchmark (163K samples) is introduced, enabling systematic comparison of VLM, heuristic, and learning-based methods.
4. **Real-time performance**: OrderMind-Mini achieves 21.3 FPS with 35.2M parameters, demonstrating strong deployment potential.

### Limitations & Future Work
1. **Static scene assumption**: The current system assumes scene stability during execution and cannot adapt to dynamically changing environments.
2. **Dependence on accurate 3D estimation**: Manipulation ordering prediction relies on precise 3D attribute estimation, which remains challenging under severe occlusion.
3. **Limited object categories**: Simulation uses only 5 YCB object categories and the real-world experiments cover only boxes and bags; generalization to a more diverse object set remains unvalidated.
4. **VLM annotation quality bottleneck**: Performance is upper-bounded by VLM annotation quality, and the model degrades to a random strategy under 70% noise.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation](../../CVPR2026/robotics/learning_to_see_and_act_task-aware_virtual_view_exploration_for_robotic_manipula.md)
- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](../../CVPR2026/robotics/palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)
- [\[NeurIPS 2025\] Understanding Prompt Tuning and In-Context Learning via Meta-Learning](understanding_prompt_tuning_and_in-context_learning_via_meta-learning.md)
- [\[NeurIPS 2025\] Spatial Understanding from Videos: Structured Prompts Meet Simulation Data](spatial_understanding_from_videos_structured_prompts_meet_simulation_data.md)
- [\[NeurIPS 2025\] MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning](mesatask_towards_task-driven_tabletop_scene_generation_via_3d_spatial_reasoning.md)

<!-- RELATED:END -->
