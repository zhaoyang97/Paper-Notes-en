---
title: >-
  [Paper Note] BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models
description: >-
  [NEURIPS2025][Multimodal VLM][Vision-Language-Action] This paper proposes BridgeVLA, which projects 3D point clouds into multi-view 2D images and uses 2D heatmaps as an intermediate representation to align the input and output spaces, enabling efficient and effective 3D robot manipulation learning.
tags:
  - NEURIPS2025
  - Multimodal VLM
  - Vision-Language-Action
  - 3D Manipulation
  - Heatmap Prediction
  - Robot Learning
  - sample efficiency
date: 2026-05-08
content_hash: 967ebf1106178641
---

# BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models

**Conference**: NEURIPS2025  
**arXiv**: [2506.07961](https://arxiv.org/abs/2506.07961)  
**Code**: [Project Page](https://bridgevla.github.io/)  
**Area**: Multimodal VLM  
**Keywords**: Vision-Language-Action, 3D Manipulation, Heatmap Prediction, Robot Learning, sample efficiency

## TL;DR
This paper proposes BridgeVLA, which projects 3D point clouds into multi-view 2D images and uses 2D heatmaps as an intermediate representation to align the input and output spaces, enabling efficient and effective 3D robot manipulation learning.

## Background & Motivation
- Leveraging pretrained vision-language models (VLMs) to build vision-language-action (VLA) models has become the dominant paradigm for learning generalizable robot manipulation policies.
- However, most VLA models rely solely on 2D image inputs and require large amounts of data collection, while 3D policies offer high sample efficiency but lack the broad semantic knowledge of VLMs.
- Existing 3D VLA methods (e.g., 3D-VLA, SpatialVLA) suffer from two critical issues:
    1. Actions are represented as token sequences without spatial structure, failing to exploit the spatial priors in 3D data and resulting in low sample efficiency.
    2. The 3D inputs used during fine-tuning are misaligned with the 2D image inputs seen during pretraining, hindering knowledge transfer.

## Core Problem
How to design a 3D VLA model that simultaneously inherits the semantic generalization of VLMs and the spatial efficiency of 3D policies, while achieving input-output alignment in a unified 2D space?

## Method

### Overall Architecture
BridgeVLA adopts a two-stage training pipeline: 2D heatmap pretraining followed by 3D action fine-tuning, using PaliGemma (SigLIP visual encoder + Gemma Transformer) as the VLM backbone.

### Stage 1: 2D Heatmap Pretraining
- **Function**: Trains the VLM to predict spatial heatmaps conditioned on text descriptions, overcoming the limitation of vanilla VLMs that can only produce token sequences.
- **Data**: 120K object detection samples from RoboPoint.
- **Heatmap Construction**: For each target object with bounding box center $\hat{\mathbf{x}_i}$, a truncated Gaussian probability map is constructed:
$$H_i^{gt}(\mathbf{x}) = \begin{cases} \exp(-\|\mathbf{x}-\hat{\mathbf{x}_i}\|^2 / 2\sigma^2) & \text{if } p_i(\mathbf{x}) \geq p_{\min} \\ 0 & \text{otherwise} \end{cases}$$
- For multiple targets, the final heatmap $H^{gt}$ is obtained by averaging and normalizing.
- **Decoding**: Image tokens output by the VLM are spatially rearranged according to patch positions into a feature grid, then restored to the original image resolution via a learnable convex upsampling module.
- **Loss**: Cross-entropy loss supervises heatmap prediction.
- **Scalability**: This pretraining strategy can in principle leverage any vision-language dataset that can be reformulated as heatmap prediction (e.g., keypoint detection, semantic segmentation).

### Stage 2: 3D Action Fine-tuning
- **Input Alignment**: RGB-D images are used to reconstruct point clouds, which are then rendered into three orthographic projection views (top, front, right) as VLM inputs, maintaining alignment with the 2D images seen during pretraining.
- **Translation Prediction**: Each of the three views produces a heatmap; back-projection is applied to score 3D grid points, and the highest-scoring point is taken as the end-effector translation.
- **Rotation/Gripper/Collision Prediction**: Global features are obtained via max-pooling over output tokens from each view, local features are extracted from heatmap peaks, and the concatenated features are passed through an MLP to predict rotation (Euler angles, 72 bins per axis), gripper state, and collision flag.
- **Coarse-to-Fine Strategy**: After an initial prediction, the point cloud is cropped and zoomed around the predicted translation, and a second forward pass is performed; the result of the second pass is used for execution.
- **Training Loss**: $L = L_{trans} + L_{rot} + L_{gripper} + L_{collision}$
- **Key Design**: No additional 3D positional information or robot state is injected into the VLM, maximally preserving the pretraining-finetuning distributional consistency.

### Core Design Philosophy
1. **Input Alignment**: 3D → multi-view orthographic 2D projections, consistent with VLM pretraining inputs.
2. **Output Alignment**: Actions → 2D heatmaps, sharing spatial structure with the input images.
3. **Stage Alignment**: Both pretraining and fine-tuning operate in a unified 2D space.

## Key Experimental Results

### RLBench (18 tasks, 100 expert demonstrations per task)

| Method | Avg. Success Rate | Avg. Rank |
|--------|-------------------|-----------|
| RVT-2 (SOTA) | 81.4% | 2.75 |
| 3D Diffuser Actor | 81.3% | 2.67 |
| **BridgeVLA** | **88.2%** | **2.03** |

- Substantial lead on Insert Peg (88.0% vs. 40.0%), demonstrating high-precision manipulation capability.
- Clear advantage on Sort Shape (60.8% vs. 35.0%).

### COLOSSEUM (12 perturbation categories for generalization)

| Method | Avg. Success Rate | Avg. Rank |
|--------|-------------------|-----------|
| RVT-2 | 56.7% | 1.92 |
| **BridgeVLA** | **64.0%** | **1.07** |

- Ranks first in 13 out of 14 perturbation settings.

### GemBench (4 generalization difficulty levels)

| Method | Avg. | L1 | L2 | L3 | L4 |
|--------|------|----|----|----|----|
| 3D-LOTUS++ | 48.0 | 68.7 | 64.5 | 41.5 | 17.4 |
| **BridgeVLA** | **50.0** | 91.1 | **65.0** | **43.8** | 0.0 |

- Achieves state of the art on L2 (novel objects) and L3 (articulated objects).

### Real-Robot Experiments
- **Data Efficiency**: Achieves 95.4% success rate with only 3 demonstrations per task, while π₀ and SpatialVLA nearly completely fail with 10 demonstrations.
- Reaches 96.9% with 10 demonstrations per task, outperforming RVT-2 (90%) by an average of 32%.
- Comprehensively surpasses RVT-2 across 6 generalization settings, with particularly large margins under lighting variation and compositional generalization.

### Ablation Study

| Variant | RLBench Avg. Success Rate |
|---------|--------------------------|
| Without heatmap (direct position regression) | 31.4% (↓56.8%) |
| With 3D positional features | 56.2% (↓32.0%) |
| **Full BridgeVLA** | **88.2%** |

## Highlights & Insights
1. **Input-output alignment paradigm**: BridgeVLA is the first 3D VLA to unify pretraining and fine-tuning input-output spaces into a common 2D domain — an elegant design choice.
2. **Exceptional sample efficiency**: 95.4% success rate with only 3 demonstrations per task, far exceeding π₀, SpatialVLA, and other methods.
3. The advantage of **heatmap intermediate representation** is thoroughly validated through ablation — removing it causes a 57-percentage-point drop in success rate.
4. **Not injecting extra 3D information yields better results** — maintaining distributional consistency with pretraining matters more than incorporating additional information.
5. Experiments are highly comprehensive: 3 simulation benchmarks + real-robot evaluation across 7 settings + 3 ablation studies.

## Limitations & Future Work
1. **Poor performance on long-horizon tasks**: Success rate of 0% on GemBench L4, lacking subtask decomposition capability; future work could incorporate LLM-based task planning.
2. **Occlusion issues**: In tasks such as Place Cups, target keypoints may be occluded in all orthographic views; dynamic projection view selection warrants exploration.
3. **Limited category-level generalization**: Absolute success rates in real-robot category-generalization settings remain modest, partly due to viewpoint discrepancies between pretraining and robot data.
4. **Pretraining data scale**: Only 120K detection samples are currently used; expanding to semantic segmentation, keypoint detection, and other tasks is expected to improve generalization.
5. **Information loss from orthographic projections**: Three fixed orthographic views may discard critical information from certain perspectives.

## Related Work & Insights

| Dimension | BridgeVLA | SpatialVLA | π₀ | RVT-2 | 3D Diffuser Actor |
|-----------|-----------|------------|-----|-------|-------------------|
| Input | Multi-view orthographic projections | 2D images + Ego3D encoding | 2D images | Multi-view orthographic projections | 3D point clouds |
| Output | 2D heatmap → 3D action | Token sequence | Flow Matching | 2D heatmap | 3D diffusion trajectory |
| VLM Backbone | PaliGemma | Qwen2-VL | PaliGemma | None | None |
| Data Efficiency | Very high (3 demos suffice) | Low (poor even at 50 demos) | Low | High | Medium |
| Input-Output Alignment | ✓ | ✗ | ✗ | ✓ (partial) | ✗ |

- Compared to SpatialVLA: BridgeVLA avoids injecting 3D positional information into the VLM, instead preserving distributional consistency through orthographic projection, yielding better results.
- Compared to RVT-2: BridgeVLA inherits the orthographic projection + heatmap design but adds a VLM backbone and pretraining, thereby acquiring semantic generalization capability.
- Compared to π₀: 3D perception combined with heatmap output substantially improves sample efficiency.

## Highlights & Insights (Extended)
1. **Maintaining pretraining distributional consistency is more important than injecting additional information** — this insight has broad implications for VLM-based robot policy design.
2. The **heatmap-as-intermediate-representation** paradigm is generalizable to other spatial prediction tasks (e.g., grasp point prediction, navigation).
3. The object detection → heatmap task design in the pretraining stage is clever, providing a general paradigm for adapting VLMs to downstream spatial tasks.
4. The coarse-to-fine two-pass inference strategy effectively improves precision, but incurs 2× inference overhead — more efficient alternatives are worth exploring.

## Rating
- Novelty: 8/10 — The paradigm of aligning inputs and outputs into a unified 2D space is novel; the heatmap pretraining strategy is cleverly designed.
- Experimental Thoroughness: 9/10 — 3 simulation benchmarks + real-robot evaluation across 7 settings + comprehensive ablation studies.
- Writing Quality: 8/10 — Clear structure with five research questions addressed systematically; strong logical coherence.
- Value: 8/10 — Achieving 95% success rate with only 3 demonstrations represents a meaningful breakthrough in sample efficiency with practical application value.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching](vla-cache_efficient_vision-language-action_manipulation_via_adaptive_token_cachi.md)
- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[NeurIPS 2025\] Learning to Steer: Input-dependent Steering for Multimodal LLMs](learning_to_steer_input-dependent_steering_for_multimodal_llms.md)
- [\[CVPR 2026\] PointAlign: Feature-Level Alignment Regularization for 3D Vision-Language Models](../../CVPR2026/multimodal_vlm/pointalign_feature-level_alignment_regularization_for_3d_vision-language_models.md)
- [\[NeurIPS 2025\] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)

<!-- RELATED:END -->
