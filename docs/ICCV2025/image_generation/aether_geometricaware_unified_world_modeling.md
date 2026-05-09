---
title: >-
  [Paper Note] Aether: Geometric-Aware Unified World Modeling
description: >-
  [ICCV 2025][Image Generation][world model] This paper proposes Aether, a unified world model that post-trains the CogVideoX video diffusion model on synthetic RGB-D data. Through a multi-task training strategy that randomly combines input/output modalities, Aether simultaneously achieves 4D reconstruction, action-conditioned video prediction, and goal-conditioned visual planning, with zero-shot transfer to real-world data reaching performance comparable to domain-specific models.
tags:
  - ICCV 2025
  - Image Generation
  - world model
  - 4D reconstruction
  - video prediction
  - visual planning
  - geometric awareness
  - synthetic data
date: 2026-05-08
content_hash: 93f5309430c4a90c
---

# Aether: Geometric-Aware Unified World Modeling

**Conference**: ICCV 2025
**arXiv**: [2503.18945](https://arxiv.org/abs/2503.18945)
**Code**: [Project](https://aether-world.github.io/)
**Area**: World Models / 4D Reconstruction / Visual Planning
**Keywords**: world model, 4D reconstruction, video prediction, visual planning, geometric awareness, synthetic data

## TL;DR

This paper proposes Aether, a unified world model that post-trains the CogVideoX video diffusion model on synthetic RGB-D data. Through a multi-task training strategy that randomly combines input/output modalities, Aether simultaneously achieves 4D reconstruction, action-conditioned video prediction, and goal-conditioned visual planning, with zero-shot transfer to real-world data reaching performance comparable to domain-specific models.

## Background & Motivation

- **State of the Field**: World models require three core capabilities—perception (4D reconstruction), prediction (action-conditioned generation), and planning (goal-conditioned reasoning)—yet existing methods typically address only one of these.
- **Limitations of Prior Work**: (1) Independent modeling of each capability lacks synergy; (2) real-world 4D annotated data is extremely scarce; (3) action representations are heterogeneous (keyboard inputs / robot actions / camera trajectories).
- **Root Cause**: The demand for unifying three capabilities conflicts with the heterogeneity of data and representations.
- **Paper Goals**: Construct a unified framework that simultaneously supports reconstruction, prediction, and planning.
- **Starting Point**: Synthetic data + camera trajectories as a unified action representation + multi-task post-training.
- **Core Idea**: Post-train a video diffusion model on synthetic 4D data, using camera trajectories as the geometric action space to unify reconstruction, prediction, and planning.

## Method

### Overall Architecture

CogVideoX-5b-I2V serves as the base model, with camera parameters automatically annotated from synthetic RGB-D data. The model produces three output modalities: color video, depth video, and action (raymap). Different tasks are realized through distinct conditioning combinations.

### Key Designs

**Design 1: 4D Synthetic Data Annotation Pipeline**
- **Function**: Automatically obtain accurate camera parameters from synthetic RGB-D videos.
- **Mechanism**: Dynamic object masking (Grounded SAM2) → video segmentation (SIFT + optical flow filtering) → coarse estimation (DroidCalib) → refinement (CoTracker3 + Bundle Adjustment + Ceres Solver).
- **Design Motivation**: 4D annotated data is scarce; an automated pipeline is a prerequisite for scaling.

**Design 2: Raymap Camera Representation**
- **Function**: Convert camera trajectories into a raymap video representation compatible with diffusion models.
- **Mechanism**: Each frame uses 6 channels (3D ray directions + 3D ray origins), with translation normalized via log-scale. The raymap is invertible—camera intrinsics and extrinsics can be recovered from the generated raymap.
- **Design Motivation**: Camera parameters must align with the spatiotemporal tokens of the DiT; raymaps naturally possess spatial structure.

**Design 3: Multi-Task Random Conditioning Training**
- **Function**: Randomly mask different conditioning combinations to enable unified multi-task training.
- **Mechanism**: Color conditioning probability allocation: 30% planning (first + last frame), 40% prediction (first frame), 28% reconstruction (full video), 2% fully unmasked. Action conditioning is retained 50% of the time and masked otherwise. Two-stage training: Stage 1 uses standard diffusion loss; Stage 2 adds decoded MS-SSIM + scale-invariant depth loss + point map loss.
- **Design Motivation**: Random conditioning enables knowledge transfer across tasks; geometric supervision ensures 3D consistency.

### Loss & Training

Stage 1: Standard diffusion MSE. Stage 2: + MS-SSIM (color) + scale-shift invariant loss (depth) + scale-shift invariant point map loss (depth + raymap). Training runs for two weeks on 80× A100 GPUs.

## Key Experimental Results

### Main Results

**Video Depth Estimation (Abs Rel↓ / $\delta<1.25$↑)**

| Method | Sintel | BONN | KITTI |
|--------|--------|------|-------|
| MonST3R | 0.378/55.8 | **0.067/96.3** | 0.168/74.4 |
| DepthCrafter | 0.590/55.5 | 0.253/56.3 | 0.124/86.5 |
| **Aether** | **0.314/60.4** | 0.273/59.4 | **0.054/97.7** |

### Ablation Study

| Configuration | Depth Abs Rel (Sintel) |
|---------------|----------------------|
| w/o multi-task training | 0.45 |
| w/o Stage 2 geometric loss | 0.38 |
| Full Aether | **0.314** |

### Key Findings

1. Training exclusively on synthetic data enables zero-shot transfer to real-world scenes, with reconstruction performance matching or exceeding specialist models on most benchmarks.
2. Multi-task training yields significant knowledge transfer—reconstruction capability promotes geometric consistency in prediction and planning.
3. Camera trajectories as the action space are particularly effective for ego-view tasks such as navigation.

## Highlights & Insights

1. This is the first work to unify reconstruction, prediction, and planning within a single video diffusion model.
2. The paradigm of synthetic data combined with an automated annotation pipeline is scalable to large-scale settings.
3. The raymap representation elegantly embeds camera parameters into the diffusion model framework.

## Limitations & Future Work

1. Camera trajectories as the action space are insufficiently general for non-ego-view tasks such as robotic arm manipulation.
2. The domain gap from synthetic data still manifests in certain real-world scenarios.
3. Planning capability is achieved solely through first- and last-frame conditioning, lacking explicit path optimization.

## Related Work & Insights

- DA-V trains depth estimation on synthetic data but does not address reconstruction or planning.
- Insight: Post-training video foundation models can efficiently inject 4D geometric reasoning capabilities.

## Rating

| Dimension | Score |
|-----------|-------|
| Novelty | ★★★★★ |
| Practicality | ★★★★☆ |
| Experimental Thoroughness | ★★★★★ |
| Writing Quality | ★★★★☆ |

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] AID: Adapting Image2Video Diffusion Models for Instruction-guided Video Prediction](aid_adapting_image2video_diffusion_models_for_instruction-guided_video_predictio.md)
- [\[ICCV 2025\] Long-Context State-Space Video World Models](long-context_state-space_video_world_models.md)
- [\[ICCV 2025\] SummDiff: Generative Modeling of Video Summarization with Diffusion](summdiff_generative_modeling_of_video_summarization_with_diffusion.md)
- [\[ICCV 2025\] UniCombine: Unified Multi-Conditional Combination with Diffusion Transformer](unicombine_unified_multi-conditional_combination_with_diffusion_transformer.md)
- [\[ICCV 2025\] A Unified Framework for Motion Reasoning and Generation in Human Interaction](a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)

<!-- RELATED:END -->
