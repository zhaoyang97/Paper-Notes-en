---
title: >-
  [Paper Note] Generating Human Interaction Motions in Scenes with Text Control
description: >-
  [ECCV 2024][Image Generation][Scene-aware motion generation] TeSMo is proposed as a text-controlled, scene-aware motion generation method. By pre-training a text-to-motion diffusion model on large-scale motion data and fine-tuning it with an enhanced scene-sensing branch, it generates realistic motion sequences of characters navigating obstacles and interacting with objects (e.g., sitting down) in 3D scenes in two stages (navigation + interaction).
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Scene-aware motion generation"
  - "text control"
  - "diffusion models"
  - "human-scene interaction"
  - "navigation and interaction"
date: 2026-05-08
content_hash: 8031ec3becc5ddbd
---

# Generating Human Interaction Motions in Scenes with Text Control

**Conference**: ECCV 2024  
**arXiv**: [2404.10685](https://arxiv.org/abs/2404.10685)  
**Code**: [Yes](https://research.nvidia.com/labs/toronto-ai/tesmo)  
**Area**: Image Generation / Human Motion Generation  
**Keywords**: Scene-aware motion generation, text control, diffusion models, human-scene interaction, navigation and interaction

## TL;DR

TeSMo is proposed as a text-controlled, scene-aware motion generation method. By pre-training a text-to-motion diffusion model on large-scale motion data and fine-tuning it with an enhanced scene-sensing branch, it generates realistic motion sequences of characters navigating obstacles and interacting with objects (e.g., sitting down) in 3D scenes in two stages (navigation + interaction).

## Background & Motivation

### Problem Introduction

Generating realistic human-environment interaction motions in 3D scenes is crucial for games, movies, and embodied AI. The ideal scenario is that users control the action style via text (e.g., "happily jump to the chair and sit down"), while the model perceives scene obstacles to generate reasonable interaction motions.

### Limitations of Prior Work

**Text-to-motion diffusion models** (MDM, MotionDiffuse, etc.): Generate high-quality motion but completely ignore environmental context, leading to characters passing through obstacles.

**Scene-aware VAE/diffusion models** (SceneDiffuser, etc.): Trained on small-scale paired human-scene datasets, leading to limited motion diversity and quality, and lacking text control.

**Reinforcement learning methods** (DIMOS, etc.): Can learn interaction motions, but reward function design is difficult, offering low motion diversity and lacking text control.

**Extreme scarcity of paired data**: Compared to large-scale mocap datasets (HumanML3D), data that simultaneously contains 3D motion, scene geometry, and text annotations is extremely scarce.

### Key Insight

The problem is decomposed into two subtasks: **navigation** and **interaction**, with diffusion models designed for each. The key strategy is to pre-train a text-to-motion model on large-scale scene-free data to obtain strong motion priors, and then fine-tune it with an enhanced scene-aware control branch (similar to ControlNet) to minimize the demand for paired data.

## Method

### Overall Architecture

TeSMo decomposes motion generation in scenes into two stages:

1. **Navigation Stage**: Inputs include starting point, ending point, 2D scene map, and text prompt $\rightarrow$ first generates pelvis root trajectory $\rightarrow$ then lifts it to full-body motion via in-painting.
2. **Interaction Stage**: Inputs include navigation end pose, target pelvis pose, 3D object geometry, and text prompt $\rightarrow$ directly generates full-body interaction motion.

Both stages employ a dual-branch architecture consisting of **pre-training + scene branch fine-tuning**.

### Key Designs

#### 1. Scene-Aware Control Branch

**Function**: Add an independent scene-aware Transformer branch on top of the frozen pre-trained text-to-motion Transformer encoder, connected to each layer of the base model via zero-initialized linear layers.

**Mechanism**: An enhanced control strategy similar to ControlNet. The pre-trained base model provides text-following capability and motion realism, while the scene branch provides environmental constraints. During fine-tuning, only the scene branch parameters are trained (approx. 20k steps), and the base model is frozen.

**Design Motivation**: Direct single-branch training from scratch (such as an adapted TRACE) performs worse in both target arrival accuracy and full-body motion quality compared to the dual-branch fine-tuning scheme. Single-stage training FID: 22.372 vs. dual-branch fine-tuning: 20.465.

#### 2. Navigation Model (Root Trajectory Generation)

The motion of each frame is represented as $\mathbf{x}^n = [x, y, z, \cos\theta, \sin\theta]$ (pelvis position + orientation), using **absolute coordinates** instead of relative velocity to facilitate target pose constraints.

- **In-painting target arrival**: Overwrites the first and last frames with the starting/ending clean pose during each denoising step.
- **Scene input**: Extracts a bird's-eye view (BEV) 2D map $\mathcal{M}$ of the walkable area from the 3D scene, encodes it into a feature grid using ResNet-18, and queries corresponding features via the 2D projected pelvis position of each frame.
- **Inference guidance**: Target arrival guidance $\mathcal{J}_g = (\hat{\mathbf{x}}_0^N - \mathbf{g})^2$ + collision guidance $\mathcal{J}_c = \text{SDF}(\hat{\mathbf{x}}_0, \mathcal{M})$.
- **A* path blending**: Supports linear blending of model-predicted trajectories with user-specified/A* paths during denoising steps: $\tilde{\mathbf{p}}_0 = s \cdot \hat{\mathbf{p}}_0 + (1-s) \cdot \mathbf{p}$.

Full-body motion is obtained by lifting from the root trajectory using PriorMDM in-painting.

#### 3. Interaction Model (Full-Body Motion Generation)

- Generates full-body motion (268 dimensions/frame) directly, including absolute pelvis pose, joint positions/velocities/rotations, and foot contacts.
- **Object representation**: Uses Basis Point Sets (BPS). 1024 points are sampled within a 1.0m radius sphere of the object's center to compute the object geometry features $\mathbf{B}_O$ and frame-wise human-object relationship features $\mathbf{B}^n$.
- Collision guidance uses 3D SDF to penalize body vertices penetrating objects.

### Loss & Training

- Pre-training stage: Standard diffusion reconstruction loss $\|\mathbf{x}_0 - \hat{\mathbf{x}}_0\|^2$.
- Fine-tuning stage: Same as above, only training the scene branch parameters.
- Inference guidance: Target arrival loss (weight 30-1000) + collision SDF loss (weight 10-1000).

### Data Construction

- **Loco-3D-FRONT**: Walk sequences from HumanML3D are embedded into 3D-FRONT indoor scenes, augmented with left-right mirroring, yielding approximately 9500 motions $\times$ 10 scenes = 95k training pairs.
- **Augmented SAMP**: Sub-sequences (walk-to-sit, stand-to-sit, etc.) are extracted from 80 sitting motion segments, randomly matched with 3D-FRONT chair geometries, annotated with text descriptions, resulting in ~200 sub-sequences/motions after left-right augmentation.

## Key Experimental Results

### Main Results

**Navigation Evaluation (Loco-3D-FRONT test set, ~1000 sequences)**:

| Method | Pos. Error ↓ | Yaw Error ↓ | Height Error ↓ | Collision Rate ↓ | FID ↓ | R-precision ↑ | Diversity ↑ | Foot Slide ↓ |
|------|----------|----------|----------|---------|------|-------------|---------|------|
| GMD | 0.374 | 1.231 | - | - | 13.160 | 0.114 | 4.488 | 0.181 |
| OmniControl | 1.226 | 1.018 | 1.159 | - | 22.930 | 0.458 | 7.128 | 0.094 |
| TRACE | 0.205 | 0.152 | 0.010 | 0.055 | 22.669 | 0.144 | 6.501 | 0.058 |
| **TeSMo** | **0.169** | **0.119** | **0.008** | **0.031** | **20.465** | **0.376** | 6.415 | **0.056** |

**Interaction Evaluation (SAMP sitting test set)**:

| Method | Pos. Error ↓ | Height Error ↓ | Yaw Error ↓ | Penetration Vol. ↓ | Penetration Ratio ↓ | User Preference ↑ |
|------|----------|----------|----------|---------|---------|----------|
| DIMOS | 0.2020 | 0.1283 | 0.4731 | 0.0193 | 0.1076 | 29.1% |
| **TeSMo** | **0.1445** | **0.0120** | **0.2410** | **0.0043** | **0.0611** | **71.9%** |

### Ablation Study

**Inference Guidance Effect**:

| Target Guidance | Collision Guidance | Nav. Pos. Error ↓ | Nav. Collision Rate ↓ | Inter. Pos. Error ↓ | Inter. Penetration Vol. ↓ | Inter. Penetration Ratio ↓ |
|---------|---------|-------------|-----------|-------------|-----------|-----------|
| ✗ | ✗ | 0.1568 | 0.0294 | 0.1445 | 0.0043 | 0.0611 |
| ✓ | ✗ | 0.118 | 0.0342 | 0.1453 | 0.0050 | 0.0554 |
| ✗ | ✓ | 0.1550 | 0.0013 | 0.1407 | 0.0040 | 0.0414 |
| ✓ | ✓ | **0.1241** | **0.0012** | **0.1404** | **0.0045** | **0.0494** |

**Training Strategy Ablation**: Single-stage training (training both branches from scratch simultaneously) results in target position error of 0.197 vs. 0.169 for two-stage, and FID of 22.372 vs. 20.465.

### Key Findings

1. **The dual-branch pre-training + fine-tuning architecture significantly outperforms training from scratch**: The dual-branch scheme leads comprehensively in target arrival, collision avoidance, and full-body motion quality.
2. **Hierarchical navigation design is effective**: Generating the root trajectory first followed by full-body motion achieves more accurate target destination arrival than end-to-end methods, while preserving the diversity of text control.
3. **Inference guidances are complementary**: Target guidance mainly improves positional accuracy ($0.156 \rightarrow 0.118$), while collision guidance mainly reduces the collision rate ($0.029 \rightarrow 0.001$). The combined usage yields the best results.
4. **The autoregressive strategy of DIMOS suffers from severe accumulated errors**: Although DIMOS takes full-body target pose as input (a stronger conditioning than TeSMo's pelvis-only pose), its target arrival accuracy is still worse.
5. **User Study** (AMT, 30 participants): TeSMo significantly outperforms DIMOS with a preference rate of 71.9%, showing more natural interaction motions and fewer penetrations.

## Highlights & Insights

- **The paradigm of pre-training + scene-aware fine-tuning** is highly practical: it maximizes the utilization of motion priors from large-scale scene-free data while minimizing dependence on scarce paired human-scene data.
- **Decomposing into navigation + interaction** is a sound engineering decision, utilizing different scene representations (2D map vs. 3D BPS) for different subtasks.
- **Clever data augmentation strategy**: Embedding existing motion data into virtual scenes provides rich training data for fine-tuning the scene branch.
- The A* path blending mechanism offers a flexible user-control interface.

## Limitations & Future Work

- The two-stage navigation process may lead to inconsistencies between the root trajectory and full-body poses.
- Using only 2D maps limits the capability to handle complex interactions, such as stepping over a low stool.
- Only sit/stand interactions are currently demonstrated; generalizing to more interaction types requires additional data.
- Object matching in data augmentation relies on contact-point constraints, which may limit certain extreme poses.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The pre-training + scene branch fine-tuning paradigm is a novel application in the field of motion generation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Quantitative evaluation, user studies, and ablations are conducted for both navigation and interaction, making it comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear framework, reasonable decomposition, and detailed data construction process.
- **Value**: ⭐⭐⭐⭐ — Provides a practical solution for scene-aware, controllable motion generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Text2Place: Affordance-aware Text Guided Human Placement](text2place_affordance-aware_text_guided_human_placement.md)
- [\[ECCV 2024\] LivePhoto: Real Image Animation with Text-guided Motion Control](livephoto_real_image_animation_with_text-guided_motion_control.md)
- [\[ECCV 2024\] COIN: Control-Inpainting Diffusion Prior for Human and Camera Motion Estimation](coin_control-inpainting_diffusion_prior_for_human_and_camera_motion_estimation.md)
- [\[ECCV 2024\] Ponymation: Learning Articulated 3D Animal Motions from Unlabeled Online Videos](ponymation_learning_articulated_3d_animal_motions_from_.md)
- [\[CVPR 2026\] ViHOI: Human-Object Interaction Synthesis with Visual Priors](../../CVPR2026/image_generation/vihoi_human-object_interaction_synthesis_with_visual_priors.md)

</div>

<!-- RELATED:END -->
