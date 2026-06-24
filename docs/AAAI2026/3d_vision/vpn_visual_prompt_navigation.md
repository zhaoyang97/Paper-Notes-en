---
title: >-
  [Paper Note] VPN: Visual Prompt Navigation
description: >-
  [AAAI 2026][3D Vision][Visual Navigation] Proposes a new paradigm of Visual Prompt Navigation (VPN): users annotate visual trajectories (arrows connecting key waypoints) on a 2D top-down view to guide agent navigation, replacing natural language instructions and image goal instructions. Constucts two datasets, R2R-VP and R2R-CE-VP, along with the VPNet baseline model. Combining view-level and trajectory-level data augmentation leads to outstanding performance in both discrete…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Visual Navigation"
  - "Visual Prompt"
  - "Top-down View"
  - "Vision-Language Navigation"
  - "Data Augmentation"
date: 2026-05-08
content_hash: 6813bfc47b7f8f83
---

# VPN: Visual Prompt Navigation

**Conference**: AAAI 2026  
**arXiv**: [2508.01766](https://arxiv.org/abs/2508.01766)  
**Code**: [github.com/farlit/VPN](https://github.com/farlit/VPN)  
**Area**: 3D Vision  
**Keywords**: Visual Navigation, Visual Prompt, Top-down View, Vision-Language Navigation, Data Augmentation

## TL;DR

Proposes a new paradigm of Visual Prompt Navigation (VPN): users annotate visual trajectories (arrows connecting key waypoints) on a 2D top-down view to guide agent navigation, replacing natural language instructions and image goal instructions. Constucts two datasets, R2R-VP and R2R-CE-VP, along with the VPNet baseline model. Combining view-level and trajectory-level data augmentation leads to outstanding performance in both discrete and continuous environments.

## Background & Motivation

### Limitations of Existing Navigation Paradigms

Visual navigation is a core research direction in AI and robotics. Existing mainstream paradigms include:

**PointGoal Navigation**: Provides the target's relative direction and distance, lacking intermediate guidance.

**ImageGoal Navigation**: Provides an image of the target location, but lacks intermediate navigation cues.

**Vision-Language Navigation (VLN)**: Describes the navigation path using natural language, currently the most active area.

**The Fundamental Dilemma of Natural Language Instructions**: Language inherently possesses **ambiguity** when describing object locations, directional turns, and distance relationships; striving for precise descriptions inevitably leads to **verbosity**. This creates a dilemma in human-computer interaction.

### Advantages of Visual Prompts

The authors propose an intuitive insight: **drawing a route on a map** is the most natural way for humans to give directions. Key advantages:

**High User Accessibility**: Non-expert users can naturally specify navigation targets by clicking or drawing trajectories.

**Rich Spatial Information**: The top-down view naturally preserves the complete spatial layout.

**High Reusability**: Top-down views are acquired via UAV aerial photography or 3D reconstruction, and can be reused repeatedly once constructed.

## Method

### Overall Architecture

The core workflow of VPN consists of three parts:
1. **Dataset Construction**: Replaces natural language instructions in R2R/R2R-CE with visual prompts.
2. **VPNet Model**: Based on the DUET/ETPNav architecture, replacing the language encoder with a ViT encoder.
3. **Data Augmentation Strategies**: View-level and trajectory-level augmentations.

### Key Designs

#### 1. Visual Prompt Construction Process

Four-step generation: ① Generate top-down view, ② Draw trajectories by connecting waypoints with arrows, ③ Crop centered on the trajectory (+60px margin), ④ Remove black borders to tightly bound the visual prompt.

Design Motivation: Center cropping is a key step. Ablation studies show that without cropping, different episodes of the same scene share the same top-down view, leading to overfitting to the scene rather than learning trajectory information (SR is only 31%).

#### 2. VPNet Model Architecture

Three core components:

**ViT Visual Prompt Encoder**: Encodes the 224x224 visual prompt map using ViT-B/16 (pretrained on ImageNet-21k). Multi-floor scenes use **Order-Aware Floor Concatenation (OAFC)**:
$$\mathcal{P}_i^o = \text{ViT}(\mathcal{P}_i) + b_i, \quad \mathcal{P} = [\mathcal{P}_1^o, ..., \mathcal{P}_k^o]$$

**Node Embedding Module**: The agent incrementally builds a topological map, where each node consists of panoramic view features (encoded by a two-layer Transformer), step embedding, and position embedding.

**Graph-Aware Cross-Modal Encoder**: A multi-layer cross-modal graph Transformer, including cross-attention layers and a Graph-Aware Self-Attention (GASA) layer:
$$\text{GASA}(X) = \text{Softmax}\left(\frac{XW_q(XW_k)^T}{\sqrt{d}} + EW_d\right)XW_v$$
where $E$ is the pairwise distance matrix of the topological graph.

#### 3. Data Augmentation Strategies

**Trajectory-level Augmentation**: Introduces PREVALENT (178k trajectories) and ScaleVLN (1.6M trajectories) to increase training data diversity.

**View-level Augmentation**:
- **Prompt View Augmentation**: Randomly rotates the top-down view (0°/90°/180°/270°).
- **Agent View Augmentation**: Randomly samples the initial orientation.

Design Motivation: In VPN, the initial orientation is independent of the visual prompt (unlike VLN where the language might imply an initial direction), allowing for free rotation.

### Loss & Training

- **Behavior Cloning + DAgger**: $\mathcal{L} = \lambda \mathcal{L}_{BC} + (1-\lambda) \mathcal{L}_{DAG}$, where $\lambda = 0.5$.
- Discrete environments: Single A5000 GPU, 400k iterations, batch=10, lr=1.5e-5.
- Continuous environments: Dual A5000 GPUs, 400k iterations, batch=16, lr=1e-5.

## Key Experimental Results

### Main Results

**Discrete Environment (R2R-VP)**:

| Method | Training Data | Val Unseen SR↑ | Val Unseen SPL↑ | Test Unseen SR↑ |
|------|---------|----------------|-----------------|-----------------|
| DUET (VLN) | R2R+PRE+SCA | 81 | 70 | 80 |
| VPNet | R2R | 51.23 | 43.47 | 52.40 |
| VPNet | R2R+PRE | 65.92 | 56.17 | 66.38 |
| **VPNet** | **R2R+PRE+SCA** | **96.68** | **94.84** | **97.56** |

VPN achieves 96.68% SR on Val Unseen, far exceeding DUET's 81%, while using only 1/3 of the ScaleVLN trajectories.

**Continuous Environment (R2R-CE-VP)**:

| Method | Setting | Val Seen SR↑ | Val Unseen SR↑ |
|------|------|-------------|----------------|
| ETPNav (VLN) | R2R+PRE | 66 | 57 |
| VPNet | R2R+PRE | **84.11** | 47.96 |

### Ablation Study

**Effect of Different Visual Prompt Types (Discrete Environment)**:

| Prompt Type | Val Seen SR | Val Unseen SR | Description |
|----------|-------------|---------------|------|
| Uncropped full top-down view | 31.68 | 33.94 | Overfitting to scenes |
| Cropped top-down view only | 83.56 | 45.83 | Similar to ImageNav |
| Cropped map + arrows + text | 95.74 | 65.36 | Text occludes details |
| **Cropped map + arrows** | **100** | **65.92** | Optimal |

**Effect of View-level Augmentation**:

| Augmentation Method | Val Unseen SR↑ | SPL↑ |
|----------|----------------|------|
| No Augmentation | 86.33 | 82.92 |
| Agent View Only | 88.18 | 85.02 |
| Prompt View Rotation Only | 96.41 | 94.37 |
| **Combination of Both** | **96.68** | **94.84** |

Prompt view rotation is significantly more effective than agent view augmentation (+10 SR vs. +2 SR).

### Key Findings

1. **Stunning Data Efficiency of Visual Prompts**: VPNet achieves 96.68% SR using only 1.6M trajectories, whereas DUET achieves only 81% SR with 4.9M trajectories.
2. **Cropping is Critical**: Without cropping, severe overfitting to the scene occurs (31% vs. 100% Val Seen).
3. **Cropping without Trajectories is also Effective**: The model can infer the approximate destination from the cropped region (similar to ImageNav).
4. **Robustness to Noise**: Under 20% salt-and-pepper noise, the SR drops from 96.68% to 90.34%.

## Highlights & Insights

1. **Paradigm Innovation**: VPN is a completely new paradigm for visual navigation, filling the gap between language navigation and image goal navigation.
2. **Data Efficiency Advantage**: The density of spatial information contained in visual prompts is significantly higher than that in language instructions.
3. **Strong Practicality**: Top-down views can be acquired once through UAVs or 3D reconstruction and reused multiple times.
4. **Excellent Ablation Study Design**: Systematically analyzes multiple dimensions including prompt types, augmentation strategies, and encoder setups.

## Limitations & Future Work

1. **Validated Only in Simulation**: Uses MP3D/HM3D scenes without physical real-world testing.
2. **Dependence on High-Quality Top-Down Views**: Unusable in some scenes where reconstruction quality is poor.
3. **Performance Gap in Continuous Environments**: Val Unseen SR is only 47.96%, which is much lower than the 96.68% in discrete environments.
4. **Coarse Handling of Multi-Floor Scenes**: Simplistic concatenation of features across floors.

## Related Work & Insights

- **VLN (R2R) Series**: DUET, BEVBert, ScaleVLN, etc., serve as the primary baselines for comparison.
- **DUET**: The architectural foundation of the discrete version of VPNet.
- **ETPNav**: The architectural foundation of the continuous version of VPNet.
- **RoVI**: Utilizes hand-drawn symbols to guide robotic manipulation, but VPN is the first work to use visual prompts as the sole navigation instruction.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — A new navigation paradigm that fills an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablations, lacking real-world experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and intuitive diagrams.
- **Value**: ⭐⭐⭐⭐ — User-friendly interaction mode, though performance in continuous environments needs improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] FALIP: Visual Prompt as Foveal Attention Boosts CLIP Zero-Shot Performance](../../ECCV2024/3d_vision/falip_visual_prompt_as_foveal_attention_boosts_clip_zero-shot_performance.md)
- [\[CVPR 2026\] EfficientVPR: Toward Efficient Visual Place Recognition via Scene-Aware Prompt Tuning and Adaptive Feature Enhancement](../../CVPR2026/3d_vision/efficientvpr_toward_efficient_visual_place_recognition_via_scene-aware_prompt_tu.md)
- [\[ICLR 2026\] Towards Physically Executable 3D Gaussian for Embodied Navigation](../../ICLR2026/3d_vision/towards_physically_executable_3d_gaussian_for_embodied_navigation.md)
- [\[CVPR 2025\] Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation](../../CVPR2025/3d_vision/vid2sim_realistic_and_interactive_simulation_from_video_for_urban_navigation.md)
- [\[ICLR 2026\] OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation](../../ICLR2026/3d_vision/openfly_a_comprehensive_platform_for_aerial_vision-language_navigation.md)

</div>

<!-- RELATED:END -->
