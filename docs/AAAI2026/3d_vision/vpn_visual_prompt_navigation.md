---
title: >-
  [Paper Note] VPN: Visual Prompt Navigation
description: >-
  [AAAI 2026][3D Vision][Visual Navigation] This paper proposes Visual Prompt Navigation (VPN), a novel navigation paradigm in which users annotate visual trajectories (keypoints connected by arrows) on 2D top-down maps to…
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Visual Navigation"
  - "Visual Prompt"
  - "Top-down View"
  - "Vision-Language Navigation"
  - "Data Augmentation"
date: 2026-05-08
content_hash: 781f7c3630239f6a
---

# VPN: Visual Prompt Navigation

**Conference**: AAAI 2026
**arXiv**: [2508.01766](https://arxiv.org/abs/2508.01766)  
**Code**: [github.com/farlit/VPN](https://github.com/farlit/VPN)  
**Area**: 3D Vision
**Keywords**: Visual Navigation, Visual Prompt, Top-down View, Vision-Language Navigation, Data Augmentation

## TL;DR

This paper proposes Visual Prompt Navigation (VPN), a novel navigation paradigm in which users annotate visual trajectories (keypoints connected by arrows) on 2D top-down maps to guide agent navigation, replacing natural language or image-goal instructions. Two datasets, R2R-VP and R2R-CE-VP, are constructed alongside a VPNet baseline model. Combined with view-level and trajectory-level data augmentation, the approach achieves strong performance in both discrete and continuous environments.

## Background & Motivation

### Limitations of Prior Navigation Paradigms

Visual navigation is a core research direction in AI and robotics. Prevailing paradigms include:

**PointGoal Navigation**: Provides the relative direction and distance to the goal, but lacks intermediate guidance.

**ImageGoal Navigation**: Supplies an image of the target location, but offers no intermediate navigation cues.

**Vision-Language Navigation (VLN)**: Describes the navigation path in natural language; currently the most active paradigm.

**Fundamental Dilemma of Natural Language Instructions**: Language is inherently **ambiguous** when describing object positions, directional changes, and distance relations; striving for precision inevitably leads to **verbosity**. This creates a fundamental trade-off in human–robot interaction.

### Advantages of Visual Prompts

The authors propose an intuitive insight: **drawing a route on a map** is the most natural way for humans to communicate navigation instructions. Key advantages:

**High User Accessibility**: Non-expert users can naturally specify navigation goals by clicking or sketching trajectories.

**Rich Spatial Information**: Top-down views inherently preserve complete spatial layouts.

**High Reusability**: Top-down views can be acquired once via drone imagery or 3D reconstruction and reused repeatedly.

## Method

### Overall Architecture

The core contributions of VPN encompass three components:
1. **Dataset Construction**: Replacing language instructions in R2R/R2R-CE with visual prompts.
2. **VPNet Model**: Built upon DUET/ETPNav architectures, substituting the language encoder with a ViT encoder.
3. **Data Augmentation Strategies**: View-level and trajectory-level augmentation.

### Key Designs

#### 1. **Visual Prompt Construction Pipeline**

Four-step generation: ① generate top-down view ② annotate trajectory by connecting waypoints with arrows ③ crop around the trajectory with a 60 px margin ④ remove black borders to tightly bound the visual prompt.

**Design Motivation**: Center cropping is a critical step. Ablation studies show that without cropping, different episodes within the same scene share an identical top-down map, causing the model to overfit to the scene rather than learning trajectory information (SR drops to only 31%).

#### 2. **VPNet Model Architecture**

Three core components:

**ViT Visual Prompt Encoder**: Uses ViT-B/16 (pretrained on ImageNet-21k) to encode 224×224 visual prompt images. For multi-floor scenarios, **Order-Aware Floor Concatenation (OAFC)** is applied:
$$\mathcal{P}_i^o = \text{ViT}(\mathcal{P}_i) + b_i, \quad \mathcal{P} = [\mathcal{P}_1^o, ..., \mathcal{P}_k^o]$$

**Node Embedding Module**: The agent incrementally builds a topological graph; each node is represented by panoramic view features (encoded via a two-layer Transformer), step embeddings, and position embeddings.

**Graph-Aware Cross-Modal Encoder**: A multi-layer cross-modal graph Transformer comprising cross-attention layers and Graph-Aware Self-Attention (GASA) layers:
$$\text{GASA}(X) = \text{Softmax}\left(\frac{XW_q(XW_k)^T}{\sqrt{d}} + EW_d\right)XW_v$$
where $E$ is the pairwise distance matrix of the topological graph.

#### 3. **Data Augmentation Strategies**

**Trajectory-Level Augmentation**: Incorporates PREVALENT (178k trajectories) and ScaleVLN (1.6M trajectories) to increase training data diversity.

**View-Level Augmentation**:
- **Prompt View Augmentation**: Random rotation of top-down views (0°/90°/180°/270°).
- **Agent View Augmentation**: Random sampling of the initial heading direction.

**Design Motivation**: In VPN, the initial heading is decoupled from the visual prompt (unlike VLN, where language may implicitly encode initial direction), making free rotation applicable.

### Loss & Training

- **Behavior Cloning + DAgger**: $\mathcal{L} = \lambda \mathcal{L}_{BC} + (1-\lambda) \mathcal{L}_{DAG}$, with $\lambda = 0.5$
- Discrete environment: single A5000 GPU, 400k iterations, batch=10, lr=1.5e-5
- Continuous environment: dual A5000 GPUs, 400k iterations, batch=16, lr=1e-5

## Key Experimental Results

### Main Results

**Discrete Environment (R2R-VP)**:

| Method | Training Data | Val Unseen SR↑ | Val Unseen SPL↑ | Test Unseen SR↑ |
|--------|--------------|----------------|-----------------|-----------------|
| DUET (VLN) | R2R+PRE+SCA | 81 | 70 | 80 |
| VPNet | R2R | 51.23 | 43.47 | 52.40 |
| VPNet | R2R+PRE | 65.92 | 56.17 | 66.38 |
| **VPNet** | **R2R+PRE+SCA** | **96.68** | **94.84** | **97.56** |

VPNet achieves 96.68% SR on Val Unseen, substantially outperforming DUET's 81%, using only 1/3 of the ScaleVLN trajectories.

**Continuous Environment (R2R-CE-VP)**:

| Method | Setting | Val Seen SR↑ | Val Unseen SR↑ |
|--------|---------|-------------|----------------|
| ETPNav (VLN) | R2R+PRE | 66 | 57 |
| VPNet | R2R+PRE | **84.11** | 47.96 |

### Ablation Study

**Effect of Different Visual Prompt Types (Discrete Environment)**:

| Prompt Type | Val Seen SR | Val Unseen SR | Notes |
|-------------|-------------|---------------|-------|
| Uncropped full top-down view | 31.68 | 33.94 | Overfits to scene |
| Cropped top-down view only | 83.56 | 45.83 | Resembles ImageNav |
| Cropped + arrows + text | 95.74 | 65.36 | Text occludes details |
| **Cropped + arrows** | **100** | **65.92** | Best |

**Effect of View-Level Augmentation**:

| Augmentation | Val Unseen SR↑ | SPL↑ |
|--------------|----------------|------|
| None | 86.33 | 82.92 |
| Agent view only | 88.18 | 85.02 |
| Prompt view rotation only | 96.41 | 94.37 |
| **Both combined** | **96.68** | **94.84** |

Prompt view rotation yields substantially larger gains than agent view augmentation (+10 SR vs. +2 SR).

### Key Findings

1. **Remarkable data efficiency of visual prompts**: VPNet achieves 96.68% SR with 1.6M trajectories, whereas DUET reaches only 81% SR with 4.9M trajectories.
2. **Cropping is critical**: Without cropping, the model severely overfits to scenes (31% vs. 100% Val Seen SR).
3. **Cropping alone (without trajectory annotation) is already effective**: The model can infer approximate destinations from the cropped region, resembling ImageNav.
4. **Moderate robustness to noise**: Under 20% salt-and-pepper noise, SR decreases from 96.68% to 90.34%.

## Highlights & Insights

1. **Paradigm Innovation**: VPN introduces a fundamentally new navigation paradigm, bridging the gap between language-guided and image-goal navigation.
2. **Data Efficiency Advantage**: Visual prompts convey spatial information at a far higher density than natural language instructions.
3. **Strong Practicality**: Top-down views can be acquired once via drone imagery or 3D reconstruction and reused across multiple deployments.
4. **Well-Designed Ablation Study**: Systematic analysis covers prompt types, augmentation strategies, encoder configurations, and multiple other dimensions.

## Limitations & Future Work

1. **Validation limited to simulated environments**: Experiments are conducted in MP3D/HM3D scenes without real-world testing.
2. **Dependence on high-quality top-down views**: Scenes with poor reconstruction quality are not amenable to this approach.
3. **Performance gap in continuous environments**: Val Unseen SR of only 47.96%, far below the 96.68% achieved in discrete environments.
4. **Coarse handling of multi-floor scenarios**: Floor features are naively concatenated.

## Related Work & Insights

- **VLN (R2R) Series**: DUET, BEVBert, and ScaleVLN serve as primary comparison baselines.
- **DUET**: Serves as the architectural foundation for the discrete-environment variant of VPNet.
- **ETPNav**: Serves as the architectural foundation for the continuous-environment variant of VPNet.
- **RoVI**: Uses hand-drawn symbols to guide robotic manipulation; however, VPN is the first work to employ visual prompts as the sole navigation instruction.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — A new navigation paradigm that fills an important gap.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablations, but lacks real-world experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — User-friendly interaction modality; continuous environment performance warrants further improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EfficientVPR: Toward Efficient Visual Place Recognition via Scene-Aware Prompt Tuning and Adaptive Feature Enhancement](../../CVPR2026/3d_vision/efficientvpr_toward_efficient_visual_place_recognition_via_scene-aware_prompt_tu.md)
- [\[CVPR 2026\] VirPro: Visual-referred Probabilistic Prompt Learning for Weakly-Supervised Monocular 3D Detection](../../CVPR2026/3d_vision/virpro_visual-referred_probabilistic_prompt_learning_for_weakly-supervised_monoc.md)
- [\[CVPR 2026\] A Survey of Spatial Memory Representations for Efficient Robot Navigation](../../CVPR2026/3d_vision/a_survey_of_spatial_memory_representations_for_efficient_robot_navigation.md)
- [\[ICLR 2026\] OpenFly: A Comprehensive Platform for Aerial Vision-Language Navigation](../../ICLR2026/3d_vision/openfly_a_comprehensive_platform_for_aerial_vision-language_navigation.md)
- [\[AAAI 2026\] MR-CoSMo: Visual-Text Memory Recall and Direct Cross-Modal Alignment Method for Query-Driven 3D Segmentation](mr-cosmo_visual-text_memory_recall_and_direct_cross-modal_alignment_method_for_q.md)

</div>

<!-- RELATED:END -->
