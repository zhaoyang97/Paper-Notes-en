---
title: >-
  [Paper Note] TesserAct: Learning 4D Embodied World Models
description: >-
  [ICCV 2025][Robotics][4D world model] TesserAct is a 4D embodied world model that trains a video generative model to jointly predict RGB, depth, and normal videos, which are subsequently converted into high-quality 4D scenes, enabling spatiotemporally consistent 3D world dynamics simulation and robot action planning.
tags:
  - "ICCV 2025"
  - "Robotics"
  - "4D world model"
  - "embodied intelligence"
  - "video diffusion model"
  - "RGB-DN"
  - "joint depth-normal prediction"
  - "robot planning"
date: 2026-05-08
content_hash: 3fd41ff645b707a0
---

# TesserAct: Learning 4D Embodied World Models

**Conference**: ICCV 2025
**arXiv**: [2504.20995](https://arxiv.org/abs/2504.20995)  
**Area**: Robotics
**Keywords**: 4D world model, embodied intelligence, video diffusion model, RGB-DN, joint depth-normal prediction, robot planning

## TL;DR

TesserAct is a 4D embodied world model that trains a video generative model to jointly predict RGB, depth, and normal videos, which are subsequently converted into high-quality 4D scenes, enabling spatiotemporally consistent 3D world dynamics simulation and robot action planning.

## Background & Motivation

World models are central components of embodied intelligence. However, existing world models suffer from fundamental limitations:

- **2D pixel-space operation**: Methods such as UniPi and SuSIE operate in 2D, providing no accurate depth or pose information.
- **Physically implausible predictions**: 2D models may produce inconsistent object shapes across time steps.
- **High cost of 4D modeling**: Directly generating outputs in the 3D+time domain incurs prohibitive computational overhead.
- **Lack of 4D annotated data**: Large-scale robot datasets generally lack depth and surface normal annotations.

**Mechanism**: RGB-DN (RGB + depth + normal) video is used as a lightweight intermediate representation of the 4D world, enabling efficient construction of 4D world models via pretrained video generative models.

## Method

### Overall Architecture

Four core components: 4D dataset construction → RGB-DN generative model → 4D scene reconstruction → action planning.

### Key Designs

**1. 4D Embodied Video Dataset (~285k videos)**

- RLBench synthetic (80k): 20 tasks × 1,000 instances × 4 viewpoints, with precise depth + DSINE normals, Colosseum randomization.
- RT1 Fractal real-world (80k): depth via RollingDepth + normals via Marigold.
- Bridge (25k): annotated with the same pipeline.
- SomethingSomethingV2 (100k): hand-object interactions with diverse language instructions.

**2. Model Architecture (fine-tuned from CogVideoX)**

- A 3D VAE encodes RGB/depth/normal separately (VAE weights frozen).
- Three independent InputProj modules extract per-modality embeddings, which are summed and fed into the DiT backbone.
- RGB retains the original pathway; depth and normals are additionally decoded via Conv3D + DNProj.
- **Zero-initialization**: all newly added modules are zero-initialized, ensuring the training starting point is equivalent to CogVideoX.
- Text conditioning: "[action instruction] + [robot arm name]".
- Multi-resolution training; 49-frame prediction.

**3. 4D Scene Reconstruction**

- Normal integration optimizes depth: a perspective camera model constrains log-depth, solved iteratively via a quadratic loss.
- RAFT optical flow segments dynamic, static, and background regions.
- Temporal consistency loss: flow-guided inter-frame depth consistency with separate weighting for dynamic and background regions.
- Regularization loss: constrains optimized depth to remain close to generated depth.
- Total loss = spatial consistency + temporal consistency + regularization.

**4. Action Planning**

PointNet encodes the 4D point cloud → combined with text embeddings → MLP outputs 7-DoF actions.

### Loss & Training

- Video generation: standard denoising loss, jointly over RGB + depth + normal.
- 4D reconstruction: $L_s$ (normal integration) + $L_c$ (flow-guided temporal consistency) + $L_r$ (depth regularization).

## Key Experimental Results

### 4D Scene Generation

**Real-world domain (RT1 + Bridge):**

| Method | AbsRel↓ | Normal Mean↓ | Chamfer↓ |
|--------|---------|-------------|---------|
| OpenSora | 31.41 | 41.82 | 0.3013 |
| CogVideoX | 26.17 | 19.53 | 0.2191 |
| **TesserAct** | **22.07** | **15.74** | **0.2030** |

**Synthetic domain (RLBench):**

| Method | AbsRel↓ | Normal Mean↓ | Chamfer↓ |
|--------|---------|-------------|---------|
| CogVideoX | 19.81 | 20.36 | 0.2884 |
| **TesserAct** | **16.02** | **14.75** | **0.0811** |

### Robot Action Planning (RLBench success rate %)

| Method | close box | open drawer | open jar | open micro | put knife |
|--------|-----------|-------------|----------|------------|-----------|
| Image-BC | 53 | 4 | 0 | 5 | 0 |
| UniPi | 81 | 67 | 38 | 72 | 66 |
| **TesserAct** | **88** | **80** | **44** | **70** | **70** |

### Novel View Synthesis

| Method | PSNR | SSIM | Time |
|--------|------|------|------|
| SoM | 10.94 | 24.02 | ~2h |
| **TesserAct** | **12.99** | **42.62** | **~1min** |

### Key Findings

- Joint RGB-DN prediction substantially outperforms generating RGB first and applying post-hoc depth/normal estimation.
- Normal integration eliminates the planar reconstruction tilt artifact.
- Both the consistency loss and regularization loss are critical for 4D reconstruction quality.
- The model generalizes to unseen scenes and novel objects.

## Highlights & Insights

1. **RGB-DN intermediate representation**: retains 3D geometric information while remaining compatible with video generative models, enabling efficient training.
2. **Zero-initialization strategy**: carefully preserves the prior knowledge of the pretrained model.
3. **Normal-assisted depth optimization**: compensates for the lack of absolute scale in affine-invariant depth estimates.
4. **Cross-platform robot generalization**: robot arm identity is specified via text, allowing a single model to adapt to multiple platforms.
5. **Automatic 4D annotation**: off-the-shelf estimators convert existing video datasets into 4D training data.
6. **Optical flow-guided dynamic/static separation**: naturally disentangles dynamic and static regions in 4D reconstruction, applying separate constraints to each.

## Limitations & Future Work

- RGB generation quality is marginally lower than RGB-only fine-tuning (SSIM drops ~3.5%).
- Depth and normals are derived from estimators rather than sensors, introducing noise.
- Prediction is limited to a fixed viewpoint; true multi-view 4D generation is not supported.
- Affine-invariant depth lacks absolute metric scale.
- The training data volume (~285k) is considerably smaller than that of large-scale video foundation models.

## Related Work & Insights

- **Embodied foundation models**: VLA approaches such as RT-2 and Octo directly output actions.
- **Video world models**: UniPi, Genie, and related methods operate in 2D space.
- **4D generation**: methods such as DreamGaussian4D rely on SDS-based optimization, which is slow.
- **Depth optimization**: normal integration methods including GeoWizard and DSINE.

## Rating

- **Novelty**: ★★★★☆ — first to propose a 4D embodied world model; RGB-DN representation is original.
- **Technical Depth**: ★★★★☆ — complete closed loop from data construction to model training, reconstruction, and planning.
- **Experimental Thoroughness**: ★★★★☆ — evaluation on both real-world and synthetic domains with downstream task validation.
- **Practicality**: ★★★★☆ — demonstrable benefit for robot action planning.
- **Overall**: 8.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making](../../ICML2025/robotics/founder_grounding_foundation_models_in_world_models_for_open-ended_embodied_deci.md)
- [\[ICLR 2026\] World-In-World: World Models in a Closed-Loop World](../../ICLR2026/robotics/world-in-world_world_models_in_a_closed-loop_world.md)
- [\[CVPR 2025\] Neural Motion Simulator: Pushing the Limit of World Models in Reinforcement Learning](../../CVPR2025/robotics/neural_motion_simulator_pushing_the_limit_of_world_models_in_reinforcement_learn.md)
- [\[CVPR 2026\] Dexterous World Models](../../CVPR2026/robotics/dexterous_world_models.md)
- [\[ICML 2025\] SENSEI: Semantic Exploration Guided by Foundation Models to Learn Versatile World Models](../../ICML2025/robotics/sensei_semantic_exploration_guided_by_foundation_models_to_learn_versatile_world.md)

</div>

<!-- RELATED:END -->
