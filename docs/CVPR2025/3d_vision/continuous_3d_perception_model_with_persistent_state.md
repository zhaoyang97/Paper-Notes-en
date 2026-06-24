---
title: >-
  [Paper Note] Continuous 3D Perception Model with Persistent State
description: >-
  [CVPR 2025][3D Vision][Online 3D Reconstruction] Proposes CUT3R (Continuous Updating Transformer for 3D Reconstruction), a recurrent model that maintains a persistent internal state, allowing online, incremental metric-scale 3D reconstruction and camera pose estimation from image streams, while enabling inference of 3D structures in unobserved regions.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Online 3D Reconstruction"
  - "Continuous Perception"
  - "Recurrent State"
  - "Pointmap Prediction"
  - "Dynamic Scenes"
date: 2026-05-08
content_hash: adcd326b5656c5a5
---

# Continuous 3D Perception Model with Persistent State

**Conference**: CVPR 2025  
**arXiv**: [2501.12387](https://arxiv.org/abs/2501.12387)  
**Code**: [cut3r.github.io](https://cut3r.github.io/)  
**Area**: 3D Vision / 3D Reconstruction  
**Keywords**: Online 3D Reconstruction, Continuous Perception, Recurrent State, Pointmap Prediction, Dynamic Scenes

## TL;DR

Proposes CUT3R (Continuous Updating Transformer for 3D Reconstruction), a recurrent model that maintains a persistent internal state, allowing online, incremental metric-scale 3D reconstruction and camera pose estimation from image streams, while enabling inference of 3D structures in unobserved regions.

## Background & Motivation

### Background

**Background**: Traditional 3D reconstruction methods (SfM, SLAM, NeRF, etc.) process each scene from scratch (tabula rasa), which makes it difficult to handle sparse observations or degenerate motions.

### Limitations of Prior Work

**Limitations of Prior Work**: Learning-based methods like DUSt3R can only handle image pairs, making scaling up to multi-view scenarios require time-consuming global alignment and preventing online updates.

### Key Challenge

**Key Challenge**: Although Spann3R supports continuous reconstruction, its spatial memory only acts as a cache and cannot infer unobserved regions.

### Core Idea

**Core Idea**: Humans are online visual learners: continuously processing visual streams, accumulating observations, and leveraging prior knowledge to infer occluded regions.

### Supplementary Notes

**Supplementary Notes**: A unified framework is needed to: (1) reconstruct 3D from sparse observations, (2) continuously refine with new observations, and (3) infer unobserved regions.

## Method

### Overall Architecture

CUT3R uses a ViT encoder to encode each frame into visual tokens, which interact bi-directionally with a set of persistently maintained state tokens—state update (integrating new information into the state) and state readout (retrieving historical context from the state), yielding dense pointmaps in the world and camera coordinate systems, as well as camera poses.

### Key Designs

1. **Persistent State Mechanism**:
    - **Function**: Encodes accumulated scene understanding into a fixed-size set of tokens.
    - **Mechanism**: The state consists of 768 learnable tokens of 768 dimensions, initialized and shared. Image tokens interact bi-directionally with the state through two interconnected Transformer decoders: state update integrates new observations, and state readout extracts historical context.
    - **Design Motivation**: The fixed-size compressed state not only caches observed content but also encodes inferences of unobserved regions, akin to a human mental model of the environment.

2. **Unseen View Query**:
    - **Function**: Infers the 3D structure and color of unseen regions from the state without adding new observations.
    - **Mechanism**: Encodes the intrinsic/extrinsic parameters of a virtual camera into a raymap (a 6-channel image where each pixel encodes the ray origin and direction), which is encoded by a ViT and then interacts with the state for readout (without updating the state) to predict the corresponding pointmaps and colors.
    - **Design Motivation**: Analogous to patch completion in MAE, this is image-level completion that exploits the global scene context encoded in the state.

3. **Redundant Prediction and Partially Labeled Training**:
    - **Function**: Predicts two pointmaps (in self-coordinate and world-coordinate systems) and 6-DoF poses.
    - **Mechanism**: $\hat{X}_t^{\text{self}}, \hat{X}_t^{\text{world}}, \hat{P}_t$ are output by HeadSelf (DPT), HeadWorld (DPT+pose token), and HeadPose (MLP), respectively.
    - **Design Motivation**: Seemingly redundant but simplifies training—each output is independently supervised, and it supports training on datasets with only poses or only single-view depth.

### Loss & Training

- **3D Regression Loss**: Confidence-aware pointmap regression $\mathcal{L}_{conf} = \sum (c \cdot \|\hat{x}/\hat{s} - x/s\|_2 - \alpha \log c)$
- **Pose Loss**: L2 loss on quaternions and translations
- **Color Loss**: MSE reconstruction loss during raymap querying
- **Curriculum Training**: 4 stages—(1) 4-view static scenes at 224×224, (2) adding dynamic scenes and partial annotations, (3) high resolution (maximum edge of 512), (4) freezing the encoder and training on 4-64 view long sequences.
- 32 training datasets covered, spanning synthetic/real, static/dynamic, and indoor/outdoor scenarios.

## Key Experimental Results

### Main Results

| Dataset | Method | Abs Rel ↓ | δ<1.25 ↑ |
|--------|------|-----------|----------|
| Bonn | DUSt3R | 0.141 | 82.5 |
| Bonn | MonST3R | 0.076 | 93.9 |
| Bonn | **CUT3R** | **0.063** | **96.2** |
| NYU-v2 | DUSt3R | 0.080 | 90.7 |
| NYU-v2 | **CUT3R** | **0.086** | **90.9** |
| KITTI | MASt3R | 0.079 | 94.7 |
| KITTI | **CUT3R** | 0.092 | 91.3 |

### Ablation Study

| Configuration | Key Metrics | Explanation |
|------|---------|------|
| No persistent state (image pairs only) | Multi-view inconsistency | Degenerates to DUSt3R-style pairwise prediction |
| No raymap query | Unable to infer unseen regions | Lacks scene prior inference capabilities |
| Short-sequence training (no Stage 4) | Performance drop on long sequences | Insufficient long-context reasoning |

### Key Findings

- Online processing speed of 16.58 FPS (A100, 512×144), significantly faster than methods requiring global alignment.
- Achieves SOTA on single-frame depth estimation on Bonn and NYU-v2.
- Video depth estimation competes with or outperforms methods requiring global alignment on multiple datasets.
- Seamlessly handles both static and dynamic scenes (including moving people/objects).
- Supports metric scale prediction (unlike DUSt3R's relative scale).

## Highlights & Insights

- Modeling 3D reconstruction as a "continuous perception" problem with a fixed-size state token is an elegant design.
- Unseen view queries, analogous to MAE's image-level completion, are conceptually simple and highly effective.
- Extremely flexible: treats video streams, unordered photo collections, and static/dynamic scenes in a unified manner.
- The redundant output design, while seemingly wasteful, is ingenious, maximizing training data utilization.
- Upgrading from DUSt3R's pairwise-to-global-alignment paradigm to an online recurrent paradigm is a major step forward for 3D reconstruction.

## Limitations & Future Work

- Fixed-size state tokens may limit the representation capacity for extremely large scenes.
- Training requires 8×A100, which has high computational costs.
- The quality of dynamic scene reconstruction is still inferior to specialized dynamic SLAM methods.
- The quality of unseen view queries depends heavily on the amount of information accumulated in the state.
- Future work can explore adaptive state sizes or hierarchical state representations.

## Related Work & Insights

- Directly inherits the pointmap prediction paradigm of DUSt3R/MASt3R, but extends it into an online recurrent architecture.
- Contemporary with Spann3R but goes a step further: the state not only caches observations but also infers unseen regions.
- Complementary to MonST3R: Ours processes arbitrary-length sequences online, while the latter extends to dynamic scenes.
- Insight: 3D perception can be modeled as a continuous process akin to "reading", constantly updating a mental model.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The unified framework of persistent state + unseen query is highly innovative in the field of 3D reconstruction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across multiple datasets for multiple tasks, including depth, pose, and reconstruction.
- Writing Quality: ⭐⭐⭐⭐⭐ Smooth narrative, clear analogies (human vision, MAE), and elegant figures/tables.
- Value: ⭐⭐⭐⭐⭐ The unified framework addresses various 3D tasks, and its online characteristic holds broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PartRM: Modeling Part-Level Dynamics with Large Cross-State Reconstruction Model](partrm_modeling_part-level_dynamics_with_large_cross-state_reconstruction_model.md)
- [\[CVPR 2025\] Mesh Mamba: A Unified State Space Model for Saliency Prediction in Non-Textured and Textured Meshes](mesh_mamba_a_unified_state_space_model_for_saliency_prediction_in_non-textured_a.md)
- [\[NeurIPS 2025\] MVSMamba: Multi-View Stereo with State Space Model](../../NeurIPS2025/3d_vision/mvsmamba_multi-view_stereo_with_state_space_model.md)
- [\[CVPR 2025\] DSPNet: Dual-vision Scene Perception for Robust 3D Question Answering](dspnet_dual-vision_scene_perception_for_robust_3d_question_answering.md)
- [\[CVPR 2025\] SphereUFormer: A U-Shaped Transformer for Spherical 360 Perception](sphereuformer_a_u-shaped_transformer_for_spherical_360_perception.md)

</div>

<!-- RELATED:END -->
