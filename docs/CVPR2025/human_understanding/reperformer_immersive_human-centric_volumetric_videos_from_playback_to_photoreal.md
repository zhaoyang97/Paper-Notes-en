---
title: >-
  [Paper Note] RePerformer: Immersive Human-centric Volumetric Videos from Playback to Photoreal Reperformance
description: >-
  [CVPR 2025][Human Understanding][Volumetric Videos] RePerformer is proposed, a 3DGS-based volumetric video representation that unifies high-fidelity playback and photorealistic reperformance under novel poses through hierarchical decoupling of motion and appearance Gaussians, Morton coding parameterization, and a semantic-aware alignment module.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Volumetric Videos"
  - "3D Gaussian Splatting"
  - "Motion Transfer"
  - "Morton Coding"
  - "Non-rigid Reconstruction"
date: 2026-05-08
content_hash: 3ed29f95bc5b6d7c
---

# RePerformer: Immersive Human-centric Volumetric Videos from Playback to Photoreal Reperformance

**Conference**: CVPR 2025  
**arXiv**: [2503.12242](https://arxiv.org/abs/2503.12242)  
**Code**: [Project Page](https://moqiyinlun.github.io/Reperformer/)  
**Area**: Human Understanding  
**Keywords**: Volumetric Videos, 3D Gaussian Splatting, Motion Transfer, Morton Coding, Non-rigid Reconstruction

## TL;DR

RePerformer is proposed, a 3DGS-based volumetric video representation that unifies high-fidelity playback and photorealistic reperformance under novel poses through hierarchical decoupling of motion and appearance Gaussians, Morton coding parameterization, and a semantic-aware alignment module.

## Background & Motivation

Human-centric volumetric video allows users to freely control the virtual camera perspective, which has important applications in telepresence, education, and entertainment. Currently, there are two complementary workflows: (1) playback-based methods can reconstruct dynamic scenes with high fidelity but cannot generalize to novel motions; (2) animatable methods (human avatars) can drive novel motions but heavily rely on parametric models such as SMPL and are mainly designed for pure human scenes.

This paper explores a new direction—the "playback-to-reperformance" paradigm: given a dense multi-view video of a dynamic sequence, the goal is to not only achieve precise free-viewpoint playback but also photorealistically reperform the entire scene (including human-object interactions) under similar but unseen novel motions. This setup requires the method to possess both high-fidelity rendering capability and generalization capability to novel motions, and must handle general non-rigid scenes instead of being limited to human bodies.

Existing animatable methods rely on the SMPL model and cannot handle human-object interaction scenes, while playback methods lack generalization capability. RePerformer simultaneously satisfies both requirements by decoupling motion and appearance and leveraging the generalization capability of 2D CNNs.

## Method

### Overall Architecture

RePerformer is a three-stage pipeline: (1) **Tracking Stage**—decoupling the dynamic scene into sparse motion Gaussians (~50K) and dense appearance Gaussians (~200K), driving the non-rigid deformation of appearance Gaussians via motion Gaussians to achieve topologically consistent tracking; (2) **Training Stage**—mapping appearance Gaussians to 2D position maps via Morton coding, and using a U-Net to learn a generalizable mapping from position maps to attribute maps; (3) **Reperformance Stage**—associating the motion Gaussians of a novel performer with the original appearance Gaussians via a semantic-aware alignment module to achieve motion transfer.

### Key Designs

#### Key Design 1: Hierarchical Motion-Appearance Decoupling

**Function**: Decoupling the dynamic scene into a topologically consistent motion representation and a generalizable appearance representation.

**Mechanism**: Sparse motion Gaussians optimize only position and rotation to capture global non-rigid motion, maintaining local rigidity via an as-rigid-as-possible (ARAP) constraint. Dense appearance Gaussians are initialized in the canonical space and associated with motion Gaussians via nearest neighbor search. Deformation is achieved via weighted interpolation: $p_{i,t}^{\mathcal{T}} = \sum_{k \in \mathcal{N}} w(p_i, p_k)(R(\Delta q_k) p_i + \Delta p_k)$.

**Design Motivation**: Hierarchical decoupling allows motion capture and appearance rendering to perform their respective duties—motion Gaussians are responsible for the generalization of geometric deformation, while appearance Gaussians are responsible for high-fidelity rendering. This design is similar to the traditional concept of Embedded Deformation Graph + Mesh Tracking, but replaces them with 3DGS.

#### Key Design 2: Morton Code Parameterization

**Function**: Efficiently encoding 3D appearance Gaussians into 2D position/attribute maps, preserving spatial proximity to support 2D CNN learning.

**Mechanism**: Quantizing the positions of appearance Gaussians in the canonical space and sorting them via Morton order (Z-order curve), interleaving the binary representations of the 3D coordinates to preserve 3D spatial continuity. Each Gaussian $i$ is assigned a $(u,v)$ coordinate, forming a spatial proximity-preserving mapping $i \to (u,v)$ that remains consistent across all frames.

**Design Motivation**: The UV atlas of SMPL cannot represent human-object interaction scenes. Morton coding is a general-purpose 3D-to-2D mapping that does not depend on any parametric human model and can handle non-rigid scenes of arbitrary topology. Meanwhile, it preserves local spatial consistency, which is beneficial for the convolution operations of 2D CNNs.

#### Key Design 3: Semantic-Aware Motion Transfer

**Function**: Transferring the motion of a novel performer to the appearance Gaussians of the original scene to achieve topology-preserving reperformance.

**Mechanism**: Using Language-SAM + GroundingDINO + SAM2 to assign semantic labels to Gaussians (e.g., head, hands, feet, etc.), establishing coarse alignment between two sequences via K-means clustering. Then, motion transfer is performed via the optimization objective $E_{\text{re}} = \mathcal{L}_2(\mathcal{G}_t^{s'}, f(\mathcal{G}_c^{s'}, \mathcal{G}_t^r)) + \lambda_2 E_{\text{arap}}$ while maintaining the original topology of appearance Gaussians.

**Design Motivation**: Traditional deformation transfer requires manually specifying mesh correspondences, which is unfeasible for large-scale Gaussian point clouds. Semantic-aware alignment automatically establishes correspondences between body parts across the two sequences, while ARAP constraints ensure topology preservation during the deformation process.

### Loss & Training

Tracking stage: $E_{\text{init}} = \lambda_{iso} E_{\text{iso}} + \lambda_{size} E_{\text{size}} + E_{\text{color}}$ plus ARAP constraints. Training stage: pre-training uses $\mathcal{L}_2$ to supervise attribute regression, and main training uses $(1-\lambda_{\text{color}}) \mathcal{L}_1 + \lambda_{\text{color}} \mathcal{L}_{\text{D-SSIM}}$. Reperformance stage: alignment loss + semantic loss + ARAP regularization.

## Key Experimental Results

### Main Results: Novel View Synthesis (DualGS Dataset, 500 frames)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Training Time (min/frame) ↓ |
|------|--------|--------|---------|------------------|
| NeuS2 | 29.59 | 0.967 | 0.056 | 3.23 |
| Spacetime Gaussian | 31.69 | 0.981 | 0.029 | 2.24 |
| DualGS | **35.51** | **0.990** | 0.019 | 12.22 |
| **RePerformer** | 34.57 | 0.986 | 0.023 | **1.68** |

### Generalization Experiment: Novel Motion Rendering (3000 frames, 2500 train/500 test)

| Method | Novel View PSNR | Novel View SSIM | Novel Motion PSNR | Novel Motion SSIM |
|------|-----------|-----------|-----------|-----------|
| AP-NeRF | 28.26 | 0.939 | 26.85 | 0.944 |
| TAVA | 21.57 | - | - | - |
| **RePerformer** | **33.57** | **0.979** | **32.88** | **0.973** |

### Key Findings

- RePerformer ranks second only to per-frame-optimized DualGS in playback quality (with a gap of ~1 dB PSNR), but runs 7.3 times faster in training speed (1.68 vs 12.22 min/frame).
- It significantly outperforms all baseline methods in novel motion generalization, with an improvement of over 5 dB in PSNR, demonstrating the generalization capability of Morton coding + U-Net.
- It successfully handles complex human-object interaction scenes (violin playing, balloon interaction, etc.), which is unfeasible for SMPL-dependent methods.

## Highlights & Insights

1. **New Paradigm Definition**: Proposes the "playback-to-reperformance" paradigm for the first time, filling the gap between playback and animatable methods, with strong practical application value.
2. **Morton Coding instead of UV Atlas**: Replaces parametric model-dependent UV mappings with space-filling curves, enabling the method to handle arbitrary non-rigid scenes.
3. **CNN Generalization instead of Per-Frame Optimization**: Utilizes a 2D CNN to learn the mapping from positions to attributes, which is fast to train and possesses generalization capabilities.

## Limitations & Future Work

- Reperformance only supports "similar" novel motions; poses with large differences may produce artifacts.
- Semantic alignment requires text prompts to specify body parts, showing limited automation.
- It relies on dense multi-view video input (up to 81 views), which places high demands on the capture apparatus.
- Although Morton coding preserves local consistency, it may still map spatially close Gaussians to distant UV coordinates.

## Related Work & Insights

- **DualGS**: Playback SOTA, whose Joint+Skin dual-Gaussian design inspired the motion-appearance decoupling of this work.
- **AnimatableGaussians**: The idea of predicting Gaussian attributes from front/back maps inspired the Morton coding + CNN regression.
- **Sumner et al. (Deformation Transfer)**: The classical deformation transfer method is extended to Gaussian point clouds.

## Rating

⭐⭐⭐⭐ — The definition of the new paradigm is highly valuable, the Morton coding parameterization is a highlight, and the technical solution is complete. Playback quality is close to per-frame-optimized SOTA, and generalization capability is strong. The limitations lie in the reliance on dense multi-view inputs and the "similarity" constraint on novel motions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] GaussianIP: Identity-Preserving Realistic 3D Human Generation via Human-Centric Diffusion Prior](gaussianip_identity-preserving_realistic_3d_human_generation_via_human-centric_d.md)
- [\[CVPR 2025\] HumanMM: Global Human Motion Recovery from Multi-shot Videos](humanmm_global_human_motion_recovery_from_multi-shot_videos.md)
- [\[ICLR 2026\] Sapiens2: High-Resolution Foundation Models for Human-Centric Vision](../../ICLR2026/human_understanding/sapiens2.md)
- [\[ICCV 2025\] What's Making That Sound Right Now? Video-centric Audio-Visual Localization](../../ICCV2025/human_understanding/whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)
- [\[CVPR 2026\] UniDex: A Robot Foundation Suite for Universal Dexterous Hand Control from Egocentric Human Videos](../../CVPR2026/human_understanding/unidex_a_robot_foundation_suite_for_universal_dexterous_hand_control_from_egocen.md)

</div>

<!-- RELATED:END -->
