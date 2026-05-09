---
title: >-
  [Paper Note] Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting
description: >-
  [CVPR 2026][Self-Supervised Learning][Monocular Depth Estimation] This paper proposes Re-Depth Anything, which refines depth predictions from Depth Anything V2/3 at inference time through self-supervised optimization: the predicted depth map is augmented via re-lighting, and a 2D diffusion model's SDS loss is used to guide the optimization without any labeled data.
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - Monocular Depth Estimation
  - Test-Time Optimization
  - Score Distillation Sampling
  - Re-lighting
  - Depth Anything
date: 2026-05-08
content_hash: dcb87e5103267d1b
---

# Re-Depth Anything: Test-Time Depth Refinement via Self-Supervised Re-lighting

**Conference**: CVPR 2026
**arXiv**: [2512.17908](https://arxiv.org/abs/2512.17908)
**Authors**: Ananta R. Bhattarai, Helge Rhodin (Bielefeld University)
**Code**: [GitHub](https://github.com/anantarb/Re-Depth-Anything)
**Area**: Self-Supervised
**Keywords**: Monocular Depth Estimation, Test-Time Optimization, Score Distillation Sampling, Re-lighting, Depth Anything

## TL;DR

This paper proposes Re-Depth Anything, which refines depth predictions from Depth Anything V2/3 at inference time through self-supervised optimization: the predicted depth map is augmented via re-lighting, and a 2D diffusion model's SDS loss is used to guide the optimization without any labeled data.

## Background & Motivation

Foundation models such as Depth Anything V2 (DA-V2) deliver strong performance, yet they still exhibit errors on in-the-wild images with large distributional shifts from training data—e.g., lighting bias causes loss of fine structures, and flat regions suffer from spurious noise. Existing test-time adaptation methods either require multi-frame temporal information or rely on specific external priors (3D meshes or sparse point clouds). Meanwhile, large-scale 2D diffusion models encode rich physical-world priors that have not been fully exploited for test-time depth refinement.

## Core Problem

How can 2D diffusion model priors be leveraged to perform **label-free** test-time refinement of depth maps from a **single image**?

## Method

### Mechanism: Re-lighting vs. Photometric Reconstruction

The key innovation is that the method deliberately **avoids** reconstructing the true lighting and material properties of the input image—an extremely ill-posed inverse rendering problem—and instead **augments** the input image. Specifically, the depth map is randomly re-lit and composited onto the original image, and a diffusion model is used to assess whether the augmented image "looks plausible."

### 4.1 Depth-Illumination Rendering

Surface normals $\mathbf{N}$ are computed from the disparity map $\hat{D}_{\text{disp}}$, and then re-lighting is applied using the Blinn-Phong model:

$$\hat{\mathbf{I}} = \tau\left(\beta_1 \max(\mathbf{N} \cdot \mathbf{l}, 0) \odot \tau^{-1}(\mathbf{I}) + \beta_2 \max(\mathbf{N} \cdot \mathbf{h}, 0)^\alpha\right)$$

- $\mathbf{l}$: randomly sampled light direction
- $\beta_1, \beta_2$: diffuse/specular reflection intensities
- $\tau(\cdot) = (\cdot)^{1/\gamma}$: tone mapping ($\gamma=2.2$)
- The input image $\mathbf{I}$ serves as an approximation of the diffuse albedo

**Handling relative depth**: DA-V2 outputs normalized disparity; surface normals are invariant to global scale, so only the offset parameter $b = ms$ needs to be optimized (in practice, fixing $b=0.1$ suffices).

### 4.2 Augmentation Objective

At each step, $\mathbf{l}$, $\beta_1$, $\beta_2$, and $\alpha$ are randomly sampled, and the SDS loss measures the plausibility of the augmented image:

$$\mathcal{L} = \mathcal{L}_{\text{SDS}}(\hat{\mathbf{I}}, c) + \frac{\lambda_1}{hw} \sum_{i,j} \|\Delta \hat{D}_{\text{disp}}^{i,j}\|^2$$

- SDS loss: a diffusion model (Stable Diffusion v1.5) evaluates the naturalness of the re-lit image
- Smoothness regularization: suppresses depth noise
- Text condition $c$: automatically generated from the input image via BLIP-2

### 4.3 Optimization Strategy

Directly optimizing the depth tensor or fine-tuning the full model both fail. The **key design** is to optimize only the intermediate embeddings $\mathbf{W}$ and the DPT decoder weights $\theta$, while freezing the ViT encoder:

$$\mathbf{W}^*, \theta^* = \arg\min_{\mathbf{W}, \theta} \mathcal{L}(\hat{\mathbf{I}}, c, \hat{D}_{\text{disp}})$$

- The frozen ViT encoder preserves its strong geometric priors
- Decoder weight optimization enables structural adjustments
- Embedding optimization enables per-image customization

**Ensemble prediction**: Due to the stochasticity of the SDS loss, $N=10$ independent optimization runs are performed and their results are averaged; most of the gain is already achieved with 3 runs.

### Implementation Details

- 1000 AdamW iterations; embedding learning rate $10^{-3}$, DPT weight learning rate $2 \times 10^{-6}$
- Scaled orthographic projection (default); regularization $\lambda_1 = 1.0$
- Approximately 80 seconds per single run (RTX 5000)

## Key Experimental Results

| Dataset | Method | AbsRel ↓ | RMSE ↓ | SI log ↓ | SqRel ↓ |
|--------|------|---------|--------|---------|---------|
| KITTI | DA-V2 | 0.305 | 7.01 | 33.6 | 2.49 |
| | **Ours + DA-V2** | **0.283** | **6.71** | **30.7** | **2.20** |
| | Relative Gain | 7.10% | 4.29% | 8.51% | 11.4% |
| ETH3D | DA-V2 | 0.113 | 0.955 | 15.1 | 0.391 |
| | **Ours + DA-V2** | **0.104** | **0.875** | **14.1** | **0.347** |
| | Relative Gain | 8.30% | 8.39% | 6.22% | 11.1% |

| Dataset | Method | AbsRel ↓ | SqRel ↓ | Normal MSE ↓ |
|--------|------|---------|---------|-------------|
| CO3D | DA3 | 0.00251 | 0.000317 | 0.000479 |
| | **Ours + DA3** | **0.00238** | **0.000294** | **0.000409** |
| | Relative Gain | 4.83% | 7.39% | **14.65%** |

*Consistent improvements are also obtained on DA3, with normal error reduction reaching 14.7%.*

## Highlights & Insights

- **Re-lighting instead of photometric reconstruction**: elegantly sidesteps the ill-posedness of inverse rendering, repurposing diffusion priors as an "augmentation plausibility check"
- **No labels, multi-view data, or additional supervision required**: purely single-image self-supervised
- **Backbone-agnostic**: effective across both DA-V2 and DA3
- **Notable fine-detail enhancement**: recovers spherical textures, balcony railings, and other fine structures; removes spurious noise on flat surfaces
- **Theoretically elegant**: transfers the SfS+SDS paradigm from DreamFusion's text-to-3D setting to depth refinement

## Limitations & Future Work

- Occasional hallucinated edges (e.g., stickers on a truck misinterpreted as geometric features)
- Sky regions may have geometry incorrectly extended
- Dark regions (e.g., under tree shadows) may be over-smoothed
- The 10-run ensemble requires approximately 13 minutes, which is prohibitive for real-time applications
- Stable Diffusion v1.5 is used as the diffusion prior, which may no longer be the optimal choice

## Related Work & Insights

- vs. **DreamFusion/RealFusion**: these methods perform full 3D reconstruction with photometric consistency; this work only applies re-lighting augmentation, avoiding the stringent requirements of photometric reconstruction
- vs. **classical Shape-from-Shading (SfS)**: SfS assumes constant albedo or known illumination and fails in many practical cases; this work replaces hand-crafted assumptions with diffusion priors
- vs. **Marigold**: Marigold uses a diffusion model directly for depth estimation; this work uses a diffusion model for test-time optimization, making it complementary to feed-forward methods
- vs. **multi-frame TTA**: the proposed single-image approach requires no temporal information, broadening its applicability

- The re-lighting augmentation + SDS paradigm is generalizable to other geometry tasks such as normal estimation and surface reconstruction
- "Augment rather than reconstruct" represents an important strategic shift for handling ill-posed inverse problems
- The design of freezing the encoder while optimizing embeddings and the decoder is transferable to test-time adaptation of other foundation models

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Self-supervised depth refinement via re-lighting SDS is entirely novel
- Experimental Thoroughness: ⭐⭐⭐⭐ — Three datasets, two backbones (DA-V2 and DA3), and detailed ablation studies
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear motivation, elegant methodology, and thorough analysis
- Value: ⭐⭐⭐⭐ — Opens a new direction for test-time refinement of foundation models

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] AcTTA: Rethinking Test-Time Adaptation via Dynamic Activation](actta_rethinking_test-time_adaptation_via_dynamic_activation.md)
- [\[CVPR 2026\] A Stitch in Time: Learning Procedural Workflow via Self-Supervised Plackett-Luce Ranking](a_stitch_in_time_learning_procedural_workflow_via_self_supervised_plackett_luce_r.md)
- [\[CVPR 2026\] MINE-JEPA: In-Domain Self-Supervised Learning for Mineral Exploration](mine-jepa_in-domain_self-supervised_learning_for_mine-like_object_classification.md)
- [\[CVPR 2026\] Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning](group_dinomics_incorporating_people_dynamics_into_dino_for_self_supervised_group_activity_feature_learning.md)
- [\[CVPR 2026\] TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation](teflow_enabling_multi-frame_supervision_for_self-supervised_feed-forward_scene_f.md)

<!-- RELATED:END -->
