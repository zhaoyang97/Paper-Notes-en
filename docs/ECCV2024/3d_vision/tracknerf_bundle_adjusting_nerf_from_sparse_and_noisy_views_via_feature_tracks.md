---
title: >-
  [Paper Note] TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks
description: >-
  [ECCV 2024][3D Vision] This paper proposes TrackNeRF, which integrates feature tracks from SfM into NeRF training. By replacing traditional pairwise correspondence losses with a global multi-view reprojection consistency loss, TrackNeRF significantly improves NeRF reconstruction quality and pose optimization accuracy under sparse views with noisy poses.
tags:
  - "ECCV 2024"
  - "3D Vision"
date: 2026-05-08
content_hash: 1394d82c9f7e9f24
---

# TrackNeRF: Bundle Adjusting NeRF from Sparse and Noisy Views via Feature Tracks

**Conference**: ECCV 2024  
**arXiv**: [2408.10739](https://arxiv.org/abs/2408.10739)  
**Code**: [Project Page](https://tracknerf.github.io/)  
**Area**: 3D Vision

## TL;DR

This paper proposes TrackNeRF, which integrates feature tracks from SfM into NeRF training. By replacing traditional pairwise correspondence losses with a global multi-view reprojection consistency loss, TrackNeRF significantly improves NeRF reconstruction quality and pose optimization accuracy under sparse views with noisy poses.

## Background & Motivation

### Key Challenge

**Key Challenge**: NeRF typically assumes a large number of images with precise poses, but in practical scenarios, views are often sparse and poses contain noise.

### Background

**Background**: BARF only performs frequency modulation without utilizing multi-view constraints, which is inherently not true bundle adjustment.

### Limitations of Prior Work

**Limitations of Prior Work**: SPARF introduces pairwise correspondence losses, but it only considers local consistency between two views and neglects global geometric constraints.

### Mechanism

**Core Problem**: Since all views originate from the same 3D scene, corresponding pixels should project back to the same 3D landmark, which requires global consistency rather than pairwise consistency.

## Method

### Overall Architecture

1. Extract dense correspondences between all image pairs using PDCNet++.
2. Construct feature tracks across all views via transitive chain propagation.
3. Perform track keypoint adjustment (TKA) on the feature tracks following the PixSfM approach.
4. Design a Track Reprojection Loss to jointly optimize NeRF parameters and camera poses.

### Key Designs

**Feature Track Extraction**: If $(u,v)$ matches between images $i$ and $j$, and $(v,q)$ matches between $j$ and $k$, a transitive relationship is established to form a connected track $T_k = \{u, v, q, \dots\}$.

**Track Keypoint Adjustment**: Minimize the weighted feature distance of all pairwise features within a track, optimizing keypoint locations via numerical gradients to obtain more accurate supervision signals.

**Track Reprojection Loss (Core)**: For each correspondence pair $(u_i, v_j)$ in track $T_k$, back-project $v_j$ to 3D via the rendered depth, and then reproject it to the image plane of $i$ to minimize the distance to $u_i$. Huber loss is used to enhance robustness.

**Depth Regularization**: Encourage depth gradients to align with rendered image gradients, mitigating geometric ambiguities and floater artifacts under sparse views.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{Photometric} + \lambda_{Depth}\mathcal{L}_{Depth} + \lambda_{Track}\mathcal{L}_{Track}$$

Photometric loss ensures appearance consistency; depth regularization mitigates geometric ambiguity; and track loss enforces global geometric consistency.

## Key Experimental Results

### Main Results

DTU dataset, 15% Gaussian noise on poses:

| Setting | Method | Rot.↓ | Trans.↓ | PSNR↑ | SSIM↑ | DE↓ |
|------|------|-------|---------|-------|-------|-----|
| 3-view | SPARF | 1.81 | 5.0 | 17.74 | 0.71 | 0.12 |
| 3-view | **TrackNeRF** | **1.12** | **2.48** | **18.53** | **0.73** | **0.11** |
| 6-view | SPARF | 1.31 | 2.7 | 21.39 | 0.81 | 0.09 |
| 6-view | **TrackNeRF** | **0.24** | **0.65** | **22.78** | **0.84** | **0.06** |
| 9-view | SPARF | 1.15 | 2.55 | 24.69 | 0.88 | 0.06 |
| 9-view | **TrackNeRF** | **0.25** | **0.70** | **25.57** | **0.89** | **0.05** |

### Ablation Study

DTU 3-view GT pose comparison (partial):

| Method | PSNR↑(masked) | SSIM↑(masked) | LPIPS↓(masked) |
|------|---------------|---------------|----------------|
| FreeNeRF | 20.46 | 0.83 | 0.17 |
| CorresNeRF | 20.58 | 0.77 | - |
| SPARF | 21.22 | 0.85 | 0.12 |
| **TrackNeRF** | **21.70** | **0.85** | **0.12** |

### Key Findings

- The 6-view setting achieves the most significant improvement (PSNR +1.65, pose error reduced by approximately 80%), as more views enable the formation of longer feature tracks.
- In the 3-view setting, the pose rotation error is almost halved (1.81 $\rightarrow$ 1.12).
- Even under the 3-view setting with GT poses, it outperforms diffusion-prior-based methods such as ReconFusion.
- The global feature track constraint is the key factor in accelerating pose optimization and improving accuracy.

## Highlights & Insights

- Seamlessly integrates the classic SfM bundle adjustment concept (track-level) into NeRF optimization, offering clear theoretical motivation.
- Does not rely on any generative priors (such as diffusion models) and instead utilizes only generic geometric cues, offering stronger generalization.
- The track keypoint adjustment step further enhances correspondence quality, providing additional robustness to feature noise.
- The substantial improvement in the 6-view setting demonstrates that global consistency constraints are of maximum value under moderately dense view settings.

## Related Work & Insights

BARF first proposed frequency modulation to optimize camera poses but does not use any multi-view correspondence constraints, meaning it is not bundle adjustment in the true sense. SPARF introduced pairwise correspondence losses, which represent a significant advancement, but remains limited to local consistency between view pairs. TrackNeRF directly integrates track-level BA loss functions within the NeRF framework, where a feature track can connect pixels across all visible views, thus enforcing global consistency.

On the LLFF dataset, TrackNeRF also demonstrates steady improvements; under the 3-view noisy pose setup, the rendered depth maps are smoother with fewer floaters. This is primarily attributed to the depth regularization loss, which encourages alignment between depth gradients and image gradients.

## Limitations & Future Work

- It depends on the quality of the dense correspondence network (PDCNet++), where mismatching errors propagate into the feature tracks.
- The training cost is comparable to SPARF, making actual deployment relatively slow.
- In extremely sparse (2-view) scenes, feature tracks degenerate into pairwise correspondences, diminishing the advantages.
- The length of feature track chains is proportional to the number of views, increasing overhead as the view count grows.

## Rating

- Novelty: ⭐⭐⭐⭐ — Elegant transfer of a classic idea
- Effectiveness: ⭐⭐⭐⭐⭐ — Comprehensive SOTA, with dual improvements in pose and PSNR
- Practicality: ⭐⭐⭐⭐
- Recommendation: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Track Everything Everywhere Fast and Robustly](track_everything_everywhere_fast_and_robustly.md)
- [\[ECCV 2024\] Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [\[ECCV 2024\] ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)
- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)
- [\[ECCV 2024\] VCD-Texture: Variance Alignment based 3D-2D Co-Denoising for Text-Guided Texturing](vcd-texture_variance_alignment_based_3d-2d_co-denoising_for_text-guided_texturin.md)

</div>

<!-- RELATED:END -->
