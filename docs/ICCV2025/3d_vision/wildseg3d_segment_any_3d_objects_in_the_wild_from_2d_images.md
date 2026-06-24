---
title: >-
  [Paper Note] WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images
description: >-
  [ICCV 2025][3D Vision][3D segmentation] This paper proposes WildSeg3D, the first feed-forward 3D segmentation model that requires no scene-specific training. It addresses multi-view pointmap alignment errors via Dynamic Global Alignment (DGA) and achieves real-time interactive 3D segmentation through Multi-view Group Mapping (MGM), outperforming the current state of the art in accuracy while being 40× faster.
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D segmentation"
  - "feed-forward"
  - "SAM2"
  - "global alignment"
  - "real-time interaction"
date: 2026-05-08
content_hash: f6464dd091544714
---

# WildSeg3D: Segment Any 3D Objects in the Wild from 2D Images

**Conference**: ICCV 2025
**arXiv**: [2503.08407](https://arxiv.org/abs/2503.08407)  
**Code**: Coming soon  
**Area**: 3D Vision
**Keywords**: 3D segmentation, feed-forward, SAM2, global alignment, real-time interaction

## TL;DR

This paper proposes WildSeg3D, the first feed-forward 3D segmentation model that requires no scene-specific training. It addresses multi-view pointmap alignment errors via Dynamic Global Alignment (DGA) and achieves real-time interactive 3D segmentation through Multi-view Group Mapping (MGM), outperforming the current state of the art in accuracy while being 40× faster.

## Background & Motivation

Interactive 3D segmentation (segmenting 3D objects from 2D images) has broad applications in VR/AR, real-time interactive systems, and automatic annotation.

**Common bottlenecks of existing methods**:

**NeRF-based methods** (SA3D, SANeRF-HQ): integrate SAM for 3D segmentation, but NeRF requires extensive scene-specific training time.

**3DGS-based methods** (SAGA, Gaussian Grouping, Feature3DGS): faster than NeRF, yet still require training to construct Gaussian feature fields.

**Shared limitation**: all methods rely on **scene-specific training** to acquire accurate 3D priors, severely hindering real-time applicability.

**Key challenge**: Feed-forward approaches (e.g., DUSt3R/MASt3R) can bypass scene-specific training, but **3D alignment errors across multi-view pointmaps accumulate**, causing target objects to be confused with the background and degrading segmentation accuracy.

## Method

### Overall Architecture

A three-stage pipeline:
1. **2D mask preprocessing** (offline): SAM2 panoptic segmentation → multi-view tracking → mask caching
2. **Dynamic Global Alignment (DGA)**: dynamic weight adjustment → optimized multi-view pointmap alignment
3. **Multi-view Group Mapping (MGM)** (real-time): user prompt → retrieve mask cache → map to 3D space

### Dynamic Global Alignment (DGA)

Standard global alignment (MASt3R) treats all pixels equally, but large background variation and target occlusion lead to misalignment. DGA introduces three innovations:

**1. Soft mask + confidence aggregation**: fuses SAM2 masks with pointmap confidence scores

$$F_i^{v,e} = \sigma(S_i^v \times C_i^{v,e})$$

**2. Dynamic adjustment function**: assigns greater attention to hard-to-match points (confidence ≈ 0.5)

$$A_i^{v,e} = \frac{F_i^{v,e} + \alpha_i^{v,e} \cdot F_i^{v,e} \cdot (1 - F_i^{v,e})}{1 + |\alpha_i^{v,e}| \cdot F_i^{v,e} \cdot (1 - F_i^{v,e}) + \epsilon}$$

where $\alpha$ takes a positive value $\alpha_p$ for matched points and a negative value $-\alpha_n$ for unmatched points.

**3. Optimized alignment loss**:

$$\chi^* = \arg\min_{\chi, P, \sigma} \sum_{e \in \mathcal{E}} \sum_{v \in e} \sum_{i=1}^{HW} W_i^{v,e} \|\chi_i^v - \sigma_e P_e X_i^{v,e}\|$$

### Multi-view Group Mapping (MGM)

At the real-time stage: the user provides a prompt in one view → the system retrieves all-view masks for the corresponding object from the mask cache → applies the transformation matrices learned by DGA to map masks into the aligned 3D space → returns results within 5–20 ms.

### Mask Caching Mechanism

Offline processing leverages SAM2's video tracking capability:
1. Perform panoptic segmentation on a single view.
2. SAM2 tracking propagates object masks to all views.
3. Results are stored as an offline cache and queried directly at runtime.

## Experiments

### Main Results on NVOS Dataset

| Method | Scene Training | mIoU (%) | mAcc (%) | Total Time |
|:---|:---:|:---:|:---:|:---:|
| NVOS | Yes | 70.1 | 92.0 | - |
| SA3D | Yes | 90.3 | 98.2 | 780s |
| SAGA | Yes | 90.9 | 98.3 | 2280s |
| OmniSeg3D | Yes | 91.7 | 98.4 | 8220s |
| FlashSplat | Yes | 91.8 | 98.6 | 1500s |
| **WildSeg3D** | **No** | **94.1** | **99.0** | **30s** |

WildSeg3D requires no scene-specific training yet surpasses all training-based methods in accuracy while achieving more than 40× speedup.

### Efficiency Comparison

| Metric | SA3D | SAGA | OmniSeg3D | WildSeg3D |
|:---|:---:|:---:|:---:|:---:|
| Scene reconstruction time | 780s | 2280s | 8220s | **<30s** |
| Interaction response time | Seconds | Seconds | Seconds | **5–20ms** |

### Ablation Study

| Ablation | Key Findings |
|:---|:---|
| Standard alignment vs. DGA | DGA significantly improves target object alignment and reduces background interference |
| Effect of soft mask | Soft masks focus attention on target regions, suppressing negative effects of background features on alignment |
| Dynamic adjustment function | Assigns higher weights to ambiguous points near the decision boundary (confidence ≈ 0.5) |
| Mask caching | Offline preprocessing eliminates online segmentation during interaction, reducing response time to milliseconds |

### Key Findings

1. **Surpassing training-based methods without training**: the feed-forward mechanism with accurate alignment matches or exceeds scene-training-dependent methods.
2. **40× acceleration**: total time reduced from 780s (SA3D) to 30s, with 5–20ms interactive response.
3. **Robust generalization**: operates across diverse scenes without adaptation.

## Highlights & Insights

1. **Feed-forward paradigm shift**: completely eliminates scene-specific training, making 3D segmentation practically deployable.
2. **Core idea of DGA**: improving alignment quality by dynamically attending to "hard" points — low-confidence points typically correspond to object boundaries or occluded regions.
3. **Offline–online decoupling**: SAM2 segmentation is performed and cached offline; at runtime, only 3D mapping lookup is required, enabling millisecond-level response.

## Limitations & Future Work

1. Relies on the quality of pointmap prediction from MASt3R; alignment may fail in texture-less regions.
2. The quality of SAM2 panoptic segmentation directly affects final results.
3. Alignment accuracy may degrade under sparse viewpoint settings.

## Related Work & Insights

- **3D reconstruction**: NeRF, 3DGS, DUSt3R, MASt3R
- **Interactive 3D segmentation**: SA3D, SAGA, Feature3DGS, FlashSplat
- **Foundation models**: SAM, SAM2

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First feed-forward 3D segmentation model; a paradigm-level innovation
- Technical depth: ⭐⭐⭐⭐ — DGA dynamic adjustment function is theoretically grounded
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive speed/accuracy comparisons with clear ablations
- Value: ⭐⭐⭐⭐⭐ — 30s reconstruction + millisecond-level interaction; genuinely deployable

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SAS: Segment Any 3D Scene with Integrated 2D Priors](sas_segment_any_3d_scene_with_integrated_2d_priors.md)
- [\[ICCV 2025\] Amodal3R: Amodal 3D Reconstruction from Occluded 2D Images](amodal3r_amodal_3d_reconstruction_from_occluded_2d_images.md)
- [\[NeurIPS 2025\] Online Segment Any 3D Thing as Instance Tracking](../../NeurIPS2025/3d_vision/online_segment_any_3d_thing_as_instance_tracking.md)
- [\[ICCV 2025\] Find Any Part in 3D](find_any_part_in_3d.md)
- [\[ICCV 2025\] TriDi: Trilateral Diffusion of 3D Humans, Objects, and Interactions](tridi_trilateral_diffusion_of_3d_humans_objects_and_interactions.md)

</div>

<!-- RELATED:END -->
