---
title: >-
  [Paper Note] DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction
description: >-
  [CVPR 2025][3D Vision][Dual Point Maps] Proposes Dual Point Maps (DualPM), which simplifies 3D shape and pose reconstruction of deformable objects into a point map prediction problem by simultaneously predicting a pair of point maps in camera space and canonical space, generalizing to real images using only synthetic training data.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Dual Point Maps"
  - "Deformable Object Reconstruction"
  - "Canonical Space"
  - "Pose Estimation"
  - "Quadrupeds"
date: 2026-05-08
content_hash: 0c8b02e0d023e5bf
---

# DualPM: Dual Posed-Canonical Point Maps for 3D Shape and Pose Reconstruction

**Conference**: CVPR 2025  
**arXiv**: [2412.04464](https://arxiv.org/abs/2412.04464)  
**Code**: [https://dualpm.github.io](https://dualpm.github.io)  
**Area**: 3D Vision / Deformable Object Reconstruction  
**Keywords**: Dual Point Maps, Deformable Object Reconstruction, Canonical Space, Pose Estimation, Quadrupeds

## TL;DR

Proposes Dual Point Maps (DualPM), which simplifies 3D shape and pose reconstruction of deformable objects into a point map prediction problem by simultaneously predicting a pair of point maps in camera space and canonical space, generalizing to real images using only synthetic training data.

## Background & Motivation

**Background**: DUSt3R has demonstrated the powerful capability of point map representation in static scene reconstruction, unifying matching, camera estimation, and triangulation into point map prediction.

**Limitations of Prior Work**: A single point map can only reconstruct the visible 3D shape and cannot recover the pose of the object (deformation field). Existing methods for deformable objects rely on large-scale weakly-supervised data or complex optimization.

**Key Challenge**: Recovering the pose requires knowing the "deformation from the canonical pose to the current pose", but a single point map lacks deformation information.

**Goal**: Design a network-friendly representation such that both shape and pose reconstruction can be achieved through simple point map prediction.

**Key Insight**: If two point maps are predicted simultaneously—one in camera space (current pose) and one in canonical space (rest pose)—the deformation field is simply the difference between the two.

**Core Idea**: DualPM = camera space point map $P$ + canonical space point map $Q$, pose/deformation field = $P - Q$.

## Method

### Overall Architecture

Given an image $I$, features $F$ are first extracted using a pre-trained feature extractor (such as DINOv2) to predict the canonical point map $Q = \Phi_Q(F)$. Then, the camera space point map $P = \Phi_P(Q)$ is predicted conditioned on $Q$. The extension to an amodal version reconstructs the complete shape through a layered representation.

### Key Designs

1. **Dual Point Maps**:

    - **Function**: Unify the encoding of 3D shape and pose information.
    - **Mechanism**: For each pixel $u$, $P(u)$ gives its 3D position in the camera coordinate system, and $Q(u)$ provides the position of the same point in the canonical space. Cross-image matching can be achieved by comparing $Q$ values (as $Q$ is pose/view-invariant), and the deformation field is directly $P - Q$.
    - **Design Motivation**: Predicting $Q$ resembles a pixel labeling problem (pose-invariant), which significantly reduces the difficulty of network learning.

2. **Canonical Point Map as an Intermediate Representation**:

    - **Function**: $Q$ is used as a conditional input to $P$, replacing the original image features.
    - **Mechanism**: Predict $Q$ first (based on DINOv2 features, which is easier to learn due to pose invariance), and then predict $P$ conditioned on $Q$. In this way, the network for $P$ does not need to learn directly from highly variable image features, but starts from $Q$, which has already decoupled the pose.
    - **Design Motivation**: Experiments demonstrate that conditioning $P$ on $Q$ yields better out-of-distribution generalization compared to directly predicting $P$ from DINOv2 features.

3. **Amodal Layered Point Maps**:

    - **Function**: Reconstruct the complete 3D shape, including self-occluded parts.
    - **Mechanism**: Each pixel is mapped to $2K$ 3D points ($K$ pairs of entry/exit points), similar to depth peeling. The first layer consists of visible points, while subsequent layers capture occluded points. An additional opacity $\sigma$ is predicted for each layer to indicate the existence of an intersection in that layer.
    - **Design Motivation**: Standard point maps can only reconstruct visible parts, while the amodal extension recovers the complete shape through layered predictions.

### Loss & Training

Predictive networks for $P$ and $Q$ are trained using a self-calibrating L2 loss, alongside a cross-entropy loss for opacity. The training data requires only 1-2 synthetic 3D models per category, utilizing synthetic renders generated by methods like Farm3D. The model is trained on synthetic data and generalizes directly to real images.

## Key Experimental Results

### Main Results

Significantly outperforms methods like 3D-Fauna and MagicPony on quadrupeds (horses, cows, dogs, etc.):
- Cross-pose correspondence: PCK@0.1 leads significantly.
- 3D Reconstruction: Chamfer distance is substantially reduced.
- Generalizes to real images with training on synthetic data only.

### Ablation Study

- $Q$ as a condition for $P$ vs. DINOv2 features: Conditioning on $Q$ yields better generalization.
- Amodal vs. modal: The amodal version provides a more complete reconstruction.
- Zero-order vs. higher-order spherical harmonics for $Q$: Zero-order is sufficient (since $Q$ should be view-independent).

### Key Findings

- DualPM can be directly applied to skeleton fitting and motion transfer.
- Synthetic data is sufficient to train a model with strong generalization capabilities.
- The canonical point map is itself an excellent feature map.

## Highlights & Insights

- Elegantly extends the point map concept of DUSt3R to the field of deformable objects.
- The design of using $Q$ as a condition for $P$ is clever—decoupling the pose first, then reconstructing.
- The amodal layered representation achieves complete 3D reconstruction.

## Limitations & Future Work

- Currently validated primarily on quadrupeds; extending to other categories such as humans requires more work.
- Relies on object masks as input.
- The amodal extension becomes increasingly difficult to predict as the number of layers increases.

## Rating

- Novelty: 9/10 — The conceptual design of dual point maps is elegant.
- Technical Depth: 8/10 — Clear theory and solid technology.
- Experimental Thoroughness: 8/10 — Multi-task validation with comprehensive ablations.
- Writing Quality: 9/10 — Concepts are explained clearly and motivation derivation is natural.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Denoising Functional Maps: Diffusion Models for Shape Correspondence](denoising_functional_maps_diffusion_models_for_shape_correspondence.md)
- [\[CVPR 2025\] Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors](pow3r_empowering_unconstrained_3d_reconstruction_with_camera_and_scene_priors.md)
- [\[CVPR 2025\] VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM](varsplat_uncertainty-aware_3d_gaussian_splatting_for_robust_rgb-d_slam.md)
- [\[CVPR 2025\] MAC-Ego3D: Multi-Agent Gaussian Consensus for Real-Time Collaborative Ego-Motion and Photorealistic 3D Reconstruction](mac-ego3d_multi-agent_gaussian_consensus_for_real-time_collaborative_ego-motion_.md)
- [\[CVPR 2025\] ESCAPE: Equivariant Shape Completion via Anchor Point Encoding](escape_equivariant_shape_completion_via_anchor_point_encoding.md)

</div>

<!-- RELATED:END -->
