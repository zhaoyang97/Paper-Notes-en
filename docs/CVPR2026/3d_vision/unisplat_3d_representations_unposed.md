---
title: >-
  [Paper Note] UniSplat: Learning 3D Representations for Spatial Intelligence from Unposed Multi-View Images
description: >-
  [CVPR 2026][3D Vision][3D representation learning] UniSplat learns unified geometry-appearance-semantic 3D representations from unposed multi-view images via three components — dual masking, coarse-to-fine Gaussian splatting, and pose-conditioned recalibration — laying a perceptual foundation for spatial intelligence.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D representation learning
  - spatial intelligence
  - Gaussian splatting
  - self-supervised learning
  - unposed multi-view
date: 2026-05-08
content_hash: 18a5ebbbf8a5f592
---

# UniSplat: Learning 3D Representations for Spatial Intelligence from Unposed Multi-View Images

**Conference**: CVPR 2026
**arXiv**: [2604.10573](https://arxiv.org/abs/2604.10573)
**Code**: [https://bobochow.github.io/UniSplat](https://bobochow.github.io/UniSplat)
**Area**: 3D Vision
**Keywords**: 3D representation learning, spatial intelligence, Gaussian splatting, self-supervised learning, unposed multi-view

## TL;DR
UniSplat learns unified geometry-appearance-semantic 3D representations from unposed multi-view images via three components — dual masking, coarse-to-fine Gaussian splatting, and pose-conditioned recalibration — laying a perceptual foundation for spatial intelligence.

## Background & Motivation

**Background**: 3D representation learning is transitioning from supervised methods (requiring calibrated poses) to self-supervised methods (learning directly from raw multi-view images), yet existing self-supervised approaches generally suffer from weak geometric awareness, insufficient appearance detail, and geometry-semantic inconsistency.

**Limitations of Prior Work**: (1) Masked autoencoding methods lack strict global 3D consistency; (2) novel view synthesis methods assume known poses or rely on dense video; (3) unposed methods jointly estimate cameras and scenes but insufficiently couple the three representation dimensions.

**Key Challenge**: Geometry, appearance, and semantics each demand different optimal granularities — semantics are inherently coarse-grained while appearance requires fine-grained detail — making naive joint learning lead to mutual interference.

**Goal**: Design a feed-forward framework that jointly learns geometry, appearance, and semantic representations from unposed sparse multi-view images.

**Core Idea**: Address geometric awareness (dual masking), appearance fidelity (coarse-to-fine splatting), and cross-task consistency (pose recalibration) through three complementary mechanisms.

## Method

### Overall Architecture
Unposed multi-view images → Transformer encoder (with dual masking) → multi-head decoder → coarse-to-fine Gaussian splatting (anchor → semantic → fine Gaussians) → pose-conditioned recalibration → output 3D representations (point cloud, normals, semantics, appearance).

### Key Designs

1. **Dual Masking Strategy**:

    - **Function**: Enhances the geometric awareness of the encoder.
    - **Mechanism**: Stage 1 applies random masking to encoder tokens to extract preliminary features; Stage 2 generates geometry-aware masks from the importance map of the coarse Gaussian field, masking structurally critical regions of the decoder tokens. This compels the decoder to infer 3D structure from incomplete evidence.
    - **Design Motivation**: Random masking may suppress unimportant regions, whereas geometry-guided masking specifically conceals structurally salient features, forcing the model to learn genuine 3D reasoning rather than local texture completion.

2. **Coarse-to-Fine Gaussian Splatting Strategy**:

    - **Function**: Progressively refines the radiance field to reconcile the granularity gap between semantics and appearance.
    - **Mechanism**: A three-level hierarchy — anchor Gaussians (position + geometry/semantic features) → semantic Gaussians (offsets + coarse appearance + semantics) → fine Gaussians (high-frequency details injected by upsampling from 2D feature maps). Semantics are rendered at coarser levels; appearance is rendered at the finest level.
    - **Design Motivation**: Semantics are coarse-grained (object-level) while appearance requires fine granularity (texture-level); hierarchical rendering prevents mutual interference between the two.

3. **Pose-Conditioned Recalibration Mechanism**:

    - **Function**: Enforces cross-task consistency between geometric and semantic predictions.
    - **Mechanism**: Camera parameters estimated by the pose head are used to reproject predictions from the 3D point cloud head and semantic head onto the 2D image plane, aligning them with corresponding RGB and semantic predictions. A reprojection consistency loss ensures geometry and semantics remain mutually coherent.
    - **Design Motivation**: In conventional multi-task learning, individual heads operate independently with no explicit mechanism to guarantee cross-task consistency; reprojection provides a natural alignment signal.

### Loss & Training
A combination of self-supervised learning and knowledge distillation: novel view synthesis photometric loss, 3D point cloud distillation loss (from DUSt3R/VGGT), semantic feature distillation loss (from DINOv2/SigLIP), and reprojection consistency loss.

## Key Experimental Results

### Main Results

| Task | Dataset | Metric | UniSplat | Prev. SOTA |
|------|---------|--------|----------|------------|
| Novel view synthesis | RealEstate10K | PSNR | Competitive | SelfSplat |
| Camera pose estimation | CO3Dv2 | RTE | Improved | RayZer |
| Depth estimation | ScanNet | Abs Rel | Improved | Baseline |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Full model | Best | Complete model |
| w/o dual masking | Degraded | Weakened geometric awareness |
| w/o coarse-to-fine | Degraded | Increased appearance-semantic inconsistency |
| w/o recalibration | Degraded | Worse cross-task consistency |

### Key Findings
- The three components are mutually complementary; removing any one leads to performance degradation.
- Geometry-guided masking more effectively enhances 3D reasoning than random masking.
- The unified representation generalizes well to downstream tasks (navigation, manipulation).

## Highlights & Insights
- **Granularity Decoupling**: The coarse-to-fine strategy elegantly resolves the granularity conflict between semantics and appearance — an idea transferable to other multi-task 3D learning settings.
- **Reprojection as Natural Alignment**: Leveraging estimated poses for cross-head consistency constraints eliminates the need for additional annotations while providing a strong supervisory signal.

## Limitations & Future Work
- Quality is dependent on the teacher models used for knowledge distillation.
- Computational overhead is non-trivial (multi-head decoder + multi-level Gaussians).
- Future work may explore lighter-weight architectures and larger-scale pretraining.

## Related Work & Insights
- **vs. RayZer**: RayZer employs an implicit renderer, whereas UniSplat uses explicit Gaussian splatting for improved interpretability.
- **vs. SelfSplat**: SelfSplat treats depth and pose modules separately; UniSplat achieves tighter coupling through pose-conditioned recalibration.

## Rating
- Novelty: ⭐⭐⭐⭐ The synergistic design of the three components is novel, though each individual component is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task evaluation is comprehensive.
- Writing Quality: ⭐⭐⭐⭐ The framework is described clearly.
- Value: ⭐⭐⭐⭐ Provides a practical perceptual foundation for spatial intelligence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning Multi-View Spatial Reasoning from Cross-View Relations](learning_multi-view_spatial_reasoning_from_cross-view_relations.md)
- [\[CVPR 2026\] BRepGaussian: CAD Reconstruction from Multi-View Images with Gaussian Splatting](brepgaussian_cad_reconstruction_from_multi-view_images_with_gaussian_splatting.md)
- [\[ICCV 2025\] Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting](../../ICCV2025/3d_vision/towards_scalable_spatial_intelligence_via_2d-to-3d_data_lifting.md)
- [\[NeurIPS 2025\] Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations](../../NeurIPS2025/3d_vision/concerto_joint_2d-3d_self-supervised_learning_emerges_spatial_representations.md)
- [\[ICLR 2026\] UFO-4D: Unposed Feedforward 4D Reconstruction from Two Images](../../ICLR2026/3d_vision/ufo-4d_unposed_feedforward_4d_reconstruction_from_two_images.md)

</div>

<!-- RELATED:END -->
