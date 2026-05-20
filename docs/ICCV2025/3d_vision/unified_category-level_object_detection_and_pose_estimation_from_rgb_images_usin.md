---
title: >-
  [Paper Note] Unified Category-Level Object Detection and Pose Estimation from RGB Images using 3D Prototypes
description: >-
  [ICCV 2025][3D Vision][Category-level pose estimation] This work presents the first RGB-only, single-model framework that unifies object detection and category-level pose estimation. By leveraging Neural Mesh Models as 3…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Category-level pose estimation"
  - "object detection"
  - "Neural Mesh Models"
  - "single-stage method"
  - "RGB-only"
date: 2026-05-08
content_hash: bdcbe11117531283
---

# Unified Category-Level Object Detection and Pose Estimation from RGB Images using 3D Prototypes

**Conference**: ICCV 2025
**arXiv**: [2508.02157](https://arxiv.org/abs/2508.02157)  
**Code**: [GitHub](https://github.com/Fischer-Tom/unified-detection-and-pose-estimation)  
**Area**: 3D Vision
**Keywords**: Category-level pose estimation, object detection, Neural Mesh Models, single-stage method, RGB-only

## TL;DR

This work presents the first RGB-only, single-model framework that unifies object detection and category-level pose estimation. By leveraging Neural Mesh Models as 3D prototypes, the method performs feature matching and multi-model RANSAC PnP to simultaneously detect objects and estimate their 9D poses. It surpasses the state of the art on all scale-agnostic metrics on REAL275.

## Background & Motivation

Category-level object pose estimation is central to 3D scene understanding, with broad applications in robotic manipulation, autonomous driving, and augmented reality. Existing methods suffer from the following key bottlenecks:

**RGB-D Dependency**: Most high-performing methods rely on depth information (i.e., RGB-D input), limiting their applicability in scenarios where depth sensors are unavailable.

**Two-Stage Pipelines**: All existing RGB-only methods adopt a two-stage architecture—first localizing objects with a standalone detector (e.g., Mask-RCNN), then feeding cropped regions into a separate pose estimator. This leads to:
   - Detection failures that directly propagate to the pose estimation stage with no possibility of recovery
   - The need to maintain two independent models and representations
   - High sensitivity to image degradation (noise, blur, etc.)

**Scale Ambiguity**: Distance and scale are inherently ambiguous in pure RGB settings, making RGB-only pose estimation substantially more challenging than RGB-D approaches.

The core motivation of this paper is: **Can a single unified model simultaneously perform detection and pose estimation?** The authors observe that Neural Mesh Models possess strong category-level generalization capabilities and can serve as a unified 3D representation to associate 2D image features with 3D geometry, thereby jointly inferring object presence and spatial pose.

## Method

### Overall Architecture

The model consists of three main components: (1) a DINOv2-based 2D feature extractor, (2) learnable per-category 3D prototypes (Neural Mesh Models), and (3) a detection and pose inference pipeline based on multi-model RANSAC.

During training, 2D image features and 3D vertex features are jointly optimized via contrastive learning. At inference, image features are matched against all prototype vertex features, and the Progressive-X multi-model fitting algorithm discovers multiple objects and estimates their 6D poses simultaneously from the correspondences. A deformation-based refinement step then upgrades each estimate to a full 9D pose.

### Key Designs

1. **Neural Mesh Models as 3D Prototype Representations**: Each category is represented by a prototype mesh $\mathbf{M}^c = (\mathbf{V}^c, \mathbf{A}^c, \mathbf{F}^c_{3D})$, where $\mathbf{V}^c \in \mathbb{R}^{V \times 3}$ denotes vertex coordinates and $\mathbf{F}^c_{3D} \in \mathbb{R}^{V \times D}$ denotes learnable vertex features. The mesh geometry is constructed by uniformly sampling the surface of a bounding box scaled to the category mean size. This representation generalizes within a category without requiring exact CAD models at test time.

2. **Dual-Stream Feature Extraction with Geometric Feature Decoder**: A frozen DINOv2 ViT serves as the backbone, augmented with parameter-efficient fine-tuning via LoRA Adapters. The adapter introduces low-rank adaptation as:
    $x'_l = \text{MLP}(\text{LN}(x_l)) + x_l + \lambda(\text{GeLU}(\text{LN}(x_l) \cdot \mathbf{D})) \cdot \mathbf{U}$
   This is followed by a shared Transformer encoder and a dual-branch Transformer decoder that produces two feature maps: $\overline{\mathbf{F}}_{2D}$ (aligned to the category mean-size prototype) for 6D pose estimation, and $\mathbf{F}_{2D}$ (aligned to the instance-size deformed prototype) for 9D refinement.

3. **Foreground Detection Module**: Leveraging the emergent foreground segmentation behavior in ViT attention maps, a foreground token initialized from the CLS token performs cross-attention with decoder features to produce a foreground probability map $\mathbf{H}$. Supervised with Dice Loss, this module effectively suppresses false-positive matches from background regions.

4. **Multi-Model RANSAC for Joint Detection and Pose Estimation**: At inference, a dense 2D-3D correspondence set $\bar{\mathcal{N}}_{3D}^{2D}$ is established by matching each pixel's feature against all vertex features across all categories, selecting the highest-similarity vertex as the match. Progressive-X (a multi-model fitting algorithm) is then applied per-category to simultaneously separate multiple instances and estimate their individual 6D poses.

5. **9D Pose Refinement**: For each detected object, a deformation parameter $\mathbf{d} \in \mathbb{R}^3$ is introduced to scale vertices along principal axes. A two-step optimization minimizes the reprojection error:
    $\mathbf{E}(\mathbf{K}, \mathbf{R}, \mathbf{t}, \mathbf{d}, \mathcal{N}_i) = \sum_{v_k, \mathbf{p}_k \in \mathcal{N}_i} \|(\mathbf{K}\mathbf{R}(\mathbf{d} \odot v_k) + \mathbf{t}) - \mathbf{p}_k\|_2^2$
   The deformation parameters are first solved with the 6D pose fixed, and the refined geometry is then used to re-estimate rotation and translation.

### Loss & Training

- **Contrastive Loss**: Feature matching probabilities are modeled under a von Mises-Fisher distribution, maximizing the likelihood of correct correspondences and minimizing that of incorrect ones, unified in an InfoNCE form:
  $$\mathcal{L}(\mathcal{Y}) = -\sum_k o_k \cdot \log\frac{e^{\kappa(f_k \cdot \theta_k)}}{\sum_{\theta_m \in \bar{\theta}_k} e^{\kappa(f_k^\top \cdot \theta_m)}}$$
- **Total Training Loss**: The average contrastive loss computed on both the mean-shape and deformed-shape prototypes.
- **Foreground Segmentation Loss**: Supervised with Dice Loss.
- **Vertex Feature Optimization**: Rather than the previously employed EMA strategy, vertex features are optimized directly via backpropagation, yielding more stable and consistent training.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA (LaPose) | Gain |
|--------|------|------|-------------------|------|
| REAL275 | NIoU25 | 75.2 | 70.7 | +4.5 |
| REAL275 | NIoU50 | 53.7 | 47.9 | +5.8 |
| REAL275 | 5°0.2d | 25.1 | 15.7 | +9.4 |
| REAL275 | 10°0.5d | 66.1 | 57.4 | +8.7 |
| REAL275 | 10° | 68.8 | 60.7 | +8.1 |
| CAMERA25 | NIoU50 | 47.6 | 45.2 | +2.4 |
| CAMERA25 | 5°0.5d | 57.6 | 53.9 | +3.7 |

The proposed method surpasses the state of the art on all 11 scale-agnostic metrics, with an average improvement of 22.9%.

### Ablation Study

| Configuration | NIoU25 | NIoU50 | 10°0.5d | Note |
|------|--------|--------|---------|------|
| GT Proto. Mask (upper bound) | 79.5 | 55.3 | 66.7 | Uses GT prototype mask as ROI |
| Ours (no ROI) | 75.2 | 53.7 | 66.1 | Single-stage, approaches upper bound |
| GT Object Mask | 72.1 | 50.3 | 61.4 | Uses GT object mask |
| Mask-RCNN | 70.4 | 48.8 | 58.4 | Uses Mask-RCNN detection |

Robustness ablation: Under 8 types of image degradation, two-stage methods suffer severe performance drops (LaPose drops an average of 25.9% under image-level degradation), while the proposed method drops only 12.6%, demonstrating substantially greater robustness.

### Key Findings

- A common source of error in two-stage methods is the detector; detection failures cause complete pose estimation breakdown.
- Single-stage methods are far more robust to image degradation than two-stage methods, particularly when image-level noise simultaneously affects both detection and pose estimation.
- Directly optimizing vertex features via backpropagation is more stable than the EMA strategy.

## Highlights & Insights

- This is the first work to achieve single-stage category-level multi-object pose estimation in an RGB-only setting, representing an important paradigm shift.
- Neural Mesh Models are elegantly extended from single-object to multi-object, multi-category scenarios.
- The incorporation of Progressive-X multi-model fitting enables graceful handling of multiple instances of the same category.
- The dual-stream feature design (category-level + instance-level) effectively decouples the requirements of detection and refinement.

## Limitations & Future Work

- Performance on absolute pose metrics is moderate, as scale estimation from RGB alone is inherently difficult.
- Accuracy on small objects (at large depth or fine scale) remains limited; using higher-resolution features could help.
- The scale prediction network, trained from prototype projections, underperforms compared to variants trained from 2D bounding boxes.
- Inference speed is constrained by the iterative nature of Progressive-X.

## Related Work & Insights

- Neural Mesh Models (Bao et al., NeurIPS 2022) provide an excellent foundation for category-level representation.
- The combination of DINOv2 self-supervised features with LoRA-based efficient fine-tuning is worth broader adoption.
- The end-to-end paradigm of contrastive learning combined with geometric solvers is generalizable to other 3D reasoning tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First single-stage RGB-only category-level multi-object pose estimation
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dataset, multi-metric, and robustness experiments are comprehensive
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear; method is described systematically
- Value: ⭐⭐⭐⭐⭐ The unified framework opens a new paradigm with tremendous potential

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] BoxDreamer: Dreaming Box Corners for Generalizable Object Pose Estimation](boxdreamer_dreaming_box_corners_for_generalizable_object_pose_estimation.md)
- [\[NeurIPS 2025\] SingRef6D: Monocular Novel Object Pose Estimation with a Single RGB Reference](../../NeurIPS2025/3d_vision/singref6d_monocular_novel_object_pose_estimation_with_a_single_rgb_reference.md)
- [\[ICCV 2025\] A Unified Interpretation of Training-Time Out-of-Distribution Detection](a_unified_interpretation_of_training-time_out-of-distribution_detection.md)
- [\[ICCV 2025\] UPP: Unified Point-Level Prompting for Robust Point Cloud Analysis](upp_unified_point-level_prompting_for_robust_point_cloud_analysis.md)
- [\[ICCV 2025\] FROSS: Faster-than-Real-Time Online 3D Semantic Scene Graph Generation from RGB-D Images](fross_faster-than-real-time_online_3d_semantic_scene_graph_generation_from_rgb-d.md)

</div>

<!-- RELATED:END -->
