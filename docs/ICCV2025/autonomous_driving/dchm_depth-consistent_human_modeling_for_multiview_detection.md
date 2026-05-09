---
title: >-
  [Paper Note] DCHM: Depth-Consistent Human Modeling for Multiview Detection
description: >-
  [ICCV 2025][Autonomous Driving][multiview pedestrian detection] This paper proposes DCHM, a depth-consistent human modeling framework that requires no 3D annotations. It generates pseudo depth labels via superpixel-level Gaussian splatting to fine-tune a monocular depth estimation network, and combines multiview label matching to achieve high-accuracy pedestrian detection under sparse-view and heavily occluded scenarios. DCHM achieves 84.2% MODA on Wildtrack and improves MODP by 31.2% over UMPD.
tags:
  - ICCV 2025
  - Autonomous Driving
  - multiview pedestrian detection
  - depth consistency
  - Gaussian splatting
  - monocular depth estimation
  - label-free method
date: 2026-05-08
content_hash: d71407779b2ae5ef
---

# DCHM: Depth-Consistent Human Modeling for Multiview Detection

**Conference**: ICCV 2025  
**arXiv**: [2507.14505](https://arxiv.org/abs/2507.14505)  
**Code**: [Project Page](https://github.com/)  
**Area**: Autonomous Driving  
**Keywords**: multiview pedestrian detection, depth consistency, Gaussian splatting, monocular depth estimation, label-free method

## TL;DR
This paper proposes DCHM, a depth-consistent human modeling framework that requires no 3D annotations. It generates pseudo depth labels via superpixel-level Gaussian splatting to fine-tune a monocular depth estimation network, and combines multiview label matching to achieve high-accuracy pedestrian detection under sparse-view and heavily occluded scenarios. DCHM achieves 84.2% MODA on Wildtrack and improves MODP by 31.2% over UMPD.

## Background & Motivation
Multiview pedestrian detection leverages multiple camera images to detect pedestrians, which is especially beneficial in heavily occluded scenes. Existing methods follow a "human modeling + pedestrian localization" strategy and face the following challenges:

**Inaccurate feature projection**: Existing methods project image features or detection results onto the ground plane, but the lack of pedestrian height information causes alignment errors for points above the ground, requiring 3D annotation-supervised training to compensate.

**Label dependency and generalization**: Supervised methods require extensive 3D annotations and generalize poorly to new scenes.

**Limitations of label-free methods**: Homography-based projection is sensitive to pixel-level errors; volumetric rendering-based methods (UMPD) perform poorly under sparse views and crowded scenes.

**Depth inconsistency**: Although monocular depth estimation methods provide detailed depth maps, depth maps predicted independently per view lack cross-view consistency, causing severe point cloud misalignment when back-projected into 3D space (e.g., the same person is estimated as multiple targets by different cameras).

**Core Idea**: Use Gaussian splatting to self-learn multiview-consistent pseudo depth labels from sparse-view images → fine-tune a monocular depth network to obtain consistent depth → 3D point cloud human modeling → multiview label matching and clustering-based detection.

## Method

### Overall Architecture
The framework consists of two phases: training and inference. The training phase involves an iterative three-step loop: (1) superpixel-level GS optimization to generate pseudo depth labels → (2) monocular depth model fine-tuning → (3) multiview detection compensation. The inference phase uses the optimized depth estimator to generate 3D point clouds, enabling pedestrian segmentation and localization via multiview matching.

### Key Designs
1. **Superpixel-level Gaussian Splatting Initialization and Optimization**:

    - Function: Achieves reliable 3D reconstruction under sparse views and generates pseudo depth labels.
    - Mechanism: Conventional SfM fails under wide-baseline sparse views. A "uniform sampling + filtering" strategy is adopted: rays are cast from each superpixel center, and points sampled uniformly along each ray are used to initialize Gaussians. The scale of each Gaussian is computed based on superpixel area and distance $t$:
    $s = \frac{tfr}{\|\mathbf{c} - \mathbf{o}\|_2 \cdot \sqrt{(\sqrt{\|\mathbf{c} - \mathbf{o}\|_2^2 - f^2} - r)^2 + f^2}}$
      A superpixel-level photometric loss $\mathcal{L}_{sp}$ (rather than pixel-level) is used to improve consistency under sparse views. Additional losses include mask loss $\mathcal{L}_m$, depth constraint loss $\mathcal{L}_d$ (encouraging small depth variance within a pedestrian mask), and opacity loss $\mathcal{L}_o$.
    - Design Motivation: Pixel-level supervision in sparse-view overlapping regions is inconsistent due to illumination and camera differences; superpixel aggregation of local features enhances consistency and accelerates optimization.

2. **Pseudo Depth Filtering and Monocular Depth Fine-tuning**:

    - Function: Selects reliable GS-rendered depths as supervision for fine-tuning the depth estimation network.
    - Mechanism: A two-step filtering strategy is applied:
        - **Cross-view foreground filtering**: Foreground pixels from source views are reprojected into the reference view; those landing in background regions are discarded.
        - **Cross-view depth consistency filtering**: Reprojected depth is compared against GS-rendered depth; only pixels with discrepancy below threshold $\tau$ are retained.
      The filtered depths serve as pseudo labels to fine-tune depth estimation networks such as Depth Anything v2.
    - Design Motivation: GS produces reliable depth only in regions where pedestrians are visible across multiple views; erroneous depths in occluded regions must be filtered out.

3. **Multiview Detection Compensation + Multiview Pedestrian Label Matching**:

    - Function: Recovers missed detections from single-view inputs; ensures consistent labeling of the same pedestrian across views.
    - Mechanism:
        - **Detection compensation**: Pedestrian masks from source views are projected into the reference view using predicted depth, generating box and point prompts that are fed into SAM to obtain compensated segmentation results.
        - **Label matching**: Starting from the first camera, Gaussians are projected onto the image plane, and IDs are assigned based on the overlap between blending weights and pedestrian masks. IDs propagate view by view (Gaussians with existing IDs falling into a new mask transfer their IDs; Gaussians without IDs are assigned new ones), ensuring cross-view consistent pedestrian identification.
    - Design Motivation: Missed detections due to occlusion degrade GS optimization and final detection; independent single-view segmentation cannot guarantee identity consistency across views.

### Loss & Training
GS optimization loss:
$$\mathcal{L} = \lambda_{sp}\mathcal{L}_{sp} + \lambda_m\mathcal{L}_m + \lambda_d\mathcal{L}_d + \lambda_o\mathcal{L}_o$$

Iterative training loop: GS optimization → pseudo depth filtering → depth network fine-tuning → detection compensation → next round of GS optimization. Accuracy improvements saturate after 3 rounds.

## Key Experimental Results

### Main Results (Comparison with Label-Free Methods)

| Method | Wildtrack MODA↑ | Wildtrack MODP↑ | Terrace MODA↑ | MultiviewX MODA↑ |
|------|----------------|----------------|---------------|-----------------|
| RCNN & clustering | 11.3 | 18.4 | -11 | 18.7 |
| BP & BB + CC | 56.9 | 67.3 | - | - |
| UMPD | 76.6 | 61.2 | 73.8 | 67.5 |
| **DCHM (Ours)** | **84.2** | **80.3** | **80.1** | **78.4** |

Wildtrack MODP surpasses UMPD by **31.2%** (80.3 vs. 61.2); MultiviewX MODA surpasses UMPD by **16.1%** (78.4 vs. 67.5).

### Ablation Study

| Configuration | MODA | Note |
|------|------|------|
| Pixel-level optimization input | 71.7 | Inconsistent under sparse views |
| **Superpixel-level optimization input** | **84.2** | +12.5 points |
| SIS + GVD+VBR (UMPD) | 76.6 | Original UMPD configuration |
| SIS + Our Recon | 82.5 | +5.9 from reconstruction |
| YOLOv11 + Our Recon | **84.2** | Better segmentation yields further gains |
| Depth estimation comparison (same Our Loc) | Depth Pro: 72.8, Our Recon: **84.2** | Depth consistency is the key factor |

### Key Findings
- Depth consistency is the critical bottleneck in multiview pedestrian detection: even the best off-the-shelf depth estimator (Depth Pro) falls far short without multiview consistency constraints.
- Superpixel-level GS supervision yields substantially larger improvements over pixel-level supervision under sparse views (MODA +12.5 points).
- Iterative training is effective: the coverage of reliable pseudo depth regions increases each round, though gains diminish after 3 rounds.
- The detection compensation mechanism effectively recovers missed detections caused by occlusion (consistent improvements with both YOLOv9 and YOLOv11).
- Inference speed is 1.2 FPS, within a practical range.

## Highlights & Insights
- **First method to achieve pedestrian reconstruction and multiview segmentation in sparse-view, large-scale, and crowded scenes.**
- The **self-learning pseudo depth label** approach is elegant: GS cross-view consistency is used to "teach" the monocular depth network.
- **Superpixel-level GS** addresses the unreliability of pixel-level photometric loss under sparse views.
- Multiview label matching is performed in 3D space via Gaussian blending weights, making it more robust than 2D tracking.
- Combining DCHM's human modeling with supervised localization methods even surpasses the supervised SOTA.

## Limitations & Future Work
- Inference speed of 1.2 FPS limits real-time applicability.
- Training is time-consuming (3 iterative rounds, each requiring per-frame GS optimization for pseudo depth).
- Ground reconstruction relies on predefined depth ranges, which may fail on complex terrain.
- Performance is sensitive to the quality of the segmentation network (notable gap between YOLOv11 and SIS).
- Dynamic scenes are not yet handled (GS optimization assumes a static scene during pedestrian movement).

## Related Work & Insights
- The superpixel-level GS optimization paradigm generalizes to other sparse-view reconstruction tasks.
- The pseudo depth filtering strategy is an effective practice for self-supervised depth learning.
- The multiview label matching method is applicable to other multi-camera object tracking and segmentation tasks.
- The SAM prompt generation strategy in the detection compensation mechanism has broad applicability.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of superpixel-level GS, self-learning pseudo depth, and detection compensation is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, multi-method comparisons, detailed ablations, and cross-method depth estimation evaluation.
- Writing Quality: ⭐⭐⭐⭐ Architecture diagrams are clear, logic is coherent, and technical details are sufficient.
- Value: ⭐⭐⭐⭐ Label-free multiview detection has significant practical value (low deployment cost), though the application scope is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] PTC-Depth: Pose-Refined Monocular Depth Estimation with Temporal Consistency](../../CVPR2026/autonomous_driving/ptc-depth_pose-refined_monocular_depth_estimation_with_temporal_consistency.md)
- [\[ICCV 2025\] DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation](dist-4d_disentangled_spatiotemporal_diffusion_with_metric_depth_for_4d_driving_s.md)
- [\[ICCV 2025\] EMD: Explicit Motion Modeling for High-Quality Street Gaussian Splatting](emd_explicit_motion_modeling_for_high-quality_street_gaussian_splatting.md)
- [\[ICCV 2025\] DriveX: Omni Scene Modeling for Learning Generalizable World Knowledge in Autonomous Driving](drivex_omni_scene_modeling_for_learning_generalizable_world_knowledge_in_autonom.md)
- [\[NeurIPS 2025\] Towards Predicting Any Human Trajectory in Context](../../NeurIPS2025/autonomous_driving/towards_predicting_any_human_trajectory_in_context.md)

</div>

<!-- RELATED:END -->
