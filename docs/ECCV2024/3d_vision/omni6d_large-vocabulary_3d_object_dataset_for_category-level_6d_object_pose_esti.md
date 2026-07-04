---
title: >-
  [Paper Note] Omni6D: Large-Vocabulary 3D Object Dataset for Category-Level 6D Object Pose Estimation
description: >-
  [ECCV 2024][3D Vision][6D Pose Estimation] Constructs **Omni6D**, the first large-scale category-level 6DoF pose estimation RGBD dataset. It covers **166 categories, 4,688 instances, and 800,000 images**, far exceeding existing datasets like NOCS (only 6 categories). It also proposes a symmetry-aware evaluation metric and a progressive fine-tuning strategy.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "6D Pose Estimation"
  - "Category-Level"
  - "Large-Vocabulary Dataset"
  - "Symmetry-Aware Metric"
  - "benchmark"
date: 2026-05-08
content_hash: 79350b17a16e28ab
---

# Omni6D: Large-Vocabulary 3D Object Dataset for Category-Level 6D Object Pose Estimation

**Conference**: ECCV 2024  
**arXiv**: [2409.18261](https://arxiv.org/abs/2409.18261)  
**Code**: Yes (Dataset publicly available)  
**Area**: 3D Vision  
**Keywords**: 6D Pose Estimation, Category-Level, Large-Vocabulary Dataset, Symmetry-Aware Metric, benchmark

## TL;DR

Constructs **Omni6D**, the first large-scale category-level 6DoF pose estimation RGBD dataset. It covers **166 categories, 4,688 instances, and 800,000 images**, far exceeding existing datasets like NOCS (only 6 categories). It also proposes a symmetry-aware evaluation metric and a progressive fine-tuning strategy.

## Background & Motivation

6D pose estimation aims to predict the position, orientation, and size of objects from RGB(D) images. Instance-level methods require known CAD models and suffer from poor generalization, while category-level methods learn category priors to generalize to unseen instances but are limited by the database's category coverage.

Existing category-level datasets suffer from critical bottlenecks:

| Dataset | Categories | Instances | Images |
|--------|:---:|:---:|:---:|
| NOCS-CAMERA | 6 | 1,085 | 0.3M |
| NOCS-REAL | 6 | 42 | 8k |
| Wild6D | 5 | 1,722 | 1M |
| **Omni6D** | **166** | **4,688** | **0.8M** |

Three core challenges: **(1)** extremely few categories ($\le 6$ categories), failing to reflect real-world object diversity; **(2)** overly simplified scenes lacking realistic challenges such as occlusions; **(3)** neglect of rotational symmetry—many daily objects (cups, bowls, etc.) maintain the same appearance when rotated around a certain axis, leading traditional metrics to incorrectly penalize such correct predictions.

## Method

### Overall Architecture

The contributions of Omni6D are three-fold: **Dataset + Evaluation Metric + Fine-tuning Strategy**.

1. Collects 4,688 real-scanned 3D models across 166 categories and uses physical simulation to render 800,000 RGBD images with occlusions, multi-views, and various lighting conditions.
2. Proposes a three-axis symmetry-aware evaluation metric to properly handle rotational invariance.
3. Designs a progressive fine-tuning method scaling from few categories to many categories.

### Key Designs

#### 1. Dataset Construction

**Function**: Creates a large-scale RGBD dataset covering 166 daily object categories.

**Data Source**: Collects 4,688 high-resolution textured mesh models from OmniObject3D, utilizing real-scanned models from Shining 3D and Artec Eva 3D scanners instead of purely synthetic models.

**Scene Rendering**: Uses BlenderProc 2.5.0 with 9 room models from the Replica dataset as backgrounds:

- Randomly selects 6-8 object instances for each scene.
- Objects perform **free-fall** inside the rooms (instead of simple placement), generating natural random dispersion and occlusions.
- Objects are randomly scaled by 0.8-1.2x.
- The camera randomly selects 10 positions within an elevation angle range of 30-90 degrees.
- Provides RGB images, depth maps, NOCS maps, instance masks, 6D poses, and size annotations.

**Design Motivation**: Free-fall placement produces a massive number of occluded scenes. Data space analysis confirms that the difference between the point cloud centroid and the object centroid in Omni6D is larger than that in CAMERA/REAL, verifying a higher occlusion rate. The more uniform angular deviation distribution in all directions forces the models to handle bottom/side views.

#### 2. Symmetry-Aware Evaluation Metric

**Function**: Designs a 6D pose evaluation metric that correctly handles rotationally symmetric objects.

**Core Problem**: Many objects possess rotational invariance. For example, a cup rotated by 90 degrees around the y-axis maintains the same appearance; traditional metrics would penalize this as a rotational prediction error.

**Symmetry Classification**:

- **Sym-0**: No rotational invariance.
- **Sym-1**: Invariant to arbitrary rotation around a certain axis (e.g., cylinders).
- **Sym-2**: Invariant to integer multiples of 90-degree rotations around a certain axis.
- **Sym-3**: Invariant to integer multiples of 180-degree rotations around a certain axis.

**Symmetry-Aware Metric**: For the three axes x, y, and z, the symmetry types $n_x, n_y, n_z$ are annotated respectively. The smallest error among all symmetry-equivalent rotations is calculated:

$$L_s = \min_{\theta_x \in \Theta_{n_x}, \theta_y \in \Theta_{n_y}, \theta_z \in \Theta_{n_z}} L(R^*_{\theta_x, \theta_y, \theta_z}, R)$$

where $\Theta_0=\{0\}$, $\Theta_2=\{0,90,180,270\}$, and $\Theta_3=\{0,180\}$. The computation is simplified by leveraging the compactness of Euler angles, requiring at most $4^3=64$ comparisons (instead of $360^3$).

**Design Motivation**: Each instance is individually annotated with symmetry (which may vary among different instances of the same category), making it more precise than category-level annotations. Experiments demonstrate that models' performance drops significantly under non-symmetry-aware metrics.

#### 3. Progressive Fine-Tuning Strategy

**Function**: Efficiently transfers models trained on NOCS to large-scale category scenarios.

**Mechanism**:

1. Starting from the CAMERA pre-trained model, first fine-tune it on Omni6D's cls3 (3 categories overlapping with CAMERA: bottle/bowl/cup).
2. In each step, double the number of categories (3->6->12->24->48), replicating the learned global features and old category parameters, and initializing parameters for new categories.
3. Continue fine-tuning on the expanded dataset.

**Design Motivation**: Training from scratch directly on 166 categories yields poor results due to data dispersion. Progressive fine-tuning reuses learned knowledge and consistently outperforms training from scratch on SPD and DualPoseNet.

### Loss & Training

- Instances are divided into training/validation/test sets in a 7:2:1 ratio.
- Ground truth masks are used to eliminate the impact of semantic segmentation on pose estimation.
- All baseline methods use a unified learning rate of 1e-4 and are trained on NVIDIA A100-80GB GPUs.

## Key Experimental Results

### Main Results

Performance of six representative methods on Omni6D (using the symmetry-aware metric):

| Method | Architecture | IoU50 | IoU75 | 5d2cm | 10d5cm | 5d | 2cm |
|------|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| SPD | implicit | 44.56 | 20.37 | 7.55 | 19.23 | 10.68 | 37.49 |
| SGPA | implicit | 36.34 | 14.44 | 4.78 | 15.03 | 8.49 | 25.57 |
| DualPoseNet | hybrid | 58.84 | 25.49 | 8.28 | 19.05 | 9.38 | 73.82 |
| RBP-Pose | hybrid | 35.92 | 4.66 | 0.37 | 0.80 | 0.75 | 39.73 |
| GPV-Pose | explicit | 15.28 | 0.26 | 0.10 | 0.96 | 2.25 | 5.31 |
| **HS-Pose** | explicit | **62.65** | **23.02** | 4.26 | 11.61 | 4.96 | **80.93** |

Key Findings: Performance discrepancies among different architectures are amplified on the large-scale dataset—implicit methods (SPD/SGPA) excel at rotation prediction, while explicit methods (HS-Pose/DualPoseNet) are superior at translation and size estimation.

### Ablation Study

Progressive fine-tuning vs. training from scratch (SPD, 5d2cm metric %):

| Categories | Training from Scratch | Progressive Fine-tuning |
|:---:|:---:|:---:|
| 3 | 9.52 | - |
| 6 | 14.96 | 18.06 |
| 12 | 12.92 | 16.83 |
| 24 | 11.90 | 14.71 |
| 48 | 8.48 | 11.01 |

Progressive fine-tuning outperforms training from scratch across all scales, showing a milder decay in performance. However, HS-Pose drops from 62.52% to 14.42% after fine-tuning, indicating that models optimal in few-category scenarios do not necessarily maintain their leading position in large-scale scenarios.

Cross-category generalization (trained on Omni6D, tested on 17 unseen categories):

| Method | IoU50 | 5d2cm | 2cm |
|------|:---:|:---:|:---:|
| SPD | 7.56 | 0.18 | 8.88 |
| DualPoseNet | **36.85** | **3.24** | **78.00** |
| HS-Pose | 36.75 | 1.54 | 79.95 |

DualPoseNet and HS-Pose exhibit the strongest generalization abilities. Translative generalization is relatively straightforward, while rotational generalization remains the primary challenge.

### Key Findings

1. **Shape-based categories are more critical than semantic categories**: After clustering by shape priors, pose estimation performance shows a positive correlation with the number of instances within each cluster, with a stronger correlation compared to semantic category analysis.
2. **Instance diversity is beneficial**: A larger within-class Chamfer distance (indicating higher diversity) leads to better pose estimation performance.
3. **Sim-to-Real is effective**: On REAL275, joint training with REAL+Omni6D ($5d2cm = 14.10\%$) significantly outperforms training solely on REAL ($8.76\%$).
4. **Visual realism**: In human evaluations, Omni6D scored $2.69 \pm 0.39$ (out of 5), surpassing CAMERA.

## Highlights & Insights

- **A nearly 30x increase in categories** (from 6 to 166), marking a vital step forward for category-level 6D pose estimation from "lab environments" to "real-world applications."
- **Symmetry-aware metric** is well-designed, filling the gap in multi-axis rotational symmetry evaluation.
- **Highly detailed dataset analysis**: Rich statistical references are provided, covering spatial distribution, angular deviation, shape prior clustering, and instance diversity.
- The progressive fine-tuning approach is simple yet effective, offering a practical solution for large-scale expansion.

## Limitations & Future Work

1. **Scenes are still rendered**: Although real-scanned models enhance visual realism, spatial layouts are still generated via simulation, maintaining a domain gap with the real world.
2. **Fine-tuning strategy is effective but naive**: Performance continues to degrade as categories increase, lacking a fundamental solution to large-scale category-level pose estimation.
3. **Lack of video sequences**: Currently, only static images are provided, leaving temporal consistency in video scenarios unaddressed.
4. **Limited scale of Omni6D-Real**: Containing only 30 scenes and 1k images, verification in real-world scenes remains insufficient.

## Related Work & Insights

- **NOCS (CVPR 2019)**: Proposed the Normalized Object Coordinate Space and a category-level dataset, but only with 6 categories.
- **Wild6D (NeurIPS 2022)**: Provided 1M real video frames with 5 categories; closer to reality but still category-limited.
- **SPD, DualPoseNet, HS-Pose**: Represent three paradigm architectures (implicit, hybrid, and explicit, respectively). Omni6D reveals their relative pros and cons under a large-scale scenario for the first time.
- **OmniObject3D**: The source of 3D models for Omni6D, providing high-quality real-scanned meshes.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Generates a leap in dataset scale and fills the gap with the symmetry-aware metric, though the methodological contribution is relatively incremental.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Exhaustive experiments with 6 baseline methods, diverse dataset splits, category-level analysis, generalization testing, fine-tuning strategies, and Sim2Real validation.
- **Writing Quality**: ⭐⭐⭐⭐ — Highly detailed dataset description with rich statistical analysis and visualization.
- **Value**: ⭐⭐⭐⭐⭐ — Serves as a vital infrastructure for category-level 6D pose estimation, pushing the field towards large-scale real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] 6DGS: 6D Pose Estimation from a Single Image and a 3D Gaussian Splatting Model](6dgs_6d_pose_estimation_from_a_single_image_and_a_3d_gaussia.md)
- [\[CVPR 2026\] PoseGaussian: 6D Pose Estimation for Unseen Objects via Sparse-View Object-Level 3D Gaussian Splatting](../../CVPR2026/3d_vision/posegaussian_6d_pose_estimation_for_unseen_objects_via_sparse-view_object-level_.md)
- [\[CVPR 2026\] Exploring 6D Object Pose Estimation with Deformation](../../CVPR2026/3d_vision/exploring_6d_object_pose_estimation_with_deformation.md)
- [\[CVPR 2026\] ComPose: A Unified Completion-Pose Framework for Robust Category-Level Object Pose Estimation](../../CVPR2026/3d_vision/compose_a_unified_completion-pose_framework_for_robust_category-level_object_pos.md)
- [\[CVPR 2026\] SE(3)-Equivariance with Geometric and Topological Guidance for Category-Level Object Pose Estimation](../../CVPR2026/3d_vision/se3-equivariance_with_geometric_and_topological_guidance_for_category-level_obje.md)

</div>

<!-- RELATED:END -->
