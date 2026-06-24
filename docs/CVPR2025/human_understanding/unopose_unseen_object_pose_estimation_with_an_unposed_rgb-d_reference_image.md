---
title: >-
  [Paper Note] UNOPose: Unseen Object Pose Estimation with an Unposed RGB-D Reference Image
description: >-
  [CVPR 2025][Human Understanding][Unseen object pose estimation] The UNOPose method and benchmark are proposed to estimate the 6DoF relative pose of unseen objects using only a single unposed RGB-D reference image. Through an $SE(3)$-invariant reference frame and overlap-aware matching, it achieves performance comparable to methods relying on CAD models.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Unseen object pose estimation"
  - "single reference image"
  - "SE(3) invariance"
  - "point cloud registration"
  - "overlap prediction"
date: 2026-05-08
content_hash: 594297c656c0efea
---

# UNOPose: Unseen Object Pose Estimation with an Unposed RGB-D Reference Image

**Conference**: CVPR 2025  
**arXiv**: [2411.16106](https://arxiv.org/abs/2411.16106)  
**Code**: [GitHub](https://github.com/shanice-l/UNOPose)  
**Area**: Human Understanding  
**Keywords**: Unseen object pose estimation, single reference image, SE(3) invariance, point cloud registration, overlap prediction

## TL;DR

The UNOPose method and benchmark are proposed to estimate the 6DoF relative pose of unseen objects using only a single unposed RGB-D reference image. Through an $SE(3)$-invariant reference frame and overlap-aware matching, it achieves performance comparable to methods relying on CAD models.

## Background & Motivation

- Most existing object pose estimation methods rely on CAD models or multiple reference views to cover the target object's appearance, leading to high annotation and preparation costs.
- Instance-level and category-level methods can only handle known objects or categories, presenting clear limitations in open-world applications.
- The single-reference-image setting faces massive challenges: the relative pose can vary across the entire $SE(3)$ space, losing the simplification of multi-view methods that select the nearest anchor.
- Occlusions, sensor noise, and extreme geometries can lead to minimal overlapping regions between viewpoints.
- Prior relative pose estimation methods (e.g., 3DAHV, DVMNet) operate only on the RGB modality and predict 3DoF rotation, failing to estimate the full 6DoF pose (including translation).
- A low-cost, general pose estimation scheme that operates with only a single RGB-D reference is highly demanded.

## Method

### Overall Architecture

UNOPose adopts a coarse-to-fine paradigm: (1) Segmenting unseen objects using SAM+DINOv2 to localize the target object in the query image; (2) Back-projecting RGB-D images into 3D point clouds, standardizing the object representation via an $SE(3)$-invariant Global Reference Frame (GRF), and performing coarse matching to obtain an initial pose estimation; (3) Executing fine matching on the initially aligned dense point clouds, utilizing Local Reference Frame (LRF) encoding to capture fine-grained geometric structures, and finally solving for the precise pose using RANSAC.

### Key Designs

**1. $SE(3)$-Invariant Global Reference Frame (GRF)**

- **Function**: Eliminates the impact of object pose and scale variations on matching, standardizing the object representation.
- **Mechanism**: Transforms the point cloud into a canonical coordinate system via a 7DoF transformation $\{\mathbf{R}_G, \mathbf{t}_G, s_G\}$. The origin is set at the object center (translation-invariant), the radius is normalized to 1 (scale-invariant), the rotation is determined by using the object center normal vector (the vector corresponding to the smallest singular value in SVD) for the z-axis, and the sum of weighted vectors projected onto the tangent plane for the x-axis.
- **Design Motivation**: Under the single-reference setting, the relative pose can cover the entire $SE(3)$ space, requiring pose and scale variations to be eliminated first to build correspondences effectively. Compared to methods requiring complex networks or PPF features, GRF transformations are highly computationally efficient.

**2. Overlap-Aware Correspondence Establishment**

- **Function**: Identifies reliable corresponding points in part-to-part matching scenarios and suppresses interference from non-overlapping regions.
- **Mechanism**: The network additionally predicts the confidence of each point being in the overlapping region, denoted as $\hat{O}_Q^c, \hat{O}_P^c$. This confidence is element-wise multiplied with the feature descriptors before computing the correlation matrix $\mathbf{X}^c = \text{softmax}[(\hat{O}_Q^c \odot \hat{F}_Q^c)(\hat{O}_P^c \odot \hat{F}_P^c)^\top]$. A learnable background token is also introduced to handle unmatched points.
- **Design Motivation**: In single-reference scenarios, occlusion and large viewpoint differences make the overlap ratio unpredictable. Indiscriminate matching introduces a large number of false correspondences, necessitating the automatic adjustment of weights for each corresponding point.

**3. Hierarchical Geometric Encoding (GRF + LRF)**

- **Function**: Performs fine matching on dense point clouds after coarse matching to capture local geometric details.
- **Mechanism**: In the fine stage, a local neighborhood is constructed for each point to calculate its Local Reference Frame (LRF), in a manner similar to GRF but applied to local point sets. Combining global positional encoding (mini-PointNet) and LRF encoding, the former provides global positional context while the latter captures fine-grained local geometric structures, making them complementary.
- **Design Motivation**: Residual errors after coarse matching need fine-grained geometric features for precise correction, and LRF guarantees the rotational invariance of local descriptors.

### Loss & Training

- Coarse stage: Negative log-likelihood loss for the correspondence matrix + binary cross-entropy loss for overlap prediction.
- Fine stage: Correspondence matrix loss + pose regression loss (ADD-style).
- Uses GeoTransformer as the geometric encoder and DINOv2 as the color feature encoder.
- Selects the optimal coarse pose via hypothesis sampling and scoring over $N_H$ triplet point pairs.

## Key Experimental Results

### Main Results

Based on the AR_BOP metric (average of YCB-V + LM-O + TUD-L) of the BOP Challenge:

| Method | Reference Type | AR_BOP |
|------|---------|--------|
| ICP (classical method) | Single Reference | 13.8 |
| FPFH + RANSAC | Single Reference | 28.5 |
| DVMNet | Single Reference | 42.9 |
| **UNOPose** | **Single Reference** | **70.9** |
| ZTE-PPF (CAD-based) | CAD model | 69.0 |
| Koenig-PPF (CAD-based) | CAD model | 75.1 |

### Ablation Study

| Configuration | YCB-V AR | LM-O AR | TUD-L AR |
|------|----------|---------|----------|
| w/o GRF | 62.1 | 43.2 | 71.8 |
| w/o Overlap Predictor | 68.3 | 49.7 | 80.5 |
| w/o LRF (fine) | 70.2 | 51.4 | 82.1 |
| Full UNOPose | **73.8** | **55.2** | **83.7** |

### Key Findings

1. UNOPose (70.9% AR_BOP) surpasses the CAD-based ZTE-PPF (69.0%), requiring only a single unposed reference.
2. GRF contributes the most; removing it significantly degrades performance, confirming the criticality of $SE(3)$-invariant standardization.
3. Compared to traditional methods (ICP 13.8%, FPFH 28.5%时), learning-based methods exhibit a massive advantage in the single-reference setting.
4. The overlap predictor brings particularly prominent improvements in low-overlap scenarios (large viewpoint differences).

## Highlights & Insights

- This work is the first to reduce the reference requirement for unseen object pose estimation to a single unposed RGB-D image, immensely simplifying the deployment workflow.
- The design of GRF is simple and elegant, constructing an invariant coordinate system via point cloud covariance matrix SVD, which is computationally efficient and learning-free.
- A standardized evaluation benchmark based on the BOP Challenge is constructed, facilitating evaluation and comparison for the community.
- The result that single-reference estimation surpasses some CAD-based methods is highly encouraging.

## Limitations & Future Work

- GRF is not robust to symmetric objects, where the normal vector directions can be ambiguous.
- The noise levels of depth data and sensor types significantly impact performance.
- Under severe occlusion, excessively small overlapping regions can still lead to failure.
- Future work can extend this to few-shot reference settings to further improve robustness.
- Extending to a pure RGB-only setting (without depth) can be explored.

## Related Work & Insights

- **FoundationPose / MegaPose**: Use CAD models to render multi-view images for pose estimation; this work demonstrates that a single reference can achieve comparable performance.
- **SAM-6D**: A method for establishing 3D-3D correspondences; UNOPose borrows its background token mechanism.
- **GeoTransformer**: A geometric Transformer for point cloud registration, which serves as the backbone of UNOPose's feature extraction.
- **Insight**: In scenarios such as robotic manipulation, users can enable robots to recognize and localize objects with just a single photo, drastically lowering the deployment barrier.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The single unposed reference setting is entirely new, and both GRF and overlap prediction designs are highly effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Standard BOP evaluations, comprehensive ablation studies, and comparison with multiple baselines.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem definition and rigorous mathematical derivation.
- **Value**: ⭐⭐⭐⭐⭐ — Drastically reduces the deployment barrier for unseen object pose estimation, offering extremely high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] One2Any: One-Reference 6D Pose Estimation for Any Object](one2any_one-reference_6d_pose_estimation_for_any_object.md)
- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](../../ICCV2025/human_understanding/raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)
- [\[ECCV 2024\] FoundPose: Unseen Object Pose Estimation with Foundation Features](../../ECCV2024/human_understanding/foundpose_unseen_object_pose_estimation_with_foundation_features.md)
- [\[CVPR 2025\] Co-op: Correspondence-based Novel Object Pose Estimation](co-op_correspondence-based_novel_object_pose_estimation.md)
- [\[ICCV 2025\] MixRI: Mixing Features of Reference Images for Novel Object Pose Estimation](../../ICCV2025/human_understanding/mixri_mixing_features_of_reference_images_for_novel_object_pose_estimation.md)

</div>

<!-- RELATED:END -->
