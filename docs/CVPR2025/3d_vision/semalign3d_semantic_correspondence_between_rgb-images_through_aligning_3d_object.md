---
title: >-
  [Paper Note] SemAlign3D: Semantic Correspondence Between RGB-Images Through Aligning 3D Object-Class Representations
description: >-
  [CVPR 2025][3D Vision][Semantic correspondence] Leverages monocular depth estimation to construct object-class 3D representations, aligning them with input images at inference time by minimizing an alignment energy function (combining semantic and spatial likelihood). This method improves the overall PCK@0.1 on SPair-71k from 85.6% to 88.9%, with gains exceeding 10 percentage points across three categories.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Semantic correspondence"
  - "3D object representations"
  - "Monocular depth estimation"
  - "Point cloud alignment"
  - "Gradient optimization"
date: 2026-05-08
content_hash: c0314effb9d17b44
---

# SemAlign3D: Semantic Correspondence Between RGB-Images Through Aligning 3D Object-Class Representations

**Conference**: CVPR 2025  
**arXiv**: [2503.22462](https://arxiv.org/abs/2503.22462)  
**Code**: [https://dub.sh/semalign3d](https://dub.sh/semalign3d)  
**Area**: 3D Vision  
**Keywords**: Semantic correspondence, 3D object representations, Monocular depth estimation, Point cloud alignment, Gradient optimization

## TL;DR

Leverages monocular depth estimation to construct object-class 3D representations, aligning them with input images at inference time by minimizing an alignment energy function (combining semantic and spatial likelihood). This method improves the overall PCK@0.1 on SPair-71k from 85.6% to 88.9%, with gains exceeding 10 percentage points across three categories.

## Background & Motivation

**Background**: Semantic correspondence aims to establish correspondences between images based on semantic meaning rather than exact visual similarity, with broad applications in robotic policy learning, image editing, style transfer, and other fields. Large vision models (such as DinoV2) have significantly advanced this field through deep features.

**Limitations of Prior Work**: Although large vision models reliably capture local semantics, they struggle to capture global geometric relationships between semantic object regions. Under extreme viewpoint changes or symmetric objects between two images, the performance of existing methods (such as GeoAware) drops significantly.

**Key Challenge**: Pure 2D feature matching lacks 3D geometric constraints and fails to handle large viewpoint changes effectively—the 2D features of the same object part from different viewpoints may have very low similarity, yet their geometric relationship in 3D space is stable.

**Goal**: To learn class-level 3D object representations and align this 3D representation with object instances in images at inference time, achieving semantic correspondence in a more robust and data-efficient manner.

**Key Insight**: To leverage recent advances in monocular depth estimation (DepthAnythingV2). Although monocular depth estimation is imperfect, it is sufficient to construct coherent class-level 3D object representations from sparsely annotated image datasets.

**Core Idea**: To construct a 3D class-level point cloud representation (containing both geometric and semantic information) from a sparsely annotated image collection using monocular depth estimation + LVM features, and perform gradient descent at inference time to minimize an alignment energy function, thereby finding the optimal alignment between the 3D representation and the image instances.

## Method

### Overall Architecture

The method consists of two stages: (1) Offline construction phase—constructing 3D class representations (sparse keypoint clouds + dense point clouds, with each point bearing a semantic feature vector) from $n$ sparsely annotated images per category; (2) Inference phase—given two input images, aligning the class-level 3D representation with each image individually (minimizing an alignment loss), and then establishing semantic correspondence between the two images using the 3D representation as a bridge.

### Key Designs

1. **Building 3D Object-Class Representations**:

    - **Function**: To construct a 3D class-level representation containing geometric and semantic information from a collection of sparsely annotated images.
    - **Mechanism**: Performed in three steps. **Keypoint world coordinates**: DepthAnythingV2 is used to estimate depth. Since camera intrinsics are unknown, the focal length is estimated by optimizing focal length parameters to minimize the variance of scale-invariant geometric features (edge angles $A_{ijkl}$ and edge length ratios $R_{ijkl}$) across images. **Sparse 3D representation**: A Beta distribution is fitted to the geometric features to iteratively compute the maximum likelihood position of each keypoint in 3D space, while extracting semantic feature vectors using a pre-trained GeoAware model. **Dense 3D representation**: Barycentric parameterization is used to align the depth point clouds of each image to the sparse canonical point cloud. After merging all aligned point clouds, k-means clustering is applied, retaining cluster centers with density exceeding a threshold, with semantic features computed as neighborhood averages.
    - **Design Motivation**: Although monocular depth estimation is imperfect, its geometric feature statistics across images are sufficiently stable to construct coherent class-level representations. The 3D representation does not model a specific instance (e.g., an A380 airliner), but rather the object category (e.g., airliner).

2. **Aligning 3D Model Representations**:

    - **Function**: To find the optimal alignment between the 3D class representation and the object instances in the images at inference time.
    - **Mechanism**: The optimization variables are the sparse keypoint coordinates $C_{sparse}$ and the focal length $f$ (with the dense point cloud following the deformation through barycentric coordinate parameterization). The alignment loss $\mathcal{L}_{align}$ consists of four terms: (1) **Reconstruction loss**: Maximize the image likelihood $P(\text{image}|C_{sparse}, f)$, where the likelihood of each pixel is defined as the maximum product of semantic likelihood $p_{sem}$ (cosine similarity between 3D point semantic features and image patch features) and spatial likelihood $p_{spatial}$ (Gaussian distribution of the projected 3D point position); (2) **Geometric consistency loss**: Maintain the angles and edge length ratios between keypoints according to the fitted Beta distribution; (3) **Background mask penalty**: Penalize points projected outside the object segmentation mask; (4) **Depth regularization**: Constrain the mean of the keypoint z-coordinate to accelerate convergence.
    - **Design Motivation**: Since the initial 3D representation is already geometrically consistent, the optimization only needs to maintain this consistency (which is much easier than finding a consistent configuration from scratch). The $\sigma$ of the spatial likelihood is annealed from large to small, enabling a coarse-to-fine alignment.

3. **Semantic Correspondence via 3D Bridge**:

    - **Function**: To bridge the semantic correspondence between two images via the 3D representation.
    - **Mechanism**: Solve for the 3D alignment for the two images separately to obtain $(C_{sparse,1}^*, f_1^*)$ and $(C_{sparse,2}^*, f_2^*)$. Given a query point $p_1$ in image 1, first find the point $i^*$ in the 3D representation with the maximum reconstruction likelihood, and then find the pixel position in image 2 with the maximum projection likelihood of that point. The variances of $p_{spatial}$ and $p_{sem}$ can be adjusted based on the degree of trust in the semantic/spatial terms.
    - **Design Motivation**: The 3D representation acts as an intermediate bridge, converting the difficult problem of extreme viewpoint changes into two relatively simple 3D-2D alignment tasks.

### Loss & Training

- **Offline construction phase**:
    - Focal length optimization: Minimize the variance of cross-image geometric features (angles and edge length ratios).
    - 3D keypoint positions: Maximum likelihood estimation based on Beta distribution.
    - Semantic features: Extracted by pre-trained GeoAware model and averaged across images.
- **Inference alignment phase**:
    - $$\mathcal{L}_{align} = w_{reconstruct} \cdot \mathcal{L}_{reconstruct} + w_{geom} \cdot \mathcal{L}_{geom} + w_{background} \cdot \mathcal{L}_{background} + w_{depth} \cdot \mathcal{L}_{depth}$$
    - Optimization strategy: Anneal $\sigma$ (spatial likelihood bandwidth) from large to small; transition weights such that $w_{dense}$ starts large and decreases, while $w_{sparse}$ starts small and increases, moving from coarse global alignment to fine keypoint localization.
    - Optimized using gradient descent.

## Key Experimental Results

### Main Results (SPair-71k, PCK@0.1)

| Method | Type | Aero | Bike | Bottle | Chair | Cat | Dog | TV | **All** |
|------|------|------|------|--------|-------|-----|-----|-----|---------|
| DINOv2+NN | U | 72.7 | 62.0 | 40.4 | 36.2 | 71.1 | 64.6 | 24.2 | 55.6 |
| SphericalMaps | U | 74.8 | 64.5 | 52.7 | 47.7 | 82.4 | 67.3 | 59.1 | 67.3 |
| GeoAware (Prev. SOTA) | S | 92.0 | 76.1 | 70.5 | 73.4 | 92.7 | 90.5 | 85.3 | 85.6 |
| **SemAlign3D (Ours)** | S | **95.6** | **80.4** | **82.2** | **88.3** | 91.4 | **91.3** | **96.1** | **88.9** |

Performance in three categories improved by more than 10 percentage points: Bottle (+11.7), Chair (+14.9), TV (+10.8).

### Ablation Study

| Loss Components | All PCK@0.1 |
|---------|------------|
| Reconstruction loss only | ~84 |
| + Geometric consistency | ~87 |
| + Background mask | ~88 |
| + Depth regularization (Full) | 88.9 |

### Key Findings

- **Most significant improvements in categories with large viewpoint changes**: Chair (+14.9), TV (+10.8), Bottle (+11.7), indicating that 3D geometric constraints are highly effective at resolving extreme viewpoint variations.
- Slight drop or comparable performance on non-rigid categories like Cat and Dog, as the 3D rigid body assumption is not fully applicable.
- Although monocular depth estimation is imperfect, its statistical properties are sufficiently stable across images.
- The method only requires sparse annotation (approx. 20 keypoints per image), demonstrating high data efficiency.
- The introduction of dense point clouds is crucial to avoiding local minima.

## Highlights & Insights

- **Elegant problem reformulation**: Reformulates the 2D-2D semantic correspondence problem into two 2D-3D alignment problems, utilizing the 3D representation as a bridge.
- **No end-to-end training required**: 3D representation construction is a purely geometric and statistical process, and alignment is optimized at inference time, requiring no GPU training.
- **Extracting useful information from imperfect depth estimation**: Avoids the impact of absolute depth errors from monocular estimations by utilizing scale-invariant geometric features and statistics.
- **Practical annealing strategies**: Annealing of spatial likelihood $\sigma$ and scheduling of dense/sparse weights effectively prevent falling into local minima.
- Significant improvements in Chair, TV, and Bottle categories are highly compelling.

## Limitations & Future Work

- The 3D rigid/quasi-rigid body assumption limits performance on non-rigid categories (e.g., cats, dogs, humans).
- Running gradient descent optimization during inference is slow (requires multiple iterations per image).
- 3D representations must be constructed independently for each category and cannot be shared across categories.
- Relies heavily on pre-trained models (GeoAware, DepthAnythingV2, SegmentAnything).
- Future directions: Explore deformable 3D representations to handle non-rigid objects; accelerate alignment optimization during inference; investigate end-to-end learning of 3D representations.

## Related Work & Insights

- **GeoAware**: Direct predecessor and main baseline, which improves semantic correspondence by fine-tuning DinoV2 features.
- **SphericalMaps**: Introduces weak geometric priors by mapping features to spherical coordinates, but fails to handle complex objects.
- **DepthAnythingV2**: Provides high-quality monocular depth estimation, making classical "3D representation from 2D images" possible.
- Insight: Even imperfect 3D information (like monocular depth estimation) can drastically improve the performance of originally pure 2D methods, as long as its geometric constraints are extracted and leveraged in an appropriate manner.

## Rating

| Dimension | Score (1-10) |
|------|------------|
| Novelty | 9 |
| Technical Depth | 8 |
| Experimental Thoroughness | 7 |
| Writing Quality | 8 |
| Value | 7 |
| Overall Rating | 7.8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection](learning_class_prototypes_for_unified_sparse-supervised_3d_object_detection.md)
- [\[CVPR 2025\] StdGEN: Semantic-Decomposed 3D Character Generation from Single Images](stdgen_semantic-decomposed_3d_character_generation_from_single_images.md)
- [\[ICCV 2025\] Unified Category-Level Object Detection and Pose Estimation from RGB Images using 3D Prototypes](../../ICCV2025/3d_vision/unified_category-level_object_detection_and_pose_estimation_from_rgb_images_usin.md)
- [\[ICCV 2025\] FROSS: Faster-than-Real-Time Online 3D Semantic Scene Graph Generation from RGB-D Images](../../ICCV2025/3d_vision/fross_faster-than-real-time_online_3d_semantic_scene_graph_generation_from_rgb-d.md)
- [\[CVPR 2025\] CADDreamer: CAD Object Generation from Single-view Images](caddreamer_cad_object_generation_from_single-view_images.md)

</div>

<!-- RELATED:END -->
