---
title: >-
  [Paper Note] GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence
description: >-
  [ECCV 2024][Human Understanding][Category-Level Pose Estimation] This paper proposes GS-Pose, a method that projects 2D semantic features from a pre-trained vision foundation model (DINOv2) into 3D space, combining them with geometric features via a Transformer matching network for category-level 9D object pose estimation. Highly data-efficient, it achieves state-of-the-art performance on multiple real-world datasets with training on only 10 synthetic 3D models.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Category-Level Pose Estimation"
  - "Geometric-Semantic Features"
  - "DINOv2"
  - "Transformer Matching"
  - "Synthetic Data Training"
date: 2026-05-08
content_hash: f5071157650a160d
---

# GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence

**Conference**: ECCV 2024  
**arXiv**: [2311.13777](https://arxiv.org/abs/2311.13777)  
**Code**: None  
**Area**: Object Pose Estimation / 3D Vision  
**Keywords**: Category-Level Pose Estimation, Geometric-Semantic Features, DINOv2, Transformer Matching, Synthetic Data Training

## TL;DR

This paper proposes GS-Pose, a method that projects 2D semantic features from a pre-trained vision foundation model (DINOv2) into 3D space, combining them with geometric features via a Transformer matching network for category-level 9D object pose estimation. Highly data-efficient, it achieves state-of-the-art performance on multiple real-world datasets with training on only 10 synthetic 3D models.

## Background & Motivation

Category-level object pose estimation aims to estimate the pose of **unseen instances** within the same object category, which is a fundamental problem in computer vision and robotics. Existing methods suffer from three major issues:

**Data Hunger of RGB Methods**: Methods using raw RGB inputs (e.g., NOCS) are effective at the instance level because texture is highly correlated with pose. However, they struggle heavily at the category level—objects in the same category vary significantly in appearance (e.g., cups with different colors and patterns). Training data must cover a vast range of appearance variations, requiring massive annotated real-world data or highly customized photorealistic simulators.

**Semantic Deficiency of Pure Geometric Methods**: To reduce the domain gap, some approaches (e.g., CPPF) rely solely on depth information. Intra-category variations in depth space are indeed smaller than in color space, allowing training solely on synthetic data and testing on real-world scenarios. However, pure geometric methods **lack semantic information**—when different parts of an object have similar geometry (e.g., matching the keyboard and screen of a laptop, or indistinguishable local geometries of a cup except for its handle), spatial ambiguities arise. For instance, CPPF even requires an auxiliary classifier specifically trained to distinguish the upper and lower parts of laptops.

**Data Efficiency Bottleneck**: Existing methods typically require a large number of training objects (e.g., CPPF uses 210 synthetic models), and their generalization to unseen instances remains limited.

**Key Insight**: Self-supervised training of pre-trained vision foundation models (such as DINOv2) enables their features to capture a **semantic understanding robust to texture and appearance variations**—analogous parts of different instances within the same category (e.g., cup handles, laptop keyboards) yield highly consistent semantic features, irrespective of color or material. Projecting these 2D semantic features into 3D space provides simultaneous access to geometric and semantic information, drastically reducing the demand for training data.

## Method

### Overall Architecture

The pipeline of GS-Pose consists of three steps:
1. **3D Semantic Feature Construction** (offline): For a single reference CAD model of each category, RGB images are rendered from multi-view perspectives to extract DINOv2 features, which are then projected onto the 3D point cloud and averaged across views to yield canonical 3D semantic features.
2. **Feature Matching** (inference): Given a single-view RGB-D observation of a query instance, partial 3D semantic features are extracted and densely matched against the complete reference 3D features using a Transformer-based matching network.
3. **Pose Estimation**: Based on the established matches, the 9D pose (rotation $R$, translation $t$, and scale $s$) is recovered using the Umeyama algorithm integrated with RANSAC.

### Key Designs

#### 1. Semantic Feature Wrapping

The detailed procedure for lifting 2D semantic features from DINOv2 to 3D space is as follows:
- Uniformly sample camera poses around the reference object to ensure each point on the model is visible in at least one view.
- Extract semantic features from the rendered RGB images using DINOv2 and resize them back to the original image dimensions.
- Evaluate the visibility of mesh vertices in each frame.
- Project visible points onto the 2D feature maps using camera intrinsic and extrinsic parameters to retrieve corresponding semantic features.
- Average the multi-view features for each point to eliminate viewing-angle discrepancies and noise.

The wrapped 3D semantic features preserve the key property of their 2D counterparts—enabling zero-shot semantic segmentation of object parts. However, directly applying them for matching yields suboptimal results, as semantic features are extremely similar across regions of the same semantic part, leading to matching ambiguities.

#### 2. Transformer Matching Network

To address the challenges in matching sparse, partial observations to a complete reference (where matching candidates expand significantly), a Transformer-matching network that integrates both geometric and semantic features is designed:

**Input Embedding**: Normalized point coordinates are embedded as geometric features using positional encoding, which are then added to the corresponding semantic features as the network input. The partial input $P$ contains $M$ points, and the complete reference input $Q$ contains $N$ points.

**Feature Fusion**: Multi-layer self-attention and cross-attention are employed:
- Self-attention layer: Independently aggregates features within the partial feature set $F^P$ and the complete feature set $F^Q$.
- Cross-attention layer: Facilitates interaction between partial and complete features to learn global cross-domain associations.

**Inlier Probability Prediction** (Key Innovation): Inspired by LightGlue, the network predicts inlier probabilities $\sigma^P$ and $\sigma^Q$ for each point to constrain the matching region and eliminate outlier candidate pairs. The assignment matrix is derived using the product of cosine similarity and the inlier probabilities:

$$\hat{A}_{i,j} = \sigma_i^P \cdot \sigma_j^Q \cdot A_{i,j} \quad \forall (i,j) \in P \times Q$$

Design Motivation: Matching a partial observation to a complete reference model is inherently harder than matching between two partial views, as the potential matching search space expands, compounding the risk of false positives. Inlier probability prediction steers the network to focus on valid matching regions, filtering out incorrect matches that lie outside the visible field of view.

#### 3. Symmetry Disambiguation

For axis-symmetric objects (such as bottles and bowls), multiple valid ground-truth (GT) poses exist, which causes conflicting training signals. The solution is to constrain the xz-plane (red-blue axis plane) of the object to always intersect with the camera origin, thereby extracting a unique GT pose. Cups are treated as axis-symmetric objects when their handles are not visible.

### Loss & Training

The training loss comprises three components:

1. **Partial Input Inlier Classification Loss** (BCE): $L_P = -\frac{1}{|P|}\sum_{i \in P}(\sigma_{i,gt}^P \log \sigma_i^P + (1-\sigma_{i,gt}^P)\log(1-\sigma_i^P))$
2. **Complete Reference Input Inlier Classification Loss** (BCE): $L_Q$, mathematically equivalent to $L_P$.
3. **Assignment Matrix Loss** (Focal Loss with $\gamma=2$): Distinguishes between positive match pairs $A_{pos}$ and negative match pairs $A_{neg}$:

$$L = -\frac{1}{|A_{pos}|}\sum_{\hat{A}_{i,j} \in A_{pos}}(1-\hat{A}_{i,j})^\gamma\log(\hat{A}_{i,j}) - \frac{1}{|A_{neg}|}\sum_{\hat{A}_{i,j} \in A_{neg}}\hat{A}_{i,j}^\gamma\log(1-\hat{A}_{i,j})$$

- Training uses only 10 ShapeNet models per category, with 40 rendered views per object.
- lr=1e-4, 100 epochs, Intel Xeon E5-2698 + Tesla V100.

## Key Experimental Results

### Main Results: NOCS REAL275 Dataset

| Method | Training Data | N(Syn) | N(Real) | 3D$_{25}$↑ | 3D$_{50}$↑ | 5°5cm↑ | 10°5cm↑ | 15°5cm↑ |
|------|---------|--------|---------|-----------|-----------|--------|---------|---------|
| NOCS | Syn+Real | 180 | 3 | 74.4 | 27.8 | 9.8 | 24.1 | 34.9 |
| DualPoseNet | Syn+Real | 180 | 3 | 82.3 | 57.3 | 36.1 | 67.8 | 76.3 |
| CPPF | Syn only | 210 | 0 | 78.2 | 26.4 | 16.9 | 44.9 | 50.8 |
| **GS-Pose** | **Syn only** | **10** | **0** | **82.1** | **63.2** | **28.8** | **60.1** | **73.6** |

Using only synthetic data (10 models), GS-Pose achieves a 3D$_{25}$ comparable to DualPoseNet trained with real-world data (82.1 vs 82.3), and significantly outperforms CPPF on 3D$_{50}$ by **36.8%** (63.2 vs 26.4).

### Ablation Study: Network Component Contribution

| Config | 3D Semantic Features | Feature Fusion Network | Inlier Probability | Symmetry Handling | 3D$_{25}$ | 3D$_{50}$ | 5°5cm | 10°5cm | 15°5cm |
|------|-----------|------------|---------|---------|----------|----------|-------|--------|--------|
| A1 | ✓ | | | | 2.2 | 0.1 | 5.8 | 20.3 | 32.0 |
| A2 | ✓ | ✓ | | ✓ | 64.9 | 48.9 | 19.9 | 49.3 | 66.9 |
| A3 | ✓ | ✓ | ✓ | | 72.3 | 44.5 | 12.7 | 40.5 | 56.7 |
| **A4** | **✓** | **✓** | **✓** | **✓** | **82.1** | **63.2** | **28.8** | **60.1** | **73.6** |

Ablation on the Number of Training Objects:

| Method | No. of Objects | 3D$_{25}$ | 3D$_{50}$ | 15°5cm |
|------|-------|----------|----------|--------|
| CPPF | 10 | 75.7 | 14.6 | 33.4 |
| CPPF | 210 | 78.2 | 26.4 | 50.8 |
| **GS-Pose** | **10** | **82.1** | **63.2** | **73.6** |
| GS-Pose | 40 | 82.1 | 63.8 | 74.9 |

### Evaluation on More Datasets

**Wild6D Dataset** (162 test objects, much larger than NOCS):

| Method | Training Data | 3D$_{25}$↑ | 3D$_{50}$↑ | 10°5cm↑ |
|------|---------|-----------|-----------|---------|
| DualPoseNet | Syn+Real | 90.0 | 70.0 | 36.5 |
| RePoNet-semi | Syn+Real+Wild | 84.7 | 70.3 | 42.5 |
| **GS-Pose** | **Syn only (10)** | **84.6** | **67.7** | **51.0** |

GS-Pose outperforms all methods (including those trained on real data) on the 10°5cm metric by 8.5%.

**SUN RGB-D Dataset** (Challenging indoor chairs category): GS-Pose outperforms CPPF by 20.8% on the 3D$_{10}$ metric (56.8 vs 36.0).

### Key Findings

1. **Directly matching raw 3D semantic features fails completely**: In the A1 configuration, 3D$_{25}$ is only 2.2%. Because semantic features exhibit minor differences within similar semantic regions, matching from partial to complete structures produces excessive false candidates, causing massive mismatches. This proves the absolute necessity of the Transformer fusion network.
2. **Inlier probability prediction is critical**: Removing it decreases 3D$_{50}$ by 14.3% and heavily degrades rotation/translation metrics.
3. **Symmetry handling is indispensable**: Without it, the 5°5cm metric drops from 28.8% to 12.7%, as conflicting ground-truth signals severely confuse the network.
4. **Exceptional data efficiency**: GS-Pose reaches saturated performance using only 10 synthetic models. In contrast, CPPF falls short of GS-Pose (trained on 10 models) even when trained with 210 models.
5. **Feature distribution alignment**: t-SNE visualization shows: Before fusion, the distributions of partial and complete features are disjointed, whereas they align accurately after fusion.

## Highlights & Insights

- **3D-fication of 2D Foundation Model Features**: Projecting DINOv2's self-supervised 2D features into 3D space is a highly clever design—harnessing the rich semantic capabilities of 2D foundation models while bypassing the limitations of immature 3D foundation models.
- **Qualitative Leap in Data Efficiency**: Reducing the required synthetic models from 210 to 10 while improving performance proves that semantic features fundamentally alleviate the dependency on geometric shape diversity.
- **Partial-to-Complete Matching Paradigm**: Unlike matching between two partial views, matching against a complete 3D reference enables 360° correspondences. Although this introduces extra difficulty due to expanded query space, the inlier probability mechanism serves as a crucial remedy.

## Limitations & Future Work

- Pose estimation is prone to failure when depth estimation is corrupted (e.g., transparent objects like bottles) or when the 2D segmentation is inaccurate.
- The method assumes that shape variations within the same category can be approximated via affine transformations (Umeyama's assumption), which may fail for objects with large non-rigid deformations.
- It is limited to single-view observations and has not yet been extended to multi-view fusion (noted as future work by the authors).
- The deformation of the target object relative to the reference model is not explicitly estimated, which may affect scale and pose accuracy.
- The symmetry handling strategy is relatively simple (fixed constraint on the xz-plane); more complex symmetry profiles may require more sophisticated solutions.

## Related Work & Insights

- **Doing More with Less**: The paradigm of leveraging the semantic richness of foundation models to substitute large amounts of annotated training data is highly generalizable to other 3D understanding tasks.
- **2D-to-3D Feature Projection**: The method of projecting multi-view averaged features into 3D space is simple and effective, mapping well to other scenarios requiring 3D semantic representations.
- **Matching Network Design**: The combination of self-attention, cross-attention, and inlier probability—originating from OnePose/LightGlue—shows remarkable effectiveness in partial-to-complete matching settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of using semantic features from 2D foundation models for 3D pose estimation is highly novel, bringing substantial gains in data efficiency.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluations across three datasets, accompanied by exceptionally detailed component, data-volume ablations, and t-SNE analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly articulated motivation, complete methodological description, and fair comparisons.
- **Value**: ⭐⭐⭐⭐⭐ — Requires only 10 synthetic models to perform robustly in real-world environments, showing exceptional practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] U-COPE: Taking a Further Step to Universal 9D Category-Level Object Pose Estimation](u-cope_taking_a_further_step_to_universal_9d_category-level_object_pose_estimati.md)
- [\[ECCV 2024\] LaPose: Laplacian Mixture Shape Modeling for RGB-Based Category-Level Object Pose Estimation](lapose_laplacian_mixture_shape_modeling_for_rgb-based_category-level_object_pose.md)
- [\[ECCV 2024\] FoundPose: Unseen Object Pose Estimation with Foundation Features](foundpose_unseen_object_pose_estimation_with_foundation_features.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](../../CVPR2025/human_understanding/gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [\[ICCV 2025\] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation](../../ICCV2025/human_understanding/cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)

</div>

<!-- RELATED:END -->
