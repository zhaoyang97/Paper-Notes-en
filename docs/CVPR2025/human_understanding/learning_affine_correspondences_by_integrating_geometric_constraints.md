---
title: >-
  [Paper Note] Learning Affine Correspondences by Integrating Geometric Constraints
description: >-
  [CVPR 2025][Human Understanding][affine correspondence] This paper proposes DenseAffine, a new framework for estimating affine correspondences that integrates dense matching with geometric constraints. It employs a two-stage decoupled training strategy: first training a dense point matcher with a Sampson distance loss, and subsequently freezing the matcher to train a local affine transformation extractor using an affine Sampson distance loss. The method achieves state-of-the-…
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "affine correspondence"
  - "dense matching"
  - "epipolar geometry"
  - "Sampson distance"
  - "pose estimation"
date: 2026-05-08
content_hash: a0ea2475d3f31a59
---

# Learning Affine Correspondences by Integrating Geometric Constraints

**Conference**: CVPR 2025  
**arXiv**: [2504.04834](https://arxiv.org/abs/2504.04834)  
**Code**: [GitHub](https://github.com/stilcrad/DenseAffine)  
**Area**: Human understanding  
**Keywords**: affine correspondence, dense matching, epipolar geometry, Sampson distance, pose estimation

## TL;DR

This paper proposes DenseAffine, a new framework for estimating affine correspondences that integrates dense matching with geometric constraints. It employs a two-stage decoupled training strategy: first training a dense point matcher with a Sampson distance loss, and subsequently freezing the matcher to train a local affine transformation extractor using an affine Sampson distance loss. The method achieves state-of-the-art (SOTA) performance on both HPatches matching and MegaDepth pose estimation.

## Background & Motivation

**Background**: Affine correspondences (ACs) capture local affine transformations (incorporating scale, rotation, and shear deformation information), providing richer geometric information than pure point correspondences. Consequently, they significantly accelerate and improve the estimation accuracy of homographies, essential matrices, and other downstream tasks.

**Limitations of Prior Work**:
1. **Traditional detector-based methods** (MSER, Hessian-Affine, AffNet): Rely on sparse keypoint detection and perform poorly in weak-textured or repetitive-textured regions.
2. **Synthetic view methods** (ASIFT): Depend on detectors and incur high computational costs.
3. **Dense matching methods** (DKM, RoMa): Provide precise point correspondences but **lack local affine transformation estimation**.
4. Existing methods fail to fully utilize **epipolar geometric constraints** to supervise the learning of affine transformations.

**Core Motivation**: To integrate the high-precision point matching capability of dense matchers with the effective supervision of geometric constraints (epipolar constraints), establishing a unified framework from point correspondence to complete affine correspondence.

## Method

### Overall Architecture

Two-stage cascade:
1. **First Sub-network (Feature Matching Module)**: Extracts precise point correspondences via dense matching.
2. **Second Sub-network (Local Affine Transformation Estimation Module)**: Estimates the local affine transformation for each matched point pair.

The training employs a **decoupled training** strategy: first training the matching sub-network until convergence, then freezing it, and subsequently training the affine sub-network.

### Key Designs

#### 1. Dense Feature Matching Module

Dense matching paradigm based on DKM:
- A ResNet-50 encoder extracts multi-scale features (coarse-to-fine levels).
- Global Gaussian process regression yields coarse matches, followed by iterative refinement using a refiner.
- Innovation: Incorporates an **epipolar constraint loss based on Sampson distance** $L_{pc}$ to encourage the network to learn rich geometric representations.
- Point correspondences and 32×32 patches are sampled from the dense warp and certainty maps.

#### 2. Local Affine Transformation Estimation Module

The affine matrix is decomposed into three components (following AffNet):
- **Scale & Orientation**: Two independent fully connected networks $E_{o,s}$ predict the orientation $O$ and scale $S$ of each patch, utilizing a **probabilistic covariant loss** (discretizing angle/scale into classification tasks).
- **Residual Deformation**: An independent network $E_{aff}$ regresses the residual affine shape matrix $\bm{A}''$ (with a determinant constraint det=1).
- Final affine correspondence = synthesized $(P_i^A, P_i^B, O_i^{A\to B}, S_i^{A\to B}, A_i'')$.

#### 3. Affine Sampson Distance Constraint Loss

Core innovation — an affine loss function based on epipolar constraints:
- Exploits the geometric relationship between affine correspondences and the fundamental matrix $\bm{F}$: $(\bm{F}^T\bm{p}_1)_{(1:2)} + (\bm{A}^T\bm{F}\bm{p}_2)_{(1:2)} = 0$
- Formulates the affine Sampson distance $SD_A(E_{AC})$ to measure the consistency of the predicted affine transformation with the epipolar geometry.
- Training target: $L_{aff} = -\frac{1}{N}\sum_i SD_A(E_{AC}^i)$

This loss requires no ground-truth (GT) annotations of affine transformations, requiring only the fundamental matrix (which can be derived from depth or pose).

### Loss & Training

**Matching sub-network**: $L_m = \sum_l L_{warp} + \lambda L_{conf} + \gamma L_{pc}$
- $L_{warp}$: L2 distance between the predicted warp and the ground truth.
- $L_{conf}$: Binary cross-entropy of confidence.
- $L_{pc}$: Sampson distance epipolar constraint loss.

**Affine sub-network**: $L_{ext} = L_{aff} + L_{ori} + L_{sca}$
- $L_{aff}$: Affine Sampson distance loss.
- $L_{ori}, L_{sca}$: Probabilistic covariant loss for orientation and scale.

## Key Experimental Results

### Main Results

**HPatches Image Matching (MMAscore)**:

| Method | Overall | Illumination | Viewpoint |
|------|---------|-------------|-----------|
| SuperPoint | 0.658 | 0.715 | 0.606 |
| DISK | 0.763 | 0.813 | 0.716 |
| PoSFeat | 0.775 | 0.826 | 0.728 |
| DKM | 0.819 | 0.869 | 0.772 |
| RoMa | 0.843 | 0.901 | 0.789 |
| **Ours** | **0.851** | **0.908** | **0.798** |

**MegaDepth Relative Pose Estimation (AUC)**:

| Method | @5° | @10° | @20° |
|------|-----|------|------|
| LoFTR | 52.8 | 69.2 | 81.2 |
| ASpanFormer | 55.3 | 71.5 | 83.1 |
| DKM | 62.1 | 76.7 | 86.4 |
| RoMa | 63.4 | 77.8 | 87.2 |
| **Ours (AC+GC-RANSAC)** | **64.7** | **78.3** | **87.6** |

### Ablation Study

**Affine Frame Accuracy (HPatches)**:

| Method | Euclidean Distance ↓ | Cosine Similarity ↑ |
|------|---------------------|---------------------|
| VLFeat | 0.202 | 0.988 |
| AffNet | 0.264 | 0.973 |
| ASIFT | 0.329 | 0.894 |
| **Ours** | **0.123** | **0.994** |

Compared to the best baseline VLFeat, the affine matrix estimation error (Euclidean distance) is reduced by **39%**.

### Key Findings

- Combining dense matching with geometric constraints significantly outperforms traditional sparse detector-based paradigms.
- The affine Sampson distance loss acts as a core innovation, enabling the network to learn geometrically consistent affine transformations **without requiring ground-truth affine annotations**.
- Decoupled training converges faster, consumes less memory, and yields superior final performance compared to joint end-to-end training.
- Utilizing affine correspondences for GC-RANSAC pose estimation provides higher accuracy than standard point correspondences paired with RANSAC.

## Highlights & Insights

1. **Elegant Unsupervised Scheme**: The affine Sampson distance loss requires only the fundamental matrix instead of ground-truth affine transformations, dramatically reducing annotation costs.
2. **New Application of Dense Matching**: It represents the first attempt to extend dense warping to full affine correspondence estimation, bypassing the benchmarks of sparse detectors.
3. **Decoupled Training Strategy**: It prevents loss ambiguities caused by weak supervision and practically outperforms end-to-end joint training.
4. **Practical Application Value**: Affine correspondences can solve homography with 2 ACs (instead of 4 PCs) and estimate essential matrices with 1 AC (instead of 5 PCs), substantially enhancing RANSAC efficiency.

## Limitations & Future Work

1. The taxonomy area designated as "human_understanding" appears to be inaccurate and would be better categorized under **image_matching / geometric estimation**.
2. The patch size is fixed at 32×32, whereas **adaptive patch scaling** could potentially boost performance under large deformation scenarios.
3. Currently, tests are constrained to rigid scenes (planar/general scenes); thus, its performance on **non-rigid deformations** remains unknown.
4. The scale and orientation estimation depends on discretized classification; a **continuous regression** formulation might yield more precise outcomes.

## Related Work & Insights

- **DKM/RoMa**: SOTA in dense matching. This work expands on them by adding a dimension of affine estimation.
- **AffNet** (Mishkin et al.): Pioneer in learning affine-covariant regions; this study substantially surpasses its accuracy.
- **GC-RANSAC** (Barath et al.): Graph-cut RANSAC, which can be integrated with affine correspondences to further improve pose estimation.
- **Insight**: Epipolar geometric constraints as annotation-free supervision signals can be extended to other tasks requiring geometric transformation learning (e.g., optical flow and scene flow estimation).

## Rating

⭐⭐⭐⭐ — Well-designed method, novel and practical geometric constraint loss, and SOTA performance in both matching and pose estimation tasks. The decoupled training strategy is simple and highly effective.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RUBIK: A Structured Benchmark for Image Matching across Geometric Challenges](rubik_a_structured_benchmark_for_image_matching_across_geometric_challenges.md)
- [\[CVPR 2025\] SemGeoMo: Dynamic Contextual Human Motion Generation with Semantic and Geometric Guidance](semgeomo_dynamic_contextual_human_motion_generation_with_semantic_and_geometric_.md)
- [\[CVPR 2026\] Geometric Neural Distance Fields for Learning Human Motion Priors](../../CVPR2026/human_understanding/geometric_neural_distance_fields_for_learning_human_motion_priors.md)
- [\[CVPR 2025\] WildAvatar: Learning In-the-Wild 3D Avatars from the Web](wildavatar_learning_in-the-wild_3d_avatars_from_the_web.md)
- [\[ICCV 2025\] HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation](../../ICCV2025/human_understanding/hcceposebf_predicting_front_back_surfaces_to_construct_ultra-dense_2d-3d_corresp.md)

</div>

<!-- RELATED:END -->
