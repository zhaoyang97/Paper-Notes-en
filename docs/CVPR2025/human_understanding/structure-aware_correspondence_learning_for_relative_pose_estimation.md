---
title: >-
  [Paper Note] Structure-Aware Correspondence Learning for Relative Pose Estimation
description: >-
  [CVPR 2025][Human Understanding][Relative Pose Estimation] Proposed a structure-aware correspondence learning method (SAC-Pose) that learns keypoints representing object structures and directly regresses 3D-3D correspondences based on inter-image structure-aware features (without explicit feature matching), significantly improving the accuracy of relative pose estimation for unseen object categories.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Relative Pose Estimation"
  - "Structure-Aware Keypoints"
  - "3D Correspondence"
  - "Feature-Matching-Free"
  - "SVD Solver"
date: 2026-05-08
content_hash: 59825e0a4a098a60
---

# Structure-Aware Correspondence Learning for Relative Pose Estimation

**Conference**: CVPR 2025  
**arXiv**: [2503.18671](https://arxiv.org/abs/2503.18671)  
**Code**: [https://github.com/Cyhhzo02/SAC-Pose-code](https://github.com/Cyhhzo02/SAC-Pose-code)  
**Area**: Human Pose/3D Vision  
**Keywords**: Relative Pose Estimation, Structure-Aware Keypoints, 3D Correspondence, Feature-Matching-Free, SVD Solver

## TL;DR

Proposed a structure-aware correspondence learning method (SAC-Pose) that learns keypoints representing object structures and directly regresses 3D-3D correspondences based on inter-image structure-aware features (without explicit feature matching), significantly improving the accuracy of relative pose estimation for unseen object categories.

## Background & Motivation

1. **Background**: Relative pose estimation aims to estimate the relative rotation of objects from a pair of images, which is of great value for achieving category-agnostic pose estimation. Mainstream methods fall into three categories: 2D correspondence (SuperGlue/LoFTR), hypothesis-and-verification (RelPose/RelPose++), and 3D correspondence (DVMNet).
2. **Limitations of Prior Work**: 2D methods fail to match under large viewpoint changes or small overlapping regions; hypothesis-and-verification methods rely on discrete sampling, which is computationally expensive and cannot model continuous pose space; 3D methods lift 2D features to 3D voxels for dense matching, but the 3D feature inference in unobserved regions is unreliable, and the cubic complexity is very high.
3. **Key Challenge**: Existing 3D correspondence methods rely on explicit feature matching, but inferring 3D features of unobserved regions from single-view 2D surface features is inherently unreliable, leading to matching errors.
4. **Goal**: Design a method that establishes reliable 3D-3D correspondences without explicit feature matching.
5. **Key Insight**: Mimic human "assembly capability"—when humans see the front and back of a suitcase, even if the overlap is minimal, they can infer the assembly configuration through structural information (shape, handle position, color patterns).
6. **Core Idea**: Represent the object structure with a set of structured keypoints, and then directly regress 3D corresponding coordinates through structure-aware feature interactions, bypassing the matching step.

## Method

### Overall Architecture

Input query and reference images → Shared feature extractor + Symmetric attention → Structure-aware keypoint extraction (extracted independently) → Structure-aware correspondence estimation (self/cross-attention) → 2D keypoints lifting to 3D + Regressing 3D coordinates in the reference coordinate system → wSVD solver for relative rotation.

### Key Designs

1. **Structure-Aware Keypoint Extraction Module (SA-KPE)**

    - **Function**: Adaptively select sparse keypoints representing the object structure from feature maps.
    - **Mechanism**: Initialize a set of learnable queries $\mathbf{Q} \in \mathbb{R}^{N_{kpt} \times C}$, interact them with image features via cross-attention to obtain image-adaptive keypoint detectors $\tilde{\mathbf{Q}}_q$. Calculate the similarity between detectors and features to generate heatmaps $\mathbf{H}_q = \text{softmax}(\tilde{\mathbf{Q}}_q \cdot \mathbf{F}_q'^{\top})$, and then weight-average over the heatmaps to obtain keypoint coordinates and features. To prevent keypoints from clustering, an image reconstruction loss (reconstructing the foreground image from keypoint features + coordinates) is designed, including L2 pixel loss + VGG perceptual loss.
    - **Design Motivation**: Dense pixel features introduce background noise and are computationally expensive, while random sampling lacks consistency; the keypoint method achieves better performance (mAE 14.2° vs 15.52°) with lower computation (50.05G vs 55.26G MACs). The image reconstruction constraint forces the keypoints to disperse into semantically rich regions.

2. **Structure-Aware Correspondence Estimation Module (SA-CE)**

    - **Function**: Extract structure-aware features from keypoint features for 3D correspondence regression.
    - **Mechanism**: First, use self-attention with ROPE positional encoding to aggregate keypoint structural information within the same image $\tilde{\mathbf{F}}_{kpt,q} = \text{MHSA}(\mathbf{F}_{kpt,q} \circledast R(\mathbf{X}_{kpt,q}))$; then use cross-attention to aggregate keypoint features from the reference image. With the structure-aware features, an MLP is used to regress the pseudo depth $d_{i,q}$ to lift 2D keypoints into the 3D space of the query coordinate system $\mathbf{x}^{(\mathcal{Q})}_{i,q}$, and another MLP is used to regress their 3D coordinates in the reference coordinate system $\mathbf{x}^{(\mathcal{R})}_{i,q}$ along with the confidence $c_i$.
    - **Design Motivation**: Self-attention + ROPE allows each keypoint to perceive its relative position (intra-image structure) within the object structure, and cross-attention allows keypoints to understand the complementary structural information between the two views (inter-image structure). The combination of both enables the network to "mentally assemble" how the two parts fit together.

3. **wSVD-based End-to-End Pose Estimation**

    - **Function**: Solve for the optimal rotation matrix from 3D-3D correspondences.
    - **Mechanism**: Calculate the weighted covariance matrix $\mathbf{H} = \sum_i c_i \mathbf{x}^{(\mathcal{Q})}_{i,q}(\mathbf{x}^{(\mathcal{R})}_{i,q})^\top$, perform SVD decomposition $\mathbf{H}=\mathbf{U}\Sigma\mathbf{V}^\top$, and obtain the optimal rotation $\Delta\mathbf{R}=\mathbf{V}\mathbf{U}^\top$. The confidence $c_i$ is predicted by the network, automatically down-weighting unreliable correspondences.
    - **Design Motivation**: wSVD provides a differentiable closed-form solution from correspondences to rotation, ensuring end-to-end training. The confidence weights allow the model to automatically identify which correspondences are trustworthy.

### Loss & Training

The total loss is $\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{pts} + \lambda_2 \mathcal{L}_{rec} + \lambda_3 \mathcal{L}_{rot} + \lambda_4 \mathcal{L}_{mask}$. Among them, $\mathcal{L}_{pts}$ uses symmetric stop-gradient to constrain the accuracy of 3D coordinate prediction (including confidence weighting); $\mathcal{L}_{rot}$ uses L1 loss to align 6D rotation representations; query and reference are used symmetrically during training to increase data efficiency.

## Key Experimental Results

### Main Results

| Method | Type | CO3D mAE↓ | CO3D Acc@15°↑ | Objaverse mAE↓ | LineMOD mAE↓ |
|------|------|-----------|-------------|---------------|-------------|
| SuperGlue | 2D | 67.2° | 37.7% | 102.4° | 64.8° |
| LoFTR | 2D | 77.5° | 33.1% | 134.1° | 84.5° |
| DVMNet | 3D | 19.9° | 62.3% | 20.2° | 36.8° |
| **Ours** | 3D | **14.2°** | **80.2%** | **15.3°** | **27.2°** |

### Ablation Study

| Configuration | mAE↓ | Acc@30°↑ | Acc@15°↑ | MACs(G) |
|------|------|---------|---------|---------|
| Dense features | 15.52 | 92.65 | 78.20 | 55.26 |
| Random points | 20.15 | 88.34 | 68.99 | 49.59 |
| **Keypoint (ours)** | **14.2** | **93.6** | **80.2** | 50.05 |
| w/o Self-Attn | 17.79 | 90.22 | 72.48 | - |
| w/o Cross-Attn | 18.53 | 89.38 | 70.99 | - |

### Key Findings

- Compared to DVMNet, the mAE on CO3D is reduced by about 6° (19.9→14.2), and Acc@15° is improved by nearly 18 percentage points, proving that directly regressing correspondences is superior to dense 3D matching.
- Removing self-attention (+3.6° mAE) and cross-attention (+4.3° mAE) both result in significant performance drops, indicating that both intra-image and inter-image structural information are crucial.
- The keypoint method achieves better performance with 10% less computation than dense features, showing that structured sparse representations are more efficient than dense ones.

## Highlights & Insights

- **The idea of "bypassing matching and directly regressing" is clever**: It avoids explicit matching in the feature space (which is vulnerable to unobserved regions) and instead allows the network to directly learn the mapping from structural features to 3D coordinates. This approach can be transferred to fields like point cloud registration.
- **Image reconstruction as keypoint dispersion constraint**: Driving keypoints to cover semantically rich areas using reconstruction objectives is more natural and effective than directly adding dispersion regularization.
- **Symmetric training strategy**: Swapping query and reference is equivalent to doubling the data, which is simple yet effective.

## Limitations & Future Work

- Currently, it only estimates rotation (3DoF); the translation part is assumed to be resolvable by 2D detection, which may not hold true in practical applications.
- The number of keypoints $N_{kpt}$ needs to be preset, which may require adaptive adjustment for objects of varying complexity.
- Testing is only conducted at the object level; its effectiveness at the scene level (large-scale, multi-object) has not been verified.
- The image reconstruction module increases training overhead; although it can be discarded during inference, training requires extra computation.

## Related Work & Insights

- **vs DVMNet**: DVMNet performs dense 3D voxel matching, while ours uses sparse keypoints for direct regression, achieving better performance with less computation.
- **vs LoFTR/SuperGlue**: These 2D methods fail under large viewpoint changes, whereas ours naturally supports large viewpoints through 3D correspondence regression.
- The concept of 3D keypoints + structure awareness can be extended to tasks such as multi-view reconstruction and 6DoF object pose estimation.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of structure-aware keypoints and direct correspondence regression is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets + detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear method description, very intuitive human assembly analogy.
- Value: ⭐⭐⭐⭐ Provides a new SOTA solution in the relative pose estimation field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Co-op: Correspondence-based Novel Object Pose Estimation](co-op_correspondence-based_novel_object_pose_estimation.md)
- [\[CVPR 2025\] Probabilistic Prompt Distribution Learning for Animal Pose Estimation](probabilistic_prompt_distribution_learning_for_animal_pose_estimation.md)
- [\[CVPR 2026\] COG: Confidence-aware Optimal Geometric Correspondence for Unsupervised Single-reference Novel Object Pose Estimation](../../CVPR2026/human_understanding/cog_confidence-aware_optimal_geometric_correspondence_for_unsupervised_single-re.md)
- [\[ICLR 2026\] GenCape: Structure-Inductive Generative Modeling for Category-Agnostic Pose Estimation](../../ICLR2026/human_understanding/gencape_structure-inductive_generative_modeling_for_category-agnostic_pose_estim.md)
- [\[ECCV 2024\] GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence](../../ECCV2024/human_understanding/gs-pose_category-level_object_pose_estimation_via_geometric_and_semantic_corresp.md)

</div>

<!-- RELATED:END -->
