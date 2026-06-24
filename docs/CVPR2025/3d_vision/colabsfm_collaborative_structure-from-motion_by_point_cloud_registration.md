---
title: >-
  [Paper Note] ColabSfM: Collaborative Structure-from-Motion by Point Cloud Registration
description: >-
  [CVPR 2025][3D Vision][Collaborative SfM] The ColabSfM paradigm is proposed, which merges distributed SfM reconstruction results via 3D point cloud registration (rather than visual descriptor matching). In addition, a dedicated SfM registration dataset generation pipeline and an improved registration model, RefineRoITr, are designed.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Collaborative SfM"
  - "Point Cloud Registration"
  - "Map Merger"
  - "Privacy Protection"
  - "3D Reconstruction"
date: 2026-05-08
content_hash: 4b0257676fef4693
---

# ColabSfM: Collaborative Structure-from-Motion by Point Cloud Registration

**Conference**: CVPR 2025  
**arXiv**: [2503.17093](https://arxiv.org/abs/2503.17093)  
**Code**: [https://github.com/EricssonResearch/ColabSfM](https://github.com/EricssonResearch/ColabSfM)  
**Area**: 3D Vision  
**Keywords**: Collaborative SfM, Point Cloud Registration, Map Merger, Privacy Protection, 3D Reconstruction

## TL;DR

The ColabSfM paradigm is proposed, which merges distributed SfM reconstruction results via 3D point cloud registration (rather than visual descriptor matching). In addition, a dedicated SfM registration dataset generation pipeline and an improved registration model, RefineRoITr, are designed.

## Background & Motivation

Robots and XR devices require precise construction and localization within environmental maps. Different manufacturers use varied SfM pipelines and feature extractors, leading to incompatible visual descriptors that prevent map sharing. Traditional map fusion methods rely on visual descriptor matching (3D-3D or 2D-3D correspondences), which present three key issues:

1. **Poor Interoperability**: Descriptors from different pipelines are incompatible (e.g., SIFT vs. SuperPoint).
2. **Privacy Risks**: Exposing visual descriptors makes them vulnerable to inversion attacks that reconstruct original images.
3. **Poor Scalability**: Storing descriptors increases the map size by 2-3 orders of magnitude.

This paper poses a fundamental question: **Can SfM maps be merged using only 3D geometric information?** The answer is to formulate this as a point cloud registration problem. However, existing registration methods are trained on RGB-D/LiDAR data and perform poorly when directly applied to SfM point clouds due to significant domain discrepancies. Moreover, training datasets for SfM point cloud registration are lacking.

## Method

### Overall Architecture

ColabSfM comprises three core contributions: (1) a scalable SfM registration dataset generation pipeline that synthesizes partial reconstruction pairs via synthetic camera trajectories from existing SfM datasets (MegaDepth); (2) the RefineRoITr model, which introduces a neural refinement stage on top of RoITr; (3) a complete definition and evaluation of the SfM point cloud registration task. The inputs are two SfM-reconstructed point clouds $\mathcal{P}, \mathcal{Q}$, and the output is the similarity transformation $(s, R, t)$ between them.

### Key Designs

1. **Synthetic SfM Registration Dataset Generation Pipeline**:
    - **Function**: Generates multiple overlapping partial reconstruction pairs from a single large-scale SfM reconstruction.
    - **Mechanism**: Two strategies are designed: (a) Random point sampling: sampling 3D point sets and their corresponding visible images from a scene; (b) Synthetic trajectories: randomly selecting a starting image and sequentially selecting subsequent images (75-300 frames) using a nearest neighbor strategy weighted by geodesic rotation distance and Euclidean distance to simulate realistic camera motion. Retriangulation is performed using the fixed camera poses from the original reconstruction to ensure accurate ground-truth correspondences.
    - **Design Motivation**: Point clouds from random image sets and video sequences exhibit distribution discrepancies (e.g., differing keypoint densities and occlusion patterns). Synthetic trajectories bridge this gap. Jointly training with both strategies enhances generalization.

2. **RefineRoITr (Neural Refinement Registration Model)**:
    - **Function**: Adds a local refinement Transformer on top of RoITr's coarse registration to improve matching accuracy.
    - **Mechanism**: RoITr achieves rotation invariance through Point Pair Features (PPF) encoding. RefineRoITr introduces a refinement Transformer $r_\theta$ over the local neighborhood features $\hat{\mathbf{G}}^X, \hat{\mathbf{G}}^Y$ extracted by the decoder. This Transformer consists of four alternating self-attention and cross-attention layers (similar to LightGlue but operating as local attention) that output enhanced neighborhood features, which are then fed into the Sinkhorn algorithm to solve for optimal transport. The computational overhead is increased by only approximately 3%.
    - **Design Motivation**: The refinement in RoITr relies solely on Sinkhorn optimization over shallow features, which is insufficiently precise for large-scale SfM point cloud scenes. Cross-point cloud interaction of local features provides richer matching cues.

3. **Normalization and Normal Vector Processing**:
    - **Function**: Addresses the scale ambiguity and inconsistent normal vector orientations in SfM reconstructions.
    - **Mechanism**: During Sim(3) training, both point clouds are normalized using their respective maximum singular values; during SE(3) training, both are normalized simultaneously using the singular values of the source point cloud. Normal orientations are aligned by randomly choosing a camera center that observes the point, instead of simply orienting towards the coordinate origin.
    - **Design Motivation**: SfM reconstructions lack metric scale, and multi-sensor acquisition leads to inconsistent normal vectors when simply aligned toward the origin; leveraging camera visibility information from 3D tracks yields consistent normal orientations.

### Loss & Training

- Superpoint matching loss $\mathcal{L}_s$: based on overlap-aware circle loss.
- Point matching loss $\mathcal{L}_p$: negative log-likelihood of ground truth correspondences after the Sinkhorn algorithm.
- Total loss $\mathcal{L} = \mathcal{L}_s + \mathcal{L}_p$.
- Training requires a point cloud overlap $> 30\%$, totaling approximately 22,000 pairs (20,000 for training and 2,000 for testing).

## Key Experimental Results

### Main Results

| Method | Dataset | IR(SE3) | FMR(SE3) | RR(SE3) | IR(Sim3) | RR(Sim3) |
|------|--------|---------|----------|---------|----------|----------|
| RoITr (3DMatch) | MegaDepth | 3.0 | 12.6 | 0.0 | 1.6 | 0.8 |
| OverlapPredator (3DMatch) | MegaDepth | 6.1 | 35.5 | 10.0 | 3.6 | 2.1 |
| RefineRoITr (3DM+Mega) | MegaDepth | 48.7 | 95.1 | 67.7 | 44.6 | 44.3 |
| **RefineRoITr (Mega only)** | **MegaDepth** | **51.0** | **96.5** | **70.2** | **44.6** | **42.7** |

### Ablation Study (Cambridge Landmarks, SE(3))

| Scene | RoITr (3DM+Mega) IR | RefineRoITr (Mega) IR | Description |
|------|---------------------|----------------------|------|
| Great Court | 52.1 | 70.9 | Significant improvement in large-scale scenes |
| Kings College | 39.6 | 57.6 | +18.0 |
| Old Hospital | 21.9 | 31.5 | +9.6 |
| Shop Facade | 28.0 | 41.6 | +13.6 |
| St Mary's Church | 64.5 | 81.8 | +17.3 |

### Key Findings

- Registration models trained on 3DMatch perform extremely poorly when directly applied to SfM point clouds (with RR near 0%), validating the severity of domain discrepancy.
- After training on the proposed dataset, RR increases from 0% to 70.2% (SE3), proving the critical role of the dataset.
- RefineRoITr consistently improves over RoITr across all scenes, while only adding 3% computational overhead.
- The synthetic trajectory strategy is crucial for generalization to video-sequence scenes (such as Cambridge Landmarks).

## Highlights & Insights

- **Paradigm Innovation**: Decouples SfM map fusion from visual descriptor matching to pure 3D point cloud registration, simultaneously addressing interoperability, privacy, and scalability issues.
- Highly reusable **dataset generation pipeline**: it generates training pairs solely from existing SfM datasets and is compatible with any local features.
- The design of RefineRoITr's local refinement Transformer is elegant, bringing significant performance improvements at only a 3% computational overhead.

## Limitations & Future Work

- It relies on sufficient point cloud overlap ($> 30\%$), and low-overlap scenes remain challenging.
- The Registration Recall (RR) in the Sim(3) setting is still insufficient (44.3%), where scale estimation remains a bottleneck.
- It relies solely on geometric information without leveraging auxiliary cues such as color or texture, which may lead to failures in geometrically similar but semantically distinct scenes.
- The quality of normal estimation relies on Open3D's 33-neighborhood estimation, which may be inaccurate in sparse point clouds.

## Related Work & Insights

- It fundamentally differs from traditional descriptor-based collaborative mapping methods (e.g., Dusmanu et al.) in that it completely eliminates the need for visual descriptors.
- The rotation invariance of Point Pair Features (PPF) is crucial for SfM registration, as the reference frame of SfM is arbitrary.
- The core idea of the dataset generation pipeline (extracting partial reconstruction pairs using synthetic trajectories from existing reconstructions) can be generalized to data generation for other 3D tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ Highly innovative paradigm, shifting SfM map fusion to a point cloud registration problem.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across MegaDepth/Cambridge/7-Scenes datasets under both SE(3) and Sim(3) settings.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, detailed pipeline descriptions, and comprehensive algorithmic pseudocode.
- Value: ⭐⭐⭐⭐ Highly practical for distributed multi-device collaborative mapping, addressing real-world demands for privacy and interoperability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TurboReg: TurboClique for Robust and Efficient Point Cloud Registration](../../ICCV2025/3d_vision/turboreg_turboclique_for_robust_and_efficient_point_cloud_registration.md)
- [\[CVPR 2025\] P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning](p-slcr_unsupervised_point_cloud_semantic_segmentation_via_prototypes_structure_l.md)
- [\[CVPR 2025\] Light3R-SfM: Towards Feed-forward Structure-from-Motion](light3r-sfm_towards_feed-forward_structure-from-motion.md)
- [\[CVPR 2026\] SuP: Sub-cloud Driven Point Cloud Registration](../../CVPR2026/3d_vision/sup_sub-cloud_driven_point_cloud_registration.md)
- [\[ICCV 2025\] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes](../../ICCV2025/3d_vision/bufferx_towards_zeroshot_point_cloud_registration_in_diverse.md)

</div>

<!-- RELATED:END -->
