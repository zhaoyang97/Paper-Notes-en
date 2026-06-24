---
title: >-
  [Paper Note] Monocular Occupancy Prediction for Scalable Indoor Scenes
description: >-
  [ECCV 2024][Autonomous Driving][3D Occupancy Prediction] Proposes the ISO (Indoor Scene Occupancy) method, which achieves monocular 3D occupancy prediction for indoor scenes using pre-trained depth models and a D-FLoSP (Dual-feature Line-of-Sight Projection) module, and constructs the Occ-ScanNet benchmark dataset that is 40 times larger than NYUv2.
tags:
  - "ECCV 2024"
  - "Autonomous Driving"
  - "3D Occupancy Prediction"
  - "Monocular Vision"
  - "Indoor Scenes"
  - "Depth Estimation"
  - "Semantic Scene Completion"
date: 2026-05-08
content_hash: f970a703b2ba2b16
---

# Monocular Occupancy Prediction for Scalable Indoor Scenes

**Conference**: ECCV 2024  
**arXiv**: [2407.11730](https://arxiv.org/abs/2407.11730)  
**Code**: [Yes](https://github.com/hongxiaoy/ISO.git)  
**Area**: Autonomous Driving / Indoor Scene Understanding  
**Keywords**: 3D Occupancy Prediction, Monocular Vision, Indoor Scenes, Depth Estimation, Semantic Scene Completion

## TL;DR

Proposes the ISO (Indoor Scene Occupancy) method, which achieves monocular 3D occupancy prediction for indoor scenes using pre-trained depth models and a D-FLoSP (Dual-feature Line-of-Sight Projection) module, and constructs the Occ-ScanNet benchmark dataset that is 40 times larger than NYUv2.

## Background & Motivation

**Key Differences Between Indoor and Outdoor**: While 3D occupancy prediction has achieved significant progress in outdoor autonomous driving scenarios (e.g., TPVFormer, FB-OCC), research on indoor scenes remains insufficient. Indoor scenes present two core challenges distinct from outdoor scenarios:

**Complexity of Scene Scale**: Indoor room sizes vary drastically—from spacious living rooms to cramped kitchens—unlike outdoor driving scenes which typically focus on a fixed 3D space. This places higher demands on the accuracy of depth prediction.

**Object Complexity**: Indoor objects have higher density, more diverse categories, and larger scale variations. While outdoor vehicles and pedestrians have relatively consistent sizes within their categories and are spaced far apart, indoor furniture exhibits significant size variations and is closely packed.

**Limitations of Prior Work**:

- **MonoScene**: Proposes FLoSP (Feature Line-of-Sight Projection) to lift 2D features to 3D. However, projecting shared 2D features along rays leads to depth ambiguity, where voxels at different depths receive the same feature, making it impossible to distinguish between foreground and background objects.
- **NDC-Scene**: Designs a depth-adaptive dual decoder, achieving some performance improvements, but still limited.
- **Dataset Bottleneck**: Existing indoor scene works mainly use the NYUv2 dataset, which contains only 795/654 training/testing samples. The scale is too small, limiting research on model scalability.

**Core Idea of ISO**:
- Leverage powerful pre-trained depth models (Depth-Anything) to obtain accurate metric depth.
- Design the D-FLoSP module to jointly project depth information and 2D features into 3D space, resolving the depth ambiguity of FLoSP.
- Introduce multi-scale feature fusion to adapt to size variations of indoor objects.
- Construct the large-scale Occ-ScanNet benchmark to advance research in this field.

## Method

### Overall Architecture

The overall architecture of ISO is based on 2D UNet + 3D UNet, with the core design focusing on the 2D$\rightarrow$3D feature transformation:

1. **2D Feature Extraction**: EfficientNet-B7 encoder + 2D UNet decoder yielding multi-scale 2D feature maps.
2. **Depth Branch**: Pre-trained Depth-Anything estimates coarse metric depth $\rightarrow$ DepthNet refines it into depth distributions.
3. **D-FLoSP Module**: Jointly projects depth distributions and 2D features into 3D voxel space.
4. **3D UNet**: Processes 3D voxel features and outputs occupancy predictions.

### Key Designs

1. **Depth Branch**

   **Function**: Estimates accurate pixel-wise depth information from a single RGB image, providing crucial spatial localization for 2D$\rightarrow$3D feature transformation.

   **Mechanism**: Split into two steps: coarse estimation and refinement:

   Coarse estimation directly uses the frozen Depth-Anything model:
    $\mathbf{D}^{\text{metric}} = \mathbf{N}_{\text{depth}}(\mathbf{I}^{\text{rgb}}) \in \mathbb{R}^{1 \times H \times W}$

   The refinement stage concatenates the coarse depth with image features to generate depth distributions via DepthNet:
    $\mathbf{D}^{\text{dist}}_{\text{s=1}} = \mathbf{F}_{\text{depth}}(\text{Concat}(\mathbf{D}^{\text{metric}}, \mathbf{X}^{\text{2d}}_{\text{s=1}})) \in \mathbb{R}^{N_{\text{bins}} \times H \times W}$

   The depth distribution is supervised using BCE loss, with GT depth converted to one-hot vectors.

   **Design Motivation**: Learning depth from scratch is highly challenging, whereas pre-trained depth models (e.g., Depth-Anything) can already estimate metric-level depth. However, pre-trained depth is not accurate enough (insufficient for high mIoU). Therefore, a learnable refinement strategy is designed—fusing depth priors with image features to learn residual depth corrections.

2. **D-FLoSP (Dual-feature Line-of-Sight Projection) Module**

   **Function**: Jointly lifts 2D image features and depth distributions to 3D voxel space, resolving the depth ambiguity of the original FLoSP.

   **Mechanism**: For each 3D voxel center $x^c$, two parallel projection paths are executed:

   Path 1 — Depth Distribution Projection: Projects 2D depth distributions to 3D, obtaining depth probabilities for each voxel:
    $\mathbf{D}^{\text{3d}}_{\text{s}=k} = \mathbf{\Phi}^{\text{3d}}_{\rho(x^c)}(\mathbf{D}^{\text{dist}}_{\text{s}=k})$

   Path 2 — Feature Projection: Similar to the original FLoSP, projects 2D features to 3D:
    $\mathbf{X}^{\text{3d}}_{\text{s}=k} = \mathbf{\Phi}^{\text{2d}}_{\rho(x^c)}(\mathbf{X}^{\text{2d}}_{\text{s}=k})$

   Finally, element-wise multiplication weighting and summation are applied across multiple scales:
    $\mathbf{X}^{\text{3d}} = \sum_{s \in \{1,2,4,8\}} \mathbf{X}^{\text{3d}}_{\text{s}=k} \odot \mathbf{D}^{\text{3d}}_{\text{s}=k}$

   **Design Motivation**: The original FLoSP only considers ray projection, where all voxels on the same ray receive identical 2D features, leading to severe depth ambiguity. D-FLoSP introduces depth distributions as weights, so that voxels close to the true depth receive higher weights, while weights of those far away approach zero, effectively resolving the ambiguity. "Dual" refers to the two parallel projection paths of features and depth.

3. **Multi-Scale Depth Fusion**

   **Function**: Leverages multi-scale (1/1, 1/2, 1/4, 1/8) depth distributions to weight 3D features at corresponding scales.

   **Mechanism**: Downsamples the depth distribution from scale 1 to obtain depth distributions at other scales. 3D features at each scale are weighted by the corresponding depth distribution and then summed.

   **Design Motivation**: Different objects in indoor scenes vary greatly in size (e.g., ceilings vs. small objects), and features at different scales have varying importance for different object sizes. Multi-scale depth weighting allows the model to independently determine depth weights at each scale, rather than applying a "one-size-fits-all" unified depth distribution to all features.

### Loss & Training

- **Depth Loss** $\mathcal{L}_{\text{depth}}$: BCE loss, supervising depth distribution prediction.
- **Occupancy Loss**: Follows the loss design of MonoScene.
- The 2D encoder uses a pre-trained EfficientNet-B7, and the Depth-Anything model is frozen.
- NYUv2 is trained for 30 epochs, with learning rates of 5e-6 (DepthNet) / 1e-4 (others), decayed by 10x at epoch 20.

## Key Experimental Results

### Main Results

**NYUv2 Dataset Performance** (voxel resolution 60×36×60):

| Method | Input | IoU | ceil | floor | wall | window | chair | bed | sofa | table | tvs | furn | obj | mIoU |
|------|------|-----|------|-------|------|--------|-------|-----|------|-------|-----|------|-----|------|
| LMSCNet | occ | 33.93 | 4.49 | 88.41 | 4.63 | 0.25 | 3.94 | 32.03 | 15.44 | 6.57 | 0.02 | 14.51 | 4.39 | 15.88 |
| MonoScene | rgb | 42.51 | 8.89 | 93.50 | 12.06 | 12.57 | 13.72 | 48.19 | 36.11 | 15.13 | 15.22 | 27.96 | 12.94 | 26.94 |
| NDC-Scene | rgb | 44.17 | 12.02 | 93.51 | 13.11 | 13.77 | 15.83 | 49.57 | 39.87 | 17.17 | 24.57 | 31.00 | 14.96 | 29.03 |
| **ISO** | **rgb** | **47.11** | **14.21** | 93.47 | **15.89** | **15.14** | **18.35** | **50.01** | **40.82** | **18.25** | **25.90** | **34.08** | **17.67** | **31.25** |

ISO achieves an mIoU of 31.25%, which is a **+2.22%** improvement over NDC-Scene (29.03%), while IoU increases from 44.17% to 47.11%.

**Occ-ScanNet Dataset Performance** (voxel resolution 60×60×36):

| Method | IoU | mIoU |
|------|-----|------|
| MonoScene* | 41.60 | 24.62 |
| **ISO** | **42.16** | **28.71** |

On the large-scale Occ-ScanNet, ISO's mIoU improves from 24.62% to 28.71% (**+4.09%**).

### Ablation Study

**Ablation on Depth Fusion Strategy** (NYUv2 + Occ-ScanNet-mini):

| Configuration | NYUv2 IoU | NYUv2 mIoU | Occ-mini IoU | Occ-mini mIoU |
|------|-----------|------------|--------------|---------------|
| baseline | 42.27 | 27.13 | 50.94 | 38.95 |
| + BEV-depth | 42.67 | 27.14 | 51.58 | 38.48 |
| + voxel-depth | **47.11** | **31.25** | 51.03 | **39.08** |

Voxel-depth fusion performs significantly better than BEV-depth (height information is more critical than BEV projections in indoor scenes).

**Ablation on Depth Models** (NYUv2):

| Depth Method | learned | multi-scale | IoU | mIoU |
|---------|---------|-------------|-----|------|
| GT Depth | ✓ | — | 53.98 | 34.47 |
| ZoeDepth | ✓ | ✓ | 45.24 | 29.40 |
| DepthAnything | ✓ | ✓ | 46.94 | 31.02 |
| DepthAnything | ✓ | ✗ | **47.11** | **31.11** | **31.25** |

DepthAnything outperforms ZoeDepth, and the learnable depth refinement strategy further improves performance.

### Key Findings

- **Voxel Depth Outperforms BEV Depth**: BEV methods perform well on ceilings/floors but poorly on furniture because indoor height info is more important than horizontal projections.
- **Multi-scale Depth is Not Always Better**: Under the DepthAnything + learned setting, single-scale (47.11/31.25) slightly outperforms multi-scale (46.94/31.02), likely because multi-scale introduces redundancy.
- **Significant Data Scale Effects**: Training on 10% of samples yields IoU=21.79/mIoU=9.57, whereas 100% yields IoU=42.16/mIoU=28.71, demonstrating that indoor occupancy prediction urgently requires large-scale data.
- **Upper Bound of GT Depth**: Using GT depth reaches mIoU=34.47, indicating substantial room for improvement in depth accuracy.

## Highlights & Insights

1. **Outstanding Contribution of Occ-ScanNet Benchmark**: With 45,755/19,764 samples, it is 40 times larger than NYUv2. Combined with a complete annotation pipeline and quality control, it is poised to become a standard benchmark for indoor occupancy prediction.
2. **Elegant D-FLoSP Design**: Fusing depth distribution to weight FLoSP features resolves the severe depth ambiguity with minimal architectural modifications.
3. **Strategy for Reusing Pre-trained Depth Models**: The combination of freezing Depth-Anything + learnable refinement balances pre-trained knowledge with task-specific learning.

## Limitations & Future Work

- **Difficulties in Semantic Learning**: Class imbalance causes prediction accuracy for small objects (e.g., TVs, windows) to remain low.
- **Only 11 Semantic Categories**: Fails to cover the category diversity of real-world scenes.
- **Monocular Limitations**: Despite utilizing depth priors, monocular input still has fundamental limitations in heavily occluded scenes.
- Combining with foundation segmentation models (such as SAM) to provide semantic priors for occupancy prediction can be explored.
- The dataset can be extended to more scenes and more semantic categories.

## Related Work & Insights

- **MonoScene** pioneered vision-only monocular SSC, featuring FLoSP as its core innovation. ISO improves upon this by introducing depth info.
- The 2D$\rightarrow$3D lifting concept from **BEVDepth/LSS** is adapted to indoor scenes, but requires a transition from BEV to voxel depth.
- Indoor occupancy prediction may benefit from depth prior transfer from 3D reconstruction (NeRF/3DGS).

## Rating

- **Novelty**: ⭐⭐⭐ — D-FLoSP is a reasonable improvement over FLoSP but with limited increment; the contribution of the Occ-ScanNet benchmark is more outstanding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Validated on two datasets with detailed ablations (depth fusion, depth models, multi-scale, data scale), offering a thorough analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear analysis of problems, with a convincing discussion of differences between indoor and outdoor scenarios.
- **Value**: ⭐⭐⭐⭐ — High long-term value of the Occ-ScanNet dataset, and the method provides a strong baseline for indoor 3D perception.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Monocular Open Vocabulary Occupancy Prediction for Indoor Scenes (LegoOcc)](../../CVPR2026/autonomous_driving/monocular_open_vocabulary_occupancy_prediction_for_indoor_scenes.md)
- [\[ECCV 2024\] OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](occgen_generative_multi-modal_3d_occupancy_prediction_for_autonomous_driving.md)
- [\[ECCV 2024\] UniTraj: A Unified Framework for Scalable Vehicle Trajectory Prediction](unitraj_a_unified_framework_for_scalable_vehicle_trajectory_prediction.md)
- [\[ECCV 2024\] Fully Sparse 3D Occupancy Prediction](fully_sparse_3d_occupancy_prediction.md)
- [\[ECCV 2024\] GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction](gaussianformer_scene_as_gaussians_for_vision-based_3d_semantic_occupancy_predict.md)

</div>

<!-- RELATED:END -->
