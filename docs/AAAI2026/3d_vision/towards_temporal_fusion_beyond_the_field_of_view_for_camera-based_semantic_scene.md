---
title: >-
  [Paper Note] Towards Temporal Fusion Beyond the Field of View for Camera-based Semantic Scene Completion
description: >-
  [AAAI 2026][3D Vision][Semantic Scene Completion] The C3DFusion module is proposed to explicitly align point features of historical and current frames in 3D space, systematically addressing the temporal completion problem of out-of-frame regions in camera-based SSC for the first time, achieving SOTA on SemanticKITTI and SSCBench-KITTI-360.
tags:
  - "AAAI 2026"
  - "3D Vision"
  - "Semantic Scene Completion"
  - "Temporal Fusion"
  - "Out-of-Frame Completion"
  - "3D Perception"
  - "Voxel Features"
date: 2026-05-08
content_hash: b320398f92876f86
---

# Towards Temporal Fusion Beyond the Field of View for Camera-based Semantic Scene Completion

**Conference**: AAAI 2026  
**arXiv**: [2511.12498](https://arxiv.org/abs/2511.12498)  
**Code**: None (Project Page available)  
**Area**: Others  
**Keywords**: Semantic Scene Completion, Temporal Fusion, Out-of-Frame Completion, 3D Perception, Voxel Features

## TL;DR

The C3DFusion module is proposed to explicitly align point features of historical and current frames in 3D space, systematically addressing the temporal completion problem of out-of-frame regions in camera-based SSC for the first time, achieving SOTA on SemanticKITTI and SSCBench-KITTI-360.

## Background & Motivation

**Background**: 3D Semantic Scene Completion (SSC) is a core perception task in autonomous driving, requiring simultaneous 3D geometry reconstruction and voxel-wise semantic label prediction. Compared to expensive LiDAR solutions, camera-based methods have developed rapidly in recent years, narrowing the performance gap. Most recent methods have started utilizing temporal information to enhance current frame features.

**Limitations of Prior Work**: Existing temporal fusion methods (such as HTCL-S, Hi-SOP, CVT-Occ, etc.) mainly focus on enhancing regions within the current camera field of view, but ignore blind spots outside the field of view—which are usually located near both sides of the ego-vehicle and are critical for safe driving. Historical frames actually contain rich contextual information of these regions, which prior methods fail to exploit effectively.

**Key Challenge**: The potential of temporal fusion lies in providing spatial information beyond the current field of view. However, most methods perform fusion in 2D feature space or BEV space, failing to naturally propagate out-of-frame information from historical frames into the 3D space of the current frame. At the same time, direct fusion in 3D space suffers from geometric inconsistency caused by depth estimation errors.

**Goal**: (1) How to effectively utilize historical frame information to complete out-of-frame regions of the current frame; (2) How to mitigate noise caused by depth estimation errors during temporal fusion in 3D space.

**Key Insight**: The authors observe that depth estimation errors are larger for distant points in historical frames, and current frame points tend to be "diluted" by historical frame points during temporal aggregation. Therefore, two complementary techniques are proposed to address these issues.

**Core Idea**: Directly align historical and current frames in the 3D point feature space, achieving high-quality out-of-frame temporal fusion through depth-aware feature decay and current frame point cloud densification.

## Method

### Overall Architecture

The model follows a standard camera-based SSC architecture consisting of three stages: viewing transformation, voxel processing, and semantic prediction. C3DFusion mainly operates in the viewing transformation stage. The input is a continuous sequence of $n$ RGB images, where 2D features are extracted by an image encoder, depth maps are obtained using a pre-trained depth estimator, and then 2D features are mapped to 3D space via back-projection. Camera poses are utilized to align historical frame points to the current frame coordinate system, eventually forming a 3D feature volume through voxelization.

### Key Designs

1. **Temporal 3D Point Feature Alignment**:

    - **Function**: Map 2D image features from multiple frames into a unified 3D space.
    - **Mechanism**: For each image frame, 2D features $\mathbf{F}_i$ and depth map $\mathbf{D}_i$ are extracted. Features are aligned to the depth map resolution via a linear layer and bilinear interpolation, and then back-projected to obtain the 3D point cloud $\mathbf{P}_i$ and its corresponding point features $\mathbf{F}_i^{pt}$. Using known camera poses, the 3D points of historical frames are transformed from their respective coordinate systems to the current frame coordinate system.
    - **Design Motivation**: Unlike LSS strategies that generate dense frustum feature volumes, this paper chooses to map point features directly. The authors assume that when extending to multi-frame scenarios, the sparse densification and long-tailed distribution features of LSS introduce geometric noise, degrading semantic prediction accuracy—an assumption validated by experiments.

2. **Historical Context Blurring**:

    - **Function**: Suppress the influence of distant point features with inaccurate depth estimation in historical frames.
    - **Mechanism**: A min-max normalization is applied to the depth map of historical frames and then inverted to obtain weights $w_i = 1 - \text{MinMax}(\mathbf{D}_i)$ in the range of $[0, 1]$. Points with larger depth (further away) receive smaller weights. The magnitude of distant point features is decayed through element-wise multiplication $\tilde{\mathbf{F}}_i^{pt} = w_i \odot \mathbf{F}_i^{pt}$.
    - **Design Motivation**: As the ego-vehicle moves forward, the historical points retained in the current coordinate system often originate from distant regions in their raw perspective. Since depth estimation error is proportional to depth, scaling down feature intensity inversely with depth mitigates geometric inconsistency.

3. **Current-Centric Feature Densification**:

    - **Function**: Increase the density of the current frame point cloud to ensure it maintains dominance during temporal aggregation.
    - **Mechanism**: Bilinear upsampling (default $2\times$) is applied to the point features and depth map of the current frame, interpolating from $(H, W)$ to $(\tilde{H}, \tilde{W}) = (2H, 2W)$, followed by back-projection to yield the densified current point cloud $\tilde{\mathbf{P}}_t$. Consequently, the current frame contributes $4HW$ points, while each historical frame still contributes $HW$ points.
    - **Design Motivation**: In overlapping field-of-view regions, multiple frames contribute a large number of points, which might dilute the current frame's fixed point count $HW$ during aggregation. Densification increases the volume contribution of the current frame in these regions, emphasizing temporally more relevant current information.

### Loss & Training

A combination of four losses is employed: cross-entropy loss $\mathcal{L}_{ce}$, geometric affinity loss $\mathcal{L}_{scal}^{geo}$, semantic affinity loss $\mathcal{L}_{scal}^{sem}$, and depth loss $\mathcal{L}_d$, with weights of 1, 1, 1, and 0.001, respectively. After voxel aggregation, an MAE-styled cross-attention and self-attention refinement is adopted, followed by 3D convolution, trilinear interpolation, and softmax to output voxel-wise predictions.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| SemanticKITTI | IoU | 47.62 | 46.21 (CF-SSC) | +1.41 |
| SemanticKITTI | mIoU | 18.98 | 18.18 (L2COcc-D) | +0.80 |
| SSCBench-KITTI-360 | IoU | 49.28 | - | - |
| SSCBench-KITTI-360 | mIoU | 21.74 | - | - |

### Ablation Study

| Configuration | IoU | mIoU | Description |
|------|-----|------|------|
| Baseline (Single-frame CGFormer) | 44.41 | 16.63 | No temporal fusion |
| + C3DFusion (Full) | 47.62 | 18.98 | IoU +3.21, mIoU +2.35 |
| LSS-style temporal fusion | Lower | Lower | Validated that direct point feature mapping outperforms LSS strategy |

### Key Findings

- The improvement of C3DFusion is particularly significant in out-of-view (OOV) regions, validating its effectiveness for out-of-frame completion.
- Integrating C3DFusion into other baseline models (e.g., VoxFormer, Symphonies) consistently yields performance boosts, demonstrating strong generalization capability.
- Both Historical Context Blurring and Current-Centric Feature Densification contribute independently to the performance gains.

## Highlights & Insights

- For the first time, the out-of-frame completion problem in camera-based SSC is explicitly proposed and systematically addressed, filling a gap in this direction.
- The two core techniques (blurring + densification) are intuitively simple yet highly effective and easy to integrate into existing architectures.
- Taking the perspective of "why LSS-style fusion performs poorly in multi-frame scenarios," the paper offers a reasonable alternative solution.

## Limitations & Future Work

- The quality of the depth estimator directly constrains the accuracy of 3D points; better depth estimation might lead to even larger improvements.
- The current frame densification strategy is relatively simple (fixed $2\times$), and adaptive densification could be explored.
- Historical context blurring uses simple inverse-depth weights; more fine-grained uncertainty modeling could yield better results.
- The method is only validated in monocular/stereo setups, without extending to multi-camera configurations.

## Related Work & Insights

- **vs HTCL-S / Hi-SOP**: These methods perform temporal alignment in 2D feature space, failing to effectively recover out-of-view regions; C3DFusion operates in 3D space, naturally supporting out-of-view information fusion.
- **vs CVT-Occ**: CVT-Occ enhances volumetric representation by constructing cross-frame cost volumes but still focuses within the field of view; C3DFusion achieves completion in regions visible in historical frames but invisible in the current frame through point cloud unification.
- **vs CGFormer / ScanSSC**: These are single-frame methods, and C3DFusion can directly enhance them as a plug-and-play module.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to systematically address out-of-frame completion, defining a novel direction, though the technical components (back-projection, point feature mapping) are relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated on two datasets with ablation studies and cross-model generalization, but lacks a separate quantitative evaluation specifically for the out-of-frame regions.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem motivation, comprehensive methodology description, and high-quality figures/tables.
- **Value**: ⭐⭐⭐⭐ Practical significance for safety-critical perception in autonomous driving; the plug-and-play nature of the module increases its empirical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TGSFormer: Scalable Temporal Gaussian Splatting for Embodied Semantic Scene Completion](../../CVPR2026/3d_vision/tgsformer_scalable_temporal_gaussian_splatting_for_embodied_semantic_scene_compl.md)
- [\[CVPR 2026\] Learning Spatial-Temporal Consistency for 3D Semantic Scene Completion](../../CVPR2026/3d_vision/learning_spatial-temporal_consistency_for_3d_semantic_scene_completion.md)
- [\[AAAI 2026\] SplatSSC: Decoupled Depth-Guided Gaussian Splatting for Semantic Scene Completion](splatssc_decoupled_depth-guided_gaussian_splatting_for_semantic_scene_completion.md)
- [\[CVPR 2026\] Multi-modal Frequency Decomposition Network for Semantic Scene Completion](../../CVPR2026/3d_vision/multi-modal_frequency_decomposition_network_for_semantic_scene_completion.md)
- [\[ICCV 2025\] Monocular Semantic Scene Completion via Masked Recurrent Networks](../../ICCV2025/3d_vision/monocular_semantic_scene_completion_via_masked_recurrent_networks.md)

</div>

<!-- RELATED:END -->
