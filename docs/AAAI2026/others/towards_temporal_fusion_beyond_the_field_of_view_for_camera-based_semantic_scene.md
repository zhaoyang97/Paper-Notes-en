---
title: >-
  [Paper Note] Towards Temporal Fusion Beyond the Field of View for Camera-based Semantic Scene Completion
description: >-
  [AAAI 2026][Semantic Scene Completion] This paper proposes C3DFusion, a module that explicitly aligns point features from historical and current frames in 3D space…
tags:
  - "AAAI 2026"
  - "Semantic Scene Completion"
  - "Temporal Fusion"
  - "Out-of-View Completion"
  - "3D Perception"
  - "Voxel Features"
date: 2026-05-08
content_hash: 5b40539846c953f0
---

# Towards Temporal Fusion Beyond the Field of View for Camera-based Semantic Scene Completion

**Conference**: AAAI 2026
**arXiv**: [2511.12498](https://arxiv.org/abs/2511.12498)
**Code**: None (Project Page available)
**Area**: Other
**Keywords**: Semantic Scene Completion, Temporal Fusion, Out-of-View Completion, 3D Perception, Voxel Features

## TL;DR

This paper proposes C3DFusion, a module that explicitly aligns point features from historical and current frames in 3D space, and is the first to systematically address temporal completion of out-of-frame (out-of-view) regions in camera-based SSC. The method achieves state-of-the-art performance on SemanticKITTI and SSCBench-KITTI-360.

## Background & Motivation

**Background**: 3D Semantic Scene Completion (SSC) is a core perception task in autonomous driving, requiring simultaneous reconstruction of 3D geometry and semantic label prediction for each voxel. Compared to expensive LiDAR-based approaches, camera-based methods have advanced rapidly in recent years, progressively closing the performance gap. Most recent methods have begun leveraging temporal information to enhance current-frame features.

**Limitations of Prior Work**: Existing temporal fusion methods (e.g., HTCL-S, Hi-SOP, CVT-Occ) primarily focus on enhancing regions within the current camera field of view, while neglecting blind spots outside the field of view—areas typically located near the sides of the ego vehicle that are critical for safe driving. Historical frames contain rich contextual information about these regions, yet existing methods fail to exploit it effectively.

**Key Challenge**: The potential of temporal fusion lies in providing spatial information beyond the current field of view; however, most methods perform fusion in 2D feature space or BEV space, making it difficult to naturally transfer out-of-view information from historical frames into the current frame's 3D space. Meanwhile, direct fusion in 3D space faces geometric inconsistency caused by depth estimation errors.

**Goal**: (1) How to effectively leverage historical frame information to complete out-of-view regions in the current frame; (2) How to mitigate noise from depth estimation errors during temporal fusion in 3D space.

**Key Insight**: The authors observe that depth estimation errors are larger for distant points in historical frames, and that current-frame points may be "diluted" by historical-frame points during temporal aggregation. Two complementary techniques are thus proposed to address these issues.

**Core Idea**: Directly align historical and current frame features in 3D point feature space, achieving high-quality out-of-view temporal fusion through depth-aware feature attenuation and current-frame point cloud densification.

## Method

### Overall Architecture

The model follows a standard camera-based SSC architecture comprising three stages: viewing transformation, voxel processing, and semantic prediction. C3DFusion primarily operates in the viewing transformation stage. The input consists of a sequence of $n$ consecutive RGB frames; 2D features are extracted via an image encoder, depth maps are obtained using a pretrained depth estimator, and 2D features are back-projected into 3D space. Historical frame points are then aligned to the current frame coordinate system using camera poses, and voxelized to produce the 3D feature volume.

### Key Designs

1. **Temporal 3D Point Feature Alignment**:

    - **Function**: Maps 2D image features from multiple frames into a unified 3D space.
    - **Mechanism**: For each frame, 2D features $\mathbf{F}_i$ and depth map $\mathbf{D}_i$ are extracted. Features are aligned to the depth map resolution via a linear layer and bilinear interpolation, then back-projected to obtain the 3D point cloud $\mathbf{P}_i$ and corresponding point features $\mathbf{F}_i^{pt}$. Using known camera poses, 3D points from historical frames are transformed from their respective coordinate systems into the current frame coordinate system.
    - **Design Motivation**: Unlike the LSS strategy that generates dense frustum feature volumes, this paper directly maps point features. The authors hypothesize that when extended to multiple frames, sparse densification and long-tail distribution of LSS introduce geometric noise that degrades semantic prediction accuracy—a hypothesis validated experimentally.

2. **Historical Context Blurring**:

    - **Function**: Suppresses the influence of point features from distant points in historical frames where depth estimation is inaccurate.
    - **Mechanism**: The historical frame depth map is min-max normalized and inverted to obtain a weight $w_i = 1 - \text{MinMax}(\mathbf{D}_i) \in [0,1]$. Points at greater depth (farther distance) receive smaller weights; feature magnitudes are attenuated via element-wise multiplication $\tilde{\mathbf{F}}_i^{pt} = w_i \odot \mathbf{F}_i^{pt}$.
    - **Design Motivation**: As the ego vehicle moves forward, points from historical frames that persist in the current coordinate system tend to originate from distant regions in the original viewpoint. Since depth estimation errors are proportional to depth, feature intensities are scaled inversely with depth to mitigate geometric inconsistency.

3. **Current-Centric Feature Densification**:

    - **Function**: Increases the density of the current-frame point cloud so that it remains dominant during temporal aggregation.
    - **Mechanism**: The current frame's point features and depth map are bilinearly upsampled (by a factor of 2 by default) from $(H, W)$ to $(\tilde{H}, \tilde{W}) = (2H, 2W)$, and then back-projected to obtain the densified current point cloud $\tilde{\mathbf{P}}_t$. The current frame thus contributes $4HW$ points, while each historical frame still contributes $HW$ points.
    - **Design Motivation**: In overlapping regions within the field of view, multiple frames contribute a large number of points, and the fixed $HW$ points from the current frame may be diluted during aggregation. Densification increases the current frame's volumetric contribution in these regions, emphasizing temporally more relevant current information.

### Loss & Training

A combination of four losses is employed: cross-entropy loss $\mathcal{L}_{ce}$, geometric affinity loss $\mathcal{L}_{scal}^{geo}$, semantic affinity loss $\mathcal{L}_{scal}^{sem}$, and depth loss $\mathcal{L}_d$, with weights 1, 1, 1, and 0.001, respectively. After voxel aggregation, MAE-style cross-attention and self-attention refinement are applied, followed by 3D convolution, trilinear interpolation, and softmax to produce per-voxel predictions.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| SemanticKITTI | IoU | 47.62 | 46.21 (CF-SSC) | +1.41 |
| SemanticKITTI | mIoU | 18.98 | 18.18 (L2COcc-D) | +0.80 |
| SSCBench-KITTI-360 | IoU | 49.28 | - | - |
| SSCBench-KITTI-360 | mIoU | 21.74 | - | - |

### Ablation Study

| Configuration | IoU | mIoU | Notes |
|------|-----|------|------|
| Baseline (single-frame CGFormer) | 44.41 | 16.63 | No temporal fusion |
| + C3DFusion (full) | 47.62 | 18.98 | IoU +3.21, mIoU +2.35 |
| LSS-style temporal fusion | lower | lower | Validates that direct point feature mapping outperforms LSS strategy |

### Key Findings

- C3DFusion achieves particularly significant improvements in out-of-view (OOV) regions, validating its effectiveness for out-of-view completion.
- Consistent performance gains are observed when integrated into other baseline models (e.g., VoxFormer, Symphonies), demonstrating strong generalizability.
- Both Historical Context Blurring and Current-Centric Feature Densification contribute independently to overall performance.

## Highlights & Insights

- This is the first work to explicitly identify and systematically address out-of-view completion in camera-based SSC, filling a gap in this direction.
- The two core techniques (blurring + densification) are intuitively simple yet effective, and are easy to integrate into existing architectures.
- The paper provides a well-motivated alternative to LSS-style fusion by analyzing why such approaches underperform in multi-frame settings.

## Limitations & Future Work

- The quality of the depth estimator directly constrains the accuracy of 3D points; better depth estimation could yield larger gains.
- The current-frame densification strategy is relatively simple (fixed 2× upsampling); adaptive densification is worth exploring.
- Historical context blurring uses a simple depth-inverse weighting scheme; more refined uncertainty modeling may yield further improvements.
- Validation is conducted only in monocular/binocular settings; multi-camera configurations remain unexplored.

## Related Work & Insights

- **vs. HTCL-S / Hi-SOP**: These methods perform temporal alignment in 2D feature space and cannot effectively recover out-of-view regions; C3DFusion operates in 3D space, naturally supporting out-of-view information fusion.
- **vs. CVT-Occ**: CVT-Occ constructs cross-frame cost volumes to enhance volumetric representations but still focuses on in-view regions; C3DFusion enables completion of regions visible in historical frames but occluded in the current frame through unified point cloud integration.
- **vs. CGFormer / ScanSSC**: These are single-frame methods; C3DFusion can directly enhance them as a plug-and-play module.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The first to systematically address out-of-view completion—a novel direction—though the underlying technical components (back-projection, point feature mapping) are relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated on two datasets with ablation studies and cross-model generalization experiments; however, separate quantitative evaluation of out-of-view regions is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clearly articulated, method description is complete, and figures are of high quality.
- **Value**: ⭐⭐⭐⭐ Practically significant for safety-critical perception in autonomous driving; the plug-and-play nature of the module enhances its practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] AdaSFormer: Adaptive Serialized Transformers for Monocular Semantic Scene Completion from Indoor Environments](../../CVPR2026/others/adasformer_adaptive_serialized_transformers_for_monocular_semantic_scene_complet.md)
- [\[AAAI 2026\] Expressive Temporal Specifications for Reward Monitoring](expressive_temporal_specifications_for_reward_monitoring.md)
- [\[AAAI 2026\] Beyond World Models: Rethinking Understanding in AI Models](beyond_world_models_rethinking_understanding_in_ai_models.md)
- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](hierarchical_semantic_alignment_for_image_clustering.md)
- [\[AAAI 2026\] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)

</div>

<!-- RELATED:END -->
