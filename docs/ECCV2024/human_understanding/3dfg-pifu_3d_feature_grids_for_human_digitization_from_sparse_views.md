---
title: >-
  [Paper Note] 3DFG-PIFu: 3D Feature Grids for Human Digitization from Sparse Views
description: >-
  [ECCV 2024][Human Understanding][Human Reconstruction] This paper proposes 3DFG-PIFu, which globally fuses multi-view features across the entire pipeline by introducing 3D Feature Grids, replacing the traditional point-wise local fusion approach. Combined with an iterative grid refinement mechanism and SDF-based SMPL-X features, it significantly outperforms state-of-the-art sparse-view human digitization methods.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Human Reconstruction"
  - "Multi-view Reconstruction"
  - "Pixel-aligned Implicit Functions"
  - "3D Feature Grids"
  - "SMPL-X"
date: 2026-05-08
content_hash: 803db40edc719a93
---

# 3DFG-PIFu: 3D Feature Grids for Human Digitization from Sparse Views

**Conference**: ECCV 2024  
**Code**: [https://github.com/kcyt/3DFG-PIFu](https://github.com/kcyt/3DFG-PIFu)  
**Area**: Other  
**Keywords**: Human Reconstruction, Multi-view Reconstruction, Pixel-aligned Implicit Functions, 3D Feature Grids, SMPL-X

## TL;DR

This paper proposes 3DFG-PIFu, which globally fuses multi-view features across the entire pipeline by introducing 3D Feature Grids, replacing the traditional point-wise local fusion approach. Combined with an iterative grid refinement mechanism and SDF-based SMPL-X features, it significantly outperforms state-of-the-art sparse-view human digitization methods.

## Background & Motivation

**Background**: Reconstructing 3D clothed human models from sparse views is an important problem in computer vision. Current mainstream methods are based on Pixel-aligned Implicit Functions (e.g., Multi-view PIFu, DeepMultiCap, DoubleField, and SeSDF), which predict occupancy fields or SDF values by aligning 2D image features with 3D query points.

**Limitations of Prior Work**: Given $V$ multi-view images, existing methods only fuse features from different views in a point-wise and localized manner at the very end of the pipeline. In other words, the $V$ images are processed independently throughout most of the pipeline, only being narrowly combined in the final step. This largely defeats the purpose of using multi-view information, essentially treating a multi-view task as a single-view task.

**Key Challenge**: Multi-view information is complementary and correlated in space, but the fusion strategies of existing methods are too localized and delayed, failing to fully leverage the global context provided by different perspectives. This delayed fusion limits the reconstruction quality to the representational capacity of single-view features, making it difficult to guarantee cross-view consistency and completeness.

**Goal**: (1) How to globally fuse multi-view features throughout the entire pipeline rather than only at the end? (2) How to utilize existing reconstruction results for iterative refinement? (3) How to more effectively integrate the prior information of parametric human models (SMPL-X) into pixel-aligned implicit models?

**Key Insight**: The authors observe that globally fusing multi-view features at any processing stage is possible if they can be projected into a shared 3D feature space. A 3D voxel grid is naturally suited as such a shared space, as it can accept feature projections from any perspective.

**Core Idea**: Utilizing 3D feature grids as a vehicle for global fusion of multi-view features, allowing multi-view information to be fully exploited at every stage of the pipeline.

## Method

### Overall Architecture

The overall pipeline of 3DFG-PIFu is as follows: the input consists of $V$ sparse-view human images and corresponding camera parameters, and the output is the reconstructed 3D human mesh. The workflow includes three core stages: (1) extracting 2D features from each image via an image encoder and unprojecting them into a 3D feature grid to achieve global fusion; (2) extracting information for each 3D query point from both the 3D feature grid and 2D pixel-aligned features to predict its occupancy/SDF value using an MLP; (3) utilizing an iterative refinement mechanism to re-project the initially reconstructed mesh onto each view to acquire updated features, iteratively improving reconstruction quality.

### Key Designs

1. **3D Feature Grids**:

    - Function: Globally fusing multi-view features throughout the entire pipeline.
    - Mechanism: First, multi-scale 2D feature maps are extracted from each input image using an image encoder (e.g., ResNet or HRNet). Then, leveraging known camera parameters, features from each feature map are unprojected into a unified 3D voxel grid. Specifically, for each voxel in the 3D grid, its projected position in each view is computed, and features are bilinearly interpolated from the corresponding feature map. Features from all views are then aggregated (e.g., via mean pooling or attention weighting). The resulting 3D feature grid encodes global spatial information from all views. To query any 3D point, trilinear interpolation within this grid is performed to obtain global multi-view features.
    - Design Motivation: Traditional methods only fuse multi-view features point-by-point in the final stage, resulting in a very narrow scope of information fusion. 3D feature grids bring fusion forward to the feature space construction stage, and the fusion scope covers the entire 3D space rather than individual query points, fundamentally solving the problem of underutilized multi-view information.

2. **Iterative Refinement**:

    - Function: Repeatedly refining the human mesh using prior reconstruction results to progressively improve quality.
    - Mechanism: After the first forward pass generates the initial human mesh, this mesh is rendered (or projected) from different viewpoints to obtain silhouette/depth information. This is concatenated with the original images as new inputs, which are passed again through the encoder and the 3D feature grid to extract richer features and generate a refined mesh. This process can be iterated multiple times, leveraging the previous reconstruction results to provide better geometric clues each time.
    - Design Motivation: A single forward pass often fails to accurately reconstruct all details, especially in occluded and geometrically complex regions. Iterative refinement is similar to a coarse-to-fine strategy, allowing the model to correct errors and complete details based on the known coarse geometry.

3. **SDF-based SMPL-X Features**:

    - Function: Effectively integrating the prior information of the parametric human model SMPL-X into the implicit reconstruction model.
    - Mechanism: Unlike previous works that directly use the surface distance of SMPL-X meshes or voxel-based inside/outside indicators, this work computes the signed distance field (SDF) values of each 3D query point to the SMPL-X mesh surface, serving as an additional input feature for the occupancy prediction MLP. SDF values are continuous and carry explicit geometric meanings: positive values denote points outside the body, negative values denote points inside, and zero represents the body surface. This representation is smoother and more informative than simple binary indicators.
    - Design Motivation: SMPL-X provides a strong human shape prior, but a suitable method is needed to combine it with free-form implicit representations. SDF representation is a natural choice because it is inherently a continuous implicit representation, mathematically aligned with the implicit prediction framework of the PIFu model, allowing seamless integration.

### Loss & Training

Training adopts standard point cloud sampling strategies, randomly sampling 3D query points near the human surface and in space, and uses Binary Cross-Entropy Loss (BCE Loss) or $L_1$ loss to supervise occupancy/SDF predictions. During training, the ground truth SMPL-X fitting results are used to compute SDF features. In the iterative refinement phase, end-to-end or progressive training strategies are adopted to gradually unlock more iteration rounds.

## Key Experimental Results

### Main Results

We evaluate our method on standard human reconstruction datasets such as THuman2.0 and RenderPeople, using Chamfer Distance (CD), Normal Consistency (NC), and Point-to-Surface (P2S) distance as metrics.

| Dataset | Metric | 3DFG-PIFu | Multi-view PIFu | SeSDF | Gain |
|--------|------|-----------|-----------------|-------|------|
| THuman2.0 | Chamfer Distance ↓ | **Best** | Baseline | Second Best | Significant Gain |
| THuman2.0 | Normal Consistency ↑ | **Best** | Baseline | Second Best | Significant Gain |
| RenderPeople | P2S ↓ | **Best** | Baseline | Second Best | Clear Improvement |

3DFG-PIFu significantly outperforms existing SOTA methods such as Multi-view PIFu, DeepMultiCap, DoubleField, and SeSDF across all metrics.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Baseline (w/o 3D Feature Grids) | Higher CD | Degenerates to traditional point-wise fusion |
| +3D Feature Grids | Significantly lowered CD | Global feature fusion brings clear improvement |
| +Iterative Refinement | Further lowered CD | Iterative refinement effectively corrects reconstruction errors |
| +SDF-based SMPL-X | Lowest CD | Human body prior features further improve geometric details |
| Different grid resolutions | Improves then degrades | An optimal 3D grid resolution exists |
| Different iteration rounds | 2 rounds yields best results | Diminishing returns with too many iterations |

### Key Findings

- The 3D feature grid is the single factor providing the largest gain, showing that global multi-view fusion is critical.
- Iterative refinement is particularly effective in occluded and geometrically complex regions (such as hands and clothing folds).
- SDF-based SMPL-X features provide consistent improvements across all body parts, especially in limb regions.
- Under extremely sparse-view conditions (e.g., 2-3 views), 3DFG-PIFu exhibits a more pronounced advantage compared to existing methods.

## Highlights & Insights

- **Generality of Global Fusion**: The concept of 3D feature grids is not only applicable to human reconstruction but can also be generalized to any multi-view 3D reconstruction task. The core insight is to shift from "delayed fusion" to "early global fusion".
- **Unified SDF Representation**: Using SDF as a bridge between parametric model priors and implicit reconstruction is an elegant design choice.
- **Lightweight Implementation of Iterative Refinement**: Instead of training an additional refinement network, it is achieved by reusing the main network, keeping the added computational overhead manageable.

## Limitations & Future Work

- The resolution of the 3D feature grid is limited by GPU memory; while high-resolution grids capture more details, they incur high computational costs.
- Iterative refinement increases inference time, which could be a bottleneck in real-time application scenarios.
- It depends on the accuracy of SMPL-X fitting; when fitting fails, SDF features can introduce noise.
- There is still room for improvement in reconstructing extreme poses and loose clothing.
- The possibility of extending the method to dynamic sequence reconstruction remains unexplored.

## Related Work & Insights

- **vs Multi-view PIFu**: Multi-view PIFu only fuses features point-by-point in the final stage, whereas 3DFG-PIFu globally fuses features across the entire pipeline using 3D feature grids, fundamentally changing the feature fusion paradigm.
- **vs SeSDF**: SeSDF introduces semantic-aware SDF prediction but still employs a localized fusion strategy. 3DFG-PIFu demonstrates that global fusion is more critical than semantic enhancement.
- **vs DeepMultiCap**: DeepMultiCap uses an attention mechanism to aggregate view features, but its scope is still restricted to the vicinity of query points. The 3D grids of 3DFG-PIFu naturally cover the global space.

## Rating

- Novelty: ⭐⭐⭐⭐ The global fusion concept of 3D feature grids is simple and effective, and SDF-based SMPL-X features represent a meaningful innovation.
- Experimental Thoroughness: ⭐⭐⭐ Ablation studies and comparative experiments are conducted on standard datasets, but in-the-wild evaluation is lacking.
- Writing Quality: ⭐⭐⭐ The problem definition is clear, and the method description is complete.
- Value: ⭐⭐⭐⭐ The global fusion concept has strong generality and inspiration, contributing to the field of multi-view reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] 3DGazeNet: Generalizing 3D Gaze Estimation with Weak-Supervision from Synthetic Views](3dgazenet_generalizing_3d_gaze_estimation_with_weak-supervision_from_synthetic_v.md)
- [\[ECCV 2024\] 3DSA: Multi-view 3D Human Pose Estimation With 3D Space Attention Mechanisms](3dsa_multi-view_3d_human_pose_estimation_with_3d_space_attention_mechanisms.md)
- [\[ECCV 2024\] TELA: Text to Layer-wise 3D Clothed Human Generation](tela_text_to_layer-wise_3d_clothed_human_generation.md)
- [\[ECCV 2024\] Wear-Any-Way: Manipulable Virtual Try-on via Sparse Correspondence Alignment](wear-any-way_manipulable_virtual_try-on_via_sparse_correspondence_alignment.md)
- [\[ECCV 2024\] ADen: Adaptive Density Representations for Sparse-view Camera Pose Estimation](aden_adaptive_density_representations_for_sparseview_camera.md)

</div>

<!-- RELATED:END -->
