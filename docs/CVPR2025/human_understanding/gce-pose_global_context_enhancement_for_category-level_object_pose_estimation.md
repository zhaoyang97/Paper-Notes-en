---
title: >-
  [Paper Note] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation
description: >-
  [CVPR 2025][Human Understanding][Category-level pose estimation] GCE-Pose proposes a "completion-then-aggregation" strategy, which reconstructs partial observations into complete geometric-semantic 3D representations via a Semantic Shape Reconstruction (SSR) module, and then injects global information into local keypoint features through a Global Context Enhancement (GCE) feature fusion module. This approach significantly outperforms existing methods on HouseCat6D and NOCS-RE…
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Category-level pose estimation"
  - "global context"
  - "semantic shape reconstruction"
  - "deep linear shape model"
  - "feature fusion"
date: 2026-05-08
content_hash: cb9a34b7355b9b23
---

# GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation

**Conference**: CVPR 2025  
**arXiv**: [2502.04293](https://arxiv.org/abs/2502.04293)  
**Code**: [https://colin-de.github.io/GCE-Pose/](https://colin-de.github.io/GCE-Pose/)  
**Area**: 3D Vision / Human Understanding  
**Keywords**: Category-level pose estimation, global context, semantic shape reconstruction, deep linear shape model, feature fusion

## TL;DR

GCE-Pose proposes a "completion-then-aggregation" strategy, which reconstructs partial observations into complete geometric-semantic 3D representations via a Semantic Shape Reconstruction (SSR) module, and then injects global information into local keypoint features through a Global Context Enhancement (GCE) feature fusion module. This approach significantly outperforms existing methods on HouseCat6D and NOCS-REAL275.

## Background & Motivation

**Background**: Category-level object pose estimation aims to predict the 6D pose and size for unseen instances of a known category. Traditional pipelines estimate Normalized Object Coordinate Space (NOCS) coordinates and solve the pose using the Umeyama algorithm. Recent approaches like AG-Pose achieve promising results using keypoint detection and DINOv2 features.

**Limitations of Prior Work**: Since category-level estimation is model-free (requiring only a single RGB-D input during inference without CAD models), partial observations captured by depth sensors inevitably suffer from occlusions and incomplete geometry. Existing methods extract features solely from partial observations, lacking global context understanding of the complete object.

**Key Challenge**: Partial observations provide limited information—for instance, a severely occluded mug might only expose its handle, making it highly challenging to establish accurate correspondences with the NOCS space. Although some methods introduce geometric shape priors, they overlook the semantic context (such as the functional semantic features of different object parts).

**Goal**: How to obtain a complete global representation of the object that contains both geometric and semantic information under partial RGB-D observations, and effectively fuse it into local features.

**Key Insight**: It is observed that geometric priors and semantic priors are complementary—geometric priors provide complete shape information, while semantic priors (from DINOv2) offer a functional understanding of different parts. Combining them provides robust global context under occlusions and shape variations.

**Core Idea**: Reconstructing partial point clouds into complete 3D shapes annotated with semantic features using a deep linear semantic shape model, and subsequently incorporating global context into local keypoint features via cross-attention.

## Method

### Overall Architecture

GCE-Pose comprises four modules: (A) extracting point cloud and image features from RGB-D inputs, and selecting discriminative keypoints and their features via a keypoint detector; (B) the SSR module completing the partial point cloud using category shape priors and attaching category-level semantic prototypes; (C) the GCE module blending global semantic shape information into keypoint features using cross-attention; (D) using the fused features to predict keypoint NOCS coordinates for pose and size regression.

### Key Designs

1. **Semantic Shape Reconstruction (SSR) Module**:

    - **Function**: Reconstructs partial point cloud observations into complete 3D shapes with semantic information.
    - **Mechanism**: Utilizing a Deep Linear Semantic Shape Model (DLSSM), each category $k$ maintains a geometric prototype $c^k$, deformation basis vectors $v^k$, and a semantic prototype $c^k_{sem}$ (built from DINOv2 features). Given a partial point cloud, a deformation network $\mathcal{D}^k$ and a scaling network $\mathcal{S}^k$ predict instance-specific shape parameters $a^k$ and scaling $s^k$. The final reconstructed shape is formulated as $\mathbf{U}_k = (s^k \odot (c^k + \sum_i a_i^k v_i^k), c^k_{sem})$. Training consists of two stages: first learning the shape space with complete point clouds, and then learning the mapping from partial to complete structures.
    - **Design Motivation**: The linear shape model offers compact and efficient parameterization, enabling semantic features to naturally propagate alongside geometric deformation without requiring extra semantic alignment steps.

2. **Semantic Prototype Construction**:

    - **Function**: Constructs 3D prototypes with part-level semantic information for each category.
    - **Mechanism**: To build the semantic prototype, RGB images of each object instance are rendered from multiple virtual camera views. 2D semantic feature maps are extracted using DINOv2 and back-projected into 3D space via depth maps and camera intrinsics to form dense semantic point clouds. These features are then aggregated onto the shape prototype points via KNN. The category-level semantic prototype is obtained by averaging across instances: $c^k_{sem} = \frac{1}{N}\sum_{i=1}^N \mathbf{F}_{instance}(P_i^k)$.
    - **Design Motivation**: Zero-shot semantic features from DINOv2 are robust to shape variations, and category-level averaging captures common semantic architectures.

3. **Global Context Enhancement (GCE) Feature Fusion**:

    - **Function**: Effectively fuses global semantic shape information into local keypoint features.
    - **Mechanism**: Position tokens are generated using learnable positional encodings for local keypoints (camera space coordinates) and global reconstructions (normalized object space coordinates), respectively. After concatenating with their respective features, they are projected into a unified space via a shared MLP. Cross-attention is then executed with keypoint features as query and global features as key/value: $\mathbf{F}_{gce} = \mathbf{F}_{kpt} + \text{CrossAttn}(\mathbf{F}''_{kpt}, \mathbf{F}''_{global})$.
    - **Design Motivation**: Partial observations (camera space) and global reconstructions (object space) suffer from inter-coordinate domain gaps, which are bridged by the learnable positional encodings.

### Loss & Training

The total loss consists of five components: $\mathcal{L}_{all} = \lambda_1 \mathcal{L}_{ocd} + \lambda_2 \mathcal{L}_{div} + \lambda_3 \mathcal{L}_{rec} + \lambda_4 \mathcal{L}_{nocs} + \lambda_5 \mathcal{L}_{pose}$. Here, $\mathcal{L}_{ocd}$ and $\mathcal{L}_{div}$ regularize keypoint quality (surface-aligned and well-distributed), $\mathcal{L}_{rec}$ is the Chamfer distance for shape reconstruction, $\mathcal{L}_{nocs}$ denotes the Smooth L1 loss for NOCS coordinates, and $\mathcal{L}_{pose}$ includes direct regression losses for rotation, translation, and size. The model is trained on a single RTX 4090 GPU for 150 epochs.

## Key Experimental Results

### Main Results

| Dataset | Method | 5°2cm | 10°5cm | IoU75 |
|--------|------|-------|--------|-------|
| HouseCat6D | AG-Pose (DINO) | 21.3 | 54.3 | 53.0 |
| HouseCat6D | SecondPose | 11.0 | 35.7 | - |
| HouseCat6D | **GCE-Pose** | **24.8** | **58.4** | **60.6** |
| NOCS-REAL275 | AG-Pose (DINO) | 57.0 | 84.7 | 80.1 |
| NOCS-REAL275 | GS-Pose | - | 60.1 | 63.2 |
| NOCS-REAL275 | **GCE-Pose** | **57.0** | **86.3** | **79.8** |

### Ablation Study

| Configuration | 5°2cm | 5°5cm | Description |
|------|-------|-------|------|
| Baseline (AG-Pose DINO) | 21.3 | 22.1 | No global prior |
| + Instance Geo. | 22.2 | 23.7 | Geometry reconstruction only +4% |
| + Categorical Sem. | 22.7 | 24.3 | Semantic prototype only +7% |
| + Mean shape + Sem. | 23.4 | 24.2 | Mean shape + Semantics +10% |
| **Full GCE-Pose** | **24.8** | **25.7** | Full model **+16%** |

### Key Findings

- The contributions of geometric and semantic priors are complementary: using geometry alone boosts performance by 4%, semantics alone by 7%, while combining them yields a 16% enhancement, displaying clear synergistic effects.
- Instance-level geometric reconstruction is more critical than using a mean shape, as it captures intra-category shape variations.
- Utilizing key tokens (instead of value tokens) in the GCE fusion module produces better results, as the keys initialize the attention weights between local and global contexts.
- The SSR module exhibits strong robustness to noise and occluded scenarios, with semantic features remaining consistent despite shape variations.

## Highlights & Insights

- **"Completion-then-aggregation" strategy**: Instead of directly extracting features from partial observations, GCE-Pose first reconstructs the complete shape and then extracts global features. This strategy enables the model to comprehend the complete structure of objects even under severe occlusions. This paradigm can be generalized to any 3D understanding tasks processing partial observations.
- **Natural propagation of semantics with geometry**: The elegance of the linear shape model lies in its fixed semantic prototype, requiring only geometric deformation to derive instance-specific semantic shape representations without additional semantic alignment or matching steps.
- **Position encoding bridging domain gaps**: Partial observations and global reconstructions reside in different coordinate networks. Employing learnable position encodings to automatically learn the coordinate transformations is both simple and effective.

## Limitations & Future Work

- The method relies on ground truth segmentation masks as input; in practical applications, segmentation quality affects performance.
- The shape models (deformation and scaling networks) must be trained separately for each category, which limits cross-category generalization.
- Linear shape models may have limited capability in representing categories with large deformations (e.g., clothing, flexible objects).
- The pipeline has not been validated on transparent or reflective objects, which often suffer from poor depth data quality.

## Related Work & Insights

- **vs AG-Pose**: While AG-Pose extracts keypoint features solely from partial observations, GCE-Pose introduces auxiliary global context, achieving a 16% improvement on strict metrics.
- **vs GS-Pose**: GS-Pose selects a single reference instance for semantic matching, making it sensitive to intra-class shape variations and noises in partial observations. GCE-Pose is more robust by learning cross-instance deformation priors.
- **vs SPD/RePoNet**: Early methods only utilized geometric shape priors, whereas GCE-Pose introduces semantic priors, achieving global context enhancement through a combination of geometry and semantics.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of semantic shape reconstruction and GCE fusion is a logical and highly effective contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluations across two datasets with extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological descriptions and intuitive diagrams.
- Value: ⭐⭐⭐⭐ Actively advances the field of category-level pose estimation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation](../../ICCV2025/human_understanding/cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)
- [\[ECCV 2024\] GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence](../../ECCV2024/human_understanding/gs-pose_category-level_object_pose_estimation_via_geometric_and_semantic_corresp.md)
- [\[ECCV 2024\] U-COPE: Taking a Further Step to Universal 9D Category-Level Object Pose Estimation](../../ECCV2024/human_understanding/u-cope_taking_a_further_step_to_universal_9d_category-level_object_pose_estimati.md)
- [\[CVPR 2025\] Co-op: Correspondence-based Novel Object Pose Estimation](co-op_correspondence-based_novel_object_pose_estimation.md)
- [\[CVPR 2025\] Recurrent Feature Mining and Keypoint Mixup Padding for Category-Agnostic Pose Estimation](recurrent_feature_mining_and_keypoint_mixup_padding_for_category-agnostic_pose_e.md)

</div>

<!-- RELATED:END -->
