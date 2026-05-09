---
title: >-
  [Paper Note] C3PO: Cross-View Cross-Modality Correspondence by Pointmap Prediction
description: >-
  [NeurIPS 2025][Remote Sensing][cross-view correspondence] This paper introduces the C3 dataset comprising 90K ground photo–floor plan pairs (597 scenes, 153M pixel-level correspondences, and 85K camera poses), exposes the limitations of existing correspondence models under cross-view cross-modality settings (e.g., ground photos vs. floor plans), and demonstrates that training on this dataset reduces the RMSE of the best-performing baseline by 34%.
tags:
  - NeurIPS 2025
  - Remote Sensing
  - cross-view correspondence
  - cross-modality
  - floor plan
  - pointmap
  - DUSt3R
  - structure-from-motion
date: 2026-05-08
content_hash: 16d9bd6ca71be237
---

# C3PO: Cross-View Cross-Modality Correspondence by Pointmap Prediction

**Conference**: NeurIPS 2025
**arXiv**: [2511.18559](https://arxiv.org/abs/2511.18559)
**Code**: To be confirmed
**Area**: Remote Sensing / Cross-View Cross-Modality Matching
**Keywords**: cross-view correspondence, cross-modality, floor plan, pointmap, DUSt3R, structure-from-motion

## TL;DR
This paper introduces the C3 dataset comprising 90K ground photo–floor plan pairs (597 scenes, 153M pixel-level correspondences, and 85K camera poses), exposes the limitations of existing correspondence models under cross-view cross-modality settings (e.g., ground photos vs. floor plans), and demonstrates that training on this dataset reduces the RMSE of the best-performing baseline by 34%.

## Background & Motivation

**State of the Field**: Geometric models such as DUSt3R have achieved significant progress in estimating 3D geometry from image pairs via dense per-pixel pointmap prediction. However, these models are inherently dependent on the viewpoint and modality distributions present in their training data.

**Limitations of Prior Work**: When input image pairs span drastically different viewpoints (e.g., aerial vs. ground-level) or distinct modalities (e.g., real photographs vs. abstract floor plans), existing geometric models degrade severely. Such cross-view cross-modality correspondence is nonetheless critical for applications including architectural navigation, indoor localization, and augmented reality.

**Root Cause**: Existing cross-view datasets are insufficient—VIGOR lacks modality diversity (all real images), while WAFFLE lacks pixel-level correspondence annotations. No large-scale, high-quality dataset exists for training and evaluating models on ground photo–floor plan correspondence.

**Paper Goals**: (1) Construct a large-scale, high-quality cross-view cross-modality correspondence dataset; (2) benchmark existing methods on this task; (3) demonstrate that training on the proposed data yields substantial performance gains.

**Starting Point**: SfM (Structure-from-Motion) is applied to internet photo collections to reconstruct 3D scenes, which are then manually registered to internet-sourced floor plans, enabling automatic derivation of pixel-level correspondences between photographs and floor plans.

**Core Idea**: A pipeline of SfM reconstruction → manual registration → automatic correspondence derivation is employed to construct C3, the first large-scale dataset providing pixel-level photo–floor plan correspondences.

## Method

### Overall Architecture
The input consists of internet photo collections and corresponding floor plans; the output is pixel-level correspondences and camera poses. The pipeline proceeds in three stages: (1) reconstruct the 3D structure of each scene from internet photos via SfM; (2) manually register the 3D reconstruction to the floor plan, establishing a 3D-to-2D mapping; (3) automatically derive pixel-level correspondences between each photograph and the floor plan using known camera poses and the 3D–floor plan mapping.

### Key Designs

1. **SfM 3D Reconstruction**:

    - **Function**: Reconstruct 3D point clouds and camera poses from multi-view internet photographs.
    - **Mechanism**: A standard SfM pipeline is applied to internet photo collections to recover scene geometry and per-image camera intrinsics and extrinsics.
    - **Design Motivation**: Internet photos are readily available and offer broad coverage; SfM provides robust 3D reconstruction and serves as the geometric foundation for subsequent registration.

2. **Manual 3D–Floor Plan Registration**:

    - **Function**: Align the SfM-reconstructed 3D point cloud with floor plans sourced from the internet.
    - **Mechanism**: Annotators manually identify correspondences between structural keypoints in the 3D point cloud (e.g., wall corners, door frames) and their counterparts in the floor plan, then solve for a rigid or affine transformation.
    - **Design Motivation**: The large appearance gap between photographs and floor plans renders automatic registration unreliable; manual registration ensures high-quality ground-truth correspondences.

3. **Pixel-Level Correspondence Derivation**:

    - **Function**: Automatically generate dense pixel-level photo–floor plan correspondences from the 3D registration results.
    - **Mechanism**: Known camera poses allow 3D points to be projected onto image pixels; the 3D–floor plan registration maps 3D points to floor plan coordinates. Composing these two mappings yields photo pixel → floor plan pixel correspondences.
    - **Design Motivation**: Automated derivation enables large-scale production of correspondence data, eliminating the need for manual per-pixel annotation.

### Loss & Training
A pointmap prediction model based on the DUSt3R architecture is trained on the C3 dataset. Given a photo–floor plan pair as input, the model predicts a per-pixel 3D pointmap. Predicted pointmaps are then used to establish photo–floor plan correspondences via nearest-neighbor matching.

Training and test scenes are disjoint to ensure generalization evaluation. The loss function is based on the L2 distance between predicted and ground-truth pointmaps, with confidence weighting to handle occluded regions and boundaries. Predicted correspondences are evaluated both via RMSE for matching accuracy and via recall for camera pose estimation and localization precision.

## Key Experimental Results

### C3 Dataset Statistics

| Metric | Value |
|--------|-------|
| Number of scenes | 597 |
| Photo–floor plan pairs | 90K |
| Pixel-level correspondences | 153M |
| Camera poses | 85K |

### Main Results

| Method | RMSE Correspondence Error | Notes |
|--------|--------------------------|-------|
| Best existing method (e.g., DUSt3R) | baseline | Poor performance on cross-modality correspondence |
| Trained on C3 | −34% RMSE | Significant improvement |

### Key Findings
- State-of-the-art correspondence models such as DUSt3R degrade severely in cross-view cross-modality settings, falling far behind their same-modality performance.
- Training on C3 reduces the RMSE of the best-performing method by 34%, indicating that the primary bottleneck for this task is training data.
- Predicted correspondences can be used for camera pose estimation, though recall metrics reveal substantial room for further improvement.

## Highlights & Insights
- **Elegant dataset construction pipeline**: SfM reconstruction serves as a geometric bridge between photographs and floor plans, transforming the intractable cross-modality annotation problem into the more manageable task of 3D–floor plan registration.
- **Exposes an important research gap**: The vulnerability of models such as DUSt3R in cross-modality settings has been largely overlooked; the C3 dataset is expected to catalyze research in this direction.
- **Practical value**: Photo–floor plan correspondence has broad applications in indoor navigation, AR overlay, and architectural inspection.

## Limitations & Future Work
- **Manual registration is a bottleneck**: Per-scene manual 3D–floor plan registration limits the scalability of the dataset; further expansion requires an automated registration pipeline.
- **Indoor scenes only**: C3 focuses on indoor architectural photo–floor plan correspondence and does not yet cover outdoor cross-view settings (e.g., aerial–ground, satellite–street-level).
- **Floor plan diversity**: Internet floor plans vary widely in style (hand-drawn, CAD, rendered) and precision; standardization may discard critical structural information.
- **Dependence on SfM quality**: Uneven density and coverage of internet photos may result in incomplete reconstructions, affecting the density and accuracy of derived correspondences.
- **Absence of semantic annotations**: Pixel-level correspondences are purely geometric and lack semantic labels such as room function (e.g., kitchen, bedroom).
- **Future directions**: (1) Develop automatic or semi-automatic 3D–floor plan registration methods to reduce annotation cost; (2) extend to additional cross-modality pairs (e.g., satellite imagery–ground photos, CAD models–photographs); (3) leverage LLMs/VLMs to assist in understanding floor plan semantics and room structure.

## Related Work & Insights
- **vs. DUSt3R**: DUSt3R targets same-modality multi-view matching (photo–photo); C3PO extends this paradigm to cross-modality settings such as photo–floor plan.
- **vs. VIGOR**: VIGOR addresses ground-to-aerial viewpoint matching but uses only real images; C3 introduces abstract modalities such as floor plans, posing a substantially greater challenge.
- **vs. WAFFLE**: WAFFLE provides photo–floor plan associations but lacks pixel-level correspondences; C3 fills this gap.
- **Insight**: Cross-modality geometric understanding is an underestimated challenge; when the modality gap is large, geometric feature matching breaks down and semantic-level matching may be necessary.

## Rating
- Novelty: ⭐⭐⭐⭐ First large-scale pixel-level photo–floor plan correspondence dataset; exposes an important research gap.
- Experimental Thoroughness: ⭐⭐⭐ Benchmarking is reasonably comprehensive, but ablation details are limited given abstract-only access.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated; pipeline description is intuitive.
- Value: ⭐⭐⭐⭐ The dataset itself is of high value and will advance research on cross-modality geometric matching.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] UniABG: Unified Adversarial View Bridging and Graph Correspondence for Unsupervised Cross-View Geo-Localization](../../AAAI2026/remote_sensing/uniabg_unified_adversarial_view_bridging_and_graph_correspondence_for_unsupervis.md)
- [\[CVPR 2026\] GeoFlow: Real-Time Fine-Grained Cross-View Geolocalization via Iterative Flow Prediction](../../CVPR2026/remote_sensing/geoflow_real-time_fine-grained_cross-view_geolocalization_via_iterative_flow_pre.md)
- [\[ICCV 2025\] GeoDistill: Geometry-Guided Self-Distillation for Weakly Supervised Cross-View Localization](../../ICCV2025/remote_sensing/geodistill_geometry-guided_self-distillation_for_weakly_supervised_cross-view_lo.md)
- [\[NeurIPS 2025\] ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning](chamaevit_unifying_channelaware_masked_autoencoders_and_mult.md)
- [\[NeurIPS 2025\] GreenHyperSpectra: A Multi-Source Hyperspectral Dataset for Global Vegetation Trait Prediction](greenhyperspectra_a_multi-source_hyperspectral_dataset_for_global_vegetation_tra.md)

<!-- RELATED:END -->
