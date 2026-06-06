---
title: >-
  [Paper Note] Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting
description: >-
  [ICCV 2025][3D Vision][Spatial Intelligence] This paper proposes a scalable data generation pipeline that automatically converts single-view 2D images into metric-scale 3D representations—including point clouds…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Spatial Intelligence"
  - "2D-to-3D"
  - "Depth Estimation"
  - "Scale Calibration"
  - "Point Cloud Dataset"
date: 2026-05-08
content_hash: 8031ef75ace747b8
---

# Towards Scalable Spatial Intelligence via 2D-to-3D Data Lifting

**Conference**: ICCV 2025
**arXiv**: [2507.18678](https://arxiv.org/abs/2507.18678)  
**Code**: [Project Page](https://ZhangGongjie.github.io/TowardsSSI-page/)  
**Area**: 3D Vision
**Keywords**: Spatial Intelligence, 2D-to-3D, Depth Estimation, Scale Calibration, Point Cloud Dataset

## TL;DR

This paper proposes a scalable data generation pipeline that automatically converts single-view 2D images into metric-scale 3D representations—including point clouds, camera poses, and depth maps—by integrating depth estimation, camera calibration, and scale calibration. The pipeline produces COCO-3D and Objects365-v2-3D datasets comprising approximately 2 million scenes, yielding significant performance gains across multiple 3D tasks.

## Background & Motivation

Spatial Intelligence—the ability to perceive, reason about, and interact with 3D environments—is widely regarded as the next frontier of AI. However, progress in this field is severely bottlenecked by the scarcity of large-scale 3D datasets. The three existing pathways for acquiring 3D data each carry fundamental limitations:

**Simulation-based generation**: Game engines (e.g., NVIDIA Isaac Gym) can produce data rapidly and cheaply, but suffer from a pronounced sim-to-real gap; their simplified geometric and physical models fail to capture real-world complexity.

**AI-generated 3D assets**: Current methods are largely confined to single-object generation; scene-level generation remains challenging, and generated scenes frequently exhibit disproportionate elements, cartoonish appearances, and implausible object layouts.

**Sensor-based capture**: LiDAR and RGB-D cameras provide high-fidelity 3D data, but acquisition and annotation costs are prohibitive. Existing datasets are small in scale (ScanNet contains only 1,503 scenes) and typically restricted to specific domains (e.g., indoors).

Meanwhile, large-scale 2D image datasets (e.g., COCO, Objects365-v2) encompass vast, diverse, and richly annotated images, yet their potential for advancing spatial intelligence remains underexplored. The central insight of this work is: **existing 2D image data can be leveraged—through depth estimation and camera parameter prediction—to automatically generate high-quality 3D training data**.

## Method

### Overall Architecture

The data generation pipeline consists of four steps: (1) relative depth estimation → (2) metric depth estimation → (3) scale calibration → (4) camera parameter prediction + 3D projection. The core idea is to exploit the fine-grained geometric precision of relative depth and the global scale information of metric depth in a complementary manner.

### Key Designs

1. **Dual Depth Estimation and Scale Calibration**: This constitutes the primary methodological contribution of the paper.

    - **Relative Depth Estimation**: MoGe is employed to first estimate a 3D point cloud and then derive a relative depth map $d_r$. MoGe employs multi-scale local geometry losses to ensure local geometric accuracy, but lacks absolute scale information.
    - **Metric Depth Estimation**: Metric3D v2 takes focal length as input and predicts metric depth $d_m$ end-to-end. This model is jointly trained on diverse indoor and outdoor scenes, mitigating overfitting to the depth distribution of any single dataset.
    - **Scale Calibration**: A scaling factor $s$ is computed over the valid point set $\mathcal{V}$ to convert relative depth into scale-calibrated depth $d_{sc}$:

$$s = \frac{\frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}} d_{m,i}}{\frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}} d_{r,i}}, \quad d_{sc,i} = s \cdot d_{r,i}$$

   The resulting depth map simultaneously captures fine geometric details and correct global scale.

2. **Camera Parameter Prediction**: For in-the-wild images lacking ground-truth camera parameters, estimation proceeds in two stages:

    - **Intrinsics**: WildCamera is adopted to predict focal length and principal point, with scale-awareness and crop-detection capabilities.
    - **Extrinsics**: PerspectiveFields is used to infer camera pose—providing per-pixel up-direction vectors and latitude values—from which a rotation matrix is constructed to align the reconstructed point cloud with a canonical 3D coordinate system (z-axis up).

3. **3D Annotation Generation**: Using the scale-calibrated depth $d_{sc}$ and camera parameters $[K, R|T]$, each valid pixel $(u_i, v_i)$ is projected into 3D space:

$$\mathbf{P}_i^{\text{cam}} = d_{sc,i} \cdot K^{-1} \begin{bmatrix} u_i \\ v_i \\ 1 \end{bmatrix}, \quad \mathbf{P}_i^{\text{world}} = R \cdot \mathbf{P}_i^{\text{cam}} + T$$

   Segmentation annotations are projected directly into 3D; bounding box annotations use the minimum and maximum depth within the region to construct 3D boxes. For Objects365-v2, which provides only box annotations, SAM is first applied to generate masks before projection.

### Loss & Training

The primary contribution of this paper is **data generation** rather than a novel training procedure. A unified hyperparameter configuration is used across all models and datasets to avoid bias introduced by dataset-specific tuning. For downstream task training:
- Instance segmentation: Uni3D + Mask3D
- Semantic segmentation: SpUNet / PTv2 / Uni3D + 2-layer MLP
- Referring instance segmentation: TGNN
- QA and dense captioning: LL3DA

## Key Experimental Results

### Main Results

| Task | Metric | ScanNet Only | COCO-3D Pretrain + ScanNet | Gain |
|------|--------|--------------|---------------------------|------|
| Instance Segmentation | mAP | 24.30% | **28.64%** | +4.34 |
| Semantic Seg. (SpUNet) | mIoU | 31.09% | **62.48%** | +31.39 |
| Semantic Seg. (PTv2) | mIoU | 51.04% | **55.81%** | +4.77 |
| Semantic Seg. (Uni3D) | mIoU | 52.14% | **55.83%** | +3.69 |
| Referring Instance Seg. | mIoU | 26.10% | **32.47%** | +6.37 |
| 3D QA (ScanQA) | CIDEr | 75.67 | **79.11** | +3.44 |

### Ablation Study

| Model | Pretrain Data | ScanNet mIoU | ScanNet mAcc | ScanNet allAcc |
|-------|--------------|-------------|-------------|---------------|
| SpUNet | None | 31.09 | 36.54 | 68.63 |
| SpUNet | COCO-3D | **62.48** (+31.39) | **70.38** (+33.84) | **84.89** (+16.26) |
| PTv2 | None | 51.04 | 58.73 | 78.17 |
| PTv2 | COCO-3D | **55.81** (+4.77) | **63.19** (+4.46) | **80.62** (+2.45) |
| Uni3D | None | 52.14 | 59.06 | 79.05 |
| Uni3D | COCO-3D | **55.83** (+3.69) | **66.10** (+7.04) | **81.31** (+2.26) |

Zero-shot generalization is also strong: models trained solely on COCO-3D can be directly applied to ScanNet, S3DIS, Matterport3D, and Structured3D, achieving, for example, over 60% mAP on the Toilet category.

### Key Findings

- SpUNet achieves a +31.39% mIoU improvement after COCO-3D pretraining, demonstrating that synthetic data can substantially complement real data.
- Even though synthetic 3D data captures only partial-view point clouds, it generalizes effectively to full-view datasets such as ScanNet.
- The unified hyperparameter setting sacrifices peak per-dataset performance but more faithfully reflects the intrinsic value of the data.
- Height distribution analysis reveals that object heights in the synthetic data conform to real-world distributions (e.g., humans: 0.5–2.0 m), validating the reliability of the pipeline.

## Highlights & Insights

- **Methodologically clean and elegant**: The dual depth estimation + scale calibration strategy is simple yet effective—relative depth contributes geometric detail while metric depth provides global scale.
- **Unprecedented scale**: The COCO-3D training set comprises 117,183 scenes, far surpassing ScanNet (1,503) and Structured3D (3,500).
- **Unified hyperparameter setting**: Identical hyperparameters are used across all experiments, avoiding dataset-specific tuning and more honestly reflecting the value of the synthetic data.
- **Substantial open-source contribution**: COCO-3D and Objects365-v2-3D cover 300+ categories and approximately 2 million scenes, broadly applicable to spatial intelligence research.

## Limitations & Future Work

- Single-view reconstruction yields point clouds that capture only partial-view geometry, with inherent occlusion and incompleteness issues.
- Depth estimation remains inadequate for large-scale outdoor scenes, particularly those involving humans.
- The current approach uses only single-frame 2D images; leveraging multi-frame temporal consistency from video could further improve 3D reconstruction quality.
- A domain gap relative to real sensor data persists.
- Camera parameter prediction itself introduces errors that may cause inaccurate 3D projections in certain extreme scenarios.

## Related Work & Insights

- SpatialRGPT and SpatialBot also attempt to generate 3D data from 2D sources, but lack fine-grained geometry or camera parameters, limiting their applicability.
- The complementary dual depth estimation strategy proposed here can be generalized to other 3D reconstruction tasks that require simultaneous local accuracy and global consistency.
- Experiments demonstrate that "2D-to-3D lifting" is a viable foundational paradigm for advancing spatial intelligence.

## Rating

- Novelty: ⭐⭐⭐⭐ The dual depth estimation + scale calibration approach is concise and effective; the dataset scale is unprecedented.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers instance segmentation, semantic segmentation, referring segmentation, QA, and dense captioning, with zero-shot generalization validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Logic is clear, experimental design is fair, and the unified hyperparameter setting reflects principled intent.
- Value: ⭐⭐⭐⭐⭐ The open-source large-scale 3D datasets will directly advance progress in spatial intelligence research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning 3D Object Spatial Relationships from Pre-trained 2D Diffusion Models](learning_3d_object_spatial_relationships_from_pre-trained_2d_diffusion_models.md)
- [\[ICCV 2025\] DAViD: Data-efficient and Accurate Vision Models from Synthetic Data](david_data-efficient_and_accurate_vision_models_from_synthetic_data.md)
- [\[CVPR 2026\] UniSplat: Learning 3D Representations for Spatial Intelligence from Unposed Multi-View Images](../../CVPR2026/3d_vision/unisplat_3d_representations_unposed.md)
- [\[CVPR 2026\] Lifting Unlabeled Internet-level Data for 3D Scene Understanding](../../CVPR2026/3d_vision/lifting_unlabeled_internet-level_data_for_3d_scene_understanding.md)
- [\[NeurIPS 2025\] TRIM: Scalable 3D Gaussian Diffusion Inference with Temporal and Spatial Trimming](../../NeurIPS2025/3d_vision/trim_scalable_3d_gaussian_diffusion_inference_with_temporal_and_spatial_trimming.md)

</div>

<!-- RELATED:END -->
