---
title: >-
  [Paper Note] Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting
description: >-
  [CVPR 2025][Autonomous Driving][Occupancy Forecasting] EfficientOCF is proposed to solve the spatial and temporal biases in occupancy forecasting through spatial decoupling (decomposing 3D occupancy into 2D BEV occupancy + height values) and temporal decoupling (achieving step-by-step OCF instead of end-to-end prediction by associating instances via optical flow), achieving SOTA 3D occupancy forecasting performance and a fast inference time of 82.33ms.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Occupancy Forecasting"
  - "Spatiotemporal Decoupling"
  - "BEV Representation"
  - "Instance-Aware"
date: 2026-05-08
content_hash: 659e20cc9528fffa
---

# Spatiotemporal Decoupling for Efficient Vision-Based Occupancy Forecasting

**Conference**: CVPR 2025  
**arXiv**: [2411.14169](https://arxiv.org/abs/2411.14169)  
**Code**: None  
**Area**: Autonomous Driving / Occupancy Forecasting  
**Keywords**: Occupancy Forecasting, Spatiotemporal Decoupling, BEV Representation, Instance-Aware, Autonomous Driving

## TL;DR

EfficientOCF is proposed to solve the spatial and temporal biases in occupancy forecasting through spatial decoupling (decomposing 3D occupancy into 2D BEV occupancy + height values) and temporal decoupling (achieving step-by-step OCF instead of end-to-end prediction by associating instances via optical flow), achieving SOTA 3D occupancy forecasting performance and a fast inference time of 82.33ms.

## Background & Motivation

3D Occupancy Forecasting (OCF) predicts the future occupancy state of the environment using past and current perception data, which is crucial for obstacle avoidance and path planning in autonomous driving. Existing 3D OCF methods suffer from two core biases:

1. **Spatial Bias**: The vast majority of voxels in 3D space are empty, but end-to-end methods still process the entire voxel grid, leading to wasted computation and biasing occupancy predictions towards the "empty" state.
2. **Temporal Bias**: Only a small number of movable objects change their positions in the short term, with static objects dominating. Consequently, end-to-end predictions of dynamic object shapes tend to diverge over time.

Existing methods (such as OCFNet) utilize dense 3D feature encoder-decoders to process all voxels, resulting in high computational costs and inaccurate predictions for movable objects. Although BEV methods are highly efficient, they lack representation of the z-axis spatial structure.

Core Idea: **Decoupling the spatial and temporal dimensions of 3D OCF**—utilizing 2D BEV + height to represent 3D in the spatial dimension, and separating instance segmentation from occupancy forecasting in the temporal dimension.

## Method

### Overall Architecture

EfficientOCF consists of four modules: (1) **Perception Module**: extracts 2D features from multi-view images and projects them into 3D voxel features using Lift-Splat-Shoot; (2) **Aggregation Module**: compresses 3D features into 2D BEV features via adaptive dual pooling and aggregates multiple frames; (3) **Prediction Module**: contains three heads sharing a 2D encoder-decoder structure (segmentation/height/optical flow); (4) **Refinement Module**: performs step-by-step refinement of OCF results by associating instances via optical flow.

### Key Designs

**1. Spatial Decoupling: BEV Occupancy + Height Representation**

- **Function**: Downscales 3D occupancy forecasting to 2D prediction, significantly improving efficiency.
- **Mechanism**: Instead of the traditional dense 3D voxels $O_t^{3D} \in \mathbb{R}^{1 \times H \times W \times L}$, it predicts 2D BEV occupancy $O_t^{2D} \in \mathbb{R}^{1 \times H \times W}$ and corresponding height values $O_t^{height} \in \mathbb{R}^{1 \times H \times W}$, storing height information only for occupied grids. The final 3D occupancy is reconstructed through height lifting.
- **Design Motivation**: Since empty voxels dominate 3D space, allocating computational resources to each of them is wasteful. The BEV + height representation preserves 3D spatial information while maintaining the computational complexity of 2D space.

**2. Temporal Decoupling: Instance Association Refinement**

- **Function**: Decouples instance segmentation from occupancy forecasting, improving prediction quality for future frames via instance association.
- **Mechanism**: The optical flow head predicts the 2D backward centripetal flow $O_t^{flow} \in \mathbb{R}^{2 \times H \times W}$ (pointing to the instance center in the previous frame). At $t=-1$, the refinement module performs NMS to extract instance centers, iteratively associates instance IDs $M_t^{2D}$ along the temporal axis using optical flow, and generates instance masks $\bar{M}_t^{2D}$ to filter the initial OCF results: $\bar{O}_t^{2D} = O_t^{2D} \cdot \bar{M}_t^{2D}$.
- **Design Motivation**: Instance segmentation based on current observations is more accurate than end-to-end future predictions. Propagating instances rather than re-predicting them preserves the temporal consistency of object shapes and reduces shape divergence.

**3. Adaptive Dual Pooling Strategy**

- **Function**: Efficiently projects 3D voxel features into 2D BEV features.
- **Mechanism**: Simultaneously utilizes average pooling (to capture global information) and max pooling (to implicitly represent the salient occupancy features corresponding to height values), and adaptively fuses them using learnable weights $\alpha_{avg}$ and $\alpha_{max}$: $F^{BEV} = \alpha_{avg} F^{avg} + \alpha_{max} F^{max}$.
- **Design Motivation**: A single pooling strategy suffers from substantial information loss. Dual pooling compresses z-axis information from different perspectives.

### Loss & Training

- **Total Loss**: $\mathcal{L}_{all} = \frac{1}{N_f+1}\sum_{t}(\lambda_1 \mathcal{L}_{occ} + \lambda_2 \mathcal{L}_{height} + \lambda_3 \mathcal{L}_{flow})$
    - $\mathcal{L}_{occ}$: Cross-entropy loss (2D BEV occupancy)
    - $\mathcal{L}_{height}$: Smooth L1 loss (height prediction)
    - $\mathcal{L}_{flow}$: Smooth L1 loss (optical flow prediction)
- A new metric **C-IoU** is proposed to mitigate the false-positive penalty within bounding boxes, providing a more reasonable evaluation under incomplete annotations.
- Evaluated and trained on three datasets: nuScenes, nuScenes-Occupancy, and Lyft-Level5.

## Key Experimental Results

### Main Results

3D occupancy forecasting comparison on nuScenes (inflated annotations):

| Method | IoU_c ↑ | IoU_f ↑ | IoU_all ↑ | Inference Time |
|------|---------|---------|-----------|---------|
| PowerBEV | 36.15 | 34.18 | 34.58 | - |
| OccFormer | 41.68 | 28.55 | 31.00 | - |
| OCFNet | 40.25 | 30.38 | 32.33 | 173ms |
| **EfficientOCF** | **43.25** | **36.11** | **37.46** | **82.33ms** |

### Ablation Study

Ablation of each component (nuScenes, 2D BEV IoU):

| Configuration | IoU_c | IoU_f | IoU_all |
|------|-------|-------|---------|
| Baseline (avg pooling) | 33.62 | 30.07 | 30.77 |
| + Dual Pooling | Gain | Gain | Gain |
| + Height Prediction | Further Gain | - | - |
| + Instance Refinement | **Highest** | **Highest** | **Highest** |

### Key Findings

1. **Over 2x inference speedup**: 82.33ms vs. 173ms of OCFNet, benefiting from replacing 3D prediction with 2D prediction.
2. **Instance refinement significantly improves future frame prediction**: IoU_f shows clear improvements over the baseline, especially in distant future timesteps.
3. **C-IoU metric is more reasonable**: On nuScenes-Occupancy, where annotations are incomplete due to sparse LiDAR, C-IoU reduces the unfair penalty on false positives inside bounding boxes.
4. **Achieves SOTA across three datasets**, verifying the generalization of the proposed method.

## Highlights & Insights

1. **The BEV + height concept for spatial decoupling is simple and efficient**: Capitalizing on the characteristics of vast empty voxels in 3D space, it records height information only for occupied grids.
2. **The insight "instance propagation is superior to end-to-end prediction" in temporal decoupling** is highly practical and clearly helps with shape preservation.
3. **The C-IoU metric** fills the gap in OCF evaluation under incomplete annotations.

## Limitations & Future Work

1. Height prediction provides only a single height value per column, which cannot represent multi-layer occupancy in the vertical direction (e.g., overpasses).
2. Instance refinement relies on the quality of optical flow; optical flow for distant or occluded features may be inaccurate.
3. Although C-IoU is more reasonable, it is still at the bounding box level and does not fully resolve annotation quality issues.
4. A multi-task framework joint with semantic prediction can be explored.

## Related Work & Insights

- **OCFNet (Cam4DOcc)**: The first visual 3D OCF benchmark. Ours proposes a more efficient decoupling paradigm based on this.
- **PowerBEV**: A 2D BEV instance prediction method, using backward centripetal flow for instance association. EfficientOCF extends this concept to 3D.
- **FIERY**: The first BEV instance prediction method from multi-camera inputs.
- **Lift-Splat-Shoot**: A classic method for lifting 2D features to 3D, adopted by the perception module of EfficientOCF.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The design concept of spatiotemporal decoupling is clear and practical, with the BEV + height representation drastically improving efficiency.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Complete evaluation across three datasets, various evaluation metrics, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear method motivation and design logic.
- **Value**: ⭐⭐⭐⭐ — Practical value for real-time occupancy forecasting in autonomous driving, achieving improvements in both inference speed and accuracy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Trajectory Mamba: Efficient Attention-Mamba Forecasting Model Based on Selective SSM](trajectory_mamba_efficient_attention-mamba_forecasting_model_based_on_selective_.md)
- [\[ICCV 2025\] Occupancy Learning with Spatiotemporal Memory](../../ICCV2025/autonomous_driving/occupancy_learning_with_spatiotemporal_memory.md)
- [\[CVPR 2025\] GaussianFormer-2: Probabilistic Gaussian Superposition for Efficient 3D Occupancy Prediction](gaussianformer-2_probabilistic_gaussian_superposition_for_efficient_3d_occupancy.md)
- [\[CVPR 2025\] DecoupledGaussian: Object-Scene Decoupling for Physics-Based Interaction](decoupledgaussian_object-scene_decoupling_for_physics-based_interaction.md)
- [\[ICCV 2025\] UniOcc: A Unified Benchmark for Occupancy Forecasting and Prediction in Autonomous Driving](../../ICCV2025/autonomous_driving/uniocc_a_unified_benchmark_for_occupancy_forecasting_and_prediction_in_autonomou.md)

</div>

<!-- RELATED:END -->
