---
title: >-
  [Paper Note] ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images
description: >-
  [CVPR 2025][Autonomous Driving][Place Recognition] This paper proposes ForestLPR, which slices point clouds at different heights to generate multiple BEV density maps. It leverages ViT to extract local features, followed by a multi-BEV interaction module to adaptively attend to discriminative features at different heights, achieving robust LiDAR place recognition in forest environments and significantly outperforming prior SOTA methods on multiple datasets.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Place Recognition"
  - "LiDAR"
  - "BEV Density Maps"
  - "Forest Environments"
  - "Multi-height Interaction"
  - "Rotation Invariance"
date: 2026-05-08
content_hash: 59bc757c06ab825d
---

# ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images

**Conference**: CVPR 2025  
**arXiv**: [2503.04475](https://arxiv.org/abs/2503.04475)  
**Code**: [shenyanqing1105/ForestLPR-CVPR2025](https://github.com/shenyanqing1105/ForestLPR-CVPR2025)  
**Area**: Autonomous Driving  
**Keywords**: Place Recognition, LiDAR, BEV Density Maps, Forest Environments, Multi-height Interaction, Rotation Invariance

## TL;DR

This paper proposes ForestLPR, which slices point clouds at different heights to generate multiple BEV density maps. It leverages ViT to extract local features, followed by a multi-BEV interaction module to adaptively attend to discriminative features at different heights, achieving robust LiDAR place recognition in forest environments and significantly outperforming prior SOTA methods on multiple datasets.

## Background & Motivation

### Background
Place recognition is a core technology for maintaining global consistency in large-scale localization systems, providing loop closure constraints for SLAM. While LiDAR-based place recognition has made significant progress in urban environments, its application in natural forest environments remains severely understudied.

### Limitations of Prior Work
1. **High self-similarity of forest environments**: Trees look extremely similar, lacking distinctive landmarks such as buildings or signs found in urban environments. Traditional feature extraction methods struggle to distinguish between different locations.
2. **Drastic temporal vegetation changes**: Seasonal changes lead to massive appearance variations in tree crowns and shrubs, making cross-temporal relocalization extremely challenging.
3. **Loss of height information**: Existing BEV-based methods typically generate a single BEV image, compressing all height information into one plane and discarding vital discriminative information along the z-axis.
4. **Sensitivity of 3D methods to point distribution**: Methods directly operating on 3D point clouds (e.g., MinkLoc3D) are sensitive to variations in point distribution, resulting in poor generalization across different sensors and tree species.

### Key Challenge
How to extract stable, highly discriminative, and scene-generalizable place descriptors in highly self-similar and temporally varying forest environments?

### Key Insight
Based on two key assumptions: (1) The unique spatial distribution of different trees in a forest can be used for place differentiation; (2) The discriminative features of different trees may reside at different heights. Therefore, the point clouds are sliced at various heights to generate multiple BEV density maps, and an interaction module is designed to adaptively focus on the most discriminative heights at each location.

### Core Idea
Replace single-layer BEV or raw 3D point clouds with multi-layer BEV density maps. Through a patch-level height-adaptive attention mechanism, stable discriminative features are implicitly identified, while robustness is enhanced by preprocessing that filters out the ground and high tree crowns.

## Method

### Overall Architecture
The pipeline of ForestLPR consists of four stages: (1) Preprocessing—ground segmentation, height offset removal, ground and tree canopy cropping; (2) Multi-BEV density map generation—slicing at different heights and projecting to generate BEV density maps; (3) Feature extraction—a shared ViT backbone extracts multi-layer local features; (4) Multi-BEV interaction and global aggregation—patch-level height attention weighting followed by GeM pooling to generate rotation-invariant global descriptors.

### Key Designs

#### 1. Point Cloud Preprocessing and Multi-BEV Density Map Generation

- **Function**: Remove high-variability elements of forest environments (ground, high tree crowns) and generate multi-layer BEV density maps within a stable height range.
- **Mechanism**:
    - After ground segmentation, the weighted average height of neighboring ground points is subtracted from each non-ground point to eliminate the influence of terrain fluctuations.
    - Points below 1m (grass, fallen leaves, snow, etc.) and above 6m (seasonal tree canopy changes) are discarded.
    - The remaining point cloud is divided into $S=5$ horizontal slices between 1m and 6m at 1m intervals.
    - Each slice is projected onto the ground and discretized, where density is calculated as $\log(V'(u,v)+1)$ and normalized to generate a BEV density map.
- **Design Motivation**: Ground and high tree crowns are the most susceptible parts of forests to weather and seasonal changes; removing them significantly improves cross-temporal consistency. Multi-layer BEVs retain the structural distribution along the height axis. The logarithmic transform prevents high-density regions from dominating the learning process.

#### 2. Multi-BEV Interaction Module

- **Function**: Achieve patch-level adaptive weighted fusion of BEV features at different heights, allowing each position to focus on the most discriminative height layers.
- **Mechanism**:
    - $S$ BEV density maps share the same DeiT backbone to extract local features respectively, yielding $\mathbf{P'} \in \mathbb{R}^{(N+2) \times S \times 3C}$.
    - For each patch, the relative difference between the height-specific feature and the mean feature is computed as $\Delta \mathbf{P'}_i$, which is then passed through a learnable weight $\mathbf{W}_a$ and Softmax to generate height attention weights $\mathbf{w}_i \in \mathbb{R}^{S \times 1}$.
    - The weighted sum of the features at different heights is computed using these weights to obtain the fused feature $\mathbf{P}^w_i$.
- **Design Motivation**: Even after removing the ground and canopy, intermediate layers in forests still contain noise such as shrub remnants. Useful information at different patch positions may come from different heights, requiring adaptive selection. Utilizing relative feature values instead of absolute values captures relative relationships between heights, and the input dimension remains fixed regardless of variations in $S$.
- **Core Characteristic**: The height weight is calculated independently for each patch position, achieving truly spatially adaptive height attention.

#### 3. Descriptor Extraction and Global Aggregation

- **Function**: Generate a rotation-invariant global place descriptor.
- **Mechanism**:
    - The DeiT backbone extracts output tokens from low, mid, and high layers, which are concatenated along the channel dimension to obtain multi-layer features.
    - After multi-BEV interaction, GeM pooling is applied to extract rotation-invariant global features from patch-level features.
    - The [class] and [distillation] tokens are concatenated with the GeM output, followed by L2 normalization and linear dimension reduction to output the final global descriptor $\mathbf{G} \in \mathbb{R}^D$ ($D=1024$).
- **Design Motivation**: Combining outputs from multiple transformer layers integrates structural information at different scales. GeM pooling inherently possesses rotation invariance, which is well-suited for reverse-revisit scenarios in forest environments.
- **Loss Function**: Triplet loss $\mathcal{L} = \max(d(\mathbf{G}^q, \mathbf{G}^p) - d(\mathbf{G}^q, \mathbf{G}^n) + m, 0)$ using Cosine distance. Positive samples are defined by volumetric overlap (calculated by Octree, with IoU > 0.9) rather than simple distance thresholds.

## Key Experimental Results

### Main Results — Intra-sequence Loop Closure Detection

| Method | Average F1 (7 Evaluation Sets) | Average R@1 (7 Evaluation Sets) |
|------|-----------------|-----------------|
| LoGG3D-Net | 72.29 | 71.07 |
| MinkLoc3Dv2 | 56.75 | 55.41 |
| BEVPlace | 50.18 | 60.56 |
| MapClosures | 57.37 | 43.64 |
| **ForestLPR** | **75.46** | **78.45** |

R@1 achieves an average improvement of **7.38%** compared to the strongest competitor LoGG3D-Net.

### Inter-sequence Relocalization
In the Wild-Places inter-sequence evaluation, ForestLPR improves average R@1 by **9.11%**.

### Zero-shot Generalization
- Trained only on the Wild-Places training set, and directly evaluated on the ANYmal and Botanic datasets **without fine-tuning**.
- ANYmal dataset: F1 81.45 (outperforming Scan-Context by 8.01%), R@1 71.87 (outperforming Scan-Context by 4.06%).
- Performs well across scenes with different sensors (VLP-16 vs. handheld), differing tree species, and varying vegetation densities.

### Key Findings
1. ForestLPR exhibits the most pronounced advantages in challenging scenarios with frequent reverse-revisit trajectories (e.g., V-03, K-03).
2. Multi-BEV density maps are keys to performance enhancement; transitioning from single-BEV to multi-BEV yields substantial gains.
3. Volumetric overlap-based positive sample mining is more accurate than simple distance thresholds, especially in the presence of occlusions and blind spots.

## Highlights & Insights

1. **Precise problem definition**: Place recognition in forest environments is an overlooked but crucial problem with unique challenges (high self-similarity, seasonal variations). This work is the first to address it systematically.
2. **Hypothesis-driven design**: Starting from the intuition that "different heights contain distinct discriminative information," a complete pipeline of multi-BEV density maps to height-adaptive attention is constructed with clear logic.
3. **High engineering value**: The preprocessing pipeline (ground segmentation + height normalization + cropping) is simple yet effective, allowing direct transfer to other forest-based perception tasks.
4. **Strong cross-domain generalization**: Excellent performance is maintained across different sensors, tree species, and terrain conditions, demonstrating the strong generalizability of the design.

## Limitations & Future Work

1. Preprocessing parameters (1m lower bound, 6m upper bound, 1m slicing interval) may not be applicable to all forest types (e.g., tropical rainforests where trees can reach dozens of meters).
2. Currently, only place recognition is supported; 6-DoF pose estimation is not addressed.
3. The resolution of the BEV density maps (0.5m) may not be sufficiently fine-grained for close-range scenarios.
4. Training relies solely on the single Wild-Places dataset; training on larger and more diverse datasets could further improve generalization.

## Related Work & Insights

- **Comparison with BEVPlace**: BEVPlace uses single-layer BEV density maps alongside group convolutions; while it performs well in some uniform forest scenes, it suffers from high computational cost. ForestLPR is more robust due to multi-layer BEVs and adaptive attention.
- **Comparison with LoGG3D-Net**: LoGG3D-Net captures local consistency based on sparse convolutions, operating directly on 3D point clouds. ForestLPR converts the 3D problem into a multi-layer 2D problem, exploiting the strong representation capability of pretrained ViTs.
- **Generality of BEV density maps**: As a 2D representation of LiDAR, BEV density maps are independent of specific sensor configurations and are more suitable for multi-frame concatenated submap scenarios compared to range images.
- **Insights for robot navigation**: Place recognition in forest environments is of direct value to applications such as search-and-rescue robots, and agricultural and forestry inspection.

## Rating

⭐⭐⭐⭐ (4/5)

The task selection is unique with realistic application value. The method design is logically clear, and the experiments are comprehensive. The multi-BEV interaction mechanism is both novel and effective. However, the core technical innovations are relatively concentrated around the introduction of multi-height BEV representation; future work could consider combining semantic information to further improve discriminative ability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] L2RSI: Cross-View LiDAR-Based Place Recognition for Large-Scale Urban Scenes via Remote Sensing Imagery](../../NeurIPS2025/autonomous_driving/l2rsi_cross-view_lidar-based_place_recognition_for_large-scale_urban_scenes_via_.md)
- [\[ECCV 2024\] Rethinking LiDAR Domain Generalization: Single Source as Multiple Density Domains](../../ECCV2024/autonomous_driving/rethinking_lidar_domain_generalization_single_source_as_multiple_density_domains.md)
- [\[CVPR 2026\] C-LaV: Conditional Latent Velocity Field Denoising for Weather-Robust LiDAR Place Recognition](../../CVPR2026/autonomous_driving/c-lav_conditional_latent_velocity_field_denoising_for_weather-robust_lidar_place.md)
- [\[CVPR 2025\] Tra-MoE: Learning Trajectory Prediction Model from Multiple Domains for Adaptive Policy Conditioning](tra-moe_learning_trajectory_prediction_model_from_multiple_domains_for_adaptive_.md)
- [\[CVPR 2025\] LiSu: A Dataset and Method for LiDAR Surface Normal Estimation](lisu_a_dataset_and_method_for_lidar_surface_normal_estimation.md)

</div>

<!-- RELATED:END -->
