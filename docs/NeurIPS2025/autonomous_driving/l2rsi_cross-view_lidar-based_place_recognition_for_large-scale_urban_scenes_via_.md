---
title: >-
  [Paper Note] L2RSI: Cross-View LiDAR-Based Place Recognition for Large-Scale Urban Scenes via Remote Sensing Imagery
description: >-
  [NeurIPS 2025][Autonomous Driving][Place Recognition] This paper proposes L2RSI, the first framework for LiDAR-based place recognition in ultra-large-scale urban scenes (100 km²) leveraging high-resolution remote sensing imagery. It aligns LiDAR BEV representations with remote sensing semantic spaces via semantic contrastive learning, and introduces Spatio-Temporal Particle Estimation (STPE) to aggregate spatio-temporal information from consecutive queries…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "Place Recognition"
  - "LiDAR"
  - "Remote Sensing Imagery"
  - "Cross-View Retrieval"
  - "Particle Estimation"
date: 2026-05-08
content_hash: 6d70e723aa1531a6
---

# L2RSI: Cross-View LiDAR-Based Place Recognition for Large-Scale Urban Scenes via Remote Sensing Imagery

**Conference**: NeurIPS 2025  
**arXiv**: [2503.11245](https://arxiv.org/abs/2503.11245)  
**Code**: [Project Page](https://shizw695.github.io/L2RSI/)  
**Area**: Autonomous Driving  
**Keywords**: Place Recognition, LiDAR, Remote Sensing Imagery, Cross-View Retrieval, Particle Estimation

## TL;DR

This paper proposes L2RSI, the first framework for LiDAR-based place recognition in ultra-large-scale urban scenes (100 km²) leveraging high-resolution remote sensing imagery. It aligns LiDAR BEV representations with remote sensing semantic spaces via semantic contrastive learning, and introduces Spatio-Temporal Particle Estimation (STPE) to aggregate spatio-temporal information from consecutive queries, achieving 83.27% Top-1 accuracy within a 100 km² retrieval range.

## Background & Motivation

Place recognition is a critical task for GPS-denied autonomous driving and robotic navigation. Conventional LiDAR-based place recognition relies on pre-built 3D maps, which are prohibitively expensive to acquire and maintain.

Limitations of existing methods:

**Uni-modal methods** (PointNetVLAD, MinkLoc3D, etc.) require reliable prior 3D maps.

**Cross-modal methods** (LIP-Loc, VXP, etc.) align LiDAR with other modalities but are limited to known routes or very small areas (<2 km²).

**Map-based methods** (OSM-based approaches) are constrained by the sparsity of map information.

**Core Innovation**: Using remote sensing imagery (satellite maps) as a reference database in place of 3D maps — remote sensing imagery offers global coverage, low cost, and good timeliness, enabling large-scale localization at minimal expense.

**Core Challenges**:
- A dual gap of **cross-view + cross-modality**: LiDAR (ground-level view, 3D point cloud) vs. remote sensing (bird's-eye view, 2D image).
- Single-query retrieval is unstable and ambiguous over large-scale search spaces.

## Method

### Overall Architecture

L2RSI consists of three modules:
1. **Data Preprocessing**: Converts remote sensing imagery and LiDAR point clouds into semantic maps respectively.
2. **Semantic Contrastive Learning Network**: Aligns global descriptors in a shared semantic space.
3. **Spatio-Temporal Particle Estimation (STPE)**: Aggregates spatio-temporal information from consecutive queries to refine retrieval results.

### Key Designs

#### 1. Data Preprocessing — Semantic Domain Unification

**Core Observation**: Although LiDAR point clouds and remote sensing imagery differ greatly in content and style, they are highly correlated in the **semantic domain** (roads, sidewalks, vegetation, and buildings are recognizable in both modalities).

**Remote Sensing Branch**:
- Applies the Alibaba AIE-SEG semantic segmentation model to segment remote sensing imagery into 4 semantic categories (road, sidewalk, vegetation, building).
- Extracts sub-images at equal intervals using a sliding window of $d=60\text{m}$ to form a semantic database.
- Filters out sub-images located far from roads.

**LiDAR Branch**:
- Registers short-sequence LiDAR scans using FastGICP to build point cloud submaps.
- Applies SphereFormer for semantic segmentation of corresponding categories.
- Projects point clouds into bird's-eye view (BEV) semantic maps using magnetometer heading (north-up alignment).

#### 2. Semantic Contrastive Learning Network

A dual-branch network where both branches **fully share weights** (since both operate in the semantic domain):
- Semantic encoder: initialized with MAE-pretrained ViT-B.
- GeM pooling + fully connected layer to generate global descriptors.

Symmetric InfoNCE loss for contrastive learning:

$$\mathcal{L} = -\log\frac{\exp(f_i^Q \cdot f_i^P / \tau)}{\sum_{j \in N} \exp(f_i^Q \cdot f_j^P / \tau)} - \log\frac{\exp(f_i^P \cdot f_i^Q / \tau)}{\sum_{j \in N} \exp(f_i^P \cdot f_j^Q / \tau)}$$

Temperature coefficient $\tau=0.1$. Compared to conventional triplet loss, increasing the number of in-batch negatives substantially accelerates training.

#### 3. Spatio-Temporal Particle Estimation (STPE)

Single-query retrieval is unstable over large-scale search spaces. STPE aggregates spatio-temporal information from a sequence of consecutive queries $\{Q_j\}_{t-L+1}^{t}$:

**Particle Modeling**: The Top-K retrieval results of each query are treated as particles. DBSCAN clustering (radius $r=30\text{m}$) determines the number of Gaussian components $M$, and the particle distribution is modeled as a GMM:

$$P(x,y) = \sum_{m=1}^{M} A_m \cdot \exp\left(-\frac{(x-\mu_{xm})^2}{2\sigma_{xm}^2} - \frac{(y-\mu_{ym})^2}{2\sigma_{ym}^2}\right)$$

**Temporal Propagation**: Inter-frame relative displacement is estimated via FastGICP to propagate historical query particle distributions to the current timestamp.

**Probabilistic Aggregation**: The propagated distributions are averaged to obtain the current location's probability density function $P_t(x,y) = \frac{1}{L}\sum_j \tilde{P}_j(x,y)$, which is then used to re-rank retrieval results.

**Key Distinction**: Unlike conventional particle filters, STPE performs particle **aggregation** rather than **filtering** — in challenging cross-view cross-modal retrieval, a single corrupted query may cause a filter to discard all reliable particles, making the aggregation strategy more robust.

### Loss & Training

- Loss: Symmetric InfoNCE (as above); positive pairs have center distance less than $d/2=30\text{m}$.
- Training is conducted on the LiRSI-XA dataset (approximately 12,194 LiDAR submaps + 47,913 remote sensing sub-images).
- Default parameters: $L=50$ (sequence length), $K=30$ (Top-K particles), $\lambda=30\%$ (sampling rate).

## Key Experimental Results

### Main Results

Recall@1 (<30 m) on the LiRSI-XA database at different scales:

| Method | 4 km² | 9 km² | 16 km² | 100 km² |
|--------|-------|-------|--------|---------|
| LIP-Loc | 16.82 | 13.69 | 11.60 | 11.02 |
| Sample4Geo | 28.63 | 25.39 | 24.23 | 22.95 |
| L2RSI (w/o STPE) | 30.05 | 23.90 | 21.93 | 20.07 |
| L2RSI (w. PF) | 71.66 | 55.17 | 50.87 | 47.85 |
| **L2RSI (w. STPE)** | **88.93** | **87.95** | **85.49** | **83.27** |

Cross-scene generalization (LiRSI-Oxford, without fine-tuning):

| Trajectory | Recall@1 | Recall@10 |
|------------|----------|-----------|
| 11-14-02-26 | 42.29 | 59.41 |
| 14-12-05-52 | 43.77 | 62.76 |

### Ablation Study

| Component | Test-S R@1 | Oxford R@1 |
|-----------|-----------|------------|
| Full L2RSI | **88.93** | **42.29** |
| w/o Semantic Segmentation | 50.43 | 8.13 |
| w/o STPE | 30.05 | 10.19 |
| w/o Orientation Information | 54.37 | 4.92 |

### Key Findings

1. **STPE is the core driver of performance gains**: R@1 improves from 30.05% without STPE to 88.93% with STPE (+58.88%), confirming the substantial value of spatio-temporal aggregation.
2. **Semantic domain unification is the key to cross-modal alignment**: Removing semantic segmentation causes R@1 to drop from 88.93% to 50.43%.
3. **STPE substantially outperforms conventional particle filtering**: 88.93% vs. 71.66%, demonstrating that aggregation is superior to filtering.
4. Expanding the retrieval range from 4 km² to 100 km² results in only a 5.66% drop in R@1, indicating strong robustness.
5. A sampling rate of 30% suffices to achieve optimal performance, with STPE adding only 31.7 ms overhead.
6. The motion model tolerates additional yaw noise of ±10° and translational noise of ±1 m.

## Highlights & Insights

1. **First cross-view LiDAR place recognition at the 100 km² scale**, representing significant practical value.
2. The idea of using the **semantic domain as a bridge** is simple yet effective: mapping two entirely different modalities into a shared semantic space.
3. **The aggregation philosophy of STPE** overturns the traditional paradigm of particle filtering, making it more suitable for challenging retrieval scenarios where individual observations are highly unreliable.
4. **Zero-shot cross-scene generalization**: the model operates in a completely different city (Xiamen → Oxford) without fine-tuning.

## Limitations & Future Work

1. Relies on an additional magnetometer for orientation information, increasing hardware requirements.
2. Cross-scene generalization accuracy (~42%) is insufficient for practical deployment and requires further improvement.
3. Semantic segmentation quality is a bottleneck — occluded areas such as underpasses and dense tree canopy cause mismatches between LiDAR and remote sensing semantics.
4. A 3-year temporal gap exists between remote sensing imagery and LiDAR data; changes in buildings and vegetation introduce noise.
5. Validation is currently limited to urban road scenes; rural or unstructured environments remain unexplored.

## Related Work & Insights

- **Uni-modal 3D place recognition**: PointNetVLAD, MinkLoc3D, and similar methods require prior 3D maps; L2RSI replaces these with remote sensing imagery.
- **Cross-view 2D place recognition**: GeoDTR, Sample4Geo, etc. perform poorly when directly applied to cross-modal retrieval, highlighting the importance of semantic domain unification.
- **Particle filter-based localization**: L2RSI's STPE reframes the particle concept from "filtering convergence" to "probabilistic aggregation."
- **Inspiration**: The approach of using remote sensing imagery as a low-cost map substitute is generalizable to urban UAV navigation, post-disaster search and rescue, and other scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — First cross-modal LiDAR place recognition at the hundred-kilometer scale; STPE design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Includes a self-constructed large-scale dataset, multi-scale evaluation, cross-scene generalization, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear; experimental organization is well-structured.
- Value: ⭐⭐⭐⭐⭐ — Strongly application-oriented; has the potential to advance large-scale localization in GPS-denied scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] ForestLPR: LiDAR Place Recognition in Forests Attentioning Multiple BEV Density Images](../../CVPR2025/autonomous_driving/forestlpr_lidar_place_recognition_in_forests_attentioning_multiple_bev_density_i.md)
- [\[NeurIPS 2025\] CuMoLoS-MAE: A Masked Autoencoder for Remote Sensing Data Reconstruction](cumolos-mae_a_masked_autoencoder_for_remote_sensing_data_reconstruction.md)
- [\[NeurIPS 2025\] X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)
- [\[CVPR 2026\] C-LaV: Conditional Latent Velocity Field Denoising for Weather-Robust LiDAR Place Recognition](../../CVPR2026/autonomous_driving/c-lav_conditional_latent_velocity_field_denoising_for_weather-robust_lidar_place.md)
- [\[NeurIPS 2025\] UrbanIng-V2X: A Large-Scale Multi-Vehicle Multi-Infrastructure Dataset Across Multiple Intersections for Cooperative Perception](urbaning-v2x_a_large-scale_multi-vehicle_multi-infrastructure_dataset_across_mul.md)

</div>

<!-- RELATED:END -->
