---
title: >-
  [Paper Note] Open-Canopy: Towards Very High Resolution Forest Monitoring
description: >-
  [CVPR 2025][Autonomous Driving][Canopy height estimation] Open-Canopy proposes the first open-access, nation-wide, very high resolution ($1.5\text{m}$) canopy height estimation benchmark dataset covering over $87,000\text{ km}^2$ in France, combining SPOT satellite imagery and airborne LiDAR data. It also introduces a benchmark task for canopy height change detection, Open-Canopy-$\Delta$, establishing a comprehensive experimental baseline across a series of SOTA models.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Canopy height estimation"
  - "very high resolution remote sensing"
  - "LiDAR"
  - "satellite imagery"
  - "forest monitoring"
date: 2026-05-08
content_hash: 082e767cbbdaf0dc
---

# Open-Canopy: Towards Very High Resolution Forest Monitoring

**Conference**: CVPR 2025  
**arXiv**: [2407.09392](https://arxiv.org/abs/2407.09392)  
**Code**: [https://github.com/fajwel/Open-Canopy](https://github.com/fajwel/Open-Canopy)  
**Area**: Autonomous Driving  
**Keywords**: Canopy height estimation, very high resolution remote sensing, LiDAR, satellite imagery, forest monitoring

## TL;DR
Open-Canopy proposes the first open-access, nation-wide, very high resolution ($1.5\text{m}$) canopy height estimation benchmark dataset covering over $87,000\text{ km}^2$ in France, combining SPOT satellite imagery and airborne LiDAR data. It also introduces a benchmark task for canopy height change detection, Open-Canopy-$\Delta$, establishing a comprehensive experimental baseline across a series of SOTA models.

## Background & Motivation

**Background**: Estimating canopy height and its temporal change from satellite imagery has critical applications in environmental monitoring—including forest health assessment, logging activity tracking, timber resource estimation, and carbon stock calculation. Although existing methods have made considerable progress at low-to-medium resolutions ($10-30\text{m}$, e.g., Sentinel-2), high-precision canopy height estimation at meter-level resolution remains highly challenging.

**Limitations of Prior Work**: (1) Lack of open datasets: Most existing remote sensing datasets for forestry are based on commercial or proprietary sources (e.g., Planet, WorldView), severely hurting model reproducibility and fair comparison; (2) Insufficient resolution: Publicly available global datasets (e.g., GEDI + Sentinel-2) only provide $10\text{m}$ resolution, failing to capture individual tree information; (3) Absence of change detection: Almost no public datasets support multi-temporal canopy height change detection tasks.

**Key Challenge**: Meter-level canopy mapping is a critical requirement for precision forest management, but there is a severe conflict between data availability and openness—high-resolution satellite data are typically commercial, while free open-source data suffer from low resolution.

**Goal**: (1) Build the first open-access, nation-wide, meter-resolution canopy height estimation benchmark; (2) Propose a multi-temporal change detection benchmark; (3) Evaluate various SOTA computer vision models under a unified framework.

**Key Insight**: France possesses two key open-access data resources—SPOT 6-7 satellite imagery ($1.5\text{m}$ resolution, open access) and IGN airborne LiDAR HD (centimeter-level precision elevation data). By combining these two, large-scale, high-quality training data can be constructed without purchasing commercial datasets.

**Core Idea**: Leverage France's publicly available SPOT satellite imagery and LiDAR data to build the first nation-wide open canopy height estimation benchmark, while introducing temporal change detection as a new challenging task.

## Method

### Overall Architecture
Open-Canopy is not a new model, but a dataset and benchmark system. Its framework consists of: (1) Data acquisition and processing—precisely aligning SPOT satellite imagery with LiDAR data; (2) Dataset construction—splitting training, validation, and test sets to cover diverse terrains and vegetation types in France; (3) Baseline evaluation—benchmarking various SOTA architectures under unified experimental settings.

### Key Designs

1. **Large-scale Dataset Construction (Open-Canopy)**:

    - **Function**: To provide standardized training and evaluation data for very high resolution canopy height estimation.
    - **Mechanism**: The dataset covers over $87,000\text{ km}^2$ of France, organized in a $1\text{km} \times 1\text{km}$ tile grid. Each tile contains: (a) SPOT 6-7 multispectral satellite imagery ($1.5\text{m}$ resolution, 4-band RGBNIR) as input; (b) a Canopy Height Model (CHM) derived from airborne LiDAR HD data as the ground truth label. The LiDAR data achieves centimeter-level precision, and the canopy height is obtained by calculating the difference between the Digital Surface Model (DSM) and the Digital Terrain Model (DTM) from the point cloud. A $1\text{km}$ buffer zone is established between the training and test sets to avoid data leakage.
    - **Design Motivation**: Both France's SPOT data and LiDAR HD data are freely accessible (open license), allowing the entire dataset to be fully reproduced by researchers worldwide. The $87,000\text{ km}^2$ coverage ensures sufficient diversity in vegetation types (broadleaf, coniferous, mixed forest, shrub, etc.) and topographic conditions.

2. **Temporal Change Detection Benchmark (Open-Canopy-$\Delta$)**:

    - **Function**: To evaluate the models' ability to detect temporal changes in canopy height.
    - **Mechanism**: Pairwise temporal observations are constructed utilizing SPOT satellite imagery and LiDAR data from different years (typically spaced 2-5 years apart). The model takes satellite images from two time points as input and outputs a canopy height change map. Types of changes include natural growth (height increase), logging activities (sudden height drop), and natural disaster impacts. This is a highly challenging task as models need to simultaneously understand single-temporal height estimation and cross-temporal difference detection.
    - **Design Motivation**: In actual forest management, changes in canopy height are much more valuable than absolute height, as they directly reflect key events such as logging, disasters, and regeneration. However, there is currently no publicly available evaluation benchmark for this task.

3. **Comprehensive Model Benchmarking**:

    - **Function**: To evaluate multiple SOTA computer vision architectures under unified settings.
    - **Mechanism**: Evaluated models include: (a) classic CNN architectures—UNet (ResNet backbone), DeepLab; (b) Transformer architectures—ViT (small/base), PVTv2; (c) hybrid architectures—Swin Transformer + UNet; (d) domain-specific methods—self-supervised pre-trained model by Tolan et al. All models use unified data augmentation, training hyperparameters, and evaluation metrics (MAE, RMSE, $R^2$, etc.) to ensure fair comparison. Evaluations are conducted separately for height estimation (Open-Canopy) and change detection (Open-Canopy-$\Delta$).
    - **Design Motivation**: The performance differences of various architectures on remote sensing data remain unclear—for example, while Transformers have surpassed CNNs on natural images, do they still hold an advantage in remote sensing scenarios? Do large pre-trained models outperform small models trained from scratch when data is abundant? These questions need to be answered on standardized benchmarks.

### Loss & Training
Baseline experiments utilize L1 loss (MAE) or L2 loss (MSE) for model training. All models are trained under the same data split and training settings (PyTorch Lightning + Hydra configuration). ImageNet pre-trained weights are used for initialization, and the learning rate is decayed using a cosine scheduler.

## Key Experimental Results

### Main Results (Canopy Height Estimation)

| Model | Backbone | MAE (m)↓ | RMSE (m)↓ | $R^2$ ↑ |
|------|---------|----------|-----------|------|
| UNet | ResNet-50 | ~2.8 | ~4.5 | ~0.72 |
| PVTv2 | PVTv2-B2 | ~2.5 | ~4.0 | ~0.78 |
| ViT-Small | DINOv2 pre-trained | ~2.6 | ~4.1 | ~0.76 |
| Tolan et al. | SSL pre-trained | ~2.7 | ~4.3 | ~0.74 |
| Swin-UNet | Swin-T | ~2.5 | ~4.1 | ~0.77 |

### Change Detection (Open-Canopy-$\Delta$)

| Method | Change MAE (m)↓ | Note |
|------|-------------|------|
| Difference between two separate predictions | ~3.5 | Severe error accumulation |
| End-to-end dual-input model | ~3.0 | Direct learning of change mapping |
| Siamese network | ~2.9 | Dual-branch structure performs slightly better |

### Key Findings
- **Transformer > CNN, but with limited margin**: PVTv2 and Swin show an approximate 10% improvement over traditional UNet, but the gap is not as prominent as in natural image tasks, indicating that local texture information remains crucial in remote sensing tasks.
- **Change detection is highly challenging**: Even the best model yields a change detection MAE of around $3\text{m}$, which is acceptable for precise logging detection (where height changes are typically $>10\text{m}$) but completely inadequate for detecting slow natural growth (annual change $<1\text{m}$).
- **Data scale is king**: The massive data scale of Open-Canopy ($87,000\text{ km}^2$) significantly outperforms models trained on small datasets, and the marginal gains of pre-trained weights diminish as the volume of training data increases.
- **Spatial generalization is a key bottleneck**: Model performance significantly drops outside training areas, especially in regions with substantial differences in vegetation types.

## Highlights & Insights
- **Value of the first large-scale open benchmark**: In high-resolution remote sensing, the lack of open datasets has long been the greatest obstacle to reproducible research. Open-Canopy leverages France's open data policy to fill this critical gap, and its data acquisition and processing pipelines can be borrowed by other countries.
- **Pioneering nature of the change detection task**: Open-Canopy-$\Delta$ standardizes temporal canopy change detection as a computer vision benchmark task for the first time, providing a clear evaluation target for future methodological research.
- **Data scale of 360GB**: This is considered extremely large-scale for remote sensing benchmark datasets, sufficient for training and evaluating large models, thereby reducing reliance on pre-training.

## Limitations & Future Work
- This study only covers France, with vegetation types biased toward temperate forests; the generalization ability to scenes like tropical rainforests and arid regions remains unknown.
- SPOT satellite imagery only has 4 bands (RGBNIR), lacking spectral bands such as Short-Wave Infrared (SWIR) that are valuable for vegetation analysis.
- The accuracy for the change detection task remains low, requiring future exploration of more robust temporal modeling methods (e.g., transformers with temporal attention).
- The LiDAR ground truth itself contains some noise (particularly in the processing of unclassified points), which may limit the upper bound of model performance.
- Future works could extend the approach to other countries (e.g., utilizing free academic data from Pléiades Neo) to build a global-level benchmark.

## Related Work & Insights
- **vs Global Canopy Height (Meta AI)**: Meta constructs a global $10\text{m}$ resolution canopy map using GEDI + Sentinel-2. Open-Canopy improves the resolution by ~7 times ($1.5\text{m}$ vs $10\text{m}$), though its geographic coverage is limited to France.
- **vs Tolan et al.**: Tolan et al. perform canopy height estimation using commercial satellite data with self-supervised pre-training. Open-Canopy demonstrates that with sufficiently large open datasets, simple supervised learning can yield comparable or even superior performance.
- **vs FLAIR Dataset**: FLAIR is a land cover classification benchmark in France. Open-Canopy serves as a complement to it for the canopy height regression task, with overlapping coverage areas.

## Rating
- Novelty: ⭐⭐⭐ The core contribution lies in the dataset rather than the methodology, but it fills an important data gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The model benchmarking is highly comprehensive, covering multiple architectures and two tasks.
- Writing Quality: ⭐⭐⭐⭐ The dataset is described in detail, and the experimental setup is transparent and reproducible.
- Value: ⭐⭐⭐⭐ The long-term value as a benchmark dataset is high, with a user-friendly open data policy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation](prompting_depth_anything_for_4k_resolution_accurate_metric_depth_estimation.md)
- [\[CVPR 2025\] ProtoOcc: 3D Occupancy Prediction with Low-Resolution Queries via Prototype-aware View Transformation](3d_occupancy_prediction_with_low-resolution_queries_via_prototype-aware_view_tra.md)
- [\[CVPR 2025\] O3N: Omnidirectional Open-Vocabulary Occupancy Prediction](o3n_omnidirectional_open-vocabulary_occupancy_prediction.md)
- [\[CVPR 2025\] DrivingSphere: Building a High-fidelity 4D World for Closed-loop Simulation](drivingsphere_building_a_high-fidelity_4d_world_for_closed-loop_simulation.md)
- [\[CVPR 2025\] RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion](racformer_towards_high-quality_3d_object_detection_via_query-based_radar-camera_.md)

</div>

<!-- RELATED:END -->
