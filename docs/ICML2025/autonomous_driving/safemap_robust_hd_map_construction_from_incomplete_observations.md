---
title: >-
  [Paper Note] SafeMap: Robust HD Map Construction from Incomplete Observations
description: >-
  [ICML 2025][Autonomous Driving][HD Map Construction] SafeMap proposes a plug-and-play robust framework for HD map construction. By utilizing two modules, Gaussian-based Perspective View Reconstruction (G-PVR) and Distillation-based BEV Correction (D-BEVC), it accurately constructs vectorized HD maps even under incomplete observations where camera views are missing.
tags:
  - "ICML 2025"
  - "Autonomous Driving"
  - "HD Map Construction"
  - "BEV Perception"
  - "Sensor Failure Robustness"
  - "View Reconstruction"
  - "knowledge distillation"
date: 2026-05-08
content_hash: 6c3082a6c1a4fa12
---

# SafeMap: Robust HD Map Construction from Incomplete Observations

**Conference**: ICML 2025  
**arXiv**: [2507.00861](https://arxiv.org/abs/2507.00861)  
**Code**: None  
**Area**: Autonomous Driving  
**Keywords**: HD Map Construction, BEV Perception, Sensor Failure Robustness, View Reconstruction, knowledge distillation

## TL;DR

SafeMap proposes a plug-and-play robust framework for HD map construction. By utilizing two modules, Gaussian-based Perspective View Reconstruction (G-PVR) and Distillation-based BEV Correction (D-BEVC), it accurately constructs vectorized HD maps even under incomplete observations where camera views are missing.

## Background & Motivation

Online High-Definition (HD) map construction is a critical task in autonomous driving, providing precise static environmental information for vehicle planning and navigation. Current mainstream methods rely on complete inputs from multi-view cameras. However, in real-world driving scenarios, cameras may suffer from occlusions, malfunctions, or physical damage, leading to missing image data from certain views.

Existing methods face the following challenges:

**Vulnerability Exposure**: MapBench evaluations demonstrate that sensor failures significantly degrade HD Map model performance, posing threats to traffic safety.

**Limitations of Prior Work**: Methods like MetaBEV and UniBEV address the sensor failure issue for 3D object detection but still rely on complete multi-view images. M-BEV only utilizes local crops of neighboring views for recovery, which requires a predefined cropping ratio and fails to fully utilize information from all available views.

**Research Gap**: Robust methods for HD Map construction under incomplete observations have not been fully explored, despite map construction being highly dependent on static environmental data captured by surrounding cameras.

SafeMap is the **first** robust framework specifically designed for HD Map construction under incomplete multi-view camera data.

## Method

### Overall Architecture

SafeMap is built upon the MapTR framework and consists of four core components:

1. **Map Encoder**: 2D feature extractor + PV-to-BEV transformation module
2. **G-PVR Module** (Gaussian-based Perspective View Reconstruction): Perspective view feature reconstruction based on Gaussian sampling
3. **D-BEVC Module** (Distillation-based BEV Correction): Distillation-based BEV feature correction
4. **Map Decoder**: MapTR-based decoder and prediction heads

During training, a **Random View Masking (RVM)** mechanism is used to randomly mask 2D image features of a specific view to simulate camera failures, followed by information recovery via the G-PVR and D-BEVC modules. During testing, the reconstruction module predicts the features of the missing views.

### Key Designs

#### G-PVR: Gaussian-based Perspective View Reconstruction Module

This module represents the core innovation of the paper, addressing the problem of "how to reconstruct missing views from multiple available views."

**Core Idea**: Different views contribute differently to the reconstruction of the missing view. Neighboring views contain the most relevant information and should receive higher weights, while distant views (such as the opposite view) contain less information and should be down-weighted.

**Specific Steps**:

1. **Perspective Panoramic View Concatenation**: Starting from the left and right neighboring views of the missing view, all available frames are arranged sequentially by spatial distance and concatenated into a perspective panoramic view $F_{PPV} = \text{Concat}(F_{PV}^a) \in \mathbb{R}^{H \times N_a W \times C}$

2. **Gaussian Reference Point Generation**: Generate reference points following a Gaussian distribution on the perspective panoramic view:

    - Horizontal direction: $p_x \sim \mathcal{N}(N_a W / 2, \sigma^2)$ (centered on the middle to concentrate sampling points on the neighboring view regions)
    - Vertical direction: $p_y \sim \mathcal{U}(0, H)$ (uniformly distributed to cover the full height)

3. **Deformable Attention Reconstruction**: Utilize a learnable query $V$ and an offset network $\theta_{\text{offset}}$ to generate sampling offsets, extracting sampled features at reference point locations via a deformable attention mechanism:
    $\hat{k} = \hat{x} W_k, \quad \hat{v} = \hat{x} W_v$
    $\Delta p = \theta_{\text{offset}}(V), \quad \hat{x} = \phi(F_{PV}^a; p + \Delta p)$

4. **MAE-style Transformer Reconstruction**: Reconstruct missing view features using MAE-like Transformer blocks.

**Advantages of G-PVR over Local PVR (M-BEV method)**:
- Utilizes all available views instead of only the two adjacent views
- Eliminates the need for predefined cropping ratios
- Naturally encodes the view importance prior via a Gaussian distribution
- Flexibly adapts to varying numbers of input views and spatial configurations

#### D-BEVC: Distillation-based BEV Correction Module

Following PV-level reconstruction, further calibration in the global BEV feature space is necessary. D-BEVC leverages the complete-observation BEV features $F_{BEV}^{com}$ as a supervision signal, calibrating the incomplete-observation BEV features $F_{BEV}^{incom}$ using MSE loss:

$$\mathcal{L}_{Cor} = \text{MSE}(F_{BEV}^{com}, F_{BEV}^{incom})$$

This module allows the incomplete-observation BEV features to implicitly benefit from the complete-observation omnidirectional BEV features during training.

### Loss & Training

The **overall optimization objective** consists of three parts:

$$L = L_{\text{map}} + \lambda_1 L_{\text{Rec}} + \lambda_2 L_{\text{Cor}}$$

| Loss Term | Definition | Weight |
|--------|------|------|
| $L_{\text{map}}$ | Map construction loss (classification + point-to-point + edge direction) | 1.0 |
| $L_{\text{Rec}}$ | PV reconstruction loss: $\|F_{PV}^{com} - F_{PV}^{incom}\|$ | $\lambda_1 = 0.05$ |
| $L_{\text{Cor}}$ | BEV correction loss: $\text{MSE}(F_{BEV}^{com}, F_{BEV}^{incom})$ | $\lambda_2 = 5$ |

**Training Strategy**:
- Optimizer: AdamW, learning rate $4.2 \times 10^{-4}$
- Training: Fine-tuned for 8 epochs on nuScenes and 2 epochs on Argoverse2
- Batch size: 4 for nuScenes, 6 for Argoverse2
- Gaussian variance: $\sigma = 3$
- Maximum of 100 map elements per frame, with 20 points per element
- BEV grid size 0.75m, Transformer decoder with 2 layers
- A random single-view RGB image is dropped per frame during training

## Key Experimental Results

### Main Results

Experiments are evaluated on nuScenes (6 views) and Argoverse2 (7 views) datasets, using mAP based on Chamfer distance (thresholds at 0.5m/1.0m/1.5m) as the assessment metric.

| Dataset | Baseline Model | Scenario | mAP Gain | Explanation |
|--------|----------|------|----------|------|
| nuScenes | HIMap | Various missing views | +2.4% ~ +18.2% | 6 single-view missing scenarios |
| Argoverse2 | MapTR | Complete observations | +1.0% | Performance gain also achieved under complete views |
| Argoverse2 | MapTR | Missing front view | +4.1% | Front view has the most significant impact |
| nuScenes | MapTR | Sensor corruptions | mRR +1.9%, mCE +9.4% | 8 real-world corruption scenarios |
| nuScenes | HIMap | Sensor corruptions | mRR +6.2%, mCE +16.8% | MapBench evaluation |

### Ablation Study

| Configuration | Key Metric | Description |
|------|----------|------|
| Baseline (w/o reconstruction module) | Reference | Original MapTR model only |
| + G-PVR only | Significant gain | PV-level reconstruction is effective |
| + D-BEVC only | Significant gain | BEV-level correction is effective |
| + G-PVR + D-BEVC (Full SafeMap) | **Best** | The two modules are complementary |
| Mean-PVR (averaging all available views) | Below baseline | Simple averaging is ineffective |
| MAE-PVR (MAE + masked token) | Outperforms Mean | But weaker than G-PVR |
| Standard-PVR (uniform reference points) | Outperforms MAE | But weaker than G-PVR |
| Gaussian-PVR (Gaussian reference points) | **Optimal** | Prior knowledge is effective |
| D-BEVC w/ L1 | Suboptimal | Manhattan distance |
| D-BEVC w/ L2 | **Optimal** | Euclidean distance (default) |
| D-BEVC w/ KL | Weakest | KL divergence performs poorly |

### Key Findings

1. **Front/Rear Views are Most Critical**: Missing the front view (CAM_FRONT) or rear view (CAM_BACK) has the heaviest impact on performance, as these views contain the most critical map elements.
2. **Negligible Parameter Count Increase**: SafeMap adds only 0.4MB to 3.6MB of parameters, having almost no effect on inference speed and GPU memory consumption.
3. **Tolerance to Multi-View Loss**: As the number of missing views increases (from 1 to 5), performance gradually degrades. SafeMap consistently and significantly outperforms MapTR under 1 to 5 missing views, exhibiting a much slower performance decay.
4. **Hyperparameter Robustness**: Performance remains robust across the ranges of $\lambda_1 \in [0.01, 0.09]$, and $\lambda_2$ & $\sigma \in [1, 5]$.

## Highlights & Insights

1. **Gaussian Sampling Encodings for View Prior**: The intuition that adjacent views are more critical is elegantly encoded into the distribution of reference points via a Gaussian distribution, which is more natural and effective than uniform sampling or local cropping.
2. **Dual-level Reconstruction Strategy**: PV-level reconstruction (G-PVR) handles local feature recovery, while BEV-level correction (D-BEVC) ensures global consistency, making them highly complementary.
3. **Plug-and-Play Design**: Can be directly integrated into existing frameworks like MapTR and HIMap without substantial architecture adjustments.
4. **Performance Gain Under Complete Scenarios**: Although primarily targeted at incomplete observations, SafeMap also improves performance under complete observations, showing that the reconstruction module enhances the model's feature representation ability.
5. **First Systematic Study**: The first robust framework specifically targeting HD Map construction under incomplete multi-view data.

## Limitations & Future Work

1. **Limitation to Camera Modality only**: Sensor failure scenarios under multi-sensor fusion (e.g., LiDAR-Camera fusion) are not investigated.
2. **Single-frame Reconstruction**: Temporal information is not exploited; incorporating history frames could potentially further improve reconstruction quality.
3. **Requirement of Complete Data During Training**: D-BEVC relies on complete observations as teacher signals, requiring the training dataset to contain complete multi-view images.
4. **Primary Evaluation Focuses on Single-View Loss**: Although multi-view failure experiments were conducted, major comparative experiments remain concentrated on single-view loss scenarios.
5. **Assumed Gaussian Distribution**: Applying a fixed variance $\sigma$ uniformly across all views might not be optimal, and an adaptive variance scheme could yield better results.

## Related Work & Insights

- **MapTR/MapTRv2**: Baseline methods for SafeMap, serving as pioneering works in end-to-end vectorized HD map construction.
- **HIMap**: An HD map construction method based on hybrid representation learning, on which SafeMap's effectiveness is also validated.
- **M-BEV**: A masked view reconstruction method in 3D object detection. SafeMap's G-PVR represents a comprehensive improvement over its Local PVR.
- **MapBench**: A robustness evaluation benchmark for HD map models under sensor corruptions.
- **MetaBEV/UniBEV**: BEV perception methods addressing sensor failures, which are, however, not applicable to HD map scenarios.

**Insight**: The idea of using Gaussian sampling as a spatial prior can be extended to other multi-view fusion tasks (e.g., 3D detection, occupancy prediction). Similarly, the distillation-based calibration strategy can be applied to other scenarios requiring robust handling of missing inputs.

## Rating

| Dimension | Rating (1-5) | Description |
|------|-----------|------|
| Novelty | 4 | Gaussian sampling view prior + dual-level reconstruction is a novel combination. |
| Value | 5 | Plug-and-play, minimal parameter overhead, high practical deployment value. |
| Experimental Thoroughness | 5 | Evaluated on two datasets, multiple baselines, extensive ablations, and hyperparameter sensitivity analysis. |
| Writing Quality | 4 | Clear structure, well-articulated motivation. |
| Overall Evaluation | 4.5 | High utility, solid experimental backup, solid ICML work. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Uncertainty-Instructed Structure Injection for Generalizable HD Map Construction](../../CVPR2025/autonomous_driving/uncertainty-instructed_structure_injection_for_generalizable_hd_map_construction.md)
- [\[ECCV 2024\] Stream Query Denoising for Vectorized HD-Map Construction](../../ECCV2024/autonomous_driving/stream_query_denoising_for_vectorized_hd-map_construction.md)
- [\[ECCV 2024\] MapDistill: Boosting Efficient Camera-based HD Map Construction via Camera-LiDAR Fusion Model Distillation](../../ECCV2024/autonomous_driving/mapdistill_boosting_efficient_camera-based_hd_map_construction_via_camera-lidar_.md)
- [\[ICCV 2025\] DAMap: Distance-aware MapNet for High Quality HD Map Construction](../../ICCV2025/autonomous_driving/damap_distance-aware_mapnet_for_high_quality_hd_map_construction.md)
- [\[CVPR 2025\] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](../../CVPR2025/autonomous_driving/mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)

</div>

<!-- RELATED:END -->
