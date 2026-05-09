---
title: >-
  [Paper Note] RTMap: Real-Time Recursive Mapping with Change Detection and Localization
description: >-
  [ICCV 2025][Autonomous Driving][HD Map] RTMap is proposed as the first end-to-end framework that simultaneously addresses three core challenges in multi-traversal online HD map construction: prior-map-based localization, road structure change detection, and probabilistic crowdsourced map fusion. It achieves improvements in both map quality and localization accuracy on TbV and nuScenes.
tags:
  - ICCV 2025
  - Autonomous Driving
  - HD Map
  - Crowdsourced Mapping
  - Change Detection
  - Map Localization
  - Uncertainty Modeling
  - Multi-Traversal Fusion
date: 2026-05-08
content_hash: e74ababecbdfe386
---

# RTMap: Real-Time Recursive Mapping with Change Detection and Localization

**Conference**: ICCV 2025
**arXiv**: [2507.00980](https://arxiv.org/abs/2507.00980)
**Code**: [github.com/CN-ADLab/RTMap](https://github.com/CN-ADLab/RTMap)
**Area**: Autonomous Driving / Online HD Map Construction
**Keywords**: HD Map, Crowdsourced Mapping, Change Detection, Map Localization, Uncertainty Modeling, Multi-Traversal Fusion

## TL;DR

RTMap is proposed as the first end-to-end framework that simultaneously addresses three core challenges in multi-traversal online HD map construction: prior-map-based localization, road structure change detection, and probabilistic crowdsourced map fusion. It achieves improvements in both map quality and localization accuracy on TbV and nuScenes.

## Background & Motivation

Online HD map construction has become a mainstream paradigm for autonomous driving, enabling vehicles to generate HD maps in real time during operation. However, existing methods suffer from critical limitations:

**Limitations of single-traversal approaches**: Methods such as MapTR/MapTRv2 treat mapping as a single-pass process and fail to leverage the rich contextual information accumulated over multiple visits to the same location. Occlusions and perceptual inaccuracies constrain the quality of resulting maps.

**Underexplored crowdsourced enhancement**: Multi-agent, multi-traversal crowdsourcing can extend the perceptual range, resolve occlusions, and improve map accuracy, but requires two key capabilities:
   - **Precise localization**: Determining the vehicle's accurate position within a prior map to align current observations.
   - **Change detection**: Detecting road structure changes (e.g., lane modifications, construction zones) to maintain map freshness.

**Fragmented existing solutions**: Localization and change detection have been studied independently, yet both fundamentally address retrieval, correspondence, and discrepancy analysis between map elements across traversals, making a unified framework a more principled choice.

RTMap is the first to unify prior-assisted online HD mapping, map localization, and change detection within a single end-to-end framework, introducing uncertainty modeling to simultaneously improve localization accuracy and crowdsourced fusion quality.

## Method

### Overall Architecture

RTMap comprises two modules: an onboard (vehicle-side) module and an offline (cloud-side) module:

- **Onboard model** $\mathrm{RTMapModel}(\mathbf{I}_t, \mathcal{M}_{t-1})$: Takes current-frame multi-sensor images and a prior map as input, and outputs map elements $\mathbf{M}_t$, uncertainties $\mathbf{U}_t$, correspondences $\mathbf{D}_t$, and an end-to-end pose $\mathbf{T}_t^{\mathbb{E}}$.
- **Localizer** $\mathrm{Localize}(\cdot)$: Performs explicit optimization using derived correspondences to solve for pose $\mathbf{T}_t^{\mathbb{R}}$.
- **Cloud-side crowdsourcing** $\mathrm{CSrc}(\cdot)$: Asynchronously fuses multi-traversal observations to update the global prior map.

GPS provides a meter-level initial pose $\mathbf{T}_t^0$, from which a local map $\mathcal{M}_{t-1}$ is cropped from the global prior map and used as input to the onboard module.

### Key Designs

#### Hybrid Queries

Three types of queries are designed to jointly handle localization, mapping, and change detection:

$$\mathbf{Q}_{\mathrm{hybrid}} = \{\mathbf{Q}_{\mathrm{map}}, \mathbf{Q}_{\mathrm{fake}}, \mathbf{Q}_{\mathrm{new}}\} + \mathbf{Q}_{\mathrm{hie}}$$

- $\mathbf{Q}_{\mathrm{prior}}$: Encoded from the prior map; each prior query uses a unified fixed-point representation (first 2 dimensions for XOY coordinates, remaining $N$ dimensions for class one-hot encoding).
- $\mathbf{Q}_{\mathrm{new}}$: Remaining queries are zero-padded and responsible for detecting newly observed elements in the current traversal.
- After training, $\mathbf{Q}_{\mathrm{prior}}$ further differentiates into $\mathbf{Q}_{\mathrm{map}}$ (matched reliable elements used for localization) and $\mathbf{Q}_{\mathrm{fake}}$ (outdated elements marked as changes).
- $\mathbf{Q}_{\mathrm{hie}}$ denotes the hierarchical query embeddings from MapTR.

#### Existence-Aware Matching

**During training**:
- $\mathbf{Q}_{\mathrm{map}}$ uses **pre-assignment** to their corresponding existing map elements.
- $\mathbf{Q}_{\mathrm{new}}$ uses standard **Hungarian matching** to pair with remaining map elements.
- $\mathbf{Q}_{\mathrm{fake}}$ receives no pre-assignment, as its corresponding map elements no longer exist.
- Different query types exhibit distinct behaviors across decoder layers: map queries progressively move toward correct positions, fake queries exhibit unstable reference points, and new queries detect newly added elements.

**During inference**:
- Without GT annotations, $\mathbf{Q}_{\mathrm{map}}$ and $\mathbf{Q}_{\mathrm{fake}}$ are intermixed within $\mathbf{Q}_{\mathrm{prior}}$.
- Since fake queries were not pre-assigned during training, their **classification confidence is significantly lower** than that of map queries.
- A confidence threshold therefore effectively distinguishes between the two types.

### Loss & Training

**Composite geometric loss** (for map element vertices):

$$\mathbf{L}_{\mathrm{pts}} = \lambda_1 \cdot \mathbf{L}_{\mathrm{nll}} + \lambda_2 \cdot \mathbf{L}_{\mathrm{mht}}$$

where the NLL loss models a Laplace distribution uncertainty for each vertex:

$$\mathbf{L}_{\mathrm{nll}} = \sum_{v=1}^{V}\sum_{k=1}^{2}\left(\log(2\sigma_v^k) + \frac{|\mathbf{m}_v^k - \mu_v^k|}{\sigma_v^k}\right)$$

- $\mu_v^k, \sigma_v^k$: position and scale parameters for the $k$-th dimension of the $v$-th vertex.
- $\mathbf{L}_{\mathrm{mht}}$: Manhattan distance loss retained from MapTRv2.
- Hyperparameters: $\lambda_1=0.03$, $\lambda_2=5.0$.

**Pose auxiliary loss**: The features of map queries in the decoder output are passed through a shared MLP with max-pooling to predict a delta pose, supervised via smooth L1 loss.

#### MAP Localization and Crowdsourcing

**Optimization-based localization**: Using matched correspondences $\mathbf{D}_t$ and uncertainties $\mathbf{U}_t$, maximum a posteriori (MAP) estimation is performed on point-to-point residuals:

$$\min_{\mathbf{T}^{\mathbb{R}}} \sum_{\mathbf{D}_t} -\log\left(\exp\left(-\frac{1}{2}\|\mathbf{T}^{\mathbb{R}}\cdot\mathbf{m}_t^i - m_{t-1}^i\|_{\mathbf{g}_t^i}^2\right)\right)$$

where the covariance is modeled as a Gaussian mixture $\mathbf{g}_t^i = \mathbf{u}_t^i \oplus u_{t-1}^i$, solved via the Levenberg–Marquardt algorithm.

**Probabilistic crowdsourcing**: On the cloud side, a union-find algorithm constructs a position solver:

$$\min_{m_t} \sum \frac{1}{2}\|m_t - \mathbf{T}_t^j \cdot \hat{\mathbf{m}}_t^j\|^2_{\mathbf{u}_t^j} + \frac{1}{2}\|m_t - m_{t-1}\|^2_{u_{t-1}}$$

Topological structure is determined via Hungarian voting, and Gaussian mixture models continuously refine the probability density.

## Key Experimental Results

### Datasets and Setup

- **TbV**: 200+ scenes containing real road changes (lane topology, road boundaries, crosswalks), specifically designed for change detection.
- **nuScenes**: 1,000 driving scenes with annotated HD maps, used for localization evaluation.
- A more realistic evaluation is introduced via pose perturbations: lateral $\mathcal{N}(0, 0.75^2)$ m, longitudinal $\mathcal{N}(0, 1.5^2)$ m, heading $\mathcal{N}(0, 0.85^2)$ °.

### Main Results: Crowdsourced Map Quality (TbV)

| Method | Scene | Traversals | Cycle(%) | Ped.(%) | Div.(%) | Avg mAP(%) |
|--------|-------|------------|----------|---------|---------|------------|
| MapTRv2 | Straight | Ave. | 31.7 | 42.0 | 37.3 | 37.1 |
| HRMapNet | Straight | Ave. | 34.2 | 43.7 | 39.8 | 39.2 |
| MapTracker | Straight | Ave. | 35.7 | 44.6 | 39.6 | 39.9 |
| RTMap(w/o U) | Straight | 2 | 28.6 | 60.5 | 31.7 | 40.2 |
| **RTMap** | **Straight** | **2** | **32.7** | **68.6** | **35.5** | **45.6** |
| RTMap(w/o U) | Straight | 3 | 35.7 | 74.4 | 42.0 | 50.7 |
| **RTMap** | **Straight** | **3** | **40.9** | **84.3** | **47.6** | **57.6** |
| MapTRv2 | Turning | Ave. | 28.2 | 31.6 | 18.3 | 26.0 |
| **RTMap** | **Turning** | **3** | **42.3** | **85.2** | **38.8** | **55.4** |

**Key Findings**:
- Existing methods suffer substantial accuracy degradation under pose noise, while RTMap progressively improves through crowdsourcing.
- After 3 traversals, RTMap achieves a mAP of 57.6% on straight-road scenes, substantially outperforming MapTracker at 39.9%.
- The probabilistic density $\mathbf{U}$ is critical to crowdsourcing, contributing approximately 7% mAP improvement.

### Ablation Study: Localization Accuracy

| Method | Lat. Mean(m) | Lat. 90th(m) | Lon. Mean(m) | Heading Mean(°) |
|--------|-------------|-------------|-------------|-----------------|
| RTMap ($\mathbf{Q}_{\mathrm{prior}}$) | 0.163 | 0.318 | 0.686 | 0.332 |
| **RTMap ($\mathbf{Q}_{\mathrm{map}}$)** | **0.125** | **0.256** | **0.633** | **0.317** |

| Method (nuScenes) | Lat. Mean(m) | Lon. Mean(m) | Heading Mean(°) |
|-------------------|-------------|-------------|-----------------|
| RTMap ($\mathbf{T}^{\mathbb{E}}$) | 0.142 | 0.589 | 0.521 |
| **RTMap ($\mathbf{T}^{\mathbb{R}}$)** | **0.121** | **0.586** | **0.368** |
| $\mathbf{T}^{\mathbb{R}}$+$\mathbf{L}_{\mathrm{pose}}$ | 0.122 | 0.590 | 0.371 |
| $\mathbf{T}^{\mathbb{R}}$+$\mathbf{L}_{\mathrm{pts}}$ | **0.118** | 0.609 | 0.376 |

**Key Findings**:
- Distinguishing $\mathbf{Q}_{\mathrm{map}}$ from $\mathbf{Q}_{\mathrm{fake}}$ via hybrid queries effectively improves localization by excluding interference from outdated elements.
- Explicit optimization $\mathbf{T}^{\mathbb{R}}$ outperforms end-to-end regression $\mathbf{T}^{\mathbb{E}}$, especially in heading angle estimation.
- The NLL loss $\mathbf{L}_{\mathrm{pts}}$ and pose auxiliary loss $\mathbf{L}_{\mathrm{pose}}$ provide complementary contributions.

### Change Detection (TbV)

| Method | Acc_c(%) | Acc_r(%) | mAcc(%) |
|--------|----------|----------|---------|
| TbV baseline | 40.0 | 68.2 | 54.1 |
| **RTMap** | **48.9** | 66.0 | **57.4** |

RTMap achieves higher recall on changed categories; in practice, higher recall is essential for safety assurance.

## Highlights & Insights

1. **First unified framework**: Localization, change detection, and mapping are integrated into a single end-to-end model, enabling mutual reinforcement rather than independent optimization.
2. **Elegant hybrid query design**: Query differentiation naturally separates existing, outdated, and newly added map elements; at inference time, confidence scores alone are sufficient to distinguish them.
3. **Multi-faceted value of uncertainty**: Probabilistic density simultaneously serves localization (Mahalanobis distance weighting) and crowdsourced fusion (noise-aware merging).
4. **Self-evolving memory**: The prior map acts as a persistent "memory" that continuously self-improves with each traversal; quality after three passes substantially exceeds single-traversal methods.

## Limitations & Future Work

1. The current system relies solely on cameras without fusing LiDAR or other multimodal sensors.
2. Validation is limited to structured roads and has not been extended to unstructured surfaces or more complex urban scenarios.
3. The crowdsourcing mechanism progressively filters transient observations via MAP voting, but does not yet distinguish permanent structural changes from temporary occlusions.
4. Training uses artificially perturbed prior maps due to the lack of multi-traversal data in public datasets, which introduces a gap relative to real crowdsourcing scenarios.

## Related Work & Insights

- **Online HD mapping**: Single-traversal methods including HDMapNet, MapTR/v2, and MapTracker; prior-utilizing methods including PrevPredMap and HRMapNet.
- **Prior-assisted methods**: P-MapNet (SD prior + HD prior), PriorDrive (unified vector encoder), SMERF, U-BEV.
- **Change detection**: ExelMap (element-level insertion/deletion detection).
- **Map localization**: BEV-Locator and EgoVM (centimeter-level localization), both treating localization as a standalone task.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ (First end-to-end framework unifying localization, change detection, and crowdsourced mapping)
- **Technical Depth**: ⭐⭐⭐⭐⭐ (Multiple sophisticated designs: hybrid queries, existence-aware matching, probabilistic modeling, MAP optimization)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Comprehensive multi-dataset, multi-task evaluation; lacks large-scale validation on real multi-traversal data)
- **Practical Value**: ⭐⭐⭐⭐⭐ (Directly targets industry-level crowdsourced HD mapping for autonomous driving)
- **Overall Recommendation**: ⭐⭐⭐⭐⭐ (A highly systematic contribution with strong potential to advance online HD mapping into the crowdsourcing era)

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](gs-livm_real-time_photo-realistic_lidar-inertial-visual_mapping_with_gaussian_sp.md)
- [\[NeurIPS 2025\] ChronoGraph: A Real-World Graph-Based Multivariate Time Series Dataset](../../NeurIPS2025/autonomous_driving/chronograph_a_real-world_graph-based_multivariate_time_series_dataset.md)
- [\[ICCV 2025\] Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](adaptive_dual_uncertainty_optimization_boosting_monocular_3d_object_detection_un.md)
- [\[ICCV 2025\] Splat-LOAM: Gaussian Splatting LiDAR Odometry and Mapping](splat-loam_gaussian_splatting_lidar_odometry_and_mapping.md)
- [\[ICCV 2025\] LookOut: Real-World Humanoid Egocentric Navigation](lookout_real-world_humanoid_egocentric_navigation.md)

<!-- RELATED:END -->
