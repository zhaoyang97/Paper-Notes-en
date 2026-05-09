---
title: >-
  [Paper Note] Robust 3D Object Detection using Probabilistic Point Clouds from Single-Photon LiDARs
description: >-
  [ICCV2025][Autonomous Driving][Probabilistic Point Cloud] This paper proposes the Probabilistic Point Cloud (PPC) representation, which attaches measurement confidence derived from raw single-photon LiDAR timing histograms as a probability attribute to each 3D point. Combined with a lightweight NPD filter and FPPS sampling strategy, PPC enables robust 3D object detection under low signal-to-background ratio (SBR) conditions, substantially outperforming point cloud denoising baselines on SUN RGB-D and KITTI with negligible computational overhead.
tags:
  - ICCV2025
  - Autonomous Driving
  - Probabilistic Point Cloud
  - Single-Photon LiDAR
  - 3D Object Detection
  - SPAD
  - Sensor Uncertainty Propagation
  - Noise Robustness
date: 2026-05-08
content_hash: 324746737abd6014
---

# Robust 3D Object Detection using Probabilistic Point Clouds from Single-Photon LiDARs

**Conference**: ICCV2025  
**arXiv**: [2508.00169](https://arxiv.org/abs/2508.00169)  
**Code**: [Project Page](https://bhavyagoyal.github.io/ppc)  
**Area**: Autonomous Driving  
**Keywords**: Probabilistic Point Cloud, Single-Photon LiDAR, 3D Object Detection, SPAD, Sensor Uncertainty Propagation, Noise Robustness

## TL;DR
This paper proposes the Probabilistic Point Cloud (PPC) representation, which attaches measurement confidence derived from raw single-photon LiDAR timing histograms as a probability attribute to each 3D point. Combined with a lightweight NPD filter and FPPS sampling strategy, PPC enables robust 3D object detection under low signal-to-background ratio (SBR) conditions, substantially outperforming point cloud denoising baselines on SUN RGB-D and KITTI with negligible computational overhead.

## Background & Motivation

**Background**: Modern LiDARs—particularly SPAD single-photon LiDARs—estimate scene depth from timing histograms. Under ideal conditions, histograms exhibit clear peaks, yielding accurate depth estimates. However, under challenging conditions such as distant objects, low-reflectance surfaces, and strong ambient light, signal peaks are buried in background noise, causing severe depth estimation degradation and resulting in sparse, noise-corrupted point clouds.

**Limitations of Prior Work**: Conventional pipelines apply threshold filtering to raw histograms, either over-filtering and discarding genuine scene points (e.g., distant pedestrians) or retaining excessive noise. The fundamental issue is that **uncertainty information is entirely discarded during the conversion from raw sensor data to point clouds**. Existing point cloud datasets (e.g., KITTI) consist of pre-filtered "clean" data, which obscures this problem.

**Key Challenge**: Existing solutions remain insufficient:
   - Point cloud denoising networks (PointCleanNet / Score-Denoising, etc.): assume locally isotropic Gaussian noise, whereas LiDAR noise is anisotropic (distributed along camera ray directions), limiting denoising effectiveness.
   - Histogram denoising (3D-CNN, etc.): requires full histogram readout, incurring prohibitive bandwidth costs (~10s GB/s), unsuitable for real-time applications.
   - Both categories introduce significant computational overhead.

**Core Idea**: Raw SPAD histograms encode rich scene information, yet conventional pipelines use only peak positions for depth estimation, discarding a large amount of useful information. This paper proposes **propagating uncertainty into downstream inference rather than discarding it**.

## Method

### Probabilistic Point Cloud (PPC) Representation
- For the timing histogram at each LiDAR pixel $(i,j)$, the point probability is defined as:

$$Pr(p^{ij}) = \frac{h_{i,j}[m]}{\sum_{n=1}^{N} h_{i,j}[n]}, \quad m = \arg\max_n h_{i,j}[n]$$

- Interpretation: the ratio of photon counts in the peak bin to total photon counts. This approaches 1 under high SBR and converges to the uniform distribution value $1/N$ under low SBR.
- This probability is appended as an additional attribute to each 3D point, forming the Probabilistic Point Cloud (PPC).
- **Computationally minimal**: requires only peak finding and normalization, executable on-sensor.

### NPD Filtering (Neighbor Probability Density)
- **Design Motivation**: Noise points tend to have both low probability and low spatial density, whereas genuine scene points—even with low probability—exhibit high local density due to neighboring points on the same surface.
- NPD score computation:

$$NPD(p_i) = \frac{\sum_{p_j \in \mathcal{BQ}_{L,r}(p_i)} Pr(p_j)}{L}$$

- $\mathcal{BQ}_{L,r}$: ball query returning at most $L$ neighbors within radius $r$.
- Normalization design: dense points with more than $L$ neighbors use the average probability; sparse points with fewer than $L$ neighbors are additionally penalized (denominator remains $L$).
- A threshold of $\alpha = 0.003$ suffices to remove the majority of noise points while retaining nearly all genuine points.
- **Assumption-free**: purely statistical, requiring no object or surface priors.

### FPPS (Farthest Probable Point Sampling)
- **Problem**: FPS (Farthest Point Sampling) used in PointNet++ and similar networks fails under noisy conditions—noise points located far from objects are preferentially sampled, yielding noisy keypoints.
- **Solution**: A high-confidence candidate set with probability $> \beta$ ($= 0.01$) is first constructed, and FPS is applied within this set.
- **Key design**: low-probability points are **not discarded**; they are retained in the point cloud for feature aggregation—they are simply excluded from serving as sampling centers.
- Applicable only to networks employing FPS (PointNet++ / Point Transformer, etc.).

### Direct PPC Integration into Models (Advanced)
Using VoteNet as a case study, three integration strategies are explored:
- **(A) Probability as point attribute input**: the network learns to exploit probability features.
- **(B) Probability-weighted feature vectors**: PointNet++ per-point features are multiplied by neighborhood-average probability, amplifying features of high-confidence points.
- **(C) Probability-weighted objectness scores**: proposal objectness is multiplied by the average probability of points within the bounding box, prioritizing high-confidence proposals.
- Combining all three yields an additional ~2% AP@25 gain on top of the already strong PPC baseline.

## Key Experimental Results

### SUN RGB-D Indoor Detection (VoteNet, AP@25 / AP@50)

| Method | Clean | SBR=0.1 | SBR=0.05 | SBR=0.02 | SBR=0.01 |
|--------|-------|---------|----------|----------|----------|
| Matched Filtering | 51.3/27.5 | 42.4/20.5 | 38.8/17.6 | 17.0/5.1 | 11.3/2.7 |
| Thresholding | 57.1/33.2 | 51.3/28.6 | 46.4/24.9 | 29.6/14.8 | 16.5/6.5 |
| PointCleanNet | 54.6/31.9 | 45.7/26.4 | 40.2/19.2 | 18.2/8.1 | 12.8/3.0 |
| Score Denoising | 57.4/34.0 | 53.2/29.5 | 48.6/25.8 | 26.4/13.7 | 14.6/4.7 |
| **PPC (Ours)** | **58.6/35.0** | **54.3/31.2** | **52.5/30.2** | **38.5/16.5** | **29.4/13.2** |

- **At extremely low SBR=0.01, PPC vs. Score Denoising: +14.9% absolute AP@25 (29.4 vs. 14.6).**
- **At SBR=0.02, AP@50 improves by 4.4% (30.2 vs. 25.8).**

### KITTI Outdoor Detection (PV-RCNN, mAP Moderate)

| SBR | Car/Ped/Cyc (Baseline) | Car/Ped/Cyc (PPC) |
|-----|------------------------|-------------------|
| 0.05 | 73.1/55.8/61.8 | 73.0/**59.1**/**64.1** |
| 0.02 | 68.2/50.0/52.9 | 68.4/**59.0**/**53.2** |
| 0.01 | 60.0/47.1/43.7 | 60.3/**55.4**/**47.8** |
| 0.005 | 50.7/37.0/35.0 | 51.3/**49.5**/**36.4** |

- **Pedestrian class at SBR=0.005: +12.5% (49.5 vs. 37.0)**—the largest gains are observed for small and distant targets.

### Computational Efficiency

| Method | Inference Time (ms/scene) |
|--------|--------------------------|
| Matched Filtering | 87 |
| PointCleanNet | 755 |
| Score Denoising | **1345** |
| PathNet | 867 |
| **PPC (Ours)** | **95** |

- PPC adds only 8 ms (87→95); denoising networks incur 10–15× overhead.

### Real Hardware Validation
- Indoor: HORIBA FLIMera SPAD camera (192×128, 4096 bins), 200–800 lux ambient light.
- Outdoor: Adaps ADS6311 commercial single-photon LiDAR (256×192, 672 bins).
- Baseline methods miss a large fraction of distant and small targets in real-world scenes, whereas PPC detects the majority of objects with accurate bounding boxes.

## Highlights & Insights

- **"Retain uncertainty" rather than "eliminate noise" philosophy**: Conventional pipelines attempt denoising prior to inference; this work takes the opposite approach—propagating uncertainty as a useful signal to downstream models. This perspective motivates end-to-end sensor-to-inference design.
- **Physically grounded confidence**: The probability definition derives directly from a photon-statistics model—it is not a learned black-box feature but an SNR indicator with clear physical meaning.
- **Plug-and-play design**: NPD filtering and FPPS require no architectural modifications and function as preprocessing modules compatible with arbitrary 3D detectors (validated on VoteNet, PV-RCNN, ImVoteNet, Uni3DETR, and PointPillars).
- **Favorable efficiency–accuracy trade-off**: 95 ms vs. 1345 ms (Score Denoising), with superior accuracy—practically deployable.
- **Bandwidth-friendly**: PPC requires only one additional scalar per pixel (the probability), amounting to ~10s MB/s, far below the ~10s GB/s needed for full histogram readout.

## Limitations & Future Work

1. **Oversimplified probability definition**: Using only the peak bin ratio ignores histogram shape information (e.g., multiple peaks, peak width). The paper identifies Cramér–Rao uncertainty estimation as a future direction.
2. **Hyperparameter sensitivity**: Thresholds $\alpha$ (NPD) and $\beta$ (FPPS) require manual tuning and may need adjustment across different sensors and scenes.
3. **Validation limited to SPAD LiDAR**: Extension to stereo vision, structured light, and iToF sensors is discussed but not experimentally validated.
4. **Performance ceiling at very low SBR**: At SBR=0.01, AP@25 remains only 29.4 (vs. 58.6 on clean data)—a 50% drop—indicating PPC also has fundamental limits.
5. **Training strategy**: The current approach uses joint training across all SBR levels; models specialized for specific SBR ranges may yield further gains.

## Related Work & Insights

- **VoteNet / PointNet++**: Primary validation backbones, demonstrating the value of uncertainty propagation for classical point cloud networks.
- **PV-RCNN**: Voxelization with point feature fusion; PPC proves effective in voxel-based pipelines as well.
- **Score-based Point Cloud Denoising** (Luo & Hu, 2021): Current SOTA denoising, but the assumption of local Gaussian noise is ill-suited for real LiDAR noise characteristics.
- **Single-Photon 3D Imaging** (Lindell et al., 2018; Heide et al., 2018): 3D reconstruction from histograms; PPC is complementary and can be applied on top of such reconstructions.
- **Insight**: Any sensor-to-inference pipeline should consider uncertainty propagation rather than simple binary filtering—a principle extensible to radar, ultrasound, and other sensing modalities.

## Rating
- Novelty: ⭐⭐⭐⭐ First work to propagate SPAD sensor uncertainty into 3D inference; the PPC concept is concise and compelling.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Simulation and real hardware, indoor and outdoor, 5 detectors, 5 SBR levels, multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Narrative flows smoothly from sensor physics to inference methodology; figures are clear.
- Value: ⭐⭐⭐⭐ Practical significance for adverse-condition 3D perception in autonomous driving and robotics; directly deployable.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Free-running vs. Synchronous: Single-Photon Lidar for High-flux 3D Imaging](free-running_vs_synchronous_single-photon_lidar_for_high-flux_3d_imaging.md)
- [\[AAAI 2026\] DriveFlow: Rectified Flow Adaptation for Robust 3D Object Detection in Autonomous Driving](../../AAAI2026/autonomous_driving/driveflow_rectified_flow_adaptation_for_robust_3d_object_detection_in_autonomous.md)
- [\[ICCV 2025\] CVFusion: Cross-View Fusion of 4D Radar and Camera for 3D Object Detection](cvfusion_cross-view_fusion_of_4d_radar_and_camera_for_3d_object_detection.md)
- [\[CVPR 2026\] BuildAnyPoint: 3D Building Structured Abstraction from Diverse Point Clouds](../../CVPR2026/autonomous_driving/buildanypoint_3d_building_structured_abstraction_from_diverse_point_clouds.md)
- [\[ICCV 2025\] A Constrained Optimization Approach for Gaussian Splatting from Coarsely-posed Images and Noisy Lidar Point Clouds](a_constrained_optimization_approach_for_gaussian_splatting_from_coarsely-posed_i.md)

<!-- RELATED:END -->
