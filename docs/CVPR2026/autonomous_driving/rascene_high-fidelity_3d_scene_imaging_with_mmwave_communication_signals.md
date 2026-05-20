---
title: >-
  [Paper Note] Rascene: High-Fidelity 3D Scene Imaging with mmWave Communication Signals
description: >-
  [CVPR 2026][Autonomous Driving][mmWave communication] This paper proposes Rascene, an Integrated Sensing and Communication (ISAC) framework for high-fidelity 3D scene imaging using mmWave OFDM communication signals (5G/W…
tags:
  - "CVPR 2026"
  - "Autonomous Driving"
  - "mmWave communication"
  - "3D scene imaging"
  - "OFDM signals"
  - "multi-frame fusion"
  - "ISAC"
date: 2026-05-08
content_hash: 4408ac8d059afdd2
---

# Rascene: High-Fidelity 3D Scene Imaging with mmWave Communication Signals

**Conference**: CVPR 2026
**arXiv**: [2604.02603](https://arxiv.org/abs/2604.02603)  
**Code**: N/A  
**Area**: Autonomous Driving / 3D Perception / Integrated Sensing and Communication
**Keywords**: mmWave communication, 3D scene imaging, OFDM signals, multi-frame fusion, ISAC

## TL;DR

This paper proposes Rascene, an Integrated Sensing and Communication (ISAC) framework for high-fidelity 3D scene imaging using mmWave OFDM communication signals (5G/Wi-Fi). It achieves geometrically consistent recovery from sparse, multipath-corrupted RF observations via confidence-weighted multi-frame fusion.

## Background & Motivation

3D environment perception is critical for autonomous driving and robotic navigation. Existing mainstream approaches exhibit notable limitations:
- **Cameras**: Strictly constrained by lighting conditions and fail in adverse environments such as smoke or fog.
- **LiDAR**: Expensive, bulky, and power-hungry, and similarly degraded by harsh weather.
- **Dedicated radar**: Although capable of penetrating obstacles, it requires ultra-wideband hardware (multi-GHz bandwidth) and dedicated spectrum licenses, resulting in high cost and poor scalability.

The core insight is that mmWave communication devices (e.g., 5G and Wi-Fi) are already widely deployed, and their OFDM waveforms naturally encode range and angle information. Reusing these existing communication signals for sensing enables low-cost, scalable 3D perception without additional dedicated sensing hardware or spectrum licensing.

A key finding is that commercial mmWave devices can perform monostatic sensing in full-duplex mode — the high directivity of phased-array antennas and short carrier wavelengths provide sufficient RF isolation between transmit and receive paths.

## Method

### Overall Architecture

Rascene consists of two core modules:
1. **RF Data Acquisition and Representation** (Sec. 3): Extracts CIR and angular information using the full-duplex monostatic capability of mmWave communication devices to generate 3D RF point clouds.
2. **Multi-Frame 3D RF Imaging Network** (Sec. 4): Performs confidence-weighted forward projection fusion over multi-frame observations to output dense voxel grids and depth maps.

The input consists of $N$ frames of RF point clouds $\mathcal{S} = \{\mathbf{S}_i\}_{i=1}^N$ with known poses $\mathcal{G} = \{\mathbf{G}_i\}_{i=1}^N$. The objective is to learn a mapping function $\mathcal{F}$ that produces a voxel grid $\hat{\mathbf{V}}_r$ and a depth map $\hat{\mathbf{D}}_r$.

### Key Designs

1. **Full-Duplex Monostatic Sensing**: Commercial mmWave devices simultaneously transmit and receive OFDM signals. Accurate range estimation is achieved via CIR estimation. The key lies in clock synchronization between co-located transmit and receive antennas, enabling the CIR to be directly used for object ranging — range is given by $r = nc/(2B)$. Combined with angle estimation from phased-array antennas (beamforming weights $w_{i,j}(\theta,\phi)$), each frame of RF data is converted into a 3D point cloud $\mathbf{S}$ in spherical coordinates.

2. **Spatially Adaptive Warping and Fusion**: This is the core of the framework. Rather than the conventional target-driven voxel query approach, Rascene employs **source-driven forward projection**: each source voxel is mapped to the reference frame coordinate system via a rigid transformation, and its contribution is distributed over a local support region using an isotropic Gaussian kernel $K_\sigma$. Fusion weights combine geometric proximity with learned confidence (sharpness controlled by raising softplus-mapped $\eta$ to a power). The unified feature representation $\mathbf{Z}_r$ is obtained via normalized weighted averaging.

3. **Coarse-to-Fine 3D Decoder**: Both encoder and decoder adopt 4-layer convolutional architectures (channel multipliers 1, 2, 4, 8), with stage-wise warping and fusion applied after each encoder stage. The decoder progressively densifies the sparse fused representation into a dense feature volume, with two task heads predicting voxel occupancy and depth maps respectively.

### Loss & Training

The total loss is a weighted sum of voxel loss and depth loss:
$$\mathcal{L} = \sum_{r=1}^N (\lambda_v \mathcal{L}_{\text{voxel}}^{(r)} + \lambda_d \mathcal{L}_{\text{depth}}^{(r)})$$

- **Voxel loss**: Binary cross-entropy (BCE) between the predicted grid and the ground truth.
- **Depth loss**: L1 loss between the predicted and ground-truth depth maps.
- Each frame within the window serves as a reference frame, and losses are accumulated over all reference frames.

Hardware prototype: 60 GHz band, 1.2288 GHz bandwidth, 16 Tx + 16 Rx antenna elements, effective sensing range of 7 m, FoV of 120°×60°. Voxel grid size: 64×64×32 (12 cm resolution).

## Key Experimental Results

### Main Results

Dataset: 20 indoor environments, 12 for training and 8 for testing (cross-scene generalization evaluation).

| Method | Frames | AbsRel | MAE (cm) | CD (cm) | CD_Diag (%) |
|--------|--------|--------|----------|---------|-------------|
| PanoRadar | 1 | 14.7% | 34.1 | 32.2 | 3.8% |
| CartoRadar | 5 | — | — | 26.8 | 3.1% |
| **Rascene** | 1 | 14.1% | 32.9 | 31.6 | 3.6% |
| **Rascene** | 5 | **9.4%** | **20.2** | **19.7** | **2.3%** |

Cross-scene generalization average: AbsRel 9.4%, MAE 20.2 cm, RMSE 38.0 cm, CD 19.7 cm, CD_Diag 2.3%.

### Ablation Study

| Frames | AbsRel | MAE (cm) | CD (cm) | CD_Diag (%) |
|--------|--------|----------|---------|-------------|
| 1 | 14.1% | 32.9 | 31.6 | 3.6% |
| 2 | 11.1% | 24.6 | 26.0 | 3.0% |
| 3 | 9.8% | 21.8 | 21.9 | 2.5% |
| 5 | 9.4% | 20.2 | 19.7 | 2.3% |

Pose robustness tests show high stability against translational perturbations (15 cm perturbations have negligible effect), while the method is more sensitive to rotational errors (5°–10° rotation errors cause significant degradation; at 10°, CD_Diag rises from 2.3% to 3.6%).

### Key Findings

- The largest performance gain occurs from 1 to 2 frames, indicating that even a single additional viewpoint provides strong geometric constraints.
- Median absolute depth error is only 6.1 cm, with 90% of pixels having errors below 37.6 cm.
- Multi-frame fusion effectively suppresses hallucinated structures and fills in missed regions.
- Even in areas where LiDAR fails due to absorption or specular reflection (e.g., dark carpets, glass), Rascene successfully recovers coherent scene geometry.

## Highlights & Insights

- **Paradigm innovation**: This work is the first to demonstrate that OFDM communication signals can support high-fidelity 3D imaging without dedicated sensing hardware or spectrum licenses.
- **Source-driven fusion** outperforms target-driven fusion — it avoids redundant sampling of empty regions and better preserves sparse but information-rich RF responses.
- **Complementarity**: RF sensing is inherently robust to failure modes of optical sensors (absorption by low-albedo surfaces, specular reflection from smooth materials), making it complementary to LiDAR.
- The confidence sharpness parameter $\eta$ allows the fusion process to be dominated by high-confidence geometric signals.

## Limitations & Future Work

- The sensing range is limited to 7 m, suitable for indoor scenarios; generalization to large-scale outdoor scenes remains to be validated.
- Known 6-DoF pose information is required (currently relying on an external IMU).
- Angular estimation resolution is constrained by the antenna array size (currently 16×16).
- Evaluation is conducted solely in indoor environments; generalization to real-world outdoor autonomous driving scenarios is unknown.
- While multipath interference is suppressed by fusion, extremely complex multipath scenarios (e.g., severe occlusions) may still pose challenges.

## Related Work & Insights

- **Comparison with PanoRadar/CartoRadar**: These methods rely on dedicated FMCW radar hardware, whereas Rascene repurposes communication devices.
- **Comparison with NeRF/multi-view reconstruction**: Vision-based methods depend on texture-rich RGB images, while RF observations are neither texture-rich nor geometrically explicit.
- **ISAC trend**: Integrated sensing and communication is a key research direction for 6G; Rascene provides a concrete instantiation for 3D imaging within this paradigm.
- **Inspiration**: The forward projection and confidence-weighted fusion strategy is generalizable to other sparse multi-view reconstruction tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to demonstrate high-fidelity 3D scene imaging from mmWave communication signals, presenting a complete system combining full-duplex monostatic sensing and source-driven fusion.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Cross-scene evaluation over 20 indoor environments is comprehensive, with detailed ablations; however, outdoor and large-scale scene validation is absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, with professional and thorough exposition of the underlying physics and system design.
- **Value**: ⭐⭐⭐⭐⭐ — Opens a new pathway toward low-cost, scalable 3D perception, with significant implications for both ISAC and autonomous driving research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] X-Scene: Large-Scale Driving Scene Generation with High Fidelity and Flexible Controllability](../../NeurIPS2025/autonomous_driving/x-scene_large-scale_driving_scene_generation_with_high_fidelity_and_flexible_con.md)
- [\[CVPR 2026\] HG-Lane: High-Fidelity Generation of Lane Scenes under Adverse Weather and Lighting Conditions without Re-annotation](hg-lane_high-fidelity_generation_of_lane_scenes_under_adverse_weather_and_lighti.md)
- [\[CVPR 2026\] CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion](colc_communication-efficient_collaborative_perception_with_lidar_completion.md)
- [\[CVPR 2026\] R4Det: 4D Radar-Camera Fusion for High-Performance 3D Object Detection](r4det_4d_radar-camera_fusion_for_high-performance_3d_object_detection.md)
- [\[ICCV 2025\] Free-running vs. Synchronous: Single-Photon Lidar for High-flux 3D Imaging](../../ICCV2025/autonomous_driving/free-running_vs_synchronous_single-photon_lidar_for_high-flux_3d_imaging.md)

</div>

<!-- RELATED:END -->
