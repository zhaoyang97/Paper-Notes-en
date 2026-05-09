---
title: >-
  [Paper Note] 3DRealCar: An In-the-wild RGB-D Car Dataset with 360-degree Views
description: >-
  [ICCV 2025][Autonomous Driving][3D Car Dataset] This paper introduces 3DRealCar, the first large-scale real-world 3D car dataset comprising high-resolution (1920×1440) 360-degree RGB-D scans of 2,500 real vehicles (averaging 200 views per car), covering 100+ brands and three lighting conditions (standard / high-reflectance / low-light). The dataset provides rich annotations including point clouds and parsing maps, and benchmarks multiple 3D reconstruction methods, revealing significant reconstruction challenges under reflective and low-light conditions.
tags:
  - ICCV 2025
  - Autonomous Driving
  - 3D Car Dataset
  - RGB-D
  - 360-degree Views
  - 3D Reconstruction
  - novel view synthesis
date: 2026-05-08
content_hash: 8a1a8cd1f145397d
---

# 3DRealCar: An In-the-wild RGB-D Car Dataset with 360-degree Views

**Conference**: ICCV 2025
**arXiv**: [2406.04875](https://arxiv.org/abs/2406.04875)
**Code**: [Project Page](https://xiaobiaodu.github.io/3drealcar/)
**Area**: 3D Vision / Autonomous Driving
**Keywords**: 3D Car Dataset, RGB-D, 360-degree Views, 3D Reconstruction, novel view synthesis

## TL;DR

This paper introduces 3DRealCar, the first large-scale real-world 3D car dataset comprising high-resolution (1920×1440) 360-degree RGB-D scans of 2,500 real vehicles (averaging 200 views per car), covering 100+ brands and three lighting conditions (standard / high-reflectance / low-light). The dataset provides rich annotations including point clouds and parsing maps, and benchmarks multiple 3D reconstruction methods, revealing significant reconstruction challenges under reflective and low-light conditions.

## Background & Motivation

- **High-quality 3D car models are essential for autonomous driving**: Photorealistic simulators require accurate 3D car assets to simulate extreme scenarios (e.g., accidents), yet existing 3D car datasets suffer from critical limitations.
- **Deficiencies of existing datasets**: SRN-Car and Objaverse-Car are synthetic with unrealistic textures; MVMC contains real data but averages only 10 views per car, insufficient for high-quality reconstruction; autonomous driving datasets (KITTI, nuScenes, etc.) offer low-resolution imagery and sparse viewpoints.
- **Bottleneck for 3D generative models**: Text-to-3D methods such as MVDream fail to generate high-quality realistic cars due to the absence of real-car priors.
- **Neglected lighting conditions**: Existing datasets do not systematically investigate the impact of different lighting conditions — particularly high reflectance and low light — on 3D reconstruction.

## Core Problem

1. How to construct the first large-scale, high-quality 3D real-world car dataset?
2. How significantly do different lighting conditions (standard / reflective / low-light) affect existing 3D reconstruction methods?
3. Can real car data provide effective priors for downstream tasks such as 3D generation, novel view synthesis, and object detection?

## Method

### Overall Architecture

Data acquisition pipeline: iPhone 14 + ARKit scanning → RGB-D images and point clouds → COLMAP for accurate pose estimation → Grounding DINO + SAM for background removal → PCA pose normalization (front of car aligned to x-axis) → point cloud scale calibration to real-world dimensions → 3DGS-based 3D model reconstruction.

### Key Designs

1. **Three-loop scanning strategy**: Each car is scanned in three concentric loops at different heights, ensuring dense 360-degree coverage (averaging 200 images per car). High-resolution 1920×1440 images preserve fine-grained details.

2. **Three lighting condition annotations**: The dataset systematically categorizes surface lighting into standard (no prominent specular highlights), reflective (strong specular highlights under direct sunlight), and low-light (underground parking), exposing substantial reconstruction quality gaps across conditions.

3. **Complete preprocessing pipeline**: Background removal (SAM segmentation) → pose normalization (PCA + manual adjustment) → scale calibration (ratio between scanned point cloud and COLMAP point cloud). Uniform car orientation supports controllable rendering.

4. **Rich annotations**: 13-class car parsing maps (headlights, windows, wheels, etc.), brand, color, and body type labels, supporting 10+ downstream tasks.

### Dataset Statistics

- 2,500 cars, 100+ brands, 6 body types (sedan, SUV, MPV, etc.)
- 20+ colors, with white and black predominating
- View count: 50–400 images per car, averaging 200
- Coverage across all three lighting conditions

## Key Experimental Results

### 3D Reconstruction Benchmark (Novel View Synthesis)

| Method | Std. PSNR↑ | Std. LPIPS↓ | Refl. PSNR↑ | Refl. LPIPS↓ | Low PSNR↑ | Low LPIPS↓ |
|--------|-----------|------------|------------|-------------|----------|-----------|
| Instant-NGP | 27.31 | 0.1264 | 24.37 | 0.1962 | 23.17 | 0.1642 |
| 3DGS | 27.47 | 0.1001 | 24.58 | 0.1852 | 23.51 | 0.1613 |
| GaussianShader | 27.53 | 0.1109 | **25.41** | **0.1423** | 23.39 | 0.1631 |
| 2DGS | 27.34 | 0.1095 | 23.19 | 0.2041 | 22.63 | 0.1681 |
| Pixel-GS | **27.67** | **0.0994** | 24.81 | 0.1541 | 23.54 | 0.1617 |
| 3DGS-MCMC | 27.63 | 0.0986 | 24.92 | 0.1621 | **23.63** | **0.1622** |

**Key Findings**: PSNR drops by approximately 3 dB under reflective conditions and approximately 4 dB under low-light conditions, demonstrating that specular highlights and low illumination pose major challenges for 3D reconstruction. GaussianShader performs best under reflective conditions (owing to its explicit reflectance modeling), yet a substantial performance gap remains across all methods.

### Downstream Task Improvements

| Task | Baseline | +3DRealCar | Gain |
|------|----------|-----------|------|
| Dreamcraft3D CLIP-I | 0.812 | 0.847 | +0.035 |
| Dreamcraft3D CD↓ | 0.587 | 0.371 | −0.216 |
| Corner-case detection (5000 data) mAP | — | 0.582 (YOLOv8x) | Significant improvement |

### Ablation Study

- Detection mAP improves consistently as synthetic data volume scales from 500 to 5,000 (YOLOv8x: 0.465→0.582), demonstrating the dataset's practical value.
- Zero123-XL fine-tuned on 3DRealCar generates car textures and geometry of substantially higher realism.
- For 2D car parsing, InternImage achieves the highest mIOU of 0.671.

## Highlights & Insights

- **First large-scale real-world 3D car dataset**: Bridges the gap between synthetic data and real-world scenes; the 2,500-car scale far surpasses existing real-data counterparts.
- **Lighting condition analysis as a unique contribution**: Systematically identifies reflective and low-light scenarios as bottlenecks for 3D reconstruction — a factor previously overlooked in the literature.
- **Practical and accessible acquisition pipeline**: High-quality scanning is achieved using only an iPhone + ARKit, without requiring specialized equipment.
- **Multi-task versatility**: A single dataset supports 10+ tasks, including 3D reconstruction, NVS, 3D generation, object detection, and car part parsing.
- **Real priors for 3D generation**: Fine-tuned Dreamcraft3D and Zero123-XL demonstrate substantial improvements in visual quality on real car inputs.

## Limitations & Future Work

- Only exterior viewpoints are covered; interior views are absent.
- Depth accuracy of iPhone scanning is limited, potentially affecting point cloud quality.
- Lighting condition labels are discrete (three categories), whereas real-world illumination varies continuously.
- Dynamic scenarios (e.g., moving vehicles) are excluded, limiting direct applicability to autonomous driving settings.
- Coverage is restricted to cars and does not extend to other traffic participants (pedestrians, cyclists, etc.).

## Related Work & Insights

- vs. **SRN-Car / Objaverse-Car**: 3DRealCar consists of real scans rather than synthetic data, yielding more realistic textures and geometry; however, synthetic data remains easier to acquire and annotate.
- vs. **MVMC**: 3DRealCar provides 200 densely sampled views (vs. 10), 1920×1440 resolution (vs. 600×450), and includes depth information and point clouds, making it substantially more suitable for high-quality 3D reconstruction.
- vs. **Autonomous driving datasets (KITTI / nuScenes)**: 3DRealCar offers complete 360-degree coverage (vs. limited viewpoints in driving datasets), making it better suited for standalone 3D car asset creation.

The "smartphone scanning + automated processing pipeline" paradigm introduced here is generalizable to other object categories. The analysis of lighting effects on reconstruction quality motivates future methods specifically designed to handle specular highlights and low-light conditions. The dataset also supports domain adaptation research (synthetic-to-real domain gap for car data).

## Rating

- **Novelty**: ⭐⭐⭐ The dataset contribution itself is incremental, but it fills an important gap; the lighting condition analysis adds meaningful novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive benchmarking of multiple state-of-the-art methods and validation across diverse downstream tasks.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with thorough statistical analysis and clearly defined task formulations.
- **Value**: ⭐⭐⭐⭐ Of practical value to both the autonomous driving and 3D vision communities; the lighting challenge analysis points to an important research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] 6DOPE-GS: Online 6D Object Pose Estimation using Gaussian Splatting](6dope-gs_online_6d_object_pose_estimation_using_gaussian_splatting.md)
- [\[ICCV 2025\] DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation](dist-4d_disentangled_spatiotemporal_diffusion_with_metric_depth_for_4d_driving_s.md)
- [\[ICCV 2025\] GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](gs-livm_real-time_photo-realistic_lidar-inertial-visual_mapping_with_gaussian_sp.md)
- [\[ICCV 2025\] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [\[ICCV 2025\] ReconDreamer++: Harmonizing Generative and Reconstructive Models for Driving Scene Representation](recondreamer_harmonizing_generative_and_reconstructive_models_for_driving_scene_.md)

</div>

<!-- RELATED:END -->
