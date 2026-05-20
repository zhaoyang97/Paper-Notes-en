---
title: >-
  [Paper Note] 3DRealCar: An In-the-wild RGB-D Car Dataset with 360-degree Views
description: >-
  [ICCV 2025][Autonomous Driving][3D vehicle dataset] This paper presents 3DRealCar, the first large-scale real-world 3D vehicle dataset comprising 2,500 vehicles from 100+ brands…
tags:
  - "ICCV 2025"
  - "Autonomous Driving"
  - "3D vehicle dataset"
  - "RGB-D"
  - "360-degree views"
  - "3D reconstruction"
date: 2026-05-08
content_hash: a1730ccd9b77a7b9
---

# 3DRealCar: An In-the-wild RGB-D Car Dataset with 360-degree Views

**Conference**: ICCV 2025
**arXiv**: [2406.04875](https://arxiv.org/abs/2406.04875)  
**Code**: N/A (dataset project)  
**Area**: Autonomous Driving
**Keywords**: 3D vehicle dataset, RGB-D, 360-degree views, 3D reconstruction, autonomous driving

## TL;DR
This paper presents 3DRealCar, the first large-scale real-world 3D vehicle dataset comprising 2,500 vehicles from 100+ brands, each with approximately 200 high-resolution 360-degree RGB-D views captured under three lighting conditions (standard, reflective, and low-light), along with 13-category vehicle parsing annotations, supporting tasks including 3D reconstruction, detection, and generation.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: High-quality 3D vehicle models are in broad demand across autonomous driving, virtual/augmented reality, and gaming. Autonomous driving systems in particular require simulation of realistic corner-case scenarios (e.g., traffic accidents), which depends on high-quality 3D vehicle assets.

Existing 3D vehicle datasets have notable deficiencies: SRN-Car and Objaverse-Car are synthetic, lacking realistic textures and geometric detail; MVMC, though real-world, provides only an average of 10 views per vehicle without pose annotations, insufficient for high-quality reconstruction. Vehicle images in autonomous driving datasets suffer from low resolution and limited viewpoints. Meanwhile, text-to-3D generation methods (e.g., MVDream) also fail to produce high-quality vehicle models.

Core limitation: the absence of a large-scale, high-quality, and diverse real-world 3D vehicle dataset. Manual 3D vehicle modeling is labor-intensive, synthetic data lacks realism, and autonomous driving datasets are insufficient in viewpoint coverage and resolution.

This paper addresses these gaps by scanning real parked vehicles at scale using a smartphone combined with ARKit, constructing a 3D real-world vehicle dataset with high quantity (2,500 vehicles), high quality (~200 dense views/vehicle), and high diversity (100+ brands, 3 lighting conditions).

## Method

### Overall Architecture
The dataset construction pipeline consists of three stages: (1) **Data Collection**: scanning vehicles using an iPhone 14 with ARKit, capturing RGB-D images and point clouds over three circuits around each vehicle; (2) **Data Preprocessing**: background removal, orientation correction, and point cloud scaling; (3) **Annotation and Quality Control**: providing brand, type, color, and parsing annotations while filtering blurred and occluded images.

### Key Designs
1. **Data Collection Strategy**:

    - Function: Efficiently acquire high-quality 3D vehicle data using consumer-grade devices.
    - Mechanism: An iPhone 14 with the ARKit API is used to scan parked vehicles, with three circuits per vehicle at varying phone heights, yielding approximately 200 RGB-D images at 1920×1440 resolution along with corresponding point clouds. Data are collected under three lighting conditions: standard (uniform illumination), reflective (strong specular highlights), and low-light (underground parking).
    - Design Motivation: Three circuits at different heights ensure full angular coverage; ARKit provides depth information and initial pose estimates; the three lighting conditions are specifically designed to expose the challenges faced by existing 3D reconstruction methods.

2. **Background Removal and Orientation Correction**:

    - Function: Extract clean vehicle data from raw scans and unify vehicle orientation.
    - Mechanism: In 2D, Grounding DINO detects vehicle bounding boxes and SAM segments vehicle masks; in 3D, point clouds are projected onto 2D and background points are removed using these masks. Orientation correction applies PCA to camera poses to compute a principal component transform $\mathcal{T} \in \mathbb{R}^{3\times3}$, aligning vehicle orientation to the x-axis; manual intervention is applied when this fails.
    - Design Motivation: Clean vehicle-only data facilitates 3D reconstruction; unified orientation enables controllable rendering.

3. **Point Cloud Scaling and Pose Refinement**:

    - Function: Restore reconstructed point clouds to real-world scale.
    - Mechanism: COLMAP is used to reconstruct denser point clouds and obtain accurate camera poses (ARKit poses lack sufficient precision). The bounding box ratio between the scanned foreground point cloud and the COLMAP point cloud is computed to scale both the point cloud and camera pose translations to real-world dimensions.
    - Design Motivation: Real-world scale point clouds are critical for autonomous driving simulation.

### Dataset Annotations
- 13-category vehicle parsing maps (body, windows, lights, tires, etc.), supporting vehicle part understanding tasks.
- Detailed metadata including brand (100+), type (Sedan/SUV/MPV/Van/Lorry/Sports Car), and color (20+ variants).
- Privacy protection: license plates and personal information are blurred.

## Key Experimental Results

### Main Results (3D Reconstruction Benchmark)

| Method | Standard PSNR↑ | Reflective PSNR↑ | Dark PSNR↑ |
|------|----------------|-------------------|------------|
| 3DGS | 27.47 | 24.58 | 23.51 |
| GaussianShader | 27.53 | **25.41** | 23.39 |
| Pixel-GS | **27.67** | 24.81 | **23.54** |
| 3DGS-MCMC | 27.63 | 24.92 | 23.63 |
| Instant-NGP | 27.31 | 24.37 | 23.17 |
| 2DGS | 27.34 | 23.19 | 22.63 |

### Ablation Study (Effect of Simulated Data Volume on Corner-Case Detection)

| Simulated Data Volume | YOLOv5n | YOLOv8s | CO-DETR | YOLOv12x |
|-----------|---------|---------|---------|----------|
| 1000 | 0.285 | 0.371 | 0.465 | 0.412 |
| 3000 | 0.345 | 0.403 | 0.517 | 0.489 |
| 5000 | 0.361 | 0.435 | 0.582 | 0.565 |

### Key Findings
- Existing methods achieve high-quality reconstruction (27+ dB PSNR) under standard lighting; however, PSNR drops significantly (~3–4 dB) under reflective and low-light conditions, exposing the limitations of current methods under challenging illumination.
- After fine-tuning with 3DRealCar data, Zero123-XL and Dreamcraft3D show substantial improvements in generation quality on real vehicles (CLIP-I: 0.812→0.847; CD: 0.587→0.371).
- Increasing simulated corner-case data from 1,000 to 5,000 raises CO-DETR's mAP@0.5 from 0.465 to 0.582, demonstrating the dataset's value for autonomous driving safety.
- GaussianShader performs best under reflective conditions, indicating that methods accounting for material properties have an advantage on glossy surfaces.

## Highlights & Insights
- The smartphone-based pipeline for high-quality 3D data collection is extremely low-cost and highly scalable.
- The three-lighting-condition design is notably forward-looking — reflective and low-light conditions pose new challenges for the 3D reconstruction community.
- The dataset supports 10+ tasks (detection, segmentation, reconstruction, generation, novel view synthesis, etc.), delivering multi-faceted value from a single data source.
- The background removal pipeline (Grounding DINO + SAM + point cloud projection filtering) offers a reusable design pattern.
- The data scale is impressive: 2,500 vehicles × 200 views = 500K high-resolution RGB-D images.
- The corner-case scene simulation approach has practical application value, using reconstructed vehicles to synthesize rare driving scenarios.

## Limitations & Future Work
- The dataset currently covers only exterior views; future work plans to include interior views to support more complete 3D vehicle modeling.
- White and black vehicles are overrepresented, resulting in an unbalanced color distribution that may affect the generalization of downstream models.
- Pose estimation relies on COLMAP, which may be inaccurate in low-texture vehicle regions (e.g., large uniform-color body panels).
- Vehicles are captured only in static states; data under motion and deformation modeling are absent.
- PCA-based orientation correction occasionally fails and requires manual intervention, leaving room for improved automation.
- Despite their challenge value, reconstruction quality under reflective and low-light conditions remains unsatisfactory, calling for new methodological advances.

## Related Work & Insights
- **vs. SRN-Car/Objaverse-Car**: Synthetic data lacks realistic textures; this work provides the first large-scale real-world 3D vehicle data.
- **vs. MVMC**: ~200 views per vehicle vs. 10; resolution 1920×1440 vs. 600×450; additionally provides pose and depth information.
- **vs. Waymo/nuScenes**: Vehicle viewpoints and resolution in these autonomous driving datasets are too limited for high-quality per-vehicle reconstruction.
- **vs. MVDream and other generation methods**: Generation methods fail to produce high-quality vehicles due to multi-view inconsistency; real data remains irreplaceable.
- **Insight**: The paradigm of large-scale 3D data collection via smartphones is generalizable to other object categories (e.g., furniture, buildings).

## Rating
- Novelty: ⭐⭐⭐⭐ First large-scale real-world 3D vehicle dataset; three-lighting-condition design is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive benchmarking across multiple 2D and 3D downstream tasks.
- Writing Quality: ⭐⭐⭐⭐ Dataset characteristics and collection pipeline are clearly described with thorough statistical analysis.
- Value: ⭐⭐⭐⭐⭐ Fills a critical gap in real-world 3D vehicle datasets with significant practical implications for autonomous driving simulation.
- Overall: A benchmark contribution for dataset papers; the complete collect–process–annotate–benchmark pipeline serves as a valuable reference for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Detect Anything 3D in the Wild](detect_anything_3d_in_the_wild.md)
- [\[ICCV 2025\] UAVScenes: A Multi-Modal Dataset for UAVs](uavscenes_a_multi-modal_dataset_for_uavs.md)
- [\[ICCV 2025\] LangTraj: Diffusion Model and Dataset for Language-Conditioned Trajectory Simulation](langtraj_diffusion_model_and_dataset_for_language-conditioned_trajectory_simulat.md)
- [\[NeurIPS 2025\] LabelAny3D: Label Any Object 3D in the Wild](../../NeurIPS2025/autonomous_driving/labelany3d_label_any_object_3d_in_the_wild.md)
- [\[ICCV 2025\] Mixed Signals: A Diverse Point Cloud Dataset for Heterogeneous LiDAR V2X Collaboration](mixed_signals_a_diverse_point_cloud_dataset_for_heterogeneous_lidar_v2x_collabor.md)

</div>

<!-- RELATED:END -->
