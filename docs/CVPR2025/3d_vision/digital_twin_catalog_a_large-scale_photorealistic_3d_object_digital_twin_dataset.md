---
title: >-
  [Paper Note] Digital Twin Catalog: A Large-Scale Photorealistic 3D Object Digital Twin Dataset
description: >-
  [CVPR 2025][3D Vision][Digital Twin] Meta Reality Labs proposes the DTC dataset, containing 2,000 3D object digital twin models with millimeter-level geometric precision and photorealistic PBR materials, paired with evaluation data captured via DSLR and egocentric AR glasses, establishing the first comprehensive real-world benchmark for 3D reconstruction and inverse rendering.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Digital Twin"
  - "3D Dataset"
  - "Inverse Rendering"
  - "PBR Material"
  - "Egocentric Reconstruction"
date: 2026-05-08
content_hash: c4f5f84a31ca1051
---

# Digital Twin Catalog: A Large-Scale Photorealistic 3D Object Digital Twin Dataset

**Conference**: CVPR 2025  
**arXiv**: [2504.08541](https://arxiv.org/abs/2504.08541)  
**Code**: [https://www.projectaria.com/datasets/dtc/](https://www.projectaria.com/datasets/dtc/)  
**Area**: 3D Vision  
**Keywords**: Digital Twin, 3D Dataset, Inverse Rendering, PBR Material, Egocentric Reconstruction

## TL;DR

Meta Reality Labs proposes the DTC dataset, containing 2,000 3D object digital twin models with millimeter-level geometric precision and photorealistic PBR materials, paired with evaluation data captured via DSLR and egocentric AR glasses, establishing the first comprehensive real-world benchmark for 3D reconstruction and inverse rendering.

## Background & Motivation

- Digital twins require virtual 3D objects to be indistinguishable from their real-world counterparts in terms of shape and appearance, which is crucial for applications such as AR/VR and robotics.
- Although methods like NeRF and 3DGS have significantly improved the quality of novel view synthesis, there is a lack of large-scale, digital-twin-level real-world datasets to quantitatively evaluate and compare different reconstruction methods.
- Existing datasets are either large-scale but of varied quality (e.g., Objaverse), or high-quality but too small (e.g., Stanford-ORB with only 14 objects).
- To advance the application of 3D reconstruction on next-generation egocentric computing platforms (such as AR glasses), egocentrically captured evaluation data is required, which has been absent from prior datasets.

## Method

### Overall Architecture

The DTC dataset consists of three parts: (1) 2,000 scan-level 3D object models; (2) DSLR evaluation data for 50 objects under two lighting conditions (100 sequences); (3) Egocentric Aria glasses evaluation data for 100 objects (200 video recordings).

### Key Designs

1. **Industrial-Grade 3D Object Scanning**:
    - **Function**: Creates 3D models with millimeter-level geometric accuracy and photorealistic PBR materials.
    - **Mechanism**: Utilizes a Covision Media industrial scanner, equipped with 8 structured light units inside a hemispherical dome for geometric scanning, and 29 spotlights and 29 cameras for material acquisition. Each object takes about 20 minutes to scan, followed by 4 hours of post-processing to generate 4K PBR textures.
    - **Design Motivation**: Structured light ensures millimeter-level accuracy, and professional technical artists correct materials of shiny/reflective objects to guarantee digital twin quality.

2. **DSLR Evaluation Data Capture System**:
    - **Function**: Provides high-quality multi-view HDR/LDR images and accurate annotations.
    - **Mechanism**: An automated DSLR camera turntable is designed where 3 cameras rotate 360° around the object, capturing one image every 9° (120 images in total). Accurate camera poses are obtained using active ChArUco boards, and environmental lighting as well as object poses are optimized through differentiable rendering.
    - **Design Motivation**: Accurate ground truth (poses, lighting, 3D models) is a prerequisite for fair evaluation of inverse rendering methods.

3. **Egocentric Evaluation Data and Alignment Pipeline**:
    - **Function**: Offers the first evaluation benchmark for 3D object reconstruction on egocentric devices.
    - **Mechanism**: Using Project Aria glasses for video recording, 3D objects are precisely aligned with egocentric videos through a pipeline of neural mesh reconstruction $\rightarrow$ reference mask generation $\rightarrow$ differentiable rendering-based alignment. Both active (360°) and passive (casual observation) recording trajectories are provided.
    - **Design Motivation**: Accelerates the integration of 3D reconstruction technologies with egocentric devices like AR glasses, enabling everyone to easily create digital twins.

### Loss & Training

This is a dataset paper and does not involve training loss functions. Benchmark evaluations adopt standard metrics: geometry (depth SI-MSE, normal error, Chamfer distance), relighting (PSNR, SSIM, LPIPS), and novel view synthesis (PSNR, SSIM, LPIPS).

## Key Experimental Results

### Main Results

| Method | Depth↓ | Normal↓ | Shape↓ | Relighting PSNR-H↑ | NVS PSNR-H↑ |
|------|-------|-------|-------|--------------|-------------|
| InvRender | 0.22 | **0.03** | **0.75** | **29.52** | **31.64** |
| NVDiffRecMC | **0.02** | 0.06 | 1.34 | 27.78 | 31.27 |
| NVDiffRec | **0.02** | 0.07 | 1.64 | 26.99 | 28.95 |
| PhySG | 0.31 | 0.16 | 11.31 | 27.28 | 28.54 |
| Neural-PIL | 5.71 | 0.25 | 25.02 | N/A | 28.42 |
| NeRD | 4.55 | 0.45 | 108.20 | 26.10 | 26.80 |

### Ablation Study

| Configuration | Push@2cm | Push@5cm | Grasp | Description |
|------|---------|---------|------|------|
| DTC Training | 36.3% | 47.0% | 42.7% | Digital twin quality objects |
| Objaverse-XL Training | 25.3% | 40.3% | 38.6% | Objects of varied quality |

### Key Findings

- Current state-of-the-art inverse rendering methods still have a significant gap under digital twin standards, especially on objects with shiny materials.
- InvRender achieves the best overall performance in relighting and NVS, but its geometric accuracy is inferior to NVDiffRec/NVDiffRecMC.
- The PSNR of 3DGS and 2DGS on egocentric data is around 28.8, indicating that there is still substantial room for improvement in egocentric reconstruction.
- Robotic policies trained on DTC high-quality objects outperform those trained on Objaverse-XL in both pushing and grasping tasks.

## Highlights & Insights

- **Balancing Scale and Quality**: 2,000 objects with both millimeter-level geometry and photorealistic PBR materials, far exceeding previous datasets.
- **First Egocentric 3D Reconstruction Benchmark**: Opens up a new direction for evaluating reconstruction on devices like AR glasses.
- **Downstream Application Validation**: Proves the value of high-quality digital twins for sim-to-real through robotic pushing/grasping experiments.
- **Complete Evaluation Framework**: Simultaneously covers geometry, materials, relighting, and novel view synthesis.

## Limitations & Future Work

- Current scanning hardware limits the size range of objects and cannot handle deformable, highly specular, or transparent objects.
- The post-processing pipeline is time-consuming (about 4 hours per object), and manual corrections remain inevitable.
- The dataset only covers 40 LVIS categories, and category diversity needs to be expanded.
- Object alignment for egocentric data may fail on objects with symmetrical geometry.

## Related Work & Insights

- Compared with Stanford-ORB (14 objects), DTC provides more objects and higher quality.
- Compared with Objaverse (818K objects), each object in DTC features high-quality PBR materials and alignment with real-world videos.
- Provides a more reliable evaluation platform for differentiable rendering methods (such as NVDiffRec, InvRender).
- Insight: High-quality digital twin data may be the key to bridging the sim-to-real gap.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The concept is clear, but it is primarily an engineering system rather than an algorithmic innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers multiple tasks such as inverse rendering, NVS, egocentric reconstruction, and robotics.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear structure, detailed data, and comprehensive comparisons.
- **Value**: ⭐⭐⭐⭐⭐ Fills the gap of large-scale, high-quality 3D digital twin datasets, driving significant progress for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Twinner: Shining Light on Digital Twins in a Few Snaps](twinner_shining_light_on_digital_twins_in_a_few_snaps.md)
- [\[CVPR 2025\] MotionAnyMesh: Physics-Grounded Articulation for Simulation-Ready Digital Twins](motionanymesh_physics-grounded_articulation_for_simulation-ready_digital_twins.md)
- [\[CVPR 2026\] OLATverse: A Large-scale Real-world Object Dataset with Precise Lighting Control](../../CVPR2026/3d_vision/olatverse_a_large-scale_real-world_object_dataset_with_precise_lighting_control.md)
- [\[CVPR 2025\] 3D-SLNR: A Super Lightweight Neural Representation for Large-scale 3D Mapping](3d-slnr_a_super_lightweight_neural_representation_for_large-scale_3d_mapping.md)
- [\[CVPR 2025\] Horizon-GS: Unified 3D Gaussian Splatting for Large-Scale Aerial-to-Ground Scenes](horizon-gs_unified_3d_gaussian_splatting_for_large-scale_aerial-to-ground_scenes.md)

</div>

<!-- RELATED:END -->
