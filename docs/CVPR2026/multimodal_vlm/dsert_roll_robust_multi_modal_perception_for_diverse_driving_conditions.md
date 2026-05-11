---
title: >-
  [Paper Note] DSERT-RoLL: Robust Multi-Modal Perception for Diverse Driving Conditions
description: >-
  [CVPR 2026][Multimodal VLM][multi-modal dataset] This paper introduces the DSERT-RoLL driving dataset, the first to integrate six sensor modalities — stereo event cameras, RGB, thermal imaging, 4D radar, and dual LiDAR — covering diverse weather and lighting conditions, along with a unified multi-modal 3D detection fusion framework.
tags:
  - CVPR 2026
  - Multimodal VLM
  - multi-modal dataset
  - event camera
  - 4D radar
  - thermal camera
  - 3D detection
date: 2026-05-08
content_hash: 0519a34ffbf0e71d
---

# DSERT-RoLL: Robust Multi-Modal Perception for Diverse Driving Conditions

**Conference**: CVPR 2026
**arXiv**: [2604.03685](https://arxiv.org/abs/2604.03685)
**Code**: [https://jeongyh98.github.io/dsert-roll](https://jeongyh98.github.io/dsert-roll)
**Area**: Autonomous Driving / Multi-Modal Perception
**Keywords**: multi-modal dataset, event camera, 4D radar, thermal camera, 3D detection

## TL;DR

This paper introduces the DSERT-RoLL driving dataset, the first to integrate six sensor modalities — stereo event cameras, RGB, thermal imaging, 4D radar, and dual LiDAR — covering diverse weather and lighting conditions, along with a unified multi-modal 3D detection fusion framework.

## Background & Motivation

Autonomous driving perception remains severely challenged under adverse weather (fog, rain, snow) and extreme lighting conditions, where conventional RGB+LiDAR systems suffer significant performance degradation. Emerging sensors such as event cameras (robust to high dynamic range and fast motion), thermal cameras (effective at night), and 4D radar (strong penetration in harsh weather) offer complementary advantages. However, existing datasets typically include only partial sensor combinations, lacking systematic cross-sensor comparison and fusion studies conducted under identical environmental conditions.

The core contribution of DSERT-RoLL lies in integrating all these novel and conventional sensors onto a single data collection platform, enabling cross-sensor comparison and fusion research for the first time.

## Method

### Overall Architecture

The dataset comprises 22K frames of multi-modal sensor data spanning highway, urban street, and suburban road scenarios. The paper also proposes a multi-modal 3D detection fusion framework: LiDAR and 4D radar voxelized features generate initial 3D box proposals, and RGB/thermal/event camera features are integrated into 3D space via confidence-based fusion.

### Key Designs

1. **Comprehensive Sensor Suite**: Stereo RGB (2448×2048), stereo event cameras (1280×720), stereo thermal cameras (640×512), 4D radar (100 m range), long-range LiDAR (150 m), and short-range high-resolution LiDAR (100 m, 360°). All cameras are configured as stereo pairs to cover the forward field of view.

2. **3D Range Sensor Fusion**: LiDAR and 4D radar are individually voxelized to extract BEV features, which are then concatenated along the channel dimension and fused via convolution to generate initial 3D box proposals. 4D radar provides Doppler velocity information in adverse weather, compensating for LiDAR degradation in fog and snow.

3. **Camera–3D Range Sensor Fusion**: A voxel-centric sampling strategy is proposed, starting from non-empty voxel indices of LiDAR and radar to construct a unified sparse voxel feature space. Each non-empty voxel is projected onto the image planes of RGB, thermal, and event cameras; neighborhood image features are sampled via deformable cross-attention and fused into 3D space, achieving confidence-weighted multi-modal integration.

### Loss & Training

Standard 3D detection losses (regression + classification) are applied to the unified multi-modal feature representation after fusion. Data is split 7:3 for training and testing, with weather, lighting, and category distributions carefully balanced across both splits.

## Key Experimental Results

### Main Results

| Modality Combination | Weather-Clear | Weather-Fog | Weather-Heavy Snow | Lighting-HDR |
|---|---|---|---|---|
| L (LiDAR only) | 82.90 | 65.67 | 54.14 | 74.51 |
| R+L | 84.67 | 66.14 | 59.43 | 79.31 |
| 4R+L | 88.26 | 67.41 | 69.96 | 82.98 |
| R+E+T+4R+L (All Modalities) | 90.30 | 71.42 | 72.94 | 86.33 |

### Key Findings

- 4D radar contributes most significantly under adverse weather (heavy snow: +15.82 vs. LiDAR only)
- Event cameras are particularly valuable under HDR and overexposed lighting conditions
- Thermal cameras complement RGB in low-light and nighttime scenarios
- Full modality fusion achieves the best performance under all conditions, confirming sensor complementarity

## Highlights & Insights

- The first driving dataset to simultaneously include six sensor modalities (including novel sensors) collected in the same environment
- Systematically reveals the strengths and weaknesses of individual sensors across diverse environmental conditions
- The voxel-centric sampling strategy elegantly resolves the heterogeneous sensor-to-unified-3D-space mapping problem
- Data distribution is carefully balanced across weather conditions, lighting scenarios, and object categories

## Limitations & Future Work

- Dataset scale (22K frames) is relatively small compared to large-scale datasets such as Waymo
- Only three object categories (vehicles, pedestrians, cyclists) are covered, limiting scope
- Sensor calibration and temporal synchronization may introduce biases under extreme conditions

## Related Work & Insights

- Complements single-modality datasets such as K-Radar (4D radar), DSEC (event camera), and KAIST (thermal imaging)
- The modular design of the fusion framework facilitates future exploration of additional sensor combinations

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First comprehensive driving dataset with multiple novel sensor modalities
- **Technical Depth**: ⭐⭐⭐⭐ — Fusion framework is well-motivated and soundly designed
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Systematic ablation across sensor combinations
- **Value**: ⭐⭐⭐⭐⭐ — Fills a critical data gap in multi-sensor research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](../../ICLR2026/multimodal_vlm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[CVPR 2026\] UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](unimmad_unified_multi-modal_and_multi-class_anomaly_detection_via_moe-driven_fea.md)
- [\[CVPR 2026\] Wan-Weaver: Interleaved Multi-modal Generation via Decoupled Training](wan-weaver_interleaved_multi-modal_generation_via_decoupled_training.md)
- [\[CVPR 2026\] Decoupling Stability and Plasticity for Multi-Modal Test-Time Adaptation](decoupling_stability_and_plasticity_for_multi-modal_test-time_adaptation.md)
- [\[CVPR 2026\] Multi-Modal Image Fusion via Intervention-Stable Feature Learning](multi-modal_image_fusion_via_intervention-stable_feature_learning.md)

</div>

<!-- RELATED:END -->
