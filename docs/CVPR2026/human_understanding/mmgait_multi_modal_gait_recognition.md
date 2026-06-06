---
title: >-
  [Paper Note] MMGait: Towards Multi-Modal Gait Recognition
description: >-
  [CVPR 2026][Human Understanding][Gait Recognition] MMGait constructs the most comprehensive multi-modal gait recognition benchmark to date (5 sensors, 12 modalities, 725 subjects, 334K sequences)…
tags:
  - "CVPR 2026"
  - "Human Understanding"
  - "Gait Recognition"
  - "Multi-Modal Benchmark"
  - "Multi-Sensor Fusion"
  - "Cross-Modal Retrieval"
  - "Omni-Modal Recognition"
date: 2026-05-08
content_hash: 5b8cbd2fe0db2692
---

# MMGait: Towards Multi-Modal Gait Recognition

**Conference**: CVPR 2026
**arXiv**: [2604.15979](https://arxiv.org/abs/2604.15979)  
**Code**: [https://github.com/BNU-IVC/MMGait](https://github.com/BNU-IVC/MMGait)  
**Area**: Human Understanding
**Keywords**: Gait Recognition, Multi-Modal Benchmark, Multi-Sensor Fusion, Cross-Modal Retrieval, Omni-Modal Recognition

## TL;DR

MMGait constructs the most comprehensive multi-modal gait recognition benchmark to date (5 sensors, 12 modalities, 725 subjects, 334K sequences), introduces the novel omni-modal gait recognition task, and proposes a unified baseline model, OmniGait.

## Background & Motivation

**Background**: Gait recognition, as a long-range, contactless biometric technology, has achieved remarkable progress in recent years. Mainstream methods focus on RGB-derived modalities (silhouettes, pose sequences) and perform well in both indoor and outdoor settings.

**Limitations of Prior Work**: RGB-derived modalities lack 3D perception and suffer severe performance degradation under adverse conditions such as occlusion, rain, fog, and low illumination. Existing multi-modal datasets (e.g., LidarGait, FreeGait) include only RGB and LiDAR sensors, which is insufficient for studying heterogeneous modality interaction and unified cross-sensor retrieval.

**Key Challenge**: Real-world deployments typically involve multiple sensor types (RGB, infrared, depth, LiDAR, 4D radar), yet the absence of a multi-sensor gait benchmark hinders research on unified multi-modal systems. Each modality exhibits unique strengths and weaknesses—RGB is information-rich but sensitive to illumination; LiDAR is precise but sparse; infrared adapts to low light but lacks texture; and 4D radar offers strong penetration but limited resolution.

**Goal**: To construct a multi-modal gait benchmark covering 5 sensor types, systematically investigate the characteristics of each modality, and propose a unified omni-modal recognition framework.

**Key Insight**: A comprehensive evaluation of each modality's recognition capability, transferability, and complementarity across three dimensions: single-modal, cross-modal, and multi-modal.

**Core Idea**: Establish a large-scale multi-sensor gait dataset and introduce the Omni Multi-Modal Gait Recognition task—enabling a single model to accept arbitrary modality inputs and retrieve targets across arbitrary modalities.

## Method

### Overall Architecture

MMGait operates at three levels: (1) the dataset level, where five sensors are synchronously captured and processed into 12 modalities; (2) the benchmark evaluation level, where systematic assessments are conducted under single-modal, cross-modal, and multi-modal paradigms; and (3) the OmniGait model level, which learns a shared cross-modal embedding space to unify all three recognition paradigms.

### Key Designs

1. **Multi-Sensor Data Acquisition System**:

    - **Function**: Provides high-quality gait data across 12 complementary modalities.
    - **Mechanism**: The system deploys an RGB camera (1280×800), an infrared camera (940 nm narrowband, designed to avoid interference with LiDAR), a depth camera (ToF-based), a 128-line LiDAR, and a 4D FMCW radar. All sensors are synchronized at 10 Hz. The RGB and depth sensors share a camera module, ensuring natural synchronization. A pentagram-shaped walking route covers 10 viewpoints spanning 0°–360°. Data are collected from 725 subjects under three conditions: normal walking (NM), carrying a bag (BG), and wearing a coat (CL).
    - **Design Motivation**: To cover the complete sensor spectrum—from visible to infrared, from 2D to 3D, and from low-cost to high-cost—enabling systematic study of each modality's properties under varying conditions.

2. **Omni Multi-Modal Gait Recognition Task**:

    - **Function**: Unifies single-modal, cross-modal, and multi-modal recognition within a single model.
    - **Mechanism**: The objective is to develop a single model capable of accepting any modality type as a query and retrieving targets across any modality. This requires learning modality-invariant gait representations, such that embeddings of the same subject remain close regardless of whether the input is an RGB silhouette, a LiDAR point cloud, or an infrared image.
    - **Design Motivation**: In practical multi-sensor systems, different sensors may be available under different conditions. A unified model eliminates the overhead of training separate models for each pair of modalities.

3. **OmniGait Baseline Model**:

    - **Function**: Serves as a strong baseline for the omni-modal task.
    - **Mechanism**: OmniGait learns a shared embedding space across multiple heterogeneous modalities. Modality-specific adapters handle different input formats (images, point clouds, sequences), from which modality-invariant gait features are extracted. All available modalities are mixed during training within a shared feature space.
    - **Design Motivation**: To provide a simple yet effective unified framework that demonstrates the feasibility and potential of the omni-modal recognition task.

### Loss & Training

A combination of triplet loss and cross-entropy loss is employed, following the standard gait recognition training paradigm. The dataset is split into 200 subjects for training and 525 for testing; the gallery uses NM-01 sequences, and queries are drawn from NM-02, BG-01, and CL-01.

## Key Experimental Results

### Main Results (Single-Modal, GaitBase Framework)

| Modality | NM R1 | BG R1 | CL R1 |
|----------|-------|-------|-------|
| RGB Silhouette | 98.5 | 96.4 | 61.0 |
| RGB Image | 98.4 | 95.3 | 51.7 |
| Infrared Silhouette | 92.1 | 82.3 | 52.0 |
| Depth | 93.5 | 89.1 | 59.1 |
| LiDAR Point Cloud | 82.7 | 78.2 | 58.5 |
| 4D Radar Point Cloud | 23.6 | 14.4 | 15.2 |

### Ablation Study (OmniGait vs. Individual Models)

| Setting | Performance |
|---------|-------------|
| Modality-specific individual models | Optimal per modality |
| OmniGait unified model | Achieves comparable or superior performance to individual models on most modalities |

### Key Findings

- **Infrared modality demonstrates a clear advantage under clothing-change conditions**: The inherent suppression of texture and color in infrared imaging makes it particularly robust in clothing-change scenarios.
- **Structural modalities such as depth and LiDAR also exhibit clothing-change robustness**: 3D structural information is naturally invariant to clothing color and texture.
- **4D radar performance lags significantly behind other modalities**: Point cloud sparsity and low resolution constrain its capacity for gait representation.
- **The performance gap between RGB and infrared silhouettes is primarily attributable to domain shift in the segmentation model**: Silhouette extractors trained on RGB data produce lower-quality masks when applied directly to infrared imagery.
- **OmniGait demonstrates the feasibility of omni-modal unification**: A single model achieves performance comparable to modality-specific models across multiple modalities.

## Highlights & Insights

- **Unmatched dataset comprehensiveness**: The combination of 5 sensors, 12 modalities, 10 viewpoints, and 3 conditions provides an unprecedented research scope.
- **Forward-looking task definition**: The omni-modal goal of a unified model capable of handling arbitrary modalities directly addresses real-world deployment requirements.
- **Value of systematic analysis**: A comprehensive evaluation across 12 modalities for the first time reveals the concrete strengths and weaknesses of each modality in gait recognition tasks, including the clothing-change robustness of infrared and the limitations of 4D radar.

## Limitations & Future Work

- Data collection was conducted in a controlled environment, leaving a gap with real-world outdoor scenarios.
- The scale of 725 subjects, while relatively large for multi-modal settings, remains considerably smaller than RGB-only datasets such as GREW (26K subjects).
- OmniGait is designed as a simple baseline; more sophisticated cross-modal alignment strategies may yield further performance gains.
- The potential of 4D radar remains underexplored and warrants dedicated network design.

## Related Work & Insights

- **vs. SUSTech1K**: SUSTech1K covers only RGB and LiDAR sensors, whereas MMGait extends coverage to 5 sensor types and 12 modalities.
- **vs. CASIA-B**: CASIA-B is a classic benchmark but contains only RGB data at a relatively small scale (124 subjects); MMGait represents a qualitative leap in the multi-modal dimension.
- **vs. FreeGait**: FreeGait also employs RGB and LiDAR but covers only a single viewpoint, whereas MMGait provides full coverage across 10 viewpoints.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The first 5-sensor, 12-modality gait benchmark; the omni-modal recognition task definition is pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Systematic evaluation across single-modal, cross-modal, and multi-modal dimensions with multi-method comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Dataset description is detailed and evaluation logic is clear.
- Value: ⭐⭐⭐⭐⭐ — Provides a landmark benchmark resource for the multi-modal gait recognition community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GaitSnippet: Gait Recognition Beyond Unordered Sets and Ordered Sequences](../../ICLR2026/human_understanding/gaitsnippet_gait_recognition_beyond_unordered_sets_and_ordered_sequences.md)
- [\[ICCV 2025\] CarGait: Cross-Attention based Re-ranking for Gait Recognition](../../ICCV2025/human_understanding/cargait_cross_attention_based_re_ranking_for_gait_recognition.md)
- [\[CVPR 2026\] OpenFS: Multi-Hand-Capable Fingerspelling Recognition with Implicit Signing-Hand Detection and Frame-Wise Letter-Conditioned Synthesis](openfs_multi-hand-capable_fingerspelling_recognition_with_implicit_signing-hand_.md)
- [\[CVPR 2026\] Sketch2Colab: Sketch-Conditioned Multi-Human Animation via Controllable Flow Distillation](sketch2colab_sketch-conditioned_multi-human_animation_via_controllable_flow_dist.md)
- [\[CVPR 2026\] GazeOnce360: Fisheye-Based 360° Multi-Person Gaze Estimation with Global-Local Feature Fusion](gazeonce360_fisheye-based_360_multi-person_gaze_estimation_with_global-local_fea.md)

</div>

<!-- RELATED:END -->
