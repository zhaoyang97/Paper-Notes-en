---
title: >-
  [Paper Note] ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users
description: >-
  [ICCV 2025][Medical Imaging][Gait Analysis] This paper presents ProGait—the first multi-purpose video dataset targeting transfemoral amputee prosthesis users—supporting three tasks: video object segmentation, 2D human pose estimation, and gait analysis. Baseline models are provided to demonstrate the dataset's effectiveness in improving prosthesis detection.
tags:
  - ICCV 2025
  - Medical Imaging
  - Gait Analysis
  - Prosthesis Detection
  - Video Dataset
  - Human Pose Estimation
  - Video Object Segmentation
date: 2026-05-08
content_hash: 01a6024d259037d6
---

# ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users

**Conference**: ICCV 2025
**arXiv**: [2507.10223](https://arxiv.org/abs/2507.10223)
**Code**: [https://github.com/pittisl/ProGait](https://github.com/pittisl/ProGait)
**Area**: Medical Imaging
**Keywords**: Gait Analysis, Prosthesis Detection, Video Dataset, Human Pose Estimation, Video Object Segmentation

## TL;DR

This paper presents ProGait—the first multi-purpose video dataset targeting transfemoral amputee prosthesis users—supporting three tasks: video object segmentation, 2D human pose estimation, and gait analysis. Baseline models are provided to demonstrate the dataset's effectiveness in improving prosthesis detection.

## Background & Motivation

### Root Cause

**Root Cause**: **State of the Field**: Prosthetic legs are critical in clinical rehabilitation, and gait analysis is foundational for optimizing prosthesis design and alignment. Traditional gait analysis relies on specialized motion capture systems or wearable sensors, which are expensive, invasive, and confined to laboratory environments. Vision-based machine learning methods offer a scalable, non-invasive alternative; however, existing visual models perform poorly when detecting and analyzing prosthesis users—primarily because training data is sourced almost entirely from able-bodied populations, rendering these models unable to handle the unique appearance and motion patterns of prosthetic limbs.

The core motivation of this paper is: **to fill the data gap in visual analysis of prosthesis users and provide dedicated training and evaluation resources for visual models**.

## Method

### Overall Architecture

ProGait is a multi-purpose video dataset supporting three core tasks:
1. **Video Object Segmentation (VOS)**: Detecting and tracking the full body of prosthesis users
2. **2D Human Pose Estimation (HPE)**: Detecting 23 keypoints including prosthetic components
3. **Gait Analysis (GA)**: Classifying gait patterns into 9 categories

### Key Designs

1. **Data Collection**: 412 video clips from 4 transfemoral amputees, each testing multiple newly fitted prostheses. Two scenarios are covered (independent walking within parallel bars + assisted walking outside bars), with both frontal and sagittal views captured simultaneously per trial. Resolution is 1920×1080 at 30fps. Diverse prosthesis types (mechanical knee joints / hydraulic / computer-controlled) ensure variability in gait patterns.

2. **Annotation Pipeline**: A Human-in-the-Loop semi-automatic annotation scheme is adopted.

    - VOS: GroundingDINO + SAM2 generate initial segmentations; annotators manually correct tracking failure frames.
    - HPE: Manual correction on ~100 frames → fine-tune RTMW → inference on 1,000 frames → second-round fine-tuning → full-data annotation; only 25% of videos require manual correction.
    - GA: Rehabilitation science researchers provide text descriptions covering gait category, deviations, prosthesis adjustment recommendations, and rationale.

3. **Privacy Protection**: GroundingDINO + SAM2 are used to detect sensitive elements (faces, identifiable markers), which are then processed with Gaussian blurring.

### Loss & Training

- VOS baseline: YOLO11 fine-tuned to learn discrete body parts separately before merging masks
- HPE baseline: RTMPose fine-tuned with a two-stage iterative fine-tuning strategy
- GA baseline: 128-dimensional LSTM classifier taking $(x, y)$ coordinate time series of 12 lower-body keypoints as input
- Data split: ~70% training / ~20% validation / ~10% test; test-set subjects do not appear in the training set

## Key Experimental Results

### Main Results (Benchmarks Across Three Tasks)

**Video Object Segmentation (mIoU)**:

| Method | Overall mIoU | Within Bars | Outside Bars |
|------|---------|------|------|
| YOLO11 (pretrained) | 0.784 | 0.831 | 0.774 |
| Grounded SAM2 ("a person") | 0.358 | 0.643 | 0.559 |
| Grounded SAM2 ("amputee") | 0.905 | 0.900 | 0.907 |
| **YOLO11-ProGait** | **0.847** | **0.815** | **0.866** |

**2D Human Pose Estimation (AP@[.5,.95])**:

| Method | Overall AP | Within Bars | Outside Bars |
|------|-------|------|------|
| HRNet | 0.750 | 0.825 | 0.733 |
| ViTPose | 0.830 | 0.845 | 0.822 |
| RTMPose | 0.855 | 0.876 | 0.850 |
| **RTMPose-ProGait** | **0.947** | **0.968** | **0.942** |

### Ablation Study (Gait Classification)

| Input Configuration | Top-1 Acc | Balanced Acc |
|---------|----------|-------------|
| Frontal view | 0.510 | 0.545 |
| **Sagittal view** | **0.826** | **0.790** |
| Within bars | 0.364 | 0.437 |
| Outside bars | 0.486 | 0.320 |
| All 23 keypoints | 0.372 | 0.403 |
| 12 lower-body keypoints only | 0.384 | 0.413 |

The sagittal view substantially outperforms the frontal view; however, combining both views leads to a performance drop. Lower-body keypoints alone are sufficient for gait classification.

### Key Findings

- Fine-tuned RTMPose-ProGait achieves approximately 29.3% improvement in lower-body keypoint AP (0.625→0.918 vs. HRNet), demonstrating the significant benefit of prosthesis-specific datasets for detection.
- Grounded SAM2 is highly prompt-sensitive: using "a person" yields only 0.358 mIoU, while the correct prompt achieves 0.905, though occasional tracking loss of the prosthesis persists.
- A simple LSTM classifier achieves 82.6% accuracy on sagittal gait classification, remaining competitive with dedicated methods such as GaitGraph2.
- Multiple gait recognition methods (GaitGraph2 / GPGait / GaitBase) can be adapted to ProGait through straightforward fine-tuning.

## Highlights & Insights

- **Filling the Data Gap**: The first multi-purpose visual dataset targeting prosthesis users, with clear clinical value.
- **Efficient Semi-Automatic Annotation Pipeline**: The two-stage iterative fine-tuning strategy reduces manual annotation effort for HPE to below 25%.
- **Clinical Insights**: Findings such as sagittal superiority over frontal view and the sufficiency of lower-body keypoints are consistent with clinical gait analysis practice.
- **IRB Approval + Privacy Protection**: A complete ethical workflow enhances the academic credibility of the dataset.

## Limitations & Future Work

- Only 4 subjects are included; the sample size is limited due to the highly specialized and vulnerable population, for whom recruitment and testing costs are high.
- Gait category distribution is imbalanced (ranging from 4 to 41 samples per class), affecting classification generalizability.
- 3D pose estimation is not supported.
- Video length varies considerably (2–40 seconds), which may affect model consistency.
- Gait classification considers only primary deviations, whereas multiple co-occurring deviations are common in clinical settings.

## Related Work & Insights

- Existing gait datasets such as GAVD and Health&Gait lack representation of prosthesis users.
- SAM2's zero-shot tracking capability is impressive but prompt-sensitive; dedicated fine-tuning remains valuable.
- Future directions include leveraging LLMs with text descriptions to enable automated gait assessment and prosthesis adjustment recommendations.

## Rating

- Novelty: ⭐⭐⭐⭐ First prosthetic gait video dataset, filling an important gap
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete benchmarks across three tasks with multiple comparison methods
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-contextualized clinical background
- Value: ⭐⭐⭐⭐ Directly advances assistive technology research for prosthesis users, though dataset scale remains limited

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] RAM-W600: A Multi-Task Wrist Dataset and Benchmark for Rheumatoid Arthritis](../../NeurIPS2025/medical_imaging/ram-w600_a_multi-task_wrist_dataset_and_benchmark_for_rheumatoid_arthritis.md)
- [\[ICCV 2025\] PVChat: Personalized Video Chat with One-Shot Learning](pvchat_personalized_video_chat_with_one-shot_learning.md)
- [\[ICCV 2025\] SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications](scivid_cross-domain_evaluation_of_video_models_in_scientific_applications.md)
- [\[NeurIPS 2025\] A Unified Solution to Video Fusion: From Multi-Frame Learning to Benchmarking](../../NeurIPS2025/medical_imaging/a_unified_solution_to_video_fusion_from_multi-frame_learning_to_benchmarking.md)
- [\[NeurIPS 2025\] Care-PD: A Multi-Site Anonymized Clinical Dataset for Parkinson's Disease Gait Assessment](../../NeurIPS2025/medical_imaging/care-pd_a_multi-site_anonymized_clinical_dataset_for_parkinsons_disease_gait_ass.md)

<!-- RELATED:END -->
