---
title: >-
  [Paper Note] MulSen-AD: Multi-Sensor Object Anomaly Detection
description: >-
  [CVPR 2025][Object Detection][Multi-Sensor Fusion] The first multi-sensor anomaly detection dataset MulSen-AD is proposed, integrating RGB camera, infrared lock-in thermography, and laser scanning modalities, alongside a baseline method MulSen-TripleAD, achieving 96.1% AUROC in object-level anomaly detection through decision-level fusion.
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Multi-Sensor Fusion"
  - "Anomaly Detection"
  - "RGB-IR-Point Cloud"
  - "Industrial Inspection"
  - "Dataset"
date: 2026-05-08
content_hash: 4449b10cb14fa255
---

# MulSen-AD: Multi-Sensor Object Anomaly Detection

**Conference**: CVPR 2025  
**arXiv**: [2412.14592](https://arxiv.org/abs/2412.14592)  
**Code**: [https://github.com/ZZZBBBZZZ/MulSen-AD](https://github.com/ZZZBBBZZZ/MulSen-AD)  
**Area**: Object Detection / Anomaly Detection  
**Keywords**: Multi-Sensor Fusion, Anomaly Detection, RGB-IR-Point Cloud, Industrial Inspection, Dataset

## TL;DR

The first multi-sensor anomaly detection dataset MulSen-AD is proposed, integrating RGB camera, infrared lock-in thermography, and laser scanning modalities, alongside a baseline method MulSen-TripleAD, achieving 96.1% AUROC in object-level anomaly detection through decision-level fusion.

## Background & Motivation

**Background**: Industrial anomaly detection mainly relies on single-sensor methods; RGB focuses on surface defects but cannot detect internal defects, 3D scanning captures geometric anomalies but ignores heating anomalies, and infrared detects internal defects but lacks color and texture information.

**Limitations of Prior Work**: Existing datasets (e.g., MVTec-AD, Real3D-AD) only contain a single modality, failing to comprehensively cover various types of anomalies in real-world factories.

**Key Challenge**: Different types of anomalies require different sensors for detection; a single sensor is insufficient for all scenarios.

**Goal**: Construct the first multi-sensor anomaly detection dataset and benchmark, and verify the necessity of multi-sensor fusion.

**Core Idea**: Integrate three modalities—RGB + Infrared + high-precision point cloud—to cover appearance, internal, and geometric anomalies.

## Method

### Overall Architecture

The MulSen-AD dataset contains 15 categories of industrial products, acquired using an industrial RGB camera ($1920 \times 1200$), infrared lock-in thermography ($640 \times 480$), and a Creaform MetraSCAN laser scanner (0.03mm accuracy). The baseline method, MulSen-TripleAD, independently processes the three modalities and performs decision-level gated fusion.

### Key Designs

1. **Multi-sensor Data Acquisition Pipeline**:

    - **Function**: Build high-quality multi-modal anomaly detection data
    - **Mechanism**: Infrared lock-in thermography detects internal anomalies (e.g., damaged capsule interiors, solar panel damage) via periodic thermal excitation; the RGB camera is mounted on a UR5 robotic arm to ensure uniform lighting; a handheld laser scanner performs 360° scans to eliminate blind spots
    - **Design Motivation**: The three sensors are complementary—RGB detects surface details, infrared detects internal states, and point clouds capture geometry

2. **Modality-Specific Annotation**:

    - **Function**: Accurately annotate anomalies visible in each modality
    - **Mechanism**: Annotations are provided for a modality only if the anomaly is visible in it. For example, internal capsule anomalies are annotated only in infrared images, not in RGB or point clouds
    - **Design Motivation**: Different anomalies exhibit varying visibility across different modalities, avoiding mandatory annotation of invisible anomalies

3. **MulSen-TripleAD Decision-level Fusion**:

    - **Function**: Fuse anomaly detection results from the three modalities
    - **Mechanism**: An independent memory bank is established for each modality to perform unsupervised anomaly detection, and anomaly scores from the three modalities are fused in the final stage through a decision gating unit
    - **Design Motivation**: Decision-level fusion minimizes the impact of data discrepancy across sensors and is more robust than feature-level fusion

### Loss & Training

Unsupervised setup, where only normal samples are used for training. About 33 anomaly samples per category are used for testing. The dataset holds a total of 2,035 samples, covering 15 product categories and 14 types of anomalies.

## Key Experimental Results

### Main Results

| Method | Modality | Object-level AUROC |
|------|------|-------------|
| Best Single Modality | RGB/IR/PC | ~90% |
| MulSen-TripleAD | RGB+IR+PC | **96.1%** |

### Key Findings

- Multi-sensor fusion yields a significant improvement compared to any single-sensor method.
- Dependency on different sensors varies across different products.
- The infrared modality is irreplaceable for detecting internal defects.

## Highlights & Insights

- The first industrial anomaly detection dataset containing RGB + Infrared + Point Cloud.
- Manually manufactured 14 types of real anomalies (cracks, holes, bends, foreign objects, etc.).
- The modality-specific annotation strategy respects the boundaries of detection capabilities of each sensor.

## Limitations & Future Work

- The dataset scale is relatively small (2,035 samples).
- Currently, only simple decision-level fusion is used; more complex fusion strategies remain to be explored.
- Acquisition across three sensors requires different equipment and procedures, making the actual deployment cost high.

## Rating

- Novelty: 8/10 — First multi-sensor AD dataset
- Technical Depth: 6/10 — The baseline method is relatively simple
- Experimental Thoroughness: 7/10 — Dataset construction is thorough, but model comparisons are limited
- Writing Quality: 7/10 — Clear and detailed

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties](multi-sensor_object_anomaly_detection_unifying_appearance_geometry_and_internal_.md)
- [\[CVPR 2025\] MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)
- [\[CVPR 2025\] Towards RAW Object Detection in Diverse Conditions](towards_raw_object_detection_in_diverse_conditions.md)
- [\[CVPR 2025\] One-for-More: Continual Diffusion Model for Anomaly Detection](one-for-more_continual_diffusion_model_for_anomaly_detection.md)
- [\[CVPR 2025\] Unseen Visual Anomaly Generation](unseen_visual_anomaly_generation.md)

</div>

<!-- RELATED:END -->
