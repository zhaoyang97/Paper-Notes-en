---
title: >-
  [Paper Note] Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties
description: >-
  [Object Detection] This work proposes MulSen-AD, the first industrial object anomaly detection dataset that integrates three sensors: RGB cameras, laser scanners, and infrared thermography (covering 15 product categories and 14 types of anomalies). Additionally, a decision-level fusion baseline method, MulSen-TripleAD, is designed, achieving a 96.1% AUROC and demonstrating that multi-sensor fusion significantly outperforms single-sensor approaches.
tags:
  - "Object Detection"
date: 2026-05-08
content_hash: e992d9ba625f8b81
---

# Multi-Sensor Object Anomaly Detection: Unifying Appearance, Geometry, and Internal Properties

## TL;DR

This work proposes MulSen-AD, the first industrial object anomaly detection dataset that integrates three sensors: RGB cameras, laser scanners, and infrared thermography (covering 15 product categories and 14 types of anomalies). Additionally, a decision-level fusion baseline method, MulSen-TripleAD, is designed, achieving a 96.1% AUROC and demonstrating that multi-sensor fusion significantly outperforms single-sensor approaches.

## Background & Motivation

Key Challenge of industrial anomaly detection:

1. **A single sensor cannot cover all anomaly types**:
    - RGB cameras excel at detecting surface defects (scratches, color anomalies) but cannot identify internal defects.
    - Laser scanners can capture geometric deformations (bends, wrinkles) but are insensitive to texture and color.
    - Infrared thermography can reveal internal defects (internal cracks, detached parts) but sacrifices color and texture information.

2. **Limitations of Prior Work**:
    - Datasets like MVTec-AD and VisA only use RGB images.
    - MVTec3D-AD uses RGB-D, but the quality of depth maps is limited.
    - Real3D-AD only uses point clouds.
    - No existing dataset provides high-quality annotations across all three modalities simultaneously.

3. **The "blind spot" issue in anomaly detection**: Single-sensor methods often miss detections in real factory scenarios—for instance, internal fractures in capsules are completely invisible under RGB, while spring pad deformations are difficult to detect via infrared.

## Method

### Overall Architecture

The MulSen-AD framework consists of two parts:
1. **MulSen-AD Dataset**: Data acquisition, annotation, and organization for the three sensors.
2. **MulSen-TripleAD Baseline Model**: A PatchCore-based decision-level fusion method for multi-sensor data.

### Key Designs

#### 1. Multi-Sensor Data Acquisition System

- **RGB Camera** (Daheng MER2-230-168U3C): 1920×1200 resolution, mounted on a UR5 robotic arm, paired with side light sources to ensure uniform illumination.
- **Infrared Thermography** (Noverlteq TWILIS-180 + FLIR A600): 640×480 resolution, 7.5-14μm wavelength range, detecting temperature anomalies via periodic thermal stimulation (30–180 seconds).
- **3D Laser Scanner** (Creaform MetraSCAN 750): 0.03mm accuracy, 0.05mm resolution, handheld 360° scanning, flip dual-side scanning + ICP fine alignment.

The 15 categories of industrial products cover metals, plastics, fibers, rubber, semiconductors, and composite materials. The 14 anomaly types include cracks, holes, fractures, wrinkles, scratches, foreign objects, mislabeling, bending, color defects, and detachment.

#### 2. Modality-Specific Annotation Strategy

Annotation follows the "annotate what is visible" principle:
- If an anomaly is only visible in the infrared image (e.g., internal capsule fracture), only the infrared image is annotated.
- Use LabelMe to annotate pixel-level masks for RGB and infrared images.
- Use Geomagic Design X to manually select the anomalous regions in point clouds.

#### 3. MulSen-TripleAD Decision-Level Fusion

Three components:
- **Multimodal Feature Extraction**: DINO extracts RGB and infrared features, and PointMAE extracts point cloud features.
- **Multimodal Memory Bank**: Establishes normal-sample-based PatchCore memory banks $\mathcal{M}_{rgb}$, $\mathcal{M}_{ir}$, and $\mathcal{M}_{pc}$ respectively for RGB, infrared, and point cloud modalities.
- **Decision Gating Unit**: A learnable OCSVM inspired by M3DM that fuses the anomaly scores from the three modalities:

$$S = \mathcal{G}_a(\phi(\mathcal{M}_{rgb}, f_{rgb}), \phi(\mathcal{M}_{pt}, f_{pt}), \phi(\mathcal{M}_{ir}, f_{ir}))$$

### Loss & Training

MulSen-TripleAD relies on distance-metric anomaly scoring from PatchCore and utilizes an OCSVM for decision fusion. The anomaly score for each modality is the $L_2$ distance from the test features to the nearest neighbor in the memory bank.

## Key Experimental Results

### Main Results

**Object-Level Anomaly Detection AUROC (Multi-Sensor vs. Single-Sensor)**:

| Method | Modality | Mean AUROC ↑ |
|------|------|-------------|
| PatchCore | RGB | 0.837 |
| InvAD | RGB | 0.892 |
| PatchCore | IR | 0.843 |
| InvAD | IR | 0.832 |
| M3DM | RGB+PC | 0.830 |
| **MulSen-TripleAD** | **RGB+IR+PC** | **0.961** |

**Detailed Performance per Category (Partial)**:

| Category | Best RGB | Best IR | Best PC | MulSen-TripleAD |
|------|---------|--------|--------|-----------------|
| Capsule | 0.940 | 0.960 | 0.923 | High Fusion |
| Screen | 0.884 | 0.572 | 0.788 | — |
| Solar Panel | 0.947 | 0.867 | 0.378 | — |
| Spring Pad | 0.980 | 0.913 | 0.512 | — |

### Key Findings

1. **Multi-sensor fusion significantly outperforms single-sensor methods**: 96.1% vs. the best single-sensor baseline of 89.2% (InvAD on RGB), a gain of nearly 7 percentage points.
2. **Complementarity of anomaly distributions**: The Venn diagram shows that while 43.7% of anomalies can be detected by all three sensors simultaneously, 9.4% (RGB-exclusive), 9.2% (infrared-exclusive), and 4.3% (point cloud-exclusive) of anomalies can only be captured by a single sensor.
3. **Different categories require different sensors**: Anomalies on Screens are primarily detected via RGB (0.884 vs. 0.572 for IR), whereas infrared is more effective for Solar Panel anomalies.
4. **Existing RGB-only methods perform poorly on infrared and point clouds**: For example, CFA achieves an average AUROC of only 0.584 on infrared data.

### Dataset Statistics

- Total samples: 2035 (1391 training + 644 testing)
- 15 categories, with an average of 4.8 anomaly types per category.
- Anomaly pixel ratio: mean of 0.372% in RGB, 0.451% in infrared, and 4.98% in point clouds.

## Highlights & Insights

1. **Fills the dataset gap in multi-sensor anomaly detection**: Previously, no dataset existed that simultaneously provided RGB, infrared, and high-precision point clouds for anomaly detection. The release of MulSen-AD enables research in this domain.
2. **Reasonableness of the modality-specific annotation strategy**: The "annotate what is visible" principle avoids the irrationality of forcing aligned annotations across all three modalities—as a surface scratch might be completely invisible in infrared.
3. **Simplicity and effectiveness of decision-level fusion**: MulSen-TripleAD performs fusion only at the final score level, processing each modality independently, which facilitates easy scaling to new sensors.
4. **Real instead of synthetic anomalies**: The 14 manually introduced anomaly types cover a broad range, and 15 representative industrial product categories are selected based on material properties.

## Limitations & Future Work

1. **Supports only object-level detection**: Current datasets and methods are primarily tailored for object-level anomaly detection, with limited capability for pixel-level localization.
2. **High annotation cost**: Data collection and annotation across three modalities require specialized equipment and significant manual labor.
3. **Overly simplistic decision-level fusion**: MulSen-TripleAD does not exploit cross-modal relationships, where feature-level or data-level fusion might be more effective.
4. **Limited data scale**: 15 categories with 2035 samples remain small compared to actual industrial demands.
5. **Sensor alignment issue**: Data from the three sensors are not spatially aligned at the pixel level, limiting the application of pixel-level fusion methods.

## Related Work & Insights

- **PatchCore** [Roth et al., 2022]: A classic memory-bank-based method for anomaly detection; MulSen-TripleAD directly extends it to multimodal settings.
- **M3DM** [Wang et al., 2023]: The first 3D anomaly detection method using RGB + point clouds, employing a learnable OCSVM for fusion.
- **MVTec-AD** [Bergmann et al., 2019]: The most classic RGB anomaly detection dataset; MulSen-AD serves as its multi-sensor counterpart.
- **Real3D-AD** [Liu et al., 2023]: A pure point cloud anomaly detection dataset, but lacks appearance and internal information.
- Insight: There is "no universal sensor" in industrial inspection, and multi-source information fusion is an inevitable path from the laboratory to real-world factories.

## Rating

⭐⭐⭐⭐ (8/10)

- Novelty: ⭐⭐⭐⭐ — The first tri-modal industrial anomaly detection dataset with a clear and important problem definition.
- Value: ⭐⭐⭐⭐⭐ — Directly targets industrial quality inspection needs, with all three sensors widely applied in actual factories.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive testing on the dataset's own benchmark, but lacks generalization validation on other datasets.
- Writing Quality: ⭐⭐⭐⭐ — Detailed dataset construction details contrast with a relatively concise method section (which is reasonable for a dataset paper).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](../../ICLR2026/object_detection/towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)
- [\[CVPR 2025\] MulSen-AD: Multi-Sensor Object Anomaly Detection](mulsen_ad_multi_sensor_anomaly_detection.md)
- [\[ICLR 2026\] PGRF-Net: A Prototype-Guided Relational Fusion Network for Diagnostic Multivariate Time-Series Anomaly Detection](../../ICLR2026/object_detection/pgrf-net_a_prototype-guided_relational_fusion_network_for_diagnostic_multivariat.md)
- [\[CVPR 2025\] MI-DETR: An Object Detection Model with Multi-time Inquiries Mechanism](mi-detr_an_object_detection_model_with_multi-time_inquiries_mechanism.md)
- [\[CVPR 2025\] Mr. DETR++: Instructive Multi-Route Training for Detection Transformers with MoE](mr_detr_instructive_multi-route_training_for_detection_transformers.md)

</div>

<!-- RELATED:END -->
