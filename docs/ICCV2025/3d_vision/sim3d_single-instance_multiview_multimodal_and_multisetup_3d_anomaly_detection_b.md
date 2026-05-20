---
title: >-
  [Paper Note] SiM3D: Single-Instance Multiview Multimodal and Multisetup 3D Anomaly Detection Benchmark
description: >-
  [ICCV 2025][3D Vision][3D anomaly detection] This paper introduces SiM3D, the first benchmark for multiview multimodal 3D anomaly detection and segmentation targeting single-instance industrial scenarios. It employs indu…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D anomaly detection"
  - "multiview"
  - "multimodal"
  - "single-instance"
  - "anomaly volume"
  - "synthetic-to-real"
date: 2026-05-08
content_hash: 620c070318a09172
---

# SiM3D: Single-Instance Multiview Multimodal and Multisetup 3D Anomaly Detection Benchmark

**Conference**: ICCV 2025
**arXiv**: [2506.21549](https://arxiv.org/abs/2506.21549)  
**Code**: [alex-costanzino/SiM3D](https://alex-costanzino.github.io/SiM3D/)  
**Area**: 3D Vision
**Keywords**: 3D anomaly detection, multiview, multimodal, single-instance, anomaly volume, synthetic-to-real

## TL;DR

This paper introduces SiM3D, the first benchmark for multiview multimodal 3D anomaly detection and segmentation targeting single-instance industrial scenarios. It employs industrial-grade sensors to acquire high-resolution data, replaces 2D anomaly maps with voxelized Anomaly Volumes, and is the first benchmark to support cross-domain synthetic-to-real evaluation.

## Background & Motivation

Industrial anomaly detection (ADS) has experienced rapid growth since the release of MVTec AD (172 top-venue papers within five years), yet existing benchmarks leave two critical industrial pain points unresolved:

**2D anomaly maps are insufficient for 3D localization**: Existing benchmarks (MVTec AD/3D-AD, etc.) produce 2D anomaly maps from a single viewpoint. Industrial automation, however, requires precise 3D localization of defects for automated repair, necessitating multi-view observation and anomaly score prediction within a 3D voxel grid—i.e., an *Anomaly Volume*.

**Multi-instance training data collection is costly and unnecessary**: Existing benchmarks require multiple training samples to capture normal-instance variability. In manufacturing, objects are highly consistent (replicated from the same CAD prototype), so a single normal instance contains all information needed for detection. Moreover, recollecting large datasets after a production line changeover is prohibitively time-consuming.

**The synthetic-to-real gap is unexplored**: No prior ADS benchmark has investigated cross-domain generalization from CAD-rendered synthetic training data to real-object testing—a setup of high practical value in industry.

**Core insight**: The paradigm shift from 2D anomaly maps to 3D Anomaly Volumes, combined with single-instance training and synthetic-to-real evaluation, better reflects real industrial requirements. SiM3D addresses this gap through industrial-grade sensors and a carefully designed acquisition pipeline.

## Method

### Overall Architecture

SiM3D contributes a benchmark and adapts existing methods rather than proposing a new algorithm. The framework comprises four core components:

**(1) Data Acquisition**
- **Sensor**: ZEISS Atos Q industrial 3D scanner (stereo pair of 12 Mpx grayscale cameras + light projector) mounted on a high-precision industrial robot arm.
- **Acquisition**: Each object is scanned 360°, covering 12–36 viewpoints sampled from a concentric hemisphere.
- **Output per viewpoint**: grayscale image ($4096 \times 3000$ px), point cloud (~7 M points, 0.04–0.15 mm spacing), and known pose; each scan also provides an integrated mesh.

**(2) Synthetic Data Generation**

Blender's Python API is used to render synthetic data consistent with the real acquisition:
- The real reference mesh is aligned to the CAD model.
- The Cycles path-tracing renderer renders from the same viewpoints.
- RGB images are converted to grayscale to match the real sensor.
- Depth maps are back-projected to 3D point clouds.

**(3) Anomaly Creation**

Three defect categories are manually introduced into purchased objects:
- Appearance anomalies: paint modifications, scratches.
- Geometric anomalies: dents, deformations.
- Mixed anomalies: contaminants (affecting both appearance and geometry).

For each object category, 50% of instances are kept defect-free and 50% are defective.

**(4) 3D Annotation Pipeline**

A two-step strategy integrating 2D and 3D information:

$$\text{2D Annotation} \xrightarrow{\text{Project onto Mesh}} \text{3D Mesh Annotation} \xrightarrow{\text{Manual Refinement}} \text{Voxelized GT (2 mm)}$$

- 2D segmentation masks are manually annotated on all viewpoint images where defects are visible.
- The integrated mesh, intrinsic matrices, and view transforms are used to project 2D annotations onto the 3D mesh.
- CloudCompare is used for 3D visualization and manual refinement.
- Annotations are finally converted to voxel grids at 2 mm resolution.

### Benchmark Design

**Two training setups**:
- **real2real**: Train on 1 real normal instance → test on the remaining real instances.
- **synth2real**: Train on synthetic data rendered from the CAD model → test on real instances.

**Metrics extended to 3D**:
- **Detection**: I-AUROC (image-level AUROC).
- **Segmentation**: Standard 2D ADS metrics are extended to operate on voxel grids (3D versions).

**Adapting single-view methods to multiview**:
1. Run a single-view ADS method independently per viewpoint to obtain 2D anomaly maps.
2. Project 2D anomaly scores into 3D space.
3. Aggregate across viewpoints into a voxel grid (per-voxel max pooling).

### Dataset Statistics

Eight industrial object categories, 333 instances in total:

| Object | Size (cm) | Total | Train | Normal Test | Anomaly Test | Views |
|--------|-----------|-------|-------|-------------|--------------|-------|
| Plastic stool | 35×35×30 | 22 | 1 (real/synth) | 10 | 10 | 12 |
| Trash bin | 26×21×33 | 42 | 1 | 20 | 20 | 12 |
| Rattan vase | 17×17×15 | 22 | 1 | 10 | 10 | 12 |
| Bathroom furniture | 33×33×50 | 20 | 1 | 8 | 10 | 36 |
| Container | 20×25×10 | 94 | 1 | 46 | 46 | 12 |
| Plastic vase | 12×12×9 | 99 | 1 | 48 | 49 | 12 |
| Wooden stool | 48×42×45 | 15 | 1 | 6 | 7 | 12 |
| Washbasin | 44×25×50 | 19 | 1 | 9 | 8 | 36 |

## Key Experimental Results

### Main Results: Anomaly Detection (I-AUROC)

| Method | Modality | real2real Mean | synth2real Mean |
|--------|----------|---------------|-----------------|
| PatchCore (WRN-101) | RGB | 0.630 | 0.600 |
| PatchCore (DINO-v2) | RGB | 0.671 | 0.596 |
| EfficientAD | RGB | 0.594 | 0.573 |
| AST | RGB | **0.687** | **0.679** |
| BTF | RGB+PC | 0.444 | 0.446 |
| M3DM (DINO-v2+FPFH) | RGB+PC | 0.621 | 0.402 |
| AST | RGB+Depth | 0.636 | 0.495 |

**Key Findings**:
- AST with RGB-only modality achieves the best performance under both setups; multimodal methods are consistently outperformed by pure RGB baselines.
- real2real uniformly outperforms synth2real, indicating that the synthetic-to-real domain gap remains a major challenge.
- Multimodal methods such as BTF and CFM degrade severely under the single-instance setting; memory-bank-based methods are unstable with extremely limited training data.

### Anomaly Segmentation (Anomaly Volume)

| Method | Modality | real2real Mean (vAUROC) | synth2real Mean |
|--------|----------|------------------------|-----------------|
| PatchCore (WRN-101) | RGB | 0.754 | 0.451 |
| PatchCore (DINO-v2) | RGB | 0.678 | 0.540 |
| AST | RGB | 0.584 | **0.544** |
| M3DM (DINO-v2+FPFH) | RGB+PC | 0.621 | 0.402 |
| AST | RGB+Depth | **0.925** | 0.495 |

**Key Findings**:
- AST (RGB+Depth) achieves a strong segmentation result under real2real (0.925) but drops substantially under synth2real (0.495).
- PatchCore (WRN-101) performs competitively on real2real segmentation (0.754).
- synth2real segmentation performance is generally poor, suggesting that precise 3D defect localization is more sensitive to domain shift than detection.
- Performance varies greatly across object categories; texturally simple objects (e.g., wooden stool, 1.000) vastly outperform complex ones (e.g., washbasin, 0.250).

## Highlights & Insights

1. **Paradigm shift from 2D anomaly maps to 3D Anomaly Volumes**: The voxelized Anomaly Volume is proposed as the standard output for 3D ADS, more directly supporting industrial automated repair than 2D maps.
2. **Practical value of single-instance + synthetic-to-real settings**: This is the first ADS work to explore cross-domain generalization from CAD-rendered training data to real-object testing, directly addressing the pain point of production line changeovers.
3. **Industrial-grade data quality**: The ZEISS Atos Q scanner and industrial robot arm yield 12 Mpx imagery and ~7 M-point clouds, far exceeding the precision of existing benchmarks.
4. **Exposing critical deficiencies of existing methods**: Multimodal methods paradoxically underperform pure RGB methods in the single-instance setting, and the large performance gap in synthetic-to-real transfer defines a clear direction for future research.

## Limitations & Future Work

- Only eight object categories are included, limiting diversity and suitability for generalization studies.
- Data acquisition and annotation costs are extremely high (industrial-grade equipment, a four-expert team, and a multi-step labeling pipeline), making large-scale expansion difficult.
- The 2 mm voxel resolution is a compromise between accuracy and memory footprint and may miss very fine defects.
- Multi-view information is currently aggregated via simple max pooling; dedicated multi-view fusion methods are absent.

## Related Work & Insights

- **2D ADS benchmarks**: MVTec AD/LOCO AD, VisA, Real-IAD, etc.
- **Multimodal ADS benchmarks**: MVTec 3D-AD (RGB + XYZ maps), Eyecandies (synthetic RGB + depth).
- **Multiview ADS benchmarks**: PAD (multiview RGB training, single-view RGB testing), Real3D-AD (point-cloud anomaly detection).
- **ADS methods**: PatchCore, EfficientAD, M3DM, BTF, CFM, AST, etc.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The 3D Anomaly Volume concept is original; the single-instance + synthetic-to-real setup is insightful; however, no new method is proposed.
- **Technical Quality**: ⭐⭐⭐⭐⭐ — Data acquisition is highly rigorous, the annotation pipeline is scientifically sound, and the benchmark design is comprehensive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Seven methods are adapted across two setups, but dedicated multiview baselines are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear; comparison tables with existing benchmarks are intuitive.
- **Overall Score**: 7.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] G2SF: Geometry-Guided Score Fusion for Multimodal Industrial Anomaly Detection](g2sf_geometry-guided_score_fusion_for_multimodal_industrial_anomaly_detection.md)
- [\[ICCV 2025\] Event-based Tiny Object Detection: A Benchmark Dataset and Baseline](event-based_tiny_object_detection_a_benchmark_dataset_and_baseline.md)
- [\[ICCV 2025\] Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation](bridging_3d_anomaly_localization_and_repair_via_high-quality_continuous_geometri.md)
- [\[ICCV 2025\] GSOT3D: Towards Generic 3D Single Object Tracking in the Wild](gsot3d_towards_generic_3d_single_object_tracking_in_the_wild.md)
- [\[CVPR 2026\] A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection](../../CVPR2026/3d_vision/a_semantically_disentangled_unified_model_for_multi-category_3d_anomaly_detectio.md)

</div>

<!-- RELATED:END -->
