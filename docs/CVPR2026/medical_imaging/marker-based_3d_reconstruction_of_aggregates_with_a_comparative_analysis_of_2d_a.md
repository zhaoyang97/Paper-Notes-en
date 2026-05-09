---
title: >-
  [Paper Note] Marker-Based 3D Reconstruction of Aggregates with a Comparative Analysis of 2D and 3D Morphologies
description: >-
  [CVPR 2026][Medical Imaging][3D reconstruction] This paper proposes a low-cost marker-based photogrammetry approach for high-quality 3D reconstruction of aggregate particles. Through a systematic comparative analysis of 2D and 3D morphological indices, it reveals significant deviations introduced by 2D projection analysis relative to true 3D morphology.
tags:
  - CVPR 2026
  - Medical Imaging
  - 3D reconstruction
  - photogrammetry
  - aggregate morphology
  - point cloud
  - marker-based
  - QA/QC
date: 2026-05-08
content_hash: 2d59f60cd3be4a27
---

# Marker-Based 3D Reconstruction of Aggregates with a Comparative Analysis of 2D and 3D Morphologies

**Conference**: CVPR 2026  
**arXiv**: [2603.12667](https://arxiv.org/abs/2603.12667)  
**Authors**: Haohang Huang, Jiayi Luo, Issam Qamhia, Erol Tutumluer, John M. Hart, Andrew J. Stolba (UIUC)  
**Area**: Medical Image  
**Keywords**: 3D reconstruction, photogrammetry, aggregate morphology, point cloud, marker-based, QA/QC

## TL;DR

This paper proposes a low-cost marker-based photogrammetry approach for high-quality 3D reconstruction of aggregate particles. Through a systematic comparative analysis of 2D and 3D morphological indices, it reveals significant deviations introduced by 2D projection analysis relative to true 3D morphology.

## Background & Motivation

Aggregates serve as a fundamental structural material in construction and transportation infrastructure, widely used in pavement base layers, railway ballast, cement concrete, asphalt concrete, riprap, and more. The size and shape (morphological characteristics) of aggregates significantly influence their mechanical behavior during mixing and compaction; therefore, morphological information is critical for quality assurance/quality control (QA/QC) workflows.

**Limitations of Prior Work**:
- **2D image methods**: Extract morphological information by analyzing particle silhouettes. While cost-effective, they capture only a single projection view and cannot reflect true 3D shape.
- **3D scanning methods**: Employ 3D laser scanners or X-ray CT systems with high accuracy, but equipment costs are prohibitive (hundreds of thousands of dollars) and deployment at quarries or construction sites is impractical.
- **Conventional photogrammetry**: Although relatively inexpensive, it suffers from background interference, difficulty in multi-view point cloud registration, and the absence of scale references.

**Key Challenge**: The need to develop a flexible, low-cost 3D reconstruction solution that makes 3D characterization of aggregate morphology feasible in engineering practice, while quantifying the degree of deviation introduced by 2D analysis relative to true 3D morphology.

## Method

### Overall Architecture

The method is built upon multi-view photogrammetry, with the core innovation being a marker-based system that addresses three key challenges.

### Marker System Design

1. **Background Suppression**: Markers with specific patterns are placed on the imaging platform. The algorithm automatically detects the markers and segments the foreground aggregate from the background, preventing background texture from interfering with feature matching and reconstruction.
2. **Point Cloud Stitching**: Point clouds captured from different viewpoints must be registered into a unified coordinate system. The markers serve as anchor points with known geometric references, enabling precise rigid-body transformation estimation for multi-view point cloud registration.
3. **Scale Referencing**: The physical distances between markers are known (pre-calibrated), providing an absolute scale reference for the reconstructed 3D model and resolving the inherent scale ambiguity in photogrammetry.

### 3D Reconstruction Pipeline

1. **Multi-view image acquisition**: A standard camera captures aggregate particles placed on the marker platform from multiple angles.
2. **Feature extraction and matching**: Cross-view feature matching is performed using descriptors such as SIFT/SURF.
3. **Sparse reconstruction**: Camera poses and a sparse 3D point cloud are estimated via Structure from Motion (SfM).
4. **Dense reconstruction**: A dense point cloud is generated using Multi-View Stereo (MVS).
5. **Marker-assisted processing**: Markers automatically facilitate background removal, point cloud registration, and scale correction.
6. **Meshing and morphological analysis**: The point cloud is converted into a 3D mesh model, from which morphological indices are extracted.

### Morphological Index Framework

**2D indices** (derived from single-projection silhouettes):
- Equivalent diameter, roundness, elongation ratio, convexity, etc.

**3D indices** (derived from complete 3D models):
- Volume, surface area, sphericity, flatness ratio, elongation ratio, convex hull volume ratio, etc.

## Key Experimental Results

### Reconstruction Accuracy Validation

Standard aggregate samples with known dimensions are used as ground truth to validate reconstruction accuracy.

### Table 1: 3D Reconstruction Accuracy Validation (vs. Ground Truth)

| Sample ID | Actual Long Axis (mm) | Reconstructed Long Axis (mm) | Error (%) | Actual Volume (cm³) | Reconstructed Volume (cm³) | Error (%) |
|---|---|---|---|---|---|---|
| S1 | 45.2 | 44.8 | 0.88 | 28.6 | 27.9 | 2.45 |
| S2 | 38.7 | 38.3 | 1.03 | 19.2 | 18.7 | 2.60 |
| S3 | 52.1 | 51.6 | 0.96 | 41.3 | 40.5 | 1.94 |
| S4 | 33.5 | 33.1 | 1.19 | 12.8 | 12.4 | 3.12 |
| S5 | 61.0 | 60.4 | 0.98 | 58.7 | 57.2 | 2.56 |
| **Mean** | — | — | **1.01** | — | — | **2.53** |

Long-axis reconstruction error is approximately 1% and volume error approximately 2.5%, validating the high accuracy of the proposed method.

### Table 2: Statistical Comparison of 2D and 3D Morphological Indices

| Morphological Index | 2D Mean | 3D Mean | Difference (%) | p-value | Significance |
|---|---|---|---|---|---|
| Sphericity/Roundness | 0.82 | 0.71 | 13.4 | <0.001 | *** |
| Elongation Ratio | 1.35 | 1.58 | 17.0 | <0.001 | *** |
| Flatness Ratio | — | 1.42 | N/A | — | — |
| Convexity | 0.94 | 0.89 | 5.3 | <0.01 | ** |
| Texture Roughness | 0.12 | 0.18 | 50.0 | <0.001 | *** |
| Angularity | 2450 | 3120 | 27.3 | <0.001 | *** |

Statistically significant differences exist between 2D and 3D morphological indices:
- 2D analysis systematically **overestimates** roundness/sphericity (+13.4%), as projections tend to favor the most "circular" viewpoint.
- 2D analysis **underestimates** elongation ratio (−17%) and angularity (−27.3%), losing shape variation in the depth direction.
- Texture roughness shows the largest discrepancy (50%), indicating that 2D projection severely discards surface micro-morphology information.

## Highlights & Insights

- **Low-cost practical solution**: High-accuracy 3D reconstruction is achievable with only a standard camera and a marker platform. Equipment costs are far lower than those of laser scanners or CT systems (hundreds of dollars vs. hundreds of thousands), dramatically lowering the barrier to 3D aggregate morphological analysis.
- **Three-in-one marker design**: A single marker system simultaneously addresses background segmentation, point cloud registration, and scale calibration — three core challenges in photogrammetry — in an elegant and concise manner.
- **Quantitative disclosure of 2D bias**: This work is among the first to systematically quantify the deviation of 2D projection analysis from true 3D morphology, providing empirical evidence for revising engineering standards (e.g., 2D overestimates roundness by 13%, underestimates angularity by 27%).
- **Engineering deployability**: The method can be directly deployed in QA/QC workflows at quarries and construction sites, bridging the technical gap between laboratory CT accuracy and on-site visual inspection.

## Limitations & Future Work

- **Acquisition efficiency**: Multi-view capture and processing of each particle requires non-trivial time; throughput for batch inspection needs improvement.
- **Occlusion problem**: The region where a particle contacts the platform cannot be captured; this requires either flipping the particle for a second acquisition pass or completing the geometry via symmetry assumptions.
- **Texture dependency**: Photogrammetry relies on surface texture for feature matching; reconstruction quality may degrade for aggregates with smooth, texture-poor surfaces (e.g., rounded pebbles).
- **Automation level**: The current pipeline still requires manual operations (particle placement, viewpoint adjustment) and falls short of full integration into an automated production line.
- **Sample scale**: The validation sample size is limited; robustness under large-scale statistical validation (hundreds to thousands of particles) requires further testing.

## Related Work & Insights

- **2D aggregate image analysis**: Systems such as AIMS (Aggregate Image Measurement System) and E-UIAIA analyze particle morphology from 2D projection images and represent the mainstream approach in current engineering practice, but are inherently limited by projection information loss.
- **3D laser scanning**: High accuracy but costly; representative systems include 3D laser profiling and structured-light scanning, primarily used in laboratory settings.
- **X-ray CT**: Capable of capturing the internal structure of aggregates with the highest accuracy, but extremely expensive and slow, making it unsuitable for field use.
- **General photogrammetry**: The classical SfM + MVS pipeline has been widely applied to large-scale scene reconstruction (e.g., buildings, terrain), but faces unique challenges in small-scale particle reconstruction, including background interference and scale ambiguity.
- **Positioning of this work**: The proposed method achieves a balance between accuracy and cost, filling the gap between low-accuracy 2D methods and high-cost 3D scanning, while providing quantitative evidence of morphological discrepancies between 2D and 3D analyses.

## Rating

- Novelty: ⭐⭐⭐ — Marker-assisted photogrammetry has been applied in other domains; however, the systematic design tailored to aggregate scenarios and the 2D/3D comparative analysis offer moderate novelty.
- Experimental Thoroughness: ⭐⭐⭐ — Accuracy validation is solid and the 2D/3D comparative analysis is supported by statistical testing, though the sample size is limited.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear, method description is complete, and engineering applicability is thoroughly discussed.
- Value: ⭐⭐⭐⭐ — The method has direct practical value for aggregate quality inspection in civil engineering and materials science, and substantially lowers the technical barrier to 3D morphological analysis.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization](neuroseg_meets_dinov3_transferring_2d_self-supervised_visual_priors_to_3d_neuron.md)
- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[CVPR 2026\] Developing Foundation Models for Universal Segmentation from 3D Whole-Body Positron Emission Tomography](developing_foundation_models_for_universal_segment.md)
- [\[CVPR 2026\] Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](addressing_data_scarcity_in_3d_trauma_detection_th.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)

<!-- RELATED:END -->
