---
title: >-
  [Paper Note] Marker-Based 3D Reconstruction of Aggregates with a Comparative Analysis of 2D and 3D Morphologies
description: >-
  [CVPR 2026][3D Vision][3D reconstruction] This paper proposes a low-cost marker-based photogrammetric pipeline for high-quality 3D reconstruction of aggregate particles. Through a systematic comparative analysis of 2D an…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "3D reconstruction"
  - "photogrammetry"
  - "aggregate morphology"
  - "point cloud"
  - "marker-based"
  - "2D-3D comparison"
date: 2026-05-08
content_hash: 0c585cc05eb75a44
---

# Marker-Based 3D Reconstruction of Aggregates with a Comparative Analysis of 2D and 3D Morphologies

**Conference**: CVPR 2026
**arXiv**: [2603.12667](https://arxiv.org/abs/2603.12667)  
**Authors**: Haohang Huang, Jiayi Luo, Issam Qamhia, Erol Tutumluer, John M. Hart, Andrew J. Stolba
**Area**: Medical Image
**Keywords**: 3D reconstruction, photogrammetry, aggregate morphology, point cloud, marker-based, 2D-3D comparison

## TL;DR

This paper proposes a low-cost marker-based photogrammetric pipeline for high-quality 3D reconstruction of aggregate particles. Through a systematic comparative analysis of 2D and 3D morphological indices, it reveals the significant limitations of relying solely on 2D images for aggregate morphology assessment.

## Background & Motivation

Aggregates serve as a core skeletal material in construction and transportation infrastructure, with broad applications in pavement base layers, railway ballast, cement concrete, asphalt concrete, and riprap. Size and morphology information of aggregate particles is critical for quality assurance/quality control (QA/QC), as morphology directly governs the mechanical behavior of aggregates during compaction and packing.

### Limitations of Prior Work

- **2D imaging methods**: Systems such as AIMS (Aggregate Imaging System) and E-UIAIA analyze only particle silhouettes, reducing three-dimensional morphology to two-dimensional projections and discarding substantial spatial information. Three-dimensional attributes such as flatness and ellipsoidal characteristics cannot be accurately retrieved from a single projection.
- **3D scanning methods**: 3D laser scanners and X-ray Computed Tomography (CT) equipment can provide complete three-dimensional information, but are prohibitively expensive (on the order of hundreds of thousands of dollars), operationally complex, and unsuitable for field deployment.
- **Key Challenge**: Industry requires low-cost, easily deployable solutions that provide complete 3D morphological information, whereas existing methods suffer from either insufficient accuracy (2D) or excessive cost (3D scanning).

### Design Motivation

This work leverages photogrammetry — which reconstructs 3D models from multi-angle images captured with an ordinary camera — combined with a marker-based design to address three key challenges: background suppression, point cloud registration, and scale calibration. The goal is to achieve low-cost, high-quality 3D reconstruction of aggregates and to quantitatively characterize the discrepancies between 2D and 3D morphological measurements.

## Method

### Overall Architecture

The proposed pipeline consists of four core stages: image acquisition → marker-assisted processing → 3D reconstruction → morphological analysis.

### 1. Marker-Based Design

The marker system serves three simultaneous functions:
- **Background suppression**: By placing aggregates on a marker board with a known pattern, the markers enable automatic segmentation of foreground aggregate from background, preventing background texture from interfering with reconstruction quality.
- **Point cloud registration**: Multi-view imaging produces multiple local point clouds; the markers provide stable feature points for accurate alignment and merging of point clouds across different viewpoints.
- **Scale calibration**: Known physical distances between markers provide an absolute scale reference, converting dimensionless reconstruction results into real physical dimensions.

### 2. Multi-View Image Acquisition and SfM Reconstruction

- An ordinary digital camera is used to capture images of aggregate samples from multiple angles.
- Structure-from-Motion (SfM) is applied to recover camera poses and generate sparse point clouds from the multi-view images.
- Multi-View Stereo (MVS) is subsequently applied to densify the sparse point cloud.
- Marker-assisted background masks ensure that only points corresponding to the aggregate region are retained.

### 3. Point Cloud Post-Processing and Mesh Generation

- Markers guide rigid-body transformation for registration of multiple point cloud sets.
- Noise points and outliers are removed.
- A closed triangular mesh is generated from the dense point cloud via Poisson surface reconstruction or an equivalent algorithm.
- Scale calibration rescales the model to true physical dimensions based on inter-marker distances.

### 4. Morphological Index Extraction

Complete 3D morphological indices are extracted from the reconstructed models:
- **Dimensional parameters**: longest axis $L$, intermediate axis $I$, shortest axis $S$
- **Flatness and elongation**: $\text{Flatness} = S/I$, $\text{Elongation} = I/L$
- **Sphericity**: $\psi = \frac{\pi^{1/3}(6V)^{2/3}}{A}$, where $V$ is volume and $A$ is surface area
- **Angularity**: quantified via 3D curvature distribution to characterize the sharpness of particle edges
- **Surface Texture (ST)**: quantified through surface roughness metrics

Corresponding 2D indices are simultaneously extracted from 2D projections for comparative analysis.

## Key Experimental Results

### Reconstruction Accuracy Validation

Multiple types of aggregate samples are selected, and reconstruction results are validated against ground truth obtained via precision calipers or CT scanning.

| Validation Metric | Reconstruction vs. Ground Truth Error | Notes |
|---|---|---|
| Dimensional accuracy (L/W/H) | < 2% relative error | High accuracy attributed to marker-based scale calibration |
| Volume accuracy | < 5% relative error | Good agreement between closed-mesh volume and CT volume |
| Surface area accuracy | < 5% relative error | Dense point cloud captures surface details reliably |
| Sphericity consistency | Highly correlated ($r > 0.95$) | 3D shape descriptors are reliably reconstructed |

### 2D vs. 3D Morphological Comparison

For the same set of aggregate samples, 2D projection morphological indices and 3D reconstruction morphological indices are computed and statistically compared.

| Morphological Index | 2D Estimate | 3D Estimate | Significance |
|---|---|---|---|
| Flatness | Systematically overestimated | Ground truth | **Significant** ($p < 0.05$) |
| Elongation | Systematic bias present | Ground truth | **Significant** |
| Sphericity | 2D approximation deviates up to ~15–20% | Exact computation | **Significant** |
| Angularity | Highly dependent on projection direction | Direction-independent | **Significant** |
| Surface Texture (ST) | Cannot be accurately retrieved | Fully quantifiable | — |

**Key Findings**: 2D projection methods exhibit statistically significant discrepancies from 3D ground truth across all morphological indices. Sphericity and flatness show the largest estimation errors under 2D methods, as these indices are inherently dependent on three-dimensional spatial information. Angularity estimated from 2D projections is highly sensitive to the projection direction; different projection angles of the same particle can yield angularity variations exceeding 30%.

## Highlights & Insights

- **Low-cost 3D solution**: Only an ordinary digital camera and a marker board are required, at a cost far below that of 3D laser scanners (tens of thousands of dollars) or CT equipment (hundreds of thousands of dollars), making 3D morphological analysis accessible at quarries and construction sites.
- **Triple-function marker design**: A single marker system simultaneously addresses background segmentation, point cloud registration, and scale calibration — an elegant and efficient design.
- **Quantitative evidence for 2D vs. 3D discrepancy**: This work provides the first systematic quantitative demonstration, in the aggregate domain, of statistically significant differences between 2D morphological indices and 3D ground truth, supplying data-driven support for the industry's transition from 2D to 3D standards.
- **Strong engineering practicality**: The method directly targets QA/QC workflows and can be applied to aggregate inspection, data collection, and morphological analysis, offering clear practical engineering value.

## Limitations & Future Work

- **Limited automation**: The current pipeline still requires manual placement of markers and manual multi-angle image capture, and does not constitute a fully automated workflow.
- **Low processing throughput**: Each aggregate sample requires multi-angle capture and reconstruction, making processing substantially slower than 2D imaging methods and unsuitable for high-volume online inspection.
- **Occlusion and concave regions**: Photogrammetric reconstruction quality degrades in deeply concave or severely self-occluded regions, potentially missing surface details.
- **Environmental dependency**: Unstable lighting conditions in field settings may adversely affect feature matching and reconstruction quality.
- **Absence of deep learning integration**: The method relies entirely on traditional geometric approaches and does not explore potential accuracy and efficiency gains from deep learning techniques such as NeRF or 3D Gaussian Splatting.
- **Limited sample scale**: The number of aggregate samples in the validation experiments is relatively small, and the representativeness of large-scale statistical conclusions warrants further strengthening.

## Related Work & Insights

- **2D aggregate imaging**: The AIMS system analyzes aggregate silhouette flatness, elongation, and angularity via three orthogonal cameras; E-UIAIA enables online analysis from 2D images on a conveyor belt. These methods are fast but discard three-dimensional information.
- **3D scanning methods**: 3D laser scanning yields high-accuracy point clouds but requires expensive and complex equipment; X-ray CT captures complete internal and external surfaces but is extremely costly and subject to radiation constraints.
- **Photogrammetry**: SfM + MVS technology has been widely adopted in cultural heritage preservation, topographic mapping, and related fields, but systematic application to aggregates with rigorous morphological validation remains scarce.
- **Positioning of this work**: By leveraging marker-based design to address the critical technical gaps of photogrammetry in aggregate settings (background segmentation, registration, and scale), this paper additionally provides the first systematic 2D-3D morphological comparison in this domain.

## Rating

- Novelty: ⭐⭐⭐ — Marker-assisted photogrammetry is not an entirely novel concept, but its systematic application to the aggregate domain and the triple-function marker design constitute a methodological contribution.
- Experimental Thoroughness: ⭐⭐⭐ — Accuracy validation and 2D-3D comparison are provided, but the sample scale is limited and direct comparison with other 3D methods is absent.
- Writing Quality: ⭐⭐⭐⭐ — The structure is clear, problem motivation is well articulated, and engineering application value is thoroughly discussed.
- Value: ⭐⭐⭐ — The work offers practical value for aggregate engineering, though its methodological contribution to the CV community is relatively limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fall Risk and Gait Analysis using World-Spaced 3D Human Mesh Recovery](fall_risk_gait_analysis_hmr.md)
- [\[CVPR 2026\] Learning 3D Reconstruction with Priors in Test Time](tco_learning_3d_reconstruction_with_priors_in_test_time.md)
- [\[CVPR 2026\] Beyond Geometry: Artistic Disparity Synthesis for Immersive 2D-to-3D](beyond_geometry_artistic_disparity_synthesis_for_immersive_2d-to-3d.md)
- [\[CVPR 2026\] FF3R: Feedforward Feature 3D Reconstruction from Unconstrained Views](ff3r_feedforward_feature_3d_reconstruction_from_unconstrained_views.md)
- [\[CVPR 2026\] Speed3R: Sparse Feed-forward 3D Reconstruction Models](speed3r_sparse_feed-forward_3d_reconstruction_models.md)

</div>

<!-- RELATED:END -->
