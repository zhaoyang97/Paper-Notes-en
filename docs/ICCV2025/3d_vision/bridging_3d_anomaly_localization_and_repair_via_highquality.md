---
title: >-
  [Paper Note] PASDF: Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation
description: >-
  [ICCV 2025][3D Vision][3D anomaly detection] This paper proposes PASDF, a unified framework that aligns point clouds to a canonical pose via a Pose Alignment Module (PAM), learns a continuous geometric representation through a neural SDF network, and scores anomalies based on SDF deviation. Anomaly repair is achieved by extracting the zero-level set via Marching Cubes as a repair template. PASDF achieves state-of-the-art performance on Real3D-AD (O-AUROC 80.2%) and Anomaly-ShapeNet (O-AUROC 90.0%).
tags:
  - ICCV 2025
  - 3D Vision
  - 3D anomaly detection
  - SDF
  - pose alignment
  - anomaly repair
  - point cloud
date: 2026-05-08
content_hash: 196e4a7fa0e0aee5
---

# PASDF: Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation

**Conference**: ICCV 2025
**arXiv**: [2505.24431](https://arxiv.org/abs/2505.24431)
**Code**: [https://github.com/ZZZBBBZZZ/PASDF](https://github.com/ZZZBBBZZZ/PASDF)
**Area**: 3D Vision / Anomaly Detection / Implicit Representation
**Keywords**: 3D anomaly detection, SDF, pose alignment, anomaly repair, point cloud

## TL;DR
This paper proposes PASDF, a unified framework that aligns point clouds to a canonical pose via a Pose Alignment Module (PAM), learns a continuous geometric representation through a neural SDF network, and scores anomalies based on SDF deviation. Anomaly repair is achieved by extracting the zero-level set via Marching Cubes as a repair template. PASDF achieves state-of-the-art performance on Real3D-AD (O-AUROC 80.2%) and Anomaly-ShapeNet (O-AUROC 90.0%).

## Background & Motivation
Existing 3D anomaly detection methods rely on discrete representations (voxels, points, or projections), which suffer from quantization artifacts and loss of geometric detail. More critically, in practical manufacturing scenarios, anomaly detection alone is insufficient — in-situ repair is also required, yet current methods lack high-fidelity repair capability. Furthermore, arbitrary pose variation at test time causes local-feature-based methods to fail easily.

## Core Problem
How to replace discrete representations with continuous geometric representations while simultaneously achieving pose-invariant 3D anomaly detection and high-quality anomaly repair?

## Method

### Overall Architecture
Input point cloud → PAM pose alignment (FPFH+RANSAC coarse alignment → ICP fine alignment → Chamfer distance feedback iteration) → Query point sampling (surface / interior / exterior) → Positional encoding → SDF network predicts signed distances → At inference: SDF deviation = anomaly score / Marching Cubes extracts zero-level set = repair template.

### Key Designs
1. **PAM Pose Alignment Module**: One normal sample is selected as the canonical pose reference; all training and test point clouds are iteratively aligned to this reference via RANSAC+ICP. A Chamfer-distance-driven dynamic threshold feedback mechanism is introduced to avoid local optima, eliminating the interference of pose variation on anomaly detection.
2. **Continuous SDF Representation**: An 8-layer MLP with skip connections and positional encoding learns the continuous signed distance field of the canonical shape. During training, 23,000 query points are sampled (10,000 surface + 10,000 inside bounding box + 3,000 within unit volume), optimized with a clamped L1 loss.
3. **Anomaly Scoring**: At inference, surface points of the test point cloud are fed into the SDF network; $|f_\theta(x)|$ serves as the anomaly score (normal points yield SDF ≈ 0, while anomalous points yield larger SDF values). The top-$K$ average is used as the object-level score.
4. **Anomaly Repair**: The Marching Cubes algorithm extracts the zero-level set mesh from the trained SDF network, which is then sampled to obtain the repaired point cloud.

### Loss & Training
- Clamped L1 loss: $\mathcal{L}_{\text{SDF}} = \frac{1}{N_q} \sum_i |\text{clamp}(\hat{s}_i, -d_{\max}, d_{\max}) - s_i|$

## Key Experimental Results

### Real3D-AD

| Method | O-AUROC↑ | P-AUROC↑ |
|--------|----------|----------|
| BTF(Raw) | 0.635 | - |
| Reg3D-AD | 0.752 | 0.715 |
| Group3AD | 0.757 | 0.741 |
| **PASDF** | **0.802** | **0.745** |

O-AUROC improves by 5% over the previous SOTA.

### Anomaly-ShapeNet (O-AUROC)

| Method | Mean |
|--------|------|
| IMRNet | 0.640 |
| R3D-AD | 0.843 |
| **PASDF** | **0.900** |

Ranks first in 37 out of 40 categories.

### Ablation Study

| Variant | O-AUROC | P-AUROC |
|---------|---------|---------|
| w/o RANSAC | 0.711 | 0.739 |
| w/o ICP | 0.727 | 0.836 |
| w/o Positional Encoding | 0.887 | 0.783 |
| PASDF (full) | 0.900 | 0.897 |

- PAM yields the largest gain for PatchCore (FPFH): O-AUROC +24.6%, P-AUROC +28.7%.

### Anomaly Repair

| Method | CD↓ | EMD↓ |
|--------|-----|------|
| w/o PE | 0.0255 / 0.0575 | 0.0133 / 0.0276 |
| with PE | 0.0203 / 0.0445 | 0.0110 / 0.0228 |

## Highlights & Insights
- **Unified detection and repair**: PASDF is the first framework to unify 3D anomaly detection and repair under an SDF formulation — detection via SDF deviation scoring, and repair via zero-level set extraction with Marching Cubes.
- **Importance of pose alignment**: The PAM module yields substantial improvements across different baselines (especially PatchCore +24.6%), demonstrating that pose normalization is a critical preprocessing step for 3D anomaly detection.
- **Impact of positional encoding on P-AUROC**: Removing positional encoding causes P-AUROC to drop from 89.7% to 78.3%, indicating that spatial frequency information is essential for fine-grained anomaly localization.

## Limitations & Future Work
- PAM is computationally expensive when the initial pose gap is large.
- The current method assumes a single normal category and has not been extended to multi-class detection.
- The approach is sensitive to point cloud quality (noise and outliers).

## Related Work & Insights
- **vs. Reg3D-AD**: A memory-based method relying on indirect feature mapping; PASDF directly models geometry, achieving +5% O-AUROC.
- **vs. IMRNet**: A reconstruction-based method using discrete point representations, resulting in discontinuous repair templates; PASDF achieves continuous, high-fidelity repair via SDF.
- **vs. R3D-AD**: A diffusion-based method with high computational cost; PASDF is simpler and achieves higher O-AUROC (0.900 vs. 0.843).

## Relevance to My Research
- The application of SDF as a continuous 3D representation for anomaly detection offers inspiring methodological insight.
- The decoupled pipeline of pose alignment → shape learning is transferable to other 3D understanding tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The unified SDF framework for anomaly detection and repair is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two benchmarks, comprehensive ablations, PAM generalization verification, and repair quality evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Theoretical modeling is clearly presented (Sec. 3.2); the pipeline diagram is intuitive.
- **Value**: ⭐⭐⭐ Primarily in the anomaly detection domain; somewhat distant from mainstream 3D vision research, but methodologically informative.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation](bridging_3d_anomaly_localization_and_repair_via_high-qualit.md)
- [\[ICCV 2025\] SiM3D: Single-Instance Multiview Multimodal and Multisetup 3D Anomaly Detection Benchmark](sim3d_single-instance_multiview_multimodal_and_multisetup_3d_anomaly_detection_b.md)
- [\[ICCV 2025\] RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians](rayletdf_raylet_distance_fields_for_generalizable_3d_surface_reconstruction_from.md)
- [\[ICCV 2025\] Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction](momentum-gs_momentum_gaussian_self-distillation_for_high-quality_large_scene_rec.md)
- [\[ICCV 2025\] Guiding Diffusion-Based Articulated Object Generation by Partial Point Cloud Alignment and Physical Plausibility Constraints](guiding_diffusion-based_articulated_object_generation_by_partial_point_cloud_ali.md)

<!-- RELATED:END -->
