---
title: >-
  [Paper Note] Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation
description: >-
  [ICCV 2025][3D Vision][3D anomaly detection] This paper proposes PASDF, a framework that employs a pose-aware signed distance function (SDF) for continuous geometric representation, unifying 3D anomaly detection and repair tasks, achieving state-of-the-art performance on Real3D-AD and Anomaly-ShapeNet.
tags:
  - ICCV 2025
  - 3D Vision
  - 3D anomaly detection
  - signed distance function
  - pose alignment
  - anomaly repair
  - point cloud
date: 2026-05-08
content_hash: 8286d605b33fff5f
---

# Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation

**Conference**: ICCV 2025
**arXiv**: [2505.24431](https://arxiv.org/abs/2505.24431)
**Code**: [https://github.com/ZZZBBBZZZ/PASDF](https://github.com/ZZZBBBZZZ/PASDF)
**Area**: 3D Vision / Anomaly Detection / Point Cloud
**Keywords**: 3D anomaly detection, signed distance function, pose alignment, anomaly repair, point cloud

## TL;DR

This paper proposes PASDF, a framework that employs a pose-aware signed distance function (SDF) for continuous geometric representation, unifying 3D anomaly detection and repair tasks, achieving state-of-the-art performance on Real3D-AD and Anomaly-ShapeNet.

## Background & Motivation

1. **Importance of 3D anomaly detection**: In manufacturing quality control, robotic manipulation, and related domains, even minor 3D anomalies—missing features, deformations, or shape irregularities—can cause complete component failure, necessitating robust 3D anomaly detection.

2. **Limitations of existing methods**:
    - **Voxel-based methods**: Discretization causes loss of fine geometric details and suffers from cubic memory growth
    - **Point cloud methods**: Sparse sampling leads to density inconsistency and incomplete surface coverage
    - **Projection-based methods**: Information loss in occluded regions and view-dependent distortions
    - All these approaches are inherently discrete representations that introduce quantization artifacts, hindering fine-grained anomaly localization

3. **Need to bridge detection and repair**: In 3D printing and advanced manufacturing, anomaly detection is only the first step; in-situ repair is equally critical. Conventional methods rely on indirect feature mappings and cannot provide explicit shape reconstruction to guide repair. Even reconstruction-based methods (e.g., IMRNet, R3D-AD) fail to generate continuous, high-fidelity repair templates due to their reliance on discrete point cloud representations.

4. **Core motivation**: Leveraging the continuity and smoothness of SDFs to bridge anomaly detection and repair, while decoupling pose to address detection under arbitrary orientations.

## Method

### Overall Architecture

PASDF consists of three core stages:

1. **Pose Alignment Module (PAM)**: Aligns input point clouds to a canonical coordinate frame, eliminating the effect of pose variation
2. **SDF Network**: Learns a continuous signed distance function representation that implicitly captures object shape
3. **Anomaly Scoring Module**: Computes anomaly scores based on the deviation of SDF values for test samples

Theoretically, anomaly detection is formulated as evaluating the likelihood that a test sample conforms to the normal shape distribution. Pose invariance is achieved by integrating over the SE(3) group, approximated in practice by aligning to a canonical pose.

### Key Designs

#### 1. Pose Alignment Module (PAM)

PAM adopts a coarse-to-fine two-stage registration strategy:

- **Coarse alignment**: Voxel downsamples the raw point cloud, extracts FPFH (Fast Point Feature Histogram) features, and applies RANSAC for global coarse registration
- **Fine alignment**: Refines the coarse alignment using ICP (Iterative Closest Point)
- **Iterative optimization**: Introduces a Chamfer Distance-driven feedback mechanism that dynamically adjusts the loss threshold τ to avoid local minima. The accumulated transformation matrix is updated iteratively as: $T^{(k)} = T_{icp}^{(k)} \cdot T_{ransac}^{(k)} \cdot T^{(k-1)}$

Key PAM parameters: loss threshold τ = 0.016, increment Δτ = 0.001, maximum iterations K = 10.

#### 2. SDF Network

The aligned point cloud is represented via a neural network-parameterized SDF:

- Query points are sampled from three regions: surface points (10k), interior bounding-box points (10k), and unit volume points (3k), totaling 23,000 points
- Query coordinates are encoded using sinusoidal positional encoding $\gamma(\mathbf{x}_i)$
- Network architecture: 8-layer MLP with weight normalization, ReLU activations, dropout (0.2) in intermediate layers, and a skip connection at the fourth layer
- SDF predictions are trained with a truncated L1 loss

#### 3. Anomaly Score Computation

- **Point-level anomaly score**: $A(\mathbf{x}_j) = |f_\theta(\mathbf{x}_j)|$, i.e., the absolute value of the SDF output
- **Object-level anomaly score**: Mean of the top-K (K = 1000) highest point-level anomaly scores

#### 4. Anomaly Repair

The trained SDF network implicitly encodes the "normal" shape manifold:
1. The anomalous input is aligned to the canonical pose via PAM
2. The Marching Cubes algorithm extracts the zero-level set of the SDF
3. A point cloud is sampled from the resulting triangular mesh as the repaired output

### Loss & Training

- **Truncated L1 loss**: $\mathcal{L}_{SDF} = \frac{1}{N_q} \sum_{i=1}^{N_q} |\text{clamp}(\hat{s}_i, -d_{max}, d_{max}) - s_i|$
- Truncation distance $d_{max} = 0.1$
- Training for 2000 epochs with learning rate $1\times10^{-5}$
- Data preprocessing: coordinates normalized to $[0,1]^3$; non-manifold detection and Poisson reconstruction ensure watertightness

## Key Experimental Results

### Main Results

**Datasets**:
- Real3D-AD: A high-resolution real-world dataset with 12 categories, 4 normal training samples and 100 test samples per category
- Anomaly-ShapeNet: A synthetic dataset with 40 categories and 1600+ samples

#### Results on Real3D-AD

| Method | O-AUROC ↑ | P-AUROC ↑ |
|--------|-----------|-----------|
| BTF(Raw) | 0.635 | 0.571 |
| BTF(FPFH) | 0.603 | 0.733 |
| M3DM(PointMAE) | 0.552 | 0.637 |
| PatchCore(FPFH+Raw) | 0.682 | 0.680 |
| RegAD | 0.704 | 0.705 |
| IMRNet | 0.725 | - |
| Group3AD | 0.751 | - |
| **PASDF (Ours)** | **0.802** | **0.745** |

PASDF surpasses Group3AD by 5.1% in O-AUROC, with particularly strong performance on Seahorse (1.000), Car (0.959), and Fish (0.989).

#### Results on Anomaly-ShapeNet

| Method | O-AUROC Mean ↑ | P-AUROC Mean ↑ |
|--------|----------------|----------------|
| BTF(Raw) | 0.493 | 0.550 |
| M3DM | 0.552 | 0.616 |
| CPMF | 0.559 | - |
| RegAD | 0.572 | 0.668 |
| IMRNet | 0.661 | 0.650 |
| R3D-AD | 0.749 | - |
| **PASDF (Ours)** | **0.900** | **0.897** |

PASDF achieves the best O-AUROC on 37 out of 40 categories and attains perfect scores of 1.000 on multiple categories.

### Ablation Study

#### Effect of PAM on Various Baselines (Anomaly-ShapeNet)

| Method | PAM | O-AUROC ↑ | P-AUROC ↑ |
|--------|-----|-----------|-----------|
| BTF(FPFH) | ✗ | 0.528 | 0.628 |
| BTF(FPFH) | ✓ | 0.579 | 0.683 |
| PatchCore(FPFH) | ✗ | 0.568 | 0.580 |
| PatchCore(FPFH) | ✓ | 0.814 | 0.867 |
| PatchCore(PointMAE) | ✗ | 0.562 | 0.577 |
| PatchCore(PointMAE) | ✓ | 0.626 | 0.681 |
| PASDF (Full) | ✓ | 0.900 | 0.897 |

Incorporating PAM into PatchCore(FPFH) yields gains of +24.6% in O-AUROC and +28.7% in P-AUROC, a remarkably substantial improvement.

#### Component Ablation (Anomaly-ShapeNet)

| Method | O-AUROC ↑ | P-AUROC ↑ |
|--------|-----------|-----------|
| w/o RANSAC | 0.711 | 0.739 |
| w/o ICP | 0.727 | 0.836 |
| w/o Iterative Optimization | 0.871 | 0.884 |
| w/o Positional Encoding | 0.887 | 0.783 |
| PASDF (Full) | 0.900 | 0.897 |

#### Anomaly Repair Quality Evaluation

| Method | Real3D-AD CD ↓ | Real3D-AD EMD ↓ | Anomaly-ShapeNet CD ↓ | Anomaly-ShapeNet EMD ↓ |
|--------|---------|---------|---------|---------|
| w/o PE | 0.0255 | 0.0133 | 0.0575 | 0.0276 |
| with PE | 0.0203 | 0.0110 | 0.0445 | 0.0228 |

### Key Findings

1. **Pose alignment is critical**: Removing RANSAC causes O-AUROC to drop sharply from 0.900 to 0.711, confirming the necessity of global coarse registration.
2. **Dual role of positional encoding**: Removing PE degrades P-AUROC by 11.4% (0.897→0.783) and simultaneously deteriorates repair quality.
3. **Generalizability of PAM**: PAM functions as a plug-and-play module that consistently boosts multiple baseline methods.
4. **Advantage of continuous representation**: PASDF outperforms all competing methods on 37/40 categories, whereas other methods exhibit unstable performance across categories.

## Highlights & Insights

1. **Unified framework**: PASDF is the first to unify 3D anomaly detection and repair under a continuous SDF representation, with both tasks sharing the same learned shape prior.
2. **Pose–shape disentanglement**: PAM explicitly decouples pose from shape, allowing the SDF network to focus on intrinsic shape variations.
3. **Continuous vs. discrete**: Replacing discrete voxel/point cloud/projection representations with a continuous SDF eliminates quantization artifacts and preserves fine geometric details.
4. **Plug-and-play PAM**: Beyond serving PASDF, PAM substantially improves the performance of other methods, demonstrating broad generalizability.

## Limitations & Future Work

1. **Computational cost**: PAM registration can be expensive in cases with challenging initial poses; learning-based or hierarchical registration approaches may be explored.
2. **Single-class assumption**: The current formulation assumes a single normal object category; extending to multi-class detection would improve practical applicability.
3. **Sensitivity to input quality**: Performance is affected by input point cloud quality, motivating enhanced robustness to noise and outliers.
4. **Absence of contextual information**: Integrating scene context could improve detection in complex real-world environments.
5. **Training efficiency**: The 2000-epoch training schedule is lengthy; more efficient training strategies warrant investigation.

## Related Work & Insights

- **DeepSDF**: The neural network parameterization of SDFs originates from DeepSDF; PASDF creatively adapts this idea for anomaly detection.
- **ICP + RANSAC**: The combination of these classical registration methods remains highly effective within PAM.
- **Marching Cubes**: This classical algorithm is employed in the repair stage to extract the zero-level set from the SDF.
- **Insight**: Continuous representations (implicit functions) hold a natural advantage over discrete counterparts in tasks requiring fine-grained geometric perception.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The unified SDF-based framework for 3D anomaly detection and repair is a novel and well-motivated contribution
- **Technical Quality**: ⭐⭐⭐⭐ — The method is thoroughly designed, ablations are comprehensive, and the generalizability of PAM is convincingly validated
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two datasets, multiple baselines, detailed ablations, and qualitative results
- **Value**: ⭐⭐⭐⭐ — The unified detection-and-repair framework has direct applicability in manufacturing scenarios
- **Overall**: ⭐⭐⭐⭐ (8/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] PASDF: Bridging 3D Anomaly Localization and Repair via High-Quality Continuous Geometric Representation](bridging_3d_anomaly_localization_and_repair_via_highquality.md)
- [\[ICCV 2025\] SiM3D: Single-Instance Multiview Multimodal and Multisetup 3D Anomaly Detection Benchmark](sim3d_single-instance_multiview_multimodal_and_multisetup_3d_anomaly_detection_b.md)
- [\[ICCV 2025\] RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians](rayletdf_raylet_distance_fields_for_generalizable_3d_surface_reconstruction_from.md)
- [\[ICCV 2025\] Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction](momentum-gs_momentum_gaussian_self-distillation_for_high-quality_large_scene_rec.md)
- [\[ICCV 2025\] DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning](dap-mae_domain-adaptive_point_cloud_masked_autoencoder_for_effective_cross-domai.md)

</div>

<!-- RELATED:END -->
