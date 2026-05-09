---
title: >-
  [Paper Note] U-CAN: Unsupervised Point Cloud Denoising with Consistency-Aware Noise2Noise Matching
description: >-
  [NeurIPS 2025][3D Vision][Point cloud denoising] This paper proposes U-CAN, an unsupervised point cloud denoising framework that infers multi-step denoising paths via a Noise2Noise matching scheme and geometric consistency constraints. The method approaches supervised performance and demonstrates that the consistency constraint generalizes to 2D image denoising.
tags:
  - NeurIPS 2025
  - 3D Vision
  - Point cloud denoising
  - unsupervised learning
  - Noise2Noise
  - consistency constraint
  - geometry reconstruction
date: 2026-05-08
content_hash: 9f8d913e3704b840
---

# U-CAN: Unsupervised Point Cloud Denoising with Consistency-Aware Noise2Noise Matching

**Conference**: NeurIPS 2025
**arXiv**: [2510.25210](https://arxiv.org/abs/2510.25210)
**Code**: Available (Project Page)
**Area**: 3D Vision / Point Cloud Processing
**Keywords**: Point cloud denoising, unsupervised learning, Noise2Noise, consistency constraint, geometry reconstruction

## TL;DR

This paper proposes U-CAN, an unsupervised point cloud denoising framework that infers multi-step denoising paths via a Noise2Noise matching scheme and geometric consistency constraints. The method approaches supervised performance and demonstrates that the consistency constraint generalizes to 2D image denoising.

## Background & Motivation

Point clouds acquired by scanning sensors inevitably contain noise, which severely degrades downstream tasks such as surface reconstruction and shape understanding. Limitations of existing methods:

**Supervised methods**: Require large numbers of noise–clean point cloud pairs, making annotation extremely costly.

**Unsupervised methods**: Existing unsupervised approaches suffer a substantial performance gap relative to supervised counterparts.

**Multi-step denoising**: Single-step denoising cannot fully eliminate noise, and multi-step schemes lack effective optimization objectives.

The core innovation lies in exploiting statistical relationships among multiple noisy observations to enable unsupervised learning under the Noise2Noise paradigm, while introducing geometric consistency constraints to further improve denoising quality.

## Method

### Overall Architecture

U-CAN consists of three core components:
1. Multi-step denoising path inference: a neural network predicts the multi-step denoising trajectory for each point.
2. Noise2Noise matching loss: leverages statistical reasoning across multiple noisy observations.
3. Geometric consistency constraint: ensures geometric coherence in the denoised output.

### Key Designs

1. **Exploitation of multiple noisy observations**:

    - Multiple noisy point cloud observations are acquired for the same shape or scene.
    - The noise across different observations is assumed to be i.i.d.
    - The N2N matching scheme avoids the need for clean targets.

2. **Noise2Noise matching loss**:

    - One noisy observation is denoised and then matched against another noisy observation.
    - Statistically, the optimal denoising result minimizes the expected distance between two noisy observations.
    - This is equivalent to denoising toward the true clean surface.

3. **Denoised Geometry Consistency**:

    - Denoised points should preserve local geometric structures (e.g., normals, curvature).
    - Neighboring points originating from the same surface should remain neighbors after denoising.
    - The constraint is formalized as local geometric invariants of the denoised output.

4. **Cross-domain generality**:

    - The authors demonstrate that the geometric consistency constraint is not restricted to 3D point clouds.
    - The same principle is applicable to 2D image denoising.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{N2N}} + \lambda_1 \mathcal{L}_{\text{geo}} + \lambda_2 \mathcal{L}_{\text{reg}}$$

- $\mathcal{L}_{\text{N2N}}$: Noise2Noise matching loss (a variant of Chamfer distance)
- $\mathcal{L}_{\text{geo}}$: Geometric consistency loss (normal consistency + local curvature preservation)
- $\mathcal{L}_{\text{reg}}$: Regularization term (to prevent degenerate solutions)

## Key Experimental Results

### Main Results (Point Cloud Denoising — PU-Net Dataset)

| Method | Supervision | CD (×10⁻⁴) ↓ | P2S (×10⁻³) ↓ | Noise σ=0.01 CD ↓ | Noise σ=0.03 CD ↓ |
|------|---------|-------------|-------------|----------------|----------------|
| PCN | Supervised | 0.68 | 1.23 | 0.58 | 1.15 |
| Score-based | Supervised | 0.52 | 0.98 | 0.45 | 0.89 |
| DMR | Supervised | 0.48 | 0.85 | 0.42 | 0.82 |
| Pointfilter | Unsupervised | 1.25 | 2.15 | 1.12 | 2.45 |
| Self2Self | Unsupervised | 0.95 | 1.72 | 0.85 | 1.85 |
| **U-CAN** | **Unsupervised** | **0.55** | **0.92** | **0.48** | **0.91** |

### Point Cloud Upsampling Experiments

| Method | PU1K CD ↓ | PU1K P2S ↓ | PU-GAN CD ↓ | Supervision |
|------|----------|-----------|------------|---------|
| PU-Net | 0.72 | 1.35 | 0.85 | Supervised |
| PU-GAN | 0.65 | 1.18 | 0.72 | Supervised |
| Dis-PU | 0.58 | 1.05 | 0.65 | Supervised |
| **U-CAN (Denoising + Upsampling)** | **0.61** | **1.08** | **0.68** | **Unsupervised** |

### 2D Image Denoising Experiments

| Method | SIDD PSNR ↑ | SIDD SSIM ↑ | BSD68 PSNR ↑ |
|------|------------|-----------|------------|
| N2N (baseline) | 38.12 | 0.952 | 29.85 |
| N2V | 37.85 | 0.948 | 29.62 |
| **N2N + Consistency Constraint** | **38.67** | **0.958** | **30.23** |

### Key Findings

1. U-CAN is the first unsupervised point cloud denoising method to approach the performance of supervised methods.
2. The geometric consistency constraint contributes approximately 15–20% performance improvement.
3. The constraint transfers successfully to 2D image denoising, demonstrating generality.
4. Multi-step denoising paths yield consistent improvements over single-step denoising.

## Highlights & Insights

- **Unsupervised approaching supervised**: Achieving near-supervised performance without annotated data offers substantial practical value.
- **Cross-domain generalizable constraint**: The transfer of the geometric consistency constraint to 2D image denoising is a particularly compelling contribution.
- **Theoretical grounding**: The statistical reasoning underlying the N2N scheme is supported by rigorous mathematical derivation.

## Limitations & Future Work

1. Multiple noisy observations of the same scene are required; the method cannot be directly applied to single-scan settings.
2. Robustness to non-uniform noise distributions remains to be validated.
3. Experiments on large-scale outdoor scenes (e.g., LiDAR point clouds) are limited.
4. Multi-step denoising incurs increased inference time.

## Related Work & Insights

- **Noise2Noise (Lehtinen et al., 2018)**: Seminal work on unsupervised denoising.
- **Score-based denoising**: Point cloud denoising via score matching.
- **DMR**: Differentiable manifold reconstruction for denoising.
- **Self2Self**: Self-supervised denoising framework.

## Rating

| Dimension | Score (1–5) |
|------|-----------|
| Novelty | 4 |
| Theoretical Depth | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Practical Value | 4 |
| Overall Recommendation | 4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Noise2Score3D: Tweedie's Approach for Unsupervised Point Cloud Denoising](../../ICCV2025/3d_vision/noise2score3d_tweedies_approach_for_unsupervised_point_cloud_denoising.md)
- [\[NeurIPS 2025\] UGM2N: An Unsupervised and Generalizable Mesh Movement Network via M-Uniform Loss](ugm2n_an_unsupervised_and_generalizable_mesh_movement_network_via_m-uniform_loss.md)
- [\[NeurIPS 2025\] Rectified Point Flow: Generic Point Cloud Pose Estimation](rectified_point_flow_generic_point_cloud_pose_estimation.md)
- [\[NeurIPS 2025\] Flux4D: Flow-based Unsupervised 4D Reconstruction](flux4d_flow-based_unsupervised_4d_reconstruction.md)
- [\[NeurIPS 2025\] PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion](pointmac_meta-learned_adaptation_for_robust_test-time_point_cloud_completion.md)

</div>

<!-- RELATED:END -->
