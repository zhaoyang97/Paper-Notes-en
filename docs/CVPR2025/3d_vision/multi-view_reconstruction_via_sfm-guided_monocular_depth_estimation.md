---
title: >-
  [Paper Note] Multi-view Reconstruction via SfM-guided Monocular Depth Estimation
description: >-
  [3D Vision] This work introduces Murre, which injects SfM sparse point clouds as conditions into diffusion-based monocular depth estimation. By generating multi-view consistent metric depth maps followed by TSDF fusion, Murre outperforms state-of-the-art MVS and neural implicit reconstruction methods across diverse real-world scenarios—including indoor, street-view, and aerial scenes—while requiring only minimal fine-tuning on synthetic data.
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: cf3bb2676b1ace6a
---

# Multi-view Reconstruction via SfM-guided Monocular Depth Estimation

## TL;DR

This work introduces Murre, which injects SfM sparse point clouds as conditions into diffusion-based monocular depth estimation. By generating multi-view consistent metric depth maps followed by TSDF fusion, Murre outperforms state-of-the-art MVS and neural implicit reconstruction methods across diverse real-world scenarios—including indoor, street-view, and aerial scenes—while requiring only minimal fine-tuning on synthetic data.

## Background & Motivation

Traditional multi-view 3D reconstruction methods face three major bottlenecks:

1. **High Memory Consumption**: Learning-based MVS methods (e.g., MVSNet, IGEV-MVS) require aggregating features in 3D space to construct cost volumes, which incurs massive GPU memory overhead and limits reconstruction resolution.
2. **Failure under Sparse Views**: When input views are sparse, large regions cannot be matched across multiple views, causing matching-based methods to fail.
3. **Limited Generalization**: Learning-based MVS methods require high-quality 3D ground truth (GT) data for training, which is scarce, leading to poor generalization across different scenes.

Although monocular depth estimation avoids multi-view matching, it faces scale ambiguity and multi-view inconsistency issues:
- Affine-invariant methods (e.g., Marigold, Depth Anything) lack global metric information.
- Metric depth methods (e.g., Metric3D) overfit to the training data domain.

Key Insight: SfM point clouds serve as a "condensed representation" of multi-view information. They capture the global structure and accurate scale of a scene, serving as prior conditions to guide diffusion models to generate depth maps that are both metrically scaled and multi-view consistent.

## Method

### Overall Architecture

The reconstruction pipeline of Murre consists of the following steps:
1. **SfM Sparse Reconstruction**: Perform detector-free SfM on input images to obtain sparse point clouds and camera poses.
2. **SfM-guided Depth Estimation**: Project sparse point clouds onto each view, densify them, and feed them as conditions into the diffusion model.
3. **Depth Alignment**: Align the predicted depth with the SfM depth using RANSAC linear regression.
4. **TSDF Fusion**: Fuse the aligned metric depth maps to reconstruct the final 3D geometry.

### Key Designs

#### 1. Injecting SfM Priors into Diffusion Models

**Sparse Depth Densification**:
- Project SfM point clouds onto each view to obtain sparse depth maps.
- Densify via k-nearest neighbors (KNN, $k=3$) interpolation: for each pixel without a value, find the $k$ nearest pixels with values and calculate their weighted average using the inverse of their distances.
- Concurrently compute a distance map representing the Euclidean distance from each pixel to the nearest pixel with a valid depth.

**Normalization Strategy**:
- Filter out the top and bottom 2% of SfM depth values to remove outliers.
- Expand the range by 20% (multiplying the minimum by 0.8 and the maximum by 1.2) to cover the full depth range.
- Ground truth (GT) depth maps are normalized using the same range.

**Conditional Input**: The RGB image and the densified depth map are mapped to the latent space via encoders. The distance map is directly downsampled to the latent resolution. All inputs are concatenated along with noise and fed into the UNet.

#### 2. Conditional Diffusion based on Stable Diffusion v2

- Initialized with Stable Diffusion V2, freezing the VAE and fine-tuning only the UNet.
- The depth map is replicated to three channels and mapped to the latent space via the VAE encoder.
- The training loss uses the standard noise prediction MSE: $\mathcal{L} = \|\epsilon - \hat{\epsilon}\|^2$.
- During inference, 5-view ensemble and pixel-wise median filtering are employed to enhance robustness.
- Supports Latent Consistency Model (LCM) distillation to achieve single-step inference.

#### 3. RANSAC Depth Alignment

A minor scale bias exists between the predicted depth and the SfM depth, which is aligned using RANSAC linear regression:
- Iteratively and randomly sample subsets to estimate the scale and shift parameters.
- Select the transformation with the most inliers as the final alignment.

### Loss & Training

$$\mathcal{L} = \|\epsilon - \hat{\epsilon}\|^2$$

This represents the standard diffusion model noise prediction loss, where $\epsilon$ is the noise added to the GT depth latent code.

## Key Experimental Results

### Main Results

**DTU Dataset (Chamfer Distance, mm, 3-view)**:

| Method | Training Data Volume | Mean CD↓ |
|------|-----------|----------|
| COLMAP | - | 2.56 |
| MonoSDF | - | 1.86 |
| Marigold | 74K | 5.46 |
| Depth-Anything | 1.5M | 3.09 |
| Metric3D | 8M | 5.01 |
| MVSNet | 27.1K | 2.38 |
| DUSt3R | 8.5M | 2.81 |
| **Murre** | **86.4K** | **1.42** |

Murre achieves the lowest Chamfer Distance (1.42mm) with the minimum amount of training data (86.4K), which is 24% lower than the second-best method, MonoSDF (1.86mm).

### Ablation Study

**Speed-Accuracy Trade-off (Replica)**:

| Configuration (Steps / Ensemble / Alignment) | Inference Time (s)↓ | F-Score↑ |
|------------------------|-------------|----------|
| 10 steps / 5 ensembles / RANSAC | 12.166 | 0.853 |
| 10 steps / 1 ensemble / RANSAC | 2.969 | 0.850 |
| 1 step (LCM) / 1 ensemble / RANSAC | 0.840 | 0.828 |
| 1 step (LCM) / 1 ensemble / No Alignment | 0.829 | 0.780 |

Single-step inference after LCM distillation takes only 0.84 seconds per view while maintaining a high F-Score of 0.828.

**Choice of SfM Methods**:

| SfM Method | F-Score↑ |
|----------|----------|
| COLMAP | 0.645 |
| DF-SfM (LoFTR) | 0.853 |
| DF-SfM (DKM) | 0.842 |

**Ablation on Depth Conditions**:

| k (KNN) | Distance Map | F-Score↑ |
|---------|--------|----------|
| 0 | ✗ | 0.543 |
| 3 | ✗ | 0.753 |
| 3 | ✓ | **0.853** |

### Key Findings

1. **Outperforming Million-Scale Models with Only 86.4K Synthetic Fine-tuning**: It leverages the powerful prior of Stable Diffusion.
2. **SfM Sparse Point Clouds as Highly Effective Multi-view Information Carriers**: Multi-view matching information is compressed into conditional signals.
3. **Crucial Role of Distance Maps**: Incorporating the distance map boosts the F-Score from 0.753 to 0.853, assisting the network in distinguishing raw SfM pixels from KNN-interpolated ones.
4. **Indispensability of KNN Densification**: Performance drops significantly when using sparse depth maps directly (0.543 vs. 0.853), since sparse signals are poorly suited for the VAE encoder.
5. **Excellent Cross-Domain Generalization**: Robust performance is demonstrated across diverse scenes including indoors (ScanNet, Replica), street-view (Waymo), and aerial (UrbanScene3D).

## Highlights & Insights

1. **Elegant Combination of SfM and Diffusion Models**: SfM provides the "what" (global structure and metric scale) and the diffusion model provides the "how much" (dense depth completion), complementing each other.
2. **Bypassing the Multi-view Matching Bottleneck**: By compressing multi-view information into SfM point cloud conditions, the reconstruction problem is recast as conditional monocular depth estimation, inherently resolving memory overhead and sparse-view issues.
3. **Effective Densification Design for Sparse Conditions**: The combination of KNN interpolation and distance maps is both simple and highly effective. The distance map serves as a confidence indicator, indicating to the model which points are reliable SfM observations versus interpolated values.
4. **Outstanding Data Efficiency**: Trained only on two synthetic datasets (Hypersim and 3D Ken Burns), the model generalizes dynamically to various real-world scenarios.
5. **Flexible Speed-Quality Trade-off**: Supports multiple inference modes ranging from 12 seconds (high quality) to 0.8 seconds (fast inference).

## Limitations & Future Work

1. **Dependency on SfM Success**: In extreme scenarios (e.g., only two images, minimal overlap), SfM fails to recover camera poses and sparse point clouds.
2. **Limited to Static Scenes**: Unable to handle dynamic scene elements such as pedestrians and vehicles.
3. **SfM Quality Caps the Optimization Ceiling**: In textureless regions, COLMAP produces noisy and sparse point clouds, which degrades depth estimation quality.
4. **Additional SfM Runtime**: While depth estimation is fast, the SfM process itself can be time-consuming.
5. **Heuristic Normalization Assumptions**: Expanding the depth range by 20% is heuristic and might be insufficient for extreme scenes.

## Related Work & Insights

- **Marigold** [Ke et al., 2024]: A diffusion-based affine-invariant monocular depth estimation model, serving as a direct predecessor to Murre. Murre injects SfM guidance on top of it to resolve metric scale.
- **Depth Anything** [Yang et al., 2024]: A large-scale pre-trained monocular depth model that lacks metric scale.
- **NeuralRecon** [Sun et al., 2021]: An elegant learning-based method that constructs TSDF volumes directly in world coordinates, though it suffers from limited generalization.
- **DF-SfM** [He et al., 2023]: A detector-free matching-based SfM framework, highly suitable for textureless scenarios.
- Insight: The union of 2D foundation models (e.g., Stable Diffusion) and 3D geometric constraints (e.g., SfM) represents a highly effective "prior + constraint" paradigm, which can potentially be extended to other 3D vision tasks.

## Rating

⭐⭐⭐⭐⭐ (9/10)

- Novelty: ⭐⭐⭐⭐⭐ — The synergy between SfM and diffusion models for depth estimation is highly novel and elegant; the KNN + distance map conditioning is simple yet effective.
- Value: ⭐⭐⭐⭐⭐ — Demonstrates outstanding performance in diverse real-world settings and supports a flexible speed-quality trade-off.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation covering 5 scene domains, multiple competitive baselines, and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐⭐ — Extremely clear motivation, excellent system pipeline illustrations, and carefully designed ablation studies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](../../ICCV2025/3d_vision/depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)
- [\[CVPR 2025\] Multi-View Pose-Agnostic Change Localization with Zero Labels](multi-view_pose-agnostic_change_localization_with_zero_labels.md)
- [\[ICCV 2025\] One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images](../../ICCV2025/3d_vision/one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)
- [\[CVPR 2025\] FLARE: Feed-forward Geometry, Appearance and Camera Estimation from Uncalibrated Sparse Views](flare_feed-forward_geometry_appearance_and_camera_estimation_from_uncalibrated_s.md)
- [\[ICCV 2025\] HORT: Monocular Hand-held Objects Reconstruction with Transformers](../../ICCV2025/3d_vision/hort_monocular_hand-held_objects_reconstruction_with_transformers.md)

</div>

<!-- RELATED:END -->
