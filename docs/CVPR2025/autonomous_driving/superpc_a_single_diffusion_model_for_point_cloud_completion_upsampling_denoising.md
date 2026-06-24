---
title: >-
  [Paper Note] SuperPC: A Single Diffusion Model for Point Cloud Completion, Upsampling, Denoising, and Colorization
description: >-
  [CVPR 2025][Autonomous Driving][Point Cloud Processing] SuperPC is proposed as the first framework to unify point cloud completion, upsampling, denoising, and colorization within a single conditional diffusion model. It effectively fuses image and point cloud modalities using a three-level conditioning (TLC) mechanism (raw/local/global) and a spatial mixed fusion (SMF) strategy.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Point Cloud Processing"
  - "Diffusion Models"
  - "Multi-task Learning"
  - "Multimodal Fusion"
  - "3D Reconstruction"
date: 2026-05-08
content_hash: 81ec19f03a80742e
---

# SuperPC: A Single Diffusion Model for Point Cloud Completion, Upsampling, Denoising, and Colorization

**Conference**: CVPR 2025  
**arXiv**: [2503.14558](https://arxiv.org/abs/2503.14558)  
**Code**: [Project Page](https://sairlab.org/superpc/)  
**Area**: Autonomous Driving / Point Cloud Processing  
**Keywords**: Point Cloud Processing, Diffusion Models, Multi-task Learning, Multimodal Fusion, 3D Reconstruction

## TL;DR

SuperPC is proposed as the first framework to unify point cloud completion, upsampling, denoising, and colorization within a single conditional diffusion model. It effectively fuses image and point cloud modalities using a three-level conditioning (TLC) mechanism (raw/local/global) and a spatial mixed fusion (SMF) strategy.

## Background & Motivation

- Point cloud processing covers four key tasks: completion, upsampling, denoising, and colorization, which are crucial in fields such as autonomous driving and 3D reconstruction.
- Existing approaches typically address each task independently using separate models, ignoring the interconnected nature of these defects.
- Sequentially applying separate models leads to error accumulation: for instance, errors from a completion model degrade upsampling quality, and noise patterns introduced by upsampling undermine denoising performance.
- Existing diffusion-based methods only handle single tasks and retain only limited input information (either global, local, or raw), failing to satisfy the diverse requirements of the four tasks.
- Most methods are restricted to unimodal input (either only point clouds or only images) and can only handle simple object-level point clouds (e.g., ShapeNet), falling short on complex scene-level data.
- A unified model is needed to leverage information from both image and point cloud modalities, guiding the diffusion process across different abstraction levels.

## Method

### Overall Architecture

SuperPC adopts a Conditional Diffusion Probabilistic Model (CDPM). Conditioned on input sparse/incomplete/noisy point clouds and corresponding images, it generates high-quality, dense, and colorized point clouds through a reverse diffusion process. The core of the framework is a Three-Level Conditioning (TLC) mechanism, which extracts and fuses image and point cloud information across raw, local, and global levels to guide the noise prediction network $\epsilon_\theta$ at each diffusion step. The loss function extends the standard noise prediction MSE loss to a three-level conditional formulation.

### Key Designs

**1. Raw Module & Dual-Spatial Early Fusion**
- **Function**: Preserves and fuses the raw texture and spatial information of images and point clouds at each diffusion step.
- **Mechanism**: Employs two branches—the image feature projection branch projects 2D image features onto the partially denoised point cloud using PyTorch3D's point rasterization ($P_t(N,3) \to P_t'(N, 3+C_1)$); the point cloud feature interpolation branch aligns the input point cloud features to the target point cloud using k-nearest neighbor interpolation based on inverse distance weighting.
- **Design Motivation**: Achieves "dual-spatial early fusion" through two spatial operations, retaining both the dense color/texture information of the images and the original geometric structure of the input point clouds, thereby providing direct, raw-level guidance for the diffusion process.

**2. Local Module & Attention-Based Deep Fusion**
- **Function**: Extracts and fuses local object-level features from images and point clouds to provide detail-rich local conditions.
- **Mechanism**: Utilizes ResNet and PointNet++ as image and point cloud encoders, respectively, to extract multi-scale features via Feature Pyramid Networks (FPN). It then fuses the intermediate feature representations of both modalities using a cross-attention mechanism: $\text{Attention}(Q, K, V) = \text{softmax}(QK/\sqrt{T}) \cdot V$, where $Q$ comes from image features, and $K, V$ are derived from point cloud features.
- **Design Motivation**: Cross-attention aligns image semantic features with point cloud spatial features to generate a unified local fused feature map. This captures multi-scale semantic information and compensates for the micro-level limitations of the raw module in high-level semantic representation.

**3. Global Module**
- **Function**: Generates a 1D global structural latent code to ensure high-level semantic consistency.
- **Mechanism**: Compresses the local fused feature map into a 1D global latent code through PointNet++ layers and max-pooling layers, highlighting high-level features while reducing dimensionality.
- **Design Motivation**: Relying solely on local or raw conditions can trap the model in local optima or cause overfitting to a single task. Global conditioning provides holistic shape and semantic constraints, enhancing model robustness against complex scenes and multi-task settings.

### Loss & Training

The standard conditional diffusion noise prediction loss is adopted, extended into a three-level conditioning format:

$$\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}_0, \epsilon, t}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, c_{\text{raw}}, c_{\text{local}}, c_{\text{global}}, t)\|^2\right]$$

where $c_{\text{raw}}$, $c_{\text{local}}$, and $c_{\text{global}}$ denote the three levels of conditioning, respectively.

## Key Experimental Results

### Main Results: ShapeNet Four-Task Comparison

| Method | Completion DCD↓ | Completion F1↑ | Upsampling DCD↓ | Upsampling F1↑ | Denoising DCD↓ | Denoising F1↑ |
|------|-----------|----------|-------------|------------|-----------|----------|
| AdaPoinTr | 0.462 | 0.423 | ✗ | ✗ | 0.562 | 0.405 |
| GradPU | ✗ | ✗ | 0.298 | 0.589 | 0.533 | 0.412 |
| ScoreDenoise | ✗ | ✗ | 0.346 | 0.537 | 0.291 | 0.812 |
| SOTA Combination | 0.462 | 0.423 | 0.281 | 0.659 | 0.280 | 0.856 |
| **SuperPC** | **0.387** | **0.557** | 0.293 | 0.631 | 0.285 | 0.837 |

*SuperPC leads by a large margin in completion, and the overall combination task DCD drops from 0.509 to 0.476.*

### Scene-Level Benchmark: KITTI-360

| Method | Completion DCD↓ | Upsampling DCD↓ | Denoising DCD↓ | Combination DCD↓ |
|------|-----------|-------------|-----------|-----------|
| SOTA Combination | 0.649 | 0.597 | 0.369 | 0.725 |
| **SuperPC** | **0.632** | **0.577** | **0.327** | **0.681** |

*On the real-world scene dataset KITTI-360, SuperPC outperforms the combination of SOTA independent models across all tasks.*

### Key Findings

- SuperPC is the only single model capable of performing all four tasks simultaneously, whereas other methods require cascading multiple models.
- SuperPC maintains its superiority in generalization experiments from ShapeNet → TartanAir (object-to-scene) and TartanAir → KITTI-360 (sim-to-real).
- Using early fusion or deep fusion in isolation yields suboptimal results; the loosely coupled combination via the SMF strategy significantly outperforms both individual designs.
- Supports arbitrary real-valued upsampling factors (e.g., 1.3425×) rather than being limited to integer multiples.

## Highlights & Insights

1. **Value of Unified Thinking**: Unifying four interrelated tasks into a single model leverages their complementary benefits, preventing error accumulation inherent in cascaded pipelines.
2. **Exquisite Multi-Level Conditioning**: The raw, local, and global levels capture pixel-level details, object-level semantics, and overall global structure respectively, forming a complete information hierarchy.
3. **New Benchmark Contributions**: Introduces three new benchmarks covering both object-level and scene-level tasks, along with two generalization evaluation tracks (object-to-scene and sim-to-real).

## Limitations & Future Work

- Quantitative evaluation of the colorization task is relatively limited, as the paper primarily focuses on the first three tasks.
- For extremely sparse or heavily occluded scenes, information from both modalities may still be insufficient.
- Incorporating more modalities (e.g., semantic labels) and scaling up to larger-scale scenes represent promising directions.

## Related Work & Insights

- Compared to dual-task methods like LiDiff, SuperPC covers more tasks and delivers stronger performance.
- The multi-level conditioning design concept can be extended to other 3D generative tasks (such as scene generation and shape editing).
- The spatial mixed fusion strategy provides a flexible and effective paradigm for multimodal 3D processing.

## Rating

⭐⭐⭐⭐ — First to unify four core point cloud processing tasks into a single diffusion model. The academy design is clearly structured, and the three-level conditioning mechanism is both logical and effective. The experiments span multiple object-level and scene-level benchmarks, demonstrating laudable generalization capabilities. However, the evaluation of colorization is relatively weak, and the gains on scene-level data are somewhat moderate.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Unlocking Generalization Power in LiDAR Point Cloud Registration](unlocking_generalization_power_in_lidar_point_cloud_registration.md)
- [\[CVPR 2025\] DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving](diffusiondrive_truncated_diffusion_model_for_end-to-end_autonomous_driving.md)
- [\[CVPR 2025\] V2X-R: Cooperative LiDAR-4D Radar Fusion with Denoising Diffusion for 3D Object Detection](v2x-r_cooperative_lidar-4d_radar_fusion_with_denoising_diffusion_for_3d_object_d.md)
- [\[CVPR 2025\] Distilling Monocular Foundation Model for Fine-grained Depth Completion](distilling_monocular_foundation_model_for_fine-grained_depth_completion.md)
- [\[CVPR 2025\] WeatherGen: A Unified Diverse Weather Generator for LiDAR Point Clouds via Spider Mamba Diffusion](weathergen_a_unified_diverse_weather_generator_for_lidar_point_clouds_via_spider.md)

</div>

<!-- RELATED:END -->
