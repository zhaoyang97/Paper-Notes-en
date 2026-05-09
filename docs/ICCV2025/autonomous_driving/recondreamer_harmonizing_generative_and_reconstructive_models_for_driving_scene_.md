---
title: >-
  [Paper Note] ReconDreamer++: Harmonizing Generative and Reconstructive Models for Driving Scene Representation
description: >-
  [ICCV 2025][Autonomous Driving][Driving scene reconstruction] Building upon ReconDreamer, ReconDreamer++ introduces a Novel Trajectory Deformation Network (NTDNet) to bridge the domain gap between generated data and real observations, and independently models the ground plane to preserve geometric priors. On Waymo, the method achieves performance on par with Street Gaussians on original trajectories while improving NTA-IoU by 6.1%, FID by 23.0% on novel trajectories.
tags:
  - ICCV 2025
  - Autonomous Driving
  - Driving scene reconstruction
  - 3D Gaussian splatting
  - novel view synthesis
  - domain gap bridging
  - ground modeling
date: 2026-05-08
content_hash: 82c1841922d2d1f2
---

# ReconDreamer++: Harmonizing Generative and Reconstructive Models for Driving Scene Representation

**Conference**: ICCV 2025
**arXiv**: [2503.18438](https://arxiv.org/abs/2503.18438)
**Code**: [https://recondreamer-plus.github.io](https://recondreamer-plus.github.io) (project page)
**Area**: Autonomous Driving
**Keywords**: Driving scene reconstruction, 3D Gaussian splatting, novel view synthesis, domain gap bridging, ground modeling

## TL;DR

Building upon ReconDreamer, ReconDreamer++ introduces a Novel Trajectory Deformation Network (NTDNet) to bridge the domain gap between generated data and real observations, and independently models the ground plane to preserve geometric priors. On Waymo, the method achieves performance on par with Street Gaussians on original trajectories while improving NTA-IoU by 6.1%, FID by 23.0% on novel trajectories.

## Background & Motivation

**Background**: Combining reconstructive models (3DGS) with generative models (video diffusion) has become the dominant paradigm for closed-loop autonomous driving simulation. ReconDreamer leverages the world model DriveRestorer to restore video quality on novel trajectories and jointly trains a 4DGS model, achieving notable success in large lateral lane-change scenarios (3–6 meters).

**Limitations of Prior Work**: (1) **Domain gap neglected**: Existing methods train a shared set of Gaussian parameters on a mixture of generated and real sensor observations, ignoring the inherent distributional difference between the two data sources. This leads to a notable performance drop on original trajectories (e.g., ReconDreamer's PSNR/SSIM/LPIPS on original trajectories are significantly lower than Street Gaussians). (2) **Poor ground modeling**: In autonomous driving, camera optical axes are typically nearly parallel to the ground, causing streaking/blurring artifacts in ground reconstruction when viewed from novel viewpoints. Using rendered results that lack 3D consistency as conditions for DriveRestorer amplifies these errors.

**Key Challenge**: Leveraging generative models to enhance novel-trajectory performance introduces synthetic data that degrades original-trajectory quality, making it difficult to simultaneously satisfy quality requirements for both trajectory types in practical simulation. The ground is the most sensitive structured element, and its quality directly affects the reliability of critical cues such as lane markings.

**Goal**: (1) How to bridge the domain gap between generated data and real observations? (2) How to improve modeling quality for structured elements such as the ground? (3) How to improve novel-trajectory performance without degrading original-trajectory quality?

**Key Insight**: Treating generated data as inputs from a different domain than real data and introducing learnable spatial deformations to align the two; separating the ground from the scene and fixing its geometric positions while optimizing only appearance parameters.

**Core Idea**: NTDNet learns spatial deformations of Gaussian parameters between original and novel trajectories to systematically bridge the domain gap, while independent ground modeling with preserved LiDAR geometric priors significantly improves novel-trajectory rendering without sacrificing original-trajectory quality.

## Method

### Overall Architecture

The driving scene is decomposed into three components: **ground**, **non-ground static background**, and **dynamic objects**. For original-trajectory camera poses, Gaussian parameters are rendered directly (bypassing NTDNet); for novel-trajectory camera poses, Gaussian parameters are first corrected by NTDNet before rendering. ReconDreamer's DriveRestorer then restores the novel-trajectory renderings, and the restored results are combined with original data to jointly train the reconstruction model.

### Key Designs

1. **Three-Component Scene Decomposition**:

    - Function: Independently models the scene as ground, non-ground background, and dynamic objects.
    - Mechanism:
        - **Ground model $\mathcal{G}^g$**: Gaussian positions $\boldsymbol{x}$ are initialized from road point cloud segmentation. During training, positions $\boldsymbol{x}$ and the number of Gaussians are fixed; only opacity $\boldsymbol{\gamma}$, covariance $\boldsymbol{\Sigma}$ (decomposed into rotation $\boldsymbol{R}$ and scaling $\boldsymbol{S}$, $\boldsymbol{\Sigma} = \boldsymbol{RSS}^T\boldsymbol{R}^T$), and color $\boldsymbol{c}$ are optimized.
        - **Non-ground background $\mathcal{G}^{bg}$**: Initialized from non-ground point clouds plus random additional points; all parameters are fully optimized.
        - **Dynamic objects $\{\mathcal{G}_i^o\}$**: Defined in local object coordinate frames and transformed to world coordinates at render time via $\boldsymbol{x}_w = R_t \boldsymbol{x}_o + T_t$.
    - Design Motivation: In autonomous driving, camera optical axes are nearly parallel to the ground, leading to unstable ground reconstruction. Fixing ground geometry reduces the search space, while multi-frame fused point clouds provide strong priors, thereby enhancing generalization.

2. **Novel Trajectory Deformation Network (NTDNet)**:

    - Function: Learns spatial deformations of Gaussian parameters between original and novel trajectories to systematically bridge the domain gap.
    - Mechanism: Comprises a pose feature module $\mathcal{F}_\phi$ and a temporal field module $\mathcal{F}_\theta$. A normalized delta pose is first computed: $\Delta p_t = \frac{p_t^{novel} - p_t^{ori}}{L}$. Gaussian parameter offsets are then generated as: $\Delta g = \mathcal{F}_{out}(\mathcal{F}_\phi(\Delta p_t) + \mathcal{F}_\theta(\text{PE}(g), \text{PE}(t)))$, with the final adjusted parameters being $g' = g + \Delta g$.
    - Design Motivation: Directly rendering novel trajectories with the same Gaussian parameters introduces a domain gap. NTDNet adaptively adjusts Gaussian parameters based on trajectory offset and timestep. Critically, **it is activated only for novel trajectories**; original trajectories use unmodified parameters, thereby avoiding any degradation of original-trajectory quality. Both modules are lightweight 8-layer MLPs with hidden dimension 256.

3. **Novel View Depth Supervision**:

    - Function: Provides additional geometric constraints for novel trajectories.
    - Mechanism: Static point clouds across all frames are fused as $Pts_{fuse} = \bigcup_{t=1}^F \{p \in Pts_t | \text{IsStatic}(p)\}$, then projected onto novel-view cameras to obtain sparse depth maps. After filtering with dynamic object masks, these serve as static depth supervision: $D_{novel} = \text{Proj}(Pts_{fuse}, C_{novel}) \odot (1 - M_{novel})$.
    - Design Motivation: Single-frame point cloud reprojection may degrade performance under domain gap conditions; multi-frame fused static point clouds are more stable and reliable.

### Loss & Training

$$\mathcal{L} = \lambda_1 \mathcal{L}_{RGB} + \lambda_2 \mathcal{L}_{SSIM} + \lambda_3 \mathcal{L}_{Depth}$$

The two-stage training pipeline from ReconDreamer is retained: (1) partially train the reconstruction model → render low-quality videos → train DriveRestorer with paired real videos; (2) DriveRestorer restores novel-trajectory renderings → mix restored results with original data to train the final model. Total training runs for 50,000 steps. NTDNet uses only 8-layer MLPs with hidden dimension 256.

## Key Experimental Results

### Main Results — Waymo Dataset

| Method | Orig. Traj. PSNR↑/SSIM↑/LPIPS↓ | 3m NTA-IoU↑ | 3m NTL-IoU↑ | 3m FID↓ |
|--------|--------------------------------|-------------|-------------|---------|
| Street Gaussians | 36.50/0.957/0.115 | 0.498 | 53.19 | 130.75 |
| DriveDreamer4D | 34.37/0.945/0.132 | 0.457 | 53.30 | 113.45 |
| ReconDreamer | 34.31/0.943/0.152 | 0.539 | 54.58 | 93.56 |
| **ReconDreamer++** | **36.29/0.957/0.108** | **0.572** | **57.06** | **72.02** |

Original trajectory: LPIPS improves by 6.4% over Street Gaussians. Novel trajectory (3m): NTA-IoU +6.1%, NTL-IoU +4.5%, FID −23.0%.

### Ablation Study — Waymo

| Depth Loss | Ground Model | NTDNet | Orig. PSNR/LPIPS | 3m NTA-IoU | 3m NTL-IoU | 3m FID |
|------------|-------------|--------|------------------|------------|------------|--------|
| ✗ | ✗ | ✗ | 34.31/0.152 | 0.539 | 54.58 | 93.56 |
| ✓ | ✗ | ✗ | 35.13/0.121 | 0.561 | 55.91 | 77.51 |
| ✓ | ✓ | ✗ | 34.93/0.117 | 0.566 | 56.89 | 75.22 |
| ✓ | ✓ | ✓ | **36.29/0.108** | **0.572** | **57.06** | **72.02** |

Each component contributes incrementally. NTDNet yields the largest gain on original trajectories (PSNR +1.36); Ground Model yields the largest gain on NTL-IoU (+1.79 for 3m lane-change).

### Cross-Dataset Validation

| Dataset | Comparison | Key Metric | Gain |
|---------|-----------|------------|------|
| nuScenes | ReconDreamer++ vs ReconDreamer | 3m NTA-IoU: 0.365 vs 0.325 | +12.3% |
| PandaSet | ReconDreamer++ vs ReconDreamer | 3m FID: 71.7 vs 74.9 | −4.3% |
| EUVS | ReconDreamer++ vs ReconDreamer | LPIPS: 0.306 vs 0.329 | −7.0% |

### Key Findings

- **NTDNet is the key component**: Its novel-trajectory-only activation design simultaneously improves novel-trajectory performance and substantially enhances original-trajectory quality (PSNR 34.93→36.29), perfectly resolving the domain gap introduced by the generative model.
- **Ground Model is critical for lane markings**: Independent ground modeling (fixed geometry, appearance-only optimization) improves NTL-IoU from 53.55 to 55.34 (+3.3%) on 6m large lane-change scenarios.
- **Depth supervision is foundational**: Strong geometric constraints provide the basis for subsequent components; introducing it alone reduces FID from 93.56 to 77.51.
- **Consistent effectiveness across four datasets**: Outperforms the predecessor on Waymo/nuScenes/PandaSet/EUVS, demonstrating the generality of the framework.

## Highlights & Insights

- **The "deform only for novel trajectories" design** is the most elegant engineering decision: NTDNet activates only on novel trajectories while original trajectories are rendered directly. This prevents additional noise from the deformation network from affecting existing quality, while allowing the network to focus on learning spatial deformations induced by trajectory differences. This principle generalizes to any domain gap problem — apply adaptation only to cross-domain inputs and leave the source domain unchanged.
- **Fix ground geometry, learn only appearance**: The strong geometric priors provided by multi-frame fused LiDAR point clouds reduce the ground optimization space from position + appearance to appearance only. For autonomous driving scenarios where camera optical axes are nearly parallel to the ground, this substantially reduces optimization difficulty.
- **Multi-frame fused static point clouds for depth supervision**: More stable and complete coverage than single-frame point clouds, while avoiding interference from dynamic objects.

## Limitations & Future Work

- NTDNet models deformations with simple MLPs without accounting for spatial locality — near and far Gaussians should exhibit different deformation patterns.
- The method still relies on the two-stage training pipeline of DriveRestorer; end-to-end training may be more efficient.
- The Ground Model depends on the accuracy of road point cloud segmentation; segmentation errors propagate into geometric priors.
- Dynamic object modeling follows Street Gaussians without special treatment for novel trajectories.
- Generalization to more extreme lane-change scenarios (>6m) has not been validated, and broader generalization may be required for practical deployment.

## Related Work & Insights

- **vs. ReconDreamer**: The direct predecessor. ReconDreamer improves novel trajectories via DriveRestorer but neglects the domain gap, degrading original-trajectory performance. ReconDreamer++'s NTDNet perfectly resolves this trade-off.
- **vs. Street Gaussians**: A purely reconstructive approach that achieves the best original-trajectory performance but cannot generalize to novel trajectories. ReconDreamer++ matches Street Gaussians on original trajectories while substantially outperforming it on novel trajectories.
- **vs. DriveDreamer4D**: Uses a world model to directly generate novel-trajectory videos for mixed 4DGS training, but similarly suffers from domain gap, causing original-trajectory degradation (PSNR 34.37 vs. 36.50).
- **vs. FreeVS**: Relies on LiDAR point clouds for free-viewpoint synthesis and cannot handle upper regions without LiDAR coverage.

## Rating

- Novelty: ⭐⭐⭐⭐ — NTDNet and independent ground modeling are elegant designs, though the improvements are incremental in scope.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive validation on four datasets with detailed ablations and both qualitative and quantitative analysis.
- Writing Quality: ⭐⭐⭐⭐ — Problem motivation is clearly articulated; ablation logic is well structured.
- Value: ⭐⭐⭐⭐ — Effectively addresses the domain gap in generative-reconstructive hybrid training, with practical value for closed-loop simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering](leveraging_2d_priors_and_sdf_guidance_for_urban_scene_rendering.md)
- [\[ICCV 2025\] DiST-4D: Disentangled Spatiotemporal Diffusion with Metric Depth for 4D Driving Scene Generation](dist-4d_disentangled_spatiotemporal_diffusion_with_metric_depth_for_4d_driving_s.md)
- [\[ICLR 2026\] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](../../ICLR2026/autonomous_driving/drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)
- [\[ICCV 2025\] Distilling Diffusion Models to Efficient 3D LiDAR Scene Completion](distilling_diffusion_models_to_efficient_3d_lidar_scene_completion.md)
- [\[ICCV 2025\] Generative Active Learning for Long-tail Trajectory Prediction via Controllable Diffusion Model](generative_active_learning_for_long-tail_trajectory_prediction_via_controllable_.md)

</div>

<!-- RELATED:END -->
