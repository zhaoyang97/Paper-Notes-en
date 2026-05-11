---
title: >-
  [Paper Note] Extrapolated Urban View Synthesis Benchmark
description: >-
  [ICCV2025][Autonomous Driving][novel view synthesis] This paper presents the first Extrapolated Urban View Synthesis (EUVS) benchmark, which leverages publicly available multi-traversal/multi-vehicle/multi-camera datasets to systematically evaluate the generalization ability of 3DGS and NeRF methods under extrapolation settings, revealing that current methods severely overfit to training viewpoints.
tags:
  - ICCV2025
  - Autonomous Driving
  - novel view synthesis
  - 3D Gaussian Splatting
  - benchmark
  - extrapolation
date: 2026-05-08
content_hash: d583281ede7c24cc
---

# Extrapolated Urban View Synthesis Benchmark

**Conference**: ICCV2025
**arXiv**: [2412.05256](https://arxiv.org/abs/2412.05256)
**Code**: [Project Page](https://ai4ce.github.io/EUVS-Benchmark)
**Area**: Autonomous Driving
**Keywords**: novel view synthesis, 3D Gaussian Splatting, benchmark, extrapolation, autonomous driving

## TL;DR

This paper presents the first Extrapolated Urban View Synthesis (EUVS) benchmark, which leverages publicly available multi-traversal/multi-vehicle/multi-camera datasets to systematically evaluate the generalization ability of 3DGS and NeRF methods under extrapolation settings, revealing that current methods severely overfit to training viewpoints.

## Background & Motivation

- **Core Problem**: Visual simulation for autonomous driving relies on novel view synthesis (NVS). Methods such as 3D Gaussian Splatting perform well under interpolation-based evaluation, yet real-world driving involves viewpoint changes far beyond the training distribution (i.e., extrapolation scenarios). Systematic evaluation of existing methods under extrapolation settings is lacking.
- **Limitations of Prior Work**:
    - Existing evaluations almost exclusively conduct interpolation tests between highly correlated training and test viewpoints, resulting in near-identical training/test distributions.
    - A small number of studies have attempted extrapolated view synthesis (e.g., GGS, VEGS, FreeSim), but the absence of real-world ground-truth data limits them to qualitative analysis only.
    - Prior methods focus on a single difficulty level, with no systematic multi-level evaluation framework.
- **Goal**: To construct a standardized, multi-difficulty extrapolated NVS benchmark that enables comprehensive quantitative and qualitative evaluation using real-world data, thereby promoting the development of more robust NVS methods.

## Method

### Data Sources and Construction

The EUVS benchmark integrates three publicly available autonomous driving datasets:

| Dataset | Characteristics | Usage |
|--------|------|------|
| **nuPlan** | 1,200 hours of driving data across 4 cities, multi-traversal + multi-camera | Level 1 (multi-lane changes) and Level 2 (multi-camera rotation) |
| **Argoverse 2** | 1,000 annotated 3D scenes across 6 U.S. cities | Level 1 and Level 3 (cross-path traversal) |
| **MARS** | Multi-vehicle collaboration + multi-traversal, asynchronous revisits of the same area | Level 3 (intersections, T-junctions, etc.) |

Total scale: **90,810 frames, 345 videos, 104 scenes**.

### Evaluation Framework: Three Difficulty Levels

The paper categorizes extrapolated viewpoint changes into three difficulty levels corresponding to different driving scenarios:

- **Level 1 (Translation Only)**: Vehicle position shifts while heading remains unchanged, corresponding to lane-change scenarios. Models are trained on data from different lanes collected via multi-traversal, and tested on adjacent lanes.
- **Level 2 (Rotation Only)**: Training and testing occur at the same location but with different headings. The front three and rear three cameras of nuPlan (6 total) are used for training, with the two lateral cameras held out for testing.
- **Level 3 (Translation + Rotation)**: Both position and heading differ simultaneously, representing the most challenging setting. Data from different traversal paths through the same intersection (e.g., entering from different directions) are used.

### Scene Preprocessing

- **COLMAP** is used for multi-frame pose estimation and sparse reconstruction to initialize 3DGS.
- **SegFormer** is used to segment and mask movable objects, which are excluded from both training and evaluation.

### Baseline Methods (9 Categories)

1. **Vanilla 3DGS**: Standard 3D Gaussian Splatting
2. **3DGM**: Distinguishes transient from permanent elements using multi-traversal consistency
3. **GaussianPro (GSPro)**: Multi-frame geometry-guided Gaussian densification with planar structural priors
4. **2DGS**: Projects volumetric 3D Gaussians onto 2D planar discs
5. **PGSR**: Unbiased depth rendering with multi-view regularization
6. **VEGS**: Diffusion prior-guided view augmentation to reduce floater artifacts
7. **Feature 3DGS**: Parallel N-dimensional Gaussian rasterization with embedded semantic features
8. **Zip-NeRF**: Anti-aliased grid-based NeRF with multi-sampling and distance normalization
9. **Instant-NGP**: Fast NeRF with multi-resolution hash encoding

### Evaluation Metrics

- **Image Quality**: PSNR, SSIM, LPIPS
- **Semantic Similarity**: DINOv2 feature cosine similarity
- **Geometric Accuracy**: RMSE, $\delta_{1.25}$ (depth estimation metrics)

## Key Experimental Results

### Performance Degradation Across Difficulty Levels (Average Over All Methods)

| Level | PSNR Drop | SSIM Drop | LPIPS Degradation | Feature Similarity Drop |
|------|----------|----------|-----------|-------------|
| Level 1 (Translation Only) | 24.6% | 12.8% | 25.3% | 11.2% |
| Level 2 (Rotation Only) | 25.6% | 14.7% | 62.5% | 14.8% |
| Level 3 (Translation + Rotation) | 30.6% | 19.5% | 70.0% | 35.9% |

→ Degradation worsens with increasing difficulty; at Level 3, LPIPS degrades by as much as 70%, indicating severe perceptual quality collapse.

### Level 1 Test Set Performance (Representative Results)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| GSPro | **16.39** | 0.7189 | **0.2450** |
| 3DGM | 16.35 | **0.7248** | 0.2542 |
| 3DGS | 16.37 | 0.7203 | 0.2599 |
| VEGS | 15.88 | 0.7047 | 0.3062 |

### Key Findings at Level 2

- **VEGS substantially outperforms others in rotation scenarios thanks to its diffusion prior**: PSNR of 23.33 vs. 19.53 for 3DGS (+19.4%), as the diffusion model can hallucinate unseen regions.
- Zip-NeRF achieves PSNR of 29.06 on the training set but collapses to 17.36 on the test set (−40.3%), exhibiting the most severe overfitting.

### Key Findings at Level 3

- All methods yield PSNR below 15, with no method showing a clear advantage.
- Feature 3DGS achieves the best LPIPS (0.3816), though the absolute value remains poor.
- 2DGS performs worst (PSNR of only 11.36), as planar representations cannot handle large viewpoint changes in complex urban scenes.

### Depth Estimation Comparison (Level 1)

3DGM achieves the best results on most depth metrics (RMSE = 0.301, $\delta_1$ = 0.651), while VEGS attains the lowest SqRel but exhibits weaker overall depth accuracy.

### Effect of Multi-Traversal Data Volume

Using GSPro on Level 1 with progressively more training traversals, PSNR and SSIM improve continuously before saturating, demonstrating that additional visual coverage does benefit extrapolation synthesis.

## Highlights & Insights

1. **First quantitative benchmark for extrapolated NVS**: Fills a long-standing gap where only qualitative evaluation was feasible; the three-level difficulty design covers typical real-world driving scenarios (lane change → camera switch → cross-intersection).
2. **Diffusion priors are not a panacea**: VEGS excels in rotation scenarios (Level 2) but similarly collapses at Level 3, suggesting that diffusion priors can compensate for unseen regions but cannot fundamentally resolve large viewpoint offsets.
3. **Geometric improvements yield limited gains**: Planar methods (2DGS, PGSR) better reconstruct road surfaces but perform worse on complex textured objects; ellipsoidal methods (3DGS, 3DGM) better preserve high-frequency details but overfit on road surfaces.
4. **GS vs. NeRF offer complementary strengths**: Implicit representations (NeRF) excel at smooth transitions in low-texture regions (roads, sky), while explicit representations (GS) better preserve high-frequency details (lane markings, guardrails), suggesting hybrid representations as a promising direction.
5. **Lighting inconsistency poses an additional challenge**: GS-W, by disentangling intrinsic and dynamic appearance features, substantially improves both training and test metrics, yet extrapolation performance still degrades significantly.

## Limitations & Future Work

- **Predominantly static scenes**: Dynamic objects are excluded from the benchmark. Preliminary results with the OmniRe dynamic baseline (PSNR of only 15.32) suggest dynamic extrapolation is even more challenging, but this direction remains underexplored.
- **Limited data scale**: Although 90K frames is a sizeable corpus, the per-scene training sets remain small; larger-scale training data may substantially improve extrapolation generalization.
- **Absence of generative baselines**: Methods based on world models or video generation, such as DriveDreamer4D and Vista, are not evaluated; these approaches may inherently be better suited for extrapolation.
- **No depth ground truth**: Depth evaluation relies on pseudo ground truth from Depth Anything, introducing additional noise.
- **Hybrid representations untested**: The paper identifies ellipsoidal–planar hybrid representations as a promising direction but does not implement or validate this approach.
- **COLMAP initialization bottleneck**: Pose alignment for multi-traversal data depends on COLMAP; registration accuracy and efficiency remain limiting factors for large-scale scenes.

## Related Work & Insights

- **RapNeRF / NerfVS**: Early attempts at extrapolated view synthesis that improve novel viewpoints via random ray sampling or geometric scaffolds, but limited to indoor settings.
- **GGS**: Employs a virtual lane generation module to address insufficient lane-change data; focuses exclusively on Level 1.
- **AutoSplat**: Enforces geometry and reflection consistency constraints in dynamic lane-change scenarios.
- **FreeSim / VEGS / SGD**: Leverage diffusion model priors to enhance 3DGS generalization.
- **GS-W**: Disentangles intrinsic properties and dynamic appearance features to handle illumination inconsistency, representing an effective strategy for multi-traversal data.
- **Inspired Directions**:
    - Large-scale pretraining with scene-level fine-tuning (akin to the foundation model paradigm)
    - Hybrid explicit/implicit representations
    - Joint use of diffusion priors and depth regularization
    - Continual learning leveraging increasing numbers of traversals

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] URB -- Urban Routing Benchmark for RL-Equipped Connected Autonomous Vehicles](../../NeurIPS2025/autonomous_driving/urb_--_urban_routing_benchmark_for_rl-equipped_connected_autonomous_vehicles.md)
- [\[ICCV 2025\] Leveraging 2D Priors and SDF Guidance for Dynamic Urban Scene Rendering](leveraging_2d_priors_and_sdf_guidance_for_urban_scene_rendering.md)
- [\[CVPR 2026\] Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](../../CVPR2026/autonomous_driving/sgnlf_spectralgeometric_neural_fields_for_posefre.md)
- [\[NeurIPS 2025\] L2RSI: Cross-View LiDAR-Based Place Recognition for Large-Scale Urban Scenes via Remote Sensing Imagery](../../NeurIPS2025/autonomous_driving/l2rsi_cross-view_lidar-based_place_recognition_for_large-scale_urban_scenes_via_.md)
- [\[CVPR 2026\] SG-NLF: Spectral-Geometric Neural Fields for Pose-Free LiDAR View Synthesis](../../CVPR2026/autonomous_driving/sgnlf_spectralgeometric_neural_fields_for_posefre.md)

</div>

<!-- RELATED:END -->
