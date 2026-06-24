---
title: >-
  [Paper Note] PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting
description: >-
  [CVPR 2025][Autonomous Driving][Panorama Synthesis] PanSplat proposes a feed-forward panorama view synthesis method. By designing a spherical 3D Gaussian pyramid, Fibonacci lattice arrangement, and hierarchical spherical cost volume, it achieves high-efficiency 4K resolution (2048×4096) panoramic generation for the first time, trainable on a single A100 GPU.
tags:
  - "CVPR 2025"
  - "Autonomous Driving"
  - "Panorama Synthesis"
  - "Gaussian Splatting"
  - "Feed-forward"
  - "4K Resolution"
  - "Spherical Representation"
date: 2026-05-08
content_hash: dc691414ba05a2b6
---

# PanSplat: 4K Panorama Synthesis with Feed-Forward Gaussian Splatting

**Conference**: CVPR 2025  
**arXiv**: [2412.12096](https://arxiv.org/abs/2412.12096)  
**Code**: [https://github.com/chengzhag/PanSplat](https://github.com/chengzhag/PanSplat)  
**Area**: 3D Vision / Autonomous Driving  
**Keywords**: Panorama Synthesis, Gaussian Splatting, Feed-forward, 4K Resolution, Spherical Representation

## TL;DR
PanSplat proposes a feed-forward panorama view synthesis method. By designing a spherical 3D Gaussian pyramid, Fibonacci lattice arrangement, and hierarchical spherical cost volume, it achieves high-efficiency 4K resolution (2048×4096) panoramic generation for the first time, trainable on a single A100 GPU.

## Background & Motivation

**Background**: With the popularity of portable 360° cameras, panorama images are widely applied in VR, virtual tourism, robotics, and autonomous driving. Wide-baseline panorama view synthesis (generating novel panoramic views from sparse input perspectives) has become a key task, requiring high resolution, fast inference, and low memory consumption.

**Limitations of Prior Work**: Existing panoramic synthesis methods are typically limited to low resolutions (512×1024) because the spherical projection of panoramas introduces huge memory and computational overhead. Directly extending perspective methods to panoramic resolutions encounters memory bottlenecks during training and inference. NeRF-based methods are too slow for inference, while direct application of 3DGS methods to panoramas faces issues with information redundancy and spherical geometry adaptation.

**Key Challenge**: High-resolution panorama synthesis requires processing 8-16 times the number of pixels compared to perspective views, but directly distributing 3D Gaussians uniformly on a sphere causes over-sampling in the polar areas (due to non-uniform spherical area). Meanwhile, the GPU memory demand for full-resolution end-to-end training far exceeds single-GPU capacity.

**Goal**: Design a feed-forward panoramic synthesis pipeline that supports high-quality panoramic image synthesis at up to 4K resolution while maintaining highly efficient training and inference.

**Key Insight**: Starting from spherical geometry, a 3D Gaussian representation adapted to panoramic characteristics is designed (spherical Gaussian pyramid + Fibonacci lattice), and memory-efficient training is achieved through hierarchical processing and deferred backpropagation.

**Core Idea**: Approximate uniformly distributed 3D Gaussian anchors on the sphere using a Fibonacci lattice, estimate depth via a hierarchical spherical cost volume, and decode attributes using a locally operated Gaussian head. This allows the entire pipeline to be trained with two-step deferred backpropagation on a single GPU.

## Method

### Overall Architecture
Inputs: Multiple panoramic images from different views. Output: Panoramic rendering results from any novel viewpoint. The overall pipeline is divided into four stages: (1) extracting multi-scale spherical features using a pre-trained feature encoder; (2) constructing a hierarchical spherical cost volume and regressing depth via a 3D CNN; (3) placing a spherical 3D Gaussian pyramid on the Fibonacci lattice based on the estimated depth; (4) predicting the attributes of each Gaussian (color, opacity, covariance) using local Gaussian heads, and finally rendering the target viewpoint via differentiable splatting.

### Key Designs

1. **Spherical 3D Gaussian Pyramid + Fibonacci Lattice**:

    - **Function**: Provide a spherically-adaptive 3D Gaussian representation for panoramic scenes.
    - **Mechanism**: Traditional methods place Gaussians on regular grids, but spherical latitude-longitude parameterization causes excessively dense Gaussians in polar regions. The Fibonacci lattice is a mathematical method for approximating uniformly distributed points on a sphere, where point intervals are nearly equal. The writers place Fibonacci lattices of multiple spherical radii layers to form a near-to-far Gaussian pyramid, where the number of Gaussians in each layer is adaptively adjusted based on the spherical area.
    - **Design Motivation**: Solve the polar redundancy problem caused by spherical parameterization, while the pyramid structure allows representing fine details nearby and coarse structures far away, reducing the total number of Gaussians while maintaining quality.

2. **Hierarchical Spherical Cost Volume**:

    - **Function**: Estimate the depth of each pixel on the sphere.
    - **Mechanism**: Drawing on the concept of cost volumes in MVS (Multi-View Stereo), but constructed on a sphere. The depth range is discretized into multiple spherical shells, and features are projected from source views to target views according to spherical geometry on each shell to calculate the matching cost. To reduce memory, a coarse-to-fine hierarchical strategy is adopted: first perform a full-range depth search at low resolution, and then perform local refinement near the initial estimation at high resolution.
    - **Design Motivation**: The GPU memory requirement for a full-resolution cost volume is $O(H \times W \times D)$, which is infeasible for 4K panoramas. The hierarchical strategy reduces the GPU memory requirement by an order of magnitude.

3. **Locally Operated Gaussian Head + Two-step Deferred Backpropagation**:

    - **Function**: Decode the attributes of each 3D Gaussian and achieve memory-efficient training.
    - **Mechanism**: The Gaussian head does not use global operations (such as self-attention) but operates only within local neighborhoods, allowing attribute prediction for each Gaussian to be performed independently. During training, a two-step deferred backpropagation is adopted: the first step only forward-propagates the cost volume part and saves intermediate results; the second step forward-propagates the Gaussian head and rendering, and back-propagates to the saved intermediate results. This avoids storing the entire pipeline in GPU memory simultaneously.
    - **Design Motivation**: End-to-end training of a 4K panoramic pipeline is infeasible on a single A100 (80GB). Two-step deferred backpropagation reduces peak GPU memory from >80GB to an acceptable range.

### Loss & Training
A weighted combination of L1 reconstruction loss and SSIM loss is used to supervise the rendering quality. The training involves pre-training on synthetic datasets and then fine-tuning on real-world scene data.

## Key Experimental Results

### Main Results (Replica Synthetic Dataset)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | Inference Time | Resolution |
|------|-------|-------|--------|---------|--------|
| PanSplat (Ours) | **Best** | **Best** | **Best** | ~0.5s | 2048×4096 |
| Feed-forward baseline | Second Best | Second Best | Second Best | ~0.5s | 512×1024 |
| NeRF-based | Lower | Lower | Higher | >10s | 512×1024 |
| Per-scene optimization | High | High | Low | >Minutes | 512×1024 |

### Ablation Study

| Configuration | PSNR | Description |
|------|------|------|
| Full model | Best | Full model |
| w/o Fibonacci lattice | -1.2dB | Replaced with regular grid, polar quality drops |
| w/o pyramid structure | -0.8dB | Single-layer Gaussian, insufficient distant details |
| w/o hierarchical cost volume | OOM | Full-resolution cost volume exceeds GPU memory |
| w/o deferred backpropagation | OOM | End-to-end training exceeds GPU memory |

### Key Findings
- **Fibonacci lattice is significantly superior to regular grids**: Near the polar regions, regular grids generate severe Gaussian clustering, resulting in rendering artifacts. The uniform distribution of the Fibonacci lattice effectively avoids this issue.
- **4K resolution brings visibly improved quality**: Upgrading from 512×1024 to 2048×4096 yields a PSNR improvement of about 2-3dB, with quality improvements particularly noticeable in detailed areas (such as texture edges and small objects).
- **Fast feed-forward inference**: Compared to per-scene optimization methods that take several minutes, PanSplat requires only about 0.5 seconds of inference time to generate a 4K panorama, making it suitable for real-time VR applications.

## Highlights & Insights
- **Clever application of Fibonacci lattice in spherical tasks**: This is a mathematically known method for uniform spherical sampling, but introducing it to the Gaussian anchor layout of 3DGS is novel. This idea can be directly transferred to all 3D vision tasks involving spherical representations (such as environmental lighting estimation, sky models, etc.).
- **Two-step deferred backpropagation is a practical engineering contribution**: It reduces the peak GPU memory to a range acceptable for a single GPU, freeing high-resolution training from the requirement of multi-GPU clusters. This strategy can be applied to other high-resolution vision tasks.
- **Hierarchical spherical cost volume bridges MVS and panoramas**: Adapting the successful experience of perspective MVS to spherical geometry is key to the effectiveness of the method.

## Limitations & Future Work
- **Dependency on multi-view inputs**: At least two panoramic images from different views are required as inputs, preventing single-image panorama generation.
- **Limited scene generalization capability**: The feed-forward model may experience performance degradation in scenes outside the training distribution (such as extreme lighting, large outdoor scenes).
- **No support for dynamic scenes**: Assumes static scenes and cannot handle panoramic videos with moving objects.
- **Anisotropic modeling of spherical Gaussians**: Currently, each Gaussian may not fully exploit the anisotropic characteristics under the spherical coordinate system. Further spherical harmonics expansion may bring quality improvements.

## Related Work & Insights
- **vs MVSGaussian**: MVSGaussian is a feed-forward 3DGS for perspective views. This paper extends a similar approach to spherical panoramas, where the key innovation lies in the spherically-adapted Gaussian representation and cost volume.
- **vs 360Roam / OmniSyn**: These methods usually employ NeRF representations or require per-scene optimization, leading to slow inference and limited resolutions. PanSplat significantly improves efficiency while maintaining quality.
- **vs Pano-NeRF**: NeRF-based methods are slow to infer in panoramic scenarios. The advantage of 3DGS's rasterization-based rendering is more prominent at high resolutions.

## Rating
- Novelty: ⭐⭐⭐⭐ The three designs (Fibonacci lattice for spherical Gaussian layout, hierarchical spherical cost volume, and deferred backpropagation) complement each other.
- Experimental Thoroughness: ⭐⭐⭐⭐ Verified on both synthetic and real-world datasets, with complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; the visualization of spherical geometry helps with understanding.
- Value: ⭐⭐⭐⭐ Achieves 4K feed-forward panorama synthesis for the first time, directing immediate application value to VR/autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] EVolSplat: Efficient Volume-based Gaussian Splatting for Urban View Synthesis](evolsplat_efficient_volume-based_gaussian_splatting_for_urban_view_synthesis.md)
- [\[CVPR 2025\] Toward Real-World BEV Perception: Depth Uncertainty Estimation via Gaussian Splatting](toward_real-world_bev_perception_depth_uncertainty_estimation_via_gaussian_splat.md)
- [\[CVPR 2025\] Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation](prompting_depth_anything_for_4k_resolution_accurate_metric_depth_estimation.md)
- [\[ICLR 2026\] WorldSplat: Gaussian-Centric Feed-Forward 4D Scene Generation for Autonomous Driving](../../ICLR2026/autonomous_driving/worldsplat_gaussian-centric_feed-forward_4d_scene_generation_for_autonomous_driv.md)
- [\[CVPR 2025\] Generative Gaussian Splatting for Unbounded 3D City Generation](generative_gaussian_splatting_for_unbounded_3d_city_generation.md)

</div>

<!-- RELATED:END -->
