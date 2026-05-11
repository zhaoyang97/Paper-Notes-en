---
title: >-
  [Paper Note] MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second
description: >-
  [CVPR 2026][3D Vision][dynamic view synthesis] This paper proposes MoVieS, a feed-forward 4D dynamic scene reconstruction framework that unifies appearance, geometry, and motion modeling via **Dynamic Splatter Pixels**…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "dynamic view synthesis"
  - "4D reconstruction"
  - "3D Gaussian splatting"
  - "point tracking"
  - "feed-forward reconstruction"
date: 2026-05-08
content_hash: b236c33c5e3fe014
---

# MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second

**Conference**: CVPR 2026
**arXiv**: [2507.10065](https://arxiv.org/abs/2507.10065)
**Code**: [Available](https://chenguolin.github.io/projects/MoVieS)
**Area**: 3D Vision
**Keywords**: dynamic view synthesis, 4D reconstruction, 3D Gaussian splatting, point tracking, feed-forward reconstruction

## TL;DR

This paper proposes MoVieS, a feed-forward 4D dynamic scene reconstruction framework that unifies appearance, geometry, and motion modeling via **Dynamic Splatter Pixels**, enabling 4D reconstruction from monocular video in approximately one second while supporting novel view synthesis, 3D point tracking, scene flow estimation, and moving object segmentation.

## Background & Motivation

Existing 3D vision methods suffer from three core limitations:

**Task fragmentation**: Depth estimation, 3D reconstruction, novel view synthesis, and point tracking are treated as independent problems, despite sharing underlying 3D priors. Separate processing wastes the complementary information between tasks.

**Static scene limitation**: Most feed-forward reconstruction methods (e.g., pixelSplat, GS-LRM, VGGT) handle only static scenes and cannot model moving objects.

**Inefficient optimization-based dynamic reconstruction**: Methods such as Shape-of-Motion and MoSca require 10–45 minutes of per-scene optimization and depend on external optical flow or point tracking models for motion supervision, making them complex and difficult to generalize.

Existing feed-forward dynamic methods also have drawbacks: BTimer predicts frames independently without temporal consistency and requires an additional enhancer module; NutWorld lacks explicit motion supervision and uses an orthographic camera that introduces projection distortion. The core motivation of MoVieS is: **Can a single unified feed-forward model jointly output appearance, geometry, and motion and complete 4D reconstruction in one second?** The key insight is that novel view synthesis and motion estimation are mutually beneficial — rendering loss provides dense spatial constraints for motion, while explicit motion supervision helps the model learn temporally consistent geometry.

## Method

### Overall Architecture

MoVieS takes a monocular video with camera parameters and timestamps $\mathcal{V} = \{\mathbf{I}_i, \mathbf{P}_i, \mathbf{K}_i, t_i\}_{i=1}^{N}$ and processes it in three stages:

1. **Feature extraction**: A pretrained image encoder (DINOv2) extracts per-frame features, fused with camera embeddings and timestamp tokens.
2. **Cross-frame attention**: VGGT's geometry-pretrained attention blocks enable inter-frame information exchange, producing shared features rich in cross-frame context.
3. **Three-head prediction**: A depth head, a splatter head (appearance), and a motion head predict all attributes of the dynamic splatter pixels in parallel.

The dynamic splatter pixels are rendered into images at target viewpoints and timestamps via a differentiable 3DGS renderer.

### Key Designs

1. **Dynamic Splatter Pixel**: The core representation. The dynamic scene is decomposed into static Gaussian primitives and time-varying deformation fields. Each pixel corresponds to a splatter pixel $\mathbf{g} = \{\mathbf{x}, \mathbf{a}\}$, where $\mathbf{x} \in \mathbb{R}^3$ is the canonical-space position and $\mathbf{a} \in \mathbb{R}^{11}$ encodes rotation quaternion, scale, opacity, and color. For dynamic content, a time-varying deformation $\mathbf{m}(t) = \{\Delta\mathbf{x}(t), \Delta\mathbf{a}(t)\}$ is appended and applied via simple addition:
$$\mathbf{x} \leftarrow \mathbf{x} + \Delta\mathbf{x}(t), \quad \mathbf{a} \leftarrow \mathbf{a} + \Delta\mathbf{a}(t)$$
Design motivation: Unlike prior work that embeds motion in implicit fields, explicitly separating static geometry from dynamic deformation allows motion to be directly supervised and visualized, and naturally yields near-zero motion for static scenes.

2. **Dual camera conditioning**: Camera parameters are encoded in two complementary forms. (a) **Plücker embeddings**: $\mathbf{P}_i$ and $\mathbf{K}_i$ are converted into pixel-aligned Plücker ray representations and added to the image feature space, providing local geometric constraints. (b) **Camera tokens**: Camera parameters are encoded via a linear layer into global tokens concatenated to the attention sequence, providing global viewpoint information. Ablation studies confirm that combining both yields the best results (PSNR 27.60 vs. Plücker alone 25.81 or camera token alone 26.81).

3. **Motion head**: Adaptive layer normalization (AdaLN) injects sinusoidal encodings of query time $t_q$ into feature tokens, followed by DPT convolutions to predict per-pixel 3D displacement $\Delta\mathbf{x}$ and attribute deformation $\Delta\mathbf{a}$. The head supports motion prediction at arbitrary query timestamps; at inference, varying $t_q$ enables continuous-time 4D reconstruction. Motion maps are visualized by normalizing XYZ coordinates to $[0,1]$ and mapping them to RGB channels.

4. **Decoupled depth head and splatter head**: Rather than predicting all Gaussian attributes with a single head, MoVieS uses a separate depth head (initialized from VGGT) for geometry and a splatter head (trained from scratch) for appearance. The splatter head also incorporates an RGB shortcut connection from the input image to the final convolutional layer to preserve high-frequency detail and color fidelity. This decoupled design better exploits VGGT's geometric priors.

5. **Motion supervision design**: Point-level L1 loss and distribution-level loss are combined:
$$\mathcal{L}_{\text{motion}} = \frac{\lambda_{\text{pt}}}{P}\sum_{i \in \Omega}\|\Delta\hat{\mathbf{x}}_i - \Delta\mathbf{x}_i\|_1 + \frac{\lambda_{\text{dist}}}{P^2}\sum_{(i,j) \in \Omega \times \Omega}\|\Delta\hat{\mathbf{x}}_i \cdot \Delta\hat{\mathbf{x}}_j^\top - \Delta\mathbf{x}_i \cdot \Delta\mathbf{x}_j^\top\|_1$$
The point-level loss provides absolute motion constraints, while the distribution-level loss preserves the relative motion structure between pixels. Ablation shows they are complementary: the point-level loss produces reasonable motion maps, and the distribution-level loss sharpens motion boundaries.

### Loss & Training

The total loss is a weighted combination of three terms: $\mathcal{L} = \lambda_d \mathcal{L}_{\text{depth}} + \lambda_r \mathcal{L}_{\text{rendering}} + \lambda_m \mathcal{L}_{\text{motion}}$

- **Depth loss**: MSE between predicted and GT depth maps plus spatial gradient L1 loss, with invalid values masked out.
- **Rendering loss**: Pixel MSE plus LPIPS perceptual loss ($\lambda_{\text{LPIPS}} = 0.5$), computed over $M$ randomly sampled target timestamps.
- **Weights**: $\lambda_d = 1, \lambda_r = 1, \lambda_m = 10, \lambda_{\text{pt}} = 1, \lambda_{\text{dist}} = 10$.
- **Curriculum training**: Three stages of progressively increasing complexity — (1) static scene pretraining, (2) dynamic scene + multi-view training, (3) high-resolution fine-tuning.
- **Datasets**: Mixed training on 8 heterogeneous datasets (RealEstate10K 70K scenes, TartanAir, MatrixCity, PointOdyssey, DynamicReplica, Spring, VKITTI2, Stereo4D 98K scenes).
- **Engineering**: gsplat rendering backend, DeepSpeed, gradient checkpointing, gradient accumulation, bf16 mixed precision; approximately 5 days on 32× H20 GPUs.

## Key Experimental Results

### Main Results: Novel View Synthesis

| Method | Type | Per-scene Time | RE10K PSNR↑ | DyCheck mPSNR↑ | DyCheck mSSIM↑ | NVIDIA PSNR↑ |
|--------|------|---------------|-------------|-----------------|----------------|--------------|
| DepthSplat | Feed-forward (static) | 0.60s | 26.57 | 13.83 | 43.64 | 17.16 |
| GS-LRM† | Feed-forward (static) | 0.57s | 26.94 | 14.60 | 45.35 | 17.83 |
| Ours (static) | Feed-forward (static) | 0.84s | **27.60** | 15.24 | 47.84 | 18.73 |
| Splatter-a-Video | Optimization-based | 37min | - | 13.61 | 31.31 | 14.39 |
| Shape-of-Motion | Optimization-based | 10min | - | 17.96 | 56.62 | 15.30 |
| MoSca | Optimization-based | 45min | - | 18.24 | 55.14 | 21.45 |
| **MoVieS** | **Feed-forward (dynamic)** | **0.93s** | 26.98 | **18.46** | **58.87** | 19.16 |

### Main Results: 3D Point Tracking (TAPVid-3D)

| Method | ADT EPE3D↓ | ADT δ0.05↑ | ADT δ0.10↑ | DriveTrack EPE3D↓ | Panoptic δ0.05↑ |
|--------|------------|------------|------------|-------------------|-----------------|
| BootsTAPIR† | 0.5539 | 17.73% | 32.97% | 0.0617 | 69.28% |
| CoTracker3† | 0.5614 | 19.88% | 35.82% | 0.0637 | 69.27% |
| SpatialTracker | 0.5413 | 18.08% | 38.23% | 0.0648 | 72.91% |
| **MoVieS** | **0.2153** | **52.05%** | **71.63%** | **0.0472** | **87.88%** |

### Ablation Study

| Motion Supervision Strategy | ADT EPE3D↓ | ADT δ0.05↑ | ADT δ0.10↑ |
|-----------------------------|-----------|-----------|-----------|
| No motion supervision | 0.7938 | 19.58% | 32.86% |
| + Point-level L1 | 0.2262 | 48.74% | 69.93% |
| + Distribution loss | 0.2496 | 45.98% | 66.87% |
| Both combined (Ours) | **0.2153** | **52.05%** | **71.63%** |

| NVS–Motion Synergy | DyCheck mPSNR↑ | NVIDIA PSNR↑ | ADT EPE3D↓ | ADT δ0.05↑ |
|--------------------|---------------|-------------|-----------|-----------|
| NVS without motion | 15.82 | 18.38 | 0.7938 | 19.58% |
| Motion without NVS | 16.26 | 18.98 | 0.3801 | 24.72% |
| **Full model** | **18.46** | **19.16** | **0.2153** | **52.05%** |

### Key Findings

1. **Remarkable speed advantage**: MoVieS completes 4D reconstruction in only 0.93 seconds — 600–2900× faster than optimization-based methods (Shape-of-Motion 10 min, MoSca 45 min) — while achieving comparable or superior performance.
2. **Strong coupling between motion and view synthesis**: Ablation experiments clearly demonstrate that the two tasks are mutually reinforcing. NVS alone cannot learn meaningful motion (EPE3D 0.79 vs. 0.22); motion prediction without NVS is blurry and low-quality. Joint training yields significant improvements on both tasks.
3. **Seamless handling of static and dynamic scenes**: When processing static inputs, predicted motion naturally converges to near zero (< 1e-3), indicating the model implicitly learns to distinguish static from dynamic regions.
4. **Large margin in 3D point tracking**: EPE3D on ADT decreases from the previous best of 0.54 to 0.22 (a 60% improvement), and δ0.05 rises from 19.88% to 52.05%, because displacement is estimated directly in 3D space, avoiding the error accumulation inherent in 2D tracking followed by depth unprojection.
5. **Zero-shot generalization**: Motion maps can be directly applied to scene flow estimation (by converting motion vectors from world to camera coordinates) and moving object segmentation (by thresholding the motion vector norm), without any task-specific fine-tuning.

## Highlights & Insights

- **Elegance of unified representation**: Dynamic Splatter Pixels naturally extend static 3DGS to 4D via simple additive deformation, preserving differentiable rendering integrity. This is more concise and efficient than implicit deformation fields or 4D primitives.
- **Proxy task philosophy**: Novel view synthesis serves as a proxy task for motion learning, providing far denser spatial supervision than sparse point tracking annotations. The idea of "supervising motion through rendering" is broadly transferable.
- **Large-scale heterogeneous training**: The flexible model design enables mixed training across 8 datasets with different annotation types; curriculum learning effectively mitigates instability arising from data heterogeneity.
- **Successful pretraining + fine-tuning for 4D**: Initializing from VGGT reduces training time by approximately 3×, though training from scratch achieves comparable results.

## Limitations & Future Work

1. **Reliance on known camera parameters**: The method assumes accurate poses and intrinsics for input video; pose-free video is not addressed (explicitly left for future work by the authors).
2. **Underperformance on the NVIDIA dataset vs. MoSca**: On multi-view dynamic scenes (NVIDIA PSNR 21.45 vs. 19.16), optimization-based methods still hold an advantage in fine detail fitting.
3. **Training instability**: The three-stage curriculum training and the cost of 32 H20 GPUs pose a significant barrier to reproduction; loss oscillations and None gradients occur during training.
4. **Motion head time complexity**: Each query timestamp requires an independent forward pass, so inference cost grows linearly with dense temporal sampling.

## Rating

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Novelty | ⭐⭐⭐⭐ | The Dynamic Splatter Pixel and the unified appearance–geometry–motion modeling framework are novel, though the base components (3DGS, VGGT, DPT) are existing techniques. |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Covers static/dynamic NVS, 3D point tracking, and zero-shot applications; ablations are carefully designed (motion supervision, NVS–motion synergy, camera conditioning); comparisons are fair (unified camera parameters). |
| Writing Quality | ⭐⭐⭐⭐ | Structure is clear, figures and tables are high quality, motivation is well articulated, and ablation visualizations effectively support design choices. |
| Value | ⭐⭐⭐⭐⭐ | Compresses 4D dynamic reconstruction from minutes to seconds while maintaining competitive performance; the unified framework's practicality and zero-shot generalizability lay a strong foundation for subsequent work. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](more_motion-aware_feed-forward_4d_reconstruction_transformer.md)
- [\[ICLR 2026\] Sharp Monocular View Synthesis in Less Than a Second](../../ICLR2026/3d_vision/sharp_monocular_view_synthesis_in_less_than_a_second.md)
- [\[CVPR 2026\] MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting](motionscale_reconstructing_appearance_geometry_and_motion_of_dynamic_scenes_with.md)
- [\[CVPR 2026\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [\[CVPR 2026\] PhysGaia: A Physics-Aware Benchmark with Multi-Body Interactions for Dynamic Novel View Synthesis](physgaia_a_physics-aware_benchmark_with_multi-body_interactions_for_dynamic_nove.md)

</div>

<!-- RELATED:END -->
