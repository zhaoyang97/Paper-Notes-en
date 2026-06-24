---
title: >-
  [Paper Note] Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction
description: >-
  [ICCV 2025][3D Vision][3D Gaussian Splatting] This paper presents the first integration of event cameras with deformable 3D Gaussian Splatting (3D-GS) for dynamic scene reconstruction. It introduces a GS-Threshold Joint Modeling (GTJM) strategy and a Dynamic-Static Decomposition (DSD) strategy, achieving state-of-the-art rendering quality and speed on a newly constructed event-4D benchmark (average PSNR improvement of 2.73 dB on synthetic data…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "event camera"
  - "dynamic scene reconstruction"
  - "threshold modeling"
  - "dynamic-static decomposition"
date: 2026-05-08
content_hash: 6cdae9008bb41095
---

# Event-boosted Deformable 3D Gaussians for Dynamic Scene Reconstruction

**Conference**: ICCV 2025  
**arXiv**: [2411.16180](https://arxiv.org/abs/2411.16180)  
**Code**: Coming soon  
**Area**: 3D Vision  
**Keywords**: 3D Gaussian Splatting, event camera, dynamic scene reconstruction, threshold modeling, dynamic-static decomposition

## TL;DR

This paper presents the first integration of event cameras with deformable 3D Gaussian Splatting (3D-GS) for dynamic scene reconstruction. It introduces a GS-Threshold Joint Modeling (GTJM) strategy and a Dynamic-Static Decomposition (DSD) strategy, achieving state-of-the-art rendering quality and speed on a newly constructed event-4D benchmark (average PSNR improvement of 2.73 dB on synthetic data, rendering speed 1.71× faster than 4D-GS).

## Background & Motivation

Dynamic scene reconstruction and novel view synthesis are fundamental to immersive applications such as VR/AR. While 3D-GS enables real-time rendering via efficient differentiable rasterization, its dynamic extensions (e.g., 4D-GS, Deformable-3DGS) are constrained by the inherent limitations of RGB cameras:

**Low frame rate**: RGB cameras lack inter-frame motion information, degrading reconstruction quality for fast-moving objects.

**Motion blur**: High-speed motion scenes further deteriorate reconstruction quality.

**Advantages of event cameras**: Microsecond-level temporal resolution enables capturing continuous inter-frame motion and near-infinite viewpoint supervision signals. However, incorporating events into 3D-GS faces a core challenge: **threshold variation modeling**. Event triggering depends on a brightness change threshold $C$ that varies complexly across polarity, spatial location, and time. Existing methods that assume a constant threshold significantly degrade event supervision quality (see Fig. 3a).

## Method

### Overall Architecture

The method consists of two core strategies:
1. **GS-Threshold Joint Modeling (GTJM)**: Addresses event threshold variation.
2. **Dynamic-Static Decomposition (DSD)**: Separates dynamic and static Gaussians to improve efficiency and quality.

### GS-Threshold Joint Modeling (GTJM)

The brightness change model for an event camera is:

$$E(t, t+\Delta t) = \int_{t}^{t+\Delta t} C \cdot e(\tau) d\tau$$

The rendered estimated brightness change is:

$$\hat{E}(t, t+\Delta t) = \log(\hat{I}(t+\Delta t)) - \log(I(t))$$

**Stage 1: RGB-assisted threshold estimation**

Ground-truth brightness changes between RGB frames are used to supervise threshold optimization. Events are accumulated into an Event Count Map $ECM_{t,f} \in \mathbb{R}^{B \times P \times H \times W}$, with learnable threshold parameter $\hat{C}_{t,f}$:

$$\hat{E}_{thres}(t,f) = \sum_{b=1}^{B}\sum_{p=1}^{P}(ECM_{t,f} \odot \hat{C}_{t,f})_{b,p,:,:}$$

Threshold modeling loss: $\mathcal{L}_{thres} = \|E_{thres}(t,f) - \hat{E}_{thres}(t,f)\|_2^2$

**Stage 2: GS-enhanced threshold refinement**

Sparse RGB frames provide insufficient supervision. The key insight is that a trained 3D-GS model can render intermediate frames as pseudo-supervision to enhance threshold optimization. The GS parameters are frozen, and the threshold is optimized jointly using $\mathcal{L}_{thres}$ and $\mathcal{L}_{event}$.

**Joint optimization**: The threshold and 3D-GS are ultimately optimized simultaneously:

$$\hat{C}^*, GS^* = \arg\min_{\hat{C}, GS}(\mathcal{L}_{thres} + \mathcal{L}_{event} + \mathcal{L}_{rgb})$$

This forms a mutually reinforcing positive cycle: optimized thresholds improve event supervision → better 3D-GS → more accurate pseudo frames → more precise threshold estimation.

### Dynamic-Static Decomposition (DSD)

**Problem**: Existing methods model the entire scene with dynamic Gaussians uniformly, wasting deformation field capacity and reducing rendering speed.

**2D decomposition**: The inherent inability of static Gaussians to represent motion is exploited. For the first 3k iterations, only static Gaussians are used for training, resulting in naturally poor reconstruction quality in dynamic regions. Multi-scale features extracted by a pretrained VGG19 are used to compute cosine similarity maps between rendered and ground-truth images. The resulting histogram exhibits a bimodal distribution, from which dynamic region masks are generated via Otsu's method.

**2D→3D correspondence**: Dynamic region pixels from multiple views are back-projected into 3D space, and spatially proximate 3D points are mapped to Gaussians.

**Buffer-based soft decomposition**: Dual radii $r_1$ and $r_2$ are employed: regions within $r_1$ are dynamic, regions beyond $r_2$ are static, and the intermediate buffer zone is pruned to allow adaptive density control for refining decomposition boundaries.

**Joint rendering**: Static Gaussians bypass the deformation field and are merged with deformed dynamic Gaussians before being passed to the rasterizer. The deformation field uses an MLP to output positional, rotational, and scaling offsets.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{thres} + \mathcal{L}_{event} + \mathcal{L}_{rgb}$$

where $\mathcal{L}_{rgb} = (1-\lambda_s)\|\hat{I}(t) - I(t)\|_1 + \lambda_s \mathcal{L}_{D-SSIM}(\hat{I}(t), I(t))$

## Key Experimental Results

### Main Results: Quantitative Results on Synthetic Dataset (Table 2, average over 8 scenes)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ |
|--------|-------|-------|--------|------|
| 3D-GS (static baseline) | ~22.7 | ~0.913 | ~0.098 | ~233 |
| K-Planes | ~23.2 | ~0.913 | ~0.044 | ~2.37 |
| 4D-GS | ~25.6 | ~0.944 | ~0.069 | ~89 |
| Deformable-3DGS | ~25.5 | ~0.938 | ~0.033 | ~70 |
| Event-4DGS | ~28.8 | ~0.950 | ~0.039 | ~55 |
| **Ours** | **~31.6** | **~0.966** | **~0.022** | **~156** |

Key findings:
- Event integration (Event-4DGS vs. Deformable-3DGS): average +3.28 dB
- Threshold modeling (Ours vs. Event-4DGS): average +2.73 dB
- Rendering speed: average 1.71× faster than 4D-GS

### Real-World Dataset (Table 3)

| Method | Excavator PSNR | Jeep PSNR | Flowers PSNR | Eagle PSNR |
|--------|---------------|-----------|-------------|-----------|
| 4D-GS | 28.35 | 28.34 | 26.82 | 27.59 |
| Event-4DGS | 29.67 | 29.64 | 27.53 | 29.08 |
| **Ours** | **31.28** | **30.41** | **28.57** | **31.29** |

FPS is also substantially higher: Ours 179/89/149/192 vs. Event-4DGS 57/47/40/63.

### Ablation Study (Table 4, average on synthetic dataset)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ |
|--------|-------|-------|--------|------|
| w/o GTJM | 29.39 | 0.956 | 0.034 | 153 |
| w/o Joint Opt. in GTJM | 30.87 | 0.963 | 0.026 | 152 |
| w/o DSD | 30.78 | 0.961 | 0.026 | **57** |
| w/o Buffer-based Soft Dec. | 31.02 | 0.963 | 0.025 | 138 |
| **Full** | **31.56** | **0.966** | **0.022** | 156 |

Key findings:
- GTJM contributes +2.17 dB PSNR improvement
- DSD increases FPS from 57 to 156 (2.74×) while maintaining quality
- Buffer-based soft decomposition contributes an additional +0.54 dB

### Mutual Enhancement Validation for Threshold Modeling (Table 1)

| Direction | Stage | Effect |
|-----------|-------|--------|
| TM→3D Rec. | RGB-assisted init. → joint optimization | PSNR: 24.46 → 26.63 |
| 3D Rec.→TM | Frozen GS assistance | MSE: 8.317 → 7.077 (×10⁻⁴) |
| Joint optimization | Simultaneous optimization of both | PSNR: 28.01, MSE: 6.322 |

## Highlights & Insights

1. **Mutually reinforcing paradigm**: GS-Threshold Joint Modeling creates a positive cycle — improved thresholds yield more accurate event supervision, while better 3D-GS provides more precise pseudo frames for threshold refinement.
2. **Clever exploitation of "static Gaussians cannot represent motion"**: Dynamic regions are automatically identified through reconstruction error in the first 3k iterations, requiring no additional semantic or motion priors.
3. **Robustness of buffer-based soft decomposition**: Reconstruction quality stabilizes once the buffer size exceeds approximately 12 basic units, reducing hyperparameter sensitivity.
4. **First event-4D benchmark**: Comprising 8 synthetic and 4 real-world scenes, it provides a standardized evaluation platform for future research.

## Limitations & Future Work

1. The real-world data acquisition system (beam splitter + event camera + frame camera + STM32) is complex and costly to deploy.
2. The monocular setup limits 3D reconstruction accuracy.
3. DSD is performed only once, which may be insufficiently flexible for scenes where dynamic regions change significantly over time.

## Related Work & Insights

- Unlike DE-NeRF (Ma et al., 2023), the first dynamic NeRF to incorporate events, this work achieves real-time rendering within the 3D-GS framework.
- The threshold modeling approach is generalizable to other event camera applications such as SLAM and optical flow estimation.
- Combining the dynamic-static decomposition strategy with scene flow estimation could further improve reconstruction quality in dynamic regions.

## Rating ⭐⭐⭐⭐

Novelty ★★★★☆: First integration of event cameras into deformable 3D-GS; both threshold joint modeling and dynamic-static decomposition are novel contributions.  
Experimental Thoroughness ★★★★★: A self-constructed benchmark with comprehensive evaluation on both synthetic and real-world data; detailed ablation studies.  
Writing Quality ★★★★☆: Method description is clear and well-illustrated.  
Value ★★★☆☆: Relies on event camera hardware, limiting applicability to relatively niche scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] TimeFormer: Capturing Temporal Relationships of Deformable 3D Gaussians for Robust Reconstruction](timeformer_capturing_temporal_relationships_of_deformable_3d_gaussians_for_robus.md)
- [\[ICCV 2025\] DeGauss: Dynamic-Static Decomposition with Gaussian Splatting for Distractor-free 3D Reconstruction](degauss_dynamic-static_decomposition_with_gaussian_splatting_for_distractor-free.md)
- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [\[CVPR 2026\] FastEventDGS: Deformable Gaussian Splatting for Fast Dynamic Scenes from a Single Event Camera](../../CVPR2026/3d_vision/fasteventdgs_deformable_gaussian_splatting_for_fast_dynamic_scenes_from_a_single.md)
- [\[ICCV 2025\] Can3Tok: Canonical 3D Tokenization and Latent Modeling of Scene-Level 3D Gaussians](can3tok_canonical_3d_tokenization_and_latent_modeling_of_scene-level_3d_gaussian.md)

</div>

<!-- RELATED:END -->
