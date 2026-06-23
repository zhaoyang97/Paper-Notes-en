---
title: >-
  [Paper Note] ARTDECO：用分层高斯结构 + 前馈先验做高保真在线 3D 重建
description: >-
  [ICLR 2026][3D Vision][On-the-fly Reconstruction] ARTDECO utilizes feed-forward 3D foundation models (MASt3R / π³) as modular pose and point cloud priors, coupled with a Gaussian decoder that decodes structured Gaussians from multi-scale features, and a hierarchical semi-implicit Gaussian representation with LoD. This system achieves SLAM-level speed, feed-forward rob
tags:
  - ICLR 2026
  - 3D Vision
  - On-the-fly Reconstruction
  - Monocular SLAM
  - 3D Gaussian Splatting
  - Feed-forward Foundation Model
  - Level-of-Detail
date: 2026-05-08
content_hash: 547e5b426e53f24e
---
# ARTDECO: High-Fidelity Online 3D Reconstruction with Hierarchical Gaussian Structure + Feed-forward Priors

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=QxvDyJP7g9](https://openreview.net/forum?id=QxvDyJP7g9)  
**Code**: [https://city-super.github.io/artdeco/](https://city-super.github.io/artdeco/) (Project page, promised to be open-sourced after acceptance)  
**Area**: 3D Vision / Online 3D Reconstruction / SLAM / 3D Gaussian Splatting  
**Keywords**: On-the-fly Reconstruction, Monocular SLAM, 3D Gaussian Splatting, Feed-forward Foundation Model, Level-of-Detail  

## TL;DR
ARTDECO utilizes feed-forward 3D foundation models (MASt3R / π³) as modular pose and point cloud priors, coupled with a Gaussian decoder that decodes structured Gaussians from multi-scale features, and a hierarchical semi-implicit Gaussian representation with LoD. This system achieves SLAM-level speed, feed-forward robustness, and rendering quality approaching per-scene optimization from monocular video streams.

## Background & Motivation
**Background**: Online (on-the-fly) 3D reconstruction from monocular image sequences is a fundamental requirement for applications such as real-to-sim, AR/VR, and robotics. 3D Gaussian Splatting (3DGS) has become the mainstream scene representation due to its explicit representation and efficient rasterization. However, in monocular settings, the lack of reliable geometric cues (scale ambiguity, limited parallax, motion blur, and insufficient overlap) makes it difficult to simultaneously achieve accuracy, speed, and robustness.

**Limitations of Prior Work**: Current 3DGS reconstruction follows two paths, each with significant drawbacks. **Per-scene optimization methods** (MonoGS, On-the-fly-NVS, etc.) rely on poses estimated by SfM/SLAM; they offer high accuracy but are computationally expensive, and their robustness is limited by the fragility of these pipelines. **Feed-forward methods** (learning monocular priors from large-scale data to directly regress poses and Gaussian primitives) exhibit fast inference and cross-scene robustness but suffer from low rendering fidelity and weak global consistency. Furthermore, 3DGS is extremely sensitive to scene scale—as the scene grows, the number of required Gaussian primitives explodes, causing efficiency to drop. Existing post-hoc anchor pruning introduces boundary artifacts and increases memory usage, while adding multi-scale Gaussians during training lacks explicit structural organization.

**Key Challenge**: The inability to reconcile efficiency and accuracy, compounded by the lack of a principled level-of-detail (LoD) mechanism for large-scale navigable scenes.

**Goal**: Build a unified system that integrates the efficiency of feed-forward models with the reliability of per-scene optimization to achieve accurate, robust, and real-time online reconstruction.

**Core Idea**: **"Feed-forward priors as modules + Hierarchical LoD Gaussians as representation."** On one hand, feed-forward foundation models are decomposed into plug-and-play modules for pose estimation, loop closure detection, and dense point cloud prediction within a SLAM-style pipeline, using them to resolve monocular ambiguities while maintaining interactive speeds. On the other hand, a hierarchical semi-implicit Gaussian structure anchored to a sparse spatial grid is designed, utilizing LoD-aware densification to achieve a principled balance between fidelity and rendering efficiency.

## Method

### Overall Architecture
ARTDECO (named after **A**ccurate localization + **R**obust reconstruction + **De**coder-based rendering, also echoing the Art Deco style's emphasis on structure and geometry) processes monocular sequences in a streaming SLAM fashion, consisting of three modules in series.

```mermaid
flowchart LR
    A[单目 RGB 帧流] --> B[前端 Frontend]
    B -->|MASt3R 匹配<br/>估相对位姿| C{帧分类}
    C -->|关键帧/建图帧| D[后端 Backend]
    C -->|普通帧| F[建图模块]
    D -->|π³ 闭环检测<br/>全局 BA| E[一致位姿 + 点云置信度]
    E --> F[建图模块 Mapping]
    F -->|LoG 选点初始化<br/>分层半隐式高斯| G[结构化 3D 高斯场]
    G -->|LoD-aware 光栅化| H[新视角渲染]
```

The frontend estimates the relative pose of each frame relative to the latest keyframe and classifies frames into three categories: ordinary frames, mapping frames, and keyframes. The backend refines keyframe poses through loop closure and global BA while estimating point cloud confidence. The mapping module uses all frame types to initialize point clouds into Gaussians and optimize them incrementally. The specialized roles of the three frame types are key to this design—unlike traditional 3DGS-SLAM which only utilizes keyframes.

### Key Designs

**1. Feed-forward foundation models as plug-and-play priors: Tracking with MASt3R, loop closure with π³.** The frontend treats MASt3R as a prior for two-view reconstruction and matching, obtaining per-frame point clouds, confidence, and pixel correspondences between the current frame and the latest keyframe. The current frame's 3D points are then projected onto the keyframe's image plane, and the Gauss–Newton method is used to minimize reprojection residuals to solve for the $T_{KC} \in \mathrm{SIM}(3)$ relative pose; focal length is jointly optimized when unknown. Since MASt3R predictions are unstable at object boundaries, the authors estimate a local covariance $\Sigma_c$ from a neighborhood of radius $\delta$ for each point to weight residuals and filter unreliable reprojections. For the backend, after ASMK coarse screening of loop closure candidates, the 3D foundation model **π³** generates point clouds for the current frame and the top-$N_a$ candidates. The three most geometrically consistent keyframes are selected based on angular error and connected to the factor graph, which is more robust to weak correspondences and noise than pure ASMK. Notably, ablation shows that replacing the frontend backbone from MASt3R (pair-wise inference) to π³ (multi-view inference) yields worse results, as π³ lacks metric-scale capability and fails to preserve object proportions under viewpoint changes.

**2. Three-way frame splitting + Reprojection confidence.** Keyframes are created when the number of valid correspondences with the latest keyframe falls below a threshold $\tau_k$, and are sent to the backend for pose refinement and to the mapping module for reconstruction. Mapping frames are selected when they provide sufficient parallax (70th percentile of pixel displacement between the current and latest keyframe exceeds $\tau_m$), used to initialize new Gaussians. Ordinary frames, which satisfy neither condition, participate only in gradient refinement of existing details without introducing new structures. Confidence does not directly rely on MASt3R's predicted values but uses reprojection error: point clouds are projected onto the $N_c$ previous keyframes with the highest ASMK scores to calculate average reprojection error $\bar e$. Confidence is defined as $C=1$ (when $\bar e \le \varepsilon_c$), otherwise $C=\frac{1}{\bar e - \varepsilon_c + 1}$, providing a more reliable measure of cross-frame geometric consistency used to lower the initial opacity of Gaussians in low-confidence regions.

**3. LoG-guided probabilistic Gaussian insertion + Semi-implicit region-individual features.** To avoid placing Gaussians at every pixel, authors insert them only in regions requiring refinement: a Laplacian of Gaussian (LoG) operator calculates insertion probability on multi-resolution maps: $P_a(u,v)=\max\big(\min(\|\nabla^2(G_\sigma)*I\|,1)-\min(\|\nabla^2(G_\sigma)*\tilde I\|,1),\,0\big)$ (where $I$ and $\tilde I$ are ground truth and rendered images), prioritizing high-frequency and poorly reconstructed areas, adding them only when exceeding threshold $\tau_a$. Each Gaussian is parameterized by center $\mu$, spherical harmonics SH, opacity $\alpha$, base scale $S_b$, individual feature $f_l$, and voxel index $v_{id}$. The base scale is derived from image-space scale $s'$ and depth as $S_b=\frac{d_i s'}{f}$, and scale and rotation are refined using two MLPs from region features $f_r$ and individual features $f_l$: $S=S_b\cdot\mathrm{MLP}_s(f_r\oplus f_l)$, $R=\mathrm{MLP}_r(f_r\oplus f_l)$. Here $f_r$ encodes voxel local context (spatial voxelization at $\epsilon$, each voxel feature initialized to zero). This "region-shared + individual-unique" semi-implicit design balances global consistency with local distinctiveness.

**4. Distance-aware hierarchical LoD Gaussians.** Gaussians are organized by level $l<L$ (level 0 being the finest, $L-1$ the coarsest). Upon initialization, a level-$l$ Gaussian corresponds to a patch of $2^{2l}$ pixels in the original image, initialized from downsampled input frames at various resolutions. Besides weighting the base scale by $2^{2l}$, each Gaussian carries a distance parameter $d_{max}=D\cdot 2^{2l}$ ($D$ is the distance from Gaussian to camera). During rendering, the inclusion is determined by the observation distance $d_r$: included if $d_r\le d_{max}$, excluded if $d_r>2d_{max}$, and smoothly faded out in between using $\alpha'=\alpha\cdot(2d_{max}-d_r)/d_{max}$. This distance-aware LoD suppresses flickering and maintains stable rendering quality across different scales while preserving efficiency. Training follows a staged streaming strategy: new Gaussians are initialized and optimized for $K$ iterations when mapping/keyframes arrive; ordinary frames trigger $K/2$ iterations without adding Gaussians. Training frames are sampled with 0.2 probability for the current frame and 0.8 for historical frames to prevent local overfitting. After streaming the sequence, a global optimization pass is performed on all frames, with position/rotation gradients backpropagated to camera poses for joint optimization.

## Key Experimental Results

### Main Results: Rendering Quality (Selection from 8 indoor/outdoor benchmarks)

| Dataset | Method | PSNR↑ | SSIM↑ | LPIPS↓ | Training Time↓ |
|---|---|---|---|---|---|
| ScanNet++ | LongSplat | 24.94 | 0.827 | 0.260 | 442.96 min |
| ScanNet++ | OnTheFly-NVS | 18.01 | 0.761 | 0.386 | 2.29 min |
| ScanNet++ | **Ours** | **29.12** | **0.918** | **0.167** | 5.33 min |
| TUM | LongSplat | 25.09 | 0.804 | 0.272 | — |
| TUM | **Ours** | **26.18** | **0.850** | **0.224** | 5.33 min |
| Fast-LIVO2 | LongSplat | 26.37 | 0.792 | 0.276 | 313.60 min |
| Fast-LIVO2 | **Ours** | **29.54** | **0.894** | **0.158** | 6.58 min |
| Waymo | S3PO-GS | 27.28 | 0.865 | 0.352 | 34.89 min |
| Waymo | **Ours** | **28.75** | **0.880** | **0.276** | 6.58 min |

ARTDECO achieves superior quality across all indoor and outdoor datasets, with significant leads on challenging sets like TUM and ScanNet++ which feature structural complexity, motion blur, and noise. Its training time is only slightly behind OnTheFly-NVS but is one to two orders of magnitude faster than LongSplat.

### Main Results: Tracking Accuracy (ATE RMSE, lower is better)

| Dataset | MonoGS | S3PO-GS | MASt3R-SLAM | OnTheFly-NVS | **Ours** |
|---|---|---|---|---|---|
| ScanNet++ | 1.217 | 0.632 | 0.025 | 0.891 | **0.018** |
| TUM | 0.244 | 0.117 | 0.031 | — | **0.025** |
| Waymo | 7.370 | 1.236 | — | 3.118 | **1.213** |

On TUM fr1, ARTDECO (0.028) also outperforms pure SLAM systems like DROID-SLAM (0.038), Go-SLAM (0.035), and MASt3R-SLAM (0.030).

### Ablation Study (ScanNet++)

| Component | Configuration | Metric Change |
|---|---|---|
| Frontend/Backend | Full (ATE 0.018) | Replace backbone with π³ → 0.374; π³→vggt loop closure → 0.096; Remove loop → 0.057; Dense keyframes → 0.094 |
| Mapping | Full (PSNR 29.12 / SSIM 0.918 / LPIPS 0.167) | Remove LoD → 28.13; Remove semi-implicit structure → 28.54; Remove global features → 28.89; Remove mapping frames → 26.38; Remove ordinary frames → 27.20 |

### Key Findings
- **MASt3R > π³ as Frontend Backbone**: Pair-wise inference preserves metric-scale, whereas multi-view inference, despite richer data, suffers scale distortion under viewpoint changes, causing ATE to jump from 0.018 to 0.374.
- **Loop Closure is Indispensable**: Removing it triples the ATE (0.018 → 0.057).
- **Mapping Frames Contribute Most**: Excluding mapping frames drops PSNR by 2.74 dB (29.12 → 26.38); multi-view constraints are critical. Ordinary frames also contribute approximately 1.9 dB.
- **Denser Frames aren't Better**: Using mapping frames and keyframes together for tracking inference actually degrades accuracy—3D foundation models generate ghosting and blur with small-parallax dense inputs, contaminating point clouds and correspondences.

## Highlights & Insights
- **Engineering Aesthetics of Modular Feed-forward Priors**: Instead of retraining large models, MASt3R and π³ are treated as replaceable "pose/loop/point cloud" components within a classic SLAM factor graph. This leverages large-scale pre-training priors while retaining the global optimization and loop closure capabilities of SLAM, making it interpretable, tunable, and interchangeable.
- **Reprojection Confidence as a Reliable Patch**: Since MASt3R's boundary predictions are unstable, the authors redefine confidence using cross-frame reprojection error and down-weight low-confidence Gaussian opacities—a simple yet effective reliability patch.
- **Continuous Fade-out of Distance-aware LoD**: Binding levels to observation distance via $d_{max}=D\cdot2^{2l}$ and linearly interpolating opacity in the $(d_{max}, 2d_{max}]$ range avoids the hard jumps and flickering during LoD transitions.
- **Three-way Frame Splitting**: Quantizing the decision of "whether to build new structures or refine" into threshold $\tau_k$ for correspondence ratios and $\tau_m$ for parallax percentiles allows ordinary frames to be utilized for gradient refinement, a major source of quality improvement.

## Limitations & Future Work
- **Strong Dependency on Feed-forward Foundation Models**: Correspondence and geometric components rely heavily on pre-trained models; robustness degrades under noise, blur, lighting changes, or inputs outside the training distribution.
- **Strong Scene Assumptions**: Implicitly assumes static, rigid scenes with consistent lighting and sufficient parallax. Low-texture surfaces, repetitive structures, or near-degenerate trajectories can cause drift or artifacts.
- **Future Work**: Introduction of uncertainty estimation, adaptive model selection, and stronger priors to improve generalization and reliability in real-world scenes.

## Related Work & Insights
- **Per-scene Optimization 3DGS-SLAM** (MonoGS, S3PO-GS, SEGS-SLAM, On-the-fly-NVS): Highly accurate but slow/fragile; these are the targets the authors aim to surpass in speed and robustness.
- **Feed-forward 3D Foundation Models** (MASt3R, π³, VGGT, pose-free reconstruction): This work does not compete with them but reuses them as modular priors—representing a pragmatic compromise in the "feed-forward vs. optimization" dilemma.
- **LoD / Large-scale 3DGS** (Anchor pruning, multi-scale Gaussians): ARTDECO provides a more principled LoD solution than post-hoc pruning by using hierarchical semi-implicit Gaussians on sparse grids with distance-aware fading, which is insightful for large-scale navigable 3DGS.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Modularly embedding feed-forward foundation models into a classic SLAM factor graph, combined with distance-aware hierarchical semi-implicit Gaussian LoD, is a novel and solid engineering integration.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across eight indoor/outdoor benchmarks, dual metrics for rendering and tracking, and exhaustive ablation of frontend/backend backbones, loop closure, frame classification, LoD, and semi-implicit structures.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with explicit mapping between motivation and design; well-placed formulas and diagrams; some symbols (e.g., source of $f_r$, staged training details) require the appendix for full clarity.
- **Value**: ⭐⭐⭐⭐ — Directly addresses the speed-accuracy-robustness trilemma of online monocular reconstruction, offering a practical path for real-to-sim, AR/VR, and robotics. Promised open-sourcing adds high utility value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Splat and Distill: Augmenting Teachers with Feed-Forward 3D Reconstruction for 3D-Aware Distillation](splat_and_distill_augmenting_teachers_with_feed-forward_3d_reconstruction_for_3d.md)
- [\[ICLR 2026\] Uncertainty-Aware 3D Reconstruction for Dynamic Underwater Scenes](uncertainty-aware_3d_reconstruction_for_dynamic_underwater_scenes.md)
- [\[ICLR 2026\] NGS-Marker: Robust Native Watermarking for 3D Gaussian Splatting](ngs-marker_robust_native_watermarking_for_3d_gaussian_splatting.md)
- [\[ICLR 2026\] 3DGEER: 3D Gaussian Rendering Made Exact and Efficient for Generic Cameras](3dgeer_3d_gaussian_rendering_made_exact_and_efficient_for_generic_cameras.md)
- [\[ICLR 2026\] One2Scene: Geometric Consistent Explorable 3D Scene Generation from a Single Image](one2scene_geometric_consistent_explorable_3d_scene_generation_from_a_single_imag.md)

</div>

<!-- RELATED:END -->
