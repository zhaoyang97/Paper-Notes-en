---
title: >-
  [Paper Note] MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting
description: >-
  [CVPR 2026][3D Vision][4D Reconstruction] This paper proposes MotionScale, a scalable 4D Gaussian Splatting framework that reconstructs the appearance, geometry, and motion of large-scale dynamic scenes from monocular video with high fidelity. Through a clustering-based adaptive motion field and a progressive optimization strategy, MotionScale achieves a PSNR of 17.98 on DyCheck and reduces 3D tracking EPE to 0.070, substantially outperforming existing methods.
tags:
  - CVPR 2026
  - 3D Vision
  - 4D Reconstruction
  - Gaussian Splatting
  - Dynamic Scenes
  - Motion Field
  - Monocular Video
date: 2026-05-08
content_hash: 1734f3c81abc493a
---

# MotionScale: Reconstructing Appearance, Geometry, and Motion of Dynamic Scenes with Scalable 4D Gaussian Splatting

**Conference**: CVPR 2026
**arXiv**: [2603.29296](https://arxiv.org/abs/2603.29296)
**Code**: [Project Page](https://hrzhou2.github.io/motion-scale-web/)
**Area**: 3D Vision
**Keywords**: 4D Reconstruction, Gaussian Splatting, Dynamic Scenes, Motion Field, Monocular Video

## TL;DR

This paper proposes MotionScale, a scalable 4D Gaussian Splatting framework that reconstructs the appearance, geometry, and motion of large-scale dynamic scenes from monocular video with high fidelity. Through a clustering-based adaptive motion field and a progressive optimization strategy, MotionScale achieves a PSNR of 17.98 on DyCheck and reduces 3D tracking EPE to 0.070, substantially outperforming existing methods.

## Background & Motivation

1. **Background**: Dynamic 4D scene reconstruction is a core challenge in computer vision. In recent years, NeRF and 3DGS have demonstrated strong performance on static or mildly dynamic scenes, particularly under multi-view settings. More recent works have begun integrating 2D geometric and motion priors (e.g., depth estimation, point tracking) with 4DGS to reconstruct scenes from monocular video.

2. **Limitations of Prior Work**: Although existing methods can produce reasonable view synthesis at observed viewpoints, they exhibit notable deficiencies in geometric accuracy and temporal consistency over long sequences, manifesting as geometric distortions, incoherent motion trajectories, and difficulty scaling to large scenes and long videos.

3. **Key Challenge**: The authors identify two key bottlenecks: (1) **Under-constrained geometry**: supervision signals rely predominantly on view-dependent appearance cues and lack the capacity to enforce 3D structural consistency; (2) **Cumulative temporal drift**: motion models rely on 2D tracking priors that lack 3D awareness, causing errors to accumulate inevitably over long sequences, leading to geometric collapse and inconsistent motion trajectories.

4. **Goal**: To design a motion representation that is both sufficiently expressive and scalable, coupled with a stable optimization strategy, enabling high-fidelity 4D reconstruction of large-scale dynamic scenes from monocular video.

5. **Key Insight**: The authors observe that global deformation fields or fixed-capacity architectures struggle to handle diverse local motions, and thus propose an adaptive scaling of motion field capacity driven by clustering.

6. **Core Idea**: The motion field is parameterized via basis transformations anchored at cluster centers, combined with an adaptive split/prune mechanism and a progressive optimization strategy that decouples foreground and background, enabling scalable 4D Gaussian Splatting.

## Method

### Overall Architecture

The input is a monocular video sequence $\{I_t\}_{t=1}^T$ without known camera parameters. Pre-trained 2D models are first applied to extract monocular depth maps, foreground masks, and dense 2D point tracks; initial camera poses are estimated using $\pi^3$. The scene is represented as a combination of a static background and a dynamic foreground, where the latter consists of a set of canonical-space 3D Gaussians driven by a scalable motion field. Optimization proceeds progressively, beginning with an initial temporal window and expanding incrementally to new frames.

### Key Designs

1. **Cluster-Centric Motion Field**:

    - **Function**: Represents the time-varying motion of dynamic Gaussians via a hierarchical motion model.
    - **Mechanism**: Dynamic Gaussians are partitioned into $K$ disjoint clusters $\{\mathcal{C}_k\}$. Each cluster is assigned a global rigid transformation $\mathbf{G}_k^t \in SE(3)$ and $B$ fine-grained basis transformations $\mathcal{B}_k^t$. Each Gaussian derives its local transformation by blending the basis transformations of its cluster using a learnable coefficient vector $\mathbf{w}_i$, which is then composed with the global transformation to yield the final state: $\boldsymbol{\mu}_i^t = \mathbf{R}_{k,g}^t(\mathbf{R}_{i,\ell}^t \boldsymbol{\mu}_i^0 + \mathbf{t}_{i,\ell}^t) + \mathbf{t}_{k,g}^t$. This design ensures that each Gaussian is influenced by only one cluster, keeping computation efficient, while the blending of basis transformations provides the capacity to express non-rigid deformations.
    - **Design Motivation**: Global MLPs or deformation fields with a fixed number of temporal basis functions lack the expressiveness needed to handle diverse local motion patterns. The clustering design allows the motion field capacity to scale adaptively with scene complexity.

2. **Adaptive Control**:

    - **Function**: Dynamically adjusts cluster topology by splitting motion-inconsistent clusters and pruning overly small ones.
    - **Mechanism**: During the long-term optimization stage, 3D trajectories of Gaussians within each cluster are extracted over the propagation window as feature descriptors. HDBSCAN is first applied to discover density-based sub-clusters, followed by agglomerative clustering to separate two candidate groups; a split is executed if the centroid distance exceeds a threshold. Upon splitting, the original motion parameters are copied to both new clusters to ensure optimization stability. Clusters that become too small are pruned to maintain a compact representation.
    - **Design Motivation**: Analogous to the densification strategy in 3DGS, when Gaussians within a single cluster exhibit significantly non-rigid motion discrepancies, the current representation lacks sufficient granularity and must be split to capture finer motion patterns.

3. **Progressive Optimization**:

    - **Function**: Decomposes the optimization of long videos into manageable stages to ensure temporal consistency.
    - **Mechanism**: Two decoupled propagation stages are employed: (a) **Background extension**: newly uncovered regions in incoming frames are detected, new Gaussians are sampled from depth maps, and camera poses are jointly optimized for sub-pixel refinement; (b) **Foreground propagation**: a three-stage refinement process — initial alignment (unidirectional tracking loss to avoid contaminating already-optimized frames) → short-term consistency (bidirectional tracking loss to enforce local temporal coherence) → long-term refinement (globally sampled frames across the full sequence to resolve cumulative drift) — progressively establishes global motion consistency.
    - **Design Motivation**: Directly optimizing over the full sequence leads to instability and drift. The progressive approach transitions optimization from local to global, and decoupling foreground and background avoids conflicts between their respective optimization objectives.

### Loss & Training

- **Tracking loss** $L_{\text{track}}$: minimizes the discrepancy between rendered 2D trajectories and CoTracker3 priors.
- **Depth consistency loss** $L_{\text{depth}}$: enforces agreement between rendered depth and monocular depth priors at tracked locations.
- **Photometric loss (RGB)**: standard image reconstruction loss.
- **ARAP regularization**: as-rigid-as-possible constraint to encourage local rigidity in motion.
- **Shadow Gaussians**: dedicated "shadow Gaussians" are introduced to model transient shadows cast by moving objects, optimized solely with photometric and segmentation constraints, without geometric or motion supervision.

## Key Experimental Results

### Main Results

| Dataset | Metric | MotionScale | Shape of Motion | GFlow | 4D-Fly |
|---------|--------|-------------|-----------------|-------|--------|
| DyCheck | PSNR↑ | **17.98** | 16.72 | - | 17.03 |
| DyCheck | SSIM↑ | **0.70** | 0.63 | - | 0.60 |
| DyCheck | LPIPS↓ | **0.40** | 0.45 | - | 0.37 |
| NVIDIA | PSNR↑ | **26.75** | 23.37 | - | 22.52 |
| NVIDIA | SSIM↑ | **0.78** | 0.75 | - | 0.69 |

| Method | EPE↓ | δ³ᴅ.05↑ | δ³ᴅ.10↑ | AJ↑ | δ_avg↑ | OA↑ |
|--------|------|---------|---------|-----|--------|-----|
| MotionScale | **0.070** | **47.0** | **76.4** | **37.7** | **50.6** | **87.4** |
| Shape of Motion | 0.082 | 43.0 | 73.3 | 34.4 | 47.0 | 86.6 |
| SpatialTracker | 0.125 | 37.7 | 63.9 | 24.9 | 36.9 | 73.5 |

### Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ | AJ↑ | δ_avg↑ | OA↑ |
|---------------|-------|-------|--------|-----|--------|-----|
| Full Model | 17.98 | 0.70 | 0.40 | 37.7 | 50.6 | 87.4 |
| Global Bases | 16.70 | 0.63 | 0.45 | 34.2 | 46.6 | 86.1 |
| w/o Adaptive Control | 17.21 | 0.67 | 0.42 | 34.9 | 47.0 | 86.6 |
| w/o Pose Ref. | 17.45 | 0.67 | 0.41 | - | - | - |
| w/o Shadow | 16.26 | 0.60 | 0.50 | - | - | - |
| w/o FG Propagation | 16.97 | 0.64 | 0.42 | 34.4 | 46.9 | 86.4 |

### Key Findings

- **Cluster motion field vs. global bases**: The clustering design improves PSNR by 1.28 and AJ by 3.5 over the baseline using globally shared bases (analogous to Shape of Motion), demonstrating that localized motion bases are essential for capturing fine-grained non-rigid deformations.
- **Removing adaptive control** results in a PSNR drop of 0.77 and an AJ drop of 2.8, indicating that dynamically adjusting cluster topology is critical for maintaining motion accuracy.
- **Removing Shadow Gaussians** has the largest impact (PSNR drops from 17.98 to 16.26); without a dedicated shadow representation, foreground Gaussians over-expand into shadowed regions, producing geometric inflation and ghosting artifacts.
- **Pose refinement** yields modest quantitative gains, but qualitative visualizations confirm its importance for preserving sharp textures.

## Highlights & Insights

- The **scalability design of the cluster-centric motion field** is particularly elegant: each Gaussian is influenced by only one cluster, keeping computational cost nearly constant, while the splitting mechanism allows unbounded capacity expansion. This "fixed computation + dynamic capacity" paradigm is transferable to other tasks requiring scalable representations.
- The **three-stage refinement for foreground propagation** represents an important engineering contribution: unidirectional alignment first (preventing noise from new frames from contaminating well-optimized results), followed by bidirectional consistency, and finally global refinement. This conservative-to-aggressive optimization ordering is worth adopting in other progressive optimization settings.
- The introduction of **Shadow Gaussians** addresses a commonly overlooked but practically significant issue: shadows cast by moving objects. Modeling shadows independently rather than forcing foreground Gaussians to absorb them both simplifies the problem and avoids geometric artifacts.

## Limitations & Future Work

- The method depends on pre-trained 2D prior models (depth, segmentation, tracking), and failure modes in these models propagate into the final reconstruction.
- The adaptive cluster splitting relies on HDBSCAN combined with a distance threshold, which may lack robustness under extreme motion patterns.
- Evaluation is conducted on a limited set of benchmarks (DAVIS, DyCheck, NVIDIA), and generalization to larger-scale outdoor scenes remains unvalidated.
- The sensitivity of K-means initialization for cluster assignment on final reconstruction quality is not thoroughly explored.

## Related Work & Insights

- **vs. Shape of Motion**: SoM employs globally shared motion basis functions, whereas this work uses cluster-localized motion bases with adaptive expansion. MotionScale significantly outperforms SoM across all metrics, particularly on long sequences and scenes with large motions.
- **vs. GFlow**: GFlow tends to produce "cloud-like" artifacts and motion discontinuities under large displacements; MotionScale maintains geometric clarity and motion continuity through progressive optimization and cluster constraints.
- **vs. 4D-Fly**: The two methods achieve comparable PSNR, but MotionScale shows clear advantages in SSIM and 3D tracking, highlighting the superiority of the cluster-centric motion field in terms of geometric consistency.

## Rating

- Novelty: ⭐⭐⭐⭐ The cluster-centric motion field with adaptive splitting presents a clear contribution, though the overall approach remains an evolution within the 4DGS framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across three datasets, covering NVS and multi-dimensional 3D/2D tracking metrics, with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and the pipeline diagram is intuitive, though the notation density is high.
- Value: ⭐⭐⭐⭐ Advances the state of the art in monocular 4D reconstruction; the scalable motion field design has practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](4dequine_disentangling_motion_and_appearance_for_4.md)
- [\[CVPR 2026\] MoRe: Motion-aware Feed-forward 4D Reconstruction Transformer](more_motion-aware_feed-forward_4d_reconstruction_transformer.md)
- [\[CVPR 2026\] MoVieS: Motion-Aware 4D Dynamic View Synthesis in One Second](movies_motion-aware_4d_dynamic_view_synthesis_in_one_second.md)
- [\[CVPR 2026\] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)
- [\[ICCV 2025\] MEGA: Memory-Efficient 4D Gaussian Splatting for Dynamic Scenes](../../ICCV2025/3d_vision/mega_memory-efficient_4d_gaussian_splatting_for_dynamic_scenes.md)

</div>

<!-- RELATED:END -->
