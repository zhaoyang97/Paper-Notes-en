---
title: >-
  [Paper Note] Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction
description: >-
  [ICCV 2025][3D Vision][Parametric curve reconstruction] This paper proposes CurveGaussian, a single-stage method that establishes a bidirectional coupling mechanism between parametric curves and edge-oriented Gaussian primitives, enabling direct end-to-end optimization of 3D parametric curves from multi-view edge maps. By eliminating the error accumulation inherent in two-stage pipelines, the method achieves comprehensive improvements over prior approaches in accuracy, efficiency, and compactness.
tags:
  - ICCV 2025
  - 3D Vision
  - Parametric curve reconstruction
  - 3DGS
  - Bézier curves
  - end-to-end optimization
  - edge reconstruction
date: 2026-05-08
content_hash: 9144d5846a6648df
---

# Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction

**Conference**: ICCV 2025
**arXiv**: [2506.21401](https://arxiv.org/abs/2506.21401)
**Code**: [Project Page](https://zhirui-gao.github.io/CurveGaussian)
**Area**: 3D Vision
**Keywords**: Parametric curve reconstruction, 3DGS, Bézier curves, end-to-end optimization, edge reconstruction

## TL;DR

This paper proposes CurveGaussian, a single-stage method that establishes a bidirectional coupling mechanism between parametric curves and edge-oriented Gaussian primitives, enabling direct end-to-end optimization of 3D parametric curves from multi-view edge maps. By eliminating the error accumulation inherent in two-stage pipelines, the method achieves comprehensive improvements over prior approaches in accuracy, efficiency, and compactness.

## Background & Motivation

Parametric curves are indispensable geometric primitives in CAD and industrial applications, and accurately reconstructing them from multi-view images is an important open problem.

**Fundamental limitations of two-stage methods**:
1. Existing methods (NEF, EMAP, EdgeGaussians) all follow a two-stage pipeline of "3D edge point cloud reconstruction → parametric curve fitting"

**Error accumulation**: Noise from 2D edge detection propagates to the 3D point cloud and further to parametric curve fitting, resulting in spurious branches and broken curves.

**Greedy fitting and local optima**: Iterative fitting methods such as RANSAC tend to produce redundant curves, with complexity growing exponentially with scene scale.

**Low efficiency**: NEF requires 1.5 hours and EMAP requires 2.5 hours of training.

**Core challenge**: Parametric curves are inherently incompatible with rendering-based multi-view optimization (lacking differentiable rendering capability), while neural rendering frameworks (NeRF/3DGS) cannot preserve the geometric continuity of curves.

**Proposed solution**: Establish a **bidirectional coupling** between parametric curves and Gaussian primitives, letting Gaussians serve as "renderable proxies" for curves, and directly optimize curve control points via backpropagation through rendering losses.

## Method

### Overall Architecture

**Input**: Multi-view 2D edge maps + camera poses
**Output**: A set of 3D parametric curves (cubic Bézier curves + first-order Bézier line segments)
**Pipeline**: Random curve initialization → curve-aware Gaussian splatting rendering → backpropagation of rendering loss to optimize control points → adaptive topology optimization

### Key Designs

1. **Bidirectional curve–Gaussian coupling**:

    - Each curve $c_j$ is uniformly sampled to generate $N=12$ Gaussian primitives.
    - **Gaussian attributes are fully determined by curve geometry**:
        - Position: anchored at curve sample points $\mathbf{p}_j(t_i)$, $t_i = \frac{i+0.5}{N}$
        - Orientation: principal axis $\mathbf{v}_0$ aligned with curve tangent $\mathbf{T}_j(t_i)$; remaining axes orthogonalized
        - Scale: $\mathbf{s}^{j,i} = [\|\Delta\mathbf{p}_j^i\|, d_j, d_j]^\top$, with the principal axis far exceeding the others, forming **stick-shaped edge-oriented Gaussians**
        - Opacity: inherited from the parent curve attribute $o_j$
    - A learnable **importance mask** $m^{j,i} \in [0,1]$ is introduced to automatically identify redundant curve segments.
    - **Bidirectionality**: Gaussian attributes are constrained by curve parameters, and rendering gradients are backpropagated to update curve control points.

2. **Parametric curve representation**:

    - **Cubic Bézier curve**: 4 control points, $\mathbf{c}_j(t) = \sum_{k=0}^{3} B_k^3(t)\mathbf{P}_j^k$
    - **First-order Bézier (line segment)**: 2 endpoints, $\mathbf{c}_j(t) = (1-t)\mathbf{P}_j^0 + t\mathbf{P}_j^1$
    - Each curve carries additional opacity $o_j$ and thickness $d_j$

3. **Adaptive topology optimization (four strategies)**:

    - **Linearization**: If a cubic Bézier approximates a straight line (mean squared error from sampled points to the fitted line $< \tau_l$), it is replaced by a first-order Bézier.
    - **Merging**: Adjacent line segments are merged when the directional angle $\theta < \tau_{la}$ and endpoint distance $d < \tau_{ld}$; adjacent Bézier curves are merged if the post-merge error $< \tau_b$.
    - **Splitting**: When a geometric discontinuity is detected (principal axis angle between adjacent Gaussians $> \theta_s$), the curve is split at the discontinuity; segments with low mask values ($m_j^i < \tau_m$) are removed while preserving both sides.
    - **Pruning**: An entire curve is removed when opacity $< \tau_d$ or all Gaussian masks fall below the threshold.
    - **Scheduling**: Linearization begins at 3k iterations, merging at 7k, and both are executed once every 1k iterations.

### Loss & Training

**Edge-aware rendering loss** — addressing gradient collapse caused by sparse edge pixels:
$$\mathcal{L}_{edge} = \frac{|M_I|}{|E_I|}\sum_{i \in N_I}\|I_i - \hat{I}_i\|_2^2 + \frac{|N_I|}{|E_I|}\sum_{i \in M_I}\|I_i - \hat{I}_i\|_2^2$$
Complementary weights balance the contributions of edge and non-edge pixels.

**Smooth connectivity regularization**: $\mathcal{L}_{conn}$ — minimizes the distance between endpoints of adjacent curves.
**Curve smoothness regularization**: $\mathcal{L}_{smo}$ — minimizes orientation differences between adjacent Gaussians.
**Compactness regularization**: $\mathcal{L}_{reg} = \sum \log(1 + o_j^2/0.5)$ — encourages low-opacity curves to be pruned.
**Mask loss**: $\mathcal{L}_m$ — eliminates redundant Gaussians.

Total loss: $\mathcal{L}_{all} = \mathcal{L}_{edge} + \lambda_1\mathcal{L}_{conn} + \lambda_2\mathcal{L}_{smo} + \lambda_3\mathcal{L}_{reg} + \lambda_4\mathcal{L}_m$

Training runs for 10k iterations; after 7k, opacity is frozen and the mask loss is activated.

## Key Experimental Results

### Main Results

ABC-NEF dataset (82 ABC models, DexiNed edge detector):

| Method | Acc.↓ | Comp.↓ | F5↑ | F10↑ | F20↑ | Time↓ |
|--------|-------|--------|-----|------|------|-------|
| NEF | 21.9 | 15.7 | 10.8 | 42.1 | 76.8 | 1.5 h |
| EMAP | 8.8 | 8.9 | 59.1 | 88.9 | 94.9 | 2.5 h |
| EdgeGaussians | 9.6 | 8.4 | 45.2 | 93.7 | 95.7 | 4 min |
| **Ours** | **8.2** | **7.5** | **73.7** | **94.0** | **96.2** | **3 min** |

Efficiency and compactness comparison:

| Method | Training Time↓ | # Bézier↓ | # Segments↓ | # Total Curves↓ |
|--------|---------------|-----------|-------------|----------------|
| NEF | 1.5 h | 22.9 | 0 | 22.9 |
| EMAP | 2.5 h | 9.2 | 36.6 | 45.8 |
| EdgeGaussians | 4 min | 13.0 | 84.9 | 97.9 |
| **Ours** | **3 min** | **6.9** | **22.0** | **28.9** |

### Ablation Study

Ablation of key components:

| Configuration | Acc.↓ | Comp.↓ | F5↑ | Note |
|---------------|-------|--------|-----|------|
| w/o edge-aware loss | degraded | degraded | degraded | gradient collapse |
| w/o topology optimization | increased | increased | decreased | cannot eliminate redundancy |
| w/o smoothness regularization | increased | increased | decreased | non-smooth curves |
| **Full model** | **8.2** | **7.5** | **73.7** | — |

### Key Findings

- **14.5% accuracy improvement**: F5 increases from 45.2% (EdgeGaussians) to 73.7% (DexiNed).
- **70.5% reduction in curve count**: Total curves reduced from 97.9 to 28.9, yielding more compact reconstructions.
- **33% training speedup**: 3 minutes vs. 4 minutes for EdgeGaussians.
- On the Replica real-world dataset, the proposed method produces parametric curves that are more complete and less redundant.

## Highlights & Insights

- **Paradigm shift**: Moving from two-stage to single-stage optimization directly in parametric space fundamentally eliminates error accumulation.
- **Elegant coupling design**: Stick-shaped Gaussians naturally conform to curve geometry (principal axis = tangent, principal scale = segment length, minor scale = line width), allowing rendering gradients to flow naturally to control points.
- **Adaptive topology**: During training, a large initial set of curves is progressively refined into a small, accurate set — the reverse of the conventional 3DGS "grow from few" strategy.
- **Extreme parametric compactness**: Each curve requires only 4 control points and 2 scalar values, far fewer parameters than independent Gaussians.

## Limitations & Future Work

- Reconstruction quality depends on the quality of the 2D edge detector (different detectors affect F5 by ~15%).
- Random initialization may require more iterations to converge to the optimal topology.
- Only cubic and first-order Bézier curves are supported; more general parametric forms such as NURBS are not covered.
- Curve grouping with semantic information has not been explored.

## Related Work & Insights

- **NEF/EMAP**: Two-stage pioneers that reconstruct edge fields with NeRF, but suffer from error accumulation in the fitting stage.
- **EdgeGaussians**: Accelerates edge reconstruction with 3DGS but remains constrained by the two-stage paradigm.
- **DiffVG**: Differentiable 2D vector graphics rendering; this work extends analogous ideas to 3D parametric curves.
- **3DGS**: Provides an efficient differentiable rendering foundation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (single-stage parametric curve optimization + curve–Gaussian coupling)
- Technical Depth: ⭐⭐⭐⭐ (comprehensive adaptive topology strategy)
- Experimental Thoroughness: ⭐⭐⭐⭐ (multi-dataset + efficiency comparison + ablation)
- Value: ⭐⭐⭐⭐ (3-minute training, compact output, well-suited for CAD pipelines)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [\[ICCV 2025\] Robust and Efficient 3D Gaussian Splatting for Urban Scene Reconstruction](robust_and_efficient_3d_gaussian_splatting_for_urban_scene_reconstruction.md)
- [\[ICCV 2025\] CstNet: Constraint-Aware Feature Learning for Parametric Point Cloud](constraint-aware_feature_learning_for_parametric_point_cloud.md)
- [\[ICCV 2025\] OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering](occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)
- [\[ICCV 2025\] Lightweight Gradient-Aware Upscaling of 3D Gaussian Splatting Images](lightweight_gradient-aware_upscaling_of_3d_gaussian_splatting_images.md)

</div>

<!-- RELATED:END -->
