---
title: >-
  [Paper Note] 3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes
description: >-
  [CVPR 2025 (Highlight)][3D Vision][3D Gaussian Splatting] Replaces Gaussian primitives with 3D smooth convex primitives for radiance field rendering. By defining convex hulls using point sets + LogSumExp smoothing + custom CUDA rasterizer, this method outperforms 3DGS on T&T and Deep Blending using fewer primitives.
tags:
  - "CVPR 2025 (Highlight)"
  - "3D Vision"
  - "3D Gaussian Splatting"
  - "Convex Primitives"
  - "Smooth Convex"
  - "Differentiable Rendering"
  - "Novel View Synthesis"
date: 2026-05-08
content_hash: 8e968b24faed60e7
---

# 3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes

**Conference**: CVPR 2025 (Highlight)  
**arXiv**: [2411.14974](https://arxiv.org/abs/2411.14974)  
**Code**: [https://convexsplatting.github.io/](https://convexsplatting.github.io/)  
**Area**: 3D Vision / Novel View Synthesis / Radiance Fields  
**Keywords**: 3D Gaussian Splatting, Convex Primitives, Smooth Convex, Differentiable Rendering, Novel View Synthesis  

## TL;DR
Replaces Gaussian primitives with 3D smooth convex primitives for radiance field rendering. By defining convex hulls using point sets + LogSumExp smoothing + custom CUDA rasterizer, this method outperforms 3DGS on T&T and Deep Blending using fewer primitives.

## Background & Motivation
3DGS achieves real-time novel view synthesis using millions of 3D Gaussian primitives. However, Gaussian primitives possess two inherent limitations: (1) they lack explicit physical boundaries, making it difficult to accurately represent flat surfaces or sharp edges; (2) their symmetric diffusion, which resembles ellipsoids, requires a massive number of primitives to fill corners and flat planes—analogous to the sphere-packing problem where filling space with spherical objects always leaves gaps. GES improves edge representation using generalized exponential functions, and 2DGS improves surface representation using 2D disks, but neither fundamentally addresses the issue of insufficient primitive shape flexibility.

## Core Problem
Can we use primitives that are more flexible than Gaussians to represent radiance fields—primitives capable of representing sharp edges, flat surfaces, and dense volumes simultaneously—while maintaining real-time rendering capabilities?

## Method

### Overall Architecture
The 3D Convex Splatting pipeline: SfM point cloud initialization $\rightarrow$ each convex primitive defined by $K$ 3D points (default $K=6$) $\rightarrow$ perspective projection to 2D $\rightarrow$ Graham Scan algorithm to compute 2D convex hulls $\rightarrow$ signed distance defined from the convex hull line segments $\rightarrow$ LogSumExp smoothing $\rightarrow$ Sigmoid to obtain the indicator function/alpha value $\rightarrow$ tile-based $\alpha$-blending rendering $\rightarrow$ optimization with L1 + D-SSIM + mask regularization.

### Key Designs
1. **Point-Set Convex Representation**: Instead of using plane normals to define the convex hull (as in CvxNet), it is implicitly defined by the convex hull of $K$ 3D points (allowing the points to move freely and the shape to deform flexibly). After projecting to 2D, the Graham Scan algorithm is used to efficiently compute the 2D convex hull, yielding a set of line segments to define the indicator function. Each point has full gradient propagation, allowing backpropagation to optimize the shape of the convex primitive, which is more natural than planar parameterization.

2. **Control of Smoothness $\delta$ and Sharpness $\sigma$**: $\delta$ controls the hardness of the convex vertices (large $\rightarrow$ sharp corners, small $\rightarrow$ rounded corners), while $\sigma$ controls the boundary diffusion of the radiance field (large $\rightarrow$ dense, small $\rightarrow$ diffuse). These two parameters allow the convex primitive to represent sharp polyhedra or degenerate into Gaussian-like shapes, making its representation capability strictly superior to that of Gaussians.

3. **Adaptive Convex Densification**: Unlike the clone/split strategy of 3DGS, 3DCS observes that regions with large $\sigma$ loss correspond to under-reconstructed or over-reconstructed areas. It splits each convex primitive directly into $K$ sub-convex primitives (a 6-point convex primitive $\rightarrow$ 6 downscaled convex primitives), where the centers of the sub-convexes correspond to the $K$ defining points of the original primitive to ensure complete spatial coverage. During splitting, $\sigma$ is increased to encourage denser reconstruction.

4. **Perspective-Aware Scaling**: Multiplies $\delta$ and $\sigma$ by the distance $d$ for scaling. This ensures that distant and near convex primitives maintain consistent visual effects in the 2D projection.

### Loss & Training
- Loss: $\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda\mathcal{L}_1 + \lambda\mathcal{L}_{D-SSIM} + \beta\mathcal{L}_m$, with $\lambda=0.2$, $\beta=0.0005$.
- Fibonacci sphere algorithm is used to initialize the point distribution, with the initial sphere radius equal to $1.2 \times$ the average distance of the nearest 3 neighbors.
- Training takes about 60-87 minutes (slightly slower than 3DGS's 42 minutes, but much faster than MipNeRF360's 48 hours).
- Each convex primitive has 69 parameters (vs 59 parameters per Gaussian in 3DGS).
- Initial $\delta=0.1$, $\sigma=0.00095$; choosing a more diffuse initial value allows the convex primitives to cover the space first before gradually sharpening.
- Densification starts from iteration 500. Densification and pruning are performed every 200 iterations, stopping densification after 9000 iterations while continuing pruning.
- Pruning criteria: opacity < 0.03 or size > $0.3 \times$ scene size.
- High-quality version uses 32-bit precision, while the lightweight version uses 16-bit precision + a higher densification threshold.

## Key Experimental Results

| Dataset | Metrics | 3DCS | 3DGS | Gain | Memory |
|--------|------|------|------|------|------|
| T&T | LPIPS/PSNR/SSIM | 0.157/23.95/0.851 | 0.183/23.14/0.841 | -0.026/+0.81/+0.01 | 282 vs 411MB |
| Deep Blending | LPIPS/PSNR/SSIM | 0.237/29.81/0.902 | 0.243/29.41/0.903 | -0.006/+0.40/-0.001 | 332 vs 676MB |
| Mip-NeRF360 | LPIPS/PSNR/SSIM | 0.207/27.29/0.802 | 0.214/27.21/0.815 | -0.007/+0.08/-0.013 | 666 vs 734MB |
| Mip-NeRF360 (Indoor) | LPIPS/PSNR | 0.166/31.33 | 0.189/30.41 | -0.023/+0.92 | - |

### Ablation Study
- **Number of defining points $K$**: $K=4$ already outperforms 3DGS, $K=6$ offers the best performance-cost ratio, and $K>6$ shows diminishing returns.
- **Number of densification splits**: Splitting into 6 new convex primitives performs best (vs. splitting into 2 or 3).
- **Perspective scaling**: Without scaling, PSNR drops by 5-6dB; $\sqrt{d}$ scaling is optimal.
- **Indoor vs. Outdoor**: Convex primitives show a greater advantage in indoor scenes (many flat planes/corners) (+0.9 PSNR), with smaller gains in outdoor natural scenes. This aligns with intuition: indoor scenes have more regular geometric structures, where the sharp corner and flat surface representation capabilities of convex primitives are better utilized. Specific indoor scene data: LPIPS 0.166 vs 3DGS 0.189, PSNR 31.33 vs 30.41, SSIM 0.927 vs 0.920.
- **Training Convergence Behavior**: 3DCS is slower in the first 5K iterations (convex primitives are initialized as spheres and need time to deform), but rapidly catches up and eventually surpasses 3DGS after 5K iterations, indicating that the stronger representation capability of convex primitives yields more thorough fitting in the later stages.
- **Perspective-Aware Scaling**: Without scaling, PSNR drops by 5-6dB; $\sqrt{d}$ scaling is optimal (Truck 25.65, Train 22.23), followed by linear $d$ scaling; $d^2$ scaling over-scales, causing PSNR to collapse to 7-9dB.
- **Inconsistency between PSNR and Perceptual Quality**: In the Flower scene, 3DCS PSNR (20.17) is lower than 3DGS (21.65), but the visual quality is visually much closer to the GT. This is because PSNR is sensitive to pixel-level discrepancies and tends to reward blurry images.

### Detailed Results per Scene

| Scene | 3DCS LPIPS | 3DGS LPIPS | 3DCS PSNR | 3DGS PSNR |
|------|-----------|-----------|----------|----------|
| Truck | 0.125 | 0.148 | 25.65 | 25.18 |
| Train | 0.187 | 0.218 | 22.23 | 21.09 |
| DrJohnson | 0.238 | 0.244 | 29.54 | 28.76 |
| Playroom | 0.237 | 0.241 | 30.08 | 30.04 |
| Bonsai | 0.182 | 0.205 | 32.50 | 31.98 |
| Kitchen | 0.117 | 0.129 | 31.96 | 30.31 |

## Highlights & Insights
- Explaining the limitations of Gaussian primitives by analogy to the sphere-packing problem is elegant and intuitive.
- The dual-parameter control ($\delta$-$\sigma$) of convex primitives allows them to continuously transition from sharp polyhedra to diffuse Gaussians, making their representation capability a strict superset of Gaussians.
- The densification strategy of splitting into $K$ sub-convex primitives is more natural than 3DGS's clone/split, guaranteeing spatial coverage.
- The lightweight version of 3DCS (16-bit precision) achieves visual quality close to 3DGS using <15% memory.
- Figure 11 demonstrates that convex primitives can decompose a tree stump into physically meaningful convex parts—not just a rendering trick, but a basis for scene understanding.

## Limitations & Future Work
- Rendering speed (25-33 FPS) is lower than 3DGS (134 FPS). Although still real-time, the gap is significant, primarily due to the overhead of Graham Scan and convex hull line segment calculations. Each frame requires 2D convex hull computation for all visible convex primitives, with complexity $O(K\log K)$ times the number of primitives.
- Training time is 40-107% longer than 3DGS, with the primary overhead in differentiating the convex hull computation during backpropagation.
- SSIM/PSNR gains are marginal in outdoor natural scenes (areas without clear edges such as vegetation/sky).
- Convex primitives cannot represent concave structures directly and require combinations of multiple convex primitives to approximate them, which may increase the primitive count in complex geometries (e.g., mirror frames, arched structures).
- Has not been compared with subsequent 3DGS improvement methods (e.g., Mini-Splatting, Scaffold-GS, etc.), which have already significantly enhanced the performance of the original 3DGS.
- Initialization relies heavily on the quality of SfM point clouds. The initial sphere radius of the Fibonacci sphere algorithm is equal to $1.2 \times$ the average distance of the nearest 3 neighbors; sparse/low-quality point clouds will directly reduce the initial coverage of the convex primitives.
- $\rightarrow$ Future directions: hybrid convex-concave primitives, adaptive $K$ values (using different numbers of points in different regions), combining with compression techniques to further reduce memory.

## Related Work & Insights

| Method | Key Differences |
|----------|----------|
| 3DGS | Gaussians are a special case of convex primitives (small $\delta$ + small $\sigma$); 3DCS has stronger representation power + fewer primitives. |
| 2DGS | 2DGS collapses 3D Gaussians into 2D disks, equivalent to a degenerate convex primitive with $K=3$; 3DCS models full 3D volumes. |
| GES | GES uses generalized exponential functions to increase edge sharpness but remains symmetric; convex primitives can be arbitrarily asymmetric. |
| CvxNet | CvxNet defines convex shapes using hyperplanes + neural network optimization, which does not support real-time rendering; 3DCS uses point sets + a custom rasterizer. |

### Insights & Connections
- The flexibility of the primitive shape directly impacts the representation efficiency—this concept can be transferred to other primitive-based 3D representations.
- Convex primitives can serve as a semantic basis for scene decomposition (where each convex corresponds to a meaningful physical part), with potential applications in editable scene representations.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to introduce smooth convex primitives into real-time radiance field rendering, establishing a new paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on 3 standard datasets with extensive ablations and synthetic experiments, but lacks comparison with other recent GS variants.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear concepts, excellent illustrations (particularly the chair comparison in Fig. 2 and the $\delta$-$\sigma$ visualization in Fig. 4).
- **Value**: ⭐⭐⭐⭐ Has the potential to become a new standard primitive following 3DGS, though the drop in rendering speed must be addressed.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Sparse Voxels Rasterization: Real-time High-fidelity Radiance Field Rendering](sparse_voxels_rasterization_real-time_high-fidelity_radiance_field_rendering.md)
- [\[CVPR 2025\] Geometry Field Splatting with Gaussian Surfels](geometry_field_splatting_with_gaussian_surfels.md)
- [\[ICLR 2026\] Horseshoe Splatting: Handling Structural Sparsity for Uncertainty-Aware Gaussian-Splatting Radiance Field Rendering](../../ICLR2026/3d_vision/horseshoe_splatting_handling_structural_sparsity_for_uncertainty-aware_gaussian-.md)
- [\[CVPR 2025\] DeSplat: Decomposed Gaussian Splatting for Distractor-Free Rendering](desplat_decomposed_gaussian_splatting_for_distractor-free_rendering.md)
- [\[CVPR 2025\] DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting](dof-gaussian_controllable_depth-of-field_for_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
