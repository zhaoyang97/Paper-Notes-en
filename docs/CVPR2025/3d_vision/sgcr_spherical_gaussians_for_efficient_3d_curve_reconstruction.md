---
title: >-
  [Paper Note] SGCR: Spherical Gaussians for Efficient 3D Curve Reconstruction
description: >-
  [CVPR 2025][3D Vision][3D Gaussians] SGCR proposes Spherical Gaussians, a concise 3D representation that simplifies the anisotropic ellipsoids of standard 3D Gaussians into uniform-sized spheres. Using only 2D edge maps for supervision, it faithfully aligns them with 3D object edges. It then reconstructs precise 3D parametric curves efficiently via a novel rational Bézier curve extraction algorithm, achieving 50 times faster speed and better accuracy than NEF and EMAP.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "3D Gaussians"
  - "Spherical Gaussians"
  - "3D Curve Reconstruction"
  - "Multi-view"
  - "Edge Detection"
date: 2026-05-08
content_hash: 793cfd8756744c14
---

# SGCR: Spherical Gaussians for Efficient 3D Curve Reconstruction

**Conference**: CVPR 2025  
**arXiv**: [2505.04668](https://arxiv.org/abs/2505.04668)  
**Code**: [https://github.com/Martinyxr/SGCR](https://github.com/Martinyxr/SGCR)  
**Area**: 3D Vision  
**Keywords**: 3D Gaussians, Spherical Gaussians, 3D Curve Reconstruction, Multi-view, Edge Detection

## TL;DR

SGCR proposes Spherical Gaussians, a concise 3D representation that simplifies the anisotropic ellipsoids of standard 3D Gaussians into uniform-sized spheres. Using only 2D edge maps for supervision, it faithfully aligns them with 3D object edges. It then reconstructs precise 3D parametric curves efficiently via a novel rational Bézier curve extraction algorithm, achieving 50 times faster speed and better accuracy than NEF and EMAP.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) achieves high-quality novel view synthesis and real-time rendering through anisotropic 3D Gaussian primitives. However, 3DGS properties (rotation, scaling, spherical harmonic colors) are primarily designed and optimized for rendering diverse 2D images, performing poorly in defining precise 3D geometric structures.

**Limitations of Prior Work**: Feature curves are key geometric clues for characterizing 3D shape structures. Traditional methods detect edges based on point clouds but rely on high-quality 3D scanning inputs. Learning-based methods require expensive 3D edge annotations as supervision. Recent multi-view methods (NEF, EMAP) can reconstruct 3D edges from 2D images but require hours of training time (~1 hour for NEF, ~2 hours for EMAP), resulting in extremely low efficiency.

**Key Challenge**: 2D edge detection is mature and easily obtainable from images, but how to efficiently merge multi-view 2D edge information into a complete 3D structure—specifically establishing edge correspondence across views—remains the key challenge.

**Goal**: Design an efficient intermediate representation that can serve as a bridge from 2D edge information to 3D parametric curves.

**Key Insight**: The explicit primitive representation of 3DGS is naturally suited for "bridging" 2D rendering and 3D geometry. By explicitly constraining Gaussian primitives to configurations suitable for representing edges (instead of rendering), each Gaussian primitive can be given a clear geometric meaning.

**Core Idea**: Simplify the anisotropic ellipsoidal Gaussians of 3DGS into equal-sized spherical Gaussians (removing the covariance matrix and simplifying colors to grayscale values). Combined with a specialized training strategy, this forces the spherical Gaussians to faithfully align with 3D edges, allowing rational Bézier curves to be extracted directly from the spherical Gaussians via an optimization algorithm.

## Method

### Overall Architecture

SGCR consists of two stages: (1) **Spherical Gaussian Generation**: starting from mesh initialization, spherical Gaussians are trained using 2D edge map supervision to distribute them along 3D edges; (2) **Parametric Curve Extraction**: first securing a rough structure through RANSAC-style segment fitting, followed by global optimization to upgrade the segments into rational Bézier curves.

### Key Designs

1. **Spherical Gaussian Representation**:

    - Function: Provide a concise explicit primitive suitable for representing 3D edges.
    - Mechanism: Make two key modifications to standard 3DGS: (a) Remove the covariance matrix (scaling $S$ and rotation $R$) and replace it with a fixed radius $r_0=0.005$, turning ellipsoids into uniform-sized spheres; (b) Remove spherical harmonic colors, simplifying them to 1D grayscale values (only used for edge map rendering). These modifications do not affect gradient backpropagation but impose strong regularization on the geometric distribution: the spherical structure ensures each Gaussian primitive corresponds to an "atomic" geometric point with a clear 3D location meaning.
    - Design Motivation: Although "thin and long" ellipsoids in anisotropic Gaussians seem more suitable for representing edges, they cannot be decomposed into meaningful geometric units—a large ellipsoid might span multiple edge segments. While a single spherical Gaussian covers a smaller range, they can be densely arranged along edges like "3D pixels", where the center of each sphere is an edge sample point, providing a natural point cloud representation for subsequent curve extraction.

2. **Edge-Aware Training Scheme**:

    - Function: Optimize the spatial distribution of spherical Gaussians under 2D edge map supervision.
    - Mechanism: Includes three loss functions and a two-stage training strategy. **Edge Loss** $\mathcal{L}_{edge}$: Since edge pixels are extremely sparse in images, directly using L1 loss would cause all Gaussians to converge to zero. Therefore, a weighted scheme is used to balance the contribution of edge/non-edge pixels: $\mathcal{L}_{edge} = \frac{N_I - |E_I|}{N_I}\sum_{i\in E_I}\|I_i - \hat{I}_i\|^2 + \frac{|E_I|}{N_I}\sum_{i\notin E_I}\|I_i - \hat{I}_i\|^2$. **Opacity-Color Consistency Loss** $\mathcal{L}_{oc}$: Constrains the opacity and color values of each Gaussian to be consistent, addressing multi-view inconsistency caused by occlusions and preventing occluded edge Gaussians from being pruned prematurely. **Regularization Loss** $\mathcal{L}_{reg}$: Penalizes opacity with a log term to control the total number of Gaussians and accelerate convergence.
    - Design Motivation: The specificity of edge maps (extreme sparsity, multi-view inconsistency) makes direct application of standard 3DGS losses completely unfeasible. The three specialized loss functions fulfill distinct roles: edge loss handles positioning, consistency loss deals with occlusions, and regularization loss controls the scale. In the two-stage training, the first stage performs densification via splitting/cloning + periodic opacity resetting, followed by large-scale pruning (retaining $o_i>0.5$ and $c_i>0.1$); the second stage only refines positions and attributes. The entire training takes only about 1 minute.

3. **SGCR Curve Extraction Algorithm**:

    - Function: Extract continuous 3D parametric curves from discrete spherical Gaussians.
    - Mechanism: Divided into segment fitting and global optimization steps. **Segment Fitting**: Uses a RANSAC-style iterative approach—randomly selecting two adjacent Gaussian centers as segment endpoints in each round, interpolating $N_s$ points along the segment, inflating them with Gaussian noise to simulate spherical Gaussian shapes, evaluating fitting quality using Chamfer distance, recording the best fit after optimizing endpoint locations, and then removing the fitted Gaussians to enter the next round. **Global Optimization**: Restores all Gaussians, inserts two control points between each pair of segment endpoints to initialize 3rd-order rational Bézier curves $B(u) = \frac{\sum B_{3,i}(u) p_i w_i}{\sum B_{3,i}(u) w_i}$, introduces opacity-weighted Chamfer distance $\mathcal{L}_{WCD}$ and endpoint connection loss $\mathcal{L}_{endpoints}$, and optimizes control point coordinates and weights for all curves simultaneously.
    - Design Motivation: Rational Bézier curves are chosen over simple Bézier curves because the latter cannot perfectly fit circular arcs. The progressive strategy of coarse segment fitting followed by curve upgrading is more stable than direct curve fitting.

### Loss & Training

Total training loss: $\mathcal{L} = (1-\lambda_1)\mathcal{L}_{edge} + \lambda_1\mathcal{L}_{D-SSIM} + \lambda_2\mathcal{L}_{oc} + \lambda_3\mathcal{L}_{reg}$, where $\lambda_1=0.2$, $\lambda_2=2$, $\lambda_3=0.01$.

## Key Experimental Results

### Main Results

| Method | CD↓ | Precision↑ | Recall↑ | F-score↑ | IoU↑ | Input | Training Time | Reconstruction Time |
|------|-----|-----------|---------|----------|------|------|---------|---------|
| RFEPS | 0.032 | 0.896 | 0.856 | 0.867 | 0.819 | Point Cloud | — | 48s |
| NerVE | 0.039 | 0.952 | 0.731 | 0.827 | 0.685 | Point Cloud | 10h | 1s |
| NEF | 0.035 | 0.939 | 0.884 | 0.904 | 0.828 | Image | 1h | 119s |
| EMAP | 0.029 | 0.951 | 0.893 | 0.921 | 0.847 | Image | 2h | 40s |
| **Ours**| **0.028** | **0.955** | **0.905** | **0.926** | **0.862** | Image | **87s** | **32s** |

Results on the ABC-NEF dataset. SGCR outperforms the state-of-the-art on all metrics, with a training speed more than 50 times faster than NEF/EMAP.

### Ablation Study

| Configuration | CD↓ | F-score↑ | IoU↑ |
|------|-----|----------|------|
| w/o Spherical Gaussians (using original 3DGS) | 0.142 | 0.330 | 0.194 |
| w/o Mesh Initialization | 0.032 | 0.901 | 0.822 |
| w/o Edge Loss | 0.052 | 0.853 | 0.714 |
| w/o Opacity-Color Loss | 0.051 | 0.859 | 0.723 |
| w/o Regularization Loss | 0.029 | 0.913 | 0.846 |
| w/o Two-Stage Training | 0.030 | 0.920 | 0.860 |
| w/o Segment Fitting | 0.030 | 0.911 | 0.822 |
| w/o Global Optimization | 0.032 | 0.876 | 0.757 |
| **Full Model** | **0.028** | **0.926** | **0.862** |

### Key Findings

- **Spherical Gaussians are the most critical design**: Removing the spherical constraint plummets the IoU from 0.862 to 0.194, indicating that standard 3DGS is completely incapable of representing meaningful 3D edge structures.
- **Edge loss and opacity-color loss contribute the most** (removing either drops the IoU by over 14%), validating the necessity of designing specialized losses for edge sparsity and multi-view occlusion.
- **Radius selection**: $r_0=0.005$ is optimal (2,361 Gaussians); too small (0.002) yields noise, too large (0.01) leads to underfitting, and variable radius (0.040) performs the worst.
- Only 10 views are needed to obtain satisfying results for simple objects, while complex objects require 30-50 views.
- Performs equally well on real-world DTU and Replica scenes, where NEF fails in complex scenes and EMAP produces chaotic segments, while SGCR remains robust.

## Highlights & Insights

- **The concept of "inverse 3DGS"** is highly ingenious: while standard 3DGS stitches a complete scene using anisotropic Gaussians, SGCR reverses this by decomposing the scene into spherical "atoms" to capture geometric structures. This inverse "rendering $\rightarrow$ geometry" thinking opens up new application directions for Gaussian primitives.
- **Relying only on 2D supervision** is the greatest practical advantage: it requires no 3D annotations or pre-training, achieving 3D curve reconstruction solely via a mature 2D edge detector (PiDiNet) and calibrated multi-view images.
- **An 87-second training time** is an order-of-magnitude leap compared to the hours required by NEF/EMAP, enabling batch processing of large numbers of objects.
- The concept of spherical Gaussians can be transferred to other 3D geometric feature extractions, such as corner detection and plane segmentation. Essentially, it uses constrained Gaussian primitives to "probe" specific types of geometric structures.

## Limitations & Future Work

- The fixed-radius assumption limits adaptation to edges of different scales—fine edges require small radii, while coarse edges require large ones.
- High dependence on the quality of the 2D edge detector; missing edges in certain views will affect the reconstruction integrity.
- The RANSAC process in segment fitting is serial, and speed may degrade when the number of Gaussians is very large.
- Future directions: adaptive radius learning, incorporating semantic information to distinguish different types of edges (creases vs. silhouettes), and extending to 4D edge reconstruction in dynamic scenes.

## Related Work & Insights

- **vs. NEF**: NEF learns edge density distributions via neural implicit fields, which is highly expressive but slow to train. SGCR directly represents edges with explicit spherical Gaussians, training 50 times faster with higher accuracy.
- **vs. EMAP**: EMAP encodes 3D edge distance and orientation in a UDF, which is also time-consuming to train. SGCR's explicit representation makes subsequent curve extraction more straightforward.
- **vs. 3DGS**: Standard 3DGS is optimized for rendering, and its Gaussian distribution has no geometric meaning. SGCR endows each Gaussian with explicit edge semantics through spherical constraints, which is a geometry-driven variant of 3DGS.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The spherical Gaussian concept is simple and elegant; the design philosophy of "limiting degrees of freedom to gain geometric meaning" is thought-provoking.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic and real datasets, complete ablations, though ModelNet is relatively small with only 120 objects.
- Writing Quality: ⭐⭐⭐⭐ Clear method description, good visualization, and complete pseudo-code.
- Value: ⭐⭐⭐⭐ Achieves double breakthroughs in both accuracy and efficiency for the specific task of 3D edge reconstruction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MEGS2: Memory-Efficient Gaussian Splatting via Spherical Gaussians and Unified Pruning](../../ICLR2026/3d_vision/megs2_memory-efficient_gaussian_splatting_via_spherical_gaussians_and_unified_pr.md)
- [\[ICCV 2025\] Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction](../../ICCV2025/3d_vision/curve-aware_gaussian_splatting_for_3d_parametric_curve_reconstruction.md)
- [\[CVPR 2025\] RigGS: Rigging of 3D Gaussians for Modeling Articulated Objects in Videos](riggs_rigging_of_3d_gaussians_for_modeling_articulated_objects_in_videos.md)
- [\[CVPR 2025\] SphereUFormer: A U-Shaped Transformer for Spherical 360 Perception](sphereuformer_a_u-shaped_transformer_for_spherical_360_perception.md)
- [\[CVPR 2025\] iSegMan: Interactive Segment-and-Manipulate 3D Gaussians](isegman_interactive_segment-and-manipulate_3d_gaussians.md)

</div>

<!-- RELATED:END -->
