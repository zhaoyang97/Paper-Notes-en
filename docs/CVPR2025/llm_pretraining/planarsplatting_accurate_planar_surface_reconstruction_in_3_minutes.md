---
title: >-
  [Paper Note] PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes
description: >-
  [CVPR 2025][LLM Pretraining][Planar Reconstruction] This paper proposes PlanarSplatting, which directly optimizes learnable 3D rectangular plane primitives. By utilizing a newly designed rectangular splatting function, planes are differentiably rendered into depth and normal maps. This enables the reconstruction of accurate indoor planar scenes from multi-view images in just 3 minutes without requiring any plane annotations.
tags:
  - "CVPR 2025"
  - "LLM Pretraining"
  - "Planar Reconstruction"
  - "Differentiable Rendering"
  - "3D Plane Primitives"
  - "Indoor Scenes"
  - "Gaussian Splatting"
date: 2026-05-08
content_hash: defa7d6fbd1c0274
---

# PlanarSplatting: Accurate Planar Surface Reconstruction in 3 Minutes

**Conference**: CVPR 2025  
**arXiv**: [2412.03451](https://arxiv.org/abs/2412.03451)  
**Code**: None (CUDA implementation will be released)  
**Area**: Video Understanding  
**Keywords**: Planar Reconstruction, Differentiable Rendering, 3D Plane Primitives, Indoor Scenes, Gaussian Splatting

## TL;DR

This paper proposes PlanarSplatting, which directly optimizes learnable 3D rectangular plane primitives. By utilizing a newly designed rectangular splatting function, planes are differentiably rendered into depth and normal maps. This enables the reconstruction of accurate indoor planar scenes from multi-view images in just 3 minutes without requiring any plane annotations.

## Background & Motivation

**Background**: Indoor planar 3D reconstruction is a classic problem in computer vision. Traditional methods fit planes from existing point clouds or meshes, while learning-based methods such as PlanarRecon and AirPlanes require training pipelines for detection, matching, and tracking on large amounts of 2D/3D plane annotations.

**Limitations of Prior Work**: (1) Methods like PlanarRecon treat planes as visual features, but planes are regional rather than local, resulting in coarse outputs that lose details; (2) existing learning-based methods rely on 2D/3D plane annotations, but large-scale plane annotations are difficult to acquire, limiting performance and scalability; (3) traditional two-stage methods (first reconstructing meshes, then fitting planes using RANSAC) are slow and heavily depend on the quality of intermediate representations.

**Key Challenge**: To obtain accurate and complete plane reconstructions, traditional methods require high-quality 3D geometry (point clouds/meshes) beforehand, while learning-based methods require abundant plane annotations. Neither approach is sufficiently efficient.

**Goal**: To design a method that directly optimizes 3D plane primitives from multi-view images without requiring plane annotations, balancing both speed and accuracy.

**Key Insight**: Leveraging the differentiable rendering concept of 3DGS, which renders 3D primitives to 2D via differentiable splatting and optimizes them using gradient descent. However, the circular/elliptical shape of 3D Gaussians is poorly suited for rectangular planes.

**Core Idea**: To design rectangular 3D plane primitives (characterized by center, rotation, and bidirectional radii) and a corresponding rectangular splatting function (based on Sigmoid rather than Gaussian). The method leverages molecular depth and normal foundation models as pseudo-supervision to directly optimize plane primitives for fitting scene geometry.

## Method

### Overall Architecture

The input consists of multi-view posed images. About 2000 3D rectangular plane primitives are initialized using a monocular depth model (Metric3Dv2). Through differentiable plane rendering, these primitives are splatted into depth and normal maps, with the depth predicted by Metric3Dv2 and the normals predicted by Omnidata serving as pseudo-label supervision for optimization. After 5000 iterations, similar planes are merged to obtain the final plane instances.

### Key Designs

1. **Learnable Rectangular Plane Primitive Representation**:

    - **Function**: Represents 3D rectangular planes with explicit parameters, supporting gradient-based optimization of position, orientation, and shape.
    - **Mechanism**: Each plane primitive $\pi$ is defined by three sets of parameters: center $\mathbf{p}_\pi \in \mathbb{R}^3$, rotation quaternion $\mathbf{q}_\pi \in \mathbb{R}^4$, and bidirectional radii $\mathbf{r}_\pi = \{r^{x+}, r^{x-}, r^{y+}, r^{y-}\} \in \mathbb{R}_+^4$. The bidirectional radii design allows asymmetric rectangular centers, providing more flexible shape-fitting capabilities. The normal is automatically determined by rotation as $\mathbf{n}_\pi = \mathbf{R}(\mathbf{q}_\pi)[0,0,1]^\top$.
    - **Design Motivation**: 3DGS Gaussian primitives are circular or elliptical with blurry boundaries, which are unsuitable for representing rectangular walls, floors, and other flat surfaces. Bidirectional radii enable a single primitive to represent asymmetric shapes such as L-shapes, thereby reducing the number of required primitives.

2. **Rectangular Plane Splatting Function**:

    - **Function**: Differentiably projects 3D rectangular plane primitives into pixel space to calculate precise rectangular boundary weights.
    - **Mechanism**: First, the intersection $\mathbf{x}_\pi^\mathbf{r}$ between the ray and the plane is computed, and then the intersection is projected onto the plane's local coordinate system to obtain $\mathcal{P}_X, \mathcal{P}_Y$. The weights are calculated using the Sigmoid function: $w_X = 2\sigma(5\lambda(r^{x+} - |\mathcal{P}_X|))$ (when $\mathcal{P}_X > 0$), where $\lambda$ grows exponentially with the iterations (up to 300). This allows the weights to gradually transition from flat gradients to approximating the hard boundary of the rectangle. The final weight is taken as $w = \min(w_X, w_Y)$.
    - **Design Motivation**: If Gaussian splatting functions are used, the boundaries of the planes will be blurry (as shown in Fig. 4/5), causing a performance drop at the boundaries of adjacent planes. The Sigmoid function can precisely approximate rectangular boundaries as $\lambda$ increases, and the curriculum strategy of gradually increasing $\lambda$ guarantees smooth gradients in the early stages of optimization and sharp boundaries later.

3. **Plane Splitting and Merging Strategy**:

    - **Function**: Adaptively adjusts the number of planes to better fit the scene.
    - **Mechanism**: **Splitting**: Every 1000 iterations, the gradients of the plane radii are checked. If the average gradient in the X direction is $> 0.2$, the plane is split along the Y-axis (and vice versa), splitting one large plane into two smaller ones. **Merging**: After optimization is complete, adjacent planes with normal angle error $< 25^\circ$ and offset distance error to the scene center $< 0.1\text{cm}$ are merged into the same plane instance.
    - **Design Motivation**: Initializing only about 2000 plane primitives may be insufficient. The splitting operation can add primitives in regions that require more detail. The merging operation aggregates multiple primitives belonging to the same physical plane into a single plane instance, yielding a compact final representation.

### Loss & Training

The rendering loss $\mathcal{L}_{\text{render}}$ consists of three components: normal cosine loss ($\alpha_1=5.0$), normal L1 loss ($\alpha_1=5.0$), and depth L1 loss ($\alpha_2=1.0$). Optimization is performed using the Adam optimizer for 5000 iterations. Depth pseudo-labels are obtained from Metric3Dv2, and normal pseudo-labels are from Omnidata. A CUDA implementation enables the entire optimization to finish within 3 minutes.

## Key Experimental Results

### Main Results

Planar reconstruction quality on the ScanNetV2 dataset (100 scenes):

| Method | Requires Annotation | Chamfer↓ | F-score↑ | SC↑ | Planar Chamfer↓ |
|---|---|---|---|---|---|
| PlanarRecon | ✓ | 9.89 | 43.47 | 0.405 | 17.53 |
| AirPlanes | ✓ | 5.30 | 64.92 | 0.568 | 8.37 |
| 2DGS+RANSAC | ✗ | 14.15 | 31.33 | 0.257 | 27.40 |
| SR+RANSAC | ✗ | 5.40 | 65.45 | 0.515 | 9.78 |
| **PlanarSplatting** | **✗** | **4.83** | **68.85** | **0.532** | **9.20** |

ScanNet++ dataset (30 scenes):

| Method | Chamfer↓ | F-score↑ | Planar Chamfer↓ |
|---|---|---|---|
| PlanarRecon | 17.85 | 31.10 | 26.90 |
| AirPlanes | 13.75 | 32.58 | 20.37 |
| **PlanarSplatting** | **9.33** | **47.04** | **14.75** |

### Ablation Study

Splatting function comparison:

| Configuration | Chamfer↓ | F-score↑ | Description |
|---|---|---|---|
| Gaussian Splatting | Poorer | Poorer | Blurry boundaries lead to imprecise plane fitting |
| Rectangular Plane Splatting | **4.83** | **68.85** | Sharp boundaries accurately fit rectangular surfaces |

### Key Findings

- PlanarSplatting surpasses PlanarRecon, which requires annotations, under unannotated conditions (Chamfer 4.83 vs 9.89), and even approaches AirPlanes in geometric metrics.
- It outperforms all methods by a large margin on ScanNet++ (F-score 47.04 vs. the previous best of 35.93).
- The rectangular splatting function shows a clear advantage over Gaussian splatting, validating the importance of shape priors.
- PlanarSplatting can serve as an initializer for 3DGS/2DGS—initializing Gaussians with its reconstruction results significantly improves rendering quality and dramatically shortens training time.
- It completes optimization in only 3 minutes, which is faster than both the inference speed of learning-based methods and the reconstruction speed of two-stage methods.

## Highlights & Insights

1. **"Shape Prior is Efficiency"**: Replacing Gaussian primitives with rectangular primitives introduces planar priors of indoor scenes from the representation level, which significantly reduces the number of required primitives and the optimization time.
2. **Design of Sigmoid Progressive Approximation to Rectangles**: The strategy of exponential growth of $\lambda$ cleverly balances optimization stability and boundary sharpness. This "soft-to-hard" curriculum strategy can be transferred to other scenarios where learning hard boundaries is required.
3. **Synergy with GS Methods**: Utilizing the method as a geometric initializer for 3DGS demonstrates the complementary value of explicit geometric reconstruction and neural rendering.

## Limitations & Future Work

- It only handles planar scenes and cannot represent curved objects (such as chairs and furniture).
- It relies on the quality of monocular depth/normal foundation models; errors from these foundation models propagate to the plane reconstruction.
- Plane merging uses simple threshold rules, which may lead to over-merging or under-merging in complex topological relationships.
- Future work could extend rectangular primitives to circular arc or NURBS primitives to handle curved surfaces.

## Related Work & Insights

- **vs PlanarRecon**: PlanarRecon learns end-to-end but requires annotations and a massive amount of training data; PlanarSplatting is training-free, annotation-free, and yields better results.
- **vs AirPlanes**: AirPlanes first reconstructs meshes and then extracts planes, which is a two-stage method; PlanarSplatting directly optimizes plane primitives, making it more efficient.
- **vs 3DGS/2DGS**: Gaussian primitives lack shape priors; PlanarSplatting's rectangular primitives are naturally suited for indoor planes and can serve as GS initialization to accelerate subsequent rendering optimization.
- **Insight**: Encoding structural priors of the scene (e.g., planes, symmetries) into the primitive design can achieve better reconstructions with fewer parameters.

## Rating

- **Novelty**: 8/10 — The design of rectangular plane primitives combined with Sigmoid splatting is novel, extending the GS concept to structured reconstruction.
- **Experimental Thoroughness**: 8/10 — Hundreds of scenes across two major datasets, offering comprehensive comparisons, though a detailed analysis of time efficiency is lacking.
- **Writing Quality**: 8/10 — Clear method descriptions, intuitive diagrams, and complete mathematical derivations.
- **Value**: 8/10 — High-quality plane reconstruction in 3 minutes holds significant value in practical applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Law of Data Reconstruction for Random Features (and Beyond)](../../ICLR2026/llm_pretraining/a_law_of_data_reconstruction_for_random_features_and_beyond.md)
- [\[CVPR 2025\] Improving Autoregressive Visual Generation with Cluster-Oriented Token Prediction](improving_autoregressive_visual_generation_with_cluster-oriented_token_predictio.md)
- [\[CVPR 2025\] ScaMo: Exploring the Scaling Law in Autoregressive Motion Generation Model](scamo_exploring_the_scaling_law_in_autoregressive_motion_generation_model.md)
- [\[CVPR 2025\] Bridging the Vision-Brain Gap with an Uncertainty-Aware Blur Prior](bridging_the_vision-brain_gap_with_an_uncertainty-aware_blur_prior.md)
- [\[CVPR 2025\] Robust Message Embedding via Attention Flow-Based Steganography](robust_message_embedding_via_attention_flow-based_steganography.md)

</div>

<!-- RELATED:END -->
