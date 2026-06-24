---
title: >-
  [Paper Note] Implicit Filtering for Learning Neural Signed Distance Functions from 3D Point Clouds
description: >-
  [ECCV 2024][3D Vision][Implicit Fields] A non-linear implicit filter is proposed to smooth the implicit field of neural SDFs without requiring normals while preserving sharp geometric details, achieving field-wide consistency regularization through extension to non-zero level sets.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Implicit Fields"
  - "Signed Distance Function"
  - "Point Cloud Reconstruction"
  - "Bilateral Filtering"
  - "Feature Preservation"
date: 2026-05-08
content_hash: efb8636728226cc2
---

# Implicit Filtering for Learning Neural Signed Distance Functions from 3D Point Clouds

**Conference**: ECCV 2024  
**arXiv**: [2407.13342](https://arxiv.org/abs/2407.13342)  
**Code**: [https://list17.github.io/ImplicitFilter](https://list17.github.io/ImplicitFilter) (with project page)  
**Area**: 3D Vision  
**Keywords**: Implicit Fields, Signed Distance Function, Point Cloud Reconstruction, Bilateral Filtering, Feature Preservation

## TL;DR

A non-linear implicit filter is proposed to smooth the implicit field of neural SDFs without requiring normals while preserving sharp geometric details, achieving field-wide consistency regularization through extension to non-zero level sets.

## Background & Motivation

**Background**: Neural Signed Distance Functions (Neural SDFs) have become a mainstream method for reconstructing surfaces from 3D point clouds. They predict the signed distance value for each point in space by overfitting an MLP on a single point cloud, and then extract the zero level set using Marching Cubes.

**Limitations of Prior Work**: Existing methods (NeuralPull, GridPull, DIGS, etc.) only impose constraints (like eikonal constraints, gradient direction constraints) on individual query points, ignoring neighborhood geometric information. This results in reconstructed surfaces that contain noise and miss geometric details like sharp edges and corners.

**Key Challenge**: Discrete point clouds lack explicit signed distance supervision. The continuity of neural networks does not guarantee correct predictions at all locations, especially in areas with insufficient point density (such as sharp edges) where reliable geometric guidance is missing.

**Goal**: How to utilize neighborhood geometric information to smooth noise in the SDF implicit field while retaining high-precision geometric features (sharp edges/corners).

**Key Insight**: Drawing inspiration from bilateral filtering in image processing, a non-linear filter acting on the implicit field is designed. It simultaneously considers the spatial positions of points and SDF gradients (as a proxy for normals), and elegantly extends to non-zero level sets to regularize the entire distance field.

**Core Idea**: The implicit field is filtered by minimizing the weighted projection distance from query points to the gradient directions of neighborhood points, while projecting the input points along the gradient to non-zero level sets to achieve field-wide consistency.

## Method

### Overall Architecture

The method is based on an unsupervised neural SDF learning framework. Given an unoriented 3D point cloud $\boldsymbol{P}=\{\boldsymbol{p}_i\}_{i=1}^N$, an MLP $f_\theta: \mathbb{R}^3 \to \mathbb{R}$ is trained to predict signed distances. The training loss consists of four terms: level set bilateral filtering loss, field-wide filtering loss, distance loss, and Chamfer distance loss. The core novelty lies in the design of a bilateral implicit filter based on SDF gradients.

### Key Designs

1. **Level Set Bilateral Filtering**: Assuming all input points lie on the zero level set $\mathcal{S}_0$, for a point $\bar{\boldsymbol{p}}$ on the zero level set, instead of simply taking the weighted average of neighboring points (which would oversmooth sharp features), the **weighted projection distance** to the gradient directions of neighboring points is minimized. The bilateral projection filtering operator is formulated as:

$$d_{bi}(\bar{\boldsymbol{p}}) = \frac{\sum_{\boldsymbol{p}_j \in \mathcal{N}}(|\boldsymbol{n}_{p_j}^T(\bar{\boldsymbol{p}}-\boldsymbol{p}_j)| + |\boldsymbol{n}_{\bar{p}}^T(\bar{\boldsymbol{p}}-\boldsymbol{p}_j)|)\phi(\|\bar{\boldsymbol{p}}-\boldsymbol{p}_j\|)\psi(\boldsymbol{n}_{\bar{p}}, \boldsymbol{n}_{p_j})}{\sum_{\boldsymbol{p}_j \in \mathcal{N}}\phi(\|\bar{\boldsymbol{p}}-\boldsymbol{p}_j\|)\psi(\boldsymbol{n}_{\bar{p}}, \boldsymbol{n}_{p_j})}$$

where $\phi$ is the Gaussian weight based on spatial distance, and $\psi$ is the Gaussian weight based on normal similarity: $\psi(\boldsymbol{n}_{\bar{p}}, \boldsymbol{n}_{p_j}) = \exp\left(-\frac{1-\boldsymbol{n}_{\bar{p}}^T\boldsymbol{n}_{p_j}}{1-\cos(\sigma_n)}\right)$, with normals obtained by normalizing SDF gradients $\boldsymbol{n} = \nabla f_\theta / \|\nabla f_\theta\|$.

Design Motivation: Unlike simple mean filtering, projecting onto the gradient direction preserves sharp features—at sharp edges, the large difference in normals causes the weight $\psi$ of neighboring points across the edge to approach zero.

2. **Sampling via NeuralPull**: Directly sampling on the zero level set is difficult. Therefore, borrowing the idea of NeuralPull, query points $\boldsymbol{q}$ are randomly sampled near the surface and then pulled to the zero level set along the gradient direction:

$$\hat{\boldsymbol{q}} = \boldsymbol{q} - f_\theta(\boldsymbol{q})\frac{\nabla f_\theta(\boldsymbol{q})}{\|\nabla f_\theta(\boldsymbol{q})\|}$$

Level set filtering loss: $L_{zero} = \sum_{\hat{\boldsymbol{q}} \in \hat{\boldsymbol{Q}}} d_{bi}(\hat{\boldsymbol{q}})$. Efficiency optimization: Since $\hat{\boldsymbol{q}}$ changes dynamically during training and requires repeated neighborhood searches, the neighborhood of the nearest neighbor $NN(\boldsymbol{q})$ of the original query point $\boldsymbol{q}$ is used as an approximation.

3. **Extension to Non-Zero Level Sets**: The filtering is extended from the zero level set to the entire SDF field. For a query point $\boldsymbol{q}$ lying on the level set $\mathcal{S}_{f_\theta(\boldsymbol{q})}$, the neighborhood is constructed by projecting the input points backward along the gradient onto that level set:

$$\mathcal{N}(\boldsymbol{q}, \mathcal{S}_{f_\theta(q)}) = \left\{\hat{\boldsymbol{p}} \mid \hat{\boldsymbol{p}} = \boldsymbol{p} + f_\theta(\boldsymbol{q})\frac{\nabla f_\theta(\boldsymbol{p})}{\|\nabla f_\theta(\boldsymbol{p})\|}, \boldsymbol{p} \in \mathcal{N}(\hat{\boldsymbol{q}}, \mathcal{S}_0)\right\}$$

Field-wide filtering loss: $L_{field} = \sum_{\boldsymbol{q} \in \boldsymbol{Q}} d_{bi}(\boldsymbol{q})$.

Design Motivation: Filtering only on the zero level set is insufficient, as inconsistencies may exist between different level sets. Filtering across level sets improves the consistency and regularity of the entire SDF field.

4. **Gradient Constraint**: Implicit filtering may degenerate into a trivial solution with zero gradients. Chamfer distance loss $L_{CD}$ is used as a gradient constraint (which is more relaxed and effective than the eikonal term), constraining the SDF values and gradients by calculating the bidirectional nearest distance between pulled points and the original point cloud.

### Loss & Training

Total loss:

$$L = L_{zero} + \alpha_1 L_{field} + \alpha_2 L_{dist} + \alpha_3 L_{CD}$$

where $L_{dist} = \frac{1}{N}\sum|f_\theta(\boldsymbol{p}_i)|$ constrains the input points to lie on the zero level set, with $\alpha_1=\alpha_2=1, \alpha_3=10$. The filtering parameters are set to $\sigma_n=15°$, and $\sigma_p$ is set to the maximum distance in the neighborhood. The network uses the OccNet architecture with SAL geometric initialization.

## Key Experimental Results

### Main Results

**ABC & FAMOUS Datasets** (F-Score threshold=0.01):

| Method | ABC $CD_{L2}$ | ABC $CD_{L1}$ | ABC F-S. | FAMOUS $CD_{L2}$ | FAMOUS $CD_{L1}$ | FAMOUS F-S. |
|------|----------|----------|------|------------|------------|-------|
| NeuralPull | 0.095 | 0.011 | 0.673 | 0.100 | 0.012 | 0.746 |
| SIREN | 0.022 | 0.012 | 0.493 | 0.025 | 0.012 | 0.561 |
| DIGS | 0.021 | 0.010 | 0.667 | 0.015 | 0.008 | 0.772 |
| **Ours** | **0.011** | **0.009** | **0.691** | **0.008** | **0.007** | **0.778** |

**ShapeNet Dataset** (3000+ objects):

| Method | $CD_{L2}\times100$ | NC | F-Score(0.002) | F-Score(0.004) |
|------|------------|------|------------|------------|
| GridPull | 0.0086 | 0.9723 | 0.9896 | 0.9923 |
| **Ours** | **0.0032** | **0.9779** | **0.9976** | **0.9985** |

**Edge Chamfer Distance** (ABC Dataset, $ECD_{L2}\times100$):

| Method | P2S | NeuralPull | DIGS | **Ours** |
|------|-----|-----------|------|----------|
| $ECD_{L1}$ | 0.0496 | 0.0501 | 0.0786 | **0.0256** |
| $ECD_{L2}$ | 1.055 | 1.255 | 2.493 | **0.399** |

### Ablation Study

**Loss Function Combinations** (FAMOUS Dataset):

| Loss Combination | $CD_{L1}$ | $CD_{L2}$ | F-S. | NC |
|---------|----------|----------|------|------|
| $L_{pull}$ only | 0.012 | 0.083 | 0.742 | 0.884 |
| $L_{CD}$ only | 0.010 | 0.031 | 0.757 | 0.891 |
| $L_{CD}+L_{zero}$ | 0.008 | 0.018 | 0.772 | 0.905 |
| $L_{CD}+L_{zero}+L_{field}$ | 0.008 | 0.011 | 0.769 | 0.908 |
| **Full (Ours)** | **0.007** | **0.008** | **0.778** | **0.911** |

### Key Findings

- Zero level set filtering $L_{zero}$ yields the most significant improvement ($CD_{L2}$ drops from 0.031 to 0.018), effectively denoising while retaining geometric features.
- Non-zero level set extension $L_{field}$ further improves field-wide consistency.
- The Chamfer distance constraint is more suitable for this method than the eikonal constraint (performance is slightly better without eikonal).
- Bidirectional projection ($d_{bi}$) demonstrates a significant improvement over unidirectional projection ($d$): F-S. increases from 0.726 to 0.778.
- On the Edge Chamfer Distance metric, Ours shows the most pronounced advantage (only 16% of DIGS), validating the effectiveness of feature preservation.

## Highlights & Insights

- **Bilateral Filtering on Implicit Fields**: Extending classic bilateral filtering from image processing to SDF implicit fields, utilizing the gradient as a proxy for the normal to avoid relying on ground-truth normals.
- **Non-Zero Level Set Regularization**: Smoothing the entire field is achieved by projecting points along the gradient to any level set, an elegant idea that effectively aids SDF consistency.
- **Feature Preservation Mechanism**: The normal similarity weight $\psi$ naturally assigns low weights to neighboring points across sharp edges, inherently preserving sharp features.
- The method is compatible with existing SDF learning frameworks (NeuralPull, GridPull, etc.), acting as a plug-and-play improvement.

## Limitations & Future Work

- Dense sampling near the surface is required, which incurs high computational overhead for searching neighborhoods and calculating gradients.
- It assumes all input points lie on the surface, which might not be robust to noisy point clouds containing outliers.
- Although the filtering parameters $\sigma_n, \sigma_p$ are robust within a certain range, they may require adaptive adjustments for extreme geometries (e.g., extremely thin structures).
- Scene-level reconstruction extracts the 0.001 level set instead of the zero level set (open scenes are not closed), a limitation that is worth exploring further.

## Related Work & Insights

- Unlike DIGS (using divergence-guided surface smoothing) and EPI (smoothing implicit surface roughness), this work is the first to optimize the implicit field via **local geometric filtering**.
- Insight: The filtering concept can be extended to other implicit representations (such as UDF, density fields of NeRF).
- A modern continuation of point cloud filtering traditions like LOP (Locally Optimal Projection operator) in neural implicit representations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Generalizing bilateral filtering to SDF implicit fields and extending it to non-zero level sets represents a solid theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive ablation and testing across five datasets (ShapeNet, ABC, FAMOUS, SRB, 3D Scene), including targeted metrics like Edge Chamfer Distance.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear mathematical derivations, and schematic diagrams intuitively explain why projection filtering outperforms mean filtering.
- **Value**: ⭐⭐⭐⭐ — Serves as a plug-and-play module that can be widely integrated into existing SDF learning frameworks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Heterogeneous Graph Learning for Scene Graph Prediction in 3D Point Clouds](heterogeneous_graph_learning_for_scene_graph_prediction_in_3d_point_clouds.md)
- [\[ECCV 2024\] Ray-Distance Volume Rendering for Neural Scene Reconstruction](ray-distance_volume_rendering_for_neural_scene_reconstruction.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] SEED: A Simple and Effective 3D DETR in Point Clouds](seed_a_simple_and_effective_3d_detr_in_point_clouds.md)
- [\[ECCV 2024\] A Probability-guided Sampler for Neural Implicit Surface Rendering](a_probabilityguided_sampler_for_neural_implicit_surface_rend.md)

</div>

<!-- RELATED:END -->
