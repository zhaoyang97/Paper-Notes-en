---
title: >-
  [Paper Note] WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction
description: >-
  [Self-Supervised Learning] > WIR3D optimizes a set of 3D Bézier curve parameters under the spatial guidance of CLIP intermediate-layer activations to faithfully represent the geometric structure and visually salient features (including texture) of 3D shapes from arbitrary viewpoints, achieving sparse yet semantically rich 3D shape abstraction.
tags:
  - Self-Supervised Learning
date: 2026-05-08
content_hash: b972b5f414921e7a
---

# WIR3D: Visually-Informed and Geometry-Aware 3D Shape Abstraction

- **Conference**: ICCV 2025
- **arXiv**: 2505.04813
- **Code**: Coming Soon
- **Area**: Self-Supervised
- **Keywords**: 3D shape abstraction, Bézier curves, CLIP guidance, texture abstraction, shape deformation

## TL;DR

> WIR3D optimizes a set of 3D Bézier curve parameters under the spatial guidance of CLIP intermediate-layer activations to faithfully represent the geometric structure and visually salient features (including texture) of 3D shapes from arbitrary viewpoints, achieving sparse yet semantically rich 3D shape abstraction.

## Background & Motivation

Abstracting 3D shapes into a sparse set of semantically meaningful curves is an important yet challenging problem. The key difficulty lies in identifying the sparse curve set that best represents the visual characteristics of a shape, encompassing both geometry and texture.

Limitations of existing methods:

**Occluding Contours**: A classical non-photorealistic rendering approach that relies entirely on surface geometry analysis, capable of extracting only low-level geometric contours while failing to capture high-level semantic concepts and textures. Being a 2D representation, it suffers from view inconsistency and produces flickering artifacts during dense-viewpoint rendering.

**Limitations of Back-projection Schemes**: Back-projecting multi-view 2D contours into 3D yields dense and visually unappealing line clusters (e.g., for simple cylinders), and cannot handle textures at all.

**3D Stroke-Based Methods such as 3Doodle**: Primarily rely on global CLIP supervision, making them insensitive to fine details (facial features, texture patterns), and lacking spatial guidance and geometric constraints.

The authors argue that genuine 3D shape abstraction must simultaneously capture: visually salient geometric structures, high-level texture concepts (e.g., dragon scales, watermelon seeds), and key features (e.g., facial features), while maintaining multi-view consistency.

## Method

### Overall Architecture

WIR3D employs a two-stage optimization:

- **Stage I (Geometry Abstraction)**: Optimizes a set of curves to represent the overall geometric structure of the shape.
- **Stage II (Texture Abstraction)**: Freezes the geometry curves and introduces additional curves to represent texture and visual details.

Different CLIP architectures are used in each stage: RN101 is sensitive to geometric structure, while RN50x64 is sensitive to high-level visual concepts.

### Curve Representation

3D strokes are modeled as a collection of cubic Bézier curves $\{B_i\}_{i=1}^n$, where each curve is defined by 4 control points:

$$B(t) = (1-t)^3 p^0 + (1-t)^2 t p^1 + (1-t)t^2 p^2 + t^3 p^3$$

3D control points are projected to 2D via perspective projection, and differentiable rasterization is performed using DiffVG.

### Semantic Loss

Following CLIPasso, the CLIP encodings of rendered strokes and the target shape are compared:

$$\mathcal{L}_{\text{semantic}} = \lambda_{\text{fc}} \text{dist}(\text{CLIP}(I_{\text{curve}}), \text{CLIP}(I_{\text{target}})) + \sum_{l=3,4} \|\text{CLIP}_l(I_{\text{curve}}) - \text{CLIP}_l(I_{\text{target}})\|_2^2$$

where $\text{CLIP}_l$ denotes the intermediate activations at layer $l$, and $\text{dist}$ is the cosine distance.

### Local Keypoint Loss (Core Innovation)

To capture fine-grained features, a spatially weighted framework based on 3D keypoints is introduced. Keypoints can be specified by the user or automatically detected via Backto3D feature back-projection followed by KMeans clustering.

A weight map is constructed for each viewpoint using Gaussian decay from keypoint centers:

$$I_{\text{weight}}(x,y) = 1 + \sum_p e^{\frac{-\|(x,y) - p\|^2}{2\sigma^2}}$$

The additive constant 1 ensures that regions far from keypoints still contribute. A z-buffer is maintained to handle keypoint occlusion.

The final local keypoint loss is:

$$\mathcal{L}_{\text{local}} = \lambda_{\text{fc}} \bar{I}_{\text{weight}} \text{dist}(\text{CLIP}(I_{\text{curve}}), \text{CLIP}(I_{\text{target}})) + \sum_{l=3,4} \|I_{\text{weight}} \cdot (\text{CLIP}_l(I_{\text{curve}}) - \text{CLIP}_l(I_{\text{target}}))\|_2^2 + \lambda_{\text{lpips}} \text{LPIPS}(I_{\text{curve}}, I_{\text{target}})$$

### SDF Regularization

A neural SDF (an MLP fitted to the signed distance field of the shape) is used to constrain the curves to remain close to the target surface:

$$\mathcal{L}_{\text{SDF}} = \frac{1}{n \cdot k}\sum_{i=1}^{n}\sum_{k=1}^{s} |\phi(B_i(t_k))|$$

### View Regularization

This term ensures that all curves remain visible across all sampled viewpoints:

$$\mathcal{L}_{\text{ndc}} = \sum_{i=1}^{n}\sum_t \text{ReLU}(\mathcal{P}(B_i(t)) - 1) + \text{ReLU}(-\mathcal{P}(B_i(t)))$$

### Two-Stage Optimization

**Stage I** (Geometry): Uses texture-free targets rendered via Freestyle; the loss is $\mathcal{L}_I = \mathcal{L}_{\text{semantic}} + 0.1 \cdot \mathcal{L}_{\text{SDF}} + \mathcal{L}_{\text{ndc}}$.

**Stage II** (Texture): Freezes Stage I curves; newly added curves are optimized using textured rendering targets with loss $\mathcal{L}_{II} = \mathcal{L}_{\text{local}} + \mathcal{L}_{\text{SDF}} + \mathcal{L}_{\text{ndc}}$.

## Key Experimental Results

### Quantitative Comparison with Baselines

| Method | LPIPS ↓ | CLIP$^{\text{img}}$ ↑ | User Rank ↑ | Coverage ↓ |
|------|---------|----------------------|-------------|------------|
| NEF | 0.313 | 0.86 | - | 0.056 |
| 3Doodle | 0.246 | 0.900 | 0.12 | 0.020 |
| **WIR3D** | **0.227** | **0.909** | **0.88** | **0.008** |

In a user study ($N=96$), WIR3D was selected as the better abstraction **88%** of the time. Coverage (one-sided Chamfer distance from curves to surface) is reduced by more than **2×** compared to 3Doodle.

### Ablation Study

| Configuration | LPIPS ↓ | CLIP$^{\text{img}}$ ↑ | Coverage ↓ |
|------|---------|----------------------|------------|
| **WIR3D (Full)** | **0.227** | **0.909** | **0.008** |
| w/o SDF | 0.229 | 0.904 | 0.012 |
| w/o Local Keypoint | 0.233 | 0.905 | 0.009 |
| w/o Stage 1 | 0.248 | 0.900 | 0.016 |
| w/o CLIP Layers | 0.294 | 0.891 | 0.012 |

### Key Findings

1. **CLIP intermediate layers are most critical**: Removing intermediate-layer activations causes the largest quality degradation (LPIPS from 0.227 to 0.294), demonstrating that spatially grounded semantic features are more informative than global CLIP encodings.
2. **Two-stage design is necessary**: Removing Stage I causes the optimization to overfit to specific viewpoints, producing a "flattening" artifact.
3. **Keypoint robustness**: Random keypoints yield results similar to using no keypoint loss (degenerating to Eq. 2), whereas semantically meaningful keypoints consistently improve results.
4. **Abstraction level control**: The number of curves naturally controls the degree of abstraction — more curves capture finer details.
5. **Robustness to in-the-wild reconstruction**: The method successfully abstracts shapes reconstructed from photographs with significant geometric artifacts.

### Applications

- **Interactive feature control**: Users progressively add details by selecting keypoints (e.g., airplane wheels, Nefertiti's headband).
- **Shape deformation**: Curves serve as intuitive deformation handles; deformations are propagated to the surface via Euclidean distance-based skinning weights. In a user study ($N=42$), WIR3D deformations were judged more preferable **80%** of the time.

## Highlights & Insights

- Exploiting the differential sensitivity of distinct CLIP architectures to geometry versus semantic concepts is an elegant design choice.
- The pipeline of keypoints → spatial weight maps → weighted CLIP intermediate-layer loss provides a principled mechanism for precisely directing 2D visual foundation model knowledge toward 3D representation.
- SDF regularization not only ensures geometric fidelity but also incidentally endows the curves with the capacity to serve as deformation handles.
- The method imposes no requirements on input mesh quality or topology, making it highly practical.

## Limitations & Future Work

- Quality is dependent on keypoint quality; random keypoints degrade performance to the level of using no keypoints.
- Preprocessing is time-consuming: neural SDF fitting, automatic keypoint detection, and Freestyle rendering can take up to 2 hours for complex models.
- The default setting of 20 curves per stage may be insufficient for shapes with highly complex geometry.
- The optimization relies purely on CLIP-based rendering comparison, with no guarantee of reliable abstraction for all texture types.

## Related Work & Insights

- **Shape decomposition**: Tulsiani (cuboids), CvxNet (convex polyhedra) — focused on reconstruction rather than abstraction.
- **Non-photorealistic rendering**: Occluding contours (view-inconsistent), NEF/implicit edge fields (focused on geometric boundaries, incapable of abstraction).
- **Curve abstraction**: CLIPasso (2D sketch abstraction), 3Doodle (3D curves but with global supervision, insensitive to fine details).

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall | 8.0/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Manual-PA: Learning 3D Part Assembly from Instruction Diagrams](manual-pa_learning_3d_part_assembly_from_instruction_diagrams.md)
- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](../../NeurIPS2025/self_supervised/soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)
- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](../../NeurIPS2025/self_supervised/starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)
- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](../../NeurIPS2025/self_supervised/seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[CVPR 2026\] Shape-of-You: Fused Gromov-Wasserstein Optimal Transport for Semantic Correspondence in-the-Wild](../../CVPR2026/self_supervised/shape-of-you_fused_gromov-wasserstein_optimal_transport_for_semantic_corresponde.md)

</div>

<!-- RELATED:END -->
