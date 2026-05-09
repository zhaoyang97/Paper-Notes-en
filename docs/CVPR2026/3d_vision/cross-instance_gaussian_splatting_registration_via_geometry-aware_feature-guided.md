---
title: >-
  [Paper Note] Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment
description: >-
  [CVPR 2026][3D Vision][3D Gaussian Splatting] This paper proposes GSA (Gaussian Splatting Alignment), the first method for category-level cross-instance registration of 3DGS models. It combines geometry-aware feature-guided coarse alignment (extending ICP to solve similarity transformations) with multi-view feature consistency fine alignment, substantially outperforming existing methods in both same-instance and cross-instance scenarios.
tags:
  - CVPR 2026
  - 3D Vision
  - 3D Gaussian Splatting
  - cross-instance registration
  - similarity transformation
  - geometry-aware features
  - inverse radiance field
date: 2026-05-08
content_hash: 4c01bc1291eded5f
---

# Cross-Instance Gaussian Splatting Registration via Geometry-Aware Feature-Guided Alignment

**Conference**: CVPR 2026
**arXiv**: [2603.21936](https://arxiv.org/abs/2603.21936)
**Code**: [https://bgu-cs-vil.github.io/GSA-project](https://bgu-cs-vil.github.io/GSA-project)
**Area**: 3D Vision / 3D Registration
**Keywords**: 3D Gaussian Splatting, cross-instance registration, similarity transformation, geometry-aware features, inverse radiance field

## TL;DR
This paper proposes GSA (Gaussian Splatting Alignment), the first method for category-level cross-instance registration of 3DGS models. It combines geometry-aware feature-guided coarse alignment (extending ICP to solve similarity transformations) with multi-view feature consistency fine alignment, substantially outperforming existing methods in both same-instance and cross-instance scenarios.

## Background & Motivation

**Background**: 3D Gaussian Splatting (3DGS) has become a powerful representation for high-fidelity novel view synthesis. However, aligning two independently reconstructed 3DGS models remains an open challenge; existing methods such as GaussReg rely on ICP and can only handle registration between models of the same object instance.

**Limitations of Prior Work**: (1) ICP fails under poor initialization (e.g., 180° rotation); (2) ICP cannot handle unknown scale and requires ground-truth scale as input; (3) in cross-instance (different object) registration, geometric discrepancies cause nearest-point matching to converge to incorrect correspondences.

**Key Challenge**: 3DGS models reconstructed via SfM inherently exhibit arbitrary differences in scale, position, and orientation; cross-instance objects further introduce shape and appearance variation, rendering traditional geometric methods entirely ineffective.

**Goal**: To align two 3DGS models—potentially representing different objects of the same category—via a similarity transformation (rotation + translation + scale) under unknown scale conditions.

**Key Insight**: (1) Replace pure geometric signals with geometry-aware viewpoint-guided features for correspondence establishment; (2) generalize the inverse radiance field framework from single-view camera pose estimation to multi-view field-to-field registration.

**Core Idea**: Geometry-aware semantic feature-guided extended ICP + multi-view feature field consistency optimization = robust cross-instance 3DGS registration.

## Method

### Overall Architecture
The pipeline consists of three stages: (1) **Preprocessing**: camera poses are estimated via COLMAP, geometry-aware features are extracted using the method of Mariotti et al., and foreground is segmented with SAM; (2) **Coarse Alignment**: a feature-guided iterative absolute orientation solver estimates the Sim(3) transformation; (3) **Fine Alignment**: multi-view feature field consistency optimization further refines registration accuracy.

### Key Designs

1. **Feature-Augmented 3DGS**:

    - **Function**: Each Gaussian is augmented with a 3-dimensional geometry-aware feature $\mathbf{f} \in \mathbb{R}^3$, lifted from 2D viewpoint-guided spherical features into 3D.
    - **Mechanism**: During training, RGB loss and feature loss are jointly optimized: $\mathcal{L} = \mathcal{L}_{\text{rgb}} + \lambda_f \mathcal{L}_f$, where $\mathcal{L}_f = \|F - F^r\|_1$. Feature and geometry optimization are decoupled: color and geometry are optimized first, then fixed before optimizing features.
    - **Design Motivation**: The geometry-aware features of Mariotti et al. are preferred over DINOv2, as the latter lacks 3D geometric awareness and suffers from spatial ambiguity (e.g., symmetric parts sharing identical features), making it unsuitable for registration tasks.

2. **Coarse Alignment: Feature-Guided Absolute Orientation Solver**:

    - **Function**: Three steps are iteratively alternated — (a) for each source point, a candidate set $\mathcal{Q}_i = \{\mathbf{q}_j \mid \|\mathbf{f}_i - \mathbf{f}_j\| \leq \tau_f\}$ is filtered by feature similarity in the target, from which the spatially nearest point is selected; (b) the optimal Sim(3) transformation is solved in closed form (Kabsch–Umeyama for rotation and translation, Horn for scale); (c) the transformation is applied.
    - **Mechanism**: $\min_{T^{(k)} \in \mathbf{Sim(3)}} \sum_i \|T^{(k)}(\mathbf{p}_i^{(k)}) - \mathbf{q}_i^{(k)}\|_2^2$
    - **Design Motivation**: Feature-constrained candidate filtering addresses all three failure modes of ICP—sensitivity to initialization, inability to handle unknown scale, and failure in cross-instance settings—while converging within only 3–6 iterations.

3. **Fine Alignment: Multi-View Feature Field Consistency**:

    - **Function**: Initialized from the coarse alignment result, the method optimizes a multi-view feature rendering consistency loss:
    $\mathcal{L}_{\text{MV-FC}} = \sum_{k=1}^N \|\text{Rend}_f(T\mathcal{G}_1, C_k^*) - \text{Rend}_f(\mathcal{G}_2, C_k^*)\|_2^2$
    - **Mechanism**: Generalized from the inverse radiance field formulation (iNeRF)—extending single-view camera pose estimation in SE(3) to multi-view field-to-field registration in Sim(3), and replacing color rendering with feature rendering to support cross-instance alignment.
    - **Design Motivation**: Multi-view constraints eliminate scale–depth ambiguity inherent to single-view settings; feature rendering enables alignment of cross-instance objects with differing appearances, as geometry-aware features are consistent across objects within the same category.

### Loss & Training
- 3DGS construction: $\mathcal{L} = \mathcal{L}_{\text{rgb}} + \lambda_f \mathcal{L}_f$, $\lambda_f=1$, $\alpha=0.2$
- Coarse alignment: iterative closest point with closed-form solver, $\tau_f=0.01$, maximum 6 iterations
- Fine alignment: multi-view feature consistency, 3 diverse viewpoints, 60 optimization iterations, learning rate 0.01

## Key Experimental Results

### Main Results — Same-Instance Registration (Objaverse, 15 objects)

| Method | Requires Ground-Truth Scale? | Mean RRE (°) ↓ | Notes |
|--------|------------------------------|----------------|-------|
| FGR | Yes | Very high | Fails on noisy data |
| REGTR | Yes | Very high | Assumes rigid transformation |
| GaussReg | Yes | High | Sensitive to initialization |
| GSA (coarse only) | **No** | SOTA | Coarse alignment alone surpasses all baselines |
| GSA (coarse + fine) | **No** | **Near-perfect** | Order-of-magnitude improvement |

### Cross-Instance Registration (ShapeNet, 6 categories × 10 pairs)

| Method | Mean RRE (°) ↓ | Notes |
|--------|----------------|-------|
| FGR | Extremely high | Complete failure |
| REGTR | Extremely high | Complete failure |
| GaussReg | Extremely high | Complete failure |
| GSA | **Lowest** | First effective cross-instance solution |

### Ablation Study

| Configuration | Effect on RRE | Notes |
|---------------|---------------|-------|
| Remove feature guidance (pure ICP) | Coarse 136.29°, fine 139.82° | Complete failure |
| Replace with DINOv2 features | Typically complete failure | Spatial ambiguity |
| Replace feature rendering with color rendering in fine alignment | Significant accuracy drop | Cross-instance appearance mismatch |
| 3 similar viewpoints (vs. 3 diverse) | Accuracy drop | Viewpoint diversity is important |

### Key Findings
- The coarse alignment stage alone achieves state-of-the-art performance; fine alignment further reduces error to near-perfect levels (same-instance setting).
- GSA successfully aligns models even when initialization involves 180° rotation and 10× scale difference.
- Geometry-aware features are critical to success—alternatives such as DINOv2 fail entirely on registration tasks.

## Highlights & Insights
- **First category-level 3DGS registration**: Fills the gap in cross-instance alignment, enabling new applications such as object substitution and synchronized novel view synthesis.
- **Elegant theoretical derivation**: The generalization from iNeRF to field-to-field registration is logically rigorous, with progressive extensions from SE(3) to Sim(3), single-view to multi-view, and color to feature rendering.
- **Practical efficiency**: Coarse alignment converges in 3 iterations and fine alignment in 60 iterations, yielding acceptable overall runtime.

## Limitations & Future Work
- Performance depends on the quality of geometry-aware features; degraded features result in reduced alignment accuracy.
- Validation is limited to object-level scenarios; extension to scene-level settings (complex multi-object environments) remains unexplored.
- The multi-view selection strategy in fine alignment could be further automated, as the current approach relies on predefined viewpoints.

## Related Work & Insights
- The comparison with GaussReg highlights the importance of feature-guided correspondence.
- The generalization of inverse radiance field frameworks (iNeRF, iComMa) to registration is broadly applicable beyond this specific task.
- The selection of geometry-aware features (Mariotti et al.) is critical for registration, suggesting a new application domain for 3D feature learning research.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First cross-instance 3DGS registration with an elegant theoretical derivation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Synthetic and real data, same-instance and cross-instance settings, comprehensive ablation
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivations, progressive structure, strong readability
- Value: ⭐⭐⭐⭐⭐ Pioneering work that unlocks new application directions

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] NG-GS: NeRF-Guided 3D Gaussian Splatting Segmentation](ng_gs_nerf_guided_3d_gaussian_splatting_segmentation.md)
- [\[CVPR 2026\] GaussianGrow: Geometry-aware Gaussian Growing from 3D Point Clouds with Text Guidance](gaussiangrow_geometry-aware_gaussian_growing_from_3d_point_clouds_with_text_guid.md)
- [\[CVPR 2026\] CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration](cmhanet_a_crossmodal_hybrid_attention_network_for.md)
- [\[AAAI 2026\] GT2-GS: Geometry-aware Texture Transfer for Gaussian Splatting](../../AAAI2026/3d_vision/gt2-gs_geometry-aware_texture_transfer_for_gaussian_splatting.md)
- [\[CVPR 2026\] DirectFisheye-GS: Enabling Native Fisheye Input in Gaussian Splatting with Cross-View Joint Optimization](directfisheye-gs_enabling_native_fisheye_input_in_gaussian_splatting_with_cross-.md)

<!-- RELATED:END -->
