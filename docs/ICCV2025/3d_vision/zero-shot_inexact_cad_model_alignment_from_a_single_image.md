---
title: >-
  [Paper Note] Zero-Shot Inexact CAD Model Alignment from a Single Image
description: >-
  [ICCV 2025][3D Vision][CAD alignment] A weakly supervised 9-DoF CAD model alignment method that enhances DINOv2 features with geometry awareness and performs dense alignment optimization in Normalized Object Coordinate (…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "CAD alignment"
  - "zero-shot"
  - "9-DoF pose estimation"
  - "foundation models"
  - "NOC"
date: 2026-05-08
content_hash: 0756bbd82bf0d647
---

# Zero-Shot Inexact CAD Model Alignment from a Single Image

**Conference**: ICCV 2025
**arXiv**: [2507.03292](https://arxiv.org/abs/2507.03292)
**Code**: [https://zerocad9d.github.io/](https://zerocad9d.github.io/)
**Area**: 3D Vision
**Keywords**: CAD alignment, zero-shot, 9-DoF pose estimation, foundation models, NOC

## TL;DR

A weakly supervised 9-DoF CAD model alignment method that enhances DINOv2 features with geometry awareness and performs dense alignment optimization in Normalized Object Coordinate (NOC) space, enabling zero-shot 3D alignment without pose annotations that generalizes to unseen categories.

## Background & Motivation

Recovering 3D scene structure from a single image is highly ill-posed (depth ambiguity + heavy occlusion). A practical solution is to retrieve an approximate 3D model from a database and align it with the target object in the image (9-DoF: 6D rigid transformation + 3D anisotropic scaling).

Limitations of existing methods:
- **Supervised methods** (ROCA, SPARC): Require annotated quadruples of RGB + depth + CAD model + 9-DoF pose, and are limited to a fixed set of trained categories.
- **Synthetic data methods** (DiffCAD): Rely on photorealistic synthetic scenes (3D-FRONT), with restricted categories and domain gaps.
- **Foundation model methods** (FoundationPose): Designed for 6-DoF tasks and **exact matching** (consistent model texture/shape), performing poorly on inexact retrieved models.
- **Inherent limitations of DINOv2 features**: (1) Symmetric parts (e.g., left and right chair legs) produce highly similar features and cannot be distinguished; (2) Sensitivity to texture variation makes handling texture-free models difficult.

Core insight: Although DINOv2 features cannot directly distinguish symmetric parts, they may implicitly encode latent information sufficient to predict part positions — which can be reorganized via a lightweight adapter.

## Method

### Overall Architecture

A coarse-to-fine pose estimation pipeline:
1. **Coarse alignment**: Encode image and 3D model into a shared feature space, establish 2D–3D correspondences via nearest-neighbor matching, and solve for an initial pose with RANSAC.
2. **Fine alignment**: Perform dense image-level alignment optimization in NOC space, backpropagating through differentiable rendering to refine pose parameters.

### Key Designs

1. **Geometry-aware feature adapter**: A lightweight MLP $E_\theta$ is trained to transform DINOv2 features into geometry-aware features. Training data consists of 9 ShapeNet CAD categories rendered and augmented with a diffusion model (300K images total), optimized with two objectives:

   **NOC prediction loss**: Encourages features to encode 3D positional information.
   $$\mathcal{L}_{\text{NOC}} = \frac{1}{n \cdot h \cdot w}\sum_{i=1}^{n}\|D_\phi(E_\theta(\text{DINO}(\mathbf{R}_i))) - \mathbf{N}_i\|_2^2$$

   **Geometric consistency triplet loss**: Enforces cross-view feature consistency for the same part and feature dissimilarity for geometrically distant parts.
   $$\mathcal{L}_{\text{triplet}} = \frac{1}{|\mathcal{T}|}\sum_{(\mathbf{a},\mathbf{p},\mathbf{n})\in\mathcal{T}}[d(\mathbf{a},\mathbf{n}) - d(\mathbf{a},\mathbf{p}) + \alpha]_+$$
   Positives: points with 3D distance $\leq \tau_{\text{dist}}^+=0.02$; negatives: points with distance $\geq \tau_{\text{dist}}^-=0.4$ and feature cosine similarity $> \tau_{\text{feat}}^-=0.75$ (**hard negative mining**, specifically targeting symmetric parts indistinguishable by DINOv2).

   Final features fuse DINOv2 and adapter outputs: $E_f(\mathbf{I}) = (1-\omega)\cdot\hat{\text{DINO}}(\mathbf{I}) \oplus \omega\cdot\hat{E}_\theta(\text{DINO}(\mathbf{I}))$ ($\omega=0.5$).

2. **3D model feature voxel grid**: Each CAD model is rendered from 36 viewpoints with 7× augmentation (288 images total); features extracted by $E_f$ are back-projected into a 3D voxel grid ($100^3$) and averaged across views to form a unified 3D feature representation. Multi-scale downsampling–upsampling smoothing is applied to the voxel grid.

3. **Dense image alignment optimization (NOC space)**: The input image is converted to a NOC map $\mathbf{N}^\mathbf{I}$ via nearest-neighbor matching (each pixel feature is matched to the closest 3D voxel, whose position serves as the NOC value). Three losses are optimized:

   - **NOC alignment loss**: $\mathcal{L}_{\text{NOC-A}} = \frac{1}{m}\|\mathbf{M} \odot (\mathbf{N}^\mathbf{I} - \mathbf{N}^t)\|_1$
   - **Silhouette loss**: $\mathcal{L}_{\text{mask}} = \frac{1}{HW}\|\mathbf{S}^\mathbf{I} - \mathbf{S}^t\|_1$ (using SAM segmentation + SoftRasterizer differentiable rendering)
   - **Depth loss**: $\mathcal{L}_{\text{depth}} = \frac{1}{m}\|\mathbf{M} \odot (\mathbf{D}^\mathbf{I} - \mathbf{D}^t)\|_1$ (using DepthAnything for metric depth prediction)

   Key advantage: The NOC map derived from nearest-neighbor matching is naturally invariant to global translation and scaling in feature space, making it more robust to domain shifts than direct neural network NOC regression.

### Loss & Training

- Adapter training: $\mathcal{L}_{\text{adapter}} = (1-\beta)\mathcal{L}_{\text{NOC}} + \beta\mathcal{L}_{\text{triplet}}$, $\beta=0.1$
- 2-layer MLP adapter + 1-layer MLP decoder (decoder discarded after training)
- AdamW optimizer, lr=3e-4, batch=140
- Coarse alignment: RANSAC + 3D–3D correspondence solving via metric depth back-projection
- Fine alignment: Adam, lr=0.005, PyTorch3D differentiable rendering

## Key Experimental Results

### Main Results (ScanNet25k, 9-DoF NMS Alignment Accuracy)

| Method | Supervision | Bathtub | Chair | Display | Sofa | Table | Avg Cat.↑ | Avg Inst.↑ |
|--------|-------------|---------|-------|---------|------|-------|-----------|------------|
| ROCA | Fully supervised | 22.5 | 41.0 | 30.4 | 15.9 | 14.6 | 21.5 | 27.4 |
| SPARC | Fully supervised | 26.7 | 52.6 | 22.5 | 32.7 | 17.7 | 27.3 | 33.9 |
| FoundationPose(9D) | Weakly supervised | 20.0 | 41.8 | 23.6 | 15.0 | 17.5 | 19.2 | 25.7 |
| **Ours** | **Weakly supervised** | **16.7** | **49.3** | **24.1** | **38.1** | **16.5** | **23.1** | **30.1** |

*The only weakly supervised method to surpass fully supervised ROCA (+1.6% avg. category, +2.7% avg. instance).*

### Ablation Study (ScanNet25k, Coarse–Fine Alignment Combinations)

| Coarse | Fine | Avg Cat.↑ | Avg Inst.↑ |
|--------|------|-----------|------------|
| DINOv2 | — | 13.1 | 18.8 |
| DINOv2 | Ours (NOC) | 17.3 | 24.2 |
| Ours | — | 18.3 | 26.0 |
| Ours | FM (feature matching) | 18.3 | 26.1 |
| **Ours** | **Ours (NOC)** | **23.1** | **30.1** |

*Geometry-aware features improve over DINOv2 by +5.2%/+7.3%; NOC dense optimization improves over no fine alignment by +4.8%/+4.1% and over feature-matching fine alignment (FM) by +4.8%/+4.0%.*

### Generalization to Unseen Categories on SUN2CAD (20 categories, single-view accuracy)

| Method | Supervision | piano | printer | lamp | mug | oven | Avg Cat.↑ | Avg Inst.↑ |
|--------|-------------|-------|---------|------|-----|------|-----------|------------|
| SPARC | Fully supervised | 27.8 | 14.1 | 3.0 | 0.0 | 0.0 | 6.9 | 4.9 |
| DINOv2 | None | 44.4 | 7.6 | 5.4 | 0.0 | 7.1 | 11.6 | 7.3 |
| **Ours** | **Weakly supervised** | **50.0** | **25.0** | **6.1** | **10.2** | **57.1** | **24.5** | **17.6** |

*Outperforms the strongest baseline by a large margin of +12.7% across 20 unseen categories, surpassing SPARC on 18 of 20 categories.*

### Key Findings

- $\beta=0.1$ (triplet loss weight at 10%) yields the lowest NOC prediction error — excessive contrastive learning sacrifices positional prediction capacity.
- Feature fusion at $\omega=0.5$ is optimal — DINOv2 semantic information and adapter geometric information are complementary.
- The three dense alignment losses are complementary: NOC improves scale and rotation, depth improves translation and rotation, and silhouette further boosts performance.
- Nearest-neighbor-based NOC prediction is more robust to domain shifts than direct neural network regression (invariant to global feature space offsets).

## Highlights & Insights

- **Elegant hard negative mining design**: Triplet negatives are specifically selected as points with high DINOv2 cosine similarity but large 3D distance, precisely targeting the weaknesses of foundation features on symmetric parts.
- **NOC space is better suited for dense alignment than feature space**: NOC maps are inherently smooth, renderable via simple rasterization (no network inference required), and nearest-neighbor matching is invariant to global shifts in feature space.
- **Strong generalization capability**: Training on only 9 categories generalizes to 20 entirely unseen categories, surpassing supervised methods that rely on category priors.
- The proposed SUN2CAD benchmark fills the gap in 9-DoF alignment evaluation for unseen categories.

## Limitations & Future Work

- Coarse alignment is not robust for heavily occluded or cropped objects (e.g., tables, bathtubs, beds), affecting scale and rotation estimation.
- Performance depends on the quality of SAM segmentation and DepthAnything depth predictions.
- The adapter still requires training on 9 ShapeNet categories — training on larger-scale CAD renderings may further improve performance.
- Building feature voxel grids for each object requires inference over 288 images, limiting real-time applicability.
- SUN2CAD annotations are derived from coarse 3D bounding box alignment followed by manual refinement, resulting in limited annotation precision.

## Related Work & Insights

- The paradigm of lightweight adapters for adapting foundation model features to specific geometric tasks is transferable to other 3D perception problems.
- The implicit hard negative mining strategy in the triplet loss is applicable to any task requiring discrimination of semantically similar but positionally distinct parts.
- Dense alignment in NOC space avoids the difficulties of RGB texture matching and is applicable to all inexact matching scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ (Effective combination of feature adaptation and NOC-space optimization)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple baselines, SUN2CAD benchmark, comprehensive ablations, hyperparameter analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear problem formulation, intuitive pipeline diagrams)
- Value: ⭐⭐⭐⭐⭐ (Zero-shot generalization substantially increases practical deployability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Diorama: Unleashing Zero-shot Single-view 3D Indoor Scene Modeling](diorama_unleashing_zeroshot_singleview_3d_indoor_scene_model.md)
- [\[ICCV 2025\] ZeroStereo: Zero-shot Stereo Matching from Single Images](zerostereo_zero-shot_stereo_matching_from_single_images.md)
- [\[ICCV 2025\] MonoMobility: Zero-Shot 3D Mobility Analysis from Monocular Videos](monomobility_zero-shot_3d_mobility_analysis_from_monocular_videos.md)
- [\[ICCV 2025\] one look is enough seamless patchwise refinement for zero-shot monocular depth e](one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)
- [\[ICCV 2025\] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes](bufferx_towards_zeroshot_point_cloud_registration_in_diverse.md)

</div>

<!-- RELATED:END -->
