---
title: >-
  [Paper Note] Learning 3D-Aware GANs from Unposed Images with Template Feature Field
description: >-
  [ECCV 2024][3D Vision][3D-aware GAN] Proposes Template Feature Field (TeFF), which jointly learns a generative radiance field and a semantic feature field to automatically extract 3D templates and estimate camera poses online from unposed in-the-wild images, thereby enabling generative adversarial learning of complete 3D geometries.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D-aware GAN"
  - "unposed images"
  - "template feature field"
  - "pose estimation"
  - "NeRF"
date: 2026-05-08
content_hash: 4317246a372f4ba3
---

# Learning 3D-Aware GANs from Unposed Images with Template Feature Field

**Conference**: ECCV 2024  
**arXiv**: [2404.05705](https://arxiv.org/abs/2404.05705)  
**Code**: [Yes](https://XDimlab.github.io/TeFF)  
**Area**: 3D Vision  
**Keywords**: 3D-aware GAN, unposed images, template feature field, pose estimation, NeRF

## TL;DR

Proposes Template Feature Field (TeFF), which jointly learns a generative radiance field and a semantic feature field to automatically extract 3D templates and estimate camera poses online from unposed in-the-wild images, thereby enabling generative adversarial learning of complete 3D geometries.

## Background & Motivation

3D-aware GANs have made significant progress in recent years. The core idea is to lift the generator into 3D space (such as NeRF) and generate 2D images via volume rendering. However, existing methods (such as EG3D) typically **assume that the camera pose distribution of the training images is known**. This poses a strong constraint in practical scenarios, as estimating precise camera poses for real-world images requires specific 3D priors, which is highly impractical for most in-the-wild object categories.

To remove the known-pose assumption, several methods (such as CAMPARI, PoF3D, 3DGP) attempt to jointly learn the camera pose distribution and the 3D content using the generator. However, these methods perform poorly on **multimodal pose distributions** (e.g., objects with a 360° field of view). The core reason is that the generated camera poses and the object orientations are **entangled** in the 2D image space. For instance, the model may rotate the object in different directions to match the target distribution (instead of moving the camera), resulting in **incomplete** 3D geometries where certain viewpoints are never observed.

The core insight of this paper is to **decouple pose estimation from GAN training**. Specifically, the proposed method leverages the cross-instance semantic alignment capability of self-supervised semantic features (such as DINO features)—where corresponding semantic parts of different instances within the same category (e.g., car wheels) remain consistent in the feature space. The authors propose to learn a 3D semantic template feature field as a canonical object space, thereby reformulating the pose estimation of real images as a 2D-3D matching problem.

## Method

### Overall Architecture

TeFF introduces key extensions to EG3D: the generator generates not only a radiance field (color + density) but also a semantic feature field (features + shared density). 2D RGB images and 2D feature maps are obtained via volume rendering. By using the mean noise input of the generator, a category-level 3D template feature field is automatically obtained. Cameras poses are then estimated online for each real image through 2D-3D matching.

### Key Designs

1. **Generating Radiance and Feature Fields**:

    - The generator $G_\psi$ maps random noise $\mathbf{z}$ to a radiance field and a feature field: $G_\psi: \mathbb{R}^3 \times \mathbb{R}^M \to \mathbb{R}^3 \times \mathbb{R}^F \times \mathbb{R}^+$, mapping each 3D point $\mathbf{x}$ to color $\mathbf{c}$, semantic feature $\mathbf{f}$, and density $\sigma$.
    - In practice, this is implemented via two sets of tri-planes: one for color and density, and another for features.
    - Volume rendering formulas: $\mathbf{c}_r = \sum_{i=1}^N T_i \alpha_i \mathbf{c}_i$, $\mathbf{f}_r = \sum_{i=1}^N T_i \alpha_i \mathbf{f}_i$, where color and features **share the density**.
    - Design Motivation: Shared density ensures that the semantic feature field is geometrically aligned with the radiance field, while the cross-instance alignment capability of the semantic features enables pose estimation.

2. **Template Feature Field**:

    - By applying EMA to the generator, $\overline{G}_\psi$ is obtained, which takes the mean noise $\mathbf{z}_0$ as input to produce the category-level template feature field.
    - The template automatically exploits the dataset's average shape discovered by the generative model.
    - DINO is used as the 2D semantic feature extractor, with PCA applied to reduce the dimensionality to the top 3 principal components.
    - Design Motivation: The mean noise naturally corresponds to the "average appearance" of the category, preventing feature bias from single instances. The cross-instance semantic alignment of DINO features makes 2D-3D matching feasible.

3. **Online Camera Pose Estimation**:

    - The camera model is parameterized as $\boldsymbol{\xi} = (\theta, \phi, \gamma, r)$, representing azimuth, elevation, in-plane rotation, and spherical radius.
    - **Azimuth-Elevation Discretization**: Discretize $\theta$ and $\phi$ into $N_\theta$ and $N_\phi$ values (e.g., 36×18), and render a set of 2D feature maps $\{\overline{\mathbf{F}}_k\}$ from the template.
    - **In-Plane Rotation and Scale Estimation via Phase Correlation**: Leverage frequency-domain methods to efficiently estimate $r$ and $\gamma$, avoiding a brute-force search over the 4D space.
    - **Pose Sampling**: Calculate the MSE between each transformed template $\tilde{\mathbf{F}}_k$ and the real feature $\mathbf{F}$, and compute a pose probability distribution using softmax temperature $\tau$: $p(k) = \text{softmax}(-e_k \cdot \tau)$.
    - The temperature is low in the early stages of training (to explore more poses) and increases in the later stages (to lock onto the best pose).
    - Design Motivation: Compared to establishing point-to-point 2D-3D correspondences (which can easily confuse symmetric parts like left and right legs), global grid search combined with phase correlation is more robust and efficient.

### Loss & Training

- **GAN Loss**: Non-saturating GAN loss + R1 regularization, incorporating an image discriminator $D_\zeta^I$ and a feature discriminator $D_\zeta^F$.
- The feature discriminator takes low-resolution RGB and semantic feature maps as input. Gradients from $D_\zeta^F$ to the RGB branch are stopped.
- **Foreground-Background Decoupling**: The foreground is generated using 3D NeRF, and the background is generated via 2D StyleGAN2, sharing the same latent code.
- **Template Update Strategy**: The template is updated every 16 steps during the first 3k iterations, and once per epoch thereafter.

## Key Experimental Results

### Main Results

Evaluated against EG3D, 3DGP, and PoF3D on 4 datasets (ShapeNet Cars, CompCars, SDIP Elephant, LSUN Plane):

| Dataset | Metric | TeFF (Ours) | EG3D | 3DGP | PoF3D |
|--------|------|-------------|------|------|-------|
| ShapeNet Cars | FID_gt↓ | **5.95** | 7.25 | 139.48 | 12.72 |
| ShapeNet Cars | Depth_gt↓ | **0.53** | 0.61 | 4.84 | 0.65 |
| CompCars | FID_360↓ | 27.71 | **7.06** | 187.20 | 44.52 |
| CompCars | Depth_360↓ | **0.31** | 0.95 | 4.02 | 10.31 |
| SDIP Elephant | FID_360↓ | **5.51** | 6.03 | 196.04 | 36.32 |
| SDIP Elephant | Depth_360↓ | **0.60** | 1.10 | 3.29 | 3.14 |
| LSUN Plane | Depth_360↓ | **0.78** | 1.19 | 3.84 | 1.37 |

Pose distribution estimation (ShapeNet Cars KL divergence):

| Method | θ KL↓ | ϕ KL↓ |
|------|-------|-------|
| 3DGP | 40.4571 | 39.3625 |
| PoF3D | 4.4829 | 0.5495 |
| **TeFF** | **0.0555** | **0.0696** |

### Ablation Study

| Configuration | θ KL↓ | ϕ KL↓ | Description |
|------|-------|-------|------|
| TeRF_RGB | 0.0663 | 0.1422 | Pose estimation using RGB template |
| TeRF_Gray | 0.0656 | 0.1490 | Grayscale RGB template |
| **TeFF (Ours)** | **0.0555** | **0.0696** | Semantic feature template, optimal |

| DoF | Depth_360↓ | FID_360↓ | FID_est↓ | Description |
|--------|-----------|---------|---------|------|
| 2 DoF (θ,ϕ) | 4.98 | 39.66 | 11.09 | Geometric errors |
| **4 DoF (θ,ϕ,γ,r)** | **0.31** | **27.31** | 20.60 | Complete geometry |

### Key Findings

- 3DGP and PoF3D achieve very low FID under their estimated pose distributions but suffer a huge spike in FID under a uniform 360° distribution (due to pose distribution collapse).
- For TeFF, FID_360 and FID_est are essentially consistent, indicating that the model learns a complete 3D object representation.
- Semantic features are better suited for cross-instance pose matching than RGB features, as they are invariant to appearance variations.
- A 4-DoF camera model (including scale and in-plane rotation) is crucial for handling scale variations in real-world data.

## Highlights & Insights

- **Core Innovation**: Leveraging the **cross-instance alignment** of DINO semantic features to construct 3D templates, effectively decoupling pose estimation from GAN training—an elegant and highly effective solution.
- **Clever Application of Phase Correlation**: Introducing traditional image registration techniques to pose estimation in 3D-aware GANs, avoiding the computational explosion associated with high-dimensional grid searches.
- **"Mean = Template" Insight**: The mean noise of the generative model naturally yields a category-level template, eliminating the need for extra annotations or clustering.

## Limitations & Future Work

- Cannot handle images with **significant perspective distortion**, as the model might distort the geometry to fit perspective effects.
- Using MSE for 2D-3D matching is susceptible to interference from geometric shape variations.
- The single-template design restricts the method to **single-category** scenarios, necessitating multi-template approaches for multi-class scenes.
- It does not model **articulated motion** of objects, which can lead to inconsistent articulation states generated from different viewpoints.

## Related Work & Insights

- **EG3D** (Chan et al., 2022): A 3D-aware GAN utilizing tri-plane representations, serving as the base architecture for this work.
- **PoF3D** (2023): A pose-free generator that does not require pose priors, though its pose distribution learning is prone to collapsing.
- **3DGP** (2023): A 3D-aware GAN with a 6DoF camera model, similarly limited by the joint learning of pose distributions.
- **Application of DINO features in 3D**: Previously used primarily for scene decomposition and editing. This is the first work to utilize them for cross-instance pose estimation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The framework design combining a template feature field with online pose estimation is highly ingenious, and the introduction of phase correlation is extremely elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluated across 4 datasets using multiple metrics, with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Logically clear and visually intuitive.
- **Value**: ⭐⭐⭐⭐ — Resolves a key limitation of 3D-aware GANs, bearing clear practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Improving 2D Feature Representations by 3D-Aware Fine-Tuning](improving_2d_feature_representations_by_3d-aware_fine-tuning.md)
- [\[ECCV 2024\] Learning 3D Geometry and Feature Consistent Gaussian Splatting for Object Removal](learning_3d_geometry_and_feature_consistent_gaussian_splatting_for_object_remova.md)
- [\[ICCV 2025\] SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images](../../ICCV2025/3d_vision/spatialsplat_efficient_semantic_3d_from_sparse_unposed_images.md)
- [\[CVPR 2026\] Learning 3D Representations for Spatial Intelligence from Unposed Multi-View Images](../../CVPR2026/3d_vision/learning_3d_representations_for_spatial_intelligence_from_unposed_multi-view_ima.md)
- [\[ECCV 2024\] Lagrangian Hashing for Compressed Neural Field Representations](lagrangian_hashing_for_compressed_neural_field_representations.md)

</div>

<!-- RELATED:END -->
