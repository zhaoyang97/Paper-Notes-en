---
title: >-
  [Paper Note] NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement
description: >-
  [3D Vision] NeuraLeaf disentangles the 3D geometry of leaves into two latent spaces — a 2D base shape space and a 3D deformation space — leveraging large-scale 2D leaf image datasets to learn the shape space…
tags:
  - "3D Vision"
date: 2026-05-08
content_hash: c0347727621c95a8
---

# NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement

## Paper Information
- **Conference**: ICCV 2025
- **arXiv**: [2507.12714](https://arxiv.org/abs/2507.12714)
- **Code**: [Project Page](https://neuraleaf-yang.github.io/)
- **Area**: 3D Vision
- **Keywords**: Neural parametric models, leaf modeling, shape and deformation disentanglement, skeleton-free skinning, 3D reconstruction

## TL;DR
NeuraLeaf disentangles the 3D geometry of leaves into two latent spaces — a 2D base shape space and a 3D deformation space — leveraging large-scale 2D leaf image datasets to learn the shape space, proposes a skeleton-free skinning model to handle highly flexible leaf deformations, and introduces DeformLeaf, the first 3D dataset dedicated to leaf deformation modeling.

## Background & Motivation

3D leaf modeling and reconstruction are critical in agriculture and computer graphics. Existing methods face the following challenges:

**Scarcity of 3D data**: Neural parametric models (NPMs) have achieved strong results in human body modeling (e.g., SMPL), but require large amounts of 3D training data, whereas 3D deformation datasets for leaves are virtually nonexistent.

**High shape diversity**: Leaf shapes vary dramatically across species, making unified modeling via traditional PCA models or Bézier curves difficult.

**High deformation degrees of freedom**: Leaf deformations are highly flexible and lack the articulated structure of the human body, making it difficult to define a universal skeleton.

**Limited existing parametric models**: PCA-based models (Barley) have limited expressiveness; Bézier curve models (Gaurav) cannot capture natural deformations.

Key insight: **A flattened leaf approximates a 2D plane** — a biological property that can be exploited to disentangle shape from deformation.

## Method

### Overall Architecture

NeuraLeaf comprises three disentangled spaces:
- **Shape space**: Represents the flattened leaf shape via a 2D SDF (learned from large-scale 2D datasets).
- **Texture space**: Generates shape-aligned textures via CycleGAN.
- **Deformation space**: Represents 3D deformations via a skeleton-free skinning model (learned from the DeformLeaf dataset).

### Key Design 1: 2D Base Shape Space

A neural SDF conditioned on a shape latent code $\mathbf{z}_s$ represents the flattened leaf:

$$\mathcal{S}_b^* = \{\mathbf{x} \in \mathbb{R}^2 \mid f_{\theta_s}(\mathbf{x}, \mathbf{z}_s) = 0\}$$

The SDF is converted to a mask $\mathbf{I}_m$ via a differentiable sigmoid, and a base mesh is constructed by thresholding. Training uses 2D leaf scan datasets, with the following loss:

$$\text{argmin}_{\theta_s, k, \{\mathbf{z}_s^i\}} \sum_i (\mathcal{L}_{\text{sdf}} + \mathcal{L}_{\text{sil}} + \mathcal{L}_{\text{eik}} + \mathcal{L}_{\text{lat}})$$

Advantage: The 2D representation can be learned from richly available biological 2D leaf scan datasets (e.g., Flavia).

### Key Design 2: Skeleton-Free Skinning Model

Unlike skeleton-based skinning in SMPL, NeuraLeaf employs a **skeleton-free skinning** approach with a large number of control points ($K=1000$):

$$\tilde{\mathbf{v}}_i = \sum_{k=1}^K w_{i,k} \mathbf{T}_k (\tilde{\mathbf{v}}_i' - \tilde{\mathbf{c}}_k)$$

Two decoders are used in a disentangled manner:
- **Transformation decoder** $f_{\theta_d}$: Predicts rigid-body transformations (quaternions) for each control point from the deformation latent code $\mathbf{z}_d$ only, independent of shape.
- **Skinning weight decoder** $f_{\theta_w}$: Predicts skinning weights from the shape latent code $\mathbf{z}_s$ and vertex positions, adapting to each leaf shape.

### Key Design 3: Deformation Mapping Loss

This loss constrains the distribution of the deformation latent space — samples with larger deformations are mapped farther from the origin:

$$\mathcal{L}_{map} = \left(\frac{d_{cham}(\mathcal{S}_b, \mathcal{S}_d)}{\|\mathbf{z}_d\|_2} - \phi\right)^2$$

where $\phi$ is a learnable scaling factor. This ensures an orderly structure in the latent space.

### DeformLeaf Dataset

Approximately 300 base–deformed leaf pairs, with 3D deformed shapes acquired via multi-view reconstruction and 2D base shapes captured by flatbed scanning. The dataset includes fine geometric details such as leaf venation and is the first 3D dataset dedicated to leaf deformation modeling.

## Experiments

### Main Results: 3D Leaf Reconstruction (Fitting to Observations)

| Method | Chamfer Distance↓ | Surface Quality |
|------|-----------------|---------|
| PCA Parametric Model | High | Lacks detail |
| DeepSDF (NPM baseline) | Medium | Lacks venation |
| **NeuraLeaf** | **Lowest** | **Clear venation** |

NeuraLeaf significantly outperforms traditional parametric models and general-purpose NPMs in reconstruction accuracy while preserving surface details such as leaf venation.

### Ablation Study: Contribution of Disentanglement Design

| Deformation Mapping Loss | Two-Stage Training | Boundary Constraint | Reconstruction Quality |
|-----------|---------|---------|---------|
| ✗ | ✗ | ✗ | Baseline |
| ✓ | ✗ | ✗ | Improved |
| ✓ | ✓ | ✗ | Further improved |
| ✓ | ✓ | ✓ | **Best** |

The deformation mapping loss is critical for imposing an orderly structure on the latent space.

### Key Findings
- Disentangling shape and deformation enables effective learning of the deformation space from only ~300 3D samples.
- The 1,000 control points far exceed the number of bones in human body models, yet are essential for capturing fine-grained leaf deformations.
- The deformation space generalizes to leaf shapes unseen during training (via second-stage training).
- The texture space naturally produces shape-aligned textures through CycleGAN.

## Highlights & Insights

1. **Clever use of domain priors**: The biological property that a flattened leaf approximates a 2D plane enables effective disentanglement.
2. **Skeleton-free skinning**: Removes dependence on predefined skeletal structures, accommodating the highly flexible deformations of leaves.
3. **Cross-dataset learning**: The shape space is learned from large-scale 2D datasets, while the deformation space is learned from a small-scale 3D dataset.
4. **First leaf deformation dataset**: DeformLeaf provides a foundational research resource for the field.

## Limitations & Future Work
- The DeformLeaf dataset is relatively small in scale (~300 pairs).
- Dynamic deformation processes (e.g., curling due to dehydration) cannot be modeled.
- The deformation mapping assumes a zero-mean Gaussian distribution, which may be unsuitable for extreme deformations.
- The current framework supports single-leaf modeling only, without considering whole-plant context.

## Related Work & Insights
- SMPL / SMAL: Parametric models for human bodies and animals.
- DeepSDF: Neural implicit shape representation.
- PhysGaussian / Leaf deformation modeling: Physics-driven leaf deformation.
- Bio-inspired leaf modeling: Leaf generation based on venation patterns.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First NPM for leaves; both the disentanglement design and skeleton-free skinning are novel contributions.
- **Practicality**: ⭐⭐⭐⭐ — High application value in agriculture and computer graphics.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive fitting, generation, and ablation experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Method is clearly presented; the dataset is a valuable contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Demeter: A Parametric Model of Crop Plant Morphology from the Real World](demeter_a_parametric_model_of_crop_plant_morphology_from_the_real_world.md)
- [\[ICCV 2025\] Unleashing Vecset Diffusion Model for Fast Shape Generation (FlashVDM)](unleashing_vecset_diffusion_model_for_fast_shape_generation.md)
- [\[ICCV 2025\] Repurposing 2D Diffusion Models with Gaussian Atlas for 3D Generation](repurposing_2d_diffusion_models_with_gaussian_atlas_for_3d_generation.md)
- [\[ICCV 2025\] SL2A-INR: Single-Layer Learnable Activation for Implicit Neural Representation](sl2a-inr_single-layer_learnable_activation_for_implicit_neural_representation.md)
- [\[ICCV 2025\] Tune-Your-Style: Intensity-Tunable 3D Style Transfer with Gaussian Splatting](tune-your-style_intensity-tunable_3d_style_transfer_with_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
