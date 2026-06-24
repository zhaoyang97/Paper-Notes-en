---
title: >-
  [Paper Note] ShapeFusion: A 3D Diffusion Model for Localized Shape Editing
description: >-
  [ECCV 2024][3D Vision] Proposes ShapeFusion, a 3D mesh localized editing method based on a masked diffusion training strategy, achieving fully localized and interpretable 3D shape editing by directly operating in vertex space without latent-space optimization.
tags:
  - "ECCV 2024"
  - "3D Vision"
date: 2026-05-08
content_hash: 0cd288d7c9f3023b
---

# ShapeFusion: A 3D Diffusion Model for Localized Shape Editing

**Conference**: ECCV 2024  
**arXiv**: [2403.19773](https://arxiv.org/abs/2403.19773)  
**Area**: Image Generation

## TL;DR

Proposes ShapeFusion, a 3D mesh localized editing method based on a masked diffusion training strategy, achieving fully localized and interpretable 3D shape editing by directly operating in vertex space without latent-space optimization.

## Background & Motivation

- Parametric 3D models (3DMMs) are widely used in digital humans, gaming, and virtual reality, where traditional methods perform global shape modeling based on PCA.
- **Limitations of Prior Work**: The orthogonality constraint and global decomposition characteristics of PCA prevent localized, decoupled 3D shape editing, meaning that editing one region inevitably affects other areas.
- Existing methods (such as SD, LED) attempt to achieve decoupling in the latent space, but factorization in the latent space cannot guarantee locality in the 3D space and leads to degraded reconstruction performance.
- Global parametric models lack interpretability, making it difficult to find latent codes that control specific regional features.

## Method

### Overall Architecture

ShapeFusion formulates localized shape modeling as an inpainting problem, using a masked training strategy so that the diffusion process acts only locally within the masked region. The framework comprises two main components:

1. **Forward Diffusion Process**: Gradually adds noise to designated regions of the input mesh.
2. **Denoising Module**: Predicts the denoised version of the added noise.

### Key Designs

**1. Masked Forward Diffusion**

- Defines a binary mask $\mathbf{M} \in \mathbb{R}^{N \times 3}$ to specify the noise addition region.
- During training, the masked region is the $k$-hop geodesic neighborhood of a randomly selected anchor point $\mathbf{x}_a$.
- Unmasked regions (including the anchor point) remain unchanged, naturally ensuring editing locality by design.

**2. Hierarchical Denoising Module based on Mesh Convolution**

- Introduces **vertex index positional encoding** $\mathbf{p}_i$ to break permutation equivariance, enabling the network to learn vertex-specific priors.
- Employs **three-tier hierarchical mesh convolution**: performs message passing on meshes of different resolutions to recursively update features from coarse to fine.
- Utilizes spiral mesh convolution (Spiral Convolution) to define neighborhoods, supporting topology-preserving generation.

**3. Feature Initialization**

The initial features of each vertex $i$ are concatenated as:
$$\mathbf{f}_i^{(0)} = [\mathbf{x}_i \| \mathbf{m}_i \| \mathbf{p}_i \| \mathbf{c}_t]$$

where $\mathbf{x}_i$ represents 3D coordinates, $\mathbf{m}_i$ is the binary mask, $\mathbf{p}_i$ denotes positional encoding, and $\mathbf{c}_t$ is the timestep embedding.

### Loss & Training

Standard diffusion model denoising loss (reparameterized form) is adopted:

$$\mathcal{L}_t = \|\epsilon_t - \epsilon_\theta(\mathbf{x}, t, \mathbf{M})\|_2$$

where $\epsilon_t$ represents the noise at the $t$-th step in the forward diffusion process, and $\mathbf{M}$ is the mask defining the editing region.

## Key Experimental Results

### Main Results

Quantitative comparisons are conducted with baseline methods on three datasets (MimicMe, UHM, and STAR) to evaluate diversity (DIV), FID, and identity preservation (ID):

| Method | MimicMe DIV↑ | MimicMe FID↓ | MimicMe ID↓ | UHM DIV↑ | UHM FID↓ | UHM ID↓ | STAR DIV↑ | STAR FID↓ | STAR ID↓ |
|------|-------------|-------------|------------|---------|---------|--------|----------|---------|--------|
| M-VAE | 0.25 | 1.21 | 0.09 | 0.61 | 1.17 | 0.21 | 0.72 | 0.71 | 0.19 |
| SD | 0.24 | 7.81 | 0.84 | 0.53 | 8.04 | 0.36 | 0.65 | 6.94 | 0.34 |
| LED | 0.10 | 3.39 | 0.23 | 0.43 | 2.30 | 0.58 | 0.47 | 2.04 | 0.56 |
| **ShapeFusion** | **0.34** | **0.30** | **0.05** | **0.71** | **0.53** | **0.11** | **0.98** | **0.43** | **0.09** |

### Ablation Study

The impact of different numbers of anchor points on reconstruction performance is evaluated on the UHM test set:

| Number of Anchors | Reconstruction Error (mm) | Comparison with PCA | Comparison with SD |
|---------|-------------|--------|-------|
| 50 | ~1.2 | Outperforms | Outperforms |
| 100 | ~0.7 | Outperforms | Outperforms |
| 200 | 0.38 | Outperforms PCA and SD | Outperforms SD |
| 500 | ~0.15 | Significantly Outperforms | Significantly Outperforms |

Furthermore, ShapeFusion achieves an inference speed of approximately 3.2 seconds, whereas baseline methods (SD, LED) require about 22 seconds (based on optimization-based fitting processes), yielding a speedup of roughly 10x.

### Key Findings

1. **Fully Localized Editing**: Heatmaps confirm that ShapeFusion is the only method that guarantees fully localized editing without affecting other regions.
2. **High Diversity Generation**: Although M-VAE can perform local editing, the generated regions are nearly identical (as autoencoders tend to reconstruct the input), whereas ShapeFusion can produce a rich variety of diverse variations.
3. **Direct Point Manipulation**: Eliminating the optimization process, localized deformed meshes can be directly generated by simply setting the anchor point positions.
4. **Expression Editing Generalization**: Generalizes successfully to out-of-distribution expressions (e.g., smirk) not present in the training data.

## Highlights & Insights

- Formulating 3D localized editing as an inpainting problem is highly elegant, bypassing the difficulties of forcing decoupling in latent space.
- The design of the masked diffusion training strategy is straightforward yet effective, ensuring locality inherently through the architectural design.
- Hierarchical mesh convolution addresses two key challenges: long-range dependencies and boundary smoothing.
- The region swapping capability has direct practical value in fields like aesthetic medicine.
- When utilized as a self-decoder, it achieves a reconstruction accuracy of 0.38mm with only 200 anchors, demonstrating strong shape prior learning capabilities.

## Limitations & Future Work

- Currently, the evaluation is primarily conducted on fixed-topology meshes; generalization to non-fixed topologies requires further investigation.
- Operating in the posed space is unnecessary and could be replaced with a pose normalization step.
- The capability for extremely fine-grained local editing (e.g., wrinkle-level) is not fully demonstrated.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying the masked diffusion strategy to 3D localized editing is novel and natural.
- **Value**: ⭐⭐⭐⭐ — Direct point manipulation and region swapping capabilities have practical utility for 3D modelers and the medical industry.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive quantitative and qualitative evaluation across three datasets and multiple application scenarios.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic and rich illustrations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [\[ECCV 2024\] UniDream: Unifying Diffusion Priors for Relightable Text-to-3D Generation](unidream_unifying_diffusion_priors_for_relightable_text-to-3d_generation.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)

</div>

<!-- RELATED:END -->
