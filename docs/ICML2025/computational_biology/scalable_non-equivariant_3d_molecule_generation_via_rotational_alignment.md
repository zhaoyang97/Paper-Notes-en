---
title: >-
  [Paper Note] Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment
description: >-
  [Computational Biology] Proposes **RADM (Rotationally Aligned Diffusion Model)**, which constructs an aligned latent space by learning sample-dependent SO(3) rotational transformations, enabling non-equivariant diffusion models to generate 3D molecules effectively, achieving generation quality comparable to SOTA equivariant models while offering better scalability and sampling efficiency.
tags:
  - "Computational Biology"
date: 2026-05-08
content_hash: d75b2a57c5dd32d4
---

# Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment

> **Conference**: ICML 2025
> **arXiv**: [2506.10186](https://arxiv.org/abs/2506.10186)
> **Code**: [GitHub](https://github.com/skeletondyh/RADM)
> **Area**: Molecule Generation / Diffusion Models
> **Keywords**: 3D Molecule Generation, Non-Equivariant, Rotational Alignment, Latent Diffusion, AutoEncoder

## TL;DR

Proposes **RADM (Rotationally Aligned Diffusion Model)**, which constructs an aligned latent space by learning sample-dependent SO(3) rotational transformations, enabling non-equivariant diffusion models to generate 3D molecules effectively, achieving generation quality comparable to SOTA equivariant models while offering better scalability and sampling efficiency.

## Background & Motivation

In 3D molecule generation, rotating a molecule in three-dimensional space does not change its chemical properties (SE(3) symmetry). Mainstream methods satisfy this constraint through equivariant networks (such as EGNN):

$$p(\mathbf{x}) = p(\mathbf{R}\mathbf{x}) \quad \forall \mathbf{R} \in \text{SO}(3)$$

But equivariant architectures have clear disadvantages:

**Complex Parameterization**: EGNN and similar models require special message passing rules to maintain equivariance.

**Lack of Standard Implementations**: Unlike the unified status of Transformers in vision/NLP.

**Poor Efficiency and Scalability**: Difficult to leverage modern acceleration techniques such as FlashAttention.

Key Challenge: **Is equivariance strictly necessary?** The probability of a molecule is determined by the total probability of all its possible 3D positions, rather than requiring equal probability at every single position.

## Method

### Overall Architecture

Trained in two stages: (1) Train an autoencoder with rotational alignment $\rightarrow$ construct an aligned latent space; (2) Train a non-equivariant diffusion model in the aligned latent space.

### Key Designs

**1. Rotation Parameterization**

Project an arbitrary matrix $\mathbf{M} \in \mathbb{R}^{3 \times 3}$ to SO(3) via SVD:

$$\mathbf{R} = \text{SVD}^+(\mathbf{M}) = \mathbf{U}\text{diag}(1, 1, \det(\mathbf{U}\mathbf{V}^\top))\mathbf{V}^\top$$

This parameterization is smooth when $\det(\mathbf{M}) \neq 0$, making it suitable for gradient optimization.

**2. Rotation Network**

A vanilla GNN (non-equivariant) is used to generate a sample-dependent rotation matrix $\mathbf{R}_\theta$ from the molecule $(\mathbf{x}, \mathbf{h})$. Atomic coordinates and features are concatenated, passed through message passing, and finally mean-pooled and processed by a 2-layer MLP to obtain $\mathbf{M}$.

**3. Non-Equivariant Autoencoder**

- **Encoder**: 1-layer EGNN (same as GeoLDM for easy ablation)
- **Decoder**: Non-equivariant GNN — **Key Design**: The decoder must be non-equivariant so that the reconstruction loss is sensitive to rotation, thereby providing gradient signals for the rotation network.

Reconstruction loss:
$$\mathcal{L} = -\mathbb{E}_{q_{\theta,\eta}(\mathbf{z}_x, \mathbf{z}_h | \mathbf{x}, \mathbf{h})}[\log p_\psi(\mathbf{R}_\theta\mathbf{x}, \mathbf{h} | \mathbf{z}_x, \mathbf{z}_h)]$$

**4. Non-Equivariant Latent Diffusion Model**

A standard denoising diffusion model is trained in the aligned latent space. The noise prediction network can use:
- Vanilla GNN (concatenating coordinates and features)
- **DiT (Diffusion Transformer)**: Directly reuse high-performance Transformer implementations from the vision field.

Training objective (standard DDPM):

$$\mathcal{L}(\mathbf{x}) = \mathbb{E}_{\boldsymbol{\epsilon}, t}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\phi\|^2]$$

**5. Translation Handling**

Coordinates are projected into an $(N-1) \times 3$-dimensional subspace by subtracting the center of mass. The center of mass of the predicted noise is also subtracted at each diffusion step.

## Key Experimental Results

### QM9 Dataset

| Model | Atom Sta (%) | Mol Sta (%) | Valid (%) | Valid & Unique (%) |
|------|-------------|------------|-----------|-------------------|
| EDM (Equivariant) | 98.7 | 82.0 | 91.9 | 90.7 |
| GeoLDM (Equivariant) | 98.9 | 89.4 | 93.8 | 92.7 |
| GDM-aug (Non-Equivariant) | 97.6 | 71.6 | 90.4 | 89.5 |
| **RADM (Non-Equivariant)** | **~98.8** | **~87** | **~93** | **~92** |

### GEOM-Drugs Dataset

| Model | Atom Sta (%) | Valid (%) |
|------|-------------|-----------|
| EDM | 81.3 | 92.6 |
| GeoLDM | 84.4 | 99.3 |
| GDM-aug | 77.7 | 91.8 |
| **RADM** | Large improvement over GDM-aug | - |

### Efficiency Comparison

- RADM sampling speed is significantly faster than equivariant diffusion models.
- Using DiT as the denoising network allows leveraging FlashAttention acceleration.
- Non-equivariant architectures are more parameter-efficient.

### Key Findings

- Non-equivariant RADM significantly outperforms all prior non-equivariant methods (GDM, GDM-aug, GraphLDM).
- Generation quality is close to SOTA equivariant models (GeoLDM).
- Rotational alignment is crucial: ablation studies prove that performance drops sharply when the rotation network is removed.
- Non-equivariant decoders are necessary (equivariant decoders would make the reconstruction loss invariant to rotation, preventing the rotation network from learning).

## Highlights & Insights

1. **Revisiting the necessity of equivariance**: Probabilistically, equivariance constraints are not strictly necessary, breaking the community's inertia.
2. **Inspiration for rotational alignment**: 3D vision datasets (like ShapeNet) are aligned; why not molecules?
3. **Autoencoder learns unsupervised alignment**: Cleverly uses the reconstruction objective to indirectly supervise the rotation network to align molecules.
4. **Potential for unified architectures**: Non-equivariant models can directly utilize general-purpose architectures like DiT, bridging molecule generation and vision generation.
5. **SVD Rotation Parameterization**: Smooth and unconstrained, suitable for end-to-end gradient learning.

## Limitations & Future Work

1. The autoencoder and diffusion model are trained separately, which might not achieve joint optimality.
2. The encoder still uses EGNN (equivariant), rendering the framework not entirely non-equivariant.
3. Only validated on small molecule datasets (QM9, GEOM-Drugs), without testing on macromolecules such as proteins.
4. Rotational alignment only handles SO(3); permutation equivariance is still guaranteed by attention mechanisms.
5. Latent space dimensions are the same as the original space, meaning real dimensional compression is not achieved.

## Related Work & Insights

- Equivariant Diffusion: EDM, GeoLDM, MiDi
- Non-Equivariant Methods: GDM, GDM-aug, GraphLDM
- Rotation Representation: SVD Parameterization, Euler Angles, Exponential Coordinates
- Latent Diffusion: LDM, GeoLDM

## Rating

⭐⭐⭐⭐ (4/5)

The argument is clear and powerful—equivariance is not mandatory. The design of the rotationally aligned autoencoder is elegant, opening up a path to connect molecular generation with general generative architectures. The experiments are thorough but limited in scale (only small molecules).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MolParser: End-to-end Visual Recognition of Molecule Structures in the Wild](../../ICCV2025/computational_biology/molparser_end-to-end_visual_recognition_of_molecule_structures_in_the_wild.md)
- [\[ICML 2025\] SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics](stofm_a_multi-scale_foundation_model_for_spatial_transcriptomics.md)
- [\[ICML 2025\] Protein Structure Tokenization: Benchmarking and New Recipe](protein_structure_tokenization_benchmarking_and_new_recipe.md)
- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](../../ICLR2026/computational_biology/thompson_sampling_via_fine-tuning_of_llms.md)
- [\[ICML 2025\] Geometric Representation Condition Improves Equivariant Molecule Generation](geometric_representation_condition_improves_equivariant_molecule_generation.md)

</div>

<!-- RELATED:END -->
