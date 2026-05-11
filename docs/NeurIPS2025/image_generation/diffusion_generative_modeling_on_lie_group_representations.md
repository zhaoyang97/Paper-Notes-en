---
title: >-
  [Paper Note] Diffusion Generative Modeling on Lie Group Representations
description: >-
  [NeurIPS 2025][Image Generation][Lie group representations] This paper proposes a novel theoretical framework for constructing diffusion processes on the **representation space** of Lie groups (rather than on the Lie gro…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Lie group representations"
  - "generalized score matching"
  - "stochastic differential equations"
  - "molecular conformation generation"
  - "manifold diffusion"
date: 2026-05-08
content_hash: a28b1106b9c8dc29
---

# Diffusion Generative Modeling on Lie Group Representations

**Conference**: NeurIPS 2025
**arXiv**: [2502.02513](https://arxiv.org/abs/2502.02513)
**Code**: None
**Area**: Image Generation / Diffusion Models / Lie Groups
**Keywords**: Lie group representations, generalized score matching, stochastic differential equations, molecular conformation generation, manifold diffusion

## TL;DR

This paper proposes a novel theoretical framework for constructing diffusion processes on the **representation space** of Lie groups (rather than on the Lie groups themselves). By mapping the curved dynamics of non-Abelian Lie groups into Euclidean space via generalized score matching, the framework enables simulation-free training of Lie group diffusion models, and demonstrates that standard score matching is a special case corresponding to the translation group.

## Background & Motivation

- **Problem**: Many scientific data types (molecular conformations, rotations, rigid-body transformations) are naturally distributed on curved manifolds (e.g., SO(3), SE(3)) rather than Euclidean space.
- **Existing Difficulties**: Diffusion algorithms on manifolds face two major challenges: (1) parameterizing vector fields on general manifolds remains unsolved; (2) Langevin updates require projection back onto the manifold to preserve geometric structure.
- **Practical Gap**: Even when data symmetry is well-defined (e.g., the torsion angle space of proteins), AlphaFold3 still performs diffusion in Cartesian coordinates, as manifold diffusion offers no clear performance advantage.
- **Core Problem**: Can the flat-space advantages of Euclidean geometry be preserved while exploiting the symmetry structure of Lie groups? This paper answers affirmatively by constructing diffusion on the **representation space** $\text{Im}(\rho_X) \subseteq GL(X)$.

## Method

### Overall Architecture

Rather than performing diffusion directly on the Lie group $G$ or on the data space $X$, the framework leverages the action representation $\rho_X: G \to GL(X)$ of $G$ on $X$:

1. Parameterize data points $\mathbf{x} \in X$ via flow coordinates $\boldsymbol{\tau}$ (i.e., identify Lie group transformation parameters).
2. Define the forward diffusion in flow coordinate space (Euclidean, amenable to Gaussian noise).
3. Map flow coordinates back to data space via the Lie group action.
4. Train a network to predict the **generalized score** (the derivative of the log-density along fundamental vector field directions).
5. The reverse SDE is guided by the generalized score to generate samples in data space $X$.

### Key Designs

#### 1. Generalized Score Matching

The standard score function $\nabla \log p(\mathbf{x})$ is replaced by derivatives along the fundamental vector fields of the Lie algebra:

$$\mathcal{L} = \mathbf{\Pi}(\mathbf{x})^\top \nabla$$

where $\mathbf{\Pi}(\mathbf{x})$ is the fundamental vector field matrix. The generalized Fisher divergence is:

$$D_{\mathcal{L}}(p||q_\theta) = \int_X p(\mathbf{x})|\mathcal{L}\log p(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x})|^2 d\mathbf{x}$$

#### 2. Three Sufficient Conditions

For generalized score matching to be applicable to Langevin dynamics, three conditions must be satisfied:

| Condition | Meaning | Mathematical Requirement |
|-----------|---------|--------------------------|
| Completeness | $\mathbf{\Pi}$ retains all information about the density | $\dim(G/G_\mathbf{x}) \geq \dim(X)$ almost everywhere |
| Homogeneity | Any two points are connected by a $G$-transformation | $X$ is a homogeneous space of $G$ |
| Commutativity | Langevin updates in each direction are mutually independent | $[\mathcal{L}_A, \mathcal{L}_B]f(\mathbf{x}) = 0$ |

A key insight is that **non-Abelian groups can also satisfy the commutativity condition**: elements that do not commute in the Lie algebra may induce commuting flows on $X$ (e.g., $\mathfrak{so}(3)$ under certain representations).

#### 3. Core Theorem: Paired SDEs

**Theorem 3.1**: There exists a pair of SDEs in which the forward process admits an exact solution and the reverse process is guided by the generalized score:

**Forward SDE**:
$$d\mathbf{x} = \left[\beta(t)\mathbf{\Pi}(\mathbf{x})\mathbf{f}(\mathbf{x}) + \frac{\gamma(t)^2}{2}\rho_X(\Omega)\right]dt + \gamma(t)\mathbf{\Pi}(\mathbf{x})d\mathbf{W}$$

**Exact Solution**: $\mathbf{x}(t) = \prod_{i=1}^n e^{\tau_i(t)A_i} \mathbf{x}(0)$, where $\boldsymbol{\tau}(t)$ follows a simple SDE.

**Reverse SDE** contains three additional terms:
- **Quadratic Casimir element** $\rho_X(\Omega) = \sum_i A_i^2$: compensates for orbital deviation induced by curvature of the flow coordinates.
- **Divergence correction** $\mathbf{\Pi}\nabla^\top \cdot \mathbf{\Pi}$: corrects the probability flow of the non-constant-coefficient SDE.
- **Generalized score** $\mathbf{\Pi}\mathcal{L}\log p_t$: guides sampling.

#### 4. Standard Score Matching as a Special Case

When $G = T(n)$ (the translation group), $\mathbf{\Pi} = I$, and the Casimir and divergence correction terms vanish, recovering standard DDPM.

#### 5. $SO(2) \times \mathbb{R}^+$ Example

Taking $G = SO(2) \times \mathbb{R}_+$ acting on $X = \mathbb{R}^2$:
- Fundamental vector field matrix: $\mathbf{\Pi}(\mathbf{x}) = \begin{pmatrix} x & -y \\ y & x \end{pmatrix}$
- The SDE decomposes along radial (scaling) and angular (rotation) directions.
- The asymptotic distribution $p_T$ takes the form $(e^{\eta_r}\cos\eta_\theta, e^{\eta_r}\sin\eta_\theta)$, obtainable by sampling two Gaussian variables.

### Loss & Training

- The training objective follows the standard denoising score matching formulation, but targets the generalized score rather than the standard score:

$$\mathbb{E}_t\left\{w(t)\mathbb{E}_{\mathbf{x}(0)}\mathbb{E}_{\mathbf{x}(t)}\left[|\mathbf{s}_\theta(\mathbf{x}(t),t) - \mathcal{L}\log p_t(\mathbf{x}(t)|\mathbf{x}(0))|^2\right]\right\}$$

- Since the conditional generalized score admits a closed-form solution $-\Sigma(t)^{-1}\boldsymbol{\eta}_t$ under Gaussian assumptions, training reduces to noise prediction (analogous to DDPM).
- A variance-preserving scheduler is used; training and sampling algorithms are given in Algorithms 1 and 2.

## Key Experimental Results

### Main Results

#### 2D/3D/4D Distribution Generation

- Using $G = SO(d) \times \mathbb{R}_+$ on $X = \mathbb{R}^d$.
- Successfully models mixture-of-Gaussians, tori, Möbius strips, and other distributions.
- Symmetry exploitation reduces the learning dimensionality (e.g., only the radial score needs to be learned for radially symmetric distributions).

#### Rotated MNIST Bridging

| Model | Mean Accuracy ↑ | Mean FID ↓ |
|-------|----------------|-----------|
| **GSM (Ours)** | **0.96 ± 0.02** | **85.8 ± 15.7** |
| BBDM | 0.80 ± 0.10 | 133.4 ± 19.0 |

- Only a 1-dimensional score (rotation angle) is learned, substantially simplifying the problem.
- BBDM operates in the full pixel space and may generate incorrect digit identities.

#### QM9 Molecular Conformation Generation

- $G = (SO(3) \times \mathbb{R}_+)^N$ acting on $X = \mathbb{R}^{3N}$.
- UFF energies of generated conformations are close to those of ground-truth conformations and are marginally lower.
- Lie group diffusion: $\Delta_\theta = -0.2159$ vs. standard diffusion: $\Delta_\gamma = -0.2144$.

#### CrossDocked2020 Molecular Docking

| Method | RMSD (Å) ↓ |
|--------|-----------|
| **GSM (Ours, SE(3))** | **2.91 ± 1.0** |
| RSGM (Riemannian diffusion) | 5.6 ± 1.2 |
| BBDM (Euclidean) | 2.92 ± 1.57 |

### Ablation Study

- In W2 distance comparisons on 2D mixture-of-Gaussians, Lie group diffusion and standard diffusion exhibit equivalent expressive capacity (both can model arbitrary distributions).
- However, when the chosen Lie group symmetry matches the data structure, the effective dimensionality is reduced and learning efficiency improves.

### Key Findings

1. An appropriate choice of Lie group can **reduce the effective dimensionality** (from 784 dimensions in rotated MNIST to 1 dimension).
2. The framework supports **bridging between non-trivial distributions** (which BBDM cannot achieve).
3. SE(3)-guided molecular docking substantially outperforms Riemannian diffusion methods.
4. The framework extends naturally to flow matching (Appendix F).

## Highlights & Insights

- **Theoretical Unification**: Standard diffusion (translation group), Riemannian diffusion (diffusion on the group), and the proposed method (diffusion in representation space) are all instances of the same unified framework.
- **Geometric Intuition of the Casimir Element**: It compensates for orbital deviation caused by curvature — in SO(2), tangential motion causes a point to drift off its circular orbit, and the Casimir element restores it.
- **Dimensionality Compression**: When the data structure aligns with the group symmetry, the effective dimensionality of score learning can be substantially reduced.
- **Simulation-Free Training**: The exact solution of the forward SDE avoids the numerical simulation difficulties associated with diffusion on non-Abelian groups.

## Limitations & Future Work

1. The commutativity condition restricts applicable group–space combinations, requiring careful selection.
2. Molecular docking experiments are conducted at a relatively small scale (no direct comparison with methods such as DiffDock).
3. Computational efficiency for high-dimensional groups (e.g., SO(n) for large $n$) remains to be verified.
4. Current validation is limited to common groups such as SO(2), SO(3), and SE(3); more complex symmetry groups warrant further exploration.
5. Generalization to non-homogeneous spaces (restricting generation within orbits) is theoretically supported but lacks experimental validation.

## Related Work & Insights

- **De Bortoli et al. (2022)**: Diffusion on Riemannian manifolds; the dynamics are formally similar to the proposed method in the Lie group setting, but require numerical simulation.
- **Corso et al. (2023) DiffDock**: SE(3) diffusion for molecular docking, operating directly on the group.
- **Kim et al. (2022)**: Linearization of nonlinear problems via bijective mappings, conceptually related to the representation space perspective of this work.
- **Insight**: Selecting the appropriate symmetry group can reduce complex problems to lower-dimensional ones — a principle with broad applicability across scientific computing.

## Rating

| Dimension | Score | Comment |
|-----------|-------|---------|
| Novelty | ★★★★★ | Entirely new theoretical framework unifying multiple diffusion paradigms with mathematical elegance |
| Technical Depth | ★★★★★ | Rigorous mathematical derivations; exact solutions for a novel class of SDEs are established |
| Experimental Thoroughness | ★★★☆☆ | Experiments are limited in scale; comparison with state-of-the-art methods is insufficient |
| Practical Value | ★★★★☆ | Promising applications in molecular conformation and docking, though engineering implementation is non-trivial |
| Writing Quality | ★★★★☆ | Mathematically rigorous but demanding to read; requires a solid background in differential geometry |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Coupling Generative Modeling and an Autoencoder with the Causal Bridge](coupling_generative_modeling_and_an_autoencoder_with_the_causal_bridge.md)
- [\[NeurIPS 2025\] CORAL: Disentangling Latent Representations in Long-Tailed Diffusion](coral_disentangling_latent_representations_in_longtailed_dif.md)
- [\[ICCV 2025\] SummDiff: Generative Modeling of Video Summarization with Diffusion](../../ICCV2025/image_generation/summdiff_generative_modeling_of_video_summarization_with_diffusion.md)
- [\[NeurIPS 2025\] Hephaestus: Mixture Generative Modeling with Energy Guidance for Large-scale QoS Degradation](hephaestus_mixture_generative_modeling_with_energy_guidance_for_large-scale_qos_.md)
- [\[NeurIPS 2025\] Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification](latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)

</div>

<!-- RELATED:END -->
