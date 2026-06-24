---
title: >-
  [Paper Note] WGFormer: An SE(3)-Transformer Driven by Wasserstein Gradient Flows for Molecular Generation
description: >-
  [ICML 2025][Computational Biology][molecular conformation] This paper proposes WGFormer, an SE(3)-Transformer driven by Wasserstein gradient flows. Operating within an autoencoder framework, WGFormer optimizes molecular conformations by minimizing energy functions on latent mixture models of atoms, consistently outperforming the state-of-the-art (SOTA) on ground-state conformation prediction tasks.
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "molecular conformation"
  - "SE(3)-Transformer"
  - "Wasserstein gradient flow"
  - "ground-state conformation"
  - "autoencoder"
date: 2026-05-08
content_hash: aa3179fbaeeffb34
---

# WGFormer: An SE(3)-Transformer Driven by Wasserstein Gradient Flows for Molecular Generation

**Conference**: ICML 2025  
**arXiv**: [2410.09795](https://arxiv.org/abs/2410.09795)  
**Code**: None  
**Area**: Molecular Generation / Geometric Deep Learning  
**Keywords**: molecular conformation, SE(3)-Transformer, Wasserstein gradient flow, ground-state conformation, autoencoder

## TL;DR
This paper proposes WGFormer, an SE(3)-Transformer driven by Wasserstein gradient flows. Operating within an autoencoder framework, WGFormer optimizes molecular conformations by minimizing energy functions on latent mixture models of atoms, consistently outperforming the state-of-the-art (SOTA) on ground-state conformation prediction tasks.

## Background & Motivation
**Background**: Predicting ground-state molecular conformations (energy-minimized conformations) is crucial for molecular docking and property prediction. Classical methods like density functional theory (DFT) and molecular mechanics simulations offer high precision but are extremely slow computationally. Recently, learning-based methods (e.g., GeoMol, ConfGF) have shown efficiency advantages but still suffer from limitations in accuracy and interpretability.

**Limitations of Prior Work**: Existing learning methods either directly regress coordinates (lacking physical constraints) or generate conformations using diffusion models (slow sampling, without directly optimizing physical energy). They lack explicit modeling of the physical energy landscape, which often causes the generated conformations to deviate from the energy minima.

**Key Challenge**: The trade-off between efficiency and accuracy/physical plausibility. Traditional simulations guarantee physical plausibility but at a high computational cost, whereas learning-based approaches are fast but lack physical guarantees.

**Goal**: To bridge energy-based simulations and learning-based strategies by designing an efficient and physically interpretable method.

**Key Insight**: Embedding Wasserstein gradient flows—an energy minimization dynamics on the space of probability distributions—directly into the architecture of an SE(3)-Transformer.

**Core Idea**: Each layer of WGFormer corresponds to a single step of the Wasserstein gradient flow that minimizes an energy function over a latent mixture model of atoms, thereby encoding the physical optimization process directly into the network architecture.

## Method

### Overall Architecture
Input: Low-quality initial conformation (such as MMFF conformations optimized by force fields) + molecular graph  
Output: 3D coordinates of the ground-state (energy-minimized) conformation  

Autoencoder framework:
1. **Encoder** (WGFormer): Encodes low-quality conformations into optimized latent representations.
2. **Decoder** (MLP): Decodes the latent representations into ground-state conformation coordinates.

### Key Designs

1. **SE(3)-Transformer Driven by Wasserstein Gradient Flows**:

    - **Function**: Designing Transformer layers such that the update in each layer corresponds to a gradient descent step of the energy function in the Wasserstein space.
    - **Mechanism**: Modeling atoms as samples of a probability distribution (a latent Gaussian mixture model $\rho = \sum_i w_i \mathcal{N}(\mu_i, \Sigma_i)$) and defining an energy function $\mathcal{E}[\rho]$. The Wasserstein gradient flow is formulated as:
    $\partial_t \rho = \nabla \cdot (\rho \nabla \frac{\delta \mathcal{E}}{\delta \rho})$
      After discretization, each Transformer layer updates the positions and features of atoms:
    $\mu_i^{(l+1)} = \mu_i^{(l)} - \eta \sum_j \alpha_{ij} (\mu_i^{(l)} - \mu_j^{(l)})$
      where the attention weights $\alpha_{ij}$ simultaneously encode the geometric and chemical relationships between atoms.
    - **Design Motivation**: Establishing a one-to-one correspondence between the network architecture and the physical process (energy minimization), significantly enhancing interpretability.

2. **SE(3)-Equivariant Design**:

    - **Function**: Ensuring that the model outputs are equivariant under rigid body transformations (rotation + translation).
    - **Mechanism**: Utilizing irreducible representations and tensor products to guarantee equivariance. Message passing between atoms is based on relative position $\mathbf{r}_{ij} = \mathbf{r}_j - \mathbf{r}_i$ and spherical harmonics.
    - **Design Motivation**: The invariance of molecular conformations under rigid body transformations is a fundamental physical symmetry.

3. **Autoencoder Framework**:

    - **Function**: The encoder optimizes conformations, and the decoder generates coordinates.
    - **Mechanism**: The encoder (WGFormer) progressively optimizes the initial conformation into a lower-energy latent representation. The decoder (a simple MLP) maps the optimized features back to 3D coordinates. The training loss is the reconstruction loss: $\mathcal{L} = \sum_i \|\hat{\mathbf{r}}_i - \mathbf{r}_i^{\text{GT}}\|^2$
    - **Design Motivation**: Decoupling encoding and decoding allows WGFormer to focus solely on learning the energy optimization process.

### Loss & Training
- Primary Loss: RMSD (Root Mean Square Deviation) $= \sqrt{\frac{1}{N} \sum_i \|\hat{\mathbf{r}}_i - \mathbf{r}_i^*\|^2}$
- Auxiliary Loss: Distance matrix loss $\mathcal{L}_{\text{dist}} = \sum_{i<j} |d_{ij}^{\text{pred}} - d_{ij}^{\text{GT}}|$
- Training Data: Ground-state conformations calculated via DFT as supervision signals.

## Key Experimental Results

### Main Results (QM9 / GEOM-Drugs)

| Dataset | Metric (RMSD↓) | WGFormer | GeoMol | ConfGF | GeoDiff |
|---|---|---|---|---|---|
| QM9 (Small Molecules) | Mean RMSD (Å) | **0.218** | 0.342 | 0.356 | 0.289 |
| QM9 | Median RMSD | **0.165** | 0.278 | 0.291 | 0.231 |
| GEOM-Drugs | Mean RMSD | **0.892** | 1.243 | 1.312 | 1.078 |
| GEOM-Drugs | COV-R (%) | **85.3** | 74.2 | 71.8 | 79.1 |
| ISO17 | Energy MAE (meV) | **12.3** | 18.7 | 19.2 | 15.5 |

### Ablation Study

| Configuration | QM9 RMSD | Description |
|---|---|---|
| WGFormer (Full) | **0.218** | Full method |
| W/o gradient flow (Standard SE(3)-Transformer) | 0.267 | Significant contribution from the gradient flow |
| W/o SE(3)-equivariance | 0.312 | Equivariance is critical |
| Direct regression (W/o encoder) | 0.295 | Effectiveness of the encoder-decoder framework |
| Fewer layers (4 layers vs 8 layers) | 0.245 | More "gradient flow steps" yield better performance |
| More layers (12 layers) | 0.215 | Diminishing returns |

### Key Findings
- WGFormer consistently outperforms all baselines on both QM9 and GEOM-Drugs, with an even more pronounced advantage on larger molecules.
- The introduction of the Wasserstein gradient flow is the primary contributor to performance improvement (ablation showing a 22% increase in RMSD when removed).
- Each Transformer layer indeed reduces latent energy—witnessed by the visualization of progressive conformation optimization across layers.
- SE(3)-equivariance is critical for molecular conformation prediction (performance drops significantly upon removal).

## Highlights & Insights
- Physically interpretable architectural design: Grounded in step-by-step energy descent across layers, rather than acting as a black-box optimizer.
- Integrating optimal transport theory into molecular generation: Wasserstein gradient flows provide a mathematical framework for optimization on distribution spaces.
- Significant improvements in conformation prediction accuracy: Consistent SOTA performance across standard benchmarks.
- Compelling visualization analysis: Shows atom coordinates progressively converging to their equilibrium states through the network layers.

## Limitations & Future Work
- Requires an initial conformation as input (relying on force-field methods to generate initial guesses).
- Scalability to very large molecules (>100 atoms) remains to be verified.
- The parameterization of the energy function can be further optimized.
- Combination with diffusion models (diffusion + gradient flows) is an interesting future direction.

## Related Work & Insights
- Directly compared against methods such as GeoMol (Ganea et al.), ConfGF (Shi et al.), and GeoDiff (Xu et al.).
- The application of Wasserstein gradient flows has been noted in other domains (e.g., GANs, particle methods).
- Provides a paradigm of physics-driven network design for AI for Science.
- Insight: Embedding physical processes into architectural designs can simultaneously enhance performance and interpretability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Novel combination of Wasserstein gradient flows and SE(3)-Transformers
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple datasets, extensive ablation, and qualitative visualization analysis
- Writing Quality: ⭐⭐⭐⭐ Clear exposition of physical motivations
- Value: ⭐⭐⭐⭐⭐ Significant contributions to both molecular conformation prediction and physically-inspired network designs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TetraGT: Tetrahedral Geometry-Driven Explicit Token Interactions with Graph Transformer for Molecular Representation Learning](../../ICLR2026/computational_biology/tetragt_tetrahedral_geometry-driven_explicit_token_interactions_with_graph_trans.md)
- [\[ICML 2025\] Compositional Flows for 3D Molecule and Synthesis Pathway Co-design](compositional_flows_for_3d_molecule_and_synthesis_pathway_co-design.md)
- [\[ICLR 2026\] SigmaDock: Untwisting Molecular Docking with Fragment-Based SE(3) Diffusion](../../ICLR2026/computational_biology/sigmadock_untwisting_molecular_docking_with_fragment-based_se3_diffusion.md)
- [\[CVPR 2025\] DiffVsgg: Diffusion-Driven Online Video Scene Graph Generation](../../CVPR2025/computational_biology/diffvsgg_diffusion-driven_online_video_scene_graph_generation.md)
- [\[ICML 2025\] Graph Generative Pre-trained Transformer (G2PT)](graph_generative_pre-trained_transformer.md)

</div>

<!-- RELATED:END -->
