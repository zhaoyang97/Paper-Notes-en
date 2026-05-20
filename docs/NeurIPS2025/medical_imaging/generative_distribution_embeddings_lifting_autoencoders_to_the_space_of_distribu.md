---
title: >-
  [Paper Note] Generative Distribution Embeddings: Lifting Autoencoders to the Space of Distributions for Multiscale Representation Learning
description: >-
  [NeurIPS 2025][Medical Imaging][distribution embeddings] This paper proposes Generative Distribution Embeddings (GDE), which lifts autoencoders to the space of distributions — the encoder operates on sets of samples whil…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "distribution embeddings"
  - "autoencoders"
  - "Wasserstein space"
  - "multiscale representation"
  - "computational biology"
date: 2026-05-08
content_hash: d864879d061fec8a
---

# Generative Distribution Embeddings: Lifting Autoencoders to the Space of Distributions for Multiscale Representation Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.18150](https://arxiv.org/abs/2505.18150)  
**Code**: Available (GitHub)  
**Area**: Medical Imaging
**Keywords**: distribution embeddings, autoencoders, Wasserstein space, multiscale representation, computational biology

## TL;DR

This paper proposes Generative Distribution Embeddings (GDE), which lifts autoencoders to the space of distributions — the encoder operates on sets of samples while the decoder is replaced by a conditional generative model — thereby learning distribution-level representations. The framework is validated on 6 computational biology tasks.

## Background & Motivation

**Background**: Modern science, particularly computational biology, increasingly requires cross-scale reasoning where the unit of analysis is not an individual data point (e.g., a single cell) but rather the distribution to which data points belong (e.g., all cells from a patient). Kernel methods, Wasserstein-space approaches, and variational autoencoders each have their own limitations.

**Limitations of Prior Work**:
- Traditional encoders operate on individual data points, discarding **population-level** signals.
- Kernel mean embeddings (KME) are nonparametric but not generative.
- Methods such as Wasserstein Wormhole are restricted to sampling a fixed number of points.
- Existing approaches lack the ability to decode back to a distribution from a distribution embedding.

**Key Challenge**: In hierarchical data (patient → cell → gene expression), unit-level noise is substantial (e.g., molecular undersampling), yet the signal of interest resides at the distribution level.

**Goal**: To construct a general framework capable of learning compact representations of distributions and resampling those distributions from the learned representations.

**Key Insight**: The autoencoder concept is lifted such that the encoder receives a set of samples (an empirical distribution) and the decoder is a conditional generative model (sampling conditioned on an embedding vector). A key constraint is that the encoder must be *distribution-invariant*.

**Core Idea**: By combining a distribution-invariant encoder with a conditional generative model, any autoencoder framework can be lifted to the space of distributions, learning representations equivalent to predictive sufficient statistics.

## Method

### Overall Architecture

GDE consists of two components:
- **Encoder $\mathcal{E}$**: Maps a sample set $S_{i,m} = \{x_{ij}\}_{j=1}^m$ to a latent representation $z_i$.
- **Conditional Generator $\mathcal{G}$**: Generates new samples given $z_i$ such that $\mathcal{G}(\mathcal{E}(S_{i,m})) \xrightarrow{m \to \infty} P_i$.

**Training Algorithm**: For each set $S_{i,m_i}$, a subsample $\tilde{S}_{i,m}$ is drawn, the embedding $z_i = \mathcal{E}(\tilde{S}_{i,m})$ is computed, and backpropagation is performed using the generator loss $\ell(\tilde{S}_{i,m}, \mathcal{G}(z_i))$.

### Key Designs

#### 1. Distribution Invariance

The encoder must satisfy two conditions:
- **Permutation invariance**: The ordering of samples does not affect the embedding.
- **Scale invariance**: Replicating each sample $K$ times does not change the embedding.

These properties ensure the encoder depends only on the empirical distribution $P_{i,m} = \frac{1}{m}\sum_{j=1}^m \delta_{x_{ij}}$.

**Theoretical Guarantees**:
- A distribution-invariant encoder can capture arbitrary distributional properties.
- Non-distribution-invariant architectures may spuriously encode noise features unrelated to the distribution.
- Distribution invariance combined with Hadamard differentiability yields a central limit theorem for the embedding: $\sqrt{m}(\mathcal{E}(S_{i,m}) - \phi(P_i)) \xrightarrow{d} \mathcal{N}(0, \Sigma_{\phi,i})$.

**Implementation**: Mean pooling and M/Z estimators satisfy distribution invariance; sum pooling does not.

#### 2. Flexibility of the Conditional Generator

Any conditional generative model is compatible with GDE, including:
- VAEs (e.g., CVAE)
- Denoising diffusion probabilistic models (DDPM)
- Sinkhorn generative models
- Sliced Wasserstein models
- Autoregressive sequence models (e.g., ProGen2, HyenaDNA)

#### 3. Generalization from Labels to Distributions

When data do not naturally form a hierarchical structure, distributions are constructed via label spaces:
- **Discrete labels**: samples are grouped by label.
- **Continuous labels**: Gaussian kernel-weighted sampling is applied.
- **Noisy labels**: likelihood weighting is used.
- All cases are unified under a general framework of sampling from a label prior $Q^{(\mathcal{Y})}$.

### Theoretical Properties

#### Predictive Sufficient Statistics

The representations learned by GDE approximate predictive sufficient statistics — conditioning on the representation enables prediction of new samples while marginalizing over sampling noise. This is validated empirically: on a Poisson distribution, the GDE estimator achieves lower MSE than the Rao–Blackwell estimator ($3.12 \times 10^{-3}$ vs. $3.79 \times 10^{-3}$ at $n=10$).

#### Wasserstein Geometry

- The $L_2$ distance in latent space is highly correlated with $W_2$ distance (Gaussian: $\rho = 0.96$; GMM: $\rho = 0.76$).
- Linear interpolation in latent space approximates optimal-transport geodesics.
- When the prior $Q$ is non-uniform, the geometry adapts accordingly.

## Key Experimental Results

### Synthetic Benchmarks

| Model | Normal | GMM | MNIST | FMNIST |
|-------|--------|-----|-------|--------|
| KME + DDPM | 0.04 | 2.17 | 80.46 | 111.01 |
| $W_2$ Wormhole | 0.20 | 2.88 | 263.29 | 320.18 |
| **GDE** | **0.02** | **1.82** | **63.79** | **102.21** |

### Application 1: Patient-Level Representation (6.3M Single-Nucleus RNA-seq)

| Metric | Supervised | Semi-supervised GDE |
|--------|-----------|-------------------|
| Accuracy | 0.8791 | **0.8887** |
| ROC AUC | 0.4872 | **0.5131** |
| F1 Score | 0.1293 | **0.1479** |

### Application 2: Clonal Population Modeling (Lineage-Traced scRNA-seq)

GDE + CVAE surpasses Wasserstein Wormhole by more than **2 bits** of mutual information.

### Application 3: Transcriptomic Perturbation Prediction

| Method | $R^2$↑ | MSE↓ |
|--------|--------|------|
| Mean (direct regression) | 0.378 | 1.855 |
| scVI | 0.421 | 1.551 |
| **GDE** | **0.458** | **1.501** |

### Application 4: Single-Cell Image Phenotype Prediction

- 5,072 genetic perturbations; 20M+ single-cell images.
- Zero-shot prediction of nuclear signal intensity for held-out perturbations: $R^2 = 0.7055$, MSE = 0.00068.

### Application 5: Yeast Promoter Design (34M Sequences)

The GDE embedding space recovers a smooth gradient of expression quantiles, and the reconstructed transcription factor binding site (TFBS) motif distributions closely match ground-truth data.

### Application 6: Spatiotemporal Modeling of Viral Proteins (SARS-CoV-2, 1M Sequences)

- Temporal prediction MAE: GDE **1.83±0.01** months vs. ESM baseline 2.24±0.01 months.
- Country classification accuracy: GDE 0.28 vs. ESM 0.25 vs. majority baseline 0.21.

### Key Findings

1. **Mean-pooled deep sets + DDPM** achieves the best performance among 30 encoder–generator combinations.
2. **GDE outperforms KME and Wasserstein Wormhole on all synthetic benchmarks.**
3. **Semi-supervised GDE outperforms purely supervised models**, leveraging the distributional structure of unlabeled data.
4. **The GDE latent space naturally exhibits Wasserstein geometry**, aligning with optimal-transport geodesics.

## Highlights & Insights

- **Conceptual elegance**: The framework distills "autoencoders over distributions" into a minimal design of distribution-invariant encoder + conditional generator.
- **Theoretical depth**: The work connects three major theoretical pillars — predictive sufficient statistics, information geometry, and Wasserstein space.
- **Broad generality**: The same framework applies across DNA sequences, protein sequences, gene expression profiles, and microscopy images.
- **CLT guarantee**: At inference time, all available samples (potentially millions) can be used, ensuring stable convergence of the embedding.
- **Prior-aware geometry**: The latent space geometry adapts to the meta-distribution $Q$, allocating higher resolution to high-density regions.

## Limitations & Future Work

1. **Set construction requires domain knowledge**: The choice of how to group samples (i.e., the meta-distribution prior $Q$) depends on domain expertise.
2. **Gradient propagation through the encoder**: Backpropagating gradients from the generator to the encoder poses engineering challenges.
3. **Scalability to large sets**: Although the CLT provides theoretical guarantees for encoding millions of samples, practical computation still requires optimization.
4. **Exchangeability assumption**: The theoretical framework does not apply to non-i.i.d. samples within a set.
5. **Insufficient mechanistic evidence for Wasserstein isometry**: The observed alignment is currently supported only by empirical findings, lacking a formal proof.

## Related Work & Insights

- The framework generalizes kernel mean embeddings (KME) and Wasserstein Wormhole as special cases of GDE.
- GDE is complementary to Meta Flow Matching and Fisher–Rao flow models.
- **Core insight**: Any conditional generative model can be "freely" upgraded into a distribution representation learner — simply by pairing it with a distribution-invariant encoder.

## Rating

⭐⭐⭐⭐⭐ (5/5)

**Rationale**: GDE demonstrates exceptional conceptual novelty (lifting autoencoders to the space of distributions), strong theoretical and empirical contributions (CLT + sufficient statistics + Wasserstein geometry, validated across 6 large-scale biological applications), and remarkable generality. The experimental scale is impressive (6M cells, 20M images, 34M sequences). This work sets a new standard for distribution-level representation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [\[NeurIPS 2025\] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs](towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)
- [\[NeurIPS 2025\] Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment](multimodal_disease_progression_modeling_via_spatiotemporal_disentanglement_and_m.md)
- [\[NeurIPS 2025\] Ordinal Label-Distribution Learning with Constrained Asymmetric Priors for Imbalanced Retinal Grading](ordinal_label-distribution_learning_with_constrained_asymmetric_priors_for_imbal.md)
- [\[NeurIPS 2025\] Multiscale Guidance of Protein Structure Prediction with Heterogeneous Cryo-EM Data](multiscale_guidance_of_protein_structure_prediction_with_heterogeneous_cryo-em_d.md)

</div>

<!-- RELATED:END -->
