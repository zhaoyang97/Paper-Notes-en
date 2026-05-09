---
title: >-
  [Paper Note] Pseudo-Nonlinear Data Augmentation: A Constrained Energy Minimization Viewpoint
description: >-
  [ICLR 2026][Image Generation][data augmentation] Leveraging the dually flat structure of energy-based models and information geometry, this work proposes a training-free, efficient, and controllable data augmentation method that performs cross-modal augmentation on statistical manifolds via forward projection (encoding) and backward projection (decoding).
tags:
  - ICLR 2026
  - Image Generation
  - data augmentation
  - information geometry
  - energy-based models
  - partially ordered sets
  - training-free methods
date: 2026-05-08
content_hash: ddd35048c1820f61
---

# Pseudo-Nonlinear Data Augmentation: A Constrained Energy Minimization Viewpoint

**Conference**: ICLR 2026
**arXiv**: [2410.00718](https://arxiv.org/abs/2410.00718)
**Code**: [GitHub](https://github.com/sleepymalc/Pseudo-Nonlinear)
**Area**: Data Augmentation / Information Geometry
**Keywords**: data augmentation, information geometry, energy-based models, partially ordered sets, training-free methods

## TL;DR

Leveraging the dually flat structure of energy-based models and information geometry, this work proposes a training-free, efficient, and controllable data augmentation method that performs cross-modal augmentation on statistical manifolds via forward projection (encoding) and backward projection (decoding).

## Background & Motivation

- **Fundamental dilemma of generative model-based augmentation**:
  1. Training a generative model under data scarcity reintroduces the very problem of insufficient data.
  2. Large-scale generation incurs prohibitive computational costs.
  3. Interpretability and controllability are lacking.
- **Limitations of linear dimensionality-reduction augmentation**: the inverse problem (reconstructing high-dimensional data from low-dimensional representations) is ill-posed.
- **Mechanism**: The dually flat structure of statistical manifolds is exploited, where projection is a linear operation in the manifold's intrinsic coordinates yet nonlinear in the ambient space.

## Method

### Log-Linear Model Framework on Partially Ordered Sets

**Three-step embedding pipeline**:
1. **Real-valued poset**: The data structure (vectors/matrices/tensors) is modeled as a partially ordered set $\Omega$.
2. **Statistical manifold embedding**: Data are embedded as probability distributions via $\varphi: \Omega_\mathbb{R} \to \mathcal{S}$.
3. **Dually flat coordinates**: Natural parameters $\theta$ and expectation parameters $\eta$ are obtained via a log-linear model.

For a positive tensor $P$, the embedding is defined as $P'_v = P_v / \sum_{w \in \Omega} P_w$.

### Forward Projection (Encoding)

Data are projected onto a low-dimensional flat submanifold $\mathcal{B} \subseteq \mathcal{S}$:

$$\mathsf{Enc} = \text{Proj}_\mathcal{B} \circ \varphi: \Omega_\mathbb{R} \to \mathcal{B}$$

The projection is unique (when $\mathcal{B}$ is a flat submanifold) and minimizes the KL divergence.

### Backward Projection (Decoding)

**Core innovation**: The pseudo-inverse of the data projection is used as an approximate inverse mapping.
1. Find the $k$ nearest neighbors $N \subseteq [n]$ of $w^*$ in the latent space.
2. Construct a local data submanifold $\mathcal{D}$ from the preimages of these neighbors.
3. Project $w^*$ onto $\mathcal{D}$: $z'^* = \text{Proj}_\mathcal{D}(w^*)$.

### Submanifold Design via Many-Body Approximation

Base submanifold ($\ell$-body approximation):

$$\mathcal{M}_\ell = \{\theta \in \mathbb{R}^{\dim(\mathcal{S})} \mid \theta_x = 0 \text{ for all non } \ell\text{-body parameters } x \in \Omega\}$$

Local data submanifold (dual construction):

$$\mathcal{M}_\ell^*(N) = \{\theta \mid \theta_x = \frac{1}{k}\sum_{i^* \in N}(\theta(z_{i^*}'))_x \text{ for all } \ell\text{-body } x\}$$

### Augmentation Algorithm

1. **Encode**: $w_i = \mathsf{Enc}(z_i) = \text{Proj}_{\mathcal{B}} \circ \varphi(z_i)$
2. **Augment**: Generate a new representation $w^*$ in the latent space $\mathcal{B}$ via kernel density sampling or controlled perturbation.
3. **Decode**: $z^* = \mathsf{Dec}(w^*) = \varphi^{-1} \circ \text{Proj}_\mathcal{B}^{-1}(w^*)$

## Key Experimental Results

### Downstream Classification Performance

| Training Set | MNIST | CIFAR-10 | Speech | Connectionist | Bankruptcy | Wine |
|---|---|---|---|---|---|---|
| OG | 97.98% | 88.57% | 84.48% | 88.10±8.58% | 96.54% | 55.00% |
| OG+STD | 97.98% | **89.89%** | 82.98% | 85.24±7.66% | 96.17% | 57.85% |
| OG+AE | 97.97% | 88.36% | 83.13% | 82.86±7.59% | 95.92% | 57.23% |
| OG+MU | 96.45% | 86.60% | 81.85% | 89.29±4.97% | 96.55% | 57.76% |
| OG+MMU | 97.52% | 88.02% | 83.06% | 91.19±5.06% | 96.44% | 58.70% |
| **OG+PNL** | 97.91% | 88.07% | **84.35%** | **93.81±4.54%** | **96.53%** | **59.03%** |

### Ablation Study: Energy-Aware vs. Ambient-Space Interpolation

| Geometry | Interpolation Energy (Interaction Energy) |
|---|---|
| Base submanifold (energy-aware) | **Consistently lower** |
| Ambient space (Euclidean) | Consistently higher |

The energy-aware method consistently yields lower energy than ambient-space geometry across all interpolation points.

### Key Findings

1. PNL consistently outperforms or matches all other augmentation methods across 6 datasets and 4 modalities.
2. **Stability advantage is pronounced**: on Connectionist Bench (208 samples), standard deviation is reduced from 8.58% to 4.54%.
3. On CIFAR-10, the 1-body approximation preserves shape, while the 5-body approximation captures fine-grained shape–color relationships.
4. Submanifold dimensionality selection involves an inherent trade-off between information retention and augmentation effectiveness.

## Highlights & Insights

1. **Theoretical elegance**: Data augmentation is naturally connected to the dually flat structure of information geometry.
2. **Multi-modal generality**: The same framework handles images, audio, and tabular data uniformly.
3. **Fine-grained controllability**: Augmentation properties are governed by the design of the poset structure and submanifold selection.
4. **Training-free**: Projection reduces to convex optimization with closed-form gradients, making computation highly efficient.
5. **Stability guarantee**: Projection minimizes KL divergence, providing clear information-theoretic guarantees.

## Limitations & Future Work

- **Lack of permutation invariance**: Posets rely on a specific index ordering, introducing bias for data without a natural order, such as graphs.
- The positive tensor assumption restricts direct application to data containing negative values.
- The method does not surpass standard augmentation (e.g., flipping/cropping) on image modalities, as standard methods encode strong modality-specific priors.
- Selecting reshape operations for higher-order tensors requires domain knowledge.

## Related Work & Insights

- **Learning-based augmentation**: VAE-, GAN-, and diffusion model-based augmentation.
- **Training-free augmentation**: Mixup, Manifold Mixup, PCA-based augmentation.
- **Information geometry**: Amari (2016), dually flat structures.
- **Log-linear models on posets**: Sugiyama et al. (2017).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The marriage of information geometry and data augmentation is highly distinctive.
- **Technical Depth**: ⭐⭐⭐⭐⭐ — Theoretically rigorous with solid mathematical derivations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-modal coverage, though large-scale validation is lacking.
- **Value**: ⭐⭐⭐ — Strong generality, but limited advantage on mainstream vision tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Verifier-Constrained Flow Expansion for Discovery Beyond the Data](verifier-constrained_flow_expansion_for_discovery_beyond_the_data.md)
- [\[NeurIPS 2025\] Graph Distance as Surprise: Free Energy Minimization in Knowledge Graph Reasoning](../../NeurIPS2025/image_generation/graph_distance_as_surprise_free_energy_minimization_in_knowledge_graph_reasoning.md)
- [\[NeurIPS 2025\] Non-Asymptotic Analysis of Data Augmentation for Precision Matrix Estimation](../../NeurIPS2025/image_generation/non-asymptotic_analysis_of_data_augmentation_for_precision_matrix_estimation.md)
- [\[ICLR 2026\] RNE: plug-and-play diffusion inference-time control and energy-based training](rne_plug-and-play_diffusion_inference-time_control_and_energy-based_training.md)
- [\[NeurIPS 2025\] UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation](../../NeurIPS2025/image_generation/utilgen_utility-centric_generative_data_augmentation_with_dual-level_task_adapta.md)

</div>

<!-- RELATED:END -->
