---
title: >-
  [Paper Note] Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction
description: >-
  [NeurIPS 2025][Image Generation][Graph Diffusion Models] This paper proposes GeoMancer, a framework that replaces numerically unstable exponential maps with a Riemannian GyroKernel autoencoder to disentangle multi-level…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Graph Diffusion Models"
  - "Riemannian Manifolds"
  - "Product Manifolds"
  - "Molecular Generation"
  - "Geometric Autoencoder"
date: 2026-05-08
content_hash: 75efdcda830793b1
---

# Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction

**Conference**: NeurIPS 2025
**arXiv**: [2510.04522](https://arxiv.org/abs/2510.04522)  
**Code**: [GitHub](https://github.com/RingBDStack/GeoMancer)  
**Area**: Diffusion Models / Graph Generation / Geometric Learning
**Keywords**: Graph Diffusion Models, Riemannian Manifolds, Product Manifolds, Molecular Generation, Geometric Autoencoder

## TL;DR

This paper proposes GeoMancer, a framework that replaces numerically unstable exponential maps with a Riemannian GyroKernel autoencoder to disentangle multi-level graph features onto task-specific product manifolds. By further introducing manifold-constrained diffusion and a self-guidance generation strategy, GeoMancer achieves unified modeling and state-of-the-art performance across molecular generation, node classification, and graph regression tasks.

## Background & Motivation

Graph diffusion models have made significant progress in learning and generating graph-structured data, enabling the unified modeling of generation and prediction tasks via latent-space diffusion — by reformulating prediction as conditional generation. However, existing methods embed node-, edge-, and graph-level features into a shared Euclidean latent space, neglecting the inherently non-Euclidean nature of graph data.

Through t-SNE visualization, the authors find that latent representations at different levels become entangled within the shared Euclidean space, despite possessing **distinct intrinsic geometric properties** (curvature heterogeneity). For instance, hierarchical structures are better modeled in hyperbolic space, while densely connected graphs are more naturally represented on spherical manifolds. Forcing features with different geometric properties into a single shared space fails to fully exploit their geometric potential.

Building an ideal Riemannian diffusion framework faces two major challenges: (1) **Numerical instability during encoding**: projecting features onto product manifolds via exponential maps tends to cause numerical explosion and is difficult to optimize; (2) **Manifold deviation during diffusion-based generation**: the generative process tends to drift away from the original data manifold, and unconditional generation tasks lack guidance signals to correct this deviation.

## Method

### Overall Architecture

GeoMancer consists of three components: (1) a Riemannian GyroKernel autoencoder — which employs a generalized Fourier transform in place of exponential maps to achieve isometry-invariant feature mapping, encoding multi-level features onto their respective product manifolds; (2) manifold-constrained diffusion — which performs standard diffusion in the geometric latent space and incorporates CFG++ for manifold-constrained conditional generation; and (3) a self-guidance strategy — which generates pseudo-labels via clustering for unconditional generation tasks, unifying all tasks under a conditional generation framework.

### Key Designs

1. **Riemannian GyroKernel Mapping**: An isometry-invariant kernel function is constructed via Bochner's theorem, replacing the numerically unstable exponential/logarithmic maps. In a gyrovector ball $\mathbb{G}^n_\kappa$ with curvature $\kappa$, the generalized Fourier feature is defined as $\text{gF}^\kappa_{\omega,b,\lambda}(x) = A_{\omega,x}\cos(\lambda\langle\omega,x\rangle_\kappa + b)$, where $\langle\omega,x\rangle_\kappa = \log\frac{1+\kappa\|x\|^2}{\|x-\omega\|^2}$ is the signed distance in the gyrovector ball. Product manifold representations $V_\kappa$ with different curvatures $\kappa_i$ are initialized, and isometry-invariant Euclidean features are obtained via $\bar{V}_\kappa = \phi_{\text{gF}}(V_\kappa)$. The key advantage is that this preserves the geometric properties of Riemannian space while enabling operations directly in Euclidean space.

2. **Multi-Level Encode–Decode Architecture**: The encoder employs a Graph Transformer (GrIT) to jointly process node and edge features, capturing relational structure via attention: $z^{l+1}_{e_{ij}} = \sigma(\rho((Qz^l_{x_i}, Kz^l_{x_j}) \odot E_w z^l_{e_{ij}}) + E_b z^l_{e_{ij}})$. After encoding, geometric priors are assigned to the embeddings via $\bar{Z} = Z\bar{V}_\kappa$, allowing each dimension to capture distinct geometric information. The decoder factorizes the complex product manifold into simpler sub-manifolds $\mathcal{M} \to \mathcal{M}_1 \times \cdots \times \mathcal{M}_m$, selecting the most suitable geometric representation for each task level.

3. **Self-Guidance Strategy**: To address the absence of label guidance in unconditional graph generation, k-means clustering is applied to graph-level representations $Z_G$ to produce pseudo-labels $C$, reformulating unconditional generation as conditional generation $P(G|C)$. During sampling, $C$ is selected randomly. This unifies all graph tasks — unconditional generation, conditional generation, and prediction — under a single conditional generation framework.

4. **Manifold-Constrained Conditional Generation**: CFG++ is adopted for manifold-constrained sampling. The generation process is formulated as: $\tilde{Z_0} = (Z_t - \sqrt{1-\bar{\alpha}_t}\tilde{\epsilon}_\theta(Z_t, \tau(y))) / \sqrt{\bar{\alpha}_t}$, where $\tilde{\epsilon}_\theta$ combines conditional and unconditional noise predictions: $\tilde{\epsilon}_\theta(\hat{Z_t}, \tau(y)) = (1-\lambda)\epsilon_\theta(\hat{Z_t}, \tau(y)) - \lambda\epsilon_\theta(\hat{Z_t})$. This ensures that generated samples remain on the clean data manifold.

### Loss & Training

The training objective combines task loss and regularization: $\mathcal{L} = \mathcal{L}_{tgt} + \mathcal{L}_{reg}$. The task loss is cross-entropy for classification/generation tasks and MSE for regression. The regularization loss is the KL divergence $D_{\text{KL}}(q(Z|(X,E)) \| \mathcal{N}(0,I))$, which prevents high variance in the latent space. Diffusion training follows the standard noise prediction objective.

## Key Experimental Results

### Main Results

Unconditional molecular generation on QM9:

| Model | Validity(%)↑ | Uniqueness(%)↑ | FCD↓ | NSPDK↓ | Novelty(%)↑ |
|-------|:-----------:|:-------------:|:----:|:------:|:-----------:|
| DiGress | 99.01 | 96.34 | 0.25 | 0.0003 | 35.46 |
| GruM | 99.69 | 96.90 | 0.11 | 0.0002 | 24.15 |
| LGD | 98.46 | 97.53 | 0.32 | 0.0004 | 56.35 |
| **GeoMancer** | **100.00** | 95.74 | **0.09** | **0.0002** | **90.43** |

Node classification accuracy:

| Model | Photo | Physics | Pubmed | Cora | Citeseer |
|-------|:-----:|:-------:|:------:|:----:|:--------:|
| NAGphormer | 95.49 | 97.34 | 91.76 | 82.13 | 71.40 |
| LGD | 96.94 | 98.55 | 92.88 | 82.81 | 72.40 |
| **GeoMancer** | **97.05** | **98.78** | **93.10** | **83.50** | **72.60** |

### Ablation Study

| Configuration | Validity(%)↑ | FCD↓ | NSPDK↓ | Novelty(%)↑ | Notes |
|---------------|:-----------:|:----:|:------:|:-----------:|-------|
| w/o self-guidance | 98.99 | 0.12 | 0.0003 | 54.06 | Pseudo-label guidance contributes significantly to novelty |
| w/o CFG++ | 100.00 | 0.09 | 0.0010 | 76.43 | Manifold constraint improves distributional fit (NSPDK) |
| w/o Riemannian | 100.00 | 0.25 | 0.0004 | 90.26 | Riemannian geometry substantially improves FCD |
| **GeoMancer (full)** | **100.00** | **0.09** | **0.0002** | **90.43** | All three components are complementary |

### Key Findings

- GeoMancer achieves 100% validity on QM9 while dramatically improving novelty from 56.35% (LGD) to 90.43%.
- High-quality conditional molecular generation is achievable without 3D geometric information.
- The self-guidance mechanism is the primary driver of novelty gains (+36%), while Riemannian geometry contributes most to distributional fit quality.
- Manifold visualizations confirm that different task levels genuinely favor sub-manifolds of different curvatures.
- GeoMancer outperforms all conventional regression models and graph Transformers on the graph regression task (ZINC12k).

## Highlights & Insights

- The unified framework design is highly ambitious: a single model simultaneously handles unconditional generation, conditional generation, node classification, and graph regression.
- The substitution of GyroKernel for exponential maps is an elegant design choice: it preserves the geometric properties of Riemannian space while avoiding numerical instability, allowing the entire diffusion process to operate in Euclidean space.
- The self-guidance strategy is simple yet effective: pseudo-labels are derived from the geometric structure already present in the latent space, unifying all tasks under conditional generation.

## Limitations & Future Work

- The number of product manifold components and the choice of curvatures remain hyperparameters; learnable curvatures partially address this, but principled theoretical guidance is lacking.
- Molecular generation is validated only on QM9 (≤9 heavy atoms); scalability to larger molecules remains to be explored.
- Gains on node classification are relatively modest, possibly because this task involves lower intrinsic geometric complexity.
- The core components (GyroKernel, product manifold disentanglement, CFG++) are drawn from existing works, which somewhat blurs the boundary of original contribution.
- No direct comparison is made against recent Riemannian diffusion methods (e.g., HypDiff) on graph generation benchmarks.

## Related Work & Insights

- **Relationship to LGD**: GeoMancer extends the latent graph diffusion framework with geometric awareness, representing a significant improvement along the same research line.
- The GyroKernel approach is adapted from MotifRGC and HyLA, but is generalized here to product manifolds and multi-level graph tasks.
- CFG++-based manifold-constrained sampling originates from image generation and is cleverly transferred to the graph generation setting.
- This work may inspire the application of geometry-aware diffusion frameworks to more complex graph-structured data in protein design, materials science, and related domains.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The unified perspective of geometry-aware graph diffusion is novel, though the individual components largely combine existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Broad task coverage (generation + classification + regression) with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured, though the dense notation requires a solid mathematical background.
- **Value**: ⭐⭐⭐⭐ Provides a unified framework for geometric understanding of graph data with practical value for molecular modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[NeurIPS 2025\] Graph Diffusion that can Insert and Delete](graph_diffusion_that_can_insert_and_delete.md)
- [\[NeurIPS 2025\] Riemannian Consistency Model](riemannian_consistency_model.md)
- [\[ICCV 2025\] A Unified Framework for Motion Reasoning and Generation in Human Interaction](../../ICCV2025/image_generation/a_unified_framework_for_motion_reasoning_and_generation_in_human_interaction.md)
- [\[NeurIPS 2025\] LLM Meets Diffusion: A Hybrid Framework for Crystal Material Generation](llm_meets_diffusion_a_hybrid_framework_for_crystal_material_generation.md)

</div>

<!-- RELATED:END -->
