---
title: >-
  [Paper Note] GGBall: Graph Generative Model on Poincaré Ball
description: >-
  [ICLR 2026][Image Generation][Hyperbolic Space] This paper proposes GGBall, the first graph generation framework operating entirely on the Poincaré ball model. By combining a hyperbolic vector-quantized variational autoencoder (HVQVAE) with a Riemannian flow matching prior, GGBall achieves state-of-the-art performance on both hierarchical and molecular graph generation, reducing the average generation error by 18% on hierarchical graph benchmarks.
tags:
  - ICLR 2026
  - Image Generation
  - Hyperbolic Space
  - Graph Generation
  - Poincaré Ball Model
  - Vector Quantization
  - Flow Matching
date: 2026-05-08
content_hash: 00469fe2ac488e2e
---

# GGBall: Graph Generative Model on Poincaré Ball

**Conference**: ICLR 2026
**arXiv**: [2506.07198](https://arxiv.org/abs/2506.07198)
**Code**: [GitHub](https://github.com/AI4Science-WestlakeU/GGBall)
**Area**: Graph Generation / Hyperbolic Geometry
**Keywords**: Hyperbolic Space, Graph Generation, Poincaré Ball Model, Vector Quantization, Flow Matching

## TL;DR
This paper proposes GGBall, the first graph generation framework operating entirely on the Poincaré ball model. By combining a hyperbolic vector-quantized variational autoencoder (HVQVAE) with a Riemannian flow matching prior, GGBall achieves state-of-the-art performance on both hierarchical and molecular graph generation, reducing the average generation error by 18% on hierarchical graph benchmarks.

## Background & Motivation
- **Background**: Graph generation is a core task in molecular design, materials discovery, and related domains. Existing methods (e.g., DiGress, GDSS) primarily operate in Euclidean space or discrete graph space.
- **Limitations of Prior Work**: Euclidean latent spaces are inherently ill-suited for capturing hierarchical structures and power-law degree distributions in graph data, leading to distortion of community structures and parent–child relationships.
- **Key Challenge**: There is a geometric mismatch between the combinatorial, hierarchical nature of graphs and the linearly growing volume of Euclidean space.
- **Key Insight**: The exponential volume growth of hyperbolic space naturally accommodates hierarchical structures (Gromov's theorem).
- **Core Idea**: The standard Euclidean latent-space generation pipeline is fully reformulated in hyperbolic space, representing graph topology uniformly via node-level latent variables.

## Method

### Overall Architecture
The framework adopts a two-stage generation process: (1) graph structures are encoded into discrete latent tokens on the Poincaré ball via HVQVAE; (2) a Riemannian flow matching model captures the latent prior, and at generation time samples are drawn from the prior, quantized, and decoded. A GNN encodes local structure, while a Transformer propagates global dependencies.

### Key Designs

1. **Poincaré Graph Neural Network (Poincaré GNN)**:

    - **Function**: Performs message passing in hyperbolic space to encode edge and node information into node representations.
    - **Mechanism**: Tangent-space aggregation with distance-modulated message functions. $\log_0^c(\cdot)$ / $\exp_0^c(\cdot)$ are used to map representations to the tangent space for aggregation and back to the manifold.
    - Message modulation: $\text{M}(\mathbf{m}_{ij}) = \gamma_{ij} \cdot \mathbf{m}_{ij} + \beta_{ij}$, where $\gamma_{ij}, \beta_{ij}$ are functions of the hyperbolic distance $d_c(\mathbf{h}_i, \mathbf{h}_j)$.
    - **Design Motivation**: Curvature-aware distance modulation enables the model to directly encode the strength of hierarchical relationships.

2. **Poincaré Diffusion Transformer**:

    - **Function**: Models global graph structure by replacing dot-product attention with geodesic distance attention.
    - Geodesic attention: $\alpha_{ij} \propto \exp(-\tau d_c(\mathbf{q}_i, \mathbf{k}_j))$
    - Value aggregation uses the Möbius gyromidpoint to maintain geometric consistency.
    - **Time modulation**: Timestep embeddings are injected to support the flow matching prior.

3. **Hyperbolic Vector-Quantized Variational Autoencoder (HVQVAE)**:

    - **Function**: Discretizes continuous hyperbolic embeddings into a learnable Poincaré codebook $\mathcal{C}$.
    - **Mechanism**: Nearest-neighbor quantization based on geodesic distance: $\mathbf{z}_q = \arg\min_{\mathbf{c}_j} d_c(\mathbf{z}, \mathbf{c}_j)$.
    - Codebook entries are initialized via hyperbolic $k$-means clustering and updated with a Riemannian optimizer.
    - **Stability mechanism**: Inactive codebook entries are replaced via an expiry threshold; weighted Einstein midpoint updates are applied.

### Loss & Training
- Autoencoder loss: reconstruction loss + degree–edge consistency loss + L2 regularization.
- HVQVAE loss: $\mathcal{L}_{\text{HVQVAE}} = \lambda_1 \mathcal{L}_{\text{AE}} + \lambda_2 \mathbb{E}[d_c^2(\text{sg}(\mathbf{z}_q), \mathbf{z})] + \lambda_3 \mathbb{E}[d_c^2(\mathbf{z}_q, \text{sg}(\mathbf{z}))]$
- Flow matching prior: Riemannian conditional flow matching objective with geodesic interpolation paths.

## Key Experimental Results

### Main Results: Abstract Graph Generation

| Method | Community-small Avg↓ | Ego-small Avg↓ | Space |
|------|---------------------|----------------|------|
| GraphVAE | 0.6233 | 0.1167 | Euclidean |
| GDSS | 0.0460 | 0.0173 | Graph |
| DiGress | 0.0380 | - | Graph |
| HGDM | 0.0240 | 0.0137 | Hybrid |
| **GGBall** | **0.0197** | **0.0117** | Hyperbolic |

### Molecular Graph Generation (QM9)

| Method | Validity↑ | Uniqueness↑ | Novelty↑ | V.U.N↑ |
|------|-----------|-------------|----------|--------|
| DiGress | 99.00 | 96.34 | 32.51 | 31.04 |
| CatFlow | 98.47 | 97.58 | 65.62 | 63.02 |
| **GGBall** | **98.33** | **96.38** | **93.77** | **88.45** |

### Key Findings
- HVQVAE significantly improves chemical validity (95.18→99.14%) and edge precision over HAE.
- The hyperbolic encoder achieves 4× lower MMD error in degree distribution preservation compared to Euclidean baselines.
- Novelty of 93.77% substantially outperforms DiGress (32.51%), indicating stronger expressiveness of hyperbolic space.

## Highlights & Insights
- GGBall is the first graph generation framework operating entirely on the Poincaré ball, unifying the encode–quantize–prior–decode pipeline end-to-end in hyperbolic space.
- Node-level latent variables provide a unified representation of graph topology, treating edge connectivity as an emergent property of latent-space geometry.
- GGBall achieves the highest V.U.N score on QM9, a composite metric combining novelty, uniqueness, and validity.

## Limitations & Future Work
- Autoregressive priors underperform flow matching in preliminary experiments; further exploration is warranted.
- Validity on molecular graphs is slightly below DiGress (98.33 vs. 99.00); tighter integration of chemical constraints remains an open challenge.
- Hyperbolic space operations introduce higher computational complexity compared to their Euclidean counterparts.

## Related Work & Insights
- **vs. HGDM**: HGDM performs edge denoising only on Poincaré embeddings while still operating in discrete graph space; GGBall operates entirely within the hyperbolic latent space.
- **vs. DiGress**: DiGress iteratively denoises in graph space and is constrained by Euclidean geometry; GGBall leverages hyperbolic latent variables to disentangle hierarchical structure.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First fully hyperbolic graph generation framework, with hyperbolic components throughout: GNN, Transformer, VQ, and flow matching.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers both abstract and molecular graphs with thorough ablations; large-scale graph benchmarks are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematically rigorous with a clear presentation.
- **Value**: ⭐⭐⭐⭐ — Opens a new direction for non-Euclidean geometry in generative modeling.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Verification of the Implicit World Model in a Generative Model via Adversarial Sequences](verification_of_the_implicit_world_model_in_a_generative_model_via_adversarial_s.md)
- [\[ICLR 2026\] PolyGraph Discrepancy: a classifier-based metric for graph generation](polygraph_discrepancy_a_classifier-based_metric_for_graph_generation.md)
- [\[ICLR 2026\] HOG-Diff: Higher-Order Guided Diffusion for Graph Generation](hog-diff_higher-order_guided_diffusion_for_graph_generation.md)
- [\[AAAI 2026\] Rectified Noise: A Generative Model Using Positive-incentive Noise](../../AAAI2026/image_generation/rectified_noise_a_generative_model_using_positive-incentive_noise.md)
- [\[ICLR 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)

<!-- RELATED:END -->
