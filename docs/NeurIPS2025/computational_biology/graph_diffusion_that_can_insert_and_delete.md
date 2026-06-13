---
title: >-
  [Paper Note] Graph Diffusion that can Insert and Delete
description: >-
  [NeurIPS 2025][Computational Biology][graph diffusion] This paper proposes GrIDDD, the first model to extend discrete denoising diffusion probabilistic models (DDPM) to support **dynamic insertion and deletion of graph n…
tags:
  - "NeurIPS 2025"
  - "Computational Biology"
  - "graph diffusion"
  - "node insertion and deletion"
  - "molecular generation"
  - "property targeting"
  - "molecular optimization"
date: 2026-05-08
content_hash: db5e52095f7e3c61
---

# Graph Diffusion that can Insert and Delete

**Conference**: NeurIPS 2025
**arXiv**: [2506.15725](https://arxiv.org/abs/2506.15725)  
**Code**: None  
**Area**: Graph Generation / Molecular Generation
**Keywords**: graph diffusion, node insertion and deletion, molecular generation, property targeting, molecular optimization

## TL;DR

This paper proposes GrIDDD, the first model to extend discrete denoising diffusion probabilistic models (DDPM) to support **dynamic insertion and deletion of graph nodes** during generation, allowing molecular graph size to adapt throughout the diffusion process. GrIDDD matches or surpasses existing methods on property targeting and molecular optimization tasks.

## Background & Motivation

### Core Problem

Graph-based molecular generation must handle discrete and combinatorially structured connectivity, where chemical constraints render the vast majority of atom–bond combinations invalid. Mainstream approaches include:

| Method Type | Representative Work | Strengths | Weaknesses |
|:---|:---|:---|:---|
| Autoregressive | You et al., JT-VAE | Naturally handles variable length | Sequential dependency, slow sampling |
| One-shot | GraphVAE | Parallel sampling | Difficulty learning high-order interactions |
| Graph Diffusion | DiGress, FreeGress | Permutation-invariant, iterative refinement | **Fixed graph size** |

### Limitations of Fixed Size

The fundamental constraint of existing graph diffusion models is that **sample size remains fixed throughout the entire generation process**, which leads to two serious problems:

**Property Targeting**: When the target property correlates with molecular size (e.g., molecular weight), fixed-size models cannot respond accordingly.

**Property Optimization**: Optimizing target properties by adding or removing atoms is impossible.

Existing workarounds:
- DiGress, E(3) Equivariant Diffusion: sample graph size from the empirical training distribution
- FreeGress: use an auxiliary classifier to predict graph size
- Lift: sample additional nodes during reverse diffusion, but without incorporating this into training

These are all expedient solutions and do not enable dynamic size adjustment within the diffusion process.

## Method

### Overall Architecture

The core innovation of GrIDDD lies in redesigning the forward and reverse diffusion processes to support **monotonic** node insertion and deletion:

- **Forward process**: progressively adds noise to a clean graph $G_0$ (with $n_0$ nodes) until reaching $G_T$ (with $n_T$ nodes)
- **Reverse process**: progressively denoises $G_T$ while adjusting size back to $G_0$
- When $n_T > n_0$, deletion is required; when $n_T < n_0$, insertion is required

### Key Designs

#### Node Deletion Mechanism

Two special atom types are introduced: DEL and DEL*

- **DEL**: represents a fully deleted node (absorbing state)
- **DEL***: an intermediate state in which a node has just been marked for deletion and will transition to DEL at the next step

Why are two states necessary?
- With only DEL, once a node enters the absorbing state it cannot be restored to a normal atom type during the reverse process
- DEL* ensures that during the reverse process, a re-inserted node can transition back to a normal atom type at the next step

#### Mixed Diffusion Process

For cases requiring deletion of $|\Delta_T|$ nodes, a mixed forward process is adopted:
- $n_0 - |\Delta_T|$ nodes use the standard diffusion transition matrix $Q_t$
- $|\Delta_T|$ nodes use a special transition matrix $Q_t^*$

The special transition matrix takes the form:

$$Q_t^* = \zeta(t)(\alpha_t A^* + (1-\alpha_t) B^*) + (1-\zeta(t)) C^*$$

where:
- $A^*, B^*$ are extended versions of the standard matrices (incorporating DEL/DEL* types)
- $C^*$ describes transitions from normal types to DEL
- $\zeta(t)$ is the logistic CDF, controlling the timing of deletion

#### Insertion/Deletion Schedule

A logistic probability density function controls when insertions/deletions occur:

$$\zeta'(t) = -\frac{e^{\frac{t-D}{w}}}{w(e^{\frac{t-D}{w}} + 1)^2}$$

Parameter $D$ controls the peak timestep and $w$ controls the steepness of the curve, ensuring that insertions/deletions are smoothly distributed across the diffusion process.

#### Target Size Sampling

During training, the target size $n_T$ is sampled from a discrete distribution $h_{n_0}(n)$:

$$h_{n_0}(n) = p_{max} + \frac{p_{min} - p_{max}}{n_{max}} (n - n_0)$$

The highest probability is assigned to $n = n_0$ (unchanged size), with probability decreasing linearly as the size difference grows.

### Loss & Training

GrIDDD employs classifier-free guidance for conditional generation:
- Condition information is injected directly into the denoising network during training
- A cross-entropy loss trains the denoiser to predict $G_0$
- Node types and edge types are optimized separately
- The reverse process requires handling posterior probabilities for three scenarios: insertion, deletion, and unchanged size

## Key Experimental Results

### Main Results: Property Targeting

Results on QM9 and ZINC-250k datasets:

| Method | QM9 MAE ↓ | QM9 Validity ↑ | ZINC MAE ↓ | ZINC Validity ↑ |
|:---|:---|:---|:---|:---|
| DiGress | High | ~95% | High | ~85% |
| FreeGress | Medium | ~97% | Medium | ~90% |
| GCPN | Medium | ~90% | Medium | ~80% |
| JT-VAE | Medium | ~100% | Medium | ~100% |
| **GrIDDD** | **Lowest or on par** | **~96–98%** | **Lowest or on par** | **~88–92%** |

GrIDDD achieves property targeting error (MAE) on par with or better than existing best methods while maintaining high chemical validity. Notably, GrIDDD solves a **harder problem** (simultaneously adjusting size and structure) yet still yields competitive results.

### Molecular Optimization

| Method | Avg. Improvement ↑ | Optimization Success Rate ↑ | Type |
|:---|:---|:---|:---|
| VJTNN | Medium | Medium | Translation model |
| Seq2Seq | Low | Low | Translation model |
| HierG2G | Medium–High | Medium | Translation model |
| GCPN | Medium | Medium | RL optimization |
| JT-VAE | Medium | Medium | VAE optimization |
| **GrIDDD** | **Highest** | **Highest** | Diffusion model |

GrIDDD significantly outperforms dedicated optimization models on molecular optimization tasks, precisely because it can optimize target properties by adding or removing atoms rather than only modifying existing atom types.

### Ablation Study

#### Effect of Size Adaptation

| Configuration | QM9 MAE | Molecular Size Variance |
|:---|:---|:---|
| Fixed size (empirical distribution sampling) | Baseline | Low (constrained by pre-sampling) |
| Fixed size (classifier-predicted) | Slightly better | Medium |
| **GrIDDD (dynamic adjustment)** | **Best** | **High (adaptive)** |

Dynamic size adjustment enables the model to generate molecules with more diverse sizes, with the advantage being most pronounced when target properties strongly correlate with molecular size.

### Key Findings

1. **Size adaptability is critical for conditional generation**: GrIDDD's advantage is most evident when target properties (e.g., molecular weight, polarizability) correlate with molecular size.

2. **Harder training does not imply worse results**: Although GrIDDD must simultaneously learn denoising and size adjustment, this added complexity does not degrade property targeting performance.

3. **Structural advantage in molecular optimization**: Conventional methods cannot optimize properties by changing the number of atoms; GrIDDD removes this limitation.

4. **Necessity of the DEL/DEL* dual-state design**: A single deletion state prevents nodes from being recovered during the reverse process; the dual-state design is a key technical contribution.

## Highlights & Insights

- **Theoretical contribution**: This work is the first to formally incorporate node-level insertion and deletion operations within the graph diffusion framework, deriving new forward/reverse transition matrices.
- **Practical value**: It opens the door to size-adaptive molecular generation with direct relevance to drug design.
- **Elegant design**: The seemingly complex functionality is realized by introducing only two special atom types (DEL, DEL*) and a mixed diffusion process.
- **Extensibility**: The method can be applied to any graph generation framework based on discrete diffusion.

## Limitations & Future Work

- Insertion/deletion is monotonic (the reverse process can only continuously insert or continuously delete), and alternating insertion and deletion within a single generation trajectory is not supported.
- The hyperparameters of the logistic schedule ($D$, $w$) require tuning.
- Computational overhead: the transition matrix dimensionality increases due to the additional deletion types.
- Validation is limited to QM9 and ZINC-250k; larger-scale datasets and more complex molecular structures (e.g., proteins) remain untested.
- The determination of threshold $p^*$ may be task-dependent.

## Related Work & Insights

- **DiGress (Vignac et al., 2023)**: categorical diffusion + Graph Transformer; the primary baseline of this work.
- **FreeGress (Ngo et al., 2024)**: classifier-free conditional generation; the training paradigm from which this work draws inspiration.
- **Johnson et al. (2021)**: first proposed insertion/deletion concepts in text diffusion, though with a more complex formulation.
- **Ketata et al. (2025)**: the Lift method samples additional nodes during the reverse process but does not incorporate this into training.
- **GCPN (You et al., 2018)**: RL-based molecular generation; a classic approach for property targeting.
- The innovations presented in this paper are generalizable to domains requiring structure-adaptive generation, such as protein design and materials discovery.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First realization of size-adaptive generation in graph diffusion
- **Technical Depth**: ⭐⭐⭐⭐ — Transition matrix design and posterior derivation are substantive
- **Practicality**: ⭐⭐⭐⭐⭐ — Direct application value for molecular design
- **Clarity**: ⭐⭐⭐⭐ — Method description is clear and figures are intuitive
- **Overall Score**: 8.5/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [\[NeurIPS 2025\] Generalizable Insights for Graph Transformers in Theory and Practice](generalizable_insights_for_graph_transformers_in_theory_and_practice.md)
- [\[NeurIPS 2025\] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs](towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)
- [\[NeurIPS 2025\] Random Search Neural Networks for Efficient and Expressive Graph Learning](random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)
- [\[NeurIPS 2025\] Constrained Discrete Diffusion](constrained_discrete_diffusion.md)

</div>

<!-- RELATED:END -->
