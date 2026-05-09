---
title: >-
  [Paper Note] Directional Sheaf Hypergraph Networks: Unifying Learning on Directed and Undirected Hypergraphs
description: >-
  [ICLR 2026][sheaf neural networks] This paper proposes Directional Sheaf Hypergraph Networks (DSHN), which combines Cellular Sheaf theory with the directional information of directed hypergraphs to construct a complex-valued Hermitian Laplacian operator. The proposed operator unifies and generalizes existing graph and hypergraph Laplacians, achieving 2%–20% relative accuracy improvements over baselines on 7 real-world datasets.
tags:
  - ICLR 2026
  - sheaf neural networks
  - directed hypergraphs
  - Laplacian
  - spectral methods
  - heterophily
date: 2026-05-08
content_hash: 12e2638186d84435
---

# Directional Sheaf Hypergraph Networks: Unifying Learning on Directed and Undirected Hypergraphs

**Conference**: ICLR 2026
**arXiv**: [2510.04727](https://arxiv.org/abs/2510.04727)
**Code**: [GitHub](https://github.com/EmaMule/DirectionalSheafHypergraphs)
**Area**: Others
**Keywords**: sheaf neural networks, directed hypergraphs, Laplacian, spectral methods, heterophily

## TL;DR

This paper proposes Directional Sheaf Hypergraph Networks (DSHN), which combines Cellular Sheaf theory with the directional information of directed hypergraphs to construct a complex-valued Hermitian Laplacian operator. The proposed operator unifies and generalizes existing graph and hypergraph Laplacians, achieving 2%–20% relative accuracy improvements over baselines on 7 real-world datasets.

## Background & Motivation

**Higher-order interaction modeling on hypergraphs**: Many real-world systems exhibit higher-order relationships among multiple entities, whereas traditional graphs can only represent pairwise interactions. Hypergraphs model multi-way interactions by connecting multiple nodes via hyperedges.

**Limitations of undirected hypergraphs**: Most HGNNs handle only undirected hypergraphs, ignoring directionality that may exist within hyperedges (e.g., reactants → products in chemical reactions, initiator → receiver in causal interactions).

**Sheaf theory for heterophily**: By assigning vector spaces and learnable restriction maps to nodes and edges, sheaf theory effectively mitigates over-smoothing and heterophily. However, existing Sheaf Hypergraph methods do not support directed hypergraphs.

**Spectral deficiency of existing SHNs**: The Sheaf Hypergraph Laplacian of Duta et al. (2023) does not satisfy positive semi-definiteness and therefore cannot serve as a valid convolution operator.

**Lessons from directed graph methods**: The Magnetic Laplacian encodes direction via complex phases but has not been extended to hypergraphs.

## Method

### Overall Architecture

DSHN proceeds in three steps: (1) define a Directed Hypergraph Cellular Sheaf with complex-valued restriction maps; (2) construct a Directed Sheaf Hypergraph Laplacian as a Hermitian positive semi-definite operator; (3) build a diffusion-based convolutional network on top of this Laplacian.

### Key Designs

1. **Directionality matrix $\mathcal{S}^{(q)}$ (Definition 1)**

   - **Function**: Assigns complex-valued coefficients to node–hyperedge pairs to encode directional roles.
   - **Mechanism**: Head nodes receive coefficient 1; tail nodes receive coefficient $e^{-2\pi i q}$; the parameter $q$ controls the strength of directional information.
   - **Design Motivation**: Setting $q=0$ reduces to the undirected case; at $q=1/4$, directional encoding resides in the imaginary part, aligning with the Magnetic Laplacian.

2. **Directed Sheaf Hypergraph Laplacian**

   - **Function**: $\mathbf{L}^{\vec{\mathcal{F}}} = \mathbf{D}_V - \mathbf{B}^{(q)\dagger} \mathbf{D}_E^{-1} \mathbf{B}^{(q)}$
   - **Mechanism**: Diagonal blocks are real-valued (self-information); off-diagonal blocks are complex-valued for directed cases. Diagonal term coefficients are corrected relative to Duta et al.
   - **Design Motivation**: Ensures positive semi-definiteness and all other spectral properties required for graph convolution.

3. **Spectral guarantees**

   - **Function**: Proves diagonalizability, real non-negative eigenvalues, positive semi-definiteness, and a spectral upper bound of 1.
   - **Mechanism**: Positive semi-definiteness follows from the non-negativity of the Dirichlet energy.
   - **Design Motivation**: Ensures a well-defined Fourier transform and stable polynomial filters.

4. **Unified generalization**

   - **Function**: Demonstrates that the proposed operator reduces to the Graph Sheaf Laplacian, Magnetic Laplacian, Zhou Hypergraph Laplacian, and GeDi Laplacian under special cases.
   - **Design Motivation**: A single framework subsumes all existing operators.

5. **DSHNLight**

   - **Function**: Detaches gradients from Laplacian construction and fixes restriction map parameters.
   - **Design Motivation**: Significantly reduces computational cost while achieving comparable or superior performance on multiple datasets.

### Loss & Training

- Standard cross-entropy loss for node classification.
- Complex-valued outputs are converted to real values by unwinding (concatenating real and imaginary parts) before being passed to the classification head.
- Restriction maps are learned via an MLP whose input is the concatenation of node and hyperedge features.

## Key Experimental Results

### Main Results

Node classification accuracy compared against 13 baselines on 7 datasets:

| Dataset | DSHN improvement over best baseline |
|--------|--------------------------------------|
| Cora (co-auth) | ~2% |
| Citeseer (co-auth) | ~5% |
| Senate-committees | ~8% |
| House-committees | ~4% |
| Walmart-trips | ~20% |
| Zoo | ~3% |
| 20Newsgroups | ~2% |

### Ablation Study

| Variant | Effect |
|---------|--------|
| $q=0$ (no direction) | Reduces to undirected sheaf method; performance degrades |
| $q=1/4$ (standard phase) | Best performance on directed datasets |
| Trivial sheaf ($\mathcal{F}=I$) | Reduces to directed hypergraph Laplacian; significant performance drop |
| DSHNLight | High computational efficiency; performance close to full model on most datasets |

### Key Findings

- Combining directionality and sheaf structure significantly outperforms using either alone.
- The Laplacian of Duta et al. (2023) indeed exhibits negative eigenvalues (a counterexample is provided in the appendix).
- The "random projection" strategy of DSHNLight proves surprisingly effective.
- The advantage is most pronounced on heterophilic datasets.

## Highlights & Insights

1. A single complex-valued Hermitian operator unifies multiple existing Laplacian definitions.
2. The paper rigorously corrects the spectral property errors in Duta et al. (2023).
3. The idea of encoding directional information via complex phases is naturally extended from directed graphs to hypergraphs.
4. DSHNLight echoes the philosophy of extreme learning machines, demonstrating the effectiveness of random features in graph learning.

## Limitations & Future Work

- Scalability challenges arise from the $nd \times nd$ Laplacian.
- The global parameter $q$ cannot learn different directional strengths per hyperedge.
- Experiments are limited to node classification tasks.
- Real-world directed hypergraph datasets remain scarce.
- Expressiveness analysis (e.g., WL hierarchy) is absent.

## Related Work & Insights

- **Hansen & Gebhart (2020)**: Graph Sheaf NN → extended in this work to directed hypergraphs.
- **Zhang et al. (2021)**: Magnetic Laplacian → extended in this work to hypergraphs with sheaves.
- **Duta et al. (2023)**: SheafHyperGNN → spectral deficiencies corrected in this work.
- **Insight**: The complex-valued Hermitian + Sheaf paradigm can be generalized to more abstract topological structures such as simplicial complexes.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of sheaves and directed hypergraphs, along with the unification results, carries significant theoretical value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 7 datasets, 13 baselines, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear; notation is heavy but precisely defined.
- **Value**: ⭐⭐⭐⭐ Corrects deficiencies in prior methods and provides a unified framework.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Sheaf Cohomology of Linear Predictive Coding Networks](../../NeurIPS2025/others/sheaf_cohomology_of_linear_predictive_coding_networks.md)
- [\[ICLR 2026\] Learning on a Razor's Edge: Identifiability and Singularity of Polynomial Neural Networks](learning_on_a_razors_edge_identifiability_and_singularity_of_polynomial_neural_n.md)
- [\[ICLR 2026\] Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)
- [\[ICLR 2026\] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)
- [\[AAAI 2026\] Learning Fair Representations with Kolmogorov-Arnold Networks](../../AAAI2026/others/learning_fair_representations_with_kolmogorov-arnold_networks.md)

<!-- RELATED:END -->
