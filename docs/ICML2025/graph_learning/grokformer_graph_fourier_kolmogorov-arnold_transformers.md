---
title: >-
  [Paper Note] GrokFormer: Graph Fourier Kolmogorov-Arnold Transformers
description: >-
  [ICML2025][Graph Learning][Graph Transformer] Proposes GrokFormer, which utilizes Fourier series-parameterized Kolmogorov-Arnold learnable activation functions to adaptively learn filter bases on multi-order spectra of the graph Laplacian. It possesses both **spectral-order adaptability** and **spectral adaptability**, making it currently the only graph Transformer filter learnable in both dimensions.
tags:
  - "ICML2025"
  - "Graph Learning"
  - "Graph Transformer"
  - "Kolmogorov-Arnold Network"
  - "Fourier Series"
  - "Spectral Graph Filters"
  - "Node Classification"
  - "Graph Classification"
date: 2026-05-08
content_hash: ab81b4c1baf2d048
---

# GrokFormer: Graph Fourier Kolmogorov-Arnold Transformers

**Conference**: ICML2025  
**arXiv**: [2411.17296](https://arxiv.org/abs/2411.17296)  
**Code**: [https://github.com/GGA23/GrokFormer](https://github.com/GGA23/GrokFormer)  
**Area**: Graph Transformer / Spectral Graph Neural Networks  
**Keywords**: Graph Transformer, Kolmogorov-Arnold Network, Fourier Series, Spectral Graph Filters, Node Classification, Graph Classification

## TL;DR

Proposes GrokFormer, which utilizes Fourier series-parameterized Kolmogorov-Arnold learnable activation functions to adaptively learn filter bases on multi-order spectra of the graph Laplacian. It possesses both **spectral-order adaptability** and **spectral adaptability**, making it currently the only graph Transformer filter learnable in both dimensions.

## Background & Motivation

- **Low-pass Bottleneck of Graph Transformers**: Self-attention is fundamentally a low-pass filter, retaining only low-frequency similarity signals between nodes and failing to capture high-frequency difference signals, which is particularly fatal on heterophilic graphs.
- **Limitations of Polynomial Filters**: Methods like ChebyNet, GPRGNN, BernNet, and JacobiConv approximate filters using $K$-order polynomials. Although order-adaptive, the frequency response of each basis is fixed on a predefined spectrum, and the receptive field is restricted to $K$-hop neighbors, lacking flexibility.
- **Limitations of Specformer**: Specformer achieves spectral adaptability (allowing arbitrary frequency response on the first-order spectrum), but is restricted to the first-order Laplacian spectrum, and its filter learning complexity is $O(N^2)$, making it difficult to scale to high-order spectral information.
- **Core Problem**: Can a GT filter be designed to adaptively learn both in **spectral order** (from 1 to $K$-th order) and **spectral location** (response at each eigenvalue)?

| Model Type | Representative Methods | Order-Adaptive | Spectral-Adaptive |
|---------|---------|---------|---------|
| Polynomial GNN | ChebyNet, GPRGNN, BernNet, JacobiConv | ✓ | ✗ |
| Polynomial GT | FeTA, PolyFormer | ✓ | ✗ |
| Spectral GT | Specformer | ✗ | ✓ |
| **GrokFormer** | **Ours** | **✓** | **✓** |

## Method

### Core Idea: Graph Fourier KAN

Inspired by Kolmogorov-Arnold Networks (KANs), GrokFormer replaces fixed basis functions with learnable activation functions. Since the original B-splines in KANs are difficult to train, **Fourier series** are used instead to parameterize each learnable function, yielding the following filter:

$$\phi_h(\lambda) = \sum_{k=1}^{K} \sum_{m=0}^{M} \left( \cos(m\lambda^k) \cdot a_{km} + \sin(m\lambda^k) \cdot b_{km} \right)$$

where $K$ represents the maximum order, $M$ is the number of frequency components (grid size), and $a_{km}, b_{km}$ are trainable Fourier coefficients.

### Order-Adaptability & Spectral-Adaptability

Decomposing the above formula into filter bases $b_k(\lambda)$ for $K$ distinct orders:

$$b_k(\lambda) = \sum_{m=0}^{M} \left( \cos(m\lambda^k) \cdot a_m + \sin(m\lambda^k) \cdot b_m \right)$$

Then, introducing learnable order coefficients $\alpha_k$ to adaptively combine each order:

$$h(\lambda) = \sum_{k=1}^{K} \alpha_k \, b_k(\lambda)$$

- $\alpha_k$ controls spectral order: determines which orders of the graph Laplacian are more important for the task
- $a_{km}, b_{km}$ control spectral location: learn arbitrary frequency responses on a specific order $K$

The final spectral graph convolution is:

$$\mathbf{X}_F^{(l)} = \mathbf{U} \, \text{diag}(h(\lambda)) \, \mathbf{U}^\top \mathbf{X}^{(l-1)}$$

### Network Architecture

Each layer of GrokFormer consists of three parts:

1. **Efficient Multi-Head Self-Attention (EMHA)**: Changes $(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ to $\mathbf{Q}(\mathbf{K}^\top \mathbf{V})$, reducing complexity from $O(N^2 d)$ to $O(Nd^2)$
2. **Graph Fourier KAN**: The aforementioned spectral graph convolution module to capture global spectral domain information
3. **Fusion**: Sums spatial domain (EMHA) and spectral domain signals, followed by LayerNorm + FFN

$$\mathbf{X}'^{(l)} = \text{EMHA}(\text{LN}(\mathbf{X}^{(l-1)})) + \mathbf{X}^{(l-1)} + \mathbf{X}_F^{(l)}$$

$$\mathbf{X}^{(l)} = \text{FFN}(\text{LN}(\mathbf{X}'^{(l)})) + \mathbf{X}'^{(l)}$$

### Theoretical Properties

- **Proposition 4.2**: The polynomial filter $h(\lambda) = \sum_{k=0}^{K} \alpha_k \lambda^k$ is a special case of the GrokFormer filter
- **Proposition 4.3**: The Specformer filter is also a special case of the GrokFormer filter
- **Proposition 4.4**: The GrokFormer filter can approximate any continuous function and construct permutation-equivariant spectral graph convolutions

### Complexity

- Forward propagation: $O(Nd(N + 2d) + KNM)$
- For large graphs, sparse generalized eigendecomposition (SGE) can be used to compute $q \ll N$ eigenvalues, reducing the complexity to $O(2Nd^2 + KqM + Nqd)$

## Key Experimental Results

### Node Classification (11 datasets, accuracy %)

| Method | Cora | Citeseer | Pubmed | Chameleon | Squirrel | Actor | Texas |
|------|------|----------|--------|-----------|----------|-------|-------|
| GCN | 87.14 | 79.86 | 86.74 | 59.61 | 46.78 | 33.23 | 77.38 |
| JacobiConv | 88.98 | 80.78 | 89.62 | 74.20 | 57.38 | 41.17 | 93.44 |
| Specformer | 88.57 | 81.49 | 90.61 | 74.72 | 64.64 | 41.93 | 88.23 |
| PolyFormer | 87.67 | 81.80 | 90.68 | 60.17 | 44.98 | 41.51 | 89.02 |
| **GrokFormer** | **89.57** | **81.92** | **91.39** | **75.58** | **65.12** | **42.98** | **94.59** |

- Homophilic graphs: Achieves best or second best across all 6 datasets, with Physics reaching 98.31%
- Heterophilic graphs: Best performance on all 5 datasets, outperforming Specformer by 0.48% on Squirrel and by 6.36% on Texas

### Graph Classification (5 datasets, accuracy %)

| Method | PROTEINS | MUTAG | PTC-MR | IMDB-B | IMDB-M |
|------|----------|-------|--------|--------|--------|
| Specformer | 70.9 | 96.3 | 82.9 | 86.6 | 58.5 |
| **GrokFormer** | **78.2** | **99.5** | **94.8** | **88.5** | **62.2** |

- MUTAG reaches 99.5%, PTC-MR reaches 94.8%, both leading other GT methods by a large margin

### Filter Fitting Capability

Across 6 standard filters (low-pass, high-pass, band-pass, band-stop, comb, low-frequency comb), GrokFormer achieves the lowest SSE and the highest $R^2$. Especially on the complex low-frequency comb filter, polynomial methods fail completely while GrokFormer achieves near-perfect fitting.

## Highlights & Insights

1. **Dual-dimensional Adaptability**: First to achieve simultaneous adaptive learning of spectral order and spectral location in graph Transformers, theoretically unifying polynomial filtering and Specformer filtering.
2. **Elegant Fourier KAN Design**: Replacing B-splines with $\sin$/$\cos$ orthogonal bases retains KAN's expressive power while significantly simplifying training, and theoretically yields optimal approximation convergence rates.
3. **Significant Advantage on Heterophilic Graphs**: Notable improvements on heterophilic graphs such as Chameleon/Squirrel/Texas, showing that multi-order spectral information is crucial for capturing node difference patterns.
4. **Theoretical Completeness**: Rigorously proves that polynomial filters and Specformer are special cases, and that the filter can approximate any continuous function while preserving permutation equivariance.

## Limitations & Future Work

1. **Spectral Decomposition Overhead**: Requires offline eigendecomposition of $O(N^3)$. Although sparse approximation is available, it remains a bottleneck for extremely large graphs.
2. **Large-scale Scalability**: Penn94 (~42K nodes) is the largest graph in the experiments, and validation on million-scale graphs is missing.
3. **Lack of Inductive Learning Scenarios**: Eigendecomposition relies on the complete graph structure; new node additions in inductive scenarios require re-decomposition.
4. **Hyperparameter Sensitivity**: $K$ (maximum order) and $M$ (number of Fourier components) require tuning, and their interaction is not fully discussed in the paper.
5. **Small Datasets for Graph Classification Experiments**: Only TU datasets were used, without evaluation on large-scale benchmarks like OGB.

## Rating

- Novelty: ⭐⭐⭐⭐ — Integrating Fourier-parameterized KANs into graph Transformer spectral filters; dual-dimensional adaptability is a substantial contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 11 node classification + 5 graph classification + filter fitting experiments + ablations, though lacking large-scale graphs.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, complete theoretical derivations, and high-quality charts.
- Value: ⭐⭐⭐⭐ — Provides a new filter design paradigm for spectral graph Transformers, inspiring learning on heterophilic graphs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Graph Meta-Network for Learning on Kolmogorov–Arnold Networks](../../ICLR2026/graph_learning/a_graph_meta-network_for_learning_on_kolmogorovarnold_networks.md)
- [\[ICLR 2026\] FS-KAN: Permutation Equivariant Kolmogorov-Arnold Networks via Function Sharing](../../ICLR2026/graph_learning/fs-kan_permutation_equivariant_kolmogorov-arnold_networks_via_function_sharing.md)
- [\[NeurIPS 2025\] Unifying and Enhancing Graph Transformers via a Hierarchical Mask Framework](../../NeurIPS2025/graph_learning/unifying_and_enhancing_graph_transformers_via_a_hierarchical_mask_framework.md)
- [\[ICML 2025\] Mitigating Over-Squashing in Graph Neural Networks by Spectrum-Preserving Sparsification](mitigating_over-squashing_in_graph_neural_networks_by_spectrum-preserving_sparsi.md)
- [\[NeurIPS 2025\] Relieving the Over-Aggregating Effect in Graph Transformers](../../NeurIPS2025/graph_learning/relieving_the_over-aggregating_effect_in_graph_transformers.md)

</div>

<!-- RELATED:END -->
