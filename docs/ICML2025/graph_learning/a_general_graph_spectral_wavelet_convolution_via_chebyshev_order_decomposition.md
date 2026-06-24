---
title: >-
  [Paper Note] A General Graph Spectral Wavelet Convolution via Chebyshev Order Decomposition
description: >-
  [ICML 2025][Graph Learning][Graph Wavelets] Proposed WaveGC, a multi-resolution graph spectral convolutional network that constructs learnable graph wavelets strictly satisfying the admissibility condition by Separating Odd and Even Chebyshev terms and combines them with matrix-valued filter kernels, achieving consistent improvements across both short- and long-range graph tasks (with a 15.7% gain on VOC).
tags:
  - "ICML 2025"
  - "Graph Learning"
  - "Graph Wavelets"
  - "Spectral Convolution"
  - "Chebyshev Polynomials"
  - "Multi-resolution"
  - "Long- and Short-range Information"
date: 2026-05-08
content_hash: 339e3018eab39925
---

# A General Graph Spectral Wavelet Convolution via Chebyshev Order Decomposition

**Conference**: ICML 2025  
**arXiv**: [2405.13806](https://arxiv.org/abs/2405.13806)  
**Code**: [https://github.com/liun-online/WaveGC](https://github.com/liun-online/WaveGC)  
**Area**: Graph Learning  
**Keywords**: Graph Wavelets, Spectral Convolution, Chebyshev Polynomials, Multi-resolution, Long- and Short-range Information

## TL;DR
Proposed WaveGC, a multi-resolution graph spectral convolutional network that constructs learnable graph wavelets strictly satisfying the admissibility condition by Separating Odd and Even Chebyshev terms and combines them with matrix-valued filter kernels, achieving consistent improvements across both short- and long-range graph tasks (with a 15.7% gain on VOC).

## Background & Motivation

**Background**: Spectral graph convolution is a core tool for filtering graph data. There are two key decisions: (a) choosing the spectral basis (signal transform domain); (b) parameterizing the filter kernel (frequency analysis method). Existing methods primarily use the standard Fourier transform and vector-valued spectral functions.

**Limitations of Prior Work**:
   - Standard Fourier basis: Global frequency analysis and insufficient flexibility—unable to simultaneously capture fine-grained short-range and coarse long-range signal features.
   - Vector-valued kernels: Limited expressive power—each frequency component is parameterized by a scalar weight.
   - Existing graph wavelet methods: Either fail to satisfy the admissibility condition (Xu et al.) or utilize fixed, non-learnable wavelet forms (Zheng et al.).

**Key Challenge**: Graph convolutions must process both short-range and long-range information simultaneously, but existing methods are either spatially restricted or lack frequency resolution.

**Goal**: Design a unified multi-resolution graph spectral convolution.

**Key Insight**: Graph wavelets provide a multi-resolution foundation (capturing information across different ranges via various scales), and matrix-valued kernels offer stronger filtering flexibility. The key challenge lies in constructing learnable wavelets that satisfy admissibility.

**Core Idea**: Properly transformed even terms of Chebyshev polynomials strictly satisfy the wavelet admissibility condition, while odd terms supplement the DC signal. By combining them, scaling functions and wavelet bases are generated with learnable coefficients.

## Method

### Overall Architecture
The architecture of WaveGC consists of:
1. Graph Wavelet Basis Construction: Constructing scaling functions and multiple wavelet bases using Chebyshev odd-even decomposition.
2. Wavelet Transform: Projecting graph signals into a multi-scale wavelet coefficient space.
3. Matrix-valued Filtering: Filtering within the wavelet domain using matrix-valued kernels.
4. Inverse Transform: Mapping signals back to the node space.

### Key Designs

1. **Chebyshev Odd-Even Decomposition for Wavelet Construction**:

    - **Function**: Strictly construct learnable graph wavelets that satisfy the admissibility condition.
    - **Mechanism**:
        - The Chebyshev polynomials $T_k(\lambda)$ are decomposed into even terms $T_{2k}$ and odd terms $T_{2k+1}$.
        - Key Discovery: $\tilde{T}_{2k}(\lambda) = T_{2k}(\lambda) - T_{2k}(0)$ automatically equals zero at $\lambda=0$, fulfilling the necessary condition for wavelet admissibility.
        - Scaling function $\phi(\lambda) = \sum_k a_k T_{2k+1}(\lambda)$ (odd terms, preserving the DC component).
        - Wavelet functions $\psi_s(\lambda) = \sum_k b_{s,k} \tilde{T}_{2k}(\lambda)$ (modified even terms, satisfying admissibility).
        - Coefficients $a_k, b_{s,k}$ are learnable.
    - **Design Motivation**: The symmetry of even terms naturally guarantees admissibility without requiring extra constraints—making this approach cleaner than projection or penalty-based methods.
    - **Theoretical Guarantee**: It is proved that any function satisfying the admissibility condition can be approximated by this basis set with arbitrary accuracy.

2. **Multi-resolution Information Capture**:

    - **Function**: Simultaneously capture short-range and long-range graph information.
    - **Mechanism**: Wavelets at different scales $s$ exhibit distinct spatial localization: small scales $s \to 0$ are highly localized (short-range), while large scales $s \to \infty$ cover the entire graph (long-range).
    - **Theoretical Guarantee (Surpassing Prior Work)**:
        - Prior work (Hammond et al.) only proved localization under small-scale limits.
        - This work provides a complete proof of both extremes: small scales yield localization within $K$-hops, while large scales yield global mixing.
        - Corollary: WaveGC can simultaneously capture both short-range and long-range information for each node.
    - **Design Motivation**: This represents the core theoretical advantage of this work—a single architecture that naturally processes information at both scales.

3. **Matrix-valued Filter Kernel**:

    - **Function**: Parameterize filtering in the wavelet domain using matrices (rather than vectors/scalars).
    - **Mechanism**: Each frequency component corresponds to a weight matrix rather than a scalar, leading to a larger parameter space and more flexible cross-channel interactions.
    - **Connection to Transformers**: Under translation-invariant conditions, the spectral function of a matrix-valued kernel is equivalent to the attention mechanism.
    - **Design Motivation**: Scalar kernels assume different feature channels are filtered identically, whereas matrix kernels allow for differentiated processing across channels.

### Loss & Training
- Standard classification/regression loss (depending on downstream tasks).
- Wavelet coefficients are learned during training, making the wavelet forms adaptive to different graph structures.
- Multi-scale features are concatenated and followed by a standard readout layer.

## Key Experimental Results

### Main Results
Short-range tasks (molecular property prediction):

| Method | ZINC (MAE↓) | Alchemy (MAE↓) |
|------|-----------|---------------|
| GCN | 0.367 | 0.203 |
| ChebNet | 0.323 | 0.184 |
| GAT | 0.341 | 0.191 |
| GraphGPS | 0.070 | - |
| **WaveGC** | **0.063** | **0.121** |

### Long-range tasks:

| Method | Peptides-func (AP↑) | PascalVOC (F1↑) |
|------|-------------------|----------------|
| GCN+RWSE | 0.654 | 0.205 |
| SAN | 0.681 | - |
| Exphormer | 0.688 | 0.356 |
| **WaveGC** | **0.695** | **0.412** (+15.7%) |

### Ablation Study

| Configuration | ZINC MAE | PascalVOC F1 | Description |
|------|---------|-------------|------|
| Fourier basis only (no wavelets) | 0.089 | 0.321 | Lacking multi-resolution |
| Fixed wavelets (non-learnable) | 0.078 | 0.375 | Unable to adapt to graph structures |
| Learnable wavelets but without admissibility | 0.071 | 0.389 | Slightly worse |
| **Full WaveGC** | **0.063** | **0.412** | Admissible + Learnable |
| Scalar kernel | 0.073 | 0.385 | Scalar kernel is less flexible |
| **Matrix kernel** | **0.063** | **0.412** | Cross-channel interaction |

### Key Findings
- The 15.7% improvement on PascalVOC demonstrates the tremendous value of multi-resolution for long-range tasks.
- Learnable vs. Fixed wavelets: Adaptive wavelets are consistently superior across all tasks.
- While the admissibility constraint is a theoretical requirement, it also yields practical performance improvements—likely because it guarantees favorable mathematical properties of wavelets (such as energy preservation).
- Matrix vs. Scalar kernels: The performance disparity is larger on long-range tasks—long-range information requires more complex channel interactions.

## Highlights & Insights
- **Chebyshev Odd-Even Decomposition** is an elegant approach for constructing graph wavelets—by leveraging the symmetry of even terms, it naturally satisfies admissibility and bypasses the difficulties of constrained optimization.
- The rigorous proof of multi-scale localization (covering both small and large scale extremes) fills a theoretical gap.
- The link to Transformers (matrix-valued spectral function $\leftrightarrow$ attention) offers a unified perspective.
- The 15.7% boost on PascalVOC indicates that long-range graph tasks remain a distinct bottleneck for existing methods.
- The methodology is generalizable—applicable not only to graph classification but also to node classification and link prediction.

## Limitations & Future Work
- Multi-scale computation increases model parameters and inference time.
- The number of scales $S$ is a hyperparameter.
- Comparisons with more recent graph architectures (such as Mamba/SSM on graphs) are omitted.
- Scalability to large graphs (100k+ nodes) remains unverified.
- Extending to directional wavelets (for directed graphs) is an open research question.

## Related Work & Insights
- **vs ChebNet**: Both utilize Chebyshev polynomials, but ChebNet performs standard convolution (without wavelets), whereas WaveGC separates odd and even terms to construct wavelets.
- **vs GraphWaveletNN**: Standard graph wavelet neural networks use a fixed wavelet form, whereas WaveGC features learnable wavelets.
- **vs Graph Transformer**: Graph Transformers employ global attention ($O(N^2)$ complexity), while WaveGC achieves a comparable global receptive field more efficiently via multi-resolution.
- **vs HyMN**: HyMN enhances graphs by sampling subgraphs based on centrality, whereas WaveGC applies frequency analysis through wavelets—representing two orthogonal directions of graph enhancement.
- **Insights**: Graph learning methods in the spectral domain (wavelets/Fourier) and spatial domain (message passing/sampling) can be highly complementary.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Constructing graph wavelets via Chebyshev odd-even decomposition is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across both short-range and long-range tasks with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Well-written with sound theoretical proofs and comprehensive comparative discussions with existing baselines.
- Value: ⭐⭐⭐⭐⭐ A significant advancement in multi-resolution graph spectral convolutions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Graph Persistence goes Spectral](../../NeurIPS2025/graph_learning/graph_persistence_goes_spectral.md)
- [\[ACL 2025\] Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning](../../ACL2025/graph_learning/beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)
- [\[ICLR 2026\] WATS: Wavelet-Aware Temperature Scaling for Reliable Graph Neural Networks](../../ICLR2026/graph_learning/wats_wavelet-aware_temperature_scaling_for_reliable_graph_neural_networks.md)
- [\[ICML 2025\] Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations](hyperbolic-pde_gnn_spectral_graph_neural_networks_in_the_perspective_of_a_system.md)
- [\[NeurIPS 2025\] OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction](../../NeurIPS2025/graph_learning/ocn_effectively_utilizing_higher-order_common_neighbors_for_better_link_predicti.md)

</div>

<!-- RELATED:END -->
