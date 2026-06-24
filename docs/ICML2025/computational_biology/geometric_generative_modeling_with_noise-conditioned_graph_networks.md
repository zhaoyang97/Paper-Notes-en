---
title: >-
  [Paper Note] Geometric Generative Modeling with Noise-Conditioned Graph Networks
description: >-
  [ICML2025][Computational Biology][Noise-Conditioned Graph Networks] Proposes Noise-Conditioned Graph Networks (NCGNs) to dynamically adjust message passing range and graph resolution in GNN architectures based on noise levels: long-range connections with low resolution at high noise levels, and local connections with high resolution at low noise levels, outperforming static architecture baselines in 3D point cloud, spatial transcriptomics, and image generation.
tags:
  - "ICML2025"
  - "Computational Biology"
  - "Noise-Conditioned Graph Networks"
  - "Diffusion Models"
  - "Flow Matching"
  - "Dynamic Message Passing"
  - "Graph Coarsening"
  - "3D Point Cloud Generation"
date: 2026-05-08
content_hash: c0398685fd1be91a
---

# Geometric Generative Modeling with Noise-Conditioned Graph Networks

**Conference**: ICML2025  
**arXiv**: [2507.09391](https://arxiv.org/abs/2507.09391)  
**Code**: [GitHub](https://github.com/peterpaohuang/ncgn)  
**Area**: Computational Biology  
**Keywords**: Noise-Conditioned Graph Networks, Diffusion Models, Flow Matching, Dynamic Message Passing, Graph Coarsening, 3D Point Cloud Generation

## TL;DR

Proposes Noise-Conditioned Graph Networks (NCGNs) to dynamically adjust message passing range and graph resolution in GNN architectures based on noise levels: long-range connections with low resolution at high noise levels, and local connections with high resolution at low noise levels, outperforming static architecture baselines in 3D point cloud, spatial transcriptomics, and image generation.

## Background & Motivation

### Background

**Background**: Importance of geometric graph generation: Applications in 3D point clouds, molecular structures, and spatial genomics all involve generating graph structures with spatial information.

### Limitations of Prior Work

**Limitations of Prior Work**: Static limitations of existing GNNs: GNNs in generative flow models use a fixed kNN or radius graph throughout the entire denoising process, ignoring the impact of noise levels on graph signals.

### Key Challenge

**Key Challenge**: Information-theoretic motivation: Theoretical analysis shows that as noise increases, recovering the signal requires information from more distant neighbors, and the graph can be represented at a lower resolution.

## Method

### Theoretical Foundation

- **Lemma 3.1 (Mutual Information Formula)**: Provides the analytical expression of the mutual information between the original node features $x_1^{(i)}$ and the aggregated noisy features $Y_t^{(i,r)}$ within a radius $r$.
- **Theorem 3.2 (Increasing Range)**: When the SNR decreases, there exists a larger radius $r_2 > r_1$ such that $I(x_1^{(i)}, Y_{c_2}^{(r_2)}) > I(x_1^{(i)}, Y_{c_2}^{(r_1)})$.
- **Proposition 3.3 (Position Noise)**: As noise increases, the expected distance between originally close nodes increases, thus necessitating a larger message passing radius.

### Dynamic Message Passing (DMP)

Given boundary conditions $(r_0, s_0)$ and $(r_1, s_1)$ and an adaptive scheduling function $f$:

$$r_t, s_t = f(t, r_0, r_1, s_0, s_1)$$

Constraints:
- **Monotonicity**: $t' < t$ (higher noise) $\Rightarrow r' \geq r, s' \leq s$
- **Boundary Consistency**: $f(0) = (r_0, s_0)$, $f(1) = (r_1, s_1)$

Steps per iteration:
1. **Graph Coarsening**: Aggregates $N$ nodes into $s_t$ supernodes (voxel clustering / average pooling).
2. **Connection Construction**: Constructs a kNN/radius graph using $r_t$.
3. **Message Passing**: Performs GCN/GAT message passing on the coarsened graph.
4. **Uncoarsening**: Maps the information back to the original nodes.

### Complexity

When $r_t \cdot s_t = r_1 N$, the message passing maintains linear time complexity throughout the entire generation process.

## Key Experimental Results

### Main Results 1: 3D Point Cloud Generation (ModelNet40)

| Method | GCN $\mathcal{W}_2$ (×10⁻²) | GAT $\mathcal{W}_2$ (×10⁻²) |
|---|---|---|
| Random | 8.624 | 8.624 |
| KNN | 5.882 | 5.598 |
| Long-Short Range | 4.315 | 4.741 |
| **DMP** | **4.215** | **4.263** |

Average improvement of 16.15%.

### Main Results 2: Spatial Transcriptomics

- DMP comprehensively outperforms kNN and fully-connected baselines on GCNs, and achieves comparable performance on GATs.

### Main Results 3: Image Generation (DiT-DMP)

| | DiT | DiT-DMP |
|---|---|---|
| FID↓ | 84.051 | **63.983** |
| IS↑ | 16.735 | **24.681** |
| Precision↑ | 0.296 | **0.446** |

Significant improvement is achieved with only two lines of code modifications (FlexiViT dynamic patching + neighborhood attention).

### Ablation Study: Scheduling Function Selection

- Exponential scheduling is the best ($\mathcal{W}_2 = 4.263 \times 10^{-2}$); linear and ReLU schedulers also outperform the baseline, whereas the logarithmic scheduler performs worse than the baseline.

## Highlights & Insights

1. **Theoretical Grounding from an Information-Theoretic Perspective**: SNR decrease $\Rightarrow$ optimal radius increase $\Rightarrow$ graph resolution can be reduced, providing a theoretical foundation for dynamic architectures.
2. **Simple yet Powerful Implementation**: DMP can be integrated into existing models (e.g., DiT) with minimal code modifications.
3. **Linear Complexity**: By balancing the expansion of the connection range through coarsening, it avoids the quadratic complexity of fully-connected structures.
4. **Empirical Evidence of Attention Weights**: The trained GAT attention distribution indeed varies with the noise levels, validating the theoretical predictions.

## Limitations & Future Work

- The scheduling function $f$ is predefined and the optimal scheduling is not learned.
- Only the connection range and resolution are adjusted, while other adjustable dimensions (number of layers, width, type of message passing) remain unexplored.
- The coarsening strategy (voxel clustering) may not be the optimal choice for all scenarios.
- The theoretical assumptions (correlation structure) of Theorem 3.2 may not strictly hold in practice.

## Related Work & Insights

- **Flow Matching (Lipman et al., 2022; Tong et al., 2023)**: DMP serves as a modular plug-in within this framework.
- **DiT (Peebles & Xie, 2023)**: State-of-the-art in image generation; the DiT-DMP in this work demonstrates rapid integration with existing models.
- **Torsional Diffusion (Jing et al., 2022)**: A representative method using fixed radius graphs, which this work aims to improve upon.
- **Insights**: The concept of dynamic architectures can be generalized to more generative flow tasks, such as molecular generation and image super-resolution.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Graph Generative Pre-trained Transformer (G2PT)](graph_generative_pre-trained_transformer.md)
- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](../../NeurIPS2025/computational_biology/generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [\[ICML 2025\] Flexibility-conditioned Protein Structure Design with Flow Matching](flexibility-conditioned_protein_structure_design_with_flow_matching.md)
- [\[NeurIPS 2025\] Random Search Neural Networks for Efficient and Expressive Graph Learning](../../NeurIPS2025/computational_biology/random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)
- [\[NeurIPS 2025\] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs](../../NeurIPS2025/computational_biology/towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)

</div>

<!-- RELATED:END -->
