---
title: >-
  [Paper Note] Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations
description: >-
  [ICML2025][Graph Learning][Graph Neural Networks] Graph message passing is modeled as a system of hyperbolic partial differential equations. It is proven that the solution space of node features is spanned by the eigenvectors of the Laplacian matrix, thereby embedding topological structure information into node representations. A bridge to spectral GNNs is established through polynomial approximation to enhance their performance.
tags:
  - "ICML2025"
  - "Graph Learning"
  - "Graph Neural Networks"
  - "Hyperbolic Partial Differential Equations"
  - "Spectral Graph Convolution"
  - "Polynomial Filters"
  - "Message Passing"
date: 2026-05-08
content_hash: b581482721205af3
---

# Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations

**Conference**: ICML2025  
**arXiv**: [2505.23014](https://arxiv.org/abs/2505.23014)  
**Code**: [GitHub](https://github.com/YueAWu/Hyperbolic-GNN)  
**Area**: Graph Learning  
**Keywords**: Graph Neural Networks, Hyperbolic Partial Differential Equations, Spectral Graph Convolution, Polynomial Filters, Message Passing

## TL;DR

Graph message passing is modeled as a system of hyperbolic partial differential equations. It is proven that the solution space of node features is spanned by the eigenvectors of the Laplacian matrix, thereby embedding topological structure information into node representations. A bridge to spectral GNNs is established through polynomial approximation to enhance their performance.

## Background & Motivation

Traditional GNNs learn the topological features of graphs through message passing, which essentially transforms node features to the spectral domain via Fourier transform for graph convolution, and then transforms them back to the spatial domain via inverse Fourier transform. However, the limitation of prior work is: **the spatial domain transformed back is independent of the graph topology**—the bases of node feature vectors are standard unit vectors $\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_d$ in Euclidean space. These bases carry no topological information, making it difficult to guarantee that the learned features truly encode the graph structure.

Existing differential equation modeling methods (such as GRAND, GraphCON) provide interpretable ways to embed nodes into spaces with specific properties. This paper further poses the question: can a dynamical system be found such that the bases of its solution space naturally describe the graph topology?

## Method

### 1. Hyperbolic PDE System on Graphs

Discretize the hyperbolic PDE from continuous domains onto graphs. For node $v_i$ on any dimension $l$, the gradient on the graph is defined as the difference in neighbor features, and the divergence is defined as the sum of all neighbor feature differences:

$$\nabla x_{il} := x_{il} - x_{jl}, \quad \nabla \cdot \nabla x_{il} = \sum_{v_j \in \mathcal{N}(v_i)} (x_{il} - x_{jl})$$

Combining equations for all nodes and dimensions yields the **hyperbolic PDE system** in matrix form:

$$\frac{\partial^2 \mathbf{X}}{\partial t^2} = a^2 \hat{\mathbf{L}} \mathbf{X}$$

where $\hat{\mathbf{L}}$ is the Laplacian matrix (taking different normalized forms), and $a$ is the propagation speed coefficient.

### 2. Derivation of the Solution Space (Core Theory)

**Theorem 3.1**: By variable substitution, the second-order PDE is transformed into a first-order system of homogeneous linear ODEs with constant coefficients, proving the existence of a solution space determined by the fundamental solution matrix.

**Theorem 3.3** (Key): The eigenvalues of the coefficient matrix $\mathbf{C} = \text{diag}(\mathbf{I}, a^2\hat{\mathbf{L}})$ are $\lambda'_1 = \dots = \lambda'_n = 1$ and $\lambda'_{n+k} = a^2 \hat{\lambda}_k$, with corresponding eigenvectors derived from the eigenvectors $\hat{\mathbf{u}}_k$ of $\hat{\mathbf{L}}$. Therefore:

$$\boldsymbol{\Phi}(t) = [e^{\lambda'_1 t}\mathbf{u}'_1, \dots, e^{\lambda'_{2n} t}\mathbf{u}'_{2n}]$$

**Implication**: Under this paradigm, node features are automatically expanded into linear combinations of the Laplacian eigenvector basis. Information propagates along the directions of the eigenvectors, naturally encoding the topological structure.

### 3. Polynomial Approximation

Direct solving requires eigendecomposition ($O(n^3)$ complexity) and suffers from a lack of flexibility when regulated solely by $a$. A polynomial approximation of the solution space is introduced:

$$\frac{\partial^2 \mathbf{X}}{\partial t^2} \approx \sum_{k=0}^{K} \theta_{tk} p_k(\mathbf{L}) \mathbf{X}$$

Taking Chebyshev polynomials as an example: $\frac{\partial^2 \mathbf{X}}{\partial t^2} = \sum_{k=0}^{K-1} T_k(\mathbf{L} - \mathbf{I}) \mathbf{X} \mathbf{W}_k$

### 4. Numerical Solving via Forward Euler Method

Discretizing time with the forward Euler method, initializing $\mathbf{X}(t_0) = \phi_0(\mathbf{X})$ and $\dot{\mathbf{X}}(t_0) = \phi_1(\mathbf{X})$, the iterative formula for node features is obtained:

$$\mathbf{X}(t_{m+1}) = (2\mathbf{I} + \tau^2 P(\mathbf{L}, t_m)) \mathbf{X}(t_m) - \mathbf{X}(t_{m-1})$$

This formula can be directly integrated with any polynomial spectral GNN (such as ChebNet, BernNet, JacobiConv), forming the **Hyperbolic-PDE enhancement paradigm**.

## Key Experimental Results

### Node Classification (Table 4: Comparison with Spectral GNN Baselines)

| Method | Cora | CiteSeer | PubMed | Actor |
|------|------|----------|--------|-------|
| GCN | 87.14 | 79.86 | 86.74 | 33.23 |
| ChebNet | 86.67 | 79.11 | 87.95 | 37.61 |
| BernNet | 88.95 | 80.09 | 88.48 | 41.79 |
| JacobiConv | 88.98 | 80.78 | 89.62 | 41.17 |
| NFGNN | 89.82 | 80.56 | 89.89 | 40.62 |
| UniFilter | 89.49 | 81.39 | **91.44** | 40.84 |
| **Ours** | **90.82** | **81.88** | 91.36 | 42.03 |

### Enhancement Effects on Existing Spectral GNNs (Selected Table 5)

| Baseline → Enhanced | Cora | PubMed | Texas | Cornell |
|--------------|------|--------|-------|---------|
| SGC → Hyp-SGC | 85.48→**86.22** | 85.36→**88.25** | 81.31→**93.61** | 72.62→**91.28** |
| APPNP → Hyp-APPNP | 88.14→**90.07** | 88.12→**90.79** | 90.98→91.31 | **91.81**→87.45 |
| GPR → Hyp-GPR | 88.57→**90.82** | 88.46→**91.36** | 92.95→**93.28** | 91.37→**92.77** |
| ChebNet → Hyp-Cheb | 86.67→**89.38** | 87.95→**90.50** | 86.22→**93.93** | 83.93→**91.06** |

The improvements on heterophilous graphs (Texas, Cornell) are particularly significant: SGC improves by **+18.66%** on Cornell, and ChebNet improves by **+7.71%** on Texas.

## Highlights & Insights

1. **Theoretical Elegance**: Starting from hyperbolic PDEs, it rigorously proves that the solution space is spanned by Laplacian eigenvectors, providing a mathematical foundation for "why message passing can capture topology."
2. **Plug-and-Play Paradigm**: It is not an independent model, but a general framework that can enhance any spectral GNN. Experiments have verified its effectiveness on SGC, APPNP, GPR, ChebNet, BernNet, etc.
3. **Significant Advantage on Heterophilous Graphs**: Traditional spectral GNNs perform poorly on heterophilous graphs, whereas the hyperbolic PDE paradigm brings a substantial improvement of 5-18%.
4. **Physical Intuition**: Hyperbolic PDEs describe wave propagation, where information propagates along the eigenvector directions with a finite speed. This is more in line with the locality of message passing than heat equation (diffusion) models.

## Limitations & Future Work

1. **Limited Gain on Homophilous Graphs**: The improvement on homophilous graphs such as Cora and CiteSeer is only 1-2%, contrasting with the substantial gains on heterophilous graphs.
2. **Partial Failure of Enhancements**: Hyperbolic-BernNet performs worse than the original BernNet on certain datasets (e.g., Cora 88.52→88.34), indicating that not all polynomial bases are compatible with the hyperbolic PDE paradigm.
3. **Hyperparameter Dependency**: Extra hyperparameters are introduced, such as propagation speed $a$, time step size $\tau$, and number of time steps $m$.
4. **Unverified on Large-Scale Graphs**: The largest dataset, DeezerEurope, has only 28,000 nodes, and scalability has not been tested on million-scale graphs.
5. **Limited to Node Classification**: Effectiveness has not been validated on other tasks such as link prediction and graph classification.

## Related Work & Insights

- **GNNs from the Perspective of Differential Equations**: GRAND (diffusion equation), GraphCON (oscillation equation); this paper provides a new perspective using hyperbolic PDEs.
- **Spectral Graph Filters**: ChebNet → BernNet → JacobiConv → ChebNetII; this paper provides a unified enhancement framework.
- **Insights**: Can other types of PDEs (such as elliptic or parabolic) be used to construct solution spaces with different properties? What kind of graph tasks do different PDE types suit?

## Rating
- Novelty: ⭐⭐⭐⭐ (Hyperbolic PDE modeling of message passing with rigorous proof of the solution space structure; theoretically novel)
- Experimental Thoroughness: ⭐⭐⭐⭐ (10 datasets + multiple baseline enhancements, but lacks large-scale graphs and other tasks)
- Writing Quality: ⭐⭐⭐⭐ (Clear theoretical derivations and good physical intuition, but with dense notation)
- Value: ⭐⭐⭐⭐ (Plug-and-play framework holds practical value, though the gains on homophilous graphs are limited)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Hyperbolic Continuous Structural Entropy for Hierarchical Clustering](../../AAAI2026/graph_learning/hyperbolic_continuous_structural_entropy_for_hierarchical_clustering.md)
- [\[ICML 2025\] A Cognac Shot To Forget Bad Memories: Corrective Unlearning for Graph Neural Networks](a_cognac_shot_to_forget_bad_memories_corrective_unlearning_for_graph_neural_netw.md)
- [\[ICLR 2026\] Bridging ML and Algorithms: Comparison of Hyperbolic Embeddings](../../ICLR2026/graph_learning/bridging_ml_and_algorithms_comparison_of_hyperbolic_embeddings.md)
- [\[ICLR 2026\] PRISM: Partial-label Relational Inference with Spatial and Spectral Cues](../../ICLR2026/graph_learning/prism_partial-label_relational_inference_with_spatial_and_spectral_cues.md)
- [\[ICML 2025\] Open Your Eyes: Vision Enhances Message Passing Neural Networks in Link Prediction](open_your_eyes_vision_enhances_message_passing_neural_networks_in_link_predictio.md)

</div>

<!-- RELATED:END -->
