---
title: >-
  [Paper Note] Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework
description: >-
  [NeurIPS 2025][3D Vision][Topological Neural Networks] This paper proposes Copresheaf Topological Neural Networks (CTNNs), which leverage the algebraic-topological notion of copresheaves to define directional…
tags:
  - "NeurIPS 2025"
  - "3D Vision"
  - "Topological Neural Networks"
  - "Copresheaf"
  - "Combinatorial Complex"
  - "Message Passing"
  - "Anisotropic Representation Learning"
date: 2026-05-08
content_hash: fafe389d02948055
---

# Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework

**Conference**: NeurIPS 2025
**arXiv**: [2505.21251](https://arxiv.org/abs/2505.21251)  
**Code**: None  
**Area**: 3D Vision
**Keywords**: Topological Neural Networks, Copresheaf, Combinatorial Complex, Message Passing, Anisotropic Representation Learning

## TL;DR
This paper proposes Copresheaf Topological Neural Networks (CTNNs), which leverage the algebraic-topological notion of copresheaves to define directional, heterogeneous message passing on combinatorial complexes. The framework unifies CNNs, GNNs, Transformers, Sheaf Neural Networks, and Topological Neural Networks as special cases, and surpasses conventional baselines on physics simulation, graph classification, and higher-order complex classification tasks.

## Background & Motivation
**Background**: Deep learning has achieved remarkable success by exploiting structural priors — CNNs for images, Transformers for sequences, and GNNs for graphs. Nevertheless, designing a unified architecture that generalizes across domains, particularly for complex, irregular, and multi-scale structured data, remains a persistent challenge.

**Limitations of Prior Work**: CNNs cannot capture local irregularities; GNNs rely on homophily assumptions and suffer from over-smoothing as depth increases; Transformers handle long-range dependencies well but assume a homogeneous embedding space, incur quadratic complexity, and lack built-in anisotropy and variable local structure. Existing Sheaf Neural Networks (SNNs), based on cellular sheaves, require intermediate edge feature spaces and symmetry constraints on restriction maps, limiting their expressive power.

**Key Challenge**: Prevailing models universally assume a single global latent space and isotropic information propagation, which is fundamentally misaligned with the heterogeneous, directional, and hierarchical nature of real-world data. A unified framework that natively encodes diverse local behaviors, respects directional coupling, and propagates information across scales is needed.

**Goal**: To construct a unified deep learning framework based on copresheaves, assigning each local region its own independent feature space and learnable information-transfer maps, thereby naturally supporting multi-scale, anisotropic, and heterogeneous message passing.

**Key Insight**: The paper exploits the algebraic-topological concept of copresheaves — each vertex is assigned an independent vector space (stalk), and each directed edge carries a learnable linear map — thereby avoiding the symmetry constraints and edge feature space requirements of sheaves. Message passing is built upon combinatorial complexes, the most general topological structure of this type.

**Core Idea**: Replace sheaves with copresheaves as the foundation for message passing, defining a directional learnable map $\rho_{y \to x}$ on each edge, and unifying CNNs, GNNs, Transformers, SNNs, and TNNs as special cases of CTNNs.

## Method

### Overall Architecture
CTNNs are organized into three levels:
1. **Combinatorial Complex (CC)**: The underlying topological structure, generalizing graphs, simplicial complexes, cell complexes, and hypergraphs. A CC consists of a finite vertex set $S$, a collection of cells $X \subset \mathcal{P}(S)$, and a rank function $\mathrm{rk}$ such that set inclusion implies non-decreasing rank.
2. **Neighborhood-Induced Copresheaf**: A neighborhood function $N$ on the CC induces a directed graph $G_N$, over which a copresheaf is defined — each vertex (cell) $x$ is assigned a vector space $\mathcal{F}(x)$, and each directed edge $y \to x$ is assigned a learnable linear map $\rho_{y \to x}: \mathcal{F}(y) \to \mathcal{F}(x)$.
3. **Copresheaf Message Passing**: Higher-order message passing is defined within the copresheaf framework, supporting cross-rank, multi-neighborhood aggregation.

### Key Designs
1. **Copresheaf Neighborhood Matrix (CNM)**: Generalizes the conventional 0/1 neighborhood matrix by replacing scalar entries with copresheaf maps $\rho_{z_i \to y_j} \in \mathrm{Hom}(\mathcal{F}(z_i), \mathcal{F}(y_j))$. Copresheaf adjacency matrices (CAM, encoding shared co-cell relations) and copresheaf incidence matrices (CIM, encoding containment relations) are further defined to drive different types of topological message passing.

2. **Copresheaf Message Passing Neural Network (CMPNN)**: The core update rule is
$$h_x^{(l+1)} = \beta\!\left(h_x^{(l)},\; \bigoplus_{(y \to x) \in E} \alpha\!\left(h_x^{(l)},\, \rho_{y \to x}\, h_y^{(l)}\right)\right).$$
Unlike standard MPNNs, $\rho_{y \to x}$ is an independently learnable linear map per edge, enabling directional, anisotropic feature transformation. It is theoretically shown that SNNs are a special case of CMPNNs on bidirectional graphs (Theorem 1: $\rho_{y \to x} = \mathcal{F}_{x \trianglelefteq e}^T \circ \mathcal{F}_{y \trianglelefteq e}$).

3. **Higher-Order Copresheaf Message Passing**: Multi-neighborhood aggregation on CCs is defined as
$$h_x^{(l+1)} = \beta\!\left(h_x^{(l)},\; \bigotimes_k \bigoplus_{y \in \mathcal{N}_k(x)} \alpha_{\mathcal{N}_k}\!\left(h_x^{(l)},\, \rho_{y \to x}^{\mathcal{N}_k}\!\left(h_y^{(l)}\right)\right)\right),$$
where $\oplus$ is a permutation-invariant aggregator and $\otimes$ combines information from different neighborhoods. This subsumes simplicial, cellular, and Hodge-stable message passing architectures as special cases.

4. **Copresheaf Transformer (CT)**: Integrates copresheaf maps into self-attention. Value vectors are transformed by copresheaf maps prior to aggregation:
$$m_x = \sum_{y \in \mathcal{N}_k(x)} a_{xy}\, \rho_{y \to x}(v_y),$$
where attention weights $a_{xy}$ are computed via standard query–key dot products. The framework reduces to standard dot-product attention when $\rho = I$. Two variants are proposed: CT-FC (directly learning a full $d \times d$ transformation matrix) and CT-SharedLoc (shared transformation with local scalar modulation).

### Loss & Training
- **Physics Simulation**: MSE loss, AdamW optimizer, learning rate $10^{-3}$, cosine scheduling.
- **Graph Classification**: Negative log-likelihood loss, Adam optimizer, learning rate $0.01$.
- **CC Classification**: Cross-entropy loss, Adam optimizer, learning rate $10^{-3}$.
- **Copresheaf Map Parameterization**: $\rho_{ij} = I + \Delta_{ij}$, where $\Delta_{ij} = \tanh(\mathrm{Linear}([h_i;\, h_j]))$ (residual form; $I$ denotes the identity matrix).

## Key Experimental Results

### Main Results: Physics Simulation (Transformer vs. Copresheaf Transformer)

| Task | Classical MSE | Copresheaf MSE | Improvement |
|------|--------------|----------------|-------------|
| Heat (Diffusion) | $2.64 \times 10^{-4} \pm 3.50 \times 10^{-5}$ | $\mathbf{9.00 \times 10^{-5} \pm 7.00 \times 10^{-6}}$ | >50% ↓ |
| Advection | $3.52 \times 10^{-4} \pm 7.70 \times 10^{-5}$ | $\mathbf{1.20 \times 10^{-4} \pm 1.20 \times 10^{-5}}$ | >50% ↓ |
| Unsteady Stokes | $1.75 \times 10^{-2} \pm 1.32 \times 10^{-3}$ | $\mathbf{1.48 \times 10^{-2} \pm 1.48 \times 10^{-4}}$ | ~15% ↓ |

### Ablation Study: Graph Classification (MUTAG Dataset)

| Model | Accuracy |
|-------|---------|
| GCN | $0.674 \pm 0.014$ |
| **CopresheafGCN** | $\mathbf{0.721 \pm 0.035}$ |
| GraphSAGE | $0.689 \pm 0.022$ |
| **CopresheafSage** | $\mathbf{0.732 \pm 0.029}$ |
| GIN | $0.700 \pm 0.039$ |
| **CopresheafGIN** | $\mathbf{0.724 \pm 0.021}$ |

### Supplementary Results: CC Classification

| Model | Accuracy |
|-------|---------|
| Classic Transformer | $0.940 \pm 0.014$ |
| CT-FC | $0.955 \pm 0.009$ |
| **CT-SharedLoc** | $\mathbf{0.970 \pm 0.010}$ |

### Key Findings
1. Copresheaf attention reduces MSE by over 50% on heat diffusion and advection tasks, with markedly improved cross-seed stability.
2. On MUTAG graph classification, copresheaf-augmented GNNs consistently outperform their vanilla counterparts; CopresheafSage achieves the largest relative gain (+4.3% absolute accuracy).
3. CT-SharedLoc's shared-transformation-plus-local-modulation strategy performs best on CC classification, underscoring the importance of balancing global shared structure with local adaptability.
4. All GNN backbones (GCN, SAGE, GIN) benefit from copresheaf augmentation, validating the generality of the framework.

## Highlights & Insights
1. **Strong Theoretical Unification**: A single mathematical concept — the copresheaf — unifies five major architecture families (CNNs, GNNs, Transformers, SNNs, TNNs), with rigorous proofs that each is a special case of CTNNs.
2. **Breaking the Single Latent Space Assumption**: Conventional deep learning assumes all nodes share a common latent space; CTNNs allow each node to have an independent latent space dimension with directional maps, better reflecting the reality of heterogeneous data.
3. **Simplification from Sheaf to Copresheaf**: Copresheaves eliminate the complexity of sheaf architectures — specifically the requirement for edge feature spaces and symmetric restriction maps — relying solely on direct vertex-to-vertex maps for a cleaner implementation.
4. **Residual Parameterization**: The design $\rho_{ij} = I + \Delta_{ij}$ guarantees graceful degradation to standard models as $\Delta \to 0$, ensuring training stability and ease of optimization.

## Limitations & Future Work
1. **Computational Overhead**: Per-edge copresheaf maps introduce additional parameters and computation; validation has so far been limited to small- and medium-scale datasets. Scaling to large graphs and high-resolution data remains a primary challenge.
2. **Limited Experimental Scale**: Physics simulation experiments use synthetic data (100–200 samples), and graph classification is evaluated only on MUTAG (188 graphs); validation on large-scale benchmarks such as OGB is absent.
3. **Restriction to Linear Maps**: Current copresheaf maps $\rho_{y \to x}$ are linear transformations; the expressive power and theoretical properties of nonlinear copresheaf maps have not been explored.
4. **Dynamic Settings Not Addressed**: The current framework assumes static topological structures and does not discuss graphs or complexes that evolve over time.
5. **Interpretability**: The geometric and topological meaning of the learned copresheaf maps has not been thoroughly analyzed.

## Related Work & Insights
- **Sheaf Neural Networks** (Hansen & Ghrist 2019; Bodnar et al. 2022): The direct precursors of CTNNs. This paper proves that SNNs are special cases of CMPNNs on bidirectional graphs, extending the sheaf diffusion framework.
- **Topological Neural Networks** (Hajij et al. 2023): Message passing architectures on combinatorial complexes; CTNNs enhance the expressive power of TNNs via copresheaf maps.
- **Graph Transformers** (graph extensions of Vaswani 2017): The copresheaf Transformer variant replaces the identity map in standard attention with a learnable directional map.
- **Geometric Deep Learning** (Bronstein et al. 2017): Articulated the vision of a unified framework; CTNNs realize a more formally grounded unification at the categorical level.

## Rating
- ⭐⭐⭐⭐ The theoretical framework is exceptionally elegant and unified, constructing a meta-framework for deep learning via copresheaves and combinatorial complexes. Experiments are limited in scale but consistent in outcome, with a clear research direction; practical scalability remains to be verified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CosmoBench: A Multiscale, Multiview, Multitask Cosmology Benchmark for Geometric Deep Learning](cosmobench_a_multiscale_multiview_multitask_cosmology_benchmark_for_geometric_de.md)
- [\[NeurIPS 2025\] Learning Neural Exposure Fields for View Synthesis](learning_neural_exposure_fields_for_view_synthesis.md)
- [\[NeurIPS 2025\] High Resolution UDF Meshing via Iterative Networks](high_resolution_udf_meshing_via_iterative_networks.md)
- [\[CVPR 2026\] Node-RF: Learning Generalized Continuous Space-Time Scene Dynamics with Neural ODE-based NeRFs](../../CVPR2026/3d_vision/node-rf_learning_generalized_continuous_space-time_scene_dynamics_with_neural_od.md)
- [\[NeurIPS 2025\] Neural Green's Functions](neural_greens_functions.md)

</div>

<!-- RELATED:END -->
