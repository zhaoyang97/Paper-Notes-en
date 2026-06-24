---
title: >-
  [Paper Note] EvoMesh: Adaptive Physical Simulation with Hierarchical Graph Evolutions
description: >-
  [ICML 2025][Graph Learning][Physical Simulation] EvoMesh proposes a fully differentiable hierarchical graph evolution framework that adaptively constructs a multi-scale graph hierarchy evolving over time based on physical inputs using Anisotropic Message Passing (AMP) and Gumbel-Softmax-based differentiable node selection (DiffSELECT), outperforming fixed-hierarchy methods by an average of approximately 20% across five physical simulation benchmarks.
tags:
  - "ICML 2025"
  - "Graph Learning"
  - "Physical Simulation"
  - "Graph Neural Networks"
  - "Hierarchical Graph Structure"
  - "Anisotropic Message Passing"
  - "Differentiable Node Selection"
date: 2026-05-08
content_hash: ab49e1d77cf4efa9
---

# EvoMesh: Adaptive Physical Simulation with Hierarchical Graph Evolutions

**Conference**: ICML 2025  
**arXiv**: [2410.03779](https://arxiv.org/abs/2410.03779)  
**Code**: [https://hbell99.github.io/evo-mesh/](https://hbell99.github.io/evo-mesh/)  
**Area**: 3D Vision  
**Keywords**: Physical Simulation, Graph Neural Networks, Hierarchical Graph Structure, Anisotropic Message Passing, Differentiable Node Selection

## TL;DR

EvoMesh proposes a fully differentiable hierarchical graph evolution framework that adaptively constructs a multi-scale graph hierarchy evolving over time based on physical inputs using Anisotropic Message Passing (AMP) and Gumbel-Softmax-based differentiable node selection (DiffSELECT), outperforming fixed-hierarchy methods by an average of approximately 20% across five physical simulation benchmarks.

## Background & Motivation

Mesh-based physical simulation using Graph Neural Networks (GNNs) has achieved significant success. The core mechanism is message passing—encoding physical quantities on mesh nodes and updating them over time by aggregating neighborhood information. However, for large-scale mesh systems, local message passing requires a high number of layers to propagate long-range information, which is computationally expensive.

Existing solutions primarily employ multi-scale hierarchical graph structures to create shortcuts for long-range information propagation (such as BSMS-GNN, HCMT, etc.). However, these methods suffer from two core limitations:

**Fixed graph hierarchy**: Constructed during the preprocessing stage via heuristic node selection, utilizing the same graph hierarchy for all input sequences. It fails to adapt to different physical contexts (e.g., in turbulent flows, small variations in initial conditions lead to completely different subsequent dynamics).

**Isotropic message passing**: Aggregation functions treat all neighbor contributions equally, ignoring the directional nature of feature propagation in physical systems (such as the directional flow of fluid around obstacles in CylinderFlow).

The core idea of EvoMesh is: **to let the graph hierarchy adaptively evolve with data and time, while enabling message passing with directional discrimination capabilities**. This simultaneously addresses the two bottlenecks of "fixed structures" and "isotropic aggregation".

## Method

### Overall Architecture

EvoMesh adopts an Encode-Process-Decode pipeline design:
- **Encoder**: Maps physical quantities of the raw mesh to a latent feature space.
- **Processor**: Performs message passing on adaptively constructed multi-scale graph hierarchies $\mathcal{G}_1, \mathcal{G}_2, \ldots, \mathcal{G}_L$. Each layer processes features using an AMP layer while constructing the next-level coarse-grained graph adaptively via DiffSELECT. REDUCE/EXPAND operations propagate information between layers.
- **Decoder**: Decodes the processed features into physical quantity predictions for the next time step.

Distinct from all prior work, EvoMesh is **the only model that simultaneously exhibits all four properties: dynamic hierarchy, adaptive hierarchy, anisotropic intra-layer propagation, and learnable inter-layer propagation**.

### Key Designs

1. **Anisotropic Message Passing (AMP) Layer**: Standard GNN aggregation functions $\psi(\cdot)$ simply sum all neighbor edge features, failing to differentiate contributions from different directions. AMP introduces a learnable importance weight network $\phi^w$ to predict directional weights based on edge features and features of nodes at both ends:

$$w_{ij} = \phi^w(\mathbf{e}_{ij}, \mathbf{v}_i, \mathbf{v}_j), \quad \alpha_{ij} = \frac{\exp(w_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(w_{ik})}$$

Nodes are then updated using weighted edge features: $\hat{\mathbf{v}}_i = \phi^v(\mathbf{v}_i, \sum_{v_j \in \mathcal{N}_{v_i}} \alpha_{ij} \hat{\mathbf{e}}_{ij})$. This enables the model to assign different contribution weights to different neighbors within the same neighborhood, naturally aligning with directionally non-uniform dynamic patterns in physical systems. $\phi^w$ is implemented using an MLP.

2. **Differentiable Node Selection (DiffSELECT)**: This is the core to achieving dynamic and adaptive hierarchy construction. The node update function $\phi^v$ of the AMP layer additionally outputs a two-dimensional probability vector $\boldsymbol{\pi}_i^l = (\pi_{i,0}^l, \pi_{i,1}^l)$, representing the probability of node $v_i$ being dropped or retained in the next layer. An approximate hard selection soft decision is then obtained via Gumbel-Softmax sampling:

$$z_{i,k}^l = \frac{\exp((\log \pi_{i,k}^l + g_{i,k}^l) / \tau)}{\sum_{k'=0}^{1} \exp((\log \pi_{i,k'}^l + g_{i,k'}^l) / \tau)}$$

where $g_{i,k}^l$ is Gumbel noise and $\tau$ is the temperature parameter. The Straight-through Gumbel-Softmax estimator ensures the differentiability of the entire process, allowing node selection to be trained end-to-end. A temperature annealing strategy is adopted during training, encouraging exploration of different hierarchical structures in the early stages and gradually refining the selection in the later stages.

Edges of the coarse-grained graph enhance connectivity via K-hop connections to prevent graph splitting issues caused by downsampling. In experiments, K=2 yields the best performance.

3. **Learnable Inter-layer Feature Propagation**: Prior works use non-parametric aggregation based on node degrees between layers (REDUCE/EXPAND). EvoMesh directly reuses the weights $\alpha_{ij}^l$ calculated by the AMP layer for inter-layer propagation:

    - **REDUCE** (Downsampling): $\mathbf{v}_i^{l+1} = \sum_{j \in \mathcal{N}_i} \alpha_{ij}^l \mathbf{v}_j^l$
    - **EXPAND** (Upsampling): $\tilde{\mathbf{v}}_i^l = \sum_{j \in \mathcal{N}_i} \mathbf{v}_j^{l+1} \alpha_{ij}^l$
    - **FeatureMixing**: Performs another AMP message passing on upsampled features, followed by skip-connection fusion with original intra-layer features to alleviate feature misalignment caused by upsampling: $\bar{\mathbf{v}}_i^l = \mathbf{v}_i^l + \text{AMP}(\tilde{\mathbf{v}}_i^l, \{\tilde{\mathbf{v}}_j^l\}_{j \in \mathcal{N}_i}, \{\mathbf{e}_{ij}^l\}_{j \in \mathcal{N}_i})$

### Loss & Training

- Uses one-step supervision, where the loss is the L2 loss between the ground truth and the prediction of the next step.
- Trained with the Adam optimizer for 1M steps, with exponential learning rate decay from $10^{-4}$ to $10^{-6}$ (over the first 500K steps).
- Autoregressive rollout is utilized for long-term prediction during inference.

## Key Experimental Results

### Main Results

Comparison on five standard physical simulation benchmarks (RMSE, lower is better):

| Dataset | Metric | EvoMesh | Prev. SOTA | Gain |
|--------|------|---------|----------|------|
| CylinderFlow | RMSE-1 (×10⁻²) | 0.1568 | 0.1733 (Eagle) | 9.53% |
| CylinderFlow | RMSE-All (×10⁻²) | 6.571 | 16.98 (BSMS) | 61.3% |
| Airfoil | RMSE-1 (×10⁻²) | 41.41 | 48.62 (HCMT) | 14.8% |
| Airfoil | RMSE-All (×10⁻²) | 2002 | 2080 (Lino) | 3.75% |
| FlyingFlag | RMSE-1 (×10⁻²) | 0.3049 | 0.3459 (MGN) | 11.9% |
| FlyingFlag | RMSE-All (×10⁻²) | 76.16 | 90.32 (HCMT) | 15.7% |
| DeformingPlate | RMSE-1 (×10⁻²) | 0.0282 | 0.0291 (Lino) | 3.10% |
| DeformingPlate | RMSE-All (×10⁻²) | 1.296 | 1.811 (BSMS) | 28.5% |

Time-varying mesh (FoldingPaper) scenario, which other hierarchical methods cannot handle due to their reliance on pre-computation:

| Model | RMSE-1 (×10⁻²) | RMSE-All (×10⁻²) |
|------|----------------|------------------|
| MGN | 0.0618 | 24.08 |
| EvoMesh | 0.0544 | 7.412 |
| Gain | 12.0% | **69.2%** |

### Ablation Study

| Configuration | Key Component Differences | Description |
|------|------------|------|
| BSMS-GNN (Baseline) | Static Hierarchy + Isotropic + Unlearnable Inter-layer | Baseline method |
| M1: Static-Aniso-Unlearnable | Static hierarchy + AMP + Unlearnable Inter-layer | Improvement with AMP alone |
| M2: Static-Aniso-Learnable | Static hierarchy + AMP + Learnable Inter-layer | Learnable inter-layer brings further improvement |
| M3: Uniform-Aniso-Learnable | Uniform Sampling + AMP + Learnable Inter-layer | Uniform sampling is inferior to adaptive sampling |
| M4: Dynamic-Aniso-Unlearnable | Dynamic Hierarchy + AMP + Unlearnable Inter-layer | Dynamic hierarchy is effective but insufficient |
| EvoMesh (Full) | Dynamically Adaptive + AMP + Learnable Inter-layer | All components are essential |

### Key Findings

- **Visualization of Adaptive Hierarchy**: Coarse-grained nodes constructed by EvoMesh tend to concentrate in regions with sharp variations in physical quantities (e.g., high-intensity regions of temporal differences in velocity fields), and the hierarchical structure evolves over time, validating the effectiveness of the adaptive construction.
- **High Correlation of Anisotropic Weights with Physical Dynamics**: The variance of edge weights predicted by AMP highly coincides with regions where physical quantities change over time.
- **OOD Generalization**: On high-resolution meshes containing 2x nodes and 3x edges compared to training data, EvoMesh performs best on CylinderFlow and Airfoil (benefiting from the resolution scalability of adaptive hierarchy construction).
- **Physical Quantity OOD Generalization**: Under conditions where the input velocity distribution shifts by 64-187%, the long-term prediction error of EvoMesh on CylinderFlow is only 0.091 (vs 0.251 for BSMS), representing a 63.7% improvement.

## Highlights & Insights

1. **Differentiable Node Selection is the Core Innovation**: Using Gumbel-Softmax to convert discrete node selection into a differentiable operation allows the graph hierarchy to be learned end-to-end. This is a key breakthrough in successfully applying graph pooling techniques to physical simulation.
2. **Killing Two Birds with One Stone with AMP Weights**: Weights computed by intra-layer AMP are directly reused for inter-layer REDUCE/EXPAND. This both reduces extra parameters and ensures consistency in information propagation within and across layers.
3. **Temperature Annealing Strategy**: Temperature annealing in Gumbel-Softmax (from exploration to refinement) is critical for training stability. Consequently, the trained model produces consistent graph hierarchies under identical inputs.
4. **Natural Support for Time-Varying Mesh Scenarios**: Because the hierarchical structure is constructed adaptively online, EvoMesh naturally supports scenarios where the mesh topology changes over time, which is impossible for all methods relying on pre-computed hierarchies.

## Limitations & Future Work

1. On systems with regular structures and smooth deformations (e.g., FlyingFlag, DeformingPlate), OOD generalization is inferior to that of simple MGN; true resolution-independent modeling remains an open question.
2. Although the randomness introduced by Gumbel-Softmax sampling has minimal impact after training, exploring more deterministic differentiable selection schemes could theoretically be beneficial.
3. The choice of K (K=2) for K-hop edge enhancement is determined experimentally, lacking an adaptive adjustment mechanism.
4. The paper only evaluates 2D/3D mesh simulation tasks; its effectiveness in more complex multi-physics coupling scenarios remains to be validated.

## Related Work & Insights

- **Difference from TopKPool/DiffPool**: Although both are differentiable graph pooling, the DiffSELECT of EvoMesh is based on independent node-level sampling from physical context rather than a global clustering assignment matrix, making it more suitable for preserving geometric structures in physical simulation.
- **AMP vs Graph Attention**: AMP weight prediction considers both edge features and end-nodes simultaneously, and weights are reused for inter-layer propagation, making it more suitable for physical simulation scenarios than standard GAT.
- **Insight**: The idea of adaptive graph structure evolution can be generalized to other fields requiring multi-scale modeling, such as molecular dynamics, weather forecasting, etc.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Graph-Based Alternatives to LLMs for Human Simulation](../../ACL2026/graph_learning/graph-based_alternatives_to_llms_for_human_simulation.md)
- [\[NeurIPS 2025\] Logical Expressiveness of Graph Neural Networks with Hierarchical Node Individualization](../../NeurIPS2025/graph_learning/logical_expressiveness_of_graph_neural_networks_with_hierarchical_node_individua.md)
- [\[ICML 2025\] GlycanAA: Modeling All-Atom Glycan Structures via Hierarchical Message Passing and Multi-Scale Pre-training](modeling_all-atom_glycan_structures_via_hierarchical_message_passing_and_multi-s.md)
- [\[NeurIPS 2025\] Unifying and Enhancing Graph Transformers via a Hierarchical Mask Framework](../../NeurIPS2025/graph_learning/unifying_and_enhancing_graph_transformers_via_a_hierarchical_mask_framework.md)
- [\[AAAI 2026\] Self-Adaptive Graph Mixture of Models](../../AAAI2026/graph_learning/self-adaptive_graph_mixture_of_models.md)

</div>

<!-- RELATED:END -->
