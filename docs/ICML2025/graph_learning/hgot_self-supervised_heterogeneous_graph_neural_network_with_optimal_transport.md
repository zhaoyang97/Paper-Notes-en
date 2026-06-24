---
title: >-
  [Paper Note] HGOT: Self-supervised Heterogeneous Graph Neural Network with Optimal Transport
description: >-
  [ICML2025][Graph Learning][Heterogeneous Graph Neural Networks] This paper proposes HGOT, which introduces optimal transport theory into heterogeneous graph self-supervised learning for the first time. It replaces data augmentation and positive/negative sample selection in traditional contrastive learning with a Fused Gromov-Wasserstein transport plan between the branch view (metapath view) and the central view (aggregated view), achieving an average improvement of over 6% in…
tags:
  - "ICML2025"
  - "Graph Learning"
  - "Heterogeneous Graph Neural Networks"
  - "Self-supervised Learning"
  - "Optimal Transport"
  - "Metapath"
  - "Gromov-Wasserstein"
date: 2026-05-08
content_hash: b6d658e24c2fb105
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# HGOT: Self-supervised Heterogeneous Graph Neural Network with Optimal Transport

**Conference**: ICML2025  
**arXiv**: [2506.02619](https://arxiv.org/abs/2506.02619)  
**Author**: Yanbei Liu, Chongxu Wang, Zhitao Xiao, Lei Geng, Yanwei Pang, Xiao Wang
**Code**: To be confirmed  
**Area**: Graph Learning  
**Keywords**: Heterogeneous Graph Neural Networks, Self-supervised Learning, Optimal Transport, Metapath, Gromov-Wasserstein

## TL;DR

This paper proposes HGOT, which introduces optimal transport theory into heterogeneous graph self-supervised learning for the first time. It replaces data augmentation and positive/negative sample selection in traditional contrastive learning with a Fused Gromov-Wasserstein transport plan between the branch view (metapath view) and the central view (aggregated view), achieving an average improvement of over 6% in node classification.

## Background & Motivation

Most existing heterogeneous graph neural networks (HGNNs) rely on supervised or semi-supervised learning, but the cost of labeling heterogeneous graphs is highly expensive. Self-supervised learning (SSL) serves as an alternative, where contrastive learning is the dominant strategy. However, contrastive learning faces two major challenges:

**Difficulty in designing graph augmentation strategies** (C1): The data structure of heterogeneous graphs is discrete, and minor perturbations can alter graph properties, leading to label inconsistency between the original and augmented graphs. Meanwhile, the concept of "maximizing/minimizing similarity" is vague and lacks a clear metric.

**Complete aggregation of heterogeneous semantic information** (C2): Heterogeneous graphs contain multiple node types, diverse relations, and different connectivity. Simple aggregation can distort the topological structure and lose contextual semantics.

The core motivation of HGOT is: **completely eliminating graph augmentation and positive/negative sample pairs**, instead discovering the alignment between the graph space and representation space through optimal transport.

## Method

The HGOT framework consists of three core modules: node feature transformation, aggregated view generation, and multi-view optimal transport alignment.

### 1. Node Feature Transformation

Features of different node types in heterogeneous graphs have diverse dimensions and need to be projected into a unified space. For the $i$-th node with type $\varphi_i$:

$$\mathbf{h}_i = \mathbf{W}_{\varphi_i} \cdot \mathbf{x}_i + \mathbf{b}_{\varphi_i}$$

where $\mathbf{W}_{\varphi_i}$ is a type-specific linear transformation matrix. After projection, all node features share the same dimension, preparing for subsequent OT calculations.

### 2. Aggregated View Generation

The original heterogeneous graph is decomposed into multiple homogeneous subgraphs (branch views) based on metapaths, which are then aggregated into a central view.

**Node Aggregation**: For each metapath $p$, a multi-head attention mechanism is used to generate the representation of node $v_i$ under this metapath:

$$\tilde{\mathbf{h}}_i^p = \text{CONCAT}_{k=1}^{K} \;\Psi\!\left(\sum_{v_j \in \mathcal{N}_p(v_i)} \alpha_{ij}^p \mathbf{W}^p \mathbf{h}_i\right)$$

The attention coefficient $\alpha_{ij}^p$ is calculated through softmax normalization.

**Metapath-level Aggregation**: Another layer of attention network is employed to weighted fuse all metapath representations:

$$\mathbf{h}_i^{\text{agg}} = \sum_{p=1}^{|\mathcal{P}|} \beta^p \cdot \tilde{\mathbf{h}}_i^p$$

The weight $\beta^p$ can be interpreted as the contribution of the $p$-th metapath, which is obtained from the type-level attention vector $\mathbf{q}$ via a tanh transformation followed by softmax.

**Edge Aggregation**: A logical OR operation is performed on the adjacency matrices of all metapaths:

$$\mathbf{A}_{\text{agg}} = \mathbf{A}_1 \lor \mathbf{A}_2 \lor \cdots \lor \mathbf{A}_{|\mathcal{P}|}$$

This ensures that as long as two nodes are connected in any metapath view, the edge is preserved in the aggregated view, without introducing duplicate edges.

### 3. Multi-view OT Alignment

Core innovation: Replacing contrastive learning with optimal transport. It calculates the transport plan between each branch view and the central view to align the graph space and the representation space.

**Wasserstein Distance on Node Distributions**:

$$\mathcal{D}_n(\mathbf{H}_p, \mathbf{H}_{\text{agg}}) = \min_{\pi \in \Pi(\mu, \nu)} \langle \mathcal{F}(\mathbf{H}_p, \mathbf{H}_{\text{agg}}), \pi \rangle$$

The cost matrix $\mathcal{F}_{ij} = \mathcal{C}_X(\mathbf{H}_p^i, \mathbf{H}_{\text{agg}}^j)$ utilizes cosine distance.

**Gromov-Wasserstein Distance on Edge Structures**:

$$\mathcal{D}_e(\mathbf{A}_p, \mathbf{A}_{\text{agg}}) = \min_{\pi \in \Pi(\mu, \nu)} \langle \mathcal{E}(\mathbf{A}_p, \mathbf{A}_{\text{agg}}) \otimes \pi, \pi \rangle$$

The cost function $\mathcal{C}_A(\mathbf{A}_p^{ik}, \mathbf{A}_{\text{agg}}^{jl}) = |\mathbf{A}_p^{ik} - \mathbf{A}_{\text{agg}}^{jl}|$ directly measures differences in edge structure.

**Fused Gromov-Wasserstein Distance**: Considering both node features and edge structures simultaneously:

$$\mathcal{D}_g(\mathcal{G}_p, \mathcal{G}_{\text{agg}}) = \min_{\pi} \;\sigma \sum_{ij} \mathcal{C}_X \cdot \pi_{ij}^{\mathcal{G}} + (1-\sigma) \sum_{ijkl} \mathcal{C}_A \cdot \pi_{ij}^{\mathcal{G}} \pi_{kl}^{\mathcal{G}}$$

The hyperparameter $\sigma \in [0,1]$ balances the contributions of node features and edge structures. The final objective function drives the encoder to learn node representations consistent with the matching relationships in the graph space.

### Training Process

- For each (branch view, central view) pair, the Sinkhorn algorithm is used to solve the optimal transport plan $\pi^{\mathcal{G}}$.
- Simultaneously, the transport plan $\pi^{\mathcal{Z}}$ is computed for the representation space.
- The two are aligned to make representation consistency coincide with graph topological consistency.
- The overall process is end-to-end unsupervised training and requires no labels.

## Key Experimental Results

**Datasets**: ACM, DBLP, Freebase, AMiner (four widely used heterogeneous graph benchmarks)

**Downstream Tasks**: Node classification, node clustering

**Main Results**:
- In node classification tasks, HGOT achieves an average accuracy improvement of over **6%** compared to SOTA methods.
- Best or highly competitive results are achieved across all four datasets.
- Baselines include mainstream self-supervised heterogeneous graph methods such as HAN, HeCo, HGCML, and HGMAE.

**Ablation Study**:
- Performance drops significantly after removing the OT alignment module, verifying the effectiveness of the OT mechanism.
- Using only the Wasserstein distance (excluding edge information) or only the GW distance (excluding node features) performs worse than the fused version.
- The logical OR operation for the aggregated view outperforms simple addition or intersection.

## Highlights & Insights

1. **First to introduce optimal transport into heterogeneous graph self-supervised learning**: It bypasses the core bottlenecks of graph augmentation design and positive/negative sample selection in contrastive learning, presenting a clear and theoretically elegant approach.
2. **Branch view + Central view design**: Local semantics (branch) are captured through metapath decomposition and then aggregated into global semantics (central), with OT aligning their relationships, exhibiting strong logical self-consistency.
3. **Fused Gromov-Wasserstein**: It simultaneously utilizes both node features and edge structures to calculate the transport distance, providing a more comprehensive representation than using either in isolation.
4. **No data augmentation required**: It avoids the label inconsistency issues caused by augmentation under discrete graph structures.

## Limitations & Future Work

1. **Computational Complexity**: The Gromov-Wasserstein distance involves fourth-order tensors. Even with Sinkhorn approximations, it may still become a bottleneck on large-scale graphs. The paper does not provide a detailed comparison of time and memory overhead.
2. **Dependence on Metapaths**: The method still relies on hand-crafted metapaths, making its applicability questionable in heterogeneous graph scenarios where metapaths are hard to define (e.g., graphs with complex schemas or dynamic changes).
3. **Limited Evaluation Tasks**: It only evaluates node classification and clustering, without involving other equally important tasks on heterogeneous graphs, such as link prediction and graph classification.
4. **Small Dataset Scale**: The four benchmark datasets are all of small to medium scale; performance on large-scale heterogeneous graphs remains unknown.
5. **Sensitivity of the $\sigma$ Hyperparameter**: The $\sigma$ in the fused GW distance requires optimization, and the discussion on hyperparameter search in the paper is insufficient.

## Related Work & Insights

- **HeCo** (Wang et al. 2021): Contrastive learning that aligns local/global information via network schemas and metapaths, which is the direct baseline to HGOT.
- **HGCML** (Wang et al. 2023): Multi-view contrastive learning + new positive sample sampling strategy.
- **HGMAE** (Tian et al. 2023): Applying masked autoencoder to heterogeneous graphs.
- **Gromov-Wasserstein** (Mémoli 2011; Vayer et al. 2020): An optimal transport distance framework between graphs, which HGOT generalizes to heterogeneous graph scenarios.

**Insights**: As a general tool for measuring distance between two distributions, OT can be further explored in scenarios such as dynamic heterogeneous graphs and knowledge graph completion. Furthermore, combining OT with generative self-supervised learning (e.g., MAEs) may yield complementary advantages.

## Reproducibility Points

- Requires implementing the Sinkhorn algorithm or calling the POT library to solve OT.
- Heterogeneous graph preprocessing requires decomposing into homogeneous subgraphs according to metapaths and constructing adjacency matrices for each metapath.
- Key hyperparameters: $\sigma$ (node/edge weight balance), $K$ (number of attention heads), $d$ (projection dimension), $d_m$ (type-level attention dimension).
- Whether the code is open-source is to be confirmed; in the absence of a code link, implementation from scratch based on the paper is required.

## Rating
- Novelty: ⭐⭐⭐⭐ — First to apply OT to heterogeneous graph SSL, offering a fresh perspective
- Experimental Thoroughness: ⭐⭐⭐ — 4 datasets and 2 tasks, complete ablation but limited scale/task coverage
- Writing Quality: ⭐⭐⭐⭐ — Clear formulation system, smooth writing logic
- Value: ⭐⭐⭐⭐ — Provides a new paradigm beyond contrastive learning for self-supervised learning on heterogeneous graphs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-Supervised Discovery of Neural Circuits in Spatially Patterned Neural Responses with Graph Neural Networks](../../NeurIPS2025/graph_learning/self-supervised_discovery_of_neural_circuits_in_spatially_patterned_neural_respo.md)
- [\[AAAI 2026\] GCL-OT: Graph Contrastive Learning with Optimal Transport for Heterophilic Text-Attributed Graphs](../../AAAI2026/graph_learning/gcl-ot_graph_contrastive_learning_with_optimal_transport_for_heterophilic_text-a.md)
- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](../../NeurIPS2025/graph_learning/sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)
- [\[AAAI 2026\] On Stealing Graph Neural Network Models](../../AAAI2026/graph_learning/on_stealing_graph_neural_network_models.md)
- [\[ICLR 2026\] HarmonyGNNs: Harmonizing Heterophily and Homophily in GNNs via Self-Supervised Node Encoding](../../ICLR2026/graph_learning/harmonygnns_harmonizing_heterophily_and_homophily_in_gnns_via_self-supervised_no.md)

</div>

<!-- RELATED:END -->
