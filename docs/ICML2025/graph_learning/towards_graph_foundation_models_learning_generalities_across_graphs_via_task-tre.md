---
title: >-
  [Paper Note] Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees
description: >-
  [ICML 2025][Graph Learning][Graph Foundation Models] The authors propose Task-Tree as a unified learning instance, aligning node/edge/graph-level tasks into a single representation space by introducing virtual task nodes. Combined with a reconstruction objective for GNN pre-training, they build the graph foundation model GIT, achieving cross-domain and cross-task generalization across fine-tuning, in-context learning, and zero-shot settings on 32 graphs from 5 domains.
tags:
  - "ICML 2025"
  - "Graph Learning"
  - "Graph Foundation Models"
  - "Task-Tree"
  - "Cross-Task Generalization"
  - "GNN Pre-training"
  - "Zero-Shot Learning"
date: 2026-05-08
content_hash: 5cda46da037d76d9
---

# Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees

**Conference**: ICML 2025  
**arXiv**: [2412.16441](https://arxiv.org/abs/2412.16441)  
**Code**: [GIT](https://github.com/Zehong-Wang/GIT)  
**Area**: Graph Learning  
**Keywords**: Graph Foundation Models, Task-Tree, Cross-Task Generalization, GNN Pre-training, Zero-Shot Learning

## TL;DR

The authors propose Task-Tree as a unified learning instance, aligning node/edge/graph-level tasks into a single representation space by introducing virtual task nodes. Combined with a reconstruction objective for GNN pre-training, they build the graph foundation model GIT, achieving cross-domain and cross-task generalization across fine-tuning, in-context learning, and zero-shot settings on 32 graphs from 5 domains.

## Background & Motivation

**Background**: Foundation models have achieved immense success in NLP (LLMs) and CV (LVMs) by pre-training on large-scale data to capture transferable patterns (e.g., texture and contours in images, token semantics in text). However, foundation models for graph-structured data are still in their infancy.

**Limitations of Prior Work**: The core challenges of graph data lie in two dimensions of heterogeneity: (1) **Structural/Feature Heterogeneity**—graphs from different domains encode completely different phenomena (e.g., social networks vs. molecular graphs); (2) **Task Heterogeneity**—graph tasks operate on different levels of learning units (node, edge, and graph), making them difficult to accommodate within a unified model. Existing methods either rely on graphon theory (which imposes overly strong assumptions and is computationally intractable) or on subgraph/substructure extraction (where MP-GNNs fail to efficiently encode substructures, suffering from high computational overhead).

**Key Challenge**: Subgraph-based methods require additional storage and encoding of induced subgraphs, increasing time and memory costs. Meanwhile, MP-GNNs have limited expressiveness in substructure learning, leading to poor cross-task generalization.

**Goal**: How can we find a unified learning instance capable of aligning node, edge, and graph-level tasks, allowing GNNs to be efficiently pre-trained on it and transferred to downstream tasks?

**Key Insight**: Starting from the learning dynamics of MP-GNNs, the prediction for any graph task relies on the embeddings of "task-related nodes" (target node for node tasks, endpoints for edge tasks, and all nodes for graph tasks). A virtual task node can be introduced to connect all task-related nodes; the computational tree rooted at this virtual node is defined as the Task-Tree.

**Core Idea**: Use Task-Trees (virtual task nodes combined with computational trees) instead of subgraphs as the unified learning instance for cross-task learning, enabling efficient and theoretically proven pre-training and transfer for graph foundation models.

## Method

### Overall Architecture

The input consists of text-attributed graphs from multiple domains (academic networks, e-commerce, knowledge graphs, molecular graphs, and temporal graphs). All node features are encoded into a shared 768-dimensional space using Sentence-BERT. A Task-Tree is constructed for each learning instance (node/edge/graph). A GNN encoder (GIT-G) is then pre-trained on multi-domain Task-Trees using reconstruction objectives. Optionally, domain specialization is conducted via instruction tuning (GIT-S). Finally, the model is evaluated on downstream tasks using fine-tuning, in-context learning, and zero-shot transfer.

### Key Designs

1. **Task-Tree Construction and Encoding**:

    - Function: Construct a unified learning unit for any graph task instance.
    - Mechanism: For node, edge, and graph-level tasks, the task-related node sets are identified first. A virtual task node is then introduced to connect all task-related nodes, forming a Task-Tree. During encoding, MEAN aggregation is used: $\mathbf{z}^t = \frac{1}{n}\sum_{i=1}^{n}\phi(T_i)$, where $T_i$ is the computational tree of the $i$-th task-related node. Operationally, this only requires adding virtual nodes and edges to the original graph, followed by standard message passing.
    - Design Motivation: Compared to subgraph-based methods, the Task-Tree has three primary advantages—(1) **Learnability**: Tree structures can be naturally and effectively encoded by MP-GNNs; (2) **Unified Nature**: It seamlessly applies to different task levels; (3) **Efficiency**: It only requires adding virtual nodes to the original graph, bypassing the overhead of subgraph extraction and storage.

2. **Task-Tree Reconstruction Pre-training (GIT-G)**:

    - Function: Pre-train the GNN on multi-domain Task-Trees using a self-supervised reconstruction objective.
    - Mechanism: Two types of data augmentation (random edge masking and attribute masking) are applied to each Task-Tree to generate two views, $\hat{T}$ and $\tilde{T}$. These views are encoded separately using the encoder $\phi$. A symmetric reconstruction loss with stop-gradient is then employed to make the two views predict each other, while KL regularization is added to project the embeddings into a shared space:
    
    $$\mathcal{L} = \frac{1}{2n}\sum_i [\|\rho(g(\hat{z}_i)) - \text{sg}[\rho(\tilde{z}_i)]\|^2 + \|\rho(g(\tilde{z}_i)) - \text{sg}[\rho(\hat{z}_i)]\|^2] + \sum_i D_{KL}(h \| z_i)$$
    
    - Design Motivation: The reconstruction objective captures corruption-invariant semantics in the Task-Trees, and the KL regularization ensures that the embeddings of different Task-Trees are projected into a shared space.

3. **Domain-Specialization Instruction Tuning (GIT-S)**:

    - Function: Adapt the general-purpose model to a specific domain.
    - Mechanism: Post-training is performed on the target domain's Task-Trees using a supervised fine-tuning loss:
    
    $$\mathcal{L}_{SFT} = \frac{1}{n}\sum_i \kappa(\phi^*(T_i), \psi(T_i))$$
    
    where $\psi(T_i)$ represents the label description embeddings encoded by an LLM to serve as instructions.
    - Design Motivation: Theoretical analysis (generalization bounds) shows that reducing the gap between the pre-training and fine-tuning distributions improves generalization. Because Task-Tree distributions from the same domain are similar, domain specialization effectively narrows this distribution gap.

### Theoretical Analysis

The paper provides three core theorems: (1) **Stability**—Task-Trees with similar subtree structures yield similar embeddings, and the width of the Task-Tree has minimal impact on representation distance; (2) **Transferability**—knowledge learned during pre-training can transfer to downstream tasks at an $O(1)$ constant ratio; (3) **Generalization Bound**—downstream risk is bounded by pre-training quality, the distribution gap, and the number of fine-tuning samples, demonstrating that few-shot fine-tuning supports robust generalization.

## Key Experimental Results

### Main Results

| Domain | Setting | GIT-G | GIT-S | OFA | GraphMAE | Sup. GNN |
|------|------|-------|-------|-----|----------|----------|
| Academic Networks | Zero-Shot | 14.88 | **23.45** | 13.98 | 15.42 | - |
| Academic Networks | 3-Shot | 54.00 | **55.18** | 45.93 | 49.25 | - |
| Academic Networks | Fine-Tuning | 75.82 | **75.88** | 72.18 | 73.81 | 73.57 |
| Molecular Graphs | Zero-Shot | 53.34 | **62.83** | 50.49 | 47.19 | - |
| Overall Avg. | Fine-Tuning | **75.37** | **75.72** | 73.08 | 72.79 | 72.25 |

### Comparison with SOTA Graph Foundation Models

| Method | Academic Networks | Knowledge Graphs | Molecular Graphs |
|------|---------|---------|--------|
| GraphPrompt+ | 74.80 | 74.78 | 72.99 |
| All in One | 75.25 | 74.92 | 71.87 |
| OpenGraph | 74.64 | 71.38 | 72.84 |
| AnyGraph | 75.01 | 74.30 | 72.49 |
| **GIT-G** | **75.82** | **75.73** | **74.57** |

### Ablation Study

| Training Strategy | Zero-Shot | 3-Shot | Fine-Tuning |
|---------|--------|--------|------|
| Base Model (GIT) | 15.36 | 53.31 | 75.53 |
| Expert Model (GIT) | 18.38 | 55.10 | 75.47 |
| General Model (GIT) | 14.88 | 54.00 | 75.82 |
| Specialized Model (GIT) | **23.45** | **55.18** | **75.88** |
| General Model (OFA) | 13.98 | 45.93 | 72.18 |
| Specialized Model (OFA) | 20.05 | 46.87 | 73.04 |

### Key Findings

- The general model of GIT maintains stable performance and does not experience the significant decline from Base to General observed in GraphMAE/OFA, indicating that the Task-Tree effectively mitigates negative transfer.
- Domain specialization (GIT-S) yields particularly distinct gains in zero-shot and few-shot settings (e.g., zero-shot performance increases from 14.88 to 23.45), while having a minor effect on full fine-tuning.
- GIT-S approaches the domain expert GIMLET in the molecular domain (62.83 vs. 64.15) and Ultra in knowledge graphs (67.80 vs. 68.53).
- Crucially, Task-Tree consistently outperforms subgraph-based methods across all evaluations while offering superior computational efficiency.

## Highlights & Insights

- **Elegant Design of Task-Tree**: Converting the cross-task heterogeneity problem into "message passing on an augmented graph" via virtual nodes delivers both theoretical unity and engineering efficiency. The core insight is that predictions in GNNs inherently depend on computational trees, and the Task-Tree serves as a natural abstraction of this computational process.
- **Theory-Driven Framework Design**: The three theorems regarding stability, transferability, and generalization are not mere post-hoc validations but actively guided the model's architecture. For instance, the distribution gap term in the generalization bound directly inspired the domain specialization strategy.
- **Transferable Instruction Tuning Concept**: Leveraging an LLM to encode label descriptions as instructions for SFT in the graph domain is a versatile strategy that can be directly applied to other structured data modalities, such as knowledge graph completion or protein function prediction.

## Limitations & Future Work

- The effectiveness of Task-Tree relies on the assumption of text-attributed graphs—all node features must first be aligned to a shared space using Sentence-BERT. However, many real-world graph datasets lack text attributes (e.g., molecular graphs with purely numerical features), leaving the feature alignment problem still unfully resolved.
- The scale of the pre-training data is relatively limited (around 30 graphs), which is vastly smaller than that of standard foundation models in NLP and CV. Whether scaling laws hold in the graph domain remains unverified.
- The domain specialization of GIT-S still depends on annotated data to some extent. Adapting the model to entirely unlabeled novel domains remains an open challenge.

## Related Work & Insights

- **vs. OFA (Liu et al., 2024a)**: OFA also targets cross-domain graph learning, but relies on subgraph extraction and a unified prompt mechanism, which incurs high computational overhead. In contrast, GIT operates directly on the original graph via Task-Trees, delivering higher efficiency backed by theoretical support.
- **vs. GFT (Wang et al., 2024b)**: GFT also introduces computational trees to align heterogeneous graph tasks but employs a model-driven design (using a learnable vocabulary and multi-faced reconstruction objectives). GIT presents a theory-driven alternative, making the two approaches complementary.
- **vs. AnyGraph (Xia & Huang, 2024)**: AnyGraph addresses cross-domain issues through a unified feature space, whereas GIT focuses on task alignment. GIT comprehensively outperforms AnyGraph in experiments.

## Rating

- Novelty: ⭐⭐⭐⭐ Although the Task-Tree concept shares connections with the computational trees in GFT, the theoretical framework and the elegant implementation constitute novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The evaluation is highly comprehensive, covering 32 graphs across 5 domains and 3 evaluation paradigms.
- Writing Quality: ⭐⭐⭐⭐ The theoretical derivations are clear, though the dense notation requires close attention.
- Value: ⭐⭐⭐⭐ Offers both a theoretical foundation and a practical framework for graph foundation models, though the feature alignment assumption limits its overall universality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] View Space: Representation Learning Across Arbitrary Graphs](../../ICML2026/graph_learning/view_space_learning_representation_across_arbitrary_graphs.md)
- [\[ICML 2025\] Mixed-Curvature Decision Trees and Random Forests](mixed-curvature_decision_trees_and_random_forests.md)
- [\[ICML 2025\] WILTing Trees: Interpreting the Distance Between MPNN Embeddings](wilting_trees_interpreting_the_distance_between_mpnn_embeddings.md)
- [\[ICML 2025\] TopInG: Topologically Interpretable Graph Learning via Persistent Rationale Filtration](toping_topologically_interpretable_graph_learning_via_persistent_rationale_filtr.md)
- [\[ICML 2025\] From RAG to Memory: Non-Parametric Continual Learning for Large Language Models](from_rag_to_memory_non-parametric_continual_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
