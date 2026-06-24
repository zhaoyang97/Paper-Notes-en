---
title: >-
  [Paper Note] RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph
description: >-
  [ACL 2025][Graph Learning][KGE] Proposed RSCF, a plug-and-play KGE method, which ensures "relation-semantics consistency" (semantically similar relations generate similar entity transformations) through three key designs: shared affine transformation, rooted entity transformation, and normalization. It significantly outperforms SOTA models on both distance-based and tensor factorization models, with consistency preservation rates validated theoretically and experimentally.
tags:
  - "ACL 2025"
  - "Graph Learning"
  - "KGE"
  - "entity transformation"
  - "relation semantics"
  - "consistency"
  - "plug-in"
date: 2026-05-08
content_hash: 0f1a8b19f75acf49
---

# RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph

**Conference**: ACL 2025  
**arXiv**: [2505.20813](https://arxiv.org/abs/2505.20813)  
**Code**: Yes  
**Area**: Graph Learning  
**Keywords**: KGE, entity transformation, relation semantics, consistency, plug-in

## TL;DR
Proposed RSCF, a plug-and-play KGE method, which ensures "relation-semantics consistency" (semantically similar relations generate similar entity transformations) through three key designs: shared affine transformation, rooted entity transformation, and normalization. It significantly outperforms SOTA models on both distance-based and tensor factorization models, with consistency preservation rates validated theoretically and experimentally.

## Background & Motivation

**Background**: KGE performs knowledge graph completion by mapping entities and relations into a low-dimensional vector space. To handle complex relation-patterns such as 1-N/N-1/N-N, Entity Transformation Models (ETMs) generate distinct entity embeddings for each relation (e.g., SFBR transforms entities using relation-specific $W_r$ matrices).

**Limitations of Prior Work**: (1) **Transformation Disconnection**: The transformation parameters $W_r$ for each relation are mutually independent; thus, semantically similar relations may learn completely different transformations and entity embeddings. Limited entity observations in sparse KGs exacerbate this disconnection. (2) **Over-concentration of Embeddings**: SFBR paired with DURA regularization leads to excessively concentrated entity embeddings, producing score distributions that are indistinguishable across relations—rendering "all relations looking the same."

**Key Challenge**: How to simultaneously leverage individual relation specificity (to distinguish different relations) and maintain consistency among semantically similar relations (similar relations should yield similar transformations)?

**Core Idea**: Employ a shared affine transformation to map all relation embeddings to the entity transformation space, ensuring a consistent "similar inputs $\rightarrow$ similar outputs" mapping.

## Method

### Overall Architecture
Entity transformation formula: $\mathbf{e_r} = (\text{N}_p(\mathbf{rA_1}) + \mathbf{1}) \otimes \mathbf{e}$, where $\mathbf{A_1}$ is the affine transformation matrix shared by all relations, $\text{N}_p$ denotes $L_p$ normalization, and $\otimes$ denotes element-wise multiplication.

### Key Designs

1. **Shared Affine Transformation (ⓐ)**:

    - All relations share a single linear transformation $\mathbf{A_1} \in \mathbb{R}^{n \times n}$ to map relation embeddings to entity transformation vectors.
    - Affine transformations preserve parallelism and distance ratios $\rightarrow$ relation embeddings with close semantic distances will map to transformation vectors that are also close.
    - Monte Carlo simulation validation: The consistency preservation rate is 100% under linear scenarios and 72.8% to 99.4% under non-linear scenarios.

2. **Rooted Entity Transformation (ⓒ)**:

    - The transformed result $\mathbf{e_r}$ = original embedding $\mathbf{e}$ + relation-specific "change amount", rather than a complete replacement.
    - The "add-one" operation: $\text{N}_p(\mathbf{rA_1}) + \mathbf{1}$ ensures that the transformation is a "fine-tuning" of the original embedding rather than a "reset".
    - Design Motivation: Decreases the required expressiveness for transformation (only needing to learn the variation rather than a completely new position), making the expressiveness loss of the shared affine transformation controllable.

3. **Normalization (ⓑ)**:

    - $\text{N}_p(\mathbf{rA}) = \frac{\mathbf{rA}}{\|\mathbf{rA}\|_p}$ prevents the scale of the change from shrinking or magnifying infinitely.
    - Constrains the transformation magnitude within a safe range, protecting the semantic preservation capability of the rooted transformation.

4. **Relation Transformation (RT) + Relation Prediction (RP)**:

    - RT: Uses the same RSCF mechanism to perform entity-specific transformation on relation embeddings: $\mathbf{r_{ht}} = \psi'(h) \otimes \psi'(t) \otimes \mathbf{r}$, to capture sub-relation semantics.
    - RP: Introduces a relation prediction loss $\phi(\mathbf{r}|\mathbf{h},\mathbf{t})$ to the training objective, enhancing the semantic clustering of relation embeddings.

## Key Experimental Results

### Main Results

| Dataset | Model | SFBR MRR | RSCF MRR | Gain |
|------|------|:---:|:---:|:---:|
| WN18RR | TransE | Baseline | Significant Gain | ✓ |
| FB15k-237 | RotatE | Baseline | Significant Gain | ✓ |
| YAGO3-10 | PairRE | Baseline | Significant Gain | ✓ |

Achieves SOTA across all combinations of TransE/SimplE/RotatE/PairRE and WN18RR/FB15k-237/YAGO3-10.

### Ablation Study & Consistency Verification

| Metric | SFBR | RSCF |
|------|:---:|:---:|
| ET Intra Cluster Dist (↓) | 1.10 | **0.47** |
| ET Inter Cluster Dist (↑) | 0.27 | **0.82** |
| EE Intra Cluster Dist (↓) | 2.35 | **0.53** |
| EE Inter Cluster Dist (↑) | 0.40 | **0.85** |

t-SNE visualization clearly demonstrates that RSCF's relation transformations and entity embeddings for semantically similar relations cluster more compactly and remain highly distinguishable.

## Highlights & Insights
- **Formal definition of the "relation-semantics consistency" concept** (the distance relationships between any pairs of relations are preserved after transformation) is a core contribution, converting vague intuition into a verifiable mathematical property.
- **Monte Carlo validation** is an elegant theoretical analysis method—verifying the probability of consistency preservation on randomly generated embeddings.
- **Minimalist design**: The entire RSCF introduces only 3 shared matrices $A_1, A_2, A_3$ (each of size $n \times n$), yielding negligible parameter overhead.

## Limitations & Future Work
- The expressiveness upper bound of the shared affine transformation is constrained—highly dissimilar relations may require more distinct transformations.
- Only tested on standard KGC datasets; scalability on ultra-large-scale KGs remains unverified.
- Whether RP and RT modules can be implemented more efficiently.

## Rating
- Novelty: ⭐⭐⭐⭐ Formal definition and theoretical validation of relation-semantics consistency
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 models $\times$ 3+ datasets + consistency quantification + visualization
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, intuitive ⓐⓑⓒ annotations
- Value: ⭐⭐⭐⭐ Plug-in design enables plug-and-play capability for existing KGE models

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Croppable Knowledge Graph Embedding](croppable_knowledge_graph_embedding.md)
- [\[ACL 2025\] A Mutual Information Perspective on Knowledge Graph Embedding](a_mutual_information_perspective_on_knowledge_graph_embedding.md)
- [\[NeurIPS 2025\] Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](../../NeurIPS2025/graph_learning/unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)
- [\[ACL 2025\] Agent Steerable Search for Knowledge Graph Question Answering](agent_steerable_search_for_knowledge_graph_question_answering.md)
- [\[CVPR 2025\] Coeff-Tuning: A Graph Filter Subspace View for Tuning Attention-Based Large Models](../../CVPR2025/graph_learning/coeff-tuning_a_graph_filter_subspace_view_for_tuning_attention-based_large_model.md)

</div>

<!-- RELATED:END -->
