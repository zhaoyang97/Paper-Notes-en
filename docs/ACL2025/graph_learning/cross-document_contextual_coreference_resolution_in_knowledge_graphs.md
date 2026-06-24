---
title: >-
  [Paper Note] Cross-Document Contextual Coreference Resolution in Knowledge Graphs
description: >-
  [ACL 2025][Graph Learning][Cross-Document Coreference Resolution] Proposes a knowledge graph-based cross-document coreference resolution method. By associating textual entity mentions with knowledge graph nodes through a dynamic linking mechanism, it combines contextual embeddings and graph message passing reasoning to improve the precision and recall of cross-document entity recognition, outperforming traditional methods on multiple benchmark datasets.
tags:
  - "ACL 2025"
  - "Graph Learning"
  - "Cross-Document Coreference Resolution"
  - "Knowledge Graph"
  - "Entity Linking"
  - "Graph Reasoning"
  - "Contextual Embeddings"
date: 2026-05-08
content_hash: 588c78d494407360
---

# Cross-Document Contextual Coreference Resolution in Knowledge Graphs

**Conference**: ACL 2025  
**arXiv**: [2504.05767](https://arxiv.org/abs/2504.05767)  
**Code**: None  
**Area**: Graph Learning  
**Keywords**: Cross-Document Coreference Resolution, Knowledge Graph, Entity Linking, Graph Reasoning, Contextual Embeddings

## TL;DR

Proposes a knowledge graph-based cross-document coreference resolution method. By associating textual entity mentions with knowledge graph nodes through a dynamic linking mechanism, it combines contextual embeddings and graph message passing reasoning to improve the precision and recall of cross-document entity recognition, outperforming traditional methods on multiple benchmark datasets.

## Background & Motivation

**Background**: Coreference resolution is a fundamental task in NLP, which requires identifying different expressions referring to the same entity across different texts. Cross-document coreference resolution (CD-CR) is particularly crucial in the field of knowledge graphs, requiring the association of entity mentions across multiple documents.

**Limitations of Prior Work**: 
   - Traditional coreference resolution methods mainly focus on intra-document reference relations, resulting in performance degradation under cross-document scenarios.
   - Existing methods lack full utilization of structural information in knowledge graphs.
   - Link accuracy decreases when entity mentions are sparse or vaguely defined.
   - The computational complexity of large-scale graph reasoning is a bottleneck for scalability.

**Key Challenge**: Cross-document entity mentions are scattered across different contexts, making it difficult to accurately judge whether they refer to the same entity using only textual features; while knowledge graphs contain rich structured entity relation information, how to effectively integrate textual context and graph structure information remains a challenge.

**Goal**: Design a method that effectively combines structured entity relationships in knowledge graphs with textual contextual embeddings to improve the precision and recall of cross-document coreference resolution.

**Key Insight**: Model cross-document coreference resolution as an entity linking task on graphs, using dynamic linking and message passing to propagate entity information.

**Core Idea**: Map textual mentions to knowledge graph entities through a dynamic linking mechanism, and then iteratively update entity representations using graph message passing to achieve cross-document coreference resolution.

## Method

### Overall Architecture

1. Extract entity mentions $M = \{m_1, m_2, \ldots, m_n\}$ from multiple documents
2. Establish links using entities and relations in the knowledge graph $\mathcal{G} = (V, E)$
3. Compute the similarity between mentions and entities through contextual embeddings
4. Iteratively update entity representations using graph message passing
5. Perform coreference determination based on the refined representations

### Key Designs

#### 1. Basic Framework for Coreference Resolution
- **Function**: Determine whether textual mentions across documents refer to the same entity
- **Mechanism**: Define a similarity function $S(m_i, e_j) = f(m_i, e_j, \mathcal{G})$ that calculates the mention-entity matching scores by combining contextual embeddings and knowledge graph information. Dynamic linking selects the best candidate:

$$\hat{e}_i = \arg\max_{e_j \in V} S(m_i, e_j)$$

Filter weak links via threshold $\theta$: $\mathcal{R} = \{(e_i, e_j) | e_i, e_j \in V, S(m_i, e_i) > \theta\}$

- **Design Motivation**: Unify entity disambiguation and knowledge graph entity alignment into a single framework, relying on graph structure constraints to resolve ambiguity.

#### 2. Knowledge Graph Integration
- **Function**: Model coreference resolution as a linking task on the graph
- **Mechanism**: Compute a contextual embedding $e_j = f(m_j, \mathcal{C})$ for each mention ($\mathcal{C}$ comes from the graph and surrounding text), and evaluate the alignment strength via a similarity matrix $S[m_j, v_i]$. The linking function is: $L(m_j) = \arg\max_{v_i \in V} S(e_j, v_i)$. After linking, integrate using a propagation algorithm: $R = \mathcal{P}(L(M), \mathcal{G})$
- **Design Motivation**: Knowledge graphs provide prior relationships between entities, which serve as strong constraints for cross-document entity disambiguation.

#### 3. Enhanced Entity Linking
- **Function**: Iteratively refine entity representations through message passing in Graph Neural Networks
- **Mechanism**: Estimate linking probability using a sigmoid function: $L(t_j, v_i) = \sigma(\mathbf{f}(t_j, v_i))$. Message passing updates entity embeddings:

$$\mathbf{h}_i^{(t+1)} = \sum_{v_j \in \mathcal{N}(v_i)} \mathbf{W} \cdot \mathbf{h}_j^{(t)} + \mathbf{b}$$

Upon iterative convergence, make coreference determinations using the refined embeddings: $C(t_j, v_i) = \mathcal{R}(\mathbf{h}_i, \mathbf{h}_j)$

- **Design Motivation**: A single-pass link might not be accurate enough; message passing can continually correct entity representations using neighbor information.

### Loss & Training

- **Model Configuration**: Llama-3 and GPT-3.5 as contextual embedding generators
- **Training**: Learning rate 3e-5, 10 epochs, batch size 32
- **Evaluation Metrics**: Precision, Recall, F1 Score

## Key Experimental Results

### Main Results

F1 scores on different models and datasets:

| Model/Method | SP-10K | CoNLL-2012 | ConceptNet | Complex SQ |
|-----------|--------|------------|------------|------------|
| Llama-3 | 71.4 | **73.9** | - | - |
| GPT-3.5 | - | - | 68.6 | 72.5 |
| CorefUD | - | - | - | - (LexGLUE: 64.4) |
| ThaiCoref | **76.8** | **78.7** | - | - |
| Major Entity ID | - | - | 61.2 | 63.0 |
| Event Coref Bank+ | - | - | - | - (GLUE: 69.4) |
| Rationale-centric | 73.9 | 75.2 | - | - |

**ThaiCoref** achieves the highest F1 of 78.7% on CoNLL-2012; Llama-3's F1 on CoNLL-2012 is 73.9%.

### Ablation Study

Performance change after removing each component (F1 Score):

| Model/Method | SP-10K | CoNLL-2012 |
|-----------|--------|------------|
| Llama-3 (full) | 71.4 | 73.9 |
| Llama-3 (ablated) | 69.6 | 72.2 |
| ThaiCoref (full) | 76.8 | 78.7 |
| ThaiCoref (ablated) | 75.3 | 76.6 |
| Rationale-centric (full) | 73.9 | 75.2 |
| Rationale-centric (ablated) | 73.8 | 73.7 |

The F1 scores of all methods decrease after ablation (1-2%), showing that each component contributes.

Entity interaction capture analysis: Llama-3 achieves a precision of 76.5 on direct links, while ThaiCoref reaches up to 80.3.

### Key Findings

1. ThaiCoref surprisingly performs the best on multiple datasets (F1 78.7%), possibly benefiting from its fine-grained annotation strategy.
2. Llama-3 performs robustly in cross-document scenarios (average F1 72-74%), outperforming GPT-3.5.
3. The integration of knowledge graph information brings improvements to all methods, with performance dropping 1-2% after ablation.
4. The Major Entity Identification method shows the lowest performance (F1 ~61-63%), indicating that identifying only major entities is insufficient for solving complex coreference.
5. Iterative updates of graph message passing make a clear contribution to refining entity representations.

## Highlights & Insights

- **Integrating Structured and Unstructured Information**: Effectively integrating the structured relationships of knowledge graphs with the unstructured context of text is a natural direction for cross-document CR.
- **Method Generality**: The framework does not rely on specific LLMs, allowing the embedding generator to be flexibly replaced.
- **Comprehensive Baseline Comparison**: Covers multiple coreference resolution paradigms, including multilingual (CorefUD, ThaiCoref), event (Event Coref Bank+), and rationale/causal (Rationale-centric).

## Limitations & Future Work

1. Dynamic linking accuracy may decrease when entity mentions are sparse or vaguely defined.
2. The generalization capability of the model is limited when training data diversity is insufficient.
3. The computational complexity of graph reasoning may become a bottleneck for large-scale datasets.
4. Mathematical symbols overlap in the method description ($e$ represents both entities and embeddings), which could be improved for better readability.
5. The ablation studies are not fine-grained enough, failing to individually evaluate the independent contributions of dynamic linking, message passing, and threshold selection.
6. There is a lack of direct comparison with the latest LLM-based coreference resolution methods (such as few-shot methods based on GPT-4).

## Related Work & Insights

- **Maverick** (Martinelli et al., 2024): An efficient coreference resolution pipeline that performs excellently in parameter-constrained environments.
- **LQCA** (Liu et al., 2024a): A long-context coreference adaptation method addressing reference relations in long texts.
- **Contrastive CR** (Hsu & Horwood, 2022): Contrastive representation learning for cross-document event/entity coreference.
- **Context Graph** (Xu et al., 2024): Uses LLMs for knowledge representation and reasoning of context graphs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of knowledge graphs and coreference resolution is not entirely new, but the dynamic linking + message passing framework offers some contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — There are many datasets, but some experimental designs are not rigorous enough, and the ablation study lacks fine granularity.
- **Writing Quality**: ⭐⭐⭐⭐ — The structure is complete, but some formula symbols are not clear, and the description of related work is somewhat verbose.
- **Value**: ⭐⭐⭐⭐ — Provides a knowledge graph enhancement perspective for cross-document CR, but the improvement margin is not highly prominent.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Extending Complex Logical Queries on Uncertain Knowledge Graphs](extending_complex_logical_queries_uncertain_knowledge_graphs.md)
- [\[ACL 2025\] Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](kg_llm_trustworthy_qa.md)
- [\[ACL 2025\] Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs](paper_2401_14640.md)
- [\[ACL 2025\] SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation](simgrag_leveraging_similar_subgraphs_for_knowledge_graphs_driven_retrieval-augme.md)
- [\[ICLR 2026\] Graphon Cross-Validation: Assessing Models on Network Data](../../ICLR2026/graph_learning/graphon_cross-validation_assessing_models_on_network_data.md)

</div>

<!-- RELATED:END -->
