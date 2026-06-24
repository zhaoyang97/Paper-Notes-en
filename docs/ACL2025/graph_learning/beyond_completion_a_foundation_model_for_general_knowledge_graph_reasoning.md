---
title: >-
  [Paper Note] Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning
description: >-
  [ACL 2025][Graph Learning][Knowledge Graph Foundation Model] This paper proposes MERRY, a foundation model for knowledge graphs (KGs) that unifiedly handles both in-KG (zero-shot KGC) and out-of-KG (KGQA) reasoning tasks. By fusing textual and structural information via multi-view conditional message passing (CMP), MERRY outperforms existing methods across 28 datasets.
tags:
  - "ACL 2025"
  - "Graph Learning"
  - "Knowledge Graph Foundation Model"
  - "KGC"
  - "KGQA"
  - "Conditional Message Passing"
  - "Text-Structure Fusion"
date: 2026-05-08
content_hash: a51d7d3a5792586a
---

# Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning

**Conference**: ACL 2025  
**arXiv**: [2505.21926](https://arxiv.org/abs/2505.21926)  
**Code**: [https://github.com/zjukg/MERRY](https://github.com/zjukg/MERRY)  
**Area**: Graph Learning / Knowledge Graphs  
**Keywords**: Knowledge Graph Foundation Model, KGC, KGQA, Conditional Message Passing, Text-Structure Fusion

## TL;DR
This paper proposes MERRY, a foundation model for knowledge graphs (KGs) that unifiedly handles both in-KG (zero-shot KGC) and out-of-KG (KGQA) reasoning tasks. By fusing textual and structural information via multi-view conditional message passing (CMP), MERRY outperforms existing methods across 28 datasets.

## Background & Motivation
**Background**: Foundation models in NLP and CV have successfully achieved cross-task transfer. Although the KG domain has seen initial attempts at foundation models like ULTRA, they only utilize structural information and are restricted to in-KG tasks (such as KGC), failing to handle out-of-KG tasks (such as KGQA).

**Limitations of Prior Work**: (a) Models like ULTRA rely solely on structural meta-topology and ignore textual information of entities/relations, which limits contextual reasoning; (b) Text-aware methods (such as StATik) require fine-tuning, resulting in poor generalization; (c) KGC and KGQA methods are largely isolated, lacking a unified framework.

**Key Challenge**: Addressing three challenges simultaneously: the semantic gap between textual and structural modalities, the varying importance of the two modalities across different tasks, and avoiding bias toward specific entities/relations when generalizing across datasets.

**Goal**: To build a KG foundation model that utilizes both text and structure to unifiedly process in-KG and out-of-KG reasoning tasks.

**Key Insight**: Design a dual-channel CMP encoder (Query-CMP for structure + Global-CMP for text) combined with a dynamic text-adaptive fusion module to balance modality weights, and a flexible edge scoring mechanism to adapt to different tasks.

**Core Idea**: Unify KGC and KGQA by encoding query-specific structural representations with QCMP, encoding global semantic representations with GCMP, and dynamically fusing them with DTAF.

## Method

### Overall Architecture
An encoder-decoder architecture:
- **Input**: Query $q$, graph $\mathcal{G}$ (with entity/relation text descriptions), candidate set $\mathcal{C}$
- **Encoding**: Dual-channel CMP $\rightarrow$ Multi-view fusion $\rightarrow$ Edge scoring
- **Decoding**: Cross-attention decoder outputs the candidate probability distribution

### Key Designs

1. **QCMP (Query-Conditional Structural Encoding)**:

    - **Function**: Encodes the KG structure based on specific query conditions to generate query-related entity and relation representations.
    - **Mechanism**: First performs CMP updates on the relation graph (using 4 types of meta-relations: h2h/h2t/t2h/t2t) to obtain the query-conditioned relation embedding $\mathbf{R}_q$, then performs CMP updates on the entity graph to obtain the query-conditioned entity embedding $\mathbf{H}_q$.
    - **Design Motivation**: Adopts the relation graph construction of ULTRA to enable cross-dataset generalization. The conditional initialization forces representations to focus on the query.

2. **GCMP (Global Structural Semantic Encoding)**:

    - **Function**: Propagates textual information across the graph via a CMP architecture to bridge the semantic gap between text and structure.
    - **Mechanism**: Uses a frozen LLM to extract entity text features $\mathcal{X}_e$ (parameter-free, taking the representation of the last token) to initialize entity nodes, which are then propagated via CMP to obtain the global semantic embedding $\mathbf{H}_g$.
    - **Design Motivation**: Direct concatenation of text and structural representations yields poor performance due to major discrepancies in their semantic spaces. Using CMP allows textual information to propagate through the graph structure, aligning the two modalities.

3. **DTAF (Dynamic Text-Adaptive Fusion)**:

    - **Function**: Dynamically balances the weights of textual and structural information to adapt to different task requirements.
    - **Mechanism**: Employs a learnable cross-attention mechanism to compress the LLM's text output into a fixed-length embedding $\mathcal{X}$, which is then fused using learnable weights $\alpha, \beta$: $\mathbf{H}_f = \beta \cdot \mathcal{X}_e + (1-\beta) \cdot \mathbf{H}_{CMP}$
    - **Design Motivation**: KGQA relies more on text, whereas KGC relies more on structure. Thus, task-adaptive fusion weights are essential.

4. **Query-Conditional Edge Scoring**:

    - **Function**: Assigns relevance weights to noisy edges in subgraphs retrieved for KGQA.
    - **Mechanism**: Computes the relevance score of each edge with the query using a bilinear layer, which is normalized and then used in CMP.
    - **Details**: In KGC, all edge weights are set to 1 (simplified), while KGQA utilizes the learned edge weights.

### Loss & Training
- Self-supervised pre-training: Link prediction on a mixed KG dataset (BCE loss).
- Zero-shot evaluation for KGC (no fine-tuning). KGQA is adapted via few-shot learning by introducing an auxiliary relation $REL\_the\_answer\_is$.

## Key Experimental Results

### Zero-shot KGC (Summary of 6 categories out of 28 datasets, MRR/Hits@10)

| Method | IndE(WN) MRR | IndE(FB) MRR | IndER(FB) MRR | IndER(WK) MRR | Total AVG MRR | SOTA Count |
|------|-------------|-------------|--------------|--------------|---------------|--------|
| Supervised SOTA | 0.640 | 0.477 | 0.166 | 0.152 | 0.366 | - |
| ULTRA(3g) | 0.517 | 0.486 | 0.386 | 0.254 | 0.433 | 4/24 |
| ProLINK | 0.553 | 0.494 | 0.372 | 0.234 | 0.433 | 8/24 |
| **MERRY** | **0.563** | 0.486 | 0.378 | **0.282** | **0.445** | **12/24** |

### KGQA (4 Commonsense Reasoning Benchmarks)

| Method | OpenBookQA | CommonsenseQA | OBQA+CSQA Avg |
|------|-----------|---------------|---------------|
| QA-GNN | 64.2 | 73.1 | - |
| GreaseLM | 66.9 | 74.2 | - |
| **MERRY** | **68.0** | **75.3** | Best |

### Key Findings
- MERRY achieves SOTA on 12 out of 24 KGC datasets, showing the broadest coverage.
- The advantage is more pronounced on the harder setting (IndER) which contains both unseen entities and unseen relations—indicating that textual info is highly valuable when handling entirely novel entities/relations.
- For the first time, a single unified model outperforms specialized models like QA-GNN and GreaseLM on KGQA, validating the feasibility of the unified framework.
- Ablation studies demonstrate that both GCMP and DTAF are critical components, as removing either degrades performance.

## Highlights & Insights
- **Unifying in-KG and out-of-KG tasks**: Successfully handles both KGC and KGQA under a single framework for the first time, cleverly transforming QA into link prediction by introducing an auxiliary relation.
- **Elegant dual-channel CMP design**: QCMP captures query-specific structural information, while GCMP aligns textual and structural semantics through graph propagation, yielding a highly clean architecture.
- **DTAF's learnable fusion weights** automatically adapt to the varying degrees of dependence on text/structure across different tasks, avoiding manual hyperparameter tuning.

## Limitations & Future Work
- The parameter-free strategy (taking the last token representation) for LLM-encoded text may result in information loss.
- Pre-training only employs the link prediction objective, without exploring richer auxiliary pre-training tasks.
- High computational overhead due to the dual-channel CMP, LLM encoding, and cross-attention mechanisms.
- Performance on generative KGQA (Open-domain QA) remains untested.
- The robustness of few-shot adaptation for KGQA has not been fully verified.

## Related Work & Insights
- **vs ULTRA**: ULTRA utilizes only structural information, whereas MERRY incorporates the textual modality and extends to KGQA, achieving 12/24 SOTA compared to ULTRA's 4/24 SOTA.
- **vs QA-GNN / GreaseLM**: These are specialized models for KGQA, whereas MERRY performs both KGC and KGQA within a unified framework with superior performance.
- **vs StATik**: StATik requires fine-tuning and has poor generalizability, whereas MERRY achieves zero-shot KGC transfer via pre-training.

## Rating
- Novelty: ⭐⭐⭐⭐ First foundation model to unify in-KG and out-of-KG reasoning; the dual-channel CMP + DTAF design is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 28 datasets with complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Systematically described methodology with rigorous mathematical formulations.
- Value: ⭐⭐⭐⭐⭐ Sets a new benchmark for KG foundation model research; the unified framework paradigm is highly influential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning](a_generative_adaptive_replay_continual_learning_model_for_temporal_knowledge_gra.md)
- [\[ACL 2025\] FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)
- [\[ICLR 2026\] HYPER: A Foundation Model for Inductive Link Prediction with Knowledge Hypergraphs](../../ICLR2026/graph_learning/hyper_a_foundation_model_for_inductive_link_prediction_with_knowledge_hypergraph.md)
- [\[ICLR 2026\] Knowledge Reasoning Language Model: Unifying Knowledge and Language for Inductive Knowledge Graph Reasoning](../../ICLR2026/graph_learning/knowledge_reasoning_language_model_unifying_knowledge_and_language_for_inductive.md)
- [\[ICLR 2026\] FLOCK: A Knowledge Graph Foundation Model via Learning on Random Walks](../../ICLR2026/graph_learning/flock_a_knowledge_graph_foundation_model_via_learning_on_random_walks.md)

</div>

<!-- RELATED:END -->
