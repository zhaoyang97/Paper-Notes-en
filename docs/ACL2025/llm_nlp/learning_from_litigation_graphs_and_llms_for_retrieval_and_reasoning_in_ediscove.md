---
title: >-
  [Paper Note] Learning from Litigation: Graphs and LLMs for Retrieval and Reasoning in eDiscovery
description: >-
  [ACL 2025][LLM (Other)][eDiscovery] This paper proposes the DISCOG (DISCOvery Graph) system, which integrates knowledge graphs with LLM-driven reasoning for document retrieval and classification in electronic discovery (eDiscovery). It outperforms strong baselines on both balanced and imbalanced datasets, reducing litigation document review costs by approximately 98% in real-world deployment.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "eDiscovery"
  - "Knowledge Graphs"
  - "Document Retrieval"
  - "Legal Reasoning"
  - "LLM Enhancement"
date: 2026-05-08
content_hash: 6401bfec982a183a
---

# Learning from Litigation: Graphs and LLMs for Retrieval and Reasoning in eDiscovery

**Conference**: ACL 2025  
**arXiv**: [2405.19164](https://arxiv.org/abs/2405.19164)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: eDiscovery, Knowledge Graphs, Document Retrieval, Legal Reasoning, LLM Enhancement

## TL;DR

This paper proposes the DISCOG (DISCOvery Graph) system, which integrates knowledge graphs with LLM-driven reasoning for document retrieval and classification in electronic discovery (eDiscovery). It outperforms strong baselines on both balanced and imbalanced datasets, reducing litigation document review costs by approximately 98% in real-world deployment.

## Background & Motivation

**Background**: Electronic discovery (eDiscovery) is a core component of legal practice, requiring the identification of documents relevant to a production request from massive document collections. Traditional practices rely on manual review by attorneys, though AI and NLP technologies have gradually improved document review efficiency.

**Limitations of Prior Work**: Existing approaches still struggle with processing legal entities (e.g., case names, court citations), legal citation relationships, and complex legal document structures. Traditional information retrieval methods like BM25 lack an understanding of the relational structure between legal documents; pure text classification methods ignore the rich networks formed by citations, parties, and courts among documents; and existing graph-based methods fail to effectively utilize the reasoning capabilities of LLMs to enhance determination.

**Key Challenge**: Relevance determination for legal documents depends not only on the semantic matching of textual content but also heavily on structured relationships between documents (such as citing the same precedent or involving the same parties). Existing methods cannot capture both levels of information simultaneously.

**Goal**: Build an eDiscovery system that integrates knowledge graphs (capturing structured relationships between documents) and LLM reasoning (providing semantic understanding and complex decision-making) to substantially reduce manual review costs while maintaining high accuracy.

**Key Insight**: Legal documents naturally contain rich structured information—citation networks, relationships between parties, and court hierarchies. Constructing a knowledge graph from this information and combining it with LLMs to reason over unstructured text content establishes a complementary dual-determination mechanism.

**Core Idea**: Design a hybrid system, DISCOG, that integrates knowledge graph retrieval and LLM reasoning. It first utilizes graph structures to narrow down candidate documents and provide relational context, and then leverages LLMs for fine-grained relevance reasoning.

## Method

### Overall Architecture

The overall pipeline of the DISCOG system consists of three core stages: (1) Knowledge Graph Construction—extracting entities (cases, courts, parties, regulatory clauses, etc.) and relationships (citations, jurisdictions, involvements, etc.) from legal documents to build a document-level knowledge graph; (2) Graph-Enhanced Retrieval—utilizing structural information on the graph (such as multi-hop relationships and community detection) to rank and pre-filter candidate documents; (3) LLM-Driven Reasoning—performing fine-grained relevance classification and reasoning using LLMs on the candidate documents returned by graph retrieval, combined with their graph context.

### Key Designs

1. **Legal Knowledge Graph Construction Module**:

    - **Function**: Automatically extract structured entities and relationships from legal documents to construct a domain-specific knowledge graph.
    - **Mechanism**: Use NER and relation extraction models to identify legal entities (case names, court names, statute citations, litigant names, etc.), and build the graph structure based on citation links, public litigants, and jurisdictional courts between documents. Nodes represent documents and entities, while edges represent various legal relationships.
    - **Design Motivation**: Legal documents form intricate relational networks through citations and common parties. This structured information is critical for determining document relevance but is overlooked by traditional text-only methods.

2. **Graph-Enhanced Document Ranking and Classification**:

    - **Function**: Enhance document retrieval and classification quality utilizing the topological structure and graph embeddings of the knowledge graph.
    - **Mechanism**: Train graph embeddings (such as TransE, ComplEx, etc.) on the knowledge graph to encode the graph structure into vector representations. Combine document text features with graph embedding features for ranking. Utilize multi-hop relationships and community structures on the graph to discover potentially relevant documents (e.g., documents citing the same precedent are more likely to be relevant to the same litigation request).
    - **Design Motivation**: Graph structures can capture implicit associations between documents—two documents may have low textual semantic similarity but are highly relevant because they cite the same precedent or involve the same statutory provision.

3. **LLM-Enhanced Legal Reasoning**:

    - **Function**: Leverage Large Language Models for fine-grained relevance decision-making on candidate documents, providing explainable reasoning incorporated with graph context.
    - **Mechanism**: Feed relationship information from the knowledge graph as additional context into the LLM, enabling the model to determine relevance not only based on textual content but also based on the document's position within the legal knowledge system. The LLM can generate chains of reasoning to explain why a document is relevant to a specific production request.
    - **Design Motivation**: Pure graph-based methods lack deep semantic understanding of text, while pure LLM methods lack structured relational information. Combining them compensates for each other's weaknesses.

### Loss & Training

Graph embedding training utilizes standard knowledge graph completion loss (such as margin-based ranking loss in TransE). The overall system adopts a cascaded training strategy, where the graph module is trained first, followed by training and optimizing the LLM reasoning module using the outputs of the graph module.

## Key Experimental Results

### Main Results

| Method | Balanced Dataset F1 | Imbalanced Dataset F1 | Precision | Recall |
|------|-------------|---------------|-----------|--------|
| BM25 | Baseline | Baseline | Low | Medium |
| BERT | Medium | Medium | Medium | Medium |
| TransE | Medium | Medium | Medium | Medium |
| GraphSAGE | Mid-High | Mid-High | Medium | High |
| DISCOG | **Highest** | **Highest** | **Highest** | **Highest** |

### Real-World Deployment Results

| Metric | Traditional Manual Review | DISCOG System | Improvement |
|------|-----------|-----------|------|
| Document Review Cost | 100% | ~2% | ↓**98%** |
| F1-score | Manual Baseline | Outperformed | Outperforms baseline methods across F1, Precision, and Recall |

### Key Findings

- The combination of knowledge graphs and LLM reasoning comprehensively outperforms methods using only one of them across F1, Precision, and Recall.
- On imbalanced datasets (which are closer to real-world scenarios where relevant documents are significantly fewer than irrelevant ones), the advantages of DISCOG are even more pronounced.
- Achieving approximately 98% cost reduction in actual deployment demonstrates extremely high commercial value.
- The graph structural info contributes significantly to recall—it can discover documents that are not textually matchable but are linked through citation networks.

## Highlights & Insights

- **Commercial Viability with 98% Cost Reduction**: This is not just an improvement in academic metrics, but a validated commercial value in real-world litigation scenarios. In the legal industry, document review costs account for a considerable proportion of total litigation costs, making a 98% reduction highly compelling.
- **Complementary Graph + LLM Framework**: The knowledge graph provides the "skeleton" of structured relationships, while the LLM provides the "flesh and blood" of semantic understanding. This architectural paradigm can be directly transferred to other domains requiring the integration of structured knowledge and natural language understanding, such as medical literature retrieval and patent analysis.
- **System Design Oriented Towards Real-World Legal Scenarios**: Rather than a pure academic demo, this system considers industrial deployment requirements such as imbalanced data and explainability.

## Limitations & Future Work

- The paper primarily focuses on the eDiscovery scenario; transferring it to other legal tasks (e.g., contract review, regulatory compliance checks) will require additional adaptation.
- The construction quality of the knowledge graph highly depends on the accuracy of entity and relation extraction, which may require extra annotations in emerging legal fields.
- The cost and latency of the LLM reasoning component could become a bottleneck on large-scale document collections.
- Privacy protection and data security issues, which are especially critical in the legal domain, were not discussed in detail.

## Related Work & Insights

- **vs. Pure BM25/TF-IDF**: Traditional retrieval cannot capture structured legal relationships between documents.
- **vs. Pure BERT Classification**: BERT only focuses on textual semantics, ignoring graph structural information such as citation networks.
- **vs. GraphSAGE and Other Graph Methods**: Pure graph methods lack deep semantic understanding of the textual content.
- The Graph + LLM paradigm of DISCOG provides an effective system architecture template for legal AI.

## Rating

- Novelty: ⭐⭐⭐ Although the combination of knowledge graphs and LLMs has been explored in other domains, its application and system integration in the eDiscovery scenario are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes performance comparisons on balanced/imbalanced datasets and real-world deployment data, though it lacks finer ablation study details.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, and the system description is comprehensive.
- Value: ⭐⭐⭐⭐⭐ The practical deployment result of a 98% cost reduction is highly compelling, making it an excellent case study combining academic research with industrial application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks](can_we_further_elicit_reasoning_in_llms_critic-guided_planning_with_retrieval-au.md)
- [\[ACL 2025\] Problem-Solving Logic Guided Curriculum In-Context Learning for LLMs Complex Reasoning](problem-solving_logic_guided_curriculum_in-context_learning_for_llms_complex_rea.md)
- [\[ACL 2025\] ATRIE: Automating Legal Interpretation with LLMs: Retrieval, Generation, and Evaluation](atrie_legal_interpretation.md)
- [\[ACL 2025\] Uni-Retrieval: A Multi-Style Retrieval Framework for STEM's Education](uni-retrieval_a_multi-style_retrieval_framework_for_stems_education.md)
- [\[ACL 2025\] CER: Confidence Enhanced Reasoning in LLMs](cer_confidence_enhanced_reasoning.md)

</div>

<!-- RELATED:END -->
