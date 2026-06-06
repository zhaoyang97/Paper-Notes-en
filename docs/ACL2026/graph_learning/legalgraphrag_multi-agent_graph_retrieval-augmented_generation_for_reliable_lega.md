---
title: >-
  [Paper Note] LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning
description: >-
  [ACL2026][Graph Learning][Legal RAG] LegalGraphRAG constructs a hierarchical legal graph using fact, ontology, and rule graphs. It employs a Researcher-Auditor-Adjudicator multi-agent workflow to perform retrieval…
tags:
  - "ACL2026"
  - "Graph Learning"
  - "Legal RAG"
  - "Hierarchical Knowledge Graph"
  - "Multi-agent"
  - "Evidence Verification"
  - "Traceable Reasoning"
date: 2026-05-08
content_hash: a37f14c05b93b644
---

# LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning

**Conference**: ACL2026  
**arXiv**: [2605.28120](https://arxiv.org/abs/2605.28120)  
**Code**: https://github.com/XMUDeepLIT/LegalGraphRAG  
**Area**: GraphRAG / Legal Reasoning  
**Keywords**: Legal RAG, Hierarchical Knowledge Graph, Multi-agent, Evidence Verification, Traceable Reasoning  

## TL;DR
LegalGraphRAG constructs a hierarchical legal graph using fact, ontology, and rule graphs. It employs a Researcher-Auditor-Adjudicator multi-agent workflow to perform retrieval, verification, and adjudication, enhancing accuracy and evidence traceability in legal judgment generation.

## Background & Motivation
**Background**: RAG is a common technique for adapting general LLMs to specialized domains. GraphRAG further organizes documents into relational graphs to support multi-hop retrieval and more coherent reasoning. Legal reasoning relies heavily on external knowledge due to the complex dependencies between case facts, legal articles, and judicial interpretations.

**Limitations of Prior Work**: Standard RAG treats text chunks as independent retrieval units, which tends to retrieve contexts based solely on surface semantic similarity. Although traditional GraphRAG provides structure, many implementations remain flat, making it difficult to distinguish between case facts, abstract legal articles, and application conditions. More critically, the retrieve-then-generate paradigm often lacks explicit evidence verification, leading models to provide seemingly correct but untraceable judgments using irrelevant materials.

**Key Challenge**: Legal tasks must simultaneously "find all relevant evidence" and "use only valid evidence." Broader retrieval introduces more noise, while narrower retrieval may miss critical legal articles or similar cases. Without hierarchical organization and verification mechanisms, it is challenging for LLMs to determine which evidence truly supports a verdict.

**Goal**: The authors aim to build a GraphRAG framework specifically for legal reasoning that can organize legal knowledge across different abstraction levels, verify evidence applicability before generating judgments, and output traceable legal bases.

**Key Insight**: The paper first conducts a preliminary study proving that flat retrieval suffers from granularity bias and that standard RAG is highly sensitive to irrelevant documents. The solution is then decomposed into two components: HierarGraph to address knowledge granularity and a multi-agent workflow to solve evidence verification.

**Core Idea**: Legal knowledge is divided into Fact/Ontology/Rule triple-layered graphs. The Researcher retrieves candidate evidence, the Auditor verifies the applicability of legal articles, and the Adjudicator synthesizes verified evidence to generate the judgment.

## Method
The key to LegalGraphRAG is not merely "putting legal documents into a graph" but structuring the legal reasoning process: evidence is first sought in the hierarchical graph based on case facts, then verified using checklists and judicial interpretations in the rule graph, and finally, judgments are generated based only on the verified subgraphs.

### Overall Architecture
The input consists of a description of criminal facts $f$ and a defendant $d$. The system first constructs a legal knowledge graph $KG=\Phi(\mathcal{D})$ based on an offline legal corpus $\mathcal{D}$. For a query, the retriever obtains the context $\mathcal{C}=\mathcal{R}(f,d,KG)$ from the graph, and the generator infers the charge $y$ based on the query and evidence. LegalGraphRAG instantiates this process into two stages: Hierarchical Knowledge Construction and Evidence-based Legal Reasoning.

Stage one constructs the HierarGraph, organizing historical cases, legal articles, judicial interpretations, case features, and charges into different subgraphs. Stage two is executed sequentially by three agents: the Researcher retrieves candidate cases and articles from the ontology/fact graphs; the Auditor checks the applicability of articles based on the rule graph; and the Adjudicator aggregates the confirmed articles, cases, and charge nodes to generate the final judgment with citations.

### Key Designs
1. **HierarGraph (Triple-layered Legal Graph)**:
    - **Function**: Explicitly separates concrete facts, abstract legal concepts, and rule application conditions to prevent flat graphs from mixing information of different granularities.
    - **Mechanism**: The Fact Graph connects Cases, Articles, and Offense nodes; the Ontology Graph abstracts original case facts into dimensions like defendant attributes, criminal acts, victim characteristics, and subjective intent, using kNN and Leiden communities to organize similar cases; the Rule Graph connects articles with judicial interpretations and attaches Diagnostic Checklists to articles.
    - **Design Motivation**: Legal judgments often depend on whether factual details satisfy legal conditions. Semantic similarity alone might find similar narratives while missing the abstract rules that determine the charge.

2. **Researcher (Multi-path Evidence Retrieval)**:
    - **Function**: Covers different types of candidate evidence from the hierarchical graph.
    - **Mechanism**: The retrieval result is the union of three operators $\mathcal{R}(q)=\mathcal{R}_{sem}(q)\cup\mathcal{R}_{com}(q)\cup\mathcal{R}_{chg}(q)$, corresponding to semantic matching retrieval, community expansion retrieval, and charge-anchored retrieval.
    - **Design Motivation**: Single-path retrieval often biases toward high-frequency facts or local similarities; multi-path retrieval ensures coverage of direct similarities, community context, and articles relevant to potential charges.

3. **Auditor & Adjudicator (Evidence Loop)**:
    - **Function**: Filters invalid evidence and binds the final verdict to verifiable grounds.
    - **Mechanism**: For each candidate article, the Auditor uses Diagnostic Checklists and Judicial Interpretations to verify if case facts meet application conditions, pruning inapplicable articles and related nodes. The Adjudicator then uses the verified $\mathcal{A}^f$, $\mathcal{C}^f$, and $\mathcal{O}^f$ to generate the judgment $\mathcal{J}=Adjudicator(q\oplus\mathcal{A}^f\oplus\mathcal{C}^f\oplus\mathcal{O}^f)$.
    - **Design Motivation**: Legal scenarios cannot tolerate "correct answers with unsupported evidence." Explicit verification transforms black-box generation into an auditable chain of evidence.

### Loss & Training
This paper primarily focuses on framework and system evaluation and does not propose an end-to-end training loss. Implementation uses GPT-4o-mini for graph construction, BGE-m3 for generating embeddings, and various backbones during inference, with Qwen3-8B as the default for main experiments. Evaluation metrics include Accuracy and Micro-F1 on CAIL2018 and CMDL datasets, covering criminal sub-domains such as Public Safety, Economic Offenses, Social Order, and Person Rights.

## Key Experimental Results

### Main Results
The paper uses a preliminary study to quantify two problems with standard RAG: flat retrieval cannot handle knowledge granularity, and the lack of verification makes it extremely sensitive to noise. Below is the decline in generation quality under retrieval noise.

| Method | Charge ACC | Articles ACC | Term MAE (Months) | vs. Correct Context |
| :--- | :--- | :--- | :--- | :--- |
| RAG (Correct Context) | 42.8 | 74.7 | 24.3 | Baseline |
| RAG + 2 Irrelevant Docs | 34.9 | 57.2 | 27.7 | Charge -7.9, Articles -17.5, MAE +3.4 |
| RAG + 4 Irrelevant Docs | 32.9 | 51.1 | 28.4 | Charge -9.9, Articles -23.6, MAE +4.1 |
| RAG + 6 Irrelevant Docs | 29.8 | 46.8 | 31.7 | Charge -13.0, Articles -27.9, MAE +7.4 |

In formal evaluations, LegalGraphRAG achieves improvements ranging from 6.3% to 19.1% over strong baselines on CAIL and CMDL; average improvements over LegalDelta and ADAPT are 7.1% and 6.7%, respectively. It reaches a peak performance of 78.7% on CMDL when combined with different backbones.

### Ablation Study
| Configuration | CAIL ACC | Δ | Description |
| :--- | :--- | :--- | :--- |
| LegalGraphRAG (Full) | 40.9 | - | Full hierarchical graph + three-agent workflow |
| w/o HierarGraph | 33.7 | -7.2 | Largest drop, showing hierarchical organization is critical |
| w/o Researcher | 36.9 | -4.0 | Insufficient coverage from multi-path retrieval |
| w/o Semantic Match | 39.1 | -1.8 | Direct semantic retrieval still contributes |
| w/o Community Exp. | 38.5 | -2.4 | Community expansion aids structural context |
| w/o Charge-Anchored | 39.3 | -1.6 | Charge anchoring complements legal grounds |
| w/o Auditor | 37.5 | -3.4 | Lack of verification reduces judgment reliability |

### Key Findings
- Flat retrieval exhibits granularity bias. The preliminary study shows that a naive hierarchical strategy outperforms a flat strategy by 25.3% in retrieval performance.
- Irrelevant documents rapidly degrade standard RAG: with 6 irrelevant docs, article prediction accuracy drops from 74.7 to 46.8, and sentence MAE increases from 24.3 to 31.7 months.
- HierarGraph is the most critical component, with its removal leading to a 7.2 drop in CAIL ACC. Researcher and Auditor contribute drops of 4.0 and 3.4 respectively, indicating that both retrieval coverage and evidence verification are essential.
- The paper emphasizes that LegalGraphRAG increases the proportion of "Traceable Correct" results and reduces "unsupported correctness" (correct answers without a supporting evidence chain).

## Highlights & Insights
- The paper accurately identifies the core problem of legal RAG: it is not about "whether there is retrieval," but whether the retrieved information is at the correct legal granularity and whether it has been verified.
- The triple-layer split of HierarGraph is highly rational for the domain. Case facts, legal concepts, and rule conditions are inherently different types of nodes; forcing them into a flat structure causes the model to get lost in noise.
- The division of labor between Researcher-Auditor-Adjudicator aligns with the legal workflow: gathering materials, checking materials, and reaching a conclusion. This structure is more auditable than a single LLM reading context for a direct answer.
- The emphasis on "unsupported correctness" is vital. High-stakes domains like law and medicine should not only look at the final answer accuracy but also whether those answers are supported by valid evidence.

## Limitations & Future Work
- The authors explicitly state that the current framework only processes unimodal textual legal evidence. Real judicial scenarios include non-textual evidence such as photos, surveillance videos, handwritten scans, and trial recordings.
- Currently, non-textual evidence requires transcription or description, which may lose visual/auditory details (e.g., cues for judging intent vs. negligence in videos).
- Graph construction depends on GPT-4o-mini and embedding models; errors in document parsing, ontology extraction, or checklist generation will be inherited by subsequent agents.
- Future directions could include incorporating multimodal nodes into the Fact Graph, allowing textual testimony, visual evidence, and audio evidence to cross-verify, moving closer to a complete evidence chain for smart courts.

## Related Work & Insights
- **vs. Naive RAG**: Naive RAG provides retrieval context directly to the LLM, lacking hierarchical structure and verification; LegalGraphRAG organizes legal knowledge before verifying evidence applicability.
- **vs. Standard GraphRAG**: Generic GraphRAG has relational structures but may not distinguish between facts, ontology, and rule layers; the hierarchical graph in this paper aligns better with legal ontologies.
- **vs. legal-specific LLM / SFT**: Specialized legal models internalize knowledge into parameters, which is costly and prone to forgetting; LegalGraphRAG enhances reasoning with external knowledge and evidence chains, offering better updatability.
- **Insights for Future Work**: Domain-specific RAG should explicitly model "evidence granularity" and "evidence verification" rather than merely optimizing top-k retrieval or reranking.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Natural combination of hierarchical legal graphs and a three-agent verification process with deep domain adaptation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Includes preliminary studies, main experiments, reliability analysis, case studies, and ablation; multimodal and cross-jurisdictional generalization remains to be verified.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear motivation and system structure, although some tables are large and complex in layout.
- **Value**: ⭐⭐⭐⭐⭐ Highly valuable for legal RAG, trustworthy QA, and evidence-grounded generation in high-stakes domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment](ea-agent_a_structured_multi-step_reasoning_agent_for_entity_alignment.md)
- [\[ACL 2026\] MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation](megarag_multimodal_knowledge_graph-based_retrieval_augmented_generation.md)
- [\[ACL 2026\] STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation](stem_structure-tracing_evidence_mining_for_knowledge_graphs-driven_retrieval-aug.md)
- [\[CVPR 2026\] M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](../../CVPR2026/graph_learning/m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)
- [\[ACL 2026\] TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation](tagrag_tag-guided_hierarchical_knowledge_graph_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->
