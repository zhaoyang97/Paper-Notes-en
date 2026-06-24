---
title: >-
  [Paper Note] GeAR: Graph-enhanced Agent for Retrieval-augmented Generation
description: >-
  [ACL 2025 (Findings)][LLM Agent][Graph-enhanced retrieval] GeAR enhances the multi-hop discovery capabilities of traditional retrievers through a graph expansion mechanism (SyncGE), and combines it with a Gist Memory agent framework to achieve multi-step retrieval reasoning. It outperforms existing SOTA by more than 10% on multi-hop QA datasets like MuSiQue, while consuming fewer tokens and iterations.
tags:
  - "ACL 2025 (Findings)"
  - "LLM Agent"
  - "Graph-enhanced retrieval"
  - "multi-hop question answering"
  - "RAG agent"
  - "knowledge graph expansion"
  - "Gist memory"
date: 2026-05-08
content_hash: 279d00b54401a0bd
---

# GeAR: Graph-enhanced Agent for Retrieval-augmented Generation

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2412.18431](https://arxiv.org/abs/2412.18431)  
**Code**: [https://gear-rag.github.io/](https://gear-rag.github.io/)  
**Area**: Agent / Retrieval-Augmented Generation  
**Keywords**: Graph-enhanced retrieval, multi-hop question answering, RAG agent, knowledge graph expansion, Gist memory

## TL;DR

GeAR enhances the multi-hop discovery capabilities of traditional retrievers through a graph expansion mechanism (SyncGE), and combines it with a Gist Memory agent framework to achieve multi-step retrieval reasoning. It outperforms existing SOTA by more than 10% on multi-hop QA datasets like MuSiQue, while consuming fewer tokens and iterations.

## Background & Motivation

**Background**: Retrieval-augmented generation (RAG) has become the mainstream solution to address the hallucination problem in large language models. Traditional RAG relies on sparse retrievers (such as BM25) or dense retrievers (such as DPR) for single-step retrieval, and then concatenates the retrieved passages into the prompt for the LLM to generate answers.

**Limitations of Prior Work**: For multi-hop reasoning scenarios (e.g., "Where did the British-Canadian researcher who won the 2024 Nobel Prize in Physics obtain their PhD?"), single-step retrieval struggles to find all relevant bridge documents simultaneously. Existing multi-step retrieval solutions (e.g., ITER-RETGEN, IRCoT) can perform iterative queries but often require high token consumption and multiple iterations, which leads to lower efficiency and a higher risk of losing key information in intermediate steps.

**Key Challenge**: Multi-hop retrieval requires establishing reasoning chains across different documents, but traditional retrievers naturally lack the ability to perceive connections between documents. Existing graph-based retrieval methods (such as those based on knowledge graphs) possess relational mapping capabilities, but constructing and maintaining them is costly, and they are difficult to integrate compatibly with existing retrievers.

**Goal**: Design a plug-and-play graph expansion mechanism that can enhance any traditional retriever, and build an efficient multi-step retrieval agent framework on top of it.

**Key Insight**: The authors observe that "proximal triples" can be extracted from the retrieved passages. These triples can then be expanded on a constructed document graph to discover bridge documents that are unreachable through keyword matching alone.

**Core Idea**: Using the approach of "extract triples $\to$ graph expansion $\to$ Gist Memory accumulation," traditional retrieval and graph structure expansion are simultaneously leveraged in each retrieval step, and Gist Memory is used to simulate human working memory to preserve crucial information across steps.

## Method

### Overall Architecture

GeAR is a multi-step iterative retrieval framework. Each retrieval step consists of three stages: (1) a base retriever (e.g., BM25) retrieves passages based on the current query; (2) the SyncGE graph expansion mechanism extracts triples from the retrieved passages and expands on the document graph to obtain more relevant passages; (3) the Gist Memory module accumulates and stores the key information obtained from the expansion in the form of triples. A reasoning module determines whether the current info is sufficient to answer the question—if insufficient, it rewrites the query and proceeds to the next round; if sufficient, it maps the triples in the Gist Memory back to their original passages and returns the final results after RRF fusion reranking.

### Key Designs

1. **SyncGE (Synchronous Graph Expansion)**:

    - **Function**: Enhances the multi-hop discovery capabilities of the base retriever.
    - **Mechanism**: First, an LLM is used to extract "proximal triples" (structured information in the form of (subject, relation, object) relevant to the query) from the retrieved passages. Then, Triple Linking is performed to map these triples to the nearest real triples in a pre-constructed document triple index. Finally, a Diverse Triple Beam Search is executed on the document graph—expanding along the graph edges starting from the linked triples and using beam search to retain the most diverse expansion paths, ultimately linking the expanded new triples back to their source passages.
    - **Design Motivation**: Traditional retrievers can only match via text similarity and cannot discover bridge documents that are semantically related but do not share lexical overlap. Graph expansion naturally bridges this gap by utilizing entity relations between documents. Using beam search ensures diversity in expansion and avoids falling into local optima.

2. **Gist Memory**:

    - **Function**: Accumulates and retains key reasoning information across retrieval steps.
    - **Mechanism**: Simulates the working memory mechanism of the human hippocampus. After each retrieval step, new proximal triples are extracted from the expanded passages and appended to the Gist Memory. In subsequent steps, the triples stored in Gist Memory are used both to assist in extracting new triples for the next round (providing context) and to help the reasoning module evaluate whether sufficient information has been gathered. In the final retrieval phase, the triples in the Gist Memory are mapped back to their original passages via Passage Linking and fused with the standard retrieved passages.
    - **Design Motivation**: The core challenge of multi-step retrieval is information loss—key clues obtained in early steps can easily be diluted or forgotten in subsequent steps. Gist Memory retains information in the form of structured triples rather than raw text, which compresses the storage footprint while preserving critical reasoning chains.

3. **Reasoning and Query Rewriting Module**:

    - **Function**: Decides when to stop retrieval and how to generate the query for the next step.
    - **Mechanism**: After each retrieval step, all current triples in the Gist Memory are fed into the LLM along with the original question to determine if there is sufficient evidence for an answer. If information is insufficient, the LLM outputs both a description of the missing information and a rewritten query. The maximum number of iterations is set to 4.
    - **Design Motivation**: Compared to fixed-step multi-step retrieval, adaptive termination saves resources for simpler questions and ensures retrieval thoroughness for complex ones. Query rewriting deduces the missing information based on known details, making it more target-oriented than simple query expansion.

### Loss & Training

GeAR does not require end-to-end training. Both the graph expansion and Gist Memory construction rely on the in-context learning capabilities of LLMs (achieving tasks like triple extraction and reasoning judgments via carefully designed prompts). The document triple index is pre-extracted and constructed from the corpus offline using an LLM.

## Key Experimental Results

### Main Results

The retrieval performance (Recall@10/20) and question-answering performance (EM/F1) are evaluated on three multi-hop QA datasets:

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| MuSiQue | Recall@10 | ~65% | ~55% (IRCoT) | +10%+ |
| HotpotQA | Recall@10 | SOTA | IRCoT/ITER-RETGEN | Comparable/Better |
| 2WikiMultiHopQA | Recall@10 | SOTA | IRCoT | Significant Gain |
| MuSiQue | EM (QA) | SOTA | Best Prior | +10%+ |
| HotpotQA | F1 (QA) | SOTA | Best Prior | Comparable |

### Ablation Study

| Configuration | MuSiQue Recall | Description |
|------|---------------|------|
| Full GeAR | Highest | Complete system |
| w/o Graph Expansion | Significant Decrease | Graph expansion is the core contribution |
| w/o Gist Memory | Moderate Decrease | Information accumulation is crucial for multi-hop QA |
| w/o Query Rewriting | Decrease | Adaptive query rewriting is beneficial |
| BM25 Baseline Only | Lowest | Single-step retrieval is highly insufficient |

### Key Findings

- **Graph expansion is the most critical module**: Performance drops significantly without SyncGE, demonstrating that graph-structured associations between documents are central to multi-hop retrieval.
- **Pronounced efficiency advantages**: GeAR requires only 2-3 iterations on average and consumes far fewer tokens than methods like IRCoT, showing that graph expansion can achieve cross-document discovery in a single step that traditional methods require multiple steps to accomplish.
- **Largest advantage is observed on the most difficult dataset, MuSiQue**: MuSiQue requires 2-4 reasoning hops which traditional methods struggle with, whereas graph expansion is naturally suited for such scenarios. The advantage is relatively smaller on the simpler HotpotQA (2 hops).
- **Plug-and-play capability**: SyncGE can enhance different base retrievers (e.g., BM25, DPR), demonstrating strong generalizability.

## Highlights & Insights

- The **plug-and-play design combining graph expansion with traditional retrievers** is highly clever. It does not require replacing existing retrievers but simply appends a graph expansion step to their output, lowering the barrier to practical deployment.
- The design idea of using **Gist Memory to compress information with triples** can be transferred to other tasks requiring multi-step reasoning (such as multi-step mathematical reasoning or long-document comprehension). Replacing raw text with structured intermediate representations can dramatically reduce context length.
- **Diverse Triple Beam Search** is a reusable trick that ensures both coverage and diversity during graph search, avoiding the local clustering issues typical of traditional BFS/DFS.

## Limitations & Future Work

- **By-design reliance on LLMs for triple extraction**: Each retrieval step requires multiple LLM calls (for triple extraction, reasoning, and query rewriting), which may lead to high latency and cost, making it less suitable for scenarios with strict real-time requirements.
- **Construction cost of the offline triple index**: Extracting triples and building graph indices for the entire corpus beforehand can be expensive for large-scale corpora.
- **Evaluation limited to multi-hop QA**: The approach has not been validated in other RAG scenarios, such as open-domain dialogue or fact verification, so its generalizability remains to be fully explored.
- **Future Directions**: Exploring lightweight models (like small specialized extractors) instead of LLMs for triple extraction to reduce cost; combining graph expansion with vector databases for more flexible hybrid retrieval.

## Related Work & Insights

- **vs IRCoT**: IRCoT uses Chain-of-Thought for iterative retrieval but relies purely on text matching, lacking a graph structure. GeAR leverages graph expansion to discover multi-hop bridge documents in a single step.
- **vs ITER-RETGEN**: ITER-RETGEN alternates iterative generation and retrieval, which incurs high computational overhead and easily loses information. GeAR's Gist Memory preserves information across steps much better.
- **vs GraphRAG (Microsoft)**: GraphRAG requires full community graph construction and global summarization, which is more expensive and designed for summary-type queries. GeAR is more lightweight and focuses on precise multi-hop retrieval.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of graph expansion and Gist Memory is novel, though multi-step RAG is not an entirely new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets with both retrieval and QA evaluations plus ablation studies, but lacks a detailed efficiency comparison.
- Writing Quality: ⭐⭐⭐⭐ The project page is well-made, the method description is clear, and pseudocode helps comprehension.
- Value: ⭐⭐⭐⭐ The plug-and-play design is highly practical, the 10%+ improvement is significant, providing good reference value for multi-hop RAG research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Evaluating Retrieval-Augmented Generation Agents for Autonomous Scientific Discovery in Astrophysics](../../ICML2025/llm_agent/evaluating_retrieval-augmented_generation_agents_for_autonomous_scientific_disco.md)
- [\[ICLR 2026\] Aria: an Agent for Retrieval and Iterative Auto-Formalization via Dependency Graph](../../ICLR2026/llm_agent/aria_an_agent_for_retrieval_and_iterative_auto-formalization_via_dependency_grap.md)
- [\[ICLR 2026\] GTool: Graph Enhanced Tool Planning with Large Language Model](../../ICLR2026/llm_agent/gtool_graph_enhanced_tool_planning_with_large_language_model.md)
- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [\[ICLR 2026\] R-WoM: Retrieval-augmented World Model for Computer-use Agents](../../ICLR2026/llm_agent/r-wom_retrieval-augmented_world_model_for_computer-use_agents.md)

</div>

<!-- RELATED:END -->
