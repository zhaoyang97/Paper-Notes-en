---
title: >-
  [Paper Note] A Comprehensive Graph Framework for Question Answering with Mode-Seeking Preference Alignment
description: >-
  [ACL 2025 Findings][NLP Understanding][RAG] This paper proposes the GraphMPA framework, which achieves global document understanding by constructing a hierarchical document graph based on general similarity metrics, and introduces mode-seeking preference optimization to replace traditional DPO for more precise human preference alignment, comprehensively outperforming existing RAG methods across six QA datasets.
tags:
  - "ACL 2025 Findings"
  - "NLP Understanding"
  - "RAG"
  - "Hierarchical Document Graph"
  - "Preference Alignment"
  - "Mode-Seeking"
  - "Community Detection"
date: 2026-05-08
content_hash: 0a63098abd77e93f
---

# A Comprehensive Graph Framework for Question Answering with Mode-Seeking Preference Alignment

**Conference**: ACL 2025 Findings  
**arXiv**: [2506.17951](https://arxiv.org/abs/2506.17951)  
**Code**: [https://github.com/tangquanwei/GraphMPA](https://github.com/tangquanwei/GraphMPA)  
**Area**: NLP Understanding / Question Answering Systems  
**Keywords**: RAG, Hierarchical Document Graph, Preference Alignment, Mode-Seeking, Community Detection

## TL;DR
This paper proposes the GraphMPA framework, which achieves global document understanding by constructing a hierarchical document graph based on general similarity metrics, and introduces mode-seeking preference optimization to replace traditional DPO for more precise human preference alignment, comprehensively outperforming existing RAG methods across six QA datasets.

## Background & Motivation
Retrieval-Augmented Generation (RAG) enhances the question-answering capabilities of LLMs by integrating external knowledge, but faces two core challenges:

**Deficit in Global Understanding**: Traditional RAG employs flat chunk retrieval, which only obtains local fragment information and fails to understand the document structure from a global-to-local perspective as humans do. Even graph-based methods like GraphRAG suffer from uneven graph construction quality and insufficient hierarchical understanding.

**Preference Alignment Discrepancy**: Standard DPO (Direct Preference Optimization) is inherently "mean-seeking", tending to generate a "compromised" response across all preferences rather than finding the truly optimal response mode. This leads to conservative and generic outputs.

**Core Idea**: Simulating the human cognitive process—first organizing documents into a multi-layered graph structure from details to concepts through community detection and hierarchical summarization (like the knowledge network naturally formed during human reading), and then utilizing mode-seeking preference optimization (finding the "mode" of the preference distribution rather than the "mean") to ensure the outputs align with the types of responses that satisfy humans most.

## Method

### Overall Architecture
GraphMPA consists of two core components: (1) hierarchical document graph construction, which organizes original documents into multi-layered graph structures for retrieval; and (2) Mode-Seeking preference alignment, which trains the model to generate responses that better align with strongest human preferences. The inputs are a document collection and a query, and the output is a high-quality, human-preference-aligned answer.

### Key Designs
1. **Hierarchical Document Graph**:

    - **Function**: Transforms documents from a flat list of chunks into a multi-layered graph structure.
    - **Mechanism**: Achieved in four steps:
        - (a) **Text Chunking and Embedding**: Chunks documents and converts them into vectors using an embedding model (e.g., BGE-M3).
        - (b) **Graph Layer Construction**: Treats chunks as nodes and establishes edges based on embedding similarity, forming a document similarity graph.
        - (c) **Community Detection**: Uses the Leiden algorithm (or Louvain) to identify tightly connected node clusters (communities) on the graph, grouping semantically related content together.
        - (d) **Summarization and Recursion**: Generates summaries for each community using an LLM, treats these summaries as new higher-level nodes, and repeats the process recursively to build a multi-layered classification graph.
    - **Design Motivation**: Simulates the progressive human understanding process of "reading paragraphs $\rightarrow$ identifying connections $\rightarrow$ summarizing themes $\rightarrow$ forming outlines", enabling retrieval to capture both detailed (bottom layer) and overview (top layer) information simultaneously.

2. **Mode-Seeking Preference Optimization (MSPO)**:

    - **Function**: Aligns model output with the "most concentrated" mode of human preferences rather than a dispersed average.
    - **Mechanism**: Unlike DPO, which minimizes the difference between preferred and non-preferred pairs (leading to "compromised" outputs), MSPO employs probability-matching constraints to let the model learn the mode of the preference distribution. Concretely, after training, the model's log probability on correct answers is higher and more concentrated (median log probability: MS is around $-5$ vs. DPO around $-25$ vs. SFT around $-150$).
    - **Design Motivation**: Human preferences are not uniformly distributed—people typically have a single "most ideal" response style and quality level. The mean-seeking optimization of DPO dilutes this preference signal, whereas mode-seeking directly targets the peak of the preference distribution.

3. **Hierarchical Retrieval Strategy**:

    - **Function**: Performs retrieval on the multi-layered graph to integrate information across different granularities.
    - **Mechanism**: Given a query, it retrieves the top-$k$ relevant nodes from each layer of the graph, aggregating detailed low-level information and abstract high-level summaries as the input context for the LLM.
    - **Design Motivation**: Retrieving only raw chunks (bottom layer) lacks a global perspective, and retrieving only summaries (top layer) lacks specific details; multi-layer retrieval balances both.

### Loss & Training
Mode-Seeking Preference Optimization modifies the optimization objective of standard DPO—switching from minimizing the log probability margin between preferred and dispreferred generations (dominated by KL divergence, which biases toward the mean) to directly maximizing the probability density at the preference mode. The models use Qwen2.5-7B-Instruct, LLaMa-3.1-8B-Instruct, and Mistral-8B as backbones.

## Key Experimental Results

### Main Results (LLaMa 8B as Backbone, Accuracy %)

| Dataset | Metric | GraphMPA (Ours) | RAPTOR | LightGraphRAG | Basic RAG | Gain |
|--------|------|----------|--------|---------------|-----------|------|
| QUALITY | Acc | **73.65** | 49.66 | 50.83 | 41.73 | +22.82 |
| PubMedQA | MIRAGE | **73.00** | 58.40 | 49.00 | 68.80 | +4.20 |
| MedQA | Acc | **66.54**| 53.10 | 45.18 | 57.34 | +9.20 |
| MedMcQA | Acc | **64.28** | 50.84 | 50.91 | 50.35 | +13.37 |
| QASPER | ROUGE-F1 | **0.3775** | 0.3657 | 0.3585 | 0.3599 | +0.012 |
| RiddleSense | Acc | 47.05 | 45.62 | 45.82 | **60.24** | -13.19 |

### Ablation Study (QUALITY Dataset)

| Configuration | Accuracy | Description |
|------|--------|------|
| Full GraphMPA | **47.05** | Full framework |
| w/ DPO (replacing MS) | 46.06 | Replacing mode-seeking with standard DPO |
| w/o Training | 46.65 | Without preference training |
| w/o Summarization | 41.73 | Without hierarchical summarization |
| w/o Retrieval | 32.10 | Without retrieval (LLM only) |

### Hyperparameter Analysis

| Parameter | Optimal Value | Trend |
|------|--------|------|
| Number of Graph Layers | 2-3 layers | Large gain from 1 layer $\rightarrow$ 2 layers, slight decline at 4 layers |
| Top-$K$ | 3-5 | Too few lacks context, too many introduces noise |

### Key Findings
- **Retrieval is the most critical component**: Performance drops from 47.05 to 32.10 (-31.7%) without retrieval, indicating that external knowledge integration is core.
- **Hierarchical summarization contributes significantly**: Performance drops from 47.05 to 41.73 (-11.3%) without summarization, confirming that global understanding is crucial for question answering.
- **MS outperforms DPO, but with a small margin**: MS is around 1 percentage point higher than DPO, but the concentration of log probability distributions shows a distinct difference.
- **Good cross-model consistency**: Similar performance improvement patterns are observed on Qwen 7B and Mistral 8B.

## Highlights & Insights
- **Precise design intuition of hierarchical document graphs**: The progressive construction of chunk $\rightarrow$ graph $\rightarrow$ community $\rightarrow$ summary $\rightarrow$ high-level graph matches closely with the cognitive process of human reading comprehension, representing a "cognitively aligned" retrieval design.
- **The comparison between Mode-Seeking and Mean-Seeking offers theoretical insights**: It reveals the fact that the implicit assumption of DPO (that preference distribution is unimodal and symmetric) does not hold in practice.
- **Document clustering achieved via Leiden/Louvain community detection** can be directly transferred to other tasks requiring structural document understanding.

## Limitations & Future Work
- GraphMPA performs worse than Basic RAG on RiddleSense (47.05 vs 60.24), potentially because common-sense reasoning tasks do not rely on deep document understanding.
- The difference between MS and DPO in the ablation study is relatively small (~1%), and the practical advantage of mode-seeking may vary across datasets.
- Graph construction relies on LLM-generated summaries, introducing additional computational overhead and fluctuations in summary quality.
- Evaluated only on open-source 7-8B level models; performance and benefits on larger models remain unknown.
- Lacks a direct comparison with the latest GraphRAG frameworks (such as Microsoft's GraphRAG).

## Related Work & Insights
- **vs. RAPTOR**: RAPTOR also adopts a hierarchical chunk summarization approach, but utilizes clustering rather than graph community detection and lacks preference alignment; GraphMPA outperforms it by 24% on QUALITY.
- **vs. LightGraphRAG**: LightGraphRAG is a lightweight graph RAG solution; GraphMPA outperforms it on most datasets through more complete graph construction and preference optimization.
- **vs. Reward-RAG**: Reward-RAG employs a reward model, while GraphMPA uses mode-seeking preference optimization; the two share similar ideas but have different optimization objectives. GraphMPA (73.00) outperforms Reward-RAG (69.20) on PubMedQA.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of hierarchical graph construction and mode-seeking preference alignment is relatively novel, though individual components have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensively evaluated across 6 datasets and 3 models, with thorough ablation and hyperparameter analyses.
- Writing Quality: ⭐⭐⭐ Standard ACL Findings level; the methodology description is relatively clear.
- Value: ⭐⭐⭐⭐ Provides a comprehensive solution of structural understanding and preference alignment for RAG systems, with strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)
- [\[ACL 2025\] SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Analysis](syngraph_a_dynamic_graph-llm_synthesis_framework_for_sparse_streaming_user_senti.md)
- [\[ACL 2025\] Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering](embqa_embedding_odqa.md)
- [\[ACL 2025\] BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering](belle_a_bi-level_multi-agent_reasoning_framework_for_multi-hop_question_answerin.md)
- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](../../ACL2026/nlp_understanding/table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)

</div>

<!-- RELATED:END -->
