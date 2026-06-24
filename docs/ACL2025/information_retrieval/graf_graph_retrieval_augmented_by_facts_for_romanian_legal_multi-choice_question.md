---
title: >-
  [Paper Note] GRAF: Graph Retrieval Augmented by Facts for Romanian Legal Multi-Choice Question Answering
description: >-
  [ACL 2025][Information Retrieval & RAG][Legal Question Answering] This paper proposes the GRAF algorithm, which integrates a legal knowledge graph (Law-RoG) and Graph Attention Networks for Romanian legal multiple-choice question answering, while open-sourcing JuRO (10,836 questions), the first Romanian legal MCQA dataset, and CROL, a legal corpus.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Legal Question Answering"
  - "Knowledge Graph"
  - "Graph Attention Network"
  - "Romanian"
  - "Low-resource NLP"
date: 2026-05-08
content_hash: 63a35c025a481231
---

# GRAF: Graph Retrieval Augmented by Facts for Romanian Legal Multi-Choice Question Answering

**Conference**: ACL 2025  
**arXiv**: [2412.04119](https://arxiv.org/abs/2412.04119)  
**Code**: [github.com/craciuncg/GRAF](https://github.com/craciuncg/GRAF)  
**Area**: Information Retrieval  
**Keywords**: Legal Question Answering, Knowledge Graph, Graph Attention Network, Romanian, Low-resource NLP  

## TL;DR

This paper proposes the GRAF algorithm, which integrates a legal knowledge graph (Law-RoG) and Graph Attention Networks for Romanian legal multiple-choice question answering, while open-sourcing JuRO (10,836 questions), the first Romanian legal MCQA dataset, and CROL, a legal corpus.

## Background & Motivation

**Background**: Legal Question Answering (Legal QA) is an emerging application area of NLP. While English has multiple datasets like PrivacyQA and JEC-QA, Romanian lacks publicly available legal QA datasets despite having pre-trained models such as jurBERT. Legal NLP for multilingual and low-resource languages is still in its infancy.

**Limitations of Prior Work**: NLP resources for Romanian are extremely scarce: there are no public legal MCQA datasets, no structured legal corpora, and no legal knowledge graphs. Although prior works by Masala et al. (2021, 2024) proposed jurBERT and judicial prediction systems, their datasets remain private.

**Key Challenge**: Standard Retrieval-Augmented Generation (RAG) methods show unstable performance in the legal domain—performing well on certain legal branches but generalizing poorly on others; fine-tuning LLMs is prone to hallucination. Traditional encoder-based methods lack the capability to handle complex questions that require cross-document legal reasoning.

**Goal**: To provide comprehensive data resources and efficient methodologies for Romanian legal question answering.

**Key Insight**: To construct a legal knowledge graph and perform graph-level alignment and matching between each answer option and the facts in the knowledge graph, rather than relying on traditional text retrieval methods.

**Core Idea**: Extracting a claim graph from question-option pairs using an LLM, encoding it along with the legal knowledge subgraph using Graph Attention Networks (GAT), and then calculating cosine alignment scores to achieve fact-verification-based MCQA.

## Method

### Overall Architecture

GRAF executes the following pipeline for each option of every question:
1. **Cross Claim Extraction**: Decomposes the (question, option) pair into a claim graph (entity-relation triplets) using an LLM.
2. **KG Sampling**: Retrieves relevant subgraphs from Law-RoG through BM25 retrieval and BFS sampling.
3. **Graph Encoding**: Encodes the claim graph and knowledge subgraph using an improved GAT.
4. **Alignment & Scoring**: Computes the alignment between claims and knowledge via cosine similarity, and outputs option scores after self-attention.

### Key Designs

**Claim Graph Construction**: Decomposes the question and options into (entity, relation, entity) triplets using an LLM (via few-shot prompting). For example, the legal question "Is criminal liability excluded in the case of self-defense?" can be decomposed into a relationship between the entities "self-defense" and "exclusion of criminal liability".

**Knowledge Graph Sampling**: Uses branch filtering + BM25 retrieval to find the top-10 entities on Law-RoG (160K nodes, 320K edges), followed by BFS expansion of depth 1, selecting at most 50 distinct entities. SpaCy is utilized for Romanian tokenization and lemmatization.

**Improved GAT Encoding**: The original GAT only processes relation-free graphs. This work extends GAT to capture both inter-node and edge topological information:

Node Attention: $e_N^{ij} = \sigma_A(a_N^T [W_N h_N^i \| W_N h_N^j])$

Edge Attention: $e_E^{ij} = \sigma_A(a_E^T [W_N h_N^i \| W_E h_E^j])$

Combining both for final node representation: $h' = h'_N + h'_E$

**Alignment Computation and Final Scoring**:

$$R^{ij} = \cos(h_c^i, h^j)$$

Aggregated knowledge representation: $\bar{H} = Rh$

Integrates the option encoding $\bar{c}$ and knowledge representation $\bar{H}$ through a self-attention mechanism, and outputs option probabilities via a fully connected layer with a sigmoid activation.

### Loss & Training

- Uses cross-entropy loss for single-choice questions and binary cross-entropy for multiple-choice questions.
- The language encoder uses jurBERT (pretrained on Romanian legal text).
- The Law-RoG knowledge graph is constructed through LLM-based few-shot entity-relation extraction, with its quality validated by 5 NLP experts.

## Key Experimental Results

### Main Results - Promotion Exam (Single-Choice, Accuracy%)

| Method | Civil Law | Criminal Law | Civil Procedure | Criminal Procedure | Administrative | Labor | Family | International | Commercial | Average |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| QBERT | 35.5 | 38.3 | 35.8 | 40.1 | 36.6 | 40.4 | 38.2 | 39.3 | 39.4 | 38.2 |
| ColBERT | 44.1 | 38.3 | 47.8 | 37.5 | 41.6 | 42.4 | 41.2 | 48.8 | 40.4 | 42.5 |
| LLM | 48.9 | 40.2 | 41.0 | 42.3 | 46.5 | 48.5 | 49.0 | 52.4 | 39.4 | 45.4 |
| LLM RAG | 53.2 | 43.8 | 38.8 | 42.8 | 57.4 | 56.6 | 63.7 | 52.4 | 57.6 | 51.8 |
| LLM LFT | 45.7 | 51.8 | 74.6 | 49.0 | 52.5 | 49.5 | 54.9 | 70.2 | 53.5 | 55.8 |
| **GRAF (Ours)** | **49.5** | **52.7** | **78.5** | **51.1** | **59.2** | **57.3** | **68.7** | **67.1** | **56.8** | **60.1** |

### Admission Exam (Multiple-Choice, Accuracy%)

| Method | Civil Law | Criminal Law | Civil Procedure | Criminal Procedure | Average |
|------|:---:|:---:|:---:|:---:|:---:|
| ColBERT | 45.1 | 43.1 | 51.0 | 41.2 | 45.1 |
| LLM RAG | 43.1 | 27.5 | 21.6 | 27.5 | 29.9 |
| **GRAF (Ours)** | **60.8** | **62.8** | **54.9** | **56.9** | **58.8** |

### Key Findings

1. **GRAF achieves the best performance in 6 out of 9 legal branches**: achieving an average accuracy of 60.1%, which outperforms the strongest baseline (LLM LFT) by 4.3 percentage points.
2. **Graph methods vs. RAG**: GRAF outperforms LLM RAG by 8.3 percentage points (60.1% vs. 51.8%), indicating that structured knowledge representations are superior to raw text retrieval.
3. **Larger advantage in multiple-choice questions**: In the admission exam, GRAF (58.8%) far outperforms ColBERT (45.1%) and LLM RAG (29.9%).
4. **Importance of domain-specific legal models**: Pre-trained domain-specific legal models significantly outperform general-purpose models (as shown in the Figure).
5. **Low agreement among LLMs**: APPA is between 40-50% and Fleiss' κ is slightly negative, indicating that legal QA remains highly challenging for LLMs.
6. **RAG is unstable in specific branches**: LLM RAG performs poorly in Civil Procedure (38.8%) but well in Administrative Law (57.4%).

## Highlights & Insights

- **Fourfold Resource Contribution**: The JuRO dataset (10,836 questions) + CROL legal corpus (330K articles, 31.5M words) + Law-RoG knowledge graph (160K nodes, 320K edges) + the GRAF algorithm.
- **Synergy of Symbolic and Numerical Methods**: Combines symbolic reasoning from the knowledge graph with numerical matching in vector space, making the two paradigms complementary.
- **Ingenious Cross Claim Extraction**: Decomposes options into verifiable claims, turning the task into a fact-checking problem.
- **Adaptability to Low-Resource Scenarios**: The entire methodology is uniquely designed for resource-scarce languages and can be transferred to legal NLP in other low-resource languages.

## Limitations & Future Work

- The best average accuracy is only 60%, which is still far from the reliability requirements for practical deployment.
- Law-RoG is generated via LLMs; although manually verified, omissions may still exist.
- Training data for individual legal branches might be insufficient.
- The potential of data augmentation (e.g., expanding training sets using LLMs) remains unexplored.
- Sensitivity analysis for BFS depth and top-k parameters in KG sampling is insufficient.
- It is unclear how the method handles questions crossing multiple legal branches that involve multiple knowledge subgraphs.

## Related Work & Insights

- The GraphRAG concept proposed by Edge et al. (2024) is the direct inspiration for the Claim Graph construction in this paper.
- The JEC-QA Chinese legal question answering dataset (26,367 questions) by Zhong et al. (2020) provides a scale reference.
- The COLIEE shared tasks (Japanese legal QA, since 2014) represent the earliest systematic evaluations in the legal NLP domain.
- The medical graph + dialogue method of He et al. (2022) demonstrates the potential of KG-based QA in vertical domains.
- Insights for low-resource NLP: Construct domain knowledge graphs first, and then leverage graph methods to compensate for data scarcity.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying graph alignment to MCQA in the GRAF algorithm is novel, along with contributing multiple original data resources.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 9 legal branches, 3 exam types, and 6 baseline methods, with detailed analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear descriptions of resources and methodologies, with well-structured algorithm pseudo-code.
- **Value**: ⭐⭐⭐⭐⭐ — The long-term value of the resource contributions (JuRO + CROL + Law-RoG) is extremely high, laying the groundwork for Romanian and low-resource legal NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ComRAG: Retrieval-Augmented Generation with Dynamic Vector Stores for Real-time Community Question Answering in Industry](comrag_retrieval-augmented_generation_with_dynamic_vector_stores_for_real-time_c.md)
- [\[ACL 2025\] NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](neusym_rag_pdf_qa.md)
- [\[ACL 2025\] VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering](voxrag_a_step_toward_transcription-free_rag_systems_in_spoken_question_answering.md)
- [\[AAAI 2026\] N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs](../../AAAI2026/information_retrieval/n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l.md)
- [\[ACL 2025\] Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs](gor_rag_long_context_summary.md)

</div>

<!-- RELATED:END -->
