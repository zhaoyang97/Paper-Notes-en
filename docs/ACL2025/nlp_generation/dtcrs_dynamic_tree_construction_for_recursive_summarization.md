---
title: >-
  [Paper Note] DTCRS: Dynamic Tree Construction for Recursive Summarization
description: >-
  [ACL 2025][Text Generation][Recursive Summarization] DTCRS is proposed to dynamically construct summary trees based on document structure and query semantics. By using question decomposition and sub-question guided clustering, it reduces redundant summary nodes and significantly outperforms the static summary tree method RAPTOR on three QA datasets.
tags:
  - "ACL 2025"
  - "Text Generation"
  - "Recursive Summarization"
  - "RAG"
  - "Summary Tree"
  - "Question Decomposition"
  - "GMM Clustering"
date: 2026-05-08
content_hash: 1fe73cf6f62fd1bf
---

tags:
  - ACL 2025
  - Text Generation
  - Recursive Summarization
  - RAG
  - Summary Tree
  - Question Decomposition
  - GMM Clustering
date: 2026-05-08
content_hash: 40eccd77d50c56d5
---
# DTCRS: Dynamic Tree Construction for Recursive Summarization

**Conference**: ACL 2025  
**arXiv**: [2604.07012](https://arxiv.org/abs/2604.07012)  
**Code**: Not released  
**Area**: NLP Generation / RAG / Text Summarization  
**Keywords**: Recursive Summarization, RAG, Summary Tree, Question Decomposition, GMM Clustering  

## TL;DR

DTCRS is proposed to dynamically construct summary trees based on document structure and query semantics. By using question decomposition and sub-question guided clustering, it reduces redundant summary nodes and significantly outperforms the static summary tree method RAPTOR on three QA datasets.

---

## Background & Motivation

### Background
Retrieval-Augmented Generation (RAG) mitigates hallucination issues in Large Language Models (LLMs) by integrating external knowledge. For abstract questions requiring multi-step reasoning, knowledge from multiple parts of a document is often necessary. Recursive summarization constructs hierarchical summary trees by performing hierarchical clustering and summarization on text chunks, thereby integrating scattered information to provide evidence support for abstract questions.

### Limitations of Prior Work

**Excessive redundant summary nodes**: Traditional summary trees (e.g., RAPTOR) are static trees built on documents, which generate a large number of redundant summary nodes irrelevant to the query. This not only increases construction time but may also negatively impact QA performance.

**Lack of correlation between summaries and queries**: Static summary trees only reflect the structure of the document itself and fail to capture query semantics, leading to weak associations between the generated summaries and user questions.

**No distinction between question types**: Recursive summarization is beneficial for abstract questions, but may offer no advantage or even introduce interference for simple extractive or boolean questions. Indiscriminately utilizing recursive summarization is suboptimal.

### Design Motivation
To design a dynamic summary tree construction method that can build summary trees on-demand based on document structure and query semantics, enhancing the relevance of summaries to questions while reducing redundant nodes and construction time.

---

## Method

### Overall Architecture
The overall workflow of DTCRS consists of three steps:
1. **Question Type Classification**: Determine whether the question requires recursive summarization.
2. **Dynamic Summary Tree Construction**: For complex questions requiring summarization, construct a summary tree via question decomposition and sub-question guided clustering.
3. **Retrieval and QA**: Retrieve relevant nodes from the summary tree using the collapsed tree method.

### Key Designs

#### 1. Question Type Classification
- Utilize the LLM to generate the Table of Contents (ToC) of the document.
- Feed the ToC and the raw question into an LLM classifier to output a binary label: $y = f_{\text{LLM}}(q, c) \in \{0, 1\}$.
- A label of 1 indicates the need to construct a dynamic summary tree, while a label of 0 leads to using DPR to retrieve top-k text chunks.
- Classification criterion: Whether the question is complex and requires synthesizing information from multiple parts of the ToC.

#### 2. Question Decomposition
- Input the ToC $c$ and the raw question $q$ into the LLM question decomposer to generate a set of sub-questions $Q' = \{q_1, q_2, \dots, q_j\}$.
- Reason for introducing ToC: Constraints the scope of sub-questions within the topic of the reference document and aligns sub-questions better with various parts of the document.
- The number of sub-questions $N_{Q'}$ is much smaller than the number of text chunks $N_d$, thereby significantly reducing the number of clusters and computational overhead.

#### 3. Text Chunking
- Use a fixed chunking method with a chunk size of 500 tokens.
- Sentences crossing the 500-token boundary are moved as a whole to the next text chunk to avoid incomplete sentences.

#### 4. Clustering
- Encode text chunks and sub-questions using SBERT to obtain embedding vectors.
- Concat the text chunk embeddings $E_T$ and the sub-question embeddings $E_{Q'}$ and perform unified dimensionality reduction using UMAP to maintain semantic consistency: $E_{\text{reduced}} = \Phi_{\text{UMAP}}(E_T \oplus E_{Q'})$.
- Employ Gaussian Mixture Models (GMM) for soft clustering, allowing text chunks to belong to multiple categories.
- **First-level clustering**: Use the number of sub-questions as the number of clusters and sub-question embeddings as the initial cluster centers.
- **Subsequent levels**: No longer consider sub-questions; use BIC to determine the number of clusters and random initial centers.
- Use global clustering instead of hierarchical clustering because the number of sub-questions is small and global clustering is more efficient.

#### 5. Recursive Summary Generation
- Input the clustered text chunks into the LLM to generate corresponding summaries.
- Recursively repeat the dimensionality reduction, clustering, and summarization processes until further clustering is no longer possible.

#### 6. Retrieval Method
- Use the collapsed tree method for retrieval: flatten all nodes into a single layer and select the top-k nodes with the highest cosine similarity to the query.

### Computational Complexity Analysis
- The total computational complexity of RAPTOR is $2N_d$ (sum of geometric series).
- The total computational complexity of DTCRS is approximately $N_d + N_{Q'} + \frac{N_{Q'}}{2} + \cdots$. Since $N_{Q'} \ll N_d$, it simplifies to approximately $N_d$.
- Time efficiency is improved by approximately 50%.

---

## Key Experimental Results

### Experimental Setup
- **LLM**: GPT-4, GPT-4o-mini, DeepSeek-V2-Lite-Chat (7B)
- **Embedding Model**: SBERT
- **GMM Clustering Threshold**: 0.5
- **DPR top-k**: 5, maximum token limit for collapsed tree: 3500
- **Hardware**: NVIDIA A800 80GB GPU

### Datasets

| Dataset | Number of Questions | Average Document Characters |
|--------|--------|---------------|
| QASPER | 1451 | 21,889 |
| QuALITY | 2128 | 24,723 |
| NarrativeQA | 10,558 | 332,134 |

### Main Results

**QASPER (F1 score)**:

| Method | F1 |
|------|-----|
| DPR + GPT-4 | 51.3 |
| CoLT5 XL | 53.9 |
| RAPTOR + GPT-4 | 55.7 |
| **DTCRS + GPT-4** | **58.5** |

**QuALITY (Accuracy)**:

| Method | Test Set | Hard Subset |
|------|----------|-------------|
| DPR + GPT-4o-mini | 62.8 | 50.3 |
| RAPTOR + GPT-4o-mini | 67.6 | 57.0 |
| **DTCRS + GPT-4o-mini** | **74.7** | **62.9** |

**NarrativeQA**: DTCRS achieves the best score of 14.4% on the METEOR metric, but is overall not superior to DPR/RAPTOR. The authors attribute this to NarrativeQA being dominated by simple extractive questions.

### Ablation Study (QASPER, F1)

| Variant | GPT-4o-mini | DeepSeek |
|------|-------------|---------|
| DPR | 33.0 | 31.3 |
| w/o Global Clustering | 43.8 | 38.6 |
| w/o Classifier | 37.2 | 33.7 |
| w/o ToC | 44.1 | 38.1 |
| w/o Question Decomposition | 37.5 | 32.9 |
| **DTCRS** | **44.5** | **38.3** |

### Key Findings
1. **Classifier is Paramount**: Removing the classifier leads to the most significant performance degradation because extractive questions are better handled by DPR; indiscriminately introducing summaries compromises performance.
2. **Question Decomposition is Important**: Performance drops significantly after removing the question decomposer.
3. **Significant Efficiency Gains**: The number of summary layer nodes is reduced from 64 in RAPTOR to 4.99 in DTCRS (a 92.2% reduction), and the construction time decreases from 214.39s to 40.82s (an 80.95% reduction).
4. **Abstract Questions Benefit More**: The advantages of DTCRS are most pronounced in abstract questions.
5. **Effect of Question Type Distribution**: QASPER and QuALITY have over 20% of abstract questions, thus showing superior performance, while NarrativeQA has almost no abstract questions, leading to limited performance gains.

---

## Highlights & Insights

1. **Query-directed dynamic summary tree** is the core innovation, converting static document summarization into dynamic query-related summarization. The idea is simple and effective.
2. **Sub-question embeddings as cluster centers** is an ingenious design, which addresses both the cluster density/count and correlation issues simultaneously.
3. **Integration of question type classification** reflects pragmatic thinking—not all questions require recursive summarization.
4. **Significant efficiency improvement** (92% reduction in nodes, 81% reduction in time) has practical significance for long-document scenarios.
5. Provides a valuable empirical analysis of "which types of questions recursive summarization is suitable for".

## Limitations & Future Work

1. Limited effectiveness on NarrativeQA indicates that the method is sensitive to the distribution of question types.
2. Reliance on multiple LLM API calls (classification, ToC generation, question decomposition, summarization) might introduce additional latency and cost.
3. The accuracy of the question classifier directly impacts the ultimate outcome, yet the paper lacks an in-depth analysis of the classifier's standalone performance.
4. Evaluated only on English datasets, leaving its cross-lingual generalizability unexplored.
5. Summary generation may introduce minor hallucinations (though the paper suggests this does not affect final QA performance).

## Related Work & Insights

- **RAG**: Self-RAG (Asai et al. 2023), Self-Adaptive RAG (Jeong et al. 2024)
- **Recursive Summarization**: RAPTOR (Sarthi et al. 2024), Recursively Summarizing Books (Wu et al. 2021), HIBRIDS (Cao & Wang 2022)
- **Query Augmentation**: Query rewriting (Mo et al. 2024), Question decomposition (Reppert et al. 2023)
- **Retrieval Methods**: DPR (Karpukhin et al. 2020), ColBERT (Khattab & Zaharia 2020)

---

## Rating ⭐⭐⭐⭐

The method design is simple yet effective; the question-driven dynamic summary tree is a reasonable improvement over the static RAPTOR, and the experiments are thorough. However, the cost analysis of multiple LLM calls is not deep enough, and the applicability of the method is highly correlated with the distribution of question types.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ThreadSumm: Summarization of Nested Discourse Threads Using Tree of Thoughts](../../ACL2026/nlp_generation/threadsumm_summarization_of_nested_discourse_threads_using_tree_of_thoughts.md)
- [\[ACL 2025\] Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation](odysseus_dynamic_focus_decoding.md)
- [\[ACL 2026\] Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search](../../ACL2026/nlp_generation/adaptive_planning_for_multi-attribute_controllable_summarization_with_monte_carl.md)
- [\[ACL 2025\] Decomposed Opinion Summarization with Verified Aspect-Aware Modules](decomposed_opinion_summarization_with_verified_aspect-aware_modules.md)
- [\[ACL 2025\] Context-Aware Hierarchical Merging for Long Document Summarization](context-aware_hierarchical_merging_for_long_document_summarization.md)

</div>

<!-- RELATED:END -->
