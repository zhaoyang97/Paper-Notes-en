---
title: >-
  [Paper Note] Logical Consistency is Vital: Neural-Symbolic Information Retrieval for Negative-Constraint Queries
description: >-
  [ACL 2025][Information Retrieval & RAG][Information Retrieval] This work proposes NS-IR, which translates natural language queries and documents into first-order logic (FOL) and optimizes dense retrieval embeddings using two techniques: logic alignment and connective constraints, significantly improving retrieval performance in complex logical scenarios such as negative-constraint queries.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Information Retrieval"
  - "First-Order Logic"
  - "Negative-Constraint Queries"
  - "Optimal Transport"
  - "Neuro-Symbolic"
date: 2026-05-08
content_hash: ecaa1fb282347f74
---

# Logical Consistency is Vital: Neural-Symbolic Information Retrieval for Negative-Constraint Queries

**Conference**: ACL 2025  
**arXiv**: [2505.22299](https://arxiv.org/abs/2505.22299)  
**Code**: [Available](https://github.com/xgl-git/NS-IR-main)  
**Area**: NLP / Information Retrieval  
**Keywords**: Information Retrieval, First-Order Logic, Negative-Constraint Queries, Optimal Transport, Neuro-Symbolic

## TL;DR

This work proposes NS-IR, which translates natural language queries and documents into first-order logic (FOL) and optimizes dense retrieval embeddings using two techniques: logic alignment and connective constraints, significantly improving retrieval performance in complex logical scenarios such as negative-constraint queries.

## Background & Motivation

Current dense retrieval mainly relies on word co-occurrence between queries and documents to calculate embedding similarity, ignoring the true intent of the query. While this is already insufficient for standard queries, it is catastrophic for **negative-constraint queries** (e.g., "introduce Yao Ming but do not mention basketball") because documents containing the excluded keywords mistakenly receive higher similarity scores.

For instance, when a search engine processes "RAG methods that do not involve prompt engineering," the retrieved documents often contain a large portion of the excluded keyword "prompt engineering." The core problem lies in the lack of modeling capability for **logical consistency** in existing retrieval methods.

First-order logic (FOL) can clearly represent negative semantics (e.g., `¬InvolvesPromptEngineering(x)`). Therefore, the authors propose integrating the logical semantics of FOL into natural language embeddings, thereby simultaneously considering semantic similarity and logical consistency.

## Method

### Overall Architecture

NS-IR is a reranking framework with the following workflow:

1. Use the BGE dense retriever to perform initial retrieval for a query, obtaining Top-K candidate documents.
2. Utilize an LLM (GPT-4o) to translate the queries and documents into first-order logic (NL-to-FOL translation).
3. Apply BGE again to encode the word-level embeddings of both NL and FOL.
4. Apply **Logic Alignment (LA)** and **Connective Constraint (CC)** to recalculate similarity scores.
5. Rerank the candidate documents according to the new scores.

### Key Designs

#### 1. Logic Alignment (LA)

**Function**: Integrate the overall logical semantics of FOL into the NL representation.

**Mechanism**: An optimal transport (OT) based method is utilized to measure the distribution discrepancy between NL and FOL embeddings. The detailed steps are:

- Retrieve NL query token embeddings $H$ and FOL query token embeddings $Z$ using BGE.
- Calculate the cost matrix $C$ (cosine distance).
- Solve the alignment matrix $P$ via optimal transport (using linear programming).
- Integrate $H$, $Z$, $P$, and the CLS token embeddings to update the CLS embedding: $\mathscr{h}^{cls} = H^T \cdot P \cdot Z \cdot h^{cls}$

**Design Motivation**: Use the alignment matrix as an intermediate bridge to synthesize word distributions and contextual features of both FOL and NL. After performing the same process for both queries and documents, compute $\text{score}_1$.

#### 2. Connective Constraint (CC)

**Function**: Accurately reflect the influence of different terms (especially logical connectives) in FOL on NL terms.

**Mechanism**: Allow different terms in FOL to exert distinct attention weights on terms in NL. The key innovation lies in the design of the $\sigma$ function:

- When the FOL term is a non-negative connective and has no alignment relationship with the NL term, $\sigma = 1$ (positive enhancement).
- When the FOL term is the negation connective $\neg$ and has no alignment relationship, $\sigma = -1$ (negative constraint, emphasizing exclusion semantics).
- Otherwise, $\sigma = 0$.

**Design Motivation**: The negation connective applies negative-constraint semantics through subtraction, forcing the embeddings to move away from the excluded concepts.

### Loss & Training

NS-IR is a zero-shot method and requires no training. The final document recommendation score is the sum of the scores from the two techniques:

$$\text{score} = \text{score}_1 + \text{score}_2$$

## Key Experimental Results

### Main Results

**Low-resource Retrieval and Web Search (nDCG@10)**:

| Method | SciFact | ArguAna | TREC-COVID | FiQA | DBPedia | NFCorpus | DL'19 | DL'20 |
|------|---------|---------|------------|------|---------|----------|-------|-------|
| BM25 | 67.1 | 43.2 | 55.5 | 25.1 | 26.1 | 31.4 | 55.4 | 50.1 |
| BGE | 71.3 | 48.4 | 75.3 | 30.6 | 38.9 | 35.4 | 64.4 | 63.4 |
| HyDE | 71.9 | 49.6 | 78.4 | 31.3 | 38.7 | 37.3 | 67.3 | 66.8 |
| InteR | 72.1 | 50.9 | 79.2 | 33.5 | 42.1 | 39.5 | 69.7 | 67.5 |
| **NS-IR (GPT-4o)** | **75.8** | **55.1** | **81.8** | **38.4** | **46.1** | **40.7** | **68.4** | **70.5** |

**Negative-Constraint Query Dataset NegConstraint (nDCG@10)**:

| Method | A-a | (A-a)∪B | (A-a)∪(B-b) | Total |
|------|-----|---------|-------------|-------|
| BM25 | 34.6 | 34.7 | 31.5 | 33.7 |
| BGE | 40.5 | 36.8 | 34.8 | 40.8 |
| HyDE | 55.3 | 51.5 | 50.6 | 53.1 |
| **NS-IR (GPT-4o)** | **57.9** | **54.2** | **53.7** | **56.5** |

### Ablation Study

| Method | NegConstraint (MAP) | NegConstraint (nDCG@10) |
|------|--------------------|-----------------------|
| BGE | 36.3 | 40.8 |
| BGE + LA | 40.8 | 47.6 |
| BGE + CC | 47.8 | 46.9 |
| NS-IR (LogicLLaMA) | 50.7 | 55.2 |
| NS-IR (GPT-4o) | 53.3 | 56.5 |

### Key Findings

1. NS-IR outperforms SOTA zero-shot methods on all standard retrieval benchmarks (with ~10% improvement over BGE).
2. In negative-constraint query scenarios, CC is more effective than LA, indicating that fine-grained logical operations at the connective level are more critical for handling negation constraints.
3. The effectiveness of NS-IR does not depend on a specific retriever, consistently outperforming baselines on bge-small, bge-base, and Contriever.
4. Performance remains stable at Top-K = 100, with no significant gains from further increasing K.
5. t-SNE visualization confirms that both LA and CC enable the query embeddings to move closer to positive documents and farther away from negative documents.

## Highlights & Insights

- **Novelty**: First to systematically investigate negative-constraint queries ("do not contain X"), a highly practical yet overlooked retrieval scenario.
- **Core Idea**: Rather than retraining the model, this work elegantly integrates FOL to optimize embeddings during the inference stage, making the framework plug-and-play.
- **Design Motivation**: Direct encoding of logical negation as a directional operation in the embedding space using a tri-valued mechanism ($\{-1, 0, +1\}$).
- **Value**: Fills the gap in negative-constraint query evaluation by offering a dataset with three types of queries of increasing complexity.

## Limitations & Future Work

1. Reliance on the OpenAI API for NL-to-FOL translation incurs high costs (while LogicLLaMA underperforms).
2. Reranking is applied only to the Top-K documents, which may lead to some recall loss.
3. The NL-to-FOL translation uses a unified prompt and is not optimized for different domains.
4. Focuses only on negative constraints and has not yet explored more complex set operations (union, intersection, complement, etc.).
5. Inference speed is limited by LLM calls and OT optimization; real-time performance needs improvement.

## Related Work & Insights

- While continuing the line of thought from query rewriting methods like HyDE and InteR, addressing it from the perspective of logical semantics is more fundamental.
- The application of Optimal Transport in NLP (sentence alignment, cross-modal representation) provides a solid theoretical foundation for embedding space alignment.
- NL-to-FOL translation has been revitalized recently due to LLM capability advancements, and introducing it to retrieval is a novel attempt.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing FOL to information retrieval, as well as the formulation and solution for negative-constraint queries, are highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive ablation and visualization on 6 BEIR datasets + 2 Web Search datasets + the self-constructed NegConstraint dataset.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly structured with a coherent flow from motivation to method and experiments; intuitive diagrams.
- **Value**: ⭐⭐⭐⭐ — Addresses a real-world pain point (negative constraints), the method is training-free, and the NegConstraint dataset is a valuable contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ComLQ: Benchmarking Complex Logical Queries in Information Retrieval](../../AAAI2026/information_retrieval/comlq_benchmarking_complex_logical_queries_in_information_retrieval.md)
- [\[ACL 2025\] NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering](neusym_rag_pdf_qa.md)
- [\[ACL 2025\] AIR-Bench: Automated Heterogeneous Information Retrieval Benchmark](air-bench_automated_heterogeneous_information_retrieval_benchmark.md)
- [\[ACL 2025\] Atomic LLM: A Fine-Grained Information Retrieval Evaluation Benchmark for Language Models](atomic_llm_a_fine-grained_information_retrieval_evaluation_benchmark_for_languag.md)
- [\[ACL 2025\] Any Information Is Just Worth One Single Screenshot: Unifying Search With Visualized Information Retrieval](any_information_is_just_worth_one_single_screenshot_unifying_search_with_visuali.md)

</div>

<!-- RELATED:END -->
