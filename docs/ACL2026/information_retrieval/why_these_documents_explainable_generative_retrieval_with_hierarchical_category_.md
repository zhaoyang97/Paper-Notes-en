---
title: >-
  [Paper Note] Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths
description: >-
  [ACL 2026][Information Retrieval & RAG][Generative Retrieval] The HyPE framework is proposed to provide query-relevant explainable paths for retrieval results in generative retrieval by first generating hierarchical cate…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Generative Retrieval"
  - "Explainable Retrieval"
  - "Hierarchical Category Paths"
  - "Docid"
  - "Path-Aware Ranking"
date: 2026-05-08
content_hash: 6ec0899c0df0d9b8
---

# Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths

**Conference**: ACL 2026  
**arXiv**: [2411.05572](https://arxiv.org/abs/2411.05572)  
**Code**: [GitHub](https://augustinlib.github.io/HyPE/)  
**Area**: Information Retrieval  
**Keywords**: Generative Retrieval, Explainable Retrieval, Hierarchical Category Paths, Docid, Path-Aware Ranking

## TL;DR
The HyPE framework is proposed to provide query-relevant explainable paths for retrieval results in generative retrieval by first generating hierarchical category paths (e.g., "Government >> Government by cities") before decoding document identifiers (docids), while simultaneously improving retrieval accuracy.

## Background & Motivation

**Background** Generative Retrieval (GR) responds to queries by directly decoding document identifiers (docids) through a single generative model, achieving end-to-end optimization and reducing reliance on external indices. Existing methods have explored two main categories of docid designs: semantic (numeric clustering indices) and lexical (titles, keywords, substrings).

**Limitations of Prior Work** Regardless of whether semantic or lexical docids are used, existing generative retrieval methods cannot answer "why a document was retrieved." For instance, for the document "Dubai," different queries might focus on "Dubai Economy" or "Dubai Government," but the retrieval system returns the same docid, failing to explain the correspondence between retrieval decisions and query intentions.

**Key Challenge** Explainability is crucial in retrieval—lack of explanation undermines user trust in retrieval results and hinders users from exploring related information. However, existing explainable retrieval methods are either limited to keyword attribution (lacking semantic context) or rely on LLM-generated natural language explanations (high inference latency, unsuitable for real-time retrieval).

**Goal** Design an explainable generative retrieval framework that provides clear and reasonable explanations during the retrieval process while maintaining or even enhancing retrieval performance.

**Key Insight** Utilize structured hierarchical category paths (such as Wikipedia category trees) as explanation carriers. By progressively generating category paths from coarse to fine before decoding the docid, the system provides an explanation for the retrieval decision and guides the model to better locate relevant documents through a coarse-to-fine reasoning process.

**Core Idea** Hierarchical category paths are a "just right" form of explanation—more semantically structured than keywords and more compact and efficient than natural language (averaging only 13.5 tokens vs. 61 tokens for natural language). They also allow for the generation of different explanatory paths for the same document based on different queries.

## Method

### Overall Architecture
HyPE consists of three stages: (1) Candidate Path Set Construction—utilizing external semantic hierarchical structures (Wikipedia category trees) and LLMs to select appropriate category paths for each document; (2) Path-Enhanced Training—associating queries with paths to build a path-enhanced training set and optimize the generative retrieval model; (3) Path-Aware Inference—generating multiple category paths first, then decoding docids under the condition of each path, and aggregating the final ranking through a path-aware ranking strategy.

### Key Designs

1. **Candidate Path Set Construction**:
    - **Function**: Assign 1-3 semantically appropriate hierarchical category paths to each document in the corpus.
    - **Mechanism**: Use the Wikipedia category tree as the backbone hierarchy (limited to 4 levels). First, a bi-encoder filters the candidate path set $\hat{\mathcal{P}}_D$ for each document from all paths based on semantic similarity. Then, an LLM selects up to 3 paths that best represent the document content from the candidates.
    - **Design Motivation**: Directly inputting all paths into an LLM exceeds context length limits. The two-stage approach (encoder filtering + LLM selection) ensures path quality while controlling costs.

2. **Path-Enhanced Training**:
    - **Function**: Train the model to generate category paths before decoding docids.
    - **Mechanism**: For each query-document pair in the training set, associate the query with the most semantically similar path from the document's candidate paths, resulting in a path-enhanced training set $\mathcal{X}^+ = \{(q, p^q, D, d)\}$. The model jointly learns two tasks: the indexing task $\mathcal{M}^\theta(p^q, d | D)$ and the retrieval task $\mathcal{M}^\theta(p^q, d | q)$.
    - **Design Motivation**: By prepending paths to docids, the model undergoes a coarse-to-fine pseudo-reasoning process during decoding, first determining the semantic category and then locating the specific document. This aligns more closely with human information retrieval logic than jumping directly to a docid.

3. **Path-Aware Ranking Strategy**:
    - **Function**: Aggregate retrieval results from multiple paths during inference to generate the final ranking.
    - **Mechanism**: Use beam search to generate $K_p$ category paths. For each path, use constrained beam search to decode $m$ docid-score pairs. Finally, retain the maximum score for each docid and rank them in descending order. Formula: $$\tilde{Y} = \{(d, s) | s = \max\{s' | (d, s') \in Y_j\}\}$$.
    - **Design Motivation**: A single path can only capture one semantic aspect of a query. A multi-path strategy covers multiple topical dimensions of the query, allowing the most relevant documents a better chance to be ranked higher.

### Loss & Training
Based on a T5-base backbone, standard seq2seq cross-entropy loss is used for multi-task learning (indexing + retrieval). The indexing task uses FirstP (the first $k$ tokens of a document) as the document representation, supplemented by 5 synthetic queries. During inference, constrained beam search with a prefix trie ensures the generation of valid docids.

## Key Experimental Results

### Main Results

| Docid Type | Dataset | Metric R@10 | Baseline | + HyPE | Gain |
|-----------|--------|----------|----------|--------|------|
| Title | NQ320K Full | R@10 | 78.7 | **83.5** | +6.1% |
| Title | NQ320K Unseen | R@10 | 68.9 | **73.6** | +6.8% |
| Summary | NQ320K Full | R@10 | 78.8 | **79.6** | +1.0% |
| Keyword | MS MARCO | R@10 | 61.2 | **62.7** | +2.5% |
| Atomic | MS MARCO | R@10 | 73.6 | **74.6** | +1.4% |

### Ablation Study

| Analysis Dimension | Result | Description |
|---------|------|------|
| Path Count K=1 vs K=3 | K=1 already outperforms none; K=3 is significantly better | Multi-path strategy effectively captures multiple topical dimensions |
| Path Quality (GPT-5 Eval) | 94.6% of paths judged relevant | High path generation quality, low risk of error propagation |
| Human Re-ranking Exp | R@1 improved by 23.7% with paths; confidence up 12% | Path explanations truly help users make better decisions |
| Inference Overhead | Only increases by ~0.1s/sample | Explainability barely impacts efficiency |
| Token Efficiency | Path 13.5 tokens vs. Natural Language 61 tokens | 4.5x more efficient |

### Key Findings
- HyPE can be orthogonally applied to all docid types (title, keyword, summary, atomic), demonstrating excellent versatility.
- Title docids benefit most from HyPE because titles encode coarse-grained semantics, which perfectly matches the coarse-to-fine structure of hierarchical paths.
- It is equally effective on non-Wikipedia corpora (MS MARCO), proving the generalization of the method does not depend on the co-origination of the corpus and the category tree.
- Even when paths are not perfectly accurate (5.4% judged irrelevant), retrieval performance does not drop significantly, indicating the method's robustness to path errors.

## Highlights & Insights
- The design of hierarchical category paths as an explanation form is ingenious: structured, compact, automatically generated, and dynamically adjustable based on the query.
- The "explain then retrieve" paradigm transforms explainability from post-hoc attribution into an organic part of the retrieval process.
- The path-aware ranking strategy cleverly utilizes multiple paths to cover various semantic aspects of a query.
- Human re-ranking experiments (R@1 improvement of 23.7%) strongly prove the value of explainability to actual users.

## Limitations & Future Work
- The current backbone hierarchy is based on the Wikipedia category tree, which might need to be replaced with domain-specific taxonomies for specialized fields (e.g., medical, legal).
- It is not applicable to semantic docids (due to their existing built-in hierarchical structure), limiting its scope somewhat.
- The path depth is fixed at 4 levels, which may be insufficient for extremely fine-grained retrieval needs.
- Future work could explore allowing the model to automatically construct hierarchical structures instead of relying on external category trees.

## Related Work & Insights
- Compared to free-text explanation methods, category paths offer a 4.5x advantage in token efficiency with almost no increase in latency.
- Orthogonal to generative retrieval methods like DSI and NCI, it can serve as a plug-and-play enhancement module.
- It provides a new paradigm for retrieval system explainability: not post-hoc explanation, but driving retrieval through explainable intermediate steps.

## Rating
- Novelty: ⭐⭐⭐⭐ Hierarchical paths as an intermediary for explainable retrieval is a novel idea.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across two datasets, four docid types, human evaluation, LLM evaluation, and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical chain with intuitive case studies.
- Value: ⭐⭐⭐⭐ Meaningful insights for both the explainable retrieval and generative retrieval fields.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)
- [\[ACL 2026\] GLIER: Generative Legal Inference and Evidence Ranking for Legal Case Retrieval](glier_generative_legal_inference_and_evidence_ranking_for_legal_case_retrieval.md)
- [\[ACL 2026\] Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy](hybrid-vector_retrieval_for_visually_rich_documents_combining_single-vector_effi.md)
- [\[ACL 2026\] IF-GEO: Conflict-Aware Instruction Fusion for Multi-Query Generative Engine Optimization](if-geo_conflict-aware_instruction_fusion_for_multi-query_generative_engine_optim.md)
- [\[ACL 2026\] Why Mean Pooling Works: Quantifying Second-Order Collapse in Text Embeddings](why_mean_pooling_works_quantifying_second-order_collapse_in_text_embeddings.md)

</div>

<!-- RELATED:END -->
