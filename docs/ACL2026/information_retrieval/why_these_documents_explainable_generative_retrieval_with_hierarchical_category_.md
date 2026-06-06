---
title: >-
  [Paper Note] Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths
description: >-
  [ACL 2026][Information Retrieval & RAG][Generative Retrieval] This paper proposes HyPE, a generative retrieval framework that first generates hierarchical category paths (e.g.…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Generative Retrieval"
  - "Explainable Retrieval"
  - "Hierarchical Category Paths"
  - "Document Identifiers"
  - "Path-Aware Ranking"
date: 2026-05-08
content_hash: 041f9f41c911068b
---

# Why These Documents? Explainable Generative Retrieval with Hierarchical Category Paths

**Conference**: ACL 2026
**arXiv**: [2411.05572](https://arxiv.org/abs/2411.05572)  
**Code**: [GitHub](https://augustinlib.github.io/HyPE/)  
**Area**: Information Retrieval
**Keywords**: Generative Retrieval, Explainable Retrieval, Hierarchical Category Paths, Document Identifiers, Path-Aware Ranking

## TL;DR
This paper proposes HyPE, a generative retrieval framework that first generates hierarchical category paths (e.g., "Government >> Government by cities") before decoding document identifiers, providing query-relevant explanations for retrieval results while simultaneously improving retrieval accuracy.

## Background & Motivation

**Background** Generative Retrieval (GR) responds to queries by directly decoding document identifiers (docids) via a single generative model, enabling end-to-end optimization and reducing reliance on external indexes. Existing methods have explored two major classes of docid design: semantic (numeric clustering-based indexes) and lexical (titles, keywords, substrings).

**Limitations of Prior Work** Neither semantic nor lexical docids can answer the question "why was this document retrieved?" For instance, given the document "Dubai," different queries may focus on "Dubai economy" or "Dubai government," yet the retrieval system returns the same docid regardless, offering no explanation of how the retrieval decision corresponds to query intent.

**Key Challenge** Explainability is critical in retrieval—the absence of explanations undermines user trust and hinders exploratory search. However, existing explainable retrieval methods are either limited to keyword attribution (lacking sufficient semantic context) or rely on LLMs to generate natural language explanations (incurring high inference latency, unsuitable for real-time retrieval).

**Goal** To design an explainable generative retrieval framework that provides clear and meaningful explanations during the retrieval process without sacrificing—and ideally improving—retrieval performance.

**Key Insight** The paper leverages structured hierarchical category paths (e.g., the Wikipedia category tree) as the explanation medium, generating coarse-to-fine semantic category paths prior to decoding docids. This both explains retrieval decisions and guides the model toward more relevant documents through a coarse-to-fine reasoning process.

**Core Idea** Hierarchical category paths represent a "just right" form of explanation—more semantically structured than keywords, yet more compact and efficient than natural language (averaging only 13.5 tokens vs. 61 tokens for natural language), and capable of producing different explanatory paths for the same document under different queries.

## Method

### Overall Architecture
HyPE consists of three stages: (1) **Candidate Path Set Construction**—using an external semantic hierarchy (Wikipedia category tree) and an LLM to assign appropriate category paths to each document; (2) **Path-Augmented Training**—associating queries with paths to construct a path-augmented training set for optimizing the generative retrieval model; (3) **Path-Aware Inference**—generating multiple category paths, then decoding docids conditioned on each path, and aggregating final rankings via a path-aware ranking strategy.

### Key Designs

1. **Candidate Path Set Construction**:
    - **Function**: Assigns 1–3 semantically appropriate hierarchical category paths to each document in the corpus.
    - **Mechanism**: Uses the Wikipedia category tree as the backbone hierarchy (depth limited to 4 levels). A bi-encoder first retrieves a candidate path set $\hat{\mathcal{P}}_D$ by semantic similarity from all available paths, then an LLM selects at most 3 paths that best represent the document's content.
    - **Design Motivation**: Feeding all paths directly to the LLM exceeds context length limits; the two-stage approach (encoder filtering + LLM selection) ensures path quality while controlling cost.

2. **Path-Augmented Training**:
    - **Function**: Trains the model to generate a category path before decoding a docid.
    - **Mechanism**: For each query–document pair in the training set, the query is associated with the most semantically similar path from the document's candidate path set, yielding a path-augmented training set $\mathcal{X}^+ = \{(q, p^q, D, d)\}$. The model is jointly trained on two tasks: an indexing task $\mathcal{M}^\theta(p^q, d | D)$ and a retrieval task $\mathcal{M}^\theta(p^q, d | q)$.
    - **Design Motivation**: Prepending the path before the docid induces a coarse-to-fine pseudo-reasoning process during decoding—the model first identifies the document's semantic category before localizing the specific document—more closely mirroring human information-seeking behavior than jumping directly to a docid.

3. **Path-Aware Ranking Strategy**:
    - **Function**: Aggregates retrieval results from multiple paths at inference time to produce a final ranking.
    - **Mechanism**: Beam search is used to generate $K_p$ category paths; for each path, constrained beam search decodes $m$ docid–score pairs; the highest score is retained for each docid and results are sorted in descending order. Formally: $\tilde{Y} = \{(d, s) | s = \max\{s' | (d, s') \in Y_j\}\}$.
    - **Design Motivation**: A single path captures only one semantic aspect of the query; a multi-path strategy covers multiple topical dimensions, giving the most relevant documents more opportunities to rank highly.

### Loss & Training
Built on a T5-base backbone, HyPE applies standard seq2seq cross-entropy loss for multi-task learning (indexing + retrieval). The indexing task uses FirstP (the first $k$ tokens of the document) as the document representation, supplemented with 5 synthetic queries. At inference time, constrained beam search with a prefix trie ensures only valid docids are generated.

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

| Analysis Dimension | Result | Notes |
|---------|------|------|
| Number of paths K=1 vs K=3 | K=1 already outperforms no-path baseline; K=3 yields significant further gains | Multi-path strategy effectively captures multiple topical dimensions |
| Path quality (GPT-based evaluation) | 94.6% of paths judged as relevant | High path generation quality; low risk of error propagation |
| Human re-ranking experiment | R@1 improves by 23.7% with paths; user confidence improves by 12% | Explanatory paths genuinely help users make better decisions |
| Inference time overhead | Only ~0.1s/sample additional latency | Explainability is gained at almost no efficiency cost |
| Token efficiency | 13.5 tokens (path) vs. 61 tokens (natural language) | 4.5× more efficient |

### Key Findings
- HyPE can be applied orthogonally to all docid types (title, keyword, summary, atomic), demonstrating strong generality.
- Title docids benefit most from HyPE, as titles encode coarse-grained semantics that align well with the coarse-to-fine structure of hierarchical paths.
- HyPE remains effective on non-Wikipedia corpora (MS MARCO), confirming that generalization does not depend on the corpus sharing the same source as the category tree.
- Even when paths are not fully accurate (5.4% judged irrelevant), retrieval performance does not degrade significantly, indicating robustness to path errors.

## Highlights & Insights
- The use of hierarchical category paths as an explanation medium is an elegant design choice: structured, compact, automatically constructible, and dynamically adaptable to different queries.
- The "explain-then-retrieve" paradigm transforms explainability from post-hoc attribution into an integral component of the retrieval process itself.
- The path-aware ranking strategy cleverly leverages multiple paths to cover diverse semantic facets of a query.
- The human re-ranking experiment (R@1 +23.7%) provides compelling evidence of the practical value of explainability for real users.

## Limitations & Future Work
- The backbone hierarchy is currently based on the Wikipedia category tree, which may need to be replaced by domain-specific taxonomies for specialized fields (e.g., medicine, law).
- The approach is not applicable to semantic docids (which already possess an inherent hierarchical structure), limiting its scope of application.
- Path depth is fixed at 4 levels, which may be insufficient for highly fine-grained retrieval needs.
- Future work could explore having the model automatically construct the hierarchical structure rather than relying on an external category tree.

## Related Work & Insights
- Compared to free-text explanation methods, category paths offer a 4.5× advantage in token efficiency with negligible additional latency.
- HyPE is orthogonal to existing generative retrieval methods such as DSI and NCI, and can serve as a plug-and-play enhancement module.
- The paper introduces a new paradigm for explainability in retrieval systems: rather than post-hoc explanation, explainability is achieved through interpretable intermediate steps that actively drive the retrieval process.

## Rating
- Novelty: ⭐⭐⭐⭐ Hierarchical paths as an interpretable retrieval intermediary is a novel idea.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two datasets, four docid types, human evaluation, LLM-based evaluation, and efficiency analysis are all included.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical flow with intuitive case analyses.
- Value: ⭐⭐⭐⭐ Offers meaningful insights for both explainable retrieval and generative retrieval research.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)
- [\[ACL 2026\] Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy](hybrid-vector_retrieval_for_visually_rich_documents_combining_single-vector_effi.md)
- [\[ICLR 2026\] Hierarchical Concept-based Interpretable Models](../../ICLR2026/information_retrieval/hierarchical_concept-based_interpretable_models.md)
- [\[ICML 2026\] Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation](../../ICML2026/information_retrieval/hierarchical_abstract_tree_for_cross-document_retrieval-augmented_generation.md)
- [\[NeurIPS 2025\] Hierarchical Retrieval: The Geometry and a Pretrain-Finetune Recipe](../../NeurIPS2025/information_retrieval/hierarchical_retrieval_the_geometry_and_a_pretrain-finetune_recipe.md)

</div>

<!-- RELATED:END -->
