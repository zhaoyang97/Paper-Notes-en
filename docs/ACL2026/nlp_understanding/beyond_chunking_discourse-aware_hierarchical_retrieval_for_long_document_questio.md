---
title: >-
  [Paper Note] Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering
description: >-
  [ACL 2026][NLP Understanding][Long Document Question Answering] The authors utilize Rhetorical Structure Theory (RST) to parse the discourse structure of long documents…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Long Document Question Answering"
  - "RST"
  - "Rhetorical Structure Theory"
  - "Hierarchical Retrieval"
  - "Cross-lingual"
date: 2026-05-08
content_hash: 2e82afdb2eab4a68
---

# Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering

**Conference**: ACL 2026  
**arXiv**: [2506.06313](https://arxiv.org/abs/2506.06313)  
**Code**: <https://github.com/DreamH1gh/DISRetrieval>  
**Area**: NLP Understanding / Long Document QA / Retrieval-Augmented Generation  
**Keywords**: Long Document Question Answering, RST, Rhetorical Structure Theory, Hierarchical Retrieval, Cross-lingual

## TL;DR
The authors utilize Rhetorical Structure Theory (RST) to parse the discourse structure of long documents, constructing a sentence-level hierarchical tree with LLM-based summary augmentation for intermediate nodes. By performing structure-aware, multi-granularity retrieval on this tree, DISRetrieval consistently outperforms fixed-size chunking and RAPTOR semantic clustering across four benchmarks: QASPER, QuALITY, NarrativeQA, and MultiFieldQA-zh.

## Background & Motivation

**Background**: The mainstream approach for long document QA is "chunking + retrieval + generation," using either flat-chunking (e.g., RAG cutting documents into 100-word segments) or recursive semantic clustering (e.g., RAPTOR) to build document trees.

**Limitations of Prior Work**: Fixed-size chunking completely ignores the discourse organization of a document—a single sentence might be split across two chunks, or two contrasting paragraphs might be separated. RAPTOR relies on semantic similarity for clustering, mixing sentences that are topically similar but rhetorically distinct, thereby losing the "Topic—Contrast—Evidence—Conclusion" hierarchy of the original document.

**Key Challenge**: "Similarity-based Organization vs. Discourse-based Organization"—linguistic theory has long established that human reading depends on rhetorical relations (contrast, elaboration, summary, etc.) rather than surface similarity. Existing chunking methods discard these structural signals.

**Goal**: Systematically inject RST discourse structure into retrieval so that long document QA no longer relies on heuristic chunking while maintaining cross-lingual support (English and Chinese).

**Key Insight**: RST represents a document as a tree with Elementary Discourse Units (EDUs) as leaves and rhetorical relations as internal nodes. Integrating the RST tree directly into a retriever provides "natural chunks" of varying granularities: leaves equal sentences (fine), intermediate nodes equal rhetorical paragraphs (coarse), and the root equals the document summary.

**Core Idea**: Implement sentence-level, cross-lingual RST parsing combined with LLM-generated summaries for intermediate nodes. This allows a single discourse tree to support both precise local retrieval (leaves) and coherent global retrieval (intermediate nodes).

## Method

### Overall Architecture
DISRetrieval consists of three stages: (1) **Discourse-Aware Tree Construction**: Performs sentence-level RST parsing for each paragraph to create intra-paragraph trees $T_i$; uses an LLM to bottom-up summarize each $T_i$ into paragraph semantic units $u_i$; applies the same RST parser to combine $\{u_i\}$ into a document-level tree $T_{doc}^*$; and finally replaces leaves of $T_{doc}^*$ with original intra-paragraph trees to obtain a unified discourse tree $T_D$. (2) **Node Encoding**: Uses `gte-multilingual-base` or OpenAI `text-embedding-3-large` to encode every node (sentence or LLM summary) in $T_D$. (3) **Structure-Aware Retrieval**: Calculates cosine similarity between the query and all nodes, scores them, and performs controlled subtree expansion for intermediate nodes while directly selecting high-scoring leaves.

### Key Designs

1.  **Granularity & Language-Adaptive RST Parser**:
    -   **Function**: Uses a single transition-based sentence-level RST parser to handle both English and Chinese long documents, avoiding the high overhead and semantic fragmentation of traditional EDU-level parsing.
    -   **Mechanism**: (a) **Granularity Adaptation**—converts existing EDU datasets by merging intra-sentence EDUs and inferring inter-sentence relations via the Lowest Common Ancestor (LCA). (b) **Language Adaptation**—uses GPT-4o to translate RST-DT training sentences into the target language (Chinese) and merges them with original English corpora to train a unified parser $f_{discourse}$. The transition system uses a stack $\sigma$ and a sentence queue $\beta$ with `shift`, `reduce`, and `pop_root` actions. The scoring model uses node representations $h_v$ and selects actions via softmax.
    -   **Design Motivation**: Traditional RST at the EDU level is computationally expensive and overly fragmented for long documents. Moving to the sentence level preserves discourse information while significantly improving speed. LLM-based translation enables cross-lingual transfer without native Chinese RST annotations.

2.  **Bottom-up LLM Node Augmentation**:
    -   **Function**: Transforms internal nodes, which initially only have "rhetorical relation labels," into retrievable representations with text content, bridging the gap between abstract relations and concrete semantics.
    -   **Mechanism**: For each internal node $v$ with children $v_l, v_r$, applies a threshold rule: $v^* = f_{LLM}(v_l, v_r)$ if $|v_l| + |v_r| \geq \tau$, otherwise $v^* = f_{merge}(v)$ (direct concatenation). This hierarchically converts internal nodes into summaries or concatenations. Finally, the same RST parser constructs the document-level tree $T_{doc}^*$ from paragraph roots $u_i$.
    -   **Design Motivation**: Pure structural nodes (e.g., Contrast, Elaboration) lack content for semantic matching. LLM summarization allows nodes to carry both structural (e.g., "this is a contrast") and semantic (e.g., "comparing X and Y") signals. The threshold $\tau$ balances fidelity and compression based on document type.

3.  **Structure-Guided Dual-Selection Retrieval**:
    -   **Function**: Retrieves evidence from the unified discourse tree to provide the generator with multi-granularity evidence (both fine-grained details and coherent paragraphs).
    -   **Mechanism**: First, calculate $\text{score}(v) = \cos(f_{enc}(q), \mathbf{e}_v)$. Then apply two strategies: (a) if $v$ is a leaf and unused, add to evidence set $E$; (b) if $v$ is an internal node, perform "controlled subtree expansion" by selecting the top $k$ unused leaves within that subtree. This continues until $|E| \geq K$.
    -   **Design Motivation**: Flat retrieval often yields redundant or fragmented chunks. This dual-selection strategy ensures that both highly relevant specific sentences and relevant discourse segments are selected while avoiding redundancy within subtrees.

### Loss & Training
The sentence-level RST parser training objective is defined as:
$$\mathcal{L}(\theta) = -\log p(a^* \mid c) + \frac{\lambda \|\theta\|_2}{2}$$
This represents the cross-entropy loss for each transition action with L2 regularization, using gold trees from RST-DT as supervision. The generation stage is entirely zero-shot.

## Key Experimental Results

### Main Results (Generation F1 / Accuracy)

| Dataset | Context | flatten-chunk | RAPTOR | Bisection | **Ours (DISRetrieval)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| QASPER (UnifiedQA-3B, OpenAI) | 400 | 39.03 | 39.53 | 39.70 | **40.74** |
| QASPER (GPT-4.1-mini, OpenAI) | 400 | 44.78 | 43.85 | 45.69 | **46.31** |
| QuALITY (Deepseek-v3, OpenAI) | 400 | 76.56 | 75.22 | 76.94 | **77.71** |
| NarrativeQA (BLEU) | — | 24.24 | 25.05 | 24.71 | **25.39** |
| MultiFieldQA-zh (Deepseek-v3, 400) | 400 | 26.70 | 27.01 | 28.24 | **29.54** |

Ours consistently outperforms all baselines across different context lengths, embedding models, and generation models.

### Ablation Study (QASPER Retrieval Performance token-level F1)

| Configuration | 200 (OpenAI) | 300 (OpenAI) | 400 (OpenAI) |
| :--- | :--- | :--- | :--- |
| flatten-chunk | 29.17 | 25.12 | 21.91 |
| RAPTOR | 27.18 | 23.57 | 20.64 |
| Bisection (Structure removed) | 29.29 | 25.16 | 21.98 |
| **Full DISRetrieval** | **30.27** | **26.00** | **22.79** |

Bisection, which shares the same LLM augmentation but uses a binary tree instead of a discourse tree, is consistently lower than Full DISRetrieval, proving the inherent value of discourse structure.

### Key Findings
-   **Discourse Structure > Semantic Clustering**: DISRetrieval consistently beats RAPTOR, indicating that rhetorical relationships are superior to embedding similarity for document organization.
-   **Gold Evidence (129 words) > Full Text (4170 words)**: On QASPER, gold evidence yields 50.71% F1 vs. 48.81% for full text, highlighting the importance of precise retrieval.
-   **Parser Bottleneck**: Retrieval recall and answer F1 increase monotonically with the amount of training data (0→100%), showing the RST parser's accuracy is the primary bottleneck.
-   **Efficiency**: Processing a 50K word document takes 103s vs. 338s for RAPTOR (a 3x speedup).

## Highlights & Insights
- Successfully integrates decades of linguistic discourse theory into neural retrieval, demonstrating that structured linguistic knowledge remains highly valuable in the LLM era.
- The nested construction (Intra-para RST → LLM summary → Inter-para RST) is an elegant engineering solution to generate document-level hierarchies using a single parser.
- LLM summarization for intermediate nodes bridges the gap between linguistic structure (abstract labels) and neural retrieval (semantic matching).
- The adaptive threshold $\tau$ (0 for academic, 50 for narrative) provides a clear heuristic: use larger $\tau$ for documents with many short sentences and smaller $\tau$ for long, independent sentences.

## Limitations & Future Work
- The performance is capped by the discourse parser; current training is on RST-DT (news), leaving room for improvement in academic or narrative domains.
- Cross-lingual capabilities were only demonstrated for English and Chinese; other languages would require additional LLM-translated training data.
- The threshold $\tau$ is a simple binary rule; future work could explore dynamic selection based on content complexity.
- Evaluation metrics (F1/BLEU) may not fully capture the coherence gains provided by discourse-aware retrieval.

## Related Work & Insights
- **vs. RAPTOR**: RAPTOR uses recursive embedding clustering while Ours uses RST relations. Ours is consistently superior, proving "why elements are connected" is more useful than "what is similar."
- **vs. Bisection**: A strong ablation that keeps all LLM enhancements but replaces the tree structure. The fact that DISRetrieval continues to outperform Bisection isolates the contribution of discourse structure.
- **vs. Traditional Chunking RAG**: Fixed-window chunking ignores document organization; this work offers a new paradigm of "understand structure first, retrieve second," particularly valuable for highly structured scenarios like legal or academic documents.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically integrating RST into long document retrieval is a first. 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive testing across four datasets, multiple context lengths, and various models.
- Writing Quality: ⭐⭐⭐⭐ Clear architectural diagrams and well-defined algorithms.
- Value: ⭐⭐⭐⭐ Provides a robust path for discourse-aware RAG improvements.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction](hcre_llm-based_hierarchical_classification_for_cross-document_relation_extractio.md)
- [\[ACL 2026\] It's High Time: A Survey of Temporal Question Answering](it39s_high_time_a_survey_of_temporal_question_answering.md)
- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)
- [\[ACL 2026\] ASTRA: Adaptive Semantic Tree Reasoning Architecture for Complex Table Question Answering](astra_adaptive_semantic_tree_reasoning_architecture_for_complex_table_question_a.md)
- [\[ACL 2026\] Knowledge-driven Augmentation and Retrieval for Integrative Temporal Adaptation](knowledge-driven_augmentation_and_retrieval_for_integrative_temporal_adaptation.md)

</div>

<!-- RELATED:END -->
