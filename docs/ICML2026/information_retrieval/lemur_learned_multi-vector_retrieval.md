---
title: >-
  [Paper Note] LEMUR: Learned Multi-Vector Retrieval
description: >-
  [ICML 2026][Information Retrieval & RAG][Multi-vector retrieval] Lemur transforms multi-vector similarity search into a supervised learning problem. It utilizes a two-layer MLP to map token-level embeddings to a low-dime…
tags:
  - "ICML 2026"
  - "Information Retrieval & RAG"
  - "Multi-vector retrieval"
  - "Approximate Nearest Neighbor Search"
  - "MaxSim"
  - "Supervised dimensionality reduction"
  - "ColBERT"
date: 2026-05-08
content_hash: ee9388ede6268c0e
---

# LEMUR: Learned Multi-Vector Retrieval

**Conference**: ICML 2026  
**arXiv**: [2601.21853](https://arxiv.org/abs/2601.21853)  
**Code**: [github.com/ejaasaari/lemur](https://github.com/ejaasaari/lemur)  
**Area**: Information Retrieval  
**Keywords**: Multi-vector retrieval, Approximate Nearest Neighbor Search, MaxSim, Supervised dimensionality reduction, ColBERT  

## TL;DR
Lemur transforms multi-vector similarity search into a supervised learning problem. It utilizes a two-layer MLP to map token-level embeddings to a low-dimensional latent space, then leverages existing single-vector ANNS indexing for retrieval, achieving speeds an order of magnitude faster than methods like PLAID or MUVERA.

## Background & Motivation

**Background**: Late interaction models, exemplified by ColBERT, achieve higher retrieval precision than single-vector models by using multi-vector representations (one embedding per token). Similarity between queries and documents is measured via MaxSim, which is the sum of inner products between each query token and its most similar document token.

**Limitations of Prior Work**: Calculating MaxSim is computationally expensive, as it requires evaluating inner products between all query and document embeddings. Current acceleration methods (PLAID, DESSERT, EMVB, IGP) rely on token-level pruning as an initial step. However, individual token similarity is an imprecise proxy for document-level MaxSim, necessitating large candidate sets to maintain recall. MUVERA reduces the problem to single-vector search using Fixed Dimensionality Encoding (FDE), but requires 10240 dimensions for sufficient accuracy, resulting in high memory and latency costs.

**Key Challenge**: There is a fundamental contradiction between the high precision of multi-vector retrieval and its high latency. Existing methods either rely on imprecise token-level proxies (PLAID series) or data-independent encodings with extremely high dimensionality (MUVERA); neither efficiently bridges this gap.

**Goal**: Design a lightweight, corpus-specific dimensionality reduction framework for search that reduces multi-vector retrieval to low-dimensional single-vector search while maintaining high recall.

**Key Insight**: MaxSim can be decomposed into the sum of per-token contributions: $\text{MaxSim}(X,C) = \sum_{x \in X} \max_{c \in C} \langle x,c \rangle$. The contribution of each token to all documents, $g(x) \in \mathbb{R}^m$, can be framed as a regression problem from $\mathbb{R}^d$ to $\mathbb{R}^m$, which is learnable via an MLP.

**Core Idea**: A two-layer MLP is employed to learn the mapping from tokens to document similarity. The structure of its linear output layer then allows reducing multi-vector search to a single-vector MIPS problem in a latent space.

## Method

### Overall Architecture
Lemur consists of an offline indexing phase and an online query phase. During the offline phase, a two-layer MLP is trained to estimate the MaxSim contribution of each token embedding to all documents. The row vectors of the output layer weight matrix are then stored in an ANNS index as single-vector document representations. During online querying, all query tokens are mapped through the MLP's hidden layer to the latent space and pooled (summed) into a single-vector query representation. This is used to retrieve $k'$ candidate documents via ANNS, followed by an exact MaxSim re-ranking to obtain the final top-$k$ results.

### Key Designs

1.  **Supervised Reconstruction of MaxSim**:
    - **Function**: Transforms multi-vector similarity estimation into a standard multi-output regression problem.
    - **Mechanism**: The target function $g_l(x) = \max_{c \in C_l} \langle x,c \rangle$ maps each token embedding $x$ to its MaxSim contribution to the $l$-th document. An MLP $\phi(x) = W \psi(x)$ is trained to fit this mapping, where $\psi$ is the feature encoder (hidden layer) and $W \in \mathbb{R}^d{m \times d'}$ is the linear output layer. Since $g_l$ is a convex piecewise linear function, a two-layer network provides sufficient fitting capacity.
    - **Design Motivation**: Directly optimizing MaxSim estimation accuracy rather than using data-independent encoding allows low-dimensional (2048-d) representations to outperform MUVERA's 10240-d FDE.

2.  **Reduction from Linear Output Layer to Single-Vector MIPS**:
    - **Function**: Reduces the inference of multi-output regression to standard single-vector approximate nearest neighbor search.
    - **Mechanism**: Because the output layer is linear, the MaxSim estimate can be expressed as $f(X) \approx W \Psi(X)$, where $\Psi(X) = \sum_{x \in X} \psi(x)$ is the pooled query vector. Finding the $k'$ documents with the largest MaxSim is equivalent to finding the $k'$ weight row vectors $w_j$ in $d'$-dimensional space that have the largest inner product with $\Psi(X)$. This is a standard single-vector MIPS problem, which can be accelerated using optimized ANNS libraries like HNSW.
    - **Design Motivation**: Avoids the complexity of designing specialized multi-vector search data structures by reusing the mature single-vector search ecosystem.

3.  **Scalable Two-Stage Training**:
    - **Function**: Scales MLP training to million-scale corpora.
    - **Mechanism**: The feature encoder $\psi$ is pre-trained on $m' \ll m$ sampled documents. After freezing $\psi$, the $j$-th row of the output layer $w_j = Z^+ y_j$ is solved analytically for each document $j$ using OLS regression, where $Z$ is the feature matrix of training samples. OLS solutions are calculated independently for each document and are naturally parallelizable.
    - **Design Motivation**: Training a multi-output MLP with $m$ outputs is memory-intensive; two-stage training decouples feature learning from linear fitting, making index construction linearly scalable.

## Key Experimental Results

### Main Results (ColBERTv2, k=100, QPS@≥80% Recall)

| Dataset | Lemur | MUVERA | IGP | DESSERT | PLAID |
|---------|-------|--------|-----|---------|-------|
| MSMARCO (8.8M docs) | **799** | 150 | 62 | — | 13 |
| HotpotQA (5.2M docs) | **426** | 22 | 37 | — | 10 |
| NQ (2.7M docs) | **869** | 79 | 107 | 38 | 16 |
| Quora (523K docs) | **4068** | 787 | 679 | 284 | 89 |
| FiQA (58K docs) | **2416** | 239 | 310 | 242 | 51 |
| SCIDOCS (26K docs) | **2591** | 391 | 320 | 285 | 85 |

### Ablation Study (Effect of Hidden Dimension $d'$)

| Configuration | MaxSim Approximation Accuracy | End-to-end Latency Trend | Description |
|---------------|-------------------------------|--------------------------|-------------|
| $d'=1024$ | Exceeds MUVERA 10240-d FDE on 7/8 datasets | Fastest ANNS | Precision already surpasses 10x larger FDE |
| $d'=2048$ (Default) | Significantly better than $d'=1024$ | Best Trade-off | Latency overhead is negligible |
| $d'=4096$ | Slightly better than $d'=2048$ | Diminishing Returns | ANNS cost begins to offset precision gains |

### Key Findings
- Lemur is 5–16x faster than the best baseline at $\geq 80\%$ recall across all 8 BEIR datasets.
- 1024-d Lemur embeddings exceed the recall of 10240-d MUVERA FDE on 7/8 datasets, indicating that supervised representations are superior to data-independent encodings.
- On non-ColBERTv2 models (e.g., answerai-colbert-small, GTE-ModernColBERT), MUVERA's recall fails to exceed 60%, while Lemur remains stable.
- Pearson and Spearman rank correlation coefficients are $> 0.94$ across all datasets, indicating extremely high MaxSim estimation quality.

## Highlights & Insights
- The "double reduction" logic—transforming multi-vector search into supervised regression and then into single-vector MIPS—is elegant. The key insight is that the row vectors of the linear output layer naturally serve as document representations in the latent space without additional projection.
- Using random weights for the feature encoder (ELM mode) remains effective, suggesting the hidden layer primarily performs non-linear feature expansion rather than learning highly specialized representations.
- The index supports incremental updates (new documents only require one OLS regression and an HNSW insertion), which is critical for production systems.
- Using corpus documents as training data (without separate training queries) is effective, lowering the barrier to deployment; however, using real queries further improves performance.

## Limitations & Future Work
- The indexing process requires OLS regression for each document, which takes more time than simple encoding. However, the two-stage design keeps training costs low (approx. 4.8 hours for 8.8M documents).
- Compatibility with extremely low-precision vector compression (e.g., 2-bit quantization) has not been explored; standard scalar and product quantization should apply but require verification.
- The performance gain is smaller for visual document retrieval (ViDoRe) because the number of embeddings per image is much larger than for text (avg. 1073 vs 68), making the re-ranking stage a bottleneck.

## Related Work & Insights
- **vs MUVERA**: MUVERA uses data-independent FDE for single-vector reduction, while Lemur uses supervised learning for corpus-specific reduction. The latter outperforms the former at 1/10th the dimensionality at the cost of training.
- **vs PLAID/DESSERT/IGP**: These methods rely on token-level pruning for candidate filtering; their proxies are imprecise and require large candidate sets. Lemur estimates MaxSim directly at the document level for more accurate, smaller candidate sets.
- **vs TCT-ColBERT**: TCT trains a general single-vector retriever to distill MaxSim. Lemur trains a lightweight search dimensionality reducer rather than an end-to-end retriever, making it more flexible for any late interaction model.
- **vs BGE-M3**: BGE-M3 is a general embedding model refined via self-distillation. Lemur does not train the encoder but builds an efficient search index for existing models.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The "double reduction" approach is highly original and provides a fresh perspective on retrieval as regression + MIPS.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated across 8 BEIR datasets, 5 text models, and 2 vision models with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematical derivations are clear; the transition from problem definition to algorithm design is seamless.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable for accelerating ColBERT-based retrieval systems; code is open-source.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy](../../ACL2026/information_retrieval/hybrid-vector_retrieval_for_visually_rich_documents_combining_single-vector_effi.md)
- [\[ACL 2026\] Prune-then-Merge: Towards Efficient Multi-Vector Visual Document Retrieval](../../ACL2026/information_retrieval/sculpting_the_vector_space_towards_efficient_multi-vector_visual_document_retrie.md)
- [\[ICML 2026\] Vector Linking based on Cross-Model Local Isometry Consistency](vector_linking_via_cross-model_local_isometric_consistency.md)
- [\[CVPR 2026\] BRIDGE: Multimodal-to-Text Retrieval via Reinforcement-Learned Query Alignment](../../CVPR2026/information_retrieval/bridge_multimodal-to-text_retrieval_via_reinforcement-learned_query_alignment.md)
- [\[ICML 2026\] HGMem: Hypergraph-based Working Memory to Improve Multi-step RAG for Long-Context Complex Relational Modeling](hgmem_hypergraph-based_working_memory_to_improve_multi-step_rag_for_long-context.md)

</div>

<!-- RELATED:END -->
