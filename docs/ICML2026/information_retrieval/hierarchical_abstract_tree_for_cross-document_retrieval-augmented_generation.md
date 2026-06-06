---
title: >-
  [Paper Note] Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation
description: >-
  [ICML 2026][Information Retrieval & RAG][Tree-RAG] Ψ-RAG replaces RAPTOR's k-means with a "merge–collapse" hierarchical clustering to construct a cross-document abstract tree…
tags:
  - "ICML 2026"
  - "Information Retrieval & RAG"
  - "Tree-RAG"
  - "Cross-document Multi-hop"
  - "Hierarchical Abstraction"
  - "Agentic Retrieval"
  - "Hybrid Sparse Retrieval"
date: 2026-05-08
content_hash: ee1f37093275565f
---

# Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation

**Conference**: ICML 2026  
**arXiv**: [2605.00529](https://arxiv.org/abs/2605.00529)  
**Code**: https://github.com/Newiz430/Psi-RAG (available)  
**Area**: Information Retrieval / Retrieval-Augmented Generation / Multi-hop QA  
**Keywords**: Tree-RAG, Cross-document Multi-hop, Hierarchical Abstraction, Agentic Retrieval, Hybrid Sparse Retrieval

## TL;DR
Ψ-RAG replaces RAPTOR's k-means with a "merge–collapse" hierarchical clustering to construct a cross-document abstract tree, paired with a retrieval-answering Agent capable of multi-turn rewriting and a hybrid sparse BM25 index. This enables Tree-RAG, for the first time, to match or even surpass Graph-RAG on corpus-level, cross-document multi-hop QA, achieving an average F1 25.9% higher than RAPTOR and 7.4% higher than HippoRAG 2.

## Background & Motivation

**Background**: Current RAG approaches mainly follow two structured paradigms. Graph-RAG (e.g., GraphRAG, HippoRAG 2) explicitly models inter-document relations via knowledge graphs, offering strong multi-hop capabilities but requiring extensive OpenIE during indexing, which is costly. Tree-RAG (represented by RAPTOR) clusters documents bottom-up into an abstract tree using k-means, enabling retrieval at token/passage/document granularity and excelling at summarization tasks, but is mainly designed for single-document scenarios.

**Limitations of Prior Work**: Directly applying Tree-RAG to "corpus-level, cross-document, multi-hop" settings exposes three issues: (1) k-means clustering assumes spherical distributions, leading to a "uniform effect" on skewed corpora, misassigning major-class documents to minority clusters and introducing noise; (2) tree leaves lack explicit interconnections, preventing causal jumps between documents as in Graph-RAG; (3) top-level abstractions are too coarse, making it difficult for dense vectors to align specific query entities with abstract concepts.

**Key Challenge**: The goal is to "retain the multi-granularity advantage of trees while gaining Graph-RAG's cross-document causal reasoning," but traditional clustering objectives and static dense matching do not support this.

**Goal**: Decompose into three subproblems—(a) design a hierarchical indexing method that does not rely on distributional assumptions and can handle skewed corpora; (b) enable the retriever to perform cross-document jumps without altering the tree structure; (c) supplement coarse-grained matching at abstract nodes with fine-grained evidence channels.

**Key Insight**: The authors leverage agglomerative hierarchical clustering (AHC), using Dasgupta cost to show that greedy merging naturally favors "skewed" over "uniform" distributions; they draw on IRCoT's iterative agent, letting the LLM decide "when to retrieve again"; at the fine-grained level, they simply add a BM25 keyword index for hybrid retrieval.

**Core Idea**: Replace k-means tree construction with "similarity ranking → iterative merging & collapse → abstraction," then add an R&A Agent and agentic sparse retrieval, upgrading Tree-RAG into a universal framework for cross-document multi-hop tasks.

## Method

### Overall Architecture
Ψ-RAG adopts a two-stage indexing + retrieval pipeline. In the indexing stage, all corpus chunks are encoded into dense vectors, pairwise cosine similarities are ranked, and an abstract tree is built via iterative "merge/collapse" in this order; tree leaves are original chunks, and each internal node is summarized by an abstraction agent (either a summary or a set of keywords) and re-encoded. In retrieval, the user query enters the R&A Agent, which first triggers hybrid retrieval (dense top-down + BM25); results are filled into the context, and the Agent decides between `<answer>` and `<retrieve>`. If `<retrieve>` is chosen, the Agent rewrites a new sub-query (e.g., expanding "David Gest's wife" to "the wife of American film producer David Gest"), driving the next retrieval round, until an answer is given or the budget is exhausted.

### Key Designs

1. **Hierarchical Abstract Tree via "Merge–Collapse"**:

    - **Function**: Organizes $n$ chunks bottom-up into a multi-branch abstract tree, without any distributional assumptions.
    - **Mechanism**: Compute the symmetric similarity matrix $S = e(D) e(D)^\top$, enumerate chunk pairs $(u,v)$ in descending similarity order. If neither has a parent, create a new abstract node $a$ with $c(a)=\{u,v\}$ (merging); if $u$ is already under $p(u)$ and $v$ is independent, attach $v$ to $p(u)$ (leaf node collapse); if both have different roots, align by depth—if depths match, create a new ancestor; if not, graft the shallower tree onto the deeper one at the corresponding level (abstract node collapse). This process takes exactly $n-1$ steps to connect $n$ chunks into a tree; overly wide nodes are split evenly to avoid overlong abstraction agent contexts.
    - **Design Motivation**: Using Dasgupta cost, the authors prove: (i) On a perfectly uniform tree, a "move leaf" operation reduces cost, so Ψ-RAG naturally avoids uniformity; (ii) On skewed trees, moving majority-class nodes to minority classes increases cost, so the algorithm preserves skewness. Thus, the "geometric–cost" analysis sidesteps k-means' uniform effect without introducing new loss functions.

2. **R&A Agent-Driven Multi-turn Retrieval**:

    - **Function**: Enables the retriever to reason causally—"check if current evidence suffices, otherwise rewrite the query and retrieve again."
    - **Mechanism**: Each Agent step outputs a triple $a = (R, \langle\text{action}\rangle, \cdot)$, where action is `<answer>` or `<retrieve>`; if retrieve, the Agent produces a new query $q'_i$. The $i$-th retrieval result $D^*_i = r(q'_i, \mathcal{T})$ and history $\{(I(D^*_j) \cup a_j)\}$ are fed back to the Agent, until an answer is given or the upper limit $i_{\max}$ is reached. Each retrieval follows RAPTOR's top-down beam: starting from the root, at each level select top-$k$ by $s(q,u)$, propagate their children to the next candidate set, and recurse to the leaves.
    - **Design Motivation**: For cross-document multi-hop queries (e.g., "Who is the wife of the producer of the documentary about the singer who influenced Beyoncé?"), initial dense matching is dominated by "Beyoncé" and "documentary," missing the true intermediary (David Gest); letting the Agent decide the next step after seeing initial evidence allows the language model to dynamically supplement the tree's missing "cross-document edges."

3. **Keyword Hybrid Index + Query Rewriting**:

    - **Function**: Addresses the issue where top-level dense abstractions are too coarse to pinpoint fine-grained facts.
    - **Mechanism**: During indexing, build an additional BM25 sparse index; at retrieval, the Agent can fuse results via a parameterized reranker (integrating dense+sparse top-$k$) or non-parametric RRF (reciprocal rank fusion). Notably, when the Agent chooses `<retrieve>`, it can not only rewrite the query but also add descriptive appositives, enabling BM25 to capture more topic keywords and helping dense retrieval locate upper-level abstractions via high-level context.
    - **Design Motivation**: Abstract trees excel at coarse summarization but cannot align specific entities to top-level summaries; BM25 complements lexical matching, and Agent query rewriting turns "short questions" into "long questions with modifiers," benefiting both retrieval paths.

### Loss & Training
The entire pipeline is training-free: encoder, reranker, and Agent LLMs (Llama-3.3-70B, Qwen3-Embedding-8B) use open-source weights directly; indexing relies on similarity ranking and LLM summarization, retrieval is controlled by prompt-driven Agent; only top-$k$, $i_{\max}$, and fusion method are tuned as hyperparameters.

## Key Experimental Results

### Main Results

| Task (Multi-hop F1) | RAPTOR | HippoRAG 2 | Ψ-RAG (Ours) | Gain over RAPTOR ↑ |
|---|---|---|---|---|
| HotpotQA / 2Wiki / MuSiQue / MultiHop-RAG Avg. | Baseline | Strong Baseline | +25.9% F1 vs RAPTOR; +7.4% vs HippoRAG 2 | Significant |
| Corpus-level Indexing Time vs Graph-RAG | — | Slow | ≈10× faster | — |
| Token-level QA (NQ / PopQA) retrieval | Baseline | — | +23.7% retrieval | Significant |

| Capability | Traditional RAG | Graph-RAG | Tree-RAG (RAPTOR) | Ψ-RAG |
|---|---|---|---|---|
| Single Document | ✓ | ✓ | ✓ | ✓ |
| Cross-document | Partial | ✓ | Partial | ✓ |
| Token-level QA | ✓ | ✓ | Weak | ✓ |
| Passage-level | Partial | ✓ | ✓ | ✓ |
| Document-level Summarization | Weak | Partial | ✓ | ✓ |

### Ablation Study

| Configuration | Key Findings |
|---|---|
| Full Ψ-RAG | Achieves best or near-best on all four task types (single-hop / multi-hop / narrative / summarization) |
| w/o R&A Agent | Multi-hop F1 drops sharply due to loss of cross-document jumps |
| w/o BM25 Hybrid | Token-level fact questions suffer most, showing coarse abstraction needs fine-grained supplementation |
| w/o "Merge–Collapse" (use k-means) | On skewed corpora, abstract nodes start mixing major classes, triggering "uniform effect" noise |

### Key Findings
- On synthetic skewed corpora like "Sports[:50] + Business[:5]", RAPTOR's top-level abstract nodes misclassify majority-class (Sports) chunks as minority, introducing "confused abstraction" noise; Ψ-RAG's ring-tree visualization shows this confusion nearly disappears, matching Dasgupta cost analysis predictions.
- Ψ-RAG's core improvement comes from the synergy of "hierarchical abstraction + Agent multi-turn": changing only indexing improves summarization tasks; adding only the Agent yields limited token-level gains; both together surpass Graph-RAG on multi-hop QA.
- Indexing is about an order of magnitude faster than GraphRAG / HippoRAG 2, as no OpenIE entity extraction is needed—crucial for real-world deployment.

## Highlights & Insights
- Theoretical analysis via Dasgupta cost clarifies that "AHC is inherently resistant to uniform effect and naturally fits skewed distributions," bridging the gap between "heuristic clustering" and "task performance" with a clean geometric argument, much stronger than empirical comparisons alone.
- "Patch-style enhancements" are applied at two precise points: geometric structure at indexing, semantic jumps at retrieval, and BM25 to fix "overly coarse abstraction"—each patch addresses a distinct issue without overlap, exemplifying an engineering "divide and conquer" worth emulating.
- The Agent's query rewriting trick (adding appositive descriptions) is nearly cost-free yet benefits both dense and sparse retrieval, a "lightweight patch at the bottleneck" with high transferability—applicable to other multi-hop search, Code RAG, and long-document QA.

## Limitations & Future Work
- The system's latency bottleneck is entirely in the LLM Agent's multi-turn calls; as $i_{\max}$ increases, latency rises linearly, and no adaptive stopping strategy is provided.
- Abstract tree quality depends on the underlying LLM's summarization ability; with smaller local models, abstract nodes may lose key entities, causing dense matching to collapse—no degradation curve for low-cost LLMs is reported.
- "Merging & collapse" is a streaming greedy algorithm, sensitive to the order of highly similar chunks; in practice, multiple shuffles and aggregation may be needed, but stability is not deeply discussed.
- In comparison with Graph-RAG, HippoRAG 2's PPR reasoning is "re-skinned" as Agent multi-turn retrieval; for hops $\geq 4$, explicit graph parallel diffusion may still outperform Agent serial calls, but no sensitivity analysis for ultra-multi-hop settings is provided.

## Related Work & Insights
- **vs RAPTOR**: Both follow the Tree-RAG paradigm, but RAPTOR uses k-means GMM for bottom-up clustering, triggering the "uniform effect" on skewed corpora; Ψ-RAG employs AHC-style merge–collapse + multi-branch rebalancing, avoids distributional assumptions, and natively supports corpus-level indexing.
- **vs HippoRAG 2 / GraphRAG**: Graph-RAG builds graphs via OpenIE entity extraction and uses PPR for multi-hop reasoning, making offline indexing very expensive; Ψ-RAG defers "cross-document relations" to retrieval-time dynamic discovery by the Agent, achieving ~10× faster indexing and higher multi-hop accuracy.
- **vs IRCoT**: IRCoT couples "multi-step reasoning + multi-step retrieval" in a single chain, but the underlying retriever is single-layer dense; Ψ-RAG applies the same multi-turn idea atop an abstract tree + sparse hybrid retrieval, making it more general for multi-granularity tasks.
- **Insights**: When a static index structure (tree/graph/inverted) alone cannot handle multi-hop, instead of modifying the index, "letting the LLM act as a temporary graph at retrieval" is a lower-cost shortcut—this idea can transfer to Code RAG (jumping by call graph), long-paper QA (jumping by citation), etc.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of "merge–collapse + Dasgupta cost theory + Agent hybrid retrieval" is a first in Tree-RAG, though each component has precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Compared across four task types, six datasets, and multiple baselines, with visualization and theory; lacks ultra-multi-hop and small-model degradation curves.
- Writing Quality: ⭐⭐⭐⭐⭐ Framework diagrams, merge step illustrations, Dasgupta proofs, and visual comparisons are all well-presented, making it highly readable.
- Value: ⭐⭐⭐⭐ Elevates Tree-RAG to Graph-RAG-level multi-hop capability, training-free and ~10× faster indexing, practical for deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering](../../NeurIPS2025/information_retrieval/benchmarking_retrievalaugmented_multimodal_generation_for_do.md)
- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](../../ACL2026/information_retrieval/feedback_adaptation_for_retrieval-augmented_generation.md)
- [\[ICLR 2026\] Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](../../ICLR2026/information_retrieval/toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)
- [\[AAAI 2026\] Neighbor-aware Instance Refining with Noisy Labels for Cross-Modal Retrieval](../../AAAI2026/information_retrieval/neighbor-aware_instance_refining_with_noisy_labels_for_cross-modal_retrieval.md)
- [\[ICLR 2026\] Hierarchical Concept-based Interpretable Models](../../ICLR2026/information_retrieval/hierarchical_concept-based_interpretable_models.md)

</div>

<!-- RELATED:END -->
