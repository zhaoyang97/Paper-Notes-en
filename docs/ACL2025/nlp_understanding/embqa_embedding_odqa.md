---
title: >-
  [Paper Note] Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering
description: >-
  [ACL 2025][NLP Understanding][open-domain QA] EmbQA proposes an embedding-level ODQA framework. It optimizes query representations using lightweight linear layers and unsupervised contrastive learning to achieve passage reranking. Furthermore, it introduces exploratory embeddings based on order statistics to expand candidate answer diversity, coupled with an entropy-based selection mechanism for automatic answer selection. EmbQA outperforms prompt-level methods like SuRe with…
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "open-domain QA"
  - "embedding-level reranking"
  - "contrastive learning"
  - "exploratory embedding"
  - "entropy selection"
date: 2026-05-08
content_hash: af051a2d27819bdc
---

# Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering

**Conference**: ACL 2025  
**arXiv**: [2503.01606](https://arxiv.org/abs/2503.01606)  
**Code**: None  
**Area**: NLP Understanding  
**Keywords**: open-domain QA, embedding-level reranking, contrastive learning, exploratory embedding, entropy selection

## TL;DR
EmbQA proposes an embedding-level ODQA framework. It optimizes query representations using lightweight linear layers and unsupervised contrastive learning to achieve passage reranking. Furthermore, it introduces exploratory embeddings based on order statistics to expand candidate answer diversity, coupled with an entropy-based selection mechanism for automatic answer selection. EmbQA outperforms prompt-level methods like SuRe with significantly lower computational cost across four ODQA datasets.

## Background & Motivation
**Background**: ODQA typically adopts a retriever-reader pipeline, which first retrieves relevant passages from a large-scale corpus and then uses an LLM as a reader to generate answers. Prompt-level frameworks (e.g., SuRe, Self-Verification, CoT) improve answer quality through multi-round prompting.

**Limitations of Prior Work**:
   - The retriever returns a large number of candidate passages, but passages containing the correct answers rank low (low top-k recall).
   - Existing prompt-level reranking methods (permitting the LLM to rate each passage individually) are computationally intensive and limited to processing a small number of passages due to the context window constraint.
   - The reader side relies on multi-round prompting (summarization, self-verification, candidate selection), each requiring a complete LLM inference run, which is inefficient and unstable.

**Key Challenge**: Improving ODQA accuracy typically requires more rounds of prompt interaction, but each round of interaction introduces significant computational overhead.

**Key Insight**: Replacing prompt-level multi-turn interactions with embedding-level operations (lightweight linear layers + single token embeddings).

**Core Idea**: Operating in the embedding space to simultaneously optimize retrieval reranking and answer generation diversity, thereby bypassing the computational overhead of multi-round prompting.

## Method

### Overall Architecture
EmbQA consists of two stages: (1) **Retriever Reranking**: It first uses a standard retriever to obtain the top-N passages, prompts the LLM to generate $K=2$ candidate answers, and then optimizes the query representation through unsupervised contrastive learning guided by these candidates. Specifically, two linear layers $W_1, W_2$ are trained to generate a new query embedding $\mathbf{e}_{q_{new}} = W_1 \mathbf{e}_y + W_2 \mathbf{e}_q$, which is then used to rerank the passages. (2) **Reader Generation**: An exploratory embedding $\mathbf{e}_r$, filtered based on order statistics, is injected to expand the semantic space and generate more diverse candidate answers. Finally, the candidate with the lowest entropy is selected as the final answer.

### Key Designs

1. **Embedding-Level Reranking (Self-Refinement Driven Reranking)**:

    - **Function**: Optimizes query representations guided by LLM-generated candidate answers to perform passage reranking.
    - **Mechanism**: Freezes retriever parameters and only trains two linear layers $W_1, W_2$. It linearly combines the candidate answer embedding and the original query embedding to obtain a new query $\mathbf{e}_{q_{new}}$. This is trained using unsupervised contrastive learning, where passages containing candidate answers serve as positive samples and those without serve as negative samples (sampled at a 5:1 ratio).
    - **Design Motivation**: Compared to prompt-level reranking (letting the LLM rate passages one by one), training only two matrices introduces extremely low overhead and allows traversing the entire database instead of processing only the top-k passages.

2. **Exploratory Embedding**:

    - **Function**: Injects a randomly sampled token-level embedding into the query during inference to guide the model to explore different semantic directions.
    - **Mechanism**: Samples $\mathbf{e}_r \in \mathbb{R}^D$ from a standard normal distribution and concatenates it with the query and retrieval context before feeding them to the LLM. To guarantee diversity, order statistics are leveraged: the hidden representations $\mathbf{h}_r$ of $\mathbf{e}_r$ after the LLM layer are sorted in descending order to compute $S_{\mathbf{e}_r} = \sum_{i=1}^p \Delta_{(i)}^2$ (the sum of squared differences between adjacent elements). Embeddings with $S$ lower than a threshold $T$ are retained.
    - **Design Motivation**: Based on the theory of Jain et al., minimizing the inner product between vectors is equivalent to maximizing orthogonality, and $S$ acts as an efficient proxy under a Gaussian approximation. A single token-sized embedding is sufficient to activate different knowledge paths in the LLM.

3. **Entropy-Based Selection**:

    - **Function**: Automatically selects the final answer using the entropy of the output logits, avoiding extra prompt rounds.
    - **Mechanism**: Computes $\hat{a} = \arg\min_{\hat{y}} \text{Entropy}(\hat{y})$ to select the candidate with the lowest entropy (highest certainty).
    - **Design Motivation**: Low entropy corresponds to high confidence, effectively replacing the summarization and voting strategies in SuRe that require multi-round LLM inference.

## Key Experimental Results

### Main Results
Using LLaMA 3.1 8B + BM25 as the baseline:

| Method | HotpotQA EM | 2Wiki EM | NQ EM | WebQ EM | Avg EM | Avg F1 |
|------|------------|----------|-------|---------|--------|--------|
| Retrieval Only | 25.4 | 16.6 | 26.0 | 22.2 | 22.6 | 30.6 |
| Chain-of-Thought | 27.0 | 15.4 | 27.2 | 28.8 | 24.6 | 33.2 |
| Self-Verification | 32.8 | 21.0 | 28.0 | 27.2 | 27.4 | 38.0 |
| SuRe | 38.8 | 23.8 | 36.6 | 34.4 | 33.4 | 45.3 |
| **EmbQA** | **42.0** | **27.4** | **42.2** | **38.2** | **37.5** | **49.7** |

EmbQA outperforms SuRe with a +4.1 gain in average EM and a +4.4 gain in F1.

### Cross-Retriever/Cross-Model

| Model + Retriever | SuRe Avg EM | EmbQA Avg EM | Gain |
|-------------|------------|-------------|------|
| LLaMA3.1 + BM25 | 33.4 | 37.5 | +4.1 |
| LLaMA3.1 + DPR | 28.6 | 31.9 | +3.3 |
| LLaMA3.1 + Contriever | 32.1 | 35.3 | +3.2 |
| Mistral + BM25 | 29.2 | 31.3 | +2.1 |

### Ablation Study

| Configuration | Avg EM | Description |
|------|--------|------|
| EmbQA Full | 37.5 | Full model |
| w/o Reranking | 34.2 | Without embedding-level reranking, drops by 3.3 |
| w/o Exploratory Embedding | 35.1 | Without exploratory embedding, drops by 2.4 |
| w/o Entropy Selection | 36.0 | Without entropy selection, drops by 1.5 |

### Key Findings
- **Embedding-level reranking is the most critical module**: Removing it leads to a 3.3 drop in EM, because retrieval quality directly affects all downstream modules.
- **Exploratory embedding is effective**: Introducing a single token embedding significantly expands candidate diversity.
- **Consistently effective across retrievers**: Outperforms SuRe on all three retrievers: BM25, DPR, and Contriever.
- **Significant efficiency advantage**: Embedding-level operations are several times faster than prompt-level ones (e.g., SuRe which requires multi-round LLM inferences).

## Highlights & Insights
- The **paradigm shift of Embedding-level vs. Prompt-level** is a core insight. Many prompt-level operations (reranking, verification, selection) can be replaced by lightweight embedding-level operations, dramatically reducing computational costs. This idea can be transferred to other RAG tasks.
- The idea of **using a single token embedding to activate different knowledge paths** is intriguing—it essentially performs controlled perturbations in the latent space to increase output diversity, which is more theoretically grounded than temperature sampling.
- **Filtering embeddings with order statistics** provides a theoretically supported metric of diversity, which is more reliable than random sampling.

## Limitations & Future Work
- In contrastive learning, positive and negative samples are determined based on "whether they contain candidate answers," which may introduce false negatives (passages containing correct information but not matched by the candidates).
- The threshold $T$ for exploratory embeddings needs to be manually adjusted, and different datasets might require different values.
- Evaluated only on 7-8B models; the behavior of the embedding space on larger models may differ.
- The Gaussian assumption of order statistics may not fully hold for LLM hidden states.

## Related Work & Insights
- **vs SuRe**: SuRe uses a prompt-level strategy of summarization + voting with multi-round LLM inferences; EmbQA uses embedding operations + entropy selection, which is both fast and stable.
- **vs RPG/KnowTrace**: Both being prompt-level methods, EmbQA outperforms them on all four datasets.
- **vs Prompt-level reranking**: EmbQA only trains two linear layers (taking seconds), whereas prompt-level methods require passage-by-passage LLM inference (taking minutes).

## Rating
- **Novelty**: ⭐⭐⭐⭐ Replacing prompt-level frameworks with embedding-level ones is a meaningful paradigm shift; the design of exploratory embedding is theoretically sound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 3 LLMs × 3 retrievers × 4 datasets, with detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, detailed method descriptions, and deep theoretical analysis.
- **Value**: ⭐⭐⭐⭐ A practical efficiency improvement scheme for RAG; the idea of embedding-level operations is broadly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering](iquest_an_iterative_question-guided_framework_for_knowledge_base_question_answer.md)
- [\[ACL 2025\] Towards a More Generalized Approach in Open Relation Extraction](generalized_open_relation_extract.md)
- [\[ACL 2025\] A Comprehensive Graph Framework for Question Answering with Mode-Seeking Preference Alignment](a_comprehensive_graph_framework_for_question_answering_with_mode-seeking_prefere.md)
- [\[ACL 2025\] BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering](belle_a_bi-level_multi-agent_reasoning_framework_for_multi-hop_question_answerin.md)
- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)

</div>

<!-- RELATED:END -->
