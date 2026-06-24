---
title: >-
  [Paper Note] RetroLLM: Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation
description: >-
  [ACL 2025][LLM (Other)][Retrieval-Augmented Generation] A unified framework, RetroLLM, is proposed to integrate retrieval and generation into a single autoregressive decoding process. Through hierarchical FM-Index constraints and forward-looking constrained decoding, the LLM is enabled to directly generate fine-grained evidence from the corpus while significantly reducing token consumption.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Retrieval-Augmented Generation"
  - "Generative Retrieval"
  - "FM-Index"
  - "Constrained Decoding"
  - "Open-domain QA"
date: 2026-05-08
content_hash: 83057ab7d88f8c49
---

# RetroLLM: Empowering Large Language Models to Retrieve Fine-grained Evidence within Generation

**Conference**: ACL 2025  
**arXiv**: [2412.11919](https://arxiv.org/abs/2412.11919)  
**Code**: [github.com/sunnynexus/RetroLLM](https://github.com/sunnynexus/RetroLLM)  
**Area**: LLM/NLP  
**Keywords**: Retrieval-Augmented Generation, Generative Retrieval, FM-Index, Constrained Decoding, Open-domain QA

## TL;DR

A unified framework, RetroLLM, is proposed to integrate retrieval and generation into a single autoregressive decoding process. Through hierarchical FM-Index constraints and forward-looking constrained decoding, the LLM is enabled to directly generate fine-grained evidence from the corpus while significantly reducing token consumption.

## Background & Motivation

Although large language models exhibit powerful generation capabilities, they often suffer from hallucinations due to their reliance on parametric memory. Retrieval-Augmented Generation (RAG) mitigates this issue by introducing external knowledge, but existing RAG methods face several limitations:

**High deployment cost**: Maintaining an independent retriever is required, which increases system complexity.

**Abundant redundant information**: Retrieved text chunks often contain database noise and irrelevant information, wasting input tokens and distracting the model.

**Poor flexibility**: Fixed retrieval granularities and counts limit the flexibility of RAG systems.

**Lack of joint optimization**: The retriever relies on independent indices, preventing joint training with the generator.

Generative retrieval eliminates the reliance on document indices by generating Document Identifiers (DocIDs), but it still requires mapping DocIDs back to document content for LLM use, which disrupts the seamless integration of retrieval and generation.

This paper proposes RetroLLM, which allows the LLM to directly generate factual evidence and the final answer from the corpus within a single autoregressive process. However, simply using the FM-Index for prefix-constrained generation suffers from severe **false pruning**—where correct evidence sequences are incorrectly pruned during the early stages of decoding.

## Method

### Overall Architecture

The inference process of RetroLLM comprises three stages: (1) Clue Generation: key phrases are generated under the constraint of a corpus-level FM-Index; (2) Evidence Generation: fine-grained evidence is generated under the constraint of candidate document-level FM-Indices; (3) Answer Generation: the final answer is freely generated based on the retrieved evidence. The entire process is completed within a single autoregressive decoding pass.

### Key Designs

1. **Hierarchical FM-Index Constraints**: Two layers of FM-Index are constructed:

    - **Corpus-level global FM-Index** $\mathcal{I}_c$: Constructed from the entire corpus, utilized in the clue generation stage to ensure that generated phrases indeed exist in the corpus.
    - **Document-level FM-Index manager** $\mathcal{I}_d$: Indices are constructed individually for each document, utilized in the evidence generation stage to restrict decoding constraints within the candidate documents.
   
   Empirical study shows that narrowing down the FM-Index constraint from the corpus level to the relevant document level can significantly reduce irrelevant decoding paths; the false pruning issue is particularly severe within the first 13 tokens.

2. **Clue Generation & Document Scoring**:

    - A set of clue phrases (key entities/topic words) is generated under the constraint of the corpus FM-Index.
    - Incorporating the TF-IDF philosophy, weights are assigned to clues: clues that appear less frequently in the corpus and cover fewer documents receive higher weights.
    - Use of auxiliary clues: Additional keywords are extracted from the query using a sparse lexical model (SPLADE-v3).
    - **Weighted Reciprocal Rank Fusion**: The clue generation ranking $R_1$ and the sparse retrieval ranking $R_2$ are fused via weighted reciprocal rank fusion to obtain the final candidate document set $\mathcal{D}_c$.

3. **Forward-Looking Constrained Evidence Generation**: Resolves the problem where the model cannot foresee the relevance of future sequences when predicting the current token, consisting of three steps:

    - **Locate future windows**: Search for text windows containing clues within candidate documents, which typically represent highly query-relevant contexts.
    - **Evaluate window relevance**: Use a re-ranking model (BGE-reranker) to compute the relevance score of each window to the query.
    - **Adjust decoding logits**: During each decoding step, associate the permitted tokens with future windows, adjusting logits based on the window relevance scores:
   
    $$\tilde{l}(t) = l(t) + \lambda \cdot \max_{w \in \mathcal{W}_t} \mathcal{S}_w(w)$$

   Special token control process: `<clue>` starts clue generation, `<evidence>` triggers evidence generation, and `</evidence>` terminates constraint and enters free generation.

4. **Training Data Construction**: Mimics the inference process to construct training data:

    - Direct clues and relevant documents are obtained using a sparse retriever.
    - Locate sentences containing the clues, and use a re-ranker to select the top-k relevant evidence.
    - Verify that the evidence indeed contains the answer and can genuinely answer the query.
    - Use the LLM to extract key entities from the query and relevant evidence as target clues.

### Loss & Training

Standard next-token prediction loss is employed, with two key designs:
- Masking the middle 80% of tokens in the evidence sequences (since evidence is long and the middle parts contribute less to learning).
- The answer generation loss is multiplied by a weight of $\gamma=2$ to strengthen answer generation capability.
- Efficient fine-tuning with LoRA is conducted for 3 epochs.
- The backbone model is Mistral-7B-Instruct.

$$\mathcal{L} = -\sum_{t=1}^{T_c+T_e} \log P(x_t|x_{<t}, q; \theta) - \gamma \sum_{t=1}^{T_a} \log P(y_t|y_{<t}, x, q; \theta)$$

## Key Experimental Results

### Main Results

**Overall QA Performance (Acc / F1 / Token Count):**

| Method | NQ Acc | TriviaQA Acc | HotpotQA Acc | PopQA Acc | 2WIKI Acc |
|------|--------|-------------|-------------|----------|----------|
| Naive RAG | 52.4 | 69.3 | 37.8 | 47.7 | 38.7 |
| IRCoT | 49.6 | 66.0 | 37.3 | 59.8 | 29.4 |
| Iter-RetGen | 51.7 | 71.0 | 37.2 | 51.7 | 29.2 |
| **RetroLLM** | **61.6** | **74.3** | **61.9** | **65.7** | **48.9** |

**Token Consumption Comparison:**

| Method | NQ | TriviaQA | HotpotQA | PopQA | 2WIKI |
|------|-----|----------|----------|-------|-------|
| Naive RAG | 919 | 915 | 960 | 944 | 1000 |
| Iter-RetGen | 3002 | 2461 | 2545 | 2509 | 2669 |
| **RetroLLM** | **302** | **287** | **607** | **355** | **661** |

### Ablation Study

| Configuration | In-domain Acc | In-domain F1 | Out-of-domain Acc | Out-of-domain F1 |
|------|---------|--------|---------|--------|
| Full RetroLLM | 66.0 | 56.6 | 57.3 | 39.6 |
| W/o forward-looking window | 44.3 | 43.2 | 40.9 | 33.8 |
| W/o clue generation | 60.6 | 52.1 | 56.4 | 38.1 |
| W/o clue expansion | 49.6 | 45.1 | 44.1 | 35.4 |
| Naive constraints only | 27.2 | 28.0 | 21.8 | 20.7 |
| W/o constraints | 41.6 | 43.0 | 31.6 | 28.1 |

### Key Findings

- **Unified framework significantly outperforms traditional RAG**: RetroLLM outperforms all RAG baselines, including complex multi-turn retrieval strategies, across all 5 datasets.
- **Token consumption is drastically reduced**: Approximately 2.1 times lower than Naive RAG on average, and about 6 times lower than Iter-RetGen.
- **Severe false pruning issue**: The naive constrained approach (Naive Constraints) yields the worst performance, even underperforming unconstrained free generation.
- **Forward-looking window contributes the most**: Removing the forward-looking window drops the in-domain Acc from 66.0 to 44.3, representing the most significant decline.
- **Excellent cross-domain generalization**: Demonstrates strong performance on the out-of-domain tasks PopQA and 2WIKI.
- **Finer retrieval granularity**: On average, single-hop QA requires only 3.29 evidence segments (vs. a fixed 5 in baselines), and multi-hop QA requires 4.24 segments.
- **Model scaling effect**: Performance steadily improves as parameter scale increases from 1B to 14B, adhering to scaling laws.

## Highlights & Insights

- **True retrieval-generation unification**: Unlike GritLM (shared parameters but decoupled attention) or OneGen (which still retrieves chunks as inputs), RetroLLM achieves a unified end-to-end autoregressive process.
- **In-depth analysis of the false pruning issue**: Empirical research reveals that prefix-constrained approaches suffer from a sharp drop in relevance within the first 13 tokens, providing the intuition for hierarchical constraints.
- **Balance between latency and performance**: RetroLLM is slightly slower than Naive RAG (582ms vs. estimated average of Naive RAG), but is significantly faster than complex RAG methods (e.g., SelfRAG at 3269ms).
- **Flexible control over the evidence count**: The model autonomously determines how much evidence to retrieve, adaptively adjusting for single-hop and multi-hop tasks.

## Limitations & Future Work

- FM-Index construction and maintenance require extra storage and preprocessing overhead, especially for document-level indices.
- The current version relies on an external sparse lexical model (SPLADE-v3) to generate auxiliary clues, falling short of absolute "unification".
- Evaluated on Wikipedia, performance on larger-scale or non-encyclopedic knowledge bases remains unverified.
- LoRA fine-tuning might limit the model's ability to thoroughly learn the joint representations of retrieval and generation.
- Multi-hop QA performance gains saturate when evidence count exceeds 6, suggesting a need for better evidence redundancy filtering mechanisms.
- Re-ranking for forward-looking windows introduces additional latency; lightweight relevance estimation methods could be explored.

## Related Work & Insights

- GritLM (Muennighoff et al., 2024) and OneGen (Zhang et al., 2024a) attempt to unify retrieval and generation, but fail to achieve end-to-end autoregression.
- DSI (Tay et al., 2022) is a pioneering work in generative retrieval, but it only generates DocIDs instead of actual evidence.
- Self-RAG (Asai et al., 2024) decides whether to retrieve via self-reflection, yet still relies on an external retriever.
- REPLUG (Shi et al., 2023) treats the retriever as a plug-in, representing another paradigm of RAG integration.
- The hierarchical constraints + forward-looking decoding scheme of RetroLLM can be generalized to other task scenarios requiring constrained generation.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Truly unifying retrieval and generation into a single autoregressive process; hierarchical FM-Index and forward-looking decoding designs exhibit strong novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive evaluation across 5 datasets, in-domain/out-of-domain assessments, detailed ablation studies, multi-scale/multi-backbone analysis, and efficiency evaluations.
- Writing Quality: ⭐⭐⭐⭐ Dense but solid technical details and clear formulations, though certain sections are packed.
- Value: ⭐⭐⭐⭐⭐ A new paradigm for RAG is proposed, exhibiting significant advantages in both performance and efficiency, offering crucial insights for future RAG system designs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MindRef: Mimicking Human Memory for Hierarchical Reference Retrieval with Fine-Grained Location Awareness](mindref_mimicking_human_memory_hierarchical_reference_retrieval.md)
- [\[ACL 2025\] Dynamic Knowledge Integration for Evidence-Driven Counter-Argument Generation with Large Language Models](dynamic_knowledge_integration_for_evidence-driven_counter-argument_generation_wi.md)
- [\[ACL 2025\] BehaviorBox: Automated Discovery of Fine-Grained Performance Differences Between Language Models](behaviorbox_automated_discovery_of_fine-grained_performance_differences_between_.md)
- [\[ACL 2025\] ChartLens: Fine-Grained Visual Attribution in Charts](chartlens_fine-grained_visual_attribution_in_charts.md)
- [\[ACL 2025\] Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method](can_we_retrieve_everything_all_at_once_arm_an_alignment-oriented_llm-based_retri.md)

</div>

<!-- RELATED:END -->
