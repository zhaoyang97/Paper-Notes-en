---
title: >-
  [Paper Note] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches
description: >-
  [ACL 2026][Retrieval-Augmented Generation] This paper systematically compares Enhanced RAG and Agentic RAG across four dimensions—user intent handling, query rewriting, document refinement, and underlying LLM selection—on four datasets. The results show that each paradigm has distinct advantages: Agentic RAG is more flexible in intent routing and query rewriting, while Enhanced RAG is more effective in document reranking. Notably, Agentic RAG incurs up to 3.3× higher cost.
tags:
  - ACL 2026
  - Retrieval-Augmented Generation
  - Agentic RAG
  - Enhanced RAG
  - Query Rewriting
  - Cost Analysis
date: 2026-05-08
content_hash: ccf4572477ad7e99
---

# Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches

**Conference**: ACL 2026
**arXiv**: [2601.07711](https://arxiv.org/abs/2601.07711)
**Code**: None
**Area**: Information Retrieval / RAG
**Keywords**: Retrieval-Augmented Generation, Agentic RAG, Enhanced RAG, Query Rewriting, Cost Analysis

## TL;DR

This paper systematically compares Enhanced RAG and Agentic RAG across four dimensions—user intent handling, query rewriting, document refinement, and underlying LLM selection—on four datasets. The results show that each paradigm has distinct advantages: Agentic RAG is more flexible in intent routing and query rewriting, while Enhanced RAG is more effective in document reranking. Notably, Agentic RAG incurs up to 3.3× higher cost.

## Background & Motivation

**State of the Field**: RAG has evolved from a research concept into a core component of production-grade language systems. Naïve RAG (retrieve-then-generate) has several well-known limitations, giving rise to two improvement paradigms: Enhanced RAG (adding modules such as routing, rewriting, and reranking into a fixed pipeline) and Agentic RAG (an LLM agent autonomously orchestrates the entire workflow).

**Limitations of Prior Work**: Despite the rapid adoption of both paradigms, a systematic empirical comparison is lacking—under what conditions should one be preferred over the other? How do performance and cost trade off? Existing work either remains at the level of theoretical definitions or evaluates only a single configuration.

**Root Cause**: Agentic RAG offers greater flexibility (dynamic decision-making, iterative retrieval), but it remains unclear whether this flexibility translates into tangible performance gains or whether the additional inference cost is justified.

**Paper Goals**: To conduct controlled experiments across four dimensions (user intent handling, query rewriting, document list refinement, and underlying LLM variation) and quantify the performance and cost differences between the two paradigms.

**Starting Point**: The four known deficiencies of Naïve RAG are mapped to four evaluation dimensions, for each of which both an Enhanced and an Agentic implementation are designed and compared.

**Core Idea**: Through rigorous controlled experimental design, the paper transforms the vague architectural choice of "Enhanced vs. Agentic RAG" into a quantifiable, dimension-wise decision guide.

## Method

### Overall Architecture

For each of the four deficiencies of Naïve RAG, both Enhanced and Agentic solutions are implemented: (1) user intent handling—Enhanced uses a semantic router, Agentic lets the agent decide autonomously; (2) query rewriting—Enhanced applies HyDE unconditionally, Agentic decides whether and how to rewrite; (3) document refinement—Enhanced uses an ELECTRA reranker, Agentic may iteratively retrieve multiple times; (4) underlying LLM—four model sizes of Qwen3 (0.6B/4B/8B/32B) are evaluated.

### Key Designs

1. **User Intent Handling**:

    - Function: Determine whether a user query should trigger retrieval.
    - Mechanism: The Enhanced approach uses the `semantic-router` framework, classifying queries by semantic similarity against predefined examples of valid/invalid queries. The Agentic approach lets the agent autonomously decide whether to invoke the RAG tool. Evaluation is conducted on 500 valid and 500 invalid queries.
    - Design Motivation: Unnecessary retrieval for queries that do not require it should be avoided—a critical requirement in production environments.

2. **Query Rewriting**:

    - Function: Bridge the semantic and format gap between user queries and knowledge-base documents.
    - Mechanism: The Enhanced approach unconditionally applies HyDE (prompting the LLM to generate a hypothetical answer passage as the query), while the Agentic approach freely chooses whether and how to rewrite. Retrieval quality is evaluated with NDCG@10.
    - Design Motivation: User queries are typically short questions, whereas knowledge-base documents are long-form text; this format mismatch degrades retrieval.

3. **Document List Refinement**:

    - Function: Optimize the retrieved document list to select the most relevant subset.
    - Mechanism: The Enhanced approach applies an ELECTRA cross-encoder to rerank the top-20 documents. The Agentic approach allows the agent to trigger multiple retrieval rounds to iteratively improve context. Experiments show that the agent opts to re-retrieve in only 10% of cases, and 53% of retrieved documents overlap with the previous round.
    - Design Motivation: Initial retrieval results may contain noise or suboptimal documents, necessitating a refinement strategy.

### Loss & Training

No model training is involved. Enhanced RAG modules use pre-trained models (OpenAI embeddings, ELECTRA reranker); Agentic RAG is implemented with the PocketFlow framework. Evaluation uses LLM-as-a-Judge (Selene-70B).

## Key Experimental Results

### Main Results

**User Intent Handling (F1)**

| Setting | FIQA | FEVER | CQA-EN |
|---------|------|-------|--------|
| Naïve | 66.7 | 66.7 | 66.7 |
| Enhanced | 95.7 | **87.9** | 96.6 |
| Agentic | **98.8** | 64.6 | **99.8** |

**Query Rewriting (NDCG@10)**

| Setting | FIQA | NQ | FEVER | CQAD | AVG |
|---------|------|----|-------|------|-----|
| Naïve | 45.3 | 43.7 | 66.2 | 45.8 | 50.3 |
| Enhanced | 43.5 | 43.9 | 81.1 | 42.8 | 52.8 |
| Agentic | 43.2 | **51.7** | **83.1** | 44.3 | **55.6** |

**Document Refinement (NDCG@10)**

| Setting | FIQA | CQA-EN | AVG |
|---------|------|--------|-----|
| Naïve | 45.0 | 46.0 | 45.5 |
| Enhanced (w/ rewriting) | **51.0** | **48.0** | **49.5** |
| Agentic | 43.4 | 44.4 | 43.9 |

### Ablation Study

**Cost Comparison**

| Metric | Agentic vs. Enhanced |
|--------|---------------------|
| Input tokens | 3.3× |
| Output tokens | 1.9× |
| Latency | 1.5× |

### Key Findings

- Agentic RAG achieves better user intent classification (requiring no manual example construction), but over-triggers retrieval on broad-domain datasets such as FEVER.
- Agentic RAG's adaptive query rewriting outperforms Enhanced RAG by an average of 2.8 NDCG@10, with a +7.8 advantage on open-domain queries (NQ).
- Enhanced RAG's explicit reranking substantially outperforms Agentic iterative retrieval—once the agent makes a decision, it rarely revises it.
- Switching the underlying LLM affects both paradigms in a consistent pattern: performance scales proportionally with model size.

## Highlights & Insights

- The vague "Enhanced vs. Agentic" architectural choice is decomposed into four independent dimensions, each with a clear recommendation.
- The final recommendation is a hybrid approach: use Agentic for intent routing and query rewriting, and use Enhanced reranking for document refinement.
- The cost analysis is highly practical—the 3.3× token overhead of Agentic RAG serves as a strong reminder that practitioners must carefully account for cost.

## Limitations & Future Work

- Only single-tool Agentic RAG (with retrieval as the sole tool) is evaluated; multi-tool agent configurations may behave differently.
- Agentic experiments on large-scale datasets (NQ, FEVER) were excluded due to excessive runtime (>7 days).
- LLM-as-a-Judge evaluation introduces potential assessment bias.
- More sophisticated Agentic strategies (e.g., reflection, planning) are not explored.

## Related Work & Insights

- **vs. CRAG/Self-RAG**: These fall under the Enhanced RAG paradigm and are incorporated into the unified comparison framework.
- **vs. LLM Agent Frameworks**: The simplest single-tool Agentic design is adopted to maintain comparability.
- **Insight**: Practical deployments should mix both paradigms according to domain characteristics, rather than adopting either one wholesale.

## Rating

- Novelty: ⭐⭐⭐ The problem is important, but there is no methodological innovation; the contribution is primarily empirical comparison.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets × four dimensions × cost analysis, though some experiments are constrained by cost.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with a strong practical orientation; well-suited as an engineering reference.
- Value: ⭐⭐⭐⭐ Provides a much-needed empirical guide for RAG architecture selection.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] DQA: Diagnostic Question Answering for IT Support](dqa_diagnostic_question_answering_for_it_support.md)
- [\[ACL 2026\] How Retrieved Context Shapes Internal Representations in RAG](how_retrieved_context_shapes_internal_representations_in_rag.md)
- [\[CVPR 2026\] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](../../CVPR2026/information_retrieval/m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ACL 2026\] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)

<!-- RELATED:END -->
