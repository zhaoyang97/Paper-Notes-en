---
title: >-
  [Paper Note] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches
description: >-
  [ACL 2026][Information Retrieval & RAG][Retrieval-Augmented Generation] This paper systematically compares Enhanced RAG and Agentic RAG across four datasets from four dimensions: user intent handling, query rewriting…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Retrieval-Augmented Generation"
  - "Agentic RAG"
  - "Enhanced RAG"
  - "Query Rewriting"
  - "Cost Analysis"
date: 2026-05-08
content_hash: 154d2c08bb02f211
---

# Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches

**Conference**: ACL 2026  
**arXiv**: [2601.07711](https://arxiv.org/abs/2601.07711)  
**Code**: None  
**Area**: Information Retrieval / RAG  
**Keywords**: Retrieval-Augmented Generation, Agentic RAG, Enhanced RAG, Query Rewriting, Cost Analysis

## TL;DR

This paper systematically compares Enhanced RAG and Agentic RAG across four datasets from four dimensions: user intent handling, query rewriting, document refinement, and underlying LLM selection. The study finds that each paradigm has distinct advantages—Agentic RAG is more flexible in intent routing and query rewriting, while Enhanced RAG is more effective in document reranking, despite Agentic RAG's cost being up to 3.3 times higher.

## Background & Motivation

**Background**: RAG has evolved from a research concept into a core component of production-grade language systems. Limitations in basic Naïve RAG (retrieve-then-generate) have led to two improved paradigms: Enhanced RAG (adding modules like routing, rewriting, and reranking to a fixed pipeline) and Agentic RAG (autonomous orchestration of the entire process by an LLM agent).

**Limitations of Prior Work**: Despite the rapid adoption of these two paradigms, there is a lack of systematic empirical comparison—under what conditions should one be chosen over the other? What are the trade-offs between performance and cost? Existing work either remains at the theoretical definition level or evaluates only under a single setting.

**Key Challenge**: Agentic RAG offers greater flexibility (dynamic decision-making, iterative retrieval), but does this flexibility translate into actual performance gains? Is the additional inference cost justifiable?

**Goal**: Conduct controlled experiments across four dimensions (user intent handling, query rewriting, document list refinement, and underlying LLM changes) to quantify the differences in performance and cost between the two paradigms.

**Key Insight**: Map four known deficiencies of Naïve RAG to four evaluation dimensions, designing specific Enhanced and Agentic implementations for comparison.

**Core Idea**: Through a strictly controlled experimental design, transform the vague architectural choice of "Enhanced vs. Agentic RAG" into a quantifiable, dimension-based decision guide.

## Method

### Overall Architecture

To address the four deficiencies of Naïve RAG, both Enhanced and Agentic versions were implemented: (1) User intent handling—Enhanced uses a semantic router, while Agentic relies on autonomous agent judgment; (2) Query rewriting—Enhanced enforces HyDE rewriting, while Agentic decides whether and how to rewrite; (3) Document refinement—Enhanced employs an ELECTRA reranker, while Agentic allows multiple iterative retrievals; (4) Underlying LLM—Tests Qwen3 across four scales: 0.6B, 4B, 8B, and 32B.

### Key Designs

1.  **User Intent Handling**:
    *   **Function**: Determines if a user query needs to trigger retrieval.
    *   **Mechanism**: The Enhanced approach uses the `semantic-router` framework, performing semantic similarity classification based on predefined valid/invalid query examples. The Agentic approach lets the agent autonomously decide whether to call RAG tools. Evaluation was conducted on 500 valid + 500 invalid queries.
    *   **Design Motivation**: To avoid meaningless retrieval operations for queries that do not require external knowledge, which is a critical requirement in production environments.

2.  **Query Rewriting**:
    *   **Function**: Bridges the semantic/format gap between user queries and knowledge base documents.
    *   **Mechanism**: Enhanced strictly uses HyDE (generating a hypothetical answer paragraph as the query), whereas Agentic can choose whether to rewrite and how to do so. Retrieval quality is evaluated using NDCG@10.
    *   **Design Motivation**: User queries are often short questions, while knowledge base documents are long-form texts, leading to poor retrieval matching due to format discrepancies.

3.  **Document List Refinement**:
    *   **Function**: Optimizes the retrieved document list to select the most relevant subset.
    *   **Mechanism**: Enhanced uses an ELECTRA cross-encoder to rerank the top-20 documents. Agentic allows the agent to trigger multi-turn retrieval to iteratively improve context. Experiments found that the Agentic version chose to re-retrieve in only 10% of cases, with 53% of documents being identical to the previous turn.
    *   **Design Motivation**: Initial retrieval results may contain noise or suboptimal documents, necessitating refinement strategies.

### Loss & Training

This paper does not involve model training. Modules in Enhanced RAG use pre-trained models (OpenAI embeddings, ELECTRA reranker), while Agentic RAG is implemented using the PocketFlow framework. Evaluation is performed via LLM-as-a-Judge (Selene-70B).

## Key Experimental Results

### Main Results

**User Intent Handling (F1)**

| Setting | FIQA | FEVER | CQA-EN |
| :--- | :--- | :--- | :--- |
| Naïve | 66.7 | 66.7 | 66.7 |
| Enhanced | 95.7 | **87.9** | 96.6 |
| Agentic | **98.8** | 64.6 | **99.8** |

**Query Rewriting (NDCG@10)**

| Setting | FIQA | NQ | FEVER | CQAD | AVG |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Naïve | 45.3 | 43.7 | 66.2 | 45.8 | 50.3 |
| Enhanced | 43.5 | 43.9 | 81.1 | 42.8 | 52.8 |
| Agentic | 43.2 | **51.7** | **83.1** | 44.3 | **55.6** |

**Document Refinement (NDCG@10)**

| Setting | FIQA | CQA-EN | AVG |
| :--- | :--- | :--- | :--- |
| Naïve | 45.0 | 46.0 | 45.5 |
| Enhanced (w/ rewriting) | **51.0** | **48.0** | **49.5** |
| Agentic | 43.4 | 44.4 | 43.9 |

### Ablation Study

**Cost Comparison**

| Metric | Agentic vs Enhanced |
| :--- | :--- |
| Input tokens | 3.3× |
| Output tokens | 1.9× |
| Latency | 1.5× |

### Key Findings

*   Agentic RAG performs better at user intent judgment (no manual examples required) but tends to overuse retrieval in broad domains (e.g., FEVER).
*   Adaptive query rewriting in Agentic RAG outperforms the Enhanced version by 2.8 NDCG@10 on average, with a +7.8 gain on open-domain queries (NQ).
*   Explicit reranking in Enhanced RAG is far superior to iterative retrieval in Agentic RAG—once an agent makes a decision, it rarely changes it.
*   Scaling the underlying LLM impacts both paradigms similarly—performance increases proportionally with model size.

## Highlights & Insights

*   Deconstructs the fuzzy "Enhanced vs. Agentic" choice into four independent dimensions, providing clear recommendations for each.
*   The final recommendation is a hybrid approach: use Agentic for intent routing and query rewriting, and use the Enhanced version's reranker for document refinement.
*   The cost analysis is highly practical—the fact that Agentic RAG consumes up to 3.3 times more tokens serves as a reminder for practitioners to carefully consider the budget.

## Limitations & Future Work

*   Only evaluates single-tool Agentic RAG (retrieval only); multi-tool agents might perform differently.
*   Agentic experiments on large-scale datasets (NQ, FEVER) were excluded due to excessive time requirements (>7 days).
*   Evaluation relies on LLM-as-a-Judge, which may introduce evaluation bias.
*   Complex Agentic strategies (e.g., reflection, planning) were not explored.

## Related Work & Insights

*   **vs. CRAG/Self-RAG**: These fall under the category of Enhanced RAG; this paper incorporates them into a unified comparison framework.
*   **vs. LLM Agent Frameworks**: This paper uses the simplest single-tool Agentic design to maintain comparability.
*   **Insight**: In practical deployments, paradigms should be mixed based on domain characteristics rather than adopting one exclusively.

## Rating

*   Novelty: ⭐⭐⭐ The problem is important, though the methodology is comparative rather than innovative.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets × four dimensions × cost analysis, though some experiments were limited by cost.
*   Writing Quality: ⭐⭐⭐⭐ Clear structure and strongly practical, suitable for engineering reference.
*   Value: ⭐⭐⭐⭐ Provides a much-needed empirical guide for RAG architectural selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)
- [\[ACL 2026\] DQA: Diagnostic Question Answering for IT Support](dqa_diagnostic_question_answering_for_it_support.md)
- [\[ACL 2026\] How Retrieved Context Shapes Internal Representations in RAG](how_retrieved_context_shapes_internal_representations_in_rag.md)
- [\[ACL 2026\] CORAL: Adaptive Retrieval Loop for Culturally-Aligned Multilingual RAG](coral_adaptive_retrieval_loop_for_culturally-aligned_multilingual_rag.md)
- [\[ACL 2026\] When Retrieval is Ineffective in Biomedical RAG: A Large-Scale Empirical Study](when_retrieval_doesnt_help_a_large-scale_study_of_biomedical_rag.md)

</div>

<!-- RELATED:END -->
