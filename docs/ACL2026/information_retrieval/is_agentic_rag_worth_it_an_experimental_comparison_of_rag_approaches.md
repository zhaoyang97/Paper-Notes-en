---
title: >-
  [Paper Note] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches
description: >-
  [ACL 2026][Information Retrieval & RAG][Agentic RAG] This paper systematically compares Enhanced RAG and Agentic RAG across four dimensions—user intent handling, query rewriting, document refinement, and underlying LLM selection—on four datasets. It finds complementary strengths: Agentic RAG is more flexible in intent routing and query rewriting, whereas Enhanced RAG is
tags:
  - ACL 2026
  - Information Retrieval & RAG
  - Agentic RAG
  - Enhanced RAG
date: 2026-05-08
content_hash: 17dd48d3e47228f8
---
# Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches

**Conference**: ACL 2026  
**arXiv**: [2601.07711](https://arxiv.org/abs/2601.07711)  
**Code**: None  
**Area**: Information Retrieval / RAG  
**Keywords**: Retrieval-Augmented Generation, Agentic RAG, Enhanced RAG, Query Rewriting, Cost Analysis

## TL;DR

This paper systematically compares Enhanced RAG and Agentic RAG across four dimensions—user intent handling, query rewriting, document refinement, and underlying LLM selection—on four datasets. It finds complementary strengths: Agentic RAG is more flexible in intent routing and query rewriting, whereas Enhanced RAG is more effective for document reranking, despite Agentic RAG incurring costs up to 3.3 times higher.

## Background & Motivation

**Background**: RAG has evolved from a research concept to a core component of production-grade language systems. Basic Naïve RAG (retrieval + generation) has multiple limitations, leading to two improved paradigms: Enhanced RAG (adding modules like routing, rewriting, and reranking into fixed pipelines) and Agentic RAG (LLMs acting as agents to autonomously orchestrate the entire process).

**Limitations of Prior Work**: Despite rapid adoption, systematic empirical comparisons are lacking—under what conditions should one choose which approach? What are the performance and cost trade-offs? Existing work either remains at a theoretical level or evaluates only under limited settings.

**Key Challenge**: Agentic RAG provides greater flexibility (dynamic decision-making, iterative retrieval), but does this flexibility translate into actual performance gains? Is the additional inference cost justified?

**Goal**: To conduct controlled experiments across four dimensions (user intent handling, query rewriting, document list refinement, underlying LLM selection) to quantify performance and cost differences between the two paradigms.

**Key Insight**: Mapping four known deficiencies of Naïve RAG to four evaluation dimensions and designing comparative implementations for Enhanced and Agentic approaches.

**Core Idea**: Transforming the vague architectural choice of "Enhanced vs. Agentic RAG" into a quantifiable, multi-dimensional decision guide through strictly controlled experimental designs.

## Method

### Overall Architecture

The authors deconstruct the vague "Enhanced vs. Agentic RAG" selection into quantifiable controlled experiments. They map four known deficiencies of Naïve RAG (retrieval + generation) to four evaluation dimensions: user intent handling, query rewriting, document list refinement, and underlying LLM selection. For each dimension, they implement an Enhanced solution (fixed pipeline with inserted modules) and an Agentic solution (LLM-based autonomous orchestration). Using identical queries and knowledge bases across four datasets, they compare retrieval/generation quality and account for token and latency costs to produce a decision guide for paradigm selection.

### Key Designs

**1. User Intent Handling: Determining whether to trigger retrieval to avoid unnecessary recalls.**

In production, many queries do not require retrieval, and blind recall introduces costs and noise. The Enhanced approach utilizes the semantic-router framework, using predefined valid/invalid query examples for semantic similarity classification. The Agentic approach removes manual examples, letting the agent autonomously decide whether to call RAG tools. Comparing them on 500 valid and 500 invalid queries tests whether flexibility (no manual examples) yields better routing. Findings show Agentic is superior in narrow domains but tends to over-trigger retrieval in broad domains like FEVER.

**2. Query Rewriting: Bridging the semantic/format gap between short queries and long documents.**

Short user questions often have a format mismatch with long knowledge base documents, degrading retrieval performance. The Enhanced approach mandates HyDE—letting the LLM generate a hypothetical answer paragraph for retrieval. The Agentic approach gives the agent freedom to decide whether and how to rewrite, adapting as needed. Retrieval quality is measured uniformly via NDCG@10 to compare fixed vs. adaptive strategies.

**3. Document List Refinement: Selecting the most relevant subset and suppressing noise from initial results.**

Initial retrieval often includes noise or suboptimal documents, requiring secondary refinement. The Enhanced approach uses an ELECTRA cross-encoder for explicit top-20 reranking. The Agentic approach allows the agent to trigger multi-turn retrieval to iteratively improve context. Comparing these "refinement philosophies" revealed a key observation: Agentic chose to re-retrieve in only 10% of cases, and 53% of those documents were identical to the previous turn—indicating that agents rarely change decisions once made, making iterative retrieval less effective than explicit reranking.

**4. Underlying LLM Selection: Investigating if switching generation models impacts the two paradigms consistently.**

Generation models in Agentic RAG serve two roles: writing the final answer and deciding on tool use/re-retrieval. A common intuition is that weaker/cheaper LLMs might hurt Agentic more than Enhanced. This study used the same queries and retrieval contexts across Qwen3 models (0.6B/4B/8B/32B), quantifying scale benefits by calculating the percentage of times larger models outperformed smaller ones (via LLM-as-a-Judge paired evaluation with Selene-70B). Results show nearly identical performance curves for both paradigms as scale increases, suggesting LLM selection is independent of the RAG paradigm choice.

### Loss & Training

This work involves no model training. Enhanced RAG modules utilize pre-trained models (OpenAI embeddings, ELECTRA reranker), while Agentic RAG is implemented via the PocketFlow framework. Generation and decision-making are handled by the four Qwen3 tiers, with final answer quality judged by LLM-as-a-Judge (Selene-70B).

## Key Experimental Results

### Main Results

**User Intent Handling (F1)**

| Setting | FIQA | FEVER | CQA-EN |
|------|------|-------|--------|
| Naïve | 66.7 | 66.7 | 66.7 |
| Enhanced | 95.7 | **87.9** | 96.6 |
| Agentic | **98.8** | 64.6 | **99.8** |

**Query Rewriting (NDCG@10)**

| Setting | FIQA | NQ | FEVER | CQAD | AVG |
|------|------|----|-------|------|-----|
| Naïve | 45.3 | 43.7 | 66.2 | 45.8 | 50.3 |
| Enhanced | 43.5 | 43.9 | 81.1 | 42.8 | 52.8 |
| Agentic | 43.2 | **51.7** | **83.1** | 44.3 | **55.6** |

**Document Refinement (NDCG@10)**

| Setting | FIQA | CQA-EN | AVG |
|------|------|--------|-----|
| Naïve | 45.0 | 46.0 | 45.5 |
| Enhanced (w/ rewriting) | **51.0** | **48.0** | **49.5** |
| Agentic | 43.4 | 44.4 | 43.9 |

### Ablation Study

**Cost Comparison**

| Metric | Agentic vs. Enhanced |
|------|-------------------|
| Input tokens | 3.3× |
| Output tokens | 1.9× |
| Time | 1.5× |

### Key Findings

- Agentic RAG performs better in user intent judgment (no manual examples needed) but over-retrieves in broad domains (e.g., FEVER).
- Adaptive query rewriting in Agentic RAG averages 2.8 NDCG@10 higher than Enhanced, with a +7.8 advantage on open-domain queries (NQ).
- Explicit reranking in Enhanced RAG is far superior to iterative retrieval in Agentic RAG—agents rarely change decisions once made.
- Changing the underlying LLM impacts both paradigms similarly—performance scales consistently with model size.

## Highlights & Insights

- Decomposes the vague "Enhanced vs. Agentic" choice into four independent dimensions, providing clear recommendations for each.
- Recommends a hybrid approach: Use Agentic RAG for intent routing and query rewriting, and Enhanced RAG's reranker for document refinement.
- Practical cost analysis—reminds practitioners of Agentic RAG's 3.3× token consumption.

## Limitations & Future Work

- Evaluation only covers single-tool Agentic RAG (retrieval only); multi-tool agents may behave differently.
- Agentic experiments on large-scale datasets (NQ, FEVER) were excluded due to excessive time (>7 days).
- Evaluation uses LLM-as-a-Judge, which has inherent biases.
- More complex agentic strategies (reflection, planning) were not explored.

## Related Work & Insights

- **vs. CRAG/Self-RAG**: These fall under the Enhanced RAG category and are included in the uniform comparison framework.
- **vs. LLM Agent Frameworks**: This study uses the simplest single-tool Agentic design to maintain comparability.
- **Insight**: Real-world deployments should hybridize both paradigms based on domain characteristics rather than adopting one exclusively.

## Rating

- Novelty: ⭐⭐⭐ Problem is important but the methodology is experimental comparison rather than architectural innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets × four dimensions × cost analysis, though some experiments were cost-constrained.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured and practical, suitable for engineering reference.
- Value: ⭐⭐⭐⭐ Provides a much-needed empirical guide for RAG architectural choices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] 生物医学 RAG 中检索何时无效：大规模实证研究](when_retrieval_doesnt_help_a_large-scale_study_of_biomedical_rag.md)
- [\[ACL 2026\] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)
- [\[ACL 2026\] DQA: Diagnostic Question Answering for IT Support](dqa_diagnostic_question_answering_for_it_support.md)
- [\[ACL 2026\] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)
- [\[CVPR 2026\] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](../../CVPR2026/information_retrieval/m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)

</div>

<!-- RELATED:END -->
