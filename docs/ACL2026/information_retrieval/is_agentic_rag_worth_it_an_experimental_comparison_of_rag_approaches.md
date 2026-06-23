---
title: >-
  [Paper Note] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches
description: >-
  [ACL 2026][Information Retrieval & RAG][Agentic RAG] This study systematically compares Enhanced RAG and Agentic RAG across four dimensions—user intent processing, query rewriting, document refinement, and underlying LLM selection—using four datasets. The findings indicate that both paradigms have distinct advantages: Agentic RAG is more flexible in intent routing and qu
tags:
  - ACL 2026
  - Information Retrieval & RAG
  - Agentic RAG
  - Enhanced RAG
date: 2026-05-08
content_hash: ee199856145518ea
---
# Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches

**Conference**: ACL 2026  
**arXiv**: [2601.07711](https://arxiv.org/abs/2601.07711)  
**Code**: None  
**Area**: Information Retrieval / RAG  
**Keywords**: Retrieval-Augmented Generation, Agentic RAG, Enhanced RAG, Query Rewriting, Cost Analysis

## TL;DR

This study systematically compares Enhanced RAG and Agentic RAG across four dimensions—user intent processing, query rewriting, document refinement, and underlying LLM selection—using four datasets. The findings indicate that both paradigms have distinct advantages: Agentic RAG is more flexible in intent routing and query rewriting, while Enhanced RAG is more effective in document reranking. However, the cost of Agentic RAG is up to 3.3 times higher.

## Background & Motivation

**Background**: RAG has evolved from a research concept into a core component of production-grade language systems. Basic Naïve RAG (retrieval + generation) faces several limitations, leading to two improved paradigms: Enhanced RAG (adding modules like routing, rewriting, and reranking into a fixed pipeline) and Agentic RAG (where an LLM acts as an agent to autonomously orchestrate the entire process).

**Limitations of Prior Work**: Despite the rapid adoption of these two paradigms, there is a lack of systematic empirical comparison—under what conditions should one be chosen over the other? What are the trade-offs between performance and cost? Existing work remains at a theoretical level or evaluates only under a single setting.

**Key Challenge**: Agentic RAG offers greater flexibility (dynamic decision-making, iterative retrieval), but does this flexibility translate into actual performance gains? Is the additional inference cost justified?

**Goal**: To conduct controlled experiments across four dimensions (user intent handling, query rewriting, document list refinement, and underlying LLM changes) to quantify the differences in performance and cost between the two paradigms.

**Key Insight**: By mapping four known deficiencies of Naïve RAG to four evaluation dimensions, the study transforms the vague architectural choice of "Enhanced vs. Agentic RAG" into a quantifiable, multi-dimensional decision guide.

**Core Idea**: Through a rigorously controlled experimental design, the "Enhanced vs. Agentic RAG" choice is converted from a conceptual architectural question into a quantifiable decision-making framework.

## Method

### Overall Architecture

The paper does not propose a new model but decomposes the architectural choice between "Enhanced vs. Agentic RAG" into quantifiable controlled experiments. Known deficiencies of Naïve RAG are mapped to four evaluation dimensions: user intent handling, query rewriting, document list refinement, and underlying LLM selection. For each dimension, an Enhanced version (inserting modules into a fixed pipeline) and an Agentic version (LLM autonomous orchestration) are implemented. Both ingest the same queries and knowledge bases to compare retrieval/generation quality across four datasets and calculate token and latency costs. The result is a decision guide on which paradigm to select for each dimension.

### Key Designs

**1. User Intent Handling: Determining whether to trigger retrieval to avoid unnecessary recalls.**

In production, many queries do not require retrieval; blind recall incurs costs and noise. The Enhanced method utilizes the `semantic-router` framework to classify semantic similarity based on pre-defined valid/invalid query examples. The Agentic method removes manual examples, allowing the agent to autonomously decide whether to invoke the RAG tool. Comparing the two on 500 valid and 500 invalid queries tests whether flexibility without manual examples yields more accurate routing—results show Agentic is superior in narrow domains but may over-trigger retrieval in broad domains like FEVER.

**2. Query Rewriting: Bridging the semantic/format gap between short queries and long documents.**

User queries are often short, while knowledge base documents are long, creating a mismatch that degrades retrieval. Evaluation uses $NDCG@10$. The Enhanced method forces the use of HyDE—LLM generates a hypothetical answer paragraph first for retrieval. The Agentic method gives the agent the freedom to decide "whether to rewrite and how to rewrite," adapting as needed.

**3. Document List Refinement: Selecting the most relevant subset from initial results to suppress noise.**

Initial retrieval often includes noise or suboptimal documents. The Enhanced method uses an ELECTRA cross-encoder to explicitly rerank top-20 documents. The Agentic method allows the agent to trigger multi-turn retrieval to iteratively improve context. A key observation emerged: the Agentic version opted to re-retrieve in only 10% of cases, and in 53% of those instances, the documents were identical to the previous round—indicating that once an agent makes a decision, it rarely backtracks. Iterative retrieval proved less effective than explicit reranking in this dimension.

**4. Underlying LLM Selection: Investigating if changing the generation model impacts both paradigms identically.**

In Agentic RAG, the model both generates answers and makes tool-calling decisions. A common intuition is that weaker/cheaper LLMs might hurt Agentic performance more than Enhanced performance. Using the same queries and retrieval contexts, results were compared across Qwen3 (0.6B/4B/8B/32B). Using "win rate of large model over small model" as a metric (judged by Selene-70B), the curves for both paradigms were nearly identical. Improvements in scale yield consistent gains for both, suggesting that LLM selection can be considered independently of the architectural paradigm.

### Loss & Training

This study does not involve model training. Enhanced RAG modules utilize pre-trained models (OpenAI embeddings, ELECTRA reranker), while Agentic RAG is implemented via the PocketFlow framework. Generation and decision-making are handled by the four Qwen3 variants, and answer quality is evaluated using LLM-as-a-Judge (Selene-70B).

## Key Experimental Results

### Main Results

**User Intent Handling (F1)**

| Setting | FIQA | FEVER | CQA-EN |
|:---|:---:|:---:|:---:|
| Naïve | 66.7 | 66.7 | 66.7 |
| Enhanced | 95.7 | **87.9** | 96.6 |
| Agentic | **98.8** | 64.6 | **99.8** |

**Query Rewriting ($NDCG@10$)**

| Setting | FIQA | NQ | FEVER | CQAD | AVG |
|:---|:---:|:---:|:---:|:---:|:---:|
| Naïve | 45.3 | 43.7 | 66.2 | 45.8 | 50.3 |
| Enhanced | 43.5 | 43.9 | 81.1 | 42.8 | 52.8 |
| Agentic | 43.2 | **51.7** | **83.1** | 44.3 | **55.6** |

**Document Refinement ($NDCG@10$)**

| Setting | FIQA | CQA-EN | AVG |
|:---|:---:|:---:|:---:|
| Naïve | 45.0 | 46.0 | 45.5 |
| Enhanced (w/ rewriting) | **51.0** | **48.0** | **49.5** |
| Agentic | 43.4 | 44.4 | 43.9 |

### Ablation Study

**Cost Comparison**

| Metric | Agentic vs Enhanced |
|:---|:---:|
| Input Tokens | 3.3× |
| Output Tokens | 1.9× |
| Latency | 1.5× |

### Key Findings

- Agentic RAG performs better in user intent judgment (without manual examples) but may over-retrieve in broad domains (e.g., FEVER).
- Agentic RAG's adaptive query rewriting is 2.8 $NDCG@10$ higher on average than Enhanced, with a jump of +7.8 in open-domain queries (NQ).
- Enhanced RAG's explicit reranking significantly outperforms Agentic's iterative retrieval—agents rarely change decisions once made.
- Changing the underlying LLM shows consistent impact patterns for both paradigms, with performance scaling linearly with model size.

## Highlights & Insights

- Decomposed the ambiguous "Enhanced vs. Agentic" choice into four independent dimensions, providing clear recommendations for each.
- The final recommendation is a hybrid approach: use Agentic for intent routing and query rewriting, and use Enhanced rerankers for document refinement.
- The cost analysis is highly practical—the 3.3x token consumption of Agentic RAG reminds practitioners to carefully weigh costs.

## Limitations & Future Work

- Evaluated only single-tool Agentic RAG (retrieval only); multi-tool agent performance may differ.
- Agentic experiments on large-scale datasets (NQ, FEVER) were excluded due to excessive runtime (>7 days).
- Evaluated using LLM-as-a-Judge, which is subject to evaluation bias.
- Did not explore more complex Agentic strategies (e.g., reflection, planning).

## Related Work & Insights

- **vs. CRAG/Self-RAG**: These fall under the category of Enhanced RAG; this paper incorporates them into a unified comparison framework.
- **vs. LLM Agent Frameworks**: This study uses a simple single-tool Agentic design to maintain comparability.
- **Insight**: In actual deployment, a hybrid of the two paradigms should be used based on domain characteristics, rather than adopting one exclusively.

## Rating

- Novelty: ⭐⭐⭐ Important problem, but the methodology relies on experimental comparison rather than technical innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets × four dimensions × cost analysis, though some experiments were limited by cost.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and strongly utility-oriented, suitable for engineering reference.
- Value: ⭐⭐⭐⭐ Provides a much-needed empirical guide for RAG architecture selection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)
- [\[ACL 2026\] 生物医学 RAG 中检索何时无效：大规模实证研究](when_retrieval_doesnt_help_a_large-scale_study_of_biomedical_rag.md)
- [\[ACL 2026\] DQA: Diagnostic Question Answering for IT Support](dqa_diagnostic_question_answering_for_it_support.md)
- [\[ACL 2026\] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)
- [\[CVPR 2026\] M4-RAG: A Massive-Scale Multilingual Multi-Cultural Multimodal RAG](../../CVPR2026/information_retrieval/m4-rag_a_massive-scale_multilingual_multi-cultural_multimodal_rag.md)

</div>

<!-- RELATED:END -->
